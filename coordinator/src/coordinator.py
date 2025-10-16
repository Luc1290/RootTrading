"""
Coordinator simplifié pour RootTrading.
Rôle : Valider la faisabilité des signaux et les transmettre au trader.
"""

import json
import logging
import threading
import time
from typing import Any

from shared.src.db_pool import DBConnectionPool
from shared.src.enums import OrderSide, SignalStrength
from shared.src.redis_client import RedisClient
from shared.src.schemas import StrategySignal

from .service_client import ServiceClient
from .trailing_sell_manager import TrailingSellManager
from .universe_manager import UniverseManager

logger = logging.getLogger(__name__)


class Coordinator:
    """
    Coordinateur simplifié : reçoit les signaux, vérifie la faisabilité, transmet au trader.
    """

    def __init__(
        self,
        trader_api_url: str = "http://trader:5002",
        portfolio_api_url: str = "http://portfolio:8000",
    ):
        """
        Initialise le coordinator.

        Args:
            trader_api_url: URL du service trader
            portfolio_api_url: URL du service portfolio
        """
        self.service_client = ServiceClient(trader_api_url, portfolio_api_url)
        self.redis_client = RedisClient()

        # Pool de connexions DB (plus robuste)
        self.db_pool = None

        # Thread de monitoring stop-loss
        self.stop_loss_active = True
        self.stop_loss_thread: threading.Thread | None = None

        # Thread de mise à jour de l'univers
        self.universe_update_active = True
        self.universe_update_thread: threading.Thread | None = None
        self.universe_update_interval = 300  # Mise à jour toutes les 5 minutes

        # Configuration dynamique basée sur l'USDC disponible
        self.fee_rate = 0.001  # 0.1% de frais estimés par trade

        # Allocation basée USDC - AUGMENTÉE pour positions plus importantes
        # 18% de l'USDC disponible (était 10%)
        self.base_allocation_usdc_percent = 18.0
        # 22% pour signaux forts (était 12%)
        self.strong_allocation_usdc_percent = 22.0
        self.max_allocation_usdc_percent = (
            28.0  # 28% maximum pour VERY_STRONG (était 15%)
        )
        # 12% pour signaux faibles (était 7%)
        self.weak_allocation_usdc_percent = 12.0
        self.usdc_safety_margin = 0.98  # Garde 2% d'USDC en sécurité
        self.min_absolute_trade_usdc = (
            15.0  # 15 USDC minimum - évite micro-positions (était 1 USDC)
        )

        # Configuration des seuils de force de signal (centralisée)
        self.signal_strength_config = {
            # Seuils pour consensus override (BUY fort = ajout immédiat à
            # l'univers)
            "consensus_override": {
                "min_force": 2.0,  # Force minimum (au lieu de 2.5 arbitraire)
                # Stratégies minimum (au lieu de 6 arbitraire)
                "min_strategies": 5,
            },
            # Seuils pour catégorisation de force (allocation)
            "categorization": {
                "very_strong_threshold": 12.0,  # Au lieu de 20
                "strong_threshold": 8.0,  # Au lieu de 15
                "moderate_threshold": 4.0,  # Au lieu de 10
                # En dessous = WEAK
            },
            # Seuils pour consensus SELL forcé
            "consensus_sell": {
                "min_strategies": 4,  # Au lieu de 5
                "min_strength": 1.8,  # Au lieu de 2.0
                "loss_multiplier": 0.6,  # Perte = -0.6xATR% pour forcer
            },
        }

        # Initialiser le pool de connexions DB
        self._init_db_pool()

        # Initialiser les symboles dans Redis si nécessaires
        self._init_symbols()

        # Obtenir une connexion DB dédiée pour le trailing manager
        trailing_db_connection = None
        if self.db_pool:
            trailing_db_connection = self.db_pool.get_connection()
            logger.info("Connexion DB dédiée créée pour TrailingSellManager")

        # Initialiser le gestionnaire de trailing sell avec une connexion
        # directe
        self.trailing_db_connection = trailing_db_connection  # Garder la référence
        self.trailing_manager = TrailingSellManager(
            redis_client=self.redis_client,
            service_client=self.service_client,
            db_connection=trailing_db_connection,  # Connexion directe pour trailing
        )

        # Initialiser le gestionnaire d'univers pour la sélection dynamique
        self.universe_manager = UniverseManager(
            redis_client=self.redis_client,  # Passer directement l'instance RedisClient
            db_pool=self.db_pool,  # Passer le pool DB au lieu d'une connexion directe
            config=None,  # Utilise la config par défaut
        )

        # Configuration stop-loss - SUPPRIMÉE : toute la logique est dans TrailingSellManager
        # self.stop_loss_percent_* supprimés pour éviter duplication de code
        # Vérification des prix toutes les 60 secondes (aligné sur la fréquence
        # des données)
        self.price_check_interval = 60

        # Démarrer le monitoring stop-loss
        self.start_stop_loss_monitoring()

        # Démarrer la mise à jour de l'univers
        self.start_universe_update()

        logger.info(
            f"✅ Coordinator initialisé - Allocation USDC AUGMENTÉE: {self.weak_allocation_usdc_percent}-{self.max_allocation_usdc_percent}% (min: {self.min_absolute_trade_usdc} USDC)"
        )

        # Stats
        self.stats = {
            "signals_received": 0,
            "signals_processed": 0,
            "signals_rejected": 0,
            "orders_sent": 0,
            "errors": 0,
        }

    def _init_db_pool(self):
        """Initialise le pool de connexions à la base de données."""
        try:
            self.db_pool = DBConnectionPool.get_instance()
            logger.info("Pool de connexions DB initialisé")
        except Exception:
            logger.exception("Erreur initialisation pool DB")
            self.db_pool = None

    def _init_symbols(self):
        """Initialise les symboles dans Redis depuis .env"""
        try:
            from shared.src.config import SYMBOLS

            # Vérifier si déjà configurés
            existing_symbols = self.redis_client.get("trading:symbols")

            # Si existants, vérifier s'ils correspondent aux symboles du .env
            if existing_symbols:
                if isinstance(existing_symbols, str):
                    existing = json.loads(existing_symbols)
                else:
                    existing = existing_symbols

                # Comparer avec les symboles du .env
                if set(existing) != set(SYMBOLS):
                    # Mettre à jour avec les nouveaux symboles
                    self.redis_client.set(
                        "trading:symbols", json.dumps(SYMBOLS))
                    logger.info(
                        f"Symboles mis à jour dans Redis: {len(existing)} → {len(SYMBOLS)} symboles"
                    )
                else:
                    logger.info(
                        f"Symboles existants dans Redis: {len(existing)} symboles (à jour)"
                    )
            else:
                # Initialiser depuis .env (SYMBOLS est déjà une liste)
                self.redis_client.set("trading:symbols", json.dumps(SYMBOLS))
                logger.info(
                    f"Symboles initialisés dans Redis: {len(SYMBOLS)} symboles")

        except Exception:
            logger.exception("Erreur initialisation symboles")
            # Fallback sur symboles par défaut
            default_symbols = ["BTCUSDC", "ETHUSDC"]
            self.redis_client.set(
                "trading:symbols",
                json.dumps(default_symbols))
            logger.info(f"Symboles par défaut configurés: {default_symbols}")

    def _calculate_unified_signal_strength(
        self, signal: StrategySignal
    ) -> tuple[float, int, float]:
        """
        Calcule la force du signal de manière unifiée.

        Args:
            signal: Signal à analyser

        Returns:
            tuple[force, strategy_count, avg_confidence]: Force calculée, nombre de stratégies, confiance moyenne
        """
        try:
            # Méthode 1 (prioritaire) : Depuis metadata (consensus
            # multi-stratégies)
            if signal.metadata:
                consensus_strength = signal.metadata.get(
                    "consensus_strength", 0)
                strategies_count = signal.metadata.get(
                    "strategies_count", signal.metadata.get(
                        "strategy_count", 1))
                avg_confidence = signal.metadata.get(
                    "avg_confidence", signal.metadata.get("confidence", 0.5)
                )

                if consensus_strength > 0 and strategies_count > 1:
                    # Formule améliorée : donner plus de poids aux stratégies multiples
                    # Force = consensus x sqrt(strategies) x confidence
                    # sqrt(strategies) pour éviter explosion linéaire, mais récompenser diversité
                    force = (consensus_strength *
                             (strategies_count**0.5) * avg_confidence)
                    logger.debug(
                        f"Force consensus: {consensus_strength} x sqrt{strategies_count} x {avg_confidence:.2f} = {force:.2f}"
                    )
                    return force, strategies_count, avg_confidence

            # Méthode 2 : Signal unique avec confidence
            if (
                hasattr(signal, "confidence")
                and signal.confidence
                and signal.confidence >= 50
            ):
                # Convertir confidence (0-100) en force (0-3)
                force = (signal.confidence / 100) * \
                    2.0  # Max 2.0 pour signal unique
                return force, 1, signal.confidence / 100

            # Méthode 3 : Enum strength
            if hasattr(signal, "strength") and signal.strength is not None:
                strength_map: dict[SignalStrength, float] = {
                    SignalStrength.VERY_STRONG: 2.5,
                    SignalStrength.STRONG: 2.0,
                    SignalStrength.MODERATE: 1.5,
                    SignalStrength.WEAK: 1.0,
                }
                force = strength_map.get(signal.strength, 1.0)
                return force, 1, 0.7  # Confiance par défaut pour enum

            else:
                # Fallback : signal basique
                return 1.0, 1, 0.5

        except Exception:
            logger.exception("Erreur calcul force signal")
            return 1.0, 1, 0.5

    def _categorize_signal_strength(self, force: float) -> str:
        """
        Catégorise la force du signal pour l'allocation.

        Args:
            force: Force calculée

        Returns:
            Catégorie: VERY_STRONG, STRONG, MODERATE, WEAK
        """
        thresholds = self.signal_strength_config["categorization"]

        if force >= thresholds["very_strong_threshold"]:
            return "VERY_STRONG"
        if force >= thresholds["strong_threshold"]:
            return "STRONG"
        if force >= thresholds["moderate_threshold"]:
            return "MODERATE"
        return "WEAK"

    def _check_consensus_sell_override(
        self, signal: StrategySignal, entry_price: float
    ) -> tuple[bool, str]:
        """
        Vérifie si un consensus SELL fort doit bypasser le trailing stop.

        Args:
            signal: Signal de vente
            entry_price: Prix d'entrée de la position

        Returns:
            tuple[should_force_sell, reason]: True si vente forcée autorisée
        """
        try:
            # Calculer la force du signal
            signal_force, strategies_count, avg_confidence = (
                self._calculate_unified_signal_strength(signal)
            )

            # Récupérer les seuils de configuration
            config = self.signal_strength_config["consensus_sell"]
            min_strategies = config["min_strategies"]
            min_strength = config["min_strength"]
            loss_multiplier = config["loss_multiplier"]

            # Vérifier si c'est un signal de consensus
            signal_type = signal.metadata.get(
                "type", "") if signal.metadata else ""

            # Calculer la perte actuelle
            current_loss_pct = (
                (signal.price - entry_price) / entry_price) * 100

            # Récupérer l'ATR pour seuil dynamique
            atr_pct = self.trailing_manager._get_atr_percentage(signal.symbol)
            if not atr_pct:
                atr_pct = 1.5  # Valeur par défaut si ATR indisponible

            loss_threshold = -loss_multiplier * atr_pct  # Seuil = -0.6xATR%

            # Conditions pour forcer la vente
            is_consensus = signal_type == "CONSENSUS"
            has_enough_strategies = strategies_count >= min_strategies
            has_enough_strength = signal_force >= min_strength
            has_significant_loss = current_loss_pct < loss_threshold

            if (
                is_consensus
                and has_enough_strategies
                and has_enough_strength
                and has_significant_loss
            ):
                reason = (
                    f"CONSENSUS_SELL_FORCED: {strategies_count} stratégies, "
                    f"force {signal_force:.1f}, perte {current_loss_pct:.2f}% < seuil {loss_threshold:.2f}%")
                return True, reason

            # Log des cas où consensus fort mais pas assez de perte
            if is_consensus and has_enough_strategies and has_enough_strength:
                logger.info(
                    f"📊 Consensus fort mais perte insuffisante {signal.symbol}: "
                    f"{current_loss_pct:.2f}% > {loss_threshold:.2f}% - trailing continue")

            else:
                return False, "Conditions consensus sell non remplies"

        except Exception as e:
            logger.exception("Erreur vérification consensus sell")
            return False, f"Erreur: {e}"

    def _mark_signal_as_processed(self, signal_id: int) -> bool:
        """
        Marque un signal comme traité en base de données.

        Args:
            signal_id: ID du signal à marquer

        Returns:
            True si le marquage a réussi, False sinon
        """
        if not self.db_pool:
            logger.warning("Pas de pool DB pour marquer le signal")
            return False

        try:
            # Utiliser le pool de connexions avec transaction auto
            from shared.src.db_pool import DBContextManager

            with DBContextManager(auto_transaction=True) as cursor:
                cursor.execute(
                    "UPDATE trading_signals SET processed = true WHERE id = %s",
                    (signal_id,),
                )

                if cursor.rowcount > 0:
                    logger.debug(f"Signal {signal_id} marqué comme traité")
                    return True
                logger.warning(f"Signal {signal_id} non trouvé pour marquage")
                return False

        except Exception:
            logger.exception("Erreur marquage signal {signal_id}")
            return False

    def process_signal(self, _channel: str, data: dict[str, Any]) -> None:
        """
        Traite un signal reçu via Redis.

        Args:
            channel: Canal Redis
            data: Données du signal
        """
        try:
            self.stats["signals_received"] += 1

            # Parser le signal
            try:
                if "side" in data and isinstance(data["side"], str):
                    data["side"] = OrderSide(data["side"])

                signal = StrategySignal(**data)
            except ValueError:
                logger.exception("❌ Erreur parsing signal")
                self.stats["signals_rejected"] += 1
                return
            except Exception:
                logger.exception("❌ Erreur création signal")
                self.stats["signals_rejected"] += 1
                return
            logger.info(
                f"📨 Signal reçu: {signal.strategy} {signal.side} {signal.symbol} @ {signal.price}"
            )
            logger.debug(f"🔍 Signal metadata: {signal.metadata}")
            if signal.metadata and "db_id" in signal.metadata:
                logger.info(
                    f"DB ID trouvé dans signal: {signal.metadata['db_id']}")
            else:
                logger.warning(
                    "Pas de db_id trouvé dans les métadonnées du signal")

            # CONSENSUS BUY OVERRIDE: Vérifier AVANT la faisabilité pour permettre le bypass
            # Cela permet d'ajouter à l'univers AVANT de vérifier si c'est
            # tradable
            if signal.side == OrderSide.BUY:
                signal_force, strategy_count, avg_confidence = (
                    self._calculate_unified_signal_strength(signal)
                )

                # Vérifier si on doit bypasser l'hystérésis pour un consensus
                # fort
                min_force = self.signal_strength_config["consensus_override"][
                    "min_force"
                ]
                min_strategies = self.signal_strength_config["consensus_override"][
                    "min_strategies"
                ]

                if signal_force >= min_force and strategy_count >= min_strategies:
                    logger.warning(
                        f"🚀 CONSENSUS BUY FORT détecté pour {signal.symbol}"
                    )
                    logger.warning(
                        f"   → {strategy_count} stratégies, force {signal_force:.2f}"
                    )
                    logger.warning(
                        "   → Ajout immédiat à l'univers tradable (bypass hystérésis)"
                    )

                    # Forcer l'ajout à l'univers tradable pour 45 minutes
                    self.universe_manager.force_pair_selection(
                        signal.symbol, duration_minutes=45
                    )

            # Vérifier la faisabilité (APRÈS le consensus override pour que
            # l'univers soit à jour)
            is_feasible, reason = self._check_feasibility(signal)

            if not is_feasible:
                logger.warning(f"❌ Signal rejeté: {reason}")
                self.stats["signals_rejected"] += 1

                # Marquer le signal comme traité même s'il est rejeté
                if signal.metadata and "db_id" in signal.metadata:
                    db_id = signal.metadata["db_id"]
                    self._mark_signal_as_processed(db_id)

                return

            # Calculer la quantité à trader (la force a déjà été calculée si
            # BUY)
            quantity = self._calculate_quantity(signal)
            if not quantity or quantity <= 0:
                logger.error("Impossible de calculer la quantité")
                self.stats["signals_rejected"] += 1

                # Marquer le signal comme traité même en cas d'erreur
                if signal.metadata and "db_id" in signal.metadata:
                    db_id = signal.metadata["db_id"]
                    self._mark_signal_as_processed(db_id)

                return

            # Vérifier l'efficacité du trade avec la quantité calculée
            is_efficient, efficiency_reason = self._check_trade_efficiency(
                signal, quantity
            )

            if not is_efficient:
                logger.warning(f"❌ Signal rejeté: {efficiency_reason}")
                self.stats["signals_rejected"] += 1

                # Marquer le signal comme traité même s'il est rejeté
                if signal.metadata and "db_id" in signal.metadata:
                    db_id = signal.metadata["db_id"]
                    self._mark_signal_as_processed(db_id)

                return

            # Préparer l'ordre pour le trader (MARKET pour exécution immédiate)
            side_value = (
                signal.side.value if hasattr(
                    signal.side,
                    "value") else str(
                    signal.side))
            order_data = {
                "symbol": signal.symbol,
                "side": side_value,
                "quantity": float(quantity),
                "price": None,  # Force ordre MARKET pour exécution immédiate
                "strategy": signal.strategy,
                "timestamp": int(time.time() * 1000),
                "metadata": signal.metadata or {},
            }

            # Ajouter les stops si disponibles
            if signal.metadata:
                if "stop_price" in signal.metadata:
                    order_data["stop_price"] = signal.metadata["stop_price"]
                if "trailing_delta" in signal.metadata:
                    order_data["trailing_delta"] = signal.metadata["trailing_delta"]

            # Envoyer l'ordre au trader
            logger.info(
                f"📤 Envoi ordre au trader: {order_data['side']} {quantity:.8f} {signal.symbol}"
            )
            order_id = self.service_client.create_order(order_data)

            if order_id:
                logger.info(f"✅ Ordre créé: {order_id}")
                self.stats["orders_sent"] += 1
                self.stats["signals_processed"] += 1

                # Marquer le signal comme traité en DB si on a le db_id
                if signal.metadata and "db_id" in signal.metadata:
                    db_id = signal.metadata["db_id"]
                    if self._mark_signal_as_processed(db_id):
                        logger.debug(
                            f"Signal {db_id} marqué comme traité en DB")
                    else:
                        logger.warning(
                            f"Impossible de marquer le signal {db_id} comme traité"
                        )
                else:
                    logger.warning(
                        "Pas de db_id dans les métadonnées du signal")
            else:
                logger.error("❌ Échec création ordre")
                self.stats["errors"] += 1

                # Marquer le signal comme traité même en cas d'échec
                if signal.metadata and "db_id" in signal.metadata:
                    db_id = signal.metadata["db_id"]
                    self._mark_signal_as_processed(db_id)

        except Exception:
            logger.exception("❌ Erreur traitement signal")
            self.stats["errors"] += 1

    def _check_feasibility(self, signal: StrategySignal) -> tuple[bool, str]:
        """
        Vérifie si un trade est faisable.

        Args:
            signal: Signal à vérifier

        Returns:
            (is_feasible, reason)
        """
        try:
            # Récupérer les balances
            balances = self.service_client.get_all_balances()
            if not balances:
                return False, "Impossible de récupérer les balances"

            # Extraire l'asset de base (quote toujours USDC)
            base_asset = self._get_base_asset(signal.symbol)

            if signal.side == OrderSide.BUY:
                # NOUVEAU: Vérifier si la paire fait partie de l'univers
                # sélectionné
                if not self.universe_manager.is_pair_tradable(signal.symbol):
                    return (
                        False, f"{signal.symbol} n'est pas dans l'univers tradable actuellement", )

                # Pour un BUY, on a besoin d'USDC
                if isinstance(balances, dict):
                    usdc_balance = balances.get("USDC", {}).get("free", 0)
                else:
                    usdc_balance = next(
                        (
                            b.get("free", 0)
                            for b in balances
                            if b.get("asset") == "USDC"
                        ),
                        0,
                    )

                if usdc_balance < self.min_absolute_trade_usdc:
                    return (
                        False,
                        f"Balance USDC insuffisante: {usdc_balance:.2f} < {self.min_absolute_trade_usdc} (minimum augmenté)",
                    )

                # Vérifier s'il y a déjà un cycle actif pour ce symbole
                active_cycle = self._check_active_cycle(signal.symbol)
                if active_cycle:
                    return (
                        False, f"Cycle déjà actif pour {signal.symbol}: {active_cycle}", )

            else:  # SELL
                # FILTRE TRAILING SELL: Vérifier si on doit vendre maintenant
                # (AVANT balance)
                active_positions = self.service_client.get_active_cycles(
                    signal.symbol)
                if active_positions:
                    position = active_positions[0]
                    entry_price = float(position.get("entry_price", 0))
                    entry_time = position.get("timestamp")
                    position_id = str(
                        position.get("id", f"pos_{signal.symbol}")
                    )  # ID unique de position

                    # Vérifier si consensus SELL fort doit bypasser le trailing
                    force_sell, sell_reason = self._check_consensus_sell_override(
                        signal, entry_price)

                    if not force_sell:
                        should_sell, trailing_reason = (
                            self.trailing_manager.check_trailing_sell(
                                symbol=signal.symbol,
                                current_price=signal.price,
                                entry_price=entry_price,
                                entry_time=entry_time,
                                position_id=position_id,
                            )
                        )
                        if not should_sell:
                            logger.info(
                                f"📝 SELL refusé pour {signal.symbol} - Raison: {trailing_reason}"
                            )
                            return False, trailing_reason
                        logger.info(
                            f"✅ SELL autorisé par trailing pour {signal.symbol} - Raison: {trailing_reason}"
                        )
                    else:
                        logger.warning(
                            f"🔥 {sell_reason}"
                        )  # sell_reason contient déjà le détail
                else:
                    # Pas de position active, autoriser le SELL
                    logger.info(
                        f"✅ Pas de position active pour {signal.symbol}, SELL autorisé (NO_POSITION)"
                    )

                # Pour un SELL, on a besoin de la crypto
                if isinstance(balances, dict):
                    crypto_balance = balances.get(
                        base_asset, {}).get("free", 0)
                else:
                    crypto_balance = next(
                        (
                            b.get("free", 0)
                            for b in balances
                            if b.get("asset") == base_asset
                        ),
                        0,
                    )

                if crypto_balance <= 0:
                    return False, f"Pas de {base_asset} à vendre"

                # Vérifier la valeur en USDC
                value_usdc = crypto_balance * signal.price
                if value_usdc < self.min_absolute_trade_usdc:
                    return (
                        False,
                        f"Valeur position trop faible: {value_usdc:.2f} USDC < {self.min_absolute_trade_usdc} USDC (minimum augmenté)",
                    )

                else:
                    return True, "OK"

        except Exception as e:
            logger.exception("Erreur vérification faisabilité")
            return False, f"Erreur: {e!s}"

    def _check_trade_efficiency(
        self, signal: StrategySignal, quantity: float
    ) -> tuple[bool, str]:
        """
        Vérifications basiques pour l'exécution du trade.
        Le Coordinator EXÉCUTE, il ne décide pas de la stratégie.

        Args:
            signal: Signal à analyser
            quantity: Quantité pré-calculée à trader

        Returns:
            (is_efficient, reason)
        """
        try:
            # Valeur totale du trade
            trade_value = quantity * signal.price

            # Filtre 1: Valeur minimum du trade (simple)
            if trade_value < self.min_absolute_trade_usdc:
                return (
                    False,
                    f"Trade trop petit: {trade_value:.2f} USDC < {self.min_absolute_trade_usdc:.2f} USDC (minimum augmenté pour éviter micro-positions)",
                )

            # Filtre 2: Ratio frais/valeur acceptable
            estimated_fees = trade_value * self.fee_rate * 2  # Aller-retour
            fee_percentage = (estimated_fees / trade_value) * 100

            if fee_percentage > 1.0:  # Si frais > 1% de la valeur du trade
                return (
                    False,
                    f"Frais trop élevés: {fee_percentage:.2f}% de la valeur du trade",
                )

            else:
                logger.info(
                    f"✅ Trade valide: {trade_value:.2f} USDC, frais {fee_percentage:.2f}%"
                )
                return True, "Trade valide"

        except Exception:
            logger.exception("❌ Erreur vérification trade")
            return True, "Erreur technique - trade autorisé par défaut"

    def _calculate_quantity(self, signal: StrategySignal) -> float | None:
        """
        Calcule la quantité à trader.

        Args:
            signal: Signal de trading

        Returns:
            Quantité à trader ou None
        """
        try:
            balances = self.service_client.get_all_balances()
            if not balances:
                return None

            base_asset = self._get_base_asset(signal.symbol)

            if signal.side == OrderSide.BUY:
                # Pour un BUY, calculer combien on peut acheter (allocation
                # dynamique)
                if isinstance(balances, dict):
                    usdc_balance = balances.get("USDC", {}).get("free", 0)
                    sum(
                        b.get("value_usdc", 0) for b in balances.values()
                    )
                else:
                    usdc_balance = next(
                        (
                            b.get("free", 0)
                            for b in balances
                            if b.get("asset") == "USDC"
                        ),
                        0,
                    )
                    sum(b.get("value_usdc", 0) for b in balances)

                # ALLOCATION USDC : Utiliser le calcul de force unifié
                # Réutiliser le calcul déjà fait si disponible dans metadata
                if signal.metadata and "calculated_force" in signal.metadata:
                    # Force déjà calculée lors du consensus override
                    signal_force = signal.metadata["calculated_force"]
                    strategies_count = signal.metadata.get(
                        "strategies_count", 1)
                    avg_confidence = signal.metadata.get(
                        "avg_confidence", signal.confidence
                    )
                else:
                    # Calculer maintenant si pas déjà fait
                    signal_force, strategies_count, avg_confidence = (
                        self._calculate_unified_signal_strength(signal)
                    )

                strength_category = self._categorize_signal_strength(
                    signal_force)

                logger.info(
                    f"💪 Force calculée {signal.symbol}: {signal_force:.2f} → {strength_category} "
                    f"(strategies:{strategies_count}, conf:{avg_confidence:.2f})")

                # Allocation selon la force calculée
                if strength_category == "VERY_STRONG":
                    allocation_percent = self.max_allocation_usdc_percent  # 15% USDC
                elif strength_category == "STRONG":
                    allocation_percent = self.strong_allocation_usdc_percent  # 12% USDC
                elif strength_category == "MODERATE":
                    allocation_percent = self.base_allocation_usdc_percent  # 10% USDC
                else:  # WEAK
                    allocation_percent = self.weak_allocation_usdc_percent  # 7% USDC

                # Calculer le montant basé sur l'USDC disponible
                trade_amount = usdc_balance * (allocation_percent / 100)

                # Limiter par la marge de sécurité USDC
                max_usdc_usable = (
                    usdc_balance * self.usdc_safety_margin
                )  # 98% de l'USDC
                trade_amount = min(trade_amount, max_usdc_usable)

                # Mais toujours respecter le minimum absolu Binance
                trade_amount = max(self.min_absolute_trade_usdc, trade_amount)

                # NOUVEAU: Si USDC insuffisant, essayer de libérer des fonds en
                # vendant la pire position
                if trade_amount > usdc_balance:
                    logger.warning(
                        f"💰 USDC insuffisant pour {signal.symbol}: besoin {trade_amount:.2f}, disponible {usdc_balance:.2f}"
                    )
                    freed_usdc = self._free_usdc_by_selling_worst_position(
                        trade_amount - usdc_balance
                    )

                    if freed_usdc > 0:
                        # Recalculer l'USDC disponible après vente
                        updated_balances = self.service_client.get_all_balances()
                        if updated_balances:
                            if isinstance(updated_balances, dict):
                                usdc_balance = updated_balances.get(
                                    "USDC", {}).get("free", 0)
                            else:
                                usdc_balance = next(
                                    (
                                        b.get("free", 0)
                                        for b in updated_balances
                                        if b.get("asset") == "USDC"
                                    ),
                                    0,
                                )

                            logger.info(
                                f"✅ USDC libéré: {freed_usdc:.2f}, nouveau solde: {usdc_balance:.2f}"
                            )

                        # Recalculer le montant de trade avec le nouvel USDC
                        trade_amount = min(
                            trade_amount, usdc_balance * self.usdc_safety_margin)
                    else:
                        logger.warning(
                            f"❌ Impossible de libérer assez d'USDC pour {signal.symbol}"
                        )
                        # Continuer avec l'USDC disponible
                        trade_amount = usdc_balance * self.usdc_safety_margin

                # Log pour debug positions augmentées
                logger.info(
                    f"💰 {signal.symbol} - USDC dispo: {usdc_balance:.0f}€, "
                    f"allocation: {allocation_percent:.0f}% = {trade_amount:.0f}€ "
                    f"(force: {strength_category}) [POSITIONS x1.8 AUGMENTÉES]")

                # Vérifier que le prix est valide avant division
                if not signal.price or signal.price <= 0:
                    logger.error(
                        f"Prix invalide pour {signal.symbol}: {signal.price}")
                    return None

                # Convertir en quantité
                quantity = trade_amount / signal.price

            # Pour un SELL, vendre toute la position
            else:  # SELL
                if isinstance(balances, dict):
                    quantity = balances.get(base_asset, {}).get("free", 0)
                else:
                    quantity = next(
                        (
                            b.get("free", 0)
                            for b in balances
                            if b.get("asset") == base_asset
                        ),
                        0,
                    )

            return quantity

        except Exception:
            logger.exception("Erreur calcul quantité")
            return None

    def _get_base_asset(self, symbol: str) -> str:
        """Extrait l'asset de base du symbole."""
        if symbol.endswith(("USDC", "USDT")):
            return symbol[:-4]
        if symbol.endswith(("BTC", "ETH")):
            return symbol[:-3]
        return symbol[:-4]  # Par défaut

    def _get_quote_asset(self, symbol: str) -> str:
        """Extrait l'asset de quote du symbole."""
        if symbol.endswith("USDC"):
            return "USDC"
        if symbol.endswith("USDT"):
            return "USDT"
        if symbol.endswith("BTC"):
            return "BTC"
        if symbol.endswith("ETH"):
            return "ETH"
        return "USDC"  # Par défaut

    def _check_active_cycle(self, symbol: str) -> str | None:
        """
        Vérifie s'il y a un cycle actif pour ce symbole via le portfolio service.

        Args:
            symbol: Symbole à vérifier (ex: 'BTCUSDC')

        Returns:
            ID du cycle actif si trouvé, None sinon
        """
        try:
            # Récupérer les positions actives depuis le portfolio service
            active_positions = self.service_client.get_active_cycles(symbol)

            if active_positions:
                # Retourner l'ID de la première position active trouvée
                position = active_positions[0]
                return position.get("id", f"position_{symbol}")

            else:
                return None

        except Exception as e:
            logger.warning(
                f"Erreur vérification cycle actif pour {symbol}: {e!s}")
            return None

    def get_stats(self) -> dict[str, Any]:
        """Retourne les statistiques du coordinator."""
        return self.stats.copy()

    def start_stop_loss_monitoring(self) -> None:
        """Démarre le thread de monitoring stop-loss."""
        if not self.stop_loss_thread or not self.stop_loss_thread.is_alive():
            self.stop_loss_thread = threading.Thread(
                target=self._stop_loss_monitor_loop, daemon=True, name="StopLossMonitor")
            self.stop_loss_thread.start()
            logger.info("🛡️ Monitoring stop-loss démarré")

    def stop_stop_loss_monitoring(self) -> None:
        """Arrête le monitoring stop-loss."""
        self.stop_loss_active = False
        if self.stop_loss_thread:
            self.stop_loss_thread.join(timeout=10)
        logger.info("🛑 Monitoring stop-loss arrêté")

    def start_universe_update(self) -> None:
        """Démarre le thread de mise à jour de l'univers."""
        if (
            not self.universe_update_thread
            or not self.universe_update_thread.is_alive()
        ):
            self.universe_update_thread = threading.Thread(
                target=self._universe_update_loop, daemon=True, name="UniverseUpdate")
            self.universe_update_thread.start()
            logger.info("🌍 Mise à jour de l'univers démarrée")

    def stop_universe_update(self) -> None:
        """Arrête la mise à jour de l'univers."""
        self.universe_update_active = False
        if self.universe_update_thread:
            self.universe_update_thread.join(timeout=10)
        logger.info("🛑 Mise à jour de l'univers arrêtée")

    def _universe_update_loop(self) -> None:
        """Boucle principale de mise à jour de l'univers."""
        logger.info("🔍 Boucle de mise à jour de l'univers active")

        # Mise à jour initiale immédiate
        self._update_universe()

        while self.universe_update_active:
            try:
                time.sleep(self.universe_update_interval)
                self._update_universe()
            except Exception:
                logger.exception("❌ Erreur dans mise à jour univers")
                time.sleep(self.universe_update_interval)

    def _update_universe(self) -> None:
        """Met à jour l'univers tradable."""
        try:
            selected, scores = self.universe_manager.update_universe()

            # Log des paires sélectionnées
            logger.info(
                f"🌍 Univers mis à jour: {len(selected)} paires sélectionnées")
            logger.info(f"📊 Core: {self.universe_manager.core_pairs}")
            satellites = selected - self.universe_manager.core_pairs
            if satellites:
                logger.info(f"🛰️ Satellites: {satellites}")

            # Log des top scores (prendre tous les scores, pas seulement > 0)
            try:
                top_scores = sorted(
                    [(s, score.score) for s, score in scores.items()],
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]

                if top_scores:
                    logger.info("📈 Top 10 scores:")
                    for symbol, score in top_scores:
                        status = "✅" if symbol in selected else "❌"
                        logger.info(f"  {status} {symbol}: {score:.2f}")

            except Exception:
                logger.exception("❌ Erreur affichage scores")
                logger.info(
                    f"Debug - selected: {selected}, scores count: {len(scores)}"
                )

        except Exception:
            logger.exception("❌ Erreur mise à jour univers")

    def _stop_loss_monitor_loop(self) -> None:
        """Boucle principale du monitoring stop-loss."""
        logger.info("🔍 Boucle de monitoring stop-loss active")

        while self.stop_loss_active:
            try:
                self._check_all_positions_stop_loss()
                time.sleep(self.price_check_interval)
            except Exception:
                logger.exception("❌ Erreur dans monitoring stop-loss")
                time.sleep(
                    self.price_check_interval * 2
                )  # Attendre plus longtemps en cas d'erreur

    def _check_all_positions_stop_loss(self) -> None:
        """Vérifie toutes les positions actives pour déclenchement stop-loss."""
        try:
            # Récupérer toutes les positions actives
            all_active_cycles = self.service_client.get_all_active_cycles()

            if not all_active_cycles:
                return

            logger.debug(
                f"🔍 Vérification stop-loss pour {len(all_active_cycles)} positions actives"
            )

            for cycle in all_active_cycles:
                try:
                    self._check_position_stop_loss(cycle)
                except Exception:
                    logger.exception(
                        f"❌ Erreur vérification stop-loss pour cycle {cycle.get('id', 'unknown')}: "
                    )

        except Exception:
            logger.exception("❌ Erreur récupération positions actives")

    def _check_position_stop_loss(self, cycle: dict[str, Any]) -> None:
        """
        Vérifie une position spécifique et déclenche un stop-loss si nécessaire.
        Met aussi à jour la référence trailing automatiquement.
        Utilise maintenant le système de stop-loss adaptatif intelligent.

        Args:
            cycle: Données du cycle de trading
        """
        try:
            symbol = cycle.get("symbol")
            entry_price = float(cycle.get("entry_price", 0))
            entry_time = cycle.get("timestamp")
            cycle_id = cycle.get("id")
            position_id = (
                str(cycle_id) if cycle_id else f"pos_{symbol}"
            )  # ID unique de position

            if not symbol or not entry_price:
                logger.warning(f"⚠️ Données cycle incomplètes: {cycle}")
                return

            # Convertir timestamp en epoch si nécessaire
            if isinstance(entry_time, str):
                from datetime import datetime

                entry_time_dt = datetime.fromisoformat(
                    entry_time.replace("Z", "+00:00")
                )
                entry_time_epoch = entry_time_dt.timestamp()
            else:
                entry_time_epoch = float(
                    entry_time) if entry_time else time.time()

            # Récupérer le prix actuel via TrailingSellManager
            current_price = self.trailing_manager.get_current_price(symbol)
            if not current_price:
                logger.warning(
                    f"Prix actuel indisponible pour {symbol} - skip monitoring"
                )
                return

            # Vérifier le hard risk en premier (forçage absolu)
            if self.universe_manager.check_hard_risk(symbol):
                logger.warning(
                    f"🚨 HARD RISK détecté pour {symbol} - vente forcée!")
                self._execute_emergency_sell(
                    symbol, current_price, position_id, "HARD_RISK"
                )
                return

            # Utiliser le TrailingSellManager pour vérifier si on doit vendre
            should_sell, sell_reason = self.trailing_manager.check_trailing_sell(
                symbol=symbol,
                current_price=current_price,
                entry_price=entry_price,
                entry_time=entry_time_epoch,
                position_id=position_id,
            )

            if should_sell:
                logger.warning(
                    f"🚨 AUTO-SELL DÉCLENCHÉ pour {symbol}: {sell_reason}")
                # Déclencher vente d'urgence
                self._execute_emergency_sell(
                    symbol, current_price, position_id, sell_reason
                )
                return

            # Mettre à jour le prix max si position gagnante
            if current_price > entry_price:
                self.trailing_manager.update_max_price_if_needed(
                    symbol, current_price, position_id
                )

        except Exception:
            logger.exception("❌ Erreur vérification position {cycle_id}")

    # _get_current_price SUPPRIMÉ - utiliser
    # trailing_manager.get_current_price() à la place

    def _execute_emergency_sell(
        self, symbol: str, current_price: float, cycle_id: str, reason: str
    ) -> None:
        """
        Exécute une vente d'urgence (stop-loss).

        Args:
            symbol: Symbole à vendre
            current_price: Prix actuel
            cycle_id: ID du cycle
            reason: Raison de la vente
        """
        try:
            logger.warning(f"🚨 VENTE D'URGENCE {symbol} - Raison: {reason}")

            # Récupérer la balance à vendre
            balances = self.service_client.get_all_balances()
            if not balances:
                logger.error(
                    f"❌ Impossible de récupérer les balances pour vente d'urgence {symbol}"
                )
                return

            base_asset = self._get_base_asset(symbol)

            if isinstance(balances, dict):
                quantity = balances.get(base_asset, {}).get("free", 0)
            else:
                quantity = next(
                    (
                        b.get("free", 0)
                        for b in balances
                        if b.get("asset") == base_asset
                    ),
                    0,
                )

            if quantity <= 0:
                logger.warning(
                    f"⚠️ Aucune quantité à vendre pour {symbol} (balance: {quantity})"
                )
                return

            # Créer ordre de vente d'urgence
            order_data = {
                "symbol": symbol,
                "side": "SELL",
                "quantity": float(quantity),
                "price": None,  # Ordre MARKET pour exécution immédiate
                "strategy": f"STOP_LOSS_AUTO_{reason}",
                "timestamp": int(time.time() * 1000),
                "metadata": {
                    "emergency_sell": True,
                    "stop_loss_trigger": True,
                    "cycle_id": cycle_id,
                    "trigger_price": current_price,
                    "reason": reason,
                },
            }

            # Envoyer l'ordre
            logger.warning(
                f"📤 Envoi ordre stop-loss: SELL {quantity:.8f} {symbol} @ MARKET"
            )
            order_id = self.service_client.create_order(order_data)

            if order_id:
                logger.warning(f"✅ Ordre stop-loss créé: {order_id}")
                self.stats["orders_sent"] += 1

                # Nettoyer les références Redis liées à ce symbole
                self.trailing_manager._clear_sell_reference(symbol)

            else:
                logger.error(f"❌ Échec création ordre stop-loss pour {symbol}")
                self.stats["errors"] += 1

        except Exception:
            logger.exception("❌ Erreur vente d'urgence {symbol}")
            self.stats["errors"] += 1

    def shutdown(self) -> None:
        """Arrête proprement le coordinator et nettoie les ressources"""
        logger.info("🛑 Arrêt du Coordinator en cours...")

        try:
            # Arrêter les threads
            self.stop_stop_loss_monitoring()
            self.stop_universe_update()

            # Libérer la connexion dédiée du trailing manager
            if hasattr(
                    self,
                    "trailing_db_connection") and self.trailing_db_connection:
                self.db_pool.release_connection(self.trailing_db_connection)
                logger.info("Connexion DB TrailingSellManager libérée")

            # Fermer le pool DB
            if self.db_pool:
                self.db_pool.close()
                logger.info("Pool DB fermé")

            # Nettoyer le client Redis (si nécessaire)
            if hasattr(self.redis_client, "close"):
                self.redis_client.close()
                logger.info("Client Redis fermé")

            logger.info("✅ Coordinator arrêté proprement")

        except Exception:
            logger.exception("❌ Erreur lors de l'arrêt du Coordinator")

    def _free_usdc_by_selling_worst_position(
            self, usdc_needed: float) -> float:
        """
        Libère de l'USDC en vendant la position avec la pire performance.

        Args:
            usdc_needed: Montant d'USDC à libérer

        Returns:
            Montant d'USDC effectivement libéré
        """
        try:
            # Récupérer toutes les positions actives
            active_cycles = self.service_client.get_all_active_cycles()
            if not active_cycles:
                logger.warning(
                    "Aucune position active à vendre pour libérer de l'USDC")
                return 0.0

            # Récupérer les balances actuelles
            balances = self.service_client.get_all_balances()
            if not balances:
                logger.error("Impossible de récupérer les balances")
                return 0.0

            # Analyser chaque position pour trouver la pire
            worst_position = None
            worst_performance = float(
                "inf"
            )  # On cherche la plus grosse perte (performance négative)
            all_positions_positive = True

            for cycle in active_cycles:
                try:
                    symbol = cycle.get("symbol")
                    entry_price = float(cycle.get("entry_price", 0))
                    if not symbol or not entry_price:
                        continue

                    # Récupérer le prix actuel
                    current_price = self.trailing_manager.get_current_price(
                        symbol)
                    if not current_price:
                        continue

                    # Calculer la performance en %
                    performance_pct = (
                        (current_price - entry_price) / entry_price
                    ) * 100

                    # Récupérer la valeur de la position
                    base_asset = self._get_base_asset(symbol)
                    if isinstance(balances, dict):
                        quantity = balances.get(base_asset, {}).get("free", 0)
                    else:
                        quantity = next(
                            (
                                b.get("free", 0)
                                for b in balances
                                if b.get("asset") == base_asset
                            ),
                            0,
                        )

                    position_value = quantity * current_price

                    # Ignorer les positions trop petites (< 5 USDC)
                    if position_value < 5.0:
                        continue

                    logger.info(
                        f"📊 Position {symbol}: {performance_pct:+.2f}% (valeur: {position_value:.2f} USDC)"
                    )

                    # Vérifier si cette position est négative
                    if performance_pct < 0:
                        all_positions_positive = False

                    # Sélectionner la position avec la pire performance
                    if performance_pct < worst_performance:
                        worst_performance = performance_pct
                        worst_position = {
                            "symbol": symbol,
                            "cycle": cycle,
                            "performance_pct": performance_pct,
                            "value_usdc": position_value,
                            "quantity": quantity,
                            "current_price": current_price,
                        }

                except Exception:
                    logger.exception("Erreur analyse position {cycle}")
                    continue

            # NOUVELLE LOGIQUE : Ne vendre que si il y a des positions en perte
            if all_positions_positive:
                logger.info(
                    "💚 Toutes les positions sont gagnantes - Pas de vente automatique"
                )
                logger.info(
                    f"💔 USDC insuffisant ({usdc_needed:.2f} requis) mais on garde les gains"
                )
                return 0.0

            # Vendre la pire position uniquement si elle est négative
            if (
                worst_position
                and worst_position["performance_pct"] < 0
                and worst_position["value_usdc"] >= usdc_needed * 0.8
            ):
                logger.warning(
                    f"🔥 VENTE AUTO de la position en PERTE: {worst_position['symbol']} "
                    f"({worst_position['performance_pct']:+.2f}%, {worst_position['value_usdc']:.2f} USDC)")

                # Créer un ordre de vente d'urgence
                order_data = {
                    "symbol": worst_position["symbol"],
                    "side": "SELL",
                    "quantity": float(worst_position["quantity"]),
                    "price": None,  # Ordre MARKET
                    "strategy": "AUTO_LIQUIDATION_WORST",
                    "timestamp": int(time.time() * 1000),
                    "metadata": {
                        "auto_liquidation": True,
                        "reason": "Libération USDC - Position en perte",
                        "performance_pct": worst_position["performance_pct"],
                        "usdc_needed": usdc_needed,
                    },
                }

                # Envoyer l'ordre
                order_id = self.service_client.create_order(order_data)
                if order_id:
                    logger.warning(f"✅ Ordre de liquidation créé: {order_id}")
                    self.stats["orders_sent"] += 1

                    # Attendre un peu pour que l'ordre s'exécute
                    time.sleep(2)

                    return worst_position["value_usdc"]
                logger.error(
                    f"❌ Échec ordre de liquidation pour {worst_position['symbol']}"
                )
                return 0.0
            if worst_position and worst_position["performance_pct"] >= 0:
                logger.info(
                    f"💚 Pire position {worst_position['symbol']} est gagnante "
                    f"({worst_position['performance_pct']:+.2f}%) - Pas de vente")
            elif worst_position:
                logger.warning(
                    f"💔 Position en perte {worst_position['symbol']} trop petite "
                    f"({worst_position['value_usdc']:.2f} USDC < {usdc_needed * 0.8:.2f} requis)")
            else:
                logger.warning(
                    "Aucune position éligible pour liquidation trouvée")
                return 0.0

        except Exception:
            logger.exception("Erreur libération USDC")
            return 0.0

    def get_universe_stats(self) -> dict[str, Any]:
        """Retourne les statistiques de l'univers tradable"""
        if self.universe_manager:
            return self.universe_manager.get_universe_stats()
        return {"status": "universe_manager_not_initialized"}
