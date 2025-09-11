"""
Coordinator simplifié pour RootTrading.
Rôle : Valider la faisabilité des signaux et les transmettre au trader.
"""
import logging
import time
import json
import asyncio
import threading
import os
from typing import Dict, Any, Optional
from shared.src.db_pool import DBConnectionPool

from shared.src.redis_client import RedisClient
from shared.src.enums import OrderSide, SignalStrength
from shared.src.schemas import StrategySignal
from service_client import ServiceClient
from trailing_sell_manager import TrailingSellManager
from universe_manager import UniverseManager

logger = logging.getLogger(__name__)


class Coordinator:
    """
    Coordinateur simplifié : reçoit les signaux, vérifie la faisabilité, transmet au trader.
    """
    
    def __init__(self, trader_api_url: str = "http://trader:5002", 
                 portfolio_api_url: str = "http://portfolio:8000"):
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
        self.stop_loss_thread: Optional[threading.Thread] = None
        
        # Thread de mise à jour de l'univers
        self.universe_update_active = True
        self.universe_update_thread: Optional[threading.Thread] = None
        self.universe_update_interval = 300  # Mise à jour toutes les 5 minutes
        
        # Configuration dynamique basée sur l'USDC disponible
        self.fee_rate = 0.001  # 0.1% de frais estimés par trade
        
        # Allocation basée USDC - optimisée pour 9 positions max (6 core + 3 satellites)
        self.base_allocation_usdc_percent = 10.0  # 10% de l'USDC disponible (9x10=90% max)
        self.strong_allocation_usdc_percent = 12.0 # 12% pour signaux forts
        self.max_allocation_usdc_percent = 15.0   # 15% maximum pour VERY_STRONG
        self.weak_allocation_usdc_percent = 7.0   # 7% pour signaux faibles
        self.usdc_safety_margin = 0.98            # Garde 2% d'USDC en sécurité
        self.min_absolute_trade_usdc = 1.0        # 1 USDC minimum (allow small position exits)
        
        # Initialiser le pool de connexions DB
        self._init_db_pool()
        
        # Initialiser les symboles dans Redis si nécessaires
        self._init_symbols()
        
        # Obtenir une connexion DB dédiée pour le trailing manager
        trailing_db_connection = None
        if self.db_pool:
            try:
                trailing_db_connection = self.db_pool.get_connection()
                logger.info("Connexion DB dédiée créée pour TrailingSellManager")
            except Exception as e:
                logger.error(f"Erreur création connexion DB pour TrailingSellManager: {e}")
        
        # Initialiser le gestionnaire de trailing sell avec une connexion directe
        self.trailing_db_connection = trailing_db_connection  # Garder la référence
        self.trailing_manager = TrailingSellManager(
            redis_client=self.redis_client,
            service_client=self.service_client,
            db_connection=trailing_db_connection  # Connexion directe pour trailing
        )
        
        # Initialiser le gestionnaire d'univers pour la sélection dynamique
        self.universe_manager = UniverseManager(
            redis_client=self.redis_client,  # Passer directement l'instance RedisClient
            db_pool=self.db_pool,  # Passer le pool DB au lieu d'une connexion directe
            config=None  # Utilise la config par défaut
        )
        
        # Configuration stop-loss - SUPPRIMÉE : toute la logique est dans TrailingSellManager
        # self.stop_loss_percent_* supprimés pour éviter duplication de code
        self.price_check_interval = 60  # Vérification des prix toutes les 60 secondes (aligné sur la fréquence des données)
        
        # Démarrer le monitoring stop-loss
        self.start_stop_loss_monitoring()
        
        # Démarrer la mise à jour de l'univers
        self.start_universe_update()
        
        logger.info(f"✅ Coordinator initialisé - Allocation USDC: {self.weak_allocation_usdc_percent}-{self.max_allocation_usdc_percent}% (positions plus grosses pour TP/SL)")
        
        # Stats
        self.stats = {
            "signals_received": 0,
            "signals_processed": 0,
            "signals_rejected": 0,
            "orders_sent": 0,
            "errors": 0
        }
        
    def _init_db_pool(self):
        """Initialise le pool de connexions à la base de données."""
        try:
            self.db_pool = DBConnectionPool.get_instance()
            logger.info("Pool de connexions DB initialisé")
        except Exception as e:
            logger.error(f"Erreur initialisation pool DB: {e}")
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
                    self.redis_client.set("trading:symbols", json.dumps(SYMBOLS))
                    logger.info(f"Symboles mis à jour dans Redis: {len(existing)} → {len(SYMBOLS)} symboles")
                else:
                    logger.info(f"Symboles existants dans Redis: {len(existing)} symboles (à jour)")
            else:
                # Initialiser depuis .env (SYMBOLS est déjà une liste)
                self.redis_client.set("trading:symbols", json.dumps(SYMBOLS))
                logger.info(f"Symboles initialisés dans Redis: {len(SYMBOLS)} symboles")
                
        except Exception as e:
            logger.error(f"Erreur initialisation symboles: {e}")
            # Fallback sur symboles par défaut
            default_symbols = ["BTCUSDC", "ETHUSDC"]
            self.redis_client.set("trading:symbols", json.dumps(default_symbols))
            logger.info(f"Symboles par défaut configurés: {default_symbols}")
            
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
                    (signal_id,)
                )
                
                if cursor.rowcount > 0:
                    logger.debug(f"Signal {signal_id} marqué comme traité")
                    return True
                else:
                    logger.warning(f"Signal {signal_id} non trouvé pour marquage")
                    return False
                    
        except Exception as e:
            logger.error(f"Erreur marquage signal {signal_id}: {e}")
            return False
    
    def process_signal(self, channel: str, data: Dict[str, Any]) -> None:
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
                if 'side' in data and isinstance(data['side'], str):
                    data['side'] = OrderSide(data['side'])
                    
                signal = StrategySignal(**data)
            except ValueError as e:
                logger.error(f"❌ Erreur parsing signal: {e}")
                self.stats["signals_rejected"] += 1
                return
            except Exception as e:
                logger.error(f"❌ Erreur création signal: {e}")
                self.stats["signals_rejected"] += 1
                return
            logger.info(f"📨 Signal reçu: {signal.strategy} {signal.side} {signal.symbol} @ {signal.price}")
            logger.debug(f"🔍 Signal metadata: {signal.metadata}")
            if signal.metadata and 'db_id' in signal.metadata:
                logger.info(f"DB ID trouvé dans signal: {signal.metadata['db_id']}")
            else:
                logger.warning("Pas de db_id trouvé dans les métadonnées du signal")
            
            # Vérifier la faisabilité
            is_feasible, reason = self._check_feasibility(signal)
            
            if not is_feasible:
                logger.warning(f"❌ Signal rejeté: {reason}")
                self.stats["signals_rejected"] += 1
                
                # Marquer le signal comme traité même s'il est rejeté
                if signal.metadata and 'db_id' in signal.metadata:
                    db_id = signal.metadata['db_id']
                    self._mark_signal_as_processed(db_id)
                
                return
            
            # CONSENSUS BUY OVERRIDE: Forcer l'ajout à l'univers si consensus très fort
            # La force est dans metadata['force'], strategy_count aussi
            signal_force = signal.metadata.get('force', 0) if signal.metadata else 0
            strategy_count = signal.metadata.get('strategy_count', 0) if signal.metadata else 0
            
            # Alternative: utiliser confidence si force n'est pas dans metadata
            if signal_force == 0 and signal.confidence and signal.confidence >= 80:
                signal_force = signal.confidence / 30  # Convertir confidence en force approximative
            
            # Alternative: utiliser strength (enum) si disponible
            if signal_force == 0 and hasattr(signal, 'strength') and signal.strength == SignalStrength.VERY_STRONG:
                signal_force = 3.0  # Considérer VERY_STRONG comme force 3.0
            
            if (signal.side == OrderSide.BUY and 
                signal_force >= 2.5 and 
                strategy_count >= 6):
                
                logger.warning(f"🚀 CONSENSUS BUY TRÈS FORT détecté pour {signal.symbol}")
                logger.warning(f"   → {strategy_count} stratégies, force {signal_force}")
                logger.warning(f"   → Ajout immédiat à l'univers tradable (bypass hystérésis)")
                
                # Forcer l'ajout à l'univers tradable
                self.universe_manager.force_pair_selection(signal.symbol, duration_minutes=60)
            
            # Vérifier l'efficacité du trade (logique simplifiée)
            is_efficient, efficiency_reason = self._check_trade_efficiency(signal)
            
            if not is_efficient:
                logger.warning(f"❌ Signal rejeté: {efficiency_reason}")
                self.stats["signals_rejected"] += 1
                
                # Marquer le signal comme traité même s'il est rejeté
                if signal.metadata and 'db_id' in signal.metadata:
                    db_id = signal.metadata['db_id']
                    self._mark_signal_as_processed(db_id)
                
                return
            
            # Calculer la quantité à trader
            quantity = self._calculate_quantity(signal)
            if not quantity or quantity <= 0:
                logger.error("Impossible de calculer la quantité")
                self.stats["signals_rejected"] += 1
                
                # Marquer le signal comme traité même en cas d'erreur
                if signal.metadata and 'db_id' in signal.metadata:
                    db_id = signal.metadata['db_id']
                    self._mark_signal_as_processed(db_id)
                
                return
            
            # Préparer l'ordre pour le trader (MARKET pour exécution immédiate)
            side_value = signal.side.value if hasattr(signal.side, 'value') else str(signal.side)
            order_data = {
                "symbol": signal.symbol,
                "side": side_value,
                "quantity": float(quantity),
                "price": None,  # Force ordre MARKET pour exécution immédiate
                "strategy": signal.strategy,
                "timestamp": int(time.time() * 1000),
                "metadata": signal.metadata or {}
            }
            
            # Ajouter les stops si disponibles
            if signal.metadata:
                if "stop_price" in signal.metadata:
                    order_data["stop_price"] = signal.metadata["stop_price"]
                if "trailing_delta" in signal.metadata:
                    order_data["trailing_delta"] = signal.metadata["trailing_delta"]
            
            # Envoyer l'ordre au trader
            logger.info(f"📤 Envoi ordre au trader: {order_data['side']} {quantity:.8f} {signal.symbol}")
            order_id = self.service_client.create_order(order_data)
            
            if order_id:
                logger.info(f"✅ Ordre créé: {order_id}")
                self.stats["orders_sent"] += 1
                self.stats["signals_processed"] += 1
                
                # Marquer le signal comme traité en DB si on a le db_id
                if signal.metadata and 'db_id' in signal.metadata:
                    db_id = signal.metadata['db_id']
                    if self._mark_signal_as_processed(db_id):
                        logger.debug(f"Signal {db_id} marqué comme traité en DB")
                    else:
                        logger.warning(f"Impossible de marquer le signal {db_id} comme traité")
                else:
                    logger.warning("Pas de db_id dans les métadonnées du signal")
            else:
                logger.error("❌ Échec création ordre")
                self.stats["errors"] += 1
                
                # Marquer le signal comme traité même en cas d'échec
                if signal.metadata and 'db_id' in signal.metadata:
                    db_id = signal.metadata['db_id']
                    self._mark_signal_as_processed(db_id)
                
        except Exception as e:
            logger.error(f"❌ Erreur traitement signal: {str(e)}")
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
                # NOUVEAU: Vérifier si la paire fait partie de l'univers sélectionné
                if not self.universe_manager.is_pair_tradable(signal.symbol):
                    return False, f"{signal.symbol} n'est pas dans l'univers tradable actuellement"
                
                # Pour un BUY, on a besoin d'USDC
                if isinstance(balances, dict):
                    usdc_balance = balances.get('USDC', {}).get('free', 0)
                else:
                    usdc_balance = next((b.get('free', 0) for b in balances if b.get('asset') == 'USDC'), 0)
                
                if usdc_balance < self.min_absolute_trade_usdc:
                    return False, f"Balance USDC insuffisante: {usdc_balance:.2f} < {self.min_absolute_trade_usdc}"
                
                # Vérifier s'il y a déjà un cycle actif pour ce symbole
                active_cycle = self._check_active_cycle(signal.symbol)
                if active_cycle:
                    return False, f"Cycle déjà actif pour {signal.symbol}: {active_cycle}"
                
            else:  # SELL
                # FILTRE TRAILING SELL: Vérifier si on doit vendre maintenant (AVANT balance)
                # Récupérer les infos de la position active pour le trailing sell
                active_positions = self.service_client.get_active_cycles(signal.symbol)
                if active_positions:
                    position = active_positions[0]
                    entry_price = float(position.get('entry_price', 0))
                    entry_time = position.get('timestamp')
                    
                    # EXCEPTION : Si signal de consensus fort ET position perdante significative
                    force_sell = False
                    if signal.metadata:
                        strategies_count = signal.metadata.get('strategies_count', 0)
                        consensus_strength = signal.metadata.get('consensus_strength', 0)
                        signal_type = signal.metadata.get('type', '')
                        
                        # Calculer la perte en %
                        current_loss_pct = ((signal.price - entry_price) / entry_price) * 100
                        
                        # Récupérer l'ATR pour le seuil dynamique
                        atr_pct = self.trailing_manager._get_atr_percentage(signal.symbol)
                        if not atr_pct:
                            atr_pct = 1.5  # Valeur par défaut si ATR indisponible
                        
                        loss_threshold = -0.6 * atr_pct  # Seuil de perte = -0.6×ATR%
                        
                        # Forcer la vente si consensus fort ET perte significative
                        if (signal_type == 'CONSENSUS' and 
                            strategies_count >= 5 and 
                            consensus_strength >= 2.0 and
                            current_loss_pct < loss_threshold):
                            logger.warning(f"⚠️ CONSENSUS FORT + PERTE SIGNIFICATIVE détectés pour {signal.symbol}")
                            logger.warning(f"   → {strategies_count} stratégies, force {consensus_strength:.1f}")
                            logger.warning(f"   → Perte: {current_loss_pct:.2f}% < seuil {loss_threshold:.2f}% (-0.6×ATR)")
                            logger.warning(f"   → SELL forcé autorisé")
                            force_sell = True
                        elif signal_type == 'CONSENSUS' and strategies_count >= 5:
                            logger.info(f"📊 Consensus fort mais position pas assez perdante: {current_loss_pct:.2f}% > {loss_threshold:.2f}%")
                            logger.info(f"   → SELL forcé refusé, laisse le trailing gérer")
                    
                    if not force_sell:
                        should_sell, sell_reason = self.trailing_manager.check_trailing_sell(
                            symbol=signal.symbol,
                            current_price=signal.price,
                            entry_price=entry_price,
                            entry_time=entry_time
                        )
                        if not should_sell:
                            # Journalisation détaillée de la raison du refus
                            logger.info(f"📝 SELL refusé pour {signal.symbol} - Raison: {sell_reason}")
                            return False, sell_reason
                        else:
                            # Journalisation si le trailing autorise la vente
                            logger.info(f"✅ SELL autorisé par trailing pour {signal.symbol} - Raison: {sell_reason}")
                    else:
                        # Journalisation du SELL forcé par consensus
                        logger.warning(f"🔥 SELL_FORCED_BY_CONSENSUS pour {signal.symbol}")
                else:
                    # Pas de position active, autoriser le SELL
                    logger.info(f"✅ Pas de position active pour {signal.symbol}, SELL autorisé (NO_POSITION)")
                
                # Pour un SELL, on a besoin de la crypto
                if isinstance(balances, dict):
                    crypto_balance = balances.get(base_asset, {}).get('free', 0)
                else:
                    crypto_balance = next((b.get('free', 0) for b in balances if b.get('asset') == base_asset), 0)
                
                if crypto_balance <= 0:
                    return False, f"Pas de {base_asset} à vendre"
                
                # Vérifier la valeur en USDC
                value_usdc = crypto_balance * signal.price
                if value_usdc < self.min_absolute_trade_usdc:
                    return False, f"Valeur trop faible: {value_usdc:.2f} USDC"
            
            return True, "OK"
            
        except Exception as e:
            logger.error(f"Erreur vérification faisabilité: {str(e)}")
            return False, f"Erreur: {str(e)}"
    
    def _check_trade_efficiency(self, signal: StrategySignal) -> tuple[bool, str]:
        """
        Vérifications basiques pour l'exécution du trade.
        Le Coordinator EXÉCUTE, il ne décide pas de la stratégie.
        
        Args:
            signal: Signal à analyser
            
        Returns:
            (is_efficient, reason)
        """
        try:
            # Calculer la quantité et valeur du trade
            quantity = self._calculate_quantity(signal)
            if not quantity:
                return False, "Impossible de calculer la quantité"
            
            # Valeur totale du trade
            trade_value = quantity * signal.price
            
            # Filtre 1: Valeur minimum du trade (simple)
            if trade_value < self.min_absolute_trade_usdc:
                return False, f"Trade trop petit: {trade_value:.2f} USDC < {self.min_absolute_trade_usdc:.2f} USDC (minimum Binance)"
            
            # Filtre 2: Ratio frais/valeur acceptable
            estimated_fees = trade_value * self.fee_rate * 2  # Aller-retour
            fee_percentage = (estimated_fees / trade_value) * 100
            
            if fee_percentage > 1.0:  # Si frais > 1% de la valeur du trade
                return False, f"Frais trop élevés: {fee_percentage:.2f}% de la valeur du trade"
            
            logger.info(f"✅ Trade valide: {trade_value:.2f} USDC, frais {fee_percentage:.2f}%")
            return True, "Trade valide"
            
        except Exception as e:
            logger.error(f"❌ Erreur vérification trade: {str(e)}")
            return True, "Erreur technique - trade autorisé par défaut"
    
    def _calculate_quantity(self, signal: StrategySignal) -> Optional[float]:
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
                # Pour un BUY, calculer combien on peut acheter (allocation dynamique)
                if isinstance(balances, dict):
                    usdc_balance = balances.get('USDC', {}).get('free', 0)
                    total_capital = sum(b.get('value_usdc', 0) for b in balances.values())
                else:
                    usdc_balance = next((b.get('free', 0) for b in balances if b.get('asset') == 'USDC'), 0)
                    total_capital = sum(b.get('value_usdc', 0) for b in balances)
                
                # ALLOCATION USDC MULTI-CRYPTO : Pourcentages adaptés pour 22 cryptos
                
                # Ajuster selon la force du signal (calculée depuis consensus_strength et strategies_count)
                strength_category = "MODERATE"  # Par défaut
                
                if signal.metadata:
                    logger.debug(f"🔍 Métadonnées {signal.symbol}: {signal.metadata}")
                    
                    # Calculer force basée sur consensus_strength et strategies_count
                    consensus_strength = signal.metadata.get('consensus_strength', 0)
                    strategies_count = signal.metadata.get('strategies_count', 1)
                    avg_confidence = signal.metadata.get('avg_confidence', 0.5)
                    
                    # Formule de force : consensus_strength * strategies_count * avg_confidence
                    force_score = consensus_strength * strategies_count * avg_confidence
                    
                    # Catégorisation basée sur le score de force
                    if force_score >= 20:
                        strength_category = "VERY_STRONG"
                    elif force_score >= 15:
                        strength_category = "STRONG" 
                    elif force_score >= 10:
                        strength_category = "MODERATE"
                    else:
                        strength_category = "WEAK"
                    
                    logger.info(f"💪 Force calculée {signal.symbol}: score={force_score:.1f} → {strength_category} "
                               f"(consensus:{consensus_strength}, strategies:{strategies_count}, conf:{avg_confidence:.2f})")
                
                # Allocation selon la force calculée
                if strength_category == "VERY_STRONG":
                    allocation_percent = self.max_allocation_usdc_percent     # 15% USDC
                elif strength_category == "STRONG":
                    allocation_percent = self.strong_allocation_usdc_percent  # 12% USDC
                elif strength_category == "MODERATE":
                    allocation_percent = self.base_allocation_usdc_percent    # 10% USDC
                else:  # WEAK
                    allocation_percent = self.weak_allocation_usdc_percent    # 7% USDC
                
                # Calculer le montant basé sur l'USDC disponible
                trade_amount = usdc_balance * (allocation_percent / 100)
                
                # Limiter par la marge de sécurité USDC
                max_usdc_usable = usdc_balance * self.usdc_safety_margin  # 98% de l'USDC
                trade_amount = min(trade_amount, max_usdc_usable)
                
                # Mais toujours respecter le minimum absolu Binance
                trade_amount = max(self.min_absolute_trade_usdc, trade_amount)
                
                # Log pour debug positions augmentées
                logger.info(f"💰 {signal.symbol} - USDC dispo: {usdc_balance:.0f}€, "
                           f"allocation: {allocation_percent:.0f}% = {trade_amount:.0f}€ "
                           f"(force: {strength_category}) [POSITIONS AUGMENTÉES]")
                
                # Convertir en quantité
                quantity = trade_amount / signal.price
                
            else:  # SELL
                # Pour un SELL, vendre toute la position
                if isinstance(balances, dict):
                    quantity = balances.get(base_asset, {}).get('free', 0)
                else:
                    quantity = next((b.get('free', 0) for b in balances if b.get('asset') == base_asset), 0)
            
            return quantity
            
        except Exception as e:
            logger.error(f"Erreur calcul quantité: {str(e)}")
            return None
    
    def _get_base_asset(self, symbol: str) -> str:
        """Extrait l'asset de base du symbole."""
        if symbol.endswith('USDC'):
            return symbol[:-4]
        elif symbol.endswith('USDT'):
            return symbol[:-4]
        elif symbol.endswith('BTC'):
            return symbol[:-3]
        elif symbol.endswith('ETH'):
            return symbol[:-3]
        else:
            return symbol[:-4]  # Par défaut
    
    def _get_quote_asset(self, symbol: str) -> str:
        """Extrait l'asset de quote du symbole."""
        if symbol.endswith('USDC'):
            return 'USDC'
        elif symbol.endswith('USDT'):
            return 'USDT'
        elif symbol.endswith('BTC'):
            return 'BTC'
        elif symbol.endswith('ETH'):
            return 'ETH'
        else:
            return 'USDC'  # Par défaut
    
    def _check_active_cycle(self, symbol: str) -> Optional[str]:
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
                return position.get('id', f"position_{symbol}")
            
            return None
            
        except Exception as e:
            logger.warning(f"Erreur vérification cycle actif pour {symbol}: {str(e)}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du coordinator."""
        return self.stats.copy()
    
    def start_stop_loss_monitoring(self) -> None:
        """Démarre le thread de monitoring stop-loss."""
        if not self.stop_loss_thread or not self.stop_loss_thread.is_alive():
            self.stop_loss_thread = threading.Thread(
                target=self._stop_loss_monitor_loop,
                daemon=True,
                name="StopLossMonitor"
            )
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
        if not self.universe_update_thread or not self.universe_update_thread.is_alive():
            self.universe_update_thread = threading.Thread(
                target=self._universe_update_loop,
                daemon=True,
                name="UniverseUpdate"
            )
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
            except Exception as e:
                logger.error(f"❌ Erreur dans mise à jour univers: {e}")
                time.sleep(self.universe_update_interval)
    
    def _update_universe(self) -> None:
        """Met à jour l'univers tradable."""
        try:
            selected, scores = self.universe_manager.update_universe()
            
            # Log des paires sélectionnées
            logger.info(f"🌍 Univers mis à jour: {len(selected)} paires sélectionnées")
            logger.info(f"📊 Core: {self.universe_manager.core_pairs}")
            satellites = selected - self.universe_manager.core_pairs
            if satellites:
                logger.info(f"🛰️ Satellites: {satellites}")
            
            # Log des top scores (prendre tous les scores, pas seulement > 0)
            try:
                top_scores = sorted(
                    [(s, score.score) for s, score in scores.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                
                if top_scores:
                    logger.info("📈 Top 10 scores:")
                    for symbol, score in top_scores:
                        status = "✅" if symbol in selected else "❌"
                        logger.info(f"  {status} {symbol}: {score:.2f}")
                        
            except Exception as e:
                logger.error(f"❌ Erreur affichage scores: {e}")
                logger.info(f"Debug - selected: {selected}, scores count: {len(scores)}")
            
        except Exception as e:
            logger.error(f"❌ Erreur mise à jour univers: {e}")
    
    def _stop_loss_monitor_loop(self) -> None:
        """Boucle principale du monitoring stop-loss."""
        logger.info("🔍 Boucle de monitoring stop-loss active")
        
        while self.stop_loss_active:
            try:
                self._check_all_positions_stop_loss()
                time.sleep(self.price_check_interval)
            except Exception as e:
                logger.error(f"❌ Erreur dans monitoring stop-loss: {e}")
                time.sleep(self.price_check_interval * 2)  # Attendre plus longtemps en cas d'erreur
    
    def _check_all_positions_stop_loss(self) -> None:
        """Vérifie toutes les positions actives pour déclenchement stop-loss."""
        try:
            # Récupérer toutes les positions actives
            all_active_cycles = self.service_client.get_all_active_cycles()
            
            if not all_active_cycles:
                return
            
            logger.debug(f"🔍 Vérification stop-loss pour {len(all_active_cycles)} positions actives")
            
            for cycle in all_active_cycles:
                try:
                    self._check_position_stop_loss(cycle)
                except Exception as e:
                    logger.error(f"❌ Erreur vérification stop-loss pour cycle {cycle.get('id', 'unknown')}: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Erreur récupération positions actives: {e}")
    
    def _check_position_stop_loss(self, cycle: Dict[str, Any]) -> None:
        """
        Vérifie une position spécifique et déclenche un stop-loss si nécessaire.
        Met aussi à jour la référence trailing automatiquement.
        Utilise maintenant le système de stop-loss adaptatif intelligent.
        
        Args:
            cycle: Données du cycle de trading
        """
        try:
            symbol = cycle.get('symbol')
            entry_price = float(cycle.get('entry_price', 0))
            entry_time = cycle.get('timestamp')
            cycle_id = cycle.get('id')
            
            if not symbol or not entry_price:
                logger.warning(f"⚠️ Données cycle incomplètes: {cycle}")
                return
            
            # Convertir timestamp en epoch si nécessaire
            if isinstance(entry_time, str):
                from datetime import datetime
                entry_time_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                entry_time_epoch = entry_time_dt.timestamp()
            else:
                entry_time_epoch = float(entry_time) if entry_time else time.time()
            
            # Récupérer le prix actuel via TrailingSellManager
            current_price = self.trailing_manager.get_current_price(symbol)
            if not current_price:
                logger.warning(f"Prix actuel indisponible pour {symbol} - skip monitoring")
                return
            
            # Vérifier le hard risk en premier (forçage absolu)
            if self.universe_manager.check_hard_risk(symbol):
                logger.warning(f"🚨 HARD RISK détecté pour {symbol} - vente forcée!")
                self._execute_emergency_sell(symbol, current_price, str(cycle_id) if cycle_id else "unknown", "HARD_RISK")
                return
            
            # Utiliser le TrailingSellManager pour vérifier si on doit vendre
            should_sell, sell_reason = self.trailing_manager.check_trailing_sell(
                symbol=symbol,
                current_price=current_price,
                entry_price=entry_price,
                entry_time=entry_time_epoch
            )
            
            if should_sell:
                logger.warning(f"🚨 AUTO-SELL DÉCLENCHÉ pour {symbol}: {sell_reason}")
                # Déclencher vente d'urgence
                self._execute_emergency_sell(symbol, current_price, str(cycle_id) if cycle_id else "unknown", sell_reason)
                return
            
            # Mettre à jour le prix max si position gagnante
            if current_price > entry_price:
                self.trailing_manager.update_max_price_if_needed(symbol, current_price)
            
                
        except Exception as e:
            logger.error(f"❌ Erreur vérification position {cycle_id}: {e}")
    
    # _get_current_price SUPPRIMÉ - utiliser trailing_manager.get_current_price() à la place
    
    def _execute_emergency_sell(self, symbol: str, current_price: float, cycle_id: str, reason: str) -> None:
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
                logger.error(f"❌ Impossible de récupérer les balances pour vente d'urgence {symbol}")
                return
            
            base_asset = self._get_base_asset(symbol)
            
            if isinstance(balances, dict):
                quantity = balances.get(base_asset, {}).get('free', 0)
            else:
                quantity = next((b.get('free', 0) for b in balances if b.get('asset') == base_asset), 0)
            
            if quantity <= 0:
                logger.warning(f"⚠️ Aucune quantité à vendre pour {symbol} (balance: {quantity})")
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
                    "reason": reason
                }
            }
            
            # Envoyer l'ordre
            logger.warning(f"📤 Envoi ordre stop-loss: SELL {quantity:.8f} {symbol} @ MARKET")
            order_id = self.service_client.create_order(order_data)
            
            if order_id:
                logger.warning(f"✅ Ordre stop-loss créé: {order_id}")
                self.stats["orders_sent"] += 1
                
                # Nettoyer les références Redis liées à ce symbole
                self.trailing_manager._clear_sell_reference(symbol)
                
            else:
                logger.error(f"❌ Échec création ordre stop-loss pour {symbol}")
                self.stats["errors"] += 1
                
        except Exception as e:
            logger.error(f"❌ Erreur vente d'urgence {symbol}: {e}")
            self.stats["errors"] += 1
    
    def shutdown(self) -> None:
        """Arrête proprement le coordinator et nettoie les ressources"""
        logger.info("🛑 Arrêt du Coordinator en cours...")
        
        try:
            # Arrêter les threads
            self.stop_stop_loss_monitoring()
            self.stop_universe_update()
            
            # Libérer la connexion dédiée du trailing manager
            if hasattr(self, 'trailing_db_connection') and self.trailing_db_connection:
                try:
                    self.db_pool.release_connection(self.trailing_db_connection)
                    logger.info("Connexion DB TrailingSellManager libérée")
                except Exception as e:
                    logger.error(f"Erreur libération connexion trailing: {e}")
            
            # Fermer le pool DB
            if self.db_pool:
                try:
                    self.db_pool.close()
                    logger.info("Pool DB fermé")
                except Exception as e:
                    logger.error(f"Erreur fermeture pool DB: {e}")
            
            # Nettoyer le client Redis (si nécessaire)
            if hasattr(self.redis_client, 'close'):
                try:
                    self.redis_client.close()
                    logger.info("Client Redis fermé")
                except Exception as e:
                    logger.error(f"Erreur fermeture Redis: {e}")
                    
            logger.info("✅ Coordinator arrêté proprement")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'arrêt du Coordinator: {e}")
    
    def get_universe_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de l'univers tradable"""
        if self.universe_manager:
            return self.universe_manager.get_universe_stats()
        else:
            return {"status": "universe_manager_not_initialized"}