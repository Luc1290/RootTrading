"""
Gestionnaire de trailing sell intelligent avec tracking du prix maximum historique.
Extrait du coordinator pour alléger le code et améliorer la maintenance.
"""

import json
import logging
import time
import traceback
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class TrailingSellManager:
    """
    Gère la logique de trailing sell avec tracking intelligent du prix maximum.
    """

    def __init__(self, redis_client, service_client, db_connection=None):
        """
        Initialise le gestionnaire de trailing sell.

        Args:
            redis_client: Client Redis pour stocker les références
            service_client: Client des services pour récupérer les positions
            db_connection: Connexion DB pour les données d'analyse (optionnel)
        """
        self.redis_client = redis_client
        self.service_client = service_client
        self.db_connection = db_connection

        # Configuration stop-loss adaptatif - SCALP OPTIMISÉ
        self.stop_loss_percent_base = 0.012  # 1.2% de base - protection rapide scalp
        self.stop_loss_percent_bullish = 0.014  # 1.4% en tendance haussière
        self.stop_loss_percent_strong_bullish = 0.016  # 1.6% en tendance très haussière

        # Note: Trailing sell utilise désormais _get_adaptive_trailing_margin() (ligne 931+)
        # Plus de seuils fixes, tout est adaptatif selon le gain atteint

        logger.info("✅ TrailingSellManager initialisé")

    def _get_redis_key(
        self, key_type: str, symbol: str, position_id: str | None = None
    ) -> str:
        """
        Génère une clé Redis unique avec namespace position_id.

        Args:
            key_type: Type de clé (cycle_max_price, sell_reference, max_tp_level)
            symbol: Symbole
            position_id: ID de position (optionnel, fallback sur symbol seul pour legacy)

        Returns:
            Clé Redis formatée
        """
        if position_id:
            return f"{key_type}:{symbol}:{position_id}"
        # Legacy fallback (pour compatibilité, mais logguer warning)
        logger.warning(
            f"⚠️ Clé Redis {key_type}:{symbol} sans position_id - risque collision!"
        )
        return f"{key_type}:{symbol}"

    def check_trailing_sell(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        entry_time: Any,
        position_id: str | None = None,
    ) -> tuple[bool, str]:
        """
        Vérifie si on doit exécuter le SELL selon la logique de trailing sell améliorée.
        Utilise le prix maximum historique du cycle pour une meilleure décision.

        Args:
            symbol: Symbole de trading
            current_price: Prix actuel
            entry_price: Prix d'entrée de la position
            entry_time: Timestamp d'entrée (ISO string ou epoch)
            position_id: ID unique de la position (évite collision multi-positions)

        Returns:
            (should_sell, reason)
        """
        logger.info(f"🔍 DEBUT check_trailing_sell pour {symbol} @ {current_price}")

        try:
            # Convertir timestamp ISO en epoch si nécessaire
            if isinstance(entry_time, str):
                entry_time_dt = datetime.fromisoformat(
                    entry_time.replace("Z", "+00:00")
                )
                entry_time_epoch = entry_time_dt.timestamp()
            else:
                entry_time_epoch = float(entry_time) if entry_time else time.time()

            precision = self._get_price_precision(current_price)
            logger.info(
                f"🔍 Prix entrée: {entry_price:.{precision}f}, Prix actuel: {current_price:.{precision}f}"
            )

            # Calculer la performance actuelle (positif = gain, négatif =
            # perte)
            performance_percent = (current_price - entry_price) / entry_price

            # Récupérer et afficher le prix max historique dès le début
            historical_max = self._get_and_update_max_price(
                symbol, current_price, entry_price, position_id
            )
            drop_from_max = (historical_max - current_price) / historical_max
            logger.info(
                f"📊 Prix max historique: {historical_max:.{precision}f}, Chute depuis max: {drop_from_max*100:.2f}%"
            )

            # === STOP-LOSS ADAPTATIF INTELLIGENT ===
            adaptive_threshold = self._calculate_adaptive_threshold(
                symbol, entry_price, entry_time_epoch
            )

            # Affichage clair selon performance
            if performance_percent >= 0:
                logger.info(
                    f"🧠 Stop-loss adaptatif pour {symbol}: {adaptive_threshold*100:.2f}% (gain actuel: +{performance_percent*100:.2f}%)"
                )
            else:
                logger.info(
                    f"🧠 Stop-loss adaptatif pour {symbol}: {adaptive_threshold*100:.2f}% (perte actuelle: {abs(performance_percent)*100:.2f}%)"
                )

            # Si perte dépasse le seuil adaptatif : SELL immédiat (perte =
            # performance négative)
            if (
                performance_percent < 0
                and abs(performance_percent) >= adaptive_threshold
            ):
                logger.info(
                    f"📉 Stop-loss adaptatif déclenché pour {symbol}: perte {abs(performance_percent)*100:.2f}% ≥ seuil {adaptive_threshold*100:.2f}%"
                )
                return (
                    True,
                    f"Stop-loss adaptatif déclenché (perte {abs(performance_percent)*100:.2f}% ≥ {adaptive_threshold*100:.2f}%)",
                )

            # Si position perdante mais dans la tolérance : garder
            if performance_percent < 0:
                logger.info(
                    f"🟡 Position perdante mais dans tolérance pour {symbol}: perte {abs(performance_percent)*100:.2f}% < seuil {adaptive_threshold*100:.2f}%"
                )
                return (
                    False,
                    f"Position perdante mais dans tolérance (perte {abs(performance_percent)*100:.2f}% < {adaptive_threshold*100:.2f}%)",
                )

            # === POSITION GAGNANTE : BREAKEVEN + TAKE PROFIT + TRAILING ===
            # Maintenant cohérent (positif = gain)
            gain_percent = performance_percent
            logger.info(
                f"🔍 Position gagnante détectée: +{gain_percent*100:.2f}%, vérification breakeven/take profit/trailing"
            )

            # === BREAKEVEN INTELLIGENT : Protection critique basée sur MAX historique ===
            # CORRECTION CRITIQUE : Breakeven basé sur le GAIN MAX atteint, pas le gain actuel
            # Sinon si max=+2% puis retour à +0.5%, le breakeven niveau 2 ne se
            # déclenche jamais
            max_gain_percent = (historical_max - entry_price) / entry_price
            fee_percent = 0.0008  # 8 bps (taker Binance standard)
            breakeven_threshold_1 = 0.012  # 1.2%
            breakeven_threshold_2 = 0.020  # 2.0%

            # Vérifier breakeven basé sur le MAX atteint, pas le gain actuel
            if max_gain_percent >= breakeven_threshold_2:
                # Max a dépassé +2% : sécuriser profit net minimum
                breakeven_price = entry_price * (1 + 0.002)  # Entry + 0.2%
                if current_price < breakeven_price:
                    logger.warning(
                        f"🛡️ BREAKEVEN NIVEAU 2 déclenché: prix {current_price:.{precision}f} < breakeven {breakeven_price:.{precision}f} (max atteint: +{max_gain_percent*100:.2f}%)"
                    )
                    self._cleanup_references(symbol, position_id)
                    return (
                        True,
                        f"Breakeven niveau 2 (entry+0.2%): {current_price:.{precision}f} < {breakeven_price:.{precision}f}",
                    )
                logger.debug(
                    f"🛡️ Breakeven niveau 2 armé @ {breakeven_price:.{precision}f} (max: +{max_gain_percent*100:.2f}%)"
                )

            elif max_gain_percent >= breakeven_threshold_1:
                # Max a dépassé +1.2% : protéger entry + frais (pas de perte)
                breakeven_price = entry_price * (1 + 2 * fee_percent)  # Entry + 2xfrais
                if current_price < breakeven_price:
                    logger.warning(
                        f"🛡️ BREAKEVEN NIVEAU 1 déclenché: prix {current_price:.{precision}f} < breakeven {breakeven_price:.{precision}f} (max atteint: +{max_gain_percent*100:.2f}%)"
                    )
                    self._cleanup_references(symbol, position_id)
                    return (
                        True,
                        f"Breakeven niveau 1 (entry+fees): {current_price:.{precision}f} < {breakeven_price:.{precision}f}",
                    )
                logger.debug(
                    f"🛡️ Breakeven niveau 1 armé @ {breakeven_price:.{precision}f} (max: +{max_gain_percent*100:.2f}%)"
                )

            # === TAKE PROFIT PROGRESSIF INTELLIGENT ===
            # Récupérer les données une seule fois
            atr_based_thresholds = self._get_atr_based_thresholds(symbol)
            market_regime = self._get_market_regime(symbol)
            atr_percent = self._get_atr_percentage(symbol) or 0.02

            # Mode PUMP RIDER : désactiver TP progressif sur vrais pumps (amplitude + vitesse)
            # Pump = fort gain rapide, pas juste une hausse normale lente
            time_elapsed_sec = time.time() - entry_time_epoch
            pump_rider_mode = (
                market_regime in ["TRENDING_BULL", "BREAKOUT_BULL"]
                # Volatilité très élevée (>3% - vrais pumps)
                and atr_percent > 0.030
                and gain_percent >= 0.05  # Déjà +5% minimum (pas 3%)
                # <10min (vitesse = confirmation pump)
                and time_elapsed_sec < 600
            )

            if pump_rider_mode:
                logger.info(
                    f"🚀 MODE PUMP RIDER activé pour {symbol}: TP progressif DÉSACTIVÉ (gain {gain_percent*100:.2f}%, ATR {atr_percent*100:.1f}%)"
                )
            elif gain_percent >= 0.025:  # Activer TP progressif à partir de +2.5%
                should_take_profit, tp_reason = self._check_progressive_take_profit(
                    symbol, gain_percent, position_id
                )
                if should_take_profit:
                    logger.info(f"💰 TAKE PROFIT PROGRESSIF DÉCLENCHÉ: {tp_reason}")
                    self._cleanup_references(symbol, position_id)
                    return True, tp_reason
            else:
                logger.debug(
                    f"TP progressif désactivé pour {symbol} (gain {gain_percent*100:.2f}% < 2.5%)"
                )

            # Utiliser les seuils déjà récupérés
            min_gain_for_trailing = atr_based_thresholds["activate_trailing_gain"]
            sell_margin = atr_based_thresholds["trailing_margin"]

            logger.info(
                f"📊 Seuils ATR pour {symbol}: activation={min_gain_for_trailing*100:.2f}%, marge={sell_margin*100:.2f}%"
            )

            # Vérifier si le gain minimum est atteint pour activer le trailing
            if gain_percent < min_gain_for_trailing:
                logger.info(
                    f"📊 Gain insuffisant pour trailing ({gain_percent*100:.2f}% < {min_gain_for_trailing*100:.2f}%), position continue"
                )
                return (
                    False,
                    f"Gain insuffisant pour activer le trailing ({gain_percent*100:.2f}% < {min_gain_for_trailing*100:.2f}%)",
                )

            # Prix max déjà récupéré plus haut
            logger.info(
                f"🎯 TRAILING LOGIC: Utilisation du prix MAX ({historical_max:.{precision}f}) pour décision, PAS le prix d'entrée"
            )

            # === TRAILING ADAPTATIF PROGRESSIF ===
            # Calculer le gain depuis le max (pour savoir à quel palier on est)
            max_gain_from_entry = (historical_max - entry_price) / entry_price

            # Marge adaptative : plus on a monté, plus on serre le trailing
            adaptive_margin = self._get_adaptive_trailing_margin(max_gain_from_entry)
            logger.info(
                f"🎯 TRAILING ADAPTATIF: max_gain={max_gain_from_entry*100:.2f}%, marge={adaptive_margin*100:.2f}%"
            )

            # Récupérer le prix SELL précédent
            previous_sell_price = self._get_previous_sell_price(symbol, position_id)
            logger.info(f"🔍 Prix SELL précédent: {previous_sell_price}")

            # === DÉCISION DE VENTE BASÉE SUR LE PRIX MAX ===

            # Si chute importante depuis le max (utiliser marge adaptative),
            # vendre immédiatement
            if drop_from_max >= adaptive_margin:
                logger.warning(
                    f"📉 CHUTE IMPORTANTE depuis max ({drop_from_max*100:.2f}%), SELL IMMÉDIAT!"
                )
                self._cleanup_references(symbol, position_id)
                return (
                    True,
                    f"Chute de {drop_from_max*100:.2f}% depuis max {historical_max:.{precision}f}, SELL immédiat",
                )

            if previous_sell_price is None:
                # Premier SELL : déjà protégé par adaptive_margin ligne 158
                # On stocke juste la référence pour le trailing classique
                self._update_sell_reference(symbol, current_price, position_id)
                logger.info(
                    f"🎯 Premier SELL @ {current_price:.{precision}f} stocké (max: {historical_max:.{precision}f}, marge adaptative: {adaptive_margin*100:.2f}%)"
                )
                return (
                    False,
                    f"Premier SELL stocké, max historique: {historical_max:.{precision}f}",
                )

            # === SELL SUIVANTS : TRAILING DÉJÀ PROTÉGÉ PAR ADAPTIVE_MARGIN LIGNE 158 ===
            # Cette section ne devrait être atteinte que si drop_from_max < adaptive_margin
            # Donc on fait juste le trailing classique sur le
            # previous_sell_price

            # Ensuite logique classique de trailing avec marge ATR
            sell_threshold = previous_sell_price * (1 - sell_margin)
            logger.info(
                f"🔍 Seuil de vente calculé: {sell_threshold:.{precision}f} (marge {sell_margin*100:.2f}%)"
            )

            if current_price > previous_sell_price:
                # Prix monte : mettre à jour référence
                self._update_sell_reference(symbol, current_price, position_id)
                logger.info(
                    f"📈 Prix monte: {current_price:.{precision}f} > {previous_sell_price:.{precision}f}, référence mise à jour"
                )
                return (
                    False,
                    f"Prix monte, référence mise à jour (max: {historical_max:.{precision}f})",
                )

            if current_price > sell_threshold:
                # Prix dans la marge de tolérance
                logger.info(
                    f"🟡 Prix stable: {current_price:.{precision}f} > seuil {sell_threshold:.{precision}f}"
                )
                return (
                    False,
                    f"Prix dans marge de tolérance (max: {historical_max:.{precision}f})",
                )

            # Prix baisse significativement : VENDRE
            logger.warning(
                f"📉 Baisse significative: {current_price:.{precision}f} ≤ {sell_threshold:.{precision}f}, SELL!"
            )
            self._cleanup_references(symbol, position_id)
            return (
                True,
                f"Baisse sous seuil trailing ({current_price:.{precision}f} ≤ {sell_threshold:.{precision}f})",
            )

        except Exception:
            logger.exception("❌ Erreur dans check_trailing_sell pour {symbol}")           

            logger.exception(f"❌ Traceback: {traceback.format_exc()}")
            # En cas d'erreur, autoriser le SELL par sécurité
            return True, "Erreur technique, SELL autorisé par défaut"

    def update_max_price_if_needed(
        self, symbol: str, current_price: float, position_id: str | None = None
    ) -> bool:
        """
        Met à jour le prix max si le prix actuel est plus élevé.
        Appelé par le monitoring automatique.

        Args:
            symbol: Symbole
            current_price: Prix actuel
            position_id: ID unique de position

        Returns:
            True si le prix max a été mis à jour
        """
        try:
            max_price_key = self._get_redis_key("cycle_max_price", symbol, position_id)
            max_price_data = self.redis_client.get(max_price_key)

            historical_max = None
            if max_price_data:
                try:
                    if isinstance(max_price_data, dict):
                        historical_max = float(max_price_data.get("price", 0))
                    elif isinstance(max_price_data, str | bytes):
                        if isinstance(max_price_data, bytes):
                            max_price_data = max_price_data.decode("utf-8")
                        max_price_dict = json.loads(max_price_data)
                        historical_max = float(max_price_dict.get("price", 0))
                except Exception:
                    logger.exception("Erreur parsing prix max")

            if historical_max is None or current_price > historical_max:
                self._update_cycle_max_price(symbol, current_price, position_id)
                precision = self._get_price_precision(current_price)
                logger.info(
                    f"📈 Nouveau max pour {symbol}: {current_price:.{precision}f}"
                )
                return True

            return False

        except Exception:
            logger.exception("Erreur mise à jour prix max {symbol}")
            return False

    def _get_and_update_max_price(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        position_id: str | None = None,
    ) -> float:
        """
        Récupère et met à jour le prix max historique du cycle.

        Args:
            symbol: Symbole
            current_price: Prix actuel
            entry_price: Prix d'entrée (utilisé si pas de max stocké)
            position_id: ID unique de position

        Returns:
            Prix maximum historique
        """
        # Récupérer le prix max historique
        max_price_key = self._get_redis_key("cycle_max_price", symbol, position_id)
        max_price_data = self.redis_client.get(max_price_key)
        historical_max = None

        if max_price_data:
            try:
                if isinstance(max_price_data, dict):
                    historical_max = float(max_price_data.get("price", 0))
                elif isinstance(max_price_data, str | bytes):
                    if isinstance(max_price_data, bytes):
                        max_price_data = max_price_data.decode("utf-8")
                    max_price_dict = json.loads(max_price_data)
                    historical_max = float(max_price_dict.get("price", 0))
            except Exception:
                logger.exception("Erreur récupération prix max")

        # Si pas de prix max, initialiser avec le prix d'entrée
        if historical_max is None:
            historical_max = entry_price
            self._update_cycle_max_price(symbol, entry_price, position_id)

        # Mettre à jour si le prix actuel est plus élevé
        if current_price > historical_max:
            historical_max = current_price
            self._update_cycle_max_price(symbol, current_price, position_id)
            precision = self._get_price_precision(current_price)
            logger.info(
                f"📊 Nouveau prix max pour {symbol}: {current_price:.{precision}f}"
            )

        return historical_max

    def _get_previous_sell_price(
        self, symbol: str, position_id: str | None = None
    ) -> float | None:
        """
        Récupère le prix du SELL précédent stocké en référence.

        Args:
            symbol: Symbole à vérifier
            position_id: ID unique de position

        Returns:
            Prix du SELL précédent ou None
        """
        try:
            ref_key = self._get_redis_key("sell_reference", symbol, position_id)
            price_data = self.redis_client.get(ref_key)

            if not price_data:
                return None

            logger.debug(
                f"🔍 Récupération sell reference {symbol}: type={type(price_data)}, data={price_data}"
            )

            # Gérer tous les cas possibles de retour Redis
            if isinstance(price_data, dict):
                if "price" in price_data:
                    return float(price_data["price"])
                logger.warning(f"Clé 'price' manquante dans dict Redis pour {symbol}")
                return None

            if isinstance(price_data, str | bytes):
                try:
                    if isinstance(price_data, bytes):
                        price_data = price_data.decode("utf-8")

                    parsed_data = json.loads(price_data)
                    if isinstance(parsed_data, dict) and "price" in parsed_data:
                        return float(parsed_data["price"])
                    logger.warning(f"Format JSON invalide pour {symbol}: {parsed_data}")
                    return None
                except json.JSONDecodeError:
                    logger.exception("Erreur JSON decode pour {symbol}")
                    return None

            else:
                logger.warning(
                    f"Type Redis inattendu pour {symbol}: {type(price_data)}"
                )
                return None

        except Exception:
            logger.exception("Erreur récupération sell reference pour {symbol}")
            return None

    def _update_sell_reference(
        self, symbol: str, price: float, position_id: str | None = None
    ) -> None:
        """
        Met à jour la référence de prix SELL pour un symbole.

        Args:
            symbol: Symbole
            price: Nouveau prix de référence
            position_id: ID unique de position
        """
        try:
            ref_key = self._get_redis_key("sell_reference", symbol, position_id)
            ref_data = {"price": price, "timestamp": int(time.time() * 1000)}
            # TTL de 7 jours (604800s) - sera refresh à chaque check
            self.redis_client.set(ref_key, json.dumps(ref_data), expiration=604800)
        except Exception:
            logger.exception("Erreur mise à jour sell reference pour {symbol}")

    def _clear_sell_reference(
        self, symbol: str, position_id: str | None = None
    ) -> None:
        """
        Supprime la référence de prix SELL pour un symbole.

        Args:
            symbol: Symbole
            position_id: ID unique de position
        """
        try:
            ref_key = self._get_redis_key("sell_reference", symbol, position_id)
            self.redis_client.delete(ref_key)
            logger.info(f"🧹 Référence SELL supprimée pour {symbol}")
        except Exception:
            logger.exception("Erreur suppression sell reference pour {symbol}")

    def _update_cycle_max_price(
        self, symbol: str, price: float, position_id: str | None = None
    ) -> None:
        """
        Met à jour le prix maximum historique d'un cycle.

        Args:
            symbol: Symbole
            price: Nouveau prix maximum
            position_id: ID unique de position
        """
        try:
            max_key = self._get_redis_key("cycle_max_price", symbol, position_id)
            max_data = {"price": price, "timestamp": int(time.time() * 1000)}
            # TTL de 7 jours (604800s) - sera refresh à chaque check
            self.redis_client.set(max_key, json.dumps(max_data), expiration=604800)
            logger.debug(f"📈 Prix max mis à jour pour {symbol}: {price}")
        except Exception:
            logger.exception("Erreur mise à jour prix max pour {symbol}")

    def _clear_cycle_max_price(
        self, symbol: str, position_id: str | None = None
    ) -> None:
        """
        Supprime le prix maximum historique d'un cycle.

        Args:
            symbol: Symbole
            position_id: ID unique de position
        """
        try:
            max_key = self._get_redis_key("cycle_max_price", symbol, position_id)
            self.redis_client.delete(max_key)
            logger.info(f"🧹 Prix max historique supprimé pour {symbol}")
        except Exception:
            logger.exception("Erreur suppression prix max pour {symbol}")

    # SUPPRIMÉ - fonction dupliquée, gardée seulement la version mise à jour
    # plus bas

    def _get_atr_based_thresholds(self, symbol: str) -> dict[str, float]:
        """
        Calcule les seuils adaptatifs OPTIMISÉS pour maximiser les gains en haussier.

        Args:
            symbol: Symbole à analyser

        Returns:
            Dict avec trailing_margin, activate_trailing_gain, adaptive_sl
        """
        try:
            # Récupérer ATR et données de marché
            atr_percent = self._get_atr_percentage(symbol)
            market_regime = self._get_market_regime(symbol)

            if atr_percent is None:
                logger.debug(f"Pas d'ATR pour {symbol}, seuils par défaut scalp")
                return {
                    "trailing_margin": 0.012,  # 1.2% par défaut
                    "activate_trailing_gain": 0.015,  # 1.5% activation scalp
                    "adaptive_sl": self.stop_loss_percent_base,
                }

            # === ACTIVATION TRAILING : Optimisée pour scalp intraday ===
            # Base 1.5% pour scalp BTC - capture gains réalistes sans
            # over-trading
            activate_trailing_gain = max(0.015, 0.8 * atr_percent)  # Min 1.5% scalp

            # === MARGES TRAILING : Adaptatives au régime de marché ===
            base_trailing_margin = max(
                0.012, 1.2 * atr_percent
            )  # Base 1.2% min au lieu de 0.8%

            # Multiplicateurs selon le régime (optimisés pour gains)
            regime_multipliers = {
                # Bull fort = marges très larges (ride les pumps)
                "TRENDING_BULL": 1.8,
                "BREAKOUT_BULL": 1.6,  # Breakout = marges larges
                "RANGING": 1.0,  # Range = marges normales
                "TRANSITION": 1.2,  # Transition = légèrement plus large
                "TRENDING_BEAR": 0.8,  # Bear = marges plus strictes
                "VOLATILE": 1.4,  # Volatile = marges modérément larges
                "BREAKOUT_BEAR": 0.7,  # Bear breakout = très strict
            }

            regime_factor = regime_multipliers.get(
                market_regime, 1.2
            )  # Défaut légèrement optimiste
            trailing_margin = base_trailing_margin * regime_factor

            # Contraintes finales scalp
            trailing_margin = max(0.012, min(0.030, trailing_margin))  # 1.2% à 3.0%
            activate_trailing_gain = max(
                0.015, min(0.025, activate_trailing_gain)
            )  # 1.5% à 2.5% scalp

            # Stop-loss adaptatif équilibré
            adaptive_sl = min(0.025, max(1.2 * atr_percent, 0.016))

            logger.debug(
                f"🚀 Seuils OPTIMISÉS {symbol}: trailing={trailing_margin*100:.2f}%, "
                f"activation={activate_trailing_gain*100:.2f}%, SL={adaptive_sl*100:.2f}% "
                f"(ATR={atr_percent*100:.2f}%, régime={market_regime})"
            )

            return {
                "trailing_margin": trailing_margin,
                "activate_trailing_gain": activate_trailing_gain,
                "adaptive_sl": adaptive_sl,
            }

        except Exception:
            logger.exception("Erreur calcul seuils optimisés {symbol}")
            return {
                "trailing_margin": 0.012,  # Fallback scalp
                "activate_trailing_gain": 0.015,  # 1.5% scalp cohérent
                "adaptive_sl": self.stop_loss_percent_base,
            }

    def _get_atr_percentage(self, symbol: str) -> float | None:
        """
        Récupère l'ATR en pourcentage depuis les données d'analyse.

        Args:
            symbol: Symbole

        Returns:
            ATR en pourcentage (ex: 0.025 = 2.5%) ou None
        """
        if not self.db_connection:
            return None

        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT ad.atr_14, md.close
                    FROM analyzer_data ad
                    JOIN market_data md ON (ad.symbol = md.symbol AND ad.timeframe = md.timeframe AND ad.time = md.time)
                    WHERE ad.symbol = %s AND ad.timeframe = '1m'
                    AND ad.atr_14 IS NOT NULL
                    ORDER BY ad.time DESC
                    LIMIT 1
                """,
                    (symbol,),
                )

                result = cursor.fetchone()
                if result and result[0] and result[1]:
                    atr_value = float(result[0])
                    close_price = float(result[1])
                    return atr_value / close_price if close_price > 0 else 0
                return None
        except Exception:
            logger.exception("Erreur récupération ATR {symbol}")
            return None

    def _get_market_regime(self, symbol: str) -> str:
        """
        Récupère le régime de marché actuel pour un symbole.

        Args:
            symbol: Symbole à analyser

        Returns:
            Régime de marché ou 'UNKNOWN'
        """
        if not self.db_connection:
            return "UNKNOWN"

        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT market_regime
                    FROM analyzer_data
                    WHERE symbol = %s AND timeframe = '1m'
                    AND market_regime IS NOT NULL
                    ORDER BY time DESC
                    LIMIT 1
                """,
                    (symbol,),
                )

                result = cursor.fetchone()
                if result and result[0]:
                    return result[0]
                return "UNKNOWN"
        except Exception:
            logger.exception("Erreur récupération régime marché {symbol}")
            return "UNKNOWN"

    def _calculate_adaptive_threshold(
        self, symbol: str, entry_price: float, entry_time: float
    ) -> float:
        """
        Calcule le seuil de stop-loss adaptatif avec ATR et données d'analyse.

        Args:
            symbol: Symbole à analyser
            entry_price: Prix d'entrée de la position
            entry_time: Timestamp d'entrée (epoch)

        Returns:
            Seuil de perte acceptable avant stop-loss (ex: 0.015 = 1.5%)
        """
        try:
            # Récupérer les seuils ATR
            atr_thresholds = self._get_atr_based_thresholds(symbol)
            atr_based_sl = atr_thresholds["adaptive_sl"]

            # Récupérer les données d'analyse si disponibles
            analysis = (
                self._get_latest_analysis_data(symbol) if self.db_connection else None
            )

            if not analysis:
                logger.debug(
                    f"Pas de données d'analyse pour {symbol}, utilisation ATR seul: {atr_based_sl*100:.2f}%"
                )
                return atr_based_sl

            # Récupérer le régime de marché
            analysis.get("market_regime", "UNKNOWN")

            # Calculer les facteurs d'ajustement (version allégée)
            regime_factor = self._calculate_regime_factor(analysis)
            support_factor = self._calculate_support_factor(analysis, entry_price)
            time_factor = self._calculate_time_factor(entry_time)

            # Combiner ATR avec moyenne pondérée (éviter l'écrasement par produit)
            # ATR = base (60%), régime = important (25%), support/temps =
            # modéré (15%)
            weighted_factor = (
                0.60 * 1.0  # ATR base (neutre à 1.0)
                + 0.25 * regime_factor
                + 0.10 * support_factor
                + 0.05 * time_factor
            )
            adaptive_threshold = float(atr_based_sl) * float(weighted_factor)

            # Contraintes finales équilibrées - coupe faux signaux, préserve
            # vrais trades
            adaptive_threshold = max(
                0.014, min(0.025, adaptive_threshold)
            )  # 1.4%-2.5% - équilibré

            logger.debug(
                f"🧠 Stop-loss adaptatif ATR+analyse {symbol}: {adaptive_threshold*100:.2f}%"
            )

            return adaptive_threshold

        except Exception:
            logger.exception("Erreur calcul stop-loss adaptatif {symbol}")
            return self.stop_loss_percent_base

    def _get_latest_analysis_data(self, symbol: str) -> dict | None:
        """
        Récupère les données d'analyse les plus récentes.
        """
        if not self.db_connection:
            return None

        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        market_regime, regime_strength, regime_confidence,
                        volatility_regime, atr_percentile,
                        nearest_support, support_strength,
                        trend_alignment, directional_bias
                    FROM analyzer_data
                    WHERE symbol = %s AND timeframe = '1m'
                    ORDER BY time DESC
                    LIMIT 1
                """,
                    (symbol,),
                )

                result = cursor.fetchone()
                if result:
                    return {
                        "market_regime": result[0],
                        "regime_strength": result[1],
                        "regime_confidence": float(result[2]) if result[2] else 50.0,
                        "volatility_regime": result[3],
                        "atr_percentile": float(result[4]) if result[4] else 50.0,
                        "nearest_support": float(result[5]) if result[5] else None,
                        "support_strength": result[6],
                        "trend_alignment": result[7],
                        "directional_bias": result[8],
                    }
                return None
        except Exception:
            logger.exception("Erreur récupération données analyse {symbol}")
            return None

    def _calculate_regime_factor(self, analysis: dict) -> float:
        """Calcule le facteur basé sur le régime de marché."""
        regime = analysis.get("market_regime", "UNKNOWN")
        strength = analysis.get("regime_strength", "WEAK")
        confidence = float(analysis.get("regime_confidence", 50))

        # LOGIQUE CORRIGÉE : en bear, SL plus large (laisser respirer
        # reversals)
        regime_multipliers = {
            "TRENDING_BULL": 1.2,  # Bull = plus tolérant (seuil plus large)
            "BREAKOUT_BULL": 1.1,  # Breakout bull = modérément tolérant
            "RANGING": 1.0,  # Range = neutre
            "TRANSITION": 0.95,  # Transition = légèrement strict
            # Bear = TOLÉRANT (laisser respirer reversals)
            "TRENDING_BEAR": 1.1,
            "VOLATILE": 0.9,  # Volatile = légèrement strict
            "BREAKOUT_BEAR": 1.0,  # Breakout bear = neutre (reversal possible)
        }

        base_factor = regime_multipliers.get(regime, 1.0)

        strength_multipliers = {
            "EXTREME": 1.2,  # Réduit de 1.3 à 1.2
            "STRONG": 1.1,
            "MODERATE": 1.0,
            "WEAK": 0.9,  # Augmenté de 0.8 à 0.9 (moins punitif)
        }

        strength_factor = strength_multipliers.get(strength, 1.0)
        confidence_factor = 0.7 + (float(confidence) / 100.0) * 0.6

        return float(base_factor * strength_factor * confidence_factor)

    def _calculate_volatility_factor(self, analysis: dict) -> float:
        """Calcule le facteur basé sur la volatilité."""
        volatility_regime = analysis.get("volatility_regime", "normal")
        atr_percentile = float(analysis.get("atr_percentile", 50))

        volatility_multipliers = {
            "low": 0.7,
            "normal": 1.0,
            "high": 1.4,
            "extreme": 1.8,
        }

        base_factor = volatility_multipliers.get(volatility_regime, 1.0)
        percentile_factor = 0.6 + (float(atr_percentile) / 100.0) * 0.8

        return float(base_factor * percentile_factor)

    def _calculate_support_factor(self, analysis: dict, entry_price: float) -> float:
        """Calcule le facteur basé sur la proximité des supports."""
        nearest_support = analysis.get("nearest_support")
        support_strength = analysis.get("support_strength", "WEAK")

        if not nearest_support:
            return 1.0

        support_price = float(nearest_support)
        entry_price_float = float(entry_price)
        support_distance = abs(entry_price_float - support_price) / entry_price_float

        strength_multipliers = {
            "MAJOR": 1.6,
            "STRONG": 1.3,
            "MODERATE": 1.1,
            "WEAK": 0.9,
        }

        strength_factor = strength_multipliers.get(support_strength, 1.0)

        if support_distance < 0.01:
            distance_factor = 1.4
        elif support_distance < 0.02:
            distance_factor = 1.2
        elif support_distance < 0.05:
            distance_factor = 1.0
        else:
            distance_factor = 0.8

        return float(strength_factor * distance_factor)

    def _calculate_time_factor(self, entry_time: float) -> float:
        """
        Calcule le facteur basé sur le temps écoulé - LOGIQUE SCALP CORRIGÉE.
        Trade récent = strict (non confirmé), trade ancien = tolérant (a résisté).
        """
        time_elapsed = float(time.time() - float(entry_time))
        minutes_elapsed = time_elapsed / 60.0

        # LOGIQUE INVERSÉE : strict sur récent, tolérant sur ancien
        if minutes_elapsed < 2:
            return 0.8  # Très récent = strict (non confirmé, risque max)
        if minutes_elapsed < 10:
            return 1.0  # Récent = neutre
        if minutes_elapsed < 60:
            return 1.1  # Confirmé = tolérant (a tenu 10-60min)
        return 1.2  # Très ancien (>1h) = très tolérant (a bien résisté)

    def get_current_price(self, symbol: str) -> float | None:
        """
        Récupère le prix actuel d'un symbole depuis Redis ou DB en fallback.

        Args:
            symbol: Symbole à récupérer

        Returns:
            Prix actuel ou None
        """
        try:
            # Essayer de récupérer depuis Redis (ticker)
            ticker_key = f"ticker:{symbol}"
            ticker_data = self.redis_client.get(ticker_key)

            if ticker_data:
                if isinstance(ticker_data, dict):
                    price = ticker_data.get("price")
                    if price:
                        return float(price)
                elif isinstance(ticker_data, str):
                    ticker_dict = json.loads(ticker_data)
                    price = ticker_dict.get("price")
                    if price:
                        return float(price)

            # Fallback: essayer market_data Redis
            market_key = f"market_data:{symbol}:1m"
            market_data = self.redis_client.get(market_key)

            if market_data:
                if isinstance(market_data, dict):
                    price = market_data.get("close")
                    if price:
                        return float(price)
                elif isinstance(market_data, str):
                    market_dict = json.loads(market_data)
                    price = market_dict.get("close")
                    if price:
                        return float(price)

            # Fallback final: récupérer depuis la base de données
            if self.db_connection:
                try:
                    with self.db_connection.cursor() as cursor:
                        cursor.execute(
                            """
                            SELECT close
                            FROM market_data
                            WHERE symbol = %s
                            ORDER BY time DESC
                            LIMIT 1
                        """,
                            (symbol,),
                        )

                        result = cursor.fetchone()
                        if result and result[0]:
                            logger.info(
                                f"💾 Prix récupéré depuis DB pour {symbol}: {result[0]}"
                            )
                            return float(result[0])
                        logger.warning(f"💾 Aucun résultat DB pour {symbol}")
                except Exception:
                    logger.exception("❌ Erreur DB pour : ")
            else:
                logger.warning(f"💾 Pas de connexion DB pour {symbol}")

            logger.warning(f"⚠️ Prix non trouvé pour {symbol} (Redis + DB)")
            return None

        except Exception:
            logger.exception("❌ Erreur récupération prix pour {symbol}")
            return None

    def _check_progressive_take_profit(
        self, symbol: str, gain_percent: float, position_id: str | None = None
    ) -> tuple[bool, str]:
        """
        Take profit progressif AMÉLIORÉ : vend si rechute significative depuis le palier atteint.
        Permet de rider les pumps tout en fermant les cycles efficacement.

        Args:
            symbol: Symbole pour tracking du palier
            gain_percent: Pourcentage de gain actuel (ex: 0.025 = 2.5%)
            position_id: ID unique de position

        Returns:
            (should_sell, reason)
        """
        # Paliers TP SIMPLIFIÉS - focus gains réels, stop micro-trading
        # destructeur
        tp_levels = [
            0.12,  # 12% - gains exceptionnels (pump majeur)
            0.08,  # 8% - gains très importants
            0.05,  # 5% - gains importants
            0.03,  # 3% - gain solide
            0.02,  # 2% - gain minimal viable (couvre frais + marge)
        ]

        # Trouver le palier le plus élevé atteint actuellement
        current_tp_level = None
        for level in tp_levels:
            if gain_percent >= level:
                current_tp_level = level
                break

        if current_tp_level is None:
            # Aucun palier atteint, pas de TP
            return False, f"Aucun palier TP atteint (+{gain_percent*100:.2f}%)"

        # Récupérer le palier max historique pour ce symbole
        historical_tp_key = self._get_redis_key("max_tp_level", symbol, position_id)
        historical_tp_data = self.redis_client.get(historical_tp_key)
        historical_max_tp = None

        if historical_tp_data:
            try:
                if isinstance(historical_tp_data, str | bytes):
                    if isinstance(historical_tp_data, bytes):
                        historical_tp_data = historical_tp_data.decode("utf-8")
                    historical_tp_dict = json.loads(historical_tp_data)
                    historical_max_tp = float(historical_tp_dict.get("level", 0))
                elif isinstance(historical_tp_data, dict):
                    historical_max_tp = float(historical_tp_data.get("level", 0))
            except Exception:
                logger.exception("Erreur récupération palier TP historique {symbol}")

        # Initialiser si pas de palier historique
        if historical_max_tp is None:
            historical_max_tp = 0

        # Mettre à jour le palier max si on a atteint un nouveau sommet
        if current_tp_level > historical_max_tp:
            self._update_max_tp_level(symbol, current_tp_level, position_id)
            logger.info(
                f"🎯 Nouveau palier TP pour {symbol}: +{current_tp_level*100:.1f}% (était +{historical_max_tp*100:.1f}%)"
            )
            historical_max_tp = current_tp_level

        # VENDRE si rechute significative depuis le palier max - tolérance INVERSÉE (strict sur gros gains)
        # Plus le gain est gros, plus on protège (tolérance serrée)
        if historical_max_tp >= 0.08:  # Gains exceptionnels (>8%)
            tolerance_factor = 0.85  # Garde 85% du palier (strict - rend 15%)
        elif historical_max_tp >= 0.05:  # Gros gains (5-8%)
            tolerance_factor = 0.80  # Garde 80% du palier (rend 20%)
        elif historical_max_tp >= 0.03:  # Gains moyens (3-5%)
            tolerance_factor = 0.75  # Garde 75% du palier (rend 25%)
        else:  # Petits gains (<3%)
            tolerance_factor = (
                # Garde 70% du palier (permissif - rend 30%, acceptable en
                # scalp)
                0.70
            )

        adjusted_threshold = historical_max_tp * tolerance_factor

        if gain_percent < adjusted_threshold:
            logger.warning(
                f"📉 Rechute significative pour {symbol}: +{gain_percent*100:.2f}% < seuil ajusté +{adjusted_threshold*100:.2f}% (palier max: +{historical_max_tp*100:.1f}%)"
            )
            self._clear_max_tp_level(symbol, position_id)  # Nettoyer après vente
            return (
                True,
                f"Rechute sous seuil TP ajusté +{adjusted_threshold*100:.2f}% (palier: +{historical_max_tp*100:.1f}%, gain: +{gain_percent*100:.2f}%)",
            )

        # Sinon, continuer à surveiller
        return (
            False,
            f"Au-dessus palier TP +{historical_max_tp*100:.1f}% (+{gain_percent*100:.2f}%), surveillance active",
        )

    def _update_max_tp_level(
        self, symbol: str, tp_level: float, position_id: str | None = None
    ) -> None:
        """
        Met à jour le palier TP maximum atteint pour un symbole.

        Args:
            symbol: Symbole
            tp_level: Nouveau palier TP maximum (ex: 0.025 = 2.5%)
            position_id: ID unique de position
        """
        try:
            tp_key = self._get_redis_key("max_tp_level", symbol, position_id)
            tp_data = {"level": tp_level, "timestamp": int(time.time() * 1000)}
            # TTL de 7 jours (604800s) - sera refresh à chaque check
            self.redis_client.set(tp_key, json.dumps(tp_data), expiration=604800)
            logger.debug(f"🎯 Palier TP mis à jour pour {symbol}: +{tp_level*100:.1f}%")
        except Exception:
            logger.exception("Erreur mise à jour palier TP pour {symbol}")

    def _clear_max_tp_level(self, symbol: str, position_id: str | None = None) -> None:
        """
        Supprime le palier TP maximum pour un symbole.

        Args:
            symbol: Symbole
            position_id: ID unique de position
        """
        try:
            tp_key = self._get_redis_key("max_tp_level", symbol, position_id)
            self.redis_client.delete(tp_key)
            logger.info(f"🧹 Palier TP max supprimé pour {symbol}")
        except Exception:
            logger.exception("Erreur suppression palier TP pour {symbol}")

    def _get_adaptive_trailing_margin(self, max_gain_percent: float) -> float:
        """
        Calcule une marge de trailing adaptative selon le gain maximum atteint.
        Plus le gain est élevé, plus la marge est serrée pour protéger les profits.

        Args:
            max_gain_percent: Gain maximum atteint depuis l'entrée (ex: 0.02 = 2%)

        Returns:
            Marge de tolérance à la baisse depuis le max (ex: 0.005 = 0.5%)
        """
        # Paliers progressifs : plus on monte, plus on protège
        if max_gain_percent >= 0.08:  # Gain ≥ 8% (exceptionnel)
            margin = 0.004  # Tolérance 0.4% seulement (verrouiller gains)
        elif max_gain_percent >= 0.05:  # Gain 5-8% (très bon)
            margin = 0.006  # Tolérance 0.6%
        elif max_gain_percent >= 0.03:  # Gain 3-5% (bon)
            margin = 0.008  # Tolérance 0.8%
        elif max_gain_percent >= 0.02:  # Gain 2-3% (solide)
            margin = 0.010  # Tolérance 1.0%
        elif max_gain_percent >= 0.015:  # Gain 1.5-2% (correct)
            margin = 0.012  # Tolérance 1.2%
        elif max_gain_percent >= 0.01:  # Gain 1-1.5% (début)
            margin = 0.014  # Tolérance 1.4%
        else:  # Gain < 1%
            margin = 0.015  # Tolérance 1.5% (large pour respirer)

        return margin

    def _cleanup_references(self, symbol: str, position_id: str | None = None) -> None:
        """
        Nettoie toutes les références pour un symbole après une vente.

        Args:
            symbol: Symbole
            position_id: ID unique de position
        """
        self._clear_sell_reference(symbol, position_id)
        self._clear_cycle_max_price(symbol, position_id)
        self._clear_max_tp_level(symbol, position_id)
        logger.info(f"🧹 Toutes les références nettoyées pour {symbol}")

    def _get_price_precision(self, price: float) -> int:
        """
        Détermine la précision d'affichage selon le niveau de prix.
        """
        if price >= 1000:
            return 2
        if price >= 100:
            return 3
        if price >= 1 or price >= 0.01:
            return 6
        if price >= 0.0001:
            return 10
        return 12
