"""
ParabolicSAR_Bounce_Strategy - VRAIE stratégie basée sur un SAR calculé correctement.
REFONTE CONCEPTUELLE COMPLÈTE - Abandon simulation Hull/ATR
"""

import logging
from typing import Any

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class ParabolicSAR_Bounce_Strategy(BaseStrategy):
    """
    Stratégie utilisant les rebonds du prix sur le VRAI Parabolic SAR.

    Le Parabolic SAR (Stop and Reverse) suit la tendance avec sa formule mathématique précise :
    SAR = SAR_prev + AF * (EP - SAR_prev)
    AF = Acceleration Factor (0.02 à 0.20, incrément 0.02)
    EP = Extreme Point (plus haut en uptrend, plus bas en downtrend)

    Signaux générés:
    - BUY: Prix rebondit sur SAR en tendance haussière (SAR < prix) + confirmations
    - SELL: Prix rebondit sur SAR en tendance baissière (SAR > prix) + confirmations
    """

    def __init__(self, symbol: str,
                 data: dict[str, Any], indicators: dict[str, Any]):
        super().__init__(symbol, data, indicators)

        # Paramètres SAR VRAIS
        self.af_initial = 0.02  # Acceleration Factor initial
        self.af_increment = 0.02  # Incrément AF
        self.af_max = 0.20  # AF maximum

        # Paramètres rebond RÉALISTES pour crypto 3m
        self.proximity_threshold = 0.005  # 0.5% proximité SAR (RÉALISTE)
        self.rebound_strength = 0.003  # 0.3% force rebond (RÉALISTE)
        self.max_sar_distance = 0.015  # 1.5% distance max SAR

        # Filtres DURCIS pour qualité
        self.min_confluence_required = 42  # Confluence relevée (35->42)
        self.min_adx_required = 18  # ADX réaliste pour tendance (12->18)
        self.required_confirmations = 2  # Confirmations minimales

        # Cache SAR pour performance
        self.sar_history: list[float] = []
        self.af_history: list[float] = []
        self.ep_history: list[float] = []
        self.trend_history: list[int] = []

    def _calculate_parabolic_sar(
        self, highs: list, lows: list, closes: list
    ) -> dict[str, Any] | None:
        """
        Calcule le Parabolic SAR avec la vraie formule mathématique.
        """
        if len(highs) < 3 or len(lows) < 3 or len(closes) < 3:
            return None

        try:
            # Conversion en float
            highs = [float(h) for h in highs[-20:]]  # Dernières 20 périodes
            lows = [float(low_val) for low_val in lows[-20:]]
            closes = [float(c) for c in closes[-20:]]

            n = len(highs)
            if n < 3:
                return None

            # Initialisation SAR
            sar_values = [0.0] * n
            af_values = [self.af_initial] * n
            ep_values = [0.0] * n
            trend_values = [1] * n  # 1 = uptrend, -1 = downtrend

            # Déterminer tendance initiale
            if highs[1] > highs[0]:
                trend_values[1] = 1
                sar_values[1] = lows[0]
                ep_values[1] = highs[1]
            else:
                trend_values[1] = -1
                sar_values[1] = highs[0]
                ep_values[1] = lows[1]

            # Calcul SAR période par période
            for i in range(2, n):
                prev_sar = sar_values[i - 1]
                prev_af = af_values[i - 1]
                prev_ep = ep_values[i - 1]
                prev_trend = trend_values[i - 1]

                # Calcul SAR suivant
                new_sar = prev_sar + prev_af * (prev_ep - prev_sar)

                # Uptrend
                if prev_trend == 1:
                    # Vérifier retournement
                    if lows[i] <= new_sar:
                        # Retournement vers downtrend
                        trend_values[i] = -1
                        sar_values[i] = prev_ep  # SAR = ancien EP
                        ep_values[i] = lows[i]  # Nouvel EP = low actuel
                        af_values[i] = self.af_initial  # Reset AF
                    else:
                        # Continuer uptrend
                        trend_values[i] = 1
                        sar_values[i] = max(
                            new_sar, lows[i - 1], lows[i - 2] if i > 2 else lows[i - 1]
                        )

                        # Mettre à jour EP et AF
                        if highs[i] > prev_ep:
                            ep_values[i] = highs[i]
                            af_values[i] = min(
                                prev_af + self.af_increment, self.af_max)
                        else:
                            ep_values[i] = prev_ep
                            af_values[i] = prev_af

                # Downtrend
                # Vérifier retournement
                elif highs[i] >= new_sar:
                    # Retournement vers uptrend
                    trend_values[i] = 1
                    sar_values[i] = prev_ep  # SAR = ancien EP
                    ep_values[i] = highs[i]  # Nouvel EP = high actuel
                    af_values[i] = self.af_initial  # Reset AF
                else:
                    # Continuer downtrend
                    trend_values[i] = -1
                    sar_values[i] = min(
                        new_sar,
                        highs[i - 1],
                        highs[i - 2] if i > 2 else highs[i - 1],
                    )

                    # Mettre à jour EP et AF
                    if lows[i] < prev_ep:
                        ep_values[i] = lows[i]
                        af_values[i] = min(
                            prev_af + self.af_increment, self.af_max)
                    else:
                        ep_values[i] = prev_ep
                        af_values[i] = prev_af

            return {
                "current_sar": sar_values[-1],
                "current_trend": trend_values[-1],
                "current_af": af_values[-1],
                "current_ep": ep_values[-1],
                "sar_history": sar_values[-3:],  # 3 dernières valeurs
                "trend_history": trend_values[-3:],
            }

        except (ValueError, TypeError, IndexError) as e:
            logger.debug(f"Erreur calcul SAR: {e}")
            return None

    def _get_current_values(self) -> dict[str, float | None]:
        """Récupère les valeurs actuelles des indicateurs."""
        return {
            # Confluence et validation
            "confluence_score": self.indicators.get("confluence_score"),
            "signal_strength": self.indicators.get("signal_strength"),
            # Tendance
            "trend_strength": self.indicators.get("trend_strength"),
            "directional_bias": self.indicators.get("directional_bias"),
            "adx_14": self.indicators.get("adx_14"),
            "plus_di": self.indicators.get("plus_di"),
            "minus_di": self.indicators.get("minus_di"),
            # Volume
            "volume_ratio": self.indicators.get("volume_ratio"),
            # Oscillateurs pour timing
            "rsi_14": self.indicators.get("rsi_14"),
            "stoch_k": self.indicators.get("stoch_k"),
            "stoch_d": self.indicators.get("stoch_d"),
            # Contexte
            "market_regime": self.indicators.get("market_regime"),
            "volatility_regime": self.indicators.get("volatility_regime"),
            "momentum_score": self.indicators.get("momentum_score"),
        }

    def _create_rejection_signal(self, reason: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Helper to create rejection signals."""
        base_metadata = {"strategy": self.name}
        if metadata:
            base_metadata.update(metadata)
        return {
            "side": None,
            "confidence": 0.0,
            "strength": "weak",
            "reason": reason,
            "metadata": base_metadata,
        }

    def _validate_signal_preconditions(self, values: dict[str, Any]) -> tuple[dict[str, Any] | None, float | None, dict[str, Any] | None]:
        """Valide les préconditions du signal. Retourne (sar_data, current_price, error_signal) ou (None, None, error)."""
        ohlc_data = self._get_ohlc_data()
        if not ohlc_data:
            return None, None, self._create_rejection_signal("Données OHLC insuffisantes pour calcul SAR", {})

        sar_data = self._calculate_parabolic_sar(
            ohlc_data["highs"], ohlc_data["lows"], ohlc_data["closes"]
        )
        if not sar_data:
            return None, None, self._create_rejection_signal("Impossible de calculer le SAR", {})

        current_price = float(ohlc_data["closes"][-1])
        current_sar = sar_data["current_sar"]

        # Validation confluence
        confluence_score = values.get("confluence_score", 0)
        if not confluence_score or float(confluence_score) < self.min_confluence_required:
            return None, None, self._create_rejection_signal(
                f"Confluence insuffisante ({confluence_score}) < {self.min_confluence_required}",
                {}
            )

        # Validation distance SAR
        distance_to_sar = abs(current_price - current_sar) / current_price
        if distance_to_sar > self.max_sar_distance:
            return None, None, self._create_rejection_signal(
                f"Prix trop éloigné du SAR ({distance_to_sar:.1%} > {self.max_sar_distance:.1%})",
                {"distance_to_sar": distance_to_sar}
            )

        return sar_data, current_price, None

    def _get_ohlc_data(self) -> dict[str, list] | None:
        """Récupère les données OHLC pour calcul SAR."""
        try:
            if not self.data:
                return None

            required_keys = ["high", "low", "close"]
            if not all(key in self.data for key in required_keys):
                return None

            # Vérifier longueur minimum
            min_length = min(len(self.data[key]) for key in required_keys)
            if min_length < 3:
                return None

            return {
                "highs": self.data["high"][-20:],  # 20 dernières périodes
                "lows": self.data["low"][-20:],
                "closes": self.data["close"][-20:],
            }
        except (KeyError, IndexError, TypeError):
            return None

    def generate_signal(self) -> dict[str, Any]:
        """
        Génère un signal basé sur les rebonds sur le VRAI Parabolic SAR.
        """
        if not self.validate_data():
            return self._create_rejection_signal("Données insuffisantes", {})

        values = self._get_current_values()

        # Valider les préconditions
        sar_data, current_price, error_signal = self._validate_signal_preconditions(values)
        if error_signal:
            return error_signal

        current_sar = sar_data["current_sar"]
        current_trend = sar_data["current_trend"]

        # Analyser le rebond selon la tendance SAR
        if current_trend == 1 and current_price > current_sar:
            return self._analyze_sar_bounce(values, current_price, sar_data, "BUY")
        if current_trend == -1 and current_price < current_sar:
            return self._analyze_sar_bounce(values, current_price, sar_data, "SELL")

        return self._create_rejection_signal(
            f"Prix/SAR non alignés pour rebond (trend={current_trend}, prix={current_price:.4f}, SAR={current_sar:.4f})",
            {"sar_data": sar_data}
        )

    def _analyze_sar_bounce(
        self,
        values: dict[str, Any],
        current_price: float,
        sar_data: dict[str, Any],
        signal_side: str,
    ) -> dict[str, Any]:
        """Analyse un rebond sur le SAR."""

        # Base confidence standardisée
        base_confidence = 0.65
        confidence_boost = 0.0

        current_sar = sar_data["current_sar"]
        current_af = sar_data["current_af"]
        current_ep = sar_data["current_ep"]

        # Distance et proximité au SAR
        distance_to_sar = abs(current_price - current_sar) / current_price

        reason = (
            f"Rebond SAR {signal_side} (SAR: {current_sar:.4f}, AF: {current_af:.3f})"
        )

        # BONUS selon proximité SAR (plus proche = meilleur)
        if distance_to_sar <= self.proximity_threshold:
            confidence_boost += 0.25
            reason += f" - TRÈS proche SAR ({distance_to_sar:.1%})"
        elif distance_to_sar <= self.proximity_threshold * 2:
            confidence_boost += 0.18
            reason += f" - proche SAR ({distance_to_sar:.1%})"
        else:
            confidence_boost += 0.10
            reason += f" - SAR acceptable ({distance_to_sar:.1%})"

        # BONUS selon AF (plus élevé = tendance plus forte)
        if current_af >= 0.15:
            confidence_boost += 0.20
            reason += f" + AF élevé ({current_af:.3f})"
        elif current_af >= 0.10:
            confidence_boost += 0.15
            reason += f" + AF modéré ({current_af:.3f})"
        elif current_af >= 0.06:
            confidence_boost += 0.10
            reason += f" + AF correct ({current_af:.3f})"

        # Détection pattern rebond dans historique
        sar_history = sar_data.get("sar_history", [])
        if len(sar_history) >= 3 and self._detect_sar_bounce_pattern(
                sar_history, current_price, signal_side):
            confidence_boost += 0.25
            reason += " + pattern rebond SAR détecté"

        # VALIDATION MARKET REGIME - REJET si contradictoire
        market_regime = values.get("market_regime")
        if market_regime:
            if (signal_side == "BUY" and market_regime == "TRENDING_BEAR") or (
                signal_side == "SELL" and market_regime == "TRENDING_BULL"
            ):
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Rejet {signal_side}: régime contradictoire ({market_regime})",
                    "metadata": {
                        "strategy": self.name,
                        "market_regime": market_regime},
                }
            # Bonus si régime aligné
            if (
                signal_side == "BUY" and market_regime in ["TRENDING_BULL", "RANGING"]
            ) or (
                signal_side == "SELL" and market_regime in ["TRENDING_BEAR", "RANGING"]
            ):
                confidence_boost += 0.15
                reason += f" + régime favorable ({market_regime})"

        # Confirmations techniques
        additional_boost, reason_additions = self._add_sar_confirmations(
            values, signal_side, current_price
        )

        # Vérifier si ADX trop faible (signal de rejet)
        if additional_boost <= -1.0:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": (
                    "Rejet SAR: ADX insuffisant" + reason_additions.split("REJET:")[1]
                    if "REJET:" in reason_additions
                    else ""
                ),
                "metadata": {"strategy": self.name},
            }

        confidence_boost += additional_boost
        reason += reason_additions

        # Compter confirmations obligatoires
        confirmations_count = self._count_sar_confirmations(
            values, signal_side)
        if confirmations_count < self.required_confirmations:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Confirmations SAR insuffisantes ({confirmations_count}/{self.required_confirmations})",
                "metadata": {
                    "strategy": self.name},
            }

        # PÉNALITÉ VOLUME - Empêcher les boosts faciles sans volume
        volume_ratio = values.get("volume_ratio")
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio < 0.8:
                    # Malus pour volume très faible
                    confidence_boost -= 0.20
                    reason += f" - volume très faible ({vol_ratio:.2f}x)"
                elif vol_ratio < 1.1:
                    # Limiter les boosts si volume insuffisant
                    confidence_boost = min(confidence_boost, 0.10)
                    reason += f" - boost limité par volume faible ({vol_ratio:.2f}x)"
            except (ValueError, TypeError):
                pass

        # Calcul final
        confidence = min(base_confidence * (1 + confidence_boost), 0.95)

        # Seuil final réaliste
        min_confidence = 0.45 if signal_side == "BUY" else 0.42
        if confidence < min_confidence:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Confiance SAR insuffisante ({confidence:.2f} < {min_confidence:.2f})",
                "metadata": {
                    "strategy": self.name},
            }

        strength = self.get_strength_from_confidence(confidence)

        return {
            "side": signal_side,
            "confidence": confidence,
            "strength": strength,
            "reason": reason,
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "current_price": current_price,
                "current_sar": current_sar,
                "current_af": current_af,
                "current_ep": current_ep,
                "distance_to_sar": distance_to_sar,
                "sar_trend": sar_data["current_trend"],
                "confirmations_count": confirmations_count,
            },
        }

    def _detect_sar_bounce_pattern(
        self, sar_history: list, current_price: float, signal_side: str
    ) -> bool:
        """Détecte un pattern de rebond sur SAR avec validation chandelier."""
        if len(sar_history) < 3:
            return False

        try:
            sar_prev3, sar_prev2, sar_prev1 = sar_history[-3:]

            # Vérifier proximité SAR d'abord
            proximity_ok = False
            if signal_side == "BUY":
                proximity_ok = (
                    current_price > sar_prev1
                    and abs(current_price - sar_prev1) / current_price
                    < self.proximity_threshold * 2
                )
            else:
                proximity_ok = (
                    current_price < sar_prev1
                    and abs(current_price - sar_prev1) / current_price
                    < self.proximity_threshold * 2
                )

            if not proximity_ok:
                return False

            # Vérifier chandelier de rebond (open/close)
            if (
                self.data
                and "open" in self.data
                and "close" in self.data
                and len(self.data["close"]) >= 1
            ):
                try:
                    current_open = float(self.data["open"][-1])
                    current_close = float(self.data["close"][-1])

                    if signal_side == "BUY":
                        # Chandelier haussier pour rebond BUY
                        return current_close > current_open
                except (ValueError, TypeError, IndexError):
                    pass
                else:
                    # Chandelier baissier pour rebond SELL
                    return current_close < current_open

            # Fallback : proximité seule si pas de données open/close
        except (ValueError, TypeError, ZeroDivisionError):
            pass
        else:
            return proximity_ok

        return False

    def _add_sar_confirmations(
        self, values: dict[str, Any], signal_side: str, _current_price: float
    ) -> tuple[float, str]:
        """Ajoute confirmations spécifiques au SAR."""
        boost = 0.0
        reason_additions = ""

        # ADX pour force tendance - REJET si trop faible
        adx_14 = values.get("adx_14")
        if adx_14:
            try:
                adx = float(adx_14)
                if adx < self.min_adx_required:
                    # Retourner directement le rejet depuis _add_sar_confirmations n'est pas possible
                    # On utilise un boost très négatif qui sera géré dans
                    # l'appelant
                    boost -= 1.0  # Signal de rejet
                    reason_additions += (
                        f" REJET: ADX trop faible ({adx:.0f} < {self.min_adx_required})"
                    )
                elif adx > 25:
                    boost += 0.20
                    reason_additions += f" + ADX fort ({adx:.0f})"
                elif adx > 20:
                    boost += 0.15
                    reason_additions += f" + ADX correct ({adx:.0f})"
                else:
                    boost += 0.08
                    reason_additions += f" + ADX minimal ({adx:.0f})"
            except (ValueError, TypeError):
                pass

        # Volume DURCI - bonus seulement si réellement élevé
        volume_ratio = values.get("volume_ratio")
        if volume_ratio:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= 2.0:
                    boost += 0.20
                    reason_additions += f" + volume EXCEPTIONNEL ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.5:
                    boost += 0.15
                    reason_additions += f" + volume élevé ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.2:
                    boost += 0.08
                    reason_additions += f" + volume modéré ({vol_ratio:.1f}x)"
                # Pas de bonus sous 1.2x
            except (ValueError, TypeError):
                pass

        # RSI pour timing - fenêtres élargies
        rsi_14 = values.get("rsi_14")
        if rsi_14:
            try:
                rsi = float(rsi_14)
                if (signal_side == "BUY" and 30 <= rsi <= 60) or (
                        signal_side == "SELL" and 40 <= rsi <= 70):  # Elargi 55->60
                    boost += 0.12
                    reason_additions += f" + RSI optimal rebond ({rsi:.0f})"
            except (ValueError, TypeError):
                pass

        return boost, reason_additions

    def _count_sar_confirmations(
            self, values: dict[str, Any], _signal_side: str) -> int:
        """Compte les confirmations obligatoires pour SAR."""
        count = 0

        # 1. ADX minimum
        adx_14 = values.get("adx_14")
        if adx_14 and float(adx_14) >= self.min_adx_required:
            count += 1

        # 2. Volume acceptable DURCI
        volume_ratio = values.get("volume_ratio")
        if volume_ratio and float(volume_ratio) >= 1.0:  # Relevé 0.8->1.0
            count += 1

        # 3. Confluence DURCIE
        confluence_score = values.get("confluence_score")
        if confluence_score and float(
                confluence_score) >= self.min_confluence_required:
            count += 1

        return count

    def validate_data(self) -> bool:
        """Valide les données pour calcul SAR."""
        if not super().validate_data():
            return False

        # Vérifier données OHLC
        required_ohlc = ["high", "low", "close"]
        for key in required_ohlc:
            if key not in self.data or not self.data[key] or len(
                    self.data[key]) < 3:
                logger.warning(
                    f"{self.name}: Données {key} insuffisantes pour SAR")
                return False

        # Vérifier indicateurs minimum
        required_indicators = ["confluence_score", "adx_14"]
        missing = 0
        for indicator in required_indicators:
            if indicator not in self.indicators or self.indicators[indicator] is None:
                missing += 1

        if missing > 1:
            logger.warning(
                f"{self.name}: Trop d'indicateurs manquants ({missing})")
            return False

        return True
