"""
Resistance_Rejection_Strategy - Stratégie basée sur le rejet au niveau de résistance.
Détecte les échecs de cassure de résistance pour signaler des ventes (retournement baissier).
"""

import contextlib
import logging
from typing import Any

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class Resistance_Rejection_Strategy(BaseStrategy):
    """
    Stratégie bidirectionnelle détectant les rejets et continuations au niveau des résistances.

    Patterns détectés :
    1. SELL: Rejet de résistance (essoufflement haussier)
       - Prix approche d'une résistance forte
       - Tentative de cassure mais échec (rejection)
       - Volume élevé au moment du rejet
       - Indicateurs techniques montrent essoufflement haussier

    2. BUY: Continuation après échec de cassure baissier
       - Prix tentait de casser en-dessous d'un support/résistance
       - Échec de la cassure baissiere (faux breakout)
       - Retour au-dessus avec volume et momentum haussier

    Signaux générés:
    - SELL: Rejet confirmé au niveau de résistance avec momentum baissier
    - BUY: Continuation haussiere après échec de cassure baissiere
    """

    def __init__(self, symbol: str,
                 data: dict[str, Any], indicators: dict[str, Any]):
        super().__init__(symbol, data, indicators)

        # Paramètres de proximité résistance - DURCIS
        self.resistance_proximity_threshold = 0.01  # 1% distance maximum
        self.tight_proximity_threshold = 0.005  # 0.5% = très proche

        # Paramètres de rejet - PLUS RÉALISTES
        # 0.15% retour minimum (plus large)
        self.min_rejection_distance = 0.0015
        self.rejection_confirmation_bars = 1  # Confirmation immédiate crypto 3m

        # Paramètres volume et momentum
        self.min_rejection_volume = 1.0  # Volume normal minimum
        self.strong_rejection_volume = 1.5  # Volume 1.5x pour rejet fort
        self.momentum_reversal_threshold = -0.2  # Momentum devient négatif

        # Paramètres RSI/oscillateurs basés sur données DB réelles
        # RSI P75 (25% des cas) au lieu de 65 (6%)
        self.overbought_rsi_threshold = 60
        # RSI entre P90-Max pour être accessible
        self.extreme_overbought_threshold = 68
        self.williams_r_overbought = -25  # Williams %R légèrement durci

        # Paramètres de résistance
        self.min_resistance_strength = 0.5  # Force minimum résistance
        self.strong_resistance_threshold = 0.8  # Résistance très forte

    def _get_current_values(self) -> dict[str, float | None]:
        """Récupère les valeurs actuelles des indicateurs pré-calculés."""
        return {
            # Support/Résistance (principal)
            "nearest_resistance": self.indicators.get("nearest_resistance"),
            "nearest_support": self.indicators.get("nearest_support"),
            "resistance_strength": self.indicators.get("resistance_strength"),
            "support_strength": self.indicators.get("support_strength"),
            "resistance_levels": self.indicators.get("resistance_levels"),
            "break_probability": self.indicators.get("break_probability"),
            "pivot_count": self.indicators.get("pivot_count"),
            # Bollinger Bands (résistance dynamique)
            "bb_upper": self.indicators.get("bb_upper"),
            "bb_lower": self.indicators.get("bb_lower"),
            "bb_position": self.indicators.get("bb_position"),
            "bb_width": self.indicators.get("bb_width"),
            # Oscillateurs momentum (essoufflement)
            "rsi_14": self.indicators.get("rsi_14"),
            "rsi_21": self.indicators.get("rsi_21"),
            "williams_r": self.indicators.get("williams_r"),
            "stoch_k": self.indicators.get("stoch_k"),
            "stoch_d": self.indicators.get("stoch_d"),
            # Volume analysis (confirmation rejet)
            "volume_ratio": self.indicators.get("volume_ratio"),
            "relative_volume": self.indicators.get("relative_volume"),
            "volume_quality_score": self.indicators.get("volume_quality_score"),
            "volume_spike_multiplier": self.indicators.get("volume_spike_multiplier"),
            "trade_intensity": self.indicators.get("trade_intensity"),
            # Momentum et trend
            "momentum_score": self.indicators.get("momentum_score"),
            "trend_strength": self.indicators.get("trend_strength"),
            "directional_bias": self.indicators.get("directional_bias"),
            "market_regime": self.indicators.get("market_regime"),
            "roc_10": self.indicators.get("roc_10"),
            "momentum_10": self.indicators.get("momentum_10"),
            # Pattern et confluence
            "pattern_detected": self.indicators.get("pattern_detected"),
            "pattern_confidence": self.indicators.get("pattern_confidence"),
            "signal_strength": self.indicators.get("signal_strength"),
            "confluence_score": self.indicators.get("confluence_score"),
        }

    def _detect_resistance_rejection(
        self, values: dict[str, Any], current_price: float
    ) -> dict[str, Any]:
        """Détecte un pattern de rejet de résistance."""
        rejection_score = 0.0
        rejection_indicators = []

        # Vérification résistance principale
        nearest_resistance = values.get("nearest_resistance")
        if nearest_resistance is None:
            return {"is_rejection": False, "score": 0.0, "indicators": []}

        try:
            resistance_level = float(nearest_resistance)
        except (ValueError, TypeError):
            return {"is_rejection": False, "score": 0.0, "indicators": []}

        # Distance à la résistance
        distance_to_resistance = (
            abs(current_price - resistance_level) / resistance_level
        )

        # Vérifier proximité de la résistance
        if distance_to_resistance > self.resistance_proximity_threshold:
            return {
                "is_rejection": False,
                "score": 0.0,
                "indicators": [
                    f"Trop loin de résistance ({distance_to_resistance*100:.2f}%)"
                ],
            }

        # Score de proximité
        if distance_to_resistance <= self.tight_proximity_threshold:
            rejection_score += 0.3
            rejection_indicators.append(
                f"Très proche résistance ({distance_to_resistance*100:.2f}%)"
            )
        else:
            rejection_score += 0.15
            rejection_indicators.append(
                f"Proche résistance ({distance_to_resistance*100:.2f}%)"
            )

        # Vérifier position par rapport à résistance (logique assouplie)
        price_vs_resistance = (
            current_price - resistance_level) / resistance_level
        if price_vs_resistance > 0.002:  # Si >0.2% au-dessus
            rejection_score += 0.1  # Léger bonus (test résistance)
            rejection_indicators.append("Test actif de la résistance")
        elif price_vs_resistance > -0.002:  # Entre -0.2% et +0.2%
            rejection_score += 0.25  # Bonus fort (juste à la résistance)
            rejection_indicators.append("Prix exactement à la résistance")
        else:  # En-dessous
            rejection_score += 0.3  # Maximum (rejet confirmé)
            rejection_indicators.append("Prix rejeté sous résistance")

        # Force de la résistance
        resistance_strength = values.get("resistance_strength")
        if resistance_strength is not None:
            try:
                # Supposer que resistance_strength est un string comme
                # "STRONG", "MODERATE", etc.
                if isinstance(resistance_strength, str):
                    strength_map = {
                        "WEAK": 0.2,
                        "MODERATE": 0.5,
                        "STRONG": 0.8,
                        "MAJOR": 1.0,
                    }
                    strength_val = strength_map.get(
                        resistance_strength.upper(), 0.5)
                else:
                    strength_val = float(resistance_strength)

                if strength_val >= self.strong_resistance_threshold:
                    rejection_score += 0.25
                    rejection_indicators.append(
                        f"Résistance très forte ({strength_val:.2f})"
                    )
                elif strength_val >= self.min_resistance_strength:
                    rejection_score += 0.15
                    rejection_indicators.append(
                        f"Résistance forte ({strength_val:.2f})"
                    )
            except (ValueError, TypeError):
                pass

        # Bollinger Band résistance dynamique
        bb_upper = values.get("bb_upper")
        bb_position = values.get("bb_position")
        if bb_upper is not None and bb_position is not None:
            try:
                bb_upper_val = float(bb_upper)
                bb_pos_val = float(bb_position)

                # Si près de BB upper ET position élevée = résistance dynamique
                bb_distance = abs(current_price - bb_upper_val) / bb_upper_val
                if bb_distance <= 0.002 and bb_pos_val >= 0.9:
                    rejection_score += 0.2
                    rejection_indicators.append(
                        f"Résistance Bollinger (pos={bb_pos_val:.2f})"
                    )
            except (ValueError, TypeError):
                pass

        return {
            "is_rejection": rejection_score
            >= 0.40,  # Seuil durci de 0.35 à 0.40 pour réduire SELL
            "score": rejection_score,
            "indicators": rejection_indicators,
            "resistance_level": resistance_level,
            "distance_pct": distance_to_resistance * 100,
        }

    def _detect_momentum_exhaustion(
            self, values: dict[str, Any]) -> dict[str, Any]:
        """Détecte l'essoufflement du momentum haussier."""
        exhaustion_score = 0.0
        exhaustion_indicators = []

        # RSI en surachat (essoufflement)
        rsi_14 = values.get("rsi_14")
        if rsi_14 is not None:
            try:
                rsi_val = float(rsi_14)
                if rsi_val >= self.extreme_overbought_threshold:
                    exhaustion_score += 0.3
                    exhaustion_indicators.append(
                        f"RSI surachat extrême ({rsi_val:.1f})"
                    )
                elif rsi_val >= self.overbought_rsi_threshold:
                    exhaustion_score += 0.2
                    exhaustion_indicators.append(
                        f"RSI surachat ({rsi_val:.1f})")
            except (ValueError, TypeError):
                pass

        # Williams %R confirme surachat
        williams_r = values.get("williams_r")
        if williams_r is not None:
            try:
                wr_val = float(williams_r)
                if wr_val >= self.williams_r_overbought:
                    exhaustion_score += 0.15
                    exhaustion_indicators.append(
                        f"Williams%R surachat ({wr_val:.1f})")
            except (ValueError, TypeError):
                pass

        # Stochastic en surachat
        stoch_k = values.get("stoch_k")
        stoch_d = values.get("stoch_d")
        if stoch_k is not None and stoch_d is not None:
            try:
                k_val = float(stoch_k)
                d_val = float(stoch_d)
                if k_val >= 80 and d_val >= 80:
                    exhaustion_score += 0.15
                    exhaustion_indicators.append(
                        f"Stoch surachat (K={k_val:.1f}, D={d_val:.1f})"
                    )
            except (ValueError, TypeError):
                pass

        # Momentum score devient négatif (format 0-100, 50=neutre) - DURCI
        momentum_score = values.get("momentum_score")
        if momentum_score is not None:
            try:
                momentum_val = float(momentum_score)
                # Momentum vraiment faible (10 cas en DB)
                if momentum_val <= 48:
                    exhaustion_score += 0.25  # Bonus augmenté car très rare
                    exhaustion_indicators.append(
                        f"Momentum très affaibli ({momentum_val:.1f})"
                    )
                elif momentum_val <= 49.5:  # Momentum affaibli (P10)
                    exhaustion_score += 0.18
                    exhaustion_indicators.append(
                        f"Momentum affaibli ({momentum_val:.1f})"
                    )
            except (ValueError, TypeError):
                pass

        # ROC ralentissement
        roc_10 = values.get("roc_10")
        if roc_10 is not None:
            try:
                roc_val = float(roc_10)
                if roc_val < 0:  # ROC négatif = retournement
                    exhaustion_score += 0.15
                    exhaustion_indicators.append(
                        f"ROC négatif ({roc_val:.2f}%)")
            except (ValueError, TypeError):
                pass

        return {
            "is_exhausted": exhaustion_score >= 0.20,  # Seuil durci de 0.15 à 0.20
            "score": exhaustion_score,
            "indicators": exhaustion_indicators,
        }

    def generate_signal(self) -> dict[str, Any]:
        """
        Génère un signal basé sur le rejet de résistance.
        """
        if not self.validate_data():
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Données insuffisantes",
                "metadata": {},
            }

        values = self._get_current_values()

        # Récupérer le prix actuel depuis les données OHLCV
        current_price = None
        if self.data.get("close"):
            with contextlib.suppress(IndexError, ValueError, TypeError):
                current_price = float(self.data["close"][-1])

        if current_price is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Prix actuel non disponible",
                "metadata": {"strategy": self.name},
            }

        # Détection du rejet de résistance
        rejection_analysis = self._detect_resistance_rejection(
            values, current_price)

        if not rejection_analysis["is_rejection"]:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Pas de rejet de résistance détecté: {', '.join(rejection_analysis['indicators'][:2]) if rejection_analysis['indicators'] else 'Aucune résistance proche'}",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "current_price": current_price,
                    "rejection_score": rejection_analysis["score"],
                },
            }

        # Détection de l'essoufflement du momentum
        exhaustion_analysis = self._detect_momentum_exhaustion(values)

        # Vérification market regime - REJET SELL si marché fortement haussier
        market_regime = values.get("market_regime")
        if market_regime == "TRENDING_BULL":
            # Au lieu de rejeter, essayer un signal BUY de continuation
            buy_signal = self._detect_continuation_buy(values, current_price)
            if buy_signal["is_continuation"]:
                return self._create_continuation_buy_signal(
                    values, current_price, buy_signal
                )
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Marché haussier - pas de rejet SELL ni continuation BUY",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "market_regime": market_regime,
                },
            }

        # Essayer d'abord un signal BUY de continuation
        continuation_analysis = self._detect_continuation_buy(
            values, current_price)
        if continuation_analysis["is_continuation"]:
            return self._create_continuation_buy_signal(
                values, current_price, continuation_analysis
            )

        # Signal SELL si rejet (essoufflement optionnel)
        if rejection_analysis["is_rejection"]:
            base_confidence = 0.65  # Base plus accessible
            confidence_boost = (
                rejection_analysis["score"] * 1.0
            )  # Multiplicateur généreux

            reason = f"Rejet résistance {rejection_analysis['resistance_level']:.2f} ({rejection_analysis['distance_pct']:.2f}%)"

            # Bonus pour essoufflement momentum (pas obligatoire)
            if exhaustion_analysis["is_exhausted"]:
                confidence_boost += exhaustion_analysis["score"] * 0.6
                reason += " + momentum épuisé"
            elif exhaustion_analysis["score"] >= 0.15:  # Essoufflement partiel
                confidence_boost += exhaustion_analysis["score"] * 0.4
                reason += " + signes essoufflement"

            # Volume de confirmation - ASSOUPLI
            volume_ratio = values.get("volume_ratio")
            if volume_ratio is not None:
                try:
                    vol_ratio = float(volume_ratio)
                    if vol_ratio >= self.strong_rejection_volume:
                        confidence_boost += 0.25
                        reason += f" + volume fort ({vol_ratio:.1f}x)"
                    elif vol_ratio >= self.min_rejection_volume:
                        confidence_boost += 0.15
                        reason += f" + volume confirmé ({vol_ratio:.1f}x)"
                    else:  # Volume faible = pénalité
                        confidence_boost -= 0.1
                        reason += f" - volume faible ({vol_ratio:.1f}x)"
                except (ValueError, TypeError):
                    pass

            # Confluence score
            confluence_score = values.get("confluence_score")
            if confluence_score is not None:
                try:
                    conf_val = float(confluence_score)
                    if conf_val > 60:  # Seuil assoupli
                        confidence_boost += 0.12
                        reason += " + haute confluence"
                    elif conf_val > 50:
                        confidence_boost += 0.08
                        reason += " + confluence modérée"
                except (ValueError, TypeError):
                    pass

            # Pattern confidence
            pattern_confidence = values.get("pattern_confidence")
            if pattern_confidence is not None:
                try:
                    pat_conf = float(pattern_confidence)
                    if pat_conf > 0.8:
                        confidence_boost += 0.1
                except (ValueError, TypeError):
                    pass

            # Calcul confidence avec clamp explicite
            raw_confidence = self.calculate_confidence(
                base_confidence, 1.0 + confidence_boost
            )
            raw_confidence = min(1.0, raw_confidence)  # Clamp à 100%

            # Vérification seuil minimum
            if raw_confidence < 0.48:  # Seuil minimum durci à 48% pour SELL
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Signal rejet résistance trop faible (conf: {raw_confidence:.2f} < 0.48)",
                    "metadata": {
                        "strategy": self.name,
                        "symbol": self.symbol,
                        "rejected_confidence": raw_confidence,
                        "rejection_score": rejection_analysis["score"],
                        "exhaustion_score": exhaustion_analysis["score"],
                    },
                }

            confidence = raw_confidence
            strength = self.get_strength_from_confidence(confidence)

            return {
                "side": "SELL",
                "confidence": confidence,
                "strength": strength,
                "reason": reason,
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "current_price": current_price,
                    "resistance_level": rejection_analysis["resistance_level"],
                    "resistance_distance_pct": rejection_analysis["distance_pct"],
                    "rejection_score": rejection_analysis["score"],
                    "exhaustion_score": exhaustion_analysis["score"],
                    "rejection_indicators": rejection_analysis["indicators"],
                    "exhaustion_indicators": exhaustion_analysis["indicators"],
                    "volume_ratio": volume_ratio,
                    "rsi_14": values.get("rsi_14"),
                    "williams_r": values.get("williams_r"),
                    "momentum_score": values.get("momentum_score"),
                    "confluence_score": confluence_score,
                    "pattern_confidence": pattern_confidence,
                },
            }

        return {
            "side": None,
            "confidence": 0.0,
            "strength": "weak",
            "reason": "Conditions de rejet non remplies",
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "current_price": current_price,
            },
        }

    def _detect_continuation_buy(
        self, values: dict[str, Any], current_price: float
    ) -> dict[str, Any]:
        """Détecte une opportunité de continuation BUY après échec de cassure baissiere."""
        continuation_score = 0.0
        continuation_indicators = []

        # Vérifier présence d'un niveau de support ou résistance
        key_level = None
        level_type = None

        nearest_support = values.get("nearest_support")
        nearest_resistance = values.get("nearest_resistance")

        if nearest_support is not None:
            try:
                support_level = float(nearest_support)
                support_distance = abs(
                    current_price - support_level) / support_level
                if support_distance <= 0.02:  # Dans les 2% du support
                    key_level = support_level
                    level_type = "support"
            except (ValueError, TypeError):
                pass

        if key_level is None and nearest_resistance is not None:
            try:
                resistance_level = float(nearest_resistance)
                resistance_distance = (
                    abs(current_price - resistance_level) / resistance_level
                )
                if resistance_distance <= 0.015:  # Dans les 1.5% de la résistance
                    key_level = resistance_level
                    level_type = "resistance"
            except (ValueError, TypeError):
                pass

        if key_level is None:
            return {"is_continuation": False, "score": 0.0, "indicators": []}

        # Vérifier que le prix a rebound du niveau (continuation pattern)
        price_vs_level = (current_price - key_level) / key_level

        if level_type == "support":
            # Pour support: prix doit être au-dessus du support (rebound)
            if price_vs_level > 0.002:  # Au moins 0.2% au-dessus
                continuation_score += 0.3
                continuation_indicators.append(
                    f"Rebound support {key_level:.2f} (+{price_vs_level*100:.2f}%)"
                )
            elif price_vs_level > -0.001:  # Juste au niveau
                continuation_score += 0.15
                continuation_indicators.append(
                    f"Test support {key_level:.2f} (hold)")
        # Pour résistance: prix peut être légèrement en-dessous (retest après
        # échec)
        elif -0.005 <= price_vs_level <= 0.002:  # Entre -0.5% et +0.2%
            continuation_score += 0.25
            continuation_indicators.append(
                f"Retest résistance {key_level:.2f} ({price_vs_level*100:.2f}%)"
            )

        # Vérifier momentum haussier pour continuation
        momentum_score = values.get("momentum_score")
        if momentum_score is not None:
            try:
                momentum_val = float(momentum_score)
                # Momentum haussier assoupli (médiane=50.02)
                if momentum_val >= 51:
                    continuation_score += 0.25  # Bonus augmenté
                    continuation_indicators.append(
                        f"Momentum haussier ({momentum_val:.1f})"
                    )
                elif momentum_val >= 49.8:  # Momentum neutre-haussier
                    continuation_score += 0.15  # Bonus augmenté
                    continuation_indicators.append(
                        f"Momentum neutre ({momentum_val:.1f})"
                    )
            except (ValueError, TypeError):
                pass

        # Vérifier RSI pour confirmation (pas en survente excessive)
        rsi_14 = values.get("rsi_14")
        if rsi_14 is not None:
            try:
                rsi_val = float(rsi_14)
                if 40 <= rsi_val <= 65:  # Zone favorable BUY élargie (P25-P75)
                    continuation_score += 0.18  # Bonus augmenté
                    continuation_indicators.append(
                        f"RSI favorable ({rsi_val:.1f})")
                elif rsi_val < 35:  # Trop survendu = risqué
                    continuation_score -= 0.1
            except (ValueError, TypeError):
                pass

        # Vérifier directional bias
        directional_bias = values.get("directional_bias")
        if directional_bias == "BULLISH":
            continuation_score += 0.25  # Bonus augmenté
            continuation_indicators.append("Bias haussier")
        elif directional_bias == "NEUTRAL":
            continuation_score += 0.12  # Nouveau bonus pour NEUTRAL
            continuation_indicators.append("Bias neutre favorable")
        elif directional_bias == "BEARISH":
            continuation_score -= 0.10  # Pénalité réduite

        return {
            "is_continuation": continuation_score
            >= 0.32,  # Seuil assoupli pour plus de BUY
            "score": continuation_score,
            "indicators": continuation_indicators,
            "key_level": key_level,
            "level_type": level_type,
        }

    def _create_continuation_buy_signal(
        self,
        values: dict[str, Any],
        current_price: float,
        continuation_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """Crée un signal BUY de continuation."""
        base_confidence = 0.60  # Base réduite pour BUY de continuation
        confidence_boost = continuation_analysis["score"] * 0.8

        key_level = continuation_analysis["key_level"]
        level_type = continuation_analysis["level_type"]

        reason = f"Continuation BUY après rebound {level_type} {key_level:.2f}"

        # Bonus volume
        volume_ratio = values.get("volume_ratio")
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= 1.5:
                    confidence_boost += 0.15
                    reason += f" + volume fort ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.0:
                    confidence_boost += 0.08
                    reason += f" + volume correct ({vol_ratio:.1f}x)"
            except (ValueError, TypeError):
                pass

        # Bonus confluence
        confluence_score = values.get("confluence_score")
        if confluence_score is not None:
            try:
                conf_val = float(confluence_score)
                if conf_val > 60:
                    confidence_boost += 0.12
                    reason += " + confluence forte"
                elif conf_val > 45:
                    confidence_boost += 0.06
                    reason += " + confluence modérée"
            except (ValueError, TypeError):
                pass

        # Calcul confidence finale
        confidence = min(
            1.0,
            self.calculate_confidence(
                base_confidence,
                1.0 +
                confidence_boost))

        # Seuil minimum pour BUY
        if confidence < 0.42:  # Seuil assoupli pour favoriser BUY
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Continuation BUY trop faible (conf: {confidence:.2f} < 0.42)",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "rejected_confidence": confidence,
                },
            }

        strength = self.get_strength_from_confidence(confidence)

        return {
            "side": "BUY",
            "confidence": confidence,
            "strength": strength,
            "reason": reason,
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "current_price": current_price,
                "key_level": key_level,
                "level_type": level_type,
                "continuation_score": continuation_analysis["score"],
                "continuation_indicators": continuation_analysis["indicators"],
                "volume_ratio": volume_ratio,
                "rsi_14": values.get("rsi_14"),
                "momentum_score": values.get("momentum_score"),
                "directional_bias": values.get("directional_bias"),
                "confluence_score": confluence_score,
                "signal_type": "continuation_buy",
            },
        }

    def validate_data(self) -> bool:
        """Valide que tous les indicateurs requis sont présents."""
        required_indicators = [
            "rsi_14",
            "volume_ratio",  # Supprimé nearest_resistance car on peut utiliser support aussi
        ]

        if not self.indicators:
            logger.warning(f"{self.name}: Aucun indicateur disponible")
            return False

        for indicator in required_indicators:
            if indicator not in self.indicators:
                logger.warning(
                    f"{self.name}: Indicateur manquant: {indicator}")
                return False
            if self.indicators[indicator] is None:
                logger.warning(f"{self.name}: Indicateur null: {indicator}")
                return False

        # Vérifier données OHLCV pour prix actuel
        if "close" not in self.data or not self.data["close"]:
            logger.warning(f"{self.name}: Données close manquantes")
            return False

        return True
