"""
RSI_Cross_Strategy - Stratégie basée sur les positions du RSI dans les zones de survente/surachat.
"""

import logging
from typing import Any

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class RSI_Cross_Strategy(BaseStrategy):
    """
    Stratégie utilisant le RSI et les indicateurs pré-calculés.

    Signaux générés:
    - BUY: RSI en zone de survente avec momentum et tendance favorables
    - SELL: RSI en zone de surachat avec momentum et tendance favorables
    """

    def __init__(self, symbol: str,
                 data: dict[str, Any], indicators: dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Seuils RSI ÉQUILIBRÉS pour crypto
        self.oversold_level = 35  # Équilibré crypto (38 -> 35)
        self.overbought_level = 65  # Équilibré crypto (62 -> 65)
        self.extreme_oversold = 25  # Zone extrême accessible (22 -> 25)
        self.extreme_overbought = 75  # Zone extrême accessible (78 -> 75)
        self.neutral_low = 42  # Zone neutre équilibrée (40 -> 42)
        self.neutral_high = 58  # Zone neutre équilibrée (60 -> 58)

    def _get_current_values(self) -> dict[str, float | None]:
        """Récupère les valeurs actuelles des indicateurs pré-calculés."""
        return {
            "rsi_14": self.indicators.get("rsi_14"),
            "rsi_21": self.indicators.get("rsi_21"),
            "signal_strength": self.indicators.get("signal_strength"),
            "momentum_score": self.indicators.get("momentum_score"),
            "trend_strength": self.indicators.get("trend_strength"),
            "directional_bias": self.indicators.get("directional_bias"),
            "confluence_score": self.indicators.get("confluence_score"),
            "pattern_confidence": self.indicators.get("pattern_confidence"),
            "volume_quality_score": self.indicators.get("volume_quality_score"),
            "volume_ratio": self.indicators.get("volume_ratio"),
            "adx_14": self.indicators.get("adx_14"),
            "volatility_regime": self.indicators.get("volatility_regime"),
            "market_regime": self.indicators.get("market_regime"),
            "trend_alignment": self.indicators.get("trend_alignment"),
            "atr_percentile": self.indicators.get("atr_percentile"),
        }

    def generate_signal(self) -> dict[str, Any]:
        """
        Génère un signal basé sur le RSI et les indicateurs pré-calculés.
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

        # Vérification des données essentielles
        rsi_14 = values["rsi_14"]
        if rsi_14 is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "RSI non disponible",
                "metadata": {"strategy": self.name},
            }

        signal_side = None
        reason = ""
        confidence_boost = 0.0

        # Logique de signal RSI simplifiée
        if rsi_14 <= self.oversold_level:
            # Zone de survente - signal BUY
            signal_side = "BUY"
            zone = "survente extrême" if rsi_14 <= self.extreme_oversold else "survente"
            reason = f"RSI ({rsi_14:.1f}) {zone}"

            # Bonus RSI survente
            if rsi_14 <= self.extreme_oversold:
                confidence_boost += 0.20
            else:
                confidence_boost += 0.15

        elif rsi_14 >= self.overbought_level:
            # Zone de surachat - signal SELL
            signal_side = "SELL"
            zone = ("surachat extrême" if rsi_14 >=
                    self.extreme_overbought else "surachat")
            reason = f"RSI ({rsi_14:.1f}) {zone}"

            # Bonus RSI surachat
            if rsi_14 >= self.extreme_overbought:
                confidence_boost += 0.20
            else:
                confidence_boost += 0.15

        if signal_side:
            # Base confidence réduite pour plus d'accessibilité
            base_confidence = 0.55

            # Ajustement avec momentum_score (format 0-100, 50=neutre) - CRYPTO
            # OPTIMISÉ
            momentum_score = values.get("momentum_score", 50)
            if momentum_score:
                try:
                    momentum_val = float(momentum_score)
                    # Pénalité momentum au lieu de rejet (momentum réel
                    # 49.5-50.5)
                    momentum_penalty = 0.0
                    if signal_side == "BUY" and momentum_val < 49.8:  # Sous moyenne
                        momentum_penalty = -0.10
                        reason += f" - momentum faible ({momentum_val:.1f})"
                    elif (
                        signal_side == "SELL" and momentum_val > 50.2
                    ):  # Au-dessus moyenne
                        momentum_penalty = -0.10
                        reason += f" - momentum fort ({momentum_val:.1f})"
                    else:
                        momentum_penalty = (
                            0.05  # Petit bonus si dans la bonne direction
                        )
                    # Bonus momentum avec seuils réalistes
                    if (signal_side == "BUY" and momentum_val > 50.3) or (
                        signal_side == "SELL" and momentum_val < 49.7
                    ):
                        confidence_boost += 0.12  # Augmenté car plus rare
                        reason += " + momentum favorable"

                    # Appliquer pénalité momentum
                    confidence_boost += momentum_penalty
                except (ValueError, TypeError):
                    pass

            # Ajustement avec trend_strength (VARCHAR:
            # weak/moderate/strong/very_strong/extreme)
            trend_strength = values.get("trend_strength")
            if trend_strength:
                trend_str = str(trend_strength).lower()
                if trend_str in ["extreme", "very_strong"]:
                    confidence_boost += 0.15
                    reason += f" et tendance {trend_str}"
                elif trend_str == "strong":
                    confidence_boost += 0.10
                    reason += f" et tendance {trend_str}"
                elif trend_str == "moderate":
                    confidence_boost += 0.05
                    reason += f" et tendance {trend_str}"

            # Pénalité bias au lieu de rejet
            directional_bias = values.get("directional_bias")
            if directional_bias:
                if (signal_side == "BUY" and directional_bias == "BEARISH") or (
                        signal_side == "SELL" and directional_bias == "BULLISH"):
                    confidence_boost -= 0.15  # Pénalité au lieu de rejet
                    reason += " - bias contraire"
                elif (signal_side == "BUY" and directional_bias == "BULLISH") or (
                    signal_side == "SELL" and directional_bias == "BEARISH"
                ):
                    confidence_boost += 0.15  # Bonus augmenté
                    reason += " + bias aligné"

            # Ajustement avec confluence_score (format 0-100)
            confluence_score = values.get("confluence_score", 0)
            if confluence_score:
                try:
                    confluence_val = float(confluence_score)
                    # Pénalité confluence faible au lieu de rejet
                    if confluence_val < 35:  # Seuil plus bas
                        confidence_boost -= 0.15
                        reason += f" - confluence très faible ({confluence_val:.0f})"
                    elif confluence_val < 45:
                        confidence_boost -= 0.08
                        reason += f" - confluence faible ({confluence_val:.0f})"
                    # Bonus confluence si bonne
                    if confluence_val > 60:
                        confidence_boost += 0.08
                        reason += " + confluence"
                except (ValueError, TypeError):
                    pass

            # Ajustement avec signal_strength pré-calculé (VARCHAR:
            # WEAK/MODERATE/STRONG)
            signal_strength_calc = values.get("signal_strength")
            if signal_strength_calc:
                sig_str = str(signal_strength_calc).upper()
                if sig_str == "STRONG":
                    confidence_boost += 0.1
                    reason += " + signal fort"
                elif sig_str == "MODERATE":
                    confidence_boost += 0.05
                    reason += " + signal modéré"

            # Confirmation avec RSI 21 pour multi-timeframe
            rsi_21 = values.get("rsi_21")
            if rsi_21 and ((signal_side == "BUY" and rsi_21 <= self.oversold_level) or (
                    signal_side == "SELL" and rsi_21 >= self.overbought_level)):
                confidence_boost += 0.1
                reason += " + RSI 21"

            # Bonus volume optionnel
            volume_quality = values.get("volume_quality_score", 50)
            if volume_quality:
                try:
                    vol_val = float(volume_quality)
                    if vol_val >= 75:
                        confidence_boost += 0.10
                        reason += " + volume"
                    elif vol_val >= 60:
                        confidence_boost += 0.05
                    elif vol_val < 40:
                        confidence_boost -= 0.08
                except (ValueError, TypeError):
                    pass

            # Vérifier divergence RSI 14/21 pour confirmation (AJUSTÉ)
            rsi_21 = values.get("rsi_21")
            if rsi_21:
                rsi_diff = abs(rsi_14 - rsi_21)
                if rsi_diff < 6:  # Tolérance augmentée (était 5)
                    confidence_boost += 0.15  # Bonus amélioré (était 0.12)
                    reason += " + multi-TF"
                elif rsi_diff > 18:  # Tolérance augmentée (était 15)
                    confidence_boost -= 0.06  # Pénalité réduite (était -0.08)
                    reason += " avec divergence multi-TF modérée"

            # ADX optionnel - pas de pénalité forte
            adx = values.get("adx_14")
            if adx and float(adx) > 30:  # Seulement bonus si ADX très fort
                confidence_boost += 0.08
                reason += " + ADX fort"
            elif adx and float(adx) < 18:  # Pénalité seulement si ADX très faible
                confidence_boost -= 0.05
                reason += " - ADX très faible"

            # Bonus volatilité optionnel
            atr_percentile = values.get("atr_percentile")
            if atr_percentile:
                try:
                    atr_val = float(atr_percentile)
                    if atr_val > 50:
                        confidence_boost += 0.05
                except (ValueError, TypeError):
                    pass

            # Market regime bonus optionnel
            market_regime = values.get("market_regime")
            if market_regime in ["TRENDING_BULL", "TRENDING_BEAR"]:
                confidence_boost += 0.08
                reason += " + trending"
            elif market_regime == "RANGING":
                confidence_boost -= 0.05  # Léger malus

            # Calcul confidence final
            confidence = min(
                base_confidence * (1 + confidence_boost), 1.0
            )  # Harmonisé avec autres strats

            # Seuil final très assoupli
            if confidence < 0.30:  # Très abaissé pour plus de signaux
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Signal RSI rejeté - confiance critique ({confidence:.2f} < 0.30)",
                    "metadata": {
                        "strategy": self.name,
                        "symbol": self.symbol,
                        "rejected_confidence": confidence,
                        "rsi_14": rsi_14,
                    },
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
                    "rsi_14": rsi_14,
                    "rsi_21": rsi_21,
                    "momentum_score": momentum_score,
                    "trend_strength": trend_strength,
                    "directional_bias": directional_bias,
                    "confluence_score": confluence_score,
                    "signal_strength_calc": signal_strength_calc,
                    "volume_quality_score": volume_quality,
                    "adx_14": adx,
                },
            }

        # RSI en zone neutre
        return {
            "side": None,
            "confidence": 0.0,
            "strength": "weak",
            "reason": f"RSI neutre ({rsi_14:.1f}) - seuils: {self.oversold_level}/{self.overbought_level}",
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "rsi_14": rsi_14,
            },
        }

    def validate_data(self) -> bool:
        """Validation minimale comme PPO."""
        if not super().validate_data():
            return False

        # Seulement RSI obligatoire
        required = ["rsi_14"]

        for indicator in required:
            if indicator not in self.indicators or self.indicators[indicator] is None:
                return False

        return True
