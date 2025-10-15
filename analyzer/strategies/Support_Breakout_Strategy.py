"""
Support_Breakout_Strategy - Version ULTRA SIMPLIFIÉE pour crypto spot.
Détecte breakouts support/résistance avec logique bidirectionnelle cohérente.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class Support_Breakout_Strategy(BaseStrategy):
    """
    Stratégie ULTRA SIMPLIFIÉE détectant breakouts support/résistance.

    Principe simplifié :
    - Support breakout (prix < support) → SELL
    - Resistance breakout (prix > résistance) → BUY
    - Confirmation volume + momentum
    - Seuil unique 0.3% pour détecter breakout

    Signaux générés:
    - SELL: Cassure support + confirmations
    - BUY: Cassure résistance + confirmations
    """

    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)

        # Paramètres DURCIS pour vrais breakouts
        self.breakout_threshold = 0.005  # 0.5% seuil durci (vs bruit 0.3%)
        self.base_confidence = 0.65  # Confiance élevée maintenue

    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère seulement les indicateurs essentiels."""
        return {
            "nearest_support": self.indicators.get("nearest_support"),
            "nearest_resistance": self.indicators.get("nearest_resistance"),
            "momentum_score": self.indicators.get("momentum_score"),
            "volume_ratio": self.indicators.get("volume_ratio"),
            "directional_bias": self.indicators.get("directional_bias"),
            "market_regime": self.indicators.get("market_regime"),
            "confluence_score": self.indicators.get("confluence_score"),
            "bb_lower": self.indicators.get("bb_lower"),
            "bb_upper": self.indicators.get("bb_upper"),
        }

    def _detect_breakout(
        self, values: Dict[str, Any], current_price: float
    ) -> Dict[str, Any]:
        """Détecte breakout support (SELL) ou résistance (BUY)."""

        # Support breakout (SELL)
        nearest_support = values.get("nearest_support") or values.get("bb_lower")
        if nearest_support is not None:
            try:
                support_level = float(nearest_support)
                if current_price < support_level * (1 - self.breakout_threshold):
                    breakdown_distance = (support_level - current_price) / support_level
                    return {
                        "is_breakout": True,
                        "signal_side": "SELL",
                        "level": support_level,
                        "distance_pct": breakdown_distance * 100,
                        "reason": f"Support breakout {support_level:.4f} ({breakdown_distance*100:.1f}%)",
                    }
            except (ValueError, TypeError):
                pass

        # Resistance breakout (BUY)
        nearest_resistance = values.get("nearest_resistance") or values.get("bb_upper")
        if nearest_resistance is not None:
            try:
                resistance_level = float(nearest_resistance)
                if current_price > resistance_level * (1 + self.breakout_threshold):
                    breakout_distance = (
                        current_price - resistance_level
                    ) / resistance_level
                    return {
                        "is_breakout": True,
                        "signal_side": "BUY",
                        "level": resistance_level,
                        "distance_pct": breakout_distance * 100,
                        "reason": f"Resistance breakout {resistance_level:.4f} ({breakout_distance*100:.1f}%)",
                    }
            except (ValueError, TypeError):
                pass

        return {"is_breakout": False, "reason": "Pas de breakout détecté"}

    def generate_signal(self) -> Dict[str, Any]:
        """Version ULTRA SIMPLIFIÉE pour crypto spot breakouts."""

        # Validation minimale
        if not self.validate_data():
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Données insuffisantes",
                "metadata": {"strategy": self.name},
            }

        values = self._get_current_values()
        confidence_boost = 0.0

        # Prix actuel
        if not ("close" in self.data and self.data["close"]):
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Prix non disponible",
                "metadata": {"strategy": self.name},
            }

        try:
            current_price = float(self.data["close"][-1])
        except (IndexError, ValueError, TypeError):
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Prix invalide",
                "metadata": {"strategy": self.name},
            }

        # Détection breakout simplifiée
        breakout_analysis = self._detect_breakout(values, current_price)

        if not breakout_analysis["is_breakout"]:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": breakout_analysis["reason"],
                "metadata": {"strategy": self.name},
            }

        signal_side = breakout_analysis["signal_side"]
        reason = breakout_analysis["reason"]

        # REJETS CRITIQUES - cohérence momentum/bias
        momentum_score = values.get("momentum_score", 50)
        directional_bias = values.get("directional_bias")

        try:
            momentum_val = float(momentum_score)
        except (ValueError, TypeError):
            momentum_val = 50

        # Rejet momentum DURCI - plus tranchant
        if signal_side == "BUY" and momentum_val < 50:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Rejet BUY: momentum trop faible ({momentum_val:.0f})",
                "metadata": {"strategy": self.name},
            }
        elif signal_side == "SELL" and momentum_val > 50:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Rejet SELL: momentum trop fort ({momentum_val:.0f})",
                "metadata": {"strategy": self.name},
            }

        # Rejet si incohérence signal/bias + BONUS si aligné
        if (signal_side == "BUY" and directional_bias == "BEARISH") or (
            signal_side == "SELL" and directional_bias == "BULLISH"
        ):
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Rejet {signal_side}: bias contradictoire ({directional_bias})",
                "metadata": {"strategy": self.name},
            }
        elif (signal_side == "BUY" and directional_bias == "BULLISH") or (
            signal_side == "SELL" and directional_bias == "BEARISH"
        ):
            confidence_boost += 0.10
            reason += f" + bias aligné ({directional_bias})"

        # Bonus simples

        # Volume DURCI - évite breakouts neutres
        volume_ratio = values.get("volume_ratio", 1.0)
        try:
            vol_ratio = float(volume_ratio)
            if vol_ratio >= 1.8:  # Seuil plus strict
                confidence_boost += 0.15
                reason += f" + volume fort ({vol_ratio:.1f}x)"
            elif vol_ratio >= 1.5:
                confidence_boost += 0.08
                reason += f" + volume ({vol_ratio:.1f}x)"
            elif vol_ratio < 1.1:  # Rejet durci (vs 1.0)
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Rejet: volume trop faible ({vol_ratio:.1f}x)",
                    "metadata": {"strategy": self.name},
                }
        except (ValueError, TypeError):
            pass

        # Confluence avec rejet
        confluence_score = values.get("confluence_score", 0)
        try:
            conf_val = float(confluence_score)
        except (ValueError, TypeError):
            conf_val = 0

        if conf_val < 40:  # Rejet si confluence trop faible
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Rejet: confluence insuffisante ({conf_val})",
                "metadata": {"strategy": self.name, "confluence_score": conf_val},
            }
        elif conf_val >= 75:  # Confluence affinée
            confidence_boost += 0.10
            reason += f" + confluence excellente ({conf_val:.0f})"
        elif conf_val >= 60:
            confidence_boost += 0.05
            reason += f" + confluence ({conf_val:.0f})"

        # Enrichir reason avec détails systématiques
        reason += f" (momentum={momentum_val:.0f}, conf={conf_val:.0f})"

        # Market regime avec rejets contradictoires + BONUS aligné
        market_regime = values.get("market_regime")
        if (signal_side == "BUY" and market_regime == "TRENDING_BEAR") or (
            signal_side == "SELL" and market_regime == "TRENDING_BULL"
        ):
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Rejet {signal_side}: régime contradictoire ({market_regime})",
                "metadata": {"strategy": self.name, "market_regime": market_regime},
            }
        elif (signal_side == "BUY" and market_regime == "TRENDING_BULL") or (
            signal_side == "SELL" and market_regime == "TRENDING_BEAR"
        ):
            confidence_boost += 0.08
            reason += f" + régime aligné ({market_regime})"

        # Calcul final
        confidence = min(1.0, self.base_confidence * (1 + confidence_boost))
        strength = self.get_strength_from_confidence(confidence)

        return {
            "side": signal_side,
            "confidence": confidence,
            "strength": strength,
            "reason": reason,
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "breakout_level": breakout_analysis["level"],
                "breakout_distance_pct": breakout_analysis["distance_pct"],
                "momentum_score": momentum_val,
                "volume_ratio": volume_ratio,
                "directional_bias": directional_bias,
                "market_regime": market_regime,
                "confluence_score": confluence_score,
                "base_confidence": self.base_confidence,
                "confidence_boost": confidence_boost,
            },
        }

    def validate_data(self) -> bool:
        """Validation ULTRA SIMPLIFIÉE - seulement essentiels."""
        if not super().validate_data():
            return False

        # Au moins un niveau support OU résistance
        has_level = any(
            self.indicators.get(ind) is not None
            for ind in ["nearest_support", "nearest_resistance", "bb_lower", "bb_upper"]
        )

        if not has_level:
            logger.warning(f"{self.name}: Aucun niveau support/résistance disponible")
            return False

        # Seulement momentum requis
        if (
            "momentum_score" not in self.indicators
            or self.indicators["momentum_score"] is None
        ):
            logger.warning(f"{self.name}: momentum_score manquant")
            return False

        return True
