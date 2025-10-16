"""
Bollinger_Touch_Strategy - Stratégie basée sur les touches des bandes de Bollinger.
OPTIMISÉE POUR CRYPTO SPOT INTRADAY
"""

import logging
from typing import Any

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class Bollinger_Touch_Strategy(BaseStrategy):
    """
    Stratégie utilisant les touches des bandes de Bollinger pour détections de retournements.

    Signaux générés:
    - BUY: Prix touche la bande basse + indicateurs de retournement haussier
    - SELL: Prix touche la bande haute + indicateurs de retournement baissier
    """

    def __init__(self, symbol: str,
                 data: dict[str, Any], indicators: dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres Bollinger Bands - OPTIMISÉS CRYPTO SPOT (CORRIGÉ)
        # 0.4% de proximité (plus sensible, corrigé)
        self.touch_threshold = 0.004
        self.bb_position_extreme_buy = (
            0.15  # Position extrême basse (15% depuis le bas)
        )
        self.bb_position_extreme_sell = (
            0.85  # Position extrême haute (85% depuis le bas)
        )
        # Position très basse (5% depuis le bas)
        self.bb_position_very_low = 0.05
        # Position très haute (95% depuis le bas)
        self.bb_position_very_high = 0.95
        self.bb_width_min_pct = 0.6  # Largeur minimum 0.6% du prix (simplifié)
        self.max_bb_width_for_trade_pct = (
            8.0  # Ne pas trader si bandes > 8% du prix (volatilité extrême)
        )

        # Paramètres volume (utilisant volume_ratio disponible) - ASSOUPLIS
        # Volume minimum 25% de la moyenne (crypto adapté)
        self.min_volume_ratio = 0.25
        # Volume élevé pour confirmation (plus accessible)
        self.high_volume_ratio = 1.3

        # Paramètres RSI adaptés crypto - ASSOUPLIS POUR PLUS DE SIGNAUX
        self.rsi_oversold_strong = 25  # Survente forte (était 22)
        self.rsi_oversold = 35  # Survente standard (était 30)
        self.rsi_overbought = 65  # Surachat standard (était 70)
        self.rsi_overbought_strong = 75  # Surachat fort (était 78)

        # Paramètres Stochastic adaptés crypto - ASSOUPLIS
        self.stoch_oversold_strong = 15  # Survente forte (était 12)
        self.stoch_oversold = 25  # Survente standard (était 20)
        self.stoch_overbought = 75  # Surachat standard (était 80)
        self.stoch_overbought_strong = 85  # Surachat fort (était 88)

    def _get_current_values(self) -> dict[str, float | None]:
        """Récupère les valeurs actuelles des indicateurs Bollinger."""
        return {
            "bb_upper": self.indicators.get("bb_upper"),
            "bb_middle": self.indicators.get("bb_middle"),
            "bb_lower": self.indicators.get("bb_lower"),
            "bb_position": self.indicators.get("bb_position"),
            "bb_width": self.indicators.get("bb_width"),
            "bb_squeeze": self.indicators.get("bb_squeeze"),
            "bb_expansion": self.indicators.get("bb_expansion"),
            "bb_breakout_direction": self.indicators.get("bb_breakout_direction"),
            "rsi_14": self.indicators.get("rsi_14"),
            "stoch_k": self.indicators.get("stoch_k"),
            "stoch_d": self.indicators.get("stoch_d"),
            "williams_r": self.indicators.get("williams_r"),
            "momentum_score": self.indicators.get("momentum_score"),
            "volatility_regime": self.indicators.get("volatility_regime"),
            "signal_strength": self.indicators.get("signal_strength"),
            "confluence_score": self.indicators.get("confluence_score"),
            "volume_ratio": self.indicators.get("volume_ratio"),
            "trend_alignment": self.indicators.get("trend_alignment"),
            "ema_7": self.indicators.get("ema_7"),
            "ema_26": self.indicators.get("ema_26"),
            "macd_histogram": self.indicators.get("macd_histogram"),
        }

    def _get_current_price(self) -> float | None:
        """Récupère le prix actuel depuis les données OHLCV."""
        try:
            if self.data and "close" in self.data and self.data["close"]:
                return float(self.data["close"][-1])
        except (IndexError, ValueError, TypeError):
            pass
        return None

    def generate_signal(self) -> dict[str, Any]:
        """
        Génère un signal basé sur les touches des bandes de Bollinger.
        """
        if not self.validate_data():
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Données insuffisantes",
                "metadata": {"strategy": self.name},
            }

        values = self._get_current_values()
        current_price = self._get_current_price()

        # Vérification des indicateurs essentiels
        try:
            bb_upper = (float(values["bb_upper"])
                        if values["bb_upper"] is not None else None)
            bb_lower = (float(values["bb_lower"])
                        if values["bb_lower"] is not None else None)
            bb_middle = (
                float(
                    values["bb_middle"]) if values["bb_middle"] is not None else None)
            bb_position = (
                float(values["bb_position"])
                if values["bb_position"] is not None
                else None
            )
            bb_width = (float(values["bb_width"])
                        if values["bb_width"] is not None else None)
        except (ValueError, TypeError) as e:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Erreur conversion Bollinger: {e}",
                "metadata": {"strategy": self.name},
            }

        if bb_upper is None or bb_lower is None or current_price is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Bollinger Bands ou prix non disponibles",
                "metadata": {"strategy": self.name},
            }

        # Vérification du volume minimum
        volume_ratio = values.get("volume_ratio")
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio < self.min_volume_ratio:
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Volume insuffisant ({vol_ratio:.2f}x < {self.min_volume_ratio}x)",
                        "metadata": {
                            "strategy": self.name,
                            "volume_ratio": vol_ratio},
                    }
            except (ValueError, TypeError):
                pass

        # Vérification bb_width corrigée avec current_price
        if bb_width is not None and current_price is not None:
            bb_width_pct = (bb_width / current_price) * \
                100  # Plus robuste pour crypto
            if bb_width_pct < self.bb_width_min_pct:
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"BB squeeze trop serré ({bb_width_pct:.2f}% < {self.bb_width_min_pct}%)",
                    "metadata": {
                        "strategy": self.name,
                        "bb_width_pct": bb_width_pct},
                }
        else:
            bb_width_pct = None

        # CORRECTION MAJEURE: Calcul des distances aux bandes (méthode
        # cohérente)
        distance_to_upper = abs(current_price - bb_upper) / current_price
        distance_to_lower = abs(current_price - bb_lower) / current_price

        signal_side = None
        reason = ""
        base_confidence = 0.65  # Base confidence raisonnable
        confidence_boost = 0.0
        touch_type = None

        # Détection des touches de bandes avec position - CORRIGÉ
        is_touching_upper = distance_to_upper <= self.touch_threshold
        is_touching_lower = distance_to_lower <= self.touch_threshold

        # Position dans les bandes pour confirmation - ÉTENDUE AUX VALEURS
        # NÉGATIVES/SUPÉRIEURES
        is_extreme_high = (
            bb_position is not None and bb_position >= self.bb_position_extreme_sell)
        is_extreme_low = (
            bb_position is not None and bb_position <= self.bb_position_extreme_buy)
        is_very_high = (bb_position is not None and bb_position >=
                        self.bb_position_very_high)
        is_very_low = (bb_position is not None and bb_position <=
                       self.bb_position_very_low)

        # SIGNAL BUY - LOGIQUE SIMPLIFIÉE
        if (is_touching_lower or is_extreme_low or is_very_low) and (
            bb_width_pct is None or bb_width_pct >= self.bb_width_min_pct
        ):
            signal_side = "BUY"
            touch_type = "lower_band"

            if is_very_low:  # Position en dehors de la bande basse (très fort)
                reason = f"Position très basse {bb_position:.3f} (hors bande)"
                confidence_boost += 0.25
            elif is_touching_lower and is_extreme_low:
                reason = f"Touche bande basse confirmée {bb_lower:.2f} (pos: {bb_position:.3f})"
                confidence_boost += 0.20
            elif is_touching_lower:
                reason = (
                    f"Touche bande basse {bb_lower:.2f} (dist: {distance_to_lower:.3f})"
                )
                confidence_boost += 0.12
            elif is_extreme_low:
                reason = f"Position extrême basse ({bb_position:.3f})"
                confidence_boost += 0.10

        # SIGNAL SELL - LOGIQUE SIMPLIFIÉE
        elif (is_touching_upper or is_extreme_high or is_very_high) and (
            bb_width_pct is None or bb_width_pct >= self.bb_width_min_pct
        ):
            signal_side = "SELL"
            touch_type = "upper_band"

            # Position en dehors de la bande haute (très fort)
            if is_very_high:
                reason = f"Position très haute {bb_position:.3f} (hors bande)"
                confidence_boost += 0.25
            elif is_touching_upper and is_extreme_high:
                reason = f"Touche bande haute confirmée {bb_upper:.2f} (pos: {bb_position:.3f})"
                confidence_boost += 0.20
            elif is_touching_upper:
                reason = (
                    f"Touche bande haute {bb_upper:.2f} (dist: {distance_to_upper:.3f})"
                )
                confidence_boost += 0.12
            elif is_extreme_high:
                reason = f"Position extrême haute ({bb_position:.3f})"
                confidence_boost += 0.10

        # Pas de touche détectée - DIAGNOSTIC AMÉLIORÉ
        if signal_side is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": (
                    f"Pas de setup BB (pos: {bb_position:.3f}, dist_up: {distance_to_upper:.4f}, dist_low: {distance_to_lower:.4f}, width: {bb_width_pct:.2f}%)"
                    if bb_width_pct
                    else f"Pas de setup BB (pos: {bb_position:.3f}, distances: {distance_to_upper:.4f}/{distance_to_lower:.4f})"
                ),
                "metadata": {
                    "strategy": self.name,
                    "bb_position": bb_position,
                    "distance_to_upper": distance_to_upper,
                    "distance_to_lower": distance_to_lower,
                    "bb_width_pct": bb_width_pct,
                    "is_touching_upper": is_touching_upper,
                    "is_touching_lower": is_touching_lower,
                    "is_extreme_high": is_extreme_high,
                    "is_extreme_low": is_extreme_low,
                    "is_very_high": is_very_high,
                    "is_very_low": is_very_low,
                },
            }

        # === CONFIRMATIONS AVEC OSCILLATEURS ===

        # RSI - Seuils adaptés crypto avec rejets contradictoires
        rsi_14 = values.get("rsi_14")
        rsi = None
        if rsi_14 is not None:
            try:
                rsi = float(rsi_14)

                # Rejets RSI contradictoires
                if signal_side == "BUY" and rsi > 55:
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Rejet BUY: RSI trop haut ({rsi:.1f}) pour retournement",
                        "metadata": {
                            "strategy": self.name,
                            "rsi": rsi},
                    }
                if signal_side == "SELL" and rsi < 45:
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Rejet SELL: RSI trop bas ({rsi:.1f}) pour retournement",
                        "metadata": {
                            "strategy": self.name,
                            "rsi": rsi},
                    }

                # Confirmations RSI
                if signal_side == "BUY":
                    if rsi <= self.rsi_oversold_strong:
                        confidence_boost += 0.20
                        reason += f" + RSI survente forte ({rsi:.1f})"
                    elif rsi <= self.rsi_oversold:
                        confidence_boost += 0.10
                        reason += f" + RSI survente ({rsi:.1f})"

                elif signal_side == "SELL":
                    if rsi >= self.rsi_overbought_strong:
                        confidence_boost += 0.20
                        reason += f" + RSI surachat fort ({rsi:.1f})"
                    elif rsi >= self.rsi_overbought:
                        confidence_boost += 0.10
                        reason += f" + RSI surachat ({rsi:.1f})"
            except (ValueError, TypeError):
                pass

        # Stochastic - Seuils adaptés crypto avec rejets contradictoires
        stoch_k = values.get("stoch_k")
        stoch_d = values.get("stoch_d")
        k = None
        d = None
        if stoch_k is not None and stoch_d is not None:
            try:
                k = float(stoch_k)
                d = float(stoch_d)

                # Rejets Stochastic contradictoires
                if signal_side == "BUY" and k > 50 and d > 50:
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Rejet BUY: Stoch trop haut ({k:.1f}/{d:.1f}) pour retournement",
                        "metadata": {
                            "strategy": self.name,
                            "stoch_k": k,
                            "stoch_d": d},
                    }
                if signal_side == "SELL" and k < 50 and d < 50:
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Rejet SELL: Stoch trop bas ({k:.1f}/{d:.1f}) pour retournement",
                        "metadata": {
                            "strategy": self.name,
                            "stoch_k": k,
                            "stoch_d": d},
                    }

                # Confirmations Stochastic
                if signal_side == "BUY":
                    if (
                        k <= self.stoch_oversold_strong
                        and d <= self.stoch_oversold_strong
                    ):
                        confidence_boost += 0.15
                        reason += f" + Stoch survente forte ({k:.1f}/{d:.1f})"
                    elif k <= self.stoch_oversold and d <= self.stoch_oversold:
                        confidence_boost += 0.08
                        reason += f" + Stoch survente ({k:.1f}/{d:.1f})"

                elif signal_side == "SELL":
                    if (
                        k >= self.stoch_overbought_strong
                        and d >= self.stoch_overbought_strong
                    ):
                        confidence_boost += 0.15
                        reason += f" + Stoch surachat fort ({k:.1f}/{d:.1f})"
                    elif k >= self.stoch_overbought and d >= self.stoch_overbought:
                        confidence_boost += 0.08
                        reason += f" + Stoch surachat ({k:.1f}/{d:.1f})"
            except (ValueError, TypeError):
                pass

        # Volume confirmation
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= self.high_volume_ratio:
                    confidence_boost += 0.12
                    reason += f" + volume élevé ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.0:
                    confidence_boost += 0.05
                    reason += f" + volume normal ({vol_ratio:.1f}x)"
            except (ValueError, TypeError):
                pass

        # Calcul final de la confiance avec clamp
        confidence = min(
            1.0, self.calculate_confidence(
                base_confidence, 1 + confidence_boost))

        # Filtre final - SEUIL ÉQUILIBRÉ
        if confidence < 0.45:  # Seuil minimum équilibré
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Signal rejeté - confiance insuffisante ({confidence:.2f} < 0.45)",
                "metadata": {
                    "strategy": self.name,
                    "rejected_signal": signal_side,
                    "final_confidence": confidence,
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
                "current_price": current_price,
                "bb_upper": bb_upper,
                "bb_middle": bb_middle,
                "bb_lower": bb_lower,
                "bb_position": bb_position,
                "bb_width": bb_width,
                "bb_width_pct": bb_width_pct,
                "touch_type": touch_type,
                "distance_to_upper": distance_to_upper,
                "distance_to_lower": distance_to_lower,
                "rsi_14": rsi,
                "stoch_k": k,
                "stoch_d": d,
                "volume_ratio": volume_ratio,
            },
        }

    def validate_data(self) -> bool:
        """Valide que tous les indicateurs Bollinger requis sont présents."""
        if not super().validate_data():
            return False

        required = ["bb_upper", "bb_lower"]

        for indicator in required:
            if indicator not in self.indicators:
                logger.warning(
                    f"{self.name}: Indicateur manquant: {indicator}")
                return False
            if self.indicators[indicator] is None:
                logger.warning(f"{self.name}: Indicateur null: {indicator}")
                return False

        # Vérifier aussi qu'on a des données de prix
        if not self.data or "close" not in self.data or not self.data["close"]:
            logger.warning(f"{self.name}: Données de prix manquantes")
            return False

        return True
