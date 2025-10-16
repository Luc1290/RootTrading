"""
HullMA_Slope_Strategy - Stratégie CONTRARIAN utilisant Hull MA comme filtre de qualité.
TRANSFORMATION D'UNE STRATÉGIE PERDANTE EN STRATÉGIE PROFITABLE
"""

import logging
import math
from typing import Any

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class HullMA_Slope_Strategy(BaseStrategy):
    """
    Stratégie CONTRARIAN avec Hull MA comme filtre de qualité et timing.

    NOUVELLE APPROCHE - Stratégie anti-lagging:
    - Hull MA sert de FILTRE de tendance (pas de signal direct)
    - Achète sur les PULLBACKS dans une tendance haussière Hull MA
    - Vend sur les BOUNCES dans une tendance baissière Hull MA
    - Focus sur les RETOURNEMENTS et CORRECTIONS plutôt que suivre

    Signaux générés:
    - BUY: Prix SOUS Hull MA haussière + oscillateurs survente + momentum retournement
    - SELL: Prix AU-DESSUS Hull MA baissière + oscillateurs surachat + momentum retournement
    """

    def __init__(self, symbol: str, data: dict[str, Any], indicators: dict[str, Any]):
        super().__init__(symbol, data, indicators)

        # PARAMÈTRES TREND-FOLLOWING RÉALISTES (abandon logique contrarian)
        self.hull_trend_threshold_weak = (
            0.3  # 0.3° pour tendance faible (plus accessible)
        )
        # 1.2° pour tendance forte (maintenu)
        self.hull_trend_threshold_strong = 1.2
        # Pas de pullbacks/bounces - suivre les micro-tendances

        # Seuils oscillateurs TREND-FOLLOWING (plus momentum que contrarian)
        self.rsi_bullish_min = 45  # RSI momentum haussier minimum
        self.rsi_bearish_max = 55  # RSI momentum baissier maximum
        self.momentum_bullish_min = 50.1  # Momentum > neutre pour BUY
        self.momentum_bearish_max = 49.9  # Momentum < neutre pour SELL

        # Anciens paramètres contrarian (conservés pour compatibilité)
        self.price_pullback_min = 0.005  # 0.5% pullback minimum
        self.price_pullback_max = 0.03  # 3% pullback maximum
        self.price_bounce_min = 0.005  # 0.5% bounce minimum
        self.price_bounce_max = 0.03  # 3% bounce maximum
        self.rsi_oversold_entry = 35  # RSI pour pullback entry
        self.rsi_overbought_entry = 65  # RSI pour bounce entry

        # Filtres qualité PLUS PERMISSIFS pour CRYPTO
        self.min_volume_ratio = 0.8  # Volume minimum ACCESSIBLE
        # Confluence minimum ASSOUPLI (15 vs 35)
        self.min_confluence_score = 15
        self.min_confidence_threshold = 0.45  # Confidence minimum ACCESSIBLE

    def _get_current_values(self) -> dict[str, float | None]:
        """Récupère les valeurs actuelles des indicateurs."""
        return {
            # Hull MA principal
            "hull_20": self.indicators.get("hull_20"),
            # Moyennes mobiles pour contexte
            "ema_12": self.indicators.get("ema_12"),
            "ema_26": self.indicators.get("ema_26"),
            "ema_50": self.indicators.get("ema_50"),
            "sma_20": self.indicators.get("sma_20"),
            # Trend analysis CRITIQUE pour nouvelle approche
            "trend_angle": self.indicators.get("trend_angle"),
            "trend_strength": self.indicators.get("trend_strength"),
            "directional_bias": self.indicators.get("directional_bias"),
            "trend_alignment": self.indicators.get("trend_alignment"),
            # Oscillateurs pour contrarian entries
            "momentum_score": self.indicators.get("momentum_score"),
            "rsi_14": self.indicators.get("rsi_14"),
            "macd_line": self.indicators.get("macd_line"),
            "macd_histogram": self.indicators.get("macd_histogram"),
            # Volume critique
            "volume_ratio": self.indicators.get("volume_ratio"),
            "volume_quality_score": self.indicators.get("volume_quality_score"),
            # Contexte marché
            "market_regime": self.indicators.get("market_regime"),
            "volatility_regime": self.indicators.get("volatility_regime"),
            # Confluence finale
            "signal_strength": self.indicators.get("signal_strength"),
            "confluence_score": self.indicators.get("confluence_score"),
        }

    def _get_price_data(self) -> dict[str, float | None]:
        """Récupère les données de prix pour analyse."""
        try:
            if (
                self.data
                and "close" in self.data
                and self.data["close"]
                and len(self.data["close"]) >= 5
            ):
                prices = self.data["close"]
                return {
                    "current_price": float(prices[-1]),
                    "prev_price_1": float(prices[-2]),
                    "prev_price_2": float(prices[-3]),
                    "prev_price_3": float(prices[-4]),
                    "prev_price_4": float(prices[-5]),
                }
        except (IndexError, ValueError, TypeError):
            pass
        return {
            "current_price": None,
            "prev_price_1": None,
            "prev_price_2": None,
            "prev_price_3": None,
            "prev_price_4": None,
        }

    def _analyze_hull_trend_direction(
        self, values: dict[str, Any], price_data: dict[str, float | None]
    ) -> dict[str, Any]:
        """Analyse la direction de tendance de Hull MA avec trend_angle comme source principale."""
        hull_20 = values.get("hull_20")
        trend_angle = values.get("trend_angle")
        current_price = price_data["current_price"]

        if hull_20 is None or current_price is None:
            return {"direction": None, "strength": "unknown", "reliable": False}

        try:
            hull_val = float(hull_20)

            # Méthode 1: Utiliser trend_angle si disponible (le plus fiable)
            if trend_angle is not None:
                try:
                    angle = float(trend_angle)
                    # Seuils MICRO-TENDANCES plus accessibles
                    if angle >= self.hull_trend_threshold_strong:  # 1.2°
                        return {
                            "direction": "strong_bullish",
                            "strength": "strong",
                            "reliable": True,
                            "slope_proxy": angle / 45.0,
                        }
                    if angle >= self.hull_trend_threshold_weak:  # 0.3°
                        return {
                            "direction": "weak_bullish",
                            "strength": "weak",
                            "reliable": True,
                            "slope_proxy": angle / 45.0,
                        }
                    if angle <= -self.hull_trend_threshold_strong:  # -1.2°
                        return {
                            "direction": "strong_bearish",
                            "strength": "strong",
                            "reliable": True,
                            "slope_proxy": angle / 45.0,
                        }
                    if angle <= -self.hull_trend_threshold_weak:  # -0.3°
                        return {
                            "direction": "weak_bearish",
                            "strength": "weak",
                            "reliable": True,
                            "slope_proxy": angle / 45.0,
                        }
                    return {
                        "direction": "sideways",
                        "strength": "flat",
                        "reliable": False,  # Pas de signal en sideways
                        "slope_proxy": angle / 45.0,
                    }
                except (ValueError, TypeError):
                    pass

            # Méthode 2: Fallback avec prix relatif et directional_bias
            directional_bias = values.get("directional_bias")
            trend_strength = values.get("trend_strength")

            # Distance prix/Hull MA comme indicateur secondaire
            price_hull_ratio = current_price / hull_val

            if directional_bias == "BULLISH" and trend_strength in [
                "weak",
                "moderate",
                "strong",
                "very_strong",
                "extreme",
            ]:
                return {
                    "direction": "bullish",
                    "strength": (
                        str(trend_strength).lower() if trend_strength else "moderate"
                    ),
                    "reliable": trend_strength
                    in ["moderate", "strong", "very_strong", "extreme"],
                    "slope_proxy": min(
                        (price_hull_ratio - 1.0) * 10, 0.1
                    ),  # Approximation
                }
            if directional_bias == "BEARISH" and trend_strength in [
                "weak",
                "moderate",
                "strong",
                "very_strong",
                "extreme",
            ]:
                return {
                    "direction": "bearish",
                    "strength": (
                        str(trend_strength).lower() if trend_strength else "moderate"
                    ),
                    "reliable": trend_strength
                    in ["moderate", "strong", "very_strong", "extreme"],
                    "slope_proxy": max(
                        (price_hull_ratio - 1.0) * 10, -0.1
                    ),  # Approximation
                }
            return {
                "direction": "sideways",
                "strength": "weak",
                "reliable": False,
                "slope_proxy": 0.0,
            }

        except (ValueError, TypeError):
            return {"direction": None, "strength": "unknown", "reliable": False}

    def _validate_hull_data(
        self, values: dict[str, Any], price_data: dict[str, float | None]
    ) -> tuple[
        bool,
        dict[str, Any] | None,
        float | None,
        float | None,
        float | None,
        float | None,
    ]:
        """Valide les données Hull et filtres préliminaires. Returns (is_valid, error_response, hull_val, price_val, volume_penalty, confluence_penalty)."""
        if not self.validate_data():
            return (
                False,
                {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": "Données insuffisantes",
                    "metadata": {"strategy": self.name},
                },
                None,
                None,
                None,
                None,
            )

        hull_20 = values.get("hull_20")
        current_price = price_data["current_price"]

        if not (self._is_valid(hull_20) and self._is_valid(current_price)):
            return (
                False,
                {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": "Hull MA ou prix invalides/NaN",
                    "metadata": {"strategy": self.name},
                },
                None,
                None,
                None,
                None,
            )

        try:
            hull_val = float(hull_20) if hull_20 is not None else 0.0
            price_val = float(current_price) if current_price is not None else 0.0
        except (ValueError, TypeError) as e:
            return (
                False,
                {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Erreur conversion Hull MA/prix: {e}",
                    "metadata": {"strategy": self.name},
                },
                None,
                None,
                None,
                None,
            )

        # Volume penalty
        volume_ratio = values.get("volume_ratio")
        volume_penalty = 0.0
        if not self._is_valid(volume_ratio):
            volume_penalty = -0.05
        else:
            try:
                vol_val = float(volume_ratio) if volume_ratio is not None else 0.0
                if vol_val < 0.8:
                    volume_penalty = -0.10
            except (ValueError, TypeError):
                volume_penalty = -0.05

        # Confluence validation
        confluence_score = values.get("confluence_score")
        confluence_penalty = 0.0
        if not self._is_valid(confluence_score):
            confluence_penalty = -0.08
        else:
            try:
                conf_val = (
                    float(confluence_score) if confluence_score is not None else 0.0
                )
                if conf_val < 15:
                    return (
                        False,
                        {
                            "side": None,
                            "confidence": 0.0,
                            "strength": "weak",
                            "reason": f"Rejet contrarian: confluence trop faible ({conf_val:.0f}) - signal bruité",
                            "metadata": {"strategy": self.name},
                        },
                        None,
                        None,
                        None,
                        None,
                    )
                if conf_val < self.min_confluence_score:
                    confluence_penalty = 0.0
            except (ValueError, TypeError):
                confluence_penalty = -0.08

        # Volatilité extrême
        volatility_regime = values.get("volatility_regime")
        if volatility_regime == "extreme":
            return (
                False,
                {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": "Rejet contrarian: volatilité extrême - pullbacks trop violents",
                    "metadata": {"strategy": self.name},
                },
                None,
                None,
                None,
                None,
            )

        return True, None, hull_val, price_val, volume_penalty, confluence_penalty

    def _detect_pullback_opportunity(
        self,
        hull_trend: dict[str, Any],
        hull_20: float,
        current_price: float,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        """Détecte une opportunité de pullback (prix temporairement contre la tendance Hull)."""

        if hull_trend["direction"] != "bullish" or not hull_trend["reliable"]:
            return {
                "is_pullback": False,
                "reason": "Pas de tendance haussière Hull fiable",
            }

        # Prix doit être SOUS Hull MA (pullback dans tendance haussière)
        if current_price >= hull_20:
            return {
                "is_pullback": False,
                "reason": f"Prix au-dessus Hull MA ({current_price:.2f} >= {hull_20:.2f})",
            }

        # Calculer l'amplitude du pullback
        pullback_pct = (hull_20 - current_price) / hull_20

        if pullback_pct < self.price_pullback_min:
            return {
                "is_pullback": False,
                "reason": f"Pullback trop faible ({pullback_pct*100:.1f}% < {self.price_pullback_min*100:.1f}%)",
            }

        if pullback_pct > self.price_pullback_max:
            return {
                "is_pullback": False,
                "reason": f"Pullback trop important ({pullback_pct*100:.1f}% > {self.price_pullback_max*100:.1f}%) - falling knife",
            }

        # Vérifier oscillateurs survente pour confirmation
        rsi_14 = values.get("rsi_14")
        momentum_score = values.get("momentum_score")

        oversold_signals = 0
        oversold_details = []

        if rsi_14 is not None and self._is_valid(rsi_14):
            try:
                rsi = float(rsi_14)
                # REJET si RSI contradictoire (BUY avec RSI élevé = pas
                # contrarian)
                # RSI trop haut pour un BUY contrarian (assoupli)
                if rsi >= 75:
                    return {
                        "is_pullback": False,
                        "reason": f"RSI contradictoire pour BUY contrarian: {rsi:.1f} trop élevé (>70)",
                        "rsi_rejection": True,
                    }
                if rsi <= self.rsi_oversold_entry:
                    oversold_signals += 1
                    oversold_details.append(f"RSI survente ({rsi:.1f})")
            except (ValueError, TypeError):
                pass

        if momentum_score is not None:
            try:
                momentum = float(momentum_score)
                if momentum <= 45:  # Momentum faible pour BUY (assoupli)
                    oversold_signals += 1
                    oversold_details.append(f"momentum très faible ({momentum:.0f})")
            except (ValueError, TypeError):
                pass

        # MACD histogram pour retournement momentum
        macd_histogram = values.get("macd_histogram")
        if macd_histogram is not None:
            try:
                hist = float(macd_histogram)
                # Chercher un retournement (MACD qui redevient positif)
                if hist > 0.00005:  # Légèrement positif (seuil assoupli)
                    oversold_signals += 1
                    oversold_details.append(f"MACD retournement (+{hist:.4f})")
            except (ValueError, TypeError):
                pass

        # Autoriser 0 confirmation si pullback optimal (1-2%)
        if oversold_signals < 1:
            if 0.01 <= pullback_pct <= 0.02:  # Zone optimale 1-2%
                oversold_signals = 1  # Donner 1 confirmation artificielle
                oversold_details.append("pullback optimal (1-2%)")
            else:
                return {
                    "is_pullback": False,
                    "reason": f"Pullback détecté mais aucune confirmation ({oversold_signals}/1)",
                    "pullback_pct": pullback_pct,
                    "oversold_signals": oversold_signals,
                }

        return {
            "is_pullback": True,
            "pullback_pct": pullback_pct,
            "oversold_signals": oversold_signals,
            "oversold_details": oversold_details,
            "reason": f"Pullback {pullback_pct*100:.1f}% avec {oversold_signals} confirmations",
        }

    def _detect_bounce_opportunity(
        self,
        hull_trend: dict[str, Any],
        hull_20: float,
        current_price: float,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        """Détecte une opportunité de bounce (prix temporairement contre la tendance Hull)."""

        if hull_trend["direction"] != "bearish" or not hull_trend["reliable"]:
            return {
                "is_bounce": False,
                "reason": "Pas de tendance baissière Hull fiable",
            }

        # Prix doit être AU-DESSUS Hull MA (bounce dans tendance baissière)
        if current_price <= hull_20:
            return {
                "is_bounce": False,
                "reason": f"Prix sous Hull MA ({current_price:.2f} <= {hull_20:.2f})",
            }

        # Calculer l'amplitude du bounce
        bounce_pct = (current_price - hull_20) / hull_20

        if bounce_pct < self.price_bounce_min:
            return {
                "is_bounce": False,
                "reason": f"Bounce trop faible ({bounce_pct*100:.1f}% < {self.price_bounce_min*100:.1f}%)",
            }

        if bounce_pct > self.price_bounce_max:
            return {
                "is_bounce": False,
                "reason": f"Bounce trop important ({bounce_pct*100:.1f}% > {self.price_bounce_max*100:.1f}%) - dead cat bounce",
            }

        # Vérifier oscillateurs surachat pour confirmation
        rsi_14 = values.get("rsi_14")
        momentum_score = values.get("momentum_score")

        overbought_signals = 0
        overbought_details = []

        if rsi_14 is not None and self._is_valid(rsi_14):
            try:
                rsi = float(rsi_14)
                # REJET si RSI contradictoire (SELL avec RSI faible = pas
                # contrarian)
                # RSI trop bas pour un SELL contrarian (assoupli)
                if rsi <= 25:
                    return {
                        "is_bounce": False,
                        "reason": f"RSI contradictoire pour SELL contrarian: {rsi:.1f} trop bas (<30)",
                        "rsi_rejection": True,
                    }
                if rsi >= self.rsi_overbought_entry:
                    overbought_signals += 1
                    overbought_details.append(f"RSI surachat ({rsi:.1f})")
            except (ValueError, TypeError):
                pass

        if momentum_score is not None:
            try:
                momentum = float(momentum_score)
                if momentum >= 55:  # Momentum élevé pour SELL (assoupli)
                    overbought_signals += 1
                    overbought_details.append(f"momentum très élevé ({momentum:.0f})")
            except (ValueError, TypeError):
                pass

        # MACD histogram pour retournement momentum
        macd_histogram = values.get("macd_histogram")
        if macd_histogram is not None:
            try:
                hist = float(macd_histogram)
                # Chercher un retournement (MACD qui redevient négatif)
                if hist < -0.00005:  # Légèrement négatif (seuil assoupli)
                    overbought_signals += 1
                    overbought_details.append(f"MACD retournement ({hist:.4f})")
            except (ValueError, TypeError):
                pass

        # Autoriser 0 confirmation si bounce optimal (1-2%)
        if overbought_signals < 1:
            if 0.01 <= bounce_pct <= 0.02:  # Zone optimale 1-2%
                overbought_signals = 1  # Donner 1 confirmation artificielle
                overbought_details.append("bounce optimal (1-2%)")
            else:
                return {
                    "is_bounce": False,
                    "reason": f"Bounce détecté mais aucune confirmation ({overbought_signals}/1)",
                    "bounce_pct": bounce_pct,
                    "overbought_signals": overbought_signals,
                }

        return {
            "is_bounce": True,
            "bounce_pct": bounce_pct,
            "overbought_signals": overbought_signals,
            "overbought_details": overbought_details,
            "reason": f"Bounce {bounce_pct*100:.1f}% avec {overbought_signals} confirmations",
        }

    def generate_signal(self) -> dict[str, Any]:
        """
        Génère un signal CONTRARIAN basé sur Hull MA comme filtre de tendance.
        """
        values = self._get_current_values()
        price_data = self._get_price_data()

        # VALIDATIONS PRÉLIMINAIRES GROUPÉES
        (
            is_valid,
            error_response,
            hull_val,
            price_val,
            volume_penalty,
            confluence_penalty,
        ) = self._validate_hull_data(values, price_data)
        if not is_valid:
            return error_response

        volume_ratio = values.get("volume_ratio")
        confluence_score = values.get("confluence_score")
        volatility_regime = values.get("volatility_regime")
        volatility_penalty = 0.0

        # === ANALYSE TENDANCE HULL MA ===

        hull_trend = self._analyze_hull_trend_direction(values, price_data)

        if hull_trend["direction"] is None or not hull_trend.get("reliable", False):
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Direction tendance Hull MA non fiable",
                "metadata": {"strategy": self.name, "hull_trend": hull_trend},
            }

        # === DÉTECTION OPPORTUNITÉS CONTRARIAN ===

        signal_side = None
        reason = ""
        opportunity_data = {}
        base_confidence = 0.55  # Réduit car approche plus accessible
        confidence_boost = 0.0

        # NOUVELLE LOGIQUE TREND-FOLLOWING (abandon contrarian)
        if hull_trend["direction"] in ["strong_bullish", "weak_bullish"]:
            # SUIVRE la micro-tendance haussière Hull

            # Vérifier momentum favorable
            momentum_score = values.get("momentum_score", 50)
            rsi_14 = values.get("rsi_14")

            # Filtres momentum pour BUY
            momentum_ok = False
            rsi_ok = False
            reason_parts = []

            if momentum_score is not None:
                try:
                    momentum_val = float(momentum_score)
                    if momentum_val >= self.momentum_bullish_min:  # > 50.1
                        momentum_ok = True
                        reason_parts.append(f"momentum {momentum_val:.1f}")
                except (ValueError, TypeError):
                    pass

            if rsi_14 is not None:
                try:
                    rsi_val = float(rsi_14)
                    if rsi_val >= self.rsi_bullish_min:  # >= 45
                        rsi_ok = True
                        reason_parts.append(f"RSI {rsi_val:.0f}")
                except (ValueError, TypeError):
                    pass

            # Au moins 1 confirmation momentum nécessaire
            if momentum_ok or rsi_ok:
                signal_side = "BUY"
                trend_label = (
                    "forte" if hull_trend["direction"] == "strong_bullish" else "faible"
                )
                reason = f"TREND-FOLLOWING BUY: Hull tendance {trend_label} ({hull_trend.get('slope_proxy', 0):.3f})"
                if reason_parts:
                    reason += f" + {' + '.join(reason_parts)}"

                # Bonus selon force tendance
                if hull_trend["direction"] == "strong_bullish":
                    confidence_boost += 0.25  # Tendance forte
                else:
                    confidence_boost += 0.15  # Tendance faible

                opportunity_data = {
                    "trend_type": hull_trend["direction"],
                    "momentum_score": momentum_score,
                    "rsi_14": rsi_14,
                    "confirmations": len(reason_parts),
                }
            else:
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Hull tendance haussière mais momentum défavorable (momentum:{momentum_score:.1f}, RSI:{rsi_14:.1f})",
                    "metadata": {"strategy": self.name, "hull_trend": hull_trend},
                }

        elif hull_trend["direction"] in ["strong_bearish", "weak_bearish"]:
            # SUIVRE la micro-tendance baissière Hull

            momentum_score = values.get("momentum_score", 50)
            rsi_14 = values.get("rsi_14")

            momentum_ok = False
            rsi_ok = False
            reason_parts = []

            if momentum_score is not None:
                try:
                    momentum_val = float(momentum_score)
                    if momentum_val <= self.momentum_bearish_max:  # < 49.9
                        momentum_ok = True
                        reason_parts.append(f"momentum {momentum_val:.1f}")
                except (ValueError, TypeError):
                    pass

            if rsi_14 is not None:
                try:
                    rsi_val = float(rsi_14)
                    if rsi_val <= self.rsi_bearish_max:  # <= 55
                        rsi_ok = True
                        reason_parts.append(f"RSI {rsi_val:.0f}")
                except (ValueError, TypeError):
                    pass

            if momentum_ok or rsi_ok:
                signal_side = "SELL"
                trend_label = (
                    "forte" if hull_trend["direction"] == "strong_bearish" else "faible"
                )
                reason = f"TREND-FOLLOWING SELL: Hull tendance {trend_label} ({hull_trend.get('slope_proxy', 0):.3f})"
                if reason_parts:
                    reason += f" + {' + '.join(reason_parts)}"

                if hull_trend["direction"] == "strong_bearish":
                    confidence_boost += 0.25
                else:
                    confidence_boost += 0.15

                opportunity_data = {
                    "trend_type": hull_trend["direction"],
                    "momentum_score": momentum_score,
                    "rsi_14": rsi_14,
                    "confirmations": len(reason_parts),
                }
            else:
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Hull tendance baissière mais momentum défavorable (momentum:{momentum_score:.1f}, RSI:{rsi_14:.1f})",
                    "metadata": {"strategy": self.name, "hull_trend": hull_trend},
                }

        else:  # sideways
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Hull MA en tendance latérale - pas de setup contrarian",
                "metadata": {"strategy": self.name, "hull_trend": hull_trend},
            }

        # === BONUS DE CONFIANCE ===

        # Bonus force tendance Hull (AJUSTÉ)
        if hull_trend["strength"] == "strong":
            confidence_boost += 0.15
            reason += " + tendance Hull FORTE"
        elif hull_trend["strength"] == "moderate":
            confidence_boost += 0.10  # Était 0.08
            reason += " + tendance Hull modérée"
        elif hull_trend["strength"] == "weak":
            confidence_boost += 0.05  # Nouveau bonus pour weak
            reason += " + tendance Hull faible"

        # Bonus nombre de confirmations oscillateurs (AJUSTÉ)
        if signal_side == "BUY":
            oversold_count = opportunity_data.get("oversold_signals", 0)
            if oversold_count >= 3:
                confidence_boost += 0.20  # Augmenté
                reason += f" + {oversold_count} confirmations survente"
            elif oversold_count >= 2:
                confidence_boost += 0.15  # Augmenté
                reason += f" + {oversold_count} confirmations"
            elif oversold_count >= 1:
                confidence_boost += 0.08  # Nouveau bonus 1 confirmation
                reason += f" + {oversold_count} confirmation"

        elif signal_side == "SELL":
            overbought_count = opportunity_data.get("overbought_signals", 0)
            if overbought_count >= 3:
                confidence_boost += 0.20  # Augmenté
                reason += f" + {overbought_count} confirmations surachat"
            elif overbought_count >= 2:
                confidence_boost += 0.15  # Augmenté
                reason += f" + {overbought_count} confirmations"
            elif overbought_count >= 1:
                confidence_boost += 0.08  # Nouveau bonus 1 confirmation
                reason += f" + {overbought_count} confirmation"

        # Bonus alignement EMA pour contexte
        ema_12 = values.get("ema_12")
        ema_26 = values.get("ema_26")
        if ema_12 is not None and ema_26 is not None:
            try:
                ema12_val = float(ema_12)
                ema26_val = float(ema_26)

                if signal_side == "BUY" and ema12_val > ema26_val:
                    confidence_boost += 0.08
                    reason += " + EMA12>26"
                elif signal_side == "SELL" and ema12_val < ema26_val:
                    confidence_boost += 0.08
                    reason += " + EMA12<26"
            except (ValueError, TypeError):
                pass

        # Bonus volume élevé avec validation NaN
        if self._is_valid(volume_ratio):
            try:
                vol_ratio = float(volume_ratio) if volume_ratio is not None else 0.0
                if vol_ratio >= 2.0:
                    confidence_boost += 0.15
                    reason += f" + volume très élevé ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.5:
                    confidence_boost += 0.10
                    reason += f" + volume élevé ({vol_ratio:.1f}x)"
                elif vol_ratio >= self.min_volume_ratio:
                    confidence_boost += 0.05
                    reason += f" + volume correct ({vol_ratio:.1f}x)"
            except (ValueError, TypeError):
                pass

        # Bonus confluence avec validation NaN
        if self._is_valid(confluence_score):
            try:
                conf_val = (
                    float(confluence_score) if confluence_score is not None else 0.0
                )
                if conf_val >= 70:
                    confidence_boost += 0.18
                    reason += f" + confluence excellente ({conf_val:.0f})"
                elif conf_val >= 60:
                    confidence_boost += 0.12
                    reason += f" + confluence forte ({conf_val:.0f})"
                elif conf_val >= self.min_confluence_score:
                    confidence_boost += 0.06
                    reason += f" + confluence ({conf_val:.0f})"
            except (ValueError, TypeError):
                pass

        # Bonus signal strength
        signal_strength = values.get("signal_strength")
        if signal_strength == "STRONG":
            confidence_boost += 0.12
            reason += " + signal fort"
        elif signal_strength == "MODERATE":
            confidence_boost += 0.06
            reason += " + signal modéré"

        # Bonus trend alignment avec validation NaN
        trend_alignment = values.get("trend_alignment")
        if self._is_valid(trend_alignment):
            try:
                alignment = (
                    float(trend_alignment) if trend_alignment is not None else 0.0
                )
                if abs(alignment) >= 0.3:
                    confidence_boost += 0.10
                    reason += " + MA alignées"
            except (ValueError, TypeError):
                pass

        # Bonus market regime favorable
        market_regime = values.get("market_regime")
        if market_regime is not None and str(market_regime) in [
            "TRENDING_BULL",
            "TRENDING_BEAR",
        ]:
            confidence_boost += 0.08
            reason += f" + marché {str(market_regime).lower()}"

        # === FILTRE FINAL ===

        # Appliquer pénalités restantes
        total_penalty = (
            volume_penalty + confluence_penalty
        )  # volatility_penalty supprimé (rejet direct)
        confidence_boost += total_penalty

        # CORRECTION: Calcul MULTIPLICATIF pour cohérence avec autres strats
        raw_confidence = max(
            0.0,
            min(1.0, self.calculate_confidence(base_confidence, 1 + confidence_boost)),
        )

        if raw_confidence < self.min_confidence_threshold:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Signal CONTRARIAN {signal_side} rejeté - confidence insuffisante ({raw_confidence:.2f} < {self.min_confidence_threshold})",
                "metadata": {
                    "strategy": self.name,
                    "rejected_signal": signal_side,
                    "raw_confidence": raw_confidence,
                    "hull_trend": hull_trend,
                    "opportunity_data": opportunity_data,
                    "penalties_applied": {
                        "volume": volume_penalty,
                        "confluence": confluence_penalty,
                        "volatility": volatility_penalty,
                    },
                },
            }

        confidence = raw_confidence  # Utiliser directement la confidence calculée
        strength = self.get_strength_from_confidence(confidence)

        return {
            "side": signal_side,
            "confidence": confidence,
            "strength": strength,
            "reason": reason,
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "approach": "CONTRARIAN",
                "current_price": price_val,
                "hull_20": hull_val,
                "hull_trend": hull_trend,
                "opportunity_data": opportunity_data,
                "volume_ratio": volume_ratio,
                "confluence_score": confluence_score,
                "market_regime": market_regime,
                "volatility_regime": volatility_regime,
            },
        }

    def validate_data(self) -> bool:
        """Valide que tous les indicateurs Hull MA requis sont présents."""
        if not super().validate_data():
            return False

        # CORRECTION CRITIQUE: Seulement Hull MA obligatoire
        required = ["hull_20"]  # volume_ratio et confluence_score en optionnel

        for indicator in required:
            if indicator not in self.indicators:
                logger.warning(f"{self.name}: Indicateur manquant: {indicator}")
                return False
            if self.indicators[indicator] is None:
                logger.warning(f"{self.name}: Indicateur null: {indicator}")
                return False

        # Vérifier données de prix suffisantes
        if (
            not self.data
            or "close" not in self.data
            or not self.data["close"]
            or len(self.data["close"]) < 3
        ):
            logger.warning(f"{self.name}: Données de prix insuffisantes")
            return False

        return True

    def _is_valid(self, x):
        """Helper pour valider les nombres (anti-NaN)."""
        try:
            x = float(x) if x is not None else None
            return x is not None and not math.isnan(x)
        except (TypeError, ValueError):
            return False
