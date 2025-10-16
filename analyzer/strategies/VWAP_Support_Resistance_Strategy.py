"""
VWAP_Support_Resistance_Strategy - Stratégie utilisant VWAP comme niveau dynamique de support/résistance.
Le VWAP (Volume-Weighted Average Price) agit comme une référence importante pour les institutions.
"""

import contextlib
import logging
from typing import Any

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class VWAP_Support_Resistance_Strategy(BaseStrategy):
    """
    Stratégie combinant VWAP avec les niveaux de support/résistance statiques.

    Principe VWAP :
    - VWAP = support dynamique quand prix au-dessus
    - VWAP = résistance dynamique quand prix en-dessous
    - Confluence VWAP + support/résistance statique = signal fort

    Signaux générés:
    - BUY: Prix rebondit sur VWAP support + confluence avec support statique
    - SELL: Prix rejette VWAP résistance + confluence avec résistance statique
    """

    def __init__(self, symbol: str, data: dict[str, Any], indicators: dict[str, Any]):
        super().__init__(symbol, data, indicators)

        # Paramètres VWAP - ADAPTÉS SCALPING
        self.vwap_distance_threshold = 0.005  # 0.5% au lieu de 0.3% (réaliste)
        # 1.5% au lieu de 1% (accessible)
        self.vwap_confluence_threshold = 0.015
        # 1.5x au lieu de 2x (atteignable)
        self.strong_vwap_volume_threshold = 1.5

        # Paramètres rebond/rejet - PLUS STRICTS
        self.min_bounce_strength = 0.002  # Rebond minimum 0.2%
        self.max_bounce_distance = 0.012  # Distance max 1.2% (plus strict)
        self.rejection_confirmation_bars = 1  # Barres pour confirmer rejet

        # Paramètres support/résistance
        self.min_sr_strength = 0.4  # Force minimum niveau S/R
        self.confluence_bonus_multiplier = 1.5  # Multiplicateur bonus confluence

        # Paramètres volume et momentum - DURCIS PREMIUM
        self.min_volume_confirmation = 1.4  # 1.4x minimum pour validation
        self.strong_volume_threshold = 2.0  # 2.0x pour bonus fort
        self.momentum_alignment_required = True  # Momentum OBLIGATOIREMENT aligné
        self.min_momentum_threshold = 0.10  # 10% au lieu de 20% (réaliste)

    def _create_rejection_signal(
        self, reason: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
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

    def _validate_signal_conditions(
        self, signal_side: str, values: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Valide les conditions critiques. Retourne un signal de rejet ou None si valide."""
        momentum_val = values.get("momentum_score")
        directional_bias = values.get("directional_bias")
        market_regime = values.get("market_regime")
        regime_strength = values.get("regime_strength")

        try:
            momentum_score = float(momentum_val) if momentum_val is not None else 50.0
        except (ValueError, TypeError):
            momentum_score = 50.0

        # Momentum contradictoire
        if (signal_side == "BUY" and momentum_score < 40) or (
            signal_side == "SELL" and momentum_score > 60
        ):
            return self._create_rejection_signal(
                f"Rejet VWAP {signal_side}: momentum {'trop faible' if signal_side == 'BUY' else 'trop fort'} ({momentum_score:.0f})",
                {"momentum_score": float(momentum_score)},
            )

        # Bias contradictoire
        if (signal_side == "BUY" and str(directional_bias).upper() == "BEARISH") or (
            signal_side == "SELL" and str(directional_bias).upper() == "BULLISH"
        ):
            return self._create_rejection_signal(
                f"Rejet VWAP {signal_side}: bias contradictoire ({directional_bias})",
                {"directional_bias": directional_bias},
            )

        # Regime contradictoire fort
        regime_str = str(regime_strength).upper() if regime_strength else "WEAK"
        if regime_str in ["STRONG", "EXTREME"]:
            if (signal_side == "BUY" and market_regime == "TRENDING_BEAR") or (
                signal_side == "SELL" and market_regime == "TRENDING_BULL"
            ):
                return self._create_rejection_signal(
                    f"Rejet VWAP {signal_side}: régime contradictoire fort ({market_regime})",
                    {
                        "market_regime": market_regime,
                        "regime_strength": regime_strength,
                    },
                )

        # Confluence insuffisante
        confluence_score = values.get("confluence_score")
        if confluence_score is not None:
            try:
                conf_val = float(confluence_score)
                if conf_val < 40:
                    return self._create_rejection_signal(
                        f"Rejet VWAP: confluence insuffisante ({conf_val:.0f})",
                        {"confluence_score": float(conf_val)},
                    )
            except (ValueError, TypeError):
                pass

        return None

    def _get_current_values(self) -> dict[str, float | None]:
        """Récupère les valeurs actuelles des indicateurs pré-calculés."""
        return {
            # VWAP principal (support/résistance dynamique)
            "vwap_10": self.indicators.get("vwap_10"),
            "anchored_vwap": self.indicators.get("anchored_vwap"),
            "vwap_upper_band": self.indicators.get("vwap_upper_band"),
            "vwap_lower_band": self.indicators.get("vwap_lower_band"),
            # Support/Résistance statiques
            "nearest_support": self.indicators.get("nearest_support"),
            "nearest_resistance": self.indicators.get("nearest_resistance"),
            "support_strength": self.indicators.get("support_strength"),
            "resistance_strength": self.indicators.get("resistance_strength"),
            "support_levels": self.indicators.get("support_levels"),
            "resistance_levels": self.indicators.get("resistance_levels"),
            "break_probability": self.indicators.get("break_probability"),
            "pivot_count": self.indicators.get("pivot_count"),
            # Volume analysis (crucial pour VWAP)
            "volume_ratio": self.indicators.get("volume_ratio"),
            "relative_volume": self.indicators.get("relative_volume"),
            "volume_quality_score": self.indicators.get("volume_quality_score"),
            "trade_intensity": self.indicators.get("trade_intensity"),
            "volume_spike_multiplier": self.indicators.get("volume_spike_multiplier"),
            "avg_volume_20": self.indicators.get("avg_volume_20"),
            # Momentum et tendance
            "momentum_score": self.indicators.get("momentum_score"),
            "directional_bias": self.indicators.get("directional_bias"),
            "trend_strength": self.indicators.get("trend_strength"),
            "trend_alignment": self.indicators.get("trend_alignment"),
            # RSI pour timing
            "rsi_14": self.indicators.get("rsi_14"),
            "rsi_21": self.indicators.get("rsi_21"),
            "williams_r": self.indicators.get("williams_r"),
            # MACD pour confirmation momentum
            "macd_line": self.indicators.get("macd_line"),
            "macd_signal": self.indicators.get("macd_signal"),
            "macd_histogram": self.indicators.get("macd_histogram"),
            "macd_trend": self.indicators.get("macd_trend"),
            # ADX pour force tendance
            "adx_14": self.indicators.get("adx_14"),
            "plus_di": self.indicators.get("plus_di"),
            "minus_di": self.indicators.get("minus_di"),
            # EMA pour contexte tendance
            "ema_12": self.indicators.get("ema_12"),
            "ema_26": self.indicators.get("ema_26"),
            "ema_50": self.indicators.get("ema_50"),
            # ATR pour volatilité
            "atr_14": self.indicators.get("atr_14"),
            "volatility_regime": self.indicators.get("volatility_regime"),
            "atr_percentile": self.indicators.get("atr_percentile"),
            # OBV pour validation volume
            "obv": self.indicators.get("obv"),
            "obv_ma_10": self.indicators.get("obv_ma_10"),
            "obv_oscillator": self.indicators.get("obv_oscillator"),
            # Contexte marché
            "market_regime": self.indicators.get("market_regime"),
            "regime_strength": self.indicators.get("regime_strength"),
            "confluence_score": self.indicators.get("confluence_score"),
            "signal_strength": self.indicators.get("signal_strength"),
            "pattern_detected": self.indicators.get("pattern_detected"),
            "pattern_confidence": self.indicators.get("pattern_confidence"),
        }

    def _detect_vwap_support_bounce(
        self, values: dict[str, Any], current_price: float
    ) -> dict[str, Any]:
        """Détecte un rebond sur VWAP agissant comme support."""
        bounce_score = 0.0
        bounce_indicators = []

        # VWAP principal
        vwap_10 = values.get("vwap_10")
        if vwap_10 is None:
            return {"is_bounce": False, "score": 0.0, "indicators": []}

        try:
            vwap_val = float(vwap_10)
        except (ValueError, TypeError):
            return {"is_bounce": False, "score": 0.0, "indicators": []}

        # Vérifier si prix est près du VWAP (potentiel support)
        if current_price <= vwap_val:
            return {
                "is_bounce": False,
                "score": 0.0,
                "indicators": ["Prix sous VWAP - pas de support"],
                "vwap_level": vwap_val,
            }

        # Distance au VWAP - CALCUL COHÉRENT
        vwap_distance = abs(current_price - vwap_val) / current_price

        if vwap_distance > self.max_bounce_distance:
            return {
                "is_bounce": False,
                "score": 0.0,
                "indicators": [f"Prix trop loin VWAP ({vwap_distance*100:.1f}%)"],
                "vwap_level": vwap_val,
            }

        # Bounce scoring selon la proximité
        if vwap_distance <= self.vwap_distance_threshold:
            bounce_score += 0.3
            bounce_indicators.append(f"Prix très près VWAP ({vwap_distance*100:.2f}%)")
        elif vwap_distance <= self.vwap_distance_threshold * 2:
            bounce_score += 0.2
            bounce_indicators.append(f"Prix près VWAP ({vwap_distance*100:.2f}%)")
        else:
            bounce_score += 0.1
            bounce_indicators.append(f"Prix proche VWAP ({vwap_distance*100:.2f}%)")

        # Confluence avec support statique
        nearest_support = values.get("nearest_support")
        if nearest_support is not None:
            try:
                support_level = float(nearest_support)
                support_vwap_distance = abs(support_level - vwap_val) / vwap_val

                if support_vwap_distance <= self.vwap_confluence_threshold:
                    bounce_score += 0.25
                    bounce_indicators.append(
                        f"Confluence VWAP/Support ({support_vwap_distance*100:.2f}%)"
                    )

                    # Bonus si support fort
                    support_strength = values.get("support_strength")
                    if support_strength is not None:
                        try:
                            if isinstance(support_strength, str):
                                strength_map = {
                                    "WEAK": 0.2,
                                    "MODERATE": 0.5,
                                    "STRONG": 0.8,
                                    "MAJOR": 1.0,
                                }
                                strength_val = strength_map.get(
                                    support_strength.upper(), 0.5
                                )
                            else:
                                strength_val = float(support_strength)

                            if strength_val >= 0.8:
                                bounce_score += 0.2
                                bounce_indicators.append(
                                    f"Support très fort ({strength_val:.2f})"
                                )
                            elif strength_val >= self.min_sr_strength:
                                bounce_score += 0.15
                                bounce_indicators.append(
                                    f"Support fort ({strength_val:.2f})"
                                )
                        except (ValueError, TypeError):
                            pass
            except (ValueError, TypeError):
                pass

        # Volume confirmation (crucial pour VWAP)
        volume_ratio = values.get("volume_ratio")
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= self.strong_vwap_volume_threshold:  # >= 2.0x
                    bounce_score += 0.25
                    bounce_indicators.append(f"Volume exceptionnel ({vol_ratio:.1f}x)")
                elif vol_ratio >= 1.8:  # Volume très fort
                    bounce_score += 0.20
                    bounce_indicators.append(f"Volume très fort ({vol_ratio:.1f}x)")
                elif vol_ratio >= self.strong_volume_threshold:  # >= 2.0x
                    bounce_score += 0.20
                    bounce_indicators.append(f"Volume fort ({vol_ratio:.1f}x)")
                elif vol_ratio >= self.min_volume_confirmation:  # >= 1.4x
                    bounce_score += 0.15
                    bounce_indicators.append(f"Volume confirmé ({vol_ratio:.1f}x)")
                else:
                    # Volume insuffisant = rejet direct
                    return {
                        "is_bounce": False,
                        "score": 0.0,
                        "indicators": [
                            f"Volume insuffisant ({vol_ratio:.1f}x) < 1.4x requis"
                        ],
                        "vwap_level": vwap_val,
                    }
            except (ValueError, TypeError):
                pass

        return {
            "is_bounce": bounce_score >= 0.45,  # Seuil plus généreux pour signaux
            "score": bounce_score,
            "indicators": bounce_indicators,
            "vwap_level": vwap_val,
            "vwap_distance_pct": vwap_distance * 100,
        }

    def _detect_vwap_resistance_rejection(
        self, values: dict[str, Any], current_price: float
    ) -> dict[str, Any]:
        """Détecte un rejet sur VWAP agissant comme résistance."""
        rejection_score = 0.0
        rejection_indicators = []

        # VWAP principal
        vwap_10 = values.get("vwap_10")
        if vwap_10 is None:
            return {"is_rejection": False, "score": 0.0, "indicators": []}

        try:
            vwap_val = float(vwap_10)
        except (ValueError, TypeError):
            return {"is_rejection": False, "score": 0.0, "indicators": []}

        # Vérifier si prix est près du VWAP (potentiel résistance)
        if current_price >= vwap_val:
            return {
                "is_rejection": False,
                "score": 0.0,
                "indicators": ["Prix au-dessus VWAP - pas de résistance"],
                "vwap_level": vwap_val,
            }

        # Distance au VWAP - CALCUL COHÉRENT
        vwap_distance = abs(current_price - vwap_val) / current_price

        if vwap_distance > self.max_bounce_distance:
            return {
                "is_rejection": False,
                "score": 0.0,
                "indicators": [f"Prix trop loin VWAP ({vwap_distance*100:.1f}%)"],
                "vwap_level": vwap_val,
            }

        # Rejection scoring selon la proximité
        if vwap_distance <= self.vwap_distance_threshold:
            rejection_score += 0.3
            rejection_indicators.append(
                f"Prix très près VWAP ({vwap_distance*100:.2f}%)"
            )
        elif vwap_distance <= self.vwap_distance_threshold * 2:
            rejection_score += 0.2
            rejection_indicators.append(f"Prix près VWAP ({vwap_distance*100:.2f}%)")
        else:
            rejection_score += 0.1
            rejection_indicators.append(f"Prix proche VWAP ({vwap_distance*100:.2f}%)")

        # Confluence avec résistance statique
        nearest_resistance = values.get("nearest_resistance")
        if nearest_resistance is not None:
            try:
                resistance_level = float(nearest_resistance)
                resistance_vwap_distance = (
                    abs(resistance_level - vwap_val) / current_price
                )

                if resistance_vwap_distance <= self.vwap_confluence_threshold:
                    rejection_score += 0.25
                    rejection_indicators.append(
                        f"Confluence VWAP/Résistance ({resistance_vwap_distance*100:.2f}%)"
                    )

                    # Bonus si résistance forte
                    resistance_strength = values.get("resistance_strength")
                    if resistance_strength is not None:
                        try:
                            if isinstance(resistance_strength, str):
                                strength_map = {
                                    "WEAK": 0.2,
                                    "MODERATE": 0.5,
                                    "STRONG": 0.8,
                                    "MAJOR": 1.0,
                                }
                                strength_val = strength_map.get(
                                    resistance_strength.upper(), 0.5
                                )
                            else:
                                strength_val = float(resistance_strength)

                            if strength_val >= 0.8:
                                rejection_score += 0.2
                                rejection_indicators.append(
                                    f"Résistance très forte ({strength_val:.2f})"
                                )
                            elif strength_val >= self.min_sr_strength:
                                rejection_score += 0.15
                                rejection_indicators.append(
                                    f"Résistance forte ({strength_val:.2f})"
                                )
                        except (ValueError, TypeError):
                            pass
            except (ValueError, TypeError):
                pass

        # Volume confirmation (crucial pour VWAP)
        volume_ratio = values.get("volume_ratio")
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= self.strong_vwap_volume_threshold:  # >= 2.0x
                    rejection_score += 0.25
                    rejection_indicators.append(
                        f"Volume exceptionnel ({vol_ratio:.1f}x)"
                    )
                elif vol_ratio >= 1.8:  # Volume très fort
                    rejection_score += 0.20
                    rejection_indicators.append(f"Volume très fort ({vol_ratio:.1f}x)")
                elif vol_ratio >= self.strong_volume_threshold:  # >= 2.0x
                    rejection_score += 0.20
                    rejection_indicators.append(f"Volume fort ({vol_ratio:.1f}x)")
                elif vol_ratio >= self.min_volume_confirmation:  # >= 1.4x
                    rejection_score += 0.15
                    rejection_indicators.append(f"Volume confirmé ({vol_ratio:.1f}x)")
                else:
                    # Volume insuffisant = rejet direct
                    return {
                        "is_rejection": False,
                        "score": 0.0,
                        "indicators": [
                            f"Volume insuffisant ({vol_ratio:.1f}x) < 1.4x requis"
                        ],
                        "vwap_level": vwap_val,
                    }
            except (ValueError, TypeError):
                pass

        return {
            "is_rejection": rejection_score >= 0.45,  # Seuil plus généreux pour signaux
            "score": rejection_score,
            "indicators": rejection_indicators,
            "vwap_level": vwap_val,
            "vwap_distance_pct": vwap_distance * 100,
        }

    def _detect_momentum_alignment(
        self, values: dict[str, Any], signal_direction: str
    ) -> dict[str, Any]:
        """Détecte l'alignement du momentum avec la direction du signal."""
        momentum_score = 0.0  # Float cohérent
        momentum_indicators = []

        # Momentum score général (format 0-100, 50=neutre)
        momentum_val = values.get("momentum_score")
        if momentum_val is not None:
            try:
                momentum = float(momentum_val)

                if signal_direction == "BUY" and momentum >= (
                    50 + self.min_momentum_threshold * 50
                ):  # >=55 (10%)
                    momentum_score += 30
                    momentum_indicators.append(
                        f"Momentum haussier fort ({momentum:.1f})"
                    )
                elif signal_direction == "SELL" and momentum <= (
                    50 - self.min_momentum_threshold * 50
                ):  # <=45 (10%)
                    momentum_score += 30
                    momentum_indicators.append(
                        f"Momentum baissier fort ({momentum:.1f})"
                    )
            except (ValueError, TypeError):
                pass

        # Directional bias alignment
        directional_bias = values.get("directional_bias")
        if directional_bias and (
            (signal_direction == "BUY" and directional_bias.upper() == "BULLISH")
            or (signal_direction == "SELL" and directional_bias.upper() == "BEARISH")
        ):
            momentum_score += 20
            momentum_indicators.append(f"Bias directionnel {directional_bias}")

        # MACD confirmation
        macd_line = values.get("macd_line")
        macd_signal = values.get("macd_signal")
        if macd_line is not None and macd_signal is not None:
            try:
                macd_val = float(macd_line)
                macd_sig = float(macd_signal)

                if signal_direction == "BUY" and macd_val > macd_sig:
                    momentum_score += 15
                    momentum_indicators.append("MACD haussier")
                elif signal_direction == "SELL" and macd_val < macd_sig:
                    momentum_score += 15
                    momentum_indicators.append("MACD baissier")
            except (ValueError, TypeError):
                pass

        # ADX force tendance
        adx_14 = values.get("adx_14")
        plus_di = values.get("plus_di")
        minus_di = values.get("minus_di")

        if adx_14 is not None and plus_di is not None and minus_di is not None:
            try:
                adx_val = float(adx_14)
                plus_di_val = float(plus_di)
                minus_di_val = float(minus_di)

                if adx_val > 25:  # Tendance forte
                    if signal_direction == "BUY" and plus_di_val > minus_di_val:
                        momentum_score += 15
                        momentum_indicators.append(f"ADX fort + DI+ ({adx_val:.1f})")
                    elif signal_direction == "SELL" and minus_di_val > plus_di_val:
                        momentum_score += 15
                        momentum_indicators.append(f"ADX fort + DI- ({adx_val:.1f})")
            except (ValueError, TypeError):
                pass

        # RSI pour timing (éviter extrêmes)
        rsi_14 = values.get("rsi_14")
        if rsi_14 is not None:
            try:
                rsi_val = float(rsi_14)
                if signal_direction == "BUY" and 40 <= rsi_val <= 60:  # Zone resserrée
                    momentum_score += 15
                    momentum_indicators.append(f"RSI optimal BUY ({rsi_val:.1f})")
                elif (
                    signal_direction == "SELL" and 40 <= rsi_val <= 65
                ):  # Zone resserrée
                    momentum_score += 15
                    momentum_indicators.append(f"RSI optimal SELL ({rsi_val:.1f})")
                elif (signal_direction == "BUY" and rsi_val >= 80) or (
                    signal_direction == "SELL" and rsi_val <= 20
                ):
                    momentum_score -= 10
                    momentum_indicators.append(f"RSI extrême ({rsi_val:.1f})")
            except (ValueError, TypeError):
                pass

        return {
            "is_aligned": momentum_score >= 40,  # Seuil plus accessible
            "score": momentum_score,
            "indicators": momentum_indicators,
        }

    def generate_signal(self) -> dict[str, Any]:
        """
        Génère un signal basé sur VWAP Support/Resistance.
        """
        if not self.validate_data():
            return self._create_rejection_signal("Données insuffisantes", {})

        values = self._get_current_values()

        # Récupérer prix actuel
        current_price = None
        if self.data.get("close"):
            with contextlib.suppress(IndexError, ValueError, TypeError):
                current_price = float(self.data["close"][-1])

        if current_price is None:
            return self._create_rejection_signal("Prix actuel non disponible", {})

        # Analyse VWAP support (signal BUY)
        support_analysis = self._detect_vwap_support_bounce(values, current_price)

        # Analyse VWAP résistance (signal SELL)
        resistance_analysis = self._detect_vwap_resistance_rejection(
            values, current_price
        )

        # Déterminer signal principal
        signal_side = None
        primary_analysis = None

        if support_analysis["is_bounce"] and resistance_analysis["is_rejection"]:
            # Conflit - prendre le score le plus élevé
            if support_analysis["score"] > resistance_analysis["score"]:
                signal_side = "BUY"
                primary_analysis = support_analysis
            else:
                signal_side = "SELL"
                primary_analysis = resistance_analysis
        elif support_analysis["is_bounce"]:
            signal_side = "BUY"
            primary_analysis = support_analysis
        elif resistance_analysis["is_rejection"]:
            signal_side = "SELL"
            primary_analysis = resistance_analysis

        if signal_side is None:
            # Diagnostic des conditions manquées
            missing_conditions = []
            if support_analysis["score"] < 0.6:
                missing_conditions.append(
                    f"Support VWAP insuffisant (score: {support_analysis['score']:.2f} < 0.6)"
                )
            if resistance_analysis["score"] < 0.6:
                missing_conditions.append(
                    f"Résistance VWAP insuffisante (score: {resistance_analysis['score']:.2f} < 0.6)"
                )

            return self._create_rejection_signal(
                f"Conditions VWAP insuffisantes: {'; '.join(missing_conditions[:2])}",
                {
                    "symbol": self.symbol,
                    "current_price": current_price,
                    "support_score": support_analysis["score"],
                    "resistance_score": resistance_analysis["score"],
                },
            )

        # Vérifier alignement momentum si requis
        momentum_analysis = self._detect_momentum_alignment(values, signal_side)

        if (
            self.momentum_alignment_required
            and momentum_analysis is not None
            and not momentum_analysis["is_aligned"]
        ):
            return self._create_rejection_signal(
                f"VWAP {signal_side} détecté mais momentum pas aligné (score: {momentum_analysis['score']:.2f})",
                {
                    "symbol": self.symbol,
                    "vwap_score": (
                        primary_analysis["score"] if primary_analysis else 0.0
                    ),
                    "momentum_score": (
                        momentum_analysis["score"] if momentum_analysis else 0.0
                    ),
                },
            )

        # Valider les conditions critiques
        rejection = self._validate_signal_conditions(signal_side, values)
        if rejection:
            return rejection

        # BASE CONFIDENCE CORRIGÉE pour aggregator
        base_confidence = 0.65  # Au lieu de 0.45 (passe aggregator)
        confidence_boost = 0.0

        # Score VWAP principal
        if primary_analysis is not None:
            confidence_boost += primary_analysis["score"] * 0.4

        # Score momentum
        if momentum_analysis is not None:
            confidence_boost += momentum_analysis["score"] * 0.3

        # Construire raison
        vwap_level = primary_analysis["vwap_level"] if primary_analysis else 0.0
        vwap_distance = (
            primary_analysis["vwap_distance_pct"] if primary_analysis else 0.0
        )

        if signal_side == "BUY":
            reason = f"VWAP support {vwap_level:.2f} (distance: {vwap_distance:.2f}%)"
        else:
            reason = (
                f"VWAP résistance {vwap_level:.2f} (distance: {vwap_distance:.2f}%)"
            )

        # Construire reason LIMITÉE (max 4-5 éléments)
        reason_elements = []
        if primary_analysis and primary_analysis["indicators"]:
            reason_elements.append(primary_analysis["indicators"][0])
        if momentum_analysis and momentum_analysis["indicators"]:
            reason_elements.append(momentum_analysis["indicators"][0])

        # Bonus confluences et confirmations supplémentaires (track pour
        # reason)

        # Trend alignment (format décimal)
        trend_alignment = values.get("trend_alignment")
        if trend_alignment is not None:
            try:
                trend_align = float(trend_alignment)
                if (
                    signal_side == "BUY" and trend_align >= 0.3
                ):  # Trend plus fort requis (30%)
                    confidence_boost += 0.12
                    if len(reason_elements) < 4:
                        reason_elements.append("trend haussier")
                elif (
                    signal_side == "SELL" and trend_align <= -0.3
                ):  # Trend plus fort requis (-30%)
                    confidence_boost += 0.12
                    if len(reason_elements) < 4:
                        reason_elements.append("trend baissier")
            except (ValueError, TypeError):
                pass

        # OBV confirmation
        obv_oscillator = values.get("obv_oscillator")
        if obv_oscillator is not None:
            try:
                obv_osc = float(obv_oscillator)
                if signal_side == "BUY" and obv_osc > 0:
                    confidence_boost += 0.08
                    if len(reason_elements) < 4:
                        reason_elements.append("OBV+")
                elif signal_side == "SELL" and obv_osc < 0:
                    confidence_boost += 0.08
                    if len(reason_elements) < 4:
                        reason_elements.append("OBV-")
            except (ValueError, TypeError):
                pass

        # EMA context
        ema_12 = values.get("ema_12")
        ema_26 = values.get("ema_26")
        if ema_12 is not None and ema_26 is not None:
            try:
                ema12_val = float(ema_12)
                ema26_val = float(ema_26)

                if signal_side == "BUY" and ema12_val > ema26_val:
                    confidence_boost += 0.08
                    reason += " + EMA haussier"
                elif signal_side == "SELL" and ema12_val < ema26_val:
                    confidence_boost += 0.08
                    reason += " + EMA baissier"
            except (ValueError, TypeError):
                pass

        # CORRECTION MAGISTRALE: Volatility context avec seuils adaptatifs
        volatility_regime = values.get("volatility_regime")
        atr_percentile = values.get("atr_percentile")

        if volatility_regime is not None:
            try:
                atr_percentile = (
                    float(atr_percentile) if atr_percentile is not None else 50
                )

                if signal_side == "BUY":
                    # BUY sur VWAP support nécessite volatilité contrôlée pour
                    # éviter faux rebonds
                    if volatility_regime == "low" and atr_percentile < 30:
                        confidence_boost += 0.12
                        reason += f" + volatilité très faible idéale support ({atr_percentile:.0f}%)"
                    elif volatility_regime == "normal" and 30 <= atr_percentile <= 60:
                        confidence_boost += 0.08
                        reason += (
                            f" + volatilité normale support ({atr_percentile:.0f}%)"
                        )
                    elif volatility_regime == "high" and atr_percentile > 70:
                        if atr_percentile > 90:  # Volatilité extrême
                            confidence_boost -= 0.12
                            reason += f" mais volatilité extrême support ({atr_percentile:.0f}%)"
                        else:  # Volatilité élevée mais gérable
                            confidence_boost -= 0.06
                            reason += f" mais volatilité élevée support ({atr_percentile:.0f}%)"
                    elif volatility_regime == "extreme":
                        confidence_boost -= (
                            0.08  # Volatilité extrême défavorable aux supports
                        )
                        reason += " mais volatilité extrême défavorable"

                # SELL sur VWAP résistance peut bénéficier de volatilité
                # modérée à élevée
                elif volatility_regime == "low" and atr_percentile < 25:
                    confidence_boost += 0.06  # Résistance solide en low vol
                    reason += f" + volatilité faible résistance solide ({atr_percentile:.0f}%)"
                elif volatility_regime == "normal" and 25 <= atr_percentile <= 70:
                    confidence_boost += 0.10
                    reason += (
                        f" + volatilité optimale résistance ({atr_percentile:.0f}%)"
                    )
                elif volatility_regime == "high" and atr_percentile > 70:
                    if (
                        atr_percentile > 85
                    ):  # Volatilité très élevée = continuation baissière
                        confidence_boost += 0.15
                        reason += f" + volatilité très élevée continuation ({atr_percentile:.0f}%)"
                    else:  # Volatilité élevée favorable
                        confidence_boost += 0.12
                        reason += (
                            f" + volatilité élevée résistance ({atr_percentile:.0f}%)"
                        )
                elif volatility_regime == "extreme":
                    confidence_boost += (
                        0.08  # Volatilité extrême favorable aux résistances
                    )
                    reason += " + volatilité extrême favorable résistance"

            except (ValueError, TypeError):
                pass

        # CORRECTION MAGISTRALE: Market regime avec logique institutionnelle
        # VWAP
        market_regime = values.get("market_regime")
        regime_strength = values.get("regime_strength")

        if market_regime is not None:
            # regime_strength est un VARCHAR (WEAK/MODERATE/STRONG/EXTREME)
            regime_str = str(regime_strength).upper() if regime_strength else "WEAK"

            if signal_side == "BUY":
                # BUY sur VWAP support : trending haussier > ranging > trending
                # baissier
                if market_regime in ["TRENDING_BULL", "TRENDING_BEAR"]:
                    if regime_str == "EXTREME":  # Trend très fort
                        confidence_boost += 0.18
                        reason += f" + trend très fort support VWAP ({regime_str})"
                    elif regime_str == "STRONG":  # Trend fort
                        confidence_boost += 0.14
                        reason += f" + trend fort support VWAP ({regime_str})"
                    elif regime_str == "MODERATE":  # Trend modéré
                        confidence_boost += 0.10
                        reason += f" + trend modéré support VWAP ({regime_str})"
                    else:  # WEAK
                        confidence_boost += 0.06
                        reason += f" + trend faible support VWAP ({regime_str})"
                elif market_regime == "RANGING":
                    if regime_str in ["EXTREME", "STRONG"]:  # Range bien défini
                        confidence_boost += 0.15
                        reason += f" + range fort rebond support ({regime_str})"
                    else:  # MODERATE/WEAK
                        confidence_boost += 0.10
                        reason += f" + range modéré support ({regime_str})"
                elif market_regime == "VOLATILE":
                    confidence_boost -= 0.05  # Marché chaotique défavorable
                    reason += " mais marché chaotique"

            # SELL sur VWAP résistance : ranging > trending baissier > trending
            # haussier
            elif market_regime == "RANGING":
                if regime_str == "EXTREME":  # Range très défini
                    confidence_boost += (
                        0.25  # BONUS AUGMENTÉ - VWAP résistance excellent
                    )
                    if len(reason_elements) < 4:
                        reason_elements.append(f"range fort ({regime_str})")
                elif regime_str == "STRONG":  # Range fort
                    confidence_boost += 0.20  # AUGMENTÉ
                    if len(reason_elements) < 4:
                        reason_elements.append(f"range ({regime_str})")
                elif regime_str == "MODERATE":  # Range modéré
                    confidence_boost += 0.15  # AUGMENTÉ
                    if len(reason_elements) < 4:
                        reason_elements.append("range modéré")
                else:  # WEAK
                    confidence_boost += 0.10  # AUGMENTÉ
                    if len(reason_elements) < 4:
                        reason_elements.append("range faible")
            elif market_regime in ["TRENDING_BULL", "TRENDING_BEAR"]:
                if regime_str in [
                    "EXTREME",
                    "STRONG",
                ]:  # Trend fort - résistance peut tenir
                    confidence_boost += 0.12
                    reason += f" + trend fort résistance VWAP ({regime_str})"
                else:  # MODERATE/WEAK
                    confidence_boost += 0.06
                    reason += f" + trend faible résistance ({regime_str})"
            elif market_regime == "VOLATILE":
                confidence_boost += 0.08  # Chaos favorable aux résistances
                reason += " + marché chaotique favorable résistance"

        # Pattern detection (format 0-100)
        pattern_detected = values.get("pattern_detected")
        pattern_confidence = values.get("pattern_confidence")
        if pattern_detected and pattern_confidence is not None:
            try:
                pattern_conf = float(pattern_confidence)
                if pattern_conf > 70:
                    confidence_boost += 0.08
                    reason += f" + pattern {pattern_detected} ({pattern_conf:.0f}%)"
                elif pattern_conf > 50:
                    confidence_boost += 0.05
                    reason += f" + pattern faible ({pattern_conf:.0f}%)"
            except (ValueError, TypeError):
                pass

        # Confluence score global (format 0-100)
        confluence_score = values.get("confluence_score")
        if confluence_score is not None:
            try:
                conf_val = float(confluence_score)
                if conf_val > 85:  # Confluence exceptionnelle uniquement
                    confidence_boost += 0.15
                    if len(reason_elements) < 4:
                        reason_elements.append(f"conf.exc.({conf_val:.0f})")
                elif conf_val > 75:  # Confluence très haute
                    confidence_boost += 0.10
                    if len(reason_elements) < 4:
                        reason_elements.append(f"conf.haute({conf_val:.0f})")
                elif conf_val > 65:  # Confluence haute
                    confidence_boost += 0.06
                    if len(reason_elements) < 4:
                        reason_elements.append(f"conf.({conf_val:.0f})")
            except (ValueError, TypeError):
                pass

        # Finaliser reason LIMITÉE (max 4 éléments)
        if reason_elements:
            reason += " + " + " + ".join(reason_elements[:4])

        # SUPPRIMER DOUBLE FILTRE - laisser aggregator filtrer
        # Filtre interne supprimé - l'aggregator filtre déjà à 60%

        # Clamp explicite pour cohérence
        confidence = min(
            1.0,
            max(
                0.0, self.calculate_confidence(base_confidence, 1.0 + confidence_boost)
            ),
        )
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
                "vwap_level": vwap_level,
                "vwap_distance_pct": vwap_distance,
                "vwap_score": primary_analysis["score"] if primary_analysis else 0.0,
                "momentum_score": float(
                    momentum_analysis["score"] if momentum_analysis else 0.0
                ),
                "vwap_indicators": (
                    primary_analysis["indicators"] if primary_analysis else []
                ),
                "momentum_indicators": (
                    momentum_analysis["indicators"] if momentum_analysis else []
                ),
                "support_analysis": support_analysis if signal_side == "BUY" else None,
                "resistance_analysis": (
                    resistance_analysis if signal_side == "SELL" else None
                ),
                "volume_ratio": (
                    float(vol_ratio_val)
                    if (vol_ratio_val := values.get("volume_ratio")) is not None
                    else None
                ),
                "trend_alignment": (
                    float(trend_align_val)
                    if (trend_align_val := values.get("trend_alignment")) is not None
                    else None
                ),
                "market_regime": values.get("market_regime"),
                "volatility_regime": values.get("volatility_regime"),
                "confluence_score": (
                    float(conf_score_val)
                    if (conf_score_val := values.get("confluence_score")) is not None
                    else None
                ),
                "pattern_detected": values.get("pattern_detected"),
                "pattern_confidence": (
                    float(pattern_conf_val)
                    if (pattern_conf_val := values.get("pattern_confidence"))
                    is not None
                    else None
                ),
            },
        }

    def validate_data(self) -> bool:
        """Valide que tous les indicateurs requis sont présents."""
        required_indicators = ["vwap_10", "volume_ratio", "momentum_score"]

        if not self.indicators:
            logger.warning(f"{self.name}: Aucun indicateur disponible")
            return False

        for indicator in required_indicators:
            if indicator not in self.indicators:
                logger.warning(f"{self.name}: Indicateur manquant: {indicator}")
                return False
            if self.indicators[indicator] is None:
                logger.warning(f"{self.name}: Indicateur null: {indicator}")
                return False

        # Vérifier données OHLCV
        if "close" not in self.data or not self.data["close"]:
            logger.warning(f"{self.name}: Données OHLCV manquantes")
            return False

        return True
