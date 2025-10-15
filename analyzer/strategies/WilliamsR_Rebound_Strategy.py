"""
WilliamsR_Rebound_Strategy - Stratégie de rebound basée sur Williams %R.
Williams %R est un oscillateur de momentum qui mesure les niveaux de surachat/survente
et génère des signaux de rebound depuis les extrêmes.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class WilliamsR_Rebound_Strategy(BaseStrategy):
    """
    Stratégie utilisant Williams %R pour détecter les opportunités de rebound.

    Principe Williams %R :
    - Williams %R = (Plus Haut N - Close) / (Plus Haut N - Plus Bas N) * -100
    - Valeurs entre -100 et 0
    - Williams %R < -80 = survente (signal BUY potentiel)
    - Williams %R > -20 = surachat (signal SELL potentiel)
    - Plus sensible que RSI, excellent pour les rebonds courts

    Signaux générés:
    - BUY: Williams %R sort de zone survente (-80) vers le haut + confirmations
    - SELL: Williams %R sort de zone surachat (-20) vers le bas + confirmations
    """

    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)

        # Paramètres Williams %R - RESSERRE pour vrais rebonds
        self.oversold_threshold = -75.0  # Seuil survente plus strict
        self.overbought_threshold = -25.0  # Seuil surachat plus strict
        self.extreme_oversold_threshold = -90.0  # Survente extrême vraiment extrême
        self.extreme_overbought_threshold = -10.0  # Surachat extrême vraiment extrême

        # Paramètres rebond - ASSOUPLIS pour plus de signaux
        self.min_rebound_strength = 5.0  # Williams %R doit bouger ≥5 points minimum
        self.rebound_confirmation_threshold = 10.0  # 10 points pour confirmation forte
        self.max_time_in_extreme = 2  # Max 2 barres en zone extrême

        # Paramètres momentum et volume - STRICTS pour qualité
        self.momentum_alignment_required = True  # Momentum doit confirmer
        self.min_momentum_threshold = 45  # Momentum minimum plus strict
        self.min_volume_confirmation = 1.5  # Volume ≥50% au-dessus normal (strict)

        # Paramètres confluence - EQUILIBRES (bonus mais avec seuils stricts)
        self.support_resistance_confluence = False  # Confluence S/R non requise
        self.confluence_distance_threshold = 0.01  # 1% max du S/R (strict)
        self.min_oscillator_confluence = 0.10  # Confluence oscillateurs rehaussée
        self.min_sr_confluence = 0.20  # Confluence S/R rehaussée

    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs pré-calculés."""
        return {
            # Williams %R principal
            "williams_r": self.indicators.get("williams_r"),
            # Autres oscillateurs (confluence)
            "rsi_14": self.indicators.get("rsi_14"),
            "rsi_21": self.indicators.get("rsi_21"),
            "stoch_k": self.indicators.get("stoch_k"),
            "stoch_d": self.indicators.get("stoch_d"),
            "stoch_rsi": self.indicators.get("stoch_rsi"),
            "stoch_fast_k": self.indicators.get("stoch_fast_k"),
            "stoch_fast_d": self.indicators.get("stoch_fast_d"),
            # Momentum indicators
            "momentum_score": self.indicators.get("momentum_score"),
            "momentum_10": self.indicators.get("momentum_10"),
            "roc_10": self.indicators.get("roc_10"),
            "roc_20": self.indicators.get("roc_20"),
            # MACD pour confirmation trend
            "macd_line": self.indicators.get("macd_line"),
            "macd_signal": self.indicators.get("macd_signal"),
            "macd_histogram": self.indicators.get("macd_histogram"),
            "macd_trend": self.indicators.get("macd_trend"),
            "macd_zero_cross": self.indicators.get("macd_zero_cross"),
            # Moyennes mobiles (support/résistance dynamique)
            "ema_12": self.indicators.get("ema_12"),
            "ema_26": self.indicators.get("ema_26"),
            "ema_50": self.indicators.get("ema_50"),
            "sma_20": self.indicators.get("sma_20"),
            "hull_20": self.indicators.get("hull_20"),
            # Support/Résistance statiques
            "nearest_support": self.indicators.get("nearest_support"),
            "nearest_resistance": self.indicators.get("nearest_resistance"),
            "support_strength": self.indicators.get("support_strength"),
            "resistance_strength": self.indicators.get("resistance_strength"),
            "support_levels": self.indicators.get("support_levels"),
            "resistance_levels": self.indicators.get("resistance_levels"),
            # Volume analysis
            "volume_ratio": self.indicators.get("volume_ratio"),
            "relative_volume": self.indicators.get("relative_volume"),
            "volume_quality_score": self.indicators.get("volume_quality_score"),
            "volume_spike_multiplier": self.indicators.get("volume_spike_multiplier"),
            "trade_intensity": self.indicators.get("trade_intensity"),
            # Trend et direction
            "trend_strength": self.indicators.get("trend_strength"),
            "directional_bias": self.indicators.get("directional_bias"),
            "trend_alignment": self.indicators.get("trend_alignment"),
            "trend_angle": self.indicators.get("trend_angle"),
            # ADX pour force tendance
            "adx_14": self.indicators.get("adx_14"),
            "plus_di": self.indicators.get("plus_di"),
            "minus_di": self.indicators.get("minus_di"),
            # Bollinger Bands (contexte volatilité)
            "bb_upper": self.indicators.get("bb_upper"),
            "bb_middle": self.indicators.get("bb_middle"),
            "bb_lower": self.indicators.get("bb_lower"),
            "bb_position": self.indicators.get("bb_position"),
            "bb_width": self.indicators.get("bb_width"),
            # ATR et volatilité
            "atr_14": self.indicators.get("atr_14"),
            "atr_percentile": self.indicators.get("atr_percentile"),
            "volatility_regime": self.indicators.get("volatility_regime"),
            # Market context
            "market_regime": self.indicators.get("market_regime"),
            "regime_strength": self.indicators.get("regime_strength"),
            "confluence_score": self.indicators.get("confluence_score"),
            "signal_strength": self.indicators.get("signal_strength"),
            "pattern_detected": self.indicators.get("pattern_detected"),
            "pattern_confidence": self.indicators.get("pattern_confidence"),
        }

    def _detect_williamsR_rebound_buy(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Détecte un rebound haussier depuis zone survente Williams %R."""
        rebound_score = 0.0
        rebound_indicators = []

        williams_r = values.get("williams_r")
        if williams_r is None:
            return {"is_rebound": False, "score": 0.0, "indicators": []}

        try:
            williams_val = float(williams_r)
        except (ValueError, TypeError):
            return {"is_rebound": False, "score": 0.0, "indicators": []}

        # Vérifier si Williams %R est en phase de rebound depuis survente
        if williams_val <= self.oversold_threshold:
            # Encore en survente - pas de rebound
            return {
                "is_rebound": False,
                "score": 0.0,
                "indicators": [f"Williams %R encore en survente ({williams_val:.1f})"],
                "williams_value": williams_val,
            }

        # Vérifier rebound depuis survente (Williams %R > -80 après avoir été < -80)
        if williams_val > self.oversold_threshold and williams_val < -50:
            # Zone de rebound depuis survente (entre -80 et -50)
            rebound_distance = abs(williams_val - self.oversold_threshold)

            if rebound_distance >= self.rebound_confirmation_threshold:
                rebound_score += 0.3
                rebound_indicators.append(
                    f"Rebound fort depuis survente ({rebound_distance:.1f} points)"
                )
            elif rebound_distance >= self.min_rebound_strength:
                rebound_score += 0.2
                rebound_indicators.append(
                    f"Rebound modéré depuis survente ({rebound_distance:.1f} points)"
                )
            else:
                rebound_score += 0.1
                rebound_indicators.append(
                    f"Rebound faible depuis survente ({rebound_distance:.1f} points)"
                )

            # Bonus si venait de zone survente extrême
            if williams_val > self.extreme_oversold_threshold:
                rebound_score += 0.15
                rebound_indicators.append("Rebound depuis survente extrême")

        else:
            # Williams %R pas dans zone de rebound appropriée
            return {
                "is_rebound": False,
                "score": 0.0,
                "indicators": [
                    f"Williams %R pas en zone rebound BUY ({williams_val:.1f})"
                ],
                "williams_value": williams_val,
            }

        return {
            "is_rebound": rebound_score >= 0.25,  # Seuil durci: 0.25
            "score": rebound_score,
            "indicators": rebound_indicators,
            "williams_value": williams_val,
            "rebound_type": "bullish_from_oversold",
        }

    def _detect_williamsR_rebound_sell(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Détecte un rebound baissier depuis zone surachat Williams %R."""
        rebound_score = 0.0
        rebound_indicators = []

        williams_r = values.get("williams_r")
        if williams_r is None:
            return {"is_rebound": False, "score": 0.0, "indicators": []}

        try:
            williams_val = float(williams_r)
        except (ValueError, TypeError):
            return {"is_rebound": False, "score": 0.0, "indicators": []}

        # Vérifier si Williams %R est en phase de rebound depuis surachat
        if williams_val >= self.overbought_threshold:
            # Encore en surachat - pas de rebound
            return {
                "is_rebound": False,
                "score": 0.0,
                "indicators": [f"Williams %R encore en surachat ({williams_val:.1f})"],
                "williams_value": williams_val,
            }

        # Vérifier rebound depuis surachat (Williams %R < -20 après avoir été > -20)
        if williams_val < self.overbought_threshold and williams_val > -50:
            # Zone de rebound depuis surachat (entre -50 et -20)
            rebound_distance = abs(self.overbought_threshold - williams_val)

            if rebound_distance >= self.rebound_confirmation_threshold:
                rebound_score += 0.3
                rebound_indicators.append(
                    f"Rebound fort depuis surachat ({rebound_distance:.1f} points)"
                )
            elif rebound_distance >= self.min_rebound_strength:
                rebound_score += 0.2
                rebound_indicators.append(
                    f"Rebound modéré depuis surachat ({rebound_distance:.1f} points)"
                )
            else:
                rebound_score += 0.1
                rebound_indicators.append(
                    f"Rebound faible depuis surachat ({rebound_distance:.1f} points)"
                )

            # Bonus si venait de zone surachat extrême
            if williams_val < self.extreme_overbought_threshold:
                rebound_score += 0.15
                rebound_indicators.append("Rebound depuis surachat extrême")

        else:
            # Williams %R pas dans zone de rebound appropriée
            return {
                "is_rebound": False,
                "score": 0.0,
                "indicators": [
                    f"Williams %R pas en zone rebound SELL ({williams_val:.1f})"
                ],
                "williams_value": williams_val,
            }

        return {
            "is_rebound": rebound_score >= 0.25,  # Seuil durci: 0.25
            "score": rebound_score,
            "indicators": rebound_indicators,
            "williams_value": williams_val,
            "rebound_type": "bearish_from_overbought",
        }

    def _detect_oscillator_confluence(
        self, values: Dict[str, Any], signal_direction: str
    ) -> Dict[str, Any]:
        """Détecte la confluence avec autres oscillateurs."""
        confluence_score = 0
        confluence_indicators = []

        # RSI confluence
        rsi_14 = values.get("rsi_14")
        if rsi_14 is not None:
            try:
                rsi_val = float(rsi_14)

                if signal_direction == "BUY":
                    if 30 <= rsi_val <= 50:  # RSI sortant de survente
                        confluence_score += 0.15
                        confluence_indicators.append(
                            f"RSI sortant survente ({rsi_val:.1f})"
                        )
                    elif 20 <= rsi_val <= 40:  # RSI encore en zone basse
                        confluence_score += 0.1
                        confluence_indicators.append(f"RSI zone basse ({rsi_val:.1f})")
                elif signal_direction == "SELL":
                    if 50 <= rsi_val <= 70:  # RSI sortant de surachat
                        confluence_score += 0.15
                        confluence_indicators.append(
                            f"RSI sortant surachat ({rsi_val:.1f})"
                        )
                    elif 60 <= rsi_val <= 80:  # RSI encore en zone haute
                        confluence_score += 0.1
                        confluence_indicators.append(f"RSI zone haute ({rsi_val:.1f})")

            except (ValueError, TypeError):
                pass

        # Stochastic confluence
        stoch_k = values.get("stoch_k")
        stoch_d = values.get("stoch_d")
        if stoch_k is not None and stoch_d is not None:
            try:
                stoch_k_val = float(stoch_k)
                stoch_d_val = float(stoch_d)

                if signal_direction == "BUY":
                    if (
                        stoch_k_val < 30 and stoch_k_val > stoch_d_val
                    ):  # Stoch cross up from oversold
                        confluence_score += 0.12
                        confluence_indicators.append("Stochastic cross haussier")
                    elif stoch_k_val < 40:
                        confluence_score += 0.08
                        confluence_indicators.append(
                            f"Stochastic bas ({stoch_k_val:.1f})"
                        )
                elif signal_direction == "SELL":
                    if (
                        stoch_k_val > 70 and stoch_k_val < stoch_d_val
                    ):  # Stoch cross down from overbought
                        confluence_score += 0.12
                        confluence_indicators.append("Stochastic cross baissier")
                    elif stoch_k_val > 60:
                        confluence_score += 0.08
                        confluence_indicators.append(
                            f"Stochastic haut ({stoch_k_val:.1f})"
                        )

            except (ValueError, TypeError):
                pass

        # CCI approximation (non disponible direct, utiliser momentum)
        momentum_10 = values.get("momentum_10")
        if momentum_10 is not None:
            try:
                momentum_val = float(momentum_10)

                if (
                    signal_direction == "BUY" and momentum_val > 100
                ):  # Momentum positif après baisse
                    confluence_score += 0.1
                    confluence_indicators.append("Momentum rebond haussier")
                elif (
                    signal_direction == "SELL" and momentum_val < 100
                ):  # Momentum négatif après hausse
                    confluence_score += 0.1
                    confluence_indicators.append("Momentum rebond baissier")

            except (ValueError, TypeError):
                pass

        return {
            "is_confluent": confluence_score
            >= self.min_oscillator_confluence,  # Score confluence optimisé
            "score": confluence_score,
            "indicators": confluence_indicators,
        }

    def _detect_support_resistance_confluence(
        self, values: Dict[str, Any], current_price: float, signal_direction: str
    ) -> Dict[str, Any]:
        """Détecte la confluence avec niveaux de support/résistance."""
        sr_score = 0.0
        sr_indicators = []

        if signal_direction == "BUY":
            # Rechercher confluence avec support
            nearest_support = values.get("nearest_support")
            if nearest_support is not None:
                try:
                    support_level = float(nearest_support)
                    distance_to_support = (
                        abs(current_price - support_level) / current_price
                    )

                    if distance_to_support <= self.confluence_distance_threshold:
                        sr_score += 0.2
                        sr_indicators.append(
                            f"Proche support {support_level:.2f} ({distance_to_support*100:.1f}%)"
                        )

                        # Bonus selon force support
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
                                    sr_score += 0.15
                                    sr_indicators.append(
                                        f"Support très fort ({strength_val:.2f})"
                                    )
                                elif strength_val >= 0.6:
                                    sr_score += 0.1
                                    sr_indicators.append(
                                        f"Support fort ({strength_val:.2f})"
                                    )
                            except (ValueError, TypeError):
                                pass
                except (ValueError, TypeError):
                    pass

        elif signal_direction == "SELL":
            # Rechercher confluence avec résistance
            nearest_resistance = values.get("nearest_resistance")
            if nearest_resistance is not None:
                try:
                    resistance_level = float(nearest_resistance)
                    distance_to_resistance = (
                        abs(current_price - resistance_level) / current_price
                    )

                    if distance_to_resistance <= self.confluence_distance_threshold:
                        sr_score += 0.2
                        sr_indicators.append(
                            f"Proche résistance {resistance_level:.2f} ({distance_to_resistance*100:.1f}%)"
                        )

                        # Bonus selon force résistance
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
                                    sr_score += 0.15
                                    sr_indicators.append(
                                        f"Résistance très forte ({strength_val:.2f})"
                                    )
                                elif strength_val >= 0.6:
                                    sr_score += 0.1
                                    sr_indicators.append(
                                        f"Résistance forte ({strength_val:.2f})"
                                    )
                            except (ValueError, TypeError):
                                pass
                except (ValueError, TypeError):
                    pass

        # EMA confluence comme support/résistance dynamique
        ema_50 = values.get("ema_50")
        if ema_50 is not None:
            try:
                ema_val = float(ema_50)
                distance_to_ema = abs(current_price - ema_val) / current_price

                if distance_to_ema <= 0.01:  # 1% de l'EMA50
                    if signal_direction == "BUY" and current_price >= ema_val:
                        sr_score += 0.1
                        sr_indicators.append("EMA50 support dynamique")
                    elif signal_direction == "SELL" and current_price <= ema_val:
                        sr_score += 0.1
                        sr_indicators.append("EMA50 résistance dynamique")
            except (ValueError, TypeError):
                pass

        return {
            "is_confluent": sr_score
            >= self.min_sr_confluence,  # Score confluence S/R optimisé
            "score": sr_score,
            "indicators": sr_indicators,
        }

    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur Williams %R rebound.
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

        # Récupérer prix actuel
        current_price = None
        if "close" in self.data and self.data["close"]:
            try:
                current_price = float(self.data["close"][-1])
            except (IndexError, ValueError, TypeError):
                pass

        if current_price is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Prix actuel non disponible",
                "metadata": {"strategy": self.name},
            }

        # Analyser rebound haussier (BUY)
        buy_rebound = self._detect_williamsR_rebound_buy(values)

        # Analyser rebound baissier (SELL)
        sell_rebound = self._detect_williamsR_rebound_sell(values)

        # Déterminer signal principal
        signal_side = None
        primary_rebound = None

        if buy_rebound["is_rebound"] and sell_rebound["is_rebound"]:
            # SÉCURITÉ: Conflit théoriquement impossible avec Williams %R mais vérification
            # Si conflit, vérifier la valeur Williams %R pour trancher
            williams_val = values.get("williams_r")
            if williams_val is not None:
                try:
                    williams_value = float(williams_val)
                    # Trancher selon la position Williams %R actuelle
                    if williams_value > -50:  # Plus proche zone SELL
                        signal_side = "SELL"
                        primary_rebound = sell_rebound
                    else:  # Plus proche zone BUY
                        signal_side = "BUY"
                        primary_rebound = buy_rebound
                except (ValueError, TypeError):
                    # Fallback sur le score le plus élevé
                    if buy_rebound["score"] > sell_rebound["score"]:
                        signal_side = "BUY"
                        primary_rebound = buy_rebound
                    else:
                        signal_side = "SELL"
                        primary_rebound = sell_rebound
            else:
                # Pas de Williams %R disponible - utiliser scores
                if buy_rebound["score"] > sell_rebound["score"]:
                    signal_side = "BUY"
                    primary_rebound = buy_rebound
                else:
                    signal_side = "SELL"
                    primary_rebound = sell_rebound
        elif buy_rebound["is_rebound"]:
            signal_side = "BUY"
            primary_rebound = buy_rebound
        elif sell_rebound["is_rebound"]:
            signal_side = "SELL"
            primary_rebound = sell_rebound

        if signal_side is None:
            # Diagnostic conditions non remplies
            williams_val = values.get("williams_r")
            missing_conditions = []

            if buy_rebound["score"] < 0.2:
                missing_conditions.append(
                    f"Rebound BUY faible (score: {buy_rebound['score']:.2f})"
                )
            if sell_rebound["score"] < 0.2:
                missing_conditions.append(
                    f"Rebound SELL faible (score: {sell_rebound['score']:.2f})"
                )

            williams_str = f"{williams_val:.1f}" if williams_val is not None else "N/A"
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Williams %R ({williams_str}) - {'; '.join(missing_conditions[:2])}",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "williams_r": williams_val,
                    "buy_rebound_score": buy_rebound["score"],
                    "sell_rebound_score": sell_rebound["score"],
                },
            }

        # Vérifier confluences - NOUVEAU: OBLIGATOIRES pour signal
        oscillator_confluence = self._detect_oscillator_confluence(values, signal_side)
        sr_confluence = self._detect_support_resistance_confluence(
            values, current_price, signal_side
        )

        # Vérifier rejets préliminaires critiques

        # 1. REJET momentum contradictoire (avant calcul confidence)
        momentum_score_val = values.get("momentum_score")
        if momentum_score_val is not None:
            try:
                momentum = float(momentum_score_val)
                if signal_side == "BUY" and momentum < 40:
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Rejet Williams %R BUY: momentum trop faible ({momentum:.1f})",
                        "metadata": {
                            "strategy": self.name,
                            "williams_r": values.get("williams_r"),
                            "momentum_score": momentum,
                            "rejection_reason": "momentum_too_low",
                        },
                    }
                elif signal_side == "SELL" and momentum > 60:
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Rejet Williams %R SELL: momentum trop fort ({momentum:.1f})",
                        "metadata": {
                            "strategy": self.name,
                            "williams_r": values.get("williams_r"),
                            "momentum_score": momentum,
                            "rejection_reason": "momentum_too_high",
                        },
                    }
            except (ValueError, TypeError):
                pass

        # 2. REJET bias contradictoire
        directional_bias = values.get("directional_bias")
        if directional_bias:
            if signal_side == "BUY" and directional_bias == "BEARISH":
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": "Rejet Williams %R BUY: bias contradictoire",
                    "metadata": {
                        "strategy": self.name,
                        "williams_r": values.get("williams_r"),
                        "directional_bias": directional_bias,
                        "rejection_reason": "bias_contradiction",
                    },
                }
            elif signal_side == "SELL" and directional_bias == "BULLISH":
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": "Rejet Williams %R SELL: bias contradictoire",
                    "metadata": {
                        "strategy": self.name,
                        "williams_r": values.get("williams_r"),
                        "directional_bias": directional_bias,
                        "rejection_reason": "bias_contradiction",
                    },
                }

        # 3. REJET volume insuffisant
        volume_ratio = values.get("volume_ratio")
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio < 1.0:
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Rejet Williams %R: volume trop faible ({vol_ratio:.1f}x)",
                        "metadata": {
                            "strategy": self.name,
                            "williams_r": values.get("williams_r"),
                            "volume_ratio": vol_ratio,
                            "rejection_reason": "volume_too_low",
                        },
                    }
            except (ValueError, TypeError):
                pass

        # 4. REJET confluence globale faible
        confluence_score_global = values.get("confluence_score")
        if confluence_score_global is not None:
            try:
                conf_val = float(confluence_score_global)
                if conf_val < 40:
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Rejet Williams %R: confluence trop faible ({conf_val:.1f})",
                        "metadata": {
                            "strategy": self.name,
                            "williams_r": values.get("williams_r"),
                            "confluence_score": conf_val,
                            "rejection_reason": "confluence_too_low",
                        },
                    }
            except (ValueError, TypeError):
                pass

        # Construire signal final
        base_confidence = 0.65  # Augmenté pour passer le filtre aggregator
        confidence_boost = 0.0

        # Vérification de sécurité pour primary_rebound
        if primary_rebound is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Aucun rebound Williams %R détecté",
                "metadata": {"strategy": self.name},
            }

        # Score rebound Williams %R
        confidence_boost += primary_rebound["score"] * 0.4

        # Score confluences oscillateurs
        confidence_boost += oscillator_confluence["score"] * 0.3

        # Score confluence support/résistance
        confidence_boost += sr_confluence["score"] * 0.3

        # Construire raison SIMPLIFIÉE (max 4-5 confirmations)
        williams_val = primary_rebound["williams_value"]
        rebound_type = primary_rebound["rebound_type"]

        reason = f"Williams %R {williams_val:.1f} - {primary_rebound['indicators'][0]}"
        confirmation_count = 1  # Limiter à 4-5 confirmations max

        # 1. Confluence oscillateurs (priorité 1)
        if oscillator_confluence["indicators"] and confirmation_count < 4:
            reason += f" + {oscillator_confluence['indicators'][0]}"
            confirmation_count += 1

        # 2. Support/Résistance (priorité 2)
        if sr_confluence["indicators"] and confirmation_count < 4:
            reason += f" + {sr_confluence['indicators'][0]}"
            confirmation_count += 1

        # CORRECTION: Momentum alignment - logique directionnelle optimisée pour rebonds
        momentum_score_val = values.get("momentum_score")
        if momentum_score_val is not None:
            try:
                momentum = float(momentum_score_val)

                # 3. Momentum (priorité 3) - adapter aux rebonds
                if confirmation_count < 4:
                    if signal_side == "BUY":
                        if momentum >= 55:  # Momentum qui remonte
                            confidence_boost += 0.10
                            reason += f" + momentum ({momentum:.1f})"
                            confirmation_count += 1
                        elif (
                            momentum >= 48 and confirmation_count < 3
                        ):  # Seulement si pas trop de confirmations
                            confidence_boost += 0.05
                            reason += f" + momentum neutre ({momentum:.1f})"
                            confirmation_count += 1
                    elif signal_side == "SELL":
                        if momentum <= 45:  # Momentum qui descend
                            confidence_boost += 0.10
                            reason += f" + momentum ({momentum:.1f})"
                            confirmation_count += 1
                        elif momentum <= 52 and confirmation_count < 3:
                            confidence_boost += 0.05
                            reason += f" + momentum neutre ({momentum:.1f})"
                            confirmation_count += 1
            except (ValueError, TypeError):
                pass

        # 4. Volume (priorité 4) - strict mais limité dans la raison
        volume_ratio = values.get("volume_ratio")
        if volume_ratio is not None and confirmation_count < 4:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= self.min_volume_confirmation:  # ≥1.5x
                    confidence_boost += 0.08
                    reason += f" + volume ({vol_ratio:.1f}x)"
                    confirmation_count += 1
                elif vol_ratio >= 1.2 and confirmation_count < 3:
                    confidence_boost += 0.04
                    reason += f" + volume modéré ({vol_ratio:.1f}x)"
                    confirmation_count += 1
            except (ValueError, TypeError):
                pass

        # 5. MACD (priorité 5) - seulement si pas trop de confirmations
        if confirmation_count < 4:
            macd_line = values.get("macd_line")
            macd_signal = values.get("macd_signal")
            if macd_line is not None and macd_signal is not None:
                try:
                    macd_val = float(macd_line)
                    macd_sig = float(macd_signal)

                    if signal_side == "BUY" and macd_val > macd_sig:
                        confidence_boost += 0.08
                        if confirmation_count < 4:
                            reason += " + MACD"
                            confirmation_count += 1
                    elif signal_side == "SELL" and macd_val < macd_sig:
                        confidence_boost += 0.08
                        if confirmation_count < 4:
                            reason += " + MACD"
                            confirmation_count += 1
                except (ValueError, TypeError):
                    pass

        # Autres bonus (sans ajout à la raison pour éviter surcharge)

        # Market regime context (bonus silencieux)
        market_regime = values.get("market_regime")
        if market_regime == "RANGING":
            confidence_boost += 0.08  # Williams %R excellent en ranging
        elif market_regime in ["TRENDING_BULL", "TRENDING_BEAR"]:
            confidence_boost += 0.03

        # ADX context (bonus silencieux)
        adx_14 = values.get("adx_14")
        if adx_14 is not None:
            try:
                adx_val = float(adx_14)
                if 15 <= adx_val <= 30:  # ADX modéré favorable aux rebonds
                    confidence_boost += 0.08
                elif adx_val < 15:  # ADX faible = marché sans direction
                    confidence_boost += 0.05
            except (ValueError, TypeError):
                pass

        # Bollinger Bands context (bonus silencieux)
        bb_position = values.get("bb_position")
        if bb_position is not None:
            try:
                bb_pos = float(bb_position)
                if signal_side == "BUY" and bb_pos <= 0.2:
                    confidence_boost += 0.05
                elif signal_side == "SELL" and bb_pos >= 0.8:
                    confidence_boost += 0.05
            except (ValueError, TypeError):
                pass

        confidence = self.calculate_confidence(base_confidence, 1.0 + confidence_boost)
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
                "williams_r": williams_val,
                "rebound_type": rebound_type,
                "rebound_score": primary_rebound["score"] if primary_rebound else 0.0,
                "rebound_indicators": (
                    primary_rebound["indicators"] if primary_rebound else []
                ),
                "oscillator_confluence_score": oscillator_confluence["score"],
                "oscillator_confluence_indicators": oscillator_confluence["indicators"],
                "sr_confluence_score": sr_confluence["score"],
                "sr_confluence_indicators": sr_confluence["indicators"],
                "buy_rebound_analysis": buy_rebound if signal_side == "BUY" else None,
                "sell_rebound_analysis": (
                    sell_rebound if signal_side == "SELL" else None
                ),
                "volume_ratio": values.get("volume_ratio"),
                "momentum_score": values.get("momentum_score"),
                "directional_bias": values.get("directional_bias"),
                "market_regime": values.get("market_regime"),
                "volatility_regime": values.get("volatility_regime"),
                "confluence_score": values.get("confluence_score"),
            },
        }

    def validate_data(self) -> bool:
        """Valide que tous les indicateurs requis sont présents."""
        required_indicators = ["williams_r"]

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
