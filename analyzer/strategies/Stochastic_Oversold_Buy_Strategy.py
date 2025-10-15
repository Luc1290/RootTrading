"""
Stochastic_Oversold_Buy_Strategy - Stratégie basée sur les conditions oversold du Stochastic.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class Stochastic_Oversold_Buy_Strategy(BaseStrategy):
    """
    Stratégie utilisant les conditions de survente du Stochastic pour identifier les opportunités d'achat.

    Le Stochastic mesure la position du prix de clôture par rapport aux highs/lows récents :
    - %K = (Close - Low_n) / (High_n - Low_n) * 100
    - %D = moyenne mobile de %K
    - Oversold = %K et %D < 20 (survente)
    - Signal d'achat = sortie de survente + croisement %K > %D + confirmations

    Signaux générés:
    - BUY: Stochastic sort de zone oversold + croisement %K > %D + confirmations
    - Pas de signaux SELL (stratégie focalisée sur les achats en survente)
    """

    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres Stochastic ASSOUPLIS - Oversolds réalistes
        self.oversold_threshold = 22  # Seuil assoupli pour plus de signaux
        self.exit_oversold_threshold = 30  # Seuil exit assoupli
        self.overbought_threshold = 80  # Seuil de surachat standard
        self.min_crossover_separation = (
            1.5  # Distance assouplie pour petits croisements
        )

    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs Stochastic et confirmation."""
        return {
            # Stochastic principal
            "stoch_k": self.indicators.get("stoch_k"),
            "stoch_d": self.indicators.get("stoch_d"),
            "stoch_fast_k": self.indicators.get("stoch_fast_k"),
            "stoch_fast_d": self.indicators.get("stoch_fast_d"),
            "stoch_rsi": self.indicators.get("stoch_rsi"),
            # RSI pour confluence
            "rsi_14": self.indicators.get("rsi_14"),
            "rsi_21": self.indicators.get("rsi_21"),
            # Williams %R (similaire au Stochastic)
            "williams_r": self.indicators.get("williams_r"),
            # Moyennes mobiles pour contexte de tendance
            "ema_12": self.indicators.get("ema_12"),
            "ema_26": self.indicators.get("ema_26"),
            "ema_50": self.indicators.get("ema_50"),
            "hull_20": self.indicators.get("hull_20"),
            # MACD pour confirmation momentum
            "macd_line": self.indicators.get("macd_line"),
            "macd_signal": self.indicators.get("macd_signal"),
            "macd_histogram": self.indicators.get("macd_histogram"),
            "macd_trend": self.indicators.get("macd_trend"),
            # Tendance et direction
            "trend_strength": self.indicators.get("trend_strength"),
            "directional_bias": self.indicators.get("directional_bias"),
            "momentum_score": self.indicators.get("momentum_score"),
            # ADX pour force de tendance
            "adx_14": self.indicators.get("adx_14"),
            "plus_di": self.indicators.get("plus_di"),
            "minus_di": self.indicators.get("minus_di"),
            # Volume pour confirmation
            "volume_ratio": self.indicators.get("volume_ratio"),
            "volume_quality_score": self.indicators.get("volume_quality_score"),
            "trade_intensity": self.indicators.get("trade_intensity"),
            # Support/Résistance pour contexte
            "nearest_support": self.indicators.get("nearest_support"),
            "support_strength": self.indicators.get("support_strength"),
            "nearest_resistance": self.indicators.get("nearest_resistance"),
            # VWAP pour niveaux institutionnels
            "vwap_10": self.indicators.get("vwap_10"),
            "anchored_vwap": self.indicators.get("anchored_vwap"),
            # Bollinger Bands pour contexte volatilité
            "bb_lower": self.indicators.get("bb_lower"),
            "bb_position": self.indicators.get("bb_position"),
            "bb_width": self.indicators.get("bb_width"),
            # Market structure
            "market_regime": self.indicators.get("market_regime"),
            "volatility_regime": self.indicators.get("volatility_regime"),
            "regime_strength": self.indicators.get("regime_strength"),
            # Pattern et confluence
            "pattern_detected": self.indicators.get("pattern_detected"),
            "pattern_confidence": self.indicators.get("pattern_confidence"),
            "signal_strength": self.indicators.get("signal_strength"),
            "confluence_score": self.indicators.get("confluence_score"),
        }

    def _get_current_price(self) -> Optional[float]:
        """Récupère le prix actuel depuis les données OHLCV."""
        try:
            if self.data and "close" in self.data and self.data["close"]:
                return float(self.data["close"][-1])
        except (IndexError, ValueError, TypeError):
            pass
        return None

    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal d'achat basé sur les conditions oversold du Stochastic.
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

        # Analyser les conditions Stochastic
        stoch_analysis = self._analyze_stochastic_conditions(values)
        if stoch_analysis is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Données Stochastic non disponibles",
                "metadata": {"strategy": self.name},
            }

        # Vérifier les conditions d'achat oversold
        buy_condition = self._check_oversold_buy_conditions(stoch_analysis)
        if buy_condition is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": stoch_analysis.get(
                    "rejection_reason", "Conditions oversold non remplies"
                ),
                "metadata": {"strategy": self.name},
            }

        # Momentum ASSOUPLI - Pénalités graduées au lieu de rejets
        momentum_penalty = 0.0
        momentum_score = values.get("momentum_score")
        if momentum_score is not None:
            try:
                momentum_val = float(momentum_score)
                # PLUS de rejet dur - seulement pénalités graduées
                if momentum_val < 20:  # Seuil critique abaissé (35 -> 20)
                    momentum_penalty = -0.20  # Pénalité forte mais pas rejet
                elif momentum_val < 35:  # Momentum très faible
                    momentum_penalty = -0.15
                elif momentum_val < 45:  # Momentum faible
                    momentum_penalty = -0.10
            except (ValueError, TypeError):
                pass

        # Bias contradictoire = pénalité au lieu de rejet
        bias_penalty = 0.0
        directional_bias = values.get("directional_bias")
        if directional_bias == "BEARISH":
            bias_penalty = -0.15  # Pénalité au lieu de rejet

        # Market regime contradictoire = pénalité au lieu de rejet
        regime_penalty = 0.0
        market_regime = values.get("market_regime")
        if market_regime == "TRENDING_BEAR":
            regime_penalty = -0.20  # Pénalité forte mais pas rejet total

        # Créer le signal d'achat avec confirmations et pénalités
        return self._create_oversold_buy_signal(
            values,
            current_price or 0.0,
            stoch_analysis,
            buy_condition,
            momentum_penalty + bias_penalty + regime_penalty,
        )

    def _analyze_stochastic_conditions(
        self, values: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Analyse les conditions actuelles du Stochastic avec fallbacks."""
        stoch_k = values.get("stoch_k")
        stoch_d = values.get("stoch_d")

        # Fallback sur Fast Stochastic si standard manquant
        if stoch_k is None or stoch_d is None:
            stoch_k = values.get("stoch_fast_k")
            stoch_d = values.get("stoch_fast_d")

        if stoch_k is None or stoch_d is None:
            return None

        try:
            k_val = float(stoch_k)
            d_val = float(stoch_d)

            # SUPPRIMÉ : 0.0 peut être une survente extrême légitime !
            # Stoch = 0 = survente maximale, pas corruption

        except (ValueError, TypeError):
            return None

        # États du Stochastic
        is_oversold = (
            k_val <= self.oversold_threshold and d_val <= self.oversold_threshold
        )
        is_exiting_oversold = (
            k_val > self.oversold_threshold or d_val > self.oversold_threshold
        ) and (
            k_val <= self.exit_oversold_threshold
            or d_val <= self.exit_oversold_threshold
        )
        is_overbought = (
            k_val >= self.overbought_threshold or d_val >= self.overbought_threshold
        )

        # Croisement %K > %D (signal haussier)
        k_above_d = k_val > d_val
        crossover_strength = abs(k_val - d_val)

        # Analyser les Stochastic Fast si disponibles
        stoch_fast_k = values.get("stoch_fast_k")
        stoch_fast_d = values.get("stoch_fast_d")
        fast_analysis = None

        if stoch_fast_k is not None and stoch_fast_d is not None:
            try:
                fast_k = float(stoch_fast_k)
                fast_d = float(stoch_fast_d)
                fast_analysis = {
                    "fast_k": fast_k,
                    "fast_d": fast_d,
                    "fast_oversold": fast_k <= self.oversold_threshold
                    and fast_d <= self.oversold_threshold,
                    "fast_crossover": fast_k > fast_d,
                }
            except (ValueError, TypeError):
                fast_analysis = None

        # Déterminer les raisons de rejet potentielles
        rejection_reasons = []
        if is_overbought:
            rejection_reasons.append(
                f"Stochastic en surachat (K:{k_val:.1f}, D:{d_val:.1f})"
            )
        if not (is_oversold or is_exiting_oversold):
            rejection_reasons.append(
                f"Stochastic pas en survente (K:{k_val:.1f}, D:{d_val:.1f})"
            )
        if not k_above_d:
            rejection_reasons.append(
                f"Pas de croisement haussier K<D ({k_val:.1f}<{d_val:.1f})"
            )
        if crossover_strength < self.min_crossover_separation:
            rejection_reasons.append(
                f"Croisement trop faible ({crossover_strength:.1f})"
            )

        return {
            "stoch_k": k_val,
            "stoch_d": d_val,
            "is_oversold": is_oversold,
            "is_exiting_oversold": is_exiting_oversold,
            "is_overbought": is_overbought,
            "k_above_d": k_above_d,
            "crossover_strength": crossover_strength,
            "fast_analysis": fast_analysis,
            "rejection_reason": (
                "; ".join(rejection_reasons) if rejection_reasons else None
            ),
        }

    def _check_oversold_buy_conditions(
        self, stoch_analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Vérifie si les conditions d'achat oversold sont remplies."""
        # Rejeter si en surachat
        if stoch_analysis["is_overbought"]:
            return None

        # Condition principale: oversold OU sortie d'oversold
        if not (stoch_analysis["is_oversold"] or stoch_analysis["is_exiting_oversold"]):
            return None

        # ASSOUPLISSEMENT: Accepter les cas sans croisement si très oversold
        k_val = stoch_analysis["stoch_k"]
        d_val = stoch_analysis["stoch_d"]
        very_oversold = k_val <= 15 and d_val <= 15  # Très profondement oversold

        # Condition croisement ASSOUPLIE: %K > %D OU très oversold
        if not stoch_analysis["k_above_d"] and not very_oversold:
            return None

        # Condition force du croisement ASSOUPLIE pour très oversold
        min_separation = self.min_crossover_separation if not very_oversold else 0.5
        if stoch_analysis["crossover_strength"] < min_separation:
            return None

        # Déterminer le type de signal
        if very_oversold:
            signal_quality = "very_strong"
            signal_type = "deep_oversold_bounce"
        elif stoch_analysis["is_oversold"]:
            signal_quality = "strong"
            signal_type = "oversold_bounce"
        else:
            signal_quality = "moderate"
            signal_type = "oversold_exit"

        return {
            "signal_type": signal_type,
            "signal_quality": signal_quality,
            "crossover_strength": stoch_analysis["crossover_strength"],
        }

    def _create_oversold_buy_signal(
        self,
        values: Dict[str, Any],
        current_price: float,
        stoch_analysis: Dict[str, Any],
        buy_condition: Dict[str, Any],
        external_penalties: float = 0.0,
    ) -> Dict[str, Any]:
        """Crée le signal d'achat oversold avec confirmations."""
        signal_side = "BUY"  # Stratégie uniquement orientée achat
        base_confidence = 0.55  # Abaissée pour plus d'accessibilité (0.65 -> 0.55)
        confidence_boost = 0.0

        # Appliquer les pénalités externes
        confidence_boost += external_penalties

        stoch_k = stoch_analysis["stoch_k"]
        stoch_d = stoch_analysis["stoch_d"]
        signal_type = buy_condition["signal_type"]
        signal_quality = buy_condition["signal_quality"]

        # Construction de la raison CONCISE
        reason = f"Stoch oversold: K={stoch_k:.1f}, D={stoch_d:.1f}"
        reason_parts = []  # Limiter à 4-5 éléments max

        # Bonus selon le type de signal
        if signal_quality == "very_strong":
            confidence_boost += 0.25  # Bonus max pour très oversold
            reason_parts.append("très profonde")
        elif signal_quality == "strong":
            confidence_boost += 0.15
            reason_parts.append("profonde")
        else:
            confidence_boost += 0.10
            reason_parts.append("sortie")

        # Bonus selon la force du croisement
        crossover_strength = buy_condition["crossover_strength"]
        if crossover_strength > 15:
            confidence_boost += 0.12
            reason_parts.append(f"fort cross({crossover_strength:.1f})")
        elif crossover_strength > 8:
            confidence_boost += 0.08
            reason_parts.append(f"cross({crossover_strength:.1f})")
        elif crossover_strength < 3:
            confidence_boost -= 0.02

        # Confirmation avec Stochastic Fast
        fast_analysis = stoch_analysis.get("fast_analysis")
        if fast_analysis is not None:
            if fast_analysis["fast_oversold"] and fast_analysis["fast_crossover"]:
                confidence_boost += 0.10
                reason_parts.append("Fast+")
            elif fast_analysis["fast_crossover"]:
                confidence_boost += 0.05

        # Confirmation avec RSI
        rsi_14 = values.get("rsi_14")
        if rsi_14 is not None:
            try:
                rsi = float(rsi_14)
                # RSI assoupli pour plus de cas
                if rsi <= 25:  # RSI survente forte
                    confidence_boost += 0.12
                    reason_parts.append(f"RSI{rsi:.0f}")
                elif rsi <= 30:  # RSI favorable (maintenu)
                    confidence_boost += 0.08  # Légèrement augmenté
                elif rsi >= 65:  # Seuil maintenu
                    confidence_boost -= 0.05  # Pénalité réduite
                    reason += f" RSI élevé ({rsi:.1f})"
            except (ValueError, TypeError):
                pass

        # Confirmation avec Williams %R
        williams_r = values.get("williams_r")
        if williams_r is not None:
            try:
                wr = float(williams_r)
                if wr <= -80:  # Seuil assoupli (-80 au lieu de -85)
                    confidence_boost += 0.10  # Augmenté
                    reason += f" + Williams R survente forte ({wr:.1f})"
                elif wr <= -65:  # Seuil assoupli (-65 au lieu de -70)
                    confidence_boost += 0.06  # Augmenté
                    reason += f" + Williams R favorable ({wr:.1f})"
            except (ValueError, TypeError):
                pass

        # Confirmation avec MACD
        macd_line = values.get("macd_line")
        macd_signal = values.get("macd_signal")
        if macd_line is not None and macd_signal is not None:
            try:
                macd_val = float(macd_line)
                macd_sig = float(macd_signal)
                macd_bullish = macd_val > macd_sig

                if macd_bullish:
                    confidence_boost += 0.08  # Réduit de 0.12
                    reason += " + MACD haussier"
                elif macd_val > macd_sig - 0.0001:  # MACD proche du croisement
                    confidence_boost += 0.03  # Réduit de 0.05
                    reason += " + MACD proche croisement"
            except (ValueError, TypeError):
                pass

        # Contexte de tendance - directional_bias déjà traité en rejet
        trend_strength = values.get("trend_strength")
        directional_bias = values.get(
            "directional_bias"
        )  # CORRECTION: variable manquante
        market_regime = values.get("market_regime")  # AJOUT: récupérer market_regime

        # Directional bias déjà vérifié en amont, ici que BULLISH ou NEUTRAL
        if directional_bias == "BULLISH":
            confidence_boost += 0.10
            reason += " + bias haussier"

        if trend_strength is not None:
            # trend_strength DB format: weak/absent/strong/very_strong/extreme (lowercase)
            trend_str = str(trend_strength).lower()
            if trend_str in ["extreme", "very_strong"]:
                confidence_boost += 0.12
                reason += f" + tendance très forte ({trend_strength})"
            elif trend_str == "strong":
                confidence_boost += 0.08
                reason += f" + tendance forte ({trend_strength})"
            elif trend_str == "weak":
                confidence_boost += 0.03
                reason += f" + tendance faible ({trend_strength})"
            # 'absent' = pas de bonus

        # Support proche pour confluence
        nearest_support = values.get("nearest_support")
        if nearest_support is not None and current_price is not None:
            try:
                support = float(nearest_support)
                distance_to_support = abs(current_price - support) / current_price

                if distance_to_support <= 0.03:  # Seuil assoupli (3% au lieu de 1.5%)
                    confidence_boost += 0.12  # Augmenté
                    reason += " + très proche support"
                elif distance_to_support <= 0.05:  # Seuil assoupli (5% au lieu de 3%)
                    confidence_boost += 0.07  # Augmenté
                    reason += " + support proche"
            except (ValueError, TypeError):
                pass

        # Bollinger Bands pour contexte
        bb_lower = values.get("bb_lower")
        bb_position = values.get("bb_position")
        if bb_lower is not None and current_price is not None:
            try:
                bb_low = float(bb_lower)
                if current_price <= bb_low * 1.02:  # Prix près BB lower
                    confidence_boost += 0.10
                    reason += " + près BB inférieure"
            except (ValueError, TypeError):
                pass

        if bb_position is not None:
            try:
                pos = float(bb_position)
                if pos <= 0.2:  # Position basse dans les BB
                    confidence_boost += 0.08
                    reason += " + position BB basse"
            except (ValueError, TypeError):
                pass

        # VWAP pour contexte institutionnel
        vwap_10 = values.get("vwap_10")
        if vwap_10 is not None and current_price is not None:
            try:
                vwap = float(vwap_10)
                if current_price < vwap:  # Prix sous VWAP = potentiel rebond
                    confidence_boost += 0.08
                    reason += " + prix < VWAP"
            except (ValueError, TypeError):
                pass

        # Volume pour confirmation
        volume_ratio = values.get("volume_ratio")
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= 2.0:  # Volume très élevé
                    confidence_boost += 0.12
                    reason += f" + volume exceptionnel ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.2:  # Volume élevé (assoupli 1.5 -> 1.2)
                    confidence_boost += 0.08
                    reason += f" + volume élevé ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.0:  # Volume normal = petit bonus
                    confidence_boost += 0.03
                    reason += f" + volume normal ({vol_ratio:.1f}x)"
            except (ValueError, TypeError):
                pass

        # Market regime
        # Market regime - TRENDING_BEAR déjà rejeté en amont
        if market_regime == "RANGING":
            confidence_boost += 0.15  # Bonus fort en ranging (oversold works)
            reason += " + ranging"
        elif market_regime == "TRENDING_BULL":
            confidence_boost += 0.10  # Bonus en trend bull
            reason += " + haussier"

        # Volatilité
        volatility_regime = values.get("volatility_regime")
        if volatility_regime == "high":
            confidence_boost += 0.05  # Haute volatilité = rebonds plus forts
            reason += " + volatilité élevée"
        elif volatility_regime == "normal":
            confidence_boost += 0.03
            reason += " + volatilité normale"

        # Pattern detection
        pattern_detected = values.get("pattern_detected")
        pattern_confidence = values.get("pattern_confidence")
        if pattern_detected and pattern_confidence is not None:
            try:
                confidence = float(pattern_confidence)
                if confidence > 70:
                    confidence_boost += 0.08
                    reason += " + pattern détecté"
            except (ValueError, TypeError):
                pass

        # Confluence score
        confluence_score = values.get("confluence_score")
        if confluence_score is not None:
            try:
                confluence = float(confluence_score)
                if confluence > 70:
                    confidence_boost += 0.10
                    reason += " + confluence élevée"
                elif confluence > 50:
                    confidence_boost += 0.05
                    reason += " + confluence modérée"
            except (ValueError, TypeError):
                pass

        # SUPPRESSION du filtre interne - laisser aggregator décider
        # raw_confidence = base_confidence * (1 + confidence_boost)
        # Pas de rejet interne, l'aggregator gère les seuils

        # Construire reason final CONCIS (max 4-5 éléments)
        final_reason = reason + " " + " ".join(reason_parts[:4])

        # Calcul final avec clamp à 1.0
        confidence = min(
            1.0, self.calculate_confidence(base_confidence, 1 + confidence_boost)
        )
        strength: str = self.get_strength_from_confidence(confidence)

        return {
            "side": signal_side,
            "confidence": confidence,
            "strength": strength,
            "reason": final_reason,
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "current_price": current_price,
                "stoch_k": stoch_k,
                "stoch_d": stoch_d,
                "signal_type": signal_type,
                "signal_quality": signal_quality,
                "crossover_strength": crossover_strength,
                "is_oversold": stoch_analysis["is_oversold"],
                "is_exiting_oversold": stoch_analysis["is_exiting_oversold"],
                "fast_analysis": fast_analysis,
                "rsi_14": values.get("rsi_14"),
                "williams_r": values.get("williams_r"),
                "macd_line": values.get("macd_line"),
                "macd_signal": values.get("macd_signal"),
                "directional_bias": values.get("directional_bias"),
                "trend_strength": values.get("trend_strength"),
                "nearest_support": values.get("nearest_support"),
                "bb_position": values.get("bb_position"),
                "volume_ratio": values.get("volume_ratio"),
                "market_regime": values.get("market_regime"),
                "volatility_regime": values.get("volatility_regime"),
                "confluence_score": values.get("confluence_score"),
            },
        }

    def validate_data(self) -> bool:
        """Valide que les données Stochastic nécessaires sont présentes."""
        if not super().validate_data():
            return False

        # Validation ASSOUPLIE avec fallbacks possibles
        # Vérifier si on a au moins un Stochastic (standard ou fast)
        stoch_available = (
            self.indicators.get("stoch_k") is not None
            and self.indicators.get("stoch_d") is not None
        ) or (
            self.indicators.get("stoch_fast_k") is not None
            and self.indicators.get("stoch_fast_d") is not None
        )

        if not stoch_available:
            logger.warning(
                f"{self.name}: Aucun Stochastic disponible (ni standard ni fast)"
            )
            return False

        # TODO: Ajouter fallback calcul Stochastic depuis OHLC si nécessaire
        # Pour l'instant, on exige au moins un type disponible

        return True
