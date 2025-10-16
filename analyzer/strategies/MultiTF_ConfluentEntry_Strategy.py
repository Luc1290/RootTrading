"""
MultiTF_ConfluentEntry_Strategy - Stratégie basée sur la confluence multi-timeframes.
OPTIMISÉE POUR CRYPTO SPOT INTRADAY
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class MultiTF_ConfluentEntry_Strategy(BaseStrategy):
    """
    Stratégie de confluence multi-timeframes pour des entrées précises.

    Utilise la confluence de plusieurs éléments techniques sur différents timeframes :
    - Trend alignment (toutes les moyennes mobiles alignées)
    - Signal strength élevé
    - Confluence score élevé
    - Support/résistance respectés
    - Volume et momentum favorables

    Signaux générés:
    - BUY: Confluence haussière sur multiple timeframes + confirmations
    - SELL: Confluence baissière sur multiple timeframes + confirmations
    """

    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres ASSOUPLIS pour plus de signaux premium
        self.min_confluence_score = 30  # Assoupli (40 -> 30)
        self.strong_confluence_score = 65  # Maintenu pour la qualité
        self.min_trend_alignment = 0.25  # Maintenu
        self.strong_trend_alignment = 0.45  # Maintenu
        self.volume_confirmation_min = 0.5  # Assoupli (1.0 -> 0.5) - plus accessible
        self.volume_strong = 1.8  # Maintenu

        # FILTRES ASSOUPLIS
        self.min_ma_count_required = 2  # Assoupli (4 -> 2)
        self.min_oscillator_count = 2  # Assoupli (3 -> 2)
        self.require_adx_confirmation = False  # ADX optionnel

        # Seuils oscillateurs ASSOUPLIS
        self.rsi_oversold = 35  # Assoupli (25 -> 35)
        self.rsi_overbought = 65  # Assoupli (75 -> 65)
        self.stoch_oversold = 25  # Assoupli (15 -> 25)
        self.stoch_overbought = 75  # Assoupli (85 -> 75)
        self.cci_oversold = -100  # Assoupli (-150 -> -100)
        self.cci_overbought = 100  # Assoupli (150 -> 100)
        self.williams_oversold = -80  # Assoupli (-90 -> -80)
        self.williams_overbought = -20  # Assoupli (-10 -> -20)

        # ADX ASSOUPLI
        self.min_adx_trend = 15  # Assoupli (20 -> 15)
        self.strong_adx_trend = 25  # Assoupli (30 -> 25)

    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs multi-TF."""
        return {
            # Confluence et force du signal
            "confluence_score": self.indicators.get("confluence_score"),
            "signal_strength": self.indicators.get("signal_strength"),
            "pattern_confidence": self.indicators.get("pattern_confidence"),
            "pattern_detected": self.indicators.get("pattern_detected"),
            # Alignement des timeframes
            "trend_alignment": self.indicators.get("trend_alignment"),
            "trend_strength": self.indicators.get("trend_strength"),
            "directional_bias": self.indicators.get("directional_bias"),
            "trend_angle": self.indicators.get("trend_angle"),
            # Régimes de marché
            "market_regime": self.indicators.get("market_regime"),
            "regime_strength": self.indicators.get("regime_strength"),
            "regime_confidence": self.indicators.get("regime_confidence"),
            "regime_duration": self.indicators.get("regime_duration"),
            "volatility_regime": self.indicators.get("volatility_regime"),
            # Support/Résistance (multi-TF)
            "nearest_support": self.indicators.get("nearest_support"),
            "nearest_resistance": self.indicators.get("nearest_resistance"),
            "support_strength": self.indicators.get("support_strength"),
            "resistance_strength": self.indicators.get("resistance_strength"),
            "break_probability": self.indicators.get("break_probability"),
            # Moyennes mobiles (alignement)
            "ema_7": self.indicators.get("ema_7"),
            "ema_12": self.indicators.get("ema_12"),
            "ema_26": self.indicators.get("ema_26"),
            "ema_50": self.indicators.get("ema_50"),
            "ema_99": self.indicators.get("ema_99"),
            "sma_20": self.indicators.get("sma_20"),
            "sma_50": self.indicators.get("sma_50"),
            "hull_20": self.indicators.get("hull_20"),
            # MACD multi-TF
            "macd_line": self.indicators.get("macd_line"),
            "macd_signal": self.indicators.get("macd_signal"),
            "macd_histogram": self.indicators.get("macd_histogram"),
            "macd_trend": self.indicators.get("macd_trend"),
            # Oscillateurs convergents
            "rsi_14": self.indicators.get("rsi_14"),
            "rsi_21": self.indicators.get("rsi_21"),
            "stoch_k": self.indicators.get("stoch_k"),
            "stoch_d": self.indicators.get("stoch_d"),
            "cci_20": self.indicators.get("cci_20"),
            "williams_r": self.indicators.get("williams_r"),
            # Volume multi-TF
            "volume_ratio": self.indicators.get("volume_ratio"),
            "relative_volume": self.indicators.get("relative_volume"),
            "volume_quality_score": self.indicators.get("volume_quality_score"),
            "volume_context": self.indicators.get("volume_context"),
            # ADX pour confirmation tendance
            "adx_14": self.indicators.get("adx_14"),
            "plus_di": self.indicators.get("plus_di"),
            "minus_di": self.indicators.get("minus_di"),
            # Momentum général
            "momentum_score": self.indicators.get("momentum_score"),
            "momentum_10": self.indicators.get("momentum_10"),
        }

    def _get_current_price(self) -> Optional[float]:
        """Récupère le prix actuel depuis les données OHLCV."""
        try:
            if self.data and "close" in self.data and self.data["close"]:
                return float(self.data["close"][-1])
        except (IndexError, ValueError, TypeError):
            pass
        return None

    def _analyze_ma_alignment(
        self, values: Dict[str, Optional[float]], current_price: float
    ) -> Dict[str, Any]:
        """Analyse l'alignement des moyennes mobiles pour détecter la tendance."""
        mas = {}
        ma_keys = [
            "ema_7",
            "ema_12",
            "ema_26",
            "ema_50",
            "ema_99",
            "sma_20",
            "sma_50",
            "hull_20",
        ]

        # Récupération des MAs disponibles
        for key in ma_keys:
            value = values.get(key)
            if value is not None:
                try:
                    mas[key] = float(value)
                except (ValueError, TypeError):
                    continue

        if len(mas) < self.min_ma_count_required:
            return {
                "alignment_score": 0.0,
                "direction": None,
                "reason": f"Pas assez de MAs ({len(mas)}/{self.min_ma_count_required})",
            }

        # Analyse spécifique pour crypto : importance des EMA courtes
        ema_7 = mas.get("ema_7")
        ema_12 = mas.get("ema_12")
        ema_26 = mas.get("ema_26")
        ema_50 = mas.get("ema_50")

        # Score d'alignement plus sophistiqué
        alignment_score = 0.0
        direction = None

        # Position du prix par rapport aux MAs
        price_above_count = sum(
            1 for _, ma_val in mas.items() if current_price > ma_val
        )
        price_ratio = price_above_count / len(mas)

        # Analyse hiérarchique des EMAs (plus important en crypto)
        if ema_7 and ema_12 and ema_26:
            # Configuration haussière parfaite : Prix > EMA7 > EMA12 > EMA26
            if current_price > ema_7 > ema_12 > ema_26:
                alignment_score += 0.4
                direction = "bullish"

                # Bonus si EMA50 aussi alignée
                if ema_50 and ema_26 > ema_50:
                    alignment_score += 0.2

            # Configuration baissière parfaite : Prix < EMA7 < EMA12 < EMA26
            elif current_price < ema_7 < ema_12 < ema_26:
                alignment_score += 0.4
                direction = "bearish"

                # Bonus si EMA50 aussi alignée
                if ema_50 and ema_26 < ema_50:
                    alignment_score += 0.2

            # Configurations partielles
            elif current_price > ema_7 and ema_7 > ema_26:
                alignment_score += 0.2
                direction = "bullish_weak"
            elif current_price < ema_7 and ema_7 < ema_26:
                alignment_score += 0.2
                direction = "bearish_weak"

        # Ajustement selon position globale du prix
        if price_ratio >= 0.75:
            if direction in ["bullish", "bullish_weak"] or direction is None:
                direction = "bullish"
                alignment_score = max(alignment_score + 0.2, 0.6)
        elif price_ratio <= 0.25:
            if direction in ["bearish", "bearish_weak"] or direction is None:
                direction = "bearish"
                alignment_score = max(alignment_score + 0.2, 0.6)
        else:
            if direction is None:
                direction = "neutral"
                alignment_score = 0.3

        # Hull MA pour confirmation (très réactive)
        hull_20 = mas.get("hull_20")
        if hull_20:
            if direction == "bullish" and current_price > hull_20:
                alignment_score = min(alignment_score + 0.1, 0.95)
            elif direction == "bearish" and current_price < hull_20:
                alignment_score = min(alignment_score + 0.1, 0.95)

        return {
            "alignment_score": alignment_score,
            "direction": direction,
            "price_above_ratio": price_ratio,
            "ma_count": len(mas),
            "ema_aligned": ema_7 is not None
            and ema_12 is not None
            and ema_26 is not None,
        }

    def _analyze_oscillator_confluence(
        self, values: Dict[str, Optional[float]]
    ) -> Dict[str, Any]:
        """Analyse la confluence des oscillateurs avec seuils crypto."""
        oscillators = {}

        # RSI avec seuils crypto
        rsi_14 = values.get("rsi_14")
        if rsi_14 is not None:
            try:
                rsi = float(rsi_14)
                if rsi <= self.rsi_oversold:
                    oscillators["rsi_14"] = "oversold"
                elif rsi >= self.rsi_overbought:
                    oscillators["rsi_14"] = "overbought"
                else:
                    oscillators["rsi_14"] = "neutral"
            except (ValueError, TypeError):
                pass

        # RSI 21 pour confirmation
        rsi_21 = values.get("rsi_21")
        if rsi_21 is not None:
            try:
                rsi = float(rsi_21)
                if rsi <= self.rsi_oversold + 2:  # Légèrement plus tolérant
                    oscillators["rsi_21"] = "oversold"
                elif rsi >= self.rsi_overbought - 2:
                    oscillators["rsi_21"] = "overbought"
                else:
                    oscillators["rsi_21"] = "neutral"
            except (ValueError, TypeError):
                pass

        # Stochastic avec seuils crypto
        stoch_k = values.get("stoch_k")
        stoch_d = values.get("stoch_d")
        if stoch_k is not None and stoch_d is not None:
            try:
                k = float(stoch_k)
                d = float(stoch_d)
                if k <= self.stoch_oversold and d <= self.stoch_oversold:
                    oscillators["stoch"] = "oversold"
                elif k >= self.stoch_overbought and d >= self.stoch_overbought:
                    oscillators["stoch"] = "overbought"
                else:
                    oscillators["stoch"] = "neutral"
            except (ValueError, TypeError):
                pass

        # CCI avec seuils crypto
        cci_20 = values.get("cci_20")
        if cci_20 is not None:
            try:
                cci = float(cci_20)
                if cci <= self.cci_oversold:
                    oscillators["cci"] = "oversold"
                elif cci >= self.cci_overbought:
                    oscillators["cci"] = "overbought"
                else:
                    oscillators["cci"] = "neutral"
            except (ValueError, TypeError):
                pass

        # Williams %R avec seuils crypto
        williams_r = values.get("williams_r")
        if williams_r is not None:
            try:
                wr = float(williams_r)
                if wr <= self.williams_oversold:
                    oscillators["williams"] = "oversold"
                elif wr >= self.williams_overbought:
                    oscillators["williams"] = "overbought"
                else:
                    oscillators["williams"] = "neutral"
            except (ValueError, TypeError):
                pass

        if len(oscillators) < self.min_oscillator_count:
            return {
                "confluence": "insufficient",
                "strength": 0.0,
                "count": len(oscillators),
            }

        # Calcul de la confluence avec pondération
        oversold_count = sum(1 for v in oscillators.values() if v == "oversold")
        overbought_count = sum(1 for v in oscillators.values() if v == "overbought")
        total_count = len(oscillators)

        # Seuils ASSOUPLIS pour plus de signaux (50%)
        if oversold_count >= total_count * 0.5:  # 50% des oscillateurs (assoupli)
            return {
                "confluence": "oversold",
                "strength": oversold_count / total_count,
                "count": total_count,
            }
        elif overbought_count >= total_count * 0.5:
            return {
                "confluence": "overbought",
                "strength": overbought_count / total_count,
                "count": total_count,
            }
        else:
            # Pas de consensus majoritaire
            return {"confluence": "insufficient", "strength": 0.0, "count": total_count}

    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur la confluence multi-timeframes.
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

        if current_price is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Prix non disponible",
                "metadata": {"strategy": self.name},
            }

        # Vérification des scores de confluence principaux
        try:
            confluence_score = (
                float(values["confluence_score"])
                if values["confluence_score"] is not None
                else None
            )
            signal_strength = values[
                "signal_strength"
            ]  # STRING: WEAK/MODERATE/STRONG/VERY_STRONG
            trend_alignment = (
                float(values["trend_alignment"])
                if values["trend_alignment"] is not None
                else None
            )
        except (ValueError, TypeError) as e:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Erreur conversion scores: {e}",
                "metadata": {"strategy": self.name},
            }

        # Confluence score ASSOUPLI avec fallback intelligent
        if confluence_score is None:
            confluence_score = 40.0  # Default optimiste au lieu de 0.0

        # Seuil confluence ABAISSÉ pour plus de signaux
        min_confluence_dynamic = (
            20 if confluence_score == 40.0 else self.min_confluence_score
        )  # 20 pour fallback, 30 pour valeurs réelles

        if confluence_score < min_confluence_dynamic:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Confluence insuffisante ({confluence_score:.1f} < {min_confluence_dynamic})",
                "metadata": {
                    "strategy": self.name,
                    "confluence_score": confluence_score,
                },
            }

        # Signal_strength ULTRA-ASSOUPLI avec fallback
        valid_strengths = ["WEAK", "MODERATE", "STRONG", "VERY_STRONG"]
        signal_strength_str = str(signal_strength) if signal_strength is not None else ""
        if signal_strength_str not in valid_strengths:
            signal_strength_str = "MODERATE"  # Default optimiste pour valeurs manquantes/invalides

        # Vérification ADX pour tendance suffisante (déplacée ici)
        adx_14 = values.get("adx_14")
        adx_value = None
        if adx_14 is not None:
            try:
                adx_value = float(adx_14)
            except (ValueError, TypeError):
                pass

        # Signal strength ASSOUPLI - Accepter WEAK avec pénalité
        weak_signal_penalty = 0.0
        if signal_strength_str == "WEAK":
            weak_signal_penalty = -0.20  # Pénalité au lieu de rejet

        # Vérification trend_alignment avec pénalité au lieu de rejet
        trend_penalty = 0.0
        if trend_alignment and trend_alignment < self.min_trend_alignment:
            trend_penalty = -0.15  # Pénalité au lieu de rejet direct

        # ADX OPTIONNEL (require_adx_confirmation = False)
        if (
            self.require_adx_confirmation
            and adx_value is not None
            and adx_value < self.min_adx_trend
        ):
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"ADX insuffisant ({adx_value:.1f} < {self.min_adx_trend}) - marché ranging",
                "metadata": {"strategy": self.name, "adx": adx_value},
            }

        # Analyse de l'alignement des moyennes mobiles
        ma_analysis = self._analyze_ma_alignment(values, current_price)

        # Analyse des oscillateurs avec validation stricte
        osc_analysis = self._analyze_oscillator_confluence(values)

        signal_side = None
        reason = ""

        # Oscillateurs ASSOUPLIS - Pénalité au lieu de rejet
        oscillator_penalty = 0.0
        if osc_analysis["confluence"] == "insufficient":
            oscillator_penalty = -0.15  # Pénalité au lieu de rejet total
            reason += f" (oscill. insuffisants {osc_analysis['count']}/{self.min_oscillator_count})"
        base_confidence = 0.65  # Harmonisé avec autres stratégies
        confidence_boost = 0.0

        # LOGIQUE ULTRA-ASSOUPLIE - Accepté même les setups neutres
        if (
            ma_analysis["direction"] == "bullish"
            and ma_analysis["alignment_score"] >= 0.3
        ):  # Encore plus assoupli 0.5 -> 0.3
            # Setup haussier
            if osc_analysis["confluence"] in [
                "oversold",
                "insufficient",
            ]:  # Accepté même insufficient
                signal_side = "BUY"
                if ma_analysis["alignment_score"] >= 0.7:
                    reason = f"Setup PARFAIT BUY: MA parfaites ({ma_analysis['alignment_score']:.2f})"
                    confidence_boost += 0.30
                elif ma_analysis["alignment_score"] >= 0.5:
                    reason = f"Setup SOLIDE BUY: MA alignées ({ma_analysis['alignment_score']:.2f})"
                    confidence_boost += 0.20
                else:
                    reason = f"Setup FAIBLE BUY: MA partielles ({ma_analysis['alignment_score']:.2f})"
                    confidence_boost += 0.10

        elif (
            ma_analysis["direction"] == "bearish"
            and ma_analysis["alignment_score"] >= 0.3
        ):
            # Setup baissier
            if osc_analysis["confluence"] in ["overbought", "insufficient"]:
                signal_side = "SELL"
                if ma_analysis["alignment_score"] >= 0.7:
                    reason = f"Setup PARFAIT SELL: MA parfaites ({ma_analysis['alignment_score']:.2f})"
                    confidence_boost += 0.30
                elif ma_analysis["alignment_score"] >= 0.5:
                    reason = f"Setup SOLIDE SELL: MA alignées ({ma_analysis['alignment_score']:.2f})"
                    confidence_boost += 0.20
                else:
                    reason = f"Setup FAIBLE SELL: MA partielles ({ma_analysis['alignment_score']:.2f})"
                    confidence_boost += 0.10

        # NOUVEAU : Accepter même les directions neutres si oscillateurs forts
        elif (
            ma_analysis["direction"] == "neutral"
            and osc_analysis["confluence"] != "insufficient"
        ):
            if (
                osc_analysis["confluence"] == "oversold"
                and osc_analysis["strength"] >= 0.6
            ):
                signal_side = "BUY"
                reason = f"Setup NEUTRE BUY: Oscillateurs survente ({osc_analysis['strength']:.2f})"
                confidence_boost += 0.08
            elif (
                osc_analysis["confluence"] == "overbought"
                and osc_analysis["strength"] >= 0.6
            ):
                signal_side = "SELL"
                reason = f"Setup NEUTRE SELL: Oscillateurs surachat ({osc_analysis['strength']:.2f})"
                confidence_boost += 0.08

        # Pas d'alignement clair ou contradictoire
        if signal_side is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Pas de confluence claire - MA: {ma_analysis['direction']}, Osc: {osc_analysis['confluence']}",
                "metadata": {
                    "strategy": self.name,
                    "ma_analysis": ma_analysis,
                    "osc_analysis": osc_analysis,
                },
            }

        # === BOOSTS SIMPLIFIÉS - 4 catégories principales ===

        # 1. Confluence score (déjà validé comme élevé)
        if confluence_score >= 85:
            confidence_boost += 0.15
            reason += f" [confluence PARFAITE: {confluence_score:.0f}]"
        elif confluence_score >= 75:
            confidence_boost += 0.10
            reason += f" [confluence excellente: {confluence_score:.0f}]"

        # 2. Signal strength (déjà validé comme STRONG+)
        if signal_strength_str == "VERY_STRONG":
            confidence_boost += 0.12
            reason += " + VERY_STRONG"
        elif signal_strength_str == "STRONG":
            confidence_boost += 0.08
            reason += " + STRONG"

        # 3. Trend alignment exceptionnel (déjà validé > 0.25)
        if trend_alignment and trend_alignment >= 0.7:
            confidence_boost += 0.15
            reason += f" + trend PARFAIT ({trend_alignment:.2f})"
        elif trend_alignment and trend_alignment >= self.strong_trend_alignment:
            confidence_boost += 0.08
            reason += f" + trend fort ({trend_alignment:.2f})"

        # 4. ADX exceptionnel (déjà validé > 20)
        if adx_value is not None:
            if adx_value >= 40:
                confidence_boost += 0.12
                reason += f" + ADX EXTRÊEME ({adx_value:.1f})"
            elif adx_value >= self.strong_adx_trend:
                confidence_boost += 0.08
                reason += f" + ADX fort ({adx_value:.1f})"

        # VALIDATION DIRECTIONAL BIAS - PÉNALITÉ au lieu de rejet
        directional_bias = values.get("directional_bias")
        if directional_bias:
            if (signal_side == "BUY" and directional_bias == "BEARISH") or (
                signal_side == "SELL" and directional_bias == "BULLISH"
            ):
                confidence_boost -= 0.15  # Pénalité au lieu de rejet
                reason += f" (bias contradictoire {directional_bias})"
            elif (signal_side == "BUY" and directional_bias == "BULLISH") or (
                signal_side == "SELL" and directional_bias == "BEARISH"
            ):
                confidence_boost += 0.20  # Boost renforcé pour alignment parfait
                reason += f" + bias PARFAITEMENT aligné"

        # VALIDATION RÉGIME MARCHÉ - REJET si contradictoire
        market_regime = values.get("market_regime")
        regime_strength = values.get("regime_strength")

        if market_regime and regime_strength:
            regime_str = str(regime_strength).upper()

            # Gestion par force du régime (STRONG/MODERATE/WEAK)
            if regime_str == "STRONG":
                # PÉNALITÉS au lieu de rejets stricts sur régimes contradictoires FORTS
                if (
                    signal_side == "BUY"
                    and market_regime in ["TRENDING_BEAR", "BREAKOUT_BEAR"]
                ) or (
                    signal_side == "SELL"
                    and market_regime in ["TRENDING_BULL", "BREAKOUT_BULL"]
                ):
                    confidence_boost -= 0.15  # Pénalité au lieu de rejet
                    reason += f" (régime contradictoire {market_regime} FORT)"
                # Confirmations régimes parfaits
                elif (
                    signal_side == "BUY"
                    and market_regime in ["TRENDING_BULL", "BREAKOUT_BULL"]
                ) or (
                    signal_side == "SELL"
                    and market_regime in ["TRENDING_BEAR", "BREAKOUT_BEAR"]
                ):
                    confidence_boost += 0.15
                    reason += f" + régime PARFAIT ({market_regime} FORT)"

            elif regime_str == "MODERATE":
                # Régimes modérés : pénalités plus douces
                if (
                    signal_side == "BUY"
                    and market_regime in ["TRENDING_BEAR", "BREAKOUT_BEAR"]
                ) or (
                    signal_side == "SELL"
                    and market_regime in ["TRENDING_BULL", "BREAKOUT_BULL"]
                ):
                    confidence_boost -= 0.08  # Pénalité réduite
                    reason += f" (régime contradictoire {market_regime} modéré)"
                elif (
                    signal_side == "BUY"
                    and market_regime in ["TRENDING_BULL", "BREAKOUT_BULL"]
                ) or (
                    signal_side == "SELL"
                    and market_regime in ["TRENDING_BEAR", "BREAKOUT_BEAR"]
                ):
                    confidence_boost += 0.08
                    reason += f" + régime aligné ({market_regime} modéré)"

            elif regime_str == "WEAK":
                # Régimes faibles : impact minimal
                if (
                    signal_side == "BUY"
                    and market_regime in ["TRENDING_BULL", "BREAKOUT_BULL"]
                ) or (
                    signal_side == "SELL"
                    and market_regime in ["TRENDING_BEAR", "BREAKOUT_BEAR"]
                ):
                    confidence_boost += 0.03  # Bonus minimal
                    reason += f" + régime aligné ({market_regime} faible)"
                # Pas de pénalité pour les contradictions faibles

        # VALIDATION VOLUME STRICTE avec rejet
        volume_ratio = values.get("volume_ratio")
        vol_ratio = None

        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                # REJET seulement si volume vraiment très faible
                if vol_ratio < self.volume_confirmation_min:
                    confidence_boost -= 0.20  # Pénalité forte mais pas rejet total
                    reason += f" (volume faible {vol_ratio:.1f}x)"
                elif vol_ratio >= self.volume_strong * 1.5:  # Volume exceptionnel
                    confidence_boost += 0.18
                    reason += f" + volume EXCEPTIONNEL ({vol_ratio:.1f}x)"
                elif vol_ratio >= self.volume_strong:
                    confidence_boost += 0.12
                    reason += f" + volume fort ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.3:
                    confidence_boost += 0.08
                    reason += f" + volume élevé ({vol_ratio:.1f}x)"
            except (ValueError, TypeError):
                pass  # Volume invalide mais pas bloquant

        # VALIDATION FINALE - Tous les autres indicateurs déjà validés
        # Plus de micros-ajustements - logique simplifiée focus winrate

        # Calcul final avec toutes les pénalités
        total_adjustment = confidence_boost + weak_signal_penalty + trend_penalty

        # Calcul final optimisé sans double calcul
        confidence = min(base_confidence * (1 + total_adjustment), 0.90)

        # Appliquer pénalité oscillateurs
        confidence = max(0.0, confidence + oscillator_penalty)

        # Seuil final ULTRA-ABAISSÉ pour maximum de signaux
        if confidence < 0.30:  # Abaissé de 0.45 à 0.30
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Setup MultiTF rejeté - confiance critique ({confidence:.2f} < 0.30)",
                "metadata": {
                    "strategy": self.name,
                    "rejected_signal": signal_side,
                    "rejected_confidence": confidence,
                    "confluence_score": confluence_score,
                    "signal_strength": signal_strength_str,
                    "trend_alignment": trend_alignment,
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
                "confluence_score": confluence_score,
                "signal_strength": signal_strength_str,
                "trend_alignment": trend_alignment,
                "ma_analysis": ma_analysis,
                "osc_analysis": osc_analysis,
                "directional_bias": directional_bias,
                "market_regime": market_regime,
                "volume_ratio": volume_ratio,
                "adx_14": adx_value,
            },
        }

    def validate_data(self) -> bool:
        """Valide que tous les indicateurs de confluence requis sont présents."""
        if not super().validate_data():
            return False

        # ASSOUPLI : confluence_score et signal_strength optionnels avec fallbacks
        # Plus de rejet dur pour indicateurs manquants au boot

        # Vérifier qu'on a au moins quelques moyennes mobiles (ASSOUPLI)
        ma_indicators = ["ema_7", "ema_12", "ema_26", "ema_50", "sma_20", "sma_50"]
        ma_available = sum(
            1
            for ma in ma_indicators
            if ma in self.indicators and self.indicators[ma] is not None
        )

        if ma_available < 1:  # Réduit de 3 à 1 - Une seule MA suffit au boot
            logger.warning(f"{self.name}: Aucune moyenne mobile ({ma_available}/6)")
            return False

        return True
