"""
Opportunity Scoring System - Professional Multi-Level Scoring
Utilise TOUS les 108 indicateurs de analyzer_data pour calculer un score précis 0-100

Architecture:
- 7 catégories de scoring (trend, momentum, volume, volatility, support/resistance, pattern, confluence)
- Pondération dynamique selon le régime de marché
- Gestion des cas edge et données manquantes
- Scoring adaptatif selon le timeframe et la volatilité

Version: 2.0 - Professional Grade
"""

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ScoreCategory(Enum):
    """Catégories de scoring."""

    TREND = "trend"
    MOMENTUM = "momentum"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    SUPPORT_RESISTANCE = "support_resistance"
    PATTERN = "pattern"
    CONFLUENCE = "confluence"


@dataclass
class CategoryScore:
    """Score d'une catégorie avec détails."""

    category: ScoreCategory
    score: float  # 0-100
    weight: float  # 0-1
    weighted_score: float  # score * weight
    details: dict[str, float]
    confidence: float  # 0-100
    issues: list[str]


@dataclass
class OpportunityScore:
    """Score global d'opportunité."""

    total_score: float  # 0-100
    grade: str  # S, A, B, C, D, F
    category_scores: dict[ScoreCategory, CategoryScore]
    confidence: float  # 0-100
    risk_level: str  # LOW, MEDIUM, HIGH, EXTREME
    recommendation: str  # BUY_NOW, BUY_DCA, WAIT, AVOID
    reasons: list[str]
    warnings: list[str]


class OpportunityScoring:
    """
    Système de scoring professionnel multi-niveaux.

    Utilise TOUS les indicateurs disponibles pour calculer un score précis.
    Pondération dynamique selon le contexte de marché.
    """

    # Pondérations par défaut (ajustées selon régime)
    DEFAULT_WEIGHTS = {
        ScoreCategory.TREND: 0.25,
        ScoreCategory.MOMENTUM: 0.20,
        ScoreCategory.VOLUME: 0.20,
        ScoreCategory.VOLATILITY: 0.10,
        ScoreCategory.SUPPORT_RESISTANCE: 0.15,
        ScoreCategory.PATTERN: 0.05,
        ScoreCategory.CONFLUENCE: 0.05,
    }

    # Pondérations selon régime
    REGIME_WEIGHTS = {
        "TRENDING_BULL": {
            ScoreCategory.TREND: 0.25,  # ↓ de 0.30
            ScoreCategory.MOMENTUM: 0.30,  # ↑ de 0.25 - PRIORITÉ pendant pump
            ScoreCategory.VOLUME: 0.20,  # ↑ de 0.15
            ScoreCategory.VOLATILITY: 0.05,
            # ↓ de 0.15 - résistance moins critique
            ScoreCategory.SUPPORT_RESISTANCE: 0.10,
            ScoreCategory.PATTERN: 0.05,
            ScoreCategory.CONFLUENCE: 0.05,
        },
        "BREAKOUT_BULL": {
            ScoreCategory.TREND: 0.15,  # ↓ de 0.20
            ScoreCategory.MOMENTUM: 0.25,  # ↑ de 0.20 - momentum clé pour breakout
            ScoreCategory.VOLUME: 0.35,  # ↑ de 0.30 - volume critique
            ScoreCategory.VOLATILITY: 0.10,
            ScoreCategory.SUPPORT_RESISTANCE: 0.05,  # ↓ de 0.10 - on casse la résistance!
            ScoreCategory.PATTERN: 0.05,
            ScoreCategory.CONFLUENCE: 0.05,
        },
        "RANGING": {
            ScoreCategory.TREND: 0.10,
            ScoreCategory.MOMENTUM: 0.15,
            ScoreCategory.VOLUME: 0.20,
            ScoreCategory.VOLATILITY: 0.15,
            ScoreCategory.SUPPORT_RESISTANCE: 0.30,
            ScoreCategory.PATTERN: 0.05,
            ScoreCategory.CONFLUENCE: 0.05,
        },
        "VOLATILE": {
            ScoreCategory.TREND: 0.15,
            ScoreCategory.MOMENTUM: 0.20,
            ScoreCategory.VOLUME: 0.25,
            ScoreCategory.VOLATILITY: 0.20,
            ScoreCategory.SUPPORT_RESISTANCE: 0.10,
            ScoreCategory.PATTERN: 0.05,
            ScoreCategory.CONFLUENCE: 0.05,
        },
    }

    def __init__(self):
        """Initialise le système de scoring."""

    @staticmethod
    def safe_float(value, default=0.0):
        """Convertir en float avec fallback."""
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    def calculate_opportunity_score(
        self, analyzer_data: dict, current_price: float = 0.0
    ) -> OpportunityScore:
        """
        Calcule le score global d'opportunité.

        Args:
            analyzer_data: Données de analyzer_data (tous les 108 indicateurs)
            current_price: Prix actuel (requis pour calcul S/R correct)

        Returns:
            OpportunityScore complet avec détails par catégorie
        """
        if not analyzer_data:
            return self._create_zero_score("Pas de données")

        # Déterminer les pondérations selon le régime
        regime = analyzer_data.get("market_regime", "UNKNOWN")
        weights = self._get_weights_for_regime(regime)

        # Calculer score par catégorie
        category_scores = {}

        # 1. TREND SCORE
        category_scores[ScoreCategory.TREND] = self._score_trend(
            analyzer_data, weights[ScoreCategory.TREND]
        )

        # 2. MOMENTUM SCORE
        category_scores[ScoreCategory.MOMENTUM] = self._score_momentum(
            analyzer_data, weights[ScoreCategory.MOMENTUM]
        )

        # 3. VOLUME SCORE
        category_scores[ScoreCategory.VOLUME] = self._score_volume(
            analyzer_data, weights[ScoreCategory.VOLUME]
        )

        # 4. VOLATILITY SCORE
        category_scores[ScoreCategory.VOLATILITY] = self._score_volatility(
            analyzer_data, weights[ScoreCategory.VOLATILITY]
        )

        # 5. SUPPORT/RESISTANCE SCORE
        category_scores[ScoreCategory.SUPPORT_RESISTANCE] = (
            self._score_support_resistance(
                analyzer_data, current_price, weights[ScoreCategory.SUPPORT_RESISTANCE]
            )
        )

        # 6. PATTERN SCORE
        category_scores[ScoreCategory.PATTERN] = self._score_pattern(
            analyzer_data, weights[ScoreCategory.PATTERN]
        )

        # 7. CONFLUENCE SCORE
        category_scores[ScoreCategory.CONFLUENCE] = self._score_confluence(
            analyzer_data, weights[ScoreCategory.CONFLUENCE]
        )

        # Calculer score total
        total_score = sum(cs.weighted_score for cs in category_scores.values())

        # Calculer confiance globale
        confidence = sum(
            cs.confidence * cs.weight for cs in category_scores.values()
        ) / sum(weights.values())

        # Déterminer grade
        grade = self._calculate_grade(total_score)

        # Déterminer niveau de risque
        risk_level = self._calculate_risk_level(analyzer_data, total_score)

        # Déterminer recommandation
        recommendation, reasons, warnings = self._calculate_recommendation(
            total_score, confidence, category_scores, analyzer_data
        )

        return OpportunityScore(
            total_score=total_score,
            grade=grade,
            category_scores=category_scores,
            confidence=confidence,
            risk_level=risk_level,
            recommendation=recommendation,
            reasons=reasons,
            warnings=warnings,
        )

    def _get_weights_for_regime(
            self, regime: str) -> dict[ScoreCategory, float]:
        """Retourne les pondérations adaptées au régime."""
        return self.REGIME_WEIGHTS.get(regime, self.DEFAULT_WEIGHTS)

    def _score_trend(self, ad: dict, weight: float) -> CategoryScore:
        """
        Score TREND (0-100) basé sur:
        - ADX + Plus DI/Minus DI (force et direction)
        - Trend alignment (EMAs alignées)
        - Trend strength (weak/moderate/strong/extreme)
        - Trend angle (pente)
        - Directional bias (BULLISH/BEARISH)
        - Régime de marché + confiance
        """
        details: dict[str, float] = {}
        issues: list[str] = []
        score = 0.0

        # 1. ADX + DI (40 points max)
        adx = self.safe_float(ad.get("adx_14"))
        plus_di = self.safe_float(ad.get("plus_di"))
        minus_di = self.safe_float(ad.get("minus_di"))

        if adx > 0:
            # ADX force (0-25 points)
            if adx > 40:
                adx_score = 25.0
            elif adx > 30:
                adx_score = 20.0
            elif adx > 25:
                adx_score = 15.0
            elif adx > 20:
                adx_score = 10.0
            else:
                adx_score = 5.0

            # Direction (0-15 points)
            if minus_di > 0:
                di_ratio = plus_di / minus_di
                if di_ratio > 3.0:
                    di_score = 15.0
                elif di_ratio > 2.0:
                    di_score = 12.0
                elif di_ratio > 1.5:
                    di_score = 8.0
                elif di_ratio > 1.0:
                    di_score = 5.0
                else:
                    di_score = 0.0
                    issues.append(f"-DI > +DI (ratio {di_ratio:.2f})")
            else:
                di_score = 10.0  # Pas de -DI, assume bullish

            score += adx_score + di_score
            details["adx"] = float(adx_score)
            details["directional"] = float(di_score)
        else:
            issues.append("ADX indisponible")

        # 2. Trend Alignment (20 points max)
        trend_alignment = self.safe_float(ad.get("trend_alignment"))
        if trend_alignment != 0:
            # trend_alignment: -100 (bear) à +100 (bull)
            if trend_alignment > 80:
                align_score = 20.0
            elif trend_alignment > 60:
                align_score = 15.0
            elif trend_alignment > 40:
                align_score = 10.0
            elif trend_alignment > 20:
                align_score = 5.0
            elif trend_alignment > 0:
                align_score = 2.0
            else:
                align_score = 0.0
                issues.append(f"EMAs baissières ({trend_alignment:.0f})")

            score += align_score
            details["ema_alignment"] = float(align_score)

        # 3. Trend Strength (15 points max)
        trend_strength = ad.get("trend_strength", "").upper()
        strength_scores = {
            "EXTREME": 15.0,
            "VERY_STRONG": 12.0,
            "STRONG": 10.0,
            "MODERATE": 6.0,
            "WEAK": 3.0,
            "ABSENT": 0.0,
        }
        strength_score = strength_scores.get(trend_strength, 0.0)
        score += strength_score
        details["trend_strength"] = float(strength_score)

        # 4. Directional Bias (15 points max)
        bias = ad.get("directional_bias", "").upper()
        if bias == "BULLISH":
            bias_score = 15.0
        elif bias == "NEUTRAL":
            bias_score = 5.0
        else:
            bias_score = 0.0
            issues.append(f"Bias {bias}")

        score += bias_score
        details["bias"] = float(bias_score)

        # 5. Régime + Confiance (10 points max)
        regime = ad.get("market_regime", "").upper()
        regime_conf = self.safe_float(ad.get("regime_confidence"))

        if regime in ["TRENDING_BULL", "BREAKOUT_BULL"]:
            regime_score = 10.0 * \
                (regime_conf / 100.0) if regime_conf > 0 else 5.0
        elif regime in ["TRANSITION", "RANGING"]:
            regime_score = 3.0
        else:
            regime_score = 0.0
            issues.append(f"Régime {regime}")

        score += regime_score
        details["regime"] = float(regime_score)

        # Confiance basée sur disponibilité des données
        confidence = 100.0 if len(details) >= 4 else 50.0

        return CategoryScore(
            category=ScoreCategory.TREND,
            score=min(score, 100),
            weight=weight,
            weighted_score=min(score, 100) * weight,
            details=details,
            confidence=confidence,
            issues=issues,
        )

    def _score_momentum(self, ad: dict, weight: float) -> CategoryScore:
        """
        Score MOMENTUM (0-100) basé sur:
        - RSI (14 et 21)
        - Williams %R
        - MACD (line, signal, histogram, trend, crosses)
        - CCI
        - MFI
        - Stochastic (K, D, RSI, divergence, signal)
        - Momentum score composite
        - ROC

        PUMP TOLERANCE: RSI/MFI élevés sont OK si pump validé
        """
        details: dict[str, float] = {}
        issues: list[str] = []
        score = 0.0

        # Détecter pump context (même logique que validator)
        vol_spike = self.safe_float(ad.get("volume_spike_multiplier"), 1.0)
        rel_volume = self.safe_float(ad.get("relative_volume"), 1.0)
        market_regime = ad.get("market_regime", "").upper()
        vol_context = ad.get("volume_context", "").upper()

        # AJUSTÉ: 2.0x (compromis 20 cryptos: P95 varie de 1.4x à 8.3x)
        is_pump = (
            (vol_spike > 2.0 or rel_volume > 2.0) and market_regime in [
                "TRENDING_BULL",
                "BREAKOUT_BULL"] and vol_context in [
                "CONSOLIDATION_BREAK",
                "BREAKOUT",
                "PUMP_START",
                "HIGH_VOLATILITY"])

        # 1. RSI (20 points max)
        rsi_14 = self.safe_float(ad.get("rsi_14"))
        rsi_21 = self.safe_float(ad.get("rsi_21"))

        if rsi_14 > 0:
            # Zone bullish: 50-70 optimal
            # PUMP: Accepter RSI élevé comme signal de force
            if 55 <= rsi_14 <= 70:
                rsi_score = 20.0
            elif 50 <= rsi_14 < 55 or 70 < rsi_14 <= 75:
                rsi_score = 15.0
            elif 75 < rsi_14 <= 85 and is_pump:
                # Pump context: RSI 75-85 = momentum fort = BONUS
                rsi_score = 18.0
                issues.append(f"RSI pump ({rsi_14:.0f})")
            elif rsi_14 > 85 and is_pump:
                # Pump extrême: RSI >85 = toujours bullish pendant pump
                rsi_score = 15.0
                issues.append(f"RSI pump extrême ({rsi_14:.0f})")
            elif 45 <= rsi_14 < 50:
                rsi_score = 10.0
            elif rsi_14 > 75 and not is_pump:
                # Sans pump context, RSI >75 = overbought = pénalité
                rsi_score = 5.0
                issues.append(f"RSI overbought ({rsi_14:.0f})")
            else:
                rsi_score = 0.0
                issues.append(f"RSI faible ({rsi_14:.0f})")

            # Bonus si RSI 14 et 21 cohérents
            if rsi_21 > 0 and abs(rsi_14 - rsi_21) < 10:
                rsi_score = min(rsi_score + 3, 20)

            score += rsi_score
            details["rsi"] = rsi_score

        # 2. MACD (20 points max)
        macd_trend = ad.get("macd_trend", "").upper()
        macd_hist = self.safe_float(ad.get("macd_histogram"))
        macd_signal_cross = ad.get("macd_signal_cross", False)

        macd_score = 0.0
        if macd_trend == "BULLISH":
            macd_score = 12.0
            # Bonus histogram positif et croissant
            if macd_hist > 0:
                macd_score += 5.0
            # Bonus croisement récent
            if macd_signal_cross:
                macd_score += 3.0
        elif macd_trend == "NEUTRAL":
            macd_score = 3.0
        else:
            issues.append(f"MACD {macd_trend}")

        score += min(macd_score, 20)
        details["macd"] = min(macd_score, 20)

        # 3. Stochastic (15 points max)
        stoch_k = self.safe_float(ad.get("stoch_k"))
        stoch_d = self.safe_float(ad.get("stoch_d"))
        stoch_signal = ad.get("stoch_signal", "").upper()
        stoch_div = ad.get("stoch_divergence", False)

        stoch_score = 0.0
        if stoch_k > 0:
            # Zone bullish: 40-80
            if 40 <= stoch_k <= 80 and 40 <= stoch_d <= 80:
                stoch_score = 10.0
            elif (stoch_k > 80 or stoch_d > 80) and is_pump:
                # Pump context: Stoch >80 acceptable
                stoch_score = 8.0
                issues.append(f"Stoch pump ({stoch_k:.0f}/{stoch_d:.0f})")
            elif stoch_k > 80 or stoch_d > 80:
                stoch_score = 3.0
                issues.append(
                    f"Stoch overbought ({stoch_k:.0f}/{stoch_d:.0f})")
            else:
                stoch_score = 5.0

            # Bonus signal
            if stoch_signal == "BULLISH":
                stoch_score += 3.0

            # Bonus divergence haussière
            if stoch_div:
                stoch_score += 2.0

            score += min(stoch_score, 15)
            details["stochastic"] = min(stoch_score, 15)

        # 4. Williams %R (10 points max)
        williams = self.safe_float(ad.get("williams_r"))
        if williams != 0:
            # Williams: -100 (oversold) à 0 (overbought)
            # Zone bullish: -50 à -20
            if -50 <= williams <= -20:
                will_score = 10.0
            elif -60 <= williams < -50 or -20 < williams <= -10:
                will_score = 7.0
            elif williams > -10:
                will_score = 2.0
                issues.append(f"Williams overbought ({williams:.0f})")
            else:
                will_score = 3.0

            score += will_score
            details["williams"] = will_score

        # 5. CCI (10 points max)
        cci = self.safe_float(ad.get("cci_20"))
        if cci != 0:
            # CCI > 0 = bullish, éviter >200
            if 0 < cci <= 100:
                cci_score = 10.0
            elif 100 < cci <= 150:
                cci_score = 7.0
            elif 150 < cci <= 200:
                cci_score = 4.0
            elif cci > 200:
                cci_score = 2.0
                issues.append(f"CCI extreme ({cci:.0f})")
            else:
                cci_score = 0.0
                issues.append(f"CCI négatif ({cci:.0f})")

            score += cci_score
            details["cci"] = cci_score

        # 6. MFI (10 points max)
        mfi = self.safe_float(ad.get("mfi_14"))
        if mfi > 0:
            # MFI: 0-100, optimal 50-70
            if 50 <= mfi <= 70:
                mfi_score = 10.0
            elif 40 <= mfi < 50 or 70 < mfi <= 80:
                mfi_score = 7.0
            elif mfi > 80 and is_pump:
                # Pump context: MFI >80 = argent entrant = BULLISH
                mfi_score = 9.0
                issues.append(f"MFI pump ({mfi:.0f})")
            elif mfi > 80:
                mfi_score = 3.0
                issues.append(f"MFI overbought ({mfi:.0f})")
            else:
                mfi_score = 2.0

            score += mfi_score
            details["mfi"] = mfi_score

        # 7. Momentum Score composite (15 points max)
        momentum_score = self.safe_float(ad.get("momentum_score"))
        if momentum_score > 0:
            # momentum_score: 0-100
            mom_contrib = (momentum_score / 100) * 15
            score += mom_contrib
            details["momentum_composite"] = mom_contrib

        # Confiance
        confidence = min(100, len(details) * 15)

        return CategoryScore(
            category=ScoreCategory.MOMENTUM,
            score=min(score, 100),
            weight=weight,
            weighted_score=min(score, 100) * weight,
            details=details,
            confidence=confidence,
            issues=issues,
        )

    def _score_volume(self, ad: dict, weight: float) -> CategoryScore:
        """
        Score VOLUME (0-100) basé sur:
        - Relative volume + spike multiplier
        - Volume context (BREAKOUT, PUMP_START, etc.)
        - Volume pattern (BUILDUP, SPIKE, SUSTAINED_HIGH)
        - Volume quality score
        - OBV + OBV oscillator + OBV MA
        - A/D Line
        - Trade intensity
        - Avg trade size
        - Quote volume ratio
        - Volume buildup periods
        """
        details: dict[str, float] = {}
        issues: list[str] = []
        score = 0.0

        # 1. Relative Volume (25 points max)
        rel_volume = self.safe_float(ad.get("relative_volume"), 1.0)
        vol_spike = self.safe_float(ad.get("volume_spike_multiplier"), 1.0)

        # Volume spike prioritaire
        if vol_spike > 3.0:
            vol_score = 25.0
        elif vol_spike > 2.5:
            vol_score = 22.0
        elif vol_spike > 2.0:
            vol_score = 18.0
        elif rel_volume > 2.0:
            vol_score = 15.0
        elif rel_volume > 1.5:
            vol_score = 12.0
        elif rel_volume > 1.2:
            vol_score = 8.0
        elif rel_volume > 1.0:
            vol_score = 5.0
        elif rel_volume > 0.8:
            vol_score = 2.0
        else:
            vol_score = 0.0
            issues.append(f"Volume faible ({rel_volume:.2f}x)")

        score += vol_score
        details["relative_volume"] = vol_score

        # 2. Volume Context (25 points max)
        vol_context = ad.get("volume_context", "").upper()
        context_scores = {
            "BREAKOUT": 25.0,
            "PUMP_START": 25.0,
            "CONSOLIDATION_BREAK": 20.0,
            "TREND_CONTINUATION": 18.0,
            "OVERSOLD_BOUNCE": 15.0,
            "HIGH_VOLATILITY": 12.0,
            "NEUTRAL": 8.0,
            "LOW_VOLATILITY": 5.0,
            "MODERATE_OVERSOLD": 5.0,
            "DEEP_OVERSOLD": 0.0,
            "REVERSAL_PATTERN": 0.0,
        }
        context_score = context_scores.get(vol_context, 5.0)
        score += context_score
        details["context"] = context_score

        if context_score == 0.0:
            issues.append(f"Context {vol_context}")

        # 3. Volume Pattern (15 points max)
        vol_pattern = ad.get("volume_pattern", "").upper()
        pattern_scores = {
            "SPIKE": 15.0,
            "SUSTAINED_HIGH": 12.0,
            "BUILDUP": 10.0,
            "NORMAL": 5.0,
            "DECLINING": 0.0,
        }
        pattern_score = pattern_scores.get(vol_pattern, 5.0)
        score += pattern_score
        details["pattern"] = pattern_score

        # 4. Volume Quality Score (15 points max)
        vol_quality = self.safe_float(ad.get("volume_quality_score"))
        if vol_quality > 0:
            quality_contrib = (vol_quality / 100) * 15
            score += quality_contrib
            details["quality"] = quality_contrib

        # 5. OBV Oscillator (10 points max)
        obv_osc = self.safe_float(ad.get("obv_oscillator"))
        if obv_osc > 0:
            # OBV positif = buying pressure
            if obv_osc > 300:
                obv_score = 10.0
            elif obv_osc > 200:
                obv_score = 8.0
            elif obv_osc > 100:
                obv_score = 6.0
            elif obv_osc > 50:
                obv_score = 4.0
            else:
                obv_score = 2.0
        else:
            obv_score = 0.0
            if obv_osc < -200:
                issues.append(f"OBV négatif ({obv_osc:.0f})")

        score += obv_score
        details["obv"] = obv_score

        # 6. Trade Intensity (5 points max)
        trade_intensity = self.safe_float(ad.get("trade_intensity"))
        if trade_intensity > 0:
            if trade_intensity > 1.5:
                intensity_score = 5.0
            elif trade_intensity > 1.2:
                intensity_score = 3.0
            else:
                intensity_score = 1.0

            score += intensity_score
            details["intensity"] = intensity_score

        # 7. Volume Buildup (5 points max)
        buildup_periods = ad.get("volume_buildup_periods", 0)
        if buildup_periods > 0:
            buildup_score = float(min(buildup_periods, 5))
            score += buildup_score
            details["buildup"] = buildup_score

        # Confiance
        confidence = min(100, len(details) * 15)

        return CategoryScore(
            category=ScoreCategory.VOLUME,
            score=min(score, 100),
            weight=weight,
            weighted_score=min(score, 100) * weight,
            details=details,
            confidence=confidence,
            issues=issues,
        )

    def _score_volatility(self, ad: dict, weight: float) -> CategoryScore:
        """
        Score VOLATILITY (0-100) basé sur:
        - ATR percentile (volatilité relative)
        - NATR (normalized ATR)
        - Volatility regime (low/normal/high/extreme)
        - Bollinger Width
        - Bollinger Squeeze/Expansion
        - Keltner Channels
        """
        details: dict[str, float] = {}
        issues: list[str] = []
        score = 0.0

        # 1. Volatility Regime (40 points max)
        vol_regime = ad.get("volatility_regime", "").lower()
        regime_scores = {
            "normal": 40.0,  # Optimal pour trading
            "high": 30.0,  # Acceptable mais plus risqué
            "low": 20.0,  # Peu de mouvement
            "extreme": 10.0,  # Trop risqué
        }
        regime_score = regime_scores.get(vol_regime, 20.0)
        score += regime_score
        details["regime"] = regime_score

        if vol_regime == "extreme":
            issues.append("Volatilité extrême")
        elif vol_regime == "low":
            issues.append("Volatilité faible")

        # 2. ATR Percentile (30 points max)
        atr_pct = self.safe_float(ad.get("atr_percentile"))
        if atr_pct > 0:
            # Percentile optimal: 40-70 (ni trop bas ni trop haut)
            if 40 <= atr_pct <= 70:
                atr_score = 30.0
            elif 30 <= atr_pct < 40 or 70 < atr_pct <= 80:
                atr_score = 20.0
            elif 20 <= atr_pct < 30 or 80 < atr_pct <= 90:
                atr_score = 10.0
            else:
                atr_score = 5.0

            score += atr_score
            details["atr_percentile"] = atr_score

        # 3. Bollinger Bands (20 points max)
        bb_squeeze = ad.get("bb_squeeze", False)
        bb_expansion = ad.get("bb_expansion", False)
        bb_width = self.safe_float(ad.get("bb_width"))

        bb_score = 0.0
        if bb_expansion:
            bb_score = 20.0  # Expansion = mouvement en cours
        elif bb_squeeze:
            bb_score = 10.0  # Squeeze = préparation mouvement
        elif bb_width > 0:
            # Width normal
            bb_score = 12.0

        score += bb_score
        details["bollinger"] = bb_score

        # 4. NATR (10 points max)
        natr = self.safe_float(ad.get("natr"))
        if natr > 0:
            # NATR optimal: 1.0-2.5%
            if 1.0 <= natr <= 2.5:
                natr_score = 10.0
            elif 0.5 <= natr < 1.0 or 2.5 < natr <= 3.5:
                natr_score = 7.0
            else:
                natr_score = 3.0

            score += natr_score
            details["natr"] = natr_score

        # Confiance
        confidence = min(100, len(details) * 25)

        return CategoryScore(
            category=ScoreCategory.VOLATILITY,
            score=min(score, 100),
            weight=weight,
            weighted_score=min(score, 100) * weight,
            details=details,
            confidence=confidence,
            issues=issues,
        )

    def _score_support_resistance(
        self, ad: dict, current_price: float, weight: float
    ) -> CategoryScore:
        """
        Score SUPPORT/RESISTANCE (0-100) basé sur:
        - Distance à la résistance
        - Break probability
        - Resistance strength
        - Support strength
        - Pivot count
        - Support/Resistance levels (JSONB)

        Args:
            ad: analyzer_data
            current_price: Prix actuel (REQUIS - ne pas utiliser nearest_support comme proxy!)
            weight: Pondération de la catégorie
        """
        details: dict[str, float] = {}
        issues: list[str] = []
        score = 0.0

        # 1. Distance à la résistance (40 points max)
        nearest_resistance = self.safe_float(ad.get("nearest_resistance"))
        break_prob = self.safe_float(ad.get("break_probability"))

        # NOUVEAU: Si nearest_resistance est NULL/0 = PAS DE PLAFOND = BULLISH!
        # Cela signifie qu'aucune résistance n'a été détectée dans les 100
        # dernières périodes
        if nearest_resistance == 0 or nearest_resistance is None:
            # Pas de résistance = bonus maximal + extra
            score += 50.0  # 50 au lieu de 40 car c'est TRÈS bullish
            details["resistance_distance"] = 50.0
            details["no_resistance_detected"] = 1.0
            issues.append("✅ Pas de résistance détectée = Ciel dégagé!")
        elif nearest_resistance > 0 and current_price > 0:
            dist_pct = (
                (nearest_resistance - current_price) / current_price) * 100

            # NOUVEAU: Si break_probability élevée, tolérer résistance proche
            # Une résistance à 0.1% avec break_prob 60%+ est un SETUP, pas un
            # problème
            if break_prob > 0.6 and dist_pct < 1.0:
                # Résistance proche MAIS cassable = score basé sur break_prob
                dist_score = min(40.0, float(break_prob * 50)
                                 )  # break_prob 0.7 → 35pts
                details["breakout_setup"] = 1.0
                issues.append(
                    f"Résistance proche ({dist_pct:.1f}%) mais cassable ({break_prob*100:.0f}%)"
                )
            # Scoring normal
            elif dist_pct > 3.0:
                dist_score = 40.0
            elif dist_pct > 2.0:
                dist_score = 30.0
            elif dist_pct > 1.5:
                dist_score = 20.0
            elif dist_pct > 1.0:
                dist_score = 10.0
            elif dist_pct > 0.5:
                dist_score = 5.0
            elif dist_pct > 0.2:
                # Résistance 0.2-0.5% = OK si momentum fort
                dist_score = 3.0
                # Pas de warning si momentum fort (sera géré dans validator)
            else:
                # < 0.2% = vraiment collé
                dist_score = 1.0
                # Pas de warning ici (sera contextuel dans validator)

            score += dist_score
            details["resistance_distance"] = float(dist_score)

        # 2. Break Probability (30 points max)
        if break_prob > 0:
            # break_probability: 0-1 (probabilité de casser la résistance)
            prob_score = break_prob * 30
            score += prob_score
            details["break_probability"] = prob_score

        # 3. Resistance Strength (15 points max) - INVERSE
        res_strength = ad.get("resistance_strength", "").upper()
        strength_scores = {
            "WEAK": 15.0,  # Résistance faible = bon pour acheter
            "MODERATE": 10.0,
            "STRONG": 5.0,
            "MAJOR": 0.0,  # Résistance majeure = risqué
        }
        res_score = strength_scores.get(res_strength, 7.0)
        score += res_score
        details["resistance_strength"] = float(res_score)

        if res_strength == "MAJOR":
            issues.append("Résistance MAJOR")

        # 4. Support Strength (10 points max)
        sup_strength = ad.get("support_strength", "").upper()
        sup_scores = {
            "MAJOR": 10.0,  # Support majeur = sécurité
            "STRONG": 8.0,
            "MODERATE": 5.0,
            "WEAK": 2.0,
        }
        sup_score = sup_scores.get(sup_strength, 5.0)
        score += sup_score
        details["support_strength"] = float(sup_score)

        # 5. Pivot Count (5 points max)
        pivot_count = ad.get("pivot_count", 0)
        if pivot_count > 0:
            # Plus de pivots = structure claire
            pivot_score = float(min(pivot_count * 0.5, 5))
            score += pivot_score
            details["pivots"] = pivot_score

        # Confiance
        confidence = min(100, len(details) * 20)

        return CategoryScore(
            category=ScoreCategory.SUPPORT_RESISTANCE,
            score=min(score, 100),
            weight=weight,
            weighted_score=min(score, 100) * weight,
            details=details,
            confidence=confidence,
            issues=issues,
        )

    def _score_pattern(self, ad: dict, weight: float) -> CategoryScore:
        """
        Score PATTERN (0-100) basé sur:
        - Pattern detected (PRICE_SPIKE_UP, LIQUIDITY_SWEEP, etc.)
        - Pattern confidence

        PUMP TOLERANCE: PRICE_SPIKE_DOWN et LIQUIDITY_SWEEP OK si pump
        """
        details: dict[str, float] = {}
        issues: list[str] = []
        score = 0.0

        pattern = ad.get("pattern_detected", "").upper()
        pattern_conf = self.safe_float(ad.get("pattern_confidence"))

        # Détecter pump context (même logique)
        vol_spike = self.safe_float(ad.get("volume_spike_multiplier"), 1.0)
        rel_volume = self.safe_float(ad.get("relative_volume"), 1.0)
        market_regime = ad.get("market_regime", "").upper()
        vol_context = ad.get("volume_context", "").upper()

        is_pump = (
            (vol_spike > 2.5 or rel_volume > 2.5) and market_regime in [
                "TRENDING_BULL",
                "BREAKOUT_BULL"] and vol_context in [
                "CONSOLIDATION_BREAK",
                "BREAKOUT",
                "PUMP_START",
                "HIGH_VOLATILITY"])

        # Patterns bullish
        bullish_patterns = [
            "PRICE_SPIKE_UP",
            "COMBINED_SPIKE",
            "VOLUME_SPIKE_UP"]

        if pattern in bullish_patterns:
            base_score = 60.0
        elif pattern == "NORMAL" or not pattern:
            # NORMAL = marché calme en TRENDING_BULL = OK, pas négatif
            base_score = 50.0  # Augmenté de 30 → 50 (neutre positif)
        elif pattern == "PRICE_SPIKE_DOWN" and is_pump:
            # Pendant un pump, PRICE_SPIKE_DOWN = pullback sain
            base_score = 40.0
            issues.append(f"Pattern pullback pump ({pattern})")
        elif pattern == "LIQUIDITY_SWEEP" and market_regime in [
            "TRENDING_BULL",
            "BREAKOUT_BULL",
        ]:
            # LIQUIDITY_SWEEP en bull = sweep des shorts = bullish
            base_score = 50.0
            issues.append(f"Pattern sweep shorts ({pattern})")
        elif pattern in ["PRICE_SPIKE_DOWN", "LIQUIDITY_SWEEP"]:
            # Vraiment baissier si pas de pump context
            base_score = 0.0
            issues.append(f"Pattern {pattern}")
        else:
            base_score = 20.0

        # Ajuster par confiance
        if pattern_conf > 0:
            score = base_score * (pattern_conf / 100)
            details["pattern"] = score
        else:
            score = base_score * 0.5
            details["pattern"] = score

        confidence = pattern_conf if pattern_conf > 0 else 50

        return CategoryScore(
            category=ScoreCategory.PATTERN,
            score=min(score, 100),
            weight=weight,
            weighted_score=min(score, 100) * weight,
            details=details,
            confidence=confidence,
            issues=issues,
        )

    def _score_confluence(self, ad: dict, weight: float) -> CategoryScore:
        """
        Score CONFLUENCE (0-100) basé sur:
        - Confluence score (calculé par market_analyzer)
        - Signal strength
        """
        details: dict[str, float] = {}
        issues: list[str] = []
        score = 0.0

        # 1. Confluence Score (60 points max)
        confluence = self.safe_float(ad.get("confluence_score"))
        if confluence > 0:
            conf_contrib = (confluence / 100) * 60
            score += conf_contrib
            details["confluence"] = conf_contrib

        # 2. Signal Strength (40 points max)
        signal_str = ad.get("signal_strength", "").upper()
        strength_scores = {
            "VERY_STRONG": 40.0,
            "STRONG": 30.0,
            "MODERATE": 20.0,
            "WEAK": 10.0,
            "VERY_WEAK": 5.0,
        }
        sig_score = strength_scores.get(signal_str, 15.0)
        score += sig_score
        details["signal_strength"] = sig_score

        confidence = 100 if confluence > 0 and signal_str else 50

        return CategoryScore(
            category=ScoreCategory.CONFLUENCE,
            score=min(score, 100),
            weight=weight,
            weighted_score=min(score, 100) * weight,
            details=details,
            confidence=confidence,
            issues=issues,
        )

    def _calculate_grade(self, score: float) -> str:
        """Calcule le grade S/A/B/C/D/F."""
        if score >= 90:
            return "S"
        if score >= 80:
            return "A"
        if score >= 70:
            return "B"
        if score >= 60:
            return "C"
        if score >= 50:
            return "D"
        return "F"

    def _calculate_risk_level(self, ad: dict, score: float) -> str:
        """Calcule le niveau de risque."""
        vol_regime = ad.get("volatility_regime", "").lower()
        regime = ad.get("market_regime", "").upper()

        # Risque extrême si volatilité extrême
        if vol_regime == "extreme":
            return "EXTREME"

        # Risque élevé si régime bearish ou score faible
        if regime in ["TRENDING_BEAR", "BREAKOUT_BEAR"] or score < 50:
            return "HIGH"

        # Risque moyen si volatilité haute ou score moyen
        if vol_regime == "high" or 50 <= score < 70:
            return "MEDIUM"

        # Risque faible si tout OK
        return "LOW"

    def _calculate_recommendation(
        self, score: float, confidence: float, category_scores: dict, _ad: dict
    ) -> tuple[str, list[str], list[str]]:
        """Calcule la recommandation finale."""
        reasons: list[str] = []
        warnings: list[str] = []

        # Récupérer scores clés
        trend_score = category_scores[ScoreCategory.TREND].score
        momentum_score = category_scores[ScoreCategory.MOMENTUM].score
        volume_score = category_scores[ScoreCategory.VOLUME].score

        # BUY_NOW: Score >80, confiance >70, trend+momentum+volume >70
        if score >= 80 and confidence >= 70:
            if trend_score >= 70 and momentum_score >= 70 and volume_score >= 70:
                reasons.append(
                    f"Score excellent: {score:.0f}/100 (Grade {self._calculate_grade(score)})"
                )
                reasons.append(f"Confiance élevée: {confidence:.0f}%")
                reasons.append(
                    f"Trend/Momentum/Volume alignés: {trend_score:.0f}/{momentum_score:.0f}/{volume_score:.0f}"
                )
                return "BUY_NOW", reasons, warnings

        # BUY_DCA: Score 70-80, bon mais pas parfait
        if 70 <= score < 80 and confidence >= 60:
            reasons.append(
                f"Score bon: {score:.0f}/100 (Grade {self._calculate_grade(score)})"
            )
            reasons.append("Entrée progressive recommandée (DCA)")

            # Avertissements si certains scores faibles
            if trend_score < 60:
                warnings.append(f"Trend score modéré: {trend_score:.0f}/100")
            if volume_score < 60:
                warnings.append(f"Volume score modéré: {volume_score:.0f}/100")

            return "BUY_DCA", reasons, warnings

        # WAIT: Score 60-70 ou confiance faible
        if 60 <= score < 70 or confidence < 60:
            reasons.append(
                f"Score moyen: {score:.0f}/100 (Grade {self._calculate_grade(score)})"
            )
            reasons.append("Attendre confirmation supplémentaire")

            # Indiquer ce qui manque
            if trend_score < 60:
                warnings.append(f"Trend faible: {trend_score:.0f}/100")
            if momentum_score < 60:
                warnings.append(f"Momentum faible: {momentum_score:.0f}/100")
            if volume_score < 60:
                warnings.append(f"Volume insuffisant: {volume_score:.0f}/100")

            return "WAIT", reasons, warnings

        # AVOID: Score <60
        reasons.append(
            f"Score faible: {score:.0f}/100 (Grade {self._calculate_grade(score)})"
        )
        reasons.append("Conditions non favorables")

        # Lister les problèmes
        for cat, cat_score in category_scores.items():
            if cat_score.score < 40:
                warnings.append(
                    f"{cat.value}: {cat_score.score:.0f}/100 - {', '.join(cat_score.issues)}"
                )

        return "AVOID", reasons, warnings

    def _create_zero_score(self, reason: str) -> OpportunityScore:
        """Crée un score de 0 avec raison."""
        zero_category = CategoryScore(
            category=ScoreCategory.TREND,
            score=0.0,
            weight=0.0,
            weighted_score=0.0,
            details={},
            confidence=0.0,
            issues=[reason],
        )

        return OpportunityScore(
            total_score=0.0,
            grade="F",
            category_scores=dict.fromkeys(ScoreCategory, zero_category),
            confidence=0.0,
            risk_level="EXTREME",
            recommendation="AVOID",
            reasons=[reason],
            warnings=[],
        )
