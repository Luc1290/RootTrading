"""
Opportunity Scoring System - CORRECTED FOR BUYING EARLY
Version: 3.0 - Buy BEFORE pump, not DURING

CHANGEMENTS MAJEURS:
1. RSI optimal: 35-55 (sortie oversold) au lieu de 60-75 (overbought)
2. Volume: BUILDUP (progression) au lieu de SPIKE (pic)
3. ROC: Faible/négatif récent = opportunité, fort = trop tard
4. Résistance: Distance >2% requise, <1% = déjà au plafond
5. MFI: 40-60 optimal, >70 = argent déjà entré
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
    """Système de scoring pour acheter AVANT le pump."""

    # Pondérations par défaut (AJUSTÉES pour early detection)
    DEFAULT_WEIGHTS = {
        ScoreCategory.TREND: 0.20,  # Réduit (trend confirme, pas prédit)
        ScoreCategory.MOMENTUM: 0.15,  # Réduit (momentum = lagging)
        ScoreCategory.VOLUME: 0.30,  # AUGMENTÉ (volume buildup = leading)
        ScoreCategory.VOLATILITY: 0.10,
        ScoreCategory.SUPPORT_RESISTANCE: 0.20,  # AUGMENTÉ (distance critique)
        ScoreCategory.PATTERN: 0.05,
        ScoreCategory.CONFLUENCE: 0.00,  # Désactivé (confluence = lagging)
    }

    # Pondérations selon régime (EARLY FOCUS)
    REGIME_WEIGHTS = {
        "TRENDING_BULL": {
            ScoreCategory.TREND: 0.15,  # Trend déjà établi
            ScoreCategory.MOMENTUM: 0.10,  # Momentum suit
            ScoreCategory.VOLUME: 0.40,  # VOLUME = prédicteur #1
            ScoreCategory.VOLATILITY: 0.05,
            ScoreCategory.SUPPORT_RESISTANCE: 0.25,  # Distance critique
            ScoreCategory.PATTERN: 0.05,
            ScoreCategory.CONFLUENCE: 0.00,
        },
        "BREAKOUT_BULL": {
            ScoreCategory.TREND: 0.10,
            ScoreCategory.MOMENTUM: 0.15,
            ScoreCategory.VOLUME: 0.45,  # VOLUME critique pour breakout
            ScoreCategory.VOLATILITY: 0.10,
            ScoreCategory.SUPPORT_RESISTANCE: 0.15,  # On casse la résistance
            ScoreCategory.PATTERN: 0.05,
            ScoreCategory.CONFLUENCE: 0.00,
        },
        "RANGING": {
            ScoreCategory.TREND: 0.10,
            ScoreCategory.MOMENTUM: 0.10,
            ScoreCategory.VOLUME: 0.25,
            ScoreCategory.VOLATILITY: 0.15,
            ScoreCategory.SUPPORT_RESISTANCE: 0.35,  # S/R critique en range
            ScoreCategory.PATTERN: 0.05,
            ScoreCategory.CONFLUENCE: 0.00,
        },
        "VOLATILE": {
            ScoreCategory.TREND: 0.15,
            ScoreCategory.MOMENTUM: 0.15,
            ScoreCategory.VOLUME: 0.30,
            ScoreCategory.VOLATILITY: 0.20,
            ScoreCategory.SUPPORT_RESISTANCE: 0.15,
            ScoreCategory.PATTERN: 0.05,
            ScoreCategory.CONFLUENCE: 0.00,
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
        """Calcule le score global d'opportunité."""
        if not analyzer_data:
            return self._create_zero_score("Pas de données")

        regime = analyzer_data.get("market_regime", "UNKNOWN")
        weights = self._get_weights_for_regime(regime)

        category_scores = {}

        # 1. TREND SCORE (LAGGING - pondération réduite)
        category_scores[ScoreCategory.TREND] = self._score_trend(
            analyzer_data, weights[ScoreCategory.TREND]
        )

        # 2. MOMENTUM SCORE (CORRECTED - cherche sortie oversold)
        category_scores[ScoreCategory.MOMENTUM] = self._score_momentum_early(
            analyzer_data, weights[ScoreCategory.MOMENTUM]
        )

        # 3. VOLUME SCORE (LEADING - pondération max)
        category_scores[ScoreCategory.VOLUME] = self._score_volume_early(
            analyzer_data, weights[ScoreCategory.VOLUME]
        )

        # 4. VOLATILITY SCORE
        category_scores[ScoreCategory.VOLATILITY] = self._score_volatility(
            analyzer_data, weights[ScoreCategory.VOLATILITY]
        )

        # 5. SUPPORT/RESISTANCE SCORE (CRITICAL - distance au plafond)
        category_scores[ScoreCategory.SUPPORT_RESISTANCE] = (
            self._score_support_resistance_early(
                analyzer_data, current_price, weights[ScoreCategory.SUPPORT_RESISTANCE]
            )
        )

        # 6. PATTERN SCORE
        category_scores[ScoreCategory.PATTERN] = self._score_pattern(
            analyzer_data, weights[ScoreCategory.PATTERN]
        )

        # 7. CONFLUENCE SCORE (désactivé)
        category_scores[ScoreCategory.CONFLUENCE] = self._score_confluence(
            analyzer_data, weights[ScoreCategory.CONFLUENCE]
        )

        # Score total
        total_score = sum(cs.weighted_score for cs in category_scores.values())
        confidence = sum(
            cs.confidence * cs.weight for cs in category_scores.values()
        ) / sum(weights.values())

        grade = self._calculate_grade(total_score)
        risk_level = self._calculate_risk_level(analyzer_data, total_score)
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

    def _get_weights_for_regime(self, regime: str) -> dict[ScoreCategory, float]:
        """Retourne les pondérations adaptées au régime."""
        return self.REGIME_WEIGHTS.get(regime, self.DEFAULT_WEIGHTS)

    def _score_trend(self, ad: dict, weight: float) -> CategoryScore:
        """Score TREND - INCHANGÉ (lagging mais nécessaire)."""
        details: dict[str, float] = {}
        issues: list[str] = []
        score = 0.0

        # Trend confirme mais ne prédit pas
        # Code identique à l'original...

        # ADX + DI
        adx = self.safe_float(ad.get("adx_14"))
        plus_di = self.safe_float(ad.get("plus_di"))
        minus_di = self.safe_float(ad.get("minus_di"))

        if adx > 0:
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
                di_score = 10.0

            score += adx_score + di_score
            details["adx"] = float(adx_score)
            details["directional"] = float(di_score)

        # Trend Alignment
        trend_alignment = self.safe_float(ad.get("trend_alignment"))
        if trend_alignment != 0:
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

        # Directional Bias
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

        confidence = 100.0 if len(details) >= 3 else 50.0

        return CategoryScore(
            category=ScoreCategory.TREND,
            score=min(score, 100),
            weight=weight,
            weighted_score=min(score, 100) * weight,
            details=details,
            confidence=confidence,
            issues=issues,
        )

    def _score_momentum_early(self, ad: dict, weight: float) -> CategoryScore:
        """
        Score MOMENTUM - CORRECTED pour EARLY detection.

        CHANGEMENTS:
        - RSI optimal: 35-55 (sortie oversold) au lieu de 60-75
        - RSI > 70 = PÉNALITÉ (déjà overbought)
        - ROC faible/négatif récent = BONUS (pas encore parti)
        - MFI 40-60 optimal, >70 = argent déjà entré
        - Stochastic <50 = meilleur qu'overbought
        """
        details: dict[str, float] = {}
        issues: list[str] = []
        score = 0.0

        # 1. RSI (25 points max) - CORRECTED
        rsi_14 = self.safe_float(ad.get("rsi_14"))
        rsi_21 = self.safe_float(ad.get("rsi_21"))

        if rsi_14 > 0:
            # Zone EARLY buy: 35-55 (sortie oversold avant pump)
            if 40 <= rsi_14 <= 55:
                rsi_score = 25.0  # OPTIMAL: ni oversold ni overbought
            elif 35 <= rsi_14 < 40:
                rsi_score = 20.0  # Sortie oversold = très bon
            elif 55 < rsi_14 <= 60:
                rsi_score = 15.0  # Début overbought mais acceptable
            elif 30 <= rsi_14 < 35:
                rsi_score = 15.0  # Encore oversold mais proche sortie
            elif 60 < rsi_14 <= 70:
                rsi_score = 8.0  # Overbought modéré = déjà en hausse
                issues.append(f"RSI élevé ({rsi_14:.0f}) - mouvement déjà commencé")
            elif rsi_14 > 70:
                rsi_score = 0.0  # Overbought = TROP TARD
                issues.append(f"RSI overbought ({rsi_14:.0f}) - TROP TARD pour acheter")
            else:
                rsi_score = 5.0  # < 30 = deep oversold, risqué
                issues.append(f"RSI très faible ({rsi_14:.0f})")

            # Bonus cohérence RSI 14 et 21
            if rsi_21 > 0 and abs(rsi_14 - rsi_21) < 10:
                rsi_score = min(rsi_score + 3, 25)

            score += rsi_score
            details["rsi"] = rsi_score

        # 2. MACD (20 points max) - légèrement réduit
        macd_trend = ad.get("macd_trend", "").upper()
        macd_hist = self.safe_float(ad.get("macd_histogram"))

        macd_score = 0.0
        if macd_trend == "BULLISH":
            macd_score = 12.0
            if macd_hist > 0:
                macd_score += 5.0
        elif macd_trend == "NEUTRAL":
            macd_score = 8.0  # Neutral acceptable pour early entry
        else:
            macd_score = 0.0
            issues.append(f"MACD {macd_trend}")

        score += min(macd_score, 20)
        details["macd"] = min(macd_score, 20)

        # 3. ROC (15 points max) - CORRECTED
        roc_10 = self.safe_float(ad.get("roc_10"))
        if roc_10 is not None:
            # ROC faible/négatif récent = PAS ENCORE PARTI = BON
            if -0.5 < roc_10 <= 0.2:
                roc_score = 15.0  # OPTIMAL: momentum flat, prêt à exploser
                issues.append(f"ROC faible ({roc_10:.2f}%) - momentum à capturer")
            elif 0.2 < roc_10 <= 0.5:
                roc_score = 10.0  # Début accélération = acceptable
            elif 0.5 < roc_10 <= 1.0:
                roc_score = 5.0  # Déjà en accélération
                issues.append(f"ROC modéré ({roc_10:.2f}%) - mouvement commencé")
            elif roc_10 > 1.0:
                roc_score = 0.0  # Forte accélération = TROP TARD
                issues.append(f"ROC fort ({roc_10:.2f}%) - TROP TARD")
            else:
                roc_score = 8.0  # Négatif = correction, mais OK si trend bull

            score += roc_score
            details["roc"] = roc_score

        # 4. MFI (15 points max) - CORRECTED
        mfi = self.safe_float(ad.get("mfi_14"))
        if mfi > 0:
            # MFI optimal: 40-60 (argent commence à entrer)
            if 40 <= mfi <= 60:
                mfi_score = 15.0  # OPTIMAL
            elif 30 <= mfi < 40:
                mfi_score = 10.0  # Peu d'argent mais acceptable
            elif 60 < mfi <= 70:
                mfi_score = 7.0  # Argent entrant mais pas trop
                issues.append(f"MFI élevé ({mfi:.0f}) - flux acheteur déjà fort")
            elif mfi > 70:
                mfi_score = 0.0  # Argent déjà massivement entré = TROP TARD
                issues.append(f"MFI overbought ({mfi:.0f}) - TROP TARD")
            else:
                mfi_score = 5.0  # < 30 = peu d'intérêt

            score += mfi_score
            details["mfi"] = mfi_score

        # 5. Stochastic (15 points max) - CORRECTED
        stoch_k = self.safe_float(ad.get("stoch_k"))
        stoch_d = self.safe_float(ad.get("stoch_d"))

        stoch_score = 0.0
        if stoch_k > 0:
            # Zone EARLY: 30-60 (sortie oversold)
            if 35 <= stoch_k <= 60 and 35 <= stoch_d <= 60:
                stoch_score = 15.0  # OPTIMAL
            elif 20 <= stoch_k < 35:
                stoch_score = 12.0  # Sortie oversold = très bon
            elif 60 < stoch_k <= 75:
                stoch_score = 8.0  # Début overbought
                issues.append(f"Stoch élevé ({stoch_k:.0f}/{stoch_d:.0f})")
            elif stoch_k > 75:
                stoch_score = 0.0  # Overbought = TROP TARD
                issues.append(f"Stoch overbought ({stoch_k:.0f}/{stoch_d:.0f}) - TROP TARD")
            else:
                stoch_score = 6.0

            score += min(stoch_score, 15)
            details["stochastic"] = min(stoch_score, 15)

        # 6. Williams %R (10 points max)
        williams = self.safe_float(ad.get("williams_r"))
        if williams != 0:
            # Williams optimal: -70 à -30 (sortie oversold)
            if -70 <= williams <= -30:
                will_score = 10.0
            elif -80 <= williams < -70:
                will_score = 8.0
            elif -30 < williams <= -10:
                will_score = 4.0
                issues.append(f"Williams élevé ({williams:.0f})")
            elif williams > -10:
                will_score = 0.0
                issues.append(f"Williams overbought ({williams:.0f}) - TROP TARD")
            else:
                will_score = 5.0

            score += will_score
            details["williams"] = will_score

        # Confiance
        confidence = min(100, len(details) * 18)

        return CategoryScore(
            category=ScoreCategory.MOMENTUM,
            score=min(score, 100),
            weight=weight,
            weighted_score=min(score, 100) * weight,
            details=details,
            confidence=confidence,
            issues=issues,
        )

    def _score_volume_early(self, ad: dict, weight: float) -> CategoryScore:
        """
        Score VOLUME - CORRECTED pour EARLY detection.

        CHANGEMENTS:
        - Volume BUILDUP (progression 1.2-2x) > SPIKE (>3x)
        - Spike >3x = PÉNALITÉ (pic, pas début)
        - Pattern BUILDUP = score max
        - Context PUMP_START = warning (déjà commencé)
        """
        details: dict[str, float] = {}
        issues: list[str] = []
        score = 0.0

        rel_volume = self.safe_float(ad.get("relative_volume"), 1.0)
        vol_spike = self.safe_float(ad.get("volume_spike_multiplier"), 1.0)
        buildup_periods = ad.get("volume_buildup_periods", 0)

        # 1. Volume BUILDUP (30 points max) - PRIORITÉ
        if buildup_periods >= 3:
            buildup_score = 30.0  # 3+ périodes buildup = OPTIMAL
            issues.append(f"Volume buildup {buildup_periods} périodes - EARLY signal")
        elif buildup_periods == 2:
            buildup_score = 25.0
        elif buildup_periods == 1:
            buildup_score = 15.0
        else:
            buildup_score = 0.0

        score += buildup_score
        details["buildup"] = buildup_score

        # 2. Relative Volume (25 points max) - CORRECTED
        # Zone EARLY: 1.2-2.5x (progression, pas explosion)
        if 1.5 <= rel_volume <= 2.5 and vol_spike < 3.0:
            vol_score = 25.0  # OPTIMAL: volume croissant mais pas spike
            issues.append(f"Volume progression {rel_volume:.1f}x - EARLY signal")
        elif 1.2 <= rel_volume < 1.5:
            vol_score = 20.0  # Début progression
        elif 2.5 < rel_volume <= 3.5 and vol_spike < 3.0:
            vol_score = 15.0  # Fort mais acceptable
        elif vol_spike >= 3.0 or rel_volume > 3.5:
            vol_score = 5.0  # SPIKE = PIC = TROP TARD
            issues.append(f"Volume spike {vol_spike:.1f}x - PIC atteint, TROP TARD")
        elif rel_volume > 1.0:
            vol_score = 12.0
        else:
            vol_score = 0.0
            issues.append(f"Volume faible ({rel_volume:.2f}x)")

        score += vol_score
        details["relative_volume"] = vol_score

        # 3. Volume Context (20 points max) - CORRECTED
        vol_context = ad.get("volume_context", "").upper()
        context_scores = {
            "CONSOLIDATION_BREAK": 20.0,  # Casse consolidation = EARLY
            "TREND_CONTINUATION": 15.0,  # Continuation = acceptable
            "NEUTRAL": 10.0,
            "BREAKOUT": 8.0,  # Breakout = déjà commencé
            "PUMP_START": 5.0,  # Pump start = déjà parti
            "HIGH_VOLATILITY": 3.0,
            "OVERSOLD_BOUNCE": 12.0,  # Bounce oversold = bon
            "LOW_VOLATILITY": 8.0,
            "MODERATE_OVERSOLD": 10.0,
            "DEEP_OVERSOLD": 0.0,
            "REVERSAL_PATTERN": 0.0,
        }
        context_score = context_scores.get(vol_context, 5.0)

        if vol_context in ["PUMP_START", "BREAKOUT"]:
            issues.append(f"Context {vol_context} - mouvement déjà commencé")

        score += context_score
        details["context"] = context_score

        # 4. Volume Pattern (15 points max) - CORRECTED
        vol_pattern = ad.get("volume_pattern", "").upper()
        pattern_scores = {
            "BUILDUP": 15.0,  # OPTIMAL
            "SUSTAINED_HIGH": 10.0,  # Soutenu mais pas spike
            "NORMAL": 8.0,
            "SPIKE": 3.0,  # Spike = pic = pas early
            "DECLINING": 0.0,
        }
        pattern_score = pattern_scores.get(vol_pattern, 5.0)

        if vol_pattern == "SPIKE":
            issues.append("Pattern SPIKE - pic volume, TROP TARD")

        score += pattern_score
        details["pattern"] = pattern_score

        # 5. OBV Oscillator (10 points max)
        obv_osc = self.safe_float(ad.get("obv_oscillator"))
        if obv_osc > 0:
            # OBV positif mais pas extrême
            if 50 < obv_osc <= 200:
                obv_score = 10.0  # OPTIMAL: buying pressure croissante
            elif 200 < obv_osc <= 300:
                obv_score = 7.0
            elif obv_osc > 300:
                obv_score = 3.0  # Trop fort = déjà avancé
                issues.append(f"OBV très élevé ({obv_osc:.0f}) - buying pressure extrême")
            else:
                obv_score = 5.0
        else:
            obv_score = 0.0
            if obv_osc < -100:
                issues.append(f"OBV négatif ({obv_osc:.0f})")

        score += obv_score
        details["obv"] = obv_score

        # Confiance
        confidence = min(100, len(details) * 20)

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
        """Score VOLATILITY - INCHANGÉ."""
        details: dict[str, float] = {}
        issues: list[str] = []
        score = 0.0

        vol_regime = ad.get("volatility_regime", "").lower()
        regime_scores = {
            "normal": 40.0,
            "high": 30.0,
            "low": 20.0,
            "extreme": 10.0,
        }
        regime_score = regime_scores.get(vol_regime, 20.0)
        score += regime_score
        details["regime"] = regime_score

        if vol_regime == "extreme":
            issues.append("Volatilité extrême")
        elif vol_regime == "low":
            issues.append("Volatilité faible")

        atr_pct = self.safe_float(ad.get("atr_percentile"))
        if atr_pct > 0:
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

        confidence = min(100, len(details) * 30)

        return CategoryScore(
            category=ScoreCategory.VOLATILITY,
            score=min(score, 100),
            weight=weight,
            weighted_score=min(score, 100) * weight,
            details=details,
            confidence=confidence,
            issues=issues,
        )

    def _score_support_resistance_early(
        self, ad: dict, current_price: float, weight: float
    ) -> CategoryScore:
        """
        Score S/R - CORRECTED pour EARLY detection.

        CHANGEMENTS:
        - Distance résistance >2% REQUISE (sinon déjà au plafond)
        - <1% = PÉNALITÉ LOURDE (collision imminente)
        - Break probability ignorée (on veut de la MARGE, pas casser)
        - Support proche = bonus (filet sécurité)
        """
        details: dict[str, float] = {}
        issues: list[str] = []
        score = 0.0

        nearest_resistance = self.safe_float(ad.get("nearest_resistance"))
        nearest_support = self.safe_float(ad.get("nearest_support"))

        # 1. Distance résistance (50 points max) - CRITIQUE
        if nearest_resistance == 0 or nearest_resistance is None:
            # Pas de résistance = bonus
            score += 50.0
            details["resistance_distance"] = 50.0
            issues.append("✅ Pas de résistance détectée = espace libre")
        elif nearest_resistance > 0 and current_price > 0:
            dist_pct = ((nearest_resistance - current_price) / current_price) * 100

            # STRICT: Distance >2% REQUISE pour early entry
            if dist_pct > 5.0:
                dist_score = 50.0  # OPTIMAL: beaucoup d'espace
            elif dist_pct > 3.0:
                dist_score = 40.0  # Bon espace
            elif dist_pct > 2.0:
                dist_score = 30.0  # Espace minimal acceptable
            elif dist_pct > 1.0:
                dist_score = 10.0  # Trop proche
                issues.append(f"⚠️ Résistance proche ({dist_pct:.1f}%) - risque rejet")
            elif dist_pct > 0.5:
                dist_score = 3.0  # Très proche = RISQUÉ
                issues.append(f"⚠️⚠️ Résistance très proche ({dist_pct:.1f}%) - COLLISION imminente")
            else:
                dist_score = 0.0  # <0.5% = déjà au plafond = TROP TARD
                issues.append(f"❌ Résistance collée ({dist_pct:.1f}%) - TROP TARD, déjà au plafond")

            score += dist_score
            details["resistance_distance"] = float(dist_score)

        # 2. Support proche (20 points max) - BONUS sécurité
        if nearest_support > 0 and current_price > nearest_support:
            support_dist_pct = ((current_price - nearest_support) / current_price) * 100

            if 1.0 <= support_dist_pct <= 3.0:
                support_score = 20.0  # OPTIMAL: support proche = filet
            elif 0.5 <= support_dist_pct < 1.0:
                support_score = 15.0
            elif support_dist_pct > 3.0:
                support_score = 10.0  # Support loin mais acceptable
            else:
                support_score = 5.0

            score += support_score
            details["support_distance"] = float(support_score)

        # 3. Resistance Strength (15 points max) - INVERSE
        res_strength_raw = ad.get("resistance_strength", "")

        # Handle both string and float formats
        if isinstance(res_strength_raw, int | float):
            # Float format (0-1): lower = weaker resistance = better
            if res_strength_raw <= 0.3:
                res_score = 15.0  # Weak
            elif res_strength_raw <= 0.5:
                res_score = 10.0  # Moderate
            elif res_strength_raw <= 0.7:
                res_score = 5.0  # Strong
            else:
                res_score = 0.0  # Major
        else:
            # String format
            res_strength = str(res_strength_raw).upper()
            strength_scores = {
                "WEAK": 15.0,
                "MODERATE": 10.0,
                "STRONG": 5.0,
                "MAJOR": 0.0,
            }
            res_score = strength_scores.get(res_strength, 7.0)
        score += res_score
        details["resistance_strength"] = float(res_score)

        if (isinstance(res_strength_raw, str) and res_strength_raw.upper() == "MAJOR") or (isinstance(res_strength_raw, int | float) and res_strength_raw > 0.8):
            issues.append("Résistance MAJOR - risque rejet élevé")

        # 4. Support Strength (15 points max)
        sup_strength_raw = ad.get("support_strength", "")

        # Handle both string and float formats
        if isinstance(sup_strength_raw, int | float):
            # Float format (0-1): higher = stronger support = better
            if sup_strength_raw >= 0.8:
                sup_score = 15.0  # Major
            elif sup_strength_raw >= 0.6:
                sup_score = 12.0  # Strong
            elif sup_strength_raw >= 0.4:
                sup_score = 8.0  # Moderate
            else:
                sup_score = 4.0  # Weak
        else:
            # String format
            sup_strength = str(sup_strength_raw).upper()
            sup_scores = {
                "MAJOR": 15.0,
                "STRONG": 12.0,
                "MODERATE": 8.0,
                "WEAK": 4.0,
            }
            sup_score = sup_scores.get(sup_strength, 8.0)
        score += sup_score
        details["support_strength"] = float(sup_score)

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
        """Score PATTERN - INCHANGÉ (contribution mineure)."""
        details: dict[str, float] = {}
        issues: list[str] = []
        score = 0.0

        pattern = ad.get("pattern_detected", "").upper()
        pattern_conf = self.safe_float(ad.get("pattern_confidence"))

        bullish_patterns = ["PRICE_SPIKE_UP", "COMBINED_SPIKE", "VOLUME_SPIKE_UP"]

        if pattern in bullish_patterns:
            base_score = 60.0
        elif pattern == "NORMAL" or not pattern:
            base_score = 50.0
        elif pattern in ["PRICE_SPIKE_DOWN", "LIQUIDITY_SWEEP"]:
            base_score = 0.0
            issues.append(f"Pattern {pattern}")
        else:
            base_score = 20.0

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

    def _score_confluence(self, _ad: dict, _weight: float) -> CategoryScore:
        """Score CONFLUENCE - DÉSACTIVÉ (lagging indicator)."""
        # Parameters prefixed with _ to indicate intentionally unused
        # Method kept for interface consistency
        details: dict[str, float] = {}
        issues: list[str] = []

        # Confluence désactivé car composite des autres indicateurs (lagging)
        confidence = 0

        return CategoryScore(
            category=ScoreCategory.CONFLUENCE,
            score=0.0,
            weight=0.0,
            weighted_score=0.0,
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

        if vol_regime == "extreme":
            return "EXTREME"

        if regime in ["TRENDING_BEAR", "BREAKOUT_BEAR"] or score < 50:
            return "HIGH"

        if vol_regime == "high" or 50 <= score < 70:
            return "MEDIUM"

        return "LOW"

    def _calculate_recommendation(
        self, score: float, confidence: float, category_scores: dict, _ad: dict
    ) -> tuple[str, list[str], list[str]]:
        """Calcule la recommandation finale."""
        reasons: list[str] = []
        warnings: list[str] = []

        volume_score = category_scores[ScoreCategory.VOLUME].score
        momentum_score = category_scores[ScoreCategory.MOMENTUM].score
        sr_score = category_scores[ScoreCategory.SUPPORT_RESISTANCE].score

        # BUY_NOW: Score >75, volume+sr OK (momentum moins critique)
        if (
            score >= 75
            and confidence >= 65
            and volume_score >= 60  # Volume critique
            and sr_score >= 50  # Distance résistance suffisante
        ):
            reasons.append(f"Score excellent: {score:.0f}/100 (Grade {self._calculate_grade(score)})")
            reasons.append(f"Volume buildup détecté: {volume_score:.0f}/100")
            reasons.append(f"Distance résistance acceptable: {sr_score:.0f}/100")
            return "BUY_NOW", reasons, warnings

        # BUY_DCA: Score 65-75
        if 65 <= score < 75 and confidence >= 60:
            reasons.append(f"Score bon: {score:.0f}/100 (Grade {self._calculate_grade(score)})")
            reasons.append("Entrée progressive recommandée (DCA)")

            if volume_score < 50:
                warnings.append(f"Volume faible: {volume_score:.0f}/100")
            if sr_score < 40:
                warnings.append(f"Distance résistance insuffisante: {sr_score:.0f}/100")

            return "BUY_DCA", reasons, warnings

        # WAIT: Score 55-65
        if 55 <= score < 65 or confidence < 60:
            reasons.append(f"Score moyen: {score:.0f}/100 (Grade {self._calculate_grade(score)})")
            reasons.append("Attendre meilleure opportunité")

            if volume_score < 50:
                warnings.append(f"Volume insuffisant: {volume_score:.0f}/100")
            if momentum_score < 50:
                warnings.append(f"Momentum faible: {momentum_score:.0f}/100")
            if sr_score < 40:
                warnings.append(f"Trop proche résistance: {sr_score:.0f}/100")

            return "WAIT", reasons, warnings

        # AVOID: Score <55
        reasons.append(f"Score faible: {score:.0f}/100 (Grade {self._calculate_grade(score)})")
        reasons.append("Conditions non favorables pour entry")

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
