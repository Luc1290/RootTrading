"""
Opportunity Scoring System - INSTITUTIONAL SCALPING INDICATORS
Version: 4.0 - Professional intraday scalping avec indicateurs reconnus

INDICATEURS UTILISÉS (7 essentiels):
1. VWAP - Price vs VWAP (THE most institutional indicator)
2. EMA 7/12/26 - Short-term trend et crossovers
3. RSI 14 - Oversold/Overbought (réaliste 30-70)
4. Volume - Relative volume et confirmation
5. Bollinger Bands - Squeeze et expansion
6. MACD - Trend confirmation
7. Support/Resistance - Niveaux clés (non bloquants)

SUPPRIMÉS:
- EMA 50/99, SMA (trop lents pour scalping)
- MFI, Williams R, CCI (redondants avec RSI)
- ROC, Momentum (déjà dans MACD)
- Confluence score (recalculé inutilement)
"""

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ScoreCategory(Enum):
    """Catégories de scoring pour scalping."""

    VWAP_POSITION = "vwap_position"  # NOUVEAU - Le plus important
    EMA_TREND = "ema_trend"  # NOUVEAU - Remplace "trend"
    RSI_MOMENTUM = "rsi_momentum"  # SIMPLIFIÉ - Juste RSI
    VOLUME = "volume"
    BOLLINGER = "bollinger"  # NOUVEAU - Séparé
    MACD = "macd"  # NOUVEAU - Séparé
    SUPPORT_RESISTANCE = "support_resistance"


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
    """Système de scoring INSTITUTIONNEL pour scalping intraday."""

    # Pondérations SCALPING (basé sur l'importance réelle)
    DEFAULT_WEIGHTS = {
        ScoreCategory.VWAP_POSITION: 0.25,  # LE PLUS IMPORTANT pour institutionnels
        ScoreCategory.EMA_TREND: 0.20,  # Trend court terme essentiel
        ScoreCategory.VOLUME: 0.20,  # Confirmation obligatoire
        ScoreCategory.RSI_MOMENTUM: 0.15,  # Oversold/overbought
        ScoreCategory.BOLLINGER: 0.10,  # Squeeze/expansion
        ScoreCategory.MACD: 0.05,  # Confirmation secondaire
        ScoreCategory.SUPPORT_RESISTANCE: 0.05,  # Informatif, non bloquant
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
        """Calcule le score d'opportunité avec indicateurs scalping."""
        if not analyzer_data:
            logger.warning("calculate_opportunity_score: analyzer_data is empty")
            return self._create_zero_score("Pas de données")

        weights = self.DEFAULT_WEIGHTS
        category_scores = {}

        # 1. VWAP POSITION (25%) - LE PLUS INSTITUTIONNEL
        category_scores[ScoreCategory.VWAP_POSITION] = self._score_vwap_position(
            analyzer_data, current_price, weights[ScoreCategory.VWAP_POSITION]
        )

        # 2. EMA TREND (20%) - Court terme seulement
        category_scores[ScoreCategory.EMA_TREND] = self._score_ema_trend(
            analyzer_data, current_price, weights[ScoreCategory.EMA_TREND]
        )

        # 3. VOLUME (20%) - Confirmation essentielle
        category_scores[ScoreCategory.VOLUME] = self._score_volume_scalping(
            analyzer_data, weights[ScoreCategory.VOLUME]
        )

        # 4. RSI (15%) - Simple et efficace
        category_scores[ScoreCategory.RSI_MOMENTUM] = self._score_rsi_scalping(
            analyzer_data, weights[ScoreCategory.RSI_MOMENTUM]
        )

        # 5. BOLLINGER BANDS (10%)
        category_scores[ScoreCategory.BOLLINGER] = self._score_bollinger(
            analyzer_data, current_price, weights[ScoreCategory.BOLLINGER]
        )

        # 6. MACD (5%) - Confirmation
        category_scores[ScoreCategory.MACD] = self._score_macd(
            analyzer_data, weights[ScoreCategory.MACD]
        )

        # 7. SUPPORT/RESISTANCE (5%) - Non bloquant
        category_scores[ScoreCategory.SUPPORT_RESISTANCE] = self._score_sr_simple(
            analyzer_data, current_price, weights[ScoreCategory.SUPPORT_RESISTANCE]
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

    def _score_vwap_position(self, ad: dict, current_price: float, weight: float) -> CategoryScore:
        """
        Score VWAP - L'indicateur #1 des traders institutionnels.

        Prix > VWAP = Force acheteur institutionnel
        Prix < VWAP = Faiblesse
        """
        details: dict[str, float] = {}
        issues: list[str] = []
        score = 0.0

        vwap = self.safe_float(ad.get("vwap_quote_10"))  # Quote VWAP plus précis
        if vwap == 0:
            vwap = self.safe_float(ad.get("vwap_10"))

        if vwap == 0 or current_price == 0:
            issues.append("VWAP indisponible")
            return CategoryScore(
                category=ScoreCategory.VWAP_POSITION,
                score=50.0,  # Neutre si pas de VWAP
                weight=weight,
                weighted_score=50.0 * weight,
                details={"vwap_missing": True},
                confidence=0.0,
                issues=issues,
            )

        # Distance vs VWAP en %
        vwap_dist_pct = ((current_price - vwap) / vwap) * 100
        details["vwap_distance_pct"] = vwap_dist_pct

        # SCALPING: On veut prix proche ou au-dessus VWAP
        if vwap_dist_pct > 0.5:  # >0.5% au dessus
            score = 100.0
            issues.append(f"✅ Prix {vwap_dist_pct:.2f}% AU DESSUS VWAP - Force acheteur institutionnelle")
        elif vwap_dist_pct > 0.2:  # Légèrement au dessus
            score = 85.0
            issues.append(f"✅ Prix au dessus VWAP (+{vwap_dist_pct:.2f}%)")
        elif vwap_dist_pct > -0.1:  # Proche du VWAP
            score = 70.0
            issues.append("Prix proche VWAP - zone neutre")
        elif vwap_dist_pct > -0.5:  # Légèrement en dessous
            score = 50.0
            issues.append(f"⚠️ Prix sous VWAP ({vwap_dist_pct:.2f}%) - faiblesse modérée")
        else:  # Trop en dessous
            score = 30.0
            issues.append(f"⚠️⚠️ Prix très sous VWAP ({vwap_dist_pct:.2f}%) - faiblesse institutionnelle")

        details["vwap_score"] = score

        confidence = 100.0  # VWAP toujours fiable

        return CategoryScore(
            category=ScoreCategory.VWAP_POSITION,
            score=score,
            weight=weight,
            weighted_score=score * weight,
            details=details,
            confidence=confidence,
            issues=issues,
        )

    def _score_ema_trend(self, ad: dict, current_price: float, weight: float) -> CategoryScore:
        """
        Score EMA Trend - COURT TERME seulement (7/12/26).

        Scalping = EMA rapides, pas 50/99
        """
        details: dict[str, float] = {}
        issues: list[str] = []
        score = 0.0

        ema7 = self.safe_float(ad.get("ema_7"))
        ema12 = self.safe_float(ad.get("ema_12"))
        ema26 = self.safe_float(ad.get("ema_26"))

        if ema7 == 0 or ema12 == 0 or current_price == 0:
            issues.append("EMA manquantes")
            return CategoryScore(
                category=ScoreCategory.EMA_TREND,
                score=50.0,
                weight=weight,
                weighted_score=50.0 * weight,
                details={},
                confidence=0.0,
                issues=issues,
            )

        # 1. Position prix vs EMAs (40 points)
        above_ema7 = current_price > ema7
        above_ema12 = current_price > ema12
        above_ema26 = current_price > ema26

        if above_ema7 and above_ema12 and above_ema26:
            price_score = 40.0
            issues.append("✅ Prix au dessus des 3 EMAs - Trend bull confirmé")
        elif above_ema7 and above_ema12:
            price_score = 30.0
            issues.append("Prix au dessus EMA7/12 - Trend court terme bull")
        elif above_ema7:
            price_score = 20.0
            issues.append("Prix au dessus EMA7 seulement - Trend très court terme")
        else:
            price_score = 10.0
            issues.append("⚠️ Prix sous EMAs - Trend bear/neutre")

        score += price_score
        details["price_vs_emas"] = price_score

        # 2. EMA alignment (30 points) - Golden cross
        ema7_above_12 = ema7 > ema12
        ema12_above_26 = ema12 > ema26

        if ema7_above_12 and ema12_above_26:
            alignment_score = 30.0
            issues.append("✅ EMAs alignées (7>12>26) - Golden cross")
        elif ema7_above_12:
            alignment_score = 20.0
            issues.append("EMA7 > EMA12 - Début hausse")
        else:
            alignment_score = 5.0
            issues.append("⚠️ EMAs désalignées - Pas de trend clair")

        score += alignment_score
        details["ema_alignment"] = alignment_score

        # 3. ADX pour force du trend (30 points)
        adx = self.safe_float(ad.get("adx_14"))
        if adx > 0:
            if adx > 40:
                adx_score = 30.0
                issues.append(f"ADX {adx:.0f} - Trend très fort")
            elif adx > 30:
                adx_score = 25.0
                issues.append(f"ADX {adx:.0f} - Trend fort")
            elif adx > 25:
                adx_score = 20.0
            elif adx > 20:
                adx_score = 10.0
            else:
                adx_score = 5.0
                issues.append(f"⚠️ ADX {adx:.0f} - Trend faible")

            score += adx_score
            details["adx"] = adx_score

        confidence = 90.0 if len(details) >= 3 else 60.0

        return CategoryScore(
            category=ScoreCategory.EMA_TREND,
            score=min(score, 100),
            weight=weight,
            weighted_score=min(score, 100) * weight,
            details=details,
            confidence=confidence,
            issues=issues,
        )

    def _score_volume_scalping(self, ad: dict, weight: float) -> CategoryScore:
        """Score VOLUME - Confirmation essentielle pour scalping."""
        details: dict[str, float] = {}
        issues: list[str] = []
        score = 0.0

        rel_volume = self.safe_float(ad.get("relative_volume"), 1.0)

        # SCALPING: Volume >0.8x acceptable, >1.2x bon
        if rel_volume > 2.0:
            vol_score = 100.0
            issues.append(f"✅ Volume fort {rel_volume:.1f}x - Excellente confirmation")
        elif rel_volume > 1.5:
            vol_score = 90.0
            issues.append(f"✅ Volume élevé {rel_volume:.1f}x")
        elif rel_volume > 1.2:
            vol_score = 75.0
            issues.append(f"Volume correct {rel_volume:.1f}x")
        elif rel_volume > 0.8:
            vol_score = 60.0
            issues.append(f"Volume moyen {rel_volume:.1f}x - acceptable")
        elif rel_volume > 0.5:
            vol_score = 40.0
            issues.append(f"⚠️ Volume faible {rel_volume:.1f}x")
        else:
            vol_score = 20.0
            issues.append(f"⚠️⚠️ Volume très faible {rel_volume:.1f}x")

        score = vol_score
        details["relative_volume"] = vol_score

        confidence = 100.0

        return CategoryScore(
            category=ScoreCategory.VOLUME,
            score=score,
            weight=weight,
            weighted_score=score * weight,
            details=details,
            confidence=confidence,
            issues=issues,
        )

    def _score_rsi_scalping(self, ad: dict, weight: float) -> CategoryScore:
        """Score RSI - Simple et réaliste pour scalping."""
        details: dict[str, float] = {}
        issues: list[str] = []
        score = 0.0

        rsi = self.safe_float(ad.get("rsi_14"))

        if rsi == 0:
            return CategoryScore(
                category=ScoreCategory.RSI_MOMENTUM,
                score=50.0,
                weight=weight,
                weighted_score=50.0 * weight,
                details={},
                confidence=0.0,
                issues=["RSI indisponible"],
            )

        # SCALPING RÉALISTE: 30-70 acceptable, 35-55 optimal
        if 40 <= rsi <= 55:
            score = 100.0
            issues.append(f"✅ RSI {rsi:.0f} - Zone optimale pour achat")
        elif 35 <= rsi < 40:
            score = 90.0
            issues.append(f"✅ RSI {rsi:.0f} - Sortie oversold, excellent")
        elif 30 <= rsi < 35:
            score = 80.0
            issues.append(f"RSI {rsi:.0f} - Oversold, bon potentiel")
        elif 55 < rsi <= 65:
            score = 70.0
            issues.append(f"RSI {rsi:.0f} - Début hausse, acceptable")
        elif 65 < rsi <= 75:
            score = 50.0
            issues.append(f"⚠️ RSI {rsi:.0f} - Élevé mais acceptable en trend fort")
        elif rsi > 75:
            score = 30.0
            issues.append(f"⚠️⚠️ RSI {rsi:.0f} - Overbought, risque correction")
        else:  # < 30
            score = 60.0
            issues.append(f"RSI {rsi:.0f} - Deep oversold, attendre rebond")

        details["rsi"] = score
        confidence = 100.0

        return CategoryScore(
            category=ScoreCategory.RSI_MOMENTUM,
            score=score,
            weight=weight,
            weighted_score=score * weight,
            details=details,
            confidence=confidence,
            issues=issues,
        )

    def _score_bollinger(self, ad: dict, current_price: float, weight: float) -> CategoryScore:  # noqa: ARG002
        """Score Bollinger Bands - Squeeze et position."""
        details: dict[str, float] = {}
        issues: list[str] = []
        score = 50.0  # Neutre par défaut

        bb_squeeze = ad.get("bb_squeeze", False)
        bb_expansion = ad.get("bb_expansion", False)
        bb_position = self.safe_float(ad.get("bb_position"))

        # Squeeze = volatilité compressée = breakout imminent
        if bb_squeeze:
            score += 30.0
            issues.append("✅ BB SQUEEZE - Breakout imminent!")
            details["squeeze"] = True

        if bb_expansion:
            score += 10.0
            details["expansion"] = True

        # Position dans les bandes
        if bb_position != 0:
            if -0.2 <= bb_position <= 0.2:  # Proche milieu
                details["position"] = "middle"
            elif bb_position < -0.5:  # Proche bande basse
                score += 10.0
                issues.append("Prix proche bande basse - rebond potentiel")
                details["position"] = "lower"
            elif bb_position > 0.5:  # Proche bande haute
                details["position"] = "upper"

        confidence = 80.0 if bb_squeeze or bb_expansion else 50.0

        return CategoryScore(
            category=ScoreCategory.BOLLINGER,
            score=min(score, 100),
            weight=weight,
            weighted_score=min(score, 100) * weight,
            details=details,
            confidence=confidence,
            issues=issues,
        )

    def _score_macd(self, ad: dict, weight: float) -> CategoryScore:
        """Score MACD - Confirmation secondaire."""
        details: dict[str, float] = {}
        issues: list[str] = []
        score = 50.0

        macd_trend = ad.get("macd_trend", "").upper()
        macd_hist = self.safe_float(ad.get("macd_histogram"))

        if macd_trend == "BULLISH":
            score = 80.0
            if macd_hist > 0:
                score = 90.0
                issues.append("✅ MACD bullish avec histogram positif")
            else:
                issues.append("MACD bullish")
        elif macd_trend == "NEUTRAL":
            score = 60.0
        else:  # BEARISH
            score = 40.0
            issues.append("⚠️ MACD bearish - attendre retournement")

        details["macd"] = score
        confidence = 70.0

        return CategoryScore(
            category=ScoreCategory.MACD,
            score=score,
            weight=weight,
            weighted_score=score * weight,
            details=details,
            confidence=confidence,
            issues=issues,
        )

    def _score_sr_simple(self, ad: dict, current_price: float, weight: float) -> CategoryScore:
        """
        Score S/R - SIMPLIFIÉ et NON BLOQUANT.

        Juste informatif, ne bloque jamais.
        """
        details: dict[str, float] = {}
        issues: list[str] = []
        score = 70.0  # Score par défaut acceptable

        nearest_resistance = self.safe_float(ad.get("nearest_resistance"))
        nearest_support = self.safe_float(ad.get("nearest_support"))

        if nearest_resistance > 0 and current_price > 0:
            res_dist_pct = ((nearest_resistance - current_price) / current_price) * 100

            if res_dist_pct > 3.0:
                score = 100.0
                issues.append(f"✅ Résistance loin ({res_dist_pct:.1f}%) - Espace libre")
            elif res_dist_pct > 1.5:
                score = 80.0
                issues.append(f"Résistance à {res_dist_pct:.1f}%")
            else:
                score = 60.0
                issues.append(f"ℹ️ Résistance proche ({res_dist_pct:.1f}%) - Potentiel breakout")

            details["resistance_distance_pct"] = res_dist_pct

        if nearest_support > 0 and current_price > nearest_support:
            sup_dist_pct = ((current_price - nearest_support) / current_price) * 100
            if sup_dist_pct < 2.0:
                issues.append(f"Support proche ({sup_dist_pct:.1f}%) - Filet sécurité")
                details["support_distance_pct"] = sup_dist_pct

        confidence = 60.0

        return CategoryScore(
            category=ScoreCategory.SUPPORT_RESISTANCE,
            score=score,
            weight=weight,
            weighted_score=score * weight,
            details=details,
            confidence=confidence,
            issues=issues,
        )

    def _calculate_grade(self, score: float) -> str:
        """Calcule le grade S/A/B/C/D/F."""
        if score >= 85:
            return "S"
        if score >= 75:
            return "A"
        if score >= 65:
            return "B"
        if score >= 55:
            return "C"
        if score >= 45:
            return "D"
        return "F"

    def _calculate_risk_level(self, ad: dict, score: float) -> str:
        """Calcule le niveau de risque."""
        vol_regime = ad.get("volatility_regime", "").lower()

        if vol_regime == "extreme":
            return "HIGH"

        if score >= 75:
            return "LOW"
        if score >= 60:
            return "MEDIUM"
        return "HIGH"

    def _calculate_recommendation(
        self, score: float, confidence: float, category_scores: dict, ad: dict  # noqa: ARG002
    ) -> tuple[str, list[str], list[str]]:
        """Calcule la recommandation finale."""
        reasons: list[str] = []
        warnings: list[str] = []

        # BUY_NOW: Score >70, confiance >60
        if score >= 70 and confidence >= 60:
            reasons.append(f"✅ Score excellent: {score:.0f}/100 (Grade {self._calculate_grade(score)})")
            reasons.append("Tous les indicateurs institutionnels alignés")
            return "BUY_NOW", reasons, warnings

        # BUY_DCA: Score 60-70
        if score >= 60:
            reasons.append(f"Score bon: {score:.0f}/100 (Grade {self._calculate_grade(score)})")
            reasons.append("Entrée progressive recommandée")
            return "BUY_DCA", reasons, warnings

        # WAIT: Score 50-60
        if score >= 50:
            reasons.append(f"Score moyen: {score:.0f}/100")
            warnings.append("Attendre meilleure configuration")
            return "WAIT", reasons, warnings

        # AVOID: Score <50
        reasons.append(f"Score faible: {score:.0f}/100")
        reasons.append("Conditions non favorables")
        return "AVOID", reasons, warnings

    def _create_zero_score(self, reason: str) -> OpportunityScore:
        """Crée un score de 0 avec raison."""
        zero_category = CategoryScore(
            category=ScoreCategory.VWAP_POSITION,
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
