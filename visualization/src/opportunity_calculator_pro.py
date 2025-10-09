"""
Opportunity Calculator PRO - Professional Trading System
Orchestre scoring + validation pour décisions de trading optimales

Architecture:
1. OpportunityScoring: Calcule score 0-100 sur 7 catégories (tous les 108 indicateurs)
2. OpportunityValidator: Validation 4 niveaux (data/market/risk/timing)
3. Décision finale: BUY_NOW / BUY_DCA / WAIT / AVOID
4. Gestion risque intelligente: TP/SL adaptatifs, taille position optimale

Version: 2.0 - Professional Grade (Real Money Ready)
"""
import logging
from typing import Optional, Dict
from dataclasses import dataclass, asdict

from opportunity_scoring import OpportunityScoring, OpportunityScore
from opportunity_validator import OpportunityValidator, ValidationSummary

logger = logging.getLogger(__name__)


@dataclass
class TradingOpportunity:
    """Opportunité de trading complète."""
    symbol: str
    action: str  # BUY_NOW, BUY_DCA, WAIT, AVOID
    confidence: float  # 0-100
    score: OpportunityScore
    validation: ValidationSummary

    # Pricing
    current_price: float
    entry_price_optimal: float  # Prix d'entrée optimal (pullback)
    entry_price_aggressive: float  # Prix d'entrée aggressif (market)

    # Targets
    tp1: float
    tp1_percent: float
    tp2: float
    tp2_percent: float
    tp3: Optional[float]
    tp3_percent: Optional[float]

    # Stop Loss
    stop_loss: float
    stop_loss_percent: float
    stop_loss_basis: str  # 'support', 'ATR', 'BB'

    # Risk Management
    rr_ratio: float
    risk_level: str  # LOW, MEDIUM, HIGH, EXTREME
    max_position_size_pct: float  # % du capital à risquer

    # Timing
    estimated_hold_time: str
    entry_urgency: str  # IMMEDIATE, SOON, PATIENT, NO_RUSH

    # Context
    market_regime: str
    volume_context: str
    volatility_regime: str

    # Reasons
    reasons: list[str]
    warnings: list[str]
    recommendations: list[str]

    # Raw data for debugging
    raw_score_details: dict
    raw_validation_details: dict
    raw_analyzer_data: dict  # Données analyzer complètes pour frontend


class OpportunityCalculatorPro:
    """
    Calculateur professionnel d'opportunités.

    Combine scoring multi-niveaux + validation stricte + gestion risque intelligente.
    Utilise TOUS les 108 indicateurs disponibles dans analyzer_data.
    """

    def __init__(self):
        """Initialise le calculateur professionnel."""
        self.scorer = OpportunityScoring()
        self.validator = OpportunityValidator()

    @staticmethod
    def safe_float(value, default=0.0):
        """Convertir en float avec fallback."""
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    def calculate_opportunity(
        self,
        symbol: str,
        current_price: float,
        analyzer_data: dict,
        higher_tf_data: Optional[dict] = None,
        signals_data: Optional[dict] = None
    ) -> TradingOpportunity:
        """
        Calcule une opportunité de trading complète.

        Args:
            symbol: Symbole (e.g. BTCUSDC)
            current_price: Prix actuel
            analyzer_data: analyzer_data complet (108 indicateurs)
            higher_tf_data: Données timeframe supérieur pour validation
            signals_data: (optionnel) Données signaux externes

        Returns:
            TradingOpportunity complète avec action, TP/SL, sizing, etc.
        """
        if not analyzer_data:
            return self._create_no_data_opportunity(symbol, current_price)

        # === ÉTAPE 1: SCORING ===
        score = self.scorer.calculate_opportunity_score(analyzer_data)

        # === ÉTAPE 2: VALIDATION ===
        validation = self.validator.validate_opportunity(
            analyzer_data, current_price, higher_tf_data
        )

        # === ÉTAPE 3: DÉCISION ===
        action, confidence, reasons, warnings, recommendations = self._make_decision(
            score, validation, analyzer_data
        )

        # === ÉTAPE 4: PRICING ===
        entry_optimal, entry_aggressive = self._calculate_entry_prices(
            current_price, analyzer_data, action
        )

        # === ÉTAPE 5: TARGETS ===
        tp1, tp2, tp3, tp_percents = self._calculate_targets(
            current_price, analyzer_data, score
        )

        # === ÉTAPE 6: STOP LOSS ===
        stop_loss, sl_percent, sl_basis = self._calculate_stop_loss(
            current_price, analyzer_data
        )

        # === ÉTAPE 7: RISK MANAGEMENT ===
        rr_ratio, risk_level, max_position_pct = self._calculate_risk_metrics(
            current_price, tp1, stop_loss, score, validation, analyzer_data
        )

        # === ÉTAPE 8: TIMING ===
        hold_time, urgency = self._calculate_timing(analyzer_data, score, action)

        # === ÉTAPE 9: CONTEXT ===
        market_regime = analyzer_data.get('market_regime', 'UNKNOWN')
        volume_context = analyzer_data.get('volume_context', 'UNKNOWN')
        volatility_regime = analyzer_data.get('volatility_regime', 'unknown')

        return TradingOpportunity(
            symbol=symbol,
            action=action,
            confidence=confidence,
            score=score,
            validation=validation,
            current_price=current_price,
            entry_price_optimal=entry_optimal,
            entry_price_aggressive=entry_aggressive,
            tp1=tp1,
            tp1_percent=tp_percents[0],
            tp2=tp2,
            tp2_percent=tp_percents[1],
            tp3=tp3,
            tp3_percent=tp_percents[2] if tp3 else None,
            stop_loss=stop_loss,
            stop_loss_percent=sl_percent,
            stop_loss_basis=sl_basis,
            rr_ratio=rr_ratio,
            risk_level=risk_level,
            max_position_size_pct=max_position_pct,
            estimated_hold_time=hold_time,
            entry_urgency=urgency,
            market_regime=market_regime,
            volume_context=volume_context,
            volatility_regime=volatility_regime,
            reasons=reasons,
            warnings=warnings,
            recommendations=recommendations,
            raw_score_details=self._serialize_score(score),
            raw_validation_details=self._serialize_validation(validation),
            raw_analyzer_data=analyzer_data  # Store analyzer_data for frontend
        )

    def _make_decision(
        self,
        score: OpportunityScore,
        validation: ValidationSummary,
        ad: dict
    ) -> tuple[str, float, list[str], list[str], list[str]]:
        """
        Décision finale combinant score + validation.

        Logique:
        - BUY_NOW: Score grade A/S + validation passée + timing urgent
        - BUY_DCA: Score grade B+ + validation passée + timing patient
        - WAIT: Score OK mais validation partielle ou timing mauvais
        - AVOID: Score faible ou validation échouée
        """
        reasons = []
        warnings = []
        recommendations = []

        # Si validation échouée = AVOID automatique
        if not validation.all_passed:
            action = 'AVOID'
            confidence = 0.0
            reasons.append(f"❌ Validation échouée: {len(validation.blocking_issues)} problèmes")
            reasons.extend(validation.blocking_issues)
            warnings.extend(validation.warnings)
            recommendations.append("Attendre résolution des problèmes bloquants")
            return action, confidence, reasons, warnings, recommendations

        # Si validation passée, décision basée sur score
        total_score = score.total_score
        score_confidence = score.confidence

        # BUY_NOW: Score S/A (>80) + confiance >70
        if score.grade in ['S', 'A'] and score_confidence >= 70:
            action = 'BUY_NOW'
            confidence = min(total_score, score_confidence)

            reasons.append(f"✅ Score excellent: {total_score:.0f}/100 (Grade {score.grade})")
            reasons.append(f"✅ Confiance élevée: {score_confidence:.0f}%")
            reasons.append(f"✅ Validation passée: {validation.overall_score:.0f}/100")

            # Ajouter détails des meilleures catégories
            from opportunity_scoring import ScoreCategory
            top_cats = sorted(
                score.category_scores.items(),
                key=lambda x: x[1].score,
                reverse=True
            )[:3]

            for cat, cat_score in top_cats:
                if cat_score.score >= 70:
                    reasons.append(f"  • {cat.value.title()}: {cat_score.score:.0f}/100")

            recommendations.append("🚀 ACHETER MAINTENANT - Setup optimal")
            recommendations.extend(score.reasons)

        # BUY_DCA: Score B/C (60-80) + confiance >60
        elif score.grade in ['B', 'C'] and score_confidence >= 60:
            action = 'BUY_DCA'
            confidence = min(total_score * 0.85, score_confidence)

            reasons.append(f"✅ Score bon: {total_score:.0f}/100 (Grade {score.grade})")
            reasons.append(f"✅ Validation passée: {validation.overall_score:.0f}/100")
            reasons.append("⚠️ Entrée progressive recommandée")

            recommendations.append("📊 ACHETER EN DCA - Diviser en 2-3 tranches")
            recommendations.append("Zone d'achat: entry_optimal → entry_aggressive")

            # Ajouter warnings des catégories faibles
            from opportunity_scoring import ScoreCategory
            for cat, cat_score in score.category_scores.items():
                if cat_score.score < 60:
                    warnings.append(f"⚠️ {cat.value.title()} faible: {cat_score.score:.0f}/100")

        # WAIT: Score D ou confiance faible
        elif score.grade == 'D' or score_confidence < 60:
            action = 'WAIT'
            confidence = max(total_score * 0.7, 40.0)

            reasons.append(f"⏸️ Score moyen: {total_score:.0f}/100 (Grade {score.grade})")
            reasons.append(f"⚠️ Confiance insuffisante: {score_confidence:.0f}%")

            recommendations.append("⏸️ ATTENDRE - Conditions pas optimales")
            recommendations.append("Surveiller amélioration du score")

            # Lister ce qui manque
            from opportunity_scoring import ScoreCategory
            weak_cats = [
                (cat, cs) for cat, cs in score.category_scores.items()
                if cs.score < 50
            ]

            for cat, cat_score in weak_cats:
                warnings.append(f"❌ {cat.value.title()}: {cat_score.score:.0f}/100")
                if cat_score.issues:
                    for issue in cat_score.issues[:2]:  # Top 2 issues
                        warnings.append(f"   {issue}")

        # AVOID: Score F
        else:
            action = 'AVOID'
            confidence = max(total_score * 0.5, 20.0)

            reasons.append(f"❌ Score faible: {total_score:.0f}/100 (Grade {score.grade})")
            reasons.append("❌ Setup non favorable")

            recommendations.append("🛑 NE PAS ACHETER - Risque élevé")
            recommendations.extend(score.warnings)

        # Ajouter warnings du validation
        warnings.extend(validation.warnings)

        return action, confidence, reasons, warnings, recommendations

    def _calculate_entry_prices(
        self, current_price: float, ad: dict, action: str
    ) -> tuple[float, float]:
        """
        Calcule prix d'entrée optimal et aggressif.

        Optimal: Pullback vers support/EMA/VWAP
        Aggressive: Market price immédiat
        """
        # Entry aggressive = current price
        entry_aggressive = current_price

        # Entry optimal = pullback vers support technique
        ema_7 = self.safe_float(ad.get('ema_7'))
        vwap = self.safe_float(ad.get('vwap_quote_10')) or self.safe_float(ad.get('vwap_10'))
        nearest_support = self.safe_float(ad.get('nearest_support'))
        bb_lower = self.safe_float(ad.get('bb_lower'))

        # Choisir le niveau de pullback le plus proche sous le prix actuel
        pullback_levels = []

        if ema_7 > 0 and ema_7 < current_price:
            pullback_levels.append(ema_7)

        if vwap > 0 and vwap < current_price:
            pullback_levels.append(vwap)

        if nearest_support > 0 and nearest_support < current_price:
            # Légèrement au-dessus du support
            pullback_levels.append(nearest_support * 1.001)

        if bb_lower > 0 and bb_lower < current_price:
            pullback_levels.append(bb_lower)

        if pullback_levels:
            # Prendre le plus haut (closest to current price)
            entry_optimal = max(pullback_levels)
        else:
            # Pas de niveau identifié, prendre 0.3% sous le prix
            entry_optimal = current_price * 0.997

        return round(entry_optimal, 8), round(entry_aggressive, 8)

    def _calculate_targets(
        self, current_price: float, ad: dict, score: OpportunityScore
    ) -> tuple[float, float, Optional[float], list[float]]:
        """
        Calcule targets adaptatifs basés sur ATR + résistances + score.

        TP1: Conservative (0.8-1.0 ATR)
        TP2: Moderate (1.2-1.5 ATR)
        TP3: Aggressive (2.0-2.5 ATR) - Seulement si score S/A
        """
        atr = self.safe_float(ad.get('atr_14'))
        natr = self.safe_float(ad.get('natr'))
        nearest_resistance = self.safe_float(ad.get('nearest_resistance'))

        # Calculer ATR%
        if atr > 0 and current_price > 0:
            atr_percent = atr / current_price
        elif natr > 0:
            atr_percent = natr / 100.0
        else:
            atr_percent = 0.015  # Fallback 1.5%

        # TP1: Conservative
        tp1_dist = max(0.01, atr_percent * 0.8)

        # TP2: Moderate
        tp2_dist = max(0.015, atr_percent * 1.2)

        # TP3: Aggressive (seulement si score excellent)
        if score.grade in ['S', 'A']:
            tp3_dist = max(0.020, atr_percent * 2.0)
        else:
            tp3_dist = None

        # Ajuster si résistance proche
        if nearest_resistance > 0 and current_price > 0:
            res_dist_pct = (nearest_resistance - current_price) / current_price

            # Si résistance entre prix et TP1, ajuster TP1 juste avant
            if res_dist_pct < tp1_dist:
                tp1_dist = res_dist_pct * 0.95

            # Si résistance entre TP1 et TP2, ajuster TP2 juste avant
            if res_dist_pct < tp2_dist:
                tp2_dist = res_dist_pct * 0.98

        tp1 = round(current_price * (1 + tp1_dist), 8)
        tp2 = round(current_price * (1 + tp2_dist), 8)
        tp3 = round(current_price * (1 + tp3_dist), 8) if tp3_dist else None

        tp_percents = [
            round(tp1_dist * 100, 2),
            round(tp2_dist * 100, 2),
            round(tp3_dist * 100, 2) if tp3_dist else None
        ]

        return tp1, tp2, tp3, tp_percents

    def _calculate_stop_loss(
        self, current_price: float, ad: dict
    ) -> tuple[float, float, str]:
        """
        Calcule SL intelligent basé sur:
        1. Support proche (priorité)
        2. ATR (0.7x)
        3. Bollinger Lower
        """
        nearest_support = self.safe_float(ad.get('nearest_support'))
        atr = self.safe_float(ad.get('atr_14'))
        natr = self.safe_float(ad.get('natr'))
        bb_lower = self.safe_float(ad.get('bb_lower'))

        # Calculer ATR%
        if atr > 0 and current_price > 0:
            atr_percent = atr / current_price
        elif natr > 0:
            atr_percent = natr / 100.0
        else:
            atr_percent = 0.012  # Fallback 1.2%

        # Méthode 1: Support
        if nearest_support > 0 and nearest_support < current_price:
            sl_dist = (current_price - nearest_support) / current_price
            # Minimum 0.7%, maximum 2%
            sl_dist = max(0.007, min(sl_dist, 0.02))
            sl_basis = "support"

        # Méthode 2: ATR
        elif atr > 0:
            sl_dist = max(0.007, atr_percent * 0.7)
            sl_basis = "ATR"

        # Méthode 3: Bollinger Lower
        elif bb_lower > 0 and bb_lower < current_price:
            sl_dist = (current_price - bb_lower) / current_price
            sl_dist = max(0.007, min(sl_dist, 0.02))
            sl_basis = "BB_lower"

        # Fallback: 1.2%
        else:
            sl_dist = 0.012
            sl_basis = "fixed"

        stop_loss = round(current_price * (1 - sl_dist), 8)
        sl_percent = round(sl_dist * 100, 2)

        return stop_loss, sl_percent, sl_basis

    def _calculate_risk_metrics(
        self,
        current_price: float,
        tp1: float,
        stop_loss: float,
        score: OpportunityScore,
        validation: ValidationSummary,
        ad: dict
    ) -> tuple[float, str, float]:
        """
        Calcule métriques de risque + position sizing.

        Returns:
            (rr_ratio, risk_level, max_position_size_pct)
        """
        # R/R Ratio
        tp_dist = abs(tp1 - current_price)
        sl_dist = abs(current_price - stop_loss)

        rr_ratio = (tp_dist / sl_dist) if sl_dist > 0 else 0.0

        # Risk Level (déjà calculé dans score)
        risk_level = score.risk_level

        # Max Position Size (% du capital)
        # Basé sur score + volatilité + risk_level
        vol_regime = ad.get('volatility_regime', '').lower()

        # Base sizing
        if score.grade == 'S' and risk_level == 'LOW':
            base_pct = 8.0  # 8% capital
        elif score.grade == 'A' and risk_level in ['LOW', 'MEDIUM']:
            base_pct = 6.0
        elif score.grade in ['A', 'B'] and risk_level == 'MEDIUM':
            base_pct = 4.0
        elif score.grade in ['B', 'C']:
            base_pct = 3.0
        else:
            base_pct = 2.0  # Minimum

        # Ajuster par volatilité
        if vol_regime == 'extreme':
            base_pct *= 0.5
        elif vol_regime == 'high':
            base_pct *= 0.75
        elif vol_regime == 'low':
            base_pct *= 1.2

        # Ajuster par R/R
        if rr_ratio > 3.0:
            base_pct *= 1.2
        elif rr_ratio < 1.8:
            base_pct *= 0.8

        max_position_pct = round(min(base_pct, 10.0), 2)  # Cap à 10%

        return round(rr_ratio, 2), risk_level, max_position_pct

    def _calculate_timing(
        self, ad: dict, score: OpportunityScore, action: str
    ) -> tuple[str, str]:
        """
        Calcule timing: durée hold estimée + urgence d'entrée.
        """
        regime = ad.get('market_regime', '').upper()
        adx = self.safe_float(ad.get('adx_14'))
        vol_regime = ad.get('volatility_regime', '').lower()

        # Durée hold
        if regime == 'BREAKOUT_BULL':
            hold_time = "5-15 min"
        elif regime == 'TRENDING_BULL':
            if adx > 35:
                hold_time = "10-20 min"
            else:
                hold_time = "15-30 min"
        elif regime in ['RANGING', 'TRANSITION']:
            hold_time = "20-45 min"
        else:
            hold_time = "15-30 min"

        # Urgence
        if action == 'BUY_NOW' and score.grade == 'S':
            urgency = 'IMMEDIATE'
        elif action == 'BUY_NOW':
            urgency = 'SOON'
        elif action == 'BUY_DCA':
            urgency = 'PATIENT'
        else:
            urgency = 'NO_RUSH'

        return hold_time, urgency

    def _serialize_score(self, score: OpportunityScore) -> dict:
        """Sérialise OpportunityScore pour export."""
        from opportunity_scoring import ScoreCategory

        return {
            'total_score': score.total_score,
            'grade': score.grade,
            'confidence': score.confidence,
            'risk_level': score.risk_level,
            'recommendation': score.recommendation,
            'category_scores': {
                cat.value: {
                    'score': cs.score,
                    'weight': cs.weight,
                    'weighted_score': cs.weighted_score,
                    'confidence': cs.confidence,
                    'details': cs.details,
                    'issues': cs.issues
                }
                for cat, cs in score.category_scores.items()
            }
        }

    def _serialize_validation(self, validation: ValidationSummary) -> dict:
        """Sérialise ValidationSummary pour export."""
        return {
            'all_passed': validation.all_passed,
            'overall_score': validation.overall_score,
            'level_results': {
                level.value: {
                    'passed': result.passed,
                    'score': result.score,
                    'reason': result.reason,
                    'details': result.details,
                    'warnings': result.warnings
                }
                for level, result in validation.level_results.items()
            },
            'blocking_issues': validation.blocking_issues,
            'warnings': validation.warnings
        }

    def _create_no_data_opportunity(self, symbol: str, current_price: float) -> TradingOpportunity:
        """Crée une opportunité vide (pas de données)."""
        from opportunity_scoring import OpportunityScore, CategoryScore, ScoreCategory
        from opportunity_validator import ValidationSummary, ValidationResult, ValidationLevel

        # Score vide
        zero_category = CategoryScore(
            category=ScoreCategory.TREND,
            score=0.0,
            weight=0.0,
            weighted_score=0.0,
            details={},
            confidence=0.0,
            issues=["Pas de données"]
        )

        score = OpportunityScore(
            total_score=0.0,
            grade='F',
            category_scores={cat: zero_category for cat in ScoreCategory},
            confidence=0.0,
            risk_level='EXTREME',
            recommendation='AVOID',
            reasons=["Pas de données"],
            warnings=[]
        )

        # Validation vide
        validation = ValidationSummary(
            all_passed=False,
            level_results={},
            overall_score=0.0,
            blocking_issues=["Pas de données analyzer_data"],
            warnings=[],
            recommendations=["Attendre disponibilité des données"]
        )

        return TradingOpportunity(
            symbol=symbol,
            action='AVOID',
            confidence=0.0,
            score=score,
            validation=validation,
            current_price=current_price,
            entry_price_optimal=current_price,
            entry_price_aggressive=current_price,
            tp1=current_price * 1.01,
            tp1_percent=1.0,
            tp2=current_price * 1.015,
            tp2_percent=1.5,
            tp3=None,
            tp3_percent=None,
            stop_loss=current_price * 0.988,
            stop_loss_percent=1.2,
            stop_loss_basis='fixed',
            rr_ratio=0.0,
            risk_level='EXTREME',
            max_position_size_pct=0.0,
            estimated_hold_time='N/A',
            entry_urgency='NO_RUSH',
            market_regime='UNKNOWN',
            volume_context='UNKNOWN',
            volatility_regime='unknown',
            reasons=["Pas de données analyzer_data"],
            warnings=[],
            recommendations=["Attendre disponibilité des données"],
            raw_score_details={},
            raw_validation_details={},
            raw_analyzer_data={}  # Empty analyzer data
        )

    def to_dict(self, opportunity: TradingOpportunity) -> dict:
        """Convertit TradingOpportunity en dict pour export/API."""
        return {
            'symbol': opportunity.symbol,
            'action': opportunity.action,
            'confidence': opportunity.confidence,

            'pricing': {
                'current_price': opportunity.current_price,
                'entry_optimal': opportunity.entry_price_optimal,
                'entry_aggressive': opportunity.entry_price_aggressive,
            },

            'targets': {
                'tp1': {'price': opportunity.tp1, 'percent': opportunity.tp1_percent},
                'tp2': {'price': opportunity.tp2, 'percent': opportunity.tp2_percent},
                'tp3': {'price': opportunity.tp3, 'percent': opportunity.tp3_percent} if opportunity.tp3 else None,
            },

            'stop_loss': {
                'price': opportunity.stop_loss,
                'percent': opportunity.stop_loss_percent,
                'basis': opportunity.stop_loss_basis
            },

            'risk': {
                'rr_ratio': opportunity.rr_ratio,
                'risk_level': opportunity.risk_level,
                'max_position_size_pct': opportunity.max_position_size_pct
            },

            'timing': {
                'estimated_hold_time': opportunity.estimated_hold_time,
                'entry_urgency': opportunity.entry_urgency
            },

            'context': {
                'market_regime': opportunity.market_regime,
                'volume_context': opportunity.volume_context,
                'volatility_regime': opportunity.volatility_regime
            },

            'score': {
                'total': opportunity.score.total_score,
                'grade': opportunity.score.grade,
                'confidence': opportunity.score.confidence
            },

            'validation': {
                'all_passed': opportunity.validation.all_passed,
                'overall_score': opportunity.validation.overall_score
            },

            'reasons': opportunity.reasons,
            'warnings': opportunity.warnings,
            'recommendations': opportunity.recommendations,

            # Debug data
            'debug': {
                'score_details': opportunity.raw_score_details,
                'validation_details': opportunity.raw_validation_details,
                'raw_data': opportunity.raw_analyzer_data  # Données analyzer pour frontend
            }
        }


# ===========================================================
# EXEMPLE D'UTILISATION
# ===========================================================
if __name__ == "__main__":
    import json

    print("="*80)
    print("OPPORTUNITY CALCULATOR PRO - Test Example")
    print("="*80)

    # Simuler des données analyzer_data complètes
    test_data = {
        # Trend indicators
        'adx_14': 32.5,
        'plus_di': 35.2,
        'minus_di': 18.3,
        'trend_alignment': 75.0,
        'trend_strength': 'STRONG',
        'directional_bias': 'BULLISH',
        'market_regime': 'TRENDING_BULL',
        'regime_confidence': 82.0,
        'regime_strength': 'STRONG',

        # Momentum indicators
        'rsi_14': 62.5,
        'rsi_21': 64.0,
        'williams_r': -28.0,
        'macd_trend': 'BULLISH',
        'macd_histogram': 0.0015,
        'macd_signal_cross': True,
        'cci_20': 85.0,
        'mfi_14': 58.0,
        'stoch_k': 68.0,
        'stoch_d': 65.0,
        'stoch_signal': 'BULLISH',
        'momentum_score': 72.0,

        # Volume indicators
        'relative_volume': 2.3,
        'volume_spike_multiplier': 2.8,
        'volume_context': 'BREAKOUT',
        'volume_pattern': 'SPIKE',
        'volume_quality_score': 78.0,
        'obv_oscillator': 250.0,
        'trade_intensity': 1.8,
        'volume_buildup_periods': 3,

        # Volatility
        'atr_14': 0.0018,
        'natr': 1.8,
        'atr_percentile': 55.0,
        'volatility_regime': 'normal',
        'bb_width': 0.025,
        'bb_squeeze': False,
        'bb_expansion': True,
        'bb_position': 0.72,

        # Support/Resistance
        'nearest_support': 1.0720,
        'nearest_resistance': 1.0920,
        'support_strength': 'STRONG',
        'resistance_strength': 'MODERATE',
        'break_probability': 0.65,
        'pivot_count': 4,

        # Pattern & Confluence
        'pattern_detected': 'PRICE_SPIKE_UP',
        'pattern_confidence': 75.0,
        'confluence_score': 78.0,
        'signal_strength': 'STRONG',

        # Quality
        'data_quality': 'EXCELLENT',
        'anomaly_detected': False,
        'cache_hit_ratio': 85.0,

        # Moving averages
        'ema_7': 1.0745,
        'vwap_quote_10': 1.0735,
        'bb_lower': 1.0710
    }

    # Créer calculateur
    calc = OpportunityCalculatorPro()

    # Calculer opportunité
    opportunity = calc.calculate_opportunity(
        symbol='BTCUSDC',
        current_price=1.0760,
        analyzer_data=test_data,
        higher_tf_data=None
    )

    # Afficher résultat
    print(f"\n🎯 OPPORTUNITÉ: {opportunity.symbol}")
    print(f"Action: {opportunity.action}")
    print(f"Confiance: {opportunity.confidence:.0f}%")
    print(f"Score: {opportunity.score.total_score:.0f}/100 (Grade {opportunity.score.grade})")
    print(f"Validation: {'✅ PASSÉE' if opportunity.validation.all_passed else '❌ ÉCHOUÉE'}")

    print(f"\n💰 PRICING:")
    print(f"  Prix actuel: {opportunity.current_price:.6f}")
    print(f"  Entrée optimale: {opportunity.entry_price_optimal:.6f}")
    print(f"  Entrée aggressive: {opportunity.entry_price_aggressive:.6f}")

    print(f"\n🎯 TARGETS:")
    print(f"  TP1: {opportunity.tp1:.6f} (+{opportunity.tp1_percent:.2f}%)")
    print(f"  TP2: {opportunity.tp2:.6f} (+{opportunity.tp2_percent:.2f}%)")
    if opportunity.tp3:
        print(f"  TP3: {opportunity.tp3:.6f} (+{opportunity.tp3_percent:.2f}%)")

    print(f"\n🛡️ STOP LOSS:")
    print(f"  SL: {opportunity.stop_loss:.6f} (-{opportunity.stop_loss_percent:.2f}%)")
    print(f"  Basis: {opportunity.stop_loss_basis}")

    print(f"\n📊 RISK:")
    print(f"  R/R Ratio: {opportunity.rr_ratio:.2f}")
    print(f"  Risk Level: {opportunity.risk_level}")
    print(f"  Max Position: {opportunity.max_position_size_pct:.2f}% du capital")

    print(f"\n⏱️ TIMING:")
    print(f"  Hold estimé: {opportunity.estimated_hold_time}")
    print(f"  Urgence: {opportunity.entry_urgency}")

    print(f"\n📋 RAISONS:")
    for reason in opportunity.reasons:
        print(f"  {reason}")

    if opportunity.warnings:
        print(f"\n⚠️ WARNINGS:")
        for warning in opportunity.warnings:
            print(f"  {warning}")

    print(f"\n💡 RECOMMANDATIONS:")
    for rec in opportunity.recommendations:
        print(f"  {rec}")

    print("\n" + "="*80)

    # Export JSON
    opportunity_dict = calc.to_dict(opportunity)
    print("\n📤 Export JSON disponible avec calc.to_dict(opportunity)")
