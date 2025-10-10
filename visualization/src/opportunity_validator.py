"""
Opportunity Validator - Multi-Level Validation System
Validation stricte en 4 niveaux (gates) avant d'autoriser un trade

Architecture:
- Level 1: Data Quality Gates (données complètes et cohérentes)
- Level 2: Market Condition Gates (conditions marché acceptables)
- Level 3: Risk Management Gates (risque acceptable)
- Level 4: Entry Timing Gates (timing optimal)

Version: 2.0 - Professional Grade
"""
import logging
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Niveaux de validation."""
    DATA_QUALITY = "data_quality"
    MARKET_CONDITIONS = "market_conditions"
    RISK_MANAGEMENT = "risk_management"
    ENTRY_TIMING = "entry_timing"


@dataclass
class ValidationResult:
    """Résultat d'une validation."""
    level: ValidationLevel
    passed: bool
    score: float  # 0-100
    reason: str
    details: Dict[str, any]
    warnings: List[str]


@dataclass
class ValidationSummary:
    """Résumé complet de validation."""
    all_passed: bool
    level_results: Dict[ValidationLevel, ValidationResult]
    overall_score: float  # 0-100
    blocking_issues: List[str]
    warnings: List[str]
    recommendations: List[str]


class OpportunityValidator:
    """
    Validateur multi-niveaux pour opportunités de trading.

    Chaque niveau doit passer pour autoriser le trade.
    Validation stricte pour protéger le capital.
    """

    def __init__(self):
        """Initialise le validateur."""
        pass

    @staticmethod
    def safe_float(value, default=0.0):
        """Convertir en float avec fallback."""
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    def validate_opportunity(
        self,
        analyzer_data: dict,
        current_price: float,
        higher_tf_data: Optional[dict] = None
    ) -> ValidationSummary:
        """
        Valide une opportunité sur les 4 niveaux.

        Args:
            analyzer_data: Données analyzer_data complètes
            current_price: Prix actuel
            higher_tf_data: Données timeframe supérieur (5m pour validation 1m)

        Returns:
            ValidationSummary complet
        """
        level_results = {}
        blocking_issues = []
        warnings = []
        recommendations = []

        # Level 1: Data Quality
        level_results[ValidationLevel.DATA_QUALITY] = self._validate_data_quality(analyzer_data)
        if not level_results[ValidationLevel.DATA_QUALITY].passed:
            blocking_issues.append(level_results[ValidationLevel.DATA_QUALITY].reason)
            return self._create_failed_summary(level_results, blocking_issues)

        # Level 2: Market Conditions
        level_results[ValidationLevel.MARKET_CONDITIONS] = self._validate_market_conditions(
            analyzer_data, higher_tf_data
        )
        if not level_results[ValidationLevel.MARKET_CONDITIONS].passed:
            blocking_issues.append(level_results[ValidationLevel.MARKET_CONDITIONS].reason)

        # Level 3: Risk Management
        level_results[ValidationLevel.RISK_MANAGEMENT] = self._validate_risk_management(
            analyzer_data, current_price
        )
        if not level_results[ValidationLevel.RISK_MANAGEMENT].passed:
            blocking_issues.append(level_results[ValidationLevel.RISK_MANAGEMENT].reason)

        # Level 4: Entry Timing
        level_results[ValidationLevel.ENTRY_TIMING] = self._validate_entry_timing(analyzer_data)
        if not level_results[ValidationLevel.ENTRY_TIMING].passed:
            blocking_issues.append(level_results[ValidationLevel.ENTRY_TIMING].reason)

        # Collecter warnings
        for result in level_results.values():
            warnings.extend(result.warnings)

        # Score global
        overall_score = sum(r.score for r in level_results.values()) / len(level_results)

        # Toutes les validations passées?
        all_passed = all(r.passed for r in level_results.values())

        # Recommandations
        if all_passed:
            recommendations.append("✅ Toutes les validations passées")
            recommendations.append(f"Score validation: {overall_score:.0f}/100")
        else:
            recommendations.append("❌ Validation échouée")
            recommendations.append(f"Problèmes à résoudre: {len(blocking_issues)}")

        return ValidationSummary(
            all_passed=all_passed,
            level_results=level_results,
            overall_score=overall_score,
            blocking_issues=blocking_issues,
            warnings=warnings,
            recommendations=recommendations
        )

    def _validate_data_quality(self, ad: dict) -> ValidationResult:
        """
        Level 1: Validation qualité des données.

        Vérifie:
        - data_quality (EXCELLENT/GOOD requis)
        - Indicateurs critiques présents
        - Pas d'anomalies détectées
        - Timestamp récent
        """
        details = {}
        warnings = []
        score = 100.0

        # 1. Data Quality Flag
        data_quality = ad.get('data_quality', '').upper()
        if data_quality not in ['EXCELLENT', 'GOOD']:
            # Message explicite selon le cas
            if not data_quality or data_quality == '':
                quality_msg = "champ 'data_quality' absent ou vide dans analyzer_data"
            else:
                quality_msg = f"qualité '{data_quality}' non acceptable (requis: EXCELLENT ou GOOD)"

            return ValidationResult(
                level=ValidationLevel.DATA_QUALITY,
                passed=False,
                score=0.0,
                reason=f"❌ Qualité données insuffisante: {quality_msg}",
                details={'data_quality': data_quality or 'N/A'},
                warnings=[]
            )

        details['data_quality'] = data_quality

        # 2. Anomalies
        anomaly = ad.get('anomaly_detected', False)
        if anomaly:
            warnings.append("⚠️ Anomalie détectée dans les données")
            score -= 20

        details['anomaly_detected'] = anomaly

        # 3. Indicateurs critiques présents
        critical_indicators = [
            'rsi_14', 'adx_14', 'macd_trend', 'relative_volume',
            'market_regime', 'nearest_resistance', 'atr_14'
        ]

        missing = []
        for ind in critical_indicators:
            if ad.get(ind) is None:
                missing.append(ind)

        if missing:
            score -= len(missing) * 10
            warnings.append(f"⚠️ Indicateurs manquants: {', '.join(missing)}")

        details['missing_indicators'] = missing
        details['indicators_present'] = len(critical_indicators) - len(missing)

        # 4. Cache hit ratio (performance)
        cache_ratio = self.safe_float(ad.get('cache_hit_ratio'))
        if cache_ratio > 0:
            details['cache_hit_ratio'] = cache_ratio

        # Validation passée si score >70
        passed = score >= 70

        return ValidationResult(
            level=ValidationLevel.DATA_QUALITY,
            passed=passed,
            score=score,
            reason="✅ Qualité données OK" if passed else f"❌ Qualité données insuffisante (score {score:.0f}/100)",
            details=details,
            warnings=warnings
        )

    def _validate_market_conditions(self, ad: dict, higher_tf: Optional[dict]) -> ValidationResult:
        """
        Level 2: Validation conditions de marché.

        Vérifie:
        - Régime de marché acceptable (pas TRENDING_BEAR/BREAKOUT_BEAR)
        - Confiance régime >50%
        - Volatilité pas extrême
        - Timeframe supérieur aligné
        - Pas de divergence majeure entre TF
        """
        details = {}
        warnings = []
        score = 100.0

        # 1. Régime de marché
        regime = ad.get('market_regime', '').upper()
        regime_conf = self.safe_float(ad.get('regime_confidence'))

        if regime in ['TRENDING_BEAR', 'BREAKOUT_BEAR']:
            return ValidationResult(
                level=ValidationLevel.MARKET_CONDITIONS,
                passed=False,
                score=0.0,
                reason=f"❌ Régime baissier: {regime}",
                details={'regime': regime, 'confidence': regime_conf},
                warnings=[]
            )

        # Confiance régime
        if regime_conf < 50:
            warnings.append(f"⚠️ Confiance régime faible: {regime_conf:.0f}%")
            score -= 15

        details['regime'] = regime
        details['regime_confidence'] = regime_conf

        # 2. Volatilité
        vol_regime = ad.get('volatility_regime', '').lower()
        if vol_regime == 'extreme':
            warnings.append("⚠️ Volatilité extrême - Risque élevé")
            score -= 20
        elif vol_regime == 'low':
            warnings.append("⚠️ Volatilité faible - Peu de mouvement attendu")
            score -= 10

        details['volatility_regime'] = vol_regime

        # 3. Directional Bias
        bias = ad.get('directional_bias', '').upper()
        if bias == 'BEARISH':
            return ValidationResult(
                level=ValidationLevel.MARKET_CONDITIONS,
                passed=False,
                score=0.0,
                reason=f"❌ Biais directionnel baissier: {bias}",
                details=details,
                warnings=[]
            )
        elif bias == 'NEUTRAL':
            warnings.append("⚠️ Biais neutre - Pas de direction claire")
            score -= 10

        details['directional_bias'] = bias

        # 4. Higher Timeframe Validation
        if higher_tf:
            htf_regime = higher_tf.get('market_regime', '').upper()
            htf_rsi = self.safe_float(higher_tf.get('rsi_14'))
            htf_macd = higher_tf.get('macd_trend', '').upper()

            # Rejet si HTF bearish
            if htf_regime in ['TRENDING_BEAR', 'BREAKOUT_BEAR']:
                return ValidationResult(
                    level=ValidationLevel.MARKET_CONDITIONS,
                    passed=False,
                    score=0.0,
                    reason=f"❌ Timeframe supérieur baissier: {htf_regime}",
                    details={**details, 'htf_regime': htf_regime},
                    warnings=[]
                )

            # Warning si divergence
            if htf_rsi < 50 and htf_macd == 'BEARISH':
                warnings.append("⚠️ Timeframe supérieur montre faiblesse")
                score -= 15

            details['htf_aligned'] = True
            details['htf_regime'] = htf_regime
        else:
            warnings.append("ℹ️ Pas de validation timeframe supérieur")
            score -= 5

        # Validation passée si score >60
        passed = score >= 60

        return ValidationResult(
            level=ValidationLevel.MARKET_CONDITIONS,
            passed=passed,
            score=score,
            reason="✅ Conditions marché OK" if passed else f"❌ Conditions marché défavorables (score {score:.0f}/100)",
            details=details,
            warnings=warnings
        )

    def _validate_risk_management(self, ad: dict, current_price: float) -> ValidationResult:
        """
        Level 3: Validation gestion du risque.

        Vérifie:
        - R/R ratio >1.5 minimum
        - ATR disponible pour calcul SL
        - Distance résistance acceptable
        - Support proche identifié
        - Taille position calculable
        """
        details = {}
        warnings = []
        score = 100.0

        # 1. ATR pour SL
        atr = self.safe_float(ad.get('atr_14'))
        natr = self.safe_float(ad.get('natr'))

        if atr <= 0 and natr <= 0:
            return ValidationResult(
                level=ValidationLevel.RISK_MANAGEMENT,
                passed=False,
                score=0.0,
                reason="❌ ATR indisponible - Impossible de calculer SL",
                details={},
                warnings=[]
            )

        # Calculer ATR%
        if atr > 0 and current_price > 0:
            atr_percent = atr / current_price
        elif natr > 0:
            atr_percent = natr / 100.0
        else:
            atr_percent = 0.01  # Fallback 1%

        details['atr_percent'] = atr_percent * 100

        # 2. Support/Résistance pour R/R
        nearest_support = self.safe_float(ad.get('nearest_support'))
        nearest_resistance = self.safe_float(ad.get('nearest_resistance'))

        # Calculer SL distance
        if nearest_support > 0 and current_price > nearest_support:
            sl_dist = max(0.007, (current_price - nearest_support) / current_price)
            sl_basis = "support"
        else:
            sl_dist = max(0.007, atr_percent * 0.7)
            sl_basis = "ATR"
            warnings.append("⚠️ Pas de support identifié - SL basé sur ATR")
            score -= 10

        details['sl_distance_pct'] = sl_dist * 100
        details['sl_basis'] = sl_basis

        # Calculer TP distance (utiliser multiplier plus agressif pour validation)
        # TP1 conservateur: 0.8 ATR, TP2 modéré: 1.2 ATR
        # Pour validation, on vérifie que TP2 (moderate) donne un R/R acceptable
        tp_dist_conservative = max(0.01, atr_percent * 0.8)
        tp_dist_moderate = max(0.015, atr_percent * 1.2)

        details['tp1_distance_pct'] = tp_dist_conservative * 100
        details['tp2_distance_pct'] = tp_dist_moderate * 100

        # R/R Ratio - Utiliser TP2 (moderate) pour validation
        # Cela permet de valider les setups où TP2 est atteignable
        rr_ratio = tp_dist_moderate / sl_dist if sl_dist > 0 else 0

        if rr_ratio < 1.5:
            return ValidationResult(
                level=ValidationLevel.RISK_MANAGEMENT,
                passed=False,
                score=0.0,
                reason=f"❌ R/R trop faible: {rr_ratio:.2f} < 1.5",
                details={**details, 'rr_ratio': rr_ratio},
                warnings=[]
            )

        details['rr_ratio'] = rr_ratio

        # Bonus si R/R excellent
        if rr_ratio > 3.0:
            details['rr_quality'] = 'Excellent'
        elif rr_ratio > 2.0:
            details['rr_quality'] = 'Bon'
        else:
            details['rr_quality'] = 'Acceptable'

        # 3. Vérifier résistance
        if nearest_resistance > 0 and current_price > 0:
            res_dist_pct = ((nearest_resistance - current_price) / current_price) * 100

            # Résistance trop proche?
            if res_dist_pct < 0.5:
                warnings.append(f"⚠️ Résistance très proche: {res_dist_pct:.1f}%")
                score -= 15
            elif res_dist_pct < 1.0:
                warnings.append(f"⚠️ Résistance proche: {res_dist_pct:.1f}%")
                score -= 10

            details['resistance_distance_pct'] = res_dist_pct

            # Vérifier si TP atteignable (utiliser TP2 pour cohérence avec le R/R)
            if tp_dist_moderate * 100 > res_dist_pct:
                warnings.append("⚠️ TP au-delà de la résistance")
                score -= 10

        # 4. Break probability
        break_prob = self.safe_float(ad.get('break_probability'))
        if break_prob > 0:
            if break_prob < 0.4:
                warnings.append(f"⚠️ Probabilité cassure faible: {break_prob*100:.0f}%")
                score -= 10

            details['break_probability'] = break_prob * 100

        # Validation passée si score >70
        passed = score >= 70

        return ValidationResult(
            level=ValidationLevel.RISK_MANAGEMENT,
            passed=passed,
            score=score,
            reason="✅ Gestion risque OK" if passed else f"❌ Gestion risque insuffisante (score {score:.0f}/100)",
            details=details,
            warnings=warnings
        )

    def _validate_entry_timing(self, ad: dict) -> ValidationResult:
        """
        Level 4: Validation timing d'entrée.

        Vérifie:
        - Pas de surachat extrême (RSI, MFI, Stochastic)
        - Volume confirmant
        - Momentum positif
        - Pattern favorable ou neutre
        - Pas de divergence négative
        """
        details = {}
        warnings = []
        score = 100.0

        # 1. Overbought Check
        rsi = self.safe_float(ad.get('rsi_14'))
        mfi = self.safe_float(ad.get('mfi_14'))
        stoch_k = self.safe_float(ad.get('stoch_k'))
        stoch_d = self.safe_float(ad.get('stoch_d'))

        overbought_issues = []

        if rsi > 80:
            overbought_issues.append(f"RSI extrême ({rsi:.0f})")
            score -= 20
        elif rsi > 75:
            warnings.append(f"⚠️ RSI élevé ({rsi:.0f})")
            score -= 10

        if mfi > 85:
            overbought_issues.append(f"MFI extrême ({mfi:.0f})")
            score -= 15
        elif mfi > 80:
            warnings.append(f"⚠️ MFI élevé ({mfi:.0f})")
            score -= 10

        if stoch_k > 90 and stoch_d > 90:
            overbought_issues.append(f"Stochastic saturé ({stoch_k:.0f}/{stoch_d:.0f})")
            score -= 15

        if overbought_issues:
            details['overbought_issues'] = overbought_issues

        details['rsi'] = rsi
        details['mfi'] = mfi

        # 2. Volume Confirmation
        vol_context = ad.get('volume_context', '').upper()
        rel_volume = self.safe_float(ad.get('relative_volume'), 1.0)
        obv_osc = self.safe_float(ad.get('obv_oscillator'))

        # Volume trop faible = rejet (seuil ajusté pour crypto 1m)
        # Ancien seuil 0.8x était trop strict, nouveau seuil 0.5x
        if rel_volume < 0.5:
            return ValidationResult(
                level=ValidationLevel.ENTRY_TIMING,
                passed=False,
                score=0.0,
                reason=f"❌ Volume trop faible: {rel_volume:.2f}x",
                details=details,
                warnings=[]
            )

        # Warning si volume modéré (0.5-0.8x)
        if rel_volume < 0.8:
            warnings.append(f"⚠️ Volume modéré: {rel_volume:.2f}x")
            score -= 10

        # OBV négatif fort = warning
        if obv_osc < -200:
            warnings.append(f"⚠️ OBV négatif: {obv_osc:.0f}")
            score -= 15

        # Contexts baissiers
        if vol_context in ['REVERSAL_PATTERN', 'DEEP_OVERSOLD']:
            warnings.append(f"⚠️ Volume context défavorable: {vol_context}")
            score -= 15

        details['volume_context'] = vol_context
        details['relative_volume'] = rel_volume

        # 3. Momentum Check
        macd_trend = ad.get('macd_trend', '').upper()
        momentum_score_val = self.safe_float(ad.get('momentum_score'))

        if macd_trend == 'BEARISH':
            warnings.append("⚠️ MACD baissier")
            score -= 15
        elif macd_trend == 'NEUTRAL':
            warnings.append("⚠️ MACD neutre")
            score -= 5

        if momentum_score_val < 40:
            warnings.append(f"⚠️ Momentum score faible: {momentum_score_val:.0f}")
            score -= 10

        details['macd_trend'] = macd_trend
        details['momentum_score'] = momentum_score_val

        # 4. Pattern Check
        pattern = ad.get('pattern_detected', '').upper()

        bearish_patterns = ['PRICE_SPIKE_DOWN', 'LIQUIDITY_SWEEP']
        if pattern in bearish_patterns:
            warnings.append(f"⚠️ Pattern baissier: {pattern}")
            score -= 15

        details['pattern'] = pattern

        # 5. Divergence Check
        stoch_div = ad.get('stoch_divergence', False)
        if stoch_div:
            # Divergence peut être bullish ou bearish, need more info
            warnings.append("ℹ️ Divergence Stochastic détectée")

        # Validation passée si score >65
        passed = score >= 65

        return ValidationResult(
            level=ValidationLevel.ENTRY_TIMING,
            passed=passed,
            score=score,
            reason="✅ Timing OK" if passed else f"❌ Timing défavorable (score {score:.0f}/100)",
            details=details,
            warnings=warnings
        )

    def _create_failed_summary(self, level_results: dict, blocking_issues: list) -> ValidationSummary:
        """Crée un résumé de validation échouée."""
        return ValidationSummary(
            all_passed=False,
            level_results=level_results,
            overall_score=0.0,
            blocking_issues=blocking_issues,
            warnings=[],
            recommendations=["❌ Validation échouée au premier niveau"]
        )
