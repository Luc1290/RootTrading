"""
Opportunity Validator - INSTITUTIONAL SCALPING
Version: 4.0 - Validation pour scalping intraday avec indicateurs institutionnels

REFONTE COMPLÈTE:
1. Aligné avec opportunity_scoring.py v4.0 (VWAP, EMA, Volume, RSI, Bollinger, MACD, S/R)
2. Validation NON-BLOQUANTE sur résistances (<1.5% ignorées)
3. RSI réaliste: 30-75 acceptable (pas strict 35-55)
4. Volume: relatif au contexte, pas de rejet sur spike >3x si justified
5. Focus: QUALITÉ DATA + COHÉRENCE INDICATEURS (pas restrictions arbitraires)
6. Support/Résistance: INFORMATIF seulement, jamais bloquant
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Niveaux de validation pour scalping institutionnel."""

    DATA_QUALITY = "data_quality"  # Qualité données uniquement
    INDICATOR_COHERENCE = "indicator_coherence"  # Cohérence entre indicateurs
    RISK_PARAMETERS = "risk_parameters"  # Paramètres R/R calculables


@dataclass
class ValidationResult:
    """Résultat d'une validation."""

    level: ValidationLevel
    passed: bool
    score: float  # 0-100
    reason: str
    details: dict[str, Any]
    warnings: list[str]


@dataclass
class ValidationSummary:
    """Résumé complet de validation."""

    all_passed: bool
    level_results: dict[ValidationLevel, ValidationResult]
    overall_score: float  # 0-100
    blocking_issues: list[str]
    warnings: list[str]
    recommendations: list[str]


class OpportunityValidator:
    """
    Validateur pour scalping intraday institutionnel.

    PHILOSOPHIE:
    - Validation MINIMALISTE: juste DATA + COHÉRENCE
    - Pas de restrictions arbitraires (RSI >70, volume >3x, etc.)
    - Support/Résistance: JAMAIS bloquant
    - Focus: laisser le scoring faire son travail
    """

    def __init__(self):
        """Initialise le validateur."""

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
        higher_tf_data: dict | None = None,  # noqa: ARG002
    ) -> ValidationSummary:
        """
        Valide une opportunité de scalping.

        VALIDATION MINIMALISTE:
        1. DATA_QUALITY: données complètes et fiables
        2. INDICATOR_COHERENCE: indicateurs cohérents entre eux
        3. RISK_PARAMETERS: R/R calculable et raisonnable
        """
        level_results = {}
        blocking_issues = []
        warnings = []
        recommendations = []

        # Level 1: Data Quality - SEULE VALIDATION VRAIMENT BLOQUANTE
        level_results[ValidationLevel.DATA_QUALITY] = self._validate_data_quality(
            analyzer_data
        )
        if not level_results[ValidationLevel.DATA_QUALITY].passed:
            blocking_issues.append(level_results[ValidationLevel.DATA_QUALITY].reason)
            return self._create_failed_summary(level_results, blocking_issues)

        # Level 2: Indicator Coherence - WARNINGS seulement, pas bloquant
        level_results[ValidationLevel.INDICATOR_COHERENCE] = (
            self._validate_indicator_coherence(analyzer_data)
        )
        # On ne bloque PAS même si cohérence faible, juste warnings
        if not level_results[ValidationLevel.INDICATOR_COHERENCE].passed:
            warnings.append(level_results[ValidationLevel.INDICATOR_COHERENCE].reason)

        # Level 3: Risk Parameters - WARNINGS si R/R faible, pas bloquant
        level_results[ValidationLevel.RISK_PARAMETERS] = self._validate_risk_parameters(
            analyzer_data, current_price
        )
        if not level_results[ValidationLevel.RISK_PARAMETERS].passed:
            warnings.append(level_results[ValidationLevel.RISK_PARAMETERS].reason)

        # Collecter tous les warnings
        for result in level_results.values():
            warnings.extend(result.warnings)

        # Score global
        overall_score = sum(r.score for r in level_results.values()) / len(
            level_results
        )

        # Validation passée si DATA_QUALITY OK (les autres sont informatifs)
        all_passed = level_results[ValidationLevel.DATA_QUALITY].passed

        # Recommandations
        if all_passed:
            recommendations.append("✅ Validation passée - Données fiables")
            recommendations.append(f"Score validation: {overall_score:.0f}/100")
            if overall_score >= 80:
                recommendations.append("✅ Excellente qualité globale")
            elif overall_score >= 60:
                recommendations.append("⚠️ Qualité acceptable avec réserves")
            else:
                recommendations.append("⚠️ Qualité limite - Vérifier warnings")
        else:
            recommendations.append("❌ Validation échouée - Données insuffisantes")

        return ValidationSummary(
            all_passed=all_passed,
            level_results=level_results,
            overall_score=overall_score,
            blocking_issues=blocking_issues,
            warnings=warnings,
            recommendations=recommendations,
        )

    def _validate_data_quality(self, ad: dict) -> ValidationResult:
        """
        Level 1: Validation qualité données - SEULE VALIDATION BLOQUANTE.

        Vérifie uniquement:
        - data_quality: EXCELLENT ou GOOD
        - Présence indicateurs critiques pour scoring institutionnel
        - Pas d'anomalies majeures
        """
        details = {}
        warnings = []
        score = 100.0

        # 1. Data Quality Flag
        data_quality = ad.get("data_quality", "").upper()
        if data_quality not in ["EXCELLENT", "GOOD"]:
            if not data_quality or data_quality == "":
                quality_msg = "champ 'data_quality' absent ou vide"
            else:
                quality_msg = f"qualité '{data_quality}' non acceptable"

            return ValidationResult(
                level=ValidationLevel.DATA_QUALITY,
                passed=False,
                score=0.0,
                reason=f"❌ Qualité données insuffisante: {quality_msg}",
                details={"data_quality": data_quality or "N/A"},
                warnings=[],
            )

        details["data_quality"] = data_quality

        # 2. Anomaly Check
        anomaly = ad.get("anomaly_detected", False)
        if anomaly:
            warnings.append("⚠️ Anomalie détectée dans les données")
            score -= 20

        details["anomaly_detected"] = anomaly

        # 3. Indicateurs CRITIQUES pour scoring institutionnel
        critical_indicators = [
            # VWAP (25% du score)
            "vwap_10",
            # EMA (20% du score)
            "ema_7",
            "ema_12",
            "ema_26",
            # Volume (20% du score)
            "relative_volume",
            # RSI (15% du score)
            "rsi_14",
            # Bollinger (10% du score)
            "bb_upper",
            "bb_lower",
            "bb_squeeze",
            # MACD (5% du score)
            "macd_trend",
            # Support/Résistance (5% du score - informatif)
            "nearest_support",
            "nearest_resistance",
            # Autres essentiels
            "atr_14",
            "current_price",
        ]

        missing = []
        for ind in critical_indicators:
            val = ad.get(ind)
            if val is None and ind not in ad:
                missing.append(ind)

        if missing:
            # Pénalité progressive selon nombre manquants
            penalty = min(len(missing) * 8, 60)
            score -= penalty
            warnings.append(f"⚠️ Indicateurs manquants ({len(missing)}): {', '.join(missing[:5])}")

            # BLOQUANT si indicateurs majeurs manquants
            critical_missing = [
                ind
                for ind in missing
                if ind in ["vwap_10", "ema_7", "rsi_14", "relative_volume", "atr_14"]
            ]
            if critical_missing:
                return ValidationResult(
                    level=ValidationLevel.DATA_QUALITY,
                    passed=False,
                    score=0.0,
                    reason=f"❌ Indicateurs MAJEURS manquants: {', '.join(critical_missing)}",
                    details={"missing_critical": critical_missing},
                    warnings=[],
                )

        details["missing_indicators"] = missing
        details["indicators_present"] = len(critical_indicators) - len(missing)

        # Validation passée si score >60
        passed = score >= 60

        return ValidationResult(
            level=ValidationLevel.DATA_QUALITY,
            passed=passed,
            score=score,
            reason=(
                "✅ Qualité données OK"
                if passed
                else f"❌ Qualité données insuffisante (score {score:.0f}/100)"
            ),
            details=details,
            warnings=warnings,
        )

    def _validate_indicator_coherence(self, ad: dict) -> ValidationResult:
        """
        Level 2: Validation cohérence indicateurs - INFORMATIF, pas bloquant.

        Vérifie que les indicateurs racontent une histoire cohérente:
        - EMA alignées avec VWAP position
        - RSI cohérent avec MACD
        - Volume cohérent avec Bollinger (squeeze/expansion)

        JAMAIS bloquant, juste warnings si incohérences.
        """
        details = {}
        warnings = []
        score = 100.0

        # 1. Cohérence VWAP <-> EMA
        vwap = self.safe_float(ad.get("vwap_10"))
        ema7 = self.safe_float(ad.get("ema_7"))
        current_price = self.safe_float(ad.get("current_price"))

        if vwap > 0 and ema7 > 0 and current_price > 0:
            above_vwap = current_price > vwap
            above_ema7 = current_price > ema7

            # Incohérence: au dessus VWAP mais sous EMA7
            if above_vwap and not above_ema7:
                warnings.append("⚠️ Incohérence: prix > VWAP mais < EMA7")
                score -= 15
                details["vwap_ema_incoherent"] = True
            # Incohérence inverse
            elif not above_vwap and above_ema7:
                warnings.append("⚠️ Incohérence: prix < VWAP mais > EMA7")
                score -= 15
                details["vwap_ema_incoherent"] = True
            else:
                details["vwap_ema_coherent"] = True

        # 2. Cohérence RSI <-> MACD
        rsi = self.safe_float(ad.get("rsi_14"))
        macd_trend = ad.get("macd_trend", "").upper()

        if rsi > 0 and macd_trend:
            # RSI bullish (>50) mais MACD bearish
            if rsi > 55 and macd_trend == "BEARISH":
                warnings.append(f"⚠️ Incohérence: RSI {rsi:.0f} > 55 mais MACD bearish")
                score -= 12
                details["rsi_macd_incoherent"] = True
            # RSI bearish (<45) mais MACD bullish
            elif rsi < 45 and macd_trend == "BULLISH":
                warnings.append(f"⚠️ Incohérence: RSI {rsi:.0f} < 45 mais MACD bullish")
                score -= 12
                details["rsi_macd_incoherent"] = True
            else:
                details["rsi_macd_coherent"] = True

        # 3. Cohérence Volume <-> Bollinger
        rel_volume = self.safe_float(ad.get("relative_volume"), 1.0)
        bb_squeeze = ad.get("bb_squeeze", False)
        bb_expansion = ad.get("bb_expansion", False)

        # Volume faible pendant expansion = incohérence
        if bb_expansion and rel_volume < 0.7:
            warnings.append(
                f"⚠️ Incohérence: BB expansion mais volume faible ({rel_volume:.2f}x)"
            )
            score -= 10
            details["volume_bb_incoherent"] = True
        # Volume fort pendant squeeze = normal (breakout imminent)
        elif bb_squeeze and rel_volume > 1.5:
            details["volume_bb_breakout_setup"] = True
            warnings.append("ℹ️ Volume fort + BB squeeze = breakout imminent")
        else:
            details["volume_bb_coherent"] = True

        # 4. Cohérence Trend (market_regime vs indicators)
        regime = ad.get("market_regime", "").upper()
        adx = self.safe_float(ad.get("adx_14"))

        # TRENDING regime mais ADX faible
        if "TRENDING" in regime and adx < 20:
            warnings.append(f"⚠️ Régime {regime} mais ADX faible ({adx:.0f})")
            score -= 10
            details["regime_adx_incoherent"] = True

        details["regime"] = regime
        details["adx"] = adx

        # Validation: jamais bloquant, juste score informatif
        passed = score >= 50  # Seuil bas car informatif seulement

        return ValidationResult(
            level=ValidationLevel.INDICATOR_COHERENCE,
            passed=passed,
            score=score,
            reason=(
                "✅ Indicateurs cohérents"
                if score >= 80
                else f"⚠️ Quelques incohérences mineures (score {score:.0f}/100)"
                if score >= 50
                else f"⚠️ Incohérences importantes (score {score:.0f}/100)"
            ),
            details=details,
            warnings=warnings,
        )

    def _validate_risk_parameters(
        self, ad: dict, current_price: float
    ) -> ValidationResult:
        """
        Level 3: Validation paramètres de risque - INFORMATIF, pas bloquant.

        Vérifie:
        - ATR disponible pour SL
        - R/R calculable et raisonnable (>1.2 acceptable, >1.8 bon)
        - Résistance: JAMAIS bloquant, juste informatif

        JAMAIS bloquant même si R/R faible.
        """
        details = {}
        warnings = []
        score = 100.0

        # 1. ATR pour SL
        atr = self.safe_float(ad.get("atr_14"))
        natr = self.safe_float(ad.get("natr"))

        if atr <= 0 and natr <= 0:
            warnings.append("⚠️ ATR indisponible - SL sera basé sur % fixe")
            score -= 30
            details["atr_available"] = False
        else:
            details["atr_available"] = True

            if atr > 0 and current_price > 0:
                atr_percent = atr / current_price
            elif natr > 0:
                atr_percent = natr / 100.0
            else:
                atr_percent = 0.01

            details["atr_percent"] = atr_percent * 100

        # 2. Support/Résistance - INFORMATIF SEULEMENT
        nearest_support = self.safe_float(ad.get("nearest_support"))
        nearest_resistance = self.safe_float(ad.get("nearest_resistance"))

        # Support: calcul SL distance
        if nearest_support > 0 and current_price > nearest_support:
            sl_dist = (current_price - nearest_support) / current_price
            sl_basis = "support"
            details["support_distance_pct"] = sl_dist * 100
        else:
            sl_dist = atr_percent * 0.8 if atr > 0 else 0.015  # 1.5% par défaut
            sl_basis = "ATR"
            warnings.append("ℹ️ Pas de support proche - SL basé sur ATR")

        details["sl_distance_pct"] = sl_dist * 100
        details["sl_basis"] = sl_basis

        # Résistance: JAMAIS bloquant, juste informatif
        if nearest_resistance > 0 and current_price > 0:
            res_dist_pct = ((nearest_resistance - current_price) / current_price) * 100

            if res_dist_pct < 0:
                # Prix AU DESSUS de la résistance = breakout!
                warnings.append(
                    f"ℹ️ Prix au dessus de la résistance (breakout de {abs(res_dist_pct):.1f}%)"
                )
                details["resistance_broken"] = True
            elif res_dist_pct < 1.5:
                # Résistance proche mais ignorée (ATH/breakout imminent)
                warnings.append(
                    f"ℹ️ Résistance proche ({res_dist_pct:.1f}%) - considérée négligeable"
                )
                details["resistance_close"] = True
            elif res_dist_pct < 3.0:
                warnings.append(f"ℹ️ Résistance à {res_dist_pct:.1f}% - marge modérée")
            else:
                warnings.append(f"ℹ️ Résistance à {res_dist_pct:.1f}% - espace libre")

            details["resistance_distance_pct"] = res_dist_pct

        # 3. R/R Ratio - WARNING si faible, pas bloquant
        # TP conservateur: 0.8 ATR pour validation
        tp_dist = atr_percent * 0.8 if atr > 0 else 0.015

        rr_ratio = tp_dist / sl_dist if sl_dist > 0 else 0

        if rr_ratio < 1.2:
            warnings.append(f"⚠️ R/R faible: {rr_ratio:.2f} < 1.2 - Risque élevé")
            score -= 30
            details["rr_quality"] = "Faible"
        elif rr_ratio < 1.8:
            warnings.append(f"⚠️ R/R modéré: {rr_ratio:.2f} < 1.8 - Acceptable")
            score -= 15
            details["rr_quality"] = "Acceptable"
        elif rr_ratio < 2.5:
            details["rr_quality"] = "Bon"
        else:
            details["rr_quality"] = "Excellent"

        details["rr_ratio"] = rr_ratio
        details["tp_distance_pct"] = tp_dist * 100

        # Validation: jamais bloquant, juste warnings
        passed = score >= 40  # Seuil très bas car informatif

        return ValidationResult(
            level=ValidationLevel.RISK_PARAMETERS,
            passed=passed,
            score=score,
            reason=(
                "✅ Paramètres risque OK"
                if score >= 80
                else f"⚠️ Paramètres risque acceptables (score {score:.0f}/100)"
                if score >= 40
                else f"⚠️ Paramètres risque limites (score {score:.0f}/100)"
            ),
            details=details,
            warnings=warnings,
        )

    def _create_failed_summary(
        self, level_results: dict, blocking_issues: list
    ) -> ValidationSummary:
        """Crée un résumé de validation échouée."""
        return ValidationSummary(
            all_passed=False,
            level_results=level_results,
            overall_score=0.0,
            blocking_issues=blocking_issues,
            warnings=[],
            recommendations=["❌ Validation échouée - Données insuffisantes"],
        )
