"""
Opportunity Validator - CORRECTED FOR EARLY BUYING
Version: 3.0 - Validation pour acheter AVANT le pump

CHANGEMENTS MAJEURS:
1. RSI/MFI élevés = REJET (pas tolérance pump)
2. Volume spike >3x = REJET (pic déjà atteint)
3. Résistance <2% = PÉNALITÉ LOURDE (trop proche plafond)
4. R/R calculation avec TP conservateur (pas optimiste)
5. Market conditions: tolérance bear SUPPRIMÉE (pas de contre-tendance)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

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
    """Validateur pour acheter AVANT le pump (early entry)."""

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
        higher_tf_data: dict | None = None,
    ) -> ValidationSummary:
        """Valide une opportunité d'achat EARLY."""
        level_results = {}
        blocking_issues = []
        warnings = []
        recommendations = []

        # Level 1: Data Quality
        level_results[ValidationLevel.DATA_QUALITY] = self._validate_data_quality(
            analyzer_data
        )
        if not level_results[ValidationLevel.DATA_QUALITY].passed:
            blocking_issues.append(level_results[ValidationLevel.DATA_QUALITY].reason)
            return self._create_failed_summary(level_results, blocking_issues)

        # Level 2: Market Conditions (STRICT - pas de contre-tendance)
        level_results[ValidationLevel.MARKET_CONDITIONS] = (
            self._validate_market_conditions_early(analyzer_data, higher_tf_data)
        )
        if not level_results[ValidationLevel.MARKET_CONDITIONS].passed:
            blocking_issues.append(
                level_results[ValidationLevel.MARKET_CONDITIONS].reason
            )

        # Level 3: Risk Management (distance résistance critique)
        level_results[ValidationLevel.RISK_MANAGEMENT] = self._validate_risk_management_early(
            analyzer_data, current_price
        )
        if not level_results[ValidationLevel.RISK_MANAGEMENT].passed:
            blocking_issues.append(
                level_results[ValidationLevel.RISK_MANAGEMENT].reason
            )

        # Level 4: Entry Timing (RSI/MFI strict)
        level_results[ValidationLevel.ENTRY_TIMING] = self._validate_entry_timing_early(
            analyzer_data
        )
        if not level_results[ValidationLevel.ENTRY_TIMING].passed:
            blocking_issues.append(level_results[ValidationLevel.ENTRY_TIMING].reason)

        # Collecter warnings
        for result in level_results.values():
            warnings.extend(result.warnings)

        # Score global
        overall_score = sum(r.score for r in level_results.values()) / len(
            level_results
        )

        # Toutes les validations passées?
        all_passed = all(r.passed for r in level_results.values())

        # Recommandations
        if all_passed:
            recommendations.append("✅ Toutes les validations passées")
            recommendations.append(f"Score validation: {overall_score:.0f}/100")
            recommendations.append("✅ Conditions EARLY entry réunies")
        else:
            recommendations.append("❌ Validation échouée")
            recommendations.append(f"Problèmes bloquants: {len(blocking_issues)}")

        return ValidationSummary(
            all_passed=all_passed,
            level_results=level_results,
            overall_score=overall_score,
            blocking_issues=blocking_issues,
            warnings=warnings,
            recommendations=recommendations,
        )

    def _validate_data_quality(self, ad: dict) -> ValidationResult:
        """Level 1: Validation qualité données - INCHANGÉ."""
        details = {}
        warnings = []
        score = 100.0

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

        anomaly = ad.get("anomaly_detected", False)
        if anomaly:
            warnings.append("⚠️ Anomalie détectée dans les données")
            score -= 20

        details["anomaly_detected"] = anomaly

        critical_indicators = [
            "rsi_14",
            "adx_14",
            "macd_trend",
            "relative_volume",
            "market_regime",
            "nearest_resistance",
            "atr_14",
        ]

        missing = []
        for ind in critical_indicators:
            val = ad.get(ind)
            if val is None and ind not in ad:
                missing.append(ind)

        if missing:
            score -= len(missing) * 10
            warnings.append(f"⚠️ Indicateurs manquants: {', '.join(missing)}")

        details["missing_indicators"] = missing
        details["indicators_present"] = len(critical_indicators) - len(missing)

        passed = score >= 70

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

    def _validate_market_conditions_early(
        self, ad: dict, higher_tf: dict | None
    ) -> ValidationResult:
        """
        Level 2: Market Conditions - CORRECTED pour EARLY entry.

        CHANGEMENTS:
        - Régime BEAR = REJET (pas de contre-tendance)
        - Directional bias BEARISH = REJET
        - HTF bear = REJET
        - Volatilité extreme = REJET
        """
        details = {}
        warnings = []
        score = 100.0

        # 1. Régime de marché - STRICT
        regime = ad.get("market_regime", "").upper()
        regime_conf = self.safe_float(ad.get("regime_confidence"))

        # REJET BINAIRE des régimes baissiers
        if regime in ["TRENDING_BEAR", "BREAKOUT_BEAR"]:
            return ValidationResult(
                level=ValidationLevel.MARKET_CONDITIONS,
                passed=False,
                score=0.0,
                reason=f"❌ RÉGIME BAISSIER: {regime} - Pas d'achat en contre-tendance",
                details={"regime": regime, "regime_confidence": regime_conf},
                warnings=[f"⚠️⚠️ Régime {regime} incompatible avec stratégie LONG"],
            )

        # Transition acceptable mais pénalisé
        if regime == "TRANSITION":
            score -= 15
            warnings.append("⚠️ Régime transition - Direction incertaine")

        # Confiance régime
        if regime_conf < 60:  # Seuil augmenté de 50 à 60
            warnings.append(f"⚠️ Confiance régime modérée: {regime_conf:.0f}%")
            score -= 15

        details["regime"] = regime
        details["regime_confidence"] = regime_conf

        # 2. Volatilité - STRICT
        vol_regime = ad.get("volatility_regime", "").lower()
        if vol_regime == "extreme":
            return ValidationResult(
                level=ValidationLevel.MARKET_CONDITIONS,
                passed=False,
                score=0.0,
                reason="❌ VOLATILITÉ EXTRÊME - Risque trop élevé",
                details={"volatility_regime": vol_regime},
                warnings=["⚠️⚠️ Volatilité extrême incompatible"],
            )
        if vol_regime == "low":
            warnings.append("⚠️ Volatilité faible - Peu de mouvement attendu")
            score -= 10

        details["volatility_regime"] = vol_regime

        # 3. Directional Bias - STRICT
        bias = ad.get("directional_bias", "").upper()
        if bias == "BEARISH":
            return ValidationResult(
                level=ValidationLevel.MARKET_CONDITIONS,
                passed=False,
                score=0.0,
                reason=f"❌ BIAIS BAISSIER: {bias} - Pas d'achat contre-tendance",
                details={"directional_bias": bias},
                warnings=[f"⚠️⚠️ Bias {bias} incompatible avec LONG"],
            )
        elif bias == "NEUTRAL":  # noqa: RET505
            warnings.append("⚠️ Biais neutre - Pas de direction claire")
            score -= 12

        details["directional_bias"] = bias

        # 4. Higher Timeframe - STRICT
        if higher_tf:
            htf_regime = higher_tf.get("market_regime", "").upper()
            htf_rsi = self.safe_float(higher_tf.get("rsi_14"))
            htf_macd = higher_tf.get("macd_trend", "").upper()

            # HTF bearish = REJET
            if htf_regime in ["TRENDING_BEAR", "BREAKOUT_BEAR"]:
                return ValidationResult(
                    level=ValidationLevel.MARKET_CONDITIONS,
                    passed=False,
                    score=0.0,
                    reason=f"❌ HTF BAISSIER: {htf_regime} - Pas d'achat contre-tendance HTF",
                    details={"htf_regime": htf_regime},
                    warnings=[f"⚠️⚠️ Timeframe supérieur {htf_regime} incompatible"],
                )

            # Warning si divergence
            if htf_rsi < 45 or htf_macd == "BEARISH":
                warnings.append(f"⚠️ HTF montre faiblesse (RSI {htf_rsi:.0f}, MACD {htf_macd})")
                score -= 15

            details["htf_aligned"] = True
            details["htf_regime"] = htf_regime
        else:
            warnings.append("ℹ️ Pas de validation timeframe supérieur")
            score -= 10

        # Validation passée si score >70 (augmenté de 60)
        passed = score >= 70

        return ValidationResult(
            level=ValidationLevel.MARKET_CONDITIONS,
            passed=passed,
            score=score,
            reason=(
                "✅ Conditions marché OK"
                if passed
                else f"❌ Conditions marché défavorables (score {score:.0f}/100)"
            ),
            details=details,
            warnings=warnings,
        )

    def _validate_risk_management_early(
        self, ad: dict, current_price: float
    ) -> ValidationResult:
        """
        Level 3: Risk Management - CORRECTED pour EARLY entry.

        CHANGEMENTS:
        - Distance résistance >2% REQUISE (sinon déjà au plafond)
        - TP calculation CONSERVATIVE (0.6 ATR au lieu de 1.2)
        - R/R minimum 1.8 (augmenté de 1.5)
        - Break probability IGNORÉE (on veut de la MARGE, pas casser)
        """
        details = {}
        warnings = []
        score = 100.0

        # 1. ATR pour SL
        atr = self.safe_float(ad.get("atr_14"))
        natr = self.safe_float(ad.get("natr"))

        if atr <= 0 and natr <= 0:
            return ValidationResult(
                level=ValidationLevel.RISK_MANAGEMENT,
                passed=False,
                score=0.0,
                reason="❌ ATR indisponible - Impossible de calculer SL",
                details={},
                warnings=[],
            )

        if atr > 0 and current_price > 0:
            atr_percent = atr / current_price
        elif natr > 0:
            atr_percent = natr / 100.0
        else:
            atr_percent = 0.01

        details["atr_percent"] = atr_percent * 100

        # 2. Support/Résistance pour R/R
        nearest_support = self.safe_float(ad.get("nearest_support"))
        nearest_resistance = self.safe_float(ad.get("nearest_resistance"))

        # SL distance
        if nearest_support > 0 and current_price > nearest_support:
            sl_dist = max(0.008, (current_price - nearest_support) / current_price)  # Min 0.8%
            sl_basis = "support"
        else:
            sl_dist = max(0.008, atr_percent * 0.8)  # 0.8 ATR
            sl_basis = "ATR"
            warnings.append("⚠️ Pas de support identifié - SL basé sur ATR")
            score -= 10

        details["sl_distance_pct"] = sl_dist * 100
        details["sl_basis"] = sl_basis

        # TP distance - CONSERVATIVE pour validation
        # TP1 ultra-conservateur: 0.6 ATR (pas 0.8)
        tp_dist_conservative = max(0.008, atr_percent * 0.6)

        details["tp1_distance_pct"] = tp_dist_conservative * 100

        # R/R Ratio - STRICT
        rr_ratio = tp_dist_conservative / sl_dist if sl_dist > 0 else 0

        # Minimum R/R 1.8 (augmenté de 1.5)
        if rr_ratio < 1.8:
            rr_penalty = max(0, min((1.8 - rr_ratio) * 40, 60))
            score -= rr_penalty

            if rr_ratio < 1.2:
                warnings.append(
                    f"⚠️⚠️ R/R TRÈS FAIBLE: {rr_ratio:.2f} < 1.8 (-{rr_penalty:.0f}pts)"
                )
                details["rr_critical_warning"] = True
            else:
                warnings.append(
                    f"⚠️ R/R sous-optimal: {rr_ratio:.2f} < 1.8 (-{rr_penalty:.0f}pts)"
                )

        details["rr_ratio"] = rr_ratio

        if rr_ratio > 3.0:
            details["rr_quality"] = "Excellent"
        elif rr_ratio > 2.5:
            details["rr_quality"] = "Très bon"
        elif rr_ratio > 2.0:
            details["rr_quality"] = "Bon"
        else:
            details["rr_quality"] = "Acceptable"

        # 3. Vérifier résistance - STRICT
        if nearest_resistance > 0 and current_price > 0:
            res_dist_pct = ((nearest_resistance - current_price) / current_price) * 100

            # Distance résistance >2% REQUISE
            if res_dist_pct < 1.0:
                # <1% = REJET
                return ValidationResult(
                    level=ValidationLevel.RISK_MANAGEMENT,
                    passed=False,
                    score=0.0,
                    reason=f"❌ RÉSISTANCE TROP PROCHE: {res_dist_pct:.1f}% < 1% - Déjà au plafond",
                    details={"resistance_distance_pct": res_dist_pct},
                    warnings=[f"❌ Résistance collée à {res_dist_pct:.1f}% - TROP TARD"],
                )
            elif res_dist_pct < 2.0:  # noqa: RET505
                # 1-2% = Pénalité lourde
                score -= 30
                warnings.append(
                    f"⚠️⚠️ Résistance TRÈS proche ({res_dist_pct:.1f}%) - Risque rejet élevé"
                )
            elif res_dist_pct < 3.0:
                score -= 15
                warnings.append(
                    f"⚠️ Résistance proche ({res_dist_pct:.1f}%) - Marge limitée"
                )

            details["resistance_distance_pct"] = res_dist_pct

        # 4. Break probability - IGNORÉE (on veut de la MARGE)
        # Break probability haute = on est PROCHE résistance = mauvais pour early entry
        # On préfère être LOIN de la résistance

        # Validation passée si score >60 (augmenté de 50)
        passed = score >= 60

        return ValidationResult(
            level=ValidationLevel.RISK_MANAGEMENT,
            passed=passed,
            score=score,
            reason=(
                "✅ Gestion risque OK"
                if passed
                else f"❌ Gestion risque insuffisante (score {score:.0f}/100)"
            ),
            details=details,
            warnings=warnings,
        )

    def _validate_entry_timing_early(self, ad: dict) -> ValidationResult:
        """
        Level 4: Entry Timing - CORRECTED pour EARLY entry.

        CHANGEMENTS:
        - RSI >70 = REJET (overbought = trop tard)
        - MFI >70 = REJET (argent déjà entré)
        - Volume spike >3x = REJET (pic déjà atteint)
        - Stochastic >80 = REJET
        - Pas de tolérance pump (on veut acheter AVANT)
        """
        details: dict[str, bool | list[str] | float] = {}
        warnings: list[str] = []
        score = 100.0

        # PAS DE TOLÉRANCE PUMP - on veut acheter AVANT, pas pendant

        # 1. Overbought Check - STRICT
        rsi = self.safe_float(ad.get("rsi_14"))
        mfi = self.safe_float(ad.get("mfi_14"))
        stoch_k = self.safe_float(ad.get("stoch_k"))
        stoch_d = self.safe_float(ad.get("stoch_d"))

        overbought_issues = []

        # RSI >70 = REJET
        if rsi > 70:
            return ValidationResult(
                level=ValidationLevel.ENTRY_TIMING,
                passed=False,
                score=0.0,
                reason=f"❌ RSI OVERBOUGHT: {rsi:.0f} > 70 - TROP TARD pour acheter",
                details={"rsi": rsi},
                warnings=[f"❌ RSI {rsi:.0f} = mouvement déjà fait"],
            )
        elif rsi > 65:  # noqa: RET505
            overbought_issues.append(f"RSI élevé ({rsi:.0f})")
            score -= 20
            warnings.append(f"⚠️ RSI {rsi:.0f} - mouvement bien avancé")

        # MFI >70 = REJET
        if mfi > 70:
            return ValidationResult(
                level=ValidationLevel.ENTRY_TIMING,
                passed=False,
                score=0.0,
                reason=f"❌ MFI OVERBOUGHT: {mfi:.0f} > 70 - Argent déjà entré massivement",
                details={"mfi": mfi},
                warnings=[f"❌ MFI {mfi:.0f} = flux acheteur épuisé"],
            )
        elif mfi > 65:  # noqa: RET505
            overbought_issues.append(f"MFI élevé ({mfi:.0f})")
            score -= 18
            warnings.append(f"⚠️ MFI {mfi:.0f} - flux acheteur déjà important")

        # Stochastic >80 = REJET
        if stoch_k > 80 or stoch_d > 80:
            return ValidationResult(
                level=ValidationLevel.ENTRY_TIMING,
                passed=False,
                score=0.0,
                reason=f"❌ STOCHASTIC OVERBOUGHT: K={stoch_k:.0f} D={stoch_d:.0f} > 80 - TROP TARD",
                details={"stoch_k": stoch_k, "stoch_d": stoch_d},
                warnings=[f"❌ Stochastic {stoch_k:.0f}/{stoch_d:.0f} = déjà overbought"],
            )
        elif stoch_k > 75 or stoch_d > 75:  # noqa: RET505
            overbought_issues.append(f"Stoch proche saturation ({stoch_k:.0f}/{stoch_d:.0f})")
            score -= 15

        if overbought_issues:
            details["overbought_issues"] = overbought_issues

        details["rsi"] = rsi
        details["mfi"] = mfi

        # 2. Volume Confirmation - STRICT
        rel_volume = self.safe_float(ad.get("relative_volume"), 1.0)
        vol_spike = self.safe_float(ad.get("volume_spike_multiplier"), 1.0)
        obv_osc = self.safe_float(ad.get("obv_oscillator"))
        vol_context = ad.get("volume_context", "").upper()

        # Volume spike >3x = REJET (pic déjà atteint)
        if vol_spike > 3.0 or rel_volume > 3.5:
            return ValidationResult(
                level=ValidationLevel.ENTRY_TIMING,
                passed=False,
                score=0.0,
                reason=f"❌ VOLUME SPIKE: {vol_spike:.1f}x > 3x - PIC volume déjà atteint, TROP TARD",
                details={"volume_spike_multiplier": vol_spike, "relative_volume": rel_volume},
                warnings=[f"❌ Volume spike {vol_spike:.1f}x = distribution en cours"],
            )

        # Volume faible = pénalité progressive
        if rel_volume < 0.8:
            vol_penalty = max(0, min((0.8 - rel_volume) * 50, 40))
            score -= vol_penalty

            if rel_volume < 0.5:
                warnings.append(
                    f"⚠️⚠️ VOLUME TRÈS FAIBLE: {rel_volume:.2f}x (-{vol_penalty:.0f}pts)"
                )
            else:
                warnings.append(
                    f"⚠️ Volume faible: {rel_volume:.2f}x (-{vol_penalty:.0f}pts)"
                )
        elif rel_volume < 1.0:
            warnings.append(f"⚠️ Volume modéré: {rel_volume:.2f}x")
            score -= 10

        # OBV négatif fort = warning
        if obv_osc < -150:
            warnings.append(f"⚠️ OBV négatif: {obv_osc:.0f}")
            score -= 15

        # Contexts baissiers
        if vol_context in ["REVERSAL_PATTERN", "DEEP_OVERSOLD"]:
            warnings.append(f"⚠️ Volume context défavorable: {vol_context}")
            score -= 15

        details["volume_context"] = vol_context
        details["relative_volume"] = rel_volume

        # 3. Momentum Check
        macd_trend = ad.get("macd_trend", "").upper()
        momentum_score_val = self.safe_float(ad.get("momentum_score"))

        if macd_trend == "BEARISH":
            warnings.append("⚠️ MACD baissier")
            score -= 18
        elif macd_trend == "NEUTRAL":
            warnings.append("⚠️ MACD neutre")
            score -= 8

        if momentum_score_val < 40:
            warnings.append(f"⚠️ Momentum score faible: {momentum_score_val:.0f}")
            score -= 12

        details["macd_trend"] = macd_trend
        details["momentum_score"] = momentum_score_val

        # 4. Pattern Check
        pattern = ad.get("pattern_detected", "").upper()

        if pattern in ["PRICE_SPIKE_DOWN", "LIQUIDITY_SWEEP"]:
            warnings.append(f"⚠️ Pattern baissier: {pattern}")
            score -= 15

        details["pattern"] = pattern

        # Validation passée si score >70 (augmenté de 65)
        passed = score >= 70

        return ValidationResult(
            level=ValidationLevel.ENTRY_TIMING,
            passed=passed,
            score=score,
            reason=(
                "✅ Timing OK pour EARLY entry"
                if passed
                else f"❌ Timing défavorable (score {score:.0f}/100)"
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
            recommendations=["❌ Validation échouée - Conditions EARLY entry non réunies"],
        )
