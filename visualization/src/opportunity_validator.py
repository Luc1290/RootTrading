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
    """
    Validateur multi-niveaux pour opportunités de trading.

    Chaque niveau doit passer pour autoriser le trade.
    Validation stricte pour protéger le capital.
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
        higher_tf_data: dict | None = None,
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
        level_results[ValidationLevel.DATA_QUALITY] = self._validate_data_quality(
            analyzer_data)
        if not level_results[ValidationLevel.DATA_QUALITY].passed:
            blocking_issues.append(
                level_results[ValidationLevel.DATA_QUALITY].reason)
            return self._create_failed_summary(level_results, blocking_issues)

        # Level 2: Market Conditions
        level_results[ValidationLevel.MARKET_CONDITIONS] = (
            self._validate_market_conditions(analyzer_data, higher_tf_data)
        )
        if not level_results[ValidationLevel.MARKET_CONDITIONS].passed:
            blocking_issues.append(
                level_results[ValidationLevel.MARKET_CONDITIONS].reason
            )

        # Level 3: Risk Management
        level_results[ValidationLevel.RISK_MANAGEMENT] = self._validate_risk_management(
            analyzer_data, current_price)
        if not level_results[ValidationLevel.RISK_MANAGEMENT].passed:
            blocking_issues.append(
                level_results[ValidationLevel.RISK_MANAGEMENT].reason
            )

        # Level 4: Entry Timing
        level_results[ValidationLevel.ENTRY_TIMING] = self._validate_entry_timing(
            analyzer_data)
        if not level_results[ValidationLevel.ENTRY_TIMING].passed:
            blocking_issues.append(
                level_results[ValidationLevel.ENTRY_TIMING].reason)

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
            recommendations.append(
                f"Score validation: {overall_score:.0f}/100")
        else:
            recommendations.append("❌ Validation échouée")
            recommendations.append(
                f"Problèmes à résoudre: {len(blocking_issues)}")

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
        data_quality = ad.get("data_quality", "").upper()
        if data_quality not in ["EXCELLENT", "GOOD"]:
            # Message explicite selon le cas
            if not data_quality or data_quality == "":
                quality_msg = "champ 'data_quality' absent ou vide dans analyzer_data"
            else:
                quality_msg = f"qualité '{data_quality}' non acceptable (requis: EXCELLENT ou GOOD)"

            return ValidationResult(
                level=ValidationLevel.DATA_QUALITY,
                passed=False,
                score=0.0,
                reason=f"❌ Qualité données insuffisante: {quality_msg}",
                details={"data_quality": data_quality or "N/A"},
                warnings=[],
            )

        details["data_quality"] = data_quality

        # 2. Anomalies
        anomaly = ad.get("anomaly_detected", False)
        if anomaly:
            warnings.append("⚠️ Anomalie détectée dans les données")
            score -= 20

        details["anomaly_detected"] = anomaly

        # 3. Indicateurs critiques présents
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
            # Considérer comme manquant seulement si None ou clé absente (pas
            # si 0 ou False)
            if val is None and ind not in ad:
                missing.append(ind)

        if missing:
            score -= len(missing) * 10
            warnings.append(f"⚠️ Indicateurs manquants: {', '.join(missing)}")

        details["missing_indicators"] = missing
        details["indicators_present"] = len(critical_indicators) - len(missing)

        # 4. Cache hit ratio (performance)
        cache_ratio = self.safe_float(ad.get("cache_hit_ratio"))
        if cache_ratio > 0:
            details["cache_hit_ratio"] = cache_ratio

        # Validation passée si score >70
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

    def _validate_market_conditions(
        self, ad: dict, higher_tf: dict | None
    ) -> ValidationResult:
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
        regime = ad.get("market_regime", "").upper()
        regime_conf = self.safe_float(ad.get("regime_confidence"))

        # CHANGEMENT: Malus progressif au lieu de rejet binaire
        # Un setup excellent peut justifier une contre-tendance
        if regime in ["TRENDING_BEAR", "BREAKOUT_BEAR"]:
            score -= 40  # Malus lourd mais pas bloquant
            warnings.append(
                f"⚠️⚠️ RÉGIME BAISSIER: {regime} - Contre-tendance très risquée!"
            )
            details["regime_bear_warning"] = True
        elif regime == "TRANSITION":
            score -= 10
            warnings.append("⚠️ Régime transition - Direction incertaine")

        # Confiance régime
        if regime_conf < 50:
            warnings.append(f"⚠️ Confiance régime faible: {regime_conf:.0f}%")
            score -= 15

        details["regime"] = regime
        details["regime_confidence"] = regime_conf

        # 2. Volatilité
        vol_regime = ad.get("volatility_regime", "").lower()
        if vol_regime == "extreme":
            warnings.append("⚠️ Volatilité extrême - Risque élevé")
            score -= 20
        elif vol_regime == "low":
            warnings.append("⚠️ Volatilité faible - Peu de mouvement attendu")
            score -= 10

        details["volatility_regime"] = vol_regime

        # 3. Directional Bias
        bias = ad.get("directional_bias", "").upper()
        # CHANGEMENT: Malus progressif au lieu de rejet binaire
        if bias == "BEARISH":
            score -= 30  # Malus mais pas bloquant
            warnings.append(
                f"⚠️⚠️ BIAIS BAISSIER: {bias} - Setup contre-tendance!")
            details["bias_bear_warning"] = True
        elif bias == "NEUTRAL":
            warnings.append("⚠️ Biais neutre - Pas de direction claire")
            score -= 10

        details["directional_bias"] = bias

        # 4. Higher Timeframe Validation
        if higher_tf:
            htf_regime = higher_tf.get("market_regime", "").upper()
            htf_rsi = self.safe_float(higher_tf.get("rsi_14"))
            htf_macd = higher_tf.get("macd_trend", "").upper()

            # CHANGEMENT: Malus progressif au lieu de rejet binaire
            # 5m bearish n'empêche pas un trade 1m si setup excellent
            if htf_regime in ["TRENDING_BEAR", "BREAKOUT_BEAR"]:
                score -= 35  # Malus lourd
                warnings.append(
                    f"⚠️⚠️ 5M BAISSIER: {htf_regime} - Trade contre-tendance HTF!"
                )
                details["htf_bear_warning"] = True

            # Warning si divergence
            if htf_rsi < 50 and htf_macd == "BEARISH":
                warnings.append("⚠️ Timeframe supérieur montre faiblesse")
                score -= 15

            details["htf_aligned"] = True
            details["htf_regime"] = htf_regime
        else:
            warnings.append("ℹ️ Pas de validation timeframe supérieur")
            score -= 5

        # Validation passée si score >30 (abaissé de 60 pour tolérer contre-tendances)
        # Avec malus progressifs: Bear -40, Bias Bear -30, HTF Bear -35 = score peut descendre à 0-30
        # Score 35-50 = contre-tendance risquée mais acceptable si reste du
        # setup excellent
        passed = score >= 30

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

    def _validate_risk_management(
        self, ad: dict, current_price: float
    ) -> ValidationResult:
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

        # Calculer ATR%
        if atr > 0 and current_price > 0:
            atr_percent = atr / current_price
        elif natr > 0:
            atr_percent = natr / 100.0
        else:
            atr_percent = 0.01  # Fallback 1%

        details["atr_percent"] = atr_percent * 100

        # 2. Support/Résistance pour R/R
        nearest_support = self.safe_float(ad.get("nearest_support"))
        nearest_resistance = self.safe_float(ad.get("nearest_resistance"))

        # Calculer SL distance
        if nearest_support > 0 and current_price > nearest_support:
            sl_dist = max(
                0.007,
                (current_price -
                 nearest_support) /
                current_price)
            sl_basis = "support"
        else:
            sl_dist = max(0.007, atr_percent * 0.7)
            sl_basis = "ATR"
            warnings.append("⚠️ Pas de support identifié - SL basé sur ATR")
            score -= 10

        details["sl_distance_pct"] = sl_dist * 100
        details["sl_basis"] = sl_basis

        # Calculer TP distance (utiliser multiplier plus agressif pour validation)
        # TP1 conservateur: 0.8 ATR, TP2 modéré: 1.2 ATR
        # Pour validation, on vérifie que TP2 (moderate) donne un R/R
        # acceptable
        tp_dist_conservative = max(0.01, atr_percent * 0.8)
        tp_dist_moderate = max(0.015, atr_percent * 1.2)

        details["tp1_distance_pct"] = tp_dist_conservative * 100
        details["tp2_distance_pct"] = tp_dist_moderate * 100

        # R/R Ratio - Utiliser TP2 (moderate) pour validation
        # Cela permet de valider les setups où TP2 est atteignable
        rr_ratio = tp_dist_moderate / sl_dist if sl_dist > 0 else 0

        # CHANGEMENT: Malus progressif au lieu de rejet binaire
        # R/R 1.4 acceptable, R/R 0.8 très mauvais
        if rr_ratio < 1.5:
            # Malus proportionnel : RR 1.4 = -3pts, RR 1.0 = -15pts, RR 0.5 =
            # -30pts
            rr_penalty = max(0, min((1.5 - rr_ratio) * 30, 50))
            score -= rr_penalty

            if rr_ratio < 1.0:
                warnings.append(
                    f"⚠️⚠️ R/R TRÈS FAIBLE: {rr_ratio:.2f} < 1.5 (-{rr_penalty:.0f}pts)"
                )
                details["rr_critical_warning"] = True
            else:
                warnings.append(
                    f"⚠️ R/R sous-optimal: {rr_ratio:.2f} < 1.5 (-{rr_penalty:.0f}pts)"
                )
                details["rr_suboptimal"] = True

        details["rr_ratio"] = rr_ratio

        # Bonus si R/R excellent
        if rr_ratio > 3.0:
            details["rr_quality"] = "Excellent"
        elif rr_ratio > 2.0:
            details["rr_quality"] = "Bon"
        else:
            details["rr_quality"] = "Acceptable"

        # 3. Vérifier résistance
        if nearest_resistance > 0 and current_price > 0:
            res_dist_pct = (
                (nearest_resistance - current_price) / current_price) * 100

            # Récupérer momentum_score pour contextualiser
            momentum_score = self.safe_float(ad.get("momentum_score"), 50)
            logger.info(
                f"Résistance {res_dist_pct:.1f}% - momentum_score={momentum_score:.1f}"
            )

            # SCALPING/DAY TRADING: Résistances <2% sont du bruit, pénalités minimales
            # On ne pénalise vraiment QUE si momentum très faible + résistance
            # proche
            if res_dist_pct < 1.0:
                if momentum_score < 45:
                    # Momentum TRÈS faible = risque réel de rejet
                    warning_msg = f"⚠️ Momentum très faible ({momentum_score:.0f}) vs résistance {res_dist_pct:.1f}%"
                    warnings.append(warning_msg)
                    logger.info(f"WARNING AJOUTÉ: {warning_msg}")
                    score -= 8  # Réduit de -15 à -8
                elif momentum_score < 55:
                    # Momentum faible mais acceptable en scalping
                    warning_msg = f"ℹ️ Momentum {momentum_score:.0f} moyen, résistance {res_dist_pct:.1f}% franchissable"
                    warnings.append(warning_msg)
                    logger.info(f"INFO AJOUTÉ: {warning_msg}")
                    score -= 3  # Réduit de -15/-10 à -3
                else:
                    # Momentum OK = setup breakout
                    warning_msg = f"✅ Setup breakout: Momentum {momentum_score:.0f} vs résistance {res_dist_pct:.1f}%"
                    warnings.append(warning_msg)
                    logger.info(f"INFO AJOUTÉ: {warning_msg}")
                    score -= 1  # Quasi aucune pénalité
            elif res_dist_pct < 2.0:
                # Entre 1-2%: pénalité uniquement si momentum vraiment faible
                if momentum_score < 40:
                    warnings.append(
                        f"⚠️ Momentum faible ({momentum_score:.0f}), résistance à {res_dist_pct:.1f}%"
                    )
                    score -= 5
                else:
                    # Sinon on ignore, c'est trop loin pour du scalping
                    score -= 1

            details["resistance_distance_pct"] = res_dist_pct

            # SUPPRIMER le warning "TP au-delà résistance" - c'est le but d'un breakout!
            # On veut justement que le TP soit au-delà pour capturer le mouvement
            # if tp_dist_moderate * 100 > res_dist_pct:
            #     warnings.append("⚠️ TP au-delà de la résistance")
            #     score -= 10

        # 4. Break probability
        break_prob = self.safe_float(ad.get("break_probability"))
        if break_prob > 0:
            if break_prob < 0.35:  # Abaissé de 0.4 à 0.35
                warnings.append(
                    f"⚠️ Probabilité cassure faible: {break_prob*100:.0f}%")
                score -= 10
            elif break_prob < 0.45:
                # Zone intermédiaire: warning informatif seulement
                warnings.append(
                    f"ℹ️ Break probability modérée: {break_prob*100:.0f}%")
                score -= 3

            details["break_probability"] = break_prob * 100

        # Validation passée si score >50 (abaissé de 70 pour tolérer R/R suboptimal)
        # Avec scoring progressif, un score de 60 = R/R 1.2-1.3 acceptable
        passed = score >= 50

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

    def _validate_entry_timing(self, ad: dict) -> ValidationResult:
        """
        Level 4: Validation timing d'entrée.

        Vérifie:
        - Pas de surachat extrême (RSI, MFI, Stochastic)
        - Volume confirmant
        - Momentum positif
        - Pattern favorable ou neutre
        - Pas de divergence négative

        EXCEPTION PUMP: Tolérance RSI/MFI élevés si pump validé
        """
        details = {}
        warnings = []
        score = 100.0

        # Détecter si c'est un pump validé (volume fort + regime bull + context
        # breakout)
        vol_spike = self.safe_float(ad.get("volume_spike_multiplier"), 1.0)
        rel_volume = self.safe_float(ad.get("relative_volume"), 1.0)
        market_regime = ad.get("market_regime", "").upper()
        vol_context = ad.get("volume_context", "").upper()

        # AJUSTÉ: 2.0x (compromis 20 cryptos: P95 varie de 1.4x à 8.3x)
        is_pump_context = (
            (vol_spike > 2.0 or rel_volume > 2.0) and market_regime in [
                "TRENDING_BULL",
                "BREAKOUT_BULL"] and vol_context in [
                "CONSOLIDATION_BREAK",
                "BREAKOUT",
                "PUMP_START",
                "HIGH_VOLATILITY"])

        # 1. Overbought Check (avec tolérance pump)
        rsi = self.safe_float(ad.get("rsi_14"))
        mfi = self.safe_float(ad.get("mfi_14"))
        stoch_k = self.safe_float(ad.get("stoch_k"))
        stoch_d = self.safe_float(ad.get("stoch_d"))

        overbought_issues = []

        # Seuils adaptatifs selon contexte
        if is_pump_context:
            # Pendant un pump validé, tolérer RSI/MFI plus élevés
            rsi_extreme_threshold = 95
            rsi_high_threshold = 85
            mfi_extreme_threshold = 95
            mfi_high_threshold = 85
            details["pump_context_detected"] = True
        else:
            # Conditions normales, seuils stricts
            rsi_extreme_threshold = 80
            rsi_high_threshold = 75
            mfi_extreme_threshold = 85
            mfi_high_threshold = 80
            details["pump_context_detected"] = False

        if rsi > rsi_extreme_threshold:
            overbought_issues.append(f"RSI extrême ({rsi:.0f})")
            score -= 20
        elif rsi > rsi_high_threshold:
            warnings.append(f"⚠️ RSI élevé ({rsi:.0f})")
            score -= 10

        if mfi > mfi_extreme_threshold:
            overbought_issues.append(f"MFI extrême ({mfi:.0f})")
            score -= 15
        elif mfi > mfi_high_threshold:
            warnings.append(f"⚠️ MFI élevé ({mfi:.0f})")
            score -= 10

        if stoch_k > 90 and stoch_d > 90:
            overbought_issues.append(
                f"Stochastic saturé ({stoch_k:.0f}/{stoch_d:.0f})")
            score -= 15

        if overbought_issues:
            details["overbought_issues"] = overbought_issues

        details["rsi"] = rsi
        details["mfi"] = mfi

        # 2. Volume Confirmation (variables déjà déclarées au début)
        obv_osc = self.safe_float(ad.get("obv_oscillator"))

        # CHANGEMENT: Malus progressif au lieu de rejet binaire
        # Volume 0.4x sur un setup excellent peut passer si tout le reste
        # compense
        if rel_volume < 0.5:
            # Malus proportionnel : 0.4x = -10pts, 0.3x = -20pts, 0.2x =
            # -30pts, 0.1x = -40pts
            vol_penalty = max(0, min((0.5 - rel_volume) * 100, 50))
            score -= vol_penalty

            if rel_volume < 0.3:
                warnings.append(
                    f"⚠️⚠️ VOLUME TRÈS FAIBLE: {rel_volume:.2f}x (-{vol_penalty:.0f}pts)"
                )
                details["volume_critical_warning"] = True
            else:
                warnings.append(
                    f"⚠️ Volume faible: {rel_volume:.2f}x (-{vol_penalty:.0f}pts)"
                )
                details["volume_low_warning"] = True

        # Warning si volume modéré (0.5-0.8x)
        elif rel_volume < 0.8:
            warnings.append(f"⚠️ Volume modéré: {rel_volume:.2f}x")
            score -= 10

        # OBV négatif fort = warning
        if obv_osc < -200:
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
            score -= 15
        elif macd_trend == "NEUTRAL":
            warnings.append("⚠️ MACD neutre")
            score -= 5

        if momentum_score_val < 40:
            warnings.append(
                f"⚠️ Momentum score faible: {momentum_score_val:.0f}")
            score -= 10

        details["macd_trend"] = macd_trend
        details["momentum_score"] = momentum_score_val

        # 4. Pattern Check
        pattern = ad.get("pattern_detected", "").upper()

        # IMPORTANT: PRICE_SPIKE_DOWN peut être un pullback sain pendant un pump
        # LIQUIDITY_SWEEP est OK si on est en TRENDING_BULL (sweep des shorts)
        # Ne pénaliser que si vraiment baissier (pas de pump context)
        if pattern == "PRICE_SPIKE_DOWN" and not is_pump_context:
            warnings.append(f"⚠️ Pattern baissier: {pattern}")
            score -= 15
        elif pattern == "LIQUIDITY_SWEEP" and market_regime not in [
            "TRENDING_BULL",
            "BREAKOUT_BULL",
        ]:
            warnings.append(f"⚠️ Pattern baissier: {pattern}")
            score -= 10

        details["pattern"] = pattern

        # 5. Divergence Check
        stoch_div = ad.get("stoch_divergence", False)
        if stoch_div:
            # Divergence peut être bullish ou bearish, need more info
            warnings.append("ℹ️ Divergence Stochastic détectée")

        # Validation passée si score >65
        passed = score >= 65

        return ValidationResult(
            level=ValidationLevel.ENTRY_TIMING,
            passed=passed,
            score=score,
            reason=(
                "✅ Timing OK"
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
            recommendations=["❌ Validation échouée au premier niveau"],
        )
