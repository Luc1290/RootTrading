"""
Opportunity Calculator PRO FIXED - Professional Trading System CORRIGÉ
Orchestre scoring + validation CORRIGÉS pour EARLY ENTRY (acheter AVANT le pump)

CORRECTIONS MAJEURES:
1. Import des modules FIXED avec logique inversée
2. Décision STRICTE: Rejette overbought, volume spikes, momentum élevé
3. Early detection PRIORITAIRE: Signaux early peuvent trigger entry même avec score modéré
4. Risk management CONSERVATEUR: R/R minimum 1.8, position sizing prudent

Architecture CORRIGÉE:
1. OpportunityEarlyDetector FIXED: Détection AVANT pump (ROC faible, volume buildup)
2. OpportunityScoring FIXED: Score setup formation (RSI 35-55, pas 60-75)
3. OpportunityValidator FIXED: Validation STRICTE (rejette overbought/spikes)
4. Décision finale: BUY_NOW / BUY_DCA / WAIT / AVOID (avec rejet strict conditions tardives)

Version: 3.0 - True Early Entry System (FIXED)
"""

import logging
from dataclasses import asdict, dataclass

from src.opportunity_early_detector import (
    EarlySignal,
    EarlySignalLevel,
    OpportunityEarlyDetector,
)
from src.opportunity_scoring import OpportunityScore, OpportunityScoring
from src.opportunity_validator import (
    OpportunityValidator,
    ValidationSummary,
)

logger = logging.getLogger(__name__)


@dataclass
class TradingOpportunity:
    """Opportunité de trading complète."""

    symbol: str
    action: str  # BUY_NOW, BUY_DCA, WAIT, AVOID
    confidence: float  # 0-100
    score: OpportunityScore
    validation: ValidationSummary

    # Early Warning
    early_signal: EarlySignal | None
    is_early_entry: bool

    # Pricing
    current_price: float
    entry_price_optimal: float
    entry_price_aggressive: float

    # Targets
    tp1: float
    tp1_percent: float
    tp2: float
    tp2_percent: float
    tp3: float | None
    tp3_percent: float | None

    # Stop Loss
    stop_loss: float
    stop_loss_percent: float
    stop_loss_basis: str

    # Risk Management
    rr_ratio: float
    risk_level: str
    max_position_size_pct: float

    # Timing
    estimated_hold_time: str
    entry_urgency: str

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
    raw_analyzer_data: dict


class OpportunityCalculatorPro:
    """
    Calculateur professionnel d'opportunités CORRIGÉ.

    CHANGEMENTS vs ancien système:
    - Utilise modules FIXED avec logique inversée (early entry, pas late entry)
    - Rejette strictement conditions overbought/spike
    - Priorité aux signaux early detector
    - Risk management conservateur (R/R 1.8+ minimum)
    """

    def __init__(self, enable_early_detection: bool = True):
        """Initialise le calculateur professionnel CORRIGÉ.

        Args:
            enable_early_detection: Active le système early warning (défaut: True)
        """
        self.scorer = OpportunityScoring()
        self.validator = OpportunityValidator()
        self.early_detector = (
            OpportunityEarlyDetector() if enable_early_detection else None
        )

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
        higher_tf_data: dict | None = None,
        _signals_data: dict | None = None,
        historical_data: list[dict] | None = None,
    ) -> TradingOpportunity:
        """
        Calcule une opportunité de trading complète CORRIGÉE pour EARLY ENTRY.

        Args:
            symbol: Symbole (e.g. BTCUSDC)
            current_price: Prix actuel
            analyzer_data: analyzer_data complet (108 indicateurs)
            higher_tf_data: Données timeframe supérieur pour validation
            signals_data: (optionnel) Données signaux externes
            historical_data: Liste des 5-10 dernières périodes pour early detection

        Returns:
            TradingOpportunity avec action BUY uniquement si AVANT le pump
        """
        if not analyzer_data:
            return self._create_no_data_opportunity(symbol, current_price)

        # === ÉTAPE 0: EARLY DETECTION (PRIORITAIRE) ===
        early_signal = None
        is_early_entry = False

        if self.early_detector and historical_data:
            early_signal = self.early_detector.detect_early_opportunity(
                current_data=analyzer_data, historical_data=historical_data
            )

            # Early signal ENTRY_NOW/PREPARE avec score 45+ = entry early confirmée
            if early_signal.level in [
                EarlySignalLevel.ENTRY_NOW,
                EarlySignalLevel.PREPARE,
            ] and early_signal.score >= 45:
                is_early_entry = True
                logger.info(
                    f"🚀 Early entry detected for {symbol}: {early_signal.level.value} (score: {early_signal.score:.0f})"
                )

        # === ÉTAPE 1: SCORING CORRIGÉ ===
        score = self.scorer.calculate_opportunity_score(analyzer_data, current_price)

        # === ÉTAPE 2: VALIDATION STRICTE ===
        validation = self.validator.validate_opportunity(
            analyzer_data, current_price, higher_tf_data
        )

        # === ÉTAPE 3: DÉCISION STRICTE (rejette late entries) ===
        action, confidence, reasons, warnings, recommendations = self._make_decision(
            score, validation, analyzer_data, early_signal, is_early_entry
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

        # === ÉTAPE 7: RISK MANAGEMENT CONSERVATEUR ===
        rr_ratio, risk_level, max_position_pct = self._calculate_risk_metrics(
            current_price, tp1, stop_loss, score, validation, analyzer_data, action
        )

        # === ÉTAPE 8: TIMING ===
        hold_time, urgency = self._calculate_timing(analyzer_data, score, action)

        # === ÉTAPE 9: CONTEXT ===
        market_regime = analyzer_data.get("market_regime", "UNKNOWN")
        volume_context = analyzer_data.get("volume_context", "UNKNOWN")
        volatility_regime = analyzer_data.get("volatility_regime", "UNKNOWN")

        return TradingOpportunity(
            symbol=symbol,
            action=action,
            confidence=confidence,
            score=score,
            validation=validation,
            early_signal=early_signal,
            is_early_entry=is_early_entry,
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
            raw_score_details=asdict(score),
            raw_validation_details=asdict(validation),
            raw_analyzer_data=analyzer_data,
        )

    def _make_decision(  # noqa: PLR0911 - Multiple returns acceptable for decision logic
        self,
        score: OpportunityScore,
        validation: ValidationSummary,
        analyzer_data: dict,
        early_signal: EarlySignal | None,
        is_early_entry: bool,
    ) -> tuple[str, float, list[str], list[str], list[str]]:
        """
        Prend décision finale CORRIGÉE (rejette late entries strictement).

        ❌ ANCIEN SYSTÈME: Acceptait RSI 60-75, volume spikes
        ✅ NOUVEAU SYSTÈME: Rejette RSI >70, volume spike >3x, momentum élevé

        Returns:
            (action, confidence, reasons, warnings, recommendations)
        """
        reasons = []
        warnings = []

        # === REJET STRICT CONDITIONS LATE ===
        rsi = self.safe_float(analyzer_data.get("rsi_14"))
        vol_spike = self.safe_float(analyzer_data.get("volume_spike_multiplier"), 1.0)
        rel_volume = self.safe_float(analyzer_data.get("relative_volume"), 1.0)
        roc_10 = self.safe_float(analyzer_data.get("roc_10"))

        # REJET: Overbought
        if rsi > 70:
            return (
                "AVOID",
                0.0,
                [f"❌ RSI OVERBOUGHT: {rsi:.0f} - TROP TARD pour entry"],
                ["Mouvement déjà avancé, risque correction"],
                ["Attendre pullback ou prochaine opportunité"],
            )

        # REJET: Volume spike (pic atteint)
        if vol_spike >= 3.0 or rel_volume > 3.0:
            return (
                "AVOID",
                0.0,
                [
                    f"❌ VOLUME SPIKE: {vol_spike:.1f}x - PIC ATTEINT",
                    "Mouvement déjà explosé",
                ],
                ["Entry tardive = high risk"],
                ["Attendre consolidation"],
            )

        # REJET: Momentum trop élevé
        if roc_10 > 0.8:
            return (
                "AVOID",
                0.0,
                [
                    f"❌ MOMENTUM ÉLEVÉ: ROC {roc_10*100:.2f}% - Déjà accéléré",
                    "Entry late = chasing",
                ],
                ["Risque d'acheter au top"],
                ["Laisser passer, attendre setup"],
            )

        # === VALIDATION STRICTE ===
        if not validation.all_passed:
            failed_msg = " | ".join(validation.blocking_issues[:3]) if validation.blocking_issues else "Validation failed"
            return (
                "AVOID",
                0.0,
                [f"Validation echouee: {failed_msg}"],
                validation.warnings,
                ["Corriger problemes avant entry"],
            )

        # === DÉCISION BASÉE SUR EARLY SIGNAL ===
        # Early signal ENTRY_NOW avec score decent = BUY_NOW
        if is_early_entry and early_signal:
            if early_signal.level == EarlySignalLevel.ENTRY_NOW:
                confidence = min(85.0, early_signal.confidence)
                reasons.extend(
                    [
                        f"🚀 EARLY ENTRY WINDOW: {early_signal.level.value}",
                        f"Early score: {early_signal.score:.0f}/100",
                        f"Setup score: {score.total_score:.0f}/100",
                    ]
                )
                reasons.extend(early_signal.reasons[:3])  # Top 3 reasons

                return (
                    "BUY_NOW",
                    confidence,
                    reasons,
                    [*early_signal.warnings, *warnings],
                    ["Entry IMMÉDIATE recommandée", "Setup early confirmé", *early_signal.recommendations[:2]],
                )

            if early_signal.level == EarlySignalLevel.PREPARE:
                confidence = min(75.0, early_signal.confidence)
                reasons.extend(
                    [
                        f"⚡ PRÉPARER ENTRY: {early_signal.level.value}",
                        f"Early score: {early_signal.score:.0f}/100",
                        f"Setup score: {score.total_score:.0f}/100",
                    ]
                )
                reasons.extend(early_signal.reasons[:3])

                return (
                    "BUY_DCA",
                    confidence,
                    reasons,
                    [*early_signal.warnings, *warnings],
                    ["Préparer entry progressive", "Window dans 30-60s", *early_signal.recommendations[:2]],
                )

        # === DÉCISION BASÉE SUR SCORE CORRIGÉ ===
        # Score 70+ avec validation = BUY_NOW
        if score.total_score >= 70 and validation.overall_score >= 75:
            confidence = min(score.total_score, validation.overall_score)
            reasons.append(f"✅ Score élevé: {score.total_score:.0f}/100")
            reasons.append(f"✅ Validation: {validation.overall_score:.0f}%")

            # Vérifier que setup est vraiment EARLY
            if rsi <= 60 and vol_spike < 2.0:
                reasons.append("✅ Setup EARLY confirmé (RSI/volume optimal)")
                return (
                    "BUY_NOW",
                    confidence,
                    reasons,
                    warnings,
                    ["Entry recommandée", "Setup optimal détecté"],
                )

            warnings.append(
                f"⚠️ Setup modéré: RSI {rsi:.0f}, vol {vol_spike:.1f}x"
            )
            return (
                "BUY_DCA",
                confidence * 0.9,
                reasons,
                warnings,
                ["Entry progressive recommandée", "Setup acceptable mais pas optimal"],
            )

        # Score 60-70 = BUY_DCA
        if score.total_score >= 60 and validation.overall_score >= 65:
            confidence = min(score.total_score, validation.overall_score) * 0.85
            reasons.append(f"📊 Score acceptable: {score.total_score:.0f}/100")

            if rsi <= 55:
                reasons.append("✅ RSI optimal pour entry early")
                return (
                    "BUY_DCA",
                    confidence,
                    reasons,
                    warnings,
                    ["Entry progressive", "Surveiller évolution"],
                )

            warnings.append(f"⚠️ RSI modéré: {rsi:.0f}")
            return (
                "WAIT",
                confidence * 0.8,
                reasons,
                [*warnings, "Setup acceptable mais attendre confirmation"],
                ["Surveiller évolution", "Préparer entry si amélioration"],
            )

        # Score 50-60 = WAIT
        if score.total_score >= 50:
            confidence = score.total_score * 0.7
            return (
                "WAIT",
                confidence,
                [
                    f"⏸️ Score modéré: {score.total_score:.0f}/100",
                    "Setup en formation",
                ],
                [*warnings, "Score insuffisant pour entry"],
                ["Surveiller évolution", "Attendre amélioration"],
            )

        # Score <50 = AVOID
        return (
            "AVOID",
            score.total_score * 0.5,
            [f"❌ Score faible: {score.total_score:.0f}/100", "Pas de setup"],
            [*warnings, "Setup non formé"],
            ["Continuer scan", "Attendre meilleur setup"],
        )

    def _calculate_entry_prices(
        self, current_price: float, analyzer_data: dict, action: str
    ) -> tuple[float, float]:
        """
        Calcule prix d'entrée optimal et agressif.

        Returns:
            (entry_optimal, entry_aggressive)
        """
        # Entry optimal = légèrement en dessous (limit order)
        nearest_support = self.safe_float(analyzer_data.get("nearest_support"))
        atr = self.safe_float(analyzer_data.get("atr"))

        if action in ["BUY_NOW", "BUY_DCA"]:
            # Optimal = entre current et support, ou current - 0.15% ATR
            if nearest_support > 0 and nearest_support < current_price:
                entry_optimal = (current_price + nearest_support) / 2
            elif atr > 0:
                entry_optimal = current_price - (0.15 * atr)
            else:
                entry_optimal = current_price * 0.999  # -0.1%

            # Aggressive = current price (market order)
            entry_aggressive = current_price
        else:
            entry_optimal = current_price
            entry_aggressive = current_price

        return entry_optimal, entry_aggressive

    def _calculate_targets(
        self, current_price: float, analyzer_data: dict, score: OpportunityScore
    ) -> tuple[float, float, float | None, tuple[float, float, float | None]]:
        """
        Calcule targets TP1/TP2/TP3 CONSERVATEURS.

        Returns:
            (tp1, tp2, tp3, (tp1_pct, tp2_pct, tp3_pct))
        """
        atr = self.safe_float(analyzer_data.get("atr"))
        nearest_resistance = self.safe_float(analyzer_data.get("nearest_resistance"))

        # TP1 conservateur: 0.6 ATR ou résistance
        if nearest_resistance > current_price:
            tp1 = min(current_price + (0.6 * atr), nearest_resistance)
        else:
            tp1 = current_price + (0.6 * atr)

        # TP2: 1.0 ATR
        tp2 = current_price + (1.0 * atr)

        # TP3: 1.5 ATR si score >75
        tp3 = current_price + (1.5 * atr) if score.total_score > 75 else None

        tp1_pct = ((tp1 - current_price) / current_price) * 100
        tp2_pct = ((tp2 - current_price) / current_price) * 100
        tp3_pct = ((tp3 - current_price) / current_price) * 100 if tp3 else None

        return tp1, tp2, tp3, (tp1_pct, tp2_pct, tp3_pct)

    def _calculate_stop_loss(
        self, current_price: float, analyzer_data: dict
    ) -> tuple[float, float, str]:
        """
        Calcule stop loss CONSERVATEUR.

        Returns:
            (stop_loss, sl_percent, sl_basis)
        """
        atr = self.safe_float(analyzer_data.get("atr"))
        nearest_support = self.safe_float(analyzer_data.get("nearest_support"))

        # SL = support - 0.3 ATR ou current - 0.8 ATR
        if nearest_support > 0 and nearest_support < current_price:
            sl = nearest_support - (0.3 * atr)
            sl_basis = "support"
        else:
            sl = current_price - (0.8 * atr)
            sl_basis = "ATR"

        sl_percent = ((current_price - sl) / current_price) * 100

        return sl, sl_percent, sl_basis

    def _calculate_risk_metrics(
        self,
        current_price: float,
        tp1: float,
        stop_loss: float,
        score: OpportunityScore,
        validation: ValidationSummary,
        analyzer_data: dict,
        action: str,
    ) -> tuple[float, str, float]:
        """
        Calcule métriques risque CONSERVATRICES.

        Returns:
            (rr_ratio, risk_level, max_position_size_pct)
        """
        # R/R ratio
        risk = current_price - stop_loss
        reward = tp1 - current_price
        rr_ratio = reward / risk if risk > 0 else 0.0

        # Risk level
        volatility = self.safe_float(analyzer_data.get("volatility_regime_strength"))
        if action == "AVOID":
            risk_level = "EXTREME"
            max_position_pct = 0.0
        elif rr_ratio < 1.8 or volatility > 0.8:
            risk_level = "HIGH"
            max_position_pct = 1.0
        elif rr_ratio < 2.5 or volatility > 0.6:
            risk_level = "MEDIUM"
            max_position_pct = 2.0
        else:
            risk_level = "LOW"
            max_position_pct = 3.0

        # Ajuster selon score/validation
        if score.total_score < 60 or validation.overall_score < 70:
            max_position_pct *= 0.5

        return rr_ratio, risk_level, max_position_pct

    def _calculate_timing(
        self, analyzer_data: dict, score: OpportunityScore, action: str
    ) -> tuple[str, str]:
        """
        Calcule timing estimé.

        Returns:
            (estimated_hold_time, entry_urgency)
        """
        volatility = self.safe_float(analyzer_data.get("volatility_regime_strength"))

        # Hold time basé sur volatilité
        if volatility > 0.7:
            hold_time = "15-30min"
        elif volatility > 0.5:
            hold_time = "30-60min"
        else:
            hold_time = "1-2h"

        # Urgency basée sur action/score
        if action == "BUY_NOW" and score.total_score >= 75:
            urgency = "IMMEDIATE"
        elif action == "BUY_NOW":
            urgency = "SOON"
        elif action == "BUY_DCA":
            urgency = "PATIENT"
        else:
            urgency = "NO_RUSH"

        return hold_time, urgency

    def _create_no_data_opportunity(
        self, symbol: str, current_price: float
    ) -> TradingOpportunity:
        """Crée opportunité vide si pas de données."""
        return TradingOpportunity(
            symbol=symbol,
            action="AVOID",
            confidence=0.0,
            score=OpportunityScore(
                total_score=0.0,
                rsi_score=0.0,
                momentum_score=0.0,
                volume_score=0.0,
                support_resistance_score=0.0,
                trend_score=0.0,
                mfi_score=0.0,
                confluence_score=0.0,
                category_scores={},
                reasons=[],
                warnings=["Pas de données disponibles"],
            ),
            validation=ValidationSummary(
                passed=False,
                confidence=0.0,
                failed_reason="Pas de données",
                data_quality=0.0,
                market_quality=0.0,
                risk_quality=0.0,
                timing_quality=0.0,
                warnings=["Pas de données disponibles"],
                rejections=[],
            ),
            early_signal=None,
            is_early_entry=False,
            current_price=current_price,
            entry_price_optimal=current_price,
            entry_price_aggressive=current_price,
            tp1=current_price,
            tp1_percent=0.0,
            tp2=current_price,
            tp2_percent=0.0,
            tp3=None,
            tp3_percent=None,
            stop_loss=current_price,
            stop_loss_percent=0.0,
            stop_loss_basis="NONE",
            rr_ratio=0.0,
            risk_level="EXTREME",
            max_position_size_pct=0.0,
            estimated_hold_time="N/A",
            entry_urgency="NO_RUSH",
            market_regime="UNKNOWN",
            volume_context="UNKNOWN",
            volatility_regime="UNKNOWN",
            reasons=["Pas de données disponibles"],
            warnings=["Impossible de calculer opportunité"],
            recommendations=["Attendre données valides"],
            raw_score_details={},
            raw_validation_details={},
            raw_analyzer_data={},
        )
