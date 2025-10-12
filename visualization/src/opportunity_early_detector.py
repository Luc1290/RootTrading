"""
Opportunity Early Detector - Leading Indicator System
Détecte les pumps AVANT la confirmation complète (20-40s d'avance)

Architecture:
- Focus LEADING indicators (velocity, acceleration, derivatives)
- Multi-timepoint analysis (dernières 3-5 périodes)
- Micro-patterns détection (higher lows, volume buildup)
- Probabilistic scoring (60-70% suffisant au lieu de 90%+)

Objectif: Signaler à T+30s au lieu de T+60s dans la séquence pump
Version: 1.0 - Early Warning System
"""
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EarlySignalLevel(Enum):
    """Niveaux de signal early."""
    NONE = "none"
    WATCH = "watch"  # Setup en formation, surveiller
    PREPARE = "prepare"  # Probable entry window bientôt
    ENTRY_NOW = "entry_now"  # Entry window MAINTENANT
    TOO_LATE = "too_late"  # Mouvement déjà avancé


@dataclass
class EarlySignal:
    """Signal early avec score et recommandation."""
    level: EarlySignalLevel
    score: float  # 0-100
    confidence: float  # 0-100
    velocity_score: float
    acceleration_score: float
    volume_buildup_score: float
    micro_pattern_score: float
    order_flow_score: float

    # Timing estimé
    estimated_entry_window_seconds: int  # Combien de secondes avant entry optimal
    estimated_move_completion_pct: float  # % du mouvement déjà fait (0-100)

    # Reasons
    reasons: List[str]
    warnings: List[str]
    recommendations: List[str]


class OpportunityEarlyDetector:
    """
    Détecteur précoce d'opportunités.

    Utilise des indicateurs LEADING au lieu de LAGGING:
    - Price velocity & acceleration (derivatives)
    - Volume buildup progression
    - Micro-patterns sur 3-5 périodes
    - Order flow pressure (bid/ask imbalance si dispo)
    """

    # Seuils pour early detection (PLUS BAS que système principal)
    THRESHOLDS = {
        'watch': 40,  # Score 40+ = worth watching
        'prepare': 55,  # Score 55+ = prepare entry
        'entry_now': 65,  # Score 65+ = entry window NOW
        'too_late': 85  # Score 85+ = déjà trop tard
    }

    def __init__(self):
        """Initialise le détecteur early."""
        pass

    @staticmethod
    def safe_float(value, default=0.0):
        """Convertir en float avec fallback."""
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    def detect_early_opportunity(
        self,
        current_data: dict,
        historical_data: Optional[List[dict]] = None
    ) -> EarlySignal:
        """
        Détecte early opportunity.

        Args:
            current_data: analyzer_data actuel (dernière période)
            historical_data: Liste des 5-10 dernières périodes analyzer_data
                             Ordre: [plus_ancien, ..., plus_recent]
                             Si None, on fait une détection limitée sur current_data seul

        Returns:
            EarlySignal avec niveau, score, et timing estimé
        """
        if not current_data:
            return self._create_no_signal("Pas de données")

        reasons = []
        warnings = []
        recommendations = []

        # === SCORE 1: VELOCITY & ACCELERATION (35 points max) ===
        velocity_score = self._score_velocity_acceleration(
            current_data, historical_data, reasons, warnings
        )

        # === SCORE 2: VOLUME BUILDUP (25 points max) ===
        volume_buildup_score = self._score_volume_buildup(
            current_data, historical_data, reasons, warnings
        )

        # === SCORE 3: MICRO-PATTERNS (20 points max) ===
        micro_pattern_score = self._score_micro_patterns(
            current_data, historical_data, reasons, warnings
        )

        # === SCORE 4: ORDER FLOW PRESSURE (13 points max) ===
        order_flow_score = self._score_order_flow(
            current_data, reasons, warnings
        )

        # === SCORE 5: EARLY MOMENTUM (15 points max) ===
        early_momentum_score = self._score_early_momentum(
            current_data, historical_data, reasons, warnings
        )

        # Score total
        total_score = (
            velocity_score +
            volume_buildup_score +
            micro_pattern_score +
            order_flow_score +
            early_momentum_score
        )

        # Confiance basée sur disponibilité des données historiques
        if historical_data and len(historical_data) >= 5:
            confidence = 90.0
        elif historical_data and len(historical_data) >= 3:
            confidence = 75.0
        elif historical_data:
            confidence = 60.0
        else:
            confidence = 40.0
            warnings.append("⚠️ Pas de données historiques - Détection limitée")

        # Déterminer niveau de signal
        level, entry_window_seconds, move_completion_pct = self._determine_signal_level(
            total_score, current_data, historical_data
        )

        # Générer recommandations
        self._generate_recommendations(
            level, total_score, entry_window_seconds,
            move_completion_pct, recommendations, warnings
        )

        return EarlySignal(
            level=level,
            score=total_score,
            confidence=confidence,
            velocity_score=velocity_score,
            acceleration_score=velocity_score,  # Inclus dans velocity
            volume_buildup_score=volume_buildup_score,
            micro_pattern_score=micro_pattern_score,
            order_flow_score=order_flow_score,
            estimated_entry_window_seconds=entry_window_seconds,
            estimated_move_completion_pct=move_completion_pct,
            reasons=reasons,
            warnings=warnings,
            recommendations=recommendations
        )

    def _score_velocity_acceleration(
        self,
        current: dict,
        historical: Optional[List[dict]],
        reasons: List[str],
        warnings: List[str]
    ) -> float:
        """
        Score velocity & acceleration (LEADING).

        Velocity: ROC_10 actuel vs moyenne
        Acceleration: Changement de ROC sur dernières périodes

        35 points max:
        - 20 pts: Price velocity
        - 15 pts: Price acceleration
        """
        score = 0.0

        # 1. Price Velocity (ROC) - 20 points max
        roc_10 = self.safe_float(current.get('roc_10'))
        roc_20 = self.safe_float(current.get('roc_20'))

        if roc_10 > 0:
            # ROC positif = momentum haussier
            if roc_10 > 0.25:  # +0.25% = 25bps = fort
                vel_score = 20
                reasons.append(f"🚀 Vélocité forte: ROC {roc_10*100:.2f}%")
            elif roc_10 > 0.15:  # +0.15% = modéré-fort
                vel_score = 15
                reasons.append(f"📈 Vélocité modérée: ROC {roc_10*100:.2f}%")
            elif roc_10 > 0.08:  # +0.08% = modéré
                vel_score = 10
            elif roc_10 > 0.03:  # +0.03% = faible
                vel_score = 5
            else:
                vel_score = 2

            score += vel_score
        elif roc_10 < -0.05:
            warnings.append(f"⚠️ Vélocité négative: ROC {roc_10*100:.2f}%")

        # 2. Acceleration (changement de ROC) - 15 points max
        if historical and len(historical) >= 3:
            # Calculer acceleration = dérivée du ROC
            recent_rocs = []
            for h in historical[-3:]:
                recent_rocs.append(self.safe_float(h.get('roc_10')))
            recent_rocs.append(roc_10)

            # Acceleration = ROC maintenant vs moyenne des 3 derniers
            avg_roc = sum(recent_rocs[:-1]) / len(recent_rocs[:-1]) if recent_rocs[:-1] else 0
            roc_change = roc_10 - avg_roc

            if roc_change > 0.10:  # Forte accélération
                accel_score = 15
                reasons.append(f"⚡ Accélération forte: +{roc_change*100:.2f}%")
            elif roc_change > 0.05:  # Accélération modérée
                accel_score = 10
                reasons.append(f"⚡ Accélération: +{roc_change*100:.2f}%")
            elif roc_change > 0.02:
                accel_score = 5
            elif roc_change < -0.05:
                accel_score = 0
                warnings.append(f"⚠️ Décélération: {roc_change*100:.2f}%")
            else:
                accel_score = 2

            score += accel_score

        return min(score, 35)

    def _score_volume_buildup(
        self,
        current: dict,
        historical: Optional[List[dict]],
        reasons: List[str],
        warnings: List[str]
    ) -> float:
        """
        Score volume buildup (LEADING).

        25 points max:
        - 10 pts: volume_buildup_periods actuel
        - 10 pts: Progression volume sur dernières périodes
        - 5 pts: relative_volume émergent (1.5x+ au lieu de 2.5x+)
        """
        score = 0.0

        # 1. Volume buildup periods - 10 points max
        buildup = current.get('volume_buildup_periods', 0)
        if buildup >= 5:
            score += 10
            reasons.append(f"📊 Volume buildup: {buildup} périodes")
        elif buildup >= 3:
            score += 7
            reasons.append(f"📊 Volume buildup: {buildup} périodes")
        elif buildup >= 2:
            score += 4

        # 2. Volume progression - 10 points max
        rel_vol = self.safe_float(current.get('relative_volume'), 1.0)

        if historical and len(historical) >= 3:
            # Calculer progression volume
            recent_vols = []
            for h in historical[-3:]:
                recent_vols.append(self.safe_float(h.get('relative_volume'), 1.0))
            recent_vols.append(rel_vol)

            # Volume en progression?
            is_increasing = all(
                recent_vols[i] <= recent_vols[i+1]
                for i in range(len(recent_vols)-1)
            )

            if is_increasing and rel_vol > recent_vols[0] * 1.5:
                score += 10
                reasons.append(f"📈 Volume en progression: {recent_vols[0]:.2f}x → {rel_vol:.2f}x")
            elif is_increasing:
                score += 5

        # 3. Relative volume émergent (seuil BAS) - 5 points max
        # On veut détecter 1.5x+ au lieu d'attendre 2.5x+
        if rel_vol >= 2.0:
            score += 5
            reasons.append(f"🔥 Volume élevé: {rel_vol:.2f}x")
        elif rel_vol >= 1.5:
            score += 3
            reasons.append(f"📊 Volume émergent: {rel_vol:.2f}x")
        elif rel_vol >= 1.2:
            score += 1
        elif rel_vol < 0.8:
            warnings.append(f"⚠️ Volume faible: {rel_vol:.2f}x")

        return min(score, 25)

    def _score_micro_patterns(
        self,
        current: dict,
        historical: Optional[List[dict]],
        reasons: List[str],
        warnings: List[str]
    ) -> float:
        """
        Score micro-patterns (LEADING).

        20 points max:
        - 10 pts: Higher lows séquence (prix)
        - 5 pts: RSI climbing (RSI en montée)
        - 5 pts: MACD histogram expansion
        """
        score = 0.0

        if not historical or len(historical) < 3:
            return 0.0

        # 1. Higher lows pattern - 10 points max
        # Utiliser nearest_support comme proxy pour "low" de la période
        recent_supports = []
        for h in historical[-3:]:
            sup = self.safe_float(h.get('nearest_support'))
            if sup > 0:
                recent_supports.append(sup)

        current_sup = self.safe_float(current.get('nearest_support'))
        if current_sup > 0:
            recent_supports.append(current_sup)

        if len(recent_supports) >= 3:
            # Vérifier higher lows
            is_higher_lows = all(
                recent_supports[i] <= recent_supports[i+1]
                for i in range(len(recent_supports)-1)
            )

            if is_higher_lows:
                score += 10
                reasons.append(f"📈 Higher lows: {recent_supports[0]:.2f} → {recent_supports[-1]:.2f}")

        # 2. RSI climbing - 5 points max
        recent_rsis = []
        for h in historical[-3:]:
            rsi = self.safe_float(h.get('rsi_14'))
            if rsi > 0:
                recent_rsis.append(rsi)

        current_rsi = self.safe_float(current.get('rsi_14'))
        if current_rsi > 0:
            recent_rsis.append(current_rsi)

        if len(recent_rsis) >= 3:
            # RSI en progression ET dans zone favorable (50-70)
            rsi_increasing = recent_rsis[-1] > recent_rsis[0]
            rsi_in_zone = 50 <= current_rsi <= 75

            if rsi_increasing and rsi_in_zone:
                rsi_delta = recent_rsis[-1] - recent_rsis[0]
                if rsi_delta > 10:
                    score += 5
                    reasons.append(f"🔼 RSI climbing: {recent_rsis[0]:.0f} → {current_rsi:.0f}")
                elif rsi_delta > 5:
                    score += 3

        # 3. MACD histogram expansion - 5 points max
        recent_macds = []
        for h in historical[-3:]:
            macd_hist = self.safe_float(h.get('macd_histogram'))
            recent_macds.append(macd_hist)

        current_macd = self.safe_float(current.get('macd_histogram'))
        recent_macds.append(current_macd)

        if len(recent_macds) >= 3:
            # MACD histogram en expansion (valeurs positives croissantes)
            if all(m > 0 for m in recent_macds[-2:]) and recent_macds[-1] > recent_macds[0]:
                score += 5
                reasons.append(f"📊 MACD expansion: {recent_macds[0]:.1f} → {current_macd:.1f}")
            elif current_macd > 0 and current_macd > recent_macds[-2]:
                score += 2

        return min(score, 20)

    def _score_order_flow(
        self,
        current: dict,
        reasons: List[str],
        warnings: List[str]
    ) -> float:
        """
        Score order flow pressure (si données disponibles).

        13 points max (augmenté de 10 pour break_probability):
        - OBV oscillator positif et croissant
        - Trade intensity élevé
        - Avg trade size en hausse (whales buying)
        - BONUS: Break probability élevée (+3 pts)
        """
        score = 0.0

        # 1. OBV Oscillator - 5 points max
        obv_osc = self.safe_float(current.get('obv_oscillator'))
        if obv_osc > 200:
            score += 5
            reasons.append(f"💰 OBV fort: {obv_osc:.0f}")
        elif obv_osc > 100:
            score += 3
        elif obv_osc > 50:
            score += 1
        elif obv_osc < -100:
            warnings.append(f"⚠️ OBV négatif: {obv_osc:.0f}")

        # 2. Trade Intensity - 3 points max
        intensity = self.safe_float(current.get('trade_intensity'))
        if intensity > 1.5:
            score += 3
            reasons.append(f"⚡ Trade intensity: {intensity:.2f}x")
        elif intensity > 1.2:
            score += 2
        elif intensity > 1.0:
            score += 1

        # 3. Quote Volume Ratio - 2 points max
        qv_ratio = self.safe_float(current.get('quote_volume_ratio'))
        if qv_ratio > 1.3:
            score += 2
        elif qv_ratio > 1.1:
            score += 1

        # 4. BREAK PROBABILITY (NOUVEAU) - 3 points max
        # Probabilité de casser la résistance au-dessus
        break_prob = self.safe_float(current.get('break_probability'), 0.5)
        if break_prob > 0.75:
            score += 3
            reasons.append(f"🔓 Résistance faible: {break_prob*100:.0f}% break prob")
        elif break_prob > 0.65:
            score += 2
            reasons.append(f"🔓 Break probable: {break_prob*100:.0f}%")
        elif break_prob > 0.55:
            score += 1

        return min(score, 13)  # Max augmenté de 10 à 13

    def _score_early_momentum(
        self,
        current: dict,
        historical: Optional[List[dict]],
        reasons: List[str],
        warnings: List[str]
    ) -> float:
        """
        Score early momentum indicators.

        15 points max (augmenté de 10 pour inclure oversold bonus):
        - RSI 50-65 avec vélocité positive (montée en cours)
        - MACD signal cross récent
        - BB position 0.5-0.8 (pas encore overbought)
        - BONUS: Sortie oversold détectée (+5 pts)
        """
        score = 0.0

        # 1. RSI early zone - 5 points max
        rsi = self.safe_float(current.get('rsi_14'))

        # 1a. BONUS OVERSOLD BOUNCE (NOUVEAU)
        # Si RSI était <35 et maintenant >45 = sortie oversold franche
        if historical and len(historical) >= 2:
            recent_rsis = [self.safe_float(h.get('rsi_14', 50)) for h in historical[-3:]]
            # Prendre le RSI le plus bas des 3 dernières périodes
            min_recent_rsi = min(recent_rsis) if recent_rsis else 50

            if min_recent_rsi < 35 and rsi > 45:
                score += 5  # Gros bonus oversold bounce
                reasons.append(f"💥 Sortie oversold: RSI {min_recent_rsi:.0f}→{rsi:.0f}")
            elif min_recent_rsi < 40 and rsi > 50:
                score += 3  # Bonus modéré
                reasons.append(f"📈 Rebond oversold: RSI {min_recent_rsi:.0f}→{rsi:.0f}")
            elif min_recent_rsi < 45 and rsi > 55:
                score += 2  # Léger bonus

        # 1b. RSI zone standard
        if 50 <= rsi <= 65:
            # Sweet spot: momentum bullish mais pas encore overbought
            score += 5
            reasons.append(f"✅ RSI zone optimal: {rsi:.0f}")
        elif 45 <= rsi < 50:
            score += 2
        elif rsi > 75:
            warnings.append(f"⚠️ RSI déjà élevé: {rsi:.0f}")

        # 2. MACD signal cross - 3 points max
        macd_cross = current.get('macd_signal_cross', False)
        macd_trend = current.get('macd_trend', '').upper()
        if macd_cross and macd_trend == 'BULLISH':
            score += 3
            reasons.append("🔄 MACD cross bullish récent")
        elif macd_trend == 'BULLISH':
            score += 1

        # 3. BB position - 2 points max
        bb_pos = self.safe_float(current.get('bb_position'))
        if 0.5 <= bb_pos <= 0.8:
            # Milieu-haut de BB, bon signe mais pas encore étendu
            score += 2
        elif bb_pos > 1.0:
            warnings.append(f"⚠️ BB position étendue: {bb_pos:.2f}")

        return min(score, 15)  # Max augmenté de 10 à 15 pour bonus oversold

    def _determine_signal_level(
        self,
        score: float,
        current: dict,
        historical: Optional[List[dict]]
    ) -> Tuple[EarlySignalLevel, int, float]:
        """
        Détermine le niveau de signal + timing estimé.

        Returns:
            (level, entry_window_seconds, move_completion_pct)
        """
        # Estimer % du mouvement déjà fait
        rsi = self.safe_float(current.get('rsi_14'))
        rel_vol = self.safe_float(current.get('relative_volume'), 1.0)
        vol_spike = self.safe_float(current.get('volume_spike_multiplier'), 1.0)

        # Heuristique: RSI et volume spike indiquent avancement
        if rsi > 80 or vol_spike > 4.0:
            move_completion_pct = 85.0  # Déjà très avancé
        elif rsi > 70 or vol_spike > 2.5:
            move_completion_pct = 60.0  # Bien avancé
        elif rsi > 60 or rel_vol > 1.8:
            move_completion_pct = 35.0  # Modérément avancé
        else:
            move_completion_pct = 15.0  # Early stage

        # Déterminer niveau et window
        if score >= self.THRESHOLDS['too_late']:
            return EarlySignalLevel.TOO_LATE, 0, move_completion_pct

        elif score >= self.THRESHOLDS['entry_now']:
            # Entry window = 10-30s
            return EarlySignalLevel.ENTRY_NOW, 20, move_completion_pct

        elif score >= self.THRESHOLDS['prepare']:
            # Préparer entry dans 20-60s
            return EarlySignalLevel.PREPARE, 40, move_completion_pct

        elif score >= self.THRESHOLDS['watch']:
            # Surveiller, entry potentielle dans 60-120s
            return EarlySignalLevel.WATCH, 90, move_completion_pct

        else:
            return EarlySignalLevel.NONE, 0, move_completion_pct

    def _generate_recommendations(
        self,
        level: EarlySignalLevel,
        score: float,
        entry_window_seconds: int,
        move_completion_pct: float,
        recommendations: List[str],
        warnings: List[str]
    ):
        """Génère recommandations selon niveau."""
        if level == EarlySignalLevel.ENTRY_NOW:
            recommendations.append(f"🚀 ENTRY WINDOW NOW - Score: {score:.0f}/100")
            recommendations.append(f"⏱️ Fenêtre entry: {entry_window_seconds}s estimé")
            recommendations.append("💡 Préparer ordre LIMIT à entry_optimal")

            if move_completion_pct > 50:
                warnings.append(f"⚠️ Mouvement déjà {move_completion_pct:.0f}% avancé")

        elif level == EarlySignalLevel.PREPARE:
            recommendations.append(f"⚡ PRÉPARER ENTRY - Score: {score:.0f}/100")
            recommendations.append(f"⏱️ Entry estimée dans: ~{entry_window_seconds}s")
            recommendations.append("💡 Vérifier entry_optimal et préparer capital")

        elif level == EarlySignalLevel.WATCH:
            recommendations.append(f"👀 SURVEILLER - Score: {score:.0f}/100")
            recommendations.append("💡 Setup en formation, rester attentif")

        elif level == EarlySignalLevel.TOO_LATE:
            recommendations.append(f"⏸️ TROP TARD - Score: {score:.0f}/100")
            recommendations.append(f"❌ Mouvement {move_completion_pct:.0f}% terminé")
            recommendations.append("💡 Attendre pullback ou prochaine opportunité")

        else:
            recommendations.append("⏸️ PAS DE SETUP - Continuer scan")

    def _create_no_signal(self, reason: str) -> EarlySignal:
        """Crée un signal vide."""
        return EarlySignal(
            level=EarlySignalLevel.NONE,
            score=0.0,
            confidence=0.0,
            velocity_score=0.0,
            acceleration_score=0.0,
            volume_buildup_score=0.0,
            micro_pattern_score=0.0,
            order_flow_score=0.0,
            estimated_entry_window_seconds=0,
            estimated_move_completion_pct=0.0,
            reasons=[reason],
            warnings=[],
            recommendations=["Pas de données disponibles"]
        )


# ===========================================================
# EXEMPLE D'UTILISATION
# ===========================================================
if __name__ == "__main__":
    import sys
    import io

    # Fix Windows encoding
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("="*80)
    print("OPPORTUNITY EARLY DETECTOR - Test Example")
    print("="*80)

    # Simuler données progression pump
    # T-3 (historical[0])
    hist_t3 = {
        'roc_10': 0.05,
        'relative_volume': 1.0,
        'volume_buildup_periods': 2,
        'rsi_14': 55,
        'macd_histogram': 5.0,
        'nearest_support': 1.0700,
        'obv_oscillator': 80,
        'trade_intensity': 1.1
    }

    # T-2 (historical[1])
    hist_t2 = {
        'roc_10': 0.08,
        'relative_volume': 1.2,
        'volume_buildup_periods': 3,
        'rsi_14': 58,
        'macd_histogram': 7.0,
        'nearest_support': 1.0710,
        'obv_oscillator': 120,
        'trade_intensity': 1.2
    }

    # T-1 (historical[2])
    hist_t1 = {
        'roc_10': 0.12,
        'relative_volume': 1.5,
        'volume_buildup_periods': 4,
        'rsi_14': 62,
        'macd_histogram': 10.0,
        'nearest_support': 1.0720,
        'obv_oscillator': 180,
        'trade_intensity': 1.4
    }

    # T (current) - ENTRY WINDOW
    current_t0 = {
        'roc_10': 0.18,
        'roc_20': 0.22,
        'relative_volume': 1.8,
        'volume_buildup_periods': 5,
        'volume_spike_multiplier': 1.8,
        'rsi_14': 66,
        'rsi_21': 64,
        'macd_histogram': 15.0,
        'macd_trend': 'BULLISH',
        'macd_signal_cross': True,
        'nearest_support': 1.0730,
        'bb_position': 0.72,
        'obv_oscillator': 250,
        'trade_intensity': 1.6,
        'quote_volume_ratio': 1.3
    }

    historical = [hist_t3, hist_t2, hist_t1]

    # Créer détecteur
    detector = OpportunityEarlyDetector()

    # Détecter
    signal = detector.detect_early_opportunity(current_t0, historical)

    # Afficher résultat
    print(f"\n🎯 EARLY SIGNAL DETECTED")
    print(f"Niveau: {signal.level.value.upper()}")
    print(f"Score: {signal.score:.0f}/100")
    print(f"Confiance: {signal.confidence:.0f}%")

    print(f"\n📊 BREAKDOWN:")
    print(f"  Velocity/Accel: {signal.velocity_score:.0f}/35")
    print(f"  Volume Buildup: {signal.volume_buildup_score:.0f}/25")
    print(f"  Micro-Patterns: {signal.micro_pattern_score:.0f}/20")
    print(f"  Order Flow: {signal.order_flow_score:.0f}/10")

    print(f"\n⏱️ TIMING:")
    print(f"  Entry window: ~{signal.estimated_entry_window_seconds}s")
    print(f"  Mouvement complété: {signal.estimated_move_completion_pct:.0f}%")

    print(f"\n📋 REASONS:")
    for reason in signal.reasons:
        print(f"  {reason}")

    if signal.warnings:
        print(f"\n⚠️ WARNINGS:")
        for warning in signal.warnings:
            print(f"  {warning}")

    print(f"\n💡 RECOMMENDATIONS:")
    for rec in signal.recommendations:
        print(f"  {rec}")

    print("\n" + "="*80)

    # Test avec données pump RÉEL (09:44 - avant spike)
    print("\n" + "="*80)
    print("TEST AVEC DONNÉES PUMP RÉEL (09:44 BTCUSDC)")
    print("="*80)

    real_current = {
        'roc_10': 0.258,  # +25.8 bps
        'roc_20': 0.300,
        'relative_volume': 0.97,
        'volume_buildup_periods': 0,
        'volume_spike_multiplier': 1.0,
        'rsi_14': 67.18,
        'rsi_21': 65.0,
        'macd_histogram': 20.06,
        'macd_trend': 'BULLISH',
        'macd_signal_cross': False,
        'nearest_support': 1.0750,
        'bb_position': 0.94,
        'obv_oscillator': 150,
        'trade_intensity': 1.3,
        'quote_volume_ratio': 1.2
    }

    real_hist = [
        {'roc_10': 0.127, 'relative_volume': 1.00, 'rsi_14': 63.4, 'macd_histogram': 20.9, 'nearest_support': 1.0740},
        {'roc_10': 0.152, 'relative_volume': 0.94, 'rsi_14': 64.6, 'macd_histogram': 19.7, 'nearest_support': 1.0742},
        {'roc_10': 0.258, 'relative_volume': 0.97, 'rsi_14': 67.2, 'macd_histogram': 20.1, 'nearest_support': 1.0750}
    ]

    real_signal = detector.detect_early_opportunity(real_current, real_hist)

    print(f"\n🎯 SIGNAL: {real_signal.level.value.upper()}")
    print(f"Score: {real_signal.score:.0f}/100")
    print(f"Entry window: ~{real_signal.estimated_entry_window_seconds}s")

    print(f"\n📋 REASONS:")
    for reason in real_signal.reasons:
        print(f"  {reason}")

    print(f"\n💡 RECOMMENDATIONS:")
    for rec in real_signal.recommendations:
        print(f"  {rec}")

    print("\n" + "="*80)
