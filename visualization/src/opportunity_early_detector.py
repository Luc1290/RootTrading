"""
Opportunity Early Detector - INSTITUTIONAL SCALPING
Détecteur optionnel pour boost de confiance basé sur patterns de formation

VERSION 4.1 - AMÉLIORATIONS:
1. Warnings volume spike contextualisés avec RSI/ROC (CRITIQUE)
2. Warnings RSI >70 contextualisés avec ADX/MACD
3. Volume 3x+ acceptable si RSI <60 (breakout institutionnel early)

VERSION 4.0 - REFONTE COMPLÈTE:
1. OPTIONNEL: Ce module booste la confiance mais n'est PAS obligatoire
2. PAS de rejets: RSI >70, volume >3x sont des WARNINGS, pas des blocages
3. Scoring informatif: Détecte patterns de buildup mais ne bloque pas
4. Compatible avec scoring institutionnel v4.0

PHILOSOPHIE:
- Détecte patterns de consolidation/buildup avant mouvement
- Ajoute +5-10pts de confiance si patterns détectés
- JAMAIS bloquant, toujours informatif
- Basé sur analyse multi-périodes (historical_data requis)

Indicateurs analysés:
- Velocity & Acceleration: ROC et sa dérivée
- Volume Buildup: Progression volume progressive
- Micro-patterns: Higher lows, RSI climbing
- Order Flow: OBV, trade intensity

Aligné avec opportunity_scoring.py v4.0 (pas de rejets arbitraires)
Version: 4.0 - Institutional Scalping (Optional Boost)
"""

import logging
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
    reasons: list[str]
    warnings: list[str]
    recommendations: list[str]


class OpportunityEarlyDetector:
    """
    Détecteur précoce d'opportunités - INSTITUTIONAL SCALPING v4.0.

    OPTIONNEL: Booste confiance mais ne bloque jamais.

    CHANGEMENTS vs v2.0:
    - PAS de rejets: RSI >70, volume >3x sont WARNINGS seulement
    - Score informatif: Détecte patterns mais n'impose pas de décision
    - Compatible avec scoring institutionnel (pas de contradictions)
    - Fonction: Détecter patterns de buildup/consolidation avant mouvement
    """

    # Seuils pour signaux informatifs (pas bloquants)
    THRESHOLDS = {
        "watch": 30,  # Score 30+ = worth watching
        "prepare": 45,  # Score 45+ = prepare entry
        "entry_now": 55,  # Score 55+ = entry window
        "too_late": 75,  # Score 75+ = mouvement avancé (WARNING, pas BLOCAGE)
    }

    def __init__(self):
        """Initialise le détecteur early."""

    @staticmethod
    def safe_float(value, default=0.0):
        """Convertir en float avec fallback."""
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default

    def detect_early_opportunity(
        self, current_data: dict, historical_data: list[dict] | None = None
    ) -> EarlySignal:
        """
        Détecte early opportunity AVANT le pump démarre.

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

        reasons: list[str] = []
        warnings: list[str] = []
        recommendations: list[str] = []

        # === SCORE 1: VELOCITY & ACCELERATION (35 points max) ===
        # INVERSÉ: Cherche momentum FAIBLE/FLAT (prêt à exploser)
        velocity_score = self._score_velocity_acceleration(
            current_data, historical_data, reasons, warnings
        )

        # === SCORE 2: VOLUME BUILDUP (30 points max, augmenté de 25) ===
        # Priorité absolue: buildup progressif, pas spike
        volume_buildup_score = self._score_volume_buildup(
            current_data, historical_data, reasons, warnings
        )

        # === SCORE 3: MICRO-PATTERNS (20 points max) ===
        # Consolidation patterns, pas breakout patterns
        micro_pattern_score = self._score_micro_patterns(
            current_data, historical_data, reasons, warnings
        )

        # === SCORE 4: ORDER FLOW PRESSURE (15 points max, augmenté de 13) ===
        order_flow_score = self._score_order_flow(current_data, reasons, warnings)

        # Score total
        total_score = (
            velocity_score
            + volume_buildup_score
            + micro_pattern_score
            + order_flow_score
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
            level,
            total_score,
            entry_window_seconds,
            move_completion_pct,
            recommendations,
            warnings,
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
            recommendations=recommendations,
        )

    def _score_velocity_acceleration(
        self,
        current: dict,
        historical: list[dict] | None,
        reasons: list[str],
        warnings: list[str],
    ) -> float:
        """
        Score velocity & acceleration INVERSÉ pour EARLY entry.

        ❌ ANCIEN SYSTÈME (FAUX):
        - ROC >0.25% = MAX score (déjà en pump!)
        - Delta ROC >0.10% = MAX score (déjà accéléré!)

        ✅ NOUVEAU SYSTÈME (CORRECT):
        - ROC -0.5 à +0.2% = MAX score (momentum flat, prêt à exploser)
        - ROC >0.5% = FAIBLE score (déjà en momentum)
        - Delta ROC -0.05 à +0.03% = MAX score (pas encore accéléré)
        - Delta ROC >0.08% = REJET (accélération déjà lancée)

        35 points max:
        - 20 pts: Price velocity (ROC actuel FAIBLE)
        - 15 pts: Price acceleration (changement ROC FAIBLE)
        """
        score = 0.0

        # 1. Price Velocity (ROC) - 20 points max
        # INVERSÉ: On veut ROC FAIBLE/FLAT (momentum disponible)
        roc_10 = self.safe_float(current.get("roc_10"))

        if -0.5 <= roc_10 <= 0.2:
            # OPTIMAL: Momentum flat/faible = prêt à exploser
            vel_score = 20
            reasons.append(
                f"✅ Vélocité FLAT (optimal): ROC {roc_10*100:.2f}% (momentum disponible)"
            )
        elif -1.0 <= roc_10 < -0.5:
            # Momentum légèrement négatif = acceptable
            vel_score = 15
            reasons.append(
                f"📊 Vélocité basse: ROC {roc_10*100:.2f}% (setup formation)"
            )
        elif 0.2 < roc_10 <= 0.4:
            # Momentum commence = moins optimal
            vel_score = 10
            reasons.append(f"⚠️ Vélocité émergente: ROC {roc_10*100:.2f}%")
        elif 0.4 < roc_10 <= 0.6:
            # Momentum en cours = trop tard pour early
            vel_score = 5
            warnings.append(f"⚠️ Vélocité modérée: ROC {roc_10*100:.2f}% (déjà lancé)")
        elif roc_10 > 0.6:
            # Fort momentum = TROP TARD
            vel_score = 0
            warnings.append(
                f"❌ Vélocité FORTE: ROC {roc_10*100:.2f}% - TROP TARD pour entry early"
            )
        elif roc_10 < -1.0:
            # Momentum très négatif = pas de setup
            vel_score = 5
            warnings.append(
                f"⚠️ Vélocité très négative: ROC {roc_10*100:.2f}% (tendance baissière)"
            )
        else:
            vel_score = 0

        score += vel_score

        # 2. Acceleration (changement de ROC) - 15 points max
        # INVERSÉ: On veut accélération FAIBLE (pas encore lancée)
        if historical and len(historical) >= 3:
            # Calculer acceleration = dérivée du ROC
            recent_rocs = []
            for h in historical[-3:]:
                recent_rocs.append(self.safe_float(h.get("roc_10")))
            recent_rocs.append(roc_10)

            # Acceleration = ROC maintenant vs moyenne des 3 derniers
            avg_roc = (
                sum(recent_rocs[:-1]) / len(recent_rocs[:-1]) if recent_rocs[:-1] else 0
            )
            roc_change = roc_10 - avg_roc

            if -0.05 <= roc_change <= 0.03:
                # OPTIMAL: Accélération faible/stable = pas encore lancé
                accel_score = 15
                reasons.append(
                    f"✅ Accélération FAIBLE (optimal): {roc_change*100:.2f}% (momentum stable)"
                )
            elif -0.10 <= roc_change < -0.05:
                # Décélération modérée = acceptable
                accel_score = 10
                reasons.append(
                    f"📊 Décélération modérée: {roc_change*100:.2f}% (consolidation)"
                )
            elif 0.03 < roc_change <= 0.08:
                # Accélération émergente = moins optimal
                accel_score = 5
                warnings.append(
                    f"⚠️ Accélération émergente: +{roc_change*100:.2f}% (momentum démarre)"
                )
            elif roc_change > 0.08:
                # Forte accélération = TROP TARD
                accel_score = 0
                warnings.append(
                    f"❌ Accélération FORTE: +{roc_change*100:.2f}% - TROP TARD (déjà accéléré)"
                )
            elif roc_change < -0.10:
                # Forte décélération = pas de setup
                accel_score = 2
                warnings.append(
                    f"⚠️ Décélération forte: {roc_change*100:.2f}% (pression vendeuse)"
                )
            else:
                accel_score = 0

            score += accel_score

        return min(score, 35)

    def _score_volume_buildup(
        self,
        current: dict,
        historical: list[dict] | None,
        reasons: list[str],
        warnings: list[str],
    ) -> float:
        """
        Score volume buildup CORRIGÉ pour EARLY entry.

        ❌ ANCIEN SYSTÈME (FAUX):
        - rel_volume >2.0x = MAX score (pic atteint!)

        ✅ NOUVEAU SYSTÈME (CORRECT):
        - volume_buildup_periods 3+ = MAX score (progression confirmée)
        - rel_volume 1.2-1.8x progressif = MAX score (buildup)
        - rel_volume >3.0x = REJET (spike = trop tard)

        30 points max (augmenté de 25):
        - 15 pts: volume_buildup_periods actuel (priorité absolue)
        - 10 pts: Progression volume sur dernières périodes
        - 5 pts: relative_volume dans zone buildup (1.2-1.8x)
        """
        score = 0.0

        # 1. Volume buildup periods - 15 points max (augmenté de 10)
        buildup = current.get("volume_buildup_periods", 0)
        if buildup >= 5:
            score += 15
            reasons.append(f"🔥 Volume buildup LONG: {buildup} périodes (setup fort)")
        elif buildup >= 3:
            score += 12
            reasons.append(f"📊 Volume buildup: {buildup} périodes (setup confirmé)")
        elif buildup >= 2:
            score += 7
            reasons.append(f"📈 Volume buildup émergent: {buildup} périodes")
        elif buildup == 1:
            score += 3

        # 2. Volume progression - 10 points max
        rel_vol = self.safe_float(current.get("relative_volume"), 1.0)

        # WARNING si volume spike - v4.1 CONTEXTUALISÉ avec RSI/ROC
        vol_spike = self.safe_float(current.get("volume_spike_multiplier"), 1.0)
        if vol_spike >= 3.0 or rel_vol > 3.0:
            # CONTEXTUALISER: Volume 3x+ n'est "trop tard" QUE si RSI et ROC aussi élevés
            rsi = self.safe_float(current.get("rsi_14"))
            roc = self.safe_float(current.get("roc_10"))

            if rsi > 70 and roc > 0.005:  # ROC >0.5%
                # TOUS les signaux lagging élevés = VRAIMENT trop tard
                warnings.append(
                    f"❌ VOLUME SPIKE {vol_spike:.1f}x + RSI {rsi:.0f} + ROC {roc*100:.2f}% - TROP TARD"
                )
            elif rsi > 60 or roc > 0.003:  # ROC >0.3%
                # Signaux modérément élevés = breakout avancé mais pas trop tard
                warnings.append(
                    f"⚠️ Volume {vol_spike:.1f}x + RSI {rsi:.0f} - Breakout avancé"
                )
            else:
                # Volume spike mais RSI/ROC modérés = BREAKOUT INSTITUTIONNEL EARLY
                warnings.append(
                    f"ℹ️ Volume spike {vol_spike:.1f}x mais RSI {rsi:.0f} / ROC {roc*100:.2f}% modérés - Potentiel breakout early"
                )
            # Continue scoring, c'est juste une info contextuelle

        if historical and len(historical) >= 3:
            # Calculer progression volume
            recent_vols = []
            for h in historical[-3:]:
                recent_vols.append(self.safe_float(h.get("relative_volume"), 1.0))
            recent_vols.append(rel_vol)

            # Volume en progression PROGRESSIVE?
            is_increasing = all(
                recent_vols[i] <= recent_vols[i + 1]
                for i in range(len(recent_vols) - 1)
            )

            # Vérifier que progression est MODÉRÉE (pas spike)
            if (
                is_increasing
                and 1.2 <= rel_vol <= 2.0
                and rel_vol > recent_vols[0] * 1.2
            ):
                score += 10
                reasons.append(
                    f"✅ Volume BUILDUP progressif: {recent_vols[0]:.2f}x → {rel_vol:.2f}x"
                )
            elif is_increasing and rel_vol <= 2.5:
                score += 5
                reasons.append(
                    f"📊 Volume en progression: {recent_vols[0]:.2f}x → {rel_vol:.2f}x"
                )
            elif rel_vol > 2.5:
                warnings.append(
                    f"⚠️ Volume élevé: {rel_vol:.2f}x (risque spike imminent)"
                )

        # 3. Relative volume dans zone buildup - 5 points max
        # ZONE OPTIMALE: 1.2-1.8x (buildup confirmé, pas spike)
        if 1.2 <= rel_vol <= 1.5:
            score += 5
            reasons.append(f"✅ Volume OPTIMAL: {rel_vol:.2f}x (buildup sain)")
        elif 1.5 < rel_vol <= 1.8:
            score += 4
            reasons.append(f"📊 Volume buildup: {rel_vol:.2f}x")
        elif 1.8 < rel_vol <= 2.2:
            score += 2
            reasons.append(f"⚠️ Volume élevé: {rel_vol:.2f}x (surveiller spike)")
        elif rel_vol < 1.0:
            warnings.append(f"⚠️ Volume faible: {rel_vol:.2f}x (pas de setup)")
        elif rel_vol > 2.5:
            warnings.append(f"❌ Volume très élevé: {rel_vol:.2f}x (trop tard)")

        return min(score, 30)

    def _score_micro_patterns(
        self,
        current: dict,
        historical: list[dict] | None,
        reasons: list[str],
        warnings: list[str],
    ) -> float:
        """
        Score micro-patterns CORRIGÉ pour EARLY entry.

        ❌ ANCIEN SYSTÈME (FAUX):
        - RSI 50-70 climbing = bonus (déjà overbought!)

        ✅ NOUVEAU SYSTÈME (CORRECT):
        - Higher lows (consolidation) = bonus
        - RSI 35-55 climbing = MAX bonus (sortie oversold)
        - RSI >70 = REJET (overbought = trop tard)
        - MACD histogram expansion modérée = bonus

        20 points max:
        - 10 pts: Higher lows séquence (consolidation avant breakout)
        - 7 pts: RSI climbing dans zone 35-55 (sortie oversold)
        - 3 pts: MACD histogram expansion MODÉRÉE
        """
        score = 0.0

        if not historical or len(historical) < 3:
            return 0.0

        # 1. Higher lows pattern - 10 points max
        # Consolidation = setup formation
        recent_supports = []
        for h in historical[-3:]:
            sup = self.safe_float(h.get("nearest_support"))
            if sup > 0:
                recent_supports.append(sup)

        current_sup = self.safe_float(current.get("nearest_support"))
        if current_sup > 0:
            recent_supports.append(current_sup)

        if len(recent_supports) >= 3:
            # Vérifier higher lows
            is_higher_lows = all(
                recent_supports[i] <= recent_supports[i + 1]
                for i in range(len(recent_supports) - 1)
            )

            if is_higher_lows:
                score += 10
                reasons.append(
                    f"✅ Higher lows (consolidation): {recent_supports[0]:.2f} → {recent_supports[-1]:.2f}"
                )

        # 2. RSI climbing dans zone EARLY - 7 points max
        # CORRIGÉ: Zone 35-55 (sortie oversold), pas 50-70 (overbought)
        recent_rsis = []
        for h in historical[-3:]:
            rsi = self.safe_float(h.get("rsi_14"))
            if rsi > 0:
                recent_rsis.append(rsi)

        current_rsi = self.safe_float(current.get("rsi_14"))
        if current_rsi > 0:
            recent_rsis.append(current_rsi)

        # WARNING si RSI overbought - v4.1 CONTEXTUALISÉ avec trend strength
        if current_rsi > 70:
            # RSI >70 n'est problématique QUE si trend fort déjà établi (ADX élevé)
            adx = self.safe_float(current.get("adx_14"))
            macd_hist = self.safe_float(current.get("macd_histogram"))

            if current_rsi > 80 and adx > 35 and macd_hist > 15:
                # RSI très élevé + trend fort confirmé = TROP TARD
                warnings.append(
                    f"❌ RSI EXTREME {current_rsi:.0f} + ADX {adx:.0f} + MACD {macd_hist:.1f} - TROP TARD"
                )
            elif current_rsi > 75 and adx > 30:
                # RSI élevé + trend modéré = déjà bien avancé
                warnings.append(
                    f"⚠️ RSI {current_rsi:.0f} + ADX {adx:.0f} - Mouvement avancé"
                )
            else:
                # RSI >70 mais ADX faible = DÉBUT de trend fort (acceptable)
                warnings.append(
                    f"ℹ️ RSI {current_rsi:.0f} mais ADX {adx:.0f} - Début trend fort possible"
                )
            # Continue scoring, c'est juste une info contextuelle

        if len(recent_rsis) >= 3:
            # RSI en progression ET dans zone EARLY (35-55)
            rsi_increasing = recent_rsis[-1] > recent_rsis[0]
            rsi_in_early_zone = 35 <= current_rsi <= 55

            if rsi_increasing and rsi_in_early_zone:
                rsi_delta = recent_rsis[-1] - recent_rsis[0]
                if rsi_delta > 8:
                    score += 7
                    reasons.append(
                        f"✅ RSI climbing OPTIMAL: {recent_rsis[0]:.0f} → {current_rsi:.0f} (sortie oversold)"
                    )
                elif rsi_delta > 4:
                    score += 5
                    reasons.append(
                        f"📊 RSI climbing: {recent_rsis[0]:.0f} → {current_rsi:.0f}"
                    )
            elif current_rsi > 65:
                warnings.append(
                    f"⚠️ RSI élevé: {current_rsi:.0f} (risque overbought)"
                )

        # 3. MACD histogram expansion MODÉRÉE - 3 points max
        # CORRIGÉ: Expansion modérée (pas forte) pour early
        recent_macds = []
        for h in historical[-3:]:
            macd_hist = self.safe_float(h.get("macd_histogram"))
            recent_macds.append(macd_hist)

        current_macd = self.safe_float(current.get("macd_histogram"))
        recent_macds.append(current_macd)

        # MACD histogram en expansion MODÉRÉE
        if len(recent_macds) >= 3 and all(m > 0 for m in recent_macds[-2:]):
            macd_change = recent_macds[-1] - recent_macds[0]
            # Expansion modérée = setup, pas explosion
            if 0 < macd_change <= 10:
                score += 3
                reasons.append(
                    f"✅ MACD expansion modérée: {recent_macds[0]:.1f} → {current_macd:.1f}"
                )
            elif macd_change > 10:
                score += 1
                warnings.append(
                    f"⚠️ MACD expansion forte: {macd_change:.1f} (déjà lancé)"
                )

        return min(score, 20)

    def _score_order_flow(
        self, current: dict, reasons: list[str], warnings: list[str]
    ) -> float:
        """
        Score order flow pressure CORRIGÉ.

        ❌ ANCIEN SYSTÈME: OBV et trade intensity comme indicateurs principaux

        ✅ NOUVEAU SYSTÈME:
        - OBV oscillator positif MODÉRÉ = optimal (pas pic)
        - Trade intensity 1.0-1.5x = optimal (buildup)
        - Trade intensity >2.0x = warning (trop actif)
        - Break probability >60% = bonus

        15 points max (augmenté de 13):
        - 6 pts: OBV oscillator positif modéré
        - 4 pts: Trade intensity dans zone buildup
        - 2 pts: Quote volume ratio favorable
        - 3 pts: Break probability élevée
        """
        score = 0.0

        # 1. OBV Oscillator - 6 points max
        # CORRIGÉ: Valeurs modérées = buildup, pas pic
        obv_osc = self.safe_float(current.get("obv_oscillator"))
        if 50 <= obv_osc <= 150:
            # OPTIMAL: OBV positif modéré
            score += 6
            reasons.append(f"✅ OBV optimal: {obv_osc:.0f} (buildup sain)")
        elif 150 < obv_osc <= 250:
            score += 4
            reasons.append(f"📊 OBV positif: {obv_osc:.0f}")
        elif obv_osc > 250:
            score += 2
            warnings.append(f"⚠️ OBV très élevé: {obv_osc:.0f} (pic possible)")
        elif 0 < obv_osc < 50:
            score += 2
        elif obv_osc < -100:
            warnings.append(f"⚠️ OBV négatif: {obv_osc:.0f} (pression vendeuse)")

        # 2. Trade Intensity - 4 points max
        # CORRIGÉ: Intensité modérée = buildup, pas frenzy
        intensity = self.safe_float(current.get("trade_intensity"))
        if 1.0 <= intensity <= 1.5:
            # OPTIMAL: Activité modérée = buildup
            score += 4
            reasons.append(f"✅ Trade intensity OPTIMAL: {intensity:.2f}x (buildup)")
        elif 1.5 < intensity <= 2.0:
            score += 2
            reasons.append(f"📊 Trade intensity: {intensity:.2f}x")
        elif intensity > 2.0:
            score += 1
            warnings.append(
                f"⚠️ Trade intensity élevée: {intensity:.2f}x (frenzy démarré)"
            )
        elif 0.8 <= intensity < 1.0:
            score += 1

        # 3. Quote Volume Ratio - 2 points max
        qv_ratio = self.safe_float(current.get("quote_volume_ratio"))
        if 1.1 <= qv_ratio <= 1.4:
            score += 2
            reasons.append(f"📊 QV ratio favorable: {qv_ratio:.2f}")
        elif qv_ratio > 1.5:
            score += 1
            warnings.append(f"⚠️ QV ratio élevé: {qv_ratio:.2f}")

        # 4. Break Probability - 3 points max
        break_prob = self.safe_float(current.get("break_probability"), 0.5)
        if break_prob > 0.70:
            score += 3
            reasons.append(
                f"✅ Résistance faible: {break_prob*100:.0f}% break probability"
            )
        elif break_prob > 0.60:
            score += 2
            reasons.append(f"📊 Break probable: {break_prob*100:.0f}%")
        elif break_prob > 0.50:
            score += 1

        return min(score, 15)

    def _determine_signal_level(
        self, score: float, current: dict, _historical: list[dict] | None
    ) -> tuple[EarlySignalLevel, int, float]:
        """
        Détermine le niveau de signal + timing estimé CORRIGÉ.

        ❌ ANCIEN SYSTÈME: RSI >70 et vol_spike >2.5 = 60% complété (FAUX)

        ✅ NOUVEAU SYSTÈME:
        - RSI >70 OU vol_spike >3.0 = 90%+ complété (TROP TARD)
        - RSI 60-70 OU vol_spike >2.0 = 70% complété (déjà bien avancé)
        - RSI 50-60 = 40% complété
        - RSI 35-50 = 15% complété (EARLY STAGE)

        Returns:
            (level, entry_window_seconds, move_completion_pct)
        """
        # Estimer % du mouvement déjà fait (CORRIGÉ)
        rsi = self.safe_float(current.get("rsi_14"))
        rel_vol = self.safe_float(current.get("relative_volume"), 1.0)
        vol_spike = self.safe_float(current.get("volume_spike_multiplier"), 1.0)

        # CORRIGÉ: Heuristique basée sur signaux LAGGING réels
        if rsi > 70 or vol_spike >= 3.0 or rel_vol > 3.0:
            move_completion_pct = 90.0  # TROP TARD
        elif rsi > 60 or vol_spike >= 2.0 or rel_vol > 2.2:
            move_completion_pct = 70.0  # Bien avancé
        elif rsi > 50 or rel_vol > 1.5:
            move_completion_pct = 40.0  # Modérément avancé
        elif rsi >= 35:
            move_completion_pct = 15.0  # Early stage - OPTIMAL
        else:
            move_completion_pct = 5.0  # Très early

        # Déterminer niveau et window (seuils ABAISSÉS)
        if score >= self.THRESHOLDS["too_late"]:
            return EarlySignalLevel.TOO_LATE, 0, move_completion_pct

        if score >= self.THRESHOLDS["entry_now"]:
            # Entry window = 10-30s
            return EarlySignalLevel.ENTRY_NOW, 20, move_completion_pct

        if score >= self.THRESHOLDS["prepare"]:
            # Préparer entry dans 20-60s
            return EarlySignalLevel.PREPARE, 40, move_completion_pct

        if score >= self.THRESHOLDS["watch"]:
            # Surveiller, entry potentielle dans 60-120s
            return EarlySignalLevel.WATCH, 90, move_completion_pct

        return EarlySignalLevel.NONE, 0, move_completion_pct

    def _generate_recommendations(
        self,
        level: EarlySignalLevel,
        score: float,
        entry_window_seconds: int,
        move_completion_pct: float,
        recommendations: list[str],
        warnings: list[str],
    ):
        """Génère recommandations selon niveau."""
        if level == EarlySignalLevel.ENTRY_NOW:
            recommendations.append(f"🚀 ENTRY WINDOW NOW - Score: {score:.0f}/100")
            recommendations.append(f"⏱️ Fenêtre entry: {entry_window_seconds}s estimé")
            recommendations.append("💡 Préparer ordre LIMIT à entry_optimal")

            if move_completion_pct > 50:
                warnings.append(
                    f"⚠️ Mouvement déjà {move_completion_pct:.0f}% avancé (risque entry tardive)"
                )

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
            recommendations=["Pas de données disponibles"],
        )
