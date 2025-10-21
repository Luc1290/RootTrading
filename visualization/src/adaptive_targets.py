"""
Adaptive Targets System - Institutional Trading v4.1
====================================================

Système de calcul de targets adaptatifs basé sur:
- Score de l'opportunité (qualité du signal)
- Volatilité du marché (ATR)
- Timeframe (1m = court terme, 1h = long terme)
- Régime de marché (trending vs ranging)

Author: RootTrading
Version: 4.1
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TargetProfile(Enum):
    """Profils de targets selon qualité du signal"""
    AGGRESSIVE = "aggressive"      # Score 80+ : Targets ambitieux
    STANDARD = "standard"          # Score 65-80 : Targets équilibrés
    CONSERVATIVE = "conservative"  # Score <65 : Targets prudents


@dataclass
class AdaptiveTargets:
    """Résultats du calcul de targets adaptatifs"""
    tp1: float
    tp2: float
    tp3: Optional[float]
    stop_loss: float

    # Métriques pour analyse
    profile_used: TargetProfile
    volatility_multiplier: float
    timeframe_multiplier: float
    risk_reward_ratio: float

    # Détails du calcul
    base_tp1_pct: float
    base_tp2_pct: float
    base_tp3_pct: Optional[float]
    adjusted_tp1_pct: float
    adjusted_tp2_pct: float
    adjusted_tp3_pct: Optional[float]


class AdaptiveTargetSystem:
    """
    Système de calcul de targets adaptatifs pour trading institutionnel.

    Philosophie:
    - Les signaux de haute qualité (score 80+) méritent des targets ambitieux
    - La volatilité doit élargir ou rétrécir les targets (ATR-based)
    - Le timeframe influence la durée de détention et donc les targets
    - Le régime de marché affecte la probabilité d'atteindre les targets
    """

    def __init__(self):
        """Initialisation avec les profils de targets par défaut"""

        # PROFILS DE TARGETS (% de mouvement depuis le prix d'entrée)
        self.target_profiles = {
            TargetProfile.AGGRESSIVE: {
                "tp1": 0.008,  # 0.8% - Premier objectif rapide
                "tp2": 0.015,  # 1.5% - Objectif principal
                "tp3": 0.025,  # 2.5% - Extension si momentum fort
                "sl": 0.004,   # 0.4% - SL serré (R/R = 2:1 minimum)
            },
            TargetProfile.STANDARD: {
                "tp1": 0.006,  # 0.6%
                "tp2": 0.012,  # 1.2%
                "tp3": 0.020,  # 2.0%
                "sl": 0.005,   # 0.5% (R/R = 1.2:1 minimum)
            },
            TargetProfile.CONSERVATIVE: {
                "tp1": 0.004,  # 0.4%
                "tp2": 0.008,  # 0.8%
                "tp3": None,   # Pas de TP3 pour signaux faibles
                "sl": 0.006,   # 0.6% (R/R = 0.66:1 minimum)
            },
        }

        # MULTIPLICATEURS DE VOLATILITÉ
        # ATR normalisé (ATR / prix) vs targets
        self.volatility_bands = {
            "very_low": (0.0, 0.003, 0.8),    # ATR <0.3% -> Réduire targets (marché calme)
            "low": (0.003, 0.006, 0.9),       # ATR 0.3-0.6%
            "normal": (0.006, 0.012, 1.0),    # ATR 0.6-1.2% -> Profil standard
            "high": (0.012, 0.020, 1.15),     # ATR 1.2-2.0% -> Augmenter targets
            "very_high": (0.020, 1.0, 1.3),   # ATR >2.0% -> Targets ambitieux (crypto volatil)
        }

        # MULTIPLICATEURS DE TIMEFRAME
        self.timeframe_multipliers = {
            "1m": 0.7,   # Court terme = targets plus serrés
            "5m": 1.0,   # Référence
            "15m": 1.3,  # Moyen terme = targets plus larges
            "1h": 1.6,   # Long terme = targets ambitieux
        }

        # AJUSTEMENTS DE RÉGIME DE MARCHÉ
        self.regime_adjustments = {
            "TRENDING_BULL": 1.1,      # Trending = laisser courir les profits
            "TRENDING_BEAR": 0.9,      # Contre-tendance = prendre profits rapidement
            "RANGING": 0.85,           # Range = targets serrés (retour à la moyenne)
            "VOLATILE": 1.15,          # Volatil = profiter des grands mouvements
            "BREAKOUT_BULL": 1.2,      # Breakout = targets ambitieux
            "BREAKOUT_BEAR": 0.85,     # Contre breakout baissier = prudence
            "TRANSITION": 0.9,         # Transition = réduire exposition
        }

    def calculate_targets(
        self,
        entry_price: float,
        score: float,
        atr: float,
        timeframe: str = "5m",
        regime: Optional[str] = None,
        side: str = "BUY"
    ) -> AdaptiveTargets:
        """
        Calcule les targets adaptatifs pour une opportunité.

        Args:
            entry_price: Prix d'entrée prévu
            score: Score de l'opportunité (0-100)
            atr: Average True Range actuel
            timeframe: Timeframe du signal ("1m", "5m", "15m", "1h")
            regime: Régime de marché détecté (optionnel)
            side: Direction du trade ("BUY" ou "SELL")

        Returns:
            AdaptiveTargets avec tous les calculs
        """

        # 1. SÉLECTION DU PROFIL SELON SCORE
        profile = self._select_profile(score)
        base_targets = self.target_profiles[profile]

        # 2. CALCUL DU MULTIPLICATEUR DE VOLATILITÉ
        volatility_mult = self._calculate_volatility_multiplier(atr, entry_price)

        # 3. MULTIPLICATEUR DE TIMEFRAME
        tf_mult = self.timeframe_multipliers.get(timeframe, 1.0)

        # 4. AJUSTEMENT DE RÉGIME (si fourni)
        regime_mult = 1.0
        if regime and regime in self.regime_adjustments:
            regime_mult = self.regime_adjustments[regime]

        # 5. MULTIPLICATEUR TOTAL
        total_mult = volatility_mult * tf_mult * regime_mult

        # 6. CALCUL DES TARGETS FINAUX
        adjusted_tp1_pct = base_targets["tp1"] * total_mult
        adjusted_tp2_pct = base_targets["tp2"] * total_mult
        adjusted_tp3_pct = base_targets["tp3"] * total_mult if base_targets["tp3"] else None
        adjusted_sl_pct = base_targets["sl"] * 0.9  # SL moins affecté par volatilité

        # 7. CONVERSION EN PRIX
        if side == "BUY":
            tp1 = entry_price * (1 + adjusted_tp1_pct)
            tp2 = entry_price * (1 + adjusted_tp2_pct)
            tp3 = entry_price * (1 + adjusted_tp3_pct) if adjusted_tp3_pct else None
            stop_loss = entry_price * (1 - adjusted_sl_pct)
        else:  # SELL
            tp1 = entry_price * (1 - adjusted_tp1_pct)
            tp2 = entry_price * (1 - adjusted_tp2_pct)
            tp3 = entry_price * (1 - adjusted_tp3_pct) if adjusted_tp3_pct else None
            stop_loss = entry_price * (1 + adjusted_sl_pct)

        # 8. CALCUL DU RISK/REWARD
        risk = abs(entry_price - stop_loss)
        reward = abs(tp2 - entry_price)  # Basé sur TP2 (objectif principal)
        rr_ratio = reward / risk if risk > 0 else 0

        logger.info(
            f"Adaptive Targets: Profile={profile.value}, Score={score:.1f}, "
            f"Mult: Vol={volatility_mult:.2f} TF={tf_mult:.2f} Regime={regime_mult:.2f} "
            f"Total={total_mult:.2f}, R/R={rr_ratio:.2f}"
        )

        return AdaptiveTargets(
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            stop_loss=stop_loss,
            profile_used=profile,
            volatility_multiplier=volatility_mult,
            timeframe_multiplier=tf_mult,
            risk_reward_ratio=rr_ratio,
            base_tp1_pct=base_targets["tp1"],
            base_tp2_pct=base_targets["tp2"],
            base_tp3_pct=base_targets["tp3"],
            adjusted_tp1_pct=adjusted_tp1_pct,
            adjusted_tp2_pct=adjusted_tp2_pct,
            adjusted_tp3_pct=adjusted_tp3_pct,
        )

    def _select_profile(self, score: float) -> TargetProfile:
        """Sélectionne le profil de targets selon le score"""
        if score >= 80:
            return TargetProfile.AGGRESSIVE
        elif score >= 65:
            return TargetProfile.STANDARD
        else:
            return TargetProfile.CONSERVATIVE

    def _calculate_volatility_multiplier(self, atr: float, price: float) -> float:
        """
        Calcule le multiplicateur basé sur la volatilité normalisée.

        Args:
            atr: Average True Range
            price: Prix actuel

        Returns:
            Multiplicateur entre 0.8 et 1.3
        """
        # ATR normalisé (en %)
        atr_pct = (atr / price) if price > 0 else 0

        # Trouver la bande de volatilité correspondante
        for band_name, (min_atr, max_atr, multiplier) in self.volatility_bands.items():
            if min_atr <= atr_pct < max_atr:
                logger.debug(f"Volatility band: {band_name}, ATR={atr_pct:.4f}, Mult={multiplier}")
                return multiplier

        # Par défaut: normal
        return 1.0

    def get_targets_explanation(self, targets: AdaptiveTargets) -> str:
        """
        Genere une explication detaillee des targets calcules.

        Args:
            targets: Resultat du calcul de targets

        Returns:
            Texte explicatif pour logging/debug
        """
        explanation = [
            f"ADAPTIVE TARGETS - Profile: {targets.profile_used.value.upper()}",
            f"",
            f"Base Targets:",
            f"  TP1: {targets.base_tp1_pct*100:.2f}% -> Adjusted: {targets.adjusted_tp1_pct*100:.2f}%",
            f"  TP2: {targets.base_tp2_pct*100:.2f}% -> Adjusted: {targets.adjusted_tp2_pct*100:.2f}%",
        ]

        if targets.base_tp3_pct:
            explanation.append(
                f"  TP3: {targets.base_tp3_pct*100:.2f}% -> Adjusted: {targets.adjusted_tp3_pct*100:.2f}%"
            )

        explanation.extend([
            f"",
            f"Multipliers Applied:",
            f"  Volatility: {targets.volatility_multiplier:.2f}x",
            f"  Timeframe: {targets.timeframe_multiplier:.2f}x",
            f"  Total: {targets.volatility_multiplier * targets.timeframe_multiplier:.2f}x",
            f"",
            f"Risk/Reward Ratio: {targets.risk_reward_ratio:.2f}:1",
        ])

        return "\n".join(explanation)

    def validate_targets(self, targets: AdaptiveTargets, entry_price: float) -> Tuple[bool, str]:
        """
        Valide que les targets calculés sont raisonnables.

        Args:
            targets: Targets calculés
            entry_price: Prix d'entrée

        Returns:
            (is_valid, reason)
        """
        # 1. Vérifier que TP1 < TP2 < TP3
        if targets.tp1 >= targets.tp2:
            return False, "TP1 must be < TP2"

        if targets.tp3 and targets.tp2 >= targets.tp3:
            return False, "TP2 must be < TP3"

        # 2. Vérifier que SL est du bon côté
        if targets.tp1 > entry_price:  # BUY
            if targets.stop_loss >= entry_price:
                return False, "Stop loss must be below entry for BUY"
        else:  # SELL
            if targets.stop_loss <= entry_price:
                return False, "Stop loss must be above entry for SELL"

        # 3. Vérifier le R/R minimum (doit être >= 0.5)
        if targets.risk_reward_ratio < 0.5:
            return False, f"Risk/Reward too low: {targets.risk_reward_ratio:.2f}"

        # 4. Vérifier que les targets ne sont pas trop éloignés (max 5% pour crypto)
        max_target_pct = 0.05  # 5%
        tp2_distance = abs(targets.tp2 - entry_price) / entry_price

        if tp2_distance > max_target_pct:
            return False, f"TP2 too far: {tp2_distance*100:.1f}% (max {max_target_pct*100}%)"

        return True, "Valid"


# EXEMPLE D'UTILISATION
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialiser le système
    target_system = AdaptiveTargetSystem()

    # Cas 1: Signal de haute qualité, volatilité normale, 5m
    print("=" * 60)
    print("CAS 1: Signal haute qualité (score 85)")
    targets = target_system.calculate_targets(
        entry_price=100.0,
        score=85,
        atr=0.8,  # ATR normalisé = 0.008 (0.8%)
        timeframe="5m",
        regime="TRENDING_BULL",
        side="BUY"
    )
    print(target_system.get_targets_explanation(targets))
    is_valid, reason = target_system.validate_targets(targets, 100.0)
    print(f"\nValidation: {is_valid} - {reason}")

    # Cas 2: Signal moyen, haute volatilité, 1h
    print("\n" + "=" * 60)
    print("CAS 2: Signal moyen (score 68), haute volatilité, 1h")
    targets = target_system.calculate_targets(
        entry_price=100.0,
        score=68,
        atr=1.5,  # ATR normalisé = 0.015 (1.5%)
        timeframe="1h",
        regime="VOLATILE",
        side="BUY"
    )
    print(target_system.get_targets_explanation(targets))
    is_valid, reason = target_system.validate_targets(targets, 100.0)
    print(f"\nValidation: {is_valid} - {reason}")

    # Cas 3: Signal faible, basse volatilité, 1m
    print("\n" + "=" * 60)
    print("CAS 3: Signal faible (score 55), basse volatilité, 1m")
    targets = target_system.calculate_targets(
        entry_price=100.0,
        score=55,
        atr=0.25,  # ATR normalisé = 0.0025 (0.25%)
        timeframe="1m",
        regime="RANGING",
        side="BUY"
    )
    print(target_system.get_targets_explanation(targets))
    is_valid, reason = target_system.validate_targets(targets, 100.0)
    print(f"\nValidation: {is_valid} - {reason}")
