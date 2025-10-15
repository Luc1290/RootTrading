"""
Classification des stratégies par type pour adaptation au régime de marché.

Ce module définit les familles de stratégies et leurs caractéristiques
pour permettre une validation adaptative selon le régime de marché.
"""

# Classification des 28 stratégies par famille principale
STRATEGY_FAMILIES = {
    "trend_following": {
        "strategies": [
            "MACD_Crossover_Strategy",
            "EMA_Cross_Strategy",
            "ADX_Direction_Strategy",
            "HullMA_Slope_Strategy",
            "TEMA_Slope_Strategy",
            "TRIX_Crossover_Strategy",
            "PPO_Crossover_Strategy",
            "ROC_Threshold_Strategy",
            "Supertrend_Reversal_Strategy",  # CORRIGÉ: Supertrend est trend-following
            "ParabolicSAR_Bounce_Strategy",  # CORRIGÉ: SAR est trend-following
        ],
        "best_regimes": ["TRENDING_BULL", "TRENDING_BEAR"],
        "acceptable_regimes": ["BREAKOUT_BULL", "BREAKOUT_BEAR", "TRANSITION"],
        "poor_regimes": ["RANGING", "VOLATILE"],
        "characteristics": "Suit les tendances établies, performances optimales en marché directionnel",
    },
    "mean_reversion": {
        "strategies": [
            "RSI_Cross_Strategy",
            "Stochastic_Oversold_Buy_Strategy",
            "StochRSI_Rebound_Strategy",
            "WilliamsR_Rebound_Strategy",
            "CCI_Reversal_Strategy",
            "Bollinger_Touch_Strategy",
            "ZScore_Extreme_Reversal_Strategy",
        ],
        "best_regimes": ["RANGING"],
        "acceptable_regimes": ["VOLATILE", "TRANSITION"],
        "poor_regimes": ["TRENDING_BULL", "TRENDING_BEAR"],
        "characteristics": "Exploite les retours à la moyenne, optimal en marché latéral",
    },
    "breakout": {
        "strategies": [
            "ATR_Breakout_Strategy",
            "Donchian_Breakout_Strategy",
            "Range_Breakout_Confirmation_Strategy",
            "Support_Breakout_Strategy",
        ],
        "best_regimes": ["BREAKOUT_BULL", "BREAKOUT_BEAR", "VOLATILE"],
        "acceptable_regimes": ["TRANSITION", "TRENDING_BULL", "TRENDING_BEAR"],
        "poor_regimes": ["RANGING"],
        "characteristics": "Détecte les cassures de niveaux, optimal en volatilité",
    },
    "volume_based": {
        "strategies": [
            "OBV_Crossover_Strategy",
            "Volume_Spike_Strategy",  # Si elle existe
            "Liquidity_Sweep_Buy_Strategy",
            "Pump_Dump_Pattern_Strategy",
        ],
        "best_regimes": ["BREAKOUT_BULL", "BREAKOUT_BEAR", "VOLATILE"],
        "acceptable_regimes": ["TRENDING_BULL", "TRENDING_BEAR", "TRANSITION"],
        "poor_regimes": ["RANGING"],
        "characteristics": "Analyse les mouvements de volume, détecte accumulation/distribution",
    },
    "structure_based": {
        "strategies": [
            "VWAP_Support_Resistance_Strategy",
            "Spike_Reaction_Buy_Strategy",
            "MultiTF_ConfluentEntry_Strategy",
            "Resistance_Rejection_Strategy",  # DÉPLACÉ: Rejection = analyse de structure/niveaux
        ],
        "best_regimes": [
            "TRENDING_BULL",
            "TRENDING_BEAR",
            "RANGING",
            "UNKNOWN",
            "TRANSITION",
        ],  # PATCH 4: +UNKNOWN/TRANSITION
        "acceptable_regimes": ["VOLATILE", "BREAKOUT_BULL", "BREAKOUT_BEAR"],
        "poor_regimes": [],  # Adaptable à tous les régimes
        "characteristics": "Analyse la structure de marché, très adaptable",
    },
    # NOUVELLES FAMILLES POUR SCALPING
    "flow": {
        "strategies": [
            # Stratégies basées sur l'analyse du flux d'ordres (à adapter selon vos stratégies)
            "Liquidity_Sweep_Buy_Strategy",  # Déjà dans volume_based mais aussi flow
            "OrderFlow_Imbalance_Strategy",  # Si existe
            "BookPressure_Strategy",  # Si existe
        ],
        "best_regimes": ["VOLATILE", "BREAKOUT_BULL", "BREAKOUT_BEAR"],
        "acceptable_regimes": ["TRENDING_BULL", "TRENDING_BEAR", "TRANSITION"],
        "poor_regimes": ["RANGING"],  # Flow moins utile en range calme
        "characteristics": "Analyse le flux d'ordres et la pression d'achat/vente",
    },
    "contrarian": {
        "strategies": [
            # Stratégies contrariennes pures (contre-tendance extrême)
            "ZScore_Extreme_Reversal_Strategy",  # Déjà dans mean_reversion mais très contrarian
            "Exhaustion_Reversal_Strategy",  # Si existe
            "Sentiment_Contrarian_Strategy",  # Si existe
        ],
        "best_regimes": ["RANGING", "TRANSITION"],  # Optimal en range et transitions
        "acceptable_regimes": ["VOLATILE"],  # Peut fonctionner en volatilité
        "poor_regimes": [
            "TRENDING_BULL",
            "TRENDING_BEAR",
            "BREAKOUT_BULL",
            "BREAKOUT_BEAR",
        ],
        "characteristics": "Prend des positions contre la tendance dominante, risqué mais rentable sur retournements",
    },
}

# Détection des doublons pour debug
from collections import defaultdict

_dups = defaultdict(list)
for fam, cfg in STRATEGY_FAMILIES.items():
    for s in cfg["strategies"]:
        _dups[s].append(fam)
DUPLICATE_MEMBERSHIPS = {s: fams for s, fams in _dups.items() if len(fams) > 1}
if DUPLICATE_MEMBERSHIPS:
    print("⚠️  Doublons de familles détectés:", DUPLICATE_MEMBERSHIPS)

# Mapping inverse : stratégie -> famille PRIMAIRE (évite l'écrasement silencieux)
STRATEGY_TO_FAMILY = {}
for family, config in STRATEGY_FAMILIES.items():
    for strategy in config["strategies"]:
        # setdefault pour ne PAS écraser la famille primaire (première occurrence)
        STRATEGY_TO_FAMILY.setdefault(strategy, family)

# Mapping complet : stratégie -> toutes ses familles (pour features avancées)
STRATEGY_FAMILY_TAGS = defaultdict(list)
for family, cfg in STRATEGY_FAMILIES.items():
    for s in cfg["strategies"]:
        if family not in STRATEGY_FAMILY_TAGS[s]:
            STRATEGY_FAMILY_TAGS[s].append(family)

# Configuration des ajustements de confidence selon le régime
REGIME_CONFIDENCE_ADJUSTMENTS = {
    "TRENDING_BULL": {
        "trend_following": {"BUY": 1.2, "SELL": 0.7},  # Boost BUY, pénalise SELL
        "mean_reversion": {"BUY": 0.6, "SELL": 0.4},  # Très pénalisé en tendance
        "breakout": {"BUY": 1.1, "SELL": 0.8},  # Léger boost BUY
        "volume_based": {"BUY": 1.1, "SELL": 0.9},  # Neutre-positif
        "structure_based": {"BUY": 1.0, "SELL": 0.9},  # Quasi-neutre
        "flow": {"BUY": 1.2, "SELL": 0.8},  # Boost BUY car suit le flux
        "contrarian": {"BUY": 0.4, "SELL": 0.3},  # Très pénalisé contre tendance
    },
    "TRENDING_BEAR": {
        "trend_following": {
            "BUY": 0.7,
            "SELL": 1.2,
        },  # CORRIGÉ: Pénalisé mais pas extrême
        "mean_reversion": {
            "BUY": 0.6,
            "SELL": 0.9,
        },  # CORRIGÉ: Permet rebonds légitimes
        "breakout": {"BUY": 0.7, "SELL": 1.2},  # CORRIGÉ: Moins pénalisant
        "volume_based": {"BUY": 0.8, "SELL": 1.2},  # CORRIGÉ: Légèrement pénalisé
        "structure_based": {"BUY": 0.8, "SELL": 1.1},  # CORRIGÉ: Moins pénalisant
        "flow": {"BUY": 0.8, "SELL": 1.2},  # Suit le flux baissier
        "contrarian": {
            "BUY": 0.6,
            "SELL": 0.4,
        },  # Pénalisé mais BUY possible pour rebond
    },
    "RANGING": {
        "trend_following": {"BUY": 0.7, "SELL": 0.7},  # Pénalisé en ranging
        "mean_reversion": {"BUY": 1.3, "SELL": 1.3},  # Fortement boosté
        "breakout": {"BUY": 0.8, "SELL": 0.8},  # Légèrement pénalisé
        "volume_based": {"BUY": 0.9, "SELL": 0.9},  # Légèrement pénalisé
        "structure_based": {"BUY": 1.1, "SELL": 1.1},  # Légèrement boosté
        "flow": {"BUY": 0.8, "SELL": 0.8},  # Moins utile en range
        "contrarian": {"BUY": 1.2, "SELL": 1.2},  # Boosté car optimal en range
    },
    "VOLATILE": {
        "trend_following": {"BUY": 0.8, "SELL": 0.8},  # Pénalisé en volatilité
        "mean_reversion": {"BUY": 1.1, "SELL": 1.1},  # Boosté en volatilité
        "breakout": {"BUY": 1.2, "SELL": 1.2},  # Fortement boosté
        "volume_based": {"BUY": 1.2, "SELL": 1.2},  # Fortement boosté
        "structure_based": {"BUY": 1.0, "SELL": 1.0},  # Neutre
        "flow": {"BUY": 1.3, "SELL": 1.3},  # Très utile en volatilité
        "contrarian": {"BUY": 0.9, "SELL": 0.9},  # Légèrement pénalisé
    },
    "BREAKOUT_BULL": {
        "trend_following": {"BUY": 1.1, "SELL": 0.6},  # Favorise BUY
        "mean_reversion": {"BUY": 0.5, "SELL": 0.3},  # Très pénalisé
        "breakout": {"BUY": 1.4, "SELL": 0.7},  # Très fort boost BUY
        "volume_based": {"BUY": 1.3, "SELL": 0.8},  # Fort boost BUY
        "structure_based": {"BUY": 1.1, "SELL": 0.8},  # Léger boost BUY
        "flow": {"BUY": 1.4, "SELL": 0.6},  # Très fort en breakout
        "contrarian": {"BUY": 0.3, "SELL": 0.2},  # Très dangereux contre breakout
    },
    "BREAKOUT_BEAR": {
        "trend_following": {"BUY": 0.6, "SELL": 1.2},  # CORRIGÉ: Pénalisé mais viable
        "mean_reversion": {
            "BUY": 0.5,
            "SELL": 0.8,
        },  # CORRIGÉ: Minimum viable pour rebonds
        "breakout": {"BUY": 0.6, "SELL": 1.4},  # CORRIGÉ: Pénalisé mais pas extrême
        "volume_based": {"BUY": 0.7, "SELL": 1.3},  # CORRIGÉ: Réduit la pénalité
        "structure_based": {"BUY": 0.7, "SELL": 1.2},  # CORRIGÉ: Plus équilibré
        "flow": {"BUY": 0.6, "SELL": 1.4},  # Suit le flux baissier
        "contrarian": {"BUY": 0.5, "SELL": 0.3},  # Très risqué contre breakout bear
    },
    "TRANSITION": {
        "trend_following": {"BUY": 0.9, "SELL": 0.9},  # Légèrement pénalisé
        "mean_reversion": {"BUY": 1.0, "SELL": 1.0},  # Neutre
        "breakout": {"BUY": 1.0, "SELL": 1.0},  # Neutre
        "volume_based": {"BUY": 1.0, "SELL": 1.0},  # Neutre
        "structure_based": {"BUY": 1.0, "SELL": 1.0},  # Neutre
        "flow": {"BUY": 1.0, "SELL": 1.0},  # Neutre
        "contrarian": {"BUY": 1.1, "SELL": 1.1},  # Légèrement boosté en transition
    },
    "UNKNOWN": {
        # Régime inconnu : on reste conservateur
        "trend_following": {"BUY": 0.9, "SELL": 0.9},
        "mean_reversion": {"BUY": 0.9, "SELL": 0.9},
        "breakout": {"BUY": 0.9, "SELL": 0.9},
        "volume_based": {"BUY": 0.9, "SELL": 0.9},
        "structure_based": {"BUY": 0.95, "SELL": 0.95},  # Moins pénalisé car adaptable
        "flow": {"BUY": 0.95, "SELL": 0.95},  # Adaptable comme volume
        "contrarian": {"BUY": 0.9, "SELL": 0.9},  # Conservateur en régime inconnu
    },
}

# Seuils minimums de confidence requis selon le régime et la direction
# DURCISSEMENT MAJEUR pour éviter les entrées prématurées en marché baissier
REGIME_MIN_CONFIDENCE = {
    "TRENDING_BULL": {
        "BUY": 0.60,  # Légèrement durci (était 0.55)
        "SELL": 0.85,  # Très strict pour SELL contre-tendance (était 0.80)
    },
    "TRENDING_BEAR": {
        "BUY": 0.75,  # CORRIGÉ: Strict mais permet rebonds légitimes
        "SELL": 0.60,  # Inchangé
    },
    "RANGING": {
        "BUY": 0.70,  # Plus strict (était 0.60)
        "SELL": 0.70,  # Plus strict (était 0.60)
    },
    "VOLATILE": {
        "BUY": 0.70,  # CORRIGÉ: Réduit la barrière d'entrée
        "SELL": 0.65,  # CORRIGÉ: Moins restrictif
    },
    "BREAKOUT_BULL": {
        "BUY": 0.55,  # Légèrement durci (était 0.50)
        "SELL": 0.90,  # Très strict pour SELL (était 0.85)
    },
    "BREAKOUT_BEAR": {
        "BUY": 0.80,  # CORRIGÉ: Strict mais réaliste
        "SELL": 0.55,  # Inchangé
    },
    "TRANSITION": {
        "BUY": 0.65,  # CORRIGÉ: Retour à niveau raisonnable
        "SELL": 0.65,  # CORRIGÉ: Retour à niveau raisonnable
    },
    "UNKNOWN": {
        "BUY": 0.70,  # CORRIGÉ: Conservateur mais pas bloquant
        "SELL": 0.70,  # CORRIGÉ: Équilibré
    },
}


def get_strategy_family(strategy_name: str) -> str:
    """
    Retourne la famille PRIMAIRE d'une stratégie.

    Args:
        strategy_name: Nom de la stratégie

    Returns:
        Famille primaire de la stratégie ou 'unknown'
    """
    return STRATEGY_TO_FAMILY.get(strategy_name, "unknown")


def get_canonical_family(family: str) -> str:
    """
    Retourne la famille canonique pour le comptage consensus.
    Gère les équivalences (ex: flow -> volume_based).

    Args:
        family: Famille à normaliser

    Returns:
        Famille canonique pour le consensus
    """
    # Équivalences pour le consensus
    equivalences = {
        "flow": "volume_based",  # flow compte comme volume_based
        # Ajouter d'autres équivalences si nécessaire
    }
    return equivalences.get(family, family)


def get_all_strategy_families(strategy_name: str) -> list:
    """
    Retourne toutes les familles d'une stratégie (primaire + secondaires).

    Args:
        strategy_name: Nom de la stratégie

    Returns:
        Liste des familles (vide si stratégie inconnue)
    """
    return STRATEGY_FAMILY_TAGS.get(strategy_name, [])


def get_regime_adjustment(
    strategy_name: str, market_regime: str, signal_side: str
) -> float:
    """
    Retourne le multiplicateur de confidence pour une stratégie selon le régime.

    Args:
        strategy_name: Nom de la stratégie
        market_regime: Régime de marché actuel
        signal_side: Direction du signal (BUY/SELL)

    Returns:
        Multiplicateur de confidence (1.0 = neutre, >1 = boost, <1 = pénalité)
    """
    family = get_strategy_family(strategy_name)
    if family == "unknown":
        return 0.95  # Légère pénalité pour stratégie inconnue

    regime = market_regime.upper() if market_regime else "UNKNOWN"
    if regime not in REGIME_CONFIDENCE_ADJUSTMENTS:
        regime = "UNKNOWN"

    adjustments = REGIME_CONFIDENCE_ADJUSTMENTS[regime].get(family, {})
    return adjustments.get(signal_side, 1.0)


def get_min_confidence_for_regime(market_regime: str, signal_side: str) -> float:
    """
    Retourne la confidence minimum requise selon le régime et la direction.

    Args:
        market_regime: Régime de marché actuel
        signal_side: Direction du signal (BUY/SELL)

    Returns:
        Confidence minimum requise (0-1)
    """
    regime = market_regime.upper() if market_regime else "UNKNOWN"
    if regime not in REGIME_MIN_CONFIDENCE:
        regime = "UNKNOWN"

    return REGIME_MIN_CONFIDENCE[regime].get(signal_side, 0.65)


def is_strategy_optimal_for_regime(strategy_name: str, market_regime: str) -> bool:
    """
    Vérifie si une stratégie est optimale pour le régime actuel.

    Args:
        strategy_name: Nom de la stratégie
        market_regime: Régime de marché actuel

    Returns:
        True si la stratégie est optimale pour ce régime
    """
    family = get_strategy_family(strategy_name)
    if family == "unknown":
        return False

    family_config = STRATEGY_FAMILIES.get(family, {})
    best_regimes = family_config.get("best_regimes", [])

    regime = market_regime.upper() if market_regime else "UNKNOWN"
    return regime in best_regimes


def is_strategy_acceptable_for_regime(strategy_name: str, market_regime: str) -> bool:
    """
    Vérifie si une stratégie est au moins acceptable pour le régime actuel.

    Args:
        strategy_name: Nom de la stratégie
        market_regime: Régime de marché actuel

    Returns:
        True si la stratégie est acceptable ou optimale pour ce régime
    """
    family = get_strategy_family(strategy_name)
    if family == "unknown":
        return True  # On accepte par défaut les stratégies inconnues

    family_config = STRATEGY_FAMILIES.get(family, {})
    best_regimes = family_config.get("best_regimes", [])
    acceptable_regimes = family_config.get("acceptable_regimes", [])

    regime = market_regime.upper() if market_regime else "UNKNOWN"
    return regime in best_regimes or regime in acceptable_regimes


def should_allow_counter_trend_rebound(
    market_regime: str, signal_side: str, market_indicators: dict = None
) -> bool:
    """
    Vérifie si on doit permettre un signal contre-tendance (rebond) basé sur les conditions de marché.

    Args:
        market_regime: Régime de marché actuel
        signal_side: Direction du signal (BUY/SELL)
        market_indicators: Dictionnaire des indicateurs de marché (RSI, volume, etc.)

    Returns:
        True si le rebond contre-tendance est autorisé
    """
    if market_indicators is None:
        market_indicators = {}

    regime = market_regime.upper() if market_regime else "UNKNOWN"

    # Permettre les rebonds BUY en marché baissier si conditions réunies
    if regime in ["TRENDING_BEAR", "BREAKOUT_BEAR"] and signal_side == "BUY":
        rsi = market_indicators.get("rsi_14", 50)
        volume_ratio = market_indicators.get("volume_ratio", 1.0)
        price_change_24h = market_indicators.get("price_change_24h", 0)

        # Conditions de rebond légitime:
        oversold = rsi < 30  # RSI oversold
        volume_spike = volume_ratio > 1.8  # Volume élevé
        significant_drop = price_change_24h < -8  # Chute significative

        # Permettre si au moins 2 conditions sont réunies
        conditions_met = sum([oversold, volume_spike, significant_drop])
        return conditions_met >= 2

    # Permettre les rebonds SELL en marché haussier si conditions réunies
    elif regime in ["TRENDING_BULL", "BREAKOUT_BULL"] and signal_side == "SELL":
        rsi = market_indicators.get("rsi_14", 50)
        volume_ratio = market_indicators.get("volume_ratio", 1.0)
        price_change_24h = market_indicators.get("price_change_24h", 0)

        # Conditions de rebond baissier:
        overbought = rsi > 70  # RSI overbought
        volume_spike = volume_ratio > 1.8  # Volume élevé
        significant_rally = price_change_24h > 8  # Hausse significative

        # Permettre si au moins 2 conditions sont réunies
        conditions_met = sum([overbought, volume_spike, significant_rally])
        return conditions_met >= 2

    return False


class AdaptiveRegimeAdjuster:
    """
    Système d'ajustement adaptatif des multiplicateurs basé sur la performance historique.
    """

    def __init__(self):
        self.performance_history = {}
        self.min_samples = 10  # Minimum de trades pour ajuster

    def record_trade_result(
        self,
        strategy_name: str,
        regime: str,
        side: str,
        success: bool,
        confidence: float,
    ):
        """
        Enregistre le résultat d'un trade pour l'apprentissage adaptatif.

        Args:
            strategy_name: Nom de la stratégie
            regime: Régime de marché
            side: Direction du trade (BUY/SELL)
            success: Succès du trade
            confidence: Confidence du signal
        """
        family = get_strategy_family(strategy_name)
        key = f"{family}_{regime}_{side}"

        if key not in self.performance_history:
            self.performance_history[key] = {
                "successes": 0,
                "total": 0,
                "confidence_sum": 0.0,
            }

        stats = self.performance_history[key]
        stats["total"] += 1
        stats["confidence_sum"] += confidence
        if success:
            stats["successes"] += 1

    def get_adaptive_multiplier(
        self, strategy_name: str, regime: str, side: str, base_multiplier: float
    ) -> float:
        """
        Retourne un multiplicateur ajusté basé sur la performance historique.

        Args:
            strategy_name: Nom de la stratégie
            regime: Régime de marché
            side: Direction (BUY/SELL)
            base_multiplier: Multiplicateur de base

        Returns:
            Multiplicateur ajusté
        """
        family = get_strategy_family(strategy_name)
        key = f"{family}_{regime}_{side}"

        if key not in self.performance_history:
            return base_multiplier

        stats = self.performance_history[key]

        # Pas assez de données pour ajuster
        if stats["total"] < self.min_samples:
            return base_multiplier

        success_rate = stats["successes"] / stats["total"]
        avg_confidence = stats["confidence_sum"] / stats["total"]

        # Ajustement basé sur le taux de succès
        if success_rate > 0.75:  # Très bon performance
            adjustment = 1.1
        elif success_rate > 0.60:  # Bonne performance
            adjustment = 1.05
        elif success_rate < 0.35:  # Mauvaise performance
            adjustment = 0.85
        elif success_rate < 0.50:  # Performance moyenne-faible
            adjustment = 0.92
        else:  # Performance acceptable
            adjustment = 1.0

        # Ajustement basé sur la confidence moyenne
        if avg_confidence > 0.80:  # Signaux haute confidence
            adjustment *= 1.02
        elif avg_confidence < 0.60:  # Signaux faible confidence
            adjustment *= 0.98

        # Limiter l'ajustement pour éviter les extrêmes
        adjustment = max(0.75, min(1.25, adjustment))

        return base_multiplier * adjustment

    def get_regime_statistics(self, regime: str = None) -> dict:
        """
        Retourne les statistiques de performance par régime.

        Args:
            regime: Régime spécifique (optionnel)

        Returns:
            Dictionnaire des statistiques
        """
        if regime:
            regime = regime.upper()
            filtered_stats = {
                k: v for k, v in self.performance_history.items() if regime in k
            }
            return filtered_stats

        return self.performance_history.copy()


# Instance globale pour l'ajustement adaptatif
adaptive_adjuster = AdaptiveRegimeAdjuster()


def get_enhanced_regime_adjustment(
    strategy_name: str,
    market_regime: str,
    signal_side: str,
    market_indicators: dict = None,
) -> float:
    """
    Version améliorée avec détection de rebonds et ajustement adaptatif.

    Args:
        strategy_name: Nom de la stratégie
        market_regime: Régime de marché actuel
        signal_side: Direction du signal (BUY/SELL)
        market_indicators: Indicateurs de marché pour détecter les rebonds

    Returns:
        Multiplicateur de confidence ajusté
    """
    # Obtenir le multiplicateur de base
    base_multiplier = get_regime_adjustment(strategy_name, market_regime, signal_side)

    # Vérifier si c'est un rebond légitime
    if should_allow_counter_trend_rebound(
        market_regime, signal_side, market_indicators
    ):
        # Réduire la pénalité pour les rebonds légitimes
        if base_multiplier < 0.8:  # Si fortement pénalisé
            base_multiplier = max(
                base_multiplier * 1.4, 0.7
            )  # Boost significatif mais limité

    # Appliquer l'ajustement adaptatif
    adaptive_multiplier = adaptive_adjuster.get_adaptive_multiplier(
        strategy_name, market_regime, signal_side, base_multiplier
    )

    return adaptive_multiplier
