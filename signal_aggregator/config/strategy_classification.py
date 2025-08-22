"""
Classification des stratégies par type pour adaptation au régime de marché.

Ce module définit les familles de stratégies et leurs caractéristiques
pour permettre une validation adaptative selon le régime de marché.
"""

# Classification des 28 stratégies par famille principale
STRATEGY_FAMILIES = {
    'trend_following': {
        'strategies': [
            'MACD_Crossover_Strategy',
            'EMA_Cross_Strategy', 
            'ADX_Direction_Strategy',
            'Supertrend_Reversal_Strategy',
            'HullMA_Slope_Strategy',
            'TEMA_Slope_Strategy',
            'TRIX_Crossover_Strategy',
            'PPO_Crossover_Strategy'
        ],
        'best_regimes': ['TRENDING_BULL', 'TRENDING_BEAR'],
        'acceptable_regimes': ['BREAKOUT_BULL', 'BREAKOUT_BEAR', 'TRANSITION'],
        'poor_regimes': ['RANGING', 'VOLATILE'],
        'characteristics': 'Suit les tendances établies, performances optimales en marché directionnel'
    },
    
    'mean_reversion': {
        'strategies': [
            'RSI_Cross_Strategy',
            'Stochastic_Oversold_Buy_Strategy',
            'StochRSI_Rebound_Strategy',
            'WilliamsR_Rebound_Strategy',
            'CCI_Reversal_Strategy',
            'Bollinger_Touch_Strategy',
            'ZScore_Extreme_Reversal_Strategy',
            'ROC_Threshold_Strategy'
        ],
        'best_regimes': ['RANGING'],
        'acceptable_regimes': ['VOLATILE', 'TRANSITION'],
        'poor_regimes': ['TRENDING_BULL', 'TRENDING_BEAR'],
        'characteristics': 'Exploite les retours à la moyenne, optimal en marché latéral'
    },
    
    'breakout': {
        'strategies': [
            'ATR_Breakout_Strategy',
            'Donchian_Breakout_Strategy',
            'Range_Breakout_Confirmation_Strategy',
            'Support_Breakout_Strategy',
            'Resistance_Rejection_Strategy'
        ],
        'best_regimes': ['BREAKOUT_BULL', 'BREAKOUT_BEAR', 'VOLATILE'],
        'acceptable_regimes': ['TRANSITION', 'TRENDING_BULL', 'TRENDING_BEAR'],
        'poor_regimes': ['RANGING'],
        'characteristics': 'Détecte les cassures de niveaux, optimal en volatilité'
    },
    
    'volume_based': {
        'strategies': [
            'OBV_Crossover_Strategy',
            'Volume_Spike_Strategy',  # Si elle existe
            'Liquidity_Sweep_Buy_Strategy',
            'Pump_Dump_Pattern_Strategy'
        ],
        'best_regimes': ['BREAKOUT_BULL', 'BREAKOUT_BEAR', 'VOLATILE'],
        'acceptable_regimes': ['TRENDING_BULL', 'TRENDING_BEAR', 'TRANSITION'],
        'poor_regimes': ['RANGING'],
        'characteristics': 'Analyse les mouvements de volume, détecte accumulation/distribution'
    },
    
    'structure_based': {
        'strategies': [
            'VWAP_Support_Resistance_Strategy',
            'ParabolicSAR_Bounce_Strategy',
            'Spike_Reaction_Buy_Strategy',
            'MultiTF_ConfluentEntry_Strategy'
        ],
        'best_regimes': ['TRENDING_BULL', 'TRENDING_BEAR', 'RANGING'],
        'acceptable_regimes': ['VOLATILE', 'TRANSITION', 'BREAKOUT_BULL', 'BREAKOUT_BEAR'],
        'poor_regimes': [],  # Adaptable à tous les régimes
        'characteristics': 'Analyse la structure de marché, très adaptable'
    }
}

# Mapping inverse : stratégie -> famille
STRATEGY_TO_FAMILY = {}
for family, config in STRATEGY_FAMILIES.items():
    for strategy in config['strategies']:
        STRATEGY_TO_FAMILY[strategy] = family

# Configuration des ajustements de confidence selon le régime
REGIME_CONFIDENCE_ADJUSTMENTS = {
    'TRENDING_BULL': {
        'trend_following': {'BUY': 1.2, 'SELL': 0.7},      # Boost BUY, pénalise SELL
        'mean_reversion': {'BUY': 0.6, 'SELL': 0.4},       # Très pénalisé en tendance
        'breakout': {'BUY': 1.1, 'SELL': 0.8},             # Léger boost BUY
        'volume_based': {'BUY': 1.1, 'SELL': 0.9},         # Neutre-positif
        'structure_based': {'BUY': 1.0, 'SELL': 0.9}       # Quasi-neutre
    },
    
    'TRENDING_BEAR': {
        'trend_following': {'BUY': 0.5, 'SELL': 1.3},      # DURCI: Très pénalisé BUY (était 0.7), boost SELL (était 1.2)
        'mean_reversion': {'BUY': 0.3, 'SELL': 0.8},       # DURCI: Très pénalisé BUY (était 0.4), amélioré SELL
        'breakout': {'BUY': 0.6, 'SELL': 1.2},             # DURCI: Très pénalisé BUY (était 0.8), boost SELL
        'volume_based': {'BUY': 0.7, 'SELL': 1.2},         # DURCI: Pénalisé BUY (était 0.9), boost SELL
        'structure_based': {'BUY': 0.7, 'SELL': 1.1}       # DURCI: Pénalisé BUY (était 0.9)
    },
    
    'RANGING': {
        'trend_following': {'BUY': 0.7, 'SELL': 0.7},      # Pénalisé en ranging
        'mean_reversion': {'BUY': 1.3, 'SELL': 1.3},       # Fortement boosté
        'breakout': {'BUY': 0.8, 'SELL': 0.8},             # Légèrement pénalisé
        'volume_based': {'BUY': 0.9, 'SELL': 0.9},         # Légèrement pénalisé
        'structure_based': {'BUY': 1.1, 'SELL': 1.1}       # Légèrement boosté
    },
    
    'VOLATILE': {
        'trend_following': {'BUY': 0.8, 'SELL': 0.8},      # Pénalisé en volatilité
        'mean_reversion': {'BUY': 1.1, 'SELL': 1.1},       # Boosté en volatilité
        'breakout': {'BUY': 1.2, 'SELL': 1.2},             # Fortement boosté
        'volume_based': {'BUY': 1.2, 'SELL': 1.2},         # Fortement boosté
        'structure_based': {'BUY': 1.0, 'SELL': 1.0}       # Neutre
    },
    
    'BREAKOUT_BULL': {
        'trend_following': {'BUY': 1.1, 'SELL': 0.6},      # Favorise BUY
        'mean_reversion': {'BUY': 0.5, 'SELL': 0.3},       # Très pénalisé
        'breakout': {'BUY': 1.4, 'SELL': 0.7},             # Très fort boost BUY
        'volume_based': {'BUY': 1.3, 'SELL': 0.8},         # Fort boost BUY
        'structure_based': {'BUY': 1.1, 'SELL': 0.8}       # Léger boost BUY
    },
    
    'BREAKOUT_BEAR': {
        'trend_following': {'BUY': 0.4, 'SELL': 1.2},      # DURCI: Très pénalisé BUY (était 0.6)
        'mean_reversion': {'BUY': 0.2, 'SELL': 0.7},       # DURCI: Extrêmement pénalisé BUY (était 0.3)
        'breakout': {'BUY': 0.5, 'SELL': 1.5},             # DURCI: Très pénalisé BUY (était 0.7), boost SELL
        'volume_based': {'BUY': 0.6, 'SELL': 1.4},         # DURCI: Très pénalisé BUY (était 0.8), boost SELL
        'structure_based': {'BUY': 0.6, 'SELL': 1.2}       # DURCI: Très pénalisé BUY (était 0.8)
    },
    
    'TRANSITION': {
        'trend_following': {'BUY': 0.9, 'SELL': 0.9},      # Légèrement pénalisé
        'mean_reversion': {'BUY': 1.0, 'SELL': 1.0},       # Neutre
        'breakout': {'BUY': 1.0, 'SELL': 1.0},             # Neutre
        'volume_based': {'BUY': 1.0, 'SELL': 1.0},         # Neutre
        'structure_based': {'BUY': 1.0, 'SELL': 1.0}       # Neutre
    },
    
    'UNKNOWN': {
        # Régime inconnu : on reste conservateur
        'trend_following': {'BUY': 0.9, 'SELL': 0.9},
        'mean_reversion': {'BUY': 0.9, 'SELL': 0.9},
        'breakout': {'BUY': 0.9, 'SELL': 0.9},
        'volume_based': {'BUY': 0.9, 'SELL': 0.9},
        'structure_based': {'BUY': 0.95, 'SELL': 0.95}     # Moins pénalisé car adaptable
    }
}

# Seuils minimums de confidence requis selon le régime et la direction
# DURCISSEMENT MAJEUR pour éviter les entrées prématurées en marché baissier
REGIME_MIN_CONFIDENCE = {
    'TRENDING_BULL': {
        'BUY': 0.60,   # Légèrement durci (était 0.55)
        'SELL': 0.85   # Très strict pour SELL contre-tendance (était 0.80)
    },
    'TRENDING_BEAR': {
        'BUY': 0.90,   # TRÈS STRICT pour BUY contre-tendance (était 0.80)
        'SELL': 0.60   # Légèrement durci (était 0.55)
    },
    'RANGING': {
        'BUY': 0.70,   # Plus strict (était 0.60)
        'SELL': 0.70   # Plus strict (était 0.60)
    },
    'VOLATILE': {
        'BUY': 0.75,   # Beaucoup plus strict (était 0.65)
        'SELL': 0.70   # Plus strict (était 0.65)
    },
    'BREAKOUT_BULL': {
        'BUY': 0.55,   # Légèrement durci (était 0.50)
        'SELL': 0.90   # Très strict pour SELL (était 0.85)
    },
    'BREAKOUT_BEAR': {
        'BUY': 0.95,   # EXTRÊMEMENT STRICT pour BUY (était 0.85)
        'SELL': 0.55   # Légèrement durci (était 0.50)
    },
    'TRANSITION': {
        'BUY': 0.75,   # Plus conservateur (était 0.65)
        'SELL': 0.70   # Plus conservateur (était 0.65)
    },
    'UNKNOWN': {
        'BUY': 0.80,   # Très conservateur (était 0.70)
        'SELL': 0.75   # Conservateur (était 0.70)
    }
}

def get_strategy_family(strategy_name: str) -> str:
    """
    Retourne la famille d'une stratégie.
    
    Args:
        strategy_name: Nom de la stratégie
        
    Returns:
        Famille de la stratégie ou 'unknown'
    """
    return STRATEGY_TO_FAMILY.get(strategy_name, 'unknown')

def get_regime_adjustment(strategy_name: str, market_regime: str, signal_side: str) -> float:
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
    if family == 'unknown':
        return 0.95  # Légère pénalité pour stratégie inconnue
        
    regime = market_regime.upper() if market_regime else 'UNKNOWN'
    if regime not in REGIME_CONFIDENCE_ADJUSTMENTS:
        regime = 'UNKNOWN'
        
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
    regime = market_regime.upper() if market_regime else 'UNKNOWN'
    if regime not in REGIME_MIN_CONFIDENCE:
        regime = 'UNKNOWN'
        
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
    if family == 'unknown':
        return False
        
    family_config = STRATEGY_FAMILIES.get(family, {})
    best_regimes = family_config.get('best_regimes', [])
    
    regime = market_regime.upper() if market_regime else 'UNKNOWN'
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
    if family == 'unknown':
        return True  # On accepte par défaut les stratégies inconnues
        
    family_config = STRATEGY_FAMILIES.get(family, {})
    best_regimes = family_config.get('best_regimes', [])
    acceptable_regimes = family_config.get('acceptable_regimes', [])
    
    regime = market_regime.upper() if market_regime else 'UNKNOWN'
    return regime in best_regimes or regime in acceptable_regimes