"""
Configuration de la hiérarchie des validators pour le signal aggregator.

Cette hiérarchie définit l'importance de chaque validator et leur pouvoir de décision :
- CRITICAL : Validators avec pouvoir de veto (doivent tous passer)
- IMPORTANT : Validators à forte influence (70% doivent passer)
- STANDARD : Validators normaux (50% doivent passer)
"""

VALIDATOR_HIERARCHY = {
    'critical': {
        'description': 'Validators critiques avec pouvoir de veto - ADAPTATIFS AU RÉGIME',
        'validators': [
            'Market_Structure_Validator',  # RÉACTIVÉ avec logique adaptative au régime
            'Volume_Spike_Validator'       # Confirme la force du mouvement
        ],
        'min_pass_rate': 1.0,     # 100% doivent passer
        'weight_multiplier': 3.0,  # Triple impact sur le score
        'veto_power': True        # Peut rejeter le signal immédiatement
    },
    
    'important': {
        'description': 'Validators importants avec forte influence sur la décision',
        'validators': [
            'Global_Trend_Validator',      # Déplacé de critical - plus permissif sans veto
            'Regime_Strength_Validator',   # Force du régime actuel
            'MultiTF_Consensus_Validator', # Consensus multi-timeframes
            'Volume_Buildup_Validator',    # Accumulation de volume
            'Trend_Alignment_Validator',   # Alignement des tendances
            'VWAP_Context_Validator'       # Position par rapport au VWAP
        ],
        'min_pass_rate': 0.70,     # 70% doivent passer (assoupli)
        'weight_multiplier': 2.0,  # Double impact sur le score
        'veto_power': False
    },
    
    'standard': {
        'description': 'Validators standards pour affiner la décision',
        'validators': [
            'RSI_Regime_Validator',
            'MACD_Regime_Validator',
            'ADX_TrendStrength_Validator',
            'ATR_Volatility_Validator',
            'Volatility_Regime_Validator',
            'Bollinger_Width_Validator',
            'Volume_Quality_Score_Validator',
            'Volume_Ratio_Validator',
            'Trend_Smoothness_Validator',
            'ZScore_Context_Validator',
            'S_R_Level_Proximity_Validator',
            'Psychological_Level_Validator',
            'Pivot_Strength_Validator',
            'Range_Validator',
            'Liquidity_Sweep_Validator',
            'Adaptive_Threshold_Validator'
        ],
        'min_pass_rate': 0.50,     # 50% doivent passer (assoupli)
        'weight_multiplier': 1.0,  # Impact normal sur le score
        'veto_power': False
    }
}

# Configuration des seuils de validation par niveau - ÉQUILIBRÉ POUR QUALITÉ
VALIDATION_THRESHOLDS = {
    'critical': {
        'min_score': 0.55,  # Score minimum équilibré (55%)
        'rejection_message': "Signal rejeté par validator critique : {validator_name} - {reason}"
    },
    'important': {
        'min_score': 0.50,  # Score minimum équilibré (50%)
        'low_score_penalty': 0.20  # Pénalité standard
    },
    'standard': {
        'min_score': 0.45,  # Score minimum équilibré (45%)
        'low_score_penalty': 0.15  # Pénalité standard
    }
}

# Configuration des bonus/malus selon les combinaisons
COMBINATION_RULES = {
    'perfect_critical': {
        'condition': 'all_critical_pass_with_high_score',
        'min_avg_score': 0.8,
        'bonus': 0.15,
        'description': 'Bonus si tous les validators critiques passent avec score élevé'
    },
    'strong_trend_volume': {
        'condition': 'trend_and_volume_aligned',
        'validators': ['Global_Trend_Validator', 'Volume_Spike_Validator'],
        'min_scores': 0.7,
        'bonus': 0.10,
        'description': 'Bonus si tendance et volume sont fortement alignés'
    },
    'multi_tf_consensus': {
        'condition': 'high_mtf_consensus',
        'validator': 'MultiTF_Consensus_Validator',
        'min_score': 0.85,
        'bonus': 0.08,
        'description': 'Bonus pour consensus multi-timeframe très fort'
    }
}

# Fonction helper pour obtenir le niveau d'un validator
def get_validator_level(validator_name: str) -> str:
    """
    Retourne le niveau hiérarchique d'un validator.
    
    Args:
        validator_name: Nom du validator
        
    Returns:
        'critical', 'important', 'standard' ou 'unknown'
    """
    for level, config in VALIDATOR_HIERARCHY.items():
        if validator_name in config['validators']:
            return level
    return 'unknown'

# Fonction helper pour vérifier si un validator a le pouvoir de veto
def has_veto_power(validator_name: str) -> bool:
    """
    Vérifie si un validator a le pouvoir de veto.
    
    Args:
        validator_name: Nom du validator
        
    Returns:
        True si le validator peut rejeter un signal immédiatement
    """
    level = get_validator_level(validator_name)
    if level in VALIDATOR_HIERARCHY:
        return VALIDATOR_HIERARCHY[level].get('veto_power', False)
    return False

# Liste de tous les validators pour validation
ALL_VALIDATORS = []
for level_config in VALIDATOR_HIERARCHY.values():
    ALL_VALIDATORS.extend(level_config['validators'])