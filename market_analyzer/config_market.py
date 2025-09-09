"""
Configuration pour Market Analyzer - ROOT Trading System

Ce fichier contient uniquement les paramètres de calcul des indicateurs techniques
pour le service market_analyzer. Les seuils de trading et pondérations sont gérés
par les services analyzer et signal_aggregator.
"""

from typing import Dict, Any

# ==================== CONFIGURATION INDICATEURS TECHNIQUES ====================

# Périodes de calcul des moyennes mobiles
MOVING_AVERAGES_CONFIG = {
    "ema": [7, 12, 26, 50, 99],      # Périodes EMA
    "sma": [20, 50, 100, 200],       # Périodes SMA
    "wma": [20],                      # WMA
    "dema": [12],                     # DEMA
    "tema": [12],                     # TEMA
    "hull": [20],                     # Hull MA
    "kama": [14]                      # KAMA
}

# RSI Configuration (calcul uniquement)
RSI_CONFIG = {
    "periods": [14, 21]               # Périodes de calcul RSI
}

# Stochastic Configuration
STOCHASTIC_CONFIG = {
    "k_period": 14,
    "k_smooth": 1,     # Fast stochastic
    "d_period": 3
}

# Stochastic RSI Configuration
STOCH_RSI_CONFIG = {
    "rsi_period": 14,
    "stoch_period": 14,
    "smooth_k": 3,
    "smooth_d": 3
}

# MACD Configuration
MACD_CONFIG = {
    "fast_period": 12,
    "slow_period": 26,
    "signal_period": 9
}

# Bollinger Bands Configuration
BOLLINGER_CONFIG = {
    "period": 20,
    "std_dev": 2.0,
    "ma_type": "sma"
}

# Keltner Channels Configuration
KELTNER_CONFIG = {
    "period": 20,
    "multiplier": 2.0,
    "atr_period": 14
}

# ATR Configuration
ATR_CONFIG = {
    "period": 14
}

# ADX Configuration
ADX_CONFIG = {
    "period": 14
}

# Williams %R Configuration
WILLIAMS_R_CONFIG = {
    "period": 14
}

# CCI Configuration
CCI_CONFIG = {
    "period": 20
}

# Volume Indicators Configuration
VOLUME_CONFIG = {
    "vwap": {
        "periods": [10]                  # Périodes VWAP
    },
    "obv": {
        "ma_period": 10                  # OBV moving average
    },
    "volume_ma": {
        "period": 20                     # Volume moving average
    }
}

# Momentum Configuration
MOMENTUM_CONFIG = {
    "period": 10,
    "roc_periods": [10, 20]             # Rate of Change périodes
}

# ==================== TIMEFRAMES ====================
TIMEFRAMES = ["1m", "3m", "5m", "15m", "1h", "1d"]

# ==================== PARAMÈTRES TECHNIQUES ====================

# Configuration pour calculs incrémentaux
INCREMENTAL_CONFIG = {
    "buffer_size": 200,     # Taille des buffers pour calculs
    "min_data_points": 100, # Minimum de points pour calculs valides
    "cache_ttl": 300        # TTL cache Redis (secondes)
}

# Configuration divergences (pour les detectors)
DIVERGENCE_CONFIG = {
    "lookback_period": 20,   # Période de recherche
    "min_peaks": 2,          # Minimum de pics/creux
    "correlation_threshold": 0.7  # Seuil de corrélation
}

# ==================== OPTIMISATIONS PERFORMANCE ====================
PERFORMANCE_CONFIG = {
    "use_talib": True,           # Utiliser TA-Lib si disponible
    "parallel_calculation": True, # Calculs parallèles
    "max_workers": 4,            # Nombre de workers
    "batch_size": 100,           # Taille des batches
    "memory_limit_mb": 500       # Limite mémoire
}

# ==================== EXPORT CONFIGURATION ====================
def get_indicator_config(indicator_name: str) -> Dict[str, Any]:
    """
    Récupère la configuration d'un indicateur spécifique pour le calcul.
    
    Args:
        indicator_name: Nom de l'indicateur
        
    Returns:
        Configuration de l'indicateur
    """
    configs = {
        "rsi": RSI_CONFIG,
        "stoch_rsi": STOCH_RSI_CONFIG,
        "macd": MACD_CONFIG,
        "bollinger": BOLLINGER_CONFIG,
        "keltner": KELTNER_CONFIG,
        "atr": ATR_CONFIG,
        "adx": ADX_CONFIG,
        "williams_r": WILLIAMS_R_CONFIG,
        "cci": CCI_CONFIG,
        "stochastic": STOCHASTIC_CONFIG,
        "volume": VOLUME_CONFIG,
        "momentum": MOMENTUM_CONFIG,
        "moving_averages": MOVING_AVERAGES_CONFIG
    }
    
    result = configs.get(indicator_name, {})
    return result if isinstance(result, dict) else {}


# Configuration complète pour export
MARKET_CONFIG = {
    "indicators": {
        "moving_averages": MOVING_AVERAGES_CONFIG,
        "rsi": RSI_CONFIG,
        "stoch_rsi": STOCH_RSI_CONFIG,
        "macd": MACD_CONFIG,
        "bollinger": BOLLINGER_CONFIG,
        "keltner": KELTNER_CONFIG,
        "atr": ATR_CONFIG,
        "adx": ADX_CONFIG,
        "williams_r": WILLIAMS_R_CONFIG,
        "cci": CCI_CONFIG,
        "stochastic": STOCHASTIC_CONFIG,
        "volume": VOLUME_CONFIG,
        "momentum": MOMENTUM_CONFIG
    },
    "timeframes": TIMEFRAMES,
    "technical": {
        "incremental": INCREMENTAL_CONFIG,
        "divergence": DIVERGENCE_CONFIG
    },
    "performance": PERFORMANCE_CONFIG
}