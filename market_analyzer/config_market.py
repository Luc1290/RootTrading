"""
Configuration pour Market Analyzer - ROOT Trading System

Ce fichier centralise toutes les configurations des indicateurs techniques
et paramètres d'analyse pour les services market_analyzer, gateway, 
signal_aggregator, etc.
"""

from typing import Dict, Any

# ==================== PÉRIODES EMA ROOT ====================
EMA_PERIODS = {
    "fast": 7,      # EMA rapide - signaux d'entrée
    "medium": 26,   # EMA moyenne - confirmation tendance
    "slow": 99      # EMA lente - tendance long terme
}

# ==================== CONFIGURATION INDICATEURS ====================

# RSI Configuration (optimisé crypto - plus volatil)
RSI_CONFIG = {
    "period": 14,
    "overbought": 75,      # Plus élevé pour crypto
    "oversold": 25,        # Plus bas pour crypto
    "extreme_overbought": 85,
    "extreme_oversold": 15
}

# Stochastic RSI Configuration
STOCH_RSI_CONFIG = {
    "rsi_period": 14,
    "stoch_period": 14,
    "smooth_k": 3,
    "smooth_d": 3,
    "overbought": 80,
    "oversold": 20
}

# MACD Configuration (adapté aux EMA ROOT)
MACD_CONFIG = {
    "fast_period": 7,   # EMA rapide ROOT
    "slow_period": 26,  # EMA moyenne ROOT  
    "signal_period": 9  # Signal classique
}

# Bollinger Bands Configuration (adapté crypto volatilité)
BOLLINGER_CONFIG = {
    "period": 20,
    "std_dev": 2.5,     # Plus large pour crypto volatilité
    "ma_type": "sma",   # SMA classique
    "squeeze_threshold": 15  # Seuil squeeze crypto
}

# ATR Configuration
ATR_CONFIG = {
    "period": 14,
    "stop_loss_multiplier": 2.0,
    "volatility_threshold": {
        "low": 25,      # percentile 25
        "high": 75,     # percentile 75
        "extreme": 90   # percentile 90
    }
}

# ADX Configuration
ADX_CONFIG = {
    "period": 14,
    "trend_strength": {
        "absent": 20,
        "weak": 25,
        "strong": 50,
        "very_strong": 75
    }
}

# Williams %R Configuration
WILLIAMS_R_CONFIG = {
    "period": 14,
    "overbought": -20,
    "oversold": -80
}

# CCI Configuration
CCI_CONFIG = {
    "period": 20,
    "overbought": 100,
    "oversold": -100
}

# Stochastic Oscillator Configuration
STOCHASTIC_CONFIG = {
    "k_period": 14,
    "k_smooth": 1,     # Fast stochastic
    "d_period": 3,
    "overbought": 80,
    "oversold": 20
}

# Volume Indicators Configuration
VOLUME_CONFIG = {
    "vwap": {
        "session_reset": True,  # Reset VWAP each session
        "std_bands": 1.0        # Standard deviation multiplier
    },
    "obv": {
        "ma_period": 10,        # OBV moving average
        "divergence_lookback": 20
    }
}

# Momentum Configuration
MOMENTUM_CONFIG = {
    "period": 10,
    "roc_period": 10,
    "divergence_lookback": 20
}

# ==================== STRATÉGIES MULTI-SIGNAUX ====================

# Configuration stratégie EMA Cross (ROOT)
EMA_CROSS_STRATEGY = {
    "name": "EMA_Cross_ROOT",
    "periods": EMA_PERIODS,
    "signals": {
        "strong_bullish": {
            "condition": "ema_7 > ema_26 > ema_99 AND price > ema_7",
            "weight": 3.0
        },
        "bullish": {
            "condition": "ema_7 > ema_26 AND price > ema_7", 
            "weight": 2.0
        },
        "weak_bullish": {
            "condition": "price > ema_26",
            "weight": 1.0
        }
    },
    "filters": {
        "volume_confirmation": True,
        "atr_volatility": True
    }
}

# Configuration stratégie RSI
RSI_STRATEGY = {
    "name": "RSI_Strategy",
    "config": RSI_CONFIG,
    "signals": {
        "oversold_bounce": {
            "condition": "rsi < 30 AND rsi > previous_rsi",
            "weight": 2.5
        },
        "overbought_rejection": {
            "condition": "rsi > 70 AND rsi < previous_rsi", 
            "weight": -2.5
        }
    }
}

# Configuration stratégie Bollinger Bands
BOLLINGER_STRATEGY = {
    "name": "Bollinger_Strategy", 
    "config": BOLLINGER_CONFIG,
    "signals": {
        "squeeze_breakout": {
            "condition": "bandwidth < 20th_percentile AND price > upper_band",
            "weight": 3.0
        },
        "mean_reversion": {
            "condition": "price < lower_band AND rsi < 30",
            "weight": 2.0
        }
    }
}

# ==================== TIMEFRAMES ====================
TIMEFRAMES = {
    "1m": "1m",
    "5m": "5m", 
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d"
}

# Timeframes principaux pour les calculs
PRIMARY_TIMEFRAMES = ["1m", "5m", "15m"]

# ==================== SEUILS DE SIGNAL ====================
SIGNAL_THRESHOLDS = {
    "strong_buy": 6.0,      # Plus bas pour crypto réactivité
    "buy": 3.5,             # Signal d'achat
    "weak_buy": 1.5,        # Signal d'achat faible
    "neutral": 0.0,         # Neutre
    "weak_sell": -1.5,      # Signal de vente faible
    "sell": -3.5,           # Signal de vente
    "strong_sell": -6.0     # Plus réactif pour crypto
}

# Pondération des indicateurs dans le score final
INDICATOR_WEIGHTS = {
    "ema_cross": 3.0,       # Poids maximum pour EMA cross ROOT
    "rsi": 2.5,             # RSI important
    "macd": 2.5,            # MACD important  
    "bollinger": 2.0,       # Bollinger bands
    "volume": 2.0,          # Confirmation volume
    "atr": 1.5,             # Volatilité
    "adx": 1.5,             # Force de tendance
    "stochastic": 1.0,      # Oscillateurs secondaires
    "williams_r": 1.0,
    "cci": 1.0,
    "momentum": 1.0
}

# ==================== PARAMÈTRES AVANCÉS ====================

# Configuration pour calculs incrémentaux
INCREMENTAL_CONFIG = {
    "buffer_size": 200,     # Taille des buffers pour calculs
    "min_data_points": 100, # Minimum de points pour calculs valides
    "cache_ttl": 300        # TTL cache Redis (secondes)
}

# Configuration divergences
DIVERGENCE_CONFIG = {
    "lookback_period": 20,   # Période de recherche
    "min_peaks": 2,          # Minimum de pics/creux
    "correlation_threshold": 0.7  # Seuil de corrélation
}

# Configuration multi-timeframe
MULTI_TIMEFRAME_CONFIG = {
    "primary": "5m",         # Timeframe principal
    "confirmation": "15m",   # Timeframe de confirmation
    "trend": "1h",          # Timeframe de tendance
    "weight_ratios": {
        "primary": 1.0,
        "confirmation": 0.8,
        "trend": 0.6
    }
}

# ==================== SYMBOLES ET MARCHÉS ====================
SUPPORTED_SYMBOLS = [
    "BTCUSDC", "ETHUSDC", "ADAUSDC", "DOTUSDC", "LINKUSDC",
    "SOLUSDC", "MATICUSDC", "AVAXUSDC", "ATOMUSDC", "NEARUSDC"
]

# Configuration par type d'actif
ASSET_CONFIG = {
    "BTC": {
        "min_volume": 1000000,  # Volume minimum pour signaux valides
        "volatility_factor": 1.0
    },
    "ETH": {
        "min_volume": 500000,
        "volatility_factor": 1.1
    },
    "ALTS": {  # Altcoins
        "min_volume": 100000,
        "volatility_factor": 1.5
    }
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
    Récupère la configuration d'un indicateur spécifique.
    
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
        "atr": ATR_CONFIG,
        "adx": ADX_CONFIG,
        "williams_r": WILLIAMS_R_CONFIG,
        "cci": CCI_CONFIG,
        "stochastic": STOCHASTIC_CONFIG,
        "volume": VOLUME_CONFIG,
        "momentum": MOMENTUM_CONFIG,
        "ema": {"periods": EMA_PERIODS}
    }
    
    return configs.get(indicator_name, {})


def get_strategy_config(strategy_name: str) -> Dict[str, Any]:
    """
    Récupère la configuration d'une stratégie spécifique.
    
    Args:
        strategy_name: Nom de la stratégie
        
    Returns:
        Configuration de la stratégie
    """
    strategies = {
        "ema_cross": EMA_CROSS_STRATEGY,
        "rsi": RSI_STRATEGY,
        "bollinger": BOLLINGER_STRATEGY
    }
    
    return strategies.get(strategy_name, {})


# Configuration complète pour export
MARKET_CONFIG = {
    "ema_periods": EMA_PERIODS,
    "indicators": {
        "rsi": RSI_CONFIG,
        "stoch_rsi": STOCH_RSI_CONFIG,
        "macd": MACD_CONFIG,
        "bollinger": BOLLINGER_CONFIG,
        "atr": ATR_CONFIG,
        "adx": ADX_CONFIG,
        "williams_r": WILLIAMS_R_CONFIG,
        "cci": CCI_CONFIG,
        "stochastic": STOCHASTIC_CONFIG,
        "volume": VOLUME_CONFIG,
        "momentum": MOMENTUM_CONFIG
    },
    "strategies": {
        "ema_cross": EMA_CROSS_STRATEGY,
        "rsi": RSI_STRATEGY, 
        "bollinger": BOLLINGER_STRATEGY
    },
    "signals": {
        "thresholds": SIGNAL_THRESHOLDS,
        "weights": INDICATOR_WEIGHTS
    },
    "timeframes": TIMEFRAMES,
    "symbols": SUPPORTED_SYMBOLS,
    "performance": PERFORMANCE_CONFIG
}