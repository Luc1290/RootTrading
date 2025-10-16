"""
Market Analyzer Indicators Package

High-performance technical indicators with automatic Redis caching.

Usage Examples:

# Simple usage (no cache)
from market_analyzer.indicators import calculate_rsi, calculate_ema
rsi = calculate_rsi(closes, 14)
ema = calculate_ema(closes, 26)

# With automatic caching (recommended for production)
rsi = calculate_rsi(closes, 14, symbol="BTCUSDC")  # 10x faster on repeated calls
ema = calculate_ema(closes, 26, symbol="BTCUSDC")  # Incremental updates

# Advanced factory interface
from market_analyzer.indicators import get_cached_indicators
indicators = get_cached_indicators("BTCUSDC")
trend_analysis = indicators.get_trend_analysis(closes)
momentum_analysis = indicators.get_momentum_analysis(closes)
"""

# Import all main indicator functions with caching support
# Import factory for advanced usage
from .cached_indicator_factory import (cached_indicators,
                                       get_cached_indicators, quick_analysis)
from .momentum.cci import calculate_cci, calculate_cci_series
from .momentum.rsi import calculate_rsi, calculate_rsi_series
from .oscillators.stochastic import (calculate_stochastic,
                                     calculate_stochastic_series)
from .oscillators.williams import (calculate_williams_r,
                                   calculate_williams_r_series)
from .trend.adx import calculate_adx, calculate_adx_series
from .trend.macd import calculate_macd, calculate_macd_series
from .trend.moving_averages import (calculate_ema, calculate_ema_series,
                                    calculate_sma, calculate_sma_series)
from .volatility.atr import calculate_atr, calculate_atr_series
from .volatility.bollinger import (calculate_bollinger_bands,
                                   calculate_bollinger_bands_series)
from .volume.obv import calculate_obv, calculate_obv_series
from .volume.vwap import calculate_vwap, calculate_vwap_series

__all__ = [
    "cached_indicators",
    "calculate_adx",
    "calculate_adx_series",
    "calculate_atr",
    "calculate_atr_series",
    "calculate_bollinger_bands",
    "calculate_bollinger_bands_series",
    "calculate_cci",
    "calculate_cci_series",
    "calculate_ema",
    "calculate_ema_series",
    "calculate_macd",
    "calculate_macd_series",
    "calculate_obv",
    "calculate_obv_series",
    # Core indicator functions (with automatic caching when symbol provided)
    "calculate_rsi",
    "calculate_rsi_series",
    "calculate_sma",
    "calculate_sma_series",
    "calculate_stochastic",
    "calculate_stochastic_series",
    "calculate_vwap",
    "calculate_vwap_series",
    "calculate_williams_r",
    "calculate_williams_r_series",
    # Advanced factory interface
    "get_cached_indicators",
    "quick_analysis",
]


# Convenience functions for common operations
def get_trend_indicators(prices, symbol=None):
    """Get all trend indicators for a price series."""
    return {
        "ema7": calculate_ema(prices, 7, symbol),
        "ema26": calculate_ema(prices, 26, symbol),
        "ema99": calculate_ema(prices, 99, symbol),
        "sma20": calculate_sma(prices, 20),
        "sma50": calculate_sma(prices, 50),
        "macd": calculate_macd_series(prices),
    }


def get_momentum_indicators(prices, symbol=None):
    """Get all momentum indicators for a price series."""
    return {
        "rsi14": calculate_rsi(prices, 14, symbol),
        "rsi21": calculate_rsi(prices, 21, symbol),
        "macd": calculate_macd_series(prices),
    }


def get_volatility_indicators(highs, lows, closes, _symbol=None):
    """Get all volatility indicators."""
    return {
        "atr14": calculate_atr(highs, lows, closes, 14),
        "bollinger": calculate_bollinger_bands_series(closes, 20, 2.0),
    }


def get_volume_indicators(highs, lows, closes, volumes):
    """Get all volume indicators."""
    return {
        "obv": calculate_obv_series(closes, volumes),
        "vwap": calculate_vwap_series(highs, lows, closes, volumes),
    }
