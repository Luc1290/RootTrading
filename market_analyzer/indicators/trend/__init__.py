"""
Trend Indicators Module

Contains trend-following technical indicators:
- Moving Averages (EMA, SMA, WMA)
- MACD (Moving Average Convergence Divergence)
- ADX (Average Directional Index)
"""

from .moving_averages import calculate_ema, calculate_sma, calculate_ema_incremental
from .macd import calculate_macd, calculate_macd_incremental
from .adx import calculate_adx, calculate_dmi

__all__ = [
    "calculate_ema",
    "calculate_sma",
    "calculate_ema_incremental",
    "calculate_macd",
    "calculate_macd_incremental",
    "calculate_adx",
    "calculate_dmi",
]
