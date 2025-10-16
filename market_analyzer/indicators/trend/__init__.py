"""
Trend Indicators Module

Contains trend-following technical indicators:
- Moving Averages (EMA, SMA, WMA)
- MACD (Moving Average Convergence Divergence)
- ADX (Average Directional Index)
"""

from .adx import calculate_adx, calculate_dmi
from .macd import calculate_macd, calculate_macd_incremental
from .moving_averages import (calculate_ema, calculate_ema_incremental,
                              calculate_sma)

__all__ = [
    "calculate_adx",
    "calculate_dmi",
    "calculate_ema",
    "calculate_ema_incremental",
    "calculate_macd",
    "calculate_macd_incremental",
    "calculate_sma",
]
