"""
Volume Indicators Module

Contains volume-based technical indicators:
- OBV (On-Balance Volume)
- VWAP (Volume Weighted Average Price)
- Volume Profile Analysis
- Advanced Volume Metrics (quote volume, trade size, intensity)
"""

from .advanced_metrics import (analyze_volume_quality,
                               calculate_avg_trade_size,
                               calculate_quote_volume_ratio,
                               calculate_trade_intensity,
                               detect_volume_anomaly)
from .obv import calculate_obv
from .vwap import (calculate_vwap, calculate_vwap_quote,
                   calculate_vwap_quote_series)

__all__ = [
    "analyze_volume_quality",
    "calculate_avg_trade_size",
    "calculate_obv",
    "calculate_quote_volume_ratio",
    "calculate_trade_intensity",
    "calculate_vwap",
    "calculate_vwap_quote",
    "calculate_vwap_quote_series",
    "detect_volume_anomaly",
]
