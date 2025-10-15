"""
Market Analysis Detectors Package

This package provides advanced market analysis capabilities:
- Regime Detection: Identifies market conditions (trending, ranging, volatile)
- Support/Resistance Detection: Finds key price levels
- Spike Detection: Identifies anomalous price/volume movements
- Range Analysis: Detailed price position and breakout analysis within ranges
- Multi-Timeframe Analysis: Aggregates signals across timeframes
- Volume Context Analysis: Intelligent volume analysis with contextual adaptation
"""

from .regime_detector import RegimeDetector, MarketRegime
from .support_resistance_detector import SupportResistanceDetector, PriceLevel
from .spike_detector import SpikeDetector, SpikeEvent
from .range_analyzer import RangeAnalyzer, RangeInfo, BreakoutAnalysis
from .multitimeframe_analyzer import MultiTimeframeAnalyzer, TimeframeSignal
from .volume_context_analyzer import (
    VolumeContextAnalyzer,
    VolumeContext,
    VolumeAnalysis,
)

__all__ = [
    "RegimeDetector",
    "MarketRegime",
    "SupportResistanceDetector",
    "PriceLevel",
    "SpikeDetector",
    "SpikeEvent",
    "RangeAnalyzer",
    "RangeInfo",
    "BreakoutAnalysis",
    "MultiTimeframeAnalyzer",
    "TimeframeSignal",
    "VolumeContextAnalyzer",
    "VolumeContext",
    "VolumeAnalysis",
]
