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

from .multitimeframe_analyzer import MultiTimeframeAnalyzer, TimeframeSignal
from .range_analyzer import BreakoutAnalysis, RangeAnalyzer, RangeInfo
from .regime_detector import MarketRegime, RegimeDetector
from .spike_detector import SpikeDetector, SpikeEvent
from .support_resistance_detector import PriceLevel, SupportResistanceDetector
from .volume_context_analyzer import VolumeAnalysis, VolumeContext, VolumeContextAnalyzer

__all__ = [
    "BreakoutAnalysis",
    "MarketRegime",
    "MultiTimeframeAnalyzer",
    "PriceLevel",
    "RangeAnalyzer",
    "RangeInfo",
    "RegimeDetector",
    "SpikeDetector",
    "SpikeEvent",
    "SupportResistanceDetector",
    "TimeframeSignal",
    "VolumeAnalysis",
    "VolumeContext",
    "VolumeContextAnalyzer",
]
