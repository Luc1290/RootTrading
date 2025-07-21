"""
Signal Aggregator Package
"""

import sys
import os

# Add path to shared modules BEFORE imports
sys.path.append(os.path.dirname(__file__))

from .signal_aggregator import SignalAggregator, EnhancedSignalAggregator
from .enhanced_regime_detector import EnhancedRegimeDetector
from .performance_tracker import PerformanceTracker
from .db_manager import DatabaseManager

__all__ = [
    'SignalAggregator',
    'EnhancedSignalAggregator', 
    'EnhancedRegimeDetector',
    'PerformanceTracker',
    'DatabaseManager'
]