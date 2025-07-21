# Signal Aggregator Package

import sys
import os

# Add path to shared modules BEFORE imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from .src.signal_aggregator import EnhancedSignalAggregator

__all__ = ['EnhancedSignalAggregator']