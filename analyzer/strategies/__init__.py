"""
Trading Signals Module

This package provides advanced trading signal detection:
- Cross Signals: EMA crosses, MACD crosses, momentum crosses
- Signal validation and confirmation
- Entry quality assessment
- Risk/reward projections
"""

from .cross_signals_pro import CrossSignalDetector, CrossSignal

__all__ = [
    'CrossSignalDetector',
    'CrossSignal'
]