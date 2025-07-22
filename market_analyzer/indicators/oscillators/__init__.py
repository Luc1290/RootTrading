"""
Oscillators Module

Contains oscillator-based technical indicators:
- Stochastic Oscillator
- Williams %R
"""

from .stochastic import calculate_stochastic
from .williams import calculate_williams_r

__all__ = [
    'calculate_stochastic',
    'calculate_williams_r'
]