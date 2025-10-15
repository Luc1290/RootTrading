"""
Freqtrade Integration Module
Adaptateurs bidirectionnels ROOT ↔ Freqtrade pour backtesting et import de stratégies.
"""

from .root_to_freqtrade import RootToFreqtradeAdapter
from .freqtrade_to_root import FreqtradeToRootAdapter
from .data_converter import DataConverter

__all__ = [
    'RootToFreqtradeAdapter',
    'FreqtradeToRootAdapter',
    'DataConverter'
]
