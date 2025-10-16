"""
Freqtrade Integration Module
Adaptateurs bidirectionnels ROOT ↔ Freqtrade pour backtesting et import de stratégies.
"""

from .data_converter import DataConverter
from .freqtrade_to_root import FreqtradeToRootAdapter
from .root_to_freqtrade import RootToFreqtradeAdapter

__all__ = ["RootToFreqtradeAdapter", "FreqtradeToRootAdapter", "DataConverter"]
