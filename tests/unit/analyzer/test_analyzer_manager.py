import pytest
from unittest.mock import MagicMock, patch
from analyzer.src.multiproc_manager import AnalyzerManager

def test_manager_initialization():
    manager = AnalyzerManager(symbols=["BTCUSDC", "ETHUSDC"], max_workers=2, use_threads=True)
    assert len(manager.symbol_groups) > 0
    assert manager.data_queue is not None
    assert manager.signal_queue is not None

def test_handle_market_data_puts_in_queue():
    manager = AnalyzerManager(symbols=["BTCUSDC"], max_workers=1, use_threads=True)
    mock_data = {"symbol": "BTCUSDC", "is_closed": True}
    manager._handle_market_data(mock_data)
    assert not manager.data_queue.empty()
