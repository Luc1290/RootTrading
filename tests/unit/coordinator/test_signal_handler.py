import pytest
from unittest.mock import MagicMock
from coordinator.src.signal_handler import SignalHandler

def test_signal_handler_init():
    handler = SignalHandler("http://mocktrader", "http://mockportfolio")
    assert handler.trader_api_url == "http://mocktrader"
    assert handler.portfolio_api_url == "http://mockportfolio"
    assert handler.signal_queue is not None
    assert isinstance(handler.market_filters, dict)
