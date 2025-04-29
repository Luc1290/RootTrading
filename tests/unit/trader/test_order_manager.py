import pytest
from trader.src.order_manager import OrderManager

def test_order_manager_initialization():
    manager = OrderManager(symbols=["BTCUSDC"])
    assert "BTCUSDC" in manager.symbols
    assert manager.signal_queue is not None

def test_pause_resume_symbol():
    manager = OrderManager(symbols=["BTCUSDC"])
    manager.pause_symbol("BTCUSDC")
    assert manager.is_trading_paused("BTCUSDC", "AnyStrategy") is True
    manager.resume_symbol("BTCUSDC")
    assert manager.is_trading_paused("BTCUSDC", "AnyStrategy") is False
