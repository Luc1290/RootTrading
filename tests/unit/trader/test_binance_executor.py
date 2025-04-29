import pytest
from trader.src.binance_executor import BinanceExecutor

def test_binance_executor_initialization():
    executor = BinanceExecutor(demo_mode=True)
    assert executor.demo_mode is True
    assert isinstance(executor.min_quantities, dict)

def test_generate_order_id():
    executor = BinanceExecutor(demo_mode=True)
    first_id = executor._generate_order_id()
    second_id = executor._generate_order_id()
    assert int(second_id) > int(first_id)
