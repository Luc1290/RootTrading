import pytest
from analyzer.src.strategy_loader import StrategyLoader

def test_strategy_loader_loads_strategies():
    loader = StrategyLoader(symbols=["BTCUSDC"])
    strategies = loader.get_strategy_list()
    assert isinstance(strategies, dict)

def test_process_market_data_no_symbol():
    loader = StrategyLoader(symbols=["BTCUSDC"])
    result = loader.process_market_data({})
    assert result == []
