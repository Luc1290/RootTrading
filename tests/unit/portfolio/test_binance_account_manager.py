import pytest
from portfolio.src.binance_account_manager import BinanceAccountManager

def test_binance_account_manager_init():
    manager = BinanceAccountManager("demo_key", "demo_secret")
    assert manager.api_key == "demo_key"
    assert manager.api_secret == "demo_secret"
    assert manager.base_url == "https://api.binance.com"
