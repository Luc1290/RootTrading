import pytest
from coordinator.src.pocket_checker import PocketChecker

def test_pocket_checker_init():
    checker = PocketChecker("http://mockportfolio")
    assert checker.portfolio_api_url == "http://mockportfolio"
    assert isinstance(checker.pocket_cache, dict)

def test_get_available_funds_cache(monkeypatch):
    checker = PocketChecker("http://mockportfolio")
    checker.pocket_cache = {"active": {"available_value": 500}}
    checker.last_cache_update = 9999999999  # Force valid cache
    assert checker.get_available_funds("active") == 500
