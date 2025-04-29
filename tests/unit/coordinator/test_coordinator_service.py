import pytest
from unittest.mock import MagicMock, patch
from coordinator.src.main import CoordinatorService

@pytest.fixture
def coordinator():
    service = CoordinatorService(trader_api_url="http://mocktrader", portfolio_api_url="http://mockportfolio", port=5053)
    service.running = True
    service.signal_handler = MagicMock()
    service.pocket_checker = MagicMock()
    return service

def test_health_check(coordinator):
    client = coordinator.app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "healthy"

def test_status_endpoint(coordinator):
    client = coordinator.app.test_client()
    response = client.get("/status")
    assert response.status_code == 200
    assert "running" in response.json

def test_force_reallocation_success(coordinator):
    coordinator.pocket_checker.reallocate_funds.return_value = True
    client = coordinator.app.test_client()
    response = client.post("/force-reallocation")
    assert response.status_code == 200
    assert response.json["status"] == "success"
