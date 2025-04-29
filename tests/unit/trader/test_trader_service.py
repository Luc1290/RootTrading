import pytest
from unittest.mock import MagicMock
from trader.src.main import TraderService

@pytest.fixture
def trader_service():
    service = TraderService(symbols=["BTCUSDC"])
    service.order_manager = MagicMock()
    service.running = True
    return service

def test_health_check(trader_service):
    client = trader_service.app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "healthy"

def test_get_orders(trader_service):
    trader_service.order_manager.get_active_orders.return_value = [{"id": "test"}]
    client = trader_service.app.test_client()
    response = client.get("/orders")
    assert response.status_code == 200
    assert isinstance(response.json, list)
