import pytest
from unittest.mock import MagicMock, patch
from analyzer.src.main import AnalyzerService
from flask.testing import FlaskClient

@pytest.fixture
def analyzer_service():
    with patch("analyzer.src.main.AnalyzerManager") as MockManager:
        service = AnalyzerService(symbols=["BTCUSDC"], use_threads=True, max_workers=2, port=5050)
        service.manager = MockManager()
        service.running = True
        yield service

def test_health_check_running(analyzer_service: AnalyzerService):
    client: FlaskClient = analyzer_service.app.test_client()
    response = client.get("/health")
    data = response.get_json()
    assert response.status_code == 200
    assert data["status"] == "healthy"
    assert data["symbols"] == ["BTCUSDC"]

def test_diagnostic_no_manager():
    service = AnalyzerService()
    client: FlaskClient = service.app.test_client()
    response = client.get("/diagnostic")
    data = response.get_json()
    assert response.status_code == 503
    assert "error" in data

def test_list_strategies_success(analyzer_service):
    mock_loader = MagicMock()
    mock_loader.get_strategy_list.return_value = {"BTCUSDC": ["EMA_Cross_Strategy"]}
    mock_loader.get_strategy_count.return_value = 1
    analyzer_service.manager.strategy_loader = mock_loader

    client = analyzer_service.app.test_client()
    response = client.get("/strategies")
    data = response.get_json()
    assert response.status_code == 200
    assert "strategies" in data
    assert data["total_count"] == 1

def test_start_calls_manager(monkeypatch):
    from analyzer.src.main import AnalyzerManager
    mock_manager = MagicMock()
    monkeypatch.setattr("analyzer.src.main.AnalyzerManager", lambda **kwargs: mock_manager)

    service = AnalyzerService()
    service.start()

    mock_manager.start.assert_called_once()
    assert service.running is True

def test_stop_cleans_up(analyzer_service):
    analyzer_service.stop()
    assert analyzer_service.running is False
    analyzer_service.manager.stop.assert_called_once()
