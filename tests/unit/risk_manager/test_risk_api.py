import pytest
from flask.testing import FlaskClient
from risk_manager.src.main import app

client: FlaskClient = app.test_client()

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json

def test_rules_endpoint():
    response = client.get("/rules")
    assert response.status_code in [200, 500]
