import pytest
from fastapi.testclient import TestClient
from portfolio.src.api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "ok"

def test_get_balances():
    response = client.get("/balances")
    assert response.status_code in [200, 404]  # vide ou non

def test_get_pockets():
    response = client.get("/pockets")
    assert response.status_code in [200, 404]
