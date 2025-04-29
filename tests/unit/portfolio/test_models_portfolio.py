import pytest
from portfolio.src.models import PortfolioModel

def test_portfolio_model_init():
    model = PortfolioModel()
    assert model.db is not None
