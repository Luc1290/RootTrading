"""
Configuration pytest et fixtures partagées pour tous les tests.
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any

import pytest
import pandas as pd
import numpy as np

# Ajouter le chemin racine pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def sample_ohlcv_data():
    """Données OHLCV fictives pour tester les stratégies."""
    dates = pd.date_range(start="2024-01-01", periods=1000, freq="1min")
    np.random.seed(42)  # Reproductibilité

    # Génération de prix réalistes avec random walk
    price_base = 50000
    returns = np.random.normal(0, 0.001, len(dates))
    prices = price_base * np.exp(np.cumsum(returns))

    # OHLCV data avec variations réalistes
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        high = close * (1 + abs(np.random.normal(0, 0.005)))
        low = close * (1 - abs(np.random.normal(0, 0.005)))
        open_price = low + (high - low) * np.random.random()
        volume = np.random.uniform(1000, 10000)

        data.append(
            {
                "timestamp": date,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def sample_indicators():
    """Indicateurs techniques fictifs pour tester les stratégies."""
    return {
        "sma_20": [50000 + i * 10 for i in range(100)],
        "sma_50": [49000 + i * 8 for i in range(100)],
        "ema_12": [50500 + i * 12 for i in range(100)],
        "ema_26": [49500 + i * 9 for i in range(100)],
        "rsi": [30 + (i % 70) for i in range(100)],
        "macd": [(i % 20) - 10 for i in range(100)],
        "macd_signal": [(i % 18) - 9 for i in range(100)],
        "macd_histogram": [((i % 20) - 10) - ((i % 18) - 9) for i in range(100)],
        "bb_upper": [51000 + i * 15 for i in range(100)],
        "bb_middle": [50000 + i * 10 for i in range(100)],
        "bb_lower": [49000 + i * 5 for i in range(100)],
        "atr": [500 + (i % 200) for i in range(100)],
        "stoch_k": [20 + (i % 60) for i in range(100)],
        "stoch_d": [25 + (i % 50) for i in range(100)],
        "adx": [20 + (i % 60) for i in range(100)],
        "cci": [-100 + (i % 200) for i in range(100)],
        "williams_r": [-80 + (i % 60) for i in range(100)],
        "obv": [1000000 + i * 1000 for i in range(100)],
        "vwap": [50000 + i * 5 + np.sin(i / 10) * 100 for i in range(100)],
        "mfi": [30 + (i % 40) for i in range(100)],
    }


@pytest.fixture
def sample_db_data():
    """Structure de données comme venant de la DB."""
    return {
        "symbol": "BTCUSDC",
        "timeframe": "1m",
        "data": [
            {
                "timestamp": datetime.now() - timedelta(minutes=i),
                "open": 50000 + i * 10,
                "high": 50100 + i * 10,
                "low": 49900 + i * 10,
                "close": 50050 + i * 10,
                "volume": 1000 + i * 100,
            }
            for i in range(100, 0, -1)
        ],
    }


@pytest.fixture
def valid_signal_format():
    """Format attendu pour un signal de stratégie valide."""
    return {
        "side": None,  # "BUY", "SELL" or None
        "confidence": 0.0,  # float 0-1
        "strength": "weak",  # "weak", "moderate", "strong", "very_strong"
        "reason": "",  # string description
        "metadata": {},  # dict with additional info
    }


@pytest.fixture
def market_data_db_format():
    """Format des données market_data en DB."""
    return {
        "id": 1,
        "symbol": "BTCUSDC",
        "time": datetime.now(),
        "start_time": int(datetime.now().timestamp() * 1000),
        "close_time": int((datetime.now() + timedelta(minutes=1)).timestamp() * 1000),
        "open": 50000.0,
        "high": 50100.0,
        "low": 49900.0,
        "close": 50050.0,
        "volume": 1500.0,
        "quote_volume": 75000000.0,
        "trades": 1234,
        "taker_buy_volume": 800.0,
        "taker_buy_quote_volume": 40000000.0,
    }


@pytest.fixture
def indicators_db_format():
    """Format des indicateurs en DB."""
    return {
        "id": 1,
        "symbol": "BTCUSDC",
        "timestamp": datetime.now(),
        "timeframe": "1m",
        "indicators": {
            "sma_20": 50000.0,
            "sma_50": 49500.0,
            "ema_12": 50200.0,
            "ema_26": 49800.0,
            "rsi": 65.5,
            "macd": 125.5,
            "macd_signal": 120.0,
            "macd_histogram": 5.5,
            "bb_upper": 51000.0,
            "bb_middle": 50000.0,
            "bb_lower": 49000.0,
            "atr": 350.0,
            "stoch_k": 75.5,
            "stoch_d": 70.2,
            "adx": 45.5,
            "cci": 150.2,
            "williams_r": -25.5,
            "obv": 1500000.0,
            "vwap": 50025.5,
            "mfi": 65.8,
        },
    }


@pytest.fixture
def mock_strategy_data():
    """Données complètes pour tester une stratégie."""
    return {
        "symbol": "BTCUSDC",
        "data": {
            "current_price": 50000.0,
            "timestamp": datetime.now(),
            "ohlcv": [
                {
                    "open": 49800,
                    "high": 50200,
                    "low": 49750,
                    "close": 50000,
                    "volume": 1500,
                },
                {
                    "open": 50000,
                    "high": 50300,
                    "low": 49900,
                    "close": 50100,
                    "volume": 1600,
                },
                {
                    "open": 50100,
                    "high": 50400,
                    "low": 50000,
                    "close": 50200,
                    "volume": 1400,
                },
            ],
        },
        "indicators": {
            "sma_20": 50000,
            "sma_50": 49800,
            "ema_12": 50150,
            "ema_26": 49950,
            "rsi": 65.5,
            "macd": 50.0,
            "macd_signal": 45.0,
            "macd_histogram": 5.0,
            "bb_upper": 50500,
            "bb_middle": 50000,
            "bb_lower": 49500,
            "atr": 300,
            "stoch_k": 75,
            "stoch_d": 70,
            "adx": 45,
            "cci": 120,
            "williams_r": -25,
            "obv": 1500000,
            "vwap": 50025,
            "mfi": 65,
        },
    }
