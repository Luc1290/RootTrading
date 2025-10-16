"""
Tests pour valider le format des données market_data en DB.
"""

import os
import sys
from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from shared.src.schemas import MarketData

# Ajouter le chemin racine pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))


class TestMarketDataFormat:
    """Tests pour valider le format MarketData."""

    def test_market_data_valid_format(self, market_data_db_format):
        """Test validation d'un format MarketData valide."""
        # Créer un objet MarketData à partir du format DB
        market_data = MarketData(
            symbol=market_data_db_format["symbol"],
            start_time=market_data_db_format["start_time"],
            close_time=market_data_db_format["close_time"],
            open=market_data_db_format["open"],
            high=market_data_db_format["high"],
            low=market_data_db_format["low"],
            close=market_data_db_format["close"],
            volume=market_data_db_format["volume"],
        )

        assert market_data.symbol == "BTCUSDC"
        assert market_data.open == 50000.0
        assert market_data.high == 50100.0
        assert market_data.low == 49900.0
        assert market_data.close == 50050.0
        assert market_data.volume == 1500.0
        assert isinstance(market_data.timestamp, datetime)

    def test_market_data_missing_required_fields(self):
        """Test validation avec champs requis manquants."""
        with pytest.raises(ValidationError) as exc_info:
            MarketData(
                symbol="BTCUSDC",
                # start_time manquant
                close_time=int(
                    (datetime.now() + timedelta(minutes=1)).timestamp() * 1000
                ),
                open=50000.0,
                high=50100.0,
                low=49900.0,
                close=50050.0,
                volume=1500.0,
            )

        assert "field required" in str(
            exc_info.value
        ).lower() or "Field required" in str(exc_info.value)

    def test_market_data_invalid_price_types(self):
        """Test validation avec types de prix invalides."""
        with pytest.raises(ValidationError):
            MarketData(
                symbol="BTCUSDC",
                start_time=int(datetime.now(timezone.utc).timestamp() * 1000),
                close_time=int(
                    (datetime.now() + timedelta(minutes=1)).timestamp() * 1000
                ),
                open="invalid",  # Type string au lieu de float
                high=50100.0,
                low=49900.0,
                close=50050.0,
                volume=1500.0,
            )

    def test_market_data_negative_values(self):
        """Test validation avec valeurs négatives."""
        # Pydantic par défaut ne valide pas les valeurs négatives
        # Créer l'objet avec volume négatif (peut être accepté sans validators
        # custom)
        market_data = MarketData(
            symbol="BTCUSDC",
            start_time=int(datetime.now(timezone.utc).timestamp() * 1000),
            close_time=int((datetime.now() + timedelta(minutes=1)).timestamp() * 1000),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=-1500.0,  # Volume négatif accepté sans validator custom
        )
        # Vérifier que l'objet est créé mais avec volume négatif
        assert market_data.volume == -1500.0

    def test_market_data_ohlc_logic_validation(self):
        """Test que la logique OHLC est cohérente."""
        # Ce test vérifie la cohérence des données mais Pydantic ne fait pas
        # cette validation par défaut
        market_data = MarketData(
            symbol="BTCUSDC",
            start_time=int(datetime.now(timezone.utc).timestamp() * 1000),
            close_time=int((datetime.now() + timedelta(minutes=1)).timestamp() * 1000),
            open=50000.0,
            high=49000.0,  # High < Open (techniquement invalide)
            low=49900.0,
            close=50050.0,
            volume=1500.0,
        )

        # Les données sont créées même si logiquement incohérentes
        # Dans un système réel, on ajouterait des validators personnalisés
        assert market_data.high == 49000.0

    def test_market_data_timestamp_auto_generation(self):
        """Test génération automatique du timestamp."""
        start_time = int(datetime.now(timezone.utc).timestamp() * 1000)

        market_data = MarketData(
            symbol="BTCUSDC",
            start_time=start_time,
            close_time=start_time + 60000,  # +1 minute
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=1500.0,
            # timestamp pas fourni
        )

        # Timestamp doit être généré automatiquement
        assert market_data.timestamp is not None
        expected_timestamp = datetime.fromtimestamp(start_time / 1000)
        assert market_data.timestamp == expected_timestamp

    def test_market_data_timestamp_provided(self):
        """Test avec timestamp fourni explicitement."""
        explicit_timestamp = datetime.now()

        market_data = MarketData(
            symbol="BTCUSDC",
            start_time=int(datetime.now(timezone.utc).timestamp() * 1000),
            close_time=int((datetime.now() + timedelta(minutes=1)).timestamp() * 1000),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=1500.0,
            timestamp=explicit_timestamp,
        )

        assert market_data.timestamp == explicit_timestamp

    def test_market_data_symbol_validation(self):
        """Test validation des symboles."""
        # Symbole vide - Pydantic accepte les strings vides par défaut
        market_data = MarketData(
            symbol="",  # Accepté sans validator custom
            start_time=int(datetime.now(timezone.utc).timestamp() * 1000),
            close_time=int((datetime.now() + timedelta(minutes=1)).timestamp() * 1000),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=1500.0,
        )
        assert market_data.symbol == ""

    def test_market_data_from_db_row_simulation(self):
        """Test simulation d'une ligne de DB."""
        # Simuler une ligne de DB comme un dict
        db_row = {
            "symbol": "ETHUSDC",
            "start_time": 1640995200000,  # timestamp spécifique
            "close_time": 1640995260000,
            "open": 3800.50,
            "high": 3820.75,
            "low": 3795.25,
            "close": 3810.00,
            "volume": 2500.75,
        }

        market_data = MarketData(**db_row)

        assert market_data.symbol == "ETHUSDC"
        assert market_data.open == 3800.50
        assert market_data.volume == 2500.75
        assert market_data.timestamp is not None

    def test_market_data_json_serialization(self, market_data_db_format):
        """Test sérialisation JSON des données market."""
        market_data = MarketData(
            **{
                k: v
                for k, v in market_data_db_format.items()
                if k
                in [
                    "symbol",
                    "start_time",
                    "close_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            }
        )

        # Test sérialisation
        json_data = market_data.dict()
        assert isinstance(json_data, dict)
        assert json_data["symbol"] == "BTCUSDC"
        assert "timestamp" in json_data

        # Test désérialisation
        market_data_2 = MarketData(**json_data)
        assert market_data_2.symbol == market_data.symbol
        assert market_data_2.close == market_data.close
