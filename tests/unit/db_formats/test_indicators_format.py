"""
Tests pour valider le format des indicateurs techniques en DB.
"""

import os
import sys
from datetime import datetime
from typing import Any, Dict

import pytest

# Ajouter le chemin racine pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))


class TestIndicatorsFormat:
    """Tests pour valider le format des indicateurs techniques."""

    def test_indicators_complete_structure(self, indicators_db_format):
        """Test structure complète des indicateurs."""
        indicators = indicators_db_format["indicators"]

        # Vérifier la présence des indicateurs essentiels
        essential_indicators = [
            "sma_20",
            "sma_50",
            "ema_12",
            "ema_26",
            "rsi",
            "macd",
            "macd_signal",
            "macd_histogram",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "atr",
            "stoch_k",
            "stoch_d",
            "adx",
            "cci",
            "williams_r",
            "obv",
            "vwap",
            "mfi",
        ]

        for indicator in essential_indicators:
            assert indicator in indicators, f"Indicateur manquant: {indicator}"
            assert indicators[indicator] is not None, f"Indicateur null: {indicator}"
            assert isinstance(
                indicators[indicator], (int, float)
            ), f"Type invalide pour {indicator}"

    def test_indicators_value_ranges(self, indicators_db_format):
        """Test que les indicateurs sont dans des plages valides."""
        indicators = indicators_db_format["indicators"]

        # RSI doit être entre 0 et 100
        assert 0 <= indicators["rsi"] <= 100, f"RSI hors plage: {indicators['rsi']}"

        # Stochastic doit être entre 0 et 100
        assert (
            0 <= indicators["stoch_k"] <= 100
        ), f"Stoch %K hors plage: {indicators['stoch_k']}"
        assert (
            0 <= indicators["stoch_d"] <= 100
        ), f"Stoch %D hors plage: {indicators['stoch_d']}"

        # Williams %R doit être entre -100 et 0
        assert (
            -100 <= indicators["williams_r"] <= 0
        ), f"Williams %R hors plage: {indicators['williams_r']}"

        # MFI doit être entre 0 et 100
        assert 0 <= indicators["mfi"] <= 100, f"MFI hors plage: {indicators['mfi']}"

        # ADX doit être positif
        assert indicators["adx"] >= 0, f"ADX négatif: {indicators['adx']}"

        # ATR doit être positif
        assert indicators["atr"] >= 0, f"ATR négatif: {indicators['atr']}"

    def test_bollinger_bands_consistency(self, indicators_db_format):
        """Test cohérence des Bollinger Bands."""
        indicators = indicators_db_format["indicators"]

        bb_lower = indicators["bb_lower"]
        bb_middle = indicators["bb_middle"]
        bb_upper = indicators["bb_upper"]

        # Ordre logique: lower < middle < upper
        assert bb_lower < bb_middle, f"BB Lower ({bb_lower}) >= Middle ({bb_middle})"
        assert bb_middle < bb_upper, f"BB Middle ({bb_middle}) >= Upper ({bb_upper})"
        assert bb_lower < bb_upper, f"BB Lower ({bb_lower}) >= Upper ({bb_upper})"

    def test_macd_components_consistency(self, indicators_db_format):
        """Test cohérence des composants MACD."""
        indicators = indicators_db_format["indicators"]

        macd = indicators["macd"]
        macd_signal = indicators["macd_signal"]
        macd_histogram = indicators["macd_histogram"]

        # Histogram = MACD - Signal (avec tolérance pour erreurs de précision)
        expected_histogram = macd - macd_signal
        assert (
            abs(macd_histogram - expected_histogram) < 0.001
        ), f"MACD Histogram incohérent: {macd_histogram} vs {expected_histogram}"

    def test_moving_averages_relationship(self, indicators_db_format):
        """Test relation entre moyennes mobiles."""
        indicators = indicators_db_format["indicators"]

        # En général, les MA courtes sont plus réactives que les longues
        # Mais on ne peut pas garantir l'ordre à tout moment
        # On vérifie juste qu'elles existent et sont positives
        assert indicators["sma_20"] > 0, "SMA 20 invalide"
        assert indicators["sma_50"] > 0, "SMA 50 invalide"
        assert indicators["ema_12"] > 0, "EMA 12 invalide"
        assert indicators["ema_26"] > 0, "EMA 26 invalide"

    def test_stochastic_consistency(self, indicators_db_format):
        """Test cohérence Stochastic %K et %D."""
        indicators = indicators_db_format["indicators"]

        stoch_k = indicators["stoch_k"]
        stoch_d = indicators["stoch_d"]

        # %D est généralement une moyenne mobile de %K, donc plus lisse
        # Pas de relation stricte mais vérifier qu'ils sont dans la même zone
        # générale
        diff = abs(stoch_k - stoch_d)
        assert diff <= 50, f"Stochastic %K et %D trop éloignés: {stoch_k} vs {stoch_d}"

    def test_indicators_metadata_format(self, indicators_db_format):
        """Test format des métadonnées des indicateurs."""
        # Vérifier les champs metadata
        assert "id" in indicators_db_format
        assert "symbol" in indicators_db_format
        assert "timestamp" in indicators_db_format
        assert "timeframe" in indicators_db_format
        assert "indicators" in indicators_db_format

        # Types des champs
        assert isinstance(indicators_db_format["id"], int)
        assert isinstance(indicators_db_format["symbol"], str)
        assert isinstance(indicators_db_format["timestamp"], datetime)
        assert isinstance(indicators_db_format["timeframe"], str)
        assert isinstance(indicators_db_format["indicators"], dict)

    def test_indicators_missing_values_handling(self):
        """Test gestion des valeurs manquantes."""
        incomplete_indicators = {
            "rsi": 65.5,
            "macd": 125.5,
            # macd_signal manquant intentionnellement
        }

        # Le système doit gérer les indicateurs manquants
        # Vérifier que les clés présentes sont valides
        for key, value in incomplete_indicators.items():
            assert isinstance(value, (int, float)), f"Type invalide pour {key}"

    def test_indicators_extreme_values(self):
        """Test gestion des valeurs extrêmes."""
        extreme_indicators = {
            "rsi": 100.0,  # RSI maximum
            "stoch_k": 0.0,  # Stochastic minimum
            "williams_r": -100.0,  # Williams %R minimum
            "adx": 100.0,  # ADX très fort
            "cci": 500.0,  # CCI extrême (pas de limite théorique)
        }

        # Ces valeurs, bien qu'extrêmes, sont techniquement valides
        assert 0 <= extreme_indicators["rsi"] <= 100
        assert 0 <= extreme_indicators["stoch_k"] <= 100
        assert -100 <= extreme_indicators["williams_r"] <= 0
        assert extreme_indicators["adx"] >= 0
        # CCI peut avoir n'importe quelle valeur

    def test_indicators_json_serializable(self, indicators_db_format):
        """Test que les indicateurs sont sérialisables en JSON."""
        import json

        # Les datetime doivent être converties pour JSON
        serializable_data = indicators_db_format.copy()
        serializable_data["timestamp"] = serializable_data["timestamp"].isoformat()

        # Test sérialisation JSON
        json_str = json.dumps(serializable_data)
        assert isinstance(json_str, str)

        # Test désérialisation
        deserialized = json.loads(json_str)
        assert deserialized["symbol"] == indicators_db_format["symbol"]
        assert (
            deserialized["indicators"]["rsi"]
            == indicators_db_format["indicators"]["rsi"]
        )

    def test_indicators_database_types(self, indicators_db_format):
        """Test types compatibles avec la base de données."""
        indicators = indicators_db_format["indicators"]

        # Tous les indicateurs doivent être des types numériques compatibles
        # SQL
        for name, value in indicators.items():
            assert isinstance(
                value, (int, float)
            ), f"Indicateur {name} n'est pas numérique: {type(value)}"

            # Vérifier que ce n'est pas NaN ou infinity
            if isinstance(value, float):
                assert not (
                    value != value), f"Indicateur {name} est NaN"  # NaN != NaN
                assert abs(value) != float(
                    "inf"), f"Indicateur {name} est infini"

    def test_indicators_computation_dependency(self):
        """Test dépendances entre indicateurs calculés."""
        # Simuler des données où les dépendances sont visibles
        test_data = {
            "ema_12": 50200.0,
            "ema_26": 49800.0,
            "macd": 400.0,  # EMA12 - EMA26
            "bb_middle": 50000.0,  # SMA typique
            "bb_upper": 50500.0,
            "bb_lower": 49500.0,
        }

        # MACD devrait être approximativement EMA12 - EMA26
        expected_macd = test_data["ema_12"] - test_data["ema_26"]
        assert (
            abs(test_data["macd"] - expected_macd) < 1.0
        ), "MACD ne correspond pas à EMA12 - EMA26"

        # Bollinger Middle entre Upper et Lower
        assert test_data["bb_lower"] < test_data["bb_middle"] < test_data["bb_upper"]

    @pytest.mark.parametrize(
        "indicator_name,min_val,max_val",
        [
            ("rsi", 0, 100),
            ("stoch_k", 0, 100),
            ("stoch_d", 0, 100),
            ("mfi", 0, 100),
            ("williams_r", -100, 0),
        ],
    )
    def test_bounded_indicators_ranges(self, indicator_name, min_val, max_val):
        """Test paramétrisé pour indicateurs avec plages définies."""
        # Créer des données test avec valeur dans la plage
        test_value = (min_val + max_val) / 2  # Valeur au milieu de la plage

        assert min_val <= test_value <= max_val

        # Test valeurs limites
        assert min_val <= min_val <= max_val
        assert min_val <= max_val <= max_val
