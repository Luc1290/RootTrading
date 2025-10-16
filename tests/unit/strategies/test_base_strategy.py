"""
Tests pour BaseStrategy - classe de base de toutes les stratégies.
"""

import os
import sys

import pytest

from analyzer.strategies.base_strategy import BaseStrategy

# Ajouter le chemin racine pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))


class TestBaseStrategy:
    """Tests pour la classe BaseStrategy."""

    def test_init(self, sample_db_data, sample_indicators):
        """Test d'initialisation de BaseStrategy."""

        # Créer une classe concrète pour tester BaseStrategy
        class ConcreteStrategy(BaseStrategy):
            def generate_signal(self):
                return {
                    "side": "BUY",
                    "confidence": 0.5,
                    "strength": "moderate",
                    "reason": "Test signal",
                    "metadata": {},
                }

        strategy = ConcreteStrategy("BTCUSDC", sample_db_data, sample_indicators)

        assert strategy.symbol == "BTCUSDC"
        assert strategy.data == sample_db_data
        assert strategy.indicators == sample_indicators
        assert strategy.name == "ConcreteStrategy"

    def test_validate_data_valid(self, sample_db_data, sample_indicators):
        """Test validate_data avec des données valides."""

        class ConcreteStrategy(BaseStrategy):
            def generate_signal(self):
                return {}

        strategy = ConcreteStrategy("BTCUSDC", sample_db_data, sample_indicators)
        assert strategy.validate_data() is True

    def test_validate_data_invalid_no_data(self, sample_indicators):
        """Test validate_data sans données."""

        class ConcreteStrategy(BaseStrategy):
            def generate_signal(self):
                return {}

        strategy = ConcreteStrategy("BTCUSDC", None, sample_indicators)
        assert strategy.validate_data() is False

    def test_validate_data_invalid_no_indicators(self, sample_db_data):
        """Test validate_data sans indicateurs."""

        class ConcreteStrategy(BaseStrategy):
            def generate_signal(self):
                return {}

        strategy = ConcreteStrategy("BTCUSDC", sample_db_data, None)
        assert strategy.validate_data() is False

    def test_calculate_confidence(self, mock_strategy_data):
        """Test du calcul de confiance avec facteurs."""

        class ConcreteStrategy(BaseStrategy):
            def generate_signal(self):
                return {}

        strategy = ConcreteStrategy(
            "BTCUSDC", mock_strategy_data["data"], mock_strategy_data["indicators"]
        )

        # Test avec facteurs positifs
        confidence = strategy.calculate_confidence(0.5, 1.2, 1.1)
        assert confidence == min(0.5 * 1.2 * 1.1, 1.0)

        # Test avec facteurs qui dépassent 1.0
        confidence = strategy.calculate_confidence(0.8, 1.5, 1.3)
        assert confidence == 1.0  # Plafonné à 1.0

        # Test avec facteurs négatifs (confiance 0)
        confidence = strategy.calculate_confidence(0.5, 0.5, 0.2)
        assert confidence == 0.05
        assert confidence >= 0.0  # Ne peut pas être négatif

    def test_get_strength_from_confidence(self, mock_strategy_data):
        """Test de la conversion confidence -> strength."""

        class ConcreteStrategy(BaseStrategy):
            def generate_signal(self):
                return {}

        strategy = ConcreteStrategy(
            "BTCUSDC", mock_strategy_data["data"], mock_strategy_data["indicators"]
        )

        # Test des différents seuils
        assert strategy.get_strength_from_confidence(0.9) == "very_strong"
        assert strategy.get_strength_from_confidence(0.8) == "very_strong"
        assert strategy.get_strength_from_confidence(0.7) == "strong"
        assert strategy.get_strength_from_confidence(0.6) == "strong"
        assert strategy.get_strength_from_confidence(0.5) == "moderate"
        assert strategy.get_strength_from_confidence(0.4) == "moderate"
        assert strategy.get_strength_from_confidence(0.3) == "weak"
        assert strategy.get_strength_from_confidence(0.1) == "weak"
        assert strategy.get_strength_from_confidence(0.0) == "weak"

    def test_abstract_generate_signal(self):
        """Test que generate_signal est abstraite et doit être implémentée."""

        # Impossible d'instancier BaseStrategy directement
        with pytest.raises(TypeError):
            BaseStrategy("BTCUSDC", {}, {})
