"""
Tests pour la gestion d'erreurs dans les stratégies.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from analyzer.strategies.MACD_Crossover_Strategy import MACD_Crossover_Strategy
from analyzer.strategies.RSI_Cross_Strategy import RSI_Cross_Strategy
from analyzer.strategies.EMA_Cross_Strategy import EMA_Cross_Strategy


class TestErrorHandling:
    """Tests pour couvrir les cas d'erreur non testés."""
    
    def test_macd_strategy_invalid_types(self):
        """Test MACD avec types invalides dans les indicateurs."""
        data = {'close': [50000, 50100, 50200]}
        indicators = {
            'macd_line': "invalid_string",  # Type invalide
            'macd_signal': 45.0,
            'confluence_score': 70
        }
        
        strategy = MACD_Crossover_Strategy("BTCUSDC", data, indicators)
        result = strategy.generate_signal()
        
        # Doit gérer l'erreur de conversion
        assert result['side'] is None
        assert "Erreur conversion MACD" in result['reason']
        
    def test_rsi_strategy_missing_critical_indicators(self):
        """Test RSI sans indicateurs critiques."""
        data = {'close': [50000]}
        indicators = {
            'rsi_14': 30,
            # market_regime manquant
            'volume_quality_score': 70,
            'confluence_score': 65
        }
        
        strategy = RSI_Cross_Strategy("BTCUSDC", data, indicators)
        result = strategy.generate_signal()
        
        # Doit gérer les indicateurs manquants
        assert result['side'] is None
        
    def test_ema_strategy_conversion_errors(self):
        """Test EMA avec erreurs de conversion."""
        data = {'close': [50000, 50100]}
        indicators = {
            'ema_12': "50000",  # String au lieu de float
            'ema_26': None,
            'confluence_score': 60
        }
        
        strategy = EMA_Cross_Strategy("BTCUSDC", data, indicators)
        result = strategy.generate_signal()
        
        # Doit gérer l'erreur - peut être détectée à différents niveaux
        assert result['side'] is None
        assert ("Erreur conversion EMA" in result['reason'] or 
                "Données insuffisantes" in result['reason'])
        
    def test_strategies_with_empty_data(self):
        """Test stratégies avec données vides."""
        strategies = [
            MACD_Crossover_Strategy,
            RSI_Cross_Strategy,
            EMA_Cross_Strategy
        ]
        
        for StrategyClass in strategies:
            strategy = StrategyClass("BTCUSDC", {}, {})
            result = strategy.generate_signal()
            
            assert result['side'] is None
            assert result['confidence'] == 0.0
            assert "insuffisant" in result['reason'].lower()
            
    def test_strategies_with_none_values(self):
        """Test stratégies avec valeurs None."""
        data = None
        indicators = None
        
        strategies = [
            MACD_Crossover_Strategy,
            RSI_Cross_Strategy, 
            EMA_Cross_Strategy
        ]
        
        for StrategyClass in strategies:
            strategy = StrategyClass("BTCUSDC", data, indicators)
            result = strategy.generate_signal()
            
            assert result['side'] is None
            assert result['confidence'] == 0.0
            
    def test_price_extraction_edge_cases(self):
        """Test extraction prix avec cas limites."""
        # Données prix vides
        data_empty = {'close': []}
        indicators = {'ema_12': 50000, 'ema_26': 49500}
        
        strategy = EMA_Cross_Strategy("BTCUSDC", data_empty, indicators)
        price = strategy._get_current_price()
        assert price is None
        
        # Données prix avec valeurs non numériques
        data_invalid = {'close': [50000, 'invalid', 50200]}
        strategy = EMA_Cross_Strategy("BTCUSDC", data_invalid, indicators)
        # Devrait gérer l'erreur sans crash
        try:
            price = strategy._get_current_price()
            # Peut être None ou raise une exception gérée
            assert price is None or isinstance(price, float)
        except (ValueError, TypeError):
            # Exception attendue et gérée
            pass
            
    def test_confidence_calculation_edge_cases(self):
        """Test calculs de confidence avec cas limites."""
        from analyzer.strategies.base_strategy import BaseStrategy
        
        class TestStrategy(BaseStrategy):
            def generate_signal(self):
                return {}
        
        strategy = TestStrategy("BTCUSDC", {}, {})
        
        # Test avec facteurs extrêmes
        confidence = strategy.calculate_confidence(0.5, 0, 1000)
        assert confidence == 0.0  # Multiplié par 0
        
        confidence = strategy.calculate_confidence(0.1, 10, 10)
        assert confidence == 1.0  # Plafonné à 1.0
        
        confidence = strategy.calculate_confidence(-0.5, 2)
        assert confidence == 0.0  # Négatif ramené à 0
        
    def test_strength_mapping_all_ranges(self):
        """Test mapping confidence -> strength sur toutes les plages."""
        from analyzer.strategies.base_strategy import BaseStrategy
        
        class TestStrategy(BaseStrategy):
            def generate_signal(self):
                return {}
        
        strategy = TestStrategy("BTCUSDC", {}, {})
        
        test_cases = [
            (0.0, "weak"),
            (0.2, "weak"),
            (0.39, "weak"),
            (0.4, "moderate"),
            (0.59, "moderate"),
            (0.6, "strong"),
            (0.79, "strong"),
            (0.8, "very_strong"),
            (0.9, "very_strong"),
            (1.0, "very_strong")
        ]
        
        for confidence, expected_strength in test_cases:
            strength = strategy.get_strength_from_confidence(confidence)
            assert strength == expected_strength, f"Confidence {confidence} -> {strength}, expected {expected_strength}"