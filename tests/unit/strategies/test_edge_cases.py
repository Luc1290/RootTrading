"""
Tests pour les cas limites et branches conditionnelles.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from analyzer.strategies.MACD_Crossover_Strategy import MACD_Crossover_Strategy
from analyzer.strategies.RSI_Cross_Strategy import RSI_Cross_Strategy


class TestEdgeCases:
    """Tests pour couvrir les branches conditionnelles manquées."""
    
    def test_macd_all_market_regimes(self):
        """Test MACD avec tous les régimes de marché."""
        base_indicators = {
            'macd_line': 50.0,
            'macd_signal': 45.0,
            'macd_histogram': 5.0,
            'confluence_score': 70
        }
        data = {'close': [50000]}
        
        regimes = ['VOLATILE', 'RANGING', 'TRENDING_BULL', 'TRENDING_BEAR', 
                  'BREAKOUT_BULL', 'BREAKOUT_BEAR', 'TRANSITION', None, 'UNKNOWN']
        
        for regime in regimes:
            indicators = base_indicators.copy()
            indicators['market_regime'] = regime
            
            strategy = MACD_Crossover_Strategy("BTCUSDC", data, indicators)
            result = strategy.generate_signal()
            
            # Vérifier que la stratégie gère tous les régimes
            assert 'side' in result
            assert 'confidence' in result
            assert 'reason' in result
            
    def test_rsi_all_momentum_ranges(self):
        """Test RSI avec différentes plages de momentum."""
        base_indicators = {
            'rsi_14': 30,  # Zone survente
            'market_regime': 'TRENDING_BULL',
            'directional_bias': 'BULLISH',
            'volume_quality_score': 70,
            'confluence_score': 65,
            'atr_percentile': 50
        }
        data = {'close': [50000]}
        
        # Test différentes valeurs de momentum
        momentum_values = [20, 35, 44, 45, 50, 55, 65, 70, 80]
        
        for momentum in momentum_values:
            indicators = base_indicators.copy()
            indicators['momentum_score'] = momentum
            
            strategy = RSI_Cross_Strategy("BTCUSDC", data, indicators)
            result = strategy.generate_signal()
            
            # Momentum < 45 pour BUY devrait rejeter
            if momentum < 45:
                assert result['side'] is None
                assert "momentum contraire" in result['reason']
            else:
                # Momentum favorable peut générer signal
                assert result['side'] in [None, "BUY"]
                
    def test_macd_trend_alignment_edge_values(self):
        """Test MACD avec valeurs limites de trend_alignment."""
        base_indicators = {
            'macd_line': 50.0,
            'macd_signal': 45.0,
            'confluence_score': 70,
            'market_regime': 'TRENDING_BULL'
        }
        data = {'close': [50000]}
        
        # Test valeurs limites de trend_alignment
        alignment_values = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, None]
        
        for alignment in alignment_values:
            indicators = base_indicators.copy()
            indicators['trend_alignment'] = alignment
            
            strategy = MACD_Crossover_Strategy("BTCUSDC", data, indicators)
            result = strategy.generate_signal()
            
            # Vérifier que toutes les valeurs sont gérées
            assert isinstance(result, dict)
            assert 'side' in result
            
    def test_rsi_extreme_values_boundaries(self):
        """Test RSI avec valeurs exactes aux frontières."""
        base_setup = {
            'market_regime': 'TRENDING_BULL',
            'directional_bias': 'BULLISH',
            'volume_quality_score': 70,
            'confluence_score': 65,
            'momentum_score': 70,
            'atr_percentile': 50
        }
        data = {'close': [50000]}
        
        # Test valeurs exactes aux seuils
        boundary_rsi_values = [22, 23, 31, 32, 33, 67, 68, 69, 77, 78, 79]
        
        for rsi_value in boundary_rsi_values:
            indicators = base_setup.copy()
            indicators['rsi_14'] = rsi_value
            
            strategy = RSI_Cross_Strategy("BTCUSDC", data, indicators)
            result = strategy.generate_signal()
            
            # Vérifier comportement selon les seuils
            if rsi_value <= 32:  # oversold_level
                assert result['side'] in [None, "BUY"]
            elif rsi_value >= 68:  # overbought_level  
                # Besoin tendance baissière pour SELL
                pass
            else:
                # Zone neutre
                assert result['side'] is None
                
    def test_strategies_with_partial_indicators(self):
        """Test stratégies avec indicateurs partiels."""
        data = {'close': [50000]}
        
        # MACD avec seulement les indicateurs essentiels
        minimal_macd = {
            'macd_line': 50.0,
            'macd_signal': 45.0,
            'confluence_score': 70
        }
        
        strategy = MACD_Crossover_Strategy("BTCUSDC", data, minimal_macd)
        result = strategy.generate_signal()
        assert isinstance(result, dict)
        
        # RSI avec indicateurs minimaux
        minimal_rsi = {
            'rsi_14': 30,
            'market_regime': 'TRENDING_BULL',
            'volume_quality_score': 70,
            'confluence_score': 65
        }
        
        strategy = RSI_Cross_Strategy("BTCUSDC", data, minimal_rsi)
        result = strategy.generate_signal()
        assert isinstance(result, dict)
        
    def test_confluence_score_boundary_values(self):
        """Test avec valeurs limites de confluence_score."""
        data = {'close': [50000]}
        
        # Test MACD avec confluence aux limites
        macd_base = {
            'macd_line': 50.0,
            'macd_signal': 45.0,
            'market_regime': 'TRENDING_BULL'
        }
        
        confluence_values = [49, 50, 51, 54, 55, 69, 70, 71, 79, 80, 81]
        
        for confluence in confluence_values:
            indicators = macd_base.copy()
            indicators['confluence_score'] = confluence
            
            strategy = MACD_Crossover_Strategy("BTCUSDC", data, indicators)
            result = strategy.generate_signal()
            
            if confluence < 50:  # En dessous du minimum requis
                assert result['side'] is None
                assert "confluence insuffisante" in result['reason']
            else:
                # Au-dessus du minimum, peut générer un signal
                assert 'side' in result
                
    def test_volume_quality_score_ranges(self):
        """Test RSI avec différentes qualités de volume."""
        base_setup = {
            'rsi_14': 30,
            'market_regime': 'TRENDING_BULL',
            'directional_bias': 'BULLISH',
            'confluence_score': 65,
            'momentum_score': 70,
            'atr_percentile': 50
        }
        data = {'close': [50000]}
        
        volume_scores = [0, 30, 59, 60, 61, 70, 80, 90, 100]
        
        for volume_score in volume_scores:
            indicators = base_setup.copy()
            indicators['volume_quality_score'] = volume_score
            
            strategy = RSI_Cross_Strategy("BTCUSDC", data, indicators)
            result = strategy.generate_signal()
            
            if volume_score < 60:  # En dessous du minimum
                assert result['side'] is None
                assert "Volume qualité insuffisante" in result['reason']
            else:
                # Volume suffisant
                assert result['side'] in [None, "BUY"]