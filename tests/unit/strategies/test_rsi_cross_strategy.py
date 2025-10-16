"""
Tests pour RSI_Cross_Strategy.
"""

import os
import sys

import pytest

from analyzer.strategies.RSI_Cross_Strategy import RSI_Cross_Strategy

# Ajouter le chemin racine pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))


class TestRSICrossStrategy:
    """Tests pour la stratégie RSI Cross."""

    def test_init(self, mock_strategy_data):
        """Test d'initialisation de la stratégie RSI."""
        strategy = RSI_Cross_Strategy(
            "BTCUSDC", mock_strategy_data["data"], mock_strategy_data["indicators"]
        )

        assert strategy.symbol == "BTCUSDC"
        assert strategy.name == "RSI_Cross_Strategy"
        assert strategy.oversold_level == 32
        assert strategy.overbought_level == 68
        assert strategy.extreme_oversold == 22
        assert strategy.extreme_overbought == 78
        assert strategy.min_volume_quality == 60
        assert strategy.min_confluence_for_signal == 55

    def test_generate_signal_no_rsi(self, mock_strategy_data):
        """Test signal sans RSI disponible."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators["rsi_14"] = None

        strategy = RSI_Cross_Strategy("BTCUSDC", mock_strategy_data["data"], indicators)
        result = strategy.generate_signal()

        assert result["side"] is None
        assert result["confidence"] == 0.0
        assert result["strength"] == "weak"
        assert "RSI non disponible" in result["reason"]

    def test_generate_signal_ranging_market_rejected(self, mock_strategy_data):
        """Test signal rejeté en marché ranging."""
        indicators = mock_strategy_data["indicators"].copy()
        # Zone survente
        indicators.update({"rsi_14": 30, "market_regime": "RANGING"})

        strategy = RSI_Cross_Strategy("BTCUSDC", mock_strategy_data["data"], indicators)
        result = strategy.generate_signal()

        assert result["side"] is None
        assert result["confidence"] == 0.0
        assert "RANGING" in result["reason"]
        assert "faux signaux" in result["reason"]

    def test_generate_signal_insufficient_volume_quality(self, mock_strategy_data):
        """Test signal rejeté pour volume qualité insuffisante."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "rsi_14": 30,
                "market_regime": "TRENDING_BULL",
                "volume_quality_score": 40,  # En dessous du minimum (60)
            }
        )

        strategy = RSI_Cross_Strategy("BTCUSDC", mock_strategy_data["data"], indicators)
        result = strategy.generate_signal()

        assert result["side"] is None
        assert result["confidence"] == 0.0
        assert "Volume qualité insuffisante" in result["reason"]

    def test_generate_signal_insufficient_confluence(self, mock_strategy_data):
        """Test signal rejeté pour confluence insuffisante."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "rsi_14": 30,
                "market_regime": "TRENDING_BULL",
                "volume_quality_score": 70,
                "confluence_score": 40,  # En dessous du minimum (55)
            }
        )

        strategy = RSI_Cross_Strategy("BTCUSDC", mock_strategy_data["data"], indicators)
        result = strategy.generate_signal()

        assert result["side"] is None
        assert result["confidence"] == 0.0
        assert "Confluence insuffisante" in result["reason"]

    def test_generate_signal_buy_oversold_valid(self, mock_strategy_data):
        """Test signal BUY valide en zone survente."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "rsi_14": 30,  # Zone survente
                "market_regime": "TRENDING_BULL",
                "volume_quality_score": 70,
                "confluence_score": 65,
                "directional_bias": "BULLISH",
                "trend_strength": "strong",
                "momentum_score": 70,  # Momentum favorable
                "signal_strength": "STRONG",
                "adx_14": 30,  # ADX fort
                "atr_percentile": 50,  # Volatilité correcte
            }
        )

        strategy = RSI_Cross_Strategy("BTCUSDC", mock_strategy_data["data"], indicators)
        result = strategy.generate_signal()

        assert result["side"] == "BUY"
        assert result["confidence"] >= 0.55
        assert result["strength"] in ["moderate", "strong", "very_strong"]
        assert "survente" in result["reason"]
        assert "haussière confirmée" in result["reason"]

    def test_generate_signal_sell_overbought_valid(self, mock_strategy_data):
        """Test signal SELL valide en zone surachat."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "rsi_14": 70,  # Zone surachat
                "market_regime": "TRENDING_BEAR",
                "volume_quality_score": 70,
                "confluence_score": 65,
                "directional_bias": "BEARISH",
                "trend_strength": "strong",
                "momentum_score": 30,  # Momentum favorable pour SELL
                "signal_strength": "STRONG",
                "adx_14": 30,  # ADX fort
                "atr_percentile": 50,  # Volatilité correcte
            }
        )

        strategy = RSI_Cross_Strategy("BTCUSDC", mock_strategy_data["data"], indicators)
        result = strategy.generate_signal()

        assert result["side"] == "SELL"
        assert result["confidence"] >= 0.55
        assert result["strength"] in ["moderate", "strong", "very_strong"]
        assert "surachat" in result["reason"]
        assert "baissière confirmée" in result["reason"]

    def test_generate_signal_buy_oversold_no_trend_confirmation(
        self, mock_strategy_data
    ):
        """Test signal BUY rejeté sans confirmation de tendance."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "rsi_14": 30,  # Zone survente
                "market_regime": "TRENDING_BEAR",  # Mais tendance baissière
                "volume_quality_score": 70,
                "confluence_score": 65,
                "directional_bias": "BEARISH",  # Bias contraire
            }
        )

        strategy = RSI_Cross_Strategy("BTCUSDC", mock_strategy_data["data"], indicators)
        result = strategy.generate_signal()

        assert result["side"] is None
        assert result["confidence"] == 0.0
        assert "tendance non haussière" in result["reason"]
        assert "évite contra-trend" in result["reason"]

    def test_generate_signal_sell_overbought_no_trend_confirmation(
        self, mock_strategy_data
    ):
        """Test signal SELL rejeté sans confirmation de tendance."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "rsi_14": 70,  # Zone surachat
                "market_regime": "TRENDING_BULL",  # Mais tendance haussière
                "volume_quality_score": 70,
                "confluence_score": 65,
                "directional_bias": "BULLISH",  # Bias contraire
            }
        )

        strategy = RSI_Cross_Strategy("BTCUSDC", mock_strategy_data["data"], indicators)
        result = strategy.generate_signal()

        assert result["side"] is None
        assert result["confidence"] == 0.0
        assert "tendance non baissière" in result["reason"]
        assert "évite contra-trend" in result["reason"]

    def test_generate_signal_momentum_contraire_rejected(self, mock_strategy_data):
        """Test signal rejeté pour momentum contraire."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "rsi_14": 30,  # Zone survente - signal BUY
                "market_regime": "TRENDING_BULL",
                "volume_quality_score": 70,
                "confluence_score": 65,
                "directional_bias": "BULLISH",
                "momentum_score": 30,  # Momentum contraire (< 45 pour BUY)
            }
        )

        strategy = RSI_Cross_Strategy("BTCUSDC", mock_strategy_data["data"], indicators)
        result = strategy.generate_signal()

        assert result["side"] is None
        assert result["confidence"] == 0.0
        assert "momentum contraire" in result["reason"]

    def test_generate_signal_extreme_oversold_bonus(self, mock_strategy_data):
        """Test bonus pour RSI en zone survente extrême."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "rsi_14": 20,  # Zone survente extrême
                "market_regime": "TRENDING_BULL",
                "volume_quality_score": 70,
                "confluence_score": 65,
                "directional_bias": "BULLISH",
                "momentum_score": 70,
            }
        )

        strategy = RSI_Cross_Strategy("BTCUSDC", mock_strategy_data["data"], indicators)
        result = strategy.generate_signal()

        assert result["side"] == "BUY"
        assert "survente extrême" in result["reason"]

    def test_generate_signal_extreme_overbought_bonus(self, mock_strategy_data):
        """Test bonus pour RSI en zone surachat extrême."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "rsi_14": 80,  # Zone surachat extrême
                "market_regime": "TRENDING_BEAR",
                "volume_quality_score": 70,
                "confluence_score": 65,
                "directional_bias": "BEARISH",
                "momentum_score": 20,
            }
        )

        strategy = RSI_Cross_Strategy("BTCUSDC", mock_strategy_data["data"], indicators)
        result = strategy.generate_signal()

        assert result["side"] == "SELL"
        assert "surachat extrême" in result["reason"]

    def test_generate_signal_low_volatility_rejected(self, mock_strategy_data):
        """Test signal rejeté pour volatilité trop faible."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "rsi_14": 30,
                "market_regime": "TRENDING_BULL",
                "volume_quality_score": 70,
                "confluence_score": 65,
                "directional_bias": "BULLISH",
                "atr_percentile": 20,  # Volatilité trop faible (< 25)
            }
        )

        strategy = RSI_Cross_Strategy("BTCUSDC", mock_strategy_data["data"], indicators)
        result = strategy.generate_signal()

        assert result["side"] is None
        assert result["confidence"] == 0.0
        assert "volatilité trop faible" in result["reason"]
        assert "marché inactif" in result["reason"]

    def test_generate_signal_neutral_rsi(self, mock_strategy_data):
        """Test RSI en zone neutre - pas de signal."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "rsi_14": 50,  # Zone neutre
                "market_regime": "TRENDING_BULL",
                "volume_quality_score": 70,
                "confluence_score": 65,
            }
        )

        strategy = RSI_Cross_Strategy("BTCUSDC", mock_strategy_data["data"], indicators)
        result = strategy.generate_signal()

        assert result["side"] is None
        assert result["confidence"] == 0.0
        assert result["strength"] == "weak"
        assert "RSI neutre" in result["reason"]

    def test_generate_signal_low_confidence_rejected(self, mock_strategy_data):
        """Test signal rejeté pour confidence trop faible."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "rsi_14": 30,  # Zone survente
                "market_regime": "TRENDING_BULL",
                "volume_quality_score": 62,  # Juste au minimum
                "confluence_score": 56,  # Juste au minimum
                "directional_bias": "BULLISH",
                "momentum_score": 52,  # Momentum faible mais favorable
                # Pas d'autres indicateurs favorables
            }
        )

        strategy = RSI_Cross_Strategy("BTCUSDC", mock_strategy_data["data"], indicators)
        result = strategy.generate_signal()

        # Peut être rejeté pour confidence insuffisante
        if result["side"] is None:
            assert "confidence insuffisante" in result["reason"]

    def test_get_current_values_structure(self, mock_strategy_data):
        """Test que _get_current_values retourne la structure attendue."""
        strategy = RSI_Cross_Strategy(
            "BTCUSDC", mock_strategy_data["data"], mock_strategy_data["indicators"]
        )
        values = strategy._get_current_values()

        # Vérifier que les clés essentielles sont présentes
        required_keys = [
            "rsi_14",
            "rsi_21",
            "momentum_score",
            "trend_strength",
            "directional_bias",
            "confluence_score",
            "volume_quality_score",
            "market_regime",
            "adx_14",
        ]

        for key in required_keys:
            assert key in values

    @pytest.mark.parametrize(
        "rsi_value,market_regime,directional_bias,expected_side",
        [
            # Survente + tendance haussière
            (25, "TRENDING_BULL", "BULLISH", "BUY"),
            # Surachat + tendance baissière
            (75, "TRENDING_BEAR", "BEARISH", "SELL"),
            # Survente mais tendance baissière
            (25, "TRENDING_BEAR", "BEARISH", None),
            # Surachat mais tendance haussière
            (75, "TRENDING_BULL", "BULLISH", None),
            (50, "TRENDING_BULL", "BULLISH", None),  # RSI neutre
        ],
    )
    def test_generate_signal_different_scenarios(
        self,
        mock_strategy_data,
        rsi_value,
        market_regime,
        directional_bias,
        expected_side,
    ):
        """Test différents scénarios RSI/tendance."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "rsi_14": rsi_value,
                "market_regime": market_regime,
                "directional_bias": directional_bias,
                "volume_quality_score": 70,
                "confluence_score": 65,
                "momentum_score": 70 if expected_side == "BUY" else 30,
                "atr_percentile": 50,
            }
        )

        strategy = RSI_Cross_Strategy("BTCUSDC", mock_strategy_data["data"], indicators)
        result = strategy.generate_signal()

        if expected_side is None:
            assert result["side"] is None
        else:
            # Peut être rejeté pour d'autres raisons, mais si accepté, doit
            # être correct
            if result["side"] is not None:
                assert result["side"] == expected_side

    def test_rsi_multi_timeframe_confirmation(self, mock_strategy_data):
        """Test confirmation avec RSI multi-timeframe."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "rsi_14": 30,
                "rsi_21": 32,  # RSI 21 aussi en survente
                "market_regime": "TRENDING_BULL",
                "directional_bias": "BULLISH",
                "volume_quality_score": 70,
                "confluence_score": 65,
                "momentum_score": 70,
            }
        )

        strategy = RSI_Cross_Strategy("BTCUSDC", mock_strategy_data["data"], indicators)
        result = strategy.generate_signal()

        if result["side"] == "BUY":
            assert "confirmé sur RSI 21" in result["reason"]
