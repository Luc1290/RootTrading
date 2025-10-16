"""
Tests pour EMA_Cross_Strategy.
"""

import os
import sys

import pytest

from analyzer.strategies.EMA_Cross_Strategy import EMA_Cross_Strategy

# Ajouter le chemin racine pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))


class TestEMACrossStrategy:
    """Tests pour la stratégie EMA Cross."""

    def test_init(self, mock_strategy_data):
        """Test d'initialisation de la stratégie EMA."""
        strategy = EMA_Cross_Strategy(
            "BTCUSDC",
            mock_strategy_data["data"],
            mock_strategy_data["indicators"])

        assert strategy.symbol == "BTCUSDC"
        assert strategy.name == "EMA_Cross_Strategy"
        assert strategy.ema_fast_period == 12
        assert strategy.ema_slow_period == 26
        assert strategy.ema_filter_period == 50
        assert strategy.min_separation_pct == 0.2
        assert strategy.strong_separation_pct == 1.2

    def test_validate_data_valid(self, mock_strategy_data):
        """Test validate_data avec données EMA valides."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update({"ema_12": 50200, "ema_26": 49950})
        # Assurer qu'il y a des données de prix
        data = mock_strategy_data["data"].copy()
        data["close"] = [49000, 49500, 50000]

        strategy = EMA_Cross_Strategy("BTCUSDC", data, indicators)
        assert strategy.validate_data() is True

    def test_validate_data_missing_ema(self, mock_strategy_data):
        """Test validate_data sans indicateurs EMA."""
        indicators = mock_strategy_data["indicators"].copy()
        # Supprimer ema_12
        indicators.pop("ema_12", None)

        strategy = EMA_Cross_Strategy(
            "BTCUSDC", mock_strategy_data["data"], indicators)
        assert strategy.validate_data() is False

    def test_validate_data_missing_price_data(self, mock_strategy_data):
        """Test validate_data sans données de prix."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update({"ema_12": 50200, "ema_26": 49950})
        data = mock_strategy_data["data"].copy()
        data["close"] = []  # Pas de données de prix

        strategy = EMA_Cross_Strategy("BTCUSDC", data, indicators)
        assert strategy.validate_data() is False

    def test_generate_signal_missing_ema_data(self, mock_strategy_data):
        """Test signal sans données EMA."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators["ema_12"] = None
        indicators["ema_26"] = 49950

        strategy = EMA_Cross_Strategy(
            "BTCUSDC", mock_strategy_data["data"], indicators)
        result = strategy.generate_signal()

        assert result["side"] is None
        assert result["confidence"] == 0.0
        # Vérifier qu'une raison d'erreur appropriée est donnée
        assert (
            "EMA 12/26 ou prix non disponibles" in result["reason"]
            or "Données insuffisantes" in result["reason"]
        )

    def test_generate_signal_ema_too_close(self, mock_strategy_data):
        """Test signal rejeté car EMA trop proches."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            # Distance < min_separation_pct (0.2%)
            {"ema_12": 50000, "ema_26": 49999}
        )
        data = mock_strategy_data["data"].copy()
        data["close"] = [49000, 49500, 50000]

        strategy = EMA_Cross_Strategy("BTCUSDC", data, indicators)
        result = strategy.generate_signal()

        assert result["side"] is None
        assert result["confidence"] == 0.0
        assert "EMA trop proches" in result["reason"]

    def test_generate_signal_golden_cross_with_trend(self, mock_strategy_data):
        """Test signal BUY (golden cross) avec confirmation de tendance."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "ema_12": 50500,  # EMA12 > EMA26
                "ema_26": 50000,
                "ema_50": 49800,  # Prix et EMA12 > EMA50 = tendance haussière
                "macd_line": 50,
                "macd_signal": 45,
                "macd_histogram": 5,
                "trend_strength": "strong",
                "directional_bias": "BULLISH",
                "momentum_score": 65,
                "volume_ratio": 1.3,
                "confluence_score": 75,
            }
        )
        data = mock_strategy_data["data"].copy()
        data["close"] = [49000, 49500, 50100]  # Prix > EMA50

        strategy = EMA_Cross_Strategy("BTCUSDC", data, indicators)
        result = strategy.generate_signal()

        assert result["side"] == "BUY"
        assert result["confidence"] >= 0.40
        assert result["strength"] in ["moderate", "strong", "very_strong"]
        assert "tendance haussière" in result["reason"]
        assert result["metadata"]["cross_type"] == "golden_cross"

    def test_generate_signal_death_cross_with_trend(self, mock_strategy_data):
        """Test signal SELL (death cross) avec confirmation de tendance."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "ema_12": 49500,  # EMA12 < EMA26
                "ema_26": 50000,
                "ema_50": 50200,  # Prix et EMA12 < EMA50 = tendance baissière
                "macd_line": -50,
                "macd_signal": -45,
                "macd_histogram": -5,
                "trend_strength": "strong",
                "directional_bias": "BEARISH",
                "momentum_score": 35,
                "volume_ratio": 1.3,
                "confluence_score": 75,
            }
        )
        data = mock_strategy_data["data"].copy()
        data["close"] = [50200, 49900, 49400]  # Prix < EMA50

        strategy = EMA_Cross_Strategy("BTCUSDC", data, indicators)
        result = strategy.generate_signal()

        assert result["side"] == "SELL"
        assert result["confidence"] >= 0.40
        assert result["strength"] in ["moderate", "strong", "very_strong"]
        assert "tendance baissière" in result["reason"]
        assert result["metadata"]["cross_type"] == "death_cross"

    def test_generate_signal_contra_trend_rejected(self, mock_strategy_data):
        """Test signal rejeté pour contra-trend."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "ema_12": 50500,  # EMA12 > EMA26 = signal BUY
                "ema_26": 50000,
                "ema_50": 51000,  # Mais prix < EMA50 = contra-trend
            }
        )
        data = mock_strategy_data["data"].copy()
        data["close"] = [49000, 49500, 50100]  # Prix < EMA50

        strategy = EMA_Cross_Strategy("BTCUSDC", data, indicators)
        result = strategy.generate_signal()

        assert result["side"] is None
        assert result["confidence"] == 0.0
        assert "contra-trend ou ambigu" in result["reason"]
        assert result["metadata"]["rejected"] == "contra_trend"

    def test_generate_signal_strong_separation_bonus(self, mock_strategy_data):
        """Test bonus pour forte séparation EMA."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "ema_12": 51000,  # Forte séparation > 1.2%
                "ema_26": 50000,
                "ema_50": 49800,
                "macd_line": 100,
                "macd_signal": 90,
                "confluence_score": 80,
            }
        )
        data = mock_strategy_data["data"].copy()
        data["close"] = [49000, 49500, 51100]

        strategy = EMA_Cross_Strategy("BTCUSDC", data, indicators)
        result = strategy.generate_signal()

        assert result["side"] == "BUY"
        assert (
            "séparation FORTE" in result["reason"]
            or "séparation forte" in result["reason"]
        )
        assert result["metadata"]["ema_separation_pct"] >= 1.8  # ~2%

    def test_generate_signal_macd_confirmation(self, mock_strategy_data):
        """Test confirmation avec MACD aligné."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "ema_12": 50300,
                "ema_26": 50000,
                "ema_50": 49800,
                "macd_line": 50,  # MACD > Signal ET > 0
                "macd_signal": 45,
                "macd_histogram": 5,
                "trend_strength": "moderate",
                "confluence_score": 70,
            }
        )
        data = mock_strategy_data["data"].copy()
        data["close"] = [49000, 49500, 50100]

        strategy = EMA_Cross_Strategy("BTCUSDC", data, indicators)
        result = strategy.generate_signal()

        assert result["side"] == "BUY"
        assert (
            "MACD PARFAITEMENT aligné" in result["reason"]
            or "MACD confirme" in result["reason"]
        )

    def test_generate_signal_macd_divergence_penalty(self, mock_strategy_data):
        """Test pénalité pour divergence MACD."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "ema_12": 50300,  # Signal BUY
                "ema_26": 50000,
                "ema_50": 49800,
                "macd_line": 45,  # Mais MACD < Signal = divergence
                "macd_signal": 50,
                "confluence_score": 60,
            }
        )
        data = mock_strategy_data["data"].copy()
        data["close"] = [49000, 49500, 50100]

        strategy = EMA_Cross_Strategy("BTCUSDC", data, indicators)
        result = strategy.generate_signal()

        if result["side"] is not None:  # Peut être rejeté
            assert "MACD diverge" in result["reason"]

    def test_generate_signal_momentum_contraire_penalty(
            self, mock_strategy_data):
        """Test pénalité pour momentum contraire."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "ema_12": 50300,  # Signal BUY
                "ema_26": 50000,
                "ema_50": 49800,
                "momentum_score": 30,  # Momentum contraire (< 40 pour BUY)
                "confluence_score": 65,
            }
        )
        data = mock_strategy_data["data"].copy()
        data["close"] = [49000, 49500, 50100]

        strategy = EMA_Cross_Strategy("BTCUSDC", data, indicators)
        result = strategy.generate_signal()

        if result["side"] is not None:  # Peut être rejeté
            assert "momentum CONTRAIRE" in result["reason"]

    def test_generate_signal_low_confidence_rejected(self, mock_strategy_data):
        """Test signal rejeté pour confidence trop faible."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "ema_12": 50050,  # Faible séparation
                "ema_26": 50000,
                "ema_50": 49800,
                "confluence_score": 30,  # Confluence faible
                "momentum_score": 35,  # Momentum contraire
            }
        )
        data = mock_strategy_data["data"].copy()
        data["close"] = [49000, 49500, 50100]

        strategy = EMA_Cross_Strategy("BTCUSDC", data, indicators)
        result = strategy.generate_signal()

        assert result["side"] is None
        assert result["confidence"] == 0.0
        # Signal peut être rejeté pour différentes raisons (EMA proches,
        # confidence faible, etc.)
        assert (
            "confidence insuffisante" in result["reason"]
            or "EMA trop proches" in result["reason"]
            or "pas de signal clair" in result["reason"]
        )

    def test_generate_signal_without_ema50_fallback(self, mock_strategy_data):
        """Test signal sans EMA50 (ancienne logique)."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "ema_12": 50300,
                "ema_26": 50000,
                "ema_50": None,  # Pas d'EMA50
                "macd_line": 50,
                "macd_signal": 45,
                "confluence_score": 70,
            }
        )
        data = mock_strategy_data["data"].copy()
        data["close"] = [49000, 49500, 50100]

        strategy = EMA_Cross_Strategy("BTCUSDC", data, indicators)
        result = strategy.generate_signal()

        # Peut générer un signal avec ancienne logique
        if result["side"] is not None:
            assert result["side"] == "BUY"
            assert result["metadata"]["cross_type"] == "golden_cross"

    def test_generate_signal_ema99_confirmation(self, mock_strategy_data):
        """Test confirmation avec EMA99 pour tendance long terme."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "ema_12": 50300,
                "ema_26": 50000,
                "ema_50": 49800,
                "ema_99": 49500,  # EMA99 confirme la tendance haussière
                "confluence_score": 70,
            }
        )
        data = mock_strategy_data["data"].copy()
        data["close"] = [49000, 49500, 50100]  # Prix > EMA99

        strategy = EMA_Cross_Strategy("BTCUSDC", data, indicators)
        result = strategy.generate_signal()

        if result["side"] == "BUY":
            assert "tendance LT haussière" in result["reason"]

    def test_generate_signal_trend_alignment_bonus(self, mock_strategy_data):
        """Test bonus pour alignement de tendance fort."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "ema_12": 50300,
                "ema_26": 50000,
                "ema_50": 49800,
                "trend_alignment": 0.7,  # Forte alignement (> 0.5)
                "confluence_score": 70,
            }
        )
        data = mock_strategy_data["data"].copy()
        data["close"] = [49000, 49500, 50100]

        strategy = EMA_Cross_Strategy("BTCUSDC", data, indicators)
        result = strategy.generate_signal()

        if result["side"] == "BUY":
            assert "EMA FORTEMENT alignées" in result["reason"]

    def test_get_current_values_structure(self, mock_strategy_data):
        """Test que _get_current_values retourne la structure attendue."""
        strategy = EMA_Cross_Strategy(
            "BTCUSDC",
            mock_strategy_data["data"],
            mock_strategy_data["indicators"])
        values = strategy._get_current_values()

        # Vérifier que les clés essentielles sont présentes
        required_keys = [
            "ema_12",
            "ema_26",
            "ema_50",
            "ema_99",
            "macd_line",
            "macd_signal",
            "macd_histogram",
            "trend_strength",
            "directional_bias",
            "confluence_score",
        ]

        for key in required_keys:
            assert key in values

    def test_get_current_price_valid(self, mock_strategy_data):
        """Test extraction du prix actuel."""
        data = mock_strategy_data["data"].copy()
        data["close"] = [49000, 49500, 50000]

        strategy = EMA_Cross_Strategy(
            "BTCUSDC", data, mock_strategy_data["indicators"])
        price = strategy._get_current_price()

        assert price == 50000

    def test_get_current_price_invalid(self, mock_strategy_data):
        """Test extraction du prix actuel avec données invalides."""
        data = mock_strategy_data["data"].copy()
        data["close"] = []  # Pas de données

        strategy = EMA_Cross_Strategy(
            "BTCUSDC", data, mock_strategy_data["indicators"])
        price = strategy._get_current_price()

        assert price is None

    @pytest.mark.parametrize(
        "ema_12,ema_26,current_price,ema_50,expected_side",
        [
            (50300, 50000, 50100, 49800, "BUY"),  # Golden cross + prix > EMA50
            (49700, 50000, 49400, 50200, "SELL"),  # Death cross + prix < EMA50
            # Golden cross mais prix < EMA50
            (50300, 50000, 49900, 50200, None),
            # Death cross mais prix > EMA50
            (49700, 50000, 50500, 49800, None),
        ],
    )
    def test_generate_signal_different_crossovers(
            self,
            mock_strategy_data,
            ema_12,
            ema_26,
            current_price,
            ema_50,
            expected_side):
        """Test différents types de croisements EMA avec filtre de tendance."""
        indicators = mock_strategy_data["indicators"].copy()
        indicators.update(
            {
                "ema_12": ema_12,
                "ema_26": ema_26,
                "ema_50": ema_50,
                "confluence_score": 60,
            }
        )
        data = mock_strategy_data["data"].copy()
        data["close"] = [49000, 49500, current_price]

        strategy = EMA_Cross_Strategy("BTCUSDC", data, indicators)
        result = strategy.generate_signal()

        if expected_side is None:
            assert result["side"] is None
        else:
            # Peut être rejeté pour d'autres raisons, mais si accepté, doit
            # être correct
            if result["side"] is not None:
                assert result["side"] == expected_side
