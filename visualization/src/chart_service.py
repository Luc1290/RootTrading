import logging
from datetime import datetime
from typing import Any

import numpy as np

from visualization.src.data_manager import DataManager

logger = logging.getLogger(__name__)


class ChartService:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager

    async def get_market_chart(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 10000,  # Augmenté pour garder plus d'historique lors du dézoom
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict[str, Any]:
        """Generate market data chart (candlestick)"""
        start_dt = datetime.fromisoformat(start_time) if start_time else None
        end_dt = datetime.fromisoformat(end_time) if end_time else None

        # Fetch market data
        candles = await self.data_manager.get_market_data(
            symbol, interval, limit, start_dt, end_dt
        )

        if not candles:
            return {
                "symbol": symbol,
                "interval": interval,
                "data": [],
                "chart_type": "candlestick",
            }

        # Format for charting
        return {
            "symbol": symbol,
            "interval": interval,
            "chart_type": "candlestick",
            "data": {
                "timestamps": [c["timestamp"] for c in candles],
                "open": [c["open"] for c in candles],
                "high": [c["high"] for c in candles],
                "low": [c["low"] for c in candles],
                "close": [c["close"] for c in candles],
                "volume": [c["volume"] for c in candles],
            },
            "latest_price": candles[-1]["close"] if candles else None,
            "price_change_24h": (
                self._calculate_price_change(candles) if len(candles) > 1 else 0
            ),
        }

    async def get_signals_chart(
        self,
        symbol: str,
        strategy: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict[str, Any]:
        """Generate chart with trading signals overlaid"""
        # Ne pas limiter par défaut aux dernières 24h - laisser la DB retourner
        # toutes les données
        start_dt = datetime.fromisoformat(start_time) if start_time else None
        end_dt = datetime.fromisoformat(end_time) if end_time else None

        # Get both market data and signals
        candles = await self.data_manager.get_market_data(
            symbol, "1m", 10000, start_dt, end_dt  # Plus d'historique pour le dézoom
        )

        signals = await self.data_manager.get_trading_signals(
            symbol, strategy, start_dt, end_dt
        )

        # Group signals by type
        buy_signals = []
        sell_signals = []

        for signal in signals:
            signal_data = {
                "timestamp": signal["timestamp"],
                "price": signal["price"],
                "strength": signal["strength"],
                "strategy": signal["strategy"],
            }

            if signal["signal_type"] == "BUY":
                buy_signals.append(signal_data)
            elif signal["signal_type"] == "SELL":
                sell_signals.append(signal_data)

        return {
            "symbol": symbol,
            "chart_type": "signals",
            "market_data": {
                "timestamps": [c["timestamp"] for c in candles],
                "close": [c["close"] for c in candles],
            },
            "signals": {"buy": buy_signals, "sell": sell_signals},
            "strategies": list({s["strategy"] for s in signals}) if signals else [],
        }

    async def get_performance_chart(
        self, period: str = "24h", metric: str = "pnl"
    ) -> dict[str, Any]:
        """Generate portfolio performance chart"""
        performance_data = await self.data_manager.get_portfolio_performance(period)

        if not performance_data:
            return {"period": period, "metric": metric, "data": []}

        # Select the appropriate metric
        metric_map = {
            "pnl": "pnl",
            "balance": "balances",
            "pnl_percentage": "pnl_percentage",
            "win_rate": "win_rate",
            "sharpe_ratio": "sharpe_ratio",
        }

        selected_metric = metric_map.get(metric, "pnl")

        return {
            "period": period,
            "metric": metric,
            "chart_type": "line",
            "data": {
                "timestamps": performance_data["timestamps"],
                "values": performance_data[selected_metric],
                "metric_name": metric.replace("_", " ").title(),
            },
            "summary": {
                "current": (
                    performance_data[selected_metric][-1]
                    if performance_data[selected_metric]
                    else 0
                ),
                "min": (
                    min(performance_data[selected_metric])
                    if performance_data[selected_metric]
                    else 0
                ),
                "max": (
                    max(performance_data[selected_metric])
                    if performance_data[selected_metric]
                    else 0
                ),
                "average": (
                    np.mean(performance_data[selected_metric])
                    if performance_data[selected_metric]
                    else 0
                ),
            },
        }

    async def get_indicators_chart(
        self,
        symbol: str,
        indicators: list[str],
        interval: str = "1m",
        limit: int = 10000,  # Augmenté pour plus d'historique sur les indicateurs
    ) -> dict[str, Any]:
        """Generate chart with technical indicators"""
        # Fetch market data with enriched indicators
        candles = await self.data_manager.get_market_data(symbol, interval, limit)

        if not candles:
            return {
                "symbol": symbol,
                "interval": interval,
                "indicators": indicators,
                "data": [],
            }

        # Use pre-calculated indicators from database instead of calculating
        # manually
        indicator_data = {}

        for indicator in indicators:
            if indicator == "sma":
                indicator_data["sma_20"] = [c.get("sma_20") for c in candles]
                indicator_data["sma_50"] = [c.get("sma_50") for c in candles]
            elif indicator == "ema":
                indicator_data["ema_7"] = [c.get("ema_7") for c in candles]
                indicator_data["ema_12"] = [c.get("ema_12") for c in candles]
                indicator_data["ema_26"] = [c.get("ema_26") for c in candles]
                indicator_data["ema_50"] = [c.get("ema_50") for c in candles]
                indicator_data["ema_99"] = [c.get("ema_99") for c in candles]
            elif indicator == "rsi":
                indicator_data["rsi_14"] = [c.get("rsi_14") for c in candles]
                indicator_data["rsi_21"] = [c.get("rsi_21") for c in candles]
            elif indicator == "macd":
                indicator_data["macd_line"] = [c.get("macd_line") for c in candles]
                indicator_data["macd_signal"] = [c.get("macd_signal") for c in candles]
                indicator_data["macd_histogram"] = [
                    c.get("macd_histogram") for c in candles
                ]
            elif indicator == "bollinger_bands":
                indicator_data["bb_upper"] = [c.get("bb_upper") for c in candles]
                indicator_data["bb_middle"] = [c.get("bb_middle") for c in candles]
                indicator_data["bb_lower"] = [c.get("bb_lower") for c in candles]
                indicator_data["bb_position"] = [c.get("bb_position") for c in candles]
                indicator_data["bb_width"] = [c.get("bb_width") for c in candles]
            elif indicator == "stochastic":
                indicator_data["stoch_k"] = [c.get("stoch_k") for c in candles]
                indicator_data["stoch_d"] = [c.get("stoch_d") for c in candles]
            elif indicator == "williams_r":
                indicator_data["williams_r"] = [c.get("williams_r") for c in candles]
            elif indicator == "cci":
                indicator_data["cci_20"] = [c.get("cci_20") for c in candles]
            elif indicator == "adx":
                indicator_data["adx_14"] = [c.get("adx_14") for c in candles]
            elif indicator == "momentum":
                indicator_data["momentum_10"] = [c.get("momentum_10") for c in candles]
                indicator_data["roc_10"] = [c.get("roc_10") for c in candles]
                indicator_data["roc_20"] = [c.get("roc_20") for c in candles]
            elif indicator == "volume":
                indicator_data["volume"] = [c["volume"] for c in candles]
                indicator_data["volume_ratio"] = [
                    c.get("volume_ratio") for c in candles
                ]
                indicator_data["avg_volume_20"] = [
                    c.get("avg_volume_20") for c in candles
                ]
            elif indicator == "volume_advanced":
                indicator_data["obv"] = [c.get("obv") for c in candles]
                indicator_data["vwap_10"] = [c.get("vwap_10") for c in candles]
                indicator_data["vwap_quote_10"] = [
                    c.get("vwap_quote_10") for c in candles
                ]
                indicator_data["quote_volume_ratio"] = [
                    c.get("quote_volume_ratio") for c in candles
                ]
                indicator_data["avg_trade_size"] = [
                    c.get("avg_trade_size") for c in candles
                ]
                indicator_data["trade_intensity"] = [
                    c.get("trade_intensity") for c in candles
                ]
            elif indicator == "atr":
                indicator_data["atr_14"] = [c.get("atr_14") for c in candles]
            elif indicator == "regime":
                indicator_data["market_regime"] = [
                    c.get("market_regime") for c in candles
                ]
                indicator_data["regime_strength"] = [
                    c.get("regime_strength") for c in candles
                ]
                indicator_data["regime_confidence"] = [
                    c.get("regime_confidence") for c in candles
                ]
            elif indicator == "volume_context":
                indicator_data["volume_context"] = [
                    c.get("volume_context") for c in candles
                ]
                indicator_data["volume_pattern"] = [
                    c.get("volume_pattern") for c in candles
                ]
                indicator_data["pattern_detected"] = [
                    c.get("pattern_detected") for c in candles
                ]
                indicator_data["data_quality"] = [
                    c.get("data_quality") for c in candles
                ]

        return {
            "symbol": symbol,
            "interval": interval,
            "chart_type": "indicators",
            "market_data": {
                "timestamps": [c["timestamp"] for c in candles],
                "open": [c["open"] for c in candles],
                "high": [c["high"] for c in candles],
                "low": [c["low"] for c in candles],
                "close": [c["close"] for c in candles],
            },
            "indicators": indicator_data,
            "requested_indicators": indicators,
        }

    def _calculate_price_change(self, candles: list[dict]) -> float:
        """Calculate 24h price change percentage"""
        if len(candles) < 2:
            return 0

        # Utiliser le premier et dernier prix disponibles
        # Si on a 24h de données (24 points en 1h), c'est vraiment 24h
        # Si on a moins, c'est la période disponible
        first_close = candles[0]["close"]
        last_close = candles[-1]["close"]

        if first_close == 0:
            return 0

        change_percent = ((last_close - first_close) / first_close) * 100

        # Log pour debug
        logger.debug(
            f"Price change calculation: {first_close} -> {last_close} = {change_percent:.2f}% ({len(candles)} points)"
        )

        return change_percent

    async def get_available_indicators(self) -> dict[str, list[str]]:
        """Get list of all available indicators organized by category"""
        return {
            "trend": [
                "sma",  # SMA 20, 50
                "ema",  # EMA 7, 12, 26, 50, 99
                "adx",  # ADX 14
            ],
            "momentum": [
                "rsi",  # RSI 14, 21
                "stochastic",  # Stochastic K, D
                "williams_r",  # Williams %R
                "cci",  # CCI 20
                "momentum",  # Momentum 10, ROC 10/20
            ],
            "volatility": [
                "bollinger_bands",  # BB Upper/Middle/Lower + Position/Width
                "atr",  # ATR 14
            ],
            "volume": [
                "volume",  # Volume + Ratio + Avg 20
                "volume_advanced",  # OBV, VWAP, VWAP Quote, Trade metrics
            ],
            "oscillators": ["macd"],  # MACD Line, Signal, Histogram
            "market_analysis": [
                "regime",  # Market Regime + Strength + Confidence
                "volume_context",  # Volume Context + Patterns + Quality
            ],
        }

    async def get_indicator_metadata(self) -> dict[str, dict[str, Any]]:
        """Get detailed metadata for each indicator"""
        return {
            "sma": {
                "name": "Simple Moving Average",
                "description": "Trend-following indicator based on average price",
                "parameters": ["20 period", "50 period"],
                "chart_type": "line_overlay",
            },
            "ema": {
                "name": "Exponential Moving Average",
                "description": "Trend indicator giving more weight to recent prices",
                "parameters": ["7, 12, 26, 50, 99 periods"],
                "chart_type": "line_overlay",
            },
            "rsi": {
                "name": "Relative Strength Index",
                "description": "Momentum oscillator measuring speed of price changes",
                "parameters": ["14 and 21 periods"],
                "chart_type": "oscillator",
                "range": [0, 100],
                "levels": [30, 70],
            },
            "macd": {
                "name": "MACD",
                "description": "Trend-following momentum indicator",
                "parameters": ["12-26-9"],
                "chart_type": "oscillator_histogram",
            },
            "bollinger_bands": {
                "name": "Bollinger Bands",
                "description": "Volatility bands around moving average",
                "parameters": ["20 period, 2 std dev"],
                "chart_type": "band_overlay",
            },
            "stochastic": {
                "name": "Stochastic Oscillator",
                "description": "Momentum indicator comparing closing price to price range",
                "parameters": ["14 period"],
                "chart_type": "oscillator",
                "range": [0, 100],
                "levels": [20, 80],
            },
            "williams_r": {
                "name": "Williams %R",
                "description": "Momentum indicator similar to stochastic",
                "parameters": ["14 period"],
                "chart_type": "oscillator",
                "range": [-100, 0],
                "levels": [-80, -20],
            },
            "cci": {
                "name": "Commodity Channel Index",
                "description": "Momentum oscillator measuring deviation from average price",
                "parameters": ["20 period"],
                "chart_type": "oscillator",
                "levels": [-100, 100],
            },
            "adx": {
                "name": "Average Directional Index",
                "description": "Trend strength indicator",
                "parameters": ["14 period"],
                "chart_type": "oscillator",
                "range": [0, 100],
                "levels": [25, 50],
            },
            "momentum": {
                "name": "Momentum & ROC",
                "description": "Rate of change and momentum indicators",
                "parameters": ["10 and 20 periods"],
                "chart_type": "oscillator",
            },
            "volume": {
                "name": "Volume Analysis",
                "description": "Volume and volume ratio analysis",
                "chart_type": "volume_chart",
            },
            "volume_advanced": {
                "name": "Advanced Volume Metrics",
                "description": "OBV, VWAP, trade size and intensity analysis",
                "chart_type": "volume_advanced",
            },
            "atr": {
                "name": "Average True Range",
                "description": "Volatility indicator measuring price range",
                "parameters": ["14 period"],
                "chart_type": "line_chart",
            },
            "regime": {
                "name": "Market Regime Analysis",
                "description": "Identifies current market regime and strength",
                "chart_type": "regime_info",
            },
            "volume_context": {
                "name": "Volume Context Analysis",
                "description": "Advanced volume pattern and context detection",
                "chart_type": "context_info",
            },
        }
