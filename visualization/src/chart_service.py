import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from data_manager import DataManager

logger = logging.getLogger(__name__)

class ChartService:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        
    async def get_market_chart(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 10000,  # Augmenté pour garder plus d'historique lors du dézoom
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Dict[str, Any]:
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
                "chart_type": "candlestick"
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
                "volume": [c["volume"] for c in candles]
            },
            "latest_price": candles[-1]["close"] if candles else None,
            "price_change_24h": self._calculate_price_change(candles) if len(candles) > 1 else 0
        }
        
    async def get_signals_chart(
        self,
        symbol: str,
        strategy: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate chart with trading signals overlaid"""
        # Ne pas limiter par défaut aux dernières 24h - laisser la DB retourner toutes les données
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
                "strategy": signal["strategy"]
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
                "close": [c["close"] for c in candles]
            },
            "signals": {
                "buy": buy_signals,
                "sell": sell_signals
            },
            "strategies": list(set(s["strategy"] for s in signals)) if signals else []
        }
        
    async def get_performance_chart(
        self,
        period: str = "24h",
        metric: str = "pnl"
    ) -> Dict[str, Any]:
        """Generate portfolio performance chart"""
        performance_data = await self.data_manager.get_portfolio_performance(period)
        
        if not performance_data:
            return {
                "period": period,
                "metric": metric,
                "data": []
            }
            
        # Select the appropriate metric
        metric_map = {
            "pnl": "pnl",
            "balance": "balances",
            "pnl_percentage": "pnl_percentage",
            "win_rate": "win_rate",
            "sharpe_ratio": "sharpe_ratio"
        }
        
        selected_metric = metric_map.get(metric, "pnl")
        
        return {
            "period": period,
            "metric": metric,
            "chart_type": "line",
            "data": {
                "timestamps": performance_data["timestamps"],
                "values": performance_data[selected_metric],
                "metric_name": metric.replace("_", " ").title()
            },
            "summary": {
                "current": performance_data[selected_metric][-1] if performance_data[selected_metric] else 0,
                "min": min(performance_data[selected_metric]) if performance_data[selected_metric] else 0,
                "max": max(performance_data[selected_metric]) if performance_data[selected_metric] else 0,
                "average": np.mean(performance_data[selected_metric]) if performance_data[selected_metric] else 0
            }
        }
        
    async def get_indicators_chart(
        self,
        symbol: str,
        indicators: List[str],
        interval: str = "1m",
        limit: int = 10000  # Augmenté pour plus d'historique sur les indicateurs
    ) -> Dict[str, Any]:
        """Generate chart with technical indicators"""
        # Fetch market data with enriched indicators
        candles = await self.data_manager.get_market_data(
            symbol, interval, limit
        )
        
        if not candles:
            return {
                "symbol": symbol,
                "interval": interval,
                "indicators": indicators,
                "data": []
            }
            
        # Use pre-calculated indicators from database instead of calculating manually
        indicator_data = {}
        
        for indicator in indicators:
            if indicator == "sma":
                indicator_data["sma_20"] = [c.get("sma_20") for c in candles]
                indicator_data["sma_50"] = [c.get("sma_50") for c in candles]
            elif indicator == "ema":
                indicator_data["ema_7"] = [c.get("ema_7") for c in candles]
                indicator_data["ema_26"] = [c.get("ema_26") for c in candles]
                indicator_data["ema_99"] = [c.get("ema_99") for c in candles]
            elif indicator == "rsi":
                indicator_data["rsi"] = [c.get("rsi_14") for c in candles]
            elif indicator == "macd":
                indicator_data["macd"] = [c.get("macd_line") for c in candles]
                indicator_data["macd_signal"] = [c.get("macd_signal") for c in candles]
                indicator_data["macd_histogram"] = [c.get("macd_histogram") for c in candles]
            elif indicator == "bollinger_bands":
                indicator_data["bb_upper"] = [c.get("bb_upper") for c in candles]
                indicator_data["bb_middle"] = [c.get("bb_middle") for c in candles]
                indicator_data["bb_lower"] = [c.get("bb_lower") for c in candles]
            elif indicator == "volume":
                indicator_data["volume"] = [c["volume"] for c in candles]
            elif indicator == "atr":
                indicator_data["atr"] = [c.get("atr_14") for c in candles]
                
        return {
            "symbol": symbol,
            "interval": interval,
            "chart_type": "indicators",
            "market_data": {
                "timestamps": [c["timestamp"] for c in candles],
                "open": [c["open"] for c in candles],
                "high": [c["high"] for c in candles],
                "low": [c["low"] for c in candles],
                "close": [c["close"] for c in candles]
            },
            "indicators": indicator_data,
            "requested_indicators": indicators
        }
        
    def _calculate_price_change(self, candles: List[Dict]) -> float:
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
        logger.debug(f"Price change calculation: {first_close} -> {last_close} = {change_percent:.2f}% ({len(candles)} points)")
        
        return change_percent