"""
Cached Indicator Factory

High-level factory for creating cached indicator calculations.
Provides a unified interface for all cached indicators with automatic fallback.
"""

from typing import Dict, List, Optional, Union, Any
import logging

# Import cached indicator functions
from .momentum.rsi import (
    calculate_rsi_cached,
    calculate_rsi_series_cached,
    clear_rsi_cache,
)
from .trend.moving_averages import (
    calculate_ema_cached,
    calculate_ema_series_cached,
    calculate_sma_cached,
    get_ma_cross_signal_cached,
    clear_ma_cache,
)
from .trend.macd import calculate_macd_series  # Will be enhanced later
from .volatility.bollinger import (
    calculate_bollinger_bands_series,
)  # Will be enhanced later
from .volume.obv import calculate_obv_series  # Will be enhanced later
from .volume.vwap import calculate_vwap_series  # Will be enhanced later

# Import cache management
from shared.src.indicator_cache import get_indicator_cache

logger = logging.getLogger(__name__)


class CachedIndicatorFactory:
    """
    Factory class for high-performance cached indicator calculations.

    Provides a unified interface for all technical indicators with:
    - Automatic caching with Redis persistence
    - Incremental calculations where possible
    - Fallback to non-cached versions on errors
    - Cache management and statistics
    """

    def __init__(self, enable_cache: bool = True, default_symbol: Optional[str] = None):
        """
        Initialize the cached indicator factory.

        Args:
            enable_cache: Global cache enable/disable
            default_symbol: Default symbol if not provided in calls
        """
        self.enable_cache = enable_cache
        self.default_symbol = default_symbol
        self._cache = get_indicator_cache() if enable_cache else None

        logger.info(
            f"CachedIndicatorFactory initialized - Cache: {'Enabled' if enable_cache else 'Disabled'}"
        )

    # ============ MOMENTUM INDICATORS ============

    def rsi(
        self,
        prices: Union[List[float], Any],
        symbol: Optional[str] = None,
        period: int = 14,
    ) -> Optional[float]:
        """Calculate RSI with caching."""
        symbol = symbol or self.default_symbol
        if not symbol:
            raise ValueError("Symbol required for cached calculations")

        return calculate_rsi_cached(prices, symbol, period, self.enable_cache)

    def rsi_series(
        self,
        prices: Union[List[float], Any],
        symbol: Optional[str] = None,
        period: int = 14,
    ) -> List[Optional[float]]:
        """Calculate RSI series with caching."""
        symbol = symbol or self.default_symbol
        if not symbol:
            raise ValueError("Symbol required for cached calculations")

        return calculate_rsi_series_cached(prices, symbol, period, self.enable_cache)

    def stochastic_rsi(
        self,
        prices: Union[List[float], Any],
        symbol: Optional[str] = None,
        rsi_period: int = 14,
        stoch_period: int = 14,
    ) -> Optional[Dict[str, Optional[float]]]:
        """Calculate Stochastic RSI (basic implementation - can be enhanced with caching)."""
        from .momentum.rsi import calculate_stoch_rsi_full

        return calculate_stoch_rsi_full(prices, rsi_period, stoch_period)

    # ============ TREND INDICATORS ============

    def ema(
        self,
        prices: Union[List[float], Any],
        symbol: Optional[str] = None,
        period: int = 26,
    ) -> Optional[float]:
        """Calculate EMA with caching."""
        symbol = symbol or self.default_symbol
        if not symbol:
            raise ValueError("Symbol required for cached calculations")

        return calculate_ema_cached(prices, symbol, period, self.enable_cache)

    def ema_series(
        self,
        prices: Union[List[float], Any],
        symbol: Optional[str] = None,
        period: int = 26,
    ) -> List[Optional[float]]:
        """Calculate EMA series with caching."""
        symbol = symbol or self.default_symbol
        if not symbol:
            raise ValueError("Symbol required for cached calculations")

        return calculate_ema_series_cached(prices, symbol, period, self.enable_cache)

    def sma(
        self,
        prices: Union[List[float], Any],
        symbol: Optional[str] = None,
        period: int = 20,
    ) -> Optional[float]:
        """Calculate SMA with caching."""
        symbol = symbol or self.default_symbol
        if not symbol:
            raise ValueError("Symbol required for cached calculations")

        return calculate_sma_cached(prices, symbol, period, self.enable_cache)

    def ma_cross_signal(
        self,
        prices: Union[List[float], Any],
        symbol: Optional[str] = None,
        fast_period: int = 7,
        slow_period: int = 26,
        ma_type: str = "ema",
    ) -> Dict:
        """Get MA crossover signals with caching."""
        symbol = symbol or self.default_symbol
        if not symbol:
            raise ValueError("Symbol required for cached calculations")

        return get_ma_cross_signal_cached(
            prices, symbol, fast_period, slow_period, ma_type
        )

    def macd(
        self,
        prices: Union[List[float], Any],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Dict[str, List[Optional[float]]]:
        """Calculate MACD (will be enhanced with caching later)."""
        return calculate_macd_series(prices, fast_period, slow_period, signal_period)

    # ============ VOLATILITY INDICATORS ============

    def bollinger_bands(
        self,
        prices: Union[List[float], Any],
        period: int = 20,
        std_multiplier: float = 2.0,
    ) -> Dict[str, List[Optional[float]]]:
        """Calculate Bollinger Bands (will be enhanced with caching later)."""
        return calculate_bollinger_bands_series(prices, period, std_multiplier)

    def atr(
        self,
        highs: Union[List[float], Any],
        lows: Union[List[float], Any],
        closes: Union[List[float], Any],
        period: int = 14,
    ) -> Optional[float]:
        """Calculate ATR (will be enhanced with caching later)."""
        from .volatility.atr import calculate_atr

        return calculate_atr(highs, lows, closes, period)

    # ============ VOLUME INDICATORS ============

    def obv(
        self, prices: Union[List[float], Any], volumes: Union[List[float], Any]
    ) -> List[Optional[float]]:
        """Calculate OBV (will be enhanced with caching later)."""
        return calculate_obv_series(prices, volumes)

    def vwap(
        self,
        highs: Union[List[float], Any],
        lows: Union[List[float], Any],
        closes: Union[List[float], Any],
        volumes: Union[List[float], Any],
        period: Optional[int] = None,
    ) -> List[Optional[float]]:
        """Calculate VWAP (will be enhanced with caching later)."""
        return calculate_vwap_series(highs, lows, closes, volumes, period)

    # ============ MULTI-INDICATOR ANALYSIS ============

    def get_trend_analysis(
        self, prices: Union[List[float], Any], symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive trend analysis with cached indicators.

        Returns:
            Dictionary with multiple trend indicators and signals
        """
        symbol = symbol or self.default_symbol
        if not symbol:
            raise ValueError("Symbol required for cached calculations")

        try:
            # EMAs
            ema7 = self.ema(prices, symbol, 7)
            ema26 = self.ema(prices, symbol, 26)
            ema99 = self.ema(prices, symbol, 99)

            # MA Cross signals
            ema_cross = self.ma_cross_signal(prices, symbol, 7, 26, "ema")

            # MACD
            macd = self.macd(prices)

            # Current price
            current_price = float(prices[-1]) if prices else None

            # Trend determination
            trend = "neutral"
            if ema7 and ema26 and ema99:
                if ema7 > ema26 > ema99:
                    trend = "bullish"
                elif ema7 < ema26 < ema99:
                    trend = "bearish"

            return {
                "trend": trend,
                "current_price": current_price,
                "ema7": ema7,
                "ema26": ema26,
                "ema99": ema99,
                "ema_cross": ema_cross,
                "macd": {
                    "line": macd["macd_line"][-1] if macd["macd_line"] else None,
                    "signal": macd["macd_signal"][-1] if macd["macd_signal"] else None,
                    "histogram": (
                        macd["macd_histogram"][-1] if macd["macd_histogram"] else None
                    ),
                },
                "symbol": symbol,
            }

        except Exception as e:
            logger.error(f"Erreur analyse trend pour {symbol}: {e}")
            return {"trend": "unknown", "error": str(e)}

    def get_momentum_analysis(
        self, prices: Union[List[float], Any], symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive momentum analysis with cached indicators.

        Returns:
            Dictionary with momentum indicators and signals
        """
        symbol = symbol or self.default_symbol
        if not symbol:
            raise ValueError("Symbol required for cached calculations")

        try:
            # RSI
            rsi14 = self.rsi(prices, symbol, 14)

            # Stochastic RSI
            stoch_rsi = self.stochastic_rsi(prices, symbol)

            # MACD momentum
            macd = self.macd(prices)

            # Momentum classification
            momentum = "neutral"
            if rsi14:
                if rsi14 > 70:
                    momentum = "overbought"
                elif rsi14 < 30:
                    momentum = "oversold"
                elif rsi14 > 50:
                    momentum = "bullish"
                else:
                    momentum = "bearish"

            return {
                "momentum": momentum,
                "rsi14": rsi14,
                "stoch_rsi_k": stoch_rsi.get("k") if stoch_rsi else None,
                "stoch_rsi_d": stoch_rsi.get("d") if stoch_rsi else None,
                "macd_momentum": (
                    "bullish"
                    if (
                        macd["macd_line"]
                        and macd["macd_signal"]
                        and macd["macd_line"][-1]
                        and macd["macd_signal"][-1]
                        and macd["macd_line"][-1] > macd["macd_signal"][-1]
                    )
                    else "bearish"
                ),
                "symbol": symbol,
            }

        except Exception as e:
            logger.error(f"Erreur analyse momentum pour {symbol}: {e}")
            return {"momentum": "unknown", "error": str(e)}

    # ============ CACHE MANAGEMENT ============

    def clear_cache(self, symbol: Optional[str] = None):
        """Clear all cached indicators for a symbol."""
        symbol = symbol or self.default_symbol
        if not symbol:
            raise ValueError("Symbol required for cache operations")

        if self._cache:
            clear_rsi_cache(symbol)
            clear_ma_cache(symbol)
            logger.info(f"Cache effacé pour {symbol}")

    def get_cache_stats(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get cache statistics for a symbol."""
        symbol = symbol or self.default_symbol
        if not symbol:
            raise ValueError("Symbol required for cache operations")

        if not self._cache:
            return {"cache_enabled": False}

        try:
            from .momentum.rsi import get_rsi_cache_stats
            from .trend.moving_averages import get_ma_cache_stats

            rsi_stats = get_rsi_cache_stats(symbol)
            ma_stats = get_ma_cache_stats(symbol)
            global_stats = self._cache.get_statistics()

            return {
                "cache_enabled": True,
                "symbol": symbol,
                "rsi_cache": rsi_stats,
                "ma_cache": ma_stats,
                "global_stats": global_stats,
                "memory_usage": self._cache.get_memory_usage(),
            }

        except Exception as e:
            logger.error(f"Erreur stats cache pour {symbol}: {e}")
            return {"cache_enabled": True, "error": str(e)}

    def warm_cache(self, prices: Union[List[float], Any], symbol: Optional[str] = None):
        """Warm up cache with common indicators."""
        symbol = symbol or self.default_symbol
        if not symbol:
            raise ValueError("Symbol required for cache operations")

        if not self.enable_cache:
            logger.info("Cache désactivé - warm up ignoré")
            return

        logger.info(f"Réchauffage cache pour {symbol}...")

        try:
            # Pre-calculate common indicators
            common_periods = {"rsi": [14, 21], "ema": [7, 26, 99], "sma": [20, 50]}

            for period in common_periods["rsi"]:
                self.rsi(prices, symbol, period)

            for period in common_periods["ema"]:
                self.ema(prices, symbol, period)

            for period in common_periods["sma"]:
                self.sma(prices, symbol, period)

            # Pre-calculate cross signals
            self.ma_cross_signal(prices, symbol, 7, 26, "ema")

            logger.info(f"Cache réchauffé pour {symbol}")

        except Exception as e:
            logger.error(f"Erreur réchauffage cache pour {symbol}: {e}")


# ============ GLOBAL FACTORY INSTANCE ============

# Create a global instance for convenient access
cached_indicators = CachedIndicatorFactory(enable_cache=True)


def get_cached_indicators(
    symbol: Optional[str] = None, enable_cache: bool = True
) -> CachedIndicatorFactory:
    """
    Get a cached indicator factory instance.

    Args:
        symbol: Default symbol for this factory
        enable_cache: Whether to enable caching

    Returns:
        CachedIndicatorFactory instance
    """
    return CachedIndicatorFactory(enable_cache=enable_cache, default_symbol=symbol)


def quick_analysis(prices: Union[List[float], Any], symbol: str) -> Dict[str, Any]:
    """
    Quick technical analysis with cached indicators.

    Args:
        prices: Price series
        symbol: Trading symbol

    Returns:
        Dictionary with trend and momentum analysis
    """
    factory = get_cached_indicators(symbol)

    trend_analysis = factory.get_trend_analysis(prices)
    momentum_analysis = factory.get_momentum_analysis(prices)

    return {
        "symbol": symbol,
        "trend": trend_analysis,
        "momentum": momentum_analysis,
        "timestamp": None,  # Will be added by calling service
    }
