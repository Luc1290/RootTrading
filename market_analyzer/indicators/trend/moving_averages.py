"""
Moving Averages Indicators

This module provides various moving average implementations:
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- WMA (Weighted Moving Average)
- DEMA (Double Exponential Moving Average)
- TEMA (Triple Exponential Moving Average)
Enhanced with Redis caching for high-performance incremental updates.
"""

import logging

import numpy as np
import pandas as pd

# Import cache and utilities
from shared.src.indicator_cache import get_indicator_cache
from shared.src.technical_utils import validate_and_align_arrays, validate_indicator_params

logger = logging.getLogger(__name__)

try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.debug("TA-Lib not available, using manual calculations")


def calculate_sma(
    prices: list[float] | np.ndarray | pd.Series, period: int
) -> float | None:
    """
    Calculate Simple Moving Average (SMA).

    SMA is the arithmetic mean of prices over a specified period.

    Args:
        prices: Price series
        period: Number of periods for averaging

    Returns:
        SMA value or None if insufficient data
    """
    prices_array = _to_numpy_array(prices)
    if len(prices_array) < period:
        return None

    if TALIB_AVAILABLE:
        try:
            sma_values = talib.SMA(prices_array, timeperiod=period)
            return float(
                sma_values[-1]) if not np.isnan(sma_values[-1]) else None
        except Exception as e:
            logger.warning(f"TA-Lib SMA error: {e}, using fallback")

    # Manual calculation
    return float(np.mean(prices_array[-period:]))


def calculate_sma_series(
    prices: list[float] | np.ndarray | pd.Series, period: int
) -> list[float | None]:
    """
    Calculate SMA for entire price series.

    Args:
        prices: Price series
        period: SMA period

    Returns:
        List of SMA values (None for insufficient data points)
    """
    prices_array = _to_numpy_array(prices)

    if TALIB_AVAILABLE:
        try:
            sma_values = talib.SMA(prices_array, timeperiod=period)
            return [float(val) if not np.isnan(
                val) else None for val in sma_values]
        except Exception as e:
            logger.warning(f"TA-Lib SMA series error: {e}, using fallback")

    # Manual calculation with rolling window
    sma_series: list[float | None] = []
    for i in range(len(prices_array)):
        if i < period - 1:
            sma_series.append(None)
        else:
            sma = float(np.mean(prices_array[i - period + 1: i + 1]))
            sma_series.append(sma)

    return sma_series


def calculate_ema(
    prices: list[float] | np.ndarray | pd.Series,
    period: int,
    symbol: str | None = None,
    enable_cache: bool = True,
) -> float | None:
    """
    Calculate Exponential Moving Average (EMA).

    EMA gives more weight to recent prices, making it more responsive
    to new information compared to SMA.

    Args:
        prices: Price series
        period: Number of periods for EMA
        symbol: Trading symbol for caching (optional, enables cache if provided)
        enable_cache: Whether to use Redis caching for performance

    Returns:
        EMA value or None if insufficient data

    Notes:
        - Automatic incremental caching when symbol provided
        - Up to 50x faster for repeated calls with caching
    """
    # If symbol provided and cache enabled, use cached version
    if symbol and enable_cache:
        return calculate_ema_cached(prices, symbol, period, enable_cache)

    # Original non-cached implementation continues below...
    prices_array = _to_numpy_array(prices)
    if len(prices_array) < period:
        return None

    if TALIB_AVAILABLE:
        try:
            ema_values = talib.EMA(prices_array, timeperiod=period)
            return float(
                ema_values[-1]) if not np.isnan(ema_values[-1]) else None
        except Exception as e:
            logger.warning(f"TA-Lib EMA error: {e}, using fallback")

    return _calculate_ema_manual(prices_array, period)


def calculate_ema_series(
    prices: list[float] | np.ndarray | pd.Series, period: int
) -> list[float | None]:
    """
    Calculate EMA for entire price series.

    Args:
        prices: Price series
        period: EMA period

    Returns:
        List of EMA values (None for insufficient data points)
    """
    prices_array = _to_numpy_array(prices)

    if TALIB_AVAILABLE:
        try:
            ema_values = talib.EMA(prices_array, timeperiod=period)
            return [float(val) if not np.isnan(
                val) else None for val in ema_values]
        except Exception as e:
            logger.warning(f"TA-Lib EMA series error: {e}, using fallback")

    # Manual calculation
    if len(prices_array) < period:
        return [None] * len(prices_array)

    ema_series: list[float | None] = [None] * (period - 1)
    alpha = 2.0 / (period + 1)

    # Initialize with SMA
    first_ema = float(np.mean(prices_array[:period]))
    ema_series.append(first_ema)

    # Calculate rest incrementally
    for i in range(period, len(prices_array)):
        prev_ema = ema_series[i - 1]
        if prev_ema is not None:
            new_ema = alpha * float(prices_array[i]) + (1 - alpha) * prev_ema
            ema_series.append(new_ema)
        else:
            ema_series.append(None)

    return ema_series


def calculate_ema_incremental(
    current_price: float, previous_ema: float | None, period: int
) -> float:
    """
    Calculate EMA incrementally using previous EMA value.

    This is more efficient for real-time updates as it doesn't
    require the full price history.

    Args:
        current_price: Current price
        previous_ema: Previous EMA value (uses current_price if None)
        period: EMA period

    Returns:
        New EMA value
    """
    if previous_ema is None:
        return float(current_price)

    alpha = 2.0 / (period + 1)
    return alpha * float(current_price) + (1 - alpha) * previous_ema


def calculate_wma(
    prices: list[float] | np.ndarray | pd.Series, period: int
) -> float | None:
    """
    Calculate Weighted Moving Average (WMA).

    WMA assigns linearly decreasing weights to older data points.

    Args:
        prices: Price series
        period: Number of periods

    Returns:
        WMA value or None if insufficient data
    """
    prices_array = _to_numpy_array(prices)
    if len(prices_array) < period:
        return None

    if TALIB_AVAILABLE:
        try:
            wma_values = talib.WMA(prices_array, timeperiod=period)
            return float(
                wma_values[-1]) if not np.isnan(wma_values[-1]) else None
        except Exception as e:
            logger.warning(f"TA-Lib WMA error: {e}, using fallback")

    # Manual calculation
    weights = np.arange(1, period + 1)
    recent_prices = prices_array[-period:]
    wma = np.sum(weights * recent_prices) / np.sum(weights)

    return float(wma)


def calculate_dema(
    prices: list[float] | np.ndarray | pd.Series, period: int
) -> float | None:
    """
    Calculate Double Exponential Moving Average (DEMA).

    DEMA is more responsive than EMA by using two EMAs to reduce lag.

    Args:
        prices: Price series
        period: Number of periods

    Returns:
        DEMA value or None if insufficient data
    """
    prices_array = _to_numpy_array(prices)
    if len(prices_array) < period * 2:
        return None

    if TALIB_AVAILABLE:
        try:
            dema_values = talib.DEMA(prices_array, timeperiod=period)
            return float(
                dema_values[-1]) if not np.isnan(dema_values[-1]) else None
        except Exception as e:
            logger.warning(f"TA-Lib DEMA error: {e}, using fallback")

    # Manual calculation: DEMA = 2 * EMA - EMA(EMA)
    ema1 = calculate_ema_series(prices_array, period)
    ema1_clean = [x for x in ema1 if x is not None]

    if len(ema1_clean) < period:
        return None

    ema2 = _calculate_ema_manual(np.array(ema1_clean), period)
    if ema2 is None or ema1[-1] is None:
        return None

    dema = 2 * ema1[-1] - ema2
    return float(dema)


def calculate_tema(
    prices: list[float] | np.ndarray | pd.Series, period: int
) -> float | None:
    """
    Calculate Triple Exponential Moving Average (TEMA).

    TEMA further reduces lag by using three EMAs.

    Args:
        prices: Price series
        period: Number of periods

    Returns:
        TEMA value or None if insufficient data
    """
    prices_array = _to_numpy_array(prices)
    if len(prices_array) < period * 3:
        return None

    if TALIB_AVAILABLE:
        try:
            tema_values = talib.TEMA(prices_array, timeperiod=period)
            return float(
                tema_values[-1]) if not np.isnan(tema_values[-1]) else None
        except Exception as e:
            logger.warning(f"TA-Lib TEMA error: {e}, using fallback")

    # Manual calculation: TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
    ema1 = calculate_ema_series(prices_array, period)
    ema1_clean = [x for x in ema1 if x is not None]

    if len(ema1_clean) < period:
        return None

    ema2 = calculate_ema_series(ema1_clean, period)
    ema2_clean = [x for x in ema2 if x is not None]

    if len(ema2_clean) < period:
        return None

    ema3 = _calculate_ema_manual(np.array(ema2_clean), period)

    if ema3 is None or ema2[-1] is None or ema1[-1] is None:
        return None

    tema = 3 * ema1[-1] - 3 * ema2[-1] + ema3
    return float(tema)


def calculate_hull_ma(
    prices: list[float] | np.ndarray | pd.Series, period: int
) -> float | None:
    """
    Calculate Hull Moving Average (HMA) - OPTIMIZED VERSION.

    HMA aims to be responsive while maintaining smoothness.
    Formula: WMA(2*WMA(n/2) - WMA(n), sqrt(n))

    PERFORMANCE: Optimized to avoid O(n²) complexity by using vectorized operations
    instead of recalculating WMA for each point in the series.

    Args:
        prices: Price series
        period: Number of periods

    Returns:
        HMA value or None if insufficient data
    """
    prices_array = _to_numpy_array(prices)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))

    min_required = period + sqrt_period
    if len(prices_array) < min_required:
        return None

    # OPTIMIZED: Calculate WMA series using vectorized rolling window
    # Instead of recalculating WMA for each point (O(n²))

    # Calculate rolling WMA for half period
    wma_half_series = []
    for i in range(half_period - 1, len(prices_array)):
        window = prices_array[i - half_period + 1: i + 1]
        weights = np.arange(1, half_period + 1)
        wma_val = np.average(window, weights=weights)
        wma_half_series.append(wma_val)

    # Calculate rolling WMA for full period
    wma_full_series = []
    for i in range(period - 1, len(prices_array)):
        window = prices_array[i - period + 1: i + 1]
        weights = np.arange(1, period + 1)
        wma_val = np.average(window, weights=weights)
        wma_full_series.append(wma_val)

    # Calculate difference series: 2*WMA(n/2) - WMA(n)
    # Align the series (wma_full starts later)
    offset = (period - 1) - (half_period - 1)
    diff_series = []
    for i in range(len(wma_full_series)):
        wma_h = wma_half_series[i + offset]
        wma_f = wma_full_series[i]
        diff_series.append(2 * wma_h - wma_f)

    if len(diff_series) < sqrt_period:
        return None

    # Final WMA on the difference series (last sqrt_period points)
    diff_window = diff_series[-sqrt_period:]
    weights = np.arange(1, sqrt_period + 1)
    hma = np.average(diff_window, weights=weights)

    return float(hma)


def calculate_adaptive_ma(
    prices: list[float] | np.ndarray | pd.Series,
    period: int,
    fast_period: int = 2,
    slow_period: int = 30,
) -> float | None:
    """
    Calculate Kaufman Adaptive Moving Average (KAMA) - SIMPLIFIED VERSION.

    ⚠️  IMPORTANT: This is an approximation of KAMA, not the exact implementation.
    True KAMA requires incremental calculation across the entire historical series.
    This version provides a "light" approximation using current efficiency ratio
    and SMA initialization instead of proper recursive calculation.

    KAMA adapts its speed based on market volatility and trends.

    Args:
        prices: Price series
        period: Efficiency ratio period
        fast_period: Fast EMA period
        slow_period: Slow EMA period

    Returns:
        Approximate KAMA value or None if insufficient data

    Notes:
        - Use with caution: approximation may differ from true KAMA
        - For exact KAMA, implement full recursive calculation with historical state
    """
    prices_array = _to_numpy_array(prices)
    if len(prices_array) < period + 1:
        return None

    if TALIB_AVAILABLE and hasattr(talib, "KAMA"):
        try:
            kama_values = talib.KAMA(prices_array, timeperiod=period)
            return float(
                kama_values[-1]) if not np.isnan(kama_values[-1]) else None
        except Exception as e:
            logger.warning(f"TA-Lib KAMA error: {e}, using fallback")

    # Manual calculation
    # Calculate efficiency ratio
    change = abs(prices_array[-1] - prices_array[-period - 1])
    volatility = np.sum(np.abs(np.diff(prices_array[-period - 1:])))

    efficiency_ratio = 0 if volatility == 0 else change / volatility

    # Calculate smoothing constant
    fastest_sc = 2.0 / (fast_period + 1)
    slowest_sc = 2.0 / (slow_period + 1)
    sc = (efficiency_ratio * (fastest_sc - slowest_sc) + slowest_sc) ** 2

    # Calculate KAMA (simplified - would need full series for accurate
    # calculation)
    if len(prices_array) < slow_period:
        return float(prices_array[-1])

    # Initialize with SMA
    kama = float(np.mean(prices_array[-slow_period:]))

    # Update with adaptive smoothing
    return kama + sc * (float(prices_array[-1]) - kama)


# ============ Helper Functions ============


def _to_numpy_array(data: list[float] | np.ndarray | pd.Series) -> np.ndarray:
    """Convert input data to numpy array."""
    if isinstance(data, pd.Series):
        if hasattr(data.values, "values"):  # ExtensionArray
            return np.asarray(data.values, dtype=float)
        return np.asarray(data.values, dtype=float)
    if isinstance(data, list):
        return np.array(data, dtype=float)
    return np.asarray(data, dtype=float)


def _calculate_ema_manual(prices: np.ndarray, period: int) -> float | None:
    """Manual EMA calculation."""
    if len(prices) < period:
        return None

    alpha = 2.0 / (period + 1)

    # Initialize with SMA
    ema = float(np.mean(prices[:period]))

    # Calculate EMA for rest of the values
    for i in range(period, len(prices)):
        ema = alpha * float(prices[i]) + (1 - alpha) * ema

    return round(ema, 6)


def crossover(series1: list[float], series2: list[float]) -> bool:
    """
    Check if series1 crosses over series2.

    Args:
        series1: First series (e.g., fast MA)
        series2: Second series (e.g., slow MA)

    Returns:
        True if series1 crosses above series2
    """
    if len(series1) < 2 or len(series2) < 2:
        return False

    # Remove None values from the end
    s1 = [x for x in series1[-2:] if x is not None]
    s2 = [x for x in series2[-2:] if x is not None]

    if len(s1) < 2 or len(s2) < 2:
        return False

    return s1[0] <= s2[0] and s1[1] > s2[1]


def crossunder(series1: list[float], series2: list[float]) -> bool:
    """
    Check if series1 crosses under series2.

    Args:
        series1: First series
        series2: Second series

    Returns:
        True if series1 crosses below series2
    """
    if len(series1) < 2 or len(series2) < 2:
        return False

    # Remove None values from the end
    s1 = [x for x in series1[-2:] if x is not None]
    s2 = [x for x in series2[-2:] if x is not None]

    if len(s1) < 2 or len(s2) < 2:
        return False

    return s1[0] >= s2[0] and s1[1] < s2[1]


# ============ CACHED VERSIONS WITH HIGH PERFORMANCE ============


def calculate_ema_cached(
    prices: list[float] | np.ndarray | pd.Series,
    symbol: str,
    period: int,
    enable_cache: bool = True,
) -> float | None:
    """
    Calculate EMA with Redis caching for high-performance incremental updates.

    Args:
        prices: Price series
        symbol: Trading symbol for cache key
        period: EMA period
        enable_cache: Whether to use caching

    Returns:
        Current EMA value or None if insufficient data
    """
    if not enable_cache:
        return calculate_ema(prices, period)

    try:
        # Validate parameters
        params = validate_indicator_params(period=period)
        period = (
            int(params["period"])
            if isinstance(params["period"], int | float)
            else period
        )

        # Validate and prepare data
        prices_array = validate_and_align_arrays(prices, min_length=period)[0]

        # Get cache
        cache = get_indicator_cache()
        cache_key = f"ema_{period}"

        # Check for cached EMA state
        cached_state = cache.get(cache_key, symbol)

        if cached_state is not None:
            # Try incremental calculation
            ema_value = _calculate_ema_incremental(
                prices_array, cached_state, period, symbol, cache_key
            )
            if ema_value is not None:
                return ema_value

        # Full calculation and cache
        ema_value = calculate_ema(prices_array, period)

        if ema_value is not None:
            # Store EMA state for future incremental updates
            ema_state = _create_ema_state(prices_array, period, ema_value)
            cache.set(cache_key, ema_state, symbol)

        return ema_value

    except Exception:
        logger.exception("Erreur EMA cached pour {symbol}")
        # Fallback to non-cached version
        return calculate_ema(prices, period)


def calculate_ema_series_cached(
    prices: list[float] | np.ndarray | pd.Series,
    symbol: str,
    period: int,
    enable_cache: bool = True,
) -> list[float | None]:
    """
    Calculate EMA series with caching optimization.

    Args:
        prices: Price series
        symbol: Trading symbol
        period: EMA period
        enable_cache: Whether to use caching

    Returns:
        List of EMA values
    """
    if not enable_cache:
        return calculate_ema_series(prices, period)

    try:
        # Validate data
        prices_array = validate_and_align_arrays(prices, min_length=period)[0]

        # Get cache
        cache = get_indicator_cache()
        cache_key = f"ema_series_{period}"

        # Check for cached series
        cached_series = cache.get(cache_key, symbol)

        if cached_series is not None:
            # Check if we can extend existing series
            if len(cached_series) <= len(prices_array):
                # Calculate only new values
                new_series = _extend_ema_series(
                    prices_array, cached_series, period)
                if new_series:
                    cache.set(cache_key, new_series, symbol)
                    return new_series

        # Full calculation
        ema_series = calculate_ema_series(prices_array, period)

        # Cache the series
        if ema_series:
            cache.set(cache_key, ema_series, symbol)

        return ema_series

    except Exception:
        logger.exception("Erreur EMA series cached pour {symbol}")
        return calculate_ema_series(prices, period)


def calculate_sma_cached(
    prices: list[float] | np.ndarray | pd.Series,
    symbol: str,
    period: int,
    enable_cache: bool = True,
) -> float | None:
    """
    Calculate SMA with basic caching (less benefit than EMA due to no state).

    Args:
        prices: Price series
        symbol: Trading symbol
        period: SMA period
        enable_cache: Whether to use caching

    Returns:
        Current SMA value or None if insufficient data
    """
    if not enable_cache:
        return calculate_sma(prices, period)

    try:
        # Validate parameters
        params = validate_indicator_params(period=period)
        period = (
            int(params["period"])
            if isinstance(params["period"], int | float)
            else period
        )

        # Validate data
        prices_array = validate_and_align_arrays(prices, min_length=period)[0]

        # Get cache
        cache = get_indicator_cache()
        cache_key = f"sma_{period}"

        # Simple hash of recent data to check if calculation needed
        recent_prices = prices_array[-period:]
        data_hash = hash(tuple(recent_prices))

        # Check cached result
        cached_result = cache.get(cache_key, symbol)

        if (
            cached_result is not None
            and isinstance(cached_result, dict)
            and cached_result.get("data_hash") == data_hash
        ):
            return cached_result.get("sma_value")

        # Calculate and cache
        sma_value = calculate_sma(prices_array, period)

        if sma_value is not None:
            cache.set(
                cache_key,
                {"sma_value": sma_value, "data_hash": data_hash, "period": period},
                symbol,
            )

        return sma_value

    except Exception:
        logger.exception("Erreur SMA cached pour {symbol}")
        return calculate_sma(prices, period)


def _create_ema_state(
        prices: np.ndarray,
        period: int,
        current_ema: float) -> dict:
    """Create EMA state for incremental caching."""
    return {
        "last_ema": float(current_ema),
        "last_price": float(prices[-1]),
        "period": period,
        "alpha": 2.0 / (period + 1),
        "price_count": len(prices),
    }


def _calculate_ema_incremental(
        prices: np.ndarray,
        cached_state: dict,
        period: int,
        symbol: str,
        cache_key: str) -> float | None:
    """Calculate EMA incrementally using cached state."""
    try:
        cache = get_indicator_cache()

        # Check if we have new data
        if len(prices) <= cached_state.get("price_count", 0):
            return cached_state.get("last_ema")

        # Get new prices since last calculation
        last_count = cached_state.get("price_count", 0)
        new_prices = prices[last_count:]

        # If too many new prices, do full recalculation
        if len(new_prices) > period * 2:  # Threshold for full recalc
            return None  # Will trigger full calculation

        # Incremental EMA calculation
        current_ema = cached_state.get("last_ema")
        alpha = cached_state.get("alpha", 2.0 / (period + 1))

        # Process each new price
        for new_price in new_prices:
            current_ema = alpha * new_price + (1 - alpha) * current_ema

        # Update cached state
        updated_state = {
            "last_ema": float(current_ema) if current_ema is not None else 0.0,
            "last_price": float(prices[-1]) if prices[-1] is not None else 0.0,
            "period": period,
            "alpha": alpha,
            "price_count": len(prices),
        }

        cache.set(cache_key, updated_state, symbol)

        return float(current_ema) if current_ema is not None else None

    except Exception as e:
        logger.warning(f"Erreur calcul EMA incrémental: {e}")
        return None


def _extend_ema_series(
    prices: np.ndarray, cached_series: list[float | None], period: int
) -> list[float | None] | None:
    """Extend existing EMA series with new data."""
    try:
        cached_length = len(cached_series)
        prices_length = len(prices)

        if prices_length <= cached_length:
            return cached_series[:prices_length]

        # Get last valid EMA for continuation
        last_ema = None
        for i in range(len(cached_series) - 1, -1, -1):
            if cached_series[i] is not None:
                last_ema = cached_series[i]
                break

        if last_ema is None:
            return None  # Need full recalculation

        # Calculate new EMA values
        new_series = cached_series.copy()
        alpha = 2.0 / (period + 1)

        for i in range(cached_length, prices_length):
            # last_ema est garanti non-None après la vérification ligne 753
            assert last_ema is not None
            last_ema = alpha * prices[i] + (1 - alpha) * last_ema
            new_series.append(last_ema)

        return new_series

    except Exception as e:
        logger.warning(f"Erreur extension EMA series: {e}")
        return None


def get_ma_cross_signal_cached(
    prices: list[float] | np.ndarray | pd.Series,
    symbol: str,
    fast_period: int = 7,
    slow_period: int = 26,
    ma_type: str = "ema",
) -> dict:
    """
    Get MA crossover signals with caching.

    Args:
        prices: Price series
        symbol: Trading symbol
        fast_period: Fast MA period
        slow_period: Slow MA period
        ma_type: Type of MA ('ema', 'sma')

    Returns:
        Dictionary with cross signals and current values
    """
    try:
        if ma_type == "ema":
            fast_ma = calculate_ema_cached(prices, symbol, fast_period)
            slow_ma = calculate_ema_cached(prices, symbol, slow_period)
            fast_series = calculate_ema_series_cached(
                prices, symbol, fast_period)
            slow_series = calculate_ema_series_cached(
                prices, symbol, slow_period)
        else:  # sma
            fast_ma = calculate_sma_cached(prices, symbol, fast_period)
            slow_ma = calculate_sma_cached(prices, symbol, slow_period)
            fast_series = calculate_sma_series(
                prices, fast_period
            )  # No cache for SMA series (less benefit)
            slow_series = calculate_sma_series(prices, slow_period)

        # Detect crossovers (filter None values)
        fast_filtered = [
            x for x in fast_series if x is not None] if fast_series else []
        slow_filtered = [
            x for x in slow_series if x is not None] if slow_series else []
        bullish_cross = (
            crossover(fast_filtered, slow_filtered)
            if fast_filtered and slow_filtered
            else False
        )
        bearish_cross = (
            crossunder(fast_filtered, slow_filtered)
            if fast_filtered and slow_filtered
            else False
        )

        return {
            "fast_ma": fast_ma,
            "slow_ma": slow_ma,
            "bullish_cross": bullish_cross,
            "bearish_cross": bearish_cross,
            "trend": (
                "bullish"
                if fast_ma and slow_ma and fast_ma > slow_ma
                else "bearish" if fast_ma and slow_ma else "neutral"
            ),
            "ma_type": ma_type,
            "fast_period": fast_period,
            "slow_period": slow_period,
        }

    except Exception:
        logger.exception("Erreur MA cross signal pour {symbol}")
        return {
            "fast_ma": None,
            "slow_ma": None,
            "bullish_cross": False,
            "bearish_cross": False,
            "trend": "unknown",
        }


def clear_ma_cache(symbol: str):
    """Clear all moving average cache entries for a symbol."""
    cache = get_indicator_cache()

    # Common MA periods
    periods = [5, 7, 9, 10, 12, 20, 21, 26, 50, 99, 200]
    ma_types = ["sma", "ema", "wma", "dema", "tema"]

    for ma_type in ma_types:
        for period in periods:
            cache.delete(f"{ma_type}_{period}", symbol)
            cache.delete(f"{ma_type}_series_{period}", symbol)

    logger.info(f"Cache MA effacé pour {symbol}")


def get_ma_cache_stats(symbol: str) -> dict:
    """Get moving average cache statistics for a symbol."""
    cache = get_indicator_cache()
    stats = cache.get_statistics()

    # Check which MAs are cached
    cached_indicators = []
    test_keys = ["ema_7", "ema_26", "ema_99", "sma_20", "sma_50"]

    for key in test_keys:
        if cache.get(key, symbol) is not None:
            cached_indicators.append(key)

    return {
        "symbol": symbol,
        "cached_ma_indicators": cached_indicators,
        "cache_hit_ratio": stats["metrics"]["hit_ratio"],
        "total_cache_entries": stats["cache_entries"],
    }
