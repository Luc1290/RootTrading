"""
Williams %R Indicator

This module provides Williams %R calculation for momentum analysis.
Williams %R is a momentum oscillator that measures overbought and oversold levels.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.debug("TA-Lib not available, using manual calculations")


def calculate_williams_r(
    highs: list[float] | np.ndarray | pd.Series,
    lows: list[float] | np.ndarray | pd.Series,
    closes: list[float] | np.ndarray | pd.Series,
    period: int = 14,
) -> float | None:
    """
    Calculate Williams %R.

    Williams %R is a momentum indicator that measures overbought and oversold levels.
    It's similar to the Stochastic oscillator but with an inverted scale.

    Formula: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: Lookback period (default: 14)

    Returns:
        Williams %R value or None if insufficient data

    Notes:
        - Values range from -100 to 0
        - %R > -20: Overbought conditions
        - %R < -80: Oversold conditions
        - Similar to Stochastic but inverted scale
    """
    highs_array = _to_numpy_array(highs)
    lows_array = _to_numpy_array(lows)
    closes_array = _to_numpy_array(closes)

    if len(highs_array) < period:
        return None

    # Ensure all arrays have same length
    min_len = min(len(highs_array), len(lows_array), len(closes_array))
    highs_array = highs_array[-min_len:]
    lows_array = lows_array[-min_len:]
    closes_array = closes_array[-min_len:]

    if TALIB_AVAILABLE:
        try:
            williams_r = talib.WILLR(
                highs_array, lows_array, closes_array, timeperiod=period
            )
            return float(
                williams_r[-1]) if not np.isnan(williams_r[-1]) else None
        except Exception as e:
            logger.warning(f"TA-Lib Williams %R error: {e}, using fallback")

    return _calculate_williams_r_manual(
        highs_array, lows_array, closes_array, period)


def calculate_williams_r_series(
    highs: list[float] | np.ndarray | pd.Series,
    lows: list[float] | np.ndarray | pd.Series,
    closes: list[float] | np.ndarray | pd.Series,
    period: int = 14,
) -> list[float | None]:
    """
    Calculate Williams %R for entire price series.

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: Lookback period

    Returns:
        List of Williams %R values (None for insufficient data points)
    """
    highs_array = _to_numpy_array(highs)
    lows_array = _to_numpy_array(lows)
    closes_array = _to_numpy_array(closes)

    # Ensure all arrays have same length
    min_len = min(len(highs_array), len(lows_array), len(closes_array))
    highs_array = highs_array[-min_len:]
    lows_array = lows_array[-min_len:]
    closes_array = closes_array[-min_len:]

    if TALIB_AVAILABLE:
        try:
            williams_r = talib.WILLR(
                highs_array, lows_array, closes_array, timeperiod=period
            )
            return [float(val) if not np.isnan(
                val) else None for val in williams_r]
        except Exception as e:
            logger.warning(
                f"TA-Lib Williams %R series error: {e}, using fallback")

    # Manual calculation
    return _calculate_williams_r_series_manual(
        highs_array, lows_array, closes_array, period
    )


def williams_r_signal(
    current_value: float | None, previous_value: float | None = None
) -> str:
    """
    Generate trading signal based on Williams %R levels.

    Args:
        current_value: Current Williams %R value
        previous_value: Previous Williams %R value (optional)

    Returns:
        'overbought', 'oversold', 'bullish_reversal', 'bearish_reversal', or 'neutral'
    """
    if current_value is None:
        return "neutral"

    # Basic overbought/oversold levels
    if current_value >= -20:
        return "overbought"
    if current_value <= -80:
        return "oversold"

    # Check for reversal signals if previous value is provided
    if previous_value is not None:
        # Bullish reversal: Moving up from oversold
        if previous_value <= -80 and current_value > -80:
            return "bullish_reversal"

        # Bearish reversal: Moving down from overbought
        if previous_value >= -20 and current_value < -20:
            return "bearish_reversal"

    return "neutral"


def calculate_williams_r_smooth(
    highs: list[float] | np.ndarray | pd.Series,
    lows: list[float] | np.ndarray | pd.Series,
    closes: list[float] | np.ndarray | pd.Series,
    period: int = 14,
    smooth_period: int = 3,
) -> float | None:
    """
    Calculate smoothed Williams %R.

    Applies simple moving average smoothing to reduce noise.

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: Williams %R period
        smooth_period: Smoothing period

    Returns:
        Smoothed Williams %R value or None
    """
    williams_series = calculate_williams_r_series(highs, lows, closes, period)

    # Remove None values and get recent values
    valid_values = [x for x in williams_series if x is not None]

    if len(valid_values) < smooth_period:
        return None

    # Calculate simple moving average
    recent_values = valid_values[-smooth_period:]
    smoothed_value = np.mean(recent_values)

    return float(smoothed_value)


def calculate_williams_r_bands(
    highs: list[float] | np.ndarray | pd.Series,
    lows: list[float] | np.ndarray | pd.Series,
    closes: list[float] | np.ndarray | pd.Series,
    period: int = 14,
    overbought_level: float = -20,
    oversold_level: float = -80,
) -> dict:
    """
    Calculate Williams %R with custom overbought/oversold levels.

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: Williams %R period
        overbought_level: Custom overbought threshold
        oversold_level: Custom oversold threshold

    Returns:
        Dictionary with williams_r, overbought_level, oversold_level, and signal
    """
    williams_r = calculate_williams_r(highs, lows, closes, period)

    signal = "neutral"
    if williams_r is not None:
        if williams_r >= overbought_level:
            signal = "overbought"
        elif williams_r <= oversold_level:
            signal = "oversold"

    return {
        "williams_r": williams_r,
        "overbought_level": overbought_level,
        "oversold_level": oversold_level,
        "signal": signal,
    }


def calculate_williams_r_divergence(
    prices: list[float] | np.ndarray | pd.Series,
    highs: list[float] | np.ndarray | pd.Series,
    lows: list[float] | np.ndarray | pd.Series,
    lookback: int = 20,
) -> str:
    """
    Detect bullish/bearish divergence in Williams %R.

    Args:
        prices: Close prices
        highs: High prices
        lows: Low prices
        lookback: Period to check for divergence

    Returns:
        'bullish_divergence', 'bearish_divergence', or 'none'
    """
    if len(prices) < lookback + 14:  # Need minimum data
        return "none"

    # Calculate Williams %R series
    williams_series = calculate_williams_r_series(highs, lows, prices)

    # Remove None values and get recent data
    valid_data = []
    for i in range(len(williams_series)):
        if williams_series[i] is not None:
            valid_data.append((i, float(prices[i]), williams_series[i]))

    if len(valid_data) < lookback:
        return "none"

    recent_data = valid_data[-lookback:]

    # Find local highs and lows in both price and Williams %R
    price_highs = []
    price_lows = []
    williams_highs = []
    williams_lows = []

    for i in range(1, len(recent_data) - 1):
        idx, price, williams = recent_data[i]
        prev_price = recent_data[i - 1][1]
        next_price = recent_data[i + 1][1]
        prev_williams = recent_data[i - 1][2]
        next_williams = recent_data[i + 1][2]

        # Price peaks and troughs
        if price > prev_price and price > next_price:
            price_highs.append((idx, price))
        elif price < prev_price and price < next_price:
            price_lows.append((idx, price))

        # Williams %R peaks and troughs (with None checks)
        if (
            williams is not None
            and prev_williams is not None
            and next_williams is not None
            and williams > prev_williams
            and williams > next_williams
        ):
            williams_highs.append((idx, williams))
        elif (
            williams is not None
            and prev_williams is not None
            and next_williams is not None
            and williams < prev_williams
            and williams < next_williams
        ):
            williams_lows.append((idx, williams))

    # Check for divergence
    # Bullish divergence: Price makes lower lows, Williams %R makes higher lows
    if len(price_lows) >= 2 and len(williams_lows) >= 2:
        price_val1, price_val2 = price_lows[-1][1], price_lows[-2][1]
        williams_val1, williams_val2 = williams_lows[-1][1], williams_lows[-2][1]
        if (
            price_val1 is not None
            and price_val2 is not None
            and williams_val1 is not None
            and williams_val2 is not None
            and price_val1 < price_val2
            and williams_val1 > williams_val2
        ):
            return "bullish_divergence"

    # Bearish divergence: Price makes higher highs, Williams %R makes lower
    # highs
    if len(price_highs) >= 2 and len(williams_highs) >= 2:
        price_val1, price_val2 = price_highs[-1][1], price_highs[-2][1]
        williams_val1, williams_val2 = williams_highs[-1][1], williams_highs[-2][1]
        if (
            price_val1 is not None
            and price_val2 is not None
            and williams_val1 is not None
            and williams_val2 is not None
            and price_val1 > price_val2
            and williams_val1 < williams_val2
        ):
            return "bearish_divergence"

    return "none"


def williams_r_trend_strength(
        values: list[float | None], period: int = 10) -> str:
    """
    Assess trend strength based on Williams %R persistence.

    Args:
        values: Recent Williams %R values
        period: Period to analyze

    Returns:
        'strong_uptrend', 'moderate_uptrend', 'strong_downtrend',
        'moderate_downtrend', or 'sideways'
    """
    if len(values) < period:
        return "sideways"

    recent_values = [x for x in values[-period:] if x is not None]
    if len(recent_values) < period // 2:
        return "sideways"

    # Count values in different zones
    overbought_count = sum(1 for x in recent_values if x >= -20)
    oversold_count = sum(1 for x in recent_values if x <= -80)
    len(recent_values) - overbought_count - oversold_count

    total = len(recent_values)
    result = "sideways"

    # Strong trends: > 70% in one zone
    if overbought_count / total > 0.7:
        result = "strong_uptrend"
    elif oversold_count / total > 0.7:
        result = "strong_downtrend"
    # Moderate trends: > 50% in one zone
    elif overbought_count / total > 0.5:
        result = "moderate_uptrend"
    elif oversold_count / total > 0.5:
        result = "moderate_downtrend"

    return result


# ============ Helper Functions ============


def _to_numpy_array(data: list[float] | np.ndarray | pd.Series) -> np.ndarray:
    """Convert input data to numpy array."""
    if isinstance(data, pd.Series):
        return np.asarray(data.values, dtype=float)
    if isinstance(data, list):
        return np.array(data, dtype=float)
    return np.asarray(data, dtype=float)


def _calculate_williams_r_manual(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int
) -> float | None:
    """Manual Williams %R calculation."""
    if len(highs) < period:
        return None

    # Get the lookback window
    high_window = highs[-period:]
    low_window = lows[-period:]
    current_close = closes[-1]

    # Calculate highest high and lowest low
    highest_high = np.max(high_window)
    lowest_low = np.min(low_window)

    # Calculate Williams %R
    if highest_high == lowest_low:
        return 0.0  # No range, neutral

    williams_r = ((highest_high - current_close) /
                  (highest_high - lowest_low)) * -100

    return float(williams_r)


def _calculate_williams_r_series_manual(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int
) -> list[float | None]:
    """Manual Williams %R series calculation."""
    williams_series: list[float | None] = []

    for i in range(len(highs)):
        if i < period - 1:
            williams_series.append(None)
        else:
            # Get window for calculation
            high_window = highs[i - period + 1: i + 1]
            low_window = lows[i - period + 1: i + 1]
            current_close = closes[i]

            # Calculate highest high and lowest low
            highest_high = np.max(high_window)
            lowest_low = np.min(low_window)

            # Calculate Williams %R
            if highest_high == lowest_low:
                williams_r = 0.0  # No range
            else:
                williams_r = ((highest_high - current_close) /
                              (highest_high - lowest_low)) * -100

            williams_series.append(float(williams_r))

    return williams_series
