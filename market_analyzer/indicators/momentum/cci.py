"""
CCI (Commodity Channel Index) Indicator

This module provides CCI calculation for momentum analysis.
CCI measures the deviation of price from its statistical average.
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


def calculate_cci(
    highs: list[float] | np.ndarray | pd.Series,
    lows: list[float] | np.ndarray | pd.Series,
    closes: list[float] | np.ndarray | pd.Series,
    period: int = 20,
) -> float | None:
    """
    Calculate Commodity Channel Index (CCI).

    CCI measures the deviation of the typical price from its statistical average.
    It identifies cyclical trends in commodities and other markets.

    Formula:
    Typical Price = (High + Low + Close) / 3
    CCI = (Typical Price - SMA of Typical Price) / (0.015 * Mean Deviation)

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: Period for CCI calculation (default: 20)

    Returns:
        CCI value or None if insufficient data

    Notes:
        - CCI > +100: Overbought conditions
        - CCI < -100: Oversold conditions
        - CCI oscillates around zero
        - Values typically range from -200 to +200
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
            cci_values = talib.CCI(
                highs_array, lows_array, closes_array, timeperiod=period
            )
            return float(
                cci_values[-1]) if not np.isnan(cci_values[-1]) else None
        except Exception as e:
            logger.warning(f"TA-Lib CCI error: {e}, using fallback")

    return _calculate_cci_manual(highs_array, lows_array, closes_array, period)


def calculate_cci_series(
    highs: list[float] | np.ndarray | pd.Series,
    lows: list[float] | np.ndarray | pd.Series,
    closes: list[float] | np.ndarray | pd.Series,
    period: int = 20,
) -> list[float | None]:
    """
    Calculate CCI for entire price series.

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: CCI period

    Returns:
        List of CCI values (None for insufficient data points)
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
            cci_values = talib.CCI(
                highs_array, lows_array, closes_array, timeperiod=period
            )
            return [float(val) if not np.isnan(
                val) else None for val in cci_values]
        except Exception as e:
            logger.warning(f"TA-Lib CCI series error: {e}, using fallback")

    # Manual calculation
    return _calculate_cci_series_manual(
        highs_array, lows_array, closes_array, period)


def cci_signal(
    current_value: float | None, previous_value: float | None = None
) -> str:
    """
    Generate trading signal based on CCI levels and movements.

    Args:
        current_value: Current CCI value
        previous_value: Previous CCI value (optional)

    Returns:
        'overbought', 'oversold', 'bullish_cross', 'bearish_cross', or 'neutral'
    """
    if current_value is None:
        return "neutral"

    # Basic overbought/oversold levels
    if current_value > 100:
        return "overbought"
    if current_value < -100:
        return "oversold"

    # Check for zero line crossovers if previous value is provided
    if previous_value is not None:
        # Bullish cross: CCI crosses above zero
        if previous_value <= 0 and current_value > 0:
            return "bullish_cross"

        # Bearish cross: CCI crosses below zero
        if previous_value >= 0 and current_value < 0:
            return "bearish_cross"

    return "neutral"


def calculate_cci_with_bands(
    highs: list[float] | np.ndarray | pd.Series,
    lows: list[float] | np.ndarray | pd.Series,
    closes: list[float] | np.ndarray | pd.Series,
    period: int = 20,
    overbought_level: float = 100,
    oversold_level: float = -100,
) -> dict:
    """
    Calculate CCI with custom overbought/oversold levels.

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: CCI period
        overbought_level: Custom overbought threshold
        oversold_level: Custom oversold threshold

    Returns:
        Dictionary with cci, overbought_level, oversold_level, and signal
    """
    cci = calculate_cci(highs, lows, closes, period)

    signal = "neutral"
    if cci is not None:
        if cci >= overbought_level:
            signal = "overbought"
        elif cci <= oversold_level:
            signal = "oversold"

    return {
        "cci": cci,
        "overbought_level": overbought_level,
        "oversold_level": oversold_level,
        "signal": signal,
    }


def calculate_cci_divergence(
    prices: list[float] | np.ndarray | pd.Series,
    highs: list[float] | np.ndarray | pd.Series,
    lows: list[float] | np.ndarray | pd.Series,
    lookback: int = 20,
) -> str:
    """
    Detect bullish/bearish divergence in CCI.

    Args:
        prices: Close prices
        highs: High prices
        lows: Low prices
        lookback: Period to check for divergence

    Returns:
        'bullish_divergence', 'bearish_divergence', or 'none'
    """
    if len(prices) < lookback + 20:  # Need minimum data
        return "none"

    # Calculate CCI series
    cci_series = calculate_cci_series(highs, lows, prices)

    # Remove None values and get recent data
    valid_data = []
    for i in range(len(cci_series)):
        if cci_series[i] is not None:
            valid_data.append((i, float(prices[i]), cci_series[i]))

    if len(valid_data) < lookback:
        return "none"

    recent_data = valid_data[-lookback:]

    # Find local highs and lows in both price and CCI
    price_highs = []
    price_lows = []
    cci_highs = []
    cci_lows = []

    for i in range(1, len(recent_data) - 1):
        idx, price, cci = recent_data[i]
        prev_price = recent_data[i - 1][1]
        next_price = recent_data[i + 1][1]
        prev_cci = recent_data[i - 1][2]
        next_cci = recent_data[i + 1][2]

        # Price peaks and troughs
        if price > prev_price and price > next_price:
            price_highs.append((idx, price))
        elif price < prev_price and price < next_price:
            price_lows.append((idx, price))

        # CCI peaks and troughs (vérifications None)
        if (
            cci is not None
            and prev_cci is not None
            and next_cci is not None
            and cci > prev_cci
            and cci > next_cci
        ):
            cci_highs.append((idx, cci))
        elif (
            cci is not None
            and prev_cci is not None
            and next_cci is not None
            and cci < prev_cci
            and cci < next_cci
        ):
            cci_lows.append((idx, cci))

    # Check for divergence
    # Bullish divergence: Price makes lower lows, CCI makes higher lows
    if len(price_lows) >= 2 and len(cci_lows) >= 2:
        price_val1, price_val2 = price_lows[-1][1], price_lows[-2][1]
        cci_val1, cci_val2 = cci_lows[-1][1], cci_lows[-2][1]
        if (
            price_val1 is not None
            and price_val2 is not None
            and cci_val1 is not None
            and cci_val2 is not None
            and price_val1 < price_val2
            and cci_val1 > cci_val2
        ):
            return "bullish_divergence"

    # Bearish divergence: Price makes higher highs, CCI makes lower highs
    if len(price_highs) >= 2 and len(cci_highs) >= 2:
        price_val1, price_val2 = price_highs[-1][1], price_highs[-2][1]
        cci_val1, cci_val2 = cci_highs[-1][1], cci_highs[-2][1]
        if (
            price_val1 is not None
            and price_val2 is not None
            and cci_val1 is not None
            and cci_val2 is not None
            and price_val1 > price_val2
            and cci_val1 < cci_val2
        ):
            return "bearish_divergence"

    return "none"


def cci_trend_direction(values: list[float | None], period: int = 10) -> str:
    """
    Determine trend direction based on CCI values.

    Args:
        values: Recent CCI values
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
    strong_bullish = sum(1 for x in recent_values if x > 100)
    moderate_bullish = sum(1 for x in recent_values if 0 < x <= 100)
    moderate_bearish = sum(1 for x in recent_values if -100 <= x < 0)
    strong_bearish = sum(1 for x in recent_values if x < -100)

    total = len(recent_values)
    result = "sideways"

    # Determine trend strength
    if strong_bullish / total > 0.6:
        result = "strong_uptrend"
    elif (strong_bullish + moderate_bullish) / total > 0.7:
        result = "moderate_uptrend"
    elif strong_bearish / total > 0.6:
        result = "strong_downtrend"
    elif (strong_bearish + moderate_bearish) / total > 0.7:
        result = "moderate_downtrend"

    return result


def calculate_cci_volatility(
    highs: list[float] | np.ndarray | pd.Series,
    lows: list[float] | np.ndarray | pd.Series,
    closes: list[float] | np.ndarray | pd.Series,
    period: int = 20,
    volatility_period: int = 14,
) -> float | None:
    """
    Calculate CCI-based volatility measure.

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: CCI period
        volatility_period: Period for volatility calculation

    Returns:
        CCI volatility measure or None
    """
    cci_series = calculate_cci_series(highs, lows, closes, period)
    valid_cci = [x for x in cci_series if x is not None]

    if len(valid_cci) < volatility_period:
        return None

    # Calculate standard deviation of recent CCI values
    recent_cci = valid_cci[-volatility_period:]
    volatility = np.std(recent_cci, ddof=1)

    return float(volatility)


# ============ Helper Functions ============


def _to_numpy_array(data: list[float] | np.ndarray | pd.Series) -> np.ndarray:
    """Convert input data to numpy array."""
    if isinstance(data, pd.Series):
        return np.asarray(data.values, dtype=float)
    if isinstance(data, list):
        return np.array(data, dtype=float)
    return np.asarray(data, dtype=float)


def _calculate_typical_price(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
) -> np.ndarray:
    """Calculate typical price series."""
    return (highs + lows + closes) / 3


def _calculate_mean_deviation(
    typical_prices: np.ndarray, sma_typical: np.ndarray, period: int
) -> np.ndarray:
    """Calculate mean deviation for CCI."""
    mean_deviations = np.zeros_like(typical_prices)

    for i in range(period - 1, len(typical_prices)):
        window = typical_prices[i - period + 1: i + 1]
        sma = sma_typical[i]

        # Calculate mean absolute deviation
        deviations = np.abs(window - sma)
        mean_deviation = np.mean(deviations)
        mean_deviations[i] = mean_deviation

    return mean_deviations


def _calculate_cci_manual(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int
) -> float | None:
    """Manual CCI calculation."""
    if len(highs) < period:
        return None

    # Calculate typical price
    typical_prices = _calculate_typical_price(highs, lows, closes)

    # Calculate SMA of typical price
    sma_typical = typical_prices[-1]  # Current typical price
    if len(typical_prices) >= period:
        sma_typical = np.mean(typical_prices[-period:])

    # Calculate mean deviation
    recent_typical = typical_prices[-period:]
    mean_deviation = np.mean(np.abs(recent_typical - sma_typical))

    if mean_deviation == 0:
        return 0.0

    # Calculate CCI
    current_typical = typical_prices[-1]
    cci = (current_typical - sma_typical) / (0.015 * mean_deviation)

    return float(cci)


def _calculate_cci_series_manual(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int
) -> list[float | None]:
    """Manual CCI series calculation."""
    cci_series: list[float | None] = []

    # Calculate typical price series
    typical_prices = _calculate_typical_price(highs, lows, closes)

    for i in range(len(typical_prices)):
        if i < period - 1:
            cci_series.append(None)
        else:
            # Get window for calculation
            window = typical_prices[i - period + 1: i + 1]

            # Calculate SMA of typical price
            sma_typical = np.mean(window)

            # Calculate mean deviation
            mean_deviation = np.mean(np.abs(window - sma_typical))

            if mean_deviation == 0:
                cci = 0.0
            else:
                # Calculate CCI
                current_typical = typical_prices[i]
                cci = (current_typical - sma_typical) / \
                    (0.015 * mean_deviation)

            cci_series.append(float(cci))

    return cci_series
