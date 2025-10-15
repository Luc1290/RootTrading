"""
Bollinger Bands Indicator

This module provides Bollinger Bands calculation including:
- Upper Band
- Middle Band (SMA)
- Lower Band
- Bandwidth
- %B (Percent B)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.debug("TA-Lib not available, using manual calculations")


def calculate_bollinger_bands(
    prices: Union[List[float], np.ndarray, pd.Series],
    period: int = 20,
    std_dev: float = 2.0,
    ma_type: str = "sma",
) -> Dict[str, Optional[float]]:
    """
    Calculate Bollinger Bands.

    Bollinger Bands consist of a middle band (SMA) and two outer bands
    at a specified number of standard deviations from the middle band.

    Args:
        prices: Price series (typically closing prices)
        period: Period for moving average and standard deviation (default: 20)
        std_dev: Number of standard deviations for bands (default: 2.0)
        ma_type: Type of moving average ('sma', 'ema') (default: 'sma')

    Returns:
        Dictionary with:
        - upper: Upper band value
        - middle: Middle band (moving average) value
        - lower: Lower band value
        - bandwidth: Band width (upper - lower)
        - percent_b: %B indicator (position within bands)

    Notes:
        - Price touching upper band may indicate overbought
        - Price touching lower band may indicate oversold
        - Band squeeze (low bandwidth) may indicate pending volatility
    """
    prices_array = _to_numpy_array(prices)
    if len(prices_array) < period:
        return {
            "upper": None,
            "middle": None,
            "lower": None,
            "bandwidth": None,
            "percent_b": None,
        }

    if TALIB_AVAILABLE and ma_type == "sma":
        try:
            upper, middle, lower = talib.BBANDS(
                prices_array,
                timeperiod=period,
                nbdevup=std_dev,
                nbdevdn=std_dev,
                matype=0,  # type: ignore  # SMA
            )

            # Calculate additional metrics
            bandwidth = None
            percent_b = None

            if not np.isnan(upper[-1]) and not np.isnan(lower[-1]):
                bandwidth = upper[-1] - lower[-1]
                if bandwidth > 0:
                    percent_b = (prices_array[-1] - lower[-1]) / bandwidth

            return {
                "upper": float(upper[-1]) if not np.isnan(upper[-1]) else None,
                "middle": float(middle[-1]) if not np.isnan(middle[-1]) else None,
                "lower": float(lower[-1]) if not np.isnan(lower[-1]) else None,
                "bandwidth": float(bandwidth) if bandwidth is not None else None,
                "percent_b": float(percent_b) if percent_b is not None else None,
            }
        except Exception as e:
            logger.warning(f"TA-Lib Bollinger Bands error: {e}, using fallback")

    return _calculate_bollinger_manual(prices_array, period, std_dev, ma_type)


def calculate_bollinger_bands_series(
    prices: Union[List[float], np.ndarray, pd.Series],
    period: int = 20,
    std_dev: float = 2.0,
    ma_type: str = "sma",
) -> Dict[str, List[Optional[float]]]:
    """
    Calculate Bollinger Bands for entire price series.

    Args:
        prices: Price series
        period: Period for calculation
        std_dev: Number of standard deviations
        ma_type: Moving average type

    Returns:
        Dictionary with lists of values for upper, middle, lower, bandwidth, percent_b
    """
    prices_array = _to_numpy_array(prices)

    if TALIB_AVAILABLE and ma_type == "sma":
        try:
            upper, middle, lower = talib.BBANDS(
                prices_array,
                timeperiod=period,
                nbdevup=std_dev,
                nbdevdn=std_dev,
                matype=0,  # type: ignore
            )

            # Calculate additional metrics series
            bandwidth_series = []
            percent_b_series = []

            for i in range(len(prices_array)):
                if not np.isnan(upper[i]) and not np.isnan(lower[i]):
                    bw = upper[i] - lower[i]
                    bandwidth_series.append(float(bw))

                    if bw > 0:
                        pb = (prices_array[i] - lower[i]) / bw
                        percent_b_series.append(float(pb))
                    else:
                        percent_b_series.append(0.0)
                else:
                    bandwidth_series.append(0.0)
                    percent_b_series.append(0.0)

            return {
                "upper": [float(val) if not np.isnan(val) else None for val in upper],
                "middle": [float(val) if not np.isnan(val) else None for val in middle],
                "lower": [float(val) if not np.isnan(val) else None for val in lower],
                "bandwidth": [x if x is not None else 0.0 for x in bandwidth_series],
                "percent_b": [x if x is not None else 0.0 for x in percent_b_series],
            }
        except Exception as e:
            logger.warning(f"TA-Lib Bollinger Bands series error: {e}, using fallback")

    # Manual calculation
    return _calculate_bollinger_series_manual(prices_array, period, std_dev, ma_type)


def calculate_bollinger_width(
    prices: Union[List[float], np.ndarray, pd.Series],
    period: int = 20,
    std_dev: float = 2.0,
) -> Optional[float]:
    """
    Calculate Bollinger Band Width indicator.

    Band Width = (Upper Band - Lower Band) / Middle Band

    Args:
        prices: Price series
        period: Period for calculation
        std_dev: Number of standard deviations

    Returns:
        Band width ratio or None
    """
    bands = calculate_bollinger_bands(prices, period, std_dev)

    if bands["upper"] is None or bands["lower"] is None or bands["middle"] is None:
        return None

    if bands["middle"] == 0:
        return None

    width = (bands["upper"] - bands["lower"]) / bands["middle"]
    return float(width)


def calculate_bollinger_squeeze(
    prices: Union[List[float], np.ndarray, pd.Series],
    period: int = 20,
    std_dev: float = 2.0,
    lookback: int = 120,
) -> Dict[str, Optional[Union[float, bool]]]:
    """
    Detect Bollinger Band squeeze conditions.

    A squeeze occurs when volatility falls to low levels and the bands narrow.

    Args:
        prices: Price series
        period: Bollinger Band period
        std_dev: Number of standard deviations
        lookback: Period to check for squeeze (default: 120)

    Returns:
        Dictionary with:
        - in_squeeze: Boolean indicating if currently in squeeze
        - squeeze_percentage: Current width as percentage of lookback average
        - width_percentile: Percentile rank of current width
    """
    # Calculate band width series
    bands_series = calculate_bollinger_bands_series(prices, period, std_dev)
    bandwidth_series = bands_series["bandwidth"]

    # Remove None values
    valid_bandwidths = [x for x in bandwidth_series if x is not None]

    if len(valid_bandwidths) < lookback:
        return {
            "in_squeeze": None,
            "squeeze_percentage": None,
            "width_percentile": None,
        }

    current_width = valid_bandwidths[-1]
    recent_widths = valid_bandwidths[-lookback:]

    # Calculate statistics
    avg_width = np.mean(recent_widths)
    min_width = np.min(recent_widths)

    # Squeeze detection (width below 20th percentile)
    percentile_20 = np.percentile(recent_widths, 20)
    in_squeeze = current_width <= percentile_20

    # Calculate metrics
    squeeze_percentage = (current_width / avg_width) * 100 if avg_width > 0 else None
    width_percentile = (
        np.sum(np.array(recent_widths) <= current_width) / len(recent_widths)
    ) * 100

    return {
        "in_squeeze": bool(in_squeeze),
        "squeeze_percentage": (
            float(squeeze_percentage) if squeeze_percentage is not None else None
        ),
        "width_percentile": float(width_percentile),
    }


def calculate_bollinger_expansion(
    prices: Union[List[float], np.ndarray, pd.Series],
    period: int = 20,
    std_dev: float = 2.0,
    lookback: int = 10,
) -> bool:
    """
    Detect Bollinger Band expansion (volatility increasing).

    Args:
        prices: Price data
        period: Period for BB calculation
        std_dev: Standard deviation multiplier
        lookback: Periods to compare for expansion detection

    Returns:
        True if bands are expanding, False otherwise
    """
    if len(prices) < period + lookback:
        return False

    prices_array = _to_numpy_array(prices)

    # Calculate BB width for recent periods
    widths = []
    for i in range(lookback):
        end_idx = len(prices_array) - i
        start_idx = max(0, end_idx - period)

        if end_idx - start_idx < period:
            continue

        window_prices = prices_array[start_idx:end_idx]
        sma = np.mean(window_prices)
        std = np.std(window_prices, ddof=0)

        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        width = upper_band - lower_band

        widths.append(width)

    if len(widths) < 3:
        return False

    # Check if width is increasing (expansion)
    current_width = widths[0]  # Most recent
    previous_width = widths[1]  # Previous period
    avg_width = np.mean(widths[2:])  # Historical average

    # Expansion if current > previous AND current > average
    is_expanding = current_width > previous_width and current_width > avg_width

    return bool(is_expanding)


def calculate_bollinger_breakout_direction(
    prices: Union[List[float], np.ndarray, pd.Series],
    period: int = 20,
    std_dev: float = 2.0,
    lookback: int = 3,
) -> str:
    """
    Detect Bollinger Band breakout direction.

    Args:
        prices: Price data
        period: Period for BB calculation
        std_dev: Standard deviation multiplier
        lookback: Periods to analyze for breakout

    Returns:
        'UP' for upward breakout, 'DOWN' for downward breakout, 'NONE' for no breakout
    """
    if len(prices) < period + lookback:
        return "NONE"

    prices_array = _to_numpy_array(prices)
    recent_prices = prices_array[-lookback:]

    # Calculate current BB levels
    bb_data = prices_array[-period:]
    sma = np.mean(bb_data)
    std = np.std(bb_data, ddof=0)

    upper_band = sma + (std_dev * std)
    lower_band = sma - (std_dev * std)

    # Check for breakouts in recent periods
    upper_breaks = np.sum(recent_prices > upper_band)
    lower_breaks = np.sum(recent_prices < lower_band)

    # Current price position
    current_price = recent_prices[-1]

    # Determine breakout direction
    if current_price > upper_band and upper_breaks >= 2:
        return "UP"
    elif current_price < lower_band and lower_breaks >= 2:
        return "DOWN"
    else:
        return "NONE"


def calculate_keltner_channels(
    prices: Union[List[float], np.ndarray, pd.Series],
    highs: Union[List[float], np.ndarray, pd.Series],
    lows: Union[List[float], np.ndarray, pd.Series],
    period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> Dict[str, Optional[float]]:
    """
    Calculate Keltner Channels.

    Keltner Channels use ATR for volatility instead of standard deviation.

    Args:
        prices: Close prices
        highs: High prices
        lows: Low prices
        period: EMA period for middle line
        atr_period: ATR period
        multiplier: ATR multiplier for bands

    Returns:
        Dictionary with upper, middle, lower values
    """
    from ..volatility.atr import calculate_atr
    from ..trend.moving_averages import calculate_ema

    prices_array = _to_numpy_array(prices)

    # Calculate middle line (EMA)
    middle = calculate_ema(prices_array, period)
    if middle is None:
        return {"upper": None, "middle": None, "lower": None}

    # Calculate ATR
    atr = calculate_atr(highs, lows, prices, atr_period)
    if atr is None:
        return {"upper": middle, "middle": middle, "lower": middle}

    # Calculate bands
    upper = middle + (atr * multiplier)
    lower = middle - (atr * multiplier)

    return {"upper": float(upper), "middle": float(middle), "lower": float(lower)}


def bollinger_band_signal(
    prices: Union[List[float], np.ndarray, pd.Series],
    period: int = 20,
    std_dev: float = 2.0,
) -> str:
    """
    Generate trading signal based on Bollinger Bands.

    Args:
        prices: Price series
        period: BB period
        std_dev: Standard deviations

    Returns:
        'overbought', 'oversold', 'squeeze', or 'neutral'
    """
    if len(prices) < 2:
        return "neutral"

    bands = calculate_bollinger_bands(prices, period, std_dev)
    if bands["upper"] is None or bands["lower"] is None:
        return "neutral"

    current_price = float(prices[-1])
    prev_price = float(prices[-2])

    # Check for band touches/breaks
    if current_price >= bands["upper"]:
        return "overbought"
    elif current_price <= bands["lower"]:
        return "oversold"

    # Check for squeeze
    squeeze = calculate_bollinger_squeeze(prices, period, std_dev)
    if squeeze["in_squeeze"] is True:
        return "squeeze"

    return "neutral"


# ============ Helper Functions ============


def _to_numpy_array(data: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
    """Convert input data to numpy array."""
    if isinstance(data, pd.Series):
        if hasattr(data.values, "values"):  # ExtensionArray
            return np.asarray(data.values, dtype=float)
        return np.asarray(data.values, dtype=float)
    elif isinstance(data, list):
        return np.array(data, dtype=float)
    return np.asarray(data, dtype=float)


def _calculate_bollinger_manual(
    prices: np.ndarray, period: int, std_dev: float, ma_type: str
) -> Dict[str, Optional[float]]:
    """Manual Bollinger Bands calculation."""
    if ma_type == "ema":
        from ..trend.moving_averages import calculate_ema

        middle = calculate_ema(prices, period)
    else:  # SMA
        middle = float(np.mean(prices[-period:]))

    if middle is None:
        return {
            "upper": None,
            "middle": None,
            "lower": None,
            "bandwidth": None,
            "percent_b": None,
        }

    # Calculate standard deviation
    std = float(np.std(prices[-period:], ddof=0))  # Population std dev

    # Calculate bands
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)

    # Calculate additional metrics
    bandwidth = upper - lower
    percent_b = None

    if bandwidth > 0:
        percent_b = (float(prices[-1]) - lower) / bandwidth

    return {
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "bandwidth": bandwidth,
        "percent_b": percent_b,
    }


def _calculate_bollinger_series_manual(
    prices: np.ndarray, period: int, std_dev: float, ma_type: str
) -> Dict[str, List[Optional[float]]]:
    """Manual Bollinger Bands series calculation."""
    upper_series: List[Optional[float]] = []
    middle_series: List[Optional[float]] = []
    lower_series: List[Optional[float]] = []
    bandwidth_series: List[Optional[float]] = []
    percent_b_series: List[Optional[float]] = []

    for i in range(len(prices)):
        if i < period - 1:
            upper_series.append(None)
            middle_series.append(None)
            lower_series.append(None)
            bandwidth_series.append(None)
            percent_b_series.append(None)
        else:
            # Calculate for window
            window = prices[i - period + 1 : i + 1]

            if ma_type == "ema":
                from ..trend.moving_averages import calculate_ema

                middle = calculate_ema(window, period)
            else:
                middle = float(np.mean(window))

            std = float(np.std(window, ddof=0))

            if middle is not None:
                upper = middle + (std_dev * std)
                lower = middle - (std_dev * std)
                bandwidth = upper - lower
            else:
                upper = None
                lower = None
                bandwidth = None

            upper_series.append(upper)
            middle_series.append(middle)
            lower_series.append(lower)
            bandwidth_series.append(bandwidth)

            if bandwidth is not None and bandwidth > 0 and lower is not None:
                percent_b = (float(prices[i]) - lower) / bandwidth
                percent_b_series.append(percent_b)
            else:
                percent_b_series.append(None)

    return {
        "upper": upper_series,
        "middle": middle_series,
        "lower": lower_series,
        "bandwidth": bandwidth_series,
        "percent_b": percent_b_series,
    }
