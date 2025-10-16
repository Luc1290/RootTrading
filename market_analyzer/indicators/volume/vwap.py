"""
VWAP (Volume Weighted Average Price) Indicator

This module provides VWAP calculation for volume-based price analysis.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_vwap_quote(
    highs: list[float] | np.ndarray | pd.Series,
    lows: list[float] | np.ndarray | pd.Series,
    closes: list[float] | np.ndarray | pd.Series,
    quote_volumes: list[float] | np.ndarray | pd.Series,
    period: int | None = None,
) -> float | None:
    """
    Calculate Volume Weighted Average Price using quote asset volume (USDC).

    More accurate than regular VWAP as it uses real monetary value exchanged
    rather than base asset quantity.

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        quote_volumes: Quote asset volume (USDC volume)
        period: Period for rolling VWAP (None for session VWAP)

    Returns:
        Quote VWAP value or None if insufficient data

    Notes:
        - Uses quote asset volume (USDC) for more accurate weighting
        - Better reflects actual monetary flow
        - More suitable for cross-asset comparisons
    """
    highs_array = _to_numpy_array(highs)
    lows_array = _to_numpy_array(lows)
    closes_array = _to_numpy_array(closes)
    quote_volumes_array = _to_numpy_array(quote_volumes)

    # Ensure all arrays have same length
    min_len = min(
        len(highs_array), len(lows_array), len(closes_array), len(quote_volumes_array)
    )
    if min_len == 0:
        return None

    highs_array = highs_array[-min_len:]
    lows_array = lows_array[-min_len:]
    closes_array = closes_array[-min_len:]
    quote_volumes_array = quote_volumes_array[-min_len:]

    # Calculate typical price
    typical_prices = (highs_array + lows_array + closes_array) / 3

    if period is None:
        # Session VWAP (all data)
        total_pv = np.sum(typical_prices * quote_volumes_array)
        total_volume = np.sum(quote_volumes_array)
    else:
        # Rolling VWAP
        if len(typical_prices) < period:
            return None

        recent_typical = typical_prices[-period:]
        recent_quote_volumes = quote_volumes_array[-period:]
        total_pv = np.sum(recent_typical * recent_quote_volumes)
        total_volume = np.sum(recent_quote_volumes)

    if total_volume == 0:
        return None

    vwap = total_pv / total_volume
    return float(vwap)


def calculate_vwap(
    highs: list[float] | np.ndarray | pd.Series,
    lows: list[float] | np.ndarray | pd.Series,
    closes: list[float] | np.ndarray | pd.Series,
    volumes: list[float] | np.ndarray | pd.Series,
    period: int | None = None,
) -> float | None:
    """
    Calculate Volume Weighted Average Price (VWAP).

    VWAP gives average price weighted by volume, providing insight into
    the "fair value" of a security based on both price and volume.

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        volumes: Volume data
        period: Period for rolling VWAP (None for session VWAP)

    Returns:
        VWAP value or None if insufficient data

    Notes:
        - VWAP is often used as a benchmark for execution quality
        - Price above VWAP suggests bullish sentiment
        - Price below VWAP suggests bearish sentiment
        - Institutional traders often use VWAP for large orders
    """
    highs_array = _to_numpy_array(highs)
    lows_array = _to_numpy_array(lows)
    closes_array = _to_numpy_array(closes)
    volumes_array = _to_numpy_array(volumes)

    # Ensure all arrays have same length
    min_len = min(
        len(highs_array), len(lows_array), len(closes_array), len(volumes_array)
    )
    if min_len == 0:
        return None

    highs_array = highs_array[-min_len:]
    lows_array = lows_array[-min_len:]
    closes_array = closes_array[-min_len:]
    volumes_array = volumes_array[-min_len:]

    # Calculate typical price
    typical_prices = (highs_array + lows_array + closes_array) / 3

    if period is None:
        # Session VWAP (all data)
        total_pv = np.sum(typical_prices * volumes_array)
        total_volume = np.sum(volumes_array)
    else:
        # Rolling VWAP
        if len(typical_prices) < period:
            return None

        recent_typical = typical_prices[-period:]
        recent_volumes = volumes_array[-period:]
        total_pv = np.sum(recent_typical * recent_volumes)
        total_volume = np.sum(recent_volumes)

    if total_volume == 0:
        return None

    vwap = total_pv / total_volume
    return float(vwap)


def calculate_vwap_series(
    highs: list[float] | np.ndarray | pd.Series,
    lows: list[float] | np.ndarray | pd.Series,
    closes: list[float] | np.ndarray | pd.Series,
    volumes: list[float] | np.ndarray | pd.Series,
    period: int | None = None,
) -> list[float | None]:
    """
    Calculate VWAP for entire price series.

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        volumes: Volume data
        period: Period for rolling VWAP

    Returns:
        List of VWAP values
    """
    highs_array = _to_numpy_array(highs)
    lows_array = _to_numpy_array(lows)
    closes_array = _to_numpy_array(closes)
    volumes_array = _to_numpy_array(volumes)

    # Ensure all arrays have same length
    min_len = min(
        len(highs_array), len(lows_array), len(closes_array), len(volumes_array)
    )
    highs_array = highs_array[-min_len:]
    lows_array = lows_array[-min_len:]
    closes_array = closes_array[-min_len:]
    volumes_array = volumes_array[-min_len:]

    # Calculate typical price
    typical_prices = (highs_array + lows_array + closes_array) / 3

    vwap_series: list[float | None] = []

    for i in range(len(typical_prices)):
        if period is None:
            # Cumulative VWAP from start
            if i == 0:
                if volumes_array[i] == 0:
                    vwap_series.append(None)
                else:
                    vwap_series.append(float(typical_prices[i]))
            else:
                cumulative_pv = np.sum(typical_prices[: i + 1] * volumes_array[: i + 1])
                cumulative_volume = np.sum(volumes_array[: i + 1])

                if cumulative_volume == 0:
                    vwap_series.append(None)
                else:
                    vwap = cumulative_pv / cumulative_volume
                    vwap_series.append(float(vwap))
        # Rolling VWAP
        elif i < period - 1:
            vwap_series.append(None)
        else:
            window_typical = typical_prices[i - period + 1 : i + 1]
            window_volumes = volumes_array[i - period + 1 : i + 1]

            window_pv = np.sum(window_typical * window_volumes)
            window_volume = np.sum(window_volumes)

            if window_volume == 0:
                vwap_series.append(None)
            else:
                vwap = window_pv / window_volume
                vwap_series.append(float(vwap))

    return vwap_series


def calculate_vwap_quote_series(
    highs: list[float] | np.ndarray | pd.Series,
    lows: list[float] | np.ndarray | pd.Series,
    closes: list[float] | np.ndarray | pd.Series,
    quote_volumes: list[float] | np.ndarray | pd.Series,
    period: int | None = None,
) -> list[float | None]:
    """
    Calculate Quote VWAP series using quote asset volume.

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        quote_volumes: Quote asset volume (USDC)
        period: Period for rolling VWAP

    Returns:
        List of Quote VWAP values
    """
    highs_array = _to_numpy_array(highs)
    lows_array = _to_numpy_array(lows)
    closes_array = _to_numpy_array(closes)
    quote_volumes_array = _to_numpy_array(quote_volumes)

    # Ensure all arrays have same length
    min_len = min(
        len(highs_array), len(lows_array), len(closes_array), len(quote_volumes_array)
    )
    if min_len == 0:
        return []

    highs_array = highs_array[-min_len:]
    lows_array = lows_array[-min_len:]
    closes_array = closes_array[-min_len:]
    quote_volumes_array = quote_volumes_array[-min_len:]

    # Calculate typical price
    typical_prices = (highs_array + lows_array + closes_array) / 3

    vwap_series: list[float | None] = []

    for i in range(len(typical_prices)):
        if period is None:
            # Cumulative VWAP from start
            if i == 0:
                if quote_volumes_array[i] == 0:
                    vwap_series.append(None)
                else:
                    vwap_series.append(float(typical_prices[i]))
            else:
                cumulative_pv = np.sum(
                    typical_prices[: i + 1] * quote_volumes_array[: i + 1]
                )
                cumulative_volume = np.sum(quote_volumes_array[: i + 1])

                if cumulative_volume == 0:
                    vwap_series.append(None)
                else:
                    vwap = cumulative_pv / cumulative_volume
                    vwap_series.append(float(vwap))
        # Rolling VWAP
        elif i < period - 1:
            vwap_series.append(None)
        else:
            window_typical = typical_prices[i - period + 1 : i + 1]
            window_quote_volumes = quote_volumes_array[i - period + 1 : i + 1]

            window_pv = np.sum(window_typical * window_quote_volumes)
            window_volume = np.sum(window_quote_volumes)

            if window_volume == 0:
                vwap_series.append(None)
            else:
                vwap = window_pv / window_volume
                vwap_series.append(float(vwap))

    return vwap_series


def calculate_vwap_bands(
    highs: list[float] | np.ndarray | pd.Series,
    lows: list[float] | np.ndarray | pd.Series,
    closes: list[float] | np.ndarray | pd.Series,
    volumes: list[float] | np.ndarray | pd.Series,
    std_multiplier: float = 1.0,
    period: int | None = None,
) -> dict[str, float | None]:
    """
    Calculate VWAP with standard deviation bands.

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        volumes: Volume data
        std_multiplier: Standard deviation multiplier for bands
        period: Period for calculation

    Returns:
        Dictionary with vwap, upper_band, lower_band
    """
    vwap = calculate_vwap(highs, lows, closes, volumes, period)
    if vwap is None:
        return {"vwap": None, "upper_band": None, "lower_band": None}

    # Calculate volume-weighted standard deviation
    highs_array = _to_numpy_array(highs)
    lows_array = _to_numpy_array(lows)
    closes_array = _to_numpy_array(closes)
    volumes_array = _to_numpy_array(volumes)

    min_len = min(
        len(highs_array), len(lows_array), len(closes_array), len(volumes_array)
    )
    typical_prices = (
        highs_array[-min_len:] + lows_array[-min_len:] + closes_array[-min_len:]
    ) / 3
    volumes = volumes_array[-min_len:]

    if period is None:
        # Use all data
        data_range = typical_prices
        vol_range = volumes
    else:
        # Use recent period
        if len(typical_prices) < period:
            return {"vwap": vwap, "upper_band": None, "lower_band": None}
        data_range = typical_prices[-period:]
        vol_range = volumes[-period:]

    # Calculate volume-weighted variance
    total_volume = np.sum(vol_range)
    if total_volume == 0:
        return {"vwap": vwap, "upper_band": None, "lower_band": None}

    weighted_variance = np.sum(vol_range * (data_range - vwap) ** 2) / total_volume
    vwap_std = np.sqrt(weighted_variance)

    upper_band = vwap + (vwap_std * std_multiplier)
    lower_band = vwap - (vwap_std * std_multiplier)

    return {
        "vwap": float(vwap),
        "upper_band": float(upper_band),
        "lower_band": float(lower_band),
    }


def calculate_anchored_vwap(
    highs: list[float] | np.ndarray | pd.Series,
    lows: list[float] | np.ndarray | pd.Series,
    closes: list[float] | np.ndarray | pd.Series,
    volumes: list[float] | np.ndarray | pd.Series,
    anchor_index: int,
) -> float | None:
    """
    Calculate Anchored VWAP from a specific point.

    Anchored VWAP starts calculation from a significant event or level.

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        volumes: Volume data
        anchor_index: Index to start VWAP calculation from

    Returns:
        Anchored VWAP value or None
    """
    if anchor_index < 0:
        return None

    # Slice data from anchor point
    highs_slice = highs[anchor_index:]
    lows_slice = lows[anchor_index:]
    closes_slice = closes[anchor_index:]
    volumes_slice = volumes[anchor_index:]

    return calculate_vwap(highs_slice, lows_slice, closes_slice, volumes_slice)


def vwap_signal(
    current_price: float,
    current_vwap: float | None,
    previous_price: float | None = None,
    previous_vwap: float | None = None,
) -> str:
    """
    Generate trading signal based on price relative to VWAP.

    Args:
        current_price: Current price
        current_vwap: Current VWAP value
        previous_price: Previous price (optional)
        previous_vwap: Previous VWAP value (optional)

    Returns:
        'bullish', 'bearish', 'bullish_cross', 'bearish_cross', or 'neutral'
    """
    if current_vwap is None:
        return "neutral"

    # Basic position relative to VWAP
    if current_price > current_vwap:
        base_signal = "bullish"
    elif current_price < current_vwap:
        base_signal = "bearish"
    else:
        return "neutral"

    # Check for crossovers if previous values available
    if previous_price is not None and previous_vwap is not None:
        # Bullish cross: Price crosses above VWAP
        if previous_price <= previous_vwap and current_price > current_vwap:
            return "bullish_cross"

        # Bearish cross: Price crosses below VWAP
        if previous_price >= previous_vwap and current_price < current_vwap:
            return "bearish_cross"

    return base_signal


def calculate_vwap_deviation(current_price: float, vwap: float | None) -> float | None:
    """
    Calculate price deviation from VWAP as percentage.

    Args:
        current_price: Current price
        vwap: VWAP value

    Returns:
        Deviation percentage or None
    """
    if vwap is None or vwap == 0:
        return None

    deviation = ((current_price - vwap) / vwap) * 100
    return float(deviation)


def calculate_volume_profile(
    prices: list[float] | np.ndarray | pd.Series,
    volumes: list[float] | np.ndarray | pd.Series,
    num_bins: int = 20,
) -> dict[str, list[float]]:
    """
    Calculate volume profile for price levels.

    Args:
        prices: Price data
        volumes: Volume data
        num_bins: Number of price bins

    Returns:
        Dictionary with price_levels and volume_profile
    """
    prices_array = _to_numpy_array(prices)
    volumes_array = _to_numpy_array(volumes)

    min_len = min(len(prices_array), len(volumes_array))
    prices_array = prices_array[-min_len:]
    volumes_array = volumes_array[-min_len:]

    if len(prices_array) == 0:
        return {"price_levels": [], "volume_profile": []}

    # Create price bins
    min_price = np.min(prices_array)
    max_price = np.max(prices_array)

    if min_price == max_price:
        return {
            "price_levels": [float(min_price)],
            "volume_profile": [float(np.sum(volumes_array))],
        }

    price_bins = np.linspace(min_price, max_price, num_bins + 1)
    volume_profile = np.zeros(num_bins)

    # Aggregate volume for each price bin
    for i, price in enumerate(prices_array):
        bin_index = np.digitize(price, price_bins) - 1
        # Clamp to valid range
        bin_index = max(0, min(bin_index, num_bins - 1))
        volume_profile[bin_index] += volumes_array[i]

    # Calculate bin centers
    price_levels = [(price_bins[i] + price_bins[i + 1]) / 2 for i in range(num_bins)]

    return {
        "price_levels": [float(level) for level in price_levels],
        "volume_profile": [float(vol) for vol in volume_profile],
    }


def find_poc(
    prices: list[float] | np.ndarray | pd.Series,
    volumes: list[float] | np.ndarray | pd.Series,
    num_bins: int = 20,
) -> float | None:
    """
    Find Point of Control (POC) - price level with highest volume.

    Args:
        prices: Price data
        volumes: Volume data
        num_bins: Number of price bins for calculation

    Returns:
        POC price level or None
    """
    profile = calculate_volume_profile(prices, volumes, num_bins)

    if not profile["volume_profile"]:
        return None

    max_volume_index = np.argmax(profile["volume_profile"])
    return float(profile["price_levels"][max_volume_index])


def calculate_value_area(
    prices: list[float] | np.ndarray | pd.Series,
    volumes: list[float] | np.ndarray | pd.Series,
    value_area_percent: float = 0.7,
    num_bins: int = 20,
) -> dict[str, float | None]:
    """
    Calculate Value Area High (VAH) and Value Area Low (VAL).

    Value Area contains the specified percentage of total volume (default 70%).

    Args:
        prices: Price data
        volumes: Volume data
        value_area_percent: Percentage of volume in value area (0.7 = 70%)
        num_bins: Number of price bins

    Returns:
        Dictionary with 'vah' (Value Area High) and 'val' (Value Area Low)
    """
    profile = calculate_volume_profile(prices, volumes, num_bins)

    if not profile["volume_profile"] or not profile["price_levels"]:
        return {"vah": None, "val": None}

    # Get total volume and target volume for value area
    total_volume = sum(profile["volume_profile"])
    target_volume = total_volume * value_area_percent

    # Find POC (Point of Control)
    poc_index = np.argmax(profile["volume_profile"])

    # Expand from POC until we reach target volume
    accumulated_volume = profile["volume_profile"][poc_index]
    low_index = poc_index
    high_index = poc_index

    while accumulated_volume < target_volume and (
        low_index > 0 or high_index < len(profile["volume_profile"]) - 1
    ):
        # Determine which direction to expand (choose side with more volume)
        left_volume = profile["volume_profile"][low_index - 1] if low_index > 0 else 0
        right_volume = (
            profile["volume_profile"][high_index + 1]
            if high_index < len(profile["volume_profile"]) - 1
            else 0
        )

        if left_volume >= right_volume and low_index > 0:
            low_index -= 1
            accumulated_volume += profile["volume_profile"][low_index]
        elif right_volume > 0 and high_index < len(profile["volume_profile"]) - 1:
            high_index += 1
            accumulated_volume += profile["volume_profile"][high_index]
        else:
            break

    return {
        "vah": float(profile["price_levels"][high_index]),  # Value Area High
        "val": float(profile["price_levels"][low_index]),  # Value Area Low
    }


# ============ Helper Functions ============


def _to_numpy_array(data: list[float] | np.ndarray | pd.Series) -> np.ndarray:
    """Convert input data to numpy array."""
    if isinstance(data, pd.Series):
        return np.asarray(data.values, dtype=float)
    if isinstance(data, list):
        return np.array(data, dtype=float)
    return np.asarray(data, dtype=float)
