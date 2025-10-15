"""
Money Flow Index (MFI) Indicator

MFI is a momentum oscillator that uses both price and volume to measure buying and selling pressure.
It's often called "volume-weighted RSI" and ranges from 0 to 100.

Calculation:
1. Typical Price = (High + Low + Close) / 3
2. Raw Money Flow = Typical Price × Volume
3. Positive/Negative Money Flow based on typical price direction
4. Money Flow Ratio = Positive Money Flow / Negative Money Flow
5. MFI = 100 - (100 / (1 + Money Flow Ratio))
"""

import numpy as np
from typing import Union, List, Optional


def calculate_mfi(
    highs: Union[List[float], np.ndarray],
    lows: Union[List[float], np.ndarray],
    closes: Union[List[float], np.ndarray],
    volumes: Union[List[float], np.ndarray],
    period: int = 14,
) -> Optional[float]:
    """
    Calculate Money Flow Index for the most recent period.

    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices
        volumes: Array of volumes
        period: Lookback period (default 14)

    Returns:
        MFI value (0-100) or None if insufficient data
    """
    # Convert to numpy arrays
    highs_array = np.array(highs, dtype=float)
    lows_array = np.array(lows, dtype=float)
    closes_array = np.array(closes, dtype=float)
    volumes_array = np.array(volumes, dtype=float)

    if (
        len(highs_array) < period + 1
        or len(lows_array) < period + 1
        or len(closes_array) < period + 1
        or len(volumes_array) < period + 1
    ):
        return None

    # Calculate typical price
    typical_prices = (highs_array + lows_array + closes_array) / 3

    # Calculate raw money flow
    money_flows = typical_prices * volumes_array

    # Get the last period+1 values (need period+1 to compare direction)
    recent_typical = typical_prices[-(period + 1) :]
    recent_flows = money_flows[-(period + 1) :]

    # Separate positive and negative money flows
    positive_flow = 0
    negative_flow = 0

    for i in range(1, len(recent_typical)):
        if recent_typical[i] > recent_typical[i - 1]:
            positive_flow += recent_flows[i]
        elif recent_typical[i] < recent_typical[i - 1]:
            negative_flow += recent_flows[i]

    # Avoid division by zero
    if negative_flow == 0:
        return 100.0
    if positive_flow == 0:
        return 0.0

    # Calculate MFI
    money_flow_ratio = positive_flow / negative_flow
    mfi = 100 - (100 / (1 + money_flow_ratio))

    return float(mfi)


def calculate_mfi_series(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 14,
) -> List[Optional[float]]:
    """
    Calculate MFI series for all data points.

    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of close prices
        volumes: Array of volumes
        period: Lookback period (default 14)

    Returns:
        List of MFI values, None for insufficient data points
    """
    if len(highs) == 0:
        return []

    result = []

    for i in range(len(highs)):
        if i < period:
            result.append(None)
        else:
            mfi_val = calculate_mfi(
                highs[: i + 1], lows[: i + 1], closes[: i + 1], volumes[: i + 1], period
            )
            result.append(mfi_val)

    return result


def interpret_mfi(mfi_value: float) -> str:
    """
    Interpret MFI signal strength.

    Args:
        mfi_value: MFI value (0-100)

    Returns:
        Signal interpretation
    """
    if mfi_value >= 80:
        return "OVERBOUGHT_STRONG"
    elif mfi_value >= 70:
        return "OVERBOUGHT"
    elif mfi_value <= 20:
        return "OVERSOLD_STRONG"
    elif mfi_value <= 30:
        return "OVERSOLD"
    else:
        return "NEUTRAL"
