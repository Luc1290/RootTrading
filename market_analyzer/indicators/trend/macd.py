"""
MACD (Moving Average Convergence Divergence) Indicator

This module provides MACD calculation including:
- MACD Line (difference between fast and slow EMA)
- Signal Line (EMA of MACD line)
- MACD Histogram (difference between MACD and Signal)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import logging
from .moving_averages import (
    calculate_ema,
    calculate_ema_incremental,
    calculate_ema_series,
)

logger = logging.getLogger(__name__)

try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.debug("TA-Lib not available, using manual calculations")


def calculate_macd(
    prices: Union[List[float], np.ndarray, pd.Series],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    normalize_high_price: bool = True,
) -> Dict[str, Optional[float]]:
    """
    Calculate MACD indicator values.

    MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages of prices.

    Args:
        prices: Price series (typically closing prices)
        fast_period: Period for fast EMA (default: 12)
        slow_period: Period for slow EMA (default: 26)
        signal_period: Period for signal line EMA (default: 9)

    Returns:
        Dictionary with:
        - macd_line: MACD line value
        - macd_signal: Signal line value
        - macd_histogram: Histogram value

    Notes:
        - MACD line = 12-period EMA - 26-period EMA
        - Signal line = 9-period EMA of MACD line
        - Histogram = MACD line - Signal line
    """
    prices_array = _to_numpy_array(prices)
    min_required = max(slow_period, fast_period) + signal_period

    if len(prices_array) < min_required:
        return {"macd_line": None, "macd_signal": None, "macd_histogram": None}

    if TALIB_AVAILABLE:
        try:
            macd_line, macd_signal, macd_hist = talib.MACD(
                prices_array,
                fastperiod=fast_period,
                slowperiod=slow_period,
                signalperiod=signal_period,
            )

            return {
                "macd_line": (
                    float(macd_line[-1]) if not np.isnan(macd_line[-1]) else None
                ),
                "macd_signal": (
                    float(macd_signal[-1]) if not np.isnan(macd_signal[-1]) else None
                ),
                "macd_histogram": (
                    float(macd_hist[-1]) if not np.isnan(macd_hist[-1]) else None
                ),
            }
        except Exception as e:
            logger.warning(f"TA-Lib MACD error: {e}, using fallback")

    return _calculate_macd_manual(
        prices_array, fast_period, slow_period, signal_period, normalize_high_price
    )


def calculate_macd_series(
    prices: Union[List[float], np.ndarray, pd.Series],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Dict[str, List[Optional[float]]]:
    """
    Calculate MACD values for entire price series.

    Args:
        prices: Price series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal EMA period

    Returns:
        Dictionary with lists of values for macd_line, macd_signal, macd_histogram
    """
    prices_array = _to_numpy_array(prices)

    if TALIB_AVAILABLE:
        try:
            macd_line, macd_signal, macd_hist = talib.MACD(
                prices_array,
                fastperiod=fast_period,
                slowperiod=slow_period,
                signalperiod=signal_period,
            )

            return {
                "macd_line": [
                    float(val) if not np.isnan(val) else None for val in macd_line
                ],
                "macd_signal": [
                    float(val) if not np.isnan(val) else None for val in macd_signal
                ],
                "macd_histogram": [
                    float(val) if not np.isnan(val) else None for val in macd_hist
                ],
            }
        except Exception as e:
            logger.warning(f"TA-Lib MACD series error: {e}, using fallback")

    # Manual calculation
    return _calculate_macd_series_manual(
        prices_array, fast_period, slow_period, signal_period
    )


def calculate_macd_incremental(
    current_price: float,
    prev_ema_fast: Optional[float],
    prev_ema_slow: Optional[float],
    prev_macd_signal: Optional[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Dict[str, Optional[float]]:
    """
    Calculate MACD incrementally using previous values.

    This method is efficient for real-time updates as it doesn't
    recalculate from the entire price history.

    Args:
        current_price: Current price
        prev_ema_fast: Previous fast EMA
        prev_ema_slow: Previous slow EMA
        prev_macd_signal: Previous MACD signal line
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal EMA period

    Returns:
        Dictionary with new MACD values and updated EMAs
    """
    # Calculate new EMAs
    new_ema_fast = calculate_ema_incremental(current_price, prev_ema_fast, fast_period)
    new_ema_slow = calculate_ema_incremental(current_price, prev_ema_slow, slow_period)

    # Calculate MACD line
    macd_line = new_ema_fast - new_ema_slow

    # Calculate signal line (EMA of MACD)
    if prev_macd_signal is None:
        macd_signal = macd_line
    else:
        macd_signal = calculate_ema_incremental(
            macd_line, prev_macd_signal, signal_period
        )

    # Calculate histogram
    macd_histogram = macd_line - macd_signal

    return {
        "macd_line": round(macd_line, 6),
        "macd_signal": round(macd_signal, 6),
        "macd_histogram": round(macd_histogram, 6),
        "ema_fast": new_ema_fast,
        "ema_slow": new_ema_slow,
    }


def calculate_ppo(
    prices: Union[List[float], np.ndarray, pd.Series],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Dict[str, Optional[float]]:
    """
    Calculate Percentage Price Oscillator (PPO).

    PPO is similar to MACD but shows the percentage difference between EMAs.

    Args:
        prices: Price series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal EMA period

    Returns:
        Dictionary with ppo_line, ppo_signal, ppo_histogram
    """
    prices_array = _to_numpy_array(prices)
    min_required = max(slow_period, fast_period) + signal_period

    if len(prices_array) < min_required:
        return {"ppo_line": None, "ppo_signal": None, "ppo_histogram": None}

    if TALIB_AVAILABLE:
        try:
            # PPO retourne seulement la ligne PPO, pas signal ni histogram
            ppo_line = talib.PPO(
                prices_array, fastperiod=fast_period, slowperiod=slow_period
            )

            # Calculer signal et histogram manuellement
            if len(ppo_line) >= signal_period:
                ppo_signal = talib.EMA(ppo_line, timeperiod=signal_period)
                ppo_hist = ppo_line - ppo_signal

                return {
                    "ppo_line": (
                        float(ppo_line[-1]) if not np.isnan(ppo_line[-1]) else None
                    ),
                    "ppo_signal": (
                        float(ppo_signal[-1])
                        if len(ppo_signal) > 0 and not np.isnan(ppo_signal[-1])
                        else None
                    ),
                    "ppo_histogram": (
                        float(ppo_hist[-1])
                        if len(ppo_hist) > 0 and not np.isnan(ppo_hist[-1])
                        else None
                    ),
                }
            else:
                return {
                    "ppo_line": (
                        float(ppo_line[-1]) if not np.isnan(ppo_line[-1]) else None
                    ),
                    "ppo_signal": None,
                    "ppo_histogram": None,
                }
        except Exception as e:
            logger.warning(f"TA-Lib PPO error: {e}, using fallback")

    # Manual calculation
    ema_fast = calculate_ema(prices_array, fast_period)
    ema_slow = calculate_ema(prices_array, slow_period)

    if ema_fast is None or ema_slow is None or ema_slow == 0:
        return {"ppo_line": None, "ppo_signal": None, "ppo_histogram": None}

    # PPO line as percentage
    ppo_line: float = float(((ema_fast - ema_slow) / ema_slow) * 100)

    # Need to calculate PPO series for signal line
    ppo_series: List[float] = []
    ema_fast_series = calculate_ema_series(prices_array, fast_period)
    ema_slow_series = calculate_ema_series(prices_array, slow_period)

    for i in range(len(ema_fast_series)):
        fast_val = ema_fast_series[i]
        slow_val = ema_slow_series[i]
        if fast_val is not None and slow_val is not None and slow_val != 0:
            ppo_val = ((fast_val - slow_val) / slow_val) * 100
            ppo_series.append(float(ppo_val))
        else:
            ppo_series.append(0.0)

    # Signal line (EMA of PPO)
    ppo_signal: Optional[float] = calculate_ema([x for x in ppo_series if x is not None], signal_period)

    if ppo_signal is None:
        return {"ppo_line": ppo_line, "ppo_signal": None, "ppo_histogram": None}

    ppo_histogram: float = float(ppo_line - ppo_signal)

    return {
        "ppo_line": round(ppo_line, 4),
        "ppo_signal": round(ppo_signal, 4),
        "ppo_histogram": round(ppo_histogram, 4),
    }


def macd_signal_cross(
    macd_values: Dict[str, Optional[float]],
    prev_macd_values: Dict[str, Optional[float]],
) -> str:
    """
    Detect MACD signal line crossovers.

    Args:
        macd_values: Current MACD values
        prev_macd_values: Previous MACD values

    Returns:
        'bullish' for MACD crossing above signal
        'bearish' for MACD crossing below signal
        'none' for no crossover
    """
    if any(
        v is None
        for v in [
            macd_values.get("macd_line"),
            macd_values.get("macd_signal"),
            prev_macd_values.get("macd_line"),
            prev_macd_values.get("macd_signal"),
        ]
    ):
        return "none"

    curr_line = macd_values["macd_line"]
    curr_signal = macd_values["macd_signal"]
    prev_line = prev_macd_values["macd_line"]
    prev_signal = prev_macd_values["macd_signal"]

    # Vérifications None avant comparaisons
    if (
        prev_line is not None
        and prev_signal is not None
        and curr_line is not None
        and curr_signal is not None
    ):

        # Bullish crossover
        if prev_line <= prev_signal and curr_line > curr_signal:
            return "bullish"

        # Bearish crossover
        if prev_line >= prev_signal and curr_line < curr_signal:
            return "bearish"

    return "none"


def macd_zero_cross(
    macd_values: Dict[str, Optional[float]],
    prev_macd_values: Dict[str, Optional[float]],
) -> str:
    """
    Detect MACD zero line crossovers.

    Args:
        macd_values: Current MACD values
        prev_macd_values: Previous MACD values

    Returns:
        'bullish' for MACD crossing above zero
        'bearish' for MACD crossing below zero
        'none' for no crossover
    """
    if (
        macd_values.get("macd_line") is None
        or prev_macd_values.get("macd_line") is None
    ):
        return "none"

    curr_line = macd_values["macd_line"]
    prev_line = prev_macd_values["macd_line"]

    # Vérifications None pour zero cross
    if prev_line is not None and curr_line is not None:
        # Bullish zero cross
        if prev_line <= 0 and curr_line > 0:
            return "bullish"

        # Bearish zero cross
        if prev_line >= 0 and curr_line < 0:
            return "bearish"

    return "none"


def calculate_macd_trend(
    macd_values: Dict[str, Optional[float]],
    prev_macd_values: Optional[Dict[str, Optional[float]]] = None,
) -> str:
    """
    Determine MACD trend direction based on MACD line and signal line.

    Args:
        macd_values: Current MACD values (macd_line, macd_signal, macd_histogram)
        prev_macd_values: Previous MACD values for trend analysis

    Returns:
        'BULLISH' for upward trend, 'BEARISH' for downward trend, 'NEUTRAL' for no clear trend
    """
    if (
        not macd_values
        or macd_values.get("macd_line") is None
        or macd_values.get("macd_signal") is None
    ):
        return "NEUTRAL"

    macd_line_val = macd_values["macd_line"]
    signal_line_val = macd_values["macd_signal"]
    histogram = macd_values.get("macd_histogram")

    # Assert non-None for type checker
    assert macd_line_val is not None
    assert signal_line_val is not None

    macd_line: float = float(macd_line_val)
    signal_line: float = float(signal_line_val)

    # Primary trend signals
    bullish_signals: float = 0.0
    bearish_signals: float = 0.0

    # 1. MACD line above/below signal line
    if macd_line > signal_line:
        bullish_signals += 1
    elif macd_line < signal_line:
        bearish_signals += 1

    # 2. MACD line above/below zero
    if macd_line > 0:
        bullish_signals += 1
    elif macd_line < 0:
        bearish_signals += 1

    # 3. Histogram analysis (if available)
    if histogram is not None:
        if histogram > 0:
            bullish_signals += 1
        elif histogram < 0:
            bearish_signals += 1

    # 4. Momentum analysis with previous values
    if prev_macd_values and prev_macd_values.get("macd_line") is not None:
        prev_macd_val = prev_macd_values["macd_line"]
        assert prev_macd_val is not None
        prev_macd: float = float(prev_macd_val)

        # MACD line direction
        if macd_line > prev_macd:
            bullish_signals += 1
        elif macd_line < prev_macd:
            bearish_signals += 1

        # Signal line comparison with previous
        if prev_macd_values.get("macd_signal") is not None:
            prev_signal_val = prev_macd_values["macd_signal"]
            assert prev_signal_val is not None
            prev_signal: float = float(prev_signal_val)
            if signal_line > prev_signal:
                bullish_signals += 0.5
            elif signal_line < prev_signal:
                bearish_signals += 0.5

    # Determine trend based on signal strength
    if bullish_signals > bearish_signals + 1:  # Need clear majority
        return "BULLISH"
    elif bearish_signals > bullish_signals + 1:
        return "BEARISH"
    else:
        return "NEUTRAL"


# ============ Helper Functions ============


def _to_numpy_array(data: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
    """Convert input data to numpy array."""
    if isinstance(data, pd.Series):
        return np.asarray(data.values, dtype=float)
    elif isinstance(data, list):
        return np.array(data, dtype=float)
    return np.asarray(data, dtype=float)


def _calculate_macd_manual(
    prices: np.ndarray,
    fast_period: int,
    slow_period: int,
    signal_period: int,
    normalize_high_price: bool = True,
) -> Dict[str, Optional[float]]:
    """Manual MACD calculation."""
    # Calculate EMAs
    ema_fast = calculate_ema(prices, fast_period)
    ema_slow = calculate_ema(prices, slow_period)

    if ema_fast is None or ema_slow is None:
        return {"macd_line": None, "macd_signal": None, "macd_histogram": None}

    # MACD line
    macd_line = ema_fast - ema_slow

    # Normalisation pour les actifs à prix élevé
    normalization_factor = 1.0
    if normalize_high_price and len(prices) > 0:
        avg_price = np.mean(
            prices[-min(50, len(prices)) :]
        )  # Moyenne des 50 derniers prix
        if avg_price > 10000:  # Pour BTC et autres cryptos à prix élevé
            normalization_factor = avg_price / 1000  # Ramener à une échelle raisonnable
        elif avg_price > 1000:
            normalization_factor = avg_price / 100

    # Appliquer la normalisation
    macd_line = macd_line / normalization_factor

    # For signal line, need MACD series
    macd_series = []
    ema_fast_series = calculate_ema_series(prices, fast_period)
    ema_slow_series = calculate_ema_series(prices, slow_period)

    for i in range(len(ema_fast_series)):
        fast_val = ema_fast_series[i]
        slow_val = ema_slow_series[i]
        if fast_val is not None and slow_val is not None:
            macd_val = (fast_val - slow_val) / normalization_factor
            macd_series.append(macd_val)
        else:
            macd_series.append(0.0)

    # Signal line (EMA of MACD)
    macd_signal = calculate_ema(
        [x for x in macd_series if x is not None], signal_period
    )

    if macd_signal is None:
        return {
            "macd_line": round(macd_line, 6),
            "macd_signal": None,
            "macd_histogram": None,
        }

    # Histogram
    macd_histogram = macd_line - macd_signal

    return {
        "macd_line": round(macd_line, 6),
        "macd_signal": round(macd_signal, 6),
        "macd_histogram": round(macd_histogram, 6),
    }


def _calculate_macd_series_manual(
    prices: np.ndarray, fast_period: int, slow_period: int, signal_period: int
) -> Dict[str, List[Optional[float]]]:
    """Manual MACD series calculation."""
    # Calculate EMA series
    ema_fast_series = calculate_ema_series(prices, fast_period)
    ema_slow_series = calculate_ema_series(prices, slow_period)

    # Calculate MACD line series
    macd_line_series: List[Optional[float]] = []
    for i in range(len(prices)):
        fast_val = ema_fast_series[i]
        slow_val = ema_slow_series[i]
        if fast_val is not None and slow_val is not None:
            macd_line_series.append(fast_val - slow_val)
        else:
            macd_line_series.append(0.0)

    # Calculate signal line series
    macd_signal_series: List[Optional[float]] = [0.0] * len(prices)
    valid_macd = [(i, x) for i, x in enumerate(macd_line_series) if x is not None]

    if len(valid_macd) >= signal_period:
        # Extract valid MACD values
        valid_indices = [x[0] for x in valid_macd]
        valid_values = [x[1] for x in valid_macd]

        # Calculate signal EMA on valid values
        signal_values = calculate_ema_series(valid_values, signal_period)

        # Map back to original indices
        for i, idx in enumerate(valid_indices):
            if i < len(signal_values) and signal_values[i] is not None:
                macd_signal_series[idx] = signal_values[i]
            else:
                macd_signal_series[idx] = 0.0

    # Calculate histogram series
    macd_histogram_series: List[Optional[float]] = []
    for i in range(len(prices)):
        line_val = macd_line_series[i]
        signal_val = macd_signal_series[i]
        if line_val is not None and signal_val is not None:
            macd_histogram_series.append(line_val - signal_val)
        else:
            macd_histogram_series.append(0.0)

    return {
        "macd_line": macd_line_series,
        "macd_signal": macd_signal_series,
        "macd_histogram": macd_histogram_series,
    }
