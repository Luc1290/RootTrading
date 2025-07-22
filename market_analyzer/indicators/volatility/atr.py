"""
ATR (Average True Range) Indicator

This module provides ATR calculation for measuring volatility.
ATR is commonly used for:
- Volatility measurement
- Stop loss placement
- Position sizing
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.debug("TA-Lib not available, using manual calculations")


def calculate_true_range(high: float, low: float, prev_close: Optional[float] = None) -> float:
    """
    Calculate True Range for a single period.
    
    True Range is the greatest of:
    - Current High - Current Low
    - |Current High - Previous Close|
    - |Current Low - Previous Close|
    
    Args:
        high: Current high price
        low: Current low price
        prev_close: Previous close price (optional)
        
    Returns:
        True Range value
    """
    tr1 = high - low
    
    if prev_close is None:
        return float(tr1)
    
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    return float(max(tr1, tr2, tr3))


def calculate_atr(highs: Union[List[float], np.ndarray, pd.Series],
                  lows: Union[List[float], np.ndarray, pd.Series],
                  closes: Union[List[float], np.ndarray, pd.Series],
                  period: int = 14) -> Optional[float]:
    """
    Calculate Average True Range (ATR).
    
    ATR is an exponential moving average of True Range values.
    
    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: Period for ATR calculation (default: 14)
        
    Returns:
        ATR value or None if insufficient data
        
    Notes:
        - Higher ATR indicates higher volatility
        - Lower ATR indicates lower volatility
        - Useful for dynamic stop loss and position sizing
    """
    highs_array = _to_numpy_array(highs)
    lows_array = _to_numpy_array(lows)
    closes_array = _to_numpy_array(closes)
    
    if len(highs_array) < period + 1:
        return None
    
    # Ensure all arrays have same length
    min_len = min(len(highs_array), len(lows_array), len(closes_array))
    highs_array = highs_array[-min_len:]
    lows_array = lows_array[-min_len:]
    closes_array = closes_array[-min_len:]
    
    if TALIB_AVAILABLE:
        try:
            atr_values = talib.ATR(highs_array, lows_array, closes_array, timeperiod=period)
            return float(atr_values[-1]) if not np.isnan(atr_values[-1]) else None
        except Exception as e:
            logger.warning(f"TA-Lib ATR error: {e}, using fallback")
    
    return _calculate_atr_manual(highs_array, lows_array, closes_array, period)


def calculate_atr_series(highs: Union[List[float], np.ndarray, pd.Series],
                         lows: Union[List[float], np.ndarray, pd.Series],
                         closes: Union[List[float], np.ndarray, pd.Series],
                         period: int = 14) -> List[Optional[float]]:
    """
    Calculate ATR for entire price series.
    
    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: ATR period
        
    Returns:
        List of ATR values (None for insufficient data points)
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
            atr_values = talib.ATR(highs_array, lows_array, closes_array, timeperiod=period)
            return [float(val) if not np.isnan(val) else None for val in atr_values]
        except Exception as e:
            logger.warning(f"TA-Lib ATR series error: {e}, using fallback")
    
    # Manual calculation
    return _calculate_atr_series_manual(highs_array, lows_array, closes_array, period)


def calculate_atr_incremental(current_high: float,
                             current_low: float,
                             current_close: float,
                             prev_close: float,
                             prev_atr: float,
                             period: int = 14) -> float:
    """
    Calculate ATR incrementally using previous ATR value.
    
    This is efficient for real-time updates.
    
    Args:
        current_high: Current high price
        current_low: Current low price
        current_close: Current close price
        prev_close: Previous close price
        prev_atr: Previous ATR value
        period: ATR period
        
    Returns:
        New ATR value
    """
    # Calculate current True Range
    tr = calculate_true_range(current_high, current_low, prev_close)
    
    # Apply Wilder's smoothing
    # ATR = ((prev_ATR * (period - 1)) + TR) / period
    new_atr = ((prev_atr * (period - 1)) + tr) / period
    
    return float(new_atr)


def calculate_atr_percent(highs: Union[List[float], np.ndarray, pd.Series],
                         lows: Union[List[float], np.ndarray, pd.Series],
                         closes: Union[List[float], np.ndarray, pd.Series],
                         period: int = 14) -> Optional[float]:
    """
    Calculate ATR as a percentage of current price.
    
    This normalizes ATR across different price levels.
    
    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: ATR period
        
    Returns:
        ATR percentage or None
    """
    atr = calculate_atr(highs, lows, closes, period)
    if atr is None:
        return None
    
    current_price = float(closes[-1])
    if current_price == 0:
        return None
    
    return (atr / current_price) * 100


def calculate_natr(highs: Union[List[float], np.ndarray, pd.Series],
                   lows: Union[List[float], np.ndarray, pd.Series],
                   closes: Union[List[float], np.ndarray, pd.Series],
                   period: int = 14) -> Optional[float]:
    """
    Calculate Normalized ATR (NATR).
    
    NATR = (ATR / Close) * 100
    
    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: ATR period
        
    Returns:
        NATR value or None
    """
    if TALIB_AVAILABLE:
        try:
            highs_array = _to_numpy_array(highs)
            lows_array = _to_numpy_array(lows)
            closes_array = _to_numpy_array(closes)
            
            natr_values = talib.NATR(highs_array, lows_array, closes_array, timeperiod=period)
            return float(natr_values[-1]) if not np.isnan(natr_values[-1]) else None
        except Exception as e:
            logger.warning(f"TA-Lib NATR error: {e}, using fallback")
    
    # Fallback to manual calculation
    return calculate_atr_percent(highs, lows, closes, period)


def calculate_atr_stop_loss(price: float,
                           atr: float,
                           multiplier: float = 2.0,
                           is_long: bool = True) -> float:
    """
    Calculate stop loss based on ATR.
    
    Args:
        price: Current price
        atr: Current ATR value
        multiplier: ATR multiplier for stop distance
        is_long: True for long position, False for short
        
    Returns:
        Stop loss price
    """
    stop_distance = atr * multiplier
    
    if is_long:
        return price - stop_distance
    else:
        return price + stop_distance


def calculate_atr_bands(prices: Union[List[float], np.ndarray, pd.Series],
                       highs: Union[List[float], np.ndarray, pd.Series],
                       lows: Union[List[float], np.ndarray, pd.Series],
                       period: int = 14,
                       multiplier: float = 2.0) -> Dict[str, Optional[float]]:
    """
    Calculate ATR-based bands around price.
    
    Args:
        prices: Close prices
        highs: High prices
        lows: Low prices
        period: ATR period
        multiplier: Band distance multiplier
        
    Returns:
        Dictionary with upper, middle, lower bands
    """
    from typing import Dict
    
    atr = calculate_atr(highs, lows, prices, period)
    if atr is None:
        return {
            'upper': None,
            'middle': None,
            'lower': None
        }
    
    current_price = float(prices[-1])
    band_distance = atr * multiplier
    
    return {
        'upper': current_price + band_distance,
        'middle': current_price,
        'lower': current_price - band_distance
    }


def volatility_regime(highs: Union[List[float], np.ndarray, pd.Series],
                      lows: Union[List[float], np.ndarray, pd.Series],
                      closes: Union[List[float], np.ndarray, pd.Series],
                      period: int = 14,
                      lookback: int = 50) -> str:
    """
    Determine current volatility regime based on ATR.
    
    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: ATR period
        lookback: Period for regime comparison
        
    Returns:
        'low', 'normal', 'high', or 'extreme'
    """
    atr_series = calculate_atr_series(highs, lows, closes, period)
    valid_atr = [x for x in atr_series if x is not None]
    
    if len(valid_atr) < lookback:
        return 'normal'
    
    current_atr = valid_atr[-1]
    recent_atr = valid_atr[-lookback:]
    
    # Calculate percentiles
    p25 = np.percentile(recent_atr, 25)
    p50 = np.percentile(recent_atr, 50)
    p75 = np.percentile(recent_atr, 75)
    p90 = np.percentile(recent_atr, 90)
    
    if current_atr <= p25:
        return 'low'
    elif current_atr <= p50:
        return 'normal'
    elif current_atr <= p90:
        return 'high'
    else:
        return 'extreme'


# ============ Helper Functions ============

def _to_numpy_array(data: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
    """Convert input data to numpy array."""
    if isinstance(data, pd.Series):
        return data.values
    elif isinstance(data, list):
        return np.array(data, dtype=float)
    return np.asarray(data, dtype=float)


def _calculate_atr_manual(highs: np.ndarray,
                         lows: np.ndarray,
                         closes: np.ndarray,
                         period: int) -> Optional[float]:
    """Manual ATR calculation using Wilder's method."""
    if len(highs) < period + 1:
        return None
    
    # Calculate True Range values
    tr_values = []
    for i in range(1, len(highs)):
        tr = calculate_true_range(
            float(highs[i]),
            float(lows[i]),
            float(closes[i-1])
        )
        tr_values.append(tr)
    
    if len(tr_values) < period:
        return None
    
    # Initial ATR (simple average)
    atr = np.mean(tr_values[:period])
    
    # Apply Wilder's smoothing for remaining values
    for i in range(period, len(tr_values)):
        atr = ((atr * (period - 1)) + tr_values[i]) / period
    
    return float(atr)


def _calculate_atr_series_manual(highs: np.ndarray,
                                lows: np.ndarray,
                                closes: np.ndarray,
                                period: int) -> List[Optional[float]]:
    """Manual ATR series calculation."""
    atr_series = []
    
    # Not enough data for first period
    if len(highs) < period + 1:
        return [None] * len(highs)
    
    # Fill initial values with None
    for i in range(period):
        atr_series.append(None)
    
    # Calculate True Range values
    tr_values = []
    for i in range(1, len(highs)):
        tr = calculate_true_range(
            float(highs[i]),
            float(lows[i]),
            float(closes[i-1])
        )
        tr_values.append(tr)
    
    # Initial ATR (simple average of first 'period' TR values)
    if len(tr_values) >= period:
        atr = np.mean(tr_values[:period])
        atr_series.append(float(atr))
        
        # Apply Wilder's smoothing for remaining values
        for i in range(period, len(tr_values)):
            atr = ((atr * (period - 1)) + tr_values[i]) / period
            atr_series.append(float(atr))
    
    return atr_series