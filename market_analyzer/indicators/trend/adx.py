"""
ADX (Average Directional Index) Indicator

This module provides ADX calculation for measuring trend strength.
Includes:
- ADX (trend strength)
- +DI (positive directional indicator)
- -DI (negative directional indicator)
- DX (directional movement index)
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


def calculate_adx(highs: Union[List[float], np.ndarray, pd.Series],
                  lows: Union[List[float], np.ndarray, pd.Series],
                  closes: Union[List[float], np.ndarray, pd.Series],
                  period: int = 14) -> Optional[float]:
    """
    Calculate Average Directional Index (ADX).
    
    ADX measures trend strength regardless of direction.
    
    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices (used for ATR calculation)
        period: Period for ADX calculation (default: 14)
        
    Returns:
        ADX value or None if insufficient data
        
    Notes:
        - ADX < 25: Weak trend or ranging market
        - ADX 25-50: Strong trend
        - ADX > 50: Very strong trend
        - ADX > 75: Extremely strong trend
    """
    highs_array = _to_numpy_array(highs)
    lows_array = _to_numpy_array(lows)
    closes_array = _to_numpy_array(closes)
    
    # Need at least 2*period for proper ADX calculation
    if len(highs_array) < period * 2:
        return None
    
    # Ensure all arrays have same length
    min_len = min(len(highs_array), len(lows_array), len(closes_array))
    highs_array = highs_array[-min_len:]
    lows_array = lows_array[-min_len:]
    closes_array = closes_array[-min_len:]
    
    if TALIB_AVAILABLE:
        try:
            adx_values = talib.ADX(highs_array, lows_array, closes_array, timeperiod=period)
            return float(adx_values[-1]) if not np.isnan(adx_values[-1]) else None
        except Exception as e:
            logger.warning(f"TA-Lib ADX error: {e}, using fallback")
    
    return _calculate_adx_manual(highs_array, lows_array, closes_array, period)


def calculate_adx_full(highs: Union[List[float], np.ndarray, pd.Series],
                       lows: Union[List[float], np.ndarray, pd.Series],
                       closes: Union[List[float], np.ndarray, pd.Series],
                       period: int = 14) -> Dict[str, Optional[float]]:
    """
    Calculate full ADX indicator set including +DI and -DI.
    
    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: Period for calculation
        
    Returns:
        Dictionary with:
        - adx: Average Directional Index
        - plus_di: Positive Directional Indicator
        - minus_di: Negative Directional Indicator
        - dx: Directional Movement Index
        - adxr: Average Directional Movement Index Rating
    """
    highs_array = _to_numpy_array(highs)
    lows_array = _to_numpy_array(lows)
    closes_array = _to_numpy_array(closes)
    
    if len(highs_array) < period * 2:
        return {
            'adx': None,
            'plus_di': None,
            'minus_di': None,
            'dx': None,
            'adxr': None
        }
    
    # Ensure all arrays have same length
    min_len = min(len(highs_array), len(lows_array), len(closes_array))
    highs_array = highs_array[-min_len:]
    lows_array = lows_array[-min_len:]
    closes_array = closes_array[-min_len:]
    
    if TALIB_AVAILABLE:
        try:
            adx = talib.ADX(highs_array, lows_array, closes_array, timeperiod=period)
            plus_di = talib.PLUS_DI(highs_array, lows_array, closes_array, timeperiod=period)
            minus_di = talib.MINUS_DI(highs_array, lows_array, closes_array, timeperiod=period)
            dx = talib.DX(highs_array, lows_array, closes_array, timeperiod=period)
            adxr = talib.ADXR(highs_array, lows_array, closes_array, timeperiod=period)
            
            return {
                'adx': float(adx[-1]) if not np.isnan(adx[-1]) else None,
                'plus_di': float(plus_di[-1]) if not np.isnan(plus_di[-1]) else None,
                'minus_di': float(minus_di[-1]) if not np.isnan(minus_di[-1]) else None,
                'dx': float(dx[-1]) if not np.isnan(dx[-1]) else None,
                'adxr': float(adxr[-1]) if not np.isnan(adxr[-1]) else None
            }
        except Exception as e:
            logger.warning(f"TA-Lib ADX full error: {e}, using fallback")
    
    return _calculate_adx_full_manual(highs_array, lows_array, closes_array, period)


def calculate_dmi(highs: Union[List[float], np.ndarray, pd.Series],
                  lows: Union[List[float], np.ndarray, pd.Series],
                  closes: Union[List[float], np.ndarray, pd.Series],
                  period: int = 14) -> Dict[str, Optional[float]]:
    """
    Calculate Directional Movement Indicators (+DI and -DI).
    
    DMI is used to identify trend direction.
    
    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: Period for calculation
        
    Returns:
        Dictionary with plus_di and minus_di
    """
    result = calculate_adx_full(highs, lows, closes, period)
    return {
        'plus_di': result['plus_di'],
        'minus_di': result['minus_di']
    }


def calculate_adx_series(highs: Union[List[float], np.ndarray, pd.Series],
                         lows: Union[List[float], np.ndarray, pd.Series],
                         closes: Union[List[float], np.ndarray, pd.Series],
                         period: int = 14) -> Dict[str, List[Optional[float]]]:
    """
    Calculate ADX indicator series for entire price history.
    
    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: Period for calculation
        
    Returns:
        Dictionary with lists for adx, plus_di, minus_di
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
            adx = talib.ADX(highs_array, lows_array, closes_array, timeperiod=period)
            plus_di = talib.PLUS_DI(highs_array, lows_array, closes_array, timeperiod=period)
            minus_di = talib.MINUS_DI(highs_array, lows_array, closes_array, timeperiod=period)
            
            return {
                'adx': [float(val) if not np.isnan(val) else None for val in adx],
                'plus_di': [float(val) if not np.isnan(val) else None for val in plus_di],
                'minus_di': [float(val) if not np.isnan(val) else None for val in minus_di]
            }
        except Exception as e:
            logger.warning(f"TA-Lib ADX series error: {e}, using fallback")
    
    # Manual calculation
    return _calculate_adx_series_manual(highs_array, lows_array, closes_array, period)


def adx_trend_strength(adx_value: Optional[float]) -> str:
    """
    Interpret ADX value for trend strength.
    
    Args:
        adx_value: Current ADX value
        
    Returns:
        Trend strength description
    """
    if adx_value is None:
        return 'unknown'
    
    if adx_value < 20:
        return 'absent'
    elif adx_value < 25:
        return 'weak'
    elif adx_value < 50:
        return 'strong'
    elif adx_value < 75:
        return 'very_strong'
    else:
        return 'extreme'


def dmi_signal(plus_di: Optional[float], 
               minus_di: Optional[float],
               adx: Optional[float] = None) -> str:
    """
    Generate trading signal based on DMI crossovers.
    
    Args:
        plus_di: Current +DI value
        minus_di: Current -DI value
        adx: Current ADX value (optional)
        
    Returns:
        'bullish', 'bearish', or 'neutral'
    """
    if plus_di is None or minus_di is None:
        return 'neutral'
    
    # Check trend strength if ADX provided
    if adx is not None and adx < 20:
        return 'neutral'  # No trend
    
    if plus_di > minus_di:
        return 'bullish'
    elif minus_di > plus_di:
        return 'bearish'
    
    return 'neutral'


def calculate_directional_bias(plus_di: Optional[float], 
                              minus_di: Optional[float],
                              adx: Optional[float] = None) -> str:
    """
    Calculate directional bias for market analysis.
    
    Returns uppercase values for consistency with market analyzer.
    
    Args:
        plus_di: Positive Directional Indicator value
        minus_di: Negative Directional Indicator value
        adx: ADX value (optional, for filtering weak trends)
        
    Returns:
        'BULLISH', 'BEARISH', or 'NEUTRAL'
    """
    signal = dmi_signal(plus_di, minus_di, adx)
    return signal.upper()


def calculate_adxr(highs: Union[List[float], np.ndarray, pd.Series],
                   lows: Union[List[float], np.ndarray, pd.Series],
                   closes: Union[List[float], np.ndarray, pd.Series],
                   period: int = 14) -> Optional[float]:
    """
    Calculate Average Directional Movement Index Rating (ADXR).
    
    ADXR = (ADX + ADX n periods ago) / 2
    
    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        period: Period for ADX calculation
        
    Returns:
        ADXR value or None
    """
    if TALIB_AVAILABLE:
        try:
            highs_array = _to_numpy_array(highs)
            lows_array = _to_numpy_array(lows)
            closes_array = _to_numpy_array(closes)
            
            adxr_values = talib.ADXR(highs_array, lows_array, closes_array, timeperiod=period)
            return float(adxr_values[-1]) if not np.isnan(adxr_values[-1]) else None
        except Exception as e:
            logger.warning(f"TA-Lib ADXR error: {e}, using fallback")
    
    # Manual calculation
    adx_series = calculate_adx_series(highs, lows, closes, period)['adx']
    valid_adx = [x for x in adx_series if x is not None]
    
    if len(valid_adx) < period:
        return None
    
    current_adx = valid_adx[-1]
    past_adx = valid_adx[-period]
    
    return (current_adx + past_adx) / 2


# ============ Helper Functions ============

def _to_numpy_array(data: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
    """Convert input data to numpy array."""
    if isinstance(data, pd.Series):
        if hasattr(data.values, 'values'):  # ExtensionArray
            return np.asarray(data.values, dtype=float)
        return np.asarray(data.values, dtype=float)
    elif isinstance(data, list):
        return np.array(data, dtype=float)
    return np.asarray(data, dtype=float)


def _calculate_directional_movement(highs: np.ndarray, 
                                   lows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate +DM and -DM arrays."""
    high_diff = np.diff(highs)
    low_diff = -np.diff(lows)
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    return plus_dm, minus_dm


def _calculate_adx_manual(highs: np.ndarray,
                         lows: np.ndarray,
                         closes: np.ndarray,
                         period: int) -> Optional[float]:
    """Manual ADX calculation."""
    if len(highs) < period * 2:
        return None
    
    # Calculate directional movements
    plus_dm, minus_dm = _calculate_directional_movement(highs, lows)
    
    # Calculate True Range
    from ..volatility.atr import calculate_atr_series
    tr_series = []
    for i in range(1, len(highs)):
        from ..volatility.atr import calculate_true_range
        tr = calculate_true_range(
            float(highs[i]),
            float(lows[i]),
            float(closes[i-1])
        )
        tr_series.append(tr)
    
    if len(tr_series) < period:
        return None
    
    # Smooth the indicators using Wilder's method
    atr = np.mean(tr_series[:period])
    plus_di_sum = np.sum(plus_dm[:period])
    minus_di_sum = np.sum(minus_dm[:period])
    
    dx_values = []
    
    for i in range(period, len(tr_series)):
        # Update sums
        atr = ((atr * (period - 1)) + tr_series[i]) / period
        plus_di_sum = ((plus_di_sum * (period - 1)) + plus_dm[i]) / period
        minus_di_sum = ((minus_di_sum * (period - 1)) + minus_dm[i]) / period
        
        # Calculate DI values
        plus_di = (plus_di_sum / atr) * 100 if atr != 0 else 0
        minus_di = (minus_di_sum / atr) * 100 if atr != 0 else 0
        
        # Calculate DX
        di_sum = plus_di + minus_di
        if di_sum != 0:
            dx = (abs(plus_di - minus_di) / di_sum) * 100
            dx_values.append(dx)
    
    if len(dx_values) < period:
        return None
    
    # Calculate ADX (average of DX)
    adx = np.mean(dx_values[:period])
    
    # Smooth ADX
    for i in range(period, len(dx_values)):
        adx = ((adx * (period - 1)) + dx_values[i]) / period
    
    return float(adx)


def _calculate_adx_full_manual(highs: np.ndarray,
                               lows: np.ndarray,
                               closes: np.ndarray,
                               period: int) -> Dict[str, Optional[float]]:
    """Manual calculation of full ADX indicators."""
    if len(highs) < period * 2:
        return {
            'adx': None,
            'plus_di': None,
            'minus_di': None,
            'dx': None,
            'adxr': None
        }
    
    # Calculate directional movements
    plus_dm, minus_dm = _calculate_directional_movement(highs, lows)
    
    # Calculate True Range
    tr_series = []
    for i in range(1, len(highs)):
        from ..volatility.atr import calculate_true_range
        tr = calculate_true_range(
            float(highs[i]),
            float(lows[i]),
            float(closes[i-1])
        )
        tr_series.append(tr)
    
    if len(tr_series) < period:
        return {
            'adx': None,
            'plus_di': None,
            'minus_di': None,
            'dx': None,
            'adxr': None
        }
    
    # Initialize with first period values
    atr = np.mean(tr_series[:period])
    plus_di_sum = np.sum(plus_dm[:period])
    minus_di_sum = np.sum(minus_dm[:period])
    
    # Smooth values
    for i in range(period, len(tr_series)):
        atr = ((atr * (period - 1)) + tr_series[i]) / period
        plus_di_sum = ((plus_di_sum * (period - 1)) + plus_dm[i]) / period
        minus_di_sum = ((minus_di_sum * (period - 1)) + minus_dm[i]) / period
    
    # Calculate final DI values
    plus_di = (plus_di_sum / atr) * 100 if atr != 0 else 0
    minus_di = (minus_di_sum / atr) * 100 if atr != 0 else 0
    
    # Calculate DX
    di_sum = plus_di + minus_di
    if di_sum != 0:
        dx = (abs(plus_di - minus_di) / di_sum) * 100
    else:
        dx = 0
    
    # Calculate ADX (would need more history for accurate ADX)
    adx = _calculate_adx_manual(highs, lows, closes, period)
    
    # Calculate ADXR manually
    adxr = calculate_adxr(highs, lows, closes, period)
    
    return {
        'adx': adx,
        'plus_di': float(plus_di),
        'minus_di': float(minus_di),
        'dx': float(dx),
        'adxr': adxr
    }


def _calculate_adx_series_manual(highs: np.ndarray,
                                lows: np.ndarray,
                                closes: np.ndarray,
                                period: int) -> Dict[str, List[Optional[float]]]:
    """Manual ADX series calculation."""
    adx_series: List[Optional[float]] = []
    plus_di_series: List[Optional[float]] = []
    minus_di_series: List[Optional[float]] = []
    
    # Need minimum data
    min_required = period * 2
    
    for i in range(len(highs)):
        if i < min_required:
            adx_series.append(None)
            plus_di_series.append(None)
            minus_di_series.append(None)
        else:
            # Calculate for current window
            window_highs = highs[:i+1]
            window_lows = lows[:i+1]
            window_closes = closes[:i+1]
            
            result = _calculate_adx_full_manual(
                window_highs, window_lows, window_closes, period
            )
            
            adx_series.append(result['adx'])
            plus_di_series.append(result['plus_di'])
            minus_di_series.append(result['minus_di'])
    
    return {
        'adx': adx_series,
        'plus_di': plus_di_series,
        'minus_di': minus_di_series
    }