"""
Stochastic Oscillator Indicator

This module provides Stochastic oscillator calculation including:
- %K (Fast Stochastic)
- %D (Slow Stochastic - smoothed %K)
- Fast and Slow variations
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


def calculate_stochastic(highs: Union[List[float], np.ndarray, pd.Series],
                        lows: Union[List[float], np.ndarray, pd.Series],
                        closes: Union[List[float], np.ndarray, pd.Series],
                        k_period: int = 14,
                        k_smooth: int = 1,
                        d_period: int = 3) -> Dict[str, Optional[float]]:
    """
    Calculate Stochastic Oscillator.
    
    The Stochastic Oscillator measures the position of a closing price
    relative to the high-low range over a specified period.
    
    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        k_period: Lookback period for %K calculation (default: 14)
        k_smooth: Smoothing period for %K (default: 1 for fast stochastic)
        d_period: Smoothing period for %D (default: 3)
        
    Returns:
        Dictionary with:
        - k: %K value (fast or smoothed)
        - d: %D value (smoothed %K)
        
    Notes:
        - Values range from 0 to 100
        - %K > 80: Overbought conditions
        - %K < 20: Oversold conditions
        - %K crossing above %D: Bullish signal
        - %K crossing below %D: Bearish signal
    """
    highs_array = _to_numpy_array(highs)
    lows_array = _to_numpy_array(lows)
    closes_array = _to_numpy_array(closes)
    
    min_required = k_period + k_smooth + d_period - 2
    if len(highs_array) < min_required:
        return {'k': None, 'd': None}
    
    # Ensure all arrays have same length
    min_len = min(len(highs_array), len(lows_array), len(closes_array))
    highs_array = highs_array[-min_len:]
    lows_array = lows_array[-min_len:]
    closes_array = closes_array[-min_len:]
    
    if TALIB_AVAILABLE:
        try:
            k_values, d_values = talib.STOCH(
                highs_array, lows_array, closes_array,
                fastk_period=k_period,
                slowk_period=k_smooth,
                slowd_period=d_period
            )
            
            return {
                'k': float(k_values[-1]) if not np.isnan(k_values[-1]) else None,
                'd': float(d_values[-1]) if not np.isnan(d_values[-1]) else None
            }
        except Exception as e:
            logger.warning(f"TA-Lib Stochastic error: {e}, using fallback")
    
    return _calculate_stochastic_manual(
        highs_array, lows_array, closes_array, k_period, k_smooth, d_period
    )


def calculate_stochastic_fast(highs: Union[List[float], np.ndarray, pd.Series],
                             lows: Union[List[float], np.ndarray, pd.Series],
                             closes: Union[List[float], np.ndarray, pd.Series],
                             k_period: int = 14,
                             d_period: int = 3) -> Dict[str, Optional[float]]:
    """
    Calculate Fast Stochastic Oscillator.
    
    Fast Stochastic uses unsmoothed %K values.
    
    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        k_period: Lookback period for %K
        d_period: Smoothing period for %D
        
    Returns:
        Dictionary with fast_k and fast_d
    """
    if TALIB_AVAILABLE:
        try:
            highs_array = _to_numpy_array(highs)
            lows_array = _to_numpy_array(lows)
            closes_array = _to_numpy_array(closes)
            
            k_values, d_values = talib.STOCHF(
                highs_array, lows_array, closes_array,
                fastk_period=k_period,
                fastd_period=d_period
            )
            
            return {
                'fast_k': float(k_values[-1]) if not np.isnan(k_values[-1]) else None,
                'fast_d': float(d_values[-1]) if not np.isnan(d_values[-1]) else None
            }
        except Exception as e:
            logger.warning(f"TA-Lib Fast Stochastic error: {e}, using fallback")
    
    # Use regular stochastic with k_smooth=1
    result = calculate_stochastic(highs, lows, closes, k_period, 1, d_period)
    return {
        'fast_k': result['k'],
        'fast_d': result['d']
    }


def calculate_stochastic_series(highs: Union[List[float], np.ndarray, pd.Series],
                               lows: Union[List[float], np.ndarray, pd.Series],
                               closes: Union[List[float], np.ndarray, pd.Series],
                               k_period: int = 14,
                               k_smooth: int = 1,
                               d_period: int = 3) -> Dict[str, List[Optional[float]]]:
    """
    Calculate Stochastic values for entire price series.
    
    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        k_period: %K period
        k_smooth: %K smoothing
        d_period: %D smoothing
        
    Returns:
        Dictionary with lists of k and d values
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
            k_values, d_values = talib.STOCH(
                highs_array, lows_array, closes_array,
                fastk_period=k_period,
                slowk_period=k_smooth,
                slowd_period=d_period
            )
            
            return {
                'k': [float(val) if not np.isnan(val) else None for val in k_values],
                'd': [float(val) if not np.isnan(val) else None for val in d_values]
            }
        except Exception as e:
            logger.warning(f"TA-Lib Stochastic series error: {e}, using fallback")
    
    # Manual calculation
    return _calculate_stochastic_series_manual(
        highs_array, lows_array, closes_array, k_period, k_smooth, d_period
    )


def stochastic_signal(k_current: Optional[float],
                     d_current: Optional[float],
                     k_previous: Optional[float],
                     d_previous: Optional[float]) -> str:
    """
    Generate trading signal based on Stochastic crossovers.
    
    Args:
        k_current: Current %K value
        d_current: Current %D value
        k_previous: Previous %K value
        d_previous: Previous %D value
        
    Returns:
        'bullish', 'bearish', 'overbought', 'oversold', or 'neutral'
    """
    if any(x is None for x in [k_current, d_current, k_previous, d_previous]):
        return 'neutral'
    
    # Assertions pour mypy - on sait que les valeurs ne sont pas None
    assert k_current is not None
    assert d_current is not None
    assert k_previous is not None
    assert d_previous is not None
    
    # Check for overbought/oversold conditions
    if k_current >= 80 and d_current >= 80:
        return 'overbought'
    elif k_current <= 20 and d_current <= 20:
        return 'oversold'
    
    # Check for crossovers
    # Bullish: %K crosses above %D
    if k_previous <= d_previous and k_current > d_current:
        return 'bullish'
    
    # Bearish: %K crosses below %D
    if k_previous >= d_previous and k_current < d_current:
        return 'bearish'
    
    return 'neutral'


def calculate_stochastic_divergence(prices: Union[List[float], np.ndarray, pd.Series],
                                   highs: Union[List[float], np.ndarray, pd.Series],
                                   lows: Union[List[float], np.ndarray, pd.Series],
                                   lookback: int = 20) -> str:
    """
    Detect bullish/bearish divergence in Stochastic oscillator.
    
    Args:
        prices: Close prices
        highs: High prices
        lows: Low prices
        lookback: Period to check for divergence
        
    Returns:
        'bullish_divergence', 'bearish_divergence', or 'none'
    """
    if len(prices) < lookback + 14:  # Need minimum data
        return 'none'
    
    # Calculate Stochastic series
    stoch_series = calculate_stochastic_series(highs, lows, prices)
    k_values = stoch_series['k']
    
    # Remove None values and get recent data
    valid_data = []
    for i in range(len(k_values)):
        if k_values[i] is not None:
            valid_data.append((i, float(prices[i]), k_values[i]))
    
    if len(valid_data) < lookback:
        return 'none'
    
    recent_data = valid_data[-lookback:]
    
    # Find local highs and lows in both price and stochastic
    price_highs = []
    price_lows = []
    stoch_highs = []
    stoch_lows = []
    
    for i in range(1, len(recent_data) - 1):
        idx, price, stoch = recent_data[i]
        prev_price = recent_data[i-1][1]
        next_price = recent_data[i+1][1]
        prev_stoch = recent_data[i-1][2]
        next_stoch = recent_data[i+1][2]
        
        # Price peaks and troughs
        if price > prev_price and price > next_price:
            price_highs.append((idx, price))
        elif price < prev_price and price < next_price:
            price_lows.append((idx, price))
        
        # Stochastic peaks and troughs
        if (stoch is not None and prev_stoch is not None and next_stoch is not None and
            stoch > prev_stoch and stoch > next_stoch):
            stoch_highs.append((idx, stoch))
        elif (stoch is not None and prev_stoch is not None and next_stoch is not None and
              stoch < prev_stoch and stoch < next_stoch):
            stoch_lows.append((idx, stoch))
    
    # Check for divergence
    # Bullish divergence: Price makes lower lows, Stochastic makes higher lows
    if len(price_lows) >= 2 and len(stoch_lows) >= 2:
        price_val1, price_val2 = price_lows[-1][1], price_lows[-2][1]
        stoch_val1, stoch_val2 = stoch_lows[-1][1], stoch_lows[-2][1]
        if (price_val1 is not None and price_val2 is not None and
            stoch_val1 is not None and stoch_val2 is not None and
            price_val1 < price_val2 and stoch_val1 > stoch_val2):
            return 'bullish_divergence'
    
    # Bearish divergence: Price makes higher highs, Stochastic makes lower highs
    if len(price_highs) >= 2 and len(stoch_highs) >= 2:
        price_val1, price_val2 = price_highs[-1][1], price_highs[-2][1]
        stoch_val1, stoch_val2 = stoch_highs[-1][1], stoch_highs[-2][1]
        if (price_val1 is not None and price_val2 is not None and
            stoch_val1 is not None and stoch_val2 is not None and
            price_val1 > price_val2 and stoch_val1 < stoch_val2):
            return 'bearish_divergence'
    
    return 'none'


# ============ Helper Functions ============

def _to_numpy_array(data: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
    """Convert input data to numpy array."""
    if isinstance(data, pd.Series):
        return np.asarray(data.values, dtype=float)
    elif isinstance(data, list):
        return np.array(data, dtype=float)
    return np.asarray(data, dtype=float)


def _calculate_raw_k(highs: np.ndarray,
                    lows: np.ndarray,
                    closes: np.ndarray,
                    period: int) -> List[Optional[float]]:
    """Calculate raw %K values."""
    k_values: List[Optional[float]] = []
    
    for i in range(len(closes)):
        if i < period - 1:
            k_values.append(None)
        else:
            # Get window
            high_window = highs[i-period+1:i+1]
            low_window = lows[i-period+1:i+1]
            current_close = closes[i]
            
            # Calculate %K
            highest_high = np.max(high_window)
            lowest_low = np.min(low_window)
            
            if highest_high == lowest_low:
                k = 50.0  # Neutral when no range
            else:
                k = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
            
            k_values.append(float(k))
    
    return k_values


def _smooth_series(values: List[Optional[float]], period: int) -> List[Optional[float]]:
    """Apply SMA smoothing to a series."""
    smoothed: List[Optional[float]] = []
    
    for i in range(len(values)):
        if values[i] is None:
            smoothed.append(None)
        elif i < period - 1:
            smoothed.append(values[i])  # Not enough data to smooth
        else:
            # Calculate SMA
            window_values = [x for x in values[i-period+1:i+1] if x is not None]
            if len(window_values) == period:
                smoothed.append(float(np.mean(window_values)))
            else:
                smoothed.append(values[i])
    
    return smoothed


def _calculate_stochastic_manual(highs: np.ndarray,
                                lows: np.ndarray,
                                closes: np.ndarray,
                                k_period: int,
                                k_smooth: int,
                                d_period: int) -> Dict[str, Optional[float]]:
    """Manual Stochastic calculation."""
    # Calculate raw %K
    raw_k = _calculate_raw_k(highs, lows, closes, k_period)
    
    # Smooth %K if needed
    if k_smooth > 1:
        smoothed_k = _smooth_series(raw_k, k_smooth)
    else:
        smoothed_k = raw_k
    
    # Calculate %D (smoothed %K)
    d_values = _smooth_series(smoothed_k, d_period)
    
    # Return last values
    k_final = None
    d_final = None
    
    for i in range(len(smoothed_k) - 1, -1, -1):
        if smoothed_k[i] is not None:
            k_final = smoothed_k[i]
            break
    
    for i in range(len(d_values) - 1, -1, -1):
        if d_values[i] is not None:
            d_final = d_values[i]
            break
    
    return {
        'k': k_final,
        'd': d_final
    }


def _calculate_stochastic_series_manual(highs: np.ndarray,
                                       lows: np.ndarray,
                                       closes: np.ndarray,
                                       k_period: int,
                                       k_smooth: int,
                                       d_period: int) -> Dict[str, List[Optional[float]]]:
    """Manual Stochastic series calculation."""
    # Calculate raw %K
    raw_k = _calculate_raw_k(highs, lows, closes, k_period)
    
    # Smooth %K if needed
    if k_smooth > 1:
        smoothed_k = _smooth_series(raw_k, k_smooth)
    else:
        smoothed_k = raw_k
    
    # Calculate %D (smoothed %K)
    d_values = _smooth_series(smoothed_k, d_period)
    
    return {
        'k': smoothed_k,
        'd': d_values
    }


def calculate_stochastic_signal(highs: Union[List[float], np.ndarray, pd.Series],
                               lows: Union[List[float], np.ndarray, pd.Series],
                               closes: Union[List[float], np.ndarray, pd.Series],
                               k_period: int = 14,
                               d_period: int = 3) -> str:
    """
    Calculate Stochastic trading signal based on %K and %D crossovers and levels.
    
    Args:
        highs: High prices
        lows: Low prices  
        closes: Close prices
        k_period: Period for %K calculation
        d_period: Period for %D smoothing
        
    Returns:
        'BULLISH' for buy signal, 'BEARISH' for sell signal, 'NEUTRAL' for no signal
    """
    if len(closes) < max(k_period, d_period) + 5:
        return 'NEUTRAL'
    
    # Calculate stochastic values
    stoch_data = calculate_stochastic(highs, lows, closes, k_period, 1, d_period)
    
    if not stoch_data or stoch_data['k'] is None or stoch_data['d'] is None:
        return 'NEUTRAL'
    
    current_k = float(stoch_data['k'])
    current_d = float(stoch_data['d'])
    
    # Get previous values for crossover detection
    if len(closes) >= max(k_period, d_period) + 6:
        prev_stoch = calculate_stochastic(
            highs[:-1], lows[:-1], closes[:-1], 
            k_period, 1, d_period
        )
        if prev_stoch and prev_stoch['k'] is not None and prev_stoch['d'] is not None:
            prev_k = float(prev_stoch['k'])
            prev_d = float(prev_stoch['d'])
        else:
            prev_k = current_k
            prev_d = current_d
    else:
        prev_k = current_k  
        prev_d = current_d
    
    # Signal conditions
    oversold_level = 20
    overbought_level = 80
    
    # Bullish signals
    if (current_k < oversold_level and current_d < oversold_level and 
        current_k > current_d and prev_k <= prev_d):
        # Bullish crossover in oversold territory
        return 'BULLISH'
    elif (current_k > current_d and prev_k <= prev_d and 
          current_k < 50):
        # Bullish crossover below midline
        return 'BULLISH'
    
    # Bearish signals  
    elif (current_k > overbought_level and current_d > overbought_level and
          current_k < current_d and prev_k >= prev_d):
        # Bearish crossover in overbought territory
        return 'BEARISH'
    elif (current_k < current_d and prev_k >= prev_d and
          current_k > 50):
        # Bearish crossover above midline
        return 'BEARISH'
    
    return 'NEUTRAL'