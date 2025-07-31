"""
OBV (On-Balance Volume) Indicator

This module provides OBV calculation for volume-momentum analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.debug("TA-Lib not available, using manual calculations")


def calculate_obv(prices: Union[List[float], np.ndarray, pd.Series],
                  volumes: Union[List[float], np.ndarray, pd.Series]) -> Optional[float]:
    """
    Calculate On-Balance Volume (OBV).
    
    OBV measures buying and selling pressure as a cumulative indicator that
    adds volume on up days and subtracts volume on down days.
    
    Formula:
    - If close > previous close: OBV = previous OBV + volume
    - If close < previous close: OBV = previous OBV - volume  
    - If close = previous close: OBV = previous OBV
    
    Args:
        prices: Price series (typically closing prices)
        volumes: Volume series
        
    Returns:
        Current OBV value or None if insufficient data
        
    Notes:
        - OBV should be used with price analysis for confirmation
        - Rising OBV suggests accumulation (bullish)
        - Falling OBV suggests distribution (bearish)
        - OBV divergence from price can signal trend changes
    """
    prices_array = _to_numpy_array(prices)
    volumes_array = _to_numpy_array(volumes)
    
    if len(prices_array) < 2 or len(volumes_array) < 2:
        return None
    
    # Ensure both arrays have same length
    min_len = min(len(prices_array), len(volumes_array))
    prices_array = prices_array[-min_len:]
    volumes_array = volumes_array[-min_len:]
    
    if TALIB_AVAILABLE:
        try:
            obv_values = talib.OBV(prices_array, volumes_array)
            return float(obv_values[-1]) if not np.isnan(obv_values[-1]) else None
        except Exception as e:
            logger.warning(f"TA-Lib OBV error: {e}, using fallback")
    
    return _calculate_obv_manual(prices_array, volumes_array)


def calculate_obv_series(prices: Union[List[float], np.ndarray, pd.Series],
                        volumes: Union[List[float], np.ndarray, pd.Series]) -> List[Optional[float]]:
    """
    Calculate OBV for entire price/volume series.
    
    Args:
        prices: Price series
        volumes: Volume series
        
    Returns:
        List of OBV values (None for first value as no previous comparison)
    """
    prices_array = _to_numpy_array(prices)
    volumes_array = _to_numpy_array(volumes)
    
    # Ensure both arrays have same length
    min_len = min(len(prices_array), len(volumes_array))
    prices_array = prices_array[-min_len:]
    volumes_array = volumes_array[-min_len:]
    
    if TALIB_AVAILABLE:
        try:
            obv_values = talib.OBV(prices_array, volumes_array)
            return [float(val) if not np.isnan(val) else None for val in obv_values]
        except Exception as e:
            logger.warning(f"TA-Lib OBV series error: {e}, using fallback")
    
    # Manual calculation
    return _calculate_obv_series_manual(prices_array, volumes_array)


def obv_signal(current_obv: Optional[float],
               previous_obv: Optional[float],
               current_price: float,
               previous_price: float) -> str:
    """
    Generate trading signal based on OBV and price relationship.
    
    Args:
        current_obv: Current OBV value
        previous_obv: Previous OBV value
        current_price: Current price
        previous_price: Previous price
        
    Returns:
        'bullish_confirmation', 'bearish_confirmation', 
        'bullish_divergence', 'bearish_divergence', or 'neutral'
    """
    if current_obv is None or previous_obv is None:
        return 'neutral'
    
    obv_direction = 'up' if current_obv > previous_obv else 'down' if current_obv < previous_obv else 'flat'
    price_direction = 'up' if current_price > previous_price else 'down' if current_price < previous_price else 'flat'
    
    # Confirmation signals (OBV and price move in same direction)
    if obv_direction == 'up' and price_direction == 'up':
        return 'bullish_confirmation'
    elif obv_direction == 'down' and price_direction == 'down':
        return 'bearish_confirmation'
    
    # Divergence signals (OBV and price move in opposite directions)
    elif obv_direction == 'up' and price_direction == 'down':
        return 'bullish_divergence'
    elif obv_direction == 'down' and price_direction == 'up':
        return 'bearish_divergence'
    
    return 'neutral'


def calculate_obv_ma(prices: Union[List[float], np.ndarray, pd.Series],
                     volumes: Union[List[float], np.ndarray, pd.Series],
                     ma_period: int = 10) -> Optional[float]:
    """
    Calculate moving average of OBV for trend smoothing.
    
    Args:
        prices: Price series
        volumes: Volume series
        ma_period: Moving average period
        
    Returns:
        OBV moving average or None
    """
    obv_series = calculate_obv_series(prices, volumes)
    valid_obv = [x for x in obv_series if x is not None]
    
    if len(valid_obv) < ma_period:
        return None
    
    recent_obv = valid_obv[-ma_period:]
    obv_ma = np.mean(recent_obv)
    
    return float(obv_ma)


def calculate_obv_oscillator(prices: Union[List[float], np.ndarray, pd.Series],
                           volumes: Union[List[float], np.ndarray, pd.Series],
                           fast_period: int = 10,
                           slow_period: int = 20) -> Optional[float]:
    """
    Calculate OBV Oscillator (fast OBV MA - slow OBV MA).
    
    Args:
        prices: Price series
        volumes: Volume series
        fast_period: Fast moving average period
        slow_period: Slow moving average period
        
    Returns:
        OBV Oscillator value or None
    """
    fast_ma = calculate_obv_ma(prices, volumes, fast_period)
    slow_ma = calculate_obv_ma(prices, volumes, slow_period)
    
    if fast_ma is None or slow_ma is None:
        return None
    
    return fast_ma - slow_ma


def detect_obv_divergence(prices: Union[List[float], np.ndarray, pd.Series],
                         volumes: Union[List[float], np.ndarray, pd.Series],
                         lookback: int = 20) -> str:
    """
    Detect bullish/bearish divergence between OBV and price.
    
    Args:
        prices: Price series
        volumes: Volume series
        lookback: Period to analyze for divergence
        
    Returns:
        'bullish_divergence', 'bearish_divergence', or 'none'
    """
    if len(prices) < lookback + 2:
        return 'none'
    
    # Calculate OBV series
    obv_series = calculate_obv_series(prices, volumes)
    
    # Remove None values and get recent data
    valid_data = []
    for i in range(len(obv_series)):
        if obv_series[i] is not None:
            valid_data.append((i, float(prices[i]), obv_series[i]))
    
    if len(valid_data) < lookback:
        return 'none'
    
    recent_data = valid_data[-lookback:]
    
    # Find local highs and lows in both price and OBV
    price_highs = []
    price_lows = []
    obv_highs = []
    obv_lows = []
    
    for i in range(1, len(recent_data) - 1):
        idx, price, obv = recent_data[i]
        prev_price = recent_data[i-1][1]
        next_price = recent_data[i+1][1]
        prev_obv = recent_data[i-1][2]
        next_obv = recent_data[i+1][2]
        
        # Price peaks and troughs
        if price > prev_price and price > next_price:
            price_highs.append((idx, price))
        elif price < prev_price and price < next_price:
            price_lows.append((idx, price))
        
        # OBV peaks and troughs (with None checks)
        if (obv is not None and prev_obv is not None and next_obv is not None and 
            obv > prev_obv and obv > next_obv):
            obv_highs.append((idx, obv))
        elif (obv is not None and prev_obv is not None and next_obv is not None and 
              obv < prev_obv and obv < next_obv):
            obv_lows.append((idx, obv))
    
    # Check for divergence
    # Bullish divergence: Price makes lower lows, OBV makes higher lows
    if len(price_lows) >= 2 and len(obv_lows) >= 2:
        price_val1, price_val2 = price_lows[-1][1], price_lows[-2][1]
        obv_val1, obv_val2 = obv_lows[-1][1], obv_lows[-2][1]
        if (price_val1 is not None and price_val2 is not None and 
            obv_val1 is not None and obv_val2 is not None and 
            price_val1 < price_val2 and obv_val1 > obv_val2):
            return 'bullish_divergence'
    
    # Bearish divergence: Price makes higher highs, OBV makes lower highs
    if len(price_highs) >= 2 and len(obv_highs) >= 2:
        price_val1, price_val2 = price_highs[-1][1], price_highs[-2][1]
        obv_val1, obv_val2 = obv_highs[-1][1], obv_highs[-2][1]
        if (price_val1 is not None and price_val2 is not None and 
            obv_val1 is not None and obv_val2 is not None and 
            price_val1 > price_val2 and obv_val1 < obv_val2):
            return 'bearish_divergence'
    
    return 'none'


def calculate_volume_accumulation_distribution(highs: Union[List[float], np.ndarray, pd.Series],
                                             lows: Union[List[float], np.ndarray, pd.Series],
                                             closes: Union[List[float], np.ndarray, pd.Series],
                                             volumes: Union[List[float], np.ndarray, pd.Series]) -> Optional[float]:
    """
    Calculate Accumulation/Distribution Line (A/D Line).
    
    Similar to OBV but considers where price closes within the day's range.
    
    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        volumes: Volume series
        
    Returns:
        Current A/D Line value or None
    """
    if TALIB_AVAILABLE:
        try:
            highs_array = _to_numpy_array(highs)
            lows_array = _to_numpy_array(lows)
            closes_array = _to_numpy_array(closes)
            volumes_array = _to_numpy_array(volumes)
            
            ad_values = talib.AD(highs_array, lows_array, closes_array, volumes_array)
            return float(ad_values[-1]) if not np.isnan(ad_values[-1]) else None
        except Exception as e:
            logger.warning(f"TA-Lib A/D Line error: {e}, using fallback")
    
    # Manual calculation
    highs_array = _to_numpy_array(highs)
    lows_array = _to_numpy_array(lows)
    closes_array = _to_numpy_array(closes)
    volumes_array = _to_numpy_array(volumes)
    
    min_len = min(len(highs_array), len(lows_array), len(closes_array), len(volumes_array))
    if min_len == 0:
        return None
    
    ad_line = 0.0
    
    for i in range(min_len):
        high = highs_array[i]
        low = lows_array[i]
        close = closes_array[i]
        volume = volumes_array[i]
        
        if high == low:
            clv = 0  # Close Location Value
        else:
            clv = ((close - low) - (high - close)) / (high - low)
        
        ad_line += clv * volume
    
    return float(ad_line)


def obv_trend_strength(obv_values: List[Optional[float]], 
                      period: int = 10) -> str:
    """
    Assess trend strength based on OBV momentum.
    
    Args:
        obv_values: Recent OBV values
        period: Period to analyze
        
    Returns:
        'strong_accumulation', 'moderate_accumulation', 'strong_distribution',
        'moderate_distribution', or 'neutral'
    """
    if len(obv_values) < period + 1:
        return 'neutral'
    
    recent_obv = [x for x in obv_values[-period-1:] if x is not None]
    if len(recent_obv) < period + 1:
        return 'neutral'
    
    # Calculate OBV momentum
    obv_change = recent_obv[-1] - recent_obv[0]
    
    # Calculate consistency (how many periods moved in same direction)
    direction_changes = 0
    for i in range(1, len(recent_obv)):
        current_change = recent_obv[i] - recent_obv[i-1]
        prev_change = recent_obv[i-1] - recent_obv[i-2] if i > 1 else current_change
        
        if (current_change > 0) != (prev_change > 0):
            direction_changes += 1
    
    consistency = 1 - (direction_changes / (len(recent_obv) - 1))
    
    # Determine strength
    if obv_change > 0:
        if consistency > 0.7:
            return 'strong_accumulation'
        elif consistency > 0.5:
            return 'moderate_accumulation'
    elif obv_change < 0:
        if consistency > 0.7:
            return 'strong_distribution'
        elif consistency > 0.5:
            return 'moderate_distribution'
    
    return 'neutral'


# ============ Helper Functions ============

def _to_numpy_array(data: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
    """Convert input data to numpy array."""
    if isinstance(data, pd.Series):
        return np.asarray(data.values, dtype=float)
    elif isinstance(data, list):
        return np.array(data, dtype=float)
    return np.asarray(data, dtype=float)


def _calculate_obv_manual(prices: np.ndarray, volumes: np.ndarray) -> Optional[float]:
    """Manual OBV calculation."""
    if len(prices) < 2:
        return None
    
    obv = 0.0
    
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            obv += volumes[i]
        elif prices[i] < prices[i-1]:
            obv -= volumes[i]
        # If prices[i] == prices[i-1], OBV doesn't change
    
    return float(obv)


def _calculate_obv_series_manual(prices: np.ndarray, volumes: np.ndarray) -> List[Optional[float]]:
    """Manual OBV series calculation."""
    if len(prices) < 2:
        return [None] * len(prices)
    
    obv_series: List[Optional[float]] = [None]  # First value is None (no previous price to compare)
    obv = 0.0
    
    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            obv += volumes[i]
        elif prices[i] < prices[i-1]:
            obv -= volumes[i]
        # If prices[i] == prices[i-1], OBV doesn't change
        
        obv_series.append(float(obv))
    
    return obv_series