"""
Momentum and Rate of Change (ROC) Indicators

This module provides momentum-based indicators for measuring price velocity.
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


def calculate_momentum(prices: Union[List[float], np.ndarray, pd.Series],
                      period: int = 10) -> Optional[float]:
    """
    Calculate Momentum indicator.
    
    Momentum measures the rate of change in price over a specified period.
    Formula: Momentum = Current Price - Price n periods ago
    
    Args:
        prices: Price series (typically closing prices)
        period: Number of periods to look back (default: 10)
        
    Returns:
        Momentum value or None if insufficient data
        
    Notes:
        - Positive momentum indicates upward price movement
        - Negative momentum indicates downward price movement
        - Zero momentum indicates no change
    """
    prices_array = _to_numpy_array(prices)
    
    if len(prices_array) < period + 1:
        return None
    
    if TALIB_AVAILABLE:
        try:
            momentum_values = talib.MOM(prices_array, timeperiod=period)
            return float(momentum_values[-1]) if not np.isnan(momentum_values[-1]) else None
        except Exception as e:
            logger.warning(f"TA-Lib Momentum error: {e}, using fallback")
    
    # Manual calculation
    current_price = float(prices_array[-1])
    past_price = float(prices_array[-period - 1])
    
    momentum = current_price - past_price
    return float(momentum)


def calculate_roc(prices: Union[List[float], np.ndarray, pd.Series],
                  period: int = 10) -> Optional[float]:
    """
    Calculate Rate of Change (ROC) indicator.
    
    ROC measures the percentage change in price over a specified period.
    Formula: ROC = ((Current Price / Price n periods ago) - 1) * 100
    
    Args:
        prices: Price series
        period: Number of periods to look back
        
    Returns:
        ROC percentage or None if insufficient data
        
    Notes:
        - Positive ROC indicates percentage increase
        - Negative ROC indicates percentage decrease
        - ROC of 0 indicates no change
    """
    prices_array = _to_numpy_array(prices)
    
    if len(prices_array) < period + 1:
        return None
    
    if TALIB_AVAILABLE:
        try:
            roc_values = talib.ROC(prices_array, timeperiod=period)
            return float(roc_values[-1]) if not np.isnan(roc_values[-1]) else None
        except Exception as e:
            logger.warning(f"TA-Lib ROC error: {e}, using fallback")
    
    # Manual calculation
    current_price = float(prices_array[-1])
    past_price = float(prices_array[-period - 1])
    
    if past_price == 0:
        return None
    
    roc = ((current_price / past_price) - 1) * 100
    return float(roc)


def calculate_momentum_series(prices: Union[List[float], np.ndarray, pd.Series],
                             period: int = 10) -> List[Optional[float]]:
    """
    Calculate Momentum for entire price series.
    
    Args:
        prices: Price series
        period: Momentum period
        
    Returns:
        List of momentum values (None for insufficient data points)
    """
    prices_array = _to_numpy_array(prices)
    
    if TALIB_AVAILABLE:
        try:
            momentum_values = talib.MOM(prices_array, timeperiod=period)
            return [float(val) if not np.isnan(val) else None for val in momentum_values]
        except Exception as e:
            logger.warning(f"TA-Lib Momentum series error: {e}, using fallback")
    
    # Manual calculation
    momentum_series: List[Optional[float]] = []
    
    for i in range(len(prices_array)):
        if i < period:
            momentum_series.append(None)
        else:
            current_price = float(prices_array[i])
            past_price = float(prices_array[i - period])
            momentum = current_price - past_price
            momentum_series.append(float(momentum))
    
    return momentum_series


def calculate_roc_series(prices: Union[List[float], np.ndarray, pd.Series],
                        period: int = 10) -> List[Optional[float]]:
    """
    Calculate ROC for entire price series.
    
    Args:
        prices: Price series
        period: ROC period
        
    Returns:
        List of ROC values (None for insufficient data points)
    """
    prices_array = _to_numpy_array(prices)
    
    if TALIB_AVAILABLE:
        try:
            roc_values = talib.ROC(prices_array, timeperiod=period)
            return [float(val) if not np.isnan(val) else None for val in roc_values]
        except Exception as e:
            logger.warning(f"TA-Lib ROC series error: {e}, using fallback")
    
    # Manual calculation
    roc_series: List[Optional[float]] = []
    
    for i in range(len(prices_array)):
        if i < period:
            roc_series.append(None)
        else:
            current_price = float(prices_array[i])
            past_price = float(prices_array[i - period])
            
            if past_price == 0:
                roc_series.append(None)
            else:
                roc = ((current_price / past_price) - 1) * 100
                roc_series.append(float(roc))
    
    return roc_series


def calculate_price_oscillator(prices: Union[List[float], np.ndarray, pd.Series],
                              fast_period: int = 12,
                              slow_period: int = 26) -> Optional[float]:
    """
    Calculate Price Oscillator (similar to MACD but using ROC).
    
    Price Oscillator = Fast ROC - Slow ROC
    
    Args:
        prices: Price series
        fast_period: Fast ROC period
        slow_period: Slow ROC period
        
    Returns:
        Price Oscillator value or None
    """
    fast_roc = calculate_roc(prices, fast_period)
    slow_roc = calculate_roc(prices, slow_period)
    
    if fast_roc is None or slow_roc is None:
        return None
    
    return fast_roc - slow_roc


def momentum_signal(current_momentum: Optional[float],
                   previous_momentum: Optional[float] = None,
                   threshold: float = 0.0) -> str:
    """
    Generate trading signal based on momentum.
    
    Args:
        current_momentum: Current momentum value
        previous_momentum: Previous momentum value (optional)
        threshold: Zero-line threshold adjustment
        
    Returns:
        'bullish', 'bearish', 'bullish_acceleration', 'bearish_acceleration', or 'neutral'
    """
    if current_momentum is None:
        return 'neutral'
    
    # Basic directional signals
    if current_momentum > threshold:
        signal = 'bullish'
    elif current_momentum < -threshold:
        signal = 'bearish'
    else:
        signal = 'neutral'
    
    # Check for acceleration if previous value available
    if previous_momentum is not None:
        if current_momentum > previous_momentum and current_momentum > threshold:
            return 'bullish_acceleration'
        elif current_momentum < previous_momentum and current_momentum < -threshold:
            return 'bearish_acceleration'
    
    return signal


def roc_signal(current_roc: Optional[float],
               previous_roc: Optional[float] = None,
               threshold: float = 1.0) -> str:
    """
    Generate trading signal based on ROC.
    
    Args:
        current_roc: Current ROC value
        previous_roc: Previous ROC value (optional)
        threshold: Minimum ROC threshold for signals
        
    Returns:
        'bullish', 'bearish', 'strong_bullish', 'strong_bearish', or 'neutral'
    """
    if current_roc is None:
        return 'neutral'
    
    # Strong signals for higher ROC values
    if current_roc > threshold * 3:
        return 'strong_bullish'
    elif current_roc < -threshold * 3:
        return 'strong_bearish'
    elif current_roc > threshold:
        return 'bullish'
    elif current_roc < -threshold:
        return 'bearish'
    
    return 'neutral'


def calculate_momentum_divergence(prices: Union[List[float], np.ndarray, pd.Series],
                                 period: int = 10,
                                 lookback: int = 20) -> str:
    """
    Detect bullish/bearish divergence in momentum.
    
    Args:
        prices: Price series
        period: Momentum period
        lookback: Period to check for divergence
        
    Returns:
        'bullish_divergence', 'bearish_divergence', or 'none'
    """
    if len(prices) < lookback + period:
        return 'none'
    
    # Calculate momentum series
    momentum_series = calculate_momentum_series(prices, period)
    
    # Remove None values and get recent data
    valid_data = []
    for i in range(len(momentum_series)):
        if momentum_series[i] is not None:
            valid_data.append((i, float(prices[i]), momentum_series[i]))
    
    if len(valid_data) < lookback:
        return 'none'
    
    recent_data = valid_data[-lookback:]
    
    # Find local highs and lows in both price and momentum
    price_highs = []
    price_lows = []
    momentum_highs = []
    momentum_lows = []
    
    for i in range(1, len(recent_data) - 1):
        idx, price, momentum = recent_data[i]
        prev_price = recent_data[i-1][1]
        next_price = recent_data[i+1][1]
        prev_momentum = recent_data[i-1][2]
        next_momentum = recent_data[i+1][2]
        
        # Price peaks and troughs
        if price > prev_price and price > next_price:
            price_highs.append((idx, price))
        elif price < prev_price and price < next_price:
            price_lows.append((idx, price))
        
        # Momentum peaks and troughs
        if (momentum is not None and prev_momentum is not None and next_momentum is not None):
            if momentum > prev_momentum and momentum > next_momentum:
                momentum_highs.append((idx, momentum))
            elif momentum < prev_momentum and momentum < next_momentum:
                momentum_lows.append((idx, momentum))
    
    # Check for divergence
    # Bullish divergence: Price makes lower lows, momentum makes higher lows
    if len(price_lows) >= 2 and len(momentum_lows) >= 2:
        if (price_lows[-1][1] is not None and price_lows[-2][1] is not None and 
            momentum_lows[-1][1] is not None and momentum_lows[-2][1] is not None):
            if price_lows[-1][1] < price_lows[-2][1] and momentum_lows[-1][1] > momentum_lows[-2][1]:
                return 'bullish_divergence'
    
    # Bearish divergence: Price makes higher highs, momentum makes lower highs
    if len(price_highs) >= 2 and len(momentum_highs) >= 2:
        if (price_highs[-1][1] is not None and price_highs[-2][1] is not None and 
            momentum_highs[-1][1] is not None and momentum_highs[-2][1] is not None):
            if price_highs[-1][1] > price_highs[-2][1] and momentum_highs[-1][1] < momentum_highs[-2][1]:
                return 'bearish_divergence'
    
    return 'none'


def calculate_trix(prices: Union[List[float], np.ndarray, pd.Series],
                   period: int = 14) -> Optional[float]:
    """
    Calculate TRIX indicator.
    
    TRIX is a momentum oscillator based on triple-smoothed EMA.
    
    Args:
        prices: Price series
        period: EMA period
        
    Returns:
        TRIX value or None
    """
    if TALIB_AVAILABLE:
        try:
            prices_array = _to_numpy_array(prices)
            trix_values = talib.TRIX(prices_array, timeperiod=period)
            return float(trix_values[-1]) if not np.isnan(trix_values[-1]) else None
        except Exception as e:
            logger.warning(f"TA-Lib TRIX error: {e}, using fallback")
    
    # Manual TRIX calculation would require triple EMA smoothing
    # For now, return None if TA-Lib not available
    logger.warning("TRIX calculation requires TA-Lib")
    return None


def momentum_strength(values: List[Optional[float]],
                     period: int = 10) -> str:
    """
    Assess momentum strength based on recent values.
    
    Args:
        values: Recent momentum values
        period: Period to analyze
        
    Returns:
        'very_strong', 'strong', 'moderate', 'weak', or 'neutral'
    """
    if len(values) < period:
        return 'neutral'
    
    recent_values = [x for x in values[-period:] if x is not None]
    if len(recent_values) < period // 2:
        return 'neutral'
    
    # Calculate statistics
    avg_momentum = np.mean([abs(x) for x in recent_values])
    positive_count = sum(1 for x in recent_values if x > 0)
    negative_count = sum(1 for x in recent_values if x < 0)
    consistency = max(positive_count, negative_count) / len(recent_values)
    
    # Determine strength
    if avg_momentum > 2 and consistency > 0.8:
        return 'very_strong'
    elif avg_momentum > 1 and consistency > 0.7:
        return 'strong'
    elif avg_momentum > 0.5 and consistency > 0.6:
        return 'moderate'
    elif avg_momentum > 0.1:
        return 'weak'
    else:
        return 'neutral'


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