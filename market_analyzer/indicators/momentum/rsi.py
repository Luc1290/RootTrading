"""
RSI (Relative Strength Index) and Stochastic RSI Indicators

This module provides optimized implementations of RSI-based indicators.
Supports both TA-Lib (when available) and manual fallback calculations.
Enhanced with Redis caching for high-performance incremental updates.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import logging
import hashlib

# Import cache and utilities
from shared.src.indicator_cache import get_indicator_cache
from shared.src.technical_utils import validate_and_align_arrays, validate_indicator_params

logger = logging.getLogger(__name__)

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.debug("TA-Lib not available, using manual calculations")


def calculate_rsi(prices: Union[List[float], np.ndarray, pd.Series], 
                  period: int = 14,
                  symbol: str = None,
                  enable_cache: bool = True) -> Optional[float]:
    """
    Calculate the Relative Strength Index (RSI).
    
    The RSI measures momentum by comparing the magnitude of recent gains 
    to recent losses. Values range from 0 to 100.
    
    Args:
        prices: Price series (typically closing prices)
        period: Number of periods for RSI calculation (default: 14)
        symbol: Trading symbol for caching (optional, enables cache if provided)
        enable_cache: Whether to use Redis caching for performance
        
    Returns:
        RSI value (0-100) or None if insufficient data
        
    Notes:
        - RSI > 70 typically indicates overbought conditions
        - RSI < 30 typically indicates oversold conditions
        - Automatic caching when symbol provided (10x faster for repeated calls)
    """
    # If symbol provided and cache enabled, use cached version
    if symbol and enable_cache:
        return calculate_rsi_cached(prices, symbol, period, enable_cache)
    
    # Original non-cached implementation continues below...
    prices_array = _to_numpy_array(prices)
    if len(prices_array) < period + 1:
        return None
        
    if TALIB_AVAILABLE:
        try:
            rsi_values = talib.RSI(prices_array, timeperiod=period)
            return float(rsi_values[-1]) if not np.isnan(rsi_values[-1]) else None
        except Exception as e:
            logger.warning(f"TA-Lib RSI error: {e}, using fallback")
            
    return _calculate_rsi_manual(prices_array, period)


def calculate_rsi_series(prices: Union[List[float], np.ndarray, pd.Series],
                         period: int = 14) -> List[Optional[float]]:
    """
    Calculate RSI for the entire price series.
    
    Args:
        prices: Price series
        period: RSI period
        
    Returns:
        List of RSI values (None for insufficient data points)
    """
    prices_array = _to_numpy_array(prices)
    
    if TALIB_AVAILABLE:
        try:
            rsi_values = talib.RSI(prices_array, timeperiod=period)
            return [float(val) if not np.isnan(val) else None for val in rsi_values]
        except Exception as e:
            logger.warning(f"TA-Lib RSI series error: {e}, using fallback")
    
    # Manual calculation for entire series
    rsi_series = []
    for i in range(len(prices_array)):
        if i < period:
            rsi_series.append(None)
        else:
            rsi = _calculate_rsi_manual(prices_array[:i+1], period)
            rsi_series.append(rsi)
    
    return rsi_series


def calculate_rsi_incremental(current_price: float,
                             prev_avg_gain: float,
                             prev_avg_loss: float,
                             prev_price: float,
                             period: int = 14) -> Tuple[float, float, float]:
    """
    Calculate RSI incrementally using previous values.
    
    This method is more efficient for real-time updates as it doesn't
    recalculate from the entire price history.
    
    Args:
        current_price: Current price
        prev_avg_gain: Previous average gain
        prev_avg_loss: Previous average loss
        prev_price: Previous price
        period: RSI period
        
    Returns:
        Tuple of (rsi_value, new_avg_gain, new_avg_loss)
    """
    price_change = current_price - prev_price
    
    # Calculate current gain/loss
    current_gain = max(price_change, 0)
    current_loss = abs(min(price_change, 0))
    
    # Update averages using Wilder's smoothing
    alpha = 1.0 / period
    new_avg_gain = (prev_avg_gain * (period - 1) + current_gain) / period
    new_avg_loss = (prev_avg_loss * (period - 1) + current_loss) / period
    
    # Calculate RSI
    if new_avg_loss == 0:
        rsi = 100.0
    else:
        rs = new_avg_gain / new_avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    return round(rsi, 2), new_avg_gain, new_avg_loss


def calculate_stoch_rsi(prices: Union[List[float], np.ndarray, pd.Series],
                        period: int = 14,
                        stoch_period: int = 14,
                        smooth_k: int = 3,
                        smooth_d: int = 3) -> Optional[float]:
    """
    Calculate the Stochastic RSI.
    
    Stochastic RSI applies the Stochastic oscillator formula to RSI values
    instead of price values. It's more sensitive than regular RSI.
    
    Args:
        prices: Price series
        period: RSI calculation period
        stoch_period: Period for stochastic calculation on RSI
        smooth_k: Smoothing period for %K line
        smooth_d: Smoothing period for %D line
        
    Returns:
        Stochastic RSI value (0-100) or None if insufficient data
        
    Notes:
        - Values > 80 indicate overbought conditions
        - Values < 20 indicate oversold conditions
        - More sensitive than regular RSI
    """
    prices_array = _to_numpy_array(prices)
    
    # Need enough data for both RSI and Stochastic calculations
    min_required = period + stoch_period
    if len(prices_array) < min_required:
        return None
    
    if TALIB_AVAILABLE:
        try:
            # First calculate RSI
            rsi_values = talib.RSI(prices_array, timeperiod=period)
            
            # Remove NaN values
            valid_rsi = rsi_values[~np.isnan(rsi_values)]
            if len(valid_rsi) < stoch_period:
                return None
            
            # Apply Stochastic to RSI values
            stoch_k, stoch_d = talib.STOCH(
                valid_rsi, valid_rsi, valid_rsi,
                fastk_period=stoch_period,
                slowk_period=smooth_k,
                slowd_period=smooth_d
            )
            
            return float(stoch_k[-1]) if not np.isnan(stoch_k[-1]) else None
            
        except Exception as e:
            logger.warning(f"TA-Lib Stoch RSI error: {e}, using fallback")
    
    # Manual calculation
    return _calculate_stoch_rsi_manual(
        prices_array, period, stoch_period, smooth_k
    )


def calculate_stoch_rsi_full(prices: Union[List[float], np.ndarray, pd.Series],
                             period: int = 14,
                             stoch_period: int = 14,
                             smooth_k: int = 3,
                             smooth_d: int = 3) -> Dict[str, Optional[float]]:
    """
    Calculate full Stochastic RSI including %K and %D lines.
    
    Args:
        prices: Price series
        period: RSI calculation period
        stoch_period: Period for stochastic calculation
        smooth_k: Smoothing for %K
        smooth_d: Smoothing for %D
        
    Returns:
        Dictionary with 'k' and 'd' values
    """
    prices_array = _to_numpy_array(prices)
    min_required = period + stoch_period + smooth_d
    
    if len(prices_array) < min_required:
        return {'k': None, 'd': None}
    
    if TALIB_AVAILABLE:
        try:
            rsi_values = talib.RSI(prices_array, timeperiod=period)
            valid_rsi = rsi_values[~np.isnan(rsi_values)]
            
            if len(valid_rsi) < stoch_period + smooth_d:
                return {'k': None, 'd': None}
            
            stoch_k, stoch_d = talib.STOCH(
                valid_rsi, valid_rsi, valid_rsi,
                fastk_period=stoch_period,
                slowk_period=smooth_k,
                slowd_period=smooth_d
            )
            
            return {
                'k': float(stoch_k[-1]) if not np.isnan(stoch_k[-1]) else None,
                'd': float(stoch_d[-1]) if not np.isnan(stoch_d[-1]) else None
            }
            
        except Exception as e:
            logger.warning(f"TA-Lib Stoch RSI full error: {e}, using fallback")
    
    # Manual calculation with both K and D
    return _calculate_stoch_rsi_kd_manual(
        prices_array, period, stoch_period, smooth_k, smooth_d
    )


# ============ Helper Functions ============

def _to_numpy_array(data: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
    """Convert input data to numpy array."""
    if isinstance(data, pd.Series):
        return data.values
    elif isinstance(data, list):
        return np.array(data, dtype=float)
    return np.asarray(data, dtype=float)


def _calculate_rsi_manual(prices: np.ndarray, period: int) -> Optional[float]:
    """Manual RSI calculation using Wilder's smoothing method."""
    if len(prices) < period + 1:
        return None
    
    # Calculate price changes
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Initial averages (SMA for first period)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    # Apply Wilder's smoothing for remaining values
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return round(float(rsi), 2)


def _calculate_stoch_rsi_manual(prices: np.ndarray, 
                                rsi_period: int,
                                stoch_period: int,
                                smooth_k: int) -> Optional[float]:
    """Manual Stochastic RSI calculation."""
    # Calculate RSI values for the price series
    rsi_values = []
    for i in range(rsi_period, len(prices)):
        rsi = _calculate_rsi_manual(prices[:i+1], rsi_period)
        if rsi is not None:
            rsi_values.append(rsi)
    
    if len(rsi_values) < stoch_period:
        return None
    
    # Calculate Stochastic on RSI values
    recent_rsi = rsi_values[-stoch_period:]
    min_rsi = min(recent_rsi)
    max_rsi = max(recent_rsi)
    
    if max_rsi == min_rsi:
        return 50.0  # Neutral when no range
    
    # Raw Stochastic RSI
    current_rsi = rsi_values[-1]
    stoch_rsi = ((current_rsi - min_rsi) / (max_rsi - min_rsi)) * 100
    
    # Apply smoothing if needed (simplified SMA smoothing)
    if smooth_k > 1 and len(rsi_values) >= stoch_period + smooth_k - 1:
        stoch_values = []
        for i in range(smooth_k):
            idx = -(smooth_k - i)
            rsi_subset = rsi_values[idx-stoch_period+1:idx+1] if idx != -1 else rsi_values[-stoch_period:]
            min_val = min(rsi_subset)
            max_val = max(rsi_subset)
            if max_val != min_val:
                val = ((rsi_subset[-1] - min_val) / (max_val - min_val)) * 100
                stoch_values.append(val)
        
        if stoch_values:
            stoch_rsi = np.mean(stoch_values)
    
    return round(float(stoch_rsi), 2)


def _calculate_stoch_rsi_kd_manual(prices: np.ndarray,
                                   rsi_period: int,
                                   stoch_period: int,
                                   smooth_k: int,
                                   smooth_d: int) -> Dict[str, Optional[float]]:
    """Calculate both %K and %D lines for Stochastic RSI manually."""
    # First calculate %K values series
    k_values = []
    
    # Calculate RSI values
    rsi_values = []
    for i in range(rsi_period, len(prices)):
        rsi = _calculate_rsi_manual(prices[:i+1], rsi_period)
        if rsi is not None:
            rsi_values.append(rsi)
    
    if len(rsi_values) < stoch_period + smooth_k + smooth_d - 2:
        return {'k': None, 'd': None}
    
    # Calculate raw Stochastic RSI values
    for i in range(stoch_period - 1, len(rsi_values)):
        rsi_window = rsi_values[i-stoch_period+1:i+1]
        min_rsi = min(rsi_window)
        max_rsi = max(rsi_window)
        
        if max_rsi == min_rsi:
            k_values.append(50.0)
        else:
            k = ((rsi_window[-1] - min_rsi) / (max_rsi - min_rsi)) * 100
            k_values.append(k)
    
    if len(k_values) < smooth_k:
        return {'k': None, 'd': None}
    
    # Smooth %K values
    smoothed_k = []
    for i in range(smooth_k - 1, len(k_values)):
        smoothed_k.append(np.mean(k_values[i-smooth_k+1:i+1]))
    
    if len(smoothed_k) < smooth_d:
        return {'k': smoothed_k[-1] if smoothed_k else None, 'd': None}
    
    # Calculate %D (SMA of %K)
    d_values = []
    for i in range(smooth_d - 1, len(smoothed_k)):
        d_values.append(np.mean(smoothed_k[i-smooth_d+1:i+1]))
    
    return {
        'k': round(float(smoothed_k[-1]), 2) if smoothed_k else None,
        'd': round(float(d_values[-1]), 2) if d_values else None
    }


# ============ CACHED VERSIONS WITH HIGH PERFORMANCE ============

def calculate_rsi_cached(prices: Union[List[float], np.ndarray, pd.Series], 
                        symbol: str,
                        period: int = 14,
                        enable_cache: bool = True) -> Optional[float]:
    """
    Calculate RSI with Redis caching for high-performance incremental updates.
    
    Args:
        prices: Price series 
        symbol: Trading symbol for cache key
        period: RSI period
        enable_cache: Whether to use caching
        
    Returns:
        Current RSI value or None if insufficient data
    """
    if not enable_cache:
        return calculate_rsi(prices, period)
    
    try:
        # Validate parameters
        params = validate_indicator_params(period=period)
        period = params['period']
        
        # Validate and prepare data
        prices_array = validate_and_align_arrays(prices, min_length=period + 1)[0]
        
        # Get cache
        cache = get_indicator_cache()
        cache_key = f"rsi_{period}"
        
        # Check for cached RSI state
        cached_state = cache.get(cache_key, symbol)
        
        if cached_state is not None:
            # Try incremental calculation
            rsi_value = _calculate_rsi_incremental(
                prices_array, cached_state, period, symbol, cache_key
            )
            if rsi_value is not None:
                return rsi_value
        
        # Full calculation and cache
        rsi_value = calculate_rsi(prices_array, period)
        
        if rsi_value is not None:
            # Store RSI state for future incremental updates
            rsi_state = _create_rsi_state(prices_array, period)
            cache.set(cache_key, rsi_state, symbol)
        
        return rsi_value
        
    except Exception as e:
        logger.error(f"Erreur RSI cached pour {symbol}: {e}")
        # Fallback to non-cached version
        return calculate_rsi(prices, period)


def calculate_rsi_series_cached(prices: Union[List[float], np.ndarray, pd.Series],
                               symbol: str,
                               period: int = 14,
                               enable_cache: bool = True) -> List[Optional[float]]:
    """
    Calculate RSI series with caching optimization.
    
    Args:
        prices: Price series
        symbol: Trading symbol
        period: RSI period  
        enable_cache: Whether to use caching
        
    Returns:
        List of RSI values
    """
    if not enable_cache:
        return calculate_rsi_series(prices, period)
    
    try:
        # Validate data
        prices_array = validate_and_align_arrays(prices, min_length=period + 1)[0]
        
        # Get cache
        cache = get_indicator_cache()
        cache_key = f"rsi_series_{period}"
        
        # Check for cached series
        cached_series = cache.get(cache_key, symbol)
        
        if cached_series is not None:
            # Check if we can extend existing series
            if len(cached_series) <= len(prices_array):
                # Calculate only new values
                new_series = _extend_rsi_series(
                    prices_array, cached_series, period
                )
                if new_series:
                    cache.set(cache_key, new_series, symbol)
                    return new_series
        
        # Full calculation
        rsi_series = calculate_rsi_series(prices_array, period)
        
        # Cache the series
        if rsi_series:
            cache.set(cache_key, rsi_series, symbol)
        
        return rsi_series
        
    except Exception as e:
        logger.error(f"Erreur RSI series cached pour {symbol}: {e}")
        return calculate_rsi_series(prices, period)


def _create_rsi_state(prices: np.ndarray, period: int) -> Dict:
    """Create RSI state for caching."""
    if len(prices) < period + 1:
        return None
    
    # Calculate initial gains and losses for incremental updates
    price_changes = np.diff(prices)
    gains = np.where(price_changes > 0, price_changes, 0)
    losses = np.where(price_changes < 0, -price_changes, 0)
    
    # Calculate current average gain and loss
    avg_gain = np.mean(gains[-period:]) if len(gains) >= period else 0
    avg_loss = np.mean(losses[-period:]) if len(losses) >= period else 0
    
    return {
        'last_price': float(prices[-1]),
        'avg_gain': float(avg_gain),
        'avg_loss': float(avg_loss),
        'period': period,
        'price_count': len(prices),
        'last_rsi': calculate_rsi(prices, period)
    }


def _calculate_rsi_incremental(prices: np.ndarray, cached_state: Dict, 
                              period: int, symbol: str, cache_key: str) -> Optional[float]:
    """Calculate RSI incrementally using cached state."""
    try:
        cache = get_indicator_cache()
        
        # Check if we have new data
        if len(prices) <= cached_state.get('price_count', 0):
            return cached_state.get('last_rsi')
        
        # Get new prices since last calculation
        last_count = cached_state.get('price_count', 0)
        new_prices = prices[last_count:]
        
        # If too many new prices, do full recalculation
        if len(new_prices) > period:
            return None  # Will trigger full calculation
        
        # Incremental RSI calculation
        avg_gain = cached_state.get('avg_gain', 0)
        avg_loss = cached_state.get('avg_loss', 0)
        last_price = cached_state.get('last_price', prices[last_count - 1] if last_count > 0 else prices[0])
        
        # Process each new price
        for new_price in new_prices:
            price_change = new_price - last_price
            
            if price_change > 0:
                gain = price_change
                loss = 0
            elif price_change < 0:
                gain = 0
                loss = -price_change
            else:
                gain = 0
                loss = 0
            
            # Update averages using Wilder's smoothing
            alpha = 1.0 / period
            avg_gain = alpha * gain + (1 - alpha) * avg_gain
            avg_loss = alpha * loss + (1 - alpha) * avg_loss
            
            last_price = new_price
        
        # Calculate RSI
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1 + rs))
        
        # Update cached state
        updated_state = {
            'last_price': float(last_price),
            'avg_gain': float(avg_gain),
            'avg_loss': float(avg_loss),
            'period': period,
            'price_count': len(prices),
            'last_rsi': float(rsi)
        }
        
        cache.set(cache_key, updated_state, symbol)
        
        return float(rsi)
        
    except Exception as e:
        logger.warning(f"Erreur calcul RSI incrémental: {e}")
        return None


def _extend_rsi_series(prices: np.ndarray, cached_series: List[Optional[float]], 
                      period: int) -> Optional[List[Optional[float]]]:
    """Extend existing RSI series with new data."""
    try:
        cached_length = len(cached_series)
        prices_length = len(prices)
        
        if prices_length <= cached_length:
            return cached_series[:prices_length]
        
        # Calculate new RSI values
        new_series = cached_series.copy()
        
        # We need at least period+1 prices for RSI calculation
        for i in range(max(period, cached_length), prices_length):
            price_window = prices[i-period:i+1]
            rsi_value = calculate_rsi(price_window, period)
            new_series.append(rsi_value)
        
        return new_series
        
    except Exception as e:
        logger.warning(f"Erreur extension RSI series: {e}")
        return None


def clear_rsi_cache(symbol: str):
    """Clear all RSI cache entries for a symbol."""
    cache = get_indicator_cache()
    
    # Clear all RSI-related cache keys
    rsi_keys = [
        "rsi_7", "rsi_14", "rsi_21", "rsi_30",  # Common periods
        "rsi_series_7", "rsi_series_14", "rsi_series_21", "rsi_series_30",
        "stoch_rsi", "stoch_rsi_series"
    ]
    
    for key in rsi_keys:
        cache.delete(key, symbol)
    
    logger.info(f"Cache RSI effacé pour {symbol}")


def get_rsi_cache_stats(symbol: str) -> Dict:
    """Get RSI cache statistics for a symbol."""
    cache = get_indicator_cache()
    stats = cache.get_statistics()
    
    # Check which RSI indicators are cached
    cached_indicators = []
    test_keys = ["rsi_14", "rsi_series_14", "stoch_rsi"]
    
    for key in test_keys:
        if cache.get(key, symbol) is not None:
            cached_indicators.append(key)
    
    return {
        'symbol': symbol,
        'cached_rsi_indicators': cached_indicators,
        'cache_hit_ratio': stats['metrics']['hit_ratio'],
        'total_cache_entries': stats['cache_entries']
    }