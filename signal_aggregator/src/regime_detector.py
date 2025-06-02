#!/usr/bin/env python3
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta
import ta
import json

logger = logging.getLogger(__name__)


class RegimeDetector:
    """Detects market regime (TREND vs RANGE) using ADX and other indicators"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        
        # ADX thresholds
        self.adx_trend_threshold = 25  # Above this = trending market
        self.adx_strong_trend_threshold = 40  # Above this = strong trend
        
        # Additional indicators for regime detection
        self.bb_squeeze_threshold = 0.02  # Bollinger Band width threshold
        self.atr_ma_ratio_threshold = 1.2  # ATR/MA ratio for volatility
        
    async def get_regime(self, symbol: str) -> str:
        """Get current market regime for a symbol"""
        try:
            # Try to get cached regime first
            cache_key = f"regime:{symbol}"
            cached = await self.redis.get(cache_key)
            
            if cached:
                return cached
                
            # Calculate regime if not cached
            regime = await self._calculate_regime(symbol)
            
            # Cache for 1 minute
            await self.redis.setex(cache_key, 60, regime)
            
            return regime
            
        except Exception as e:
            logger.error(f"Error getting regime for {symbol}: {e}")
            return "UNDEFINED"  # Default to undefined
            
    async def _calculate_regime(self, symbol: str) -> str:
        """Calculate market regime based on multiple indicators"""
        try:
            # Get recent price data from Redis
            candles = await self._get_recent_candles(symbol, limit=50)
            
            if not candles or len(candles) < 30:
                return "UNDEFINED"
                
            df = pd.DataFrame(candles)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            
            # Calculate ADX
            adx = ta.trend.ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            )
            current_adx = adx.adx().iloc[-1]
            
            # Calculate Bollinger Band width
            bb = ta.volatility.BollingerBands(
                close=df['close'],
                window=20,
                window_dev=2
            )
            bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
            current_bb_width = bb_width.iloc[-1]
            
            # Calculate ATR/SMA ratio
            atr = ta.volatility.AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            )
            sma = ta.trend.SMAIndicator(close=df['close'], window=20)
            atr_sma_ratio = atr.average_true_range() / sma.sma_indicator()
            current_atr_ratio = atr_sma_ratio.iloc[-1]
            
            # Determine regime
            regime_score = 0
            
            # ADX contribution (most important)
            if current_adx > self.adx_strong_trend_threshold:
                regime_score += 2  # Strong trend
            elif current_adx > self.adx_trend_threshold:
                regime_score += 1  # Moderate trend
            else:
                regime_score -= 1  # No trend
                
            # Bollinger Band width contribution
            if current_bb_width < self.bb_squeeze_threshold:
                regime_score -= 1  # Squeeze = range
            else:
                regime_score += 0.5  # Expansion = potential trend
                
            # ATR ratio contribution
            if current_atr_ratio > self.atr_ma_ratio_threshold:
                regime_score += 0.5  # High volatility = trend
                
            # Final decision
            if regime_score >= 1.5:
                regime = "TREND"
            elif regime_score <= -0.5:
                regime = "RANGE"
            else:
                regime = "UNDEFINED"
                
            logger.info(f"Regime for {symbol}: {regime} (score={regime_score:.2f}, "
                       f"ADX={current_adx:.2f}, BB_width={current_bb_width:.4f}, "
                       f"ATR_ratio={current_atr_ratio:.4f})")
                       
            return regime
            
        except Exception as e:
            logger.error(f"Error calculating regime: {e}")
            return "UNDEFINED"
            
    async def _get_recent_candles(self, symbol: str, limit: int = 50) -> list:
        """Get recent candles from Redis"""
        try:
            # Get from Redis sorted set
            key = f"candles:1m:{symbol}"
            candles = await self.redis.zrevrange(key, 0, limit - 1)
            
            if not candles:
                return []
                
            # Parse candles
            parsed = []
            for candle_data in candles:
                candle = json.loads(candle_data)
                parsed.append(candle)
                
            return parsed
            
        except Exception as e:
            logger.error(f"Error getting candles: {e}")
            return []
            
    async def update_all_regimes(self):
        """Update regimes for all active symbols"""
        try:
            # Get all active symbols from Redis
            symbols = await self.redis.smembers("active_symbols")
            
            for symbol in symbols:
                await self.get_regime(symbol)  # This will calculate and cache
                
            logger.info(f"Updated regimes for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error updating regimes: {e}")
            
    async def get_regime_stats(self, symbol: str) -> Dict[str, float]:
        """Get detailed regime statistics"""
        try:
            stats_key = f"regime_stats:{symbol}"
            stats = await self.redis.hgetall(stats_key)
            
            if not stats:
                return {
                    'trend_percentage': 0.0,
                    'range_percentage': 0.0,
                    'undefined_percentage': 0.0,
                    'current_regime_duration': 0
                }
                
            return {
                'trend_percentage': float(stats.get('trend_pct', 0)),
                'range_percentage': float(stats.get('range_pct', 0)),
                'undefined_percentage': float(stats.get('undefined_pct', 0)),
                'current_regime_duration': int(stats.get('current_duration', 0))
            }
            
        except Exception as e:
            logger.error(f"Error getting regime stats: {e}")
            return {}