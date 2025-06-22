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
        
        # Market danger thresholds
        self.volatility_danger_threshold = 8.0  # % volatility in 24h
        self.rsi_oversold_threshold = 30
        self.rsi_overbought_threshold = 70
        self.volume_spike_threshold = 2.0  # 2x average volume
        
    async def get_regime(self, symbol: str) -> str:
        """Get current market regime for a symbol"""
        try:
            # Try to get cached regime first
            cache_key = f"regime:{symbol}"
            cached = self.redis.get(cache_key)
            
            if cached:
                return cached
                
            # Calculate regime if not cached
            regime = await self._calculate_regime(symbol)
            
            # Cache for 1 minute
            self.redis.set(cache_key, regime, expiration=60)
            
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
            
    async def get_danger_level(self, symbol: str) -> float:
        """
        Calculate market danger level from 0 to 10
        0-3: Safe for trading
        4-6: Caution, reduce position sizes
        7-10: Dangerous, avoid new positions
        """
        try:
            # Get cached danger level first
            cache_key = f"danger:{symbol}"
            cached = self.redis.get(cache_key)
            
            if cached:
                return float(cached)
                
            # Calculate danger level
            danger_level = await self._calculate_danger_level(symbol)
            
            # Cache for 1 minute
            self.redis.set(cache_key, str(danger_level), expiration=60)
            
            return danger_level
            
        except Exception as e:
            logger.error(f"Error getting danger level for {symbol}: {e}")
            return 5.0  # Default to medium danger
            
    async def _calculate_danger_level(self, symbol: str) -> float:
        """Calculate market danger level based on multiple factors"""
        try:
            candles = await self._get_recent_candles(symbol, limit=100)
            
            if not candles or len(candles) < 50:
                return 5.0  # Default medium danger if not enough data
                
            df = pd.DataFrame(candles)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float) if 'volume' in df.columns else 0
            df['open'] = df['open'].astype(float) if 'open' in df.columns else df['close']
            
            danger_score = 0.0
            
            # 1. Calculate 24h volatility
            high_24h = df['high'].tail(288).max()  # 288 candles = 24h on 5min
            low_24h = df['low'].tail(288).min()
            current_price = df['close'].iloc[-1]
            volatility_24h = ((high_24h - low_24h) / current_price) * 100
            
            if volatility_24h > self.volatility_danger_threshold:
                danger_score += 3.0
            elif volatility_24h > self.volatility_danger_threshold * 0.75:
                danger_score += 2.0
            elif volatility_24h > self.volatility_danger_threshold * 0.5:
                danger_score += 1.0
                
            # 2. RSI extremes
            rsi = ta.momentum.RSIIndicator(close=df['close'], window=14)
            current_rsi = rsi.rsi().iloc[-1]
            
            if current_rsi < self.rsi_oversold_threshold or current_rsi > self.rsi_overbought_threshold:
                danger_score += 2.0
            elif current_rsi < 35 or current_rsi > 65:
                danger_score += 1.0
                
            # 3. Price trend (MA direction)
            ma_25 = ta.trend.SMAIndicator(close=df['close'], window=25).sma_indicator()
            ma_99 = ta.trend.SMAIndicator(close=df['close'], window=99).sma_indicator()
            
            if len(ma_99) > 0:
                ma_25_current = ma_25.iloc[-1]
                ma_99_current = ma_99.iloc[-1]
                ma_25_prev = ma_25.iloc[-10]
                ma_99_prev = ma_99.iloc[-10]
                
                # Bearish divergence = dangerous
                if ma_25_current < ma_99_current and ma_25_prev > ma_99_prev:
                    danger_score += 2.5
                # Strong downtrend
                elif ma_25_current < ma_25_prev and ma_99_current < ma_99_prev:
                    danger_score += 1.5
                    
            # 4. Volume analysis
            avg_volume = df['volume'].tail(50).mean()
            recent_volume = df['volume'].tail(5).mean()
            
            if recent_volume > avg_volume * self.volume_spike_threshold:
                danger_score += 1.5  # High volume in volatile market = danger
                
            # 5. Consecutive red candles
            close_prices = df['close'].tail(10)
            open_prices = df['open'].tail(10)
            red_candles = sum(close_prices.values < open_prices.values)
            
            if red_candles >= 7:
                danger_score += 1.5
            elif red_candles >= 5:
                danger_score += 0.5
                
            # Normalize to 0-10 scale
            danger_level = min(danger_score, 10.0)
            
            logger.info(f"Danger level for {symbol}: {danger_level:.1f} "
                       f"(vol={volatility_24h:.2f}%, RSI={current_rsi:.1f}, "
                       f"red_candles={red_candles}/10)")
            
            # NOUVEAU: Tracker l'historique du danger pour détecter les sorties de crise
            self._track_danger_history(symbol, danger_level)
                       
            return danger_level
            
        except Exception as e:
            logger.error(f"Error calculating danger level: {e}")
            return 5.0  # Default medium danger
            
    def _track_danger_history(self, symbol: str, danger_level: float):
        """Track danger level history to detect recovery periods"""
        try:
            history_key = f"danger_history:{symbol}"
            history_data = {
                'timestamp': datetime.now().isoformat(),
                'level': danger_level
            }
            
            # Store in Redis list (keep last 100 entries)
            self.redis.lpush(history_key, json.dumps(history_data))
            self.redis.ltrim(history_key, 0, 99)
            
            # Also track if we're exiting a danger period
            previous_danger_key = f"previous_danger:{symbol}"
            previous_danger = self.redis.get(previous_danger_key)
            
            if previous_danger:
                previous_level = float(previous_danger)
                # Détection de sortie de crise
                if previous_level >= 7.0 and danger_level < 5.0:
                    # On sort d'une période très dangereuse
                    recovery_key = f"recovery_period:{symbol}"
                    self.redis.set(recovery_key, "true", expiration=600)  # 10 minutes de période de récupération
                    logger.info(f"🔄 {symbol} sort d'une période dangereuse - période de récupération activée")
            
            # Update current danger level
            self.redis.set(previous_danger_key, str(danger_level), expiration=300)
            
        except Exception as e:
            logger.error(f"Error tracking danger history: {e}")
            
    async def is_in_recovery(self, symbol: str) -> bool:
        """Check if symbol is in recovery period after danger"""
        recovery_key = f"recovery_period:{symbol}"
        return bool(self.redis.get(recovery_key))
            
    async def _get_recent_candles(self, symbol: str, limit: int = 50) -> list:
        """Get recent candles from Redis"""
        try:
            # Get from Redis sorted set
            key = f"candles:1m:{symbol}"
            candles = self.redis.smembers(key)  # Fallback to simple get
            
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
            # Use common trading symbols since active_symbols might not exist
            symbols = ["XRPUSDC", "SOLUSDC"]
            
            for symbol in symbols:
                await self.get_regime(symbol)  # This will calculate and cache
                await self.get_danger_level(symbol)  # Also update danger level
                
            logger.info(f"Updated regimes and danger levels for {len(symbols)} symbols")
            
        except Exception as e:
            logger.error(f"Error updating regimes: {e}")
            
    async def get_regime_stats(self, symbol: str) -> Dict[str, float]:
        """Get detailed regime statistics"""
        try:
            stats_key = f"regime_stats:{symbol}"
            stats = self.redis.hgetall(stats_key)
            
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