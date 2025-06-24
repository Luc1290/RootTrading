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
        Calculate market risk/opportunity level from 0 to 10
        0-2: Excellent opportunity (rebounds, low danger)
        3-4: Good opportunity 
        5-6: Neutral (balanced risk/opportunity)
        7-8: Risky, reduce position sizes
        9-10: Very dangerous, avoid new positions
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
                
            # NOUVEAU: Calculate opportunity factors (subtract from danger)
            opportunity_score = 0.0
            
            # 1. Rebound detection - check if recovering from recent low
            recent_low = df['low'].tail(50).min()
            current_high = df['high'].tail(10).max()
            rebound_pct = ((current_price - recent_low) / recent_low) * 100
            
            if rebound_pct > 5.0:  # Strong rebound from low
                opportunity_score += 3.0
                logger.info(f"🚀 Strong rebound detected for {symbol}: +{rebound_pct:.1f}% from recent low")
            elif rebound_pct > 3.0:  # Moderate rebound
                opportunity_score += 2.0
                logger.info(f"📈 Moderate rebound detected for {symbol}: +{rebound_pct:.1f}% from recent low")
            elif rebound_pct > 1.5:  # Small rebound
                opportunity_score += 1.0
                
            # 2. Volatility trend - is volatility decreasing?
            vol_recent = ((df['high'].tail(10).max() - df['low'].tail(10).min()) / current_price) * 100
            vol_previous = ((df['high'].tail(20).head(10).max() - df['low'].tail(20).head(10).min()) / current_price) * 100
            
            if vol_recent < vol_previous * 0.7:  # Volatility decreased significantly
                opportunity_score += 1.5
                logger.info(f"📉 Volatility decreasing for {symbol}: {vol_recent:.1f}% vs {vol_previous:.1f}%")
            elif vol_recent < vol_previous * 0.85:  # Moderate volatility decrease
                opportunity_score += 1.0
                
            # 3. RSI recovery from oversold
            if 35 < current_rsi < 45 and df['close'].tail(5).is_monotonic_increasing:
                opportunity_score += 2.0  # Recovering from oversold
                logger.info(f"💡 RSI recovery from oversold for {symbol}: {current_rsi:.1f}")
            elif 30 < current_rsi < 40:
                opportunity_score += 1.0
                
            # 4. Consecutive green candles after red period
            close_prices = df['close'].tail(10)
            open_prices = df['open'].tail(10)
            green_candles_recent = sum(close_prices.tail(3).values > open_prices.tail(3).values)
            red_candles_before = sum(close_prices.head(7).values < open_prices.head(7).values)
            
            if green_candles_recent >= 2 and red_candles_before >= 4:
                opportunity_score += 1.5  # Recovery pattern
                logger.info(f"🟢 Recovery pattern for {symbol}: {green_candles_recent} green after {red_candles_before} red")
                
            # Final calculation: danger - opportunity
            final_score = danger_score - opportunity_score
            risk_opportunity_level = max(0.0, min(final_score, 10.0))  # Clamp to 0-10
            
            # Enhanced logging
            logger.info(f"Risk/Opportunity level for {symbol}: {risk_opportunity_level:.1f} "
                       f"(danger={danger_score:.1f}, opportunity={opportunity_score:.1f}) "
                       f"[vol={volatility_24h:.2f}%, RSI={current_rsi:.1f}, rebound={rebound_pct:.1f}%]")
            
            # Track history with new score
            self._track_danger_history(symbol, risk_opportunity_level)
                       
            return risk_opportunity_level
            
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
            self.redis.redis.lpush(history_key, json.dumps(history_data))
            self.redis.redis.ltrim(history_key, 0, 99)
            
            # Also track if we're exiting a danger period
            previous_danger_key = f"previous_danger:{symbol}"
            previous_danger = self.redis.get(previous_danger_key)
            
            if previous_danger:
                previous_level = float(previous_danger)
                # Détection de sortie de crise ou entrée en opportunité
                if previous_level >= 7.0 and danger_level < 5.0:
                    # On sort d'une période très dangereuse vers neutre/opportunité
                    recovery_key = f"recovery_period:{symbol}"
                    self.redis.set(recovery_key, "true", expiration=600)  # 10 minutes de période de récupération
                    logger.info(f"🔄 {symbol} sort d'une période dangereuse - période de récupération activée")
                elif previous_level >= 5.0 and danger_level <= 2.0:
                    # Entrée en zone d'excellente opportunité
                    opportunity_key = f"opportunity_period:{symbol}"
                    self.redis.set(opportunity_key, "true", expiration=900)  # 15 minutes d'opportunité
                    logger.info(f"🚀 {symbol} entre en zone d'excellente opportunité (score={danger_level:.1f})")
            
            # Update current danger level
            self.redis.set(previous_danger_key, str(danger_level), expiration=300)
            
        except Exception as e:
            logger.error(f"Error tracking danger history: {e}")
            
    async def is_in_recovery(self, symbol: str) -> bool:
        """Check if symbol is in recovery period after danger"""
        recovery_key = f"recovery_period:{symbol}"
        return bool(self.redis.get(recovery_key))
        
    async def is_opportunity_period(self, symbol: str) -> bool:
        """Check if symbol is in excellent opportunity period"""
        opportunity_key = f"opportunity_period:{symbol}"
        return bool(self.redis.get(opportunity_key))
            
    async def _get_recent_candles(self, symbol: str, limit: int = 50) -> list:
        """Get recent candles from Redis market_data"""
        try:
            # Get from Redis market_data key (JSON format)
            key = f"market_data:{symbol}:5m"
            market_data = self.redis.get(key)
            
            if not market_data:
                logger.warning(f"No market data found for {symbol} in key {key}")
                return []
                
            # Parse data (might already be dict from Redis client)
            if isinstance(market_data, str):
                data = json.loads(market_data)
            else:
                data = market_data  # Already a dict
            
            # Convert to candle format expected by pandas
            prices = data.get('prices', [])
            rsi_values = data.get('rsi', [])
            atr_values = data.get('atr', [])
            volumes = data.get('volumes', [])
            
            if not prices:
                logger.warning(f"No price data found for {symbol}")
                return []
                
            # Create candle objects with OHLCV data
            # Since we only have close prices, we'll approximate OHLC
            candles = []
            for i in range(min(len(prices), limit)):
                close_price = prices[-(i+1)]  # Start from most recent
                # Approximate OHLC from close prices
                open_price = prices[-(i+2)] if i+1 < len(prices) else close_price
                high_price = max(open_price, close_price) * 1.001  # Small approximation
                low_price = min(open_price, close_price) * 0.999   # Small approximation
                volume = volumes[-(i+1)] if i < len(volumes) else 1000  # Default volume
                
                candles.insert(0, {  # Insert at beginning to maintain chronological order
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
                
            logger.debug(f"Retrieved {len(candles)} candles for {symbol} from market_data")
            return candles
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
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