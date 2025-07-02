import asyncio
import aioredis
import asyncpg
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self):
        self.redis_client = None
        self.postgres_pool = None
        self.redis_pubsub = None
        self.subscriptions = {}
        
        # Configuration from environment
        self.redis_host = os.getenv("REDIS_HOST", "redis")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        
        self.postgres_host = os.getenv("POSTGRES_HOST", "postgres")
        self.postgres_port = int(os.getenv("POSTGRES_PORT", "5432"))
        self.postgres_db = os.getenv("POSTGRES_DB", "root_trading")
        self.postgres_user = os.getenv("POSTGRES_USER", "postgres")
        self.postgres_password = os.getenv("POSTGRES_PASSWORD", "postgres")
        
    async def initialize(self):
        """Initialize database connections"""
        await self._init_redis()
        await self._init_postgres()
        
    async def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = await aioredis.create_redis_pool(
                f"redis://{self.redis_host}:{self.redis_port}",
                encoding="utf-8"
            )
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            
    async def _init_postgres(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.postgres_pool = await asyncpg.create_pool(
                host=self.postgres_host,
                port=self.postgres_port,
                database=self.postgres_db,
                user=self.postgres_user,
                password=self.postgres_password,
                min_size=2,
                max_size=10
            )
            logger.info("PostgreSQL connection pool established")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            
    async def close(self):
        """Close all connections"""
        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()
            
        if self.postgres_pool:
            await self.postgres_pool.close()
            
    def is_redis_connected(self) -> bool:
        """Check if Redis is connected"""
        return self.redis_client is not None and not self.redis_client.closed
        
    def is_postgres_connected(self) -> bool:
        """Check if PostgreSQL is connected"""
        return self.postgres_pool is not None
        
    async def get_market_data(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Fetch market data from PostgreSQL"""
        if not self.postgres_pool:
            return []
            
        query = """
            SELECT 
                time as timestamp,
                open,
                high,
                low,
                close,
                volume,
                -- Indicateurs techniques enrichis
                rsi_14,
                ema_12,
                ema_26,
                ema_50,
                sma_20,
                sma_50,
                macd_line,
                macd_signal,
                macd_histogram,
                bb_upper,
                bb_middle,
                bb_lower,
                bb_position,
                bb_width,
                atr_14,
                momentum_10,
                volume_ratio,
                avg_volume_20,
                enhanced,
                ultra_enriched
            FROM market_data
            WHERE symbol = $1
        """
        
        params = [symbol]
        
        if start_time:
            query += f" AND time >= ${len(params) + 1}"
            params.append(start_time)
            
        if end_time:
            query += f" AND time <= ${len(params) + 1}"
            params.append(end_time)
            
        query += " ORDER BY time DESC"
        
        if limit:
            query += f" LIMIT ${len(params) + 1}"
            params.append(limit)
            
        try:
            async with self.postgres_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
                return [
                    {
                        "timestamp": row["timestamp"].isoformat(),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                        # Indicateurs techniques (peuvent être NULL)
                        "rsi_14": float(row["rsi_14"]) if row["rsi_14"] is not None else None,
                        "ema_12": float(row["ema_12"]) if row["ema_12"] is not None else None,
                        "ema_26": float(row["ema_26"]) if row["ema_26"] is not None else None,
                        "ema_50": float(row["ema_50"]) if row["ema_50"] is not None else None,
                        "sma_20": float(row["sma_20"]) if row["sma_20"] is not None else None,
                        "sma_50": float(row["sma_50"]) if row["sma_50"] is not None else None,
                        "macd_line": float(row["macd_line"]) if row["macd_line"] is not None else None,
                        "macd_signal": float(row["macd_signal"]) if row["macd_signal"] is not None else None,
                        "macd_histogram": float(row["macd_histogram"]) if row["macd_histogram"] is not None else None,
                        "bb_upper": float(row["bb_upper"]) if row["bb_upper"] is not None else None,
                        "bb_middle": float(row["bb_middle"]) if row["bb_middle"] is not None else None,
                        "bb_lower": float(row["bb_lower"]) if row["bb_lower"] is not None else None,
                        "bb_position": float(row["bb_position"]) if row["bb_position"] is not None else None,
                        "bb_width": float(row["bb_width"]) if row["bb_width"] is not None else None,
                        "atr_14": float(row["atr_14"]) if row["atr_14"] is not None else None,
                        "momentum_10": float(row["momentum_10"]) if row["momentum_10"] is not None else None,
                        "volume_ratio": float(row["volume_ratio"]) if row["volume_ratio"] is not None else None,
                        "avg_volume_20": float(row["avg_volume_20"]) if row["avg_volume_20"] is not None else None,
                        "enhanced": row["enhanced"] if row["enhanced"] is not None else False,
                        "ultra_enriched": row["ultra_enriched"] if row["ultra_enriched"] is not None else False
                    }
                    for row in rows
                ][::-1]  # Reverse to get chronological order
                
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return []
            
    async def get_trading_signals(
        self,
        symbol: str,
        strategy: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Fetch trading signals from PostgreSQL"""
        if not self.postgres_pool:
            return []
            
        query = """
            SELECT 
                timestamp,
                strategy,
                side as signal_type,
                strength,
                price,
                metadata
            FROM trading_signals
            WHERE symbol = $1
        """
        
        params = [symbol]
        
        if strategy:
            query += f" AND strategy = ${len(params) + 1}"
            params.append(strategy)
            
        if start_time:
            query += f" AND timestamp >= ${len(params) + 1}"
            params.append(start_time)
            
        if end_time:
            query += f" AND timestamp <= ${len(params) + 1}"
            params.append(end_time)
            
        query += " ORDER BY timestamp DESC LIMIT 1000"
        
        try:
            async with self.postgres_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
                return [
                    {
                        "timestamp": row["timestamp"].isoformat(),
                        "strategy": row["strategy"],
                        "signal_type": row["signal_type"],
                        "strength": float(row["strength"]),
                        "price": float(row["price"]),
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
                    }
                    for row in rows
                ][::-1]
                
        except Exception as e:
            logger.error(f"Error fetching trading signals: {e}")
            return []
            
    async def get_portfolio_performance(
        self,
        period: str = "24h"
    ) -> Dict[str, Any]:
        """Fetch portfolio performance data"""
        if not self.postgres_pool:
            return {}
            
        # Convert period to timedelta
        period_map = {
            "1h": timedelta(hours=1),
            "24h": timedelta(days=1),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30)
        }
        
        delta = period_map.get(period, timedelta(days=1))
        start_time = datetime.utcnow() - delta
        
        query = """
            SELECT 
                created_at as timestamp,
                profit_loss as pnl_usdt,
                profit_loss_percent as pnl_percentage
            FROM trade_cycles
            WHERE created_at >= $1 AND status = 'completed'
            ORDER BY created_at ASC
        """
        
        try:
            async with self.postgres_pool.acquire() as conn:
                rows = await conn.fetch(query, start_time)
                
                pnl_cumsum = 0
                balances = []
                pnl_values = []
                pnl_percentages = []
                
                for row in rows:
                    pnl_cumsum += float(row["pnl_usdt"]) if row["pnl_usdt"] else 0
                    balances.append(1000 + pnl_cumsum)  # Starting balance assumption
                    pnl_values.append(pnl_cumsum)
                    pnl_percentages.append(float(row["pnl_percentage"]) if row["pnl_percentage"] else 0)
                
                return {
                    "timestamps": [row["timestamp"].isoformat() for row in rows],
                    "balances": balances,
                    "pnl": pnl_values,
                    "pnl_percentage": pnl_percentages,
                    "win_rate": [0] * len(rows),  # Placeholder
                    "sharpe_ratio": [0] * len(rows)  # Placeholder
                }
                
        except Exception as e:
            logger.error(f"Error fetching portfolio performance: {e}")
            return {}
            
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price from Redis"""
        if not self.redis_client:
            return None
            
        try:
            key = f"ticker:{symbol}"
            data = await self.redis_client.get(key)
            if data:
                ticker = json.loads(data)
                return float(ticker.get("price", 0))
        except Exception as e:
            logger.error(f"Error getting latest price: {e}")
            
        return None
        
    async def subscribe_to_channel(self, channel: str, callback):
        """Subscribe to Redis channel for real-time updates"""
        if not self.redis_client:
            return
            
        try:
            if channel not in self.subscriptions:
                ch = await self.redis_client.subscribe(channel)
                self.subscriptions[channel] = {
                    "channel": ch[0],
                    "callbacks": []
                }
                
                # Start listening to the channel
                asyncio.create_task(self._listen_to_channel(channel))
                
            self.subscriptions[channel]["callbacks"].append(callback)
            
        except Exception as e:
            logger.error(f"Error subscribing to channel {channel}: {e}")
            
    async def _listen_to_channel(self, channel_name: str):
        """Listen to a Redis channel and call callbacks"""
        if channel_name not in self.subscriptions:
            return
            
        channel = self.subscriptions[channel_name]["channel"]
        
        try:
            async for message in channel.iter():
                if message:
                    data = json.loads(message)
                    callbacks = self.subscriptions[channel_name]["callbacks"]
                    
                    for callback in callbacks:
                        try:
                            await callback(data)
                        except Exception as e:
                            logger.error(f"Error in callback: {e}")
                            
        except Exception as e:
            logger.error(f"Error listening to channel {channel_name}: {e}")
            
    async def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols"""
        if not self.postgres_pool:
            # Return default symbols if no database connection
            return ["XRPUSDC", "SOLUSDC", "BTCUSDT", "ETHUSDT", "ADAUSDT"]
            
        query = """
            SELECT DISTINCT symbol 
            FROM market_data 
            ORDER BY symbol
        """
        
        try:
            async with self.postgres_pool.acquire() as conn:
                rows = await conn.fetch(query)
                symbols = [row["symbol"] for row in rows]
                
                # If no symbols in database, return default ones
                if not symbols:
                    return ["XRPUSDC", "SOLUSDC", "BTCUSDT", "ETHUSDT", "ADAUSDT"]
                
                return symbols
                
        except Exception as e:
            logger.error(f"Error fetching available symbols: {e}")
            # Return default symbols on error
            return ["XRPUSDC", "SOLUSDC", "BTCUSDT", "ETHUSDT", "ADAUSDT"]