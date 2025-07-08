import asyncio
import aioredis
import asyncpg
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone
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
        
        self.postgres_host = os.getenv("POSTGRES_HOST", "db")
        self.postgres_port = int(os.getenv("POSTGRES_PORT", "5432"))
        self.postgres_db = os.getenv("POSTGRES_DB", "trading")
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
        limit: int = 10000,  # Augmenté pour garder plus d'historique
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Fetch market data from PostgreSQL"""
        if not self.postgres_pool:
            return []
            
        # Convertir l'intervalle en minutes pour l'agrégation
        interval_minutes = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240
        }.get(interval, 1)
        
        if interval_minutes == 1:
            # Requête directe pour 1m (pas d'agrégation nécessaire)
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
                WHERE symbol = $1 AND timeframe = $2
            """
        else:
            # Requête avec agrégation simple pour les autres intervalles
            query = f"""
                WITH aggregated AS (
                    SELECT 
                        date_trunc('hour', time) + 
                        INTERVAL '{interval_minutes} minutes' * 
                        FLOOR(EXTRACT(MINUTE FROM time) / {interval_minutes}) as period,
                        (array_agg(open ORDER BY time))[1] as open,
                        MAX(high) as high,
                        MIN(low) as low,
                        (array_agg(close ORDER BY time DESC))[1] as close,
                        SUM(volume) as volume,
                        (array_agg(rsi_14 ORDER BY time DESC NULLS LAST))[1] as rsi_14,
                        (array_agg(ema_12 ORDER BY time DESC NULLS LAST))[1] as ema_12,
                        (array_agg(ema_26 ORDER BY time DESC NULLS LAST))[1] as ema_26,
                        (array_agg(ema_50 ORDER BY time DESC NULLS LAST))[1] as ema_50,
                        (array_agg(sma_20 ORDER BY time DESC NULLS LAST))[1] as sma_20,
                        (array_agg(sma_50 ORDER BY time DESC NULLS LAST))[1] as sma_50,
                        (array_agg(macd_line ORDER BY time DESC NULLS LAST))[1] as macd_line,
                        (array_agg(macd_signal ORDER BY time DESC NULLS LAST))[1] as macd_signal,
                        (array_agg(macd_histogram ORDER BY time DESC NULLS LAST))[1] as macd_histogram,
                        (array_agg(bb_upper ORDER BY time DESC NULLS LAST))[1] as bb_upper,
                        (array_agg(bb_middle ORDER BY time DESC NULLS LAST))[1] as bb_middle,
                        (array_agg(bb_lower ORDER BY time DESC NULLS LAST))[1] as bb_lower,
                        (array_agg(bb_position ORDER BY time DESC NULLS LAST))[1] as bb_position,
                        (array_agg(bb_width ORDER BY time DESC NULLS LAST))[1] as bb_width,
                        (array_agg(atr_14 ORDER BY time DESC NULLS LAST))[1] as atr_14,
                        (array_agg(momentum_10 ORDER BY time DESC NULLS LAST))[1] as momentum_10,
                        (array_agg(volume_ratio ORDER BY time DESC NULLS LAST))[1] as volume_ratio,
                        (array_agg(avg_volume_20 ORDER BY time DESC NULLS LAST))[1] as avg_volume_20,
                        (array_agg(enhanced ORDER BY time DESC NULLS LAST))[1] as enhanced,
                        (array_agg(ultra_enriched ORDER BY time DESC NULLS LAST))[1] as ultra_enriched
                    FROM market_data
                    WHERE symbol = $1 AND timeframe = $2
                    GROUP BY period
                )
                SELECT 
                    period as timestamp,
                    open, high, low, close, volume,
                    rsi_14, ema_12, ema_26, ema_50, sma_20, sma_50,
                    macd_line, macd_signal, macd_histogram,
                    bb_upper, bb_middle, bb_lower, bb_position, bb_width,
                    atr_14, momentum_10, volume_ratio, avg_volume_20,
                    enhanced, ultra_enriched
                FROM aggregated
            """
        
        params = [symbol, interval]
        
        if start_time:
            query += f" AND time >= ${len(params) + 1}"
            params.append(start_time)
            
        if end_time:
            query += f" AND time <= ${len(params) + 1}"
            params.append(end_time)
            
        if interval_minutes == 1:
            query += " ORDER BY time DESC"
        else:
            query += " ORDER BY period DESC"
        
        if limit:
            query += f" LIMIT ${len(params) + 1}"
            params.append(limit)
            
        try:
            async with self.postgres_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
                # Pour les intervalles > 1m SEULEMENT, ajouter la bougie courante
                if interval_minutes > 1 and rows:
                    current_candle = await self._build_current_candle(conn, symbol, interval_minutes)
                    if current_candle:
                        # Insérer la bougie courante au début (plus récente)
                        rows = [current_candle] + list(rows)
                
                result_data = []
                for row in rows:
                    # Pour les bougies courantes, row est déjà un dict
                    if isinstance(row, dict):
                        data_point = row.copy()
                        if not isinstance(data_point["timestamp"], str):
                            data_point["timestamp"] = data_point["timestamp"].isoformat()
                    else:
                        # Pour les données de la DB, row est un Record
                        data_point = {
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
                    
                    result_data.append(data_point)
                
                # Trier par timestamp pour garantir l'ordre chronologique
                result_data.sort(key=lambda x: x['timestamp'])
                return result_data
                
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
            
        query += " ORDER BY timestamp DESC LIMIT 10000"  # Plus de signaux historiques
        
        try:
            async with self.postgres_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
                return [
                    {
                        "timestamp": row["timestamp"].isoformat(),
                        "strategy": row["strategy"],
                        "signal_type": row["signal_type"],
                        "strength": row["strength"],
                        "price": float(row["price"]),
                        "metadata": json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"] if row["metadata"] else {}
                    }
                    for row in rows
                ][::-1]
                
        except Exception as e:
            logger.error(f"Error fetching trading signals: {e}")
            return []
    
    async def _build_current_candle(self, conn, symbol: str, interval_minutes: int):
        """Construit la bougie courante en temps réel pour les intervalles > 1m"""
        from datetime import datetime, timezone
        
        try:
            now = datetime.now(timezone.utc)
            
            # Calculer le début de la période courante
            if interval_minutes == 5:
                # Pour 5m: 12:00, 12:05, 12:10...
                period_start_minute = (now.minute // 5) * 5
            elif interval_minutes == 15:
                # Pour 15m: 12:00, 12:15, 12:30...
                period_start_minute = (now.minute // 15) * 15
            elif interval_minutes == 30:
                # Pour 30m: 12:00, 12:30...
                period_start_minute = (now.minute // 30) * 30
            elif interval_minutes == 60:
                # Pour 1h: 12:00, 13:00...
                period_start_minute = 0
            elif interval_minutes == 240:
                # Pour 4h: 00:00, 04:00, 08:00...
                period_start_minute = 0
                now = now.replace(hour=(now.hour // 4) * 4)
            else:
                return None
            
            period_start = now.replace(minute=period_start_minute, second=0, microsecond=0)
            
            # Convertir en timezone-naive pour PostgreSQL
            period_start_naive = period_start.replace(tzinfo=None)
            
            # Récupérer toutes les données 1m depuis le début de la période
            query = """
                SELECT time, open, high, low, close, volume,
                       rsi_14, ema_12, ema_26, ema_50, sma_20, sma_50,
                       macd_line, macd_signal, macd_histogram,
                       bb_upper, bb_middle, bb_lower, bb_position, bb_width,
                       atr_14, momentum_10, volume_ratio, avg_volume_20,
                       enhanced, ultra_enriched
                FROM market_data 
                WHERE symbol = $1 AND time >= $2 
                ORDER BY time ASC
            """
            
            rows = await conn.fetch(query, symbol, period_start_naive)
            
            if not rows:
                return None
            
            # Construire la bougie en cours
            first_row = rows[0]
            last_row = rows[-1]
            
            current_candle = {
                "timestamp": period_start_naive,
                "open": float(first_row["open"]),
                "high": max(float(row["high"]) for row in rows),
                "low": min(float(row["low"]) for row in rows),
                "close": float(last_row["close"]),
                "volume": sum(float(row["volume"]) for row in rows),
                # Indicateurs: prendre les dernières valeurs
                "rsi_14": float(last_row["rsi_14"]) if last_row["rsi_14"] is not None else None,
                "ema_12": float(last_row["ema_12"]) if last_row["ema_12"] is not None else None,
                "ema_26": float(last_row["ema_26"]) if last_row["ema_26"] is not None else None,
                "ema_50": float(last_row["ema_50"]) if last_row["ema_50"] is not None else None,
                "sma_20": float(last_row["sma_20"]) if last_row["sma_20"] is not None else None,
                "sma_50": float(last_row["sma_50"]) if last_row["sma_50"] is not None else None,
                "macd_line": float(last_row["macd_line"]) if last_row["macd_line"] is not None else None,
                "macd_signal": float(last_row["macd_signal"]) if last_row["macd_signal"] is not None else None,
                "macd_histogram": float(last_row["macd_histogram"]) if last_row["macd_histogram"] is not None else None,
                "bb_upper": float(last_row["bb_upper"]) if last_row["bb_upper"] is not None else None,
                "bb_middle": float(last_row["bb_middle"]) if last_row["bb_middle"] is not None else None,
                "bb_lower": float(last_row["bb_lower"]) if last_row["bb_lower"] is not None else None,
                "bb_position": float(last_row["bb_position"]) if last_row["bb_position"] is not None else None,
                "bb_width": float(last_row["bb_width"]) if last_row["bb_width"] is not None else None,
                "atr_14": float(last_row["atr_14"]) if last_row["atr_14"] is not None else None,
                "momentum_10": float(last_row["momentum_10"]) if last_row["momentum_10"] is not None else None,
                "volume_ratio": float(last_row["volume_ratio"]) if last_row["volume_ratio"] is not None else None,
                "avg_volume_20": float(last_row["avg_volume_20"]) if last_row["avg_volume_20"] is not None else None,
                "enhanced": bool(last_row["enhanced"]) if last_row["enhanced"] is not None else False,
                "ultra_enriched": bool(last_row["ultra_enriched"]) if last_row["ultra_enriched"] is not None else False
            }
            
            logger.info(f"🕯️ Bougie courante {interval_minutes}m construite pour {symbol}: {period_start_naive} -> close={current_candle['close']}")
            return current_candle
            
        except Exception as e:
            logger.error(f"❌ Erreur construction bougie courante: {e}")
            return None
            
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
        start_time = datetime.now(timezone.utc).replace(tzinfo=None) - delta
        
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