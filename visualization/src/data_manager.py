import asyncio
import aioredis
import asyncpg
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta, timezone
import os

logger = logging.getLogger(__name__)

def format_timestamp(dt):
    """Format timestamp correctly for JSON, avoiding double timezone indicators"""
    if dt is None:
        return None
    iso_str = dt.isoformat()
    # Si le timestamp a déjà un timezone, ne pas ajouter Z
    if '+' in iso_str or 'Z' in iso_str:
        return iso_str
    else:
        return iso_str + 'Z'

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
            "3m": 3,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1d": 1440
        }.get(interval, 1)
        
        # Vérifier si des données natives existent pour ce timeframe
        native_timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d']
        use_native = interval in native_timeframes
        
        if use_native:
            # Requête directe pour timeframes natifs - JOIN entre market_data et analyzer_data
            query = """
                SELECT 
                    md.time as timestamp,
                    md.open,
                    md.high,
                    md.low,
                    md.close,
                    md.volume,
                    -- Indicateurs techniques depuis analyzer_data
                    ad.rsi_14,
                    ad.rsi_21,
                    ad.ema_7,
                    ad.ema_12,
                    ad.ema_26,
                    ad.ema_50,
                    ad.ema_99,
                    ad.sma_20,
                    ad.sma_50,
                    ad.macd_line,
                    ad.macd_signal,
                    ad.macd_histogram,
                    ad.bb_upper,
                    ad.bb_middle,
                    ad.bb_lower,
                    ad.bb_position,
                    ad.bb_width,
                    ad.atr_14,
                    ad.adx_14,
                    ad.stoch_k,
                    ad.stoch_d,
                    ad.williams_r,
                    ad.cci_20,
                    ad.momentum_10,
                    ad.roc_10,
                    ad.roc_20,
                    ad.obv,
                    ad.vwap_10,
                    ad.vwap_quote_10,
                    ad.volume_ratio,
                    ad.avg_volume_20,
                    ad.quote_volume_ratio,
                    ad.avg_trade_size,
                    ad.trade_intensity,
                    ad.market_regime,
                    ad.regime_strength,
                    ad.regime_confidence,
                    ad.volume_context,
                    ad.volume_pattern,
                    ad.pattern_detected,
                    ad.data_quality
                FROM market_data md
                LEFT JOIN analyzer_data ad ON (md.time = ad.time AND md.symbol = ad.symbol AND md.timeframe = ad.timeframe)
                WHERE md.symbol = $1 AND md.timeframe = $2
            """
        else:
            # Requête avec agrégation pour les autres intervalles - JOIN avec analyzer_data
            query = f"""
                WITH aggregated AS (
                    SELECT 
                        date_trunc('hour', md.time) + 
                        INTERVAL '{interval_minutes} minutes' * 
                        FLOOR(EXTRACT(MINUTE FROM md.time) / {interval_minutes}) as period,
                        (array_agg(md.open ORDER BY md.time))[1] as open,
                        MAX(md.high) as high,
                        MIN(md.low) as low,
                        (array_agg(md.close ORDER BY md.time DESC))[1] as close,
                        SUM(md.volume) as volume,
                        (array_agg(ad.rsi_14 ORDER BY md.time DESC NULLS LAST))[1] as rsi_14,
                        (array_agg(ad.rsi_21 ORDER BY md.time DESC NULLS LAST))[1] as rsi_21,
                        (array_agg(ad.ema_7 ORDER BY md.time DESC NULLS LAST))[1] as ema_7,
                        (array_agg(ad.ema_12 ORDER BY md.time DESC NULLS LAST))[1] as ema_12,
                        (array_agg(ad.ema_26 ORDER BY md.time DESC NULLS LAST))[1] as ema_26,
                        (array_agg(ad.ema_50 ORDER BY md.time DESC NULLS LAST))[1] as ema_50,
                        (array_agg(ad.ema_99 ORDER BY md.time DESC NULLS LAST))[1] as ema_99,
                        (array_agg(ad.sma_20 ORDER BY md.time DESC NULLS LAST))[1] as sma_20,
                        (array_agg(ad.sma_50 ORDER BY md.time DESC NULLS LAST))[1] as sma_50,
                        (array_agg(ad.macd_line ORDER BY md.time DESC NULLS LAST))[1] as macd_line,
                        (array_agg(ad.macd_signal ORDER BY md.time DESC NULLS LAST))[1] as macd_signal,
                        (array_agg(ad.macd_histogram ORDER BY md.time DESC NULLS LAST))[1] as macd_histogram,
                        (array_agg(ad.bb_upper ORDER BY md.time DESC NULLS LAST))[1] as bb_upper,
                        (array_agg(ad.bb_middle ORDER BY md.time DESC NULLS LAST))[1] as bb_middle,
                        (array_agg(ad.bb_lower ORDER BY md.time DESC NULLS LAST))[1] as bb_lower,
                        (array_agg(ad.bb_position ORDER BY md.time DESC NULLS LAST))[1] as bb_position,
                        (array_agg(ad.bb_width ORDER BY md.time DESC NULLS LAST))[1] as bb_width,
                        (array_agg(ad.atr_14 ORDER BY md.time DESC NULLS LAST))[1] as atr_14,
                        (array_agg(ad.adx_14 ORDER BY md.time DESC NULLS LAST))[1] as adx_14,
                        (array_agg(ad.stoch_k ORDER BY md.time DESC NULLS LAST))[1] as stoch_k,
                        (array_agg(ad.stoch_d ORDER BY md.time DESC NULLS LAST))[1] as stoch_d,
                        (array_agg(ad.williams_r ORDER BY md.time DESC NULLS LAST))[1] as williams_r,
                        (array_agg(ad.cci_20 ORDER BY md.time DESC NULLS LAST))[1] as cci_20,
                        (array_agg(ad.momentum_10 ORDER BY md.time DESC NULLS LAST))[1] as momentum_10,
                        (array_agg(ad.roc_10 ORDER BY md.time DESC NULLS LAST))[1] as roc_10,
                        (array_agg(ad.roc_20 ORDER BY md.time DESC NULLS LAST))[1] as roc_20,
                        (array_agg(ad.obv ORDER BY md.time DESC NULLS LAST))[1] as obv,
                        (array_agg(ad.vwap_10 ORDER BY md.time DESC NULLS LAST))[1] as vwap_10,
                        (array_agg(ad.vwap_quote_10 ORDER BY md.time DESC NULLS LAST))[1] as vwap_quote_10,
                        (array_agg(ad.volume_ratio ORDER BY md.time DESC NULLS LAST))[1] as volume_ratio,
                        (array_agg(ad.avg_volume_20 ORDER BY md.time DESC NULLS LAST))[1] as avg_volume_20,
                        (array_agg(ad.quote_volume_ratio ORDER BY md.time DESC NULLS LAST))[1] as quote_volume_ratio,
                        (array_agg(ad.avg_trade_size ORDER BY md.time DESC NULLS LAST))[1] as avg_trade_size,
                        (array_agg(ad.trade_intensity ORDER BY md.time DESC NULLS LAST))[1] as trade_intensity,
                        (array_agg(ad.market_regime ORDER BY md.time DESC NULLS LAST))[1] as market_regime,
                        (array_agg(ad.regime_strength ORDER BY md.time DESC NULLS LAST))[1] as regime_strength,
                        (array_agg(ad.regime_confidence ORDER BY md.time DESC NULLS LAST))[1] as regime_confidence,
                        (array_agg(ad.volume_context ORDER BY md.time DESC NULLS LAST))[1] as volume_context,
                        (array_agg(ad.volume_pattern ORDER BY md.time DESC NULLS LAST))[1] as volume_pattern,
                        (array_agg(ad.pattern_detected ORDER BY md.time DESC NULLS LAST))[1] as pattern_detected,
                        (array_agg(ad.data_quality ORDER BY md.time DESC NULLS LAST))[1] as data_quality
                    FROM market_data md
                    LEFT JOIN analyzer_data ad ON (md.time = ad.time AND md.symbol = ad.symbol AND md.timeframe = ad.timeframe)
                    WHERE md.symbol = $1 AND md.timeframe = $2
                    GROUP BY period
                )
                SELECT 
                    period as timestamp,
                    open, high, low, close, volume,
                    rsi_14, rsi_21, ema_7, ema_12, ema_26, ema_50, ema_99, sma_20, sma_50,
                    macd_line, macd_signal, macd_histogram,
                    bb_upper, bb_middle, bb_lower, bb_position, bb_width,
                    atr_14, adx_14, stoch_k, stoch_d, williams_r, cci_20,
                    momentum_10, roc_10, roc_20, obv, vwap_10, vwap_quote_10,
                    volume_ratio, avg_volume_20, quote_volume_ratio, avg_trade_size, trade_intensity,
                    market_regime, regime_strength, regime_confidence,
                    volume_context, volume_pattern, pattern_detected, data_quality
                FROM aggregated
            """
        
        params = [symbol, interval]  # Utiliser le timeframe demandé
        
        if start_time:
            query += f" AND md.time >= ${len(params) + 1}"
            params.append(start_time.isoformat() if isinstance(start_time, datetime) else start_time)
            
        if end_time:
            query += f" AND md.time <= ${len(params) + 1}"
            params.append(end_time.isoformat() if isinstance(end_time, datetime) else end_time)
            
        if use_native:
            query += " ORDER BY md.time DESC"
        else:
            query += " ORDER BY period DESC"
        
        if limit:
            query += f" LIMIT ${len(params) + 1}"
            params.append(int(limit))
            
        try:
            async with self.postgres_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
                # Pour les intervalles agrégés SEULEMENT, ajouter la bougie courante
                if not use_native and rows:
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
                            iso_str = data_point["timestamp"].isoformat()
                            # Si le timestamp a déjà un timezone, ne pas ajouter Z
                            if '+' in iso_str or 'Z' in iso_str:
                                data_point["timestamp"] = iso_str
                            else:
                                data_point["timestamp"] = iso_str + 'Z'
                    else:
                        # Pour les données de la DB, row est un Record
                        iso_str = row["timestamp"].isoformat()
                        # Si le timestamp a déjà un timezone, ne pas ajouter Z
                        if '+' in iso_str or 'Z' in iso_str:
                            timestamp_str = iso_str
                        else:
                            timestamp_str = iso_str + 'Z'
                            
                        data_point = {
                            "timestamp": timestamp_str,
                            "open": float(row["open"]),
                            "high": float(row["high"]),
                            "low": float(row["low"]),
                            "close": float(row["close"]),
                            "volume": float(row["volume"]),
                            # Indicateurs RSI et EMAs
                            "rsi_14": float(row["rsi_14"]) if row["rsi_14"] is not None else None,
                            "rsi_21": float(row["rsi_21"]) if row["rsi_21"] is not None else None,
                            "ema_7": float(row["ema_7"]) if row["ema_7"] is not None else None,
                            "ema_12": float(row["ema_12"]) if row["ema_12"] is not None else None,
                            "ema_26": float(row["ema_26"]) if row["ema_26"] is not None else None,
                            "ema_50": float(row["ema_50"]) if row["ema_50"] is not None else None,
                            "ema_99": float(row["ema_99"]) if row["ema_99"] is not None else None,
                            "sma_20": float(row["sma_20"]) if row["sma_20"] is not None else None,
                            "sma_50": float(row["sma_50"]) if row["sma_50"] is not None else None,
                            # MACD
                            "macd_line": float(row["macd_line"]) if row["macd_line"] is not None else None,
                            "macd_signal": float(row["macd_signal"]) if row["macd_signal"] is not None else None,
                            "macd_histogram": float(row["macd_histogram"]) if row["macd_histogram"] is not None else None,
                            # Bollinger Bands
                            "bb_upper": float(row["bb_upper"]) if row["bb_upper"] is not None else None,
                            "bb_middle": float(row["bb_middle"]) if row["bb_middle"] is not None else None,
                            "bb_lower": float(row["bb_lower"]) if row["bb_lower"] is not None else None,
                            "bb_position": float(row["bb_position"]) if row["bb_position"] is not None else None,
                            "bb_width": float(row["bb_width"]) if row["bb_width"] is not None else None,
                            # Volatilité et tendance
                            "atr_14": float(row["atr_14"]) if row["atr_14"] is not None else None,
                            "adx_14": float(row["adx_14"]) if row["adx_14"] is not None else None,
                            # Oscillateurs
                            "stoch_k": float(row["stoch_k"]) if row["stoch_k"] is not None else None,
                            "stoch_d": float(row["stoch_d"]) if row["stoch_d"] is not None else None,
                            "williams_r": float(row["williams_r"]) if row["williams_r"] is not None else None,
                            "cci_20": float(row["cci_20"]) if row["cci_20"] is not None else None,
                            # Momentum
                            "momentum_10": float(row["momentum_10"]) if row["momentum_10"] is not None else None,
                            "roc_10": float(row["roc_10"]) if row["roc_10"] is not None else None,
                            "roc_20": float(row["roc_20"]) if row["roc_20"] is not None else None,
                            # Volume
                            "obv": float(row["obv"]) if row["obv"] is not None else None,
                            "vwap_10": float(row["vwap_10"]) if row["vwap_10"] is not None else None,
                            "vwap_quote_10": float(row["vwap_quote_10"]) if row["vwap_quote_10"] is not None else None,
                            "volume_ratio": float(row["volume_ratio"]) if row["volume_ratio"] is not None else None,
                            "avg_volume_20": float(row["avg_volume_20"]) if row["avg_volume_20"] is not None else None,
                            "quote_volume_ratio": float(row["quote_volume_ratio"]) if row["quote_volume_ratio"] is not None else None,
                            "avg_trade_size": float(row["avg_trade_size"]) if row["avg_trade_size"] is not None else None,
                            "trade_intensity": float(row["trade_intensity"]) if row["trade_intensity"] is not None else None,
                            # Régime et contexte
                            "market_regime": row["market_regime"] if row["market_regime"] is not None else None,
                            "regime_strength": row["regime_strength"] if row["regime_strength"] is not None else None,
                            "regime_confidence": float(row["regime_confidence"]) if row["regime_confidence"] is not None else None,
                            "volume_context": row["volume_context"] if row["volume_context"] is not None else None,
                            "volume_pattern": row["volume_pattern"] if row["volume_pattern"] is not None else None,
                            "pattern_detected": row["pattern_detected"] if row["pattern_detected"] is not None else None,
                            "data_quality": row["data_quality"] if row["data_quality"] is not None else None
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
                confidence,
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
            params.append(start_time.isoformat() if isinstance(start_time, datetime) else start_time)
            
        if end_time:
            query += f" AND timestamp <= ${len(params) + 1}"
            params.append(end_time.isoformat() if isinstance(end_time, datetime) else end_time)
            
        query += " ORDER BY timestamp DESC LIMIT $" + str(len(params) + 1)
        params.append(10000)  # Plus de signaux historiques
        
        try:
            async with self.postgres_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
                return [
                    {
                        "timestamp": format_timestamp(row["timestamp"]),
                        "strategy": row["strategy"],
                        "signal_type": row["signal_type"],
                        "strength": row["confidence"],  # Utiliser confidence comme strength
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
            
            # Récupérer toutes les données 1m depuis le début de la période avec JOIN
            query = """
                SELECT md.time, md.open, md.high, md.low, md.close, md.volume,
                       ad.rsi_14, ad.rsi_21, ad.ema_7, ad.ema_12, ad.ema_26, ad.ema_50, ad.ema_99, 
                       ad.sma_20, ad.sma_50, ad.macd_line, ad.macd_signal, ad.macd_histogram,
                       ad.bb_upper, ad.bb_middle, ad.bb_lower, ad.bb_position, ad.bb_width,
                       ad.atr_14, ad.adx_14, ad.stoch_k, ad.stoch_d, ad.williams_r, ad.cci_20,
                       ad.momentum_10, ad.roc_10, ad.roc_20, ad.obv, ad.vwap_10, ad.vwap_quote_10,
                       ad.volume_ratio, ad.avg_volume_20, ad.quote_volume_ratio, ad.avg_trade_size, ad.trade_intensity,
                       ad.market_regime, ad.regime_strength, ad.regime_confidence,
                       ad.volume_context, ad.volume_pattern, ad.pattern_detected, ad.data_quality
                FROM market_data md
                LEFT JOIN analyzer_data ad ON (md.time = ad.time AND md.symbol = ad.symbol AND md.timeframe = ad.timeframe)
                WHERE md.symbol = $1 AND md.time >= $2 
                ORDER BY md.time ASC
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
                "rsi_21": float(last_row["rsi_21"]) if last_row["rsi_21"] is not None else None,
                "ema_7": float(last_row["ema_7"]) if last_row["ema_7"] is not None else None,
                "ema_12": float(last_row["ema_12"]) if last_row["ema_12"] is not None else None,
                "ema_26": float(last_row["ema_26"]) if last_row["ema_26"] is not None else None,
                "ema_50": float(last_row["ema_50"]) if last_row["ema_50"] is not None else None,
                "ema_99": float(last_row["ema_99"]) if last_row["ema_99"] is not None else None,
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
                "adx_14": float(last_row["adx_14"]) if last_row["adx_14"] is not None else None,
                "stoch_k": float(last_row["stoch_k"]) if last_row["stoch_k"] is not None else None,
                "stoch_d": float(last_row["stoch_d"]) if last_row["stoch_d"] is not None else None,
                "williams_r": float(last_row["williams_r"]) if last_row["williams_r"] is not None else None,
                "cci_20": float(last_row["cci_20"]) if last_row["cci_20"] is not None else None,
                "momentum_10": float(last_row["momentum_10"]) if last_row["momentum_10"] is not None else None,
                "roc_10": float(last_row["roc_10"]) if last_row["roc_10"] is not None else None,
                "roc_20": float(last_row["roc_20"]) if last_row["roc_20"] is not None else None,
                "obv": float(last_row["obv"]) if last_row["obv"] is not None else None,
                "vwap_10": float(last_row["vwap_10"]) if last_row["vwap_10"] is not None else None,
                "vwap_quote_10": float(last_row["vwap_quote_10"]) if last_row["vwap_quote_10"] is not None else None,
                "volume_ratio": float(last_row["volume_ratio"]) if last_row["volume_ratio"] is not None else None,
                "avg_volume_20": float(last_row["avg_volume_20"]) if last_row["avg_volume_20"] is not None else None,
                "quote_volume_ratio": float(last_row["quote_volume_ratio"]) if last_row["quote_volume_ratio"] is not None else None,
                "avg_trade_size": float(last_row["avg_trade_size"]) if last_row["avg_trade_size"] is not None else None,
                "trade_intensity": float(last_row["trade_intensity"]) if last_row["trade_intensity"] is not None else None,
                "market_regime": last_row["market_regime"] if last_row["market_regime"] is not None else None,
                "regime_strength": last_row["regime_strength"] if last_row["regime_strength"] is not None else None,
                "regime_confidence": float(last_row["regime_confidence"]) if last_row["regime_confidence"] is not None else None,
                "volume_context": last_row["volume_context"] if last_row["volume_context"] is not None else None,
                "volume_pattern": last_row["volume_pattern"] if last_row["volume_pattern"] is not None else None,
                "pattern_detected": last_row["pattern_detected"] if last_row["pattern_detected"] is not None else None,
                "data_quality": last_row["data_quality"] if last_row["data_quality"] is not None else None
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
                
                pnl_cumsum = 0.0
                balances = []
                pnl_values = []
                pnl_percentages = []
                
                for row in rows:
                    pnl_cumsum = float(pnl_cumsum + (float(row["pnl_usdt"]) if row["pnl_usdt"] else 0))
                    balances.append(1000 + pnl_cumsum)  # Starting balance assumption
                    pnl_values.append(pnl_cumsum)
                    pnl_percentages.append(float(row["pnl_percentage"]) if row["pnl_percentage"] else 0)
                
                return {
                    "timestamps": [format_timestamp(row["timestamp"]) for row in rows],
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
    
    async def get_trade_cycles(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get trade cycles from database"""
        if not self.postgres_pool:
            return []
            
        query = """
            SELECT 
                id,
                symbol,
                strategy,
                status,
                side,
                entry_order_id,
                exit_order_id,
                entry_price,
                exit_price,
                quantity,
                profit_loss,
                profit_loss_percent,
                created_at,
                updated_at,
                completed_at
            FROM trade_cycles
            WHERE 1=1
        """
        
        params = []
        
        if symbol:
            query += f" AND symbol = ${len(params) + 1}"
            params.append(symbol)
            
        if status:
            query += f" AND status = ${len(params) + 1}"
            params.append(status)
            
        query += " ORDER BY created_at DESC"
        
        if limit:
            query += f" LIMIT ${len(params) + 1}"
            params.append(limit)
            
        try:
            async with self.postgres_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
                return [
                    {
                        "id": row["id"],
                        "symbol": row["symbol"],
                        "strategy": row["strategy"],
                        "status": row["status"],
                        "side": row["side"],
                        "entry_order_id": row["entry_order_id"],
                        "exit_order_id": row["exit_order_id"],
                        "entry_price": float(row["entry_price"]) if row["entry_price"] else None,
                        "exit_price": float(row["exit_price"]) if row["exit_price"] else None,
                        "quantity": float(row["quantity"]) if row["quantity"] else None,
                        "profit_loss": float(row["profit_loss"]) if row["profit_loss"] else None,
                        "profit_loss_percent": float(row["profit_loss_percent"]) if row["profit_loss_percent"] else None,
                        "created_at": format_timestamp(row["created_at"]),
                        "updated_at": format_timestamp(row["updated_at"]),
                        "completed_at": format_timestamp(row["completed_at"]) if row["completed_at"] else None
                    }
                    for row in rows
                ]
                
        except Exception as e:
            logger.error(f"Error fetching trade cycles: {e}")
            return []