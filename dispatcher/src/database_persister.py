"""
Module de persistance des données de marché en base de données.
Sauve les données Kafka vers PostgreSQL/TimescaleDB.
"""
import logging
import asyncio
import asyncpg
from datetime import datetime
from typing import Dict, Any, Optional
from shared.src.config import get_db_config

logger = logging.getLogger(__name__)

class DatabasePersister:
    """
    Service de persistance des données de marché en base.
    """
    
    def __init__(self) -> None:
        self.db_pool: Optional[asyncpg.Pool] = None
        self.running = False
        self.loop = None
        
    async def initialize(self):
        """Initialise la connexion à la base de données."""
        try:
            db_config = get_db_config()
            self.db_pool = await asyncpg.create_pool(
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                user=db_config['user'],
                password=db_config['password'],
                min_size=2,
                max_size=5
            )
            logger.info("✅ Connexion base de données initialisée pour persistance")
            self.running = True
        except Exception as e:
            logger.error(f"❌ Erreur initialisation base de données: {e}")
            self.db_pool = None
    
    def start_persister(self):
        """Démarre le thread de persistance."""
        import threading
        
        def _run_async_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.initialize())
            self.loop.run_forever()
        
        if not self.loop:
            thread = threading.Thread(target=_run_async_loop, daemon=True)
            thread.start()
            # Attendre que la loop soit prête
            import time
            time.sleep(1)
    
    def save_market_data(self, topic: str, message: Dict[str, Any]):
        """
        Sauve les données de marché en base.
        
        Args:
            topic: Topic Kafka (ex: market.data.btcusdc.1m)
            message: Message contenant les données OHLCV
        """
        if not self.running or not self.db_pool:
            return
            
        # Extraire symbole et timeframe du topic ou du message
        symbol = None
        timeframe = None
        
        # D'abord essayer d'extraire du topic (format: market.data.{symbol}.{timeframe})
        parts = topic.split(".")
        if len(parts) >= 4:
            symbol = parts[2].upper()
            timeframe = parts[3]
        
        # Sinon, utiliser les valeurs du message
        if not symbol:
            symbol = message.get("symbol", "").upper()
        if not timeframe:
            timeframe = message.get("timeframe", "1m")
            
        if not symbol:
            logger.error(f"Impossible d'extraire le symbole du topic {topic} ou du message")
            return
            
        # Préparer les données OHLCV brutes pour l'insertion
        # Le Gateway n'envoie plus d'indicateurs - seulement les données brutes
        data = {
            "time": message.get("time"),
            "symbol": symbol,
            "timeframe": timeframe,
            # OHLCV de base uniquement
            "open": message.get("open"),
            "high": message.get("high"),
            "low": message.get("low"),
            "close": message.get("close"),
            "volume": message.get("volume"),
            # Métadonnées sources
            "quote_asset_volume": message.get("quote_asset_volume"),
            "number_of_trades": message.get("number_of_trades"),
            "taker_buy_base_asset_volume": message.get("taker_buy_base_asset_volume"),
            "taker_buy_quote_asset_volume": message.get("taker_buy_quote_asset_volume"),
            "is_closed": message.get("is_closed", True),
            "source": message.get("source", "gateway")
        }
        
        # Vérifier les données essentielles OHLCV
        if not data["time"] or data["close"] is None:
            logger.error(f"Données OHLCV essentielles manquantes pour {symbol}")
            return
            
        # Traiter le timestamp pour PostgreSQL
        if isinstance(data["time"], int):
            # Convertir timestamp milliseconds en objet datetime
            data["time"] = datetime.fromtimestamp(data["time"] / 1000)
        elif isinstance(data["time"], str):
            # Convertir string ISO en objet datetime
            try:
                data["time"] = datetime.fromisoformat(data["time"].replace('Z', '+00:00'))
            except ValueError:
                # Fallback pour format sans timezone
                data["time"] = datetime.fromisoformat(data["time"])
            
        # Exécuter l'insertion en async
        if self.loop is not None:
            future = asyncio.run_coroutine_threadsafe(
                self._insert_market_data(data),
                self.loop
            )
            
            # Attendre le résultat avec timeout
            try:
                future.result(timeout=5.0)
            except Exception as e:
                import traceback
                logger.error(f"Erreur lors de l'insertion des données de marché: {e}")
                logger.error(f"Type d'erreur: {type(e).__name__}")
                logger.error(f"Traceback complet: {traceback.format_exc()}")
                logger.error(f"Données problématiques: {data}")

    
    async def _insert_market_data(self, data: Dict[str, Any]):
        """
        Insert les données de marché BRUTES en base (méthode async).
        ARCHITECTURE PROPRE: Seulement OHLCV + métadonnées Binance.
        
        Args:
            data: Dictionnaire avec les données OHLCV brutes uniquement
        """
        query = """
            INSERT INTO market_data (
                time, symbol, timeframe, open, high, low, close, volume,
                quote_asset_volume, number_of_trades, 
                taker_buy_base_asset_volume, taker_buy_quote_asset_volume
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12
            )
            ON CONFLICT (time, symbol, timeframe) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                quote_asset_volume = EXCLUDED.quote_asset_volume,
                number_of_trades = EXCLUDED.number_of_trades,
                taker_buy_base_asset_volume = EXCLUDED.taker_buy_base_asset_volume,
                taker_buy_quote_asset_volume = EXCLUDED.taker_buy_quote_asset_volume,
                updated_at = CURRENT_TIMESTAMP
        """
        
        try:
            if self.db_pool is not None:
                async with self.db_pool.acquire() as conn:
                    await conn.execute(
                        query,
                        data["time"], data["symbol"], data["timeframe"],
                        data["open"], data["high"], data["low"], data["close"], data["volume"],
                        data.get("quote_asset_volume"), data.get("number_of_trades"),
                        data.get("taker_buy_base_asset_volume"), data.get("taker_buy_quote_asset_volume")
                    )
                    
                    # Log pour les données OHLCV brutes sauvegardées
                    logger.debug(f"✓ Données OHLCV brutes sauvegardées: {data['symbol']} {data['timeframe']} @ {data['close']}")
                    
        except Exception as e:
            import traceback
            logger.error(f"Erreur lors de l'insertion OHLCV en base: {e}")
            logger.error(f"Type d'erreur: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.error(f"Données OHLCV: {data}")
            raise

    async def close(self):
        """Ferme les connexions."""
        self.running = False
        if self.db_pool:
            await self.db_pool.close()
            logger.info("🔌 Connexions base fermées")