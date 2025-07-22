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
            
        # Extraire symbole et timeframe du topic
  
            
        # Extraire les données OHLCV + indicateurs techniques
     

            # Log pour déboguer
            indicators_count = len(db_indicators)
            if indicators_count > 2:  # Plus que enhanced/ultra_enriched
                logger.debug(f"💾 Sauvegarde {symbol} avec {indicators_count} indicateurs")
            
            # Sauvegarder de manière asynchrone
            if self.loop:
                asyncio.run_coroutine_threadsafe(
                    self._insert_market_data(market_data),
                    self.loop
                )
                
        except Exception as e:
            logger.error(f"❌ Erreur extraction données marché: {e}")
    
    async def _insert_market_data(self, data: Dict[str, Any]):
        """
        Insert les données de marché enrichies en base (méthode async).
        
        Args:
            data: Dictionnaire avec les données OHLCV + indicateurs techniques
        """

    async def close(self):
        """Ferme les connexions."""
        self.running = False
        if self.db_pool:
            await self.db_pool.close()
            logger.info("🔌 Connexions base fermées")