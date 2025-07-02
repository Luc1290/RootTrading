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
    
    def __init__(self):
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
        parts = topic.split('.')
        if len(parts) < 4 or parts[0] != "market" or parts[1] != "data":
            return
            
        symbol = parts[2].upper()
        timeframe = parts[3]
        
        # Vérifier que la bougie est fermée
        if not message.get('is_closed', False):
            return
            
        # Extraire les données OHLCV + indicateurs techniques
        try:
            market_data = {
                'time': datetime.fromtimestamp(message['start_time'] / 1000),
                'symbol': symbol,
                'open': float(message['open']),
                'high': float(message['high']),
                'low': float(message['low']),
                'close': float(message['close']),
                'volume': float(message['volume'])
            }
            
            # **NOUVEAU**: Ajouter les indicateurs techniques si disponibles
            # RSI
            if 'rsi_14' in message and message['rsi_14'] is not None:
                market_data['rsi_14'] = float(message['rsi_14'])
            
            # EMAs
            for period in [12, 26, 50]:
                key = f'ema_{period}'
                if key in message and message[key] is not None:
                    market_data[key] = float(message[key])
            
            # SMAs  
            for period in [20, 50]:
                key = f'sma_{period}'
                if key in message and message[key] is not None:
                    market_data[key] = float(message[key])
            
            # MACD
            for macd_field in ['macd_line', 'macd_signal', 'macd_histogram']:
                if macd_field in message and message[macd_field] is not None:
                    market_data[macd_field] = float(message[macd_field])
            
            # Bollinger Bands
            for bb_field in ['bb_upper', 'bb_middle', 'bb_lower', 'bb_position', 'bb_width']:
                if bb_field in message and message[bb_field] is not None:
                    market_data[bb_field] = float(message[bb_field])
            
            # Autres indicateurs
            for indicator in ['atr_14', 'momentum_10', 'volume_ratio', 'avg_volume_20']:
                if indicator in message and message[indicator] is not None:
                    market_data[indicator] = float(message[indicator])
            
            # Métadonnées d'enrichissement
            market_data['enhanced'] = message.get('enhanced', False)
            market_data['ultra_enriched'] = message.get('ultra_enriched', False)
            
            # Log pour déboguer
            indicators_count = len([k for k in market_data.keys() if k not in ['time', 'symbol', 'open', 'high', 'low', 'close', 'volume']])
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
        if not self.db_pool:
            return
            
        try:
            # Construire la requête dynamiquement en fonction des champs disponibles
            columns = list(data.keys())
            placeholders = [f"${i+1}" for i in range(len(columns))]
            values = list(data.values())
            
            # Construire les clauses UPDATE pour ON CONFLICT
            update_clauses = [f"{col} = EXCLUDED.{col}" for col in columns if col not in ['time', 'symbol']]
            
            query = f"""
                INSERT INTO market_data ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
                ON CONFLICT (time, symbol) DO UPDATE SET
                    {', '.join(update_clauses)}
            """
            
            async with self.db_pool.acquire() as conn:
                await conn.execute(query, *values)
                
                # Log avec nombre d'indicateurs
                indicators_count = len([k for k in data.keys() if k not in ['time', 'symbol', 'open', 'high', 'low', 'close', 'volume']])
                if indicators_count > 2:  # Plus que enhanced/ultra_enriched
                    logger.debug(f"💾 Sauvé {data['symbol']} avec {indicators_count} indicateurs: {data['close']}")
                else:
                    logger.debug(f"💾 Sauvé {data['symbol']} OHLCV seulement: {data['close']}")
                
        except Exception as e:
            logger.error(f"❌ Erreur insertion base enrichie: {e}")
            logger.error(f"Données: {list(data.keys())}")
    
    async def close(self):
        """Ferme les connexions."""
        self.running = False
        if self.db_pool:
            await self.db_pool.close()
            logger.info("🔌 Connexions base fermées")