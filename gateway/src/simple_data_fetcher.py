"""
Simple Data Fetcher - Service de récupération de données OHLCV brutes depuis Binance
Ne fait AUCUN calcul d'indicateur - transmet uniquement les données brutes
"""

import asyncio
import logging
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import aiohttp
from aiohttp import ClientTimeout
import sys
import os

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import SYMBOLS
from shared.src.redis_client import RedisClient
from gateway.src.kafka_producer import get_producer

logger = logging.getLogger(__name__)

class SimpleDataFetcher:
    """
    Récupérateur de données OHLCV brutes multi-timeframes depuis Binance.
    Architecture propre : AUCUN calcul d'indicateur, transmission de données brutes uniquement.
    """
    
    def __init__(self):
        self.symbols = SYMBOLS
        self.timeframes = ['1m', '3m', '5m', '15m', '1h', '1d']
        self.redis_client = RedisClient()
        self.kafka_producer = get_producer()
        self.running = False
        
        # URLs Binance API
        self.base_url = "https://api.binance.com"
        self.klines_endpoint = "/api/v3/klines"
        
        # Configuration des timeouts
        self.timeout = ClientTimeout(total=30)
        
        # Limits de récupération par timeframe (optimisées pour EMA 99 stable)
        self.limits = {
            '1m': 300,   # 300 minutes = 5h (suffisant pour EMA 99)
            '3m': 300,   # 900 minutes = 15h (bon équilibre)
            '5m': 300,   # 1500 minutes = 25h (journée + 1h)
            '15m': 300,  # 4500 minutes = 75h (3+ jours)
            '1h': 300,   # 18000 minutes = 300h (12+ jours)
            '1d': 300    # 300 jours = 10 mois (EMA 99 très stable)
        }
        
        logger.info("📡 SimpleDataFetcher initialisé - données brutes uniquement")

    async def start(self):
        """Démarre le service de récupération de données."""
        self.running = True
        logger.info("🚀 SimpleDataFetcher démarré")
        
        try:
            # Récupération initiale des données historiques pour tous les symboles/timeframes
            await self._fetch_initial_data()
            
            # Ensuite, lancer la surveillance en continu
            await self._continuous_fetch()
            
        except Exception as e:
            logger.error(f"❌ Erreur dans SimpleDataFetcher: {e}")
        finally:
            self.running = False

    async def _fetch_initial_data(self):
        """Récupère les données historiques initiales pour tous les symboles/timeframes."""
        logger.info("📚 Récupération des données historiques initiales...")
        
        tasks = []
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                task = self._fetch_symbol_timeframe_data(symbol, timeframe)
                tasks.append(task)
        
        # Exécuter toutes les tâches en parallèle
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        total_count = len(tasks)
        
        logger.info(f"✅ Données initiales récupérées: {success_count}/{total_count}")

    async def _fetch_symbol_timeframe_data(self, symbol: str, timeframe: str):
        """Récupère les données pour un symbole/timeframe spécifique."""
        try:
            limit = self.limits.get(timeframe, 200)
            
            # URL de requête Binance
            url = f"{self.base_url}{self.klines_endpoint}"
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'limit': str(limit)
            }
            
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        klines = await response.json()
                        
                        # Traiter les données OHLCV brutes
                        processed_data = self._process_raw_klines(klines, symbol, timeframe)
                        
                        # Publier sur Kafka via Redis
                        await self._publish_to_kafka(processed_data, symbol, timeframe)
                        
                        logger.debug(f"✅ Données récupérées: {symbol} {timeframe} ({len(klines)} bougies)")
                        return True
                    else:
                        logger.error(f"❌ Erreur API Binance {response.status} pour {symbol} {timeframe}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Erreur récupération {symbol} {timeframe}: {e}")
            return False

    def _process_raw_klines(self, klines: List, symbol: str, timeframe: str) -> List[Dict]:
        """
        Traite les klines brutes pour en extraire uniquement les données OHLCV.
        AUCUN calcul d'indicateur technique.
        """
        processed_candles = []
        
        for kline in klines:
            # Extraire uniquement les données OHLCV
            candle_data = {
                'time': datetime.fromtimestamp(kline[0] / 1000).isoformat(),
                'symbol': symbol,
                'timeframe': timeframe,
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5]),
                'close_time': kline[6],
                'quote_asset_volume': float(kline[7]),
                'number_of_trades': kline[8],
                'taker_buy_base_asset_volume': float(kline[9]),
                'taker_buy_quote_asset_volume': float(kline[10]),
                'is_closed': True,  # Données historiques sont fermées
                'source': 'binance_historical'
            }
            
            processed_candles.append(candle_data)
        
        return processed_candles

    async def _publish_to_kafka(self, candles: List[Dict], symbol: str, timeframe: str):
        """Publie les données brutes sur Kafka via KafkaProducer."""
        try:
            for candle in candles:
                # Utiliser le KafkaProducer pour publier
                self.kafka_producer.publish_market_data(candle, key=symbol)
                
                logger.debug(f"📤 Données historiques publiées: {symbol} {timeframe}")
                
        except Exception as e:
            logger.error(f"❌ Erreur publication Kafka: {e}")

    async def _continuous_fetch(self):
        """Surveillance continue des nouvelles données."""
        logger.info("🔄 Démarrage de la surveillance continue...")
        
        while self.running:
            try:
                # Attendre 60 secondes entre les vérifications (moins de pression sur l'API)
                await asyncio.sleep(60)
                
                # Récupérer les dernières données pour tous les symboles avec un petit délai
                for symbol in self.symbols:
                    for timeframe in self.timeframes:
                        await self._fetch_latest_data(symbol, timeframe)
                        # Petit délai entre chaque requête pour éviter les rate limits
                        await asyncio.sleep(0.1)
                        
            except Exception as e:
                logger.error(f"❌ Erreur surveillance continue: {e}")
                await asyncio.sleep(60)  # Attendre plus longtemps en cas d'erreur

    async def _fetch_latest_data(self, symbol: str, timeframe: str):
        """Récupère uniquement les dernières données pour un symbole/timeframe."""
        try:
            # Récupérer les 5 dernières bougies pour éviter les gaps
            url = f"{self.base_url}{self.klines_endpoint}"
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'limit': '5'
            }
            
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        klines = await response.json()
                        
                        # Traiter toutes les bougies fermées (pas celle en cours)
                        if len(klines) >= 2:
                            closed_klines = klines[:-1]  # Toutes sauf la dernière (en cours)
                            
                            processed_data = self._process_raw_klines(closed_klines, symbol, timeframe)
                            await self._publish_to_kafka(processed_data, symbol, timeframe)
                            
        except Exception as e:
            logger.error(f"❌ Erreur récupération latest {symbol} {timeframe}: {e}")

    async def _fetch_period_data(self, symbol: str, timeframe: str, start_time, end_time):
        """Récupère les données pour une période spécifique (utilisé par SmartDataFetcher)."""
        try:
            # Convertir les timestamps en millisecondes pour Binance API
            start_ts = int(start_time.timestamp() * 1000)
            end_ts = int(end_time.timestamp() * 1000)
            
            url = f"{self.base_url}{self.klines_endpoint}"
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'startTime': str(start_ts),
                'endTime': str(end_ts),
                'limit': '1000'
            }
            
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        klines = await response.json()
                        
                        if klines:
                            processed_data = self._process_raw_klines(klines, symbol, timeframe)
                            await self._publish_to_kafka(processed_data, symbol, timeframe)
                            
                            logger.debug(f"✅ Période remplie: {symbol} {timeframe} ({len(klines)} bougies)")
                            return True
                    else:
                        logger.error(f"❌ Erreur API Binance {response.status} pour {symbol} {timeframe}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Erreur fetch période {symbol} {timeframe}: {e}")
            return False

    async def stop(self):
        """Arrête le service."""
        self.running = False
        logger.info("🛑 SimpleDataFetcher arrêté")

async def main():
    """Point d'entrée principal."""
    fetcher = SimpleDataFetcher()
    
    try:
        await fetcher.start()
    except KeyboardInterrupt:
        logger.info("🛑 Arrêt demandé par l'utilisateur")
    finally:
        await fetcher.stop()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())