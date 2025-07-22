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

logger = logging.getLogger(__name__)

class SimpleDataFetcher:
    """
    Récupérateur de données OHLCV brutes multi-timeframes depuis Binance.
    Architecture propre : AUCUN calcul d'indicateur, transmission de données brutes uniquement.
    """
    
    def __init__(self):
        self.symbols = SYMBOLS
        self.timeframes = ['1m', '3m', '5m', '15m', '1d']
        self.redis_client = RedisClient()
        self.running = False
        
        # URLs Binance API
        self.base_url = "https://api.binance.com"
        self.klines_endpoint = "/api/v3/klines"
        
        # Configuration des timeouts
        self.timeout = ClientTimeout(total=30)
        
        # Limits de récupération par timeframe (optimisées)
        self.limits = {
            '1m': 100,   # 100 minutes = 1h40 (buffer technique)
            '3m': 100,   # 300 minutes = 5h (session trading)
            '5m': 100,   # 500 minutes = 8h20 (journée complète)
            '15m': 200,  # 3000 minutes = 50h (quelques jours)
            '1d': 365    # 365 jours = 1 an (analyse long terme)
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
                'limit': limit
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
        """Publie les données brutes sur Kafka via Redis."""
        try:
            topic = f"market.data.{symbol.lower()}.{timeframe}"
            
            for candle in candles:
                # Publier chaque bougie individuellement
                message = json.dumps(candle)
                
                # Utiliser Redis pour transmettre au dispatcher/Kafka
                redis_channel = f"roottrading:raw_market_data:{symbol.lower()}:{timeframe}"
                
                self.redis_client.publish(redis_channel, candle)
                
                logger.debug(f"📤 Données publiées: {topic}")
                
        except Exception as e:
            logger.error(f"❌ Erreur publication Kafka: {e}")

    async def _continuous_fetch(self):
        """Surveillance continue des nouvelles données."""
        logger.info("🔄 Démarrage de la surveillance continue...")
        
        while self.running:
            try:
                # Attendre 30 secondes entre les vérifications
                await asyncio.sleep(30)
                
                # Récupérer les dernières données pour tous les symboles
                for symbol in self.symbols:
                    for timeframe in self.timeframes:
                        await self._fetch_latest_data(symbol, timeframe)
                        
            except Exception as e:
                logger.error(f"❌ Erreur surveillance continue: {e}")
                await asyncio.sleep(60)  # Attendre plus longtemps en cas d'erreur

    async def _fetch_latest_data(self, symbol: str, timeframe: str):
        """Récupère uniquement les dernières données pour un symbole/timeframe."""
        try:
            # Récupérer seulement les 2 dernières bougies (dont celle en cours)
            url = f"{self.base_url}{self.klines_endpoint}"
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'limit': 2
            }
            
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        klines = await response.json()
                        
                        # Ne traiter que la bougie fermée (pas celle en cours)
                        if len(klines) >= 2:
                            closed_kline = [klines[-2]]  # Avant-dernière = fermée
                            
                            processed_data = self._process_raw_klines(closed_kline, symbol, timeframe)
                            await self._publish_to_kafka(processed_data, symbol, timeframe)
                            
        except Exception as e:
            logger.error(f"❌ Erreur récupération latest {symbol} {timeframe}: {e}")

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