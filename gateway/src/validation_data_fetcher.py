"""
Service dédié pour récupérer les données de validation 15m.
Stocke les données dans Redis pour usage par le signal aggregator.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List
import aiohttp
import sys
import os

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import BINANCE_API_KEY, BINANCE_SECRET_KEY, SYMBOLS, VALIDATION_INTERVAL
from shared.src.redis_client import RedisClient

logger = logging.getLogger(__name__)


class ValidationDataFetcher:
    """
    Récupère les données 15m depuis Binance pour la validation des signaux.
    """
    
    def __init__(self):
        self.symbols = SYMBOLS
        self.interval = VALIDATION_INTERVAL
        self.redis_client = RedisClient()
        self.running = False
        
        # URLs Binance API
        self.base_url = "https://api.binance.com"
        self.klines_endpoint = "/api/v3/klines"
        
        logger.info(f"ValidationDataFetcher initialisé pour {self.symbols} en {self.interval}")
    
    async def start(self):
        """Démarre la récupération périodique des données de validation"""
        self.running = True
        logger.info("🚀 Démarrage du ValidationDataFetcher")
        
        while self.running:
            try:
                # Récupérer les données pour tous les symboles
                for symbol in self.symbols:
                    await self._fetch_and_store_validation_data(symbol)
                
                # Attendre avant la prochaine mise à jour
                # Pour 5m, on met à jour toutes les 30 secondes
                await asyncio.sleep(30)  # 30 secondes entre chaque récupération
                
            except Exception as e:
                logger.error(f"❌ Erreur dans la boucle ValidationDataFetcher: {e}")
                await asyncio.sleep(60)  # Attendre 1 minute avant de réessayer
    
    async def stop(self):
        """Arrête la récupération des données"""
        self.running = False
        logger.info("⏹️ Arrêt du ValidationDataFetcher")
    
    async def _fetch_and_store_validation_data(self, symbol: str):
        """
        Récupère les données de validation pour un symbole et les stocke dans Redis.
        
        Args:
            symbol: Symbole à analyser (ex: BTCUSDC)
        """
        try:
            # Récupérer les klines récentes (50 dernières bougies 15m)
            klines = await self._fetch_klines(symbol, limit=50)
            
            if not klines:
                logger.warning(f"Aucune donnée reçue pour {symbol}")
                return
            
            # Traiter les données
            validation_data = self._process_klines(klines)
            
            # Stocker dans Redis
            redis_key = f"market_data:{symbol}:{self.interval}"
            self._store_in_redis(redis_key, validation_data)
            
            logger.debug(f"✅ Données {self.interval} mises à jour pour {symbol}")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la mise à jour des données {symbol}: {e}")
    
    async def _fetch_klines(self, symbol: str, limit: int = 50) -> List:
        """
        Récupère les klines depuis l'API Binance.
        
        Args:
            symbol: Symbole à récupérer
            limit: Nombre de klines à récupérer
            
        Returns:
            Liste des klines
        """
        try:
            params = {
                'symbol': symbol,
                'interval': self.interval.lower(),  # 15m
                'limit': limit
            }
            
            url = f"{self.base_url}{self.klines_endpoint}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        logger.error(f"❌ Erreur API Binance {response.status} pour {symbol}")
                        return []
                        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des klines {symbol}: {e}")
            return []
    
    def _process_klines(self, klines: List) -> Dict:
        """
        Traite les klines pour extraire les données utiles.
        
        Args:
            klines: Données brutes de Binance
            
        Returns:
            Données processées pour la validation
        """
        try:
            prices = []
            volumes = []
            
            for kline in klines:
                # Format Binance: [open_time, open, high, low, close, volume, ...]
                close_price = float(kline[4])
                volume = float(kline[5])
                
                prices.append(close_price)
                volumes.append(volume)
            
            # Calculer RSI 14 si on a assez de données
            rsi_14 = self._calculate_rsi(prices, 14) if len(prices) >= 20 else None
            
            # Calculer volatilité (ATR approximé)
            atr = self._calculate_atr(klines) if len(klines) >= 14 else None
            
            validation_data = {
                'prices': prices,
                'volumes': volumes,
                'rsi_14': rsi_14,
                'atr': atr,
                'last_price': prices[-1] if prices else 0,
                'avg_volume': sum(volumes) / len(volumes) if volumes else 0,
                'timestamp': time.time(),
                'count': len(prices)
            }
            
            return validation_data
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement klines: {e}")
            return {}
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calcule le RSI"""
        if len(prices) < period + 1:
            return 50.0  # Valeur neutre
        
        try:
            deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            gains = [d if d > 0 else 0 for d in deltas]
            losses = [-d if d < 0 else 0 for d in deltas]
            
            avg_gain = sum(gains[-period:]) / period
            avg_loss = sum(losses[-period:]) / period
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return round(rsi, 2)
            
        except Exception:
            return 50.0
    
    def _calculate_atr(self, klines: List, period: int = 14) -> float:
        """Calcule l'ATR (Average True Range)"""
        if len(klines) < period + 1:
            return 0.0
        
        try:
            true_ranges = []
            
            for i in range(1, len(klines)):
                high = float(klines[i][2])
                low = float(klines[i][3])
                prev_close = float(klines[i-1][4])
                
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                true_ranges.append(tr)
            
            # ATR = moyenne des true ranges
            atr = sum(true_ranges[-period:]) / period
            return round(atr, 6)
            
        except Exception:
            return 0.0
    
    def _store_in_redis(self, key: str, data: Dict):
        """Stocke les données dans Redis avec expiration"""
        try:
            json_data = json.dumps(data)
            # Expiration de 1 heure (les données 15m se périmment vite)
            # RedisClient n'est pas async, on l'utilise de manière synchrone
            self.redis_client.set(key, json_data, expiration=3600)
            
        except Exception as e:
            logger.error(f"❌ Erreur stockage Redis {key}: {e}")


async def main():
    """Point d'entrée principal pour le service de validation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    fetcher = ValidationDataFetcher()
    
    try:
        await fetcher.start()
    except KeyboardInterrupt:
        logger.info("⏹️ Arrêt demandé par l'utilisateur")
    finally:
        await fetcher.stop()


if __name__ == "__main__":
    asyncio.run(main())