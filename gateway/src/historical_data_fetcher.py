"""
Module de récupération des données historiques de Binance.
Permet d'initialiser le système avec suffisamment de données historiques.
"""
import logging
import time
from typing import List, Dict, Any
import aiohttp
from datetime import datetime, timedelta
import asyncio

# Configuration du logging
logger = logging.getLogger(__name__)

class HistoricalDataFetcher:
    """
    Récupère les données historiques de Binance pour initialiser l'Analyzer.
    """
    
    def __init__(self, kafka_producer=None):
        """
        Initialise le récupérateur de données historiques.
        
        Args:
            kafka_producer: Producteur Kafka pour publier les données
        """
        self.kafka_producer = kafka_producer
        self.base_url = "https://api.binance.com/api/v3"
        
        # Limites de l'API
        self.rate_limit_per_second = 20
        self.last_request_time = 0
        
        # Session HTTP pour les requêtes
        self.session = None
        
        logger.info("✅ HistoricalDataFetcher initialisé")
    
    async def _respect_rate_limit(self):
        """
        Respecte les limites de taux de l'API Binance.
        """
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        # S'assurer d'au moins 50ms entre les requêtes (20 requêtes/s)
        if elapsed < 0.05:
            await asyncio.sleep(0.05 - elapsed)
        
        self.last_request_time = time.time()
    
    async def _ensure_session(self):
        """
        S'assure qu'une session HTTP est disponible pour les requêtes.
        """
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 1000, start_time: int = None) -> List[Dict[str, Any]]:
        """
        Récupère les chandeliers historiques de Binance.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            interval: Intervalle (ex: '1m', '5m', '1h')
            limit: Nombre max de chandeliers à récupérer (max 1000)
            start_time: Timestamp de début en millisecondes
            
        Returns:
            Liste des chandeliers
        """
        await self._respect_rate_limit()
        await self._ensure_session()
        
        url = f"{self.base_url}/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000)  # Max 1000 par requête
        }
        
        if start_time:
            params["startTime"] = start_time
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    error_msg = await response.text()
                    logger.error(f"❌ Erreur API Binance ({response.status}): {error_msg}")
                    return []
                
                response_data = await response.json()
                
                # Vérifier le format de la réponse
                if not isinstance(response_data, list):
                    logger.error(f"❌ Format de réponse inattendu: {response_data}")
                    return []
                
                # Formater les données
                klines = []
                for k in response_data:
                    if len(k) < 7:  # Vérifier le nombre minimal de champs
                        logger.warning(f"⚠️ Chandelier incomplet ignoré: {k}")
                        continue
                    
                    try:
                        kline = {
                            "symbol": symbol,
                            "start_time": k[0],  # Open time
                            "close_time": k[6],  # Close time
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[5]),
                            "is_closed": True,  # Toutes ces données sont des chandeliers fermés
                        }
                        klines.append(kline)
                    except (ValueError, IndexError) as e:
                        logger.warning(f"⚠️ Erreur lors du traitement du chandelier {k}: {str(e)}")
                
                return klines
            
        except aiohttp.ClientError as e:
            logger.error(f"❌ Erreur HTTP lors de la récupération des chandeliers: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des chandeliers: {str(e)}")
            return []
    
    async def fetch_and_publish_history(self, symbols: List[str], interval: str, days_back: int = 3) -> int:
        """
        Récupère l'historique et publie les données via Kafka.
        
        Args:
            symbols: Liste des symboles
            interval: Intervalle des chandeliers
            days_back: Nombre de jours d'historique à récupérer
            
        Returns:
            Nombre total de chandeliers récupérés
        """
        if not self.kafka_producer:
            logger.error("❌ Kafka Producer non configuré")
            return 0
        
        total_klines = 0
        
        try:
            # Créer une session HTTP pour toutes les requêtes
            async with aiohttp.ClientSession() as session:
                self.session = session
                
                # Calculer le timestamp de début
                start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
                
                for symbol in symbols:
                    logger.info(f"📈 Récupération de l'historique pour {symbol} @ {interval} (depuis {days_back} jours)...")
                    
                    # Récupérer les données par lots
                    current_start = start_time
                    all_klines = []
                    
                    while True:
                        klines = await self.get_klines(symbol, interval, 1000, current_start)
                        
                        if not klines:
                            break
                        
                        all_klines.extend(klines)
                        
                        # Mettre à jour le timestamp de début pour la prochaine requête
                        current_start = klines[-1]["close_time"] + 1
                        
                        # Si on a atteint le présent, arrêter
                        if current_start > int(time.time() * 1000):
                            break
                        
                        # Petite pause pour respecter les limites
                        await asyncio.sleep(0.1)
                    
                    # Publier les données
                    logger.info(f"📊 Publication de {len(all_klines)} chandeliers pour {symbol}...")
                    
                    for kline in all_klines:
                        # Publier via Kafka
                        topic = f"market.data.{symbol.lower()}"
                        self.kafka_producer.publish_market_data(kline)
                    
                    total_klines += len(all_klines)
                    
                    # Pause entre les symboles
                    await asyncio.sleep(1)
        finally:
            # S'assurer que la session est fermée
            if self.session and not self.session.closed:
                await self.session.close()
                self.session = None
        
        logger.info(f"✅ Total: {total_klines} chandeliers historiques récupérés et publiés")
        return total_klines
    
    async def close(self):
        """Ferme proprement les ressources."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None