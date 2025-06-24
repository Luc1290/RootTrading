"""
Module de gestion des abonnements Redis pour l'analyzer.
S'abonne aux canaux Redis pour recevoir les données de marché et publier les signaux.
"""
import datetime
import json
import logging
import threading
import time
from typing import Dict, Any, Callable, List, Optional
import queue

# Ajouter le répertoire parent au path pour les imports
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB, CHANNEL_PREFIX, SYMBOLS
from shared.src.redis_client import RedisClient
from shared.src.kafka_client import KafkaClient
from shared.src.schemas import StrategySignal
from analyzer.src.bar_aggregator import BarAggregator

# Configuration du logging
logger = logging.getLogger(__name__)

class RedisSubscriber:
    """
    Gestionnaire d'abonnements Redis pour l'analyzer.
    Reçoit les données de marché et publie les signaux de trading.
    """
    
    def __init__(self, symbols: List[str] = None):
        """
        Initialise le subscriber Redis.
        
        Args:
            symbols: Liste des symboles à surveiller
        """
        self.symbols = symbols or SYMBOLS
        self.redis_client = RedisClient()
        self.kafka_client = KafkaClient()
        
        # Topics Kafka multi-timeframes au lieu de Redis
        self.timeframes = ['1m', '5m', '15m', '1h', '4h']
        self.kafka_topics = []
        for symbol in self.symbols:
            for tf in self.timeframes:
                self.kafka_topics.append(f"market.data.{symbol.lower()}.{tf}")
        
        # Cache des données par symbole et timeframe
        self.data_cache = {}
        for symbol in self.symbols:
            self.data_cache[symbol] = {}
            for tf in self.timeframes:
                self.data_cache[symbol][tf] = []
        
        self.signal_channel = f"{CHANNEL_PREFIX}:analyze:signal"
        
        # Canaux Redis pour les données de marché (multi-timeframes)
        self.market_data_channels = []
        for symbol in self.symbols:
            for tf in self.timeframes:
                channel = f"{CHANNEL_PREFIX}:market:data:{symbol.lower()}:{tf}"
                self.market_data_channels.append(channel)
        
        # File d'attente thread-safe pour les données de marché
        self.market_data_queue = queue.Queue()
        
        # Thread pour le traitement des données
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        logger.info(f"✅ RedisSubscriber enrichi initialisé pour {len(self.kafka_topics)} topics")
        
        logger.info(f"✅ RedisSubscriber initialisé pour {len(self.symbols)} symboles: {', '.join(self.symbols)}")
    
    def _process_market_data(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Callback pour traiter les données de marché reçues de Redis.
        Ajoute les données à la file d'attente pour traitement.
        
        Args:
            channel: Canal Redis d'où proviennent les données
            data: Données de marché
        """
        try:
            # Traitement direct des données ultra-enrichies
            symbol = data.get("symbol")
            if not symbol:
                logger.warning(f"Données reçues sans symbole: {data}")
                return

            # Ne traiter que les bougies fermées pour éviter le spam
            if data.get("is_closed", False):
                self.market_data_queue.put((channel, data))
                logger.debug(f"📊 {symbol} : bougie fermée close={data.get('close')}")
            else:
                logger.debug(f"📊 {symbol} : bougie en cours, ignorée")

        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement des données de marché: {str(e)}")
    
    def _load_historical_data(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Charge les données historiques depuis Redis pour initialiser les stratégies.
        """
        try:
            logger.info("🔄 Chargement des données historiques depuis Redis...")
            
            for symbol in self.symbols:
                # Charger les données de marché 1m (timeframe principal)
                redis_key = f"market_data:{symbol}:1m"
                
                try:
                    raw_data = self.redis_client.get(redis_key)
                    if raw_data:
                        data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
                        
                        # Formater les données pour les stratégies
                        formatted_data = {
                            'symbol': symbol,
                            'close': data.get('close', 0),
                            'start_time': int(data.get('timestamp', time.time()) * 1000),
                            'is_closed': True,
                            'rsi_14': data.get('rsi_14'),
                            'macd_line': data.get('macd_line'),
                            'macd_signal': data.get('macd_signal'),
                            'bb_upper': data.get('bb_upper'),
                            'bb_lower': data.get('bb_lower'),
                            'ema_12': data.get('ema_12'),
                            'ema_26': data.get('ema_26'),
                            'volume': data.get('volume', 0)
                        }
                        
                        # Simuler plusieurs points de données historiques en distribuant dans le temps
                        base_time = int(time.time() * 1000) - (100 * 60 * 1000)  # 100 minutes ago
                        
                        for i in range(100):
                            historical_data = formatted_data.copy()
                            historical_data['start_time'] = base_time + (i * 60 * 1000)  # 1 minute intervals
                            # Légère variation des prix pour simuler l'historique
                            price_variation = (i % 10 - 5) * 0.001  # ±0.5% variation
                            historical_data['close'] = formatted_data['close'] * (1 + price_variation)
                            
                            callback(historical_data)
                        
                        logger.info(f"✅ Données historiques chargées pour {symbol} (100 points simulés)")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Erreur chargement données historiques {symbol}: {e}")
            
            logger.info("✅ Chargement des données historiques terminé")
            
        except Exception as e:
            logger.error(f"❌ Erreur générale lors du chargement historique: {e}")
    
    def start_listening(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Démarre l'écoute des canaux Redis pour les données de marché.
        
        Args:
            callback: Fonction appelée pour chaque donnée de marché reçue
        """
        try:
            # Charger les données historiques avant de commencer l'écoute temps réel
            self._load_historical_data(callback)
            
            # S'abonner aux canaux de données de marché
            self.redis_client.subscribe(self.market_data_channels, self._process_market_data)
            logger.info(f"✅ Abonné aux canaux Redis: {', '.join(self.market_data_channels)}")
            
            # Démarrer le thread de traitement des données
            self.stop_event.clear()
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                args=(callback,),
                daemon=True
            )
            self.processing_thread.start()
            logger.info("✅ Thread de traitement des données démarré")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du démarrage de l'écoute Redis: {str(e)}")
            raise
    
    def _processing_loop(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Boucle de traitement des données de marché.
        Cette méthode s'exécute dans un thread séparé.
        
        Args:
            callback: Fonction appelée pour chaque donnée de marché
        """
        while not self.stop_event.is_set():
            try:
                # Récupérer une donnée de la file d'attente avec timeout
                try:
                    channel, data = self.market_data_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Appeler le callback avec les données
                callback(data)
                
                # Marquer la tâche comme terminée
                self.market_data_queue.task_done()
                
            except Exception as e:
                logger.error(f"❌ Erreur dans la boucle de traitement: {str(e)}")
                time.sleep(1)  # Pause pour éviter une boucle d'erreur infinie
    
    def publish_signal(self, signal: StrategySignal) -> None:
        try:
            # Vérifier que tous les champs requis sont présents
            required_fields = ['symbol', 'strategy', 'side', 'timestamp', 'price']
            missing_fields = [field for field in required_fields if not hasattr(signal, field) or getattr(signal, field) is None]
        
            if missing_fields:
                logger.error(f"❌ Signal incomplet, ne sera pas publié. Champs manquants: {missing_fields}")
                return
            
            # Convertir le signal en dictionnaire
            signal_dict = signal.dict()
        
            # S'assurer que timestamp est converti en chaîne ISO
            if "timestamp" in signal_dict and isinstance(signal_dict["timestamp"], datetime.datetime):
                signal_dict["timestamp"] = signal_dict["timestamp"].isoformat()
        
            # S'assurer que les énums sont convertis en chaînes
            if "side" in signal_dict and not isinstance(signal_dict["side"], str):
                signal_dict["side"] = signal_dict["side"].value
        
            if "strength" in signal_dict and not isinstance(signal_dict["strength"], str):
                signal_dict["strength"] = signal_dict["strength"].value
        
            # Vérification finale avant publication
            for field in required_fields:
                if field not in signal_dict or signal_dict[field] is None:
                    logger.error(f"❌ Champ {field} manquant ou nul après conversion")
                    return
        
            # Publier sur Redis (pour le coordinator actuel)
            self.redis_client.publish(self.signal_channel, signal_dict)
            
            # Publier aussi sur Kafka (pour le signal_aggregator)
            try:
                self.kafka_client.produce('analyzer.signals', signal_dict)
                logger.info(f"✅ Signal publié sur Redis et Kafka: {signal.side} pour {signal.symbol} @ {signal.price}")
            except Exception as e:
                logger.warning(f"⚠️ Erreur publication Kafka: {e}. Signal publié sur Redis seulement.")
                logger.info(f"✅ Signal publié sur {self.signal_channel}: {signal.side} pour {signal.symbol} @ {signal.price}")
    
        except Exception as e:
            logger.error(f"❌ Erreur lors de la publication du signal: {str(e)}")
    
    def stop(self) -> None:
        """
        Arrête l'écoute Redis et nettoie les ressources.
        """
        logger.info("Arrêt du subscriber Redis...")
        
        # Signaler aux threads de s'arrêter
        self.stop_event.set()
        
        # Attendre que le thread de traitement se termine
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
            logger.info("Thread de traitement arrêté")
        
        # Se désabonner des canaux Redis
        self.redis_client.unsubscribe()
        
        # Fermer la connexion Redis
        self.redis_client.close()
        logger.info("✅ Subscriber Redis arrêté")