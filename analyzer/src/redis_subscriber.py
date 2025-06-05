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
        self.market_data_channels = [f"{CHANNEL_PREFIX}:market:data:{symbol.lower()}" for symbol in self.symbols]
        # Agrégateurs par symbole
        self.aggregators = {sym: BarAggregator() for sym in self.symbols}
        self.signal_channel = f"{CHANNEL_PREFIX}:analyze:signal"
        
        # File d'attente thread-safe pour les données de marché
        self.market_data_queue = queue.Queue()
        
        # Thread pour le traitement des données
        self.processing_thread = None
        self.stop_event = threading.Event()
        
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
            # Agrégation : ne pousser que les bougies fermées
            symbol = data.get("symbol")
            if symbol not in self.aggregators:
                return

            bar = self.aggregators[symbol].add(data)
            if bar is None:
                return          # bougie pas encore fermée

            self.market_data_queue.put((channel, bar))

            # Log uniquement sur bougie fermée
            if bar.get("is_closed", False):
                logger.info(f"📊 {symbol} : bougie 1 min close={bar['close']}")

        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement des données de marché: {str(e)}")
    
    def start_listening(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Démarre l'écoute des canaux Redis pour les données de marché.
        
        Args:
            callback: Fonction appelée pour chaque donnée de marché reçue
        """
        try:
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