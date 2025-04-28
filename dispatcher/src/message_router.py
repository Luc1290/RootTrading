"""
Module de routage des messages entre Kafka et Redis.
Transforme et route les messages vers les canaux appropriés.
"""
import logging
import json
import time
import threading
from collections import deque
from typing import Dict, Any, List, Optional, Tuple

from shared.src.config import SYMBOLS
from shared.src.redis_client import RedisClient

# Configuration du logging
logger = logging.getLogger(__name__)

class MessageRouter:
    """
    Router de messages entre Kafka et Redis.
    Reçoit les messages Kafka et les publie sur les canaux Redis appropriés.
    """
    
    def __init__(self, redis_client: RedisClient = None):
        """
        Initialise le router de messages.
        
        Args:
            redis_client: Client Redis préexistant (optionnel)
        """
        self.redis_client = redis_client or RedisClient()
        
        # Préfixe pour les canaux Redis
        self.redis_prefix = "roottrading"
        
        # Mappings des topics vers les canaux
        self.topic_to_channel = {
            "market.data": "market:data",  # Le symbole sera ajouté dynamiquement
            "signals": "analyze:signal",
            "executions": "trade:execution",
            "orders": "trade:order"
        }
        
        # Précalculer les mappings pour les topics de données de marché
        self.market_data_channel_cache = {}
        for symbol in SYMBOLS:
            topic = f"market.data.{symbol.lower()}"
            self.market_data_channel_cache[topic] = f"{self.redis_prefix}:{self.topic_to_channel['market.data']}:{symbol.lower()}"
        
        # File d'attente pour les messages en cas de panne Redis
        self.message_queue = deque(maxlen=10000)  # Limiter à 10 000 messages
        self.queue_lock = threading.Lock()
        self.queue_processor_running = True
        
        # Démarrer le thread de traitement de la file d'attente
        self.queue_processor_thread = threading.Thread(
            target=self._process_queue, 
            daemon=True,
            name="MessageQueueProcessor"
        )
        self.queue_processor_thread.start()
        
        # Compteurs de statistiques
        self.stats = {
            "messages_received": 0,
            "messages_routed": 0,
            "messages_queued": 0,
            "queue_processed": 0,
            "errors": 0,
            "last_reset": time.time()
        }
        
        # Initialiser le compteur de séquence des messages
        self.message_sequence = 0
        
        logger.info("✅ MessageRouter initialisé")
    
    def _process_queue(self) -> None:
        """
        Traite les messages dans la file d'attente en cas de panne Redis.
        Fonctionne dans un thread séparé.
        """
        while self.queue_processor_running:
            try:
                # Traiter la file d'attente si elle n'est pas vide
                with self.queue_lock:
                    queue_size = len(self.message_queue)
                    if queue_size > 0:
                        # Récupérer le message le plus ancien
                        channel, message = self.message_queue.popleft()
                        
                        try:
                            # Tenter de publier sur Redis
                            self.redis_client.publish(channel, message)
                            
                            # Incrémenter le compteur de messages traités
                            self.stats["queue_processed"] += 1
                            
                            if queue_size > 1:
                                logger.info(f"Message envoyé depuis la file d'attente. Restants: {queue_size - 1}")
                        except Exception as e:
                            # En cas d'échec, remettre le message dans la file d'attente
                            self.message_queue.append((channel, message))
                            logger.warning(f"Échec de publication depuis la file d'attente: {str(e)}")
                            
                            # Pause plus longue en cas d'erreur
                            time.sleep(1.0)
            except Exception as e:
                logger.error(f"❌ Erreur dans le traitement de la file d'attente: {str(e)}")
            
            # Courte pause pour éviter de consommer trop de CPU
            time.sleep(0.1)
    
    def _get_redis_channel(self, topic: str) -> Optional[str]:
        """
        Détermine le canal Redis correspondant à un topic Kafka.
        
        Args:
            topic: Topic Kafka
            
        Returns:
            Canal Redis ou None si non mappé
        """
        # Vérifier d'abord dans le cache précalculé
        if topic in self.market_data_channel_cache:
            return self.market_data_channel_cache[topic]
        
        # Chercher le type de topic (avant le premier point)
        parts = topic.split('.')
        topic_type = parts[0]
        
        # Cas spécial pour les données de marché (inclut le symbole)
        if topic_type == "market" and len(parts) >= 3:
            symbol = parts[2]
            channel = f"{self.redis_prefix}:{self.topic_to_channel['market.data']}:{symbol}"
            
            # Ajouter au cache pour les futures requêtes
            self.market_data_channel_cache[topic] = channel
            return channel
        
        # Autres types de topics
        for key, channel in self.topic_to_channel.items():
            if topic.startswith(key):
                return f"{self.redis_prefix}:{channel}"
        
        return None
    
    def _transform_message(self, topic: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transforme le message si nécessaire et assure que tous les champs essentiels sont présents.
    
        Args:
            topic: Topic Kafka source
            message: Message original
        
        Returns:
            Message transformé
        """
        # Créer une copie pour éviter de modifier l'original
        transformed = message.copy() if message else {}
    
        # S'assurer que le message est complet pour les données de marché
        if topic.startswith("market.data"):
            # S'assurer que tous les champs essentiels sont présents
            if 'symbol' not in transformed:
                # Extraire le symbole du topic (market.data.btcusdc -> BTCUSDC)
                parts = topic.split('.')
                if len(parts) >= 3:
                    transformed['symbol'] = parts[2].upper()
        
            # S'assurer que is_closed est présent et de type booléen
            if 'is_closed' not in transformed:
                transformed['is_closed'] = False
        
            # S'assurer que les champs numériques sont bien des nombres
            for field in ['open', 'high', 'low', 'close', 'volume']:
                if field in transformed and not isinstance(transformed[field], (int, float)):
                    try:
                        transformed[field] = float(transformed[field])
                    except (ValueError, TypeError):
                        transformed[field] = 0.0
    
        # Ajouter des métadonnées de routage
        transformed["_routing"] = {
            "source_topic": topic,
            "timestamp": time.time(),
            "sequence": self.message_sequence + 1
        }
    
        # Incrémenter la séquence pour le prochain message
        self.message_sequence += 1
    
        # Ajouter plus de logs pour le débogage (uniquement pour les chandeliers fermés)
        if topic.startswith("market.data") and transformed.get('is_closed', False):
            logger.info(f"Message transformé: {topic} -> {transformed.get('symbol')} @ {transformed.get('close')}")
    
        return transformed
    
    def route_message(self, topic: str, message: Dict[str, Any]) -> bool:
        """
        Route un message Kafka vers le canal Redis approprié.
        
        Args:
            topic: Topic Kafka source
            message: Message à router
            
        Returns:
            True si le routage a réussi, False sinon
        """
        try:
            # Incrémenter le compteur de messages reçus
            self.stats["messages_received"] += 1
            
            # Déterminer le canal Redis
            channel = self._get_redis_channel(topic)
            
            if not channel:
                logger.warning(f"Aucun canal Redis trouvé pour le topic {topic}")
                return False
            
            # Transformer le message si nécessaire
            transformed_message = self._transform_message(topic, message)
            
            # Tentative de publication directe
            try:
                # Publier sur Redis
                self.redis_client.publish(channel, transformed_message)
                
                # Incrémenter le compteur de messages routés
                self.stats["messages_routed"] += 1
                
                # Log détaillé en niveau debug
                logger.debug(f"Message routé: {topic} -> {channel}")
                
                return True
            except Exception as e:
                # En cas d'échec, mettre le message en file d'attente
                with self.queue_lock:
                    self.message_queue.append((channel, transformed_message))
                    self.stats["messages_queued"] += 1
                
                logger.warning(f"❗ Publication Redis échouée, message mis en file d'attente: {str(e)}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du routage du message: {str(e)}")
            self.stats["errors"] += 1
            return False
    
    def batch_route_messages(self, messages: List[Tuple[str, Dict[str, Any]]]) -> int:
        """
        Route un lot de messages.
        
        Args:
            messages: Liste de tuples (topic, message)
            
        Returns:
            Nombre de messages routés avec succès
        """
        success_count = 0
        
        for topic, message in messages:
            if self.route_message(topic, message):
                success_count += 1
        
        return success_count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Récupère les statistiques du router.
        
        Returns:
            Dictionnaire de statistiques
        """
        # Calculer les statistiques dérivées
        now = time.time()
        elapsed = now - self.stats["last_reset"]
        
        stats = self.stats.copy()
        stats["uptime"] = elapsed
        stats["queue_size"] = len(self.message_queue)
        
        if elapsed > 0:
            stats["messages_per_second"] = stats["messages_routed"] / elapsed
        else:
            stats["messages_per_second"] = 0
        
        stats["success_rate"] = (
            stats["messages_routed"] / max(stats["messages_received"], 1) * 100
        )
        
        return stats
    
    def reset_stats(self) -> None:
        """
        Réinitialise les compteurs de statistiques.
        """
        self.stats = {
            "messages_received": 0,
            "messages_routed": 0,
            "messages_queued": 0,
            "queue_processed": 0,
            "errors": 0,
            "last_reset": time.time()
        }
        
        logger.info("Statistiques du router réinitialisées")
    
    def close(self) -> None:
        """
        Ferme proprement les ressources du router.
        """
        logger.info("Arrêt du MessageRouter...")
        
        # Arrêter le thread de traitement de la file d'attente
        self.queue_processor_running = False
        
        # Attendre la fin du thread (avec un timeout)
        if self.queue_processor_thread and self.queue_processor_thread.is_alive():
            self.queue_processor_thread.join(timeout=2.0)
        
        # Vider la file d'attente si possible
        try:
            with self.queue_lock:
                queue_size = len(self.message_queue)
                if queue_size > 0:
                    logger.warning(f"⚠️ {queue_size} messages restent dans la file d'attente lors de la fermeture")
        except:
            pass
        
        # Fermer le client Redis
        if self.redis_client:
            self.redis_client.close()
            logger.info("Client Redis fermé")