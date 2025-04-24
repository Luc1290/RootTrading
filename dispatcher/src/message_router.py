"""
Module de routage des messages entre Kafka et Redis.
Transforme et route les messages vers les canaux appropriés.
"""
import logging
import json
import time
from typing import Dict, Any, List, Optional

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
        
        # Compteurs de statistiques
        self.stats = {
            "messages_received": 0,
            "messages_routed": 0,
            "errors": 0,
            "last_reset": time.time()
        }
        
        logger.info("✅ MessageRouter initialisé")
    
    def _get_redis_channel(self, topic: str) -> Optional[str]:
        """
        Détermine le canal Redis correspondant à un topic Kafka.
        
        Args:
            topic: Topic Kafka
            
        Returns:
            Canal Redis ou None si non mappé
        """
        # Chercher le type de topic (avant le premier point)
        parts = topic.split('.')
        topic_type = parts[0]
        
        # Cas spécial pour les données de marché (inclut le symbole)
        if topic_type == "market" and len(parts) >= 3:
            symbol = parts[2]
            return f"{self.redis_prefix}:{self.topic_to_channel['market.data']}:{symbol}"
        
        # Autres types de topics
        for key, channel in self.topic_to_channel.items():
            if topic.startswith(key):
                return f"{self.redis_prefix}:{channel}"
        
        return None
    
    def _transform_message(self, topic: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transforme le message si nécessaire.
        
        Args:
            topic: Topic Kafka source
            message: Message original
            
        Returns:
            Message transformé
        """
        # Par défaut, pas de transformation
        transformed = message.copy()
        
        # Ajouter des métadonnées de routage
        transformed["_routing"] = {
            "source_topic": topic,
            "timestamp": time.time()
        }
        
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
            
            # Publier sur Redis
            self.redis_client.publish(channel, transformed_message)
            
            # Incrémenter le compteur de messages routés
            self.stats["messages_routed"] += 1
            
            # Log détaillé si nécessaire
            logger.info(f"Message routé: {topic} -> {channel}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du routage du message: {str(e)}")
            self.stats["errors"] += 1
            return False
    
    def batch_route_messages(self, messages: List[Dict[str, Any]]) -> int:
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
            "errors": 0,
            "last_reset": time.time()
        }
        
        logger.info("Statistiques du router réinitialisées")
    
    def close(self) -> None:
        """
        Ferme proprement les ressources du router.
        """
        if self.redis_client:
            self.redis_client.close()
            logger.info("Client Redis fermé")