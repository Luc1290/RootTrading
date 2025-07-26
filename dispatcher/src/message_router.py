"""
Module de routage des messages entre Kafka et Redis.
Transforme et route les messages vers les canaux appropriés.
"""
import logging
import time
import threading
from collections import deque
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from shared.src.config import SYMBOLS
from shared.src.redis_client import RedisClient
from dispatcher.src.database_persister import DatabasePersister

# Configuration du logging
logger = logging.getLogger(__name__)

class MessageRouter:
    """
    Router de messages entre Kafka et Redis.
    Reçoit les messages Kafka et les publie sur les canaux Redis appropriés.
    """
    
    def __init__(self, redis_client: Optional[RedisClient] = None):
        """
        Initialise le router de messages.
        
        Args:
            redis_client: Client Redis préexistant (optionnel)
        """
        self.redis_client = redis_client or RedisClient()
        
        # Client Redis dédié pour les messages haute priorité
        self.high_priority_redis = RedisClient()
        
        # Ajout du persister pour sauvegarder en base
        self.db_persister = DatabasePersister()
        self.db_persister.start_persister()
        
        # Préfixe pour les canaux Redis
        self.redis_prefix = "roottrading"
        
        # Mappings des topics vers les canaux
        self.topic_to_channel = {
            "market.data": "market:data",  # Multi-timeframes enrichies
            "signals.filtered": "signals:filtered",  # Signaux filtrés du signal_aggregator
            "executions": "trade:execution",
            "orders": "trade:order"
        }
        
        # Support multi-timeframes
        self.timeframes = ['1m', '3m', '5m', '15m']
        
        # Cache pour les données enrichies par symbole/timeframe
        self.enriched_data_cache: Dict[str, Dict[str, Any]] = {}
        for symbol in SYMBOLS:
            self.enriched_data_cache[symbol] = {}
            for tf in self.timeframes:
                self.enriched_data_cache[symbol][tf] = deque(maxlen=50)
        
        # Précalculer les mappings pour les topics de données de marché multi-timeframes
        self.market_data_channel_cache = {}
        for symbol in SYMBOLS:
            for tf in self.timeframes:
                topic = f"market.data.{symbol.lower()}.{tf}"
                self.market_data_channel_cache[topic] = f"{self.redis_prefix}:{self.topic_to_channel['market.data']}:{symbol.lower()}:{tf}"
        
        # File d'attente séparée pour les messages haute priorité
        self.high_priority_queue: deque = deque(maxlen=5000)
        
        # File d'attente pour les messages en cas de panne Redis
        self.message_queue: deque = deque(maxlen=10000)  # Limiter à 10 000 messages
        self.queue_lock = threading.Lock()
        self.queue_processor_running = True
        
        # Démarrer le thread de traitement de la file d'attente
        self.queue_processor_thread = threading.Thread(
            target=self._process_queue, 
            daemon=True,
            name="MessageQueueProcessor"
        )
        self.queue_processor_thread.start()
        
        # Démarrer le thread de traitement prioritaire
        self.high_priority_processor_thread = threading.Thread(
            target=self._process_high_priority_queue, 
            daemon=True,
            name="HighPriorityProcessor"
        )
        self.high_priority_processor_thread.start()
        
        # Compteurs de statistiques
        self.stats = {
            "messages_received": 0,
            "messages_routed": 0,
            "high_priority_routed": 0,
            "messages_queued": 0,
            "queue_processed": 0,
            "errors": 0,
            "last_reset": time.time()
        }
        
        # Initialiser le compteur de séquence des messages
        self.message_sequence = 0
        
        # --- Déduplication courte (1 s) --------------------
        self._dedup_cache: Dict[str, float] = {}       # {clé: timestamp}
        self._dedup_ttl   = 1.0      # secondes
        
        logger.info("✅ MessageRouter initialisé avec support priorité")

    def _determine_priority(self, topic: str) -> str:
        """
        Détermine la priorité d'un message basé sur son topic.
        
        Args:
            topic: Topic Kafka
            
        Returns:
            Priorité du message ('high' ou 'normal')
        """
        # Données de marché en temps réel = haute priorité
        if topic.startswith("market.data"):
            return 'high'
        # Signaux de trading = haute priorité
        elif topic.startswith("signals") or topic.startswith("analyzer.signals"):
            return 'high'
        # Par défaut = priorité normale
        else:
            return 'normal'

    def _publish_high_priority(self, channel: str, message: Dict[str, Any]) -> None:
        """
        Publie un message haute priorité sur Redis.
        Utilise une connexion dédiée pour les messages critiques.
        
        Args:
            channel: Canal Redis
            message: Message à publier
        """
        try:
            # Utiliser le client Redis dédié haute priorité
            self.high_priority_redis.publish(channel, message)
            self.stats["high_priority_routed"] += 1
        except Exception as e:
            # En cas d'échec, mettre en file d'attente prioritaire
            with self.queue_lock:
                self.high_priority_queue.append((channel, message))
            logger.warning(f"❗ Publication haute priorité échouée, message mis en file d'attente: {str(e)}")
            raise

    def _process_high_priority_queue(self) -> None:
        """
        Traite les messages haute priorité en file d'attente.
        Fonctionne dans un thread séparé avec traitement plus fréquent.
        """
        while self.queue_processor_running:
            try:
                # Traiter la file d'attente prioritaire si elle n'est pas vide
                with self.queue_lock:
                    queue_size = len(self.high_priority_queue)
                    if queue_size > 0:
                        # Récupérer le message le plus ancien
                        channel, message = self.high_priority_queue.popleft()
                        
                        try:
                            # Tenter de publier sur Redis
                            self.high_priority_redis.publish(channel, message)
                            
                            # Incrémenter le compteur de messages haute priorité traités
                            self.stats["high_priority_routed"] += 1
                            
                            if queue_size > 1:
                                logger.debug(f"Message haute priorité envoyé depuis la file d'attente. Restants: {queue_size - 1}")
                        except Exception as e:
                            # En cas d'échec, remettre le message dans la file d'attente
                            self.high_priority_queue.append((channel, message))
                            logger.warning(f"Échec de publication haute priorité depuis la file d'attente: {str(e)}")
                            
                            # Pause plus courte en cas d'erreur pour les messages haute priorité
                            time.sleep(0.2)
            except Exception as e:
                logger.error(f"❌ Erreur dans le traitement de la file d'attente haute priorité: {str(e)}")
            
            # Pause très courte pour les messages haute priorité
            time.sleep(0.01)  # 10 ms
    
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
                        if len(self.message_queue[0]) == 3:  # Nouveau format avec priorité
                            channel, message, priority = self.message_queue.popleft()
                        else:  # Ancien format pour rétrocompatibilité
                            channel, message = self.message_queue.popleft()
                            priority = 'normal'
                        
                        try:
                            # Tenter de publier sur Redis
                            if priority == 'high':
                                self._publish_high_priority(channel, message)
                            else:
                                self.redis_client.publish(channel, message)
                            
                            # Incrémenter le compteur de messages traités
                            self.stats["queue_processed"] += 1
                            
                            if queue_size > 1:
                                logger.info(f"Message envoyé depuis la file d'attente. Restants: {queue_size - 1}")
                        except Exception as e:
                            # En cas d'échec, remettre le message dans la file d'attente
                            self.message_queue.append((channel, message, priority))
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
        Optimisé pour les données ultra-enrichies avec indicateurs techniques.

        Args:
            topic: Topic Kafka source
            message: Message original
        
        Returns:
            Message transformé et enrichi
        """
        transformed = message.copy()
        
        # Ajouter les métadonnées de routage
        transformed["_routing"] = {
            "source_topic": topic,
            "received_at": datetime.now().isoformat(),
            "priority": self._determine_priority(topic)
        }
        
        # Traitement spécifique selon le type de topic
        if any(topic.startswith(f"market.data.") and topic.endswith(f".{tf}") for tf in ["1m", "3m", "5m", "15m", "1d"]):
            # Extraction du symbole et timeframe depuis le topic
            parts = topic.split(".")
            if len(parts) >= 4:
                symbol = parts[2]
                timeframe = parts[3]
                
                # S'assurer que le symbole et timeframe sont dans le message
                if "symbol" not in transformed:
                    transformed["symbol"] = symbol
                if "timeframe" not in transformed:
                    transformed["timeframe"] = timeframe
                    
                # Validation des champs essentiels pour les données de marché
                required_fields = ["time", "open", "high", "low", "close", "volume"]
                for field in required_fields:
                    if field not in transformed:
                        logger.warning(f"Champ requis manquant dans les données de marché: {field}")
                        
                # Architecture PROPRE: Le Gateway n'envoie que des données OHLCV brutes
                # Pas d'indicateurs techniques dans les messages du Gateway
                transformed["_routing"]["data_type"] = "raw_market_data"
                
                # Vérifier que c'est bien des données brutes (pas d'indicateurs)
                indicator_fields = ["rsi_14", "ema_12", "macd_line", "bb_upper"]
                has_indicators = any(ind in transformed for ind in indicator_fields)
                if has_indicators:
                    logger.warning(f"ATTENTION: Données avec indicateurs reçues du Gateway - architecture corrompue!")
                    transformed["_routing"]["data_type"] = "corrupted_enriched_data"
                        
        elif topic == "market.data" or topic.startswith("market.data.") and not any(topic.endswith(f".{tf}") for tf in ["1m", "3m", "5m", "15m", "1d"]):
            # Format legacy ou données brutes génériques
            transformed["_routing"]["data_type"] = "raw_market_data_legacy"
            
        elif topic == "analyzer.signals" or topic == "signals":
            transformed["_routing"]["data_type"] = "signal"
            # Validation des champs de signal
            signal_fields = ["symbol", "side", "strategy", "timestamp"]
            for field in signal_fields:
                if field not in transformed:
                    logger.warning(f"Champ signal manquant: {field}")
                    
        elif topic == "executions":
            transformed["_routing"]["data_type"] = "execution"
            
        elif topic == "orders":
            transformed["_routing"]["data_type"] = "order"
            
        return transformed

    
    def route_message(self, topic: str, message: Dict[str, Any], priority: Optional[str] = None) -> bool:
        """
        Route un message Kafka vers le canal Redis approprié avec gestion de priorité.
        
        Args:
            topic: Topic Kafka source
            message: Message à router
            priority: Priorité du message ('high', 'normal', ou None pour auto-détection)
            
        Returns:
            True si le routage a réussi, False sinon
        """
        try:
            # Incrémenter le compteur de messages reçus
            self.stats["messages_received"] += 1
            
            # ----- Déduplication ---------------------------------
            now = time.time()
            # Nettoyage des clés expirées
            self._dedup_cache = {k: t for k, t in self._dedup_cache.items() if now - t < self._dedup_ttl}

            # Clé de déduplication adaptée au type de message
            if topic.startswith("analyzer.signals"):
                # Pour les signaux : topic + symbol + side + strategy + timestamp
                dedup_key = f"{topic}:{message.get('symbol')}:{message.get('side')}:{message.get('strategy')}:{message.get('timestamp')}"
            else:
                # Pour les données de marché : topic + timestamp + close (si présent)
                dedup_key = f"{topic}:{message.get('close_time') or message.get('start_time')}:{message.get('close')}"
            if dedup_key in self._dedup_cache:
                logger.debug(f"Message ignoré (doublon) : {dedup_key}")
                return False

            # Marquer comme vu
            self._dedup_cache[dedup_key] = now
            # ----------------------------------------------------
            
            # === PERSISTANCE EN BASE DE DONNÉES ===
            # Sauvegarder les données de marché en base
            if topic.startswith("market.data") and self.db_persister:
                try:
                    self.db_persister.save_market_data(topic, message)
                    logger.debug(f"💾 Données sauvées: {topic}")
                except Exception as e:
                    logger.error(f"❌ Erreur sauvegarde base: {e}")
            # =====================================
            
            # Déterminer la priorité si non spécifiée
            if priority is None:
                # Données de marché en temps réel = haute priorité
                if topic.startswith("market.data") and message.get('is_closed', False):
                    priority = 'high'
                # Signaux = haute priorité
                elif topic.startswith("signals"):
                    priority = 'high'
                else:
                    priority = 'normal'
            
            # Déterminer le canal Redis
            channel = self._get_redis_channel(topic)
            
            if not channel:
                logger.warning(f"Aucun canal Redis trouvé pour le topic {topic}")
                return False
            
            # Transformer le message si nécessaire
            transformed_message = self._transform_message(topic, message)
            
            # Ajouter l'information de priorité
            transformed_message["_routing"]["priority"] = priority
            
            # Tentative de publication directe
            try:
                # Publier sur Redis avec priorité
                if priority == 'high':
                    # Pour les messages haute priorité, utiliser une connexion dédiée
                    self._publish_high_priority(channel, transformed_message)
                else:
                    # Publication normale
                    self.redis_client.publish(channel, transformed_message)
                
                # Incrémenter le compteur de messages routés
                self.stats["messages_routed"] += 1
                
                # Log détaillé en niveau debug
                logger.debug(f"Message routé ({priority}): {topic} -> {channel}")
                
                return True
            except Exception as e:
                # En cas d'échec, mettre le message en file d'attente
                with self.queue_lock:
                    # Ajouter la priorité dans le tuple
                    self.message_queue.append((channel, transformed_message, priority))
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
        Récupère les statistiques du router avec informations de priorité.
        
        Returns:
            Dictionnaire de statistiques
        """
        # Calculer les statistiques dérivées
        now = time.time()
        elapsed = now - self.stats["last_reset"]
        
        stats = self.stats.copy()
        stats["uptime"] = elapsed
        stats["queue_size"] = len(self.message_queue)
        stats["high_priority_queue_size"] = len(self.high_priority_queue)
        
        if elapsed > 0:
            stats["messages_per_second"] = stats["messages_routed"] / elapsed
            stats["high_priority_per_second"] = stats["high_priority_routed"] / elapsed
        else:
            stats["messages_per_second"] = 0
            stats["high_priority_per_second"] = 0
        
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
        
        # Attendre la fin des threads (avec un timeout)
        if self.queue_processor_thread and self.queue_processor_thread.is_alive():
            self.queue_processor_thread.join(timeout=2.0)
        
        if self.high_priority_processor_thread and self.high_priority_processor_thread.is_alive():
            self.high_priority_processor_thread.join(timeout=2.0)
        
        # Vider la file d'attente si possible
        try:
            with self.queue_lock:
                queue_size = len(self.message_queue)
                high_priority_size = len(self.high_priority_queue)
                if queue_size > 0 or high_priority_size > 0:
                    logger.warning(f"⚠️ Messages non traités lors de la fermeture: {queue_size} normaux, {high_priority_size} haute priorité")
        except Exception:
            pass
        
        # Fermer les clients Redis
        if self.redis_client:
            self.redis_client.close()
        
        if self.high_priority_redis:
            self.high_priority_redis.close()
        
        logger.info("Clients Redis fermés")