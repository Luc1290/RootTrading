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
from dispatcher.src.database_persister import DatabasePersister

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
        
        # Client Redis dédié pour les messages haute priorité
        self.high_priority_redis = RedisClient()
        
        # Ajout du persister pour sauvegarder en base
        self.db_persister = DatabasePersister()
        self.db_persister.start_persister()
        
        # Préfixe pour les canaux Redis
        self.redis_prefix = "roottrading"
        
        # Mappings des topics vers les canaux (ULTRA-ENRICHI)
        self.topic_to_channel = {
            "market.data": "market:data",  # Multi-timeframes enrichies
            "analyzer.signals": "analyze:signal",  # Signaux bruts de l'analyzer
            "analyzer.signals.enhanced": "analyze:signal:enhanced",  # Signaux ultra-confluents
            "analyzer.signals.scored": "analyze:signal:scored",  # Signaux avec scoring ultra-avancé
            "signals.filtered": "signals:filtered",  # Signaux filtrés du signal_aggregator
            "signals.scored": "signals:scored",  # Signaux avec scoring
            "executions": "trade:execution",
            "orders": "trade:order"
        }
        
        # Support multi-timeframes
        self.timeframes = ['1m', '5m', '15m', '1h', '4h']
        
        # Cache pour les données enrichies par symbole/timeframe
        self.enriched_data_cache = {}
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
        self.high_priority_queue = deque(maxlen=5000)
        
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
        self._dedup_cache = {}       # {clé: timestamp}
        self._dedup_ttl   = 1.0      # secondes
        
        logger.info("✅ MessageRouter initialisé avec support priorité")

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
        Optimisé pour les données ultra-enrichies avec 20+ indicateurs techniques.

        Args:
            topic: Topic Kafka source
            message: Message original
        
        Returns:
            Message transformé et enrichi
        """
        # Cas spécial pour les données de marché en temps réel (optimisation)
        is_market_data = topic.startswith("market.data")
        is_signal_data = topic.startswith("analyzer.signals")
        is_closed_candle = is_market_data and message.get('is_closed', False)
        is_enhanced_data = message.get('enhanced', False)
        
        # Créer une copie pour éviter de modifier l'original
        transformed = message.copy() if message else {}
        
        # Traitement spécialisé pour données de marché ultra-enrichies
        if is_market_data:
            # S'assurer que tous les champs essentiels sont présents
            if 'symbol' not in transformed:
                # Extraire le symbole du topic (market.data.solusdc.5m -> SOLUSDC)
                parts = topic.split('.')
                if len(parts) >= 3:
                    transformed['symbol'] = parts[2].upper()
            
            # Extraire timeframe du topic si pas présent
            if 'timeframe' not in transformed:
                parts = topic.split('.')
                if len(parts) >= 4:
                    transformed['timeframe'] = parts[3]
            
            # S'assurer que is_closed est présent et de type booléen
            if 'is_closed' not in transformed:
                transformed['is_closed'] = False
            
            # Traitement optimisé pour données enrichies
            if is_enhanced_data and is_closed_candle:
                # Validation rapide des indicateurs critiques
                critical_indicators = ['rsi_14', 'macd_line', 'ema_12', 'bb_position', 'adx_14']
                for indicator in critical_indicators:
                    if indicator in transformed and not isinstance(transformed[indicator], (int, float, type(None))):
                        try:
                            transformed[indicator] = float(transformed[indicator]) if transformed[indicator] is not None else None
                        except (ValueError, TypeError):
                            transformed[indicator] = None
                            
                # Mise en cache des données enrichies pour confluence
                symbol = transformed.get('symbol')
                timeframe = transformed.get('timeframe')
                if symbol and timeframe and symbol in self.enriched_data_cache:
                    if timeframe in self.enriched_data_cache[symbol]:
                        self.enriched_data_cache[symbol][timeframe].append({
                            'timestamp': transformed.get('close_time', time.time()),
                            'close': transformed.get('close'),
                            'rsi_14': transformed.get('rsi_14'),
                            'macd_line': transformed.get('macd_line'),
                            'adx_14': transformed.get('adx_14'),
                            'volume_spike': transformed.get('volume_spike', False)
                        })
            else:
                # Traitement standard pour données non-enrichies
                for field in ['open', 'high', 'low', 'close', 'volume']:
                    if field in transformed and not isinstance(transformed[field], (int, float)):
                        try:
                            transformed[field] = float(transformed[field])
                        except (ValueError, TypeError):
                            transformed[field] = 0.0
        
        # Traitement spécialisé pour signaux avec scoring
        elif is_signal_data:
            # Valider les champs de signal
            if 'confidence' in transformed and not isinstance(transformed['confidence'], (int, float)):
                try:
                    transformed['confidence'] = float(transformed['confidence'])
                except (ValueError, TypeError):
                    transformed['confidence'] = 0.5
                    
            # Traitement des métadonnées ultra-confluentes
            if 'metadata' in transformed and isinstance(transformed['metadata'], dict):
                metadata = transformed['metadata']
                
                # Valider les scores si présents
                if 'total_score' in metadata:
                    try:
                        metadata['total_score'] = float(metadata['total_score'])
                    except (ValueError, TypeError):
                        metadata['total_score'] = 50.0
                        
                # Compter les confirmations
                if 'total_confirmations' in metadata and isinstance(metadata['total_confirmations'], list):
                    transformed['confirmation_count'] = len(metadata['total_confirmations'])
        
        # Ajouter des métadonnées de routage ultra-enrichies
        transformed["_routing"] = {
            "source_topic": topic,
            "timestamp": time.time(),
            "sequence": self.message_sequence + 1,
            "enhanced": is_enhanced_data,
            "priority": "high" if (is_closed_candle and is_enhanced_data) or is_signal_data else "normal"
        }
        
        # Incrémenter la séquence pour le prochain message
        self.message_sequence += 1
        
        # Logs optimisés pour données critiques
        if is_closed_candle and is_enhanced_data:
            symbol = transformed.get('symbol')
            timeframe = transformed.get('timeframe')
            close = transformed.get('close')
            rsi = transformed.get('rsi_14')
            logger.info(f"📊 ENRICHI {symbol} {timeframe}: close={close:.4f} RSI={rsi:.1f}" if rsi else f"📊 ENRICHI {symbol} {timeframe}: close={close:.4f}")
        elif is_signal_data:
            symbol = transformed.get('symbol')
            side = transformed.get('side')
            score = transformed.get('metadata', {}).get('total_score')
            conf_count = transformed.get('confirmation_count', 0)
            logger.info(f"🎯 SIGNAL {symbol}: {side} score={score:.1f} conf={conf_count}" if score else f"🎯 SIGNAL {symbol}: {side} conf={conf_count}")
        
        return transformed
    
    def route_message(self, topic: str, message: Dict[str, Any], priority: str = None) -> bool:
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

            # Clé = topic + timestamp + close (si présent)
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
        except:
            pass
        
        # Fermer les clients Redis
        if self.redis_client:
            self.redis_client.close()
        
        if self.high_priority_redis:
            self.high_priority_redis.close()
        
        logger.info("Clients Redis fermés")