"""
Client Redis amélioré avec pool de connexions pour la communication entre services.
Optimisé pour les performances et la résilience.
"""
import json
import logging
from typing import Any, Dict, Callable, Optional, List, Union, Tuple
import threading
import time
import random
from queue import Queue, Empty

import redis
from redis import Redis, ConnectionPool
from redis.exceptions import ConnectionError, TimeoutError, RedisError

from .config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB
import numpy as np
from decimal import Decimal

class NumpyEncoder(json.JSONEncoder):
    """
    Convertit automatiquement les types NumPy et Decimal
    pour qu'ils soient sérialisables en JSON.
    """
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

# Configuration du logging
logger = logging.getLogger(__name__)

class RedisMetrics:
    """Collecte des métriques sur l'utilisation de Redis."""
    
    def __init__(self):
        self.operations = 0
        self.publish_count = 0
        self.errors = 0
        self.reconnections = 0
        self.operation_times = []  # Durées des 100 dernières opérations
        self.lock = threading.RLock()
    
    def record_operation(self, duration: float):
        """Enregistre la durée d'une opération."""
        with self.lock:
            self.operations += 1
            self.operation_times.append(duration)
            # Garder seulement les 100 dernières opérations
            if len(self.operation_times) > 100:
                self.operation_times.pop(0)
    
    def record_publish(self):
        """Incrémente le compteur de publications."""
        with self.lock:
            self.publish_count += 1
    
    def record_error(self):
        """Incrémente le compteur d'erreurs."""
        with self.lock:
            self.errors += 1
    
    def record_reconnection(self):
        """Incrémente le compteur de reconnexions."""
        with self.lock:
            self.reconnections += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'utilisation."""
        with self.lock:
            avg_time = sum(self.operation_times) / max(len(self.operation_times), 1)
            return {
                "operations": self.operations,
                "publishes": self.publish_count,
                "errors": self.errors,
                "reconnections": self.reconnections,
                "avg_operation_time": avg_time,
                "max_operation_time": max(self.operation_times) if self.operation_times else 0
            }

class RedisClientPool:
    """
    Client Redis avec pool de connexions pour performances accrues et résilience.
    """
    _instance = None
    
    @classmethod
    def get_instance(cls, host: str = REDIS_HOST, port: int = REDIS_PORT, 
                    password: str = REDIS_PASSWORD, db: int = REDIS_DB):
        """Implémentation Singleton pour accès global."""
        if cls._instance is None:
            cls._instance = RedisClientPool(host, port, password, db)
        return cls._instance
    
    def __init__(self, host: str = REDIS_HOST, port: int = REDIS_PORT, 
                password: str = REDIS_PASSWORD, db: int = REDIS_DB):
        """Initialise le client Redis avec un pool de connexions."""
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        
        # Métriques
        self.metrics = RedisMetrics()
        
        # Créer le pool de connexions
        self.connection_pool = self._create_connection_pool()
        
        # Créer une connexion partagée pour les opérations simples
        self.redis = Redis(connection_pool=self.connection_pool)
        
        # Pour stocker les abonnements PubSub actifs et leurs messages
        self.pubsub_connections: Dict[str, Any] = {}
        self.pubsub_threads: Dict[str, threading.Thread] = {}
        self.pubsub_callbacks: Dict[str, Callable] = {}
        self.pubsub_channels: Dict[str, List[str]] = {}
        self.pubsub_stop_events: Dict[str, threading.Event] = {}
        
        # Message queue pour chaque PubSub (évite les blocages lors des callbacks)
        self.message_queues: Dict[str, Queue] = {}
        self.processor_threads: Dict[str, threading.Thread] = {}
        
        # Verrou pour les opérations pubsub
        self.pubsub_lock = threading.RLock()
        
        logger.info(f"✅ Pool de connexions Redis initialisé pour {host}:{port} (DB: {db})")
    
    def _create_connection_pool(self) -> ConnectionPool:
        """Crée et retourne un pool de connexions Redis."""
        try:
            connection_params = {
                'host': self.host,
                'port': self.port,
                'db': self.db,
                'socket_timeout': 5,
                'socket_connect_timeout': 5,
                'retry_on_timeout': True,
                'decode_responses': True,  # Décode les réponses en strings
                'health_check_interval': 30  # Vérifie la santé des connexions toutes les 30s
            }
        
            # Ajouter le mot de passe si défini
            if self.password:
                connection_params['password'] = self.password
        
            # Créer le pool avec 5 connexions minimum et 50 maximum
            pool = ConnectionPool(max_connections=50, **connection_params)
            
            # Tester le pool de connexions
            test_redis = Redis(connection_pool=pool)
            test_redis.ping()
            
            logger.info(f"✅ Pool de connexions Redis créé à {self.host}:{self.port} (DB: {self.db})")
            return pool
            
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"❌ Erreur de création du pool de connexions Redis: {str(e)}")
            raise
    
    def decrbyfloat(self, key: str, value: float) -> float:
        """
        Décrémente la valeur d'une clé par un nombre à virgule flottante.
        Équivalent à la commande Redis DECRBY mais pour les floats.
        
        Args:
            key: Clé dont la valeur sera décrémentée
            value: Valeur à décrémenter (float)
            
        Returns:
            Nouvelle valeur de la clé après décrémentation
        """
        # Redis n'a pas de commande native DECRBYFLOAT, nous utilisons INCRBYFLOAT avec une valeur négative
        return self._retry_operation(self.redis.incrbyfloat, key, -abs(float(value)))

    def _retry_operation(self, operation_func, *args, max_retries=3, **kwargs):
        """
        Exécute une opération Redis avec retry automatique.
        
        Args:
            operation_func: Fonction Redis à appeler
            *args: Arguments pour la fonction
            max_retries: Nombre maximum de tentatives
            **kwargs: Arguments nommés pour la fonction
            
        Returns:
            Résultat de l'opération
        """
        retry_count = 0
        last_error = None
        start_time = time.time()
        
        while retry_count <= max_retries:
            try:
                result = operation_func(*args, **kwargs)
                
                # Enregistrer les métriques (succès)
                duration = time.time() - start_time
                self.metrics.record_operation(duration)
                
                return result
                
            except (ConnectionError, TimeoutError) as e:
                retry_count += 1
                last_error = e
                self.metrics.record_error()
                
                # Pause exponentielle avec jitter pour éviter la tempête de reconnexions
                if retry_count <= max_retries:
                    wait_time = (0.1 * (2 ** retry_count)) + (random.random() * 0.1)
                    logger.warning(f"⚠️ Erreur Redis (tentative {retry_count}/{max_retries}), "
                                   f"nouvelle tentative dans {wait_time:.2f}s: {str(e)}")
                    time.sleep(wait_time)
                    
                    # Tenter une reconnexion
                    try:
                        self._reconnect()
                    except Exception as reconnect_error:
                        logger.error(f"❌ Échec de reconnexion: {str(reconnect_error)}")
                else:
                    logger.error(f"❌ Échec de l'opération Redis après {max_retries} tentatives")
                    raise last_error
                    
            except Exception as e:
                # Autres erreurs non liées à la connexion
                self.metrics.record_error()
                logger.error(f"❌ Erreur Redis: {str(e)}")
                raise
        
        # Si on arrive ici, c'est que l'erreur persiste
        raise last_error
    
    def _reconnect(self):
        """Réinitialise le pool de connexions."""
        try:
            logger.info("⚙️ Tentative de reconnexion Redis...")
            self.metrics.record_reconnection()
            
            # Créer un nouveau pool
            self.connection_pool = self._create_connection_pool()
            
            # Réinitialiser la connexion principale
            self.redis = Redis(connection_pool=self.connection_pool)
            
            # Tester la connexion
            self.redis.ping()
            
            logger.info("✅ Reconnexion Redis réussie!")
            
            # Réinitialiser les connexions pubsub actives
            with self.pubsub_lock:
                for client_id in list(self.pubsub_connections.keys()):
                    channels = self.pubsub_channels.get(client_id, [])
                    callback = self.pubsub_callbacks.get(client_id, None)
                    
                    if channels and callback:
                        # Arrêter le thread existant
                        if client_id in self.pubsub_stop_events:
                            self.pubsub_stop_events[client_id].set()
                        
                        # Attendre que le thread se termine
                        if client_id in self.pubsub_threads:
                            thread = self.pubsub_threads[client_id]
                            if thread.is_alive():
                                thread.join(timeout=2.0)
                        
                        # Créer une nouvelle connexion et s'abonner
                        self._subscribe_internal(client_id, channels, callback)
                        logger.info(f"✅ Réabonnement aux canaux pour client {client_id}: {channels}")
                        
        except Exception as e:
            logger.error(f"❌ Échec de la reconnexion Redis: {str(e)}")
            raise
    
    def _format_value(self, value: Any) -> Any:
        """
        Formate une valeur pour le stockage dans Redis.
        
        Args:
            value: Valeur à formater
            
        Returns:
            Valeur formatée pour Redis
        """
        if isinstance(value, (dict, list)):
            return json.dumps(value, cls=NumpyEncoder)
        elif not isinstance(value, (str, int, float, bool, bytes, bytearray)):
            return str(value)
        return value
    
    def _parse_value(self, value: Any) -> Any:
        """
        Parse une valeur récupérée depuis Redis.
        
        Args:
            value: Valeur à analyser
            
        Returns:
            Valeur analysée
        """
        if not value:
            return value
            
        if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value
    
    def _subscribe_internal(self, client_id: str, channels: List[str], callback: Callable):
        """
        Implémentation interne de l'abonnement aux canaux.
        
        Args:
            client_id: Identifiant unique du client PubSub
            channels: Liste des canaux à suivre
            callback: Fonction de rappel pour les messages
        """
        # Créer un nouvel objet PubSub
        pubsub = self.redis.pubsub(ignore_subscribe_messages=True)
        
        # S'abonner aux canaux
        pubsub.subscribe(*channels)
        
        # Créer une queue de messages et un drapeau d'arrêt
        message_queue: Queue = Queue()
        stop_event = threading.Event()
        
        # Stocker les références
        self.pubsub_connections[client_id] = pubsub
        self.pubsub_channels[client_id] = channels
        self.pubsub_callbacks[client_id] = callback
        self.pubsub_stop_events[client_id] = stop_event
        self.message_queues[client_id] = message_queue
        
        # Démarrer le thread d'écoute
        listen_thread = threading.Thread(
            target=self._listen_for_messages,
            args=(client_id, pubsub, message_queue, stop_event),
            daemon=True
        )
        self.pubsub_threads[client_id] = listen_thread
        listen_thread.start()
        
        # Démarrer le thread de traitement
        processor_thread = threading.Thread(
            target=self._process_messages,
            args=(client_id, message_queue, callback, stop_event),
            daemon=True
        )
        self.processor_threads[client_id] = processor_thread
        processor_thread.start()
    
    def _listen_for_messages(self, client_id: str, pubsub, message_queue: Queue, stop_event: threading.Event):
        """
        Écoute les messages sur un pubsub et les met dans une queue.
        
        Args:
            client_id: Identifiant du client
            pubsub: Objet PubSub Redis
            message_queue: Queue pour stocker les messages
            stop_event: Event pour signaler l'arrêt
        """
        retry_count = 0
        max_retries = 10
        delay = 0.1
        
        while not stop_event.is_set():
            try:
                # Récupérer les messages avec un court timeout
                message = pubsub.get_message(timeout=0.1)
                
                if message and message['type'] == 'message':
                    # Mettre le message dans la queue
                    message_queue.put((message['channel'], message['data']))
                
                # Courte pause pour éviter de monopoliser le CPU
                time.sleep(0.001)
                
                # Réinitialiser le compteur de tentatives après un succès
                retry_count = 0
                delay = 0.1
                
            except (ConnectionError, TimeoutError) as e:
                retry_count += 1
                
                if retry_count > max_retries:
                    logger.error(f"❌ Trop d'erreurs de connexion dans le thread PubSub {client_id}, arrêt")
                    stop_event.set()
                    break
                
                logger.warning(f"⚠️ Erreur de connexion dans le thread PubSub {client_id} "
                               f"(tentative {retry_count}/{max_retries}): {str(e)}")
                
                # Backoff exponentiel
                time.sleep(delay)
                delay = min(delay * 2, 30)  # Max 30 secondes
                
                # Tenter une reconnexion à Redis
                try:
                    # Créer une nouvelle connexion PubSub
                    new_pubsub = self.redis.pubsub(ignore_subscribe_messages=True)
                    
                    # S'abonner aux mêmes canaux
                    channels = self.pubsub_channels.get(client_id, [])
                    if channels:
                        new_pubsub.subscribe(*channels)
                    
                    # Remplacer l'ancien pubsub
                    with self.pubsub_lock:
                        self.pubsub_connections[client_id] = new_pubsub
                    
                    # Remplacer le pubsub local
                    pubsub = new_pubsub
                    
                    logger.info(f"✅ Reconnexion PubSub réussie pour {client_id}")
                except Exception as reconnect_error:
                    logger.error(f"❌ Échec de reconnexion PubSub pour {client_id}: {str(reconnect_error)}")
                
            except Exception as e:
                logger.error(f"❌ Erreur dans le thread d'écoute PubSub {client_id}: {str(e)}")
                time.sleep(1)  # Pause pour éviter de consommer trop de CPU
    
    def _process_messages(self, client_id: str, message_queue: Queue, callback: Callable, stop_event: threading.Event):
        """
        Traite les messages de la queue et appelle le callback.
        
        Args:
            client_id: Identifiant du client
            message_queue: Queue contenant les messages
            callback: Fonction à appeler avec (channel, data)
            stop_event: Event pour signaler l'arrêt
        """
        while not stop_event.is_set():
            try:
                # Attendre un message avec timeout
                try:
                    channel, data = message_queue.get(timeout=0.5)
                except Empty:
                    continue
                
                # Parser le message si c'est du JSON
                data = self._parse_value(data)
                
                # Appeler le callback
                try:
                    callback(channel, data)
                except Exception as e:
                    logger.error(f"❌ Erreur dans le callback PubSub pour {client_id}: {str(e)}")
                
                # Indiquer que le traitement est terminé
                message_queue.task_done()
                
            except Exception as e:
                logger.error(f"❌ Erreur dans le thread de traitement PubSub {client_id}: {str(e)}")
                time.sleep(0.1)  # Courte pause pour éviter de consommer trop de CPU
    
    # === API PUBLIQUE ===
    
    def publish(self, channel: str, message: Any) -> int:
        """
        Publie un message sur un canal Redis.
        
        Args:
            channel: Canal de publication
            message: Message à publier
            
        Returns:
            Nombre de clients qui ont reçu le message
        """
        # Formater le message
        formatted_message = self._format_value(message)
        
        # Publier avec retry
        result = self._retry_operation(self.redis.publish, channel, formatted_message)
        
        # Enregistrer la métrique
        self.metrics.record_publish()
        
        return result
    
    def subscribe(self, channels: Union[str, List[str]], callback: Callable[[str, Any], None]) -> str:
        """
        S'abonne à un ou plusieurs canaux Redis.
        
        Args:
            channels: Canal unique ou liste de canaux
            callback: Fonction appelée avec (channel, message)
            
        Returns:
            ID du client PubSub (à utiliser pour se désabonner)
        """
        if isinstance(channels, str):
            channels = [channels]
        
        # Générer un ID unique pour ce client
        client_id = f"pubsub-{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
        
        # S'abonner de manière thread-safe
        with self.pubsub_lock:
            self._subscribe_internal(client_id, channels, callback)
        
        logger.info(f"✅ Abonné aux canaux Redis: {', '.join(channels)} (ID: {client_id})")
        return client_id
    
    def unsubscribe(self, client_id: str) -> None:
        """
        Désabonne un client PubSub.
        
        Args:
            client_id: ID du client retourné par subscribe()
        """
        with self.pubsub_lock:
            if client_id in self.pubsub_stop_events:
                # Signaler l'arrêt des threads
                self.pubsub_stop_events[client_id].set()
            
            # Fermer la connexion PubSub
            if client_id in self.pubsub_connections:
                try:
                    pubsub = self.pubsub_connections[client_id]
                    pubsub.unsubscribe()
                    pubsub.close()
                except Exception as e:
                    logger.warning(f"⚠️ Erreur lors de la fermeture PubSub {client_id}: {str(e)}")
            
            # Attendre la fin des threads
            for thread_dict, thread_name in [(self.pubsub_threads, "listener"), 
                                           (self.processor_threads, "processor")]:
                if client_id in thread_dict:
                    thread = thread_dict[client_id]
                    if thread.is_alive():
                        thread.join(timeout=2.0)
                        if thread.is_alive():
                            logger.warning(f"⚠️ Thread {thread_name} pour {client_id} "
                                          f"ne s'est pas terminé proprement")
            
            # Nettoyer les références
            for container in [self.pubsub_connections, self.pubsub_channels, 
                             self.pubsub_callbacks, self.pubsub_stop_events,
                             self.message_queues, self.pubsub_threads,
                             self.processor_threads]:
                if client_id in container:
                    del container[client_id]
        
        logger.info(f"✅ Désabonné du client PubSub {client_id}")
    
    def get(self, key: str) -> Any:
        """
        Récupère une valeur depuis Redis.
        
        Args:
            key: Clé à récupérer
            
        Returns:
            Valeur récupérée ou None si non trouvée
        """
        value = self._retry_operation(self.redis.get, key)
        return self._parse_value(value)
    
    def set(self, key: str, value: Any, expiration: Optional[int] = None) -> bool:
        """
        Stocke une valeur dans Redis.
        
        Args:
            key: Clé Redis
            value: Valeur à stocker
            expiration: Temps d'expiration en secondes
            
        Returns:
            True si réussi
        """
        formatted_value = self._format_value(value)
        
        if expiration:
            return self._retry_operation(self.redis.setex, key, expiration, formatted_value)
        else:
            return self._retry_operation(self.redis.set, key, formatted_value)
    
    def delete(self, key: str) -> int:
        """
        Supprime une clé de Redis.
        
        Args:
            key: Clé à supprimer
            
        Returns:
            Nombre de clés supprimées (0 ou 1)
        """
        return self._retry_operation(self.redis.delete, key)
    
    def hset(self, name: str, key: str, value: Any) -> int:
        """
        Stocke une valeur dans un champ d'une table de hachage.
        
        Args:
            name: Nom de la table
            key: Champ
            value: Valeur
            
        Returns:
            1 si nouveau champ, 0 si mise à jour
        """
        formatted_value = self._format_value(value)
        return self._retry_operation(self.redis.hset, name, key, formatted_value)
    
    def hget(self, name: str, key: str) -> Any:
        """
        Récupère un champ d'une table de hachage.
        
        Args:
            name: Nom de la table
            key: Champ
            
        Returns:
            Valeur du champ ou None
        """
        value = self._retry_operation(self.redis.hget, name, key)
        return self._parse_value(value)
    
    def hmset(self, key: str, mapping: Dict[str, Any]) -> bool:
        """
        Stocke plusieurs champs dans une table de hachage.
        
        Args:
            key: Clé Redis (nom de la table)
            mapping: Dictionnaire {champ: valeur}
            
        Returns:
            True si réussi
        """
        # Formater chaque valeur
        formatted_mapping = {k: self._format_value(v) for k, v in mapping.items()}
        return self._retry_operation(self.redis.hset, key, mapping=formatted_mapping)
    
    def hgetall(self, name: str) -> Dict[str, Any]:
        """
        Récupère tous les champs d'une table de hachage.
        
        Args:
            name: Nom de la table
            
        Returns:
            Dictionnaire {champ: valeur}
        """
        result = self._retry_operation(self.redis.hgetall, name)
        
        # Parser chaque valeur
        return {k: self._parse_value(v) for k, v in result.items()} if result else {}
    
    def hdel(self, name: str, *keys) -> int:
        """
        Supprime un ou plusieurs champs d'une table de hachage.
        
        Args:
            name: Nom de la table
            *keys: Clés des champs à supprimer
            
        Returns:
            Nombre de champs supprimés
        """
        return self._retry_operation(self.redis.hdel, name, *keys)
    
    def sadd(self, key: str, *values) -> int:
        """
        Ajoute un ou plusieurs éléments à un set.
        
        Args:
            key: Clé du set
            *values: Valeurs à ajouter
            
        Returns:
            Nombre d'éléments ajoutés
        """
        formatted_values = [self._format_value(v) for v in values]
        return self._retry_operation(self.redis.sadd, key, *formatted_values)
    
    def srem(self, key: str, *values) -> int:
        """
        Supprime un ou plusieurs éléments d'un set.
        
        Args:
            key: Clé du set
            *values: Valeurs à supprimer
            
        Returns:
            Nombre d'éléments supprimés
        """
        formatted_values = [self._format_value(v) for v in values]
        return self._retry_operation(self.redis.srem, key, *formatted_values)
    
    def smembers(self, key: str) -> List[Any]:
        """
        Récupère tous les membres d'un set.
        
        Args:
            key: Clé du set
            
        Returns:
            Liste des membres
        """
        result = self._retry_operation(self.redis.smembers, key)
        return [self._parse_value(v) for v in result] if result else []
    
    def incrbyfloat(self, key: str, value: float) -> float:
        """
        Incrémente la valeur d'une clé par un nombre à virgule flottante.
        
        Args:
            key: Clé dont la valeur sera incrémentée
            value: Valeur à incrémenter (float)
            
        Returns:
            Nouvelle valeur de la clé après incrémentation
        """
        return self._retry_operation(self.redis.incrbyfloat, key, float(value))
    
    def pipeline(self):
        """
        Crée un pipeline Redis pour des transactions ou des opérations groupées.
        
        Returns:
            RedisPipeline: Wrapper autour du pipeline Redis
        """
        return RedisPipeline(self)
    
    def close(self) -> None:
        """Ferme toutes les connexions Redis."""
        # Désabonner tous les clients PubSub
        with self.pubsub_lock:
            for client_id in list(self.pubsub_connections.keys()):
                self.unsubscribe(client_id)
        
        # Fermer le pool de connexions
        if hasattr(self.connection_pool, 'disconnect'):
            self.connection_pool.disconnect()
        
        logger.info("✅ Toutes les connexions Redis fermées")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Récupère les métriques d'utilisation Redis.
        
        Returns:
            Dictionnaire des métriques
        """
        metrics = self.metrics.get_stats()
        
        # Ajouter des informations sur les connexions actives
        with self.pubsub_lock:
            metrics["active_pubsub_connections"] = len(self.pubsub_connections)
            metrics["active_channels"] = sum(len(channels) for channels in self.pubsub_channels.values())
        
        return metrics

class RedisPipeline:
    """Wrapper autour du Pipeline Redis avec retry et formatting."""
    
    def __init__(self, client: RedisClientPool):
        """
        Initialise le pipeline.
        
        Args:
            client: Instance RedisClientPool
        """
        self.client = client
        self.pipeline = client.redis.pipeline()
    
    def execute(self):
        """
        Exécute le pipeline avec retry.
        
        Returns:
            Résultat de l'exécution
        """
        return self.client._retry_operation(self.pipeline.execute)
    
    def __getattr__(self, name):
        """
        Délègue les méthodes au pipeline Redis sous-jacent.
        
        Args:
            name: Nom de la méthode
            
        Returns:
            Méthode du pipeline
        """
        # Récupérer la méthode du pipeline
        method = getattr(self.pipeline, name)
        
        # Si c'est une méthode, wrapper pour formater les valeurs
        if callable(method):
            def wrapped_method(*args, **kwargs):
                # Formater les arguments
                formatted_args = []
                for arg in args:
                    if isinstance(arg, (dict, list, tuple)) and name not in ('hmset', 'hset'):
                        formatted_args.append(self.client._format_value(arg))
                    elif name == 'hset' and len(args) > 2 and arg == args[2]:
                        # Pour hset(name, key, value), formater la valeur
                        formatted_args.append(self.client._format_value(arg))
                    else:
                        formatted_args.append(arg)
                
                # Formater les kwargs
                formatted_kwargs = {}
                for key, value in kwargs.items():
                    if key == 'mapping' and name in ('hmset', 'hset'):
                        # Pour hmset/hset avec mapping, formater chaque valeur
                        formatted_mapping = {k: self.client._format_value(v) for k, v in value.items()}
                        formatted_kwargs[key] = formatted_mapping
                    else:
                        formatted_kwargs[key] = value
                
                # Appeler la méthode avec les arguments formatés
                result = method(*formatted_args, **formatted_kwargs)
                return self if result == self.pipeline else result
            
            return wrapped_method
        
        return method

# Alias pour la compatibilité
RedisClient = RedisClientPool