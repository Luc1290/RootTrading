"""
Client Redis partagé pour la communication entre services.
Fournit des fonctions pour publier et s'abonner aux canaux Redis.
"""
import json
import logging
from typing import Any, Dict, Callable, Optional, List, Union
import threading
import time

import redis
from redis import Redis
from redis.exceptions import ConnectionError, TimeoutError

from .config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB

# Configuration du logging
logger = logging.getLogger(__name__)

class RedisClient:
    """Client Redis avec des fonctionnalités pour publier et s'abonner aux messages."""
    
    def __init__(self, host: str = REDIS_HOST, port: int = REDIS_PORT, 
                 password: str = REDIS_PASSWORD, db: int = REDIS_DB):
        """Initialise le client Redis avec les paramètres de connexion."""
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.redis = self._create_redis_connection()
        
        # Pour stocker les abonnements PubSub actifs
        self.pubsub = None
        self.subscription_thread = None
        self.stop_event = threading.Event()
        
        # Stocker les canaux d'origine pour la reconnexion
        self._original_channels = []
        self._callback = None
    
    def _create_redis_connection(self) -> Redis:
        """Crée et retourne une connexion Redis."""
        try:
            connection_params = {
                'host': self.host,
                'port': self.port,
                'db': self.db,
                'socket_timeout': 5,
                'socket_connect_timeout': 5,
                'retry_on_timeout': True,
                'decode_responses': True  # Décode les réponses en strings
            }
        
            # Ajouter le mot de passe si défini
            if self.password:
                connection_params['password'] = self.password
        
            connection = Redis(**connection_params)
        
            # Test de la connexion
            connection.ping()
            logger.info(f"✅ Connexion Redis établie à {self.host}:{self.port} (DB: {self.db})")
            return connection
            
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"❌ Erreur de connexion à Redis: {str(e)}")
            raise
    
    def reconnect(self) -> None:
        """Tente de se reconnecter à Redis en cas de perte de connexion."""
        retry_count = 0
        max_retries = 5
        retry_delay = 2  # secondes
        
        while retry_count < max_retries:
            try:
                logger.info(f"Tentative de reconnexion à Redis ({retry_count+1}/{max_retries})...")
                self.redis = self._create_redis_connection()
                logger.info("Reconnexion à Redis réussie!")
                return
            except (ConnectionError, TimeoutError) as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Échec de reconnexion après {max_retries} tentatives.")
                    raise
                logger.warning(f"Échec de reconnexion, nouvelle tentative dans {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Backoff exponentiel
    
    def publish(self, channel: str, message: Union[Dict[str, Any], str]) -> int:
        """
        Publie un message sur un canal Redis.
        
        Args:
            channel: Le canal sur lequel publier
            message: Le message à publier (dictionnaire ou chaîne)
            
        Returns:
            Nombre de clients qui ont reçu le message
        """
        try:
            # Si le message est un dictionnaire, le convertir en JSON
            if isinstance(message, dict):
                message = json.dumps(message)
                
            result = self.redis.publish(channel, message)
            return result
            
        except (ConnectionError, TimeoutError):
            logger.warning("Perte de connexion Redis pendant la publication, tentative de reconnexion...")
            self.reconnect()
            # Réessayer après reconnexion
            return self.redis.publish(channel, message)
        except Exception as e:
            logger.error(f"Erreur lors de la publication du message: {str(e)}")
            return 0
    
    def subscribe(self, channels: Union[str, List[str]], callback: Callable[[str, Any], None]) -> None:
        """
        S'abonne à un ou plusieurs canaux Redis et traite les messages via un callback.
        
        Args:
            channels: Canal unique ou liste de canaux à écouter
            callback: Fonction appelée avec (channel, message) pour chaque message reçu
        """
        if isinstance(channels, str):
            channels = [channels]
        
        # Stocker les canaux et le callback pour la reconnexion
        self._original_channels = channels
        self._callback = callback
            
        # Créer un objet PubSub
        self.pubsub = self.redis.pubsub(ignore_subscribe_messages=True)
        
        # S'abonner aux canaux
        self.pubsub.subscribe(*channels)
        
        # Réinitialiser l'événement d'arrêt s'il était actif
        self.stop_event.clear()
        
        # Lancer l'écoute dans un thread séparé
        self.subscription_thread = threading.Thread(
            target=self._listen_for_messages, 
            args=(callback,),
            daemon=True
        )
        self.subscription_thread.start()
        
        logger.info(f"✅ Abonné aux canaux Redis: {', '.join(channels)}")
    
    def _listen_for_messages(self, callback: Callable[[str, Any], None]) -> None:
        """
        Écoute les messages sur les canaux souscrits et appelle le callback.
        Cette méthode s'exécute dans un thread séparé.
        """
        retry_count = 0
        max_reconnect_retries = 10
        reconnect_delay = 1  # secondes
        
        while not self.stop_event.is_set():
            try:
                # Récupérer le prochain message avec timeout
                message = self.pubsub.get_message(timeout=1.0)
                
                if message and message['type'] == 'message':
                    channel = message['channel']
                    data = message['data']
                    
                    # Essayer de parser le message JSON
                    try:
                        if isinstance(data, str) and (data.startswith('{') or data.startswith('[')):
                            data = json.loads(data)
                    except json.JSONDecodeError:
                        # Si ce n'est pas du JSON, garder la chaîne
                        pass
                    
                    # Appeler le callback avec le canal et les données
                    try:
                        callback(channel, data)
                    except Exception as e:
                        logger.error(f"Erreur dans le callback pour le canal {channel}: {str(e)}")
                
                # Réinitialiser le compteur de tentatives après un message réussi
                retry_count = 0
                    
            except (ConnectionError, TimeoutError):
                logger.warning("Perte de connexion Redis pendant l'écoute, tentative de reconnexion...")
                try:
                    # Incrémenter le compteur de tentatives
                    retry_count += 1
                    if retry_count > max_reconnect_retries:
                        logger.error(f"Abandonnement après {max_reconnect_retries} tentatives de reconnexion")
                        self.stop_event.set()
                        break
                        
                    # Attendre un peu avec backoff exponentiel
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(30, reconnect_delay * 1.5)  # Maximum 30 secondes
                    
                    # Reconnecter à Redis
                    self.reconnect()
                    
                    # Recréer l'objet PubSub et se réabonner aux canaux d'origine
                    if self._original_channels:
                        self.pubsub = self.redis.pubsub(ignore_subscribe_messages=True)
                        self.pubsub.subscribe(*self._original_channels)
                        logger.info(f"Réabonnement aux canaux: {', '.join(self._original_channels)}")
                except Exception as e:
                    logger.error(f"Échec de reconnexion lors de l'écoute: {str(e)}")
            
            except Exception as e:
                logger.error(f"Erreur durant l'écoute Redis: {str(e)}")
                time.sleep(1)  # Pause pour éviter de consommer trop de CPU
    
    def unsubscribe(self) -> None:
        """Désabonne de tous les canaux et arrête le thread d'écoute."""
        if self.pubsub:
            # Signaler au thread de s'arrêter
            self.stop_event.set()
            
            try:
                # Désabonner de tous les canaux
                self.pubsub.unsubscribe()
                self.pubsub.close()
            except Exception as e:
                logger.warning(f"Erreur lors du désabonnement: {str(e)}")
            
            # Attendre que le thread se termine proprement
            if self.subscription_thread and self.subscription_thread.is_alive():
                self.subscription_thread.join(timeout=2.0)
            
            logger.info("Désabonnement de tous les canaux Redis")
    
    def get(self, key: str) -> Any:
        """Récupère une valeur depuis Redis."""
        try:
            value = self.redis.get(key)
            if value and isinstance(value, str) and value.startswith('{') and value.endswith('}'):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return value
        except (ConnectionError, TimeoutError):
            logger.warning("Perte de connexion Redis pendant get(), tentative de reconnexion...")
            self.reconnect()
            return self.redis.get(key)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la clé {key}: {str(e)}")
            return None
    
    def set(self, key: str, value: Any, expiration: Optional[int] = None) -> bool:
        """
        Stocke une valeur dans Redis.
        
        Args:
            key: Clé Redis
            value: Valeur à stocker (sera convertie en JSON si c'est un dictionnaire)
            expiration: Temps d'expiration en secondes (optionnel)
            
        Returns:
            True si réussi
        """
        try:
            if isinstance(value, dict):
                value = json.dumps(value)
                
            if expiration:
                return self.redis.setex(key, expiration, value)
            else:
                return self.redis.set(key, value)
                
        except (ConnectionError, TimeoutError):
            logger.warning("Perte de connexion Redis pendant set(), tentative de reconnexion...")
            self.reconnect()
            # Réessayer après reconnexion
            if expiration:
                return self.redis.setex(key, expiration, value)
            else:
                return self.redis.set(key, value)
        except Exception as e:
            logger.error(f"Erreur lors de la définition de la clé {key}: {str(e)}")
            return False
    
    def delete(self, key: str) -> int:
        """Supprime une clé de Redis."""
        try:
            return self.redis.delete(key)
        except (ConnectionError, TimeoutError):
            logger.warning("Perte de connexion Redis pendant delete(), tentative de reconnexion...")
            self.reconnect()
            return self.redis.delete(key)
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de la clé {key}: {str(e)}")
            return 0
    
    def close(self) -> None:
        """Ferme la connexion Redis et nettoie les ressources."""
        try:
            self.unsubscribe()
            if self.redis:
                self.redis.close()
                logger.info("Connexion Redis fermée")
        except Exception as e:
            logger.error(f"Erreur lors de la fermeture de la connexion Redis: {str(e)}")