"""
Client Kafka partagé pour la communication entre services.
Fournit des fonctions pour produire et consommer des messages Kafka.
"""
import json
import logging
import threading
import time
from typing import Dict, Any, Callable, List, Optional, Union

from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic

from .config import KAFKA_BROKER, KAFKA_GROUP_ID

# Configuration du logging
logger = logging.getLogger(__name__)

class KafkaClient:
    """Client Kafka avec des fonctionnalités pour produire et consommer des messages."""
    
    def __init__(self, broker: str = KAFKA_BROKER, group_id: str = KAFKA_GROUP_ID):
        """
        Initialise le client Kafka avec les paramètres de connexion.
        
        Args:
            broker: Adresse du broker Kafka (host:port)
            group_id: ID du groupe de consommateurs
        """
        self.broker = broker
        self.group_id = group_id
        self.producer = None
        self.consumer = None
        self.admin_client = None
        self.stop_flag = threading.Event()
        self.consumer_thread = None
    
    def _create_producer(self) -> Producer:
        """Crée et retourne un producteur Kafka."""
        conf = {
            'bootstrap.servers': self.broker,
            'client.id': f'roottrading-producer-{time.time()}',
            'queue.buffering.max.messages': 10000,
            'queue.buffering.max.ms': 100,
            'batch.num.messages': 100,
            'message.max.bytes': 1000000,  # ~1MB max message size
            'default.topic.config': {'acks': 'all'}
        }
        
        try:
            producer = Producer(conf)
            logger.info(f"✅ Producteur Kafka créé pour {self.broker}")
            return producer
        except KafkaException as e:
            logger.error(f"❌ Erreur lors de la création du producteur Kafka: {str(e)}")
            raise
    
    def _create_consumer(self, topics: List[str]) -> Consumer:
        """
        Crée et retourne un consommateur Kafka.
        
        Args:
            topics: Liste des topics à suivre
        """
        conf = {
            'bootstrap.servers': self.broker,
            'group.id': self.group_id,
            'auto.offset.reset': 'latest',  # 'earliest' pour traiter tous les messages depuis le début
            'enable.auto.commit': True,
            'max.poll.interval.ms': 300000,  # 5 minutes
            'session.timeout.ms': 30000,  # 30 secondes
        }
        
        try:
            consumer = Consumer(conf)
            consumer.subscribe(topics)
            logger.info(f"✅ Consommateur Kafka créé pour {self.broker}, topics: {', '.join(topics)}")
            return consumer
        except KafkaException as e:
            logger.error(f"❌ Erreur lors de la création du consommateur Kafka: {str(e)}")
            raise

    def _resolve_wildcard_topics(self, topics: List[str]) -> List[str]:
        """
        Résout les topics contenant des caractères joker (*) en les correspondant aux topics existants.
    
        Args:
            topics: Liste des topics, certains pouvant contenir des caractères joker
        
        Returns:
            Liste de topics résolus sans caractères joker
        """
        resolved_topics = []
    
        # Récupérer tous les topics existants
        if not self.admin_client:
            self.admin_client = AdminClient({'bootstrap.servers': self.broker})
    
        try:
            existing_topics = list(self.admin_client.list_topics(timeout=10).topics.keys())
        except Exception as e:
            logger.error(f"Impossible de lister les topics existants: {str(e)}")
            # En cas d'erreur, retourner les topics tels quels (sauf ceux avec *)
            return [t for t in topics if '*' not in t]
    
        # Résoudre chaque topic
        for topic in topics:
            if '*' in topic:
                # C'est un pattern, convertir en expression régulière
                pattern = topic.replace('.', '\.').replace('*', '.*')
                import re
                regex = re.compile(f"^{pattern}$")
            
                # Trouver tous les topics correspondants
                matches = [t for t in existing_topics if regex.match(t)]
            
                if matches:
                    resolved_topics.extend(matches)
                    logger.info(f"Pattern {topic} résolu en {len(matches)} topics: {', '.join(matches[:3])}...")
                else:
                    # Créer des topics par défaut pour le pattern
                    base_topic = topic.split('*')[0]
                    default_topics = [f"{base_topic}info", f"{base_topic}error", f"{base_topic}debug"]
                    logger.info(f"Aucun topic correspondant à {topic}, création des topics par défaut: {default_topics}")
                    self._ensure_topics_exist(default_topics)
                    resolved_topics.extend(default_topics)
            else:
                resolved_topics.append(topic)
    
        return resolved_topics
    
    def _ensure_topics_exist(self, topics: List[str]) -> None:
        """
        S'assure que les topics existent, les crée si nécessaire.
        
        Args:
            topics: Liste des topics à vérifier/créer
        """
        if not self.admin_client:
            self.admin_client = AdminClient({'bootstrap.servers': self.broker})
        
        # Récupérer les topics existants
        existing_topics = self.admin_client.list_topics(timeout=10).topics
        
        topics_to_create = []
        for topic in topics:
            if topic not in existing_topics:
                logger.info(f"Le topic {topic} n'existe pas, création en cours...")
                topics_to_create.append(NewTopic(
                    topic,
                    num_partitions=3,  # Nombre de partitions
                    replication_factor=1  # Facteur de réplication (1 pour dev)
                ))
        
        if topics_to_create:
            try:
                futures = self.admin_client.create_topics(topics_to_create)
                
                # Attendre la création des topics
                for topic, future in futures.items():
                    future.result()  # Bloque jusqu'à ce que le topic soit créé
                    logger.info(f"Topic {topic} créé avec succès")
            except KafkaException as e:
                logger.error(f"Erreur lors de la création des topics: {str(e)}")
    
    def _delivery_report(self, err, msg) -> None:
        """
        Callback appelé pour chaque message produit pour indiquer le succès ou l'échec.
        
        Args:
            err: Erreur de livraison (None si succès)
            msg: L'objet message produit
        """
        if err is not None:
            logger.error(f"❌ Échec de la livraison du message: {str(err)}")
        else:
            topic = msg.topic()
            partition = msg.partition()
            offset = msg.offset()
            logger.info(f"✅ Message livré au topic {topic} [{partition}] @ offset {offset}")
    
    def produce(self, topic: str, message: Union[Dict[str, Any], str], key: Optional[str] = None) -> None:
        """
        Produit un message sur un topic Kafka.
        
        Args:
            topic: Topic sur lequel publier
            message: Message à publier (dictionnaire ou chaîne)
            key: Clé optionnelle pour le message (pour le partitionnement)
        """
        # Créer le producteur si nécessaire
        if not self.producer:
            conf = {
                'bootstrap.servers': self.broker,
                'client.id': f'roottrading-producer-{time.time()}',
                'queue.buffering.max.messages': 100000,  # Augmenté à 100k
                'queue.buffering.max.ms': 50,  # Réduit à 50ms pour un équilibre latence/débit
                'batch.num.messages': 1000,  # Augmenté à 1000
                'linger.ms': 5,  # Attendre 5ms pour collecter plus de messages
                'compression.type': 'snappy',  # Ajouter la compression
                'message.max.bytes': 2000000,  # 2MB
                'default.topic.config': {'acks': 'all'}
            }
            self.producer = Producer(conf)
        
        # S'assurer que le topic existe
        self._ensure_topics_exist([topic])
        
        # Convertir le dictionnaire en JSON si nécessaire
        if isinstance(message, dict):
            message = json.dumps(message)
        
        try:
            # Produire le message avec callback
            self.producer.produce(
                topic=topic,
                key=key.encode('utf-8') if key else None,
                value=message.encode('utf-8') if isinstance(message, str) else message,
                callback=self._delivery_report
            )
            
            # Appeler poll pour traiter les événements de livraison
            self.producer.poll(0)
            
        except BufferError:
            logger.warning("File d'attente du producteur Kafka pleine, vidage en cours...")
            self.producer.flush()
            
            # Réessayer après le flush
            self.producer.produce(
                topic=topic,
                key=key.encode('utf-8') if key else None,
                value=message.encode('utf-8') if isinstance(message, str) else message,
                callback=self._delivery_report
            )
        except Exception as e:
            logger.error(f"Erreur lors de la production du message Kafka: {str(e)}")
            raise
    
    def flush(self) -> None:
        """Force l'envoi de tous les messages en attente."""
        if self.producer:
            self.producer.flush()
            logger.info("Producteur Kafka vidé")
    
    def consume(self, topics: List[str], callback: Callable[[str, Dict[str, Any]], None], 
               batch_size: int = 100, poll_timeout: float = 1.0) -> None:
        # Réinitialiser le drapeau d'arrêt
        self.stop_flag.clear()
    
        # S'assurer que les topics existent
        try:
            resolved_topics = self._resolve_wildcard_topics(topics)
            self._ensure_topics_exist(resolved_topics)
        except Exception as e:
            logger.warning(f"⚠️ Impossible de vérifier/créer les topics: {str(e)}")
            resolved_topics = [t for t in topics if '*' not in t]  # Utiliser seulement les topics sans wildcard
    
        # Créer le consommateur
        self.consumer = self._create_consumer(resolved_topics)
    
        # Lancer la consommation dans un thread séparé
        self.consumer_thread = threading.Thread(
            target=self._consume_loop,
            args=(callback, batch_size, poll_timeout),
            daemon=True
        )
        self.consumer_thread.start()
    
        logger.info(f"✅ Démarrage de la consommation depuis les topics: {', '.join(topics)}")
    
    def _consume_loop(self, callback: Callable[[str, Dict[str, Any]], None], 
                     batch_size: int, poll_timeout: float) -> None:
        """
        Boucle principale de consommation de messages.
        Cette méthode s'exécute dans un thread séparé.
        
        Args:
            callback: Fonction appelée pour chaque message
            batch_size: Nombre maximum de messages à traiter par lot
            poll_timeout: Timeout en secondes pour le poll Kafka
        """
        try:
            while not self.stop_flag.is_set():
                # Récupérer un lot de messages
                messages = self.consumer.consume(num_messages=batch_size, timeout=poll_timeout)
                
                if not messages:
                    continue
                
                for msg in messages:
                    # Vérifier les erreurs
                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            # Fin de partition, rien à faire
                            continue
                        else:
                            logger.error(f"❌ Erreur de consommation Kafka: {msg.error()}")
                            continue
                    
                    # Traiter le message
                    topic = msg.topic()
                    value = msg.value()
                    
                    # Essayer de parser le JSON
                    try:
                        if isinstance(value, bytes):
                            value = value.decode('utf-8')
                        
                        if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                            value = json.loads(value)
                        
                        # Appeler le callback avec le topic et la valeur
                        callback(topic, value)
                        
                    except json.JSONDecodeError:
                        logger.warning(f"Message non-JSON reçu sur {topic}: {value[:100]}...")
                        # Appeler le callback avec la valeur brute
                        callback(topic, value)
                    except Exception as e:
                        logger.error(f"Erreur lors du traitement du message Kafka: {str(e)}")
        
        except KafkaException as e:
            logger.error(f"Erreur Kafka durant la consommation: {str(e)}")
        except Exception as e:
            logger.error(f"Erreur générale dans la boucle de consommation: {str(e)}")
        finally:
            logger.info("Arrêt de la boucle de consommation Kafka")
    
    def stop_consuming(self) -> None:
        """Arrête la consommation de messages Kafka."""
        if self.consumer_thread and self.consumer_thread.is_alive():
            logger.info("Arrêt du consommateur Kafka...")
            self.stop_flag.set()
            
            # Attendre que le thread se termine proprement
            self.consumer_thread.join(timeout=10.0)
            
            if self.consumer:
                self.consumer.close()
                self.consumer = None
            
            logger.info("Consommateur Kafka arrêté")
    
    def close(self) -> None:
        """Ferme les connexions Kafka et nettoie les ressources."""
        self.stop_consuming()
        
        if self.producer:
            self.producer.flush()
            logger.info("Producteur Kafka fermé")
            self.producer = None