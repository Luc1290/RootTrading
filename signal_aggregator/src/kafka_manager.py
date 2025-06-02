#!/usr/bin/env python3
"""
Gestionnaire Kafka asynchrone pour le signal aggregator.
"""
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from kafka.errors import KafkaError

logger = logging.getLogger(__name__)


class KafkaManager:
    """Gestionnaire Kafka asynchrone."""
    
    def __init__(self, bootstrap_servers: str = 'kafka:9092'):
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self.consumer = None
        self.admin_client = None
        self._running = False
        
    async def start(self):
        """Démarre les connexions Kafka."""
        try:
            # Créer le producteur
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                compression_type='lz4',
                acks='all',
                enable_idempotence=True,
                max_batch_size=16384,
                linger_ms=10
            )
            await self.producer.start()
            logger.info("✅ Producteur Kafka démarré")
            
            # Créer le client admin
            self.admin_client = AIOKafkaAdminClient(
                bootstrap_servers=self.bootstrap_servers
            )
            await self.admin_client.start()
            logger.info("✅ Client admin Kafka démarré")
            
            self._running = True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du démarrage de Kafka: {e}")
            raise
            
    async def stop(self):
        """Arrête les connexions Kafka."""
        self._running = False
        
        if self.consumer:
            await self.consumer.stop()
            
        if self.producer:
            await self.producer.stop()
            
        if self.admin_client:
            await self.admin_client.close()
            
        logger.info("✅ Connexions Kafka fermées")
        
    async def ensure_topics_exist(self, topics: List[str], 
                                num_partitions: int = 3,
                                replication_factor: int = 1,
                                config: Optional[Dict[str, str]] = None):
        """S'assure que les topics existent."""
        try:
            # Obtenir la liste des topics existants
            existing_topics = await self.admin_client.list_topics()
            
            # Identifier les topics à créer
            topics_to_create = []
            for topic in topics:
                if topic not in existing_topics:
                    new_topic = NewTopic(
                        name=topic,
                        num_partitions=num_partitions,
                        replication_factor=replication_factor,
                        topic_configs=config or {}
                    )
                    topics_to_create.append(new_topic)
                    
            if topics_to_create:
                logger.info(f"Création de {len(topics_to_create)} topics: {[t.name for t in topics_to_create]}")
                await self.admin_client.create_topics(topics_to_create)
                logger.info("✅ Topics créés avec succès")
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création des topics: {e}")
            # Ne pas faire échouer si les topics existent déjà
            if "already exists" not in str(e):
                raise
                
    async def subscribe(self, topics: List[str], group_id: str = 'signal-aggregator'):
        """S'abonne aux topics spécifiés."""
        try:
            self.consumer = AIOKafkaConsumer(
                *topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=group_id,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                max_poll_records=100
            )
            await self.consumer.start()
            logger.info(f"✅ Souscription aux topics: {topics}")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la souscription: {e}")
            raise
            
    async def consume(self, timeout: float = 1.0) -> List[Dict[str, Any]]:
        """Consomme des messages depuis Kafka."""
        messages = []
        
        if not self.consumer:
            return messages
            
        try:
            # Récupérer les messages avec timeout
            records = await self.consumer.getmany(timeout_ms=int(timeout * 1000))
            
            for topic_partition, msgs in records.items():
                for msg in msgs:
                    messages.append({
                        'topic': msg.topic,
                        'value': msg.value,
                        'timestamp': msg.timestamp,
                        'key': msg.key.decode('utf-8') if msg.key else None
                    })
                    
        except Exception as e:
            logger.error(f"❌ Erreur lors de la consommation: {e}")
            
        return messages
        
    async def produce(self, topic: str, value: Dict[str, Any], key: Optional[str] = None):
        """Produit un message sur Kafka."""
        if not self.producer:
            logger.error("Producteur non initialisé")
            return
            
        try:
            # Encoder la clé si fournie
            key_bytes = key.encode('utf-8') if key else None
            
            # Envoyer le message
            await self.producer.send_and_wait(
                topic=topic,
                value=value,
                key=key_bytes
            )
            
            logger.debug(f"Message envoyé sur {topic}")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'envoi du message: {e}")
            raise