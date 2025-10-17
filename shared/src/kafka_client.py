"""
Client Kafka optimisé avec meilleure gestion des erreurs et performances accrues.
"""

import json
import logging
import queue
import random
import re
import threading
import time
from collections.abc import Callable
from queue import Empty, Queue
from typing import Any

from confluent_kafka import (
    Consumer,  # type: ignore[import-untyped]
    KafkaError,
    KafkaException,
    Producer,
)
from confluent_kafka.admin import (
    AdminClient,  # type: ignore[import-untyped]
    NewTopic,
)

from .config import KAFKA_BROKER, KAFKA_GROUP_ID

# Configuration du logging
logger = logging.getLogger(__name__)


class KafkaMetrics:
    """
    Collecte des métriques sur l'utilisation du client Kafka.
    """

    def __init__(self):
        self.produced_messages = 0
        self.successful_deliveries = 0
        self.failed_deliveries = 0
        self.consumed_messages = 0
        self.processing_errors = 0
        self.reconnections = 0
        self.lock = threading.RLock()

    def record_produce(self):
        with self.lock:
            self.produced_messages += 1

    def record_delivery_success(self):
        with self.lock:
            self.successful_deliveries += 1

    def record_delivery_failure(self):
        with self.lock:
            self.failed_deliveries += 1

    def record_consume(self):
        with self.lock:
            self.consumed_messages += 1

    def record_processing_error(self):
        with self.lock:
            self.processing_errors += 1

    def record_reconnection(self):
        with self.lock:
            self.reconnections += 1

    def get_stats(self) -> dict[str, Any]:
        with self.lock:
            return {
                "produced_messages": self.produced_messages,
                "successful_deliveries": self.successful_deliveries,
                "failed_deliveries": self.failed_deliveries,
                "consumed_messages": self.consumed_messages,
                "processing_errors": self.processing_errors,
                "reconnections": self.reconnections,
                "delivery_success_rate": (
                    self.successful_deliveries / max(1, self.produced_messages)
                )
                * 100,
            }

    def reset(self):
        with self.lock:
            self.produced_messages = 0
            self.successful_deliveries = 0
            self.failed_deliveries = 0
            self.consumed_messages = 0
            self.processing_errors = 0
            self.reconnections = 0


class KafkaClientPool:
    """
    Client Kafka optimisé avec gestion améliorée des erreurs et performances accrues.
    Implémente un pattern singleton pour un accès global.
    """

    _instance = None

    @classmethod
    def get_instance(cls, broker: str = KAFKA_BROKER, group_id: str = KAFKA_GROUP_ID):
        """
        Obtient l'instance unique du client Kafka.

        Args:
            broker: Adresse du broker Kafka
            group_id: ID du groupe de consommateurs

        Returns:
            Instance KafkaClientPool
        """
        if cls._instance is None:
            cls._instance = KafkaClientPool(broker, group_id)
        return cls._instance

    def __init__(self, broker: str = KAFKA_BROKER, group_id: str = KAFKA_GROUP_ID):
        """
        Initialise le client Kafka.

        Args:
            broker: Adresse du broker Kafka
            group_id: ID du groupe de consommateurs
        """
        self.broker = broker
        self.group_id = group_id

        # Producteurs/consommateurs lazily initialized
        self._producer = None
        self._admin_client = None

        # Dictionnaires pour suivre les consommateurs
        self.consumers: dict[str, Any] = {}
        self.consumer_threads: dict[str, threading.Thread] = {}
        self.processor_threads: dict[str, threading.Thread] = {}
        self.message_queues: dict[str, queue.Queue] = {}
        self.stop_events: dict[str, threading.Event] = {}
        self.topic_maps: dict[str, dict[str, Any]] = {}

        # Métriques
        self.metrics = KafkaMetrics()

        # Cache des topics existants
        self._existing_topics_cache: set[str] = set()
        self._topics_cache_time = 0
        self._topics_cache_lock = threading.RLock()

        # Thread de statistiques périodiques
        self._start_stats_thread()

        logger.info(f"✅ Client Kafka initialisé pour {broker}")

    def _start_stats_thread(self):
        """Démarre un thread pour enregistrer périodiquement les statistiques."""

        def stats_reporter():
            while True:
                try:
                    time.sleep(300)  # Toutes les 5 minutes
                    stats = self.metrics.get_stats()
                    logger.info(f"📊 Statistiques Kafka: {stats}")

                    # Réinitialiser les métriques
                    self.metrics.reset()
                except Exception:
                    logger.exception("Erreur dans le thread de statistiques Kafka: ")

        thread = threading.Thread(target=stats_reporter, daemon=True)
        thread.start()

    @property
    def producer(self) -> Producer:
        """
        Obtient le producteur Kafka, en l'initialisant si nécessaire.

        Returns:
            Instance Producer Kafka
        """
        if self._producer is None:
            self._producer = self._create_producer()
        return self._producer

    @property
    def admin_client(self) -> AdminClient:
        """
        Obtient le client d'administration Kafka, en l'initialisant si nécessaire.

        Returns:
            Instance AdminClient Kafka
        """
        if self._admin_client is None:
            self._admin_client = AdminClient({"bootstrap.servers": self.broker})
        return self._admin_client

    def _create_producer(self) -> Producer:
        """
        Crée et configure un producteur Kafka.

        Returns:
            Instance Producer Kafka
        """
        conf = {
            "bootstrap.servers": self.broker,
            "client.id": f"roottrading-producer-{time.time()}",
            "queue.buffering.max.messages": 100000,
            "queue.buffering.max.ms": 50,
            "batch.num.messages": 1000,
            "linger.ms": 5,
            "compression.type": "snappy",
            "message.max.bytes": 2000000,  # 2MB
            "acks": "all",
            # Configuration de fiabilité
            "retries": 5,
            "retry.backoff.ms": 200,
            "max.in.flight.requests.per.connection": 5,
            "enable.idempotence": True,
            # CORRECTION: Attendre confirmation des leaders avant de continuer
            # pour éviter les conflits de séquence après restart
            "request.timeout.ms": 30000,  # 30s timeout
            "delivery.timeout.ms": 120000,  # 2min total timeout
        }

        try:
            producer = Producer(conf)
            logger.info(f"✅ Producteur Kafka créé pour {self.broker}")
        except KafkaException:
            logger.exception("❌ Erreur lors de la création du producteur Kafka: ")
            raise
        else:
            return producer

    def _create_consumer(
        self, topics: list[str], group_id: str | None = None
    ) -> Consumer:
        """
        Crée et configure un consommateur Kafka.

        Args:
            topics: Liste des topics à consommer
            group_id: ID du groupe de consommateurs

        Returns:
            Instance Consumer Kafka
        """
        effective_group_id = group_id or f"{self.group_id}-{time.time()}"

        conf = {
            "bootstrap.servers": self.broker,
            "group.id": effective_group_id,
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
            "auto.commit.interval.ms": 5000,
            "max.poll.interval.ms": 300000,  # 5 minutes
            "session.timeout.ms": 30000,  # 30 seconds
            "heartbeat.interval.ms": 10000,  # 10 seconds
            "fetch.min.bytes": 1,
            "fetch.max.bytes": 52428800,  # 50MB
            # 'fetch.max.wait.ms': 500,
            "fetch.message.max.bytes": 1048576,  # 1MB
            "max.partition.fetch.bytes": 1048576,  # 1MB
            "socket.timeout.ms": 60000,  # 60 seconds
        }

        try:
            consumer = Consumer(conf)
            consumer.subscribe(topics)
            logger.info(
                f"✅ Consommateur Kafka créé pour {self.broker}, topics: {', '.join(topics)}"
            )
        except KafkaException:
            logger.exception("❌ Erreur lors de la création du consommateur Kafka: ")
            raise
        else:
            return consumer

    def get_existing_topics(self, force_refresh: bool = False) -> set[str]:
        """
        Récupère la liste des topics existants sur Kafka avec mise en cache.

        Args:
            force_refresh: Force la mise à jour du cache

        Returns:
            Ensemble des topics existants
        """
        current_time = time.time()

        with self._topics_cache_lock:
            # Utiliser le cache s'il est récent (moins de 5 minutes)
            if (
                not force_refresh
                and self._existing_topics_cache
                and (current_time - self._topics_cache_time) < 300
            ):
                return self._existing_topics_cache

            try:
                topics = self.admin_client.list_topics(timeout=10).topics.keys()
                self._existing_topics_cache = set(topics)
                self._topics_cache_time = int(current_time)
            except Exception:
                logger.exception("❌ Erreur lors de la récupération des topics")
                # Retourner le cache même s'il est périmé en cas d'erreur
                return (
                    self._existing_topics_cache
                    if self._existing_topics_cache
                    else set()
                )
            else:
                return self._existing_topics_cache

    def _resolve_wildcards(self, patterns: list[str]) -> list[str]:
        """
        Résout les patterns de topics contenant des wildcards.

        Args:
            patterns: Liste des patterns de topics

        Returns:
            Liste des topics résolus
        """
        resolved_topics = []
        wildcard_patterns = []

        # Séparer les patterns contenant des wildcards
        for pattern in patterns:
            if "*" in pattern:
                wildcard_patterns.append(pattern)
            else:
                resolved_topics.append(pattern)

        if not wildcard_patterns:
            return resolved_topics

        # Récupérer tous les topics existants
        existing_topics = self.get_existing_topics()

        # Résoudre chaque pattern
        for pattern in wildcard_patterns:
            # Convertir le pattern en regex
            regex_pattern = pattern.replace(".", r"\.").replace("*", ".*")
            regex = re.compile(f"^{regex_pattern}$")

            # Trouver les correspondances
            matches = [t for t in existing_topics if regex.match(t)]

            if matches:
                resolved_topics.extend(matches)
                logger.info(f"Pattern '{pattern}' résolu en {len(matches)} topics")
            else:
                logger.warning(f"Aucun topic correspondant au pattern '{pattern}'")

        return resolved_topics

    def ensure_topics_exist(
        self,
        topics: list[str],
        num_partitions: int = 3,
        replication_factor: int = 1,
        config: dict[str, str] | None = None,
    ) -> None:
        """
        S'assure que les topics existent, les crée si nécessaire.

        Args:
            topics: Liste des topics à vérifier/créer
            num_partitions: Nombre de partitions pour les nouveaux topics
            replication_factor: Facteur de réplication pour les nouveaux topics
            config: Configuration supplémentaire pour les topics
        """
        # Exclure les patterns avec wildcards
        topics_to_check = [t for t in topics if "*" not in t]

        if not topics_to_check:
            return

        # Récupérer les topics existants
        existing_topics = self.get_existing_topics()

        # Identifier les topics à créer
        topics_to_create = []
        for topic in topics_to_check:
            if topic not in existing_topics:
                new_topic = NewTopic(
                    topic,
                    num_partitions=num_partitions,
                    replication_factor=replication_factor,
                    config=config or {},
                )
                topics_to_create.append(new_topic)

        if not topics_to_create:
            return

        # Créer les topics
        logger.info(
            f"Création de {len(topics_to_create)} topics Kafka: {[t.topic for t in topics_to_create]}"
        )

        try:
            futures = self.admin_client.create_topics(topics_to_create)

            # Attendre et vérifier les résultats
            for topic, future in futures.items():
                try:
                    future.result(timeout=30)  # 30s timeout
                    logger.info(f"✅ Topic '{topic}' créé avec succès")

                    # Ajouter au cache
                    with self._topics_cache_lock:
                        self._existing_topics_cache.add(topic)
                except Exception:
                    logger.exception("❌ Échec de création du topic '{topic}'")
        except Exception:
            logger.exception("❌ Erreur lors de la création des topics")

    def _delivery_callback(self, err, msg):
        """
        Callback pour les confirmations de livraison des messages.

        Args:
            err: Erreur éventuelle
            msg: Message produit
        """
        if err:
            logger.error(
                f"❌ Échec de livraison du message vers {msg.topic()}: {err!s}"
            )
            self.metrics.record_delivery_failure()
        else:
            self.metrics.record_delivery_success()
            logger.debug(
                f"✓ Message livré à {msg.topic()} [{msg.partition()}] @ {msg.offset()}"
            )

    def produce(
        self,
        topic: str,
        message: dict[str, Any] | str,
        key: str | None = None,
        headers: list[tuple] | None = None,
    ) -> None:
        """
        Produit un message sur un topic Kafka.

        Args:
            topic: Topic sur lequel publier
            message: Message à publier (dictionnaire ou chaîne)
            key: Clé du message (pour le partitionnement)
            headers: En-têtes du message (liste de tuples (nom, valeur))
        """
        # S'assurer que le topic existe
        self.ensure_topics_exist([topic])

        # Convertir le dictionnaire en JSON si nécessaire
        if isinstance(message, dict):
            try:
                message = json.dumps(message)
            except Exception:
                logger.exception("❌ Erreur JSON serialization")
                # Log le contenu problématique pour déboguer
                logger.exception(
                    f"Message data: {type(message)} = {str(message)[:500]}..."
                )
                raise

        # Sérialiser la clé et la valeur
        serialized_key = key.encode("utf-8") if key else None
        serialized_value = (
            message.encode("utf-8") if isinstance(message, str) else message
        )

        # Enregistrer la métrique
        self.metrics.record_produce()

        try:
            # Produire le message
            self.producer.produce(
                topic=topic,
                key=serialized_key,
                value=serialized_value,
                headers=headers,
                callback=self._delivery_callback,
            )

            # Appeler poll pour traiter les événements de livraison
            self.producer.poll(0)

        except BufferError:
            logger.warning("⚠️ File d'attente du producteur pleine, vidage...")
            self.producer.flush(timeout=10.0)

            # Réessayer
            self.producer.produce(
                topic=topic,
                key=serialized_key,
                value=serialized_value,
                headers=headers,
                callback=self._delivery_callback,
            )

        except KafkaException as e:
            # CORRECTION: Si erreur FATAL, recréer le producteur
            if "_FATAL" in str(e) or "Fatal error" in str(e):
                logger.exception(
                    "❌ Erreur FATALE Kafka détectée, recréation du producteur: "
                )

                # Fermer l'ancien producteur
                try:
                    if self._producer:
                        self._producer.flush(timeout=5.0)
                except BaseException:
                    pass

                # Recréer un nouveau producteur
                self._producer = None
                time.sleep(1.0)  # Pause avant recréation

                try:
                    # Réessayer avec le nouveau producteur
                    self.producer.produce(
                        topic=topic,
                        key=serialized_key,
                        value=serialized_value,
                        headers=headers,
                        callback=self._delivery_callback,
                    )
                    logger.info(
                        "✅ Message reproduit avec succès après recréation du producteur"
                    )
                except Exception:
                    logger.exception("❌ Échec après recréation du producteur: ")
                    self.metrics.record_delivery_failure()
                    raise
            else:
                logger.exception("❌ Erreur lors de la production du message: ")
                self.metrics.record_delivery_failure()
                raise

        except Exception:
            logger.exception("❌ Erreur lors de la production du message: ")
            self.metrics.record_delivery_failure()
            raise

    def flush(self, timeout: float = 30.0) -> None:
        """
        Force l'envoi de tous les messages en attente.

        Args:
            timeout: Timeout en secondes
        """
        if self._producer:
            remaining = self.producer.flush(timeout=timeout)
            if remaining > 0:
                logger.warning(
                    f"⚠️ {remaining} messages non envoyés après timeout de {timeout}s"
                )

    def consume(
        self,
        topics: list[str],
        callback: Callable[[str, dict[str, Any]], None],
        group_id: str | None = None,
        batch_size: int = 100,
        poll_timeout: float = 1.0,
    ) -> str:
        """
        Consomme des messages depuis des topics Kafka.

        Args:
            topics: Liste des topics à consommer
            callback: Fonction à appeler pour chaque message
            group_id: ID du groupe de consommateurs
            batch_size: Nombre de messages à traiter par lot
            poll_timeout: Timeout pour le poll en secondes

        Returns:
            ID du consommateur (à utiliser pour arrêter la consommation)
        """
        # Générer un ID unique pour ce consommateur
        consumer_id = f"consumer-{int(time.time())}-{random.randint(1000, 9999)}"

        # Résoudre les wildcards et s'assurer que les topics existent
        resolved_topics = self._resolve_wildcards(topics)
        self.ensure_topics_exist(resolved_topics)

        # Stocker la correspondance entre les patterns et les topics résolus
        self.topic_maps[consumer_id] = {"patterns": topics, "resolved": resolved_topics}

        # Créer une file d'attente pour les messages (augmentée pour
        # multi-crypto)
        message_queue: Queue = Queue(maxsize=batch_size * 50)
        self.message_queues[consumer_id] = message_queue

        # Créer un event pour signaler l'arrêt
        stop_event = threading.Event()
        self.stop_events[consumer_id] = stop_event

        # Créer le consommateur
        consumer = self._create_consumer(resolved_topics, group_id)
        self.consumers[consumer_id] = consumer

        # Démarrer le thread de consommation
        consumer_thread = threading.Thread(
            target=self._consume_messages,
            args=(
                consumer_id,
                consumer,
                message_queue,
                stop_event,
                batch_size,
                poll_timeout,
            ),
            daemon=True,
        )
        self.consumer_threads[consumer_id] = consumer_thread
        consumer_thread.start()

        # Démarrer le thread de traitement
        processor_thread = threading.Thread(
            target=self._process_messages,
            args=(consumer_id, message_queue, callback, stop_event),
            daemon=True,
        )
        self.processor_threads[consumer_id] = processor_thread
        processor_thread.start()

        logger.info(
            f"✅ Démarrage de la consommation depuis {len(resolved_topics)} topics (ID: {consumer_id})"
        )
        return consumer_id

    def _consume_messages(
        self,
        consumer_id: str,
        consumer: Consumer,
        message_queue: Queue,
        stop_event: threading.Event,
        batch_size: int,
        poll_timeout: float,
    ):
        """
        Thread de consommation de messages Kafka.

        Args:
            consumer_id: ID du consommateur
            consumer: Instance Consumer Kafka
            message_queue: Queue pour les messages
            stop_event: Event pour signaler l'arrêt
            batch_size: Nombre de messages à traiter par lot
            poll_timeout: Timeout pour le poll en secondes
        """
        retry_count = 0
        max_retries = 10
        retry_delay = 1.0

        while not stop_event.is_set():
            try:
                # Récupérer un lot de messages
                messages = consumer.consume(
                    num_messages=batch_size, timeout=poll_timeout
                )

                if not messages:
                    # Pas de messages, continuer la boucle
                    # Courte pause pour éviter de surcharger CPU
                    time.sleep(0.01)
                    continue

                # Réinitialiser le compteur de tentatives après un poll réussi
                retry_count = 0
                retry_delay = 1.0

                # Traiter les messages
                for msg in messages:
                    if stop_event.is_set():
                        break

                    # Vérifier les erreurs
                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            # Fin de partition, rien à faire
                            continue
                        logger.error(f"❌ Erreur de consommation: {msg.error()}")
                        continue

                    # Mettre le message dans la queue
                    topic = msg.topic()
                    value = msg.value()
                    key = msg.key()
                    headers = msg.headers()
                    timestamp = msg.timestamp()
                    partition = msg.partition()
                    offset = msg.offset()

                    # Créer un dictionnaire avec toutes les métadonnées utiles
                    message_data = {
                        "topic": topic,
                        "raw_value": value,
                        "key": key.decode("utf-8") if key else None,
                        "headers": dict(headers) if headers else {},
                        "timestamp": timestamp,
                        "partition": partition,
                        "offset": offset,
                    }

                    # Essayer de décoder le message
                    try:
                        if isinstance(value, bytes):
                            decoded_value = value.decode("utf-8")

                            # Essayer de parser le JSON
                            if decoded_value and (decoded_value.startswith(("{", "["))):
                                message_data["value"] = json.loads(decoded_value)
                            else:
                                message_data["value"] = decoded_value
                        else:
                            message_data["value"] = value
                    except Exception as e:
                        logger.warning(f"⚠️ Impossible de décoder le message: {e!s}")
                        message_data["value"] = None

                    # Mettre le message dans la queue avec timeout
                    try:
                        message_queue.put(message_data, timeout=1.0)
                        self.metrics.record_consume()
                    except Exception:
                        logger.exception(
                            "❌ Impossible d'ajouter le message à la queue: "
                        )

            except (KafkaException, RuntimeError):
                retry_count += 1
                logger.exception("❌ Erreur Kafka dans le thread de consommation : ")

                if retry_count > max_retries:
                    logger.critical(
                        f"🔥 Trop d'erreurs dans le thread de consommation {consumer_id}, arrêt"
                    )
                    stop_event.set()
                    break

                # Pause exponentielle avant de réessayer
                wait_time = retry_delay * (1 + random.random())
                logger.info(
                    f"⏳ Pause de {wait_time:.2f}s avant nouvelle tentative ({retry_count}/{max_retries})"
                )
                time.sleep(wait_time)
                retry_delay = min(retry_delay * 2, 30.0)  # Max 30s

                # Essayer de recréer le consommateur
                try:
                    consumer.close()
                    resolved_topics = self.topic_maps[consumer_id]["resolved"]
                    new_consumer = self._create_consumer(resolved_topics)
                    self.consumers[consumer_id] = new_consumer
                    consumer = new_consumer
                    self.metrics.record_reconnection()
                    logger.info(f"✅ Consommateur {consumer_id} reconnecté")
                except Exception:
                    logger.exception("❌ Échec de reconnexion du consommateur : ")

            except Exception:
                logger.exception(
                    "❌ Erreur inattendue dans le thread de consommation : "
                )
                # Pause pour éviter une boucle d'erreurs trop rapide
                time.sleep(1.0)

    def _process_messages(
        self,
        _consumer_id: str,
        message_queue: Queue,
        callback: Callable,
        stop_event: threading.Event,
    ):
        """
        Thread de traitement des messages.

        Args:
            consumer_id: ID du consommateur
            message_queue: Queue contenant les messages
            callback: Fonction à appeler pour chaque message
            stop_event: Event pour signaler l'arrêt
        """
        while not stop_event.is_set():
            try:
                # Récupérer un message de la queue avec timeout
                try:
                    message_data = message_queue.get(timeout=0.5)
                except Empty:
                    continue

                # Extraire le topic et la valeur
                topic = message_data["topic"]
                value = message_data.get("value", message_data.get("raw_value"))

                # Appeler le callback
                try:
                    callback(topic, value)
                except Exception:
                    self.metrics.record_processing_error()
                    logger.exception("❌ Erreur dans le callback pour {topic}")
                finally:
                    # Marquer le message comme traité
                    message_queue.task_done()

            except Exception:
                logger.exception("❌ Erreur dans le thread de traitement : ")
                time.sleep(0.1)  # Pause pour éviter de surcharger le CPU

    def stop_consuming(self, consumer_id: str) -> None:
        """
        Arrête la consommation pour un consommateur spécifique.

        Args:
            consumer_id: ID du consommateur à arrêter
        """
        logger.info(f"Arrêt du consommateur {consumer_id}...")

        # Signaler l'arrêt
        if consumer_id in self.stop_events:
            self.stop_events[consumer_id].set()

        # Attendre la fin des threads
        for thread_dict, thread_name in [
            (self.consumer_threads, "consommation"),
            (self.processor_threads, "traitement"),
        ]:
            if consumer_id in thread_dict:
                thread = thread_dict[consumer_id]
                if thread and thread.is_alive():
                    logger.info(
                        f"Attente de la fin du thread de {thread_name} {consumer_id}..."
                    )
                    thread.join(timeout=5.0)
                    if thread.is_alive():
                        logger.warning(
                            f"⚠️ Le thread de {thread_name} {consumer_id} ne s'est pas terminé proprement"
                        )

        # Fermer le consommateur
        if consumer_id in self.consumers:
            try:
                self.consumers[consumer_id].close()
                logger.info(f"✅ Consommateur {consumer_id} fermé")
            except Exception:
                logger.exception("❌ Erreur lors de la fermeture du consommateur : ")

        # Nettoyer les ressources
        for container in [
            self.consumers,
            self.consumer_threads,
            self.processor_threads,
            self.message_queues,
            self.stop_events,
            self.topic_maps,
        ]:
            if consumer_id in container:
                del container[consumer_id]

    def close(self) -> None:
        """Ferme toutes les ressources Kafka."""
        # Arrêter tous les consommateurs
        for consumer_id in list(self.consumers.keys()):
            self.stop_consuming(consumer_id)

        # Vider le producteur
        if self._producer:
            logger.info("Vidage du producteur Kafka...")
            self.producer.flush(timeout=5.0)
            self._producer = None

        # Réinitialiser les caches
        with self._topics_cache_lock:
            self._existing_topics_cache = set()
            self._topics_cache_time = 0

        logger.info("✅ Client Kafka fermé")

    def get_metrics(self) -> dict[str, Any]:
        """
        Récupère les métriques d'utilisation Kafka.

        Returns:
            Dictionnaire des métriques
        """
        metrics = self.metrics.get_stats()

        # Ajouter des informations sur les consommateurs actifs
        metrics["active_consumers"] = len(self.consumers)
        metrics["subscribed_topics"] = sum(
            len(data["resolved"]) for data in self.topic_maps.values()
        )

        return metrics


# Alias pour compatibilité
KafkaClient = KafkaClientPool
