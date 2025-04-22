"""
Module de consommation des logs.
Collecte les logs de tous les services et les centralise.
"""
import logging
import json
import time
import threading
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable

# Importer les modules partagés
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.kafka_client import KafkaClient
from shared.src.redis_client import RedisClient
from shared.src.config import KAFKA_BROKER, KAFKA_GROUP_ID, KAFKA_TOPIC_ERRORS

from logger.src.db_exporter import DBExporter

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logger.log')
    ]
)
logger = logging.getLogger(__name__)

class LogConsumer:
    """
    Consommateur de logs centralisé.
    Collecte les logs de tous les services via Kafka et Redis, puis les exporte en base de données.
    """
    
    def __init__(self, kafka_broker: str = KAFKA_BROKER, 
                 db_exporter: Optional[DBExporter] = None):
        """
        Initialise le consommateur de logs.
        
        Args:
            kafka_broker: Adresse du broker Kafka
            db_exporter: Exporteur de logs vers la base de données (optionnel)
        """
        self.kafka_broker = kafka_broker
        self.kafka_client = None
        self.redis_client = None
        
        # Exporteur de logs vers la base de données
        self.db_exporter = db_exporter or DBExporter()
        
        # File d'attente pour le traitement asynchrone des logs
        self.log_queue = []
        self.queue_lock = threading.Lock()
        
        # Threads pour la consommation des messages
        self.kafka_thread = None
        self.redis_thread = None
        self.processing_thread = None
        
        # Flag pour arrêter les threads
        self.running = False
        
        # Nombre maximum de logs à traiter par lot
        self.batch_size = 100
        
        # Filtrage des logs
        self.min_log_level = os.getenv("MIN_LOG_LEVEL", "info").lower()
        self.log_levels_priority = {
            "debug": 0,
            "info": 1,
            "warning": 2,
            "error": 3,
            "critical": 4
        }
        
        logger.info(f"✅ LogConsumer initialisé (niveau min: {self.min_log_level})")
    
    def _should_process_log(self, log_level: str) -> bool:
        """
        Détermine si un log doit être traité en fonction de son niveau.
        
        Args:
            log_level: Niveau de log ('debug', 'info', 'warning', 'error', 'critical')
            
        Returns:
            True si le log doit être traité, False sinon
        """
        level_priority = self.log_levels_priority.get(log_level.lower(), 0)
        min_priority = self.log_levels_priority.get(self.min_log_level, 0)
        return level_priority >= min_priority
    
    def _sanitize_log_message(self, message: str) -> str:
        """
        Nettoie un message de log pour éviter les problèmes d'encodage et les injections SQL.
        
        Args:
            message: Message de log brut
            
        Returns:
            Message nettoyé
        """
        if not message:
            return ""
        
        # Limiter la taille du message
        if len(message) > 10000:
            message = message[:10000] + "... [tronqué]"
        
        # Échapper les caractères spéciaux
        return message.replace("'", "''").replace("\0", "")
    
    def _normalize_log_data(self, log_data: Dict[str, Any], source: str) -> Dict[str, Any]:
        """
        Normalise les données de log provenant de différentes sources.
        
        Args:
            log_data: Données de log brutes
            source: Source du log ('kafka', 'redis')
            
        Returns:
            Données de log normalisées
        """
        normalized = {
            "service": log_data.get("service", "unknown"),
            "level": log_data.get("level", "info").lower(),
            "message": self._sanitize_log_message(log_data.get("message", "")),
            "timestamp": log_data.get("timestamp") or datetime.now().isoformat(),
            "source": source,
            "data": None
        }
        
        # Extraire les données supplémentaires
        data = log_data.get("data") or log_data.get("details")
        if data:
            if isinstance(data, dict):
                normalized["data"] = json.dumps(data)
            elif isinstance(data, str):
                normalized["data"] = data
        
        return normalized
    
    def _process_kafka_log(self, topic: str, log_data: Dict[str, Any]) -> None:
        """
        Traite un log reçu via Kafka.
        
        Args:
            topic: Topic Kafka source
            log_data: Données du log
        """
        try:
            # Normaliser le log
            normalized = self._normalize_log_data(log_data, "kafka")
            
            # Vérifier si le log doit être traité
            if not self._should_process_log(normalized["level"]):
                return
            
            # Ajouter à la file d'attente
            with self.queue_lock:
                self.log_queue.append(normalized)
            
            # Traiter la file d'attente si elle atteint la taille maximale
            if len(self.log_queue) >= self.batch_size:
                self._process_log_queue()
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement du log Kafka: {str(e)}")
    
    def _process_redis_log(self, channel: str, log_data: Dict[str, Any]) -> None:
        """
        Traite un log reçu via Redis.
        
        Args:
            channel: Canal Redis source
            log_data: Données du log
        """
        try:
            # Normaliser le log
            normalized = self._normalize_log_data(log_data, "redis")
            
            # Vérifier si le log doit être traité
            if not self._should_process_log(normalized["level"]):
                return
            
            # Ajouter à la file d'attente
            with self.queue_lock:
                self.log_queue.append(normalized)
            
            # Traiter la file d'attente si elle atteint la taille maximale
            if len(self.log_queue) >= self.batch_size:
                self._process_log_queue()
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement du log Redis: {str(e)}")
    
    def _process_log_queue(self) -> None:
        """
        Traite les logs en attente dans la file d'attente.
        """
        if not self.log_queue:
            return
        
        with self.queue_lock:
            # Extraire les logs de la file d'attente
            logs_to_process = self.log_queue.copy()
            self.log_queue.clear()
        
        if not logs_to_process:
            return
        
        try:
            # Stocker les logs en base de données
            self.db_exporter.store_logs(logs_to_process)
            
            # Loguer des statistiques de traitement
            counts_by_level = {}
            for log in logs_to_process:
                level = log.get("level", "unknown")
                counts_by_level[level] = counts_by_level.get(level, 0) + 1
            
            level_counts = ", ".join([f"{level}: {count}" for level, count in counts_by_level.items()])
            logger.info(f"✅ {len(logs_to_process)} logs traités ({level_counts})")
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement de la file d'attente: {str(e)}")
            
            # Remettre les logs dans la file d'attente pour un nouvel essai
            with self.queue_lock:
                self.log_queue.extend(logs_to_process)
    
    def _kafka_consumer_loop(self) -> None:
        """
        Boucle de consommation de messages Kafka.
        """
        try:
            # Initialiser le client Kafka
            self.kafka_client = KafkaClient(
                broker=self.kafka_broker,
                group_id=f"{KAFKA_GROUP_ID}-logger"
            )
            
            # Construire la liste des topics à suivre
            topics = [KAFKA_TOPIC_ERRORS, "logs.*"]
            
            logger.info(f"Démarrage de la consommation Kafka sur les topics: {', '.join(topics)}")
            
            # Démarrer la consommation Kafka
            self.kafka_client.consume(topics, self._process_kafka_log)
            
            # La boucle continue automatiquement dans le thread du client Kafka
        
        except Exception as e:
            logger.error(f"❌ Erreur dans la boucle de consommation Kafka: {str(e)}")
    
    def _redis_consumer_loop(self) -> None:
        """
        Boucle de consommation de messages Redis.
        """
        try:
            # Initialiser le client Redis
            self.redis_client = RedisClient()
            
            # Construire la liste des canaux à suivre
            channels = ["roottrading:logs.*", "roottrading:errors.*"]
            
            logger.info(f"Démarrage de la consommation Redis sur les canaux: {', '.join(channels)}")
            
            # Démarrer la consommation Redis
            self.redis_client.subscribe(channels, self._process_redis_log)
            
            # La boucle continue automatiquement dans le thread du client Redis
        
        except Exception as e:
            logger.error(f"❌ Erreur dans la boucle de consommation Redis: {str(e)}")
    
    def _processing_loop(self) -> None:
        """
        Boucle de traitement périodique des logs.
        """
        while self.running:
            try:
                # Traiter les logs toutes les secondes
                time.sleep(1)
                
                # Traiter la file d'attente si elle contient des logs
                if self.log_queue:
                    self._process_log_queue()
            
            except Exception as e:
                logger.error(f"❌ Erreur dans la boucle de traitement: {str(e)}")
    
    def start(self) -> None:
        """
        Démarre la consommation de logs.
        """
        if self.running:
            logger.warning("⚠️ LogConsumer déjà démarré")
            return
        
        self.running = True
        
        # Démarrer le thread de consommation Kafka
        self.kafka_thread = threading.Thread(target=self._kafka_consumer_loop)
        self.kafka_thread.daemon = True
        self.kafka_thread.start()
        
        # Démarrer le thread de consommation Redis
        self.redis_thread = threading.Thread(target=self._redis_consumer_loop)
        self.redis_thread.daemon = True
        self.redis_thread.start()
        
        # Démarrer le thread de traitement périodique
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("✅ LogConsumer démarré")
    
    def stop(self) -> None:
        """
        Arrête la consommation de logs.
        """
        if not self.running:
            return
        
        logger.info("🛑 Arrêt du LogConsumer...")
        self.running = False
        
        # Arrêter les clients Kafka et Redis
        if self.kafka_client:
            self.kafka_client.stop_consuming()
        
        if self.redis_client:
            self.redis_client.unsubscribe()
            self.redis_client.close()
        
        # Traiter les derniers logs
        self._process_log_queue()
        
        # Attendre la fin des threads (avec timeout)
        if self.kafka_thread and self.kafka_thread.is_alive():
            self.kafka_thread.join(timeout=5)
        
        if self.redis_thread and self.redis_thread.is_alive():
            self.redis_thread.join(timeout=5)
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
        
        # Fermer l'exporteur de logs
        if self.db_exporter:
            self.db_exporter.close()
        
        logger.info("✅ LogConsumer arrêté")
    
    def add_log(self, service: str, level: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Ajoute un log directement à la file d'attente.
        
        Args:
            service: Nom du service source
            level: Niveau de log ('debug', 'info', 'warning', 'error', 'critical')
            message: Message de log
            data: Données supplémentaires (optionnel)
        """
        log_data = {
            "service": service,
            "level": level.lower(),
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "source": "direct",
            "data": json.dumps(data) if data else None
        }
        
        # Vérifier si le log doit être traité
        if not self._should_process_log(log_data["level"]):
            return
        
        # Ajouter à la file d'attente
        with self.queue_lock:
            self.log_queue.append(log_data)
        
        # Traiter la file d'attente si elle atteint la taille maximale
        if len(self.log_queue) >= self.batch_size:
            self._process_log_queue()

# Point d'entrée pour les tests
if __name__ == "__main__":
    # Initialiser et démarrer le consommateur de logs
    consumer = LogConsumer()
    consumer.start()
    
    try:
        # Rester en vie jusqu'à Ctrl+C
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        # Arrêter proprement
        consumer.stop()
        logger.info("Programme terminé")