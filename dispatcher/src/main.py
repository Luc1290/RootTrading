"""
Point d'entrée principal pour le microservice Dispatcher.
Reçoit les messages Kafka et les route vers les bons destinataires.
"""
import logging
import signal
import sys
import time
import os
import threading
from typing import Dict, Any

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import KAFKA_BROKER, KAFKA_GROUP_ID, SYMBOLS, LOG_LEVEL
from shared.src.kafka_client import KafkaClient
from shared.src.redis_client import RedisClient

from dispatcher.src.message_router import MessageRouter

# Configuration du logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dispatcher.log')
    ]
)
logger = logging.getLogger("dispatcher")

# Variables pour le contrôle du service
running = True
kafka_client = None
redis_client = None
router = None

def handle_kafka_message(topic: str, message: Dict[str, Any]) -> None:
    """
    Callback pour traiter les messages Kafka.
    
    Args:
        topic: Topic Kafka source
        message: Message reçu
    """
    if not router:
        logger.error("Router non initialisé, message ignoré")
        return
    
    try:
        # Router le message
        success = router.route_message(topic, message)
        
        if success:
            logger.debug(f"Message routé depuis {topic}")
        else:
            logger.warning(f"Échec du routage pour le message de {topic}")
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement du message Kafka: {str(e)}")

def signal_handler(sig, frame):
    """
    Gestionnaire de signal pour l'arrêt propre.
    
    Args:
        sig: Type de signal reçu
        frame: Frame actuelle
    """
    global running
    logger.info(f"Signal {sig} reçu, arrêt en cours...")
    running = False

def main():
    """
    Fonction principale du service Dispatcher.
    """
    global running, kafka_client, redis_client, router
    
    # Configurer les gestionnaires de signaux
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("🚀 Démarrage du service Dispatcher RootTrading...")
    
    try:
        # Initialiser les clients
        kafka_client = KafkaClient(broker=KAFKA_BROKER, group_id=f"{KAFKA_GROUP_ID}-dispatcher")
        redis_client = RedisClient()
        
        # Initialiser le router de messages
        router = MessageRouter(redis_client=redis_client)
        
        # Construire la liste des topics à suivre
        topics = []
        
        # Topics de données de marché pour chaque symbole
        for symbol in SYMBOLS:
            topics.append(f"market.data.{symbol.lower()}")
        
        # Autres topics à suivre
        topics.extend(["signals", "executions", "orders"])
        
        logger.info(f"Abonnement aux topics Kafka: {', '.join(topics)}")
        
        # Démarrer la consommation Kafka
        kafka_client.consume(topics, handle_kafka_message)
        
        # Boucle principale
        while running:
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur critique dans le service Dispatcher: {str(e)}")
    finally:
        # Nettoyage et arrêt propre
        logger.info("Arrêt du service Dispatcher...")
        
        if kafka_client:
            kafka_client.stop_consuming()
        
        if redis_client:
            redis_client.close()
        
        logger.info("Service Dispatcher terminé")

if __name__ == "__main__":
    main()