"""
Point d'entrée principal pour le microservice Logger.
Collecte et centralise les logs de tous les services.
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

from shared.src.config import LOG_LEVEL
from logger.src.consumer import LogConsumer
from logger.src.db_exporter import DBExporter

# Configuration du logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logger_service.log')
    ]
)
logger = logging.getLogger("logger_service")

# Variables pour le contrôle du service
log_consumer = None
running = True

def signal_handler(signum, frame):
    """
    Gestionnaire de signal pour l'arrêt propre.
    
    Args:
        signum: Numéro du signal
        frame: Frame actuelle
    """
    global running
    logger.info(f"Signal {signum} reçu, arrêt en cours...")
    running = False

def main():
    """
    Fonction principale du service Logger.
    """
    global log_consumer, running
    
    # Configurer les gestionnaires de signaux
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Paramètres de configuration
    retention_days = int(os.getenv("LOG_RETENTION_DAYS", "30"))
    export_dir = os.getenv("LOG_EXPORT_DIR", "./exports")
    
    logger.info("🚀 Démarrage du service Logger RootTrading...")
    
    try:
        # Créer l'exporteur de logs vers la base de données
        db_exporter = DBExporter(retention_days=retention_days, export_dir=export_dir)
        
        # Initialiser le consommateur de logs
        log_consumer = LogConsumer(db_exporter=db_exporter)
        
        # Démarrer la consommation des logs
        log_consumer.start()
        
        logger.info("✅ Service Logger démarré")
        
        # Boucle principale pour garder le service actif
        while running:
            time.sleep(1)
            
            # Journaliser périodiquement des statistiques sur les logs traités
            if int(time.time()) % 3600 == 0:  # Toutes les heures
                stats = db_exporter.get_stats()
                logger.info(f"Statistiques de logs: {stats['logs_stored']} stockés, {stats['logs_rotated']} archivés")
    
    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur critique dans le service Logger: {str(e)}")
    finally:
        # Arrêter le consommateur de logs
        if log_consumer:
            log_consumer.stop()
        
        logger.info("Service Logger terminé")

if __name__ == "__main__":
    main()