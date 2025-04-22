"""
Point d'entrée principal pour le microservice Coordinator.
Gère la coordination entre les signaux et les processus de trading.
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

from coordinator.src.signal_handler import SignalHandler
from coordinator.src.pocket_checker import PocketChecker

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('coordinator.log')
    ]
)
logger = logging.getLogger("coordinator")

# Variables pour le contrôle du service
signal_handler = None
pocket_checker = None
running = True

def shutdown_handler(signum, frame):
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
    Fonction principale du service Coordinator.
    """
    global signal_handler, pocket_checker
    
    # Configurer les gestionnaires de signaux
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    logger.info("🚀 Démarrage du service Coordinator RootTrading...")
    
    # Paramètres des services
    trader_api_url = os.getenv("TRADER_API_URL", "http://trader:5002")
    portfolio_api_url = os.getenv("PORTFOLIO_API_URL", "http://portfolio:8000")
    
    try:
        # Initialiser le vérificateur de poches
        pocket_checker = PocketChecker(portfolio_api_url=portfolio_api_url)
        
        # Initialiser le gestionnaire de signaux
        signal_handler = SignalHandler(
            trader_api_url=trader_api_url,
            portfolio_api_url=portfolio_api_url
        )
        
        # Démarrer les composants
        signal_handler.start()
        
        # Réallouer les fonds initialement
        logger.info("Réallocation initiale des fonds...")
        pocket_checker.reallocate_funds()
        
        logger.info("✅ Service Coordinator démarré")
        
        # Boucle principale pour garder le service actif
        while running:
            time.sleep(1)
            
            # Toutes les 15 minutes, vérifier et réallouer les fonds si nécessaire
            if int(time.time()) % 900 == 0:
                try:
                    pocket_checker.reallocate_funds()
                except Exception as e:
                    logger.error(f"Erreur lors de la réallocation des fonds: {str(e)}")
    
    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur critique dans le service Coordinator: {str(e)}")
    finally:
        # Nettoyage des ressources
        if signal_handler:
            signal_handler.stop()
        
        logger.info("Service Coordinator terminé")

if __name__ == "__main__":
    main()