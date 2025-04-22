"""
Point d'entrée principal pour le microservice Risk Manager.
Surveille et gère les risques du système de trading.
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
from risk_manager.src.checker import RuleChecker

# Configuration du logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('risk_manager.log')
    ]
)
logger = logging.getLogger("risk_manager")

# Variables pour le contrôle du service
rule_checker = None
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
    Fonction principale du service RiskManager.
    """
    global rule_checker, running
    
    # Configurer les gestionnaires de signaux
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Obtenir le chemin du fichier de règles
    rules_file = os.getenv("RISK_RULES_FILE", "risk_manager/src/rules.yaml")
    portfolio_api_url = os.getenv("PORTFOLIO_API_URL", "http://portfolio:8000")
    trader_api_url = os.getenv("TRADER_API_URL", "http://trader:5002")
    
    logger.info("🚀 Démarrage du service Risk Manager RootTrading...")
    
    try:
        # Initialiser le vérificateur de règles
        rule_checker = RuleChecker(
            rules_file=rules_file,
            portfolio_api_url=portfolio_api_url,
            trader_api_url=trader_api_url
        )
        
        # Démarrer le vérificateur
        check_interval = int(os.getenv("RISK_CHECK_INTERVAL", "60"))  # Par défaut: toutes les 60 secondes
        rule_checker.start()
        
        logger.info(f"✅ Vérification des règles démarrée (intervalle: {check_interval}s)")
        
        # Boucle principale pour garder le service actif
        while running:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur critique dans le service Risk Manager: {str(e)}")
    finally:
        # Arrêter le vérificateur de règles
        if rule_checker:
            rule_checker.stop()
        
        logger.info("Service Risk Manager terminé")

if __name__ == "__main__":
    main()