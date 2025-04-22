"""
Point d'entrée principal pour le microservice Scheduler.
Gère la surveillance du système et les tâches planifiées.
"""
import logging
import signal
import sys
import time
import os
import threading
import argparse
import json
from datetime import datetime, timedelta
from typing import Dict, Any

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import LOG_LEVEL
from scheduler.src.monitor import ServiceMonitor
from scheduler.src.health_check import HealthChecker

# Configuration du logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scheduler.log')
    ]
)
logger = logging.getLogger("scheduler")

# Variables pour le contrôle du service
service_monitor = None
health_checker = None
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

def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description='Scheduler et Health Checker')
    parser.add_argument(
        '--check-interval', 
        type=int, 
        default=60,
        help='Intervalle de vérification des services (en secondes)'
    )
    parser.add_argument(
        '--monitor-config', 
        type=str, 
        default=None,
        help='Chemin vers le fichier de configuration du moniteur de services'
    )
    parser.add_argument(
        '--health-config', 
        type=str, 
        default=None,
        help='Chemin vers le fichier de configuration du health checker'
    )
    parser.add_argument(
        '--health-interval', 
        type=int, 
        default=300,  # 5 minutes
        help='Intervalle de vérification de santé (en secondes)'
    )
    parser.add_argument(
        '--report-dir', 
        type=str, 
        default='./reports',
        help='Répertoire pour les rapports de santé'
    )
    return parser.parse_args()

def generate_health_report(checker: HealthChecker, report_dir: str) -> None:
    """
    Génère un rapport de santé du système.
    
    Args:
        checker: Instance du health checker
        report_dir: Répertoire des rapports
    """
    try:
        # Créer le répertoire des rapports s'il n'existe pas
        os.makedirs(report_dir, exist_ok=True)
        
        # Exécuter toutes les vérifications
        results = checker.run_all_checks()
        
        # Générer le nom du fichier avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"health_report_{timestamp}.json"
        filepath = os.path.join(report_dir, filename)
        
        # Enregistrer les résultats dans un fichier JSON
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Journaliser un résumé
        global_status = results.get("global", {}).get("status", "unknown")
        logger.info(f"Rapport de santé généré: {filepath} (status: {global_status})")
        
        # Si le statut global est critique, journaliser un avertissement
        if global_status == "critical":
            logger.warning("⚠️ ALERTE: Statut de santé critique détecté")
            
            # Enregistrer les détails des erreurs critiques
            for check_name, result in results.items():
                if check_name != "global" and result.get("status") == "critical":
                    logger.error(f"❌ Vérification critique: {check_name} - {result.get('message')}")
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de la génération du rapport de santé: {str(e)}")

def main():
    """
    Fonction principale du service Scheduler.
    """
    global service_monitor, health_checker, running
    
    # Parser les arguments
    args = parse_arguments()
    
    # Configurer les gestionnaires de signaux
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Créer le répertoire des rapports s'il n'existe pas
    os.makedirs(args.report_dir, exist_ok=True)
    
    logger.info("🚀 Démarrage du service Scheduler RootTrading...")
    
    try:
        # Initialiser le moniteur de services
        service_monitor = ServiceMonitor(config_file=args.monitor_config)
        
        # Initialiser le health checker
        health_checker = HealthChecker(config_file=args.health_config)
        
        # Démarrer le moniteur de services
        service_monitor.start()
        
        # Initialiser les variables de planification
        last_health_check = 0
        health_interval = args.health_interval
        
        logger.info("✅ Service Scheduler démarré")
        
        # Boucle principale pour garder le service actif
        while running:
            time.sleep(1)
            
            # Vérifier périodiquement la santé du système
            current_time = time.time()
            if current_time - last_health_check > health_interval:
                try:
                    # Générer un rapport de santé
                    generate_health_report(health_checker, args.report_dir)
                    
                    # Mettre à jour le timestamp de dernière vérification
                    last_health_check = current_time
                except Exception as e:
                    logger.error(f"❌ Erreur lors de la vérification de santé: {str(e)}")
            
            # Vérifier et planifier d'autres tâches
            # Par exemple, synchroniser les bases de données, nettoyer les logs, etc.
    
    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur critique dans le service Scheduler: {str(e)}")
    finally:
        # Arrêter le moniteur de services
        if service_monitor:
            service_monitor.stop()
        
        # Fermer le health checker
        if health_checker:
            health_checker.close()
        
        logger.info("Service Scheduler terminé")

if __name__ == "__main__":
    main()