"""
Point d'entrée principal pour le microservice Analyzer.
Démarre le gestionnaire d'analyse multiprocessus pour traiter les données de marché et générer des signaux.
"""
import argparse
import logging
import signal
import sys
import time
import os

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import SYMBOLS, LOG_LEVEL

from analyzer.src.multiproc_manager import AnalyzerManager

# Configuration du logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('analyzer.log')
    ]
)
logger = logging.getLogger("analyzer")

def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description='Analyzer de trading RootTrading')
    parser.add_argument(
        '--threads', 
        action='store_true', 
        help='Utiliser des threads au lieu de processus'
    )
    parser.add_argument(
        '--workers', 
        type=int, 
        default=None, 
        help='Nombre de workers (processus/threads)'
    )
    parser.add_argument(
        '--symbols', 
        type=str, 
        default=None, 
        help='Liste de symboles séparés par des virgules (ex: BTCUSDC,ETHUSDC)'
    )
    return parser.parse_args()

def main():
    """Fonction principale du service Analyzer."""
    # Parser les arguments
    args = parse_arguments()
    
    # Configurer les symboles
    symbols = args.symbols.split(',') if args.symbols else SYMBOLS
    
    logger.info("🚀 Démarrage du service Analyzer RootTrading...")
    logger.info(f"Configuration: {len(symbols)} symboles, "
               f"mode {'threads' if args.threads else 'processus'}, "
               f"{args.workers or 'auto'} workers")
    
    # Variables pour le contrôle du service
    manager = None
    stop_signal = False
    
    # Gestionnaire de signaux pour l'arrêt propre
    def signal_handler(sig, frame):
        nonlocal stop_signal
        logger.info(f"Signal {sig} reçu, arrêt en cours...")
        stop_signal = True
    
    # Configurer les gestionnaires de signaux
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Créer et démarrer le gestionnaire d'analyse
        manager = AnalyzerManager(
            symbols=symbols,
            max_workers=args.workers,
            use_threads=args.threads
        )
        manager.start()
        
        # Boucle principale
        logger.info("✅ Service Analyzer démarré et en attente de données")
        while not stop_signal:
            time.sleep(1.0)
        
    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur critique dans le service Analyzer: {str(e)}")
    finally:
        # Arrêter le gestionnaire proprement
        if manager:
            logger.info("Arrêt du service Analyzer...")
            manager.stop()
        
        logger.info("Service Analyzer terminé")

if __name__ == "__main__":
    main()