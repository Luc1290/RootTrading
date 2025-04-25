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
import threading
import psutil
from flask import Flask, jsonify

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

# Créer l'application Flask
app = Flask(__name__)

# Variables globales pour suivre l'état du service
start_time = time.time()
manager = None
process = psutil.Process(os.getpid())

@app.route('/health', methods=['GET'])
def health_check():
    """
    Point de terminaison pour vérifier l'état du service.
    """
    global manager, start_time
    
    return jsonify({
        "status": "ok",
        "timestamp": time.time(),
        "uptime": time.time() - start_time,
        "symbols": SYMBOLS
    })

@app.route('/diagnostic', methods=['GET'])
def diagnostic():
    """
    Point de terminaison pour le diagnostic du service.
    """
    global manager, start_time, process
    
    # Vérifier l'état du gestionnaire d'analyse
    manager_status = {
        "running": manager is not None,
        "workers": manager.max_workers if manager else 0,
        "use_threads": manager.use_threads if manager else False,
        "symbol_groups": len(manager.symbol_groups) if manager else 0,
        "queue_sizes": {
            "data_queue": manager.data_queue.qsize() if manager else 0,
            "signal_queue": manager.signal_queue.qsize() if manager else 0
        }
    }
    
    # Construire la réponse
    diagnostic_info = {
        "status": "operational" if manager else "stopped",
        "timestamp": time.time(),
        "uptime": time.time() - start_time,
        "manager": manager_status,
        "symbols": SYMBOLS,
        "memory_usage_mb": round(process.memory_info().rss / (1024 * 1024), 2),
        "cpu_percent": process.cpu_percent(interval=0.1),
        "thread_count": threading.active_count()
    }
    
    return jsonify(diagnostic_info)

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
    parser.add_argument(
        '--port', 
        type=int, 
        default=5001, 
        help='Port pour l\'API REST'
    )
    parser.add_argument(
        '--no-api', 
        action='store_true', 
        help='Désactive l\'API REST'
    )
    return parser.parse_args()

def main():
    """Fonction principale du service Analyzer."""
    global manager
    
    # Parser les arguments
    args = parse_arguments()
    
    # Configurer les symboles
    symbols = args.symbols.split(',') if args.symbols else SYMBOLS
    
    logger.info("🚀 Démarrage du service Analyzer RootTrading...")
    logger.info(f"Configuration: {len(symbols)} symboles, "
               f"mode {'threads' if args.threads else 'processus'}, "
               f"{args.workers or 'auto'} workers")
    
    # Variables pour le contrôle du service
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
        
        # Démarrer l'API REST si activée
        if not args.no_api:
            # Démarrer l'API dans un thread séparé
            api_thread = threading.Thread(
                target=lambda: app.run(host='0.0.0.0', port=args.port),
                daemon=True
            )
            api_thread.start()
            logger.info(f"✅ API REST démarrée sur le port {args.port}")
        
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