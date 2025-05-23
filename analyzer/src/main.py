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
from flask import Flask, jsonify, request

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

# Classe principale pour gérer le service Analyzer
class AnalyzerService:
    """
    Service principal Analyzer qui gère l'API REST et le cycle de vie du gestionnaire d'analyse.
    """
    
    def __init__(self, symbols=None, use_threads=False, max_workers=None, port=5001):
        """
        Initialise le service Analyzer.
        
        Args:
            symbols: Liste des symboles à analyser
            use_threads: Utiliser des threads au lieu de processus
            max_workers: Nombre maximum de workers
            port: Port pour l'API REST
        """
        self.symbols = symbols or SYMBOLS
        self.use_threads = use_threads
        self.max_workers = max_workers
        self.port = port
        self.start_time = time.time()
        self.running = False
        self.manager = None
        self.process = psutil.Process(os.getpid())
        self.app = Flask(__name__)
        self.setup_routes()
    
    def setup_routes(self):
        """
        Configure les routes de l'API Flask.
        """
        self.app.route('/health', methods=['GET'])(self.health_check)
        self.app.route('/diagnostic', methods=['GET'])(self.diagnostic)
        self.app.route('/strategies', methods=['GET'])(self.list_strategies)
    
    def health_check(self):
        """
        Point de terminaison pour vérifier l'état du service.
        """
        return jsonify({
            "status": "healthy" if self.running else "stopped",
            "timestamp": time.time(),
            "uptime": time.time() - self.start_time,
            "symbols": self.symbols
        })
    
    def diagnostic(self):
        """
        Point de terminaison pour le diagnostic du service.
        """
        if not self.manager:
            return jsonify({
                "status": "stopped",
                "timestamp": time.time(),
                "uptime": time.time() - self.start_time,
                "error": "Analyzer manager not running"
            }), 503
            
        # Vérifier l'état du gestionnaire d'analyse
        manager_status = {
            "running": self.running,
            "workers": self.manager.max_workers,
            "use_threads": self.manager.use_threads,
            "symbol_groups": len(self.manager.symbol_groups),
            "queue_sizes": {
                "data_queue": self.manager.data_queue.qsize() if hasattr(self.manager.data_queue, 'qsize') else 'unknown',
                "signal_queue": self.manager.signal_queue.qsize() if hasattr(self.manager.signal_queue, 'qsize') else 'unknown'
            }
        }
        
        # Construire la réponse
        diagnostic_info = {
            "status": "operational" if self.running else "stopped",
            "timestamp": time.time(),
            "uptime": time.time() - self.start_time,
            "manager": manager_status,
            "symbols": self.symbols,
            "memory_usage_mb": round(self.process.memory_info().rss / (1024 * 1024), 2),
            "cpu_percent": self.process.cpu_percent(interval=0.1),
            "thread_count": threading.active_count()
        }
        
        return jsonify(diagnostic_info)
    
    def list_strategies(self):
        """
        Liste toutes les stratégies chargées.
        """
        if not self.manager:
            return jsonify({
                "error": "Analyzer manager not running"
            }), 503
            
        try:
            strategy_loader = self.manager.strategy_loader
            strategies = strategy_loader.get_strategy_list()
            return jsonify({
                "strategies": strategies,
                "total_count": strategy_loader.get_strategy_count()
            })
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des stratégies: {str(e)}")
            return jsonify({
                "error": f"Failed to retrieve strategies: {str(e)}"
            }), 500
    
    def start(self):
        """
        Démarre le service Analyzer.
        """
        if self.running:
            logger.warning("Le service est déjà en cours d'exécution")
            return
            
        self.running = True
        
        logger.info("🚀 Démarrage du service Analyzer RootTrading...")
        logger.info(f"Configuration: {len(self.symbols)} symboles, "
                   f"mode {'threads' if self.use_threads else 'processus'}, "
                   f"{self.max_workers or 'auto'} workers")
        
        try:
            # Créer et démarrer le gestionnaire d'analyse
            self.manager = AnalyzerManager(
                symbols=self.symbols,
                max_workers=self.max_workers,
                use_threads=self.use_threads
            )
            self.manager.start()
            
            logger.info("✅ Service Analyzer démarré et en attente de données")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du démarrage du gestionnaire: {str(e)}")
            self.running = False
            raise
    
    def start_api(self, debug=False):
        """
        Démarre l'API REST dans un thread séparé.
        
        Args:
            debug: Activer le mode debug pour Flask
        """
        api_thread = threading.Thread(
            target=lambda: self.app.run(host='0.0.0.0', port=self.port, debug=debug, use_reloader=False, threaded=True),
            daemon=True
        )
        api_thread.start()
        logger.info(f"✅ API REST démarrée sur le port {self.port}")
        return api_thread
    
    def stop(self):
        """
        Arrête proprement le service Analyzer.
        """
        if not self.running:
            return
            
        logger.info("Arrêt du service Analyzer...")
        self.running = False
        
        # Arrêter le gestionnaire proprement
        if self.manager:
            self.manager.stop()
            self.manager = None
            
        logger.info("Service Analyzer terminé")
        

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
        default=5012, 
        help='Port pour l\'API REST'
    )
    parser.add_argument(
        '--no-api', 
        action='store_true', 
        help='Désactive l\'API REST'
    )
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Active le mode debug'
    )
    return parser.parse_args()


def main():
    """Fonction principale du service Analyzer."""
    # Parser les arguments
    args = parse_arguments()
    
    # Configurer les symboles
    symbols = args.symbols.split(',') if args.symbols else SYMBOLS
    
    # Gestionnaire de signaux pour l'arrêt propre
    service = AnalyzerService(
        symbols=symbols,
        use_threads=args.threads,
        max_workers=args.workers,
        port=args.port
    )
    
    # Configurer les gestionnaires de signaux
    def signal_handler(sig, frame):
        logger.info(f"Signal {sig} reçu, arrêt en cours...")
        service.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Démarrer le service
        service.start()
        
        # Démarrer l'API REST si activée
        if not args.no_api:
            service.start_api(debug=args.debug)
        
        # Boucle principale
        while service.running:
            time.sleep(1.0)
        
    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur critique dans le service Analyzer: {str(e)}")
    finally:
        service.stop()


if __name__ == "__main__":
    main()