"""
Point d'entrée principal pour le microservice Coordinator.
Valide et transmet les signaux de trading au service Trader.
"""
import logging
import signal
import sys
import time
import os
import threading
from flask import Flask, jsonify
import requests  # type: ignore
from urllib.parse import urljoin

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from coordinator import Coordinator
from shared.src.redis_client import RedisClient

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


class CoordinatorService:
    """
    Service principal du Coordinator simplifié.
    Valide les signaux et les transmet au trader.
    """
    
    def __init__(self, trader_api_url=None, portfolio_api_url=None, port=5003):
        """
        Initialise le service Coordinator.
        
        Args:
            trader_api_url: URL de l'API du service Trader
            portfolio_api_url: URL de l'API du service Portfolio
            port: Port pour l'API HTTP
        """
        self.trader_api_url = trader_api_url or os.getenv("TRADER_API_URL", "http://trader:5002")
        self.portfolio_api_url = portfolio_api_url or os.getenv("PORTFOLIO_API_URL", "http://portfolio:8000")
        self.port = port
        
        self.coordinator = None
        self.redis_client = None
        self.running = False
        self.start_time = time.time()
        
        # Initialiser l'API Flask
        self.app = Flask(__name__)
        self.setup_routes()
        
        logger.info("✅ CoordinatorService initialisé (version simplifiée)")
    
    def setup_routes(self):
        """Configure les routes de l'API Flask."""
        self.app.route('/health', methods=['GET'])(self.health_check)
        self.app.route('/diagnostic', methods=['GET'])(self.diagnostic)
        self.app.route('/status', methods=['GET'])(self.get_status)
        self.app.route('/stats', methods=['GET'])(self.get_stats)   
    
    def health_check(self):
        """
        Point de terminaison pour vérifier l'état du service.
        """
        return jsonify({
            "status": "healthy" if self.running else "stopped",
            "timestamp": time.time(),
            "uptime": time.time() - self.start_time,
            "components": {
                "coordinator": self.coordinator is not None,
                "redis": self.redis_client is not None
            }
        })
    
    def diagnostic(self):
        """
        Point de terminaison pour le diagnostic du service.
        """
        # Vérifier la santé des services dépendants
        trader_health = self._check_service_health(self.trader_api_url)
        portfolio_health = self._check_service_health(self.portfolio_api_url)        
                
        # Construire la réponse de diagnostic
        diagnostic_info = {
            "status": "operational" if self.running else "stopped",
            "timestamp": time.time(),
            "uptime": time.time() - self.start_time,
            "services": {
                "trader": trader_health,
                "portfolio": portfolio_health
            },
            "stats": self.coordinator.get_stats() if self.coordinator else {}
        }
        
        return jsonify(diagnostic_info)
        
    def get_status(self):
        """
        Renvoie l'état actuel du Coordinator.
        """
        if not self.running:
            return jsonify({
                "status": "stopped"
            })
        
        # Collecter les informations d'état
        status_data = {
            "running": self.running,
            "uptime": time.time() - self.start_time,
            "version": "simplified-1.0"
        }
        
        return jsonify(status_data)
    
    def get_stats(self):
        """
        Récupère les statistiques du coordinator.
        """
        if not self.coordinator:
            return jsonify({
                "status": "not_initialized"
            }), 503
        
        return jsonify(self.coordinator.get_stats())
    
    def _check_service_health(self, service_url):
        """
        Vérifie l'état de santé d'un service dépendant.
        
        Args:
            service_url: URL de base du service
            
        Returns:
            Dict avec les informations d'état
        """
        try:
            health_url = urljoin(service_url, "/health")
            response = requests.get(health_url, timeout=2.0)
            
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "response_time": response.elapsed.total_seconds(),
                    "details": response.json() if response.headers.get('content-type') == 'application/json' else None
                }
            else:
                return {
                    "status": "unhealthy",
                    "status_code": response.status_code
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }       
    
    def check_portfolio_health(self) -> bool:
        """
        Vérifie l'état de santé du service Portfolio.
            
        Returns:
            True si le service est en bonne santé, False sinon
        """
        try:
            health_url = urljoin(self.portfolio_api_url, "/health")
            response = requests.get(health_url, timeout=2.0)
            
            if response.status_code == 200:
                # Log de santé réussi supprimé pour réduire la verbosité
                return True
            else:
                logger.warning(f"⚠️ Service Portfolio a retourné un code d'état non-OK: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la vérification de santé du service Portfolio: {str(e)}")
            return False
    
    def start_api_server(self):
        """
        Démarre le serveur API dans un thread séparé.
        """
        api_thread = threading.Thread(
            target=lambda: self.app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False),
            daemon=True
        )
        api_thread.start()
        logger.info(f"✅ API REST démarrée sur le port {self.port}")
        return api_thread
    
    def start(self):
        """
        Démarre le service Coordinator.
        """
        if self.running:
            logger.warning("Le service est déjà en cours d'exécution")
            return
            
        self.running = True
        logger.info("🚀 Démarrage du service Coordinator RootTrading (version simplifiée)...")
        
        try:
            # Vérifier que le portfolio est en ligne avant de continuer
            portfolio_health_checks = 0
            while portfolio_health_checks < 5:  # Essayer 5 fois maximum
                if self.check_portfolio_health():
                    break
                portfolio_health_checks += 1
                logger.warning(f"Tentative {portfolio_health_checks}/5: Service Portfolio non disponible, nouvelle tentative dans 5 secondes...")
                time.sleep(5)
            
            if portfolio_health_checks >= 5:
                logger.error("❌ Service Portfolio non disponible après 5 tentatives")
                logger.info("💡 Le Coordinator attendra que le Portfolio soit disponible...")
                # Continuer à attendre mais avec un intervalle plus long
                while not self.check_portfolio_health():
                    logger.info("⏳ En attente du service Portfolio... (vérification toutes les 30s)")
                    time.sleep(30)
                logger.info("✅ Service Portfolio maintenant disponible!")
            
            # Initialiser le coordinator
            self.coordinator = Coordinator(
                trader_api_url=self.trader_api_url,
                portfolio_api_url=self.portfolio_api_url
            )
            
            # Initialiser Redis
            self.redis_client = RedisClient()
            
            # S'abonner aux signaux filtrés
            self.redis_client.subscribe("roottrading:signals:filtered", self.coordinator.process_signal)
            logger.info("📡 Abonné aux signaux filtrés")
            
            # Démarrer le serveur API
            self.start_api_server()
            
            logger.info("✅ Service Coordinator démarré")
        
        except Exception as e:
            logger.error(f"❌ Erreur critique lors du démarrage: {str(e)}")
            self.running = False
            raise
    
    def run_maintenance_loop(self):
        """
        Exécute la boucle principale de maintenance du coordinateur.
        """
        # Compteurs pour les vérifications périodiques
        last_health_check = 0
        
        # Boucle principale pour garder le service actif
        while self.running:
            time.sleep(1)
            current_time = int(time.time())
            
            # Vérifier la santé du portfolio toutes les minutes
            if current_time - last_health_check >= 60:
                try:
                    if not self.check_portfolio_health():
                        logger.warning("Le service Portfolio n'est pas en bonne santé.")
                    last_health_check = current_time
                except Exception as e:
                    logger.error(f"❌ Erreur lors de la vérification de santé du portfolio: {str(e)}")
    
    def stop(self):
        """
        Arrête proprement le service Coordinator.
        """
        if not self.running:
            return
            
        logger.info("Arrêt du service Coordinator...")
        self.running = False
        
        # Arrêter les composants
        if self.redis_client:
            self.redis_client.unsubscribe_all()
            self.redis_client = None
        
        if self.coordinator:
            self.coordinator = None
        
        logger.info("Service Coordinator terminé")
    


def main():
    """
    Fonction principale du service Coordinator.
    """
    # Créer le service
    coordinator = CoordinatorService()
    
    # Configurer les gestionnaires de signaux
    def shutdown_handler(signum, frame):
        logger.info(f"Signal {signum} reçu, arrêt en cours...")
        coordinator.stop()
    
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    
    try:
        # Démarrer le service
        coordinator.start()
        
        # Exécuter la boucle de maintenance
        coordinator.run_maintenance_loop()
    
    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur critique dans le service Coordinator: {str(e)}")
    finally:
        # Arrêter le service
        coordinator.stop()


if __name__ == "__main__":
    main()