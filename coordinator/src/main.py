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
import json
from typing import Dict, Any
from flask import Flask, jsonify, request
import requests
from urllib.parse import urljoin

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from coordinator.src.signal_handler import SignalHandler
from coordinator.src.cycle_sync_monitor import CycleSyncMonitor

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
    Service principal du Coordinator.
    Coordonne les signaux de trading et la gestion des poches de capital.
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
        
        self.signal_handler = None
        self.cycle_sync_monitor = None
        self.running = False
        self.start_time = time.time()
        
        # Initialiser l'API Flask
        self.app = Flask(__name__)
        self.setup_routes()
        
        logger.info(f"✅ CoordinatorService initialisé")
    
    def setup_routes(self):
        """Configure les routes de l'API Flask."""
        self.app.route('/health', methods=['GET'])(self.health_check)
        self.app.route('/diagnostic', methods=['GET'])(self.diagnostic)
        self.app.route('/status', methods=['GET'])(self.get_status)
        self.app.route('/sync-monitor/status', methods=['GET'])(self.get_sync_monitor_status)
        self.app.route('/sync-monitor/force', methods=['POST'])(self.force_sync_monitor)   
    
    def health_check(self):
        """
        Point de terminaison pour vérifier l'état du service.
        """
        return jsonify({
            "status": "healthy" if self.running else "stopped",
            "timestamp": time.time(),
            "uptime": time.time() - self.start_time,
            "components": {
                "signal_handler": self.signal_handler is not None
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
            "signal_processing": {
                "running": self.signal_handler is not None and self.signal_handler.processing_thread is not None,
                "queue_size": self.signal_handler.signal_queue.qsize() if self.signal_handler else 0
            },
            "cycle_sync_monitor": self.cycle_sync_monitor.get_stats() if self.cycle_sync_monitor else None
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
            "market_filters": {}
        }
        
        # Ajouter les filtres de marché s'ils existent
        if self.signal_handler and hasattr(self.signal_handler, 'market_filters'):
            status_data["market_filters"] = self.signal_handler.market_filters
        
        return jsonify(status_data)
    
    def get_sync_monitor_status(self):
        """
        Récupère l'état du moniteur de synchronisation des cycles.
        """
        if not self.cycle_sync_monitor:
            return jsonify({
                "status": "not_initialized"
            }), 503
        
        return jsonify(self.cycle_sync_monitor.get_stats())
    
    def force_sync_monitor(self):
        """
        Force une synchronisation immédiate via le moniteur de cycles.
        """
        if not self.cycle_sync_monitor:
            return jsonify({
                "status": "error",
                "message": "Sync monitor not initialized"
            }), 503
        
        success = self.cycle_sync_monitor.force_sync()
        return jsonify({
            "status": "success" if success else "failed",
            "stats": self.cycle_sync_monitor.get_stats()
        })
    
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
                logger.info("✅ Service Portfolio en bonne santé")
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
        logger.info("🚀 Démarrage du service Coordinator RootTrading...")
        
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
            
            # Initialiser le gestionnaire de signaux
            self.signal_handler = SignalHandler(
                trader_api_url=self.trader_api_url,
                portfolio_api_url=self.portfolio_api_url
            )
            
            # Démarrer le listener pour les cycles créés (libération automatique des fonds)
            self._start_cycle_listener()
            
            # DÉSACTIVÉ: Le moniteur de synchronisation n'est plus nécessaire
            # car le Trader est maintenant la SEULE source de vérité pour les cycles
            # self.cycle_sync_monitor = CycleSyncMonitor(
            #     trader_api_url=self.trader_api_url,
            #     check_interval=30  # Vérifier toutes les 30 secondes
            # )
            
            # Démarrer les composants
            self.signal_handler.start()
            # self.cycle_sync_monitor.start()  # DÉSACTIVÉ
            
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
        last_reallocation = 0
        last_sync_check = 0
        
        # Boucle principale pour garder le service actif
        while self.running:
            time.sleep(1)
            current_time = int(time.time())
            
            # Vérifier la santé du portfolio toutes les minutes
            if current_time - last_health_check >= 60:
                try:
                    logger.info("Vérification de santé du portfolio...")
                    if not self.check_portfolio_health():
                        logger.warning("Le service Portfolio n'est pas en bonne santé.")
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
        if self.signal_handler:
            self.signal_handler.stop()
            self.signal_handler = None
        
        if self.cycle_sync_monitor:
            self.cycle_sync_monitor.stop()
            self.cycle_sync_monitor = None
        
        logger.info("Service Coordinator terminé")
    
    def _start_cycle_listener(self):
        """
        Démarre le listener pour les cycles créés afin de libérer automatiquement les fonds.
        """
        def cycle_listener_thread():
            from shared.src.redis_client import RedisClient
            redis = RedisClient()
            
            def handle_cycle_created(channel, data):
                """
                Libère automatiquement les fonds USDC après confirmation d'achat.
                """
                try:
                    cycle_id = data.get("cycle_id")                    
                    symbol = data.get("symbol", "")
                    status = data.get("status", "")                                                
                except Exception as e:
                    logger.error(f"Erreur lors du traitement du cycle créé: {str(e)}")

                # S'abonner au canal des cycles créés
                    redis.subscribe("roottrading:cycle:created", handle_cycle_created)                    
        
        # Démarrer le thread
        listener_thread = threading.Thread(target=cycle_listener_thread, daemon=True)
        listener_thread.start()


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