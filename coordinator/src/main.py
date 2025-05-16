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
        self.pocket_checker = None
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
        self.app.route('/force-reallocation', methods=['POST'])(self.force_reallocation)
        self.app.route('/status', methods=['GET'])(self.get_status)
        self.app.route('/force-reconcile', methods=['POST'])(self.force_reconciliation)

    def force_reconciliation(self):
        """
        Force une réconciliation complète des poches avec les cycles actifs.
        """
        if not self.running or not self.pocket_checker:
            return jsonify({
                "status": "error",
                "message": "Service not running"
            }), 503
        
        try:
            success = self.pocket_checker.reconcile_pockets(force=True)
            
            if success:
                return jsonify({
                    "status": "success",
                    "message": "Forced reconciliation completed successfully"
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Forced reconciliation failed"
                }), 500
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la réconciliation forcée: {str(e)}")
            return jsonify({
                "status": "error",
                "message": f"Exception: {str(e)}"
            }), 500
    
    def health_check(self):
        """
        Point de terminaison pour vérifier l'état du service.
        """
        return jsonify({
            "status": "healthy" if self.running else "stopped",
            "timestamp": time.time(),
            "uptime": time.time() - self.start_time,
            "components": {
                "signal_handler": self.signal_handler is not None,
                "pocket_checker": self.pocket_checker is not None
            }
        })
    
    def diagnostic(self):
        """
        Point de terminaison pour le diagnostic du service.
        """
        # Vérifier la santé des services dépendants
        trader_health = self._check_service_health(self.trader_api_url)
        portfolio_health = self._check_service_health(self.portfolio_api_url)
        
        # Obtenir des informations sur les poches
        pockets_info = self._get_pockets_info() if self.pocket_checker else None
        
        # Construire la réponse de diagnostic
        diagnostic_info = {
            "status": "operational" if self.running else "stopped",
            "timestamp": time.time(),
            "uptime": time.time() - self.start_time,
            "services": {
                "trader": trader_health,
                "portfolio": portfolio_health
            },
            "pockets": pockets_info,
            "signal_processing": {
                "running": self.signal_handler is not None and self.signal_handler.processing_thread is not None,
                "queue_size": self.signal_handler.signal_queue.qsize() if self.signal_handler else 0
            }
        }
        
        return jsonify(diagnostic_info)
    
    def force_reallocation(self):
        """
        Force une réallocation des fonds entre les poches.
        """
        if not self.running or not self.pocket_checker:
            return jsonify({
                "status": "error",
                "message": "Service not running"
            }), 503
        
        try:
            success = self.pocket_checker.reallocate_funds()
            
            if success:
                self.pocket_checker.check_pocket_synchronization()
                return jsonify({
                    "status": "success",
                    "message": "Reallocation completed"
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Reallocation failed"
                }), 500
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la réallocation forcée: {str(e)}")
            return jsonify({
                "status": "error",
                "message": f"Exception: {str(e)}"
            }), 500
    
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
    
    def _get_pockets_info(self):
        """
        Récupère les informations sur les poches.
        
        Returns:
            Dict avec les informations sur les poches
        """
        if not self.pocket_checker:
            return None
            
        try:
            # Forcer un rafraîchissement du cache
            self.pocket_checker._refresh_cache()
            
            # Préparer les informations de poche
            pockets_info = {}
            
            for pocket_type in ["active", "buffer", "safety"]:
                available = self.pocket_checker.get_available_funds(pocket_type)
                
                if pocket_type in self.pocket_checker.pocket_cache:
                    pocket_data = self.pocket_checker.pocket_cache[pocket_type]
                    pockets_info[pocket_type] = {
                        "available": available,
                        "total": pocket_data.get("total_value", 0),
                        "used": pocket_data.get("used_value", 0),
                        "active_cycles": pocket_data.get("active_cycles", 0)
                    }
                else:
                    pockets_info[pocket_type] = {
                        "available": available,
                        "status": "not_in_cache"
                    }
            
            return pockets_info
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des infos de poche: {str(e)}")
            return {"error": str(e)}
    
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
                logger.warning("⚠️ Service Portfolio non disponible après 5 tentatives, démarrage malgré tout...")
            
            # Initialiser le vérificateur de poches
            self.pocket_checker = PocketChecker(portfolio_api_url=self.portfolio_api_url)
            
            # Initialiser le gestionnaire de signaux
            self.signal_handler = SignalHandler(
                trader_api_url=self.trader_api_url,
                portfolio_api_url=self.portfolio_api_url
            )
            
            # Démarrer les composants
            self.signal_handler.start()
            
            # Réallouer les fonds initialement
            logger.info("Réallocation initiale des fonds...")
            self.pocket_checker.reallocate_funds()
            
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
                        # Essayer de forcer une réconciliation si le portfolio est en ligne mais non-OK
                        self.pocket_checker.check_pocket_synchronization()
                    last_health_check = current_time
                except Exception as e:
                    logger.error(f"❌ Erreur lors de la vérification de santé: {str(e)}")
            
            # Réduire l'intervalle à 2 minutes (120 secondes) au lieu de 5
            if current_time - last_reallocation >= 120:
                try:
                    logger.info("Réallocation périodique des fonds...")
                    self.pocket_checker.reallocate_funds()
                    last_reallocation = current_time
                except Exception as e:
                    logger.error(f"❌ Erreur lors de la réallocation des fonds: {str(e)}")
            
            # Vérifier la synchronisation des poches toutes les 5 minutes (300s) au lieu de 15
            if current_time - last_sync_check >= 300:
                try:
                    logger.info("Vérification de la synchronisation des poches...")
                    self.pocket_checker.check_pocket_synchronization()
                    last_sync_check = current_time
                except Exception as e:
                    logger.error(f"❌ Erreur lors de la vérification de synchronisation: {str(e)}")
    
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