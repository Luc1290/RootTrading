"""
Point d'entrée principal pour le microservice Dispatcher.
Reçoit les messages Kafka et les route vers les bons destinataires.
"""
import logging
import signal
import sys
import time
import os
import threading
from typing import Dict, Any
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import KAFKA_BROKER, KAFKA_GROUP_ID, SYMBOLS, LOG_LEVEL
from shared.src.kafka_client import KafkaClient
from shared.src.redis_client import RedisClient

from dispatcher.src.message_router import MessageRouter

# Configuration du logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dispatcher.log')
    ]
)
logger = logging.getLogger("dispatcher")


class HealthHandler(BaseHTTPRequestHandler):
    """Gestionnaire des requêtes HTTP pour les endpoints de santé."""
    
    def do_GET(self):
        """Gère les requêtes GET pour les endpoints de santé et diagnostic."""
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Récupérer les statistiques du routeur
            router_stats = self.server.dispatcher_service.router.get_stats() if self.server.dispatcher_service.router else {}
            
            health_info = {
                "status": "healthy" if self.server.dispatcher_service.running else "stopping",
                "timestamp": time.time(),
                "uptime": time.time() - self.server.dispatcher_service.start_time,
                "stats": router_stats
            }
            
            self.wfile.write(json.dumps(health_info).encode())
        
        elif self.path == '/diagnostic':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Informations diagnostiques plus détaillées
            kafka_status = "connected" if self.server.dispatcher_service.kafka_client else "disconnected"
            redis_status = "connected" if self.server.dispatcher_service.redis_client else "disconnected"
            
            diagnostic_info = {
                "status": "operational" if self.server.dispatcher_service.running else "stopping",
                "timestamp": time.time(),
                "uptime": time.time() - self.server.dispatcher_service.start_time,
                "connections": {
                    "kafka": kafka_status,
                    "redis": redis_status
                },
                "topics": self.server.dispatcher_service.topics,
                "stats": self.server.dispatcher_service.router.get_stats() if self.server.dispatcher_service.router else {}
            }
            
            self.wfile.write(json.dumps(diagnostic_info).encode())
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Rediriger les logs HTTP vers le logger du dispatcher."""
        logger.debug(f"HTTP Server: {format % args}")


class DispatcherService:
    """
    Service principal du Dispatcher.
    Coordonne la réception des messages Kafka et leur routage vers Redis.
    """
    
    def __init__(self, broker=KAFKA_BROKER, symbols=SYMBOLS):
        """
        Initialise le service Dispatcher.
        
        Args:
            broker: Adresse du broker Kafka
            symbols: Liste des symboles à surveiller
        """
        self.running = False
        self.kafka_client = None
        self.redis_client = None
        self.router = None
        self.symbols = symbols
        self.broker = broker
        self.start_time = time.time()
        self.topics = []
        self.http_server = None
    
    def handle_kafka_message(self, topic: str, message: Dict[str, Any]) -> None:
        """
        Callback pour traiter les messages Kafka.
        
        Args:
            topic: Topic Kafka source
            message: Message reçu
        """
        if not self.router or not self.running:
            logger.warning("Router non initialisé ou service arrêté, message ignoré")
            return
        
        try:
            # Router le message
            success = self.router.route_message(topic, message)
            
            if success:
                logger.debug(f"Message routé depuis {topic}")
            else:
                logger.warning(f"Échec du routage pour le message de {topic}")
        
        except Exception as e:
            logger.error(f"Erreur lors du traitement du message Kafka: {str(e)}")
    
    def start_http_server(self, port=5004):
        """
        Démarre un serveur HTTP pour les endpoints de santé.
        
        Args:
            port: Port pour le serveur HTTP
        """
        try:
            self.http_server = HTTPServer(('0.0.0.0', port), HealthHandler)
            self.http_server.dispatcher_service = self
            
            # Démarrer dans un thread séparé pour ne pas bloquer
            self.http_thread = threading.Thread(target=self.http_server.serve_forever, daemon=True)
            self.http_thread.start()
            
            logger.info(f"✅ Serveur HTTP démarré sur le port {port}")
        except Exception as e:
            logger.error(f"❌ Erreur lors du démarrage du serveur HTTP: {str(e)}")
    
    def setup_signal_handlers(self):
        """Configure les gestionnaires de signaux pour l'arrêt propre."""
        def signal_handler(sig, frame):
            logger.info(f"Signal {sig} reçu, arrêt en cours...")
            self.stop()
        
        # Enregistrer les gestionnaires pour SIGINT et SIGTERM
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def stop(self):
        """Arrête proprement le service Dispatcher."""
        if not self.running:
            return
        
        logger.info("Arrêt du service Dispatcher...")
        self.running = False
        
        # Arrêt du serveur HTTP s'il est démarré
        if self.http_server:
            self.http_server.shutdown()
            logger.info("Serveur HTTP arrêté")
        
        # Arrêt du client Kafka
        if self.kafka_client:
            self.kafka_client.close()
            logger.info("Client Kafka fermé")
        
        # Fermeture du client Redis
        if self.redis_client:
            self.redis_client.close()
            logger.info("Client Redis fermé")
        
        # Fermeture du router
        if self.router:
            self.router.close()
            logger.info("Router fermé")
        
        logger.info("Service Dispatcher terminé")
    
    def run(self):
        """
        Fonction principale du service Dispatcher.
        """
        if self.running:
            logger.warning("Le service est déjà en cours d'exécution")
            return
        
        self.running = True
        self.setup_signal_handlers()
        
        logger.info("🚀 Démarrage du service Dispatcher RootTrading...")
        
        try:
            # Initialiser les clients
            self.kafka_client = KafkaClient(broker=self.broker, group_id=f"{KAFKA_GROUP_ID}-dispatcher")
            self.redis_client = RedisClient()
            
            # Initialiser le router de messages
            self.router = MessageRouter(redis_client=self.redis_client)
            
            # Construire la liste des topics à suivre
            self.topics = []
            
            # Topics de données de marché multi-timeframes pour chaque symbole
            timeframes = ['1m', '3m', '5m', '15m', '1h', '1d']
            for symbol in self.symbols:
                # Topics multi-timeframes
                for tf in timeframes:
                    self.topics.append(f"market.data.{symbol.lower()}.{tf}")
                # Garder aussi l'ancien format pour compatibilité
                self.topics.append(f"market.data.{symbol.lower()}")
            
            # Autres topics à suivre
            self.topics.extend(["signals", "executions", "orders", "analyzer.signals"])
            
            logger.info(f"Abonnement aux topics Kafka: {', '.join(self.topics)}")
            
            # Démarrer le serveur HTTP pour les endpoints de santé
            self.start_http_server()
            
            # Démarrer la consommation Kafka
            self.kafka_client.consume(self.topics, self.handle_kafka_message)
            
            # Boucle principale
            while self.running:
                time.sleep(1.0)
        
        except KeyboardInterrupt:
            logger.info("Programme interrompu par l'utilisateur")
        except Exception as e:
            logger.error(f"❌ Erreur critique dans le service Dispatcher: {str(e)}")
            self.running = False
        finally:
            # Nettoyage et arrêt propre
            self.stop()


def main():
    """Point d'entrée principal pour le service Dispatcher."""
    dispatcher = DispatcherService()
    dispatcher.run()


if __name__ == "__main__":
    main()