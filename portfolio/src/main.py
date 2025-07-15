"""
Point d'entrée principal pour le microservice Portfolio.
Gère la synchronisation des balances et expose une API REST.
"""
import argparse
import asyncio
import logging
import signal
import sys
import time
import os
import uvicorn
from contextlib import asynccontextmanager

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Maintenant que le chemin est configuré, importer les modules nécessaires
from shared.src.config import LOG_LEVEL
from utils.logging_config import setup_logging
from models import DBManager
from sync import start_sync_tasks
from redis_subscriber import start_redis_subscriptions
from startup import initial_sync_binance
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# Configuration du logging
logger = logging.getLogger("portfolio")

class PortfolioService:
    """
    Service principal pour le microservice Portfolio.
    Gère la synchronisation des balances et expose une API REST.
    """
    
    def __init__(self, host="0.0.0.0", port=8000):
        """
        Initialise le service Portfolio.
        
        Args:
            host: Adresse IP pour l'API REST
            port: Port pour l'API REST
        """
        self.host = host
        self.port = port
        self.app = None
        self.running = False
        self.start_time = time.time()
        
        logger.info(f"✅ PortfolioService initialisé sur {host}:{port}")
    
    async def start(self):
        """
        Démarre le service Portfolio.
        """
        if self.running:
            logger.warning("Le service est déjà en cours d'exécution")
            return
        
        self.running = True
        logger.info("🚀 Démarrage du service Portfolio RootTrading...")
        
        try:
            # Vérifier la connexion à la base de données
            await self._check_database_connection()
            
            # Initialiser la synchronisation Binance
            await initial_sync_binance()
            
            # Démarrer les tâches de synchronisation
            start_sync_tasks()
            
            # Démarrer les abonnements Redis
            start_redis_subscriptions()
            
            # Créer l'application FastAPI
            self.app = self._create_fastapi_app()
            
            logger.info("✅ Service Portfolio démarré")
        
        except Exception as e:
            logger.error(f"❌ Erreur critique lors du démarrage: {str(e)}")
            self.running = False
            raise
    
    async def _check_database_connection(self):
        """
        Vérifie la connexion à la base de données.
        """
        try:
            db = DBManager()
            result = db.execute_query("SELECT 1 as test", fetch_one=True)
            db.close()
            if result and result.get('test') == 1:
                logger.info("✅ Connexion à la base de données vérifiée")
            else:
                raise Exception("Réponse de test invalide")
        except Exception as e:
            logger.error(f"❌ Erreur de connexion à la base de données: {str(e)}")
            raise
    
    def _create_fastapi_app(self):
        """
        Crée et configure l'application FastAPI.
        """
        app = FastAPI(
            title="RootTrading Portfolio API",
            description="API pour la gestion du portefeuille et des poches de capital",
            version="1.0.0"
        )
        
        # Configurer CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Ajouter la compression gzip
        app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Importer et enregistrer les routes
        from api import register_routes
        register_routes(app)
        
        return app
    
    def stop(self):
        """
        Arrête proprement le service Portfolio.
        """
        if not self.running:
            return
        
        logger.info("Arrêt du service Portfolio...")
        self.running = False
        
        logger.info("Service Portfolio terminé")
    
    def run_server(self):
        """
        Lance le serveur uvicorn.
        """
        if not self.app:
            raise RuntimeError("L'application FastAPI n'est pas initialisée")
        
        # Configuration du logging pour réduire la verbosité
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=False  # Désactiver les logs d'accès pour les health checks
        )


def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description='Portfolio RootTrading')
    parser.add_argument(
        '--host', 
        type=str, 
        default="0.0.0.0", 
        help='Adresse IP pour l\'API REST'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=8000, 
        help='Port pour l\'API REST'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default=LOG_LEVEL,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Niveau de journalisation'
    )
    return parser.parse_args()


def setup_signal_handlers(portfolio_service):
    """
    Configure les gestionnaires de signaux pour arrêter proprement le service.
    
    Args:
        portfolio_service: Instance du service Portfolio
    """
    def signal_handler(sig, frame):
        logger.info(f"Signal {sig} reçu, arrêt en cours...")
        portfolio_service.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Fonction principale du service Portfolio."""
    # Parser les arguments
    args = parse_arguments()
    
    # Configurer le logging
    setup_logging(args.log_level)
    
    # Créer le service
    portfolio_service = PortfolioService(host=args.host, port=args.port)
    
    # Configurer les gestionnaires de signaux
    setup_signal_handlers(portfolio_service)
    
    try:
        # Démarrer le service
        await portfolio_service.start()
        
        # Lancer le serveur
        portfolio_service.run_server()
    
    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur critique dans le service Portfolio: {str(e)}")
    finally:
        # Arrêter le service
        portfolio_service.stop()


if __name__ == "__main__":
    asyncio.run(main())