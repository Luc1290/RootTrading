"""
Point d'entrée principal pour le microservice Signal Aggregator - VERSION PROPRE.
Utilise le nouveau système d'agrégation intelligent.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
import psycopg2
from aiohttp import web

# Ajouter les répertoires nécessaires au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))  # Pour shared
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))     # Pour validators
sys.path.append(os.path.dirname(__file__))  # Pour les modules src locaux

from validator_loader import ValidatorLoader
from context_manager import ContextManager
from signal_aggregator_service import SignalAggregatorService
from database_manager import DatabaseManager

# Configuration du logging
log_level = logging.DEBUG if os.getenv('DEBUG_LOGS', 'false').lower() == 'true' else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SignalAggregatorApp:
    """Application principale du signal aggregator."""
    
    def __init__(self):
        # Configuration base de données
        self.db_config = {
            'host': os.getenv('DB_HOST', 'db'),
            'port': os.getenv('DB_PORT', 5432),
            'database': os.getenv('DB_NAME', 'trading'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres')
        }
        
        # Modules principaux
        self.db_connection = None
        self.validator_loader = None
        self.context_manager = None
        self.database_manager = None
        self.aggregator_service = None
        
        # Web server pour health checks
        self.web_app = None
        self.web_runner = None
        
        # Statistiques
        self.start_time = datetime.utcnow()
        
    async def initialize(self):
        """Initialise tous les composants."""
        logger.info("Initialisation du Signal Aggregator...")
        
        # Connexion base de données
        await self.connect_db()
        
        # Chargement des validators
        self.validator_loader = ValidatorLoader()
        self.validator_loader.load_validators()
        logger.info(f"Validators chargés: {len(self.validator_loader.get_all_validators())}")
        
        # Gestionnaires
        self.context_manager = ContextManager(self.db_connection)
        self.database_manager = DatabaseManager(self.db_connection)
        
        # Service d'agrégation principal
        self.aggregator_service = SignalAggregatorService(
            self.validator_loader, 
            self.context_manager, 
            self.database_manager
        )
        
        # Web server pour health checks
        await self.setup_web_server()
        
        logger.info("Signal Aggregator initialisé avec succès")
        
    async def connect_db(self):
        """Établit la connexion à la base de données."""
        try:
            self.db_connection = psycopg2.connect(**self.db_config)
            logger.info("Connexion à la base de données établie")
        except Exception as e:
            logger.error(f"Erreur connexion DB: {e}")
            raise
            
    async def setup_web_server(self):
        """Configure le serveur web pour les health checks."""
        self.web_app = web.Application()
        
        # Routes
        self.web_app.router.add_get('/health', self.health_check)
        self.web_app.router.add_get('/stats', self.get_stats)
        self.web_app.router.add_get('/metrics', self.get_metrics)
        
        # Démarrage du serveur
        self.web_runner = web.AppRunner(self.web_app)
        await self.web_runner.setup()
        
        site = web.TCPSite(self.web_runner, '0.0.0.0', 8080)
        await site.start()
        
        logger.info("Serveur web health check démarré sur le port 8080")
        
    async def health_check(self, request):
        """Endpoint de health check."""
        try:
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            # Test connexion DB
            with self.db_connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                db_status = "OK"
                
            # Stats du service d'agrégation
            stats = self.aggregator_service.get_stats() if self.aggregator_service else {}
            
            return web.json_response({
                'status': 'healthy',
                'uptime_seconds': uptime,
                'database_status': db_status,
                'aggregator_stats': stats.get('aggregator_stats', {}),
                'validation_stats': stats.get('validation_stats', {}),
                'timestamp': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Erreur health check: {e}")
            return web.json_response({
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }, status=500)
            
    async def get_stats(self, request):
        """Endpoint pour les statistiques détaillées."""
        try:
            if not self.aggregator_service:
                return web.json_response({'error': 'Service non initialisé'}, status=503)
                
            stats = self.aggregator_service.get_stats()
            return web.json_response(stats)
            
        except Exception as e:
            logger.error(f"Erreur récupération stats: {e}")
            return web.json_response({'error': str(e)}, status=500)
            
    async def get_metrics(self, request):
        """Endpoint pour les métriques Prometheus-style."""
        try:
            if not self.aggregator_service:
                return web.Response(text="# Service non initialisé\n", content_type='text/plain', status=503)
                
            stats = self.aggregator_service.get_stats()
            aggregator_stats = stats.get('aggregator_stats', {})
            validation_stats = stats.get('validation_stats', {})
            
            metrics = []
            metrics.append("# HELP signal_aggregator_signals_received Total signals received")
            metrics.append("# TYPE signal_aggregator_signals_received counter")
            metrics.append(f"signal_aggregator_signals_received {aggregator_stats.get('signals_received', 0)}")
            
            metrics.append("# HELP signal_aggregator_signals_sent Total signals sent to coordinator")
            metrics.append("# TYPE signal_aggregator_signals_sent counter")
            metrics.append(f"signal_aggregator_signals_sent {aggregator_stats.get('signals_sent', 0)}")
            
            metrics.append("# HELP signal_aggregator_veto_rate Rate of signals vetoed")
            metrics.append("# TYPE signal_aggregator_veto_rate gauge")
            metrics.append(f"signal_aggregator_veto_rate {validation_stats.get('veto_rate', 0)}")
            
            return web.Response(text="\n".join(metrics) + "\n", content_type='text/plain')
            
        except Exception as e:
            logger.error(f"Erreur génération métriques: {e}")
            return web.Response(text=f"# Erreur: {str(e)}\n", content_type='text/plain', status=500)
            
    async def run(self):
        """Lance le service d'agrégation."""
        try:
            logger.info("Démarrage du service d'agrégation...")
            await self.aggregator_service.start()
            
        except KeyboardInterrupt:
            logger.info("Arrêt demandé par l'utilisateur")
        except Exception as e:
            logger.error(f"Erreur service d'agrégation: {e}")
            raise
        finally:
            await self.shutdown()
            
    async def shutdown(self):
        """Arrêt propre de l'application."""
        logger.info("Arrêt du Signal Aggregator...")
        
        # Arrêt du service d'agrégation
        if self.aggregator_service:
            await self.aggregator_service.shutdown()
            
        # Arrêt du serveur web
        if self.web_runner:
            await self.web_runner.cleanup()
            
        # Fermeture DB
        if self.db_connection:
            self.db_connection.close()
            
        logger.info("Signal Aggregator arrêté")


async def main():
    """Point d'entrée principal."""
    app = SignalAggregatorApp()
    
    try:
        await app.initialize()
        await app.run()
    except KeyboardInterrupt:
        logger.info("Arrêt par Ctrl+C")
    except Exception as e:
        logger.error(f"Erreur fatale: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())