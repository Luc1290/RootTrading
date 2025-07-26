"""
Point d'entr�e principal pour le microservice Signal Aggregator.
Valide et score les signaux avant de les transmettre au coordinator.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any
import psycopg2

# Ajouter les r�pertoires n�cessaires au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))  # Pour shared
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))     # Pour validators
sys.path.append(os.path.dirname(__file__))  # Pour les modules src locaux

from validator_loader import ValidatorLoader
from redis_handler import RedisHandler
from context_manager import ContextManager
from signal_processor import SignalProcessor
from database_manager import DatabaseManager

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SignalAggregatorService:
    """Service principal du signal aggregator."""
    
    def __init__(self):
        # Modules principaux
        self.validator_loader = ValidatorLoader()
        self.redis_handler = RedisHandler()
        self.context_manager = None  # Initialisé après connexion DB
        self.database_manager = None  # Initialisé après connexion DB
        self.signal_processor = None  # Initialis� apr�s les autres modules
        
        # Configuration base de donn�es
        self.db_config = {
            'host': os.getenv('DB_HOST', 'db'),  # Nom du service Docker
            'port': os.getenv('DB_PORT', 5432),
            'database': os.getenv('DB_NAME', 'trading'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres')
        }
        
        # Configuration Redis
        self.redis_url = os.getenv('REDIS_URL', 'redis://redis:6379')
        
        # Configuration de l'analyse
        self.health_check_interval = int(os.getenv('HEALTH_CHECK_INTERVAL', 60))  # secondes
        
        # Statistiques globales
        self.start_time = datetime.utcnow()
        self.last_health_check = None
        
    async def connect_db(self):
        """�tablit la connexion � la base de donn�es."""
        try:
            self.db_connection = psycopg2.connect(**self.db_config)
            logger.info("Connexion � la base de donn�es �tablie")
            
            # Initialiser le gestionnaire de contexte et de base de données
            self.context_manager = ContextManager(self.db_connection)
            self.database_manager = DatabaseManager(self.db_connection)
            
        except Exception as e:
            logger.error(f"Erreur connexion DB: {e}")
            raise
            
    async def connect_redis(self):
        """�tablit la connexion Redis."""
        try:
            await self.redis_handler.connect(self.redis_url)
            logger.info("Connexion Redis �tablie")
        except Exception as e:
            logger.error(f"Erreur connexion Redis: {e}")
            raise
            
    def initialize_components(self):
        """Initialise tous les composants du service."""
        try:
            # Chargement des validators
            self.validator_loader.load_validators()
            validators = self.validator_loader.get_all_validators()
            logger.info(f"Validators charg�s: {len(validators)}")
            for name in validators.keys():
                logger.info(f"  - {name}")
                
            # Initialisation du processeur de signaux
            self.signal_processor = SignalProcessor(
                self.validator_loader, 
                self.context_manager,
                self.database_manager
            )
            
            logger.info("Tous les composants initialis�s avec succ�s")
            
        except Exception as e:
            logger.error(f"Erreur initialisation composants: {e}")
            raise
            
    async def process_signal_callback(self, signal_data: str):
        """
        Callback pour traiter les signaux re�us depuis Redis.
        
        Args:
            signal_data: Donn�es du signal au format JSON
        """
        try:
            if not self.signal_processor:
                logger.error("Signal processor non initialis�")
                return
                
            # Traitement du signal (peut être individuel ou batch)
            result = await self.signal_processor.process_signal(signal_data)
            
            if result:
                # Vérifier si c'est un signal individuel ou une liste (batch)
                if isinstance(result, list):
                    # Batch de signaux validés
                    if result:  # Si la liste n'est pas vide
                        # Publication des signaux validés vers le coordinator
                        await self.redis_handler.publish_multiple_signals(result)
                        
                        logger.info(f"Batch de {len(result)} signaux publié vers coordinator")
                        
                        # Log détaillé de chaque signal
                        for validated_signal in result:
                            timeframe = validated_signal.get('metadata', {}).get('timeframe', 'N/A')
                            final_score = validated_signal.get('metadata', {}).get('final_score', 0.0)
                            logger.debug(f"Signal publié: {validated_signal['strategy']} "
                                       f"{validated_signal['symbol']} {timeframe} "
                                       f"{validated_signal['side']} (final_score={final_score:.2f})")
                else:
                    # Signal individuel validé
                    await self.redis_handler.publish_validated_signal(result)
                    
                    timeframe = result.get('metadata', {}).get('timeframe', 'N/A')
                    final_score = result.get('metadata', {}).get('final_score', 0.0)
                    logger.info(f"Signal publié vers coordinator: {result['strategy']} "
                              f"{result['symbol']} {timeframe} "
                              f"{result['side']} (final_score={final_score:.2f})")
            
        except Exception as e:
            logger.error(f"Erreur traitement callback signal: {e}")
            
    async def health_check(self):
        """Effectue un contr�le de sant� du service."""
        try:
            logger.debug("Ex�cution health check")
            
            # V�rification Redis
            redis_health = await self.redis_handler.health_check()
            if not redis_health:
                logger.warning("Health check Redis �chou�")
                
            # V�rification DB (simple ping)
            try:
                with self.db_connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                db_health = True
            except Exception as e:
                logger.error(f"Health check DB �chou�: {e}")
                db_health = False
                
            # Log des statistiques
            if self.signal_processor:
                stats = self.signal_processor.get_stats()
                logger.info(f"Stats - Trait�s: {stats['signals_processed']}, "
                          f"Valid�s: {stats['signals_validated']}, "
                          f"Rejet�s: {stats['signals_rejected']}, "
                          f"Erreurs: {stats['validation_errors']}")
                          
            self.last_health_check = datetime.utcnow()
            
            return redis_health and db_health
            
        except Exception as e:
            logger.error(f"Erreur health check: {e}")
            return False
            
    def get_service_stats(self) -> Dict[str, Any]:
        """R�cup�re les statistiques compl�tes du service."""
        try:
            uptime = (datetime.utcnow() - self.start_time).total_seconds()
            
            stats = {
                'service': 'signal_aggregator',
                'status': 'running',
                'uptime_seconds': uptime,
                'start_time': self.start_time.isoformat(),
                'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
                'validators_loaded': len(self.validator_loader.get_all_validators()),
                'redis_connected': self.redis_handler.is_connected if self.redis_handler else False,
                'db_connected': hasattr(self, 'db_connection') and self.db_connection is not None
            }
            
            # Ajout des stats Redis
            if self.redis_handler:
                stats['redis_stats'] = self.redis_handler.get_stats()
                
            # Ajout des stats de validation
            if self.signal_processor:
                stats['validation_stats'] = self.signal_processor.get_stats()
                
            # Ajout des stats de cache
            if self.context_manager:
                stats['cache_stats'] = self.context_manager.get_cache_stats()
                
            # Ajout des stats de base de données
            if self.database_manager:
                stats['database_stats'] = self.database_manager.get_storage_stats()
                
            return stats
            
        except Exception as e:
            logger.error(f"Erreur r�cup�ration stats: {e}")
            return {'error': str(e)}
            
    async def start(self):
        """D�marre le service signal aggregator."""
        logger.info("=� D�marrage du service Signal Aggregator RootTrading")
        
        try:
            # Connexions
            await self.connect_db()
            await self.connect_redis()
            
            # Initialisation des composants
            self.initialize_components()
            
            # Configuration du callback Redis
            await self.redis_handler.set_callback(self.process_signal_callback)
            
            # D�marrage de l'�coute Redis
            logger.info("=� D�marrage de l'�coute des signaux Redis...")
            
            # Cr�er une t�che pour l'�coute Redis
            redis_task = asyncio.create_task(
                self.redis_handler.subscribe_to_signals(self.process_signal_callback)
            )
            
            # Cr�er une t�che pour les health checks p�riodiques
            health_task = asyncio.create_task(self._run_health_checks())
            
            logger.info(" Service Signal Aggregator d�marr� et en �coute")
            
            # Attendre que l'une des t�ches se termine (ce qui ne devrait pas arriver en fonctionnement normal)
            done, pending = await asyncio.wait(
                [redis_task, health_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Annuler les t�ches en cours
            for task in pending:
                task.cancel()
                
            # V�rifier si une t�che s'est termin�e avec une erreur
            for task in done:
                try:
                    await task
                except Exception as e:
                    logger.error(f"T�che termin�e avec erreur: {e}")
                    
        except KeyboardInterrupt:
            logger.info("Arr�t du service demand� par l'utilisateur")
        except Exception as e:
            logger.error(f"Erreur dans le service principal: {e}")
            raise
        finally:
            await self.cleanup()
            
    async def _run_health_checks(self):
        """Ex�cute les health checks p�riodiques."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self.health_check()
            except asyncio.CancelledError:
                logger.info("Health checks arr�t�s")
                break
            except Exception as e:
                logger.error(f"Erreur health check p�riodique: {e}")
                
    async def cleanup(self):
        """Nettoie les ressources du service."""
        logger.info(">� Nettoyage des ressources...")
        
        try:
            # Fermeture connexion Redis
            if self.redis_handler:
                await self.redis_handler.disconnect()
                logger.info("Connexion Redis ferm�e")
                
            # Fermeture connexion DB
            if hasattr(self, 'db_connection') and self.db_connection:
                self.db_connection.close()
                logger.info("Connexion DB ferm�e")
                
            # Nettoyage du cache
            if self.context_manager:
                self.context_manager.clear_cache()
                logger.info("Cache contexte vid�")
                
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage: {e}")
            
        logger.info(" Nettoyage termin�")


async def main():
    """Fonction principale du service."""
    service = SignalAggregatorService()
    
    try:
        await service.start()
    except Exception as e:
        logger.error(f"Erreur fatale dans le service: {e}")
        raise


if __name__ == "__main__":
    # Configuration des logs pour le debug si n�cessaire
    if os.getenv('DEBUG', '').lower() in ['true', '1', 'yes']:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Mode DEBUG activ�")
        
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Service arr�t� par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur critique: {e}")
        sys.exit(1)