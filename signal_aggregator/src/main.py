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
log_level = logging.DEBUG if os.getenv('DEBUG_LOGS', 'false').lower() == 'true' else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Logger spécialisé pour les statistiques
stats_logger = logging.getLogger('signal_aggregator.stats')
validation_logger = logging.getLogger('signal_aggregator.validation')


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
        
        # Statistiques détaillées
        self.cycle_stats = {
            'total_signals_received': 0,
            'total_signals_processed': 0,
            'total_signals_validated': 0,
            'total_signals_rejected': 0,
            'total_validation_errors': 0,
            'processing_times': [],
            'last_cycle_time': 0.0,
            'avg_processing_time': 0.0
        }
        
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
        Callback pour traiter les signaux reçus depuis Redis.
        
        Args:
            signal_data: Données du signal au format JSON
        """
        import time
        start_time = time.time()
        
        try:
            self.cycle_stats['total_signals_received'] += 1
            
            if not self.signal_processor:
                logger.error("Signal processor non initialisé")
                return
            
            # Log de réception du signal (DEBUG uniquement)
            if logger.isEnabledFor(logging.DEBUG):
                try:
                    import json
                    message = json.loads(signal_data)
                    if isinstance(message, dict):
                        if message.get('type') == 'signal_batch':
                            signals = message.get('signals', [])
                            logger.debug(f"Batch reçu: {len(signals)} signaux")
                        else:
                            strategy = message.get('strategy', 'N/A')
                            symbol = message.get('symbol', 'N/A')
                            side = message.get('side', 'N/A')
                            confidence = message.get('confidence', 0)
                            logger.debug(f"Signal reçu: {strategy} {symbol} {side} (conf={confidence:.2f})")
                except:
                    logger.debug("Signal reçu (format non parsable)")
                
            # Traitement du signal (peut être individuel ou batch)
            result = await self.signal_processor.process_signal(signal_data)
            
            # Timing du traitement
            processing_time = time.time() - start_time
            self.cycle_stats['processing_times'].append(processing_time)
            self.cycle_stats['last_cycle_time'] = processing_time
            
            # Limiter l'historique des temps de traitement
            if len(self.cycle_stats['processing_times']) > 100:
                self.cycle_stats['processing_times'] = self.cycle_stats['processing_times'][-100:]
                
            self.cycle_stats['avg_processing_time'] = sum(self.cycle_stats['processing_times']) / len(self.cycle_stats['processing_times'])
            
            if result:
                # Vérifier si c'est un signal individuel ou une liste (batch)
                if isinstance(result, list):
                    # Batch de signaux validés
                    if result:  # Si la liste n'est pas vide
                        self.cycle_stats['total_signals_validated'] += len(result)
                        
                        # Publication des signaux validés vers le coordinator
                        await self.redis_handler.publish_multiple_signals(result)
                        
                        validation_logger.info(f"Batch validé: {len(result)} signaux publiés vers coordinator ({processing_time*1000:.1f}ms)")
                        
                        # Log détaillé de chaque signal (DEBUG uniquement)
                        if logger.isEnabledFor(logging.DEBUG):
                            for validated_signal in result:
                                timeframe = validated_signal.get('metadata', {}).get('timeframe', 'N/A')
                                final_score = validated_signal.get('metadata', {}).get('final_score', 0.0)
                                validation_score = validated_signal.get('metadata', {}).get('validation_score', 0.0)
                                validators_passed = validated_signal.get('metadata', {}).get('validators_passed', 0)
                                logger.debug(f"Signal validé: {validated_signal['strategy']} "
                                           f"{validated_signal['symbol']} {timeframe} "
                                           f"{validated_signal['side']} (final={final_score:.2f}, "
                                           f"validation={validation_score:.2f}, validators={validators_passed})")
                else:
                    # Signal individuel validé
                    self.cycle_stats['total_signals_validated'] += 1
                    
                    await self.redis_handler.publish_validated_signal(result)
                    
                    timeframe = result.get('metadata', {}).get('timeframe', 'N/A')
                    final_score = result.get('metadata', {}).get('final_score', 0.0)
                    validation_score = result.get('metadata', {}).get('validation_score', 0.0)
                    validators_passed = result.get('metadata', {}).get('validators_passed', 0)
                    validation_strength = result.get('metadata', {}).get('validation_strength', 'N/A')
                    
                    validation_logger.info(f"Signal VALIDÉ: {result['strategy']} {result['symbol']} {timeframe} "
                                         f"{result['side']} (final={final_score:.2f}, validation={validation_score:.2f}, "
                                         f"strength={validation_strength}, validators={validators_passed}) - {processing_time*1000:.1f}ms")
            else:
                self.cycle_stats['total_signals_rejected'] += 1
                
            self.cycle_stats['total_signals_processed'] += 1
            
        except Exception as e:
            self.cycle_stats['total_validation_errors'] += 1
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
                
            # Log des statistiques détaillées
            if self.signal_processor:
                processor_stats = self.signal_processor.get_stats()
                
                # Calculer le taux de validation
                validation_rate = (processor_stats['signals_validated'] / processor_stats['signals_processed'] * 100) if processor_stats['signals_processed'] > 0 else 0
                
                stats_logger.info(f"Stats Processeur: {processor_stats['signals_validated']}/{processor_stats['signals_processed']} validés "
                                 f"({validation_rate:.1f}%) - Rejetés: {processor_stats['signals_rejected']}, "
                                 f"Erreurs: {processor_stats['validation_errors']}, "
                                 f"Score moyen: {processor_stats['avg_validation_score']:.3f}")
                
                # Stats globales du service
                global_validation_rate = (self.cycle_stats['total_signals_validated'] / self.cycle_stats['total_signals_received'] * 100) if self.cycle_stats['total_signals_received'] > 0 else 0
                
                stats_logger.info(f"Stats Service: {self.cycle_stats['total_signals_received']} reçus, "
                                 f"{self.cycle_stats['total_signals_validated']} validés ({global_validation_rate:.1f}%), "
                                 f"Temps moy: {self.cycle_stats['avg_processing_time']*1000:.1f}ms")
                
                # Log des performances par validator (DEBUG uniquement)
                if logger.isEnabledFor(logging.DEBUG) and processor_stats.get('validator_performance'):
                    validator_perf = processor_stats['validator_performance']
                    top_validators = sorted(validator_perf.items(), key=lambda x: x[1]['avg_score'], reverse=True)[:3]
                    
                    if top_validators:
                        perf_info = []
                        for validator_name, perf in top_validators:
                            success_rate = (perf['successful_validations'] / perf['total_runs'] * 100) if perf['total_runs'] > 0 else 0
                            perf_info.append(f"{validator_name}: {success_rate:.0f}% ({perf['avg_score']:.2f})")
                        
                        logger.debug(f"Top validators: {'; '.join(perf_info)}")
                          
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
                'db_connected': hasattr(self, 'db_connection') and self.db_connection is not None,
                
                # Statistiques détaillées du service
                'cycle_stats': self.cycle_stats.copy(),
                'validation_rate': (self.cycle_stats['total_signals_validated'] / self.cycle_stats['total_signals_received'] * 100) if self.cycle_stats['total_signals_received'] > 0 else 0,
                'rejection_rate': (self.cycle_stats['total_signals_rejected'] / self.cycle_stats['total_signals_received'] * 100) if self.cycle_stats['total_signals_received'] > 0 else 0,
                'error_rate': (self.cycle_stats['total_validation_errors'] / self.cycle_stats['total_signals_received'] * 100) if self.cycle_stats['total_signals_received'] > 0 else 0
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
    # Configuration des logs pour le debug si nécessaire  
    if os.getenv('DEBUG', '').lower() in ['true', '1', 'yes'] or os.getenv('DEBUG_LOGS', 'false').lower() == 'true':
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Mode DEBUG activé")
        
    # Log de la configuration
    logger.info(f"Signal Aggregator démarrage - Log level: {logging.getLevelName(log_level)}")
    if os.getenv('DEBUG_LOGS', 'false').lower() == 'true':
        logger.info("Logs détaillés activés (DEBUG_LOGS=true)")
        
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Service arr�t� par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur critique: {e}")
        sys.exit(1)