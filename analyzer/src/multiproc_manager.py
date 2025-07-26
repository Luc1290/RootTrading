"""
Gestionnaire de processus multiples pour l'analyzer.
Permet d'exécuter plusieurs stratégies en parallèle sur différents cœurs CPU.
"""

import asyncio
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict, List, Any, Callable, Optional
import time
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class MultiProcessManager:
    """Gestionnaire de parallélisation pour l'analyzer."""
    
    def __init__(self, max_workers: Optional[int] = None):
        # Nombre de workers basé sur les CPUs disponibles
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        
        # Executors pour différents types de tâches
        self.process_executor = None
        self.thread_executor = None
        
        # Métriques de performance
        self.metrics = {
            'tasks_executed': 0,
            'total_execution_time': 0,
            'average_execution_time': 0,
            'errors': 0,
            'active_workers': 0
        }
        
        logger.info(f"MultiProcessManager initialisé avec {self.max_workers} workers")
        
    async def start(self):
        """Démarre les executors."""
        try:
            # Executor pour les tâches CPU-intensives (calculs de stratégies)
            self.process_executor = ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=mp.get_context('spawn')  # Plus stable que fork
            )
            
            # Executor pour les tâches I/O (DB, Redis)
            self.thread_executor = ThreadPoolExecutor(
                max_workers=self.max_workers * 2
            )
            
            logger.info("Executors démarrés")
            
        except Exception as e:
            logger.error(f"Erreur démarrage executors: {e}")
            raise
            
    async def stop(self):
        """Arrête les executors proprement."""
        try:
            if self.process_executor:
                self.process_executor.shutdown(wait=True)
                logger.info("Process executor arrêté")
                
            if self.thread_executor:
                self.thread_executor.shutdown(wait=True)
                logger.info("Thread executor arrêté")
                
        except Exception as e:
            logger.error(f"Erreur arrêt executors: {e}")
            
    async def execute_strategies_parallel(self, 
                                        strategy_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Exécute plusieurs tâches de stratégies en parallèle.
        
        Args:
            strategy_tasks: Liste des tâches à exécuter
            
        Returns:
            Liste des résultats
        """
        if not strategy_tasks:
            return []
            
        start_time = time.time()
        results = []
        
        try:
            # Création des tâches asyncio
            tasks = []
            for task_data in strategy_tasks:
                task = self._execute_strategy_task(task_data)
                tasks.append(task)
                
            # Exécution en parallèle avec timeout
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Traitement des résultats et gestion des erreurs
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Erreur tâche {i}: {result}")
                    self.metrics['errors'] += 1
                else:
                    processed_results.append(result)
                    
            # Mise à jour des métriques
            execution_time = time.time() - start_time
            self.metrics['tasks_executed'] += len(strategy_tasks)
            self.metrics['total_execution_time'] += execution_time
            self.metrics['average_execution_time'] = (
                self.metrics['total_execution_time'] / self.metrics['tasks_executed']
            )
            
            logger.info(f"Exécution parallèle terminée: {len(processed_results)}/{len(strategy_tasks)} "
                       f"réussies en {execution_time:.2f}s")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Erreur exécution parallèle: {e}")
            return []
            
    async def _execute_strategy_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exécute une tâche de stratégie individuelle.
        
        Args:
            task_data: Données de la tâche
            
        Returns:
            Résultat de la tâche
        """
        try:
            # Extraction des données de la tâche
            strategy_class = task_data['strategy_class']
            symbol = task_data['symbol']
            data = task_data['data']
            indicators = task_data['indicators']
            
            # Exécution de la stratégie dans un thread séparé
            # (Les stratégies sont légères, pas besoin de processus séparé)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.thread_executor,
                self._run_strategy,
                strategy_class,
                symbol,
                data,
                indicators
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur exécution tâche stratégie: {e}")
            raise
            
    def _run_strategy(self, 
                      strategy_class: type, 
                      symbol: str, 
                      data: Dict[str, Any], 
                      indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exécute une stratégie (fonction synchrone pour l'executor).
        
        Args:
            strategy_class: Classe de la stratégie
            symbol: Symbole à analyser
            data: Données de marché
            indicators: Indicateurs pré-calculés
            
        Returns:
            Résultat de la stratégie
        """
        try:
            # Instanciation de la stratégie
            strategy = strategy_class(symbol=symbol, data=data, indicators=indicators)
            
            # Génération du signal
            signal = strategy.generate_signal()
            
            # Enrichissement du résultat
            result = {
                'strategy_name': strategy_class.__name__,
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'signal': signal,
                'execution_time': time.time()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur dans _run_strategy: {e}")
            raise
            
    async def execute_db_operations_parallel(self, 
                                           db_operations: List[Callable]) -> List[Any]:
        """
        Exécute plusieurs opérations de base de données en parallèle.
        
        Args:
            db_operations: Liste des opérations à exécuter
            
        Returns:
            Liste des résultats
        """
        if not db_operations:
            return []
            
        try:
            # Exécution des opérations DB dans des threads
            loop = asyncio.get_event_loop()
            tasks = []
            
            for operation in db_operations:
                task = loop.run_in_executor(self.thread_executor, operation)
                tasks.append(task)
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filtrage des erreurs
            valid_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Erreur opération DB: {result}")
                else:
                    valid_results.append(result)
                    
            return valid_results
            
        except Exception as e:
            logger.error(f"Erreur exécution DB parallèle: {e}")
            return []
            
    async def batch_process_symbols(self, 
                                   symbols: List[str], 
                                   timeframes: List[str],
                                   processor_func: Callable,
                                   batch_size: int = 10) -> List[Any]:
        """
        Traite les symboles par batch pour éviter la surcharge.
        
        Args:
            symbols: Liste des symboles
            timeframes: Liste des timeframes
            processor_func: Fonction de traitement
            batch_size: Taille des batches
            
        Returns:
            Liste des résultats
        """
        all_results = []
        
        # Création des combinaisons symbole/timeframe
        combinations = [(symbol, tf) for symbol in symbols for tf in timeframes]
        
        # Traitement par batch
        for i in range(0, len(combinations), batch_size):
            batch = combinations[i:i + batch_size]
            
            logger.info(f"Traitement batch {i//batch_size + 1}: "
                       f"{len(batch)} combinaisons")
            
            # Exécution du batch
            batch_tasks = []
            for symbol, timeframe in batch:
                task = processor_func(symbol, timeframe)
                batch_tasks.append(task)
                
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Collecte des résultats valides
            for result in batch_results:
                if not isinstance(result, Exception):
                    all_results.append(result)
                    
            # Pause entre les batches pour éviter la surcharge
            if i + batch_size < len(combinations):
                await asyncio.sleep(0.1)
                
        return all_results
        
    def get_metrics(self) -> Dict[str, Any]:
        """Récupère les métriques de performance."""
        return {
            **self.metrics,
            'max_workers': self.max_workers,
            'active_processes': len(mp.active_children()) if hasattr(mp, 'active_children') else 0,
            'cpu_count': mp.cpu_count(),
            'memory_usage_mb': self._get_memory_usage()
        }
        
    def _get_memory_usage(self) -> float:
        """Récupère l'utilisation mémoire du processus."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0
        except Exception:
            return 0.0
            
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé du gestionnaire de processus."""
        try:
            # Test simple d'exécution
            test_task = asyncio.create_task(asyncio.sleep(0.001))
            await asyncio.wait_for(test_task, timeout=1.0)
            
            health_status = {
                'status': 'healthy',
                'executors_running': (
                    self.process_executor is not None and 
                    self.thread_executor is not None
                ),
                'metrics': self.get_metrics(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check échoué: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }