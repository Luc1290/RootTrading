"""
Gestionnaire de processus multiples pour l'analyzer.
Permet d'exécuter plusieurs stratégies en parallèle sur différents cœurs CPU.
"""
import logging
import multiprocessing as mp
import os
import sys
import time
import signal
import threading
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import queue
from functools import partial

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import SYMBOLS
from shared.src.schemas import StrategySignal

from analyzer.src.strategy_loader import StrategyLoader, get_strategy_loader
from analyzer.src.redis_subscriber import RedisSubscriber

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('analyzer.log')
    ]
)
logger = logging.getLogger(__name__)

class AnalyzerManager:
    """
    Gestionnaire de processus pour l'analyzer.
    Distribue l'analyse des données de marché sur plusieurs processus/threads.
    """
    
    def __init__(self, symbols: List[str] = None, max_workers: int = None, use_threads: bool = False):
        """
        Initialise le gestionnaire d'analyse.
        
        Args:
            symbols: Liste des symboles à analyser
            max_workers: Nombre maximum de workers (processus/threads)
            use_threads: Utiliser des threads au lieu de processus
        """
        self.symbols = symbols or SYMBOLS
        if max_workers is None:
            max_workers = max(1, int(mp.cpu_count() * 0.75))
        self.max_workers = max_workers
        self.use_threads = use_threads

        # Regrouper les symboles par worker pour réduire la surcharge
        symbols_per_worker = max(1, len(self.symbols) // self.max_workers)
        self.symbol_groups = [
            self.symbols[i:i+symbols_per_worker] 
            for i in range(0, len(self.symbols), symbols_per_worker)
        ]
    
        # Ajuster le nombre de workers en fonction des groupes
        self.max_workers = min(self.max_workers, len(self.symbol_groups))
        
        # File d'attente de données à analyser
        self.data_queue = mp.Queue() if not use_threads else queue.Queue()
        
        # File d'attente de signaux générés
        self.signal_queue = mp.Queue() if not use_threads else queue.Queue()
        
        # Créer le chargeur de stratégies
        self.strategy_loader = get_strategy_loader()
        
        # Événement d'arrêt
        self.stop_event = mp.Event() if not use_threads else threading.Event()
        
        # Subscriber Redis
        self.redis_subscriber = RedisSubscriber(symbols=self.symbols)
        
        # Pool de workers
        self.executor = None
        
        logger.info(f"✅ AnalyzerManager initialisé avec {self.max_workers} workers "
                   f"({'threads' if use_threads else 'processus'}) pour {len(self.symbols)} symboles")
    
    def _worker_analyze(self, data: Dict[str, Any]) -> List[StrategySignal]:
        """
        Fonction de worker qui analyse les données de marché.
        
        Args:
            data: Données de marché à analyser
            
        Returns:
            Liste des signaux générés
        """
        try:
            # Récupérer le chargeur de stratégies (local à chaque processus)
            strategy_loader = get_strategy_loader()
            
            # Traiter les données avec toutes les stratégies
            signals = strategy_loader.process_market_data(data)
            
            return signals
        
        except Exception as e:
            logger.error(f"❌ Erreur dans le worker d'analyse: {str(e)}")
            return []
    
    def _handle_market_data(self, data: Dict[str, Any]) -> None:
        """
        Callback appelé pour chaque donnée de marché reçue.
        Ajoute les données à la file d'attente pour analyse.
        
        Args:
            data: Données de marché
        """
        # Ajouter à la file d'attente d'analyse
        self.data_queue.put(data)
    
    def _publish_signals(self, signals: List[StrategySignal]) -> None:
        """
        Publie les signaux générés sur Redis.
        
        Args:
            signals: Liste des signaux à publier
        """
        if not signals:
            return
        
        for signal in signals:
            self.redis_subscriber.publish_signal(signal)
    
    def _process_data_queue(self) -> None:
        """
        Processus/thread qui traite la file d'attente de données et soumet les tâches aux workers.
        """
        logger.info("Démarrage du processeur de file d'attente de données")
        
        # Utiliser ProcessPoolExecutor pour les processus ou ThreadPoolExecutor pour les threads
        executor_class = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            self.executor = executor
            
            while not self.stop_event.is_set():
                try:
                    # Ne pas bloquer indéfiniment pour pouvoir vérifier stop_event
                    try:
                        # Récupérer les données avec timeout
                        data = self.data_queue.get(timeout=0.1)
                    except (queue.Empty, mp.queues.Empty):
                        continue
                    
                    # Ne traiter que les chandeliers fermés
                    if not data.get('is_closed', False):
                        continue
                    
                    # Soumettre la tâche à un worker
                    future = executor.submit(self._worker_analyze, data)
                    
                    # Ajouter un callback pour traiter les résultats
                    future.add_done_callback(
                        lambda f: self._publish_signals(f.result())
                    )
                
                except Exception as e:
                    logger.error(f"❌ Erreur dans le processeur de file d'attente: {str(e)}")
                    time.sleep(0.1)  # Pause pour éviter une boucle d'erreur infinie
        
        logger.info("Processeur de file d'attente de données arrêté")
    
    def start(self) -> None:
        """
        Démarre le gestionnaire d'analyse.
        """
        logger.info("🚀 Démarrage du gestionnaire d'analyse...")
        
        # Réinitialiser l'événement d'arrêt
        self.stop_event.clear()
        
        # Démarrer le subscriber Redis
        self.redis_subscriber.start_listening(self._handle_market_data)
        
        # Démarrer le processus/thread de traitement de file d'attente
        if self.use_threads:
            import threading
            self.queue_processor = threading.Thread(
                target=self._process_data_queue,
                daemon=True
            )
        else:
            self.queue_processor = mp.Process(
                target=self._process_data_queue,
                daemon=True
            )
        
        self.queue_processor.start()
        logger.info("✅ Gestionnaire d'analyse démarré")
    
    def stop(self) -> None:
        """
        Arrête le gestionnaire d'analyse.
        """
        logger.info("Arrêt du gestionnaire d'analyse...")
        
        # Signaler l'arrêt
        self.stop_event.set()
        
        # Attendre que le processeur de file d'attente se termine
        if self.queue_processor.is_alive():
            if self.use_threads:
                self.queue_processor.join(timeout=5.0)
            else:
                self.queue_processor.join(timeout=5.0)
                if self.queue_processor.is_alive():
                    self.queue_processor.terminate()
            
            logger.info("Processeur de file d'attente arrêté")
        
        # Arrêter le subscriber Redis
        self.redis_subscriber.stop()
        
        # Vider les files d'attente
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except (queue.Empty, mp.queues.Empty):
                break
        
        while not self.signal_queue.empty():
            try:
                self.signal_queue.get_nowait()
            except (queue.Empty, mp.queues.Empty):
                break
        
        logger.info("✅ Gestionnaire d'analyse arrêté")

# Point d'entrée pour exécution directe
if __name__ == "__main__":
    try:
        # Configurer le gestionnaire de signaux pour l'arrêt propre
        stop_event = mp.Event()
        
        def signal_handler(sig, frame):
            logger.info(f"Signal {sig} reçu, arrêt en cours...")
            stop_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Créer et démarrer le gestionnaire d'analyse
        manager = AnalyzerManager()
        manager.start()
        
        # Attendre le signal d'arrêt
        while not stop_event.is_set():
            time.sleep(1.0)
        
        # Arrêter le gestionnaire
        manager.stop()
        logger.info("Fin du programme")
        
    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur critique: {str(e)}")