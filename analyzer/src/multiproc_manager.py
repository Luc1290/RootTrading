"""
Gestionnaire de processus multiples pour l'analyzer.
Permet d'exécuter plusieurs stratégies en parallèle sur différents cœurs CPU.
"""
import logging
import multiprocessing as mp
import os
import sys
import time
import datetime
import signal
import threading
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import queue
from functools import partial
from multiprocessing import Manager

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
        # Filtrer les symboles vides ou uniquement des espaces
        self.symbols = [s.strip() for s in (symbols or SYMBOLS) if s and s.strip()]
    
        # Log pour déboguer
        logger.info(f"Symboles après filtrage: {self.symbols}")

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

        # Créer un Manager pour partager les files d'attente entre processus
        if not use_threads:
            self.mp_manager = Manager()
            self.data_queue = self.mp_manager.Queue()
            self.signal_queue = self.mp_manager.Queue()
        else:
            self.data_queue = queue.Queue()
            self.signal_queue = queue.Queue()
    
        # Ajuster le nombre de workers en fonction des groupes
        self.max_workers = min(self.max_workers, len(self.symbol_groups))       
             
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
    
    def _handle_future_result(self, future_result):
        """
        Méthode statique pour traiter les résultats futurs sans capturer self
        """
        # Extraire les signaux du résultat
        signals = future_result
    
        # Au lieu de publier directement, mettre dans une file d'attente que le processus principal va gérer
        self.signal_queue.put(signals)

    def _process_signal_queue(self):
        """
        Processus/thread qui traite la file d'attente des signaux
        """
        logger.info("Démarrage du processeur de file d'attente des signaux")
    
        # Importer localement pour éviter les problèmes de pickling
        from shared.src.schemas import StrategySignal
        from shared.src.enums import OrderSide, SignalStrength
        from datetime import datetime

        while not self.stop_event.is_set():
            try:
                # Récupérer les signaux avec timeout
                try:
                    signal_dicts = self.signal_queue.get(timeout=0.1)
                except (queue.Empty, mp.queues.Empty):
                    continue
        
                if signal_dicts:
                    logger.info(f"Traitement de {len(signal_dicts)} signal(s) reçus")
                
                    # Traiter chaque dictionnaire de signal
                    for signal_dict in signal_dicts:
                        try:
                            # Convertir les chaînes en objets enum
                            if isinstance(signal_dict.get('side'), str):
                                try:
                                    signal_dict['side'] = OrderSide(signal_dict['side'])
                                except (ValueError, TypeError):
                                    signal_dict['side'] = OrderSide.BUY  # Valeur par défaut
                        
                            if isinstance(signal_dict.get('strength'), str):
                                try:
                                    signal_dict['strength'] = SignalStrength(signal_dict['strength'])
                                except (ValueError, TypeError):
                                    signal_dict['strength'] = SignalStrength.MODERATE  # Valeur par défaut
                        
                            # Convertir le timestamp
                            if isinstance(signal_dict.get('timestamp'), str):
                                try:
                                    signal_dict['timestamp'] = datetime.fromisoformat(signal_dict['timestamp'])
                                except (ValueError, TypeError):
                                    signal_dict['timestamp'] = datetime.now()
                        
                            # Recréer l'objet StrategySignal
                            signal = StrategySignal(**signal_dict)
                        
                            # Vérifier les champs obligatoires
                            required_fields = ['symbol', 'strategy', 'side', 'timestamp', 'price']
                            missing_fields = [field for field in required_fields 
                                            if not hasattr(signal, field) or getattr(signal, field) is None]
                        
                            if missing_fields:
                                logger.info(f"❌ Signal incomplet, ne sera pas publié. Champs manquants: {missing_fields}")
                                continue
                        
                            # Publier le signal si valide
                            self.redis_subscriber.publish_signal(signal)
                            logger.info(f"✅ Signal publié: {signal.side} pour {signal.symbol} @ {signal.price}")
                    
                        except Exception as e:
                            logger.info(f"❌ Erreur lors du traitement du signal: {str(e)}", exc_info=True)
            
                # Marquer comme traité
                self.signal_queue.task_done()
        
            except Exception as e:
                logger.info(f"❌ Erreur dans le processeur de file d'attente des signaux: {str(e)}")
                time.sleep(0.1)

        logger.info("Processeur de file d'attente des signaux arrêté")

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
        Processus/thread qui traite la file d'attente de données.
        Cette version n'utilise pas de pool d'executors pour éviter les problèmes de pickling.
        """
        logger.info("Démarrage du processeur de file d'attente de données")
    
        # Créer un loader de stratégies local à ce processus
        local_strategy_loader = None
    
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

                # Ajouter ces logs pour déboguer
                logger.info(f"Données reçues pour {data.get('symbol')}: is_closed={data.get('is_closed', False)}, close={data.get('close')}")
            
                # Extraire uniquement les données nécessaires
                analysis_data = {
                    'symbol': data.get('symbol', ''),
                    'open': data.get('open', 0.0),
                    'high': data.get('high', 0.0),
                    'low': data.get('low', 0.0),
                    'close': data.get('close', 0.0),
                    'volume': data.get('volume', 0.0),
                    'timestamp': data.get('timestamp', ''),
                    'start_time': data.get('start_time', 0),
                    'interval': data.get('interval', ''),
                    'is_closed': data.get('is_closed', True)
                }
            
                # Initialiser le strategy loader au besoin (seulement au premier appel)
                if local_strategy_loader is None:
                    from analyzer.src.strategy_loader import get_strategy_loader
                    local_strategy_loader = get_strategy_loader()
                    logger.info("Loader de stratégies local créé dans le processus d'analyse")
            
                # Analyser les données directement dans ce processus
                try:
                    signals = local_strategy_loader.process_market_data(analysis_data)
                
                    # Si des signaux sont générés, les convertir en dictionnaires et les envoyer
                    if signals:
                        # Convertir les signaux en dictionnaires pour éviter les problèmes de pickling
                        signal_dicts = []
                        for signal in signals:
                            try:
                                # Assurez-vous que tous les champs requis sont présents et correctement formatés
                                side_value = signal.side.value if hasattr(signal.side, 'value') else str(signal.side)
                                strength_value = signal.strength.value if hasattr(signal.strength, 'value') else "MODERATE"
                                timestamp_value = signal.timestamp.isoformat() if hasattr(signal.timestamp, 'isoformat') else datetime.now().isoformat()
            
                                # Créer un dictionnaire avec les données du signal
                                signal_dict = {
                                    'symbol': signal.symbol,
                                    'strategy': signal.strategy,
                                    'side': side_value,
                                    'timestamp': timestamp_value,
                                    'price': float(signal.price),
                                    'confidence': float(signal.confidence) if hasattr(signal, 'confidence') else 0.5,
                                    'strength': strength_value,
                                    'metadata': dict(signal.metadata) if hasattr(signal, 'metadata') and signal.metadata else {}
                                }
            
                                # Vérifier explicitement que tous les champs requis sont présents
                                required_fields = ['symbol', 'strategy', 'side', 'timestamp', 'price']
                                missing_fields = [field for field in required_fields if field not in signal_dict or signal_dict[field] is None]
            
                                if missing_fields:
                                    logger.warning(f"Signal incomplet, ne sera pas ajouté. Champs manquants: {missing_fields}")
                                    continue
                
                                signal_dicts.append(signal_dict)
                            except Exception as e:
                                logger.error(f"Erreur lors de la conversion du signal: {str(e)}")
                    
                            # Mettre les dictionnaires sur la file d'attente
                            if signal_dicts:
                                self.signal_queue.put(signal_dicts)
                            logger.info(f"Mis {len(signal_dicts)} signaux sur la file d'attente")
            
                except Exception as e:
                    logger.info(f"❌ Erreur lors de l'analyse des données: {str(e)}")
        
            except Exception as e:
                logger.error(f"❌ Erreur dans le processeur de file d'attente: {str(e)}")
                time.sleep(0.1)  # Pause pour éviter une boucle d'erreur infinie
    
        logger.info("Processeur de file d'attente de données arrêté")
      
    def start(self):
        """
        Démarre le gestionnaire d'analyse.
        """
        logger.info("🚀 Démarrage du gestionnaire d'analyse...")
    
        # Réinitialiser l'événement d'arrêt
        self.stop_event.clear()
    
        # Démarrer le subscriber Redis
        self.redis_subscriber.start_listening(self._handle_market_data)
    
        # Démarrer le processus/thread de traitement de file d'attente des données
        if self.use_threads:
            self.queue_processor = threading.Thread(
                target=self._process_data_queue,
                daemon=True
            )
        
            # Ajouter le processeur de signaux
            self.signal_processor = threading.Thread(
                target=self._process_signal_queue,
                daemon=True
            )
        else:
            self.queue_processor = mp.Process(
                target=self._process_data_queue,
                daemon=False
            )
        
            # Ajouter le processeur de signaux
            self.signal_processor = threading.Thread(  # Utiliser un thread même en mode processus
                target=self._process_signal_queue,
                daemon=True
            )
    
        self.queue_processor.start()
        self.signal_processor.start()
        logger.info("✅ Gestionnaire d'analyse démarré")
    
    def stop(self):
        """
        Arrête le gestionnaire d'analyse.
        """
        logger.info("Arrêt du gestionnaire d'analyse...")
    
        # Signaler l'arrêt
        self.stop_event.set()
    
        # Attendre que les processeurs se terminent
        if self.queue_processor.is_alive():
            if self.use_threads:
                self.queue_processor.join(timeout=5.0)
            else:
                self.queue_processor.join(timeout=5.0)
                if self.queue_processor.is_alive():
                    self.queue_processor.terminate()
        
            logger.info("Processeur de file d'attente de données arrêté")
    
        # Attendre que le processeur de signaux se termine
        if self.signal_processor.is_alive():
            self.signal_processor.join(timeout=5.0)
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