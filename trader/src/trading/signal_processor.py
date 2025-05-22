"""
Processeur de signaux de trading.
Analyse et filtre les signaux avant de créer des cycles de trading.
"""
import logging
import json
import time
import queue
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from shared.src.redis_client import RedisClient
from shared.src.enums import OrderSide, SignalStrength
from shared.src.schemas import StrategySignal

# Configuration du logging
logger = logging.getLogger(__name__)

def parse_signal_strength(strength_str):
    """Convertit une chaîne de force de signal en énumération SignalStrength."""
    if isinstance(strength_str, str):
        # Tenter de convertir directement via l'énumération
        try:
            return SignalStrength[strength_str.upper()]
        except (KeyError, AttributeError):
            # Mapping de fallback pour les anciennes valeurs ou formats différents
            mapping = {
                "WEAK": SignalStrength.WEAK,
                "MODERATE": SignalStrength.MODERATE,
                "STRONG": SignalStrength.STRONG,
                "VERY_STRONG": SignalStrength.VERY_STRONG,
                # Versions minuscules
                "weak": SignalStrength.WEAK,
                "moderate": SignalStrength.MODERATE,
                "strong": SignalStrength.STRONG,
                "very_strong": SignalStrength.VERY_STRONG
            }
            return mapping.get(strength_str, SignalStrength.MODERATE)  # Valeur par défaut
    elif strength_str is None:
        return SignalStrength.MODERATE  # Valeur par défaut si None
    return strength_str  # Si c'est déjà une énumération

class SignalProcessor:
    """
    Processeur de signaux de trading.
    Reçoit, valide et traite les signaux pour créer des cycles de trading.
    """
    
    def __init__(self, symbols: List[str], signal_callback: Callable[[StrategySignal], None],
                 min_signal_strength: SignalStrength = SignalStrength.MODERATE):
        """
        Initialise le processeur de signaux.
        
        Args:
            symbols: Liste des symboles autorisés
            signal_callback: Fonction de rappel pour les signaux validés
            min_signal_strength: Force minimale des signaux à traiter
        """
        # Log les valeurs possibles de SignalStrength pour débogage
        logger.debug(f"Valeurs possibles de SignalStrength: {[e.name for e in SignalStrength]}")
        logger.debug(f"Force minimale des signaux configurée: {min_signal_strength.name}")
        self.symbols = symbols
        self.signal_callback = signal_callback
        self.min_signal_strength = min_signal_strength
        
        # Client Redis pour recevoir les signaux
        self.redis_client = RedisClient()
        
        # Canal Redis pour les signaux
        self.signal_channel = "roottrading:analyze:signal"
        
        # File d'attente thread-safe pour les signaux
        self.signal_queue = queue.Queue()
        
        # Thread de traitement des signaux
        self.processing_thread = None
        self.running = False
        
        # Dictionnaire pour suivre les signaux récents (anti-doublon)
        self.recent_signals = {}
        self.signal_lock = threading.RLock()
        
        # Compteurs pour les statistiques
        self.stats = {
            "signals_received": 0,
            "signals_processed": 0,
            "signals_rejected": 0,
            "signals_duplicated": 0
        }
        
        logger.info(f"✅ SignalProcessor initialisé pour {len(symbols)} symboles")
    
    def _process_signal_message(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Callback pour traiter les signaux reçus de Redis.
        Valide et ajoute les signaux à la file d'attente pour traitement.

        Args:
            channel: Canal Redis d'où provient le signal
            data: Données du signal
        """
        try:
            # Logging pour debug - afficher les données brutes reçues
            logger.debug(f"Signal brut reçu: {json.dumps(data)}")
            
            with self.signal_lock:
                self.stats["signals_received"] += 1

            # Filtres de protection : ignorer les données de prix ou autres messages non pertinents
            if not isinstance(data, dict) or "strategy" not in data or "side" not in data:
                logger.debug(f"⏭️ Message ignoré: n'est pas un signal valide")
                return
        
            # Vérifier les champs obligatoires
            required_fields = ["strategy", "symbol", "side", "timestamp", "price"]
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                logger.warning(f"❌ Champs obligatoires manquants dans le signal: {missing_fields}")
                with self.signal_lock:
                    self.stats["signals_rejected"] += 1
                return
            
            # Vérifier que le symbole est supporté
            if data["symbol"] not in self.symbols:
                logger.debug(f"⏭️ Signal ignoré: symbole non supporté {data['symbol']}")
                with self.signal_lock:
                    self.stats["signals_rejected"] += 1
                return
            
            # Convertir la chaîne ISO en objet datetime
            if "timestamp" in data and isinstance(data["timestamp"], str):
                try:
                    data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                except ValueError:
                    logger.error(f"❌ Format de timestamp invalide: {data['timestamp']}")
                    with self.signal_lock:
                        self.stats["signals_rejected"] += 1
                    return
            
            # Vérifier les doublons
            signal_fingerprint = f"{data['symbol']}:{data['strategy']}:{data['side']}:{int(data['price'])}"
            
            with self.signal_lock:
                if signal_fingerprint in self.recent_signals:
                    recent_time = self.recent_signals[signal_fingerprint]
                    # Ignorer si moins de 30 secondes
                    if time.time() - recent_time < 30:
                        logger.debug(f"⏭️ Signal ignoré: doublon récent {signal_fingerprint}")
                        self.stats["signals_duplicated"] += 1
                        return

                # Mettre à jour l'horodatage du signal
                self.recent_signals[signal_fingerprint] = time.time()
                
                # Nettoyer les signaux trop anciens (plus de 5 minutes)
                current_time = time.time()
                expired_signals = [fp for fp, ts in self.recent_signals.items() if current_time - ts > 300]
                for fp in expired_signals:
                    self.recent_signals.pop(fp, None)
            
            # Convertir la valeur strength en enum avant validation
            if "strength" in data and isinstance(data["strength"], str):
                data["strength"] = parse_signal_strength(data["strength"])
            
            # Valider le signal avec Pydantic
            try:
                signal = StrategySignal(**data)
            except Exception as e:
                logger.error(f"❌ Erreur lors de la validation du signal: {str(e)}")
                logger.error(f"Données reçues problématiques: {data}")
                with self.signal_lock:
                    self.stats["signals_rejected"] += 1
                return
            
            # Vérifier la force du signal
            # S'assurer que strength est bien un objet enum et pas une chaîne
            signal_strength = signal.strength
            if isinstance(signal_strength, str):
                signal_strength = parse_signal_strength(signal_strength)
                
            if signal_strength.value < self.min_signal_strength.value:
                logger.debug(f"⏭️ Signal ignoré: force insuffisante ({signal_strength.name})")
                with self.signal_lock:
                    self.stats["signals_rejected"] += 1
                return
            
            # Ajouter à la file d'attente pour traitement
            self.signal_queue.put(signal)
            logger.info(f"📨 Signal reçu: {signal.side} {signal.symbol} @ {signal.price} ({signal.strategy})")

        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement du signal: {str(e)}")
            logger.error(f"Données reçues problématiques: {data}")
    
    def _signal_processing_loop(self) -> None:
        """
        Boucle de traitement des signaux de trading.
        Cette méthode s'exécute dans un thread séparé.
        """
        logger.info("Démarrage de la boucle de traitement des signaux")
        
        while self.running:
            try:
                # Récupérer un signal de la file d'attente avec timeout
                try:
                    signal = self.signal_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Traiter le signal
                try:
                    self.signal_callback(signal)
                    with self.signal_lock:
                        self.stats["signals_processed"] += 1
                except Exception as e:
                    logger.error(f"❌ Erreur lors du traitement du signal: {str(e)}")
                    with self.signal_lock:
                        self.stats["signals_rejected"] += 1
                
                # Marquer la tâche comme terminée
                self.signal_queue.task_done()
                
            except Exception as e:
                logger.error(f"❌ Erreur dans la boucle de traitement des signaux: {str(e)}")
                time.sleep(1)  # Pause pour éviter une boucle d'erreur infinie
        
        logger.info("Boucle de traitement des signaux arrêtée")
    
    def start(self) -> None:
        """
        Démarre le processeur de signaux.
        """
        if self.running:
            logger.warning("Le processeur de signaux est déjà en cours d'exécution")
            return
        
        self.running = True
        
        # S'abonner au canal des signaux et sauvegarder le client_id
        self.client_id = self.redis_client.subscribe(self.signal_channel, self._process_signal_message)
        logger.info(f"✅ Abonné au canal Redis des signaux: {self.signal_channel}")
        
        # Démarrer le thread de traitement des signaux
        self.processing_thread = threading.Thread(
            target=self._signal_processing_loop,
            daemon=True,
            name="SignalProcessor"
        )
        self.processing_thread.start()
        
        logger.info("✅ Processeur de signaux démarré")
    
    def stop(self) -> None:
        """
        Arrête le processeur de signaux.
        """
        if not self.running:
            return
        
        logger.info("Arrêt du processeur de signaux...")
        self.running = False
        
        # Attendre que le thread de traitement se termine
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        # Se désabonner du canal Redis
        if hasattr(self, 'client_id') and self.client_id:
            self.redis_client.unsubscribe(self.client_id)
        self.redis_client.close()
        
        logger.info("✅ Processeur de signaux arrêté")
    
    def get_stats(self) -> Dict[str, int]:
        """
        Récupère les statistiques du processeur de signaux.
        
        Returns:
            Dictionnaire des statistiques
        """
        with self.signal_lock:
            return self.stats.copy()
    
    def set_min_signal_strength(self, strength: SignalStrength) -> None:
        """
        Modifie la force minimale des signaux à traiter.
        
        Args:
            strength: Nouvelle force minimale
        """
        self.min_signal_strength = strength
        logger.info(f"Force minimale des signaux modifiée: {strength}")