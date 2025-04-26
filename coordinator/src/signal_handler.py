"""
Module de gestion des signaux de trading.
Reçoit les signaux, les valide, et coordonne la création des cycles de trading.
"""
import logging
import json
import requests
import threading
import time
from typing import Dict, Any, Optional, List
import queue

# Importer les modules partagés
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.redis_client import RedisClient
from shared.src.config import TRADING_MODE
from shared.src.enums import OrderSide, SignalStrength, CycleStatus
from shared.src.schemas import StrategySignal, TradeOrder

from coordinator.src.pocket_checker import PocketChecker

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SignalHandler:
    """
    Gestionnaire des signaux de trading.
    Reçoit les signaux, les valide, et coordonne la création des cycles de trading.
    """
    
    def __init__(self, trader_api_url: str = "http://trader:5002", 
                 portfolio_api_url: str = "http://portfolio:8000"):
        """
        Initialise le gestionnaire de signaux.
        
        Args:
            trader_api_url: URL de l'API du service Trader
            portfolio_api_url: URL de l'API du service Portfolio
        """
        self.trader_api_url = trader_api_url
        self.portfolio_api_url = portfolio_api_url
        
        # Client Redis pour les communications
        self.redis_client = RedisClient()
        
        # Canal Redis pour les signaux
        self.signal_channel = "roottrading:analyze:signal"
        
        # File d'attente thread-safe pour les signaux
        self.signal_queue = queue.Queue()
        
        # Thread pour le traitement des signaux
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        # Gestionnaire de poches
        self.pocket_checker = PocketChecker(portfolio_api_url)
        
        # Cache des prix actuels
        self.price_cache = {}
        
        # Configuration du mode de trading
        self.demo_mode = TRADING_MODE.lower() == 'demo'
        
        # Stratégies spéciales pour le filtrage
        self.filter_strategies = ['Ride_or_React_Strategy']
        self.market_filters = {}  # {symbol: {filter_data}}

        # Circuit breakers pour éviter les appels répétés à des services en échec
        self.trader_circuit = CircuitBreaker()
        self.portfolio_circuit = CircuitBreaker()
        
        logger.info(f"✅ SignalHandler initialisé en mode {'DÉMO' if self.demo_mode else 'RÉEL'}")
    
    def _process_signal(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Callback pour traiter les signaux reçus de Redis.
        Ajoute les signaux à la file d'attente pour traitement.
        
        Args:
            channel: Canal Redis d'où provient le signal
            data: Données du signal
        """
        try:
            # Valider le signal
            signal = StrategySignal(**data)
            
            # Traiter les signaux de filtrage séparément
            if signal.strategy in self.filter_strategies:
                self._update_market_filters(signal)
                return
            
            # Ajouter à la file d'attente pour traitement
            self.signal_queue.put(signal)
            
            # Mettre à jour le cache des prix
            self.price_cache[signal.symbol] = signal.price
            
            logger.info(f"📨 Signal reçu: {signal.side} {signal.symbol} @ {signal.price} ({signal.strategy})")
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement du signal: {str(e)}")
    
    def _update_market_filters(self, signal: StrategySignal) -> None:
        """
        Met à jour les filtres de marché basés sur des stratégies spéciales comme Ride or React.
        
        Args:
            signal: Signal de la stratégie de filtrage
        """
        if signal.strategy == 'Ride_or_React_Strategy':
            # Stocker les informations de mode dans le dictionnaire de filtres
            self.market_filters[signal.symbol] = {
                'mode': signal.metadata.get('mode', 'react'),
                'action': signal.metadata.get('action', 'normal_trading'),
                'updated_at': time.time()
            }
            
            logger.info(f"🔍 Filtre de marché mis à jour pour {signal.symbol}: "
                       f"mode={signal.metadata.get('mode', 'react')}")
    
    def _should_filter_signal(self, signal: StrategySignal) -> bool:
        """
        Détermine si un signal doit être filtré en fonction des conditions de marché.
        
        Args:
            signal: Signal à évaluer
            
        Returns:
            True si le signal doit être filtré (ignoré), False sinon
        """
        # Vérifier si nous avons des informations de filtrage pour ce symbole
        if signal.symbol not in self.market_filters:
            return False
        
        filter_info = self.market_filters[signal.symbol]
        
        # Vérifier si les informations de filtrage sont récentes (moins de 30 minutes)
        if time.time() - filter_info.get('updated_at', 0) > 1800:
            logger.warning(f"Informations de filtrage obsolètes pour {signal.symbol}, ignorées")
            return False
        
        # En mode "ride", filtrer certains signaux
        if filter_info.get('mode') == 'ride':
            # Si dans une tendance haussière forte, filtrer les signaux SELL (sauf très forts)
            if signal.side == OrderSide.SELL and signal.strength != SignalStrength.VERY_STRONG:
                logger.info(f"🔍 Signal {signal.side} filtré: marché en mode RIDE pour {signal.symbol}")
                return True
        
        return False
    
    def _calculate_trade_amount(self, signal: StrategySignal) -> float:
        """
        Calcule le montant à trader basé sur le signal.
        
        Args:
            signal: Signal de trading
            
        Returns:
            Montant en USDC à réserver
        """
        # Valeurs par défaut
        default_amount = 100.0  # 100 USDC par défaut
        
        # Ajuster en fonction de la force du signal
        if signal.strength == SignalStrength.WEAK:
            amount = default_amount * 0.5
        elif signal.strength == SignalStrength.MODERATE:
            amount = default_amount * 0.8
        elif signal.strength == SignalStrength.STRONG:
            amount = default_amount * 1.0
        elif signal.strength == SignalStrength.VERY_STRONG:
            amount = default_amount * 1.2
        else:
            amount = default_amount
        
        # TODO: Logique d'ajustement plus complexe basée sur le portefeuille total
        # et les limites de risque par trade
        
        return amount
    
    def _create_trade_cycle(self, signal: StrategySignal) -> Optional[str]:
        """
        Crée un cycle de trading à partir d'un signal.

        Args:
            signal: Signal de trading validé

        Returns:
            ID du cycle créé ou None en cas d'échec
        """
        # Vérifier le circuit breaker pour le portfolio
        if not self.portfolio_circuit.can_execute():
            logger.warning(f"Circuit ouvert pour le service Portfolio, signal ignoré")
            return None
    
        try:
            # Calculer le montant à trader
            trade_amount = self._calculate_trade_amount(signal)
        
            # Déterminer la poche à utiliser
            try:
                pocket_type = self.pocket_checker.determine_best_pocket(trade_amount)
                # Appel au portfolio réussi
                self.portfolio_circuit.record_success()
            except Exception as e:
                self.portfolio_circuit.record_failure()
                logger.error(f"❌ Erreur lors de l'interaction avec Portfolio: {str(e)}")
                return None
        
            if not pocket_type:
                logger.warning(f"❌ Aucune poche disponible pour un trade de {trade_amount:.2f} USDC")
                return None
        
            # Convertir le montant en quantité
            quantity = trade_amount / signal.price
        
            # Calculer le stop-loss et take-profit
            stop_price = signal.metadata.get('stop_price')
            target_price = signal.metadata.get('target_price')
        
            # Préparer la requête pour le Trader
            order_data = {
                "symbol": signal.symbol,
                "side": signal.side.value,
                "quantity": quantity,
                "price": signal.price,
                "strategy": signal.strategy,
                "timestamp": int(time.time() * 1000)  # Un timestamp actuel en millisecondes
}
        
            # Réserver les fonds dans la poche
            temp_cycle_id = f"temp_{int(time.time())}"
            try:
                reserved = self.pocket_checker.reserve_funds(trade_amount, temp_cycle_id, pocket_type)
                # Autre appel au portfolio réussi
                self.portfolio_circuit.record_success()
            except Exception as e:
                self.portfolio_circuit.record_failure()
                logger.error(f"❌ Erreur lors de la réservation des fonds: {str(e)}")
                return None
        
            if not reserved:
                logger.error(f"❌ Échec de réservation des fonds pour le trade")
                return None
        
            # Vérifier le circuit breaker pour le Trader
            if not self.trader_circuit.can_execute():
                logger.warning(f"Circuit ouvert pour le service Trader, libération des fonds réservés")
                self.pocket_checker.release_funds(trade_amount, temp_cycle_id, pocket_type)
                return None
        
            # Créer le cycle via l'API du Trader
            try:
                response = requests.post(f"{self.trader_api_url}/order", json=order_data)
                response.raise_for_status()
            
                result = response.json()
                cycle_id = result.get('order_id')
            
                # Appel au trader réussi
                self.trader_circuit.record_success()
            
                if not cycle_id:
                    logger.error("❌ Réponse invalide du Trader: pas d'ID de cycle")
                    # Libérer les fonds réservés
                    self.pocket_checker.release_funds(trade_amount, temp_cycle_id, pocket_type)
                    return None
            
                # Mettre à jour la réservation avec l'ID réel du cycle
                try:
                    self.pocket_checker.release_funds(trade_amount, temp_cycle_id, pocket_type)
                    self.pocket_checker.reserve_funds(trade_amount, cycle_id, pocket_type)
                    self.portfolio_circuit.record_success()
                except Exception as e:
                    self.portfolio_circuit.record_failure()
                    logger.error(f"❌ Erreur lors de la mise à jour de la réservation: {str(e)}")
                    # Tenter d'annuler le cycle créé
                    try:
                        requests.delete(f"{self.trader_api_url}/order/{cycle_id}")
                    except:
                        pass
                    return None
            
                logger.info(f"✅ Cycle de trading créé: {cycle_id} ({signal.side} {signal.symbol})")
                return cycle_id
                
            except requests.RequestException as e:
                self.trader_circuit.record_failure()
                logger.error(f"❌ Erreur lors de la création du cycle: {str(e)}")
                # Libérer les fonds réservés
                self.pocket_checker.release_funds(trade_amount, temp_cycle_id, pocket_type)
                return None
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création du cycle de trading: {str(e)}")
            return None
    
    def _signal_processing_loop(self) -> None:
        """
        Boucle de traitement des signaux de trading.
        Cette méthode s'exécute dans un thread séparé.
        """
        logger.info("Démarrage de la boucle de traitement des signaux")
        
        while not self.stop_event.is_set():
            try:
                # Récupérer un signal de la file d'attente avec timeout
                try:
                    signal = self.signal_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Vérifier si le signal doit être filtré
                if self._should_filter_signal(signal):
                    self.signal_queue.task_done()
                    continue
                
                # Vérifier la force du signal
                if signal.strength in [SignalStrength.WEAK]:
                    logger.info(f"⚠️ Signal ignoré: trop faible ({signal.strength})")
                    self.signal_queue.task_done()
                    continue
                
                # Créer un cycle de trading
                cycle_id = self._create_trade_cycle(signal)
                
                if cycle_id:
                    logger.info(f"✅ Trade exécuté pour le signal {signal.strategy} sur {signal.symbol}")
                else:
                    logger.warning(f"⚠️ Échec d'exécution du trade pour le signal {signal.strategy}")
                
                # Marquer la tâche comme terminée
                self.signal_queue.task_done()
                
            except Exception as e:
                logger.error(f"❌ Erreur dans la boucle de traitement des signaux: {str(e)}")
                time.sleep(1)  # Pause pour éviter une boucle d'erreur infinie
        
        logger.info("Boucle de traitement des signaux arrêtée")
    
    def start(self) -> None:
        """
        Démarre le gestionnaire de signaux.
        """
        logger.info("🚀 Démarrage du gestionnaire de signaux...")
        
        # S'abonner au canal des signaux
        self.redis_client.subscribe(self.signal_channel, self._process_signal)
        
        # Démarrer le thread de traitement des signaux
        self.stop_event.clear()
        self.processing_thread = threading.Thread(
            target=self._signal_processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        
        logger.info("✅ Gestionnaire de signaux démarré")
    
    def stop(self) -> None:
        """
        Arrête le gestionnaire de signaux.
        """
        logger.info("Arrêt du gestionnaire de signaux...")
        
        # Signaler l'arrêt aux threads
        self.stop_event.set()
        
        # Attendre que le thread de traitement se termine
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        # Se désabonner du canal Redis
        self.redis_client.unsubscribe()
        
        logger.info("✅ Gestionnaire de signaux arrêté")

class CircuitBreaker:
    """Circuit breaker pour éviter les appels répétés à des services en échec."""
    
    def __init__(self, max_failures=3, reset_timeout=60):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.open_since = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def record_success(self):
        """Enregistre un succès et réinitialise le circuit."""
        self.failure_count = 0
        self.state = "CLOSED"
        self.open_since = None
    
    def record_failure(self):
        """Enregistre un échec et ouvre le circuit si nécessaire."""
        self.failure_count += 1
        if self.failure_count >= self.max_failures:
            self.state = "OPEN"
            self.open_since = time.time()
    
    def can_execute(self):
        """Vérifie si une opération peut être exécutée."""
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            # Vérifier si le temps de reset est écoulé
            if time.time() - self.open_since > self.reset_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        
        # HALF_OPEN: permettre un essai
        return True
