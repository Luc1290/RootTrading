"""
Gestionnaire d'ordres pour le trader.
Reçoit les signaux, les valide, et crée des cycles de trading.
"""
import logging
import threading
import json
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import queue

# Importer les modules partagés
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import SYMBOLS, TRADE_QUANTITY
from shared.src.redis_client import RedisClient
from shared.src.enums import OrderSide, SignalStrength, CycleStatus  # Ajout de CycleStatus
from shared.src.schemas import StrategySignal, TradeOrder

from trader.src.cycle_manager import CycleManager
from trader.src.binance_executor import BinanceExecutor

# Configuration du logging
logger = logging.getLogger(__name__)

class OrderManager:
    """
    Gestionnaire d'ordres.
    Reçoit les signaux de trading, les valide, et crée des cycles de trading.
    """
    
    def __init__(self, symbols: List[str] = None, redis_client: RedisClient = None, 
                 cycle_manager: CycleManager = None):
        """
        Initialise le gestionnaire d'ordres.
        
        Args:
            symbols: Liste des symboles à surveiller
            redis_client: Client Redis préexistant (optionnel)
            cycle_manager: Gestionnaire de cycles préexistant (optionnel)
        """
        self.symbols = symbols or SYMBOLS
        self.redis_client = redis_client or RedisClient()
        self.cycle_manager = cycle_manager or CycleManager()
        
        # Configuration du canal Redis pour les signaux
        self.signal_channel = "roottrading:analyze:signal"
        
        # Configuration du canal Redis pour les mises à jour de prix
        self.price_channels = [f"roottrading:market:data:{symbol.lower()}" for symbol in self.symbols]
        
        # File d'attente thread-safe pour les signaux
        self.signal_queue = queue.Queue()
        
        # Dictionnaire pour stocker les derniers prix par symbole
        self.last_prices: Dict[str, float] = {}
        
        # Thread pour le traitement des signaux
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        # Thread pour le traitement des mises à jour de prix
        self.price_thread = None
        
        logger.info(f"✅ OrderManager initialisé pour {len(self.symbols)} symboles: {', '.join(self.symbols)}")
    
    def _process_signal(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Callback pour traiter les signaux reçus de Redis.
        Ajoute les signaux à la file d'attente pour traitement.
        
        Args:
            channel: Canal Redis d'où provient le signal
            data: Données du signal
        """
        try:
            # Valider le signal avec Pydantic
            signal = StrategySignal(**data)
            
            # Ajouter à la file d'attente pour traitement
            self.signal_queue.put(signal)
            
            logger.info(f"📨 Signal reçu: {signal.side} {signal.symbol} @ {signal.price} ({signal.strategy})")
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement du signal: {str(e)}")
    
    def _process_price_update(self, channel: str, data: Dict[str, Any]) -> None:
        """
        Callback pour traiter les mises à jour de prix reçues de Redis.
        Met à jour les derniers prix et vérifie les stop-loss/take-profit.
        
        Args:
            channel: Canal Redis d'où provient la mise à jour
            data: Données de prix
        """
        try:
            # Ne traiter que les chandeliers fermés
            if not data.get('is_closed', False):
                return
            
            symbol = data.get('symbol')
            price = data.get('close')
            
            if not symbol or price is None:
                return
            
            # Mettre à jour le dernier prix
            self.last_prices[symbol] = price
            
            # Traiter la mise à jour de prix dans le gestionnaire de cycles
            self.cycle_manager.process_price_update(symbol, price)
            
            logger.debug(f"💰 Prix mis à jour: {symbol} @ {price}")
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement de la mise à jour de prix: {str(e)}")
    
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
                
                # Traiter le signal
                self._handle_signal(signal)
                
                # Marquer la tâche comme terminée
                self.signal_queue.task_done()
                
            except Exception as e:
                logger.error(f"❌ Erreur dans la boucle de traitement des signaux: {str(e)}")
                time.sleep(1)  # Pause pour éviter une boucle d'erreur infinie
        
        logger.info("Boucle de traitement des signaux arrêtée")

    # Calculer le pourcentage minimal de changement de prix nécessaire pour être rentable
    def calculate_min_profitable_change(self, symbol: str) -> float:
        """
        Calcule le pourcentage minimal de changement de prix nécessaire pour être rentable,
        en tenant compte des frais d'achat et de vente.
    
        Args:
            symbol: Le symbole de la paire de trading
        
        Returns:
            Le pourcentage minimal nécessaire (ex: 0.3 pour 0.3%)
        """
        maker_fee, taker_fee = self.binance_executor.get_trade_fee(symbol)
    
        # Calculer l'impact total des frais (achat + vente)
        # Pour être sûr, on considère les frais taker qui sont généralement plus élevés
        total_fee_impact = (1 + taker_fee) * (1 + taker_fee) - 1
    
        # Ajouter une petite marge de sécurité (20% supplémentaire)
        min_profitable_change = total_fee_impact * 1.2 * 100  # Convertir en pourcentage
    
        return min_profitable_change
    
    def _handle_signal(self, signal: StrategySignal) -> None:
        """
        Traite un signal de trading et crée un cycle si nécessaire.
        
        Args:
            signal: Signal de trading à traiter
        """
        try:
            # Vérifier si le symbole est supporté
            if signal.symbol not in self.symbols:
                logger.warning(f"⚠️ Signal ignoré: symbole non supporté {signal.symbol}")
                return
            
            # Vérifier si le signal est assez fort
            if signal.strength in [SignalStrength.WEAK]:
                logger.info(f"⚠️ Signal ignoré: force insuffisante ({signal.strength})")
                return
            
            # Récupérer le dernier prix du symbole (utiliser le prix du signal si non disponible)
            current_price = self.last_prices.get(signal.symbol, signal.price)
            
            # Calculer la quantité à trader (utiliser la configuration par défaut)
            quantity = TRADE_QUANTITY
            
            # Calculer les prix cibles et stop-loss
            target_price = None
            stop_price = None
            trailing_delta = None
            
            # Si le signal contient des métadonnées spécifiques
            metadata = signal.metadata or {}
            
            # Récupérer target/stop des métadonnées si présents
            if 'target_price' in metadata:
                target_price = float(metadata['target_price'])
            if 'stop_price' in metadata:
                stop_price = float(metadata['stop_price'])
            if 'trailing_delta' in metadata:
                trailing_delta = float(metadata['trailing_delta'])
            
            # Sinon, calculer des valeurs par défaut
            if target_price is None and signal.side == OrderSide.BUY:
                # Pour un achat, cible = prix + 3%
                target_price = current_price * 1.03
            elif target_price is None and signal.side == OrderSide.SELL:
                # Pour une vente, cible = prix - 3%
                target_price = current_price * 0.97
            
            if stop_price is None and signal.side == OrderSide.BUY:
                # Pour un achat, stop = prix - 2%
                stop_price = current_price * 0.98
            elif stop_price is None and signal.side == OrderSide.SELL:
                # Pour une vente, stop = prix + 2%
                stop_price = current_price * 1.02

             # Vérifier que le gain potentiel est supérieur au seuil minimal
            min_change = self.calculate_min_profitable_change(signal.symbol)
            target_price_percent = abs((target_price - current_price) / current_price * 100)
        
            if target_price_percent < min_change:
                logger.info(f"⚠️ Signal ignoré: gain potentiel {target_price_percent:.2f}% inférieur au seuil minimal {min_change:.2f}%")
                return
            
            # Déterminer la poche à utiliser (par défaut: active)
            pocket = metadata.get('pocket', 'active')
            
            # Créer un nouveau cycle de trading
            cycle = self.cycle_manager.create_cycle(
                symbol=signal.symbol,
                strategy=signal.strategy,
                side=signal.side,
                price=current_price,
                quantity=quantity,
                pocket=pocket,
                target_price=target_price,
                stop_price=stop_price,
                trailing_delta=trailing_delta
            )
            
            if cycle:
                logger.info(f"✅ Cycle créé pour le signal: {cycle.id}")
            else:
                logger.error("❌ Échec de création du cycle pour le signal")
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du traitement du signal: {str(e)}")
    
    def start(self) -> None:
        """
        Démarre le gestionnaire d'ordres.
        S'abonne aux canaux Redis et lance les threads de traitement.
        """
        logger.info("🚀 Démarrage du gestionnaire d'ordres...")
        
        # Réinitialiser l'événement d'arrêt
        self.stop_event.clear()
        
        # S'abonner au canal des signaux
        self.redis_client.subscribe(self.signal_channel, self._process_signal)
        logger.info(f"✅ Abonné au canal Redis des signaux: {self.signal_channel}")
        
        # S'abonner aux canaux des mises à jour de prix
        self.redis_client.subscribe(self.price_channels, self._process_price_update)
        logger.info(f"✅ Abonné aux canaux Redis des prix: {len(self.price_channels)} canaux")
        
        # Démarrer le thread de traitement des signaux
        self.processing_thread = threading.Thread(
            target=self._signal_processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        
        logger.info("✅ Gestionnaire d'ordres démarré")
    
    def stop(self) -> None:
        """
        Arrête le gestionnaire d'ordres.
        """
        logger.info("Arrêt du gestionnaire d'ordres...")
        
        # Signaler l'arrêt aux threads
        self.stop_event.set()
        
        # Attendre que le thread de traitement se termine
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
            logger.info("Thread de traitement des signaux arrêté")
        
        # Se désabonner des canaux Redis
        self.redis_client.unsubscribe()
        
        # Fermer le client Redis
        self.redis_client.close()
        
        # Fermer le gestionnaire de cycles
        self.cycle_manager.close()
        
        logger.info("✅ Gestionnaire d'ordres arrêté")
    
    def create_manual_order(self, symbol: str, side: OrderSide, quantity: float, 
                           price: Optional[float] = None) -> str:
        """
        Crée un ordre manuel (hors signal).
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            side: Côté de l'ordre (BUY ou SELL)
            quantity: Quantité à trader
            price: Prix (optionnel, sinon au marché)
            
        Returns:
            ID du cycle créé ou message d'erreur
        """
        try:
            # Vérifier si le symbole est supporté
            if symbol not in self.symbols:
                return f"Symbole non supporté: {symbol}"
            
            # Utiliser le dernier prix connu si non spécifié
            if price is None:
                price = self.last_prices.get(symbol)
                if price is None:
                    return "Prix non spécifié et aucun prix récent disponible"
            
            # Créer un cycle avec la stratégie "Manual"
            cycle = self.cycle_manager.create_cycle(
                symbol=symbol,
                strategy="Manual",
                side=side,
                price=price,
                quantity=quantity,
                pocket="active"  # Utiliser la poche active par défaut
            )
            
            if cycle:
                return cycle.id
            else:
                return "Échec de création du cycle"
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création de l'ordre manuel: {str(e)}")
            return f"Erreur: {str(e)}"
    
    def get_active_orders(self) -> List[Dict[str, Any]]:
        """
        Récupère la liste des ordres actifs.
        
        Returns:
            Liste des ordres actifs au format JSON
        """
        active_cycles = self.cycle_manager.get_active_cycles()
        
        # Convertir les cycles en dictionnaires pour JSON
        orders = []
        for cycle in active_cycles:
            orders.append({
                "id": cycle.id,
                "symbol": cycle.symbol,
                "strategy": cycle.strategy,
                "status": cycle.status.value,
                "side": "BUY" if cycle.status in [CycleStatus.ACTIVE_BUY, CycleStatus.WAITING_BUY] else "SELL",
                "entry_price": cycle.entry_price,
                "current_price": self.last_prices.get(cycle.symbol),
                "quantity": cycle.quantity,
                "target_price": cycle.target_price,
                "stop_price": cycle.stop_price,
                "created_at": cycle.created_at.isoformat() if cycle.created_at else None
            })
        
        return orders