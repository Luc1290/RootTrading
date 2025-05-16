# trader/src/trading/cycle_manager.py
"""
Gestionnaire des cycles de trading.
Version simplifiée qui délègue à d'autres modules.
"""
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import threading
from threading import RLock

from shared.src.config import get_db_url, TRADING_MODE
from shared.src.enums import OrderSide, OrderStatus, CycleStatus
from shared.src.schemas import TradeOrder, TradeExecution, TradeCycle
from shared.src.db_pool import DBContextManager, DBConnectionPool, transaction

from trader.src.exchange.binance_executor import BinanceExecutor
from trader.src.trading.cycle_repository import CycleRepository
from trader.src.trading.stop_manager import StopManager

# Configuration du logging
logger = logging.getLogger(__name__)

class CycleManager:
    """
    Gestionnaire des cycles de trading.
    Crée, met à jour et suit l'état des cycles de trading.
    """
    
    def __init__(self, db_url: str = None, binance_executor: BinanceExecutor = None):
        """
        Initialise le gestionnaire de cycles.
        
        Args:
            db_url: URL de connexion à la base de données
            binance_executor: Exécuteur Binance préexistant (optionnel)
        """
        self.db_url = db_url or get_db_url()
        self.binance_executor = binance_executor or BinanceExecutor()
        self.demo_mode = TRADING_MODE.lower() == 'demo'
        
        # Initialiser les composants
        self.repository = CycleRepository(self.db_url)
        self.stop_manager = StopManager(self.repository)
        
        # Dictionnaire des cycles actifs {id_cycle: cycle}
        self.active_cycles: Dict[str, TradeCycle] = {}
        
        # Mutex pour l'accès concurrent aux cycles
        self.cycles_lock = RLock()
        
        # Initialiser le pool de connexions DB
        try:
            self.db_pool = DBConnectionPool.get_instance()
            self._load_active_cycles_from_db()
            # Démarrer le thread de nettoyage périodique
            self._start_cleanup_thread()
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation de la base de données: {str(e)}")
        
        logger.info(f"✅ CycleManager initialisé en mode {'DÉMO' if self.demo_mode else 'RÉEL'}")
    
    def _start_cleanup_thread(self):
        """Démarre un thread de nettoyage périodique des cycles inactifs."""
        def cleanup_routine():
            while True:
                try:
                    # Nettoyer les cycles inactifs toutes les heures
                    time.sleep(3600)
                    self._cleanup_inactive_cycles()
                except Exception as e:
                    logger.error(f"❌ Erreur dans le thread de nettoyage: {str(e)}")
        
        cleanup_thread = threading.Thread(target=cleanup_routine, daemon=True)
        cleanup_thread.start()
        logger.info("Thread de nettoyage des cycles démarré")

    def _cleanup_inactive_cycles(self):
        """Nettoie les cycles inactifs qui sont restés en mémoire trop longtemps."""
        now = datetime.now()
        cycles_to_remove = []
        
        with self.cycles_lock:
            for cycle_id, cycle in self.active_cycles.items():
                # Si le cycle est en état terminal depuis plus de 24h, le supprimer de la mémoire
                if (cycle.status in [CycleStatus.COMPLETED, CycleStatus.CANCELED, CycleStatus.FAILED] and
                    (now - cycle.updated_at).total_seconds() > 24 * 3600):
                    cycles_to_remove.append(cycle_id)
        
        # Supprimer les cycles identifiés
        if cycles_to_remove:
            with self.cycles_lock:
                for cycle_id in cycles_to_remove:
                    self.active_cycles.pop(cycle_id, None)
            
            logger.info(f"🧹 {len(cycles_to_remove)} cycles inactifs nettoyés de la mémoire")
    
    def _load_active_cycles_from_db(self) -> None:
        """
        Charge les cycles actifs depuis la base de données.
        """
        try:
            cycles = self.repository.get_active_cycles()
            
            with self.cycles_lock:
                self.active_cycles = {cycle.id: cycle for cycle in cycles}
                
            logger.info(f"✅ {len(self.active_cycles)} cycles actifs chargés depuis la base de données")
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement des cycles actifs: {str(e)}")
    
    def create_cycle(self, symbol: str, strategy: str, side: Union[OrderSide, str], 
                    price: float, quantity: float, pocket: Optional[str] = None,
                    target_price: Optional[float] = None, stop_price: Optional[float] = None,
                    trailing_delta: Optional[float] = None) -> Optional[TradeCycle]:
        """
        Crée un nouveau cycle de trading et exécute l'ordre d'entrée.
        
        Returns:
            Cycle créé ou None si l'ordre Binance échoue.
        """
        try:
            if isinstance(side, str):
                side = OrderSide(side)

            # Valider la quantité avant d'aller plus loin
            if quantity <= 0:
                logger.error(f"❌ Quantité invalide pour création de cycle: {quantity}")
                return None

            cycle_id = f"cycle_{uuid.uuid4().hex[:16]}"
            now = datetime.now()

            # Créer l'objet cycle
            cycle = TradeCycle(
                id=cycle_id,
                symbol=symbol,
                strategy=strategy,
                status=CycleStatus.INITIATING,
                entry_price=None,
                quantity=quantity,
                target_price=target_price,
                stop_price=stop_price,
                trailing_delta=trailing_delta,
                created_at=now,
                updated_at=now,
                pocket=pocket,
                demo=self.demo_mode
            )

            # Créer l'ordre
            entry_order = TradeOrder(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                client_order_id=f"entry_{cycle_id}",
                strategy=strategy,
                demo=self.demo_mode
            )

            logger.info(f"🔄 Envoi de l'ordre d'entrée pour le cycle {cycle_id}")
            execution = self.binance_executor.execute_order(entry_order)

            # Vérifier si l'exécution a réussi
            if not execution or not execution.order_id or execution.status != OrderStatus.FILLED:
                logger.error(f"❌ L'ordre d'entrée pour le cycle {cycle_id} a échoué ou n'est pas FILLED")
                return None

            # Mise à jour du cycle avec données exécutées
            with self.cycles_lock:
                cycle.entry_order_id = execution.order_id
                cycle.entry_price = execution.price
                cycle.status = CycleStatus.ACTIVE_BUY if side == OrderSide.BUY else CycleStatus.ACTIVE_SELL
                cycle.confirmed = True
                cycle.updated_at = datetime.now()
                self.active_cycles[cycle_id] = cycle

            # Enregistrer l'exécution et le cycle
            self.repository.save_execution(execution, cycle_id)
            self.repository.save_cycle(cycle)

            # Publier sur Redis
            self._publish_cycle_event(cycle, "created")

            logger.info(f"✅ Cycle {cycle_id} créé avec succès: {side.value} {quantity} {symbol} @ {execution.price}")
            return cycle

        except Exception as e:
            logger.error(f"❌ Erreur lors de la création du cycle: {str(e)}")
            return None
    
    def _publish_cycle_event(self, cycle: TradeCycle, event_type: str) -> None:
        """
        Publie un événement de cycle sur Redis.
        
        Args:
            cycle: Cycle concerné
            event_type: Type d'événement (created, updated, closed, etc.)
        """
        try:
            from shared.src.redis_client import RedisClient
            redis = RedisClient()
            
            # Convertir les valeurs NumPy ou Decimal si présentes
            cycle_data = {
                "cycle_id": cycle.id,
                "symbol": cycle.symbol,
                "strategy": cycle.strategy,
                "quantity": float(cycle.quantity) if hasattr(cycle.quantity, 'dtype') else cycle.quantity,
                "entry_price": float(cycle.entry_price) if hasattr(cycle.entry_price, 'dtype') else cycle.entry_price,
                "timestamp": int(cycle.created_at.timestamp() * 1000),
                "pocket": cycle.pocket
            }
            
            # Ajouter des infos supplémentaires selon le type d'événement
            if event_type == "closed" and cycle.exit_price:
                cycle_data["exit_price"] = float(cycle.exit_price) if hasattr(cycle.exit_price, 'dtype') else cycle.exit_price
                cycle_data["profit_loss"] = float(cycle.profit_loss) if hasattr(cycle.profit_loss, 'dtype') else cycle.profit_loss
                cycle_data["profit_loss_percent"] = float(cycle.profit_loss_percent) if hasattr(cycle.profit_loss_percent, 'dtype') else cycle.profit_loss_percent
            
            redis.publish(f"roottrading:cycle:{event_type}", cycle_data)
            logger.info(f"📢 Événement {event_type} publié pour le cycle {cycle.id}")
        except Exception as e:
            logger.warning(f"⚠️ Impossible de publier l'événement de cycle sur Redis: {str(e)}")
    
    def close_cycle(self, cycle_id: str, exit_price: Optional[float] = None) -> bool:
        """
        Ferme un cycle de trading en exécutant l'ordre de sortie.
        
        Args:
            cycle_id: ID du cycle à fermer
            exit_price: Prix de sortie (optionnel, sinon au marché)
            
        Returns:
            True si la fermeture a réussi, False sinon
        """
        try:
            # Récupérer le cycle
            with self.cycles_lock:
                if cycle_id not in self.active_cycles:
                    logger.warning(f"⚠️ Cycle {cycle_id} non trouvé dans les cycles actifs")
                    return False
                
                cycle = self.active_cycles[cycle_id]
            
            # Vérifier que le cycle peut être fermé
            if cycle.status not in [CycleStatus.WAITING_BUY, CycleStatus.ACTIVE_BUY, 
                                   CycleStatus.WAITING_SELL, CycleStatus.ACTIVE_SELL]:
                logger.warning(f"⚠️ Impossible de fermer le cycle {cycle_id} avec le statut {cycle.status}")
                return False
            
            # Déterminer le côté de l'ordre de sortie (inverse de l'entrée)
            if cycle.status in [CycleStatus.WAITING_BUY, CycleStatus.ACTIVE_BUY]:
                exit_side = OrderSide.SELL
            else:
                exit_side = OrderSide.BUY
            
            # Créer l'ordre de sortie
            exit_order = TradeOrder(
                symbol=cycle.symbol,
                side=exit_side,
                quantity=cycle.quantity,
                price=exit_price,  # None pour un ordre au marché
                client_order_id=f"exit_{cycle_id}",
                strategy=cycle.strategy,
                demo=cycle.demo
            )
            
            # Exécuter l'ordre de sortie
            logger.info(f"🔄 Exécution de l'ordre de sortie pour le cycle {cycle_id}")
            execution = self.binance_executor.execute_order(exit_order)
            
            # Calculer le profit/perte
            entry_value = cycle.entry_price * cycle.quantity
            exit_value = execution.price * execution.quantity
            
            if exit_side == OrderSide.SELL:
                # Si on vend, profit = sortie - entrée
                profit_loss = exit_value - entry_value
            else:
                # Si on achète (pour clôturer une vente), profit = entrée - sortie
                profit_loss = entry_value - exit_value
            
            # Calculer le pourcentage de profit/perte
            profit_loss_percent = (profit_loss / entry_value) * 100
            
            # Mettre à jour le cycle
            with self.cycles_lock:
                cycle.exit_order_id = execution.order_id
                cycle.exit_price = execution.price
                cycle.status = CycleStatus.COMPLETED
                cycle.profit_loss = profit_loss
                cycle.profit_loss_percent = profit_loss_percent
                cycle.completed_at = datetime.now()
                cycle.updated_at = datetime.now()
            
            # Enregistrer l'exécution et le cycle mis à jour
            self.repository.save_execution(execution, cycle_id)
            self.repository.save_cycle(cycle)
            
            # Publier sur Redis
            self._publish_cycle_event(cycle, "closed")
            
            # Supprimer le cycle des cycles actifs
            with self.cycles_lock:
                self.active_cycles.pop(cycle_id, None)
            
            logger.info(f"✅ Cycle {cycle_id} fermé avec succès: P&L = {profit_loss:.2f} ({profit_loss_percent:.2f}%)")
            return True
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la fermeture du cycle {cycle_id}: {str(e)}")
            return False
    
    def cancel_cycle(self, cycle_id: str, reason: str = "Annulation manuelle") -> bool:
        """
        Annule un cycle de trading.
        Si un ordre est actif, il est annulé sur Binance.
        
        Args:
            cycle_id: ID du cycle à annuler
            reason: Raison de l'annulation
            
        Returns:
            True si l'annulation a réussi, False sinon
        """
        try:
            # Récupérer le cycle
            with self.cycles_lock:
                if cycle_id not in self.active_cycles:
                    logger.warning(f"⚠️ Cycle {cycle_id} non trouvé dans les cycles actifs")
                    return False
                
                cycle = self.active_cycles[cycle_id]
            
            # Vérifier si des ordres doivent être annulés
            if cycle.status in [CycleStatus.ACTIVE_BUY, CycleStatus.ACTIVE_SELL]:
                # Déterminer l'ordre à annuler
                order_id = cycle.entry_order_id if cycle.exit_order_id is None else cycle.exit_order_id
                
                if order_id:
                    # Annuler l'ordre sur Binance
                    logger.info(f"🔄 Annulation de l'ordre {order_id} pour le cycle {cycle_id}")
                    self.binance_executor.cancel_order(cycle.symbol, order_id)
            
            # Mettre à jour le cycle
            with self.cycles_lock:
                cycle.status = CycleStatus.CANCELED
                cycle.updated_at = datetime.now()
                # Ajouter la raison comme commentaire
                if not hasattr(cycle, 'metadata'):
                    cycle.metadata = {}
                cycle.metadata['cancel_reason'] = reason
            
            # Enregistrer le cycle mis à jour
            self.repository.save_cycle(cycle)
            
            # Publier sur Redis
            self._publish_cycle_event(cycle, "canceled")
            
            # Supprimer le cycle des cycles actifs
            with self.cycles_lock:
                self.active_cycles.pop(cycle_id, None)
            
            logger.info(f"✅ Cycle {cycle_id} annulé: {reason}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'annulation du cycle {cycle_id}: {str(e)}")
            return False
    
    def update_stop_loss(self, cycle_id: str, new_stop_price: float) -> bool:
        """
        Met à jour le stop-loss d'un cycle.
        
        Args:
            cycle_id: ID du cycle
            new_stop_price: Nouveau prix de stop-loss
            
        Returns:
            True si la mise à jour a réussi, False sinon
        """
        try:
            # Récupérer le cycle
            with self.cycles_lock:
                if cycle_id not in self.active_cycles:
                    logger.warning(f"⚠️ Cycle {cycle_id} non trouvé dans les cycles actifs")
                    return False
                
                cycle = self.active_cycles[cycle_id]
                
                # Mettre à jour le stop-loss
                cycle.stop_price = new_stop_price
                cycle.updated_at = datetime.now()
            
            # Enregistrer le cycle mis à jour
            self.repository.save_cycle(cycle)
            
            logger.info(f"✅ Stop-loss mis à jour pour le cycle {cycle_id}: {new_stop_price}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la mise à jour du stop-loss pour le cycle {cycle_id}: {str(e)}")
            return False
    
    def get_cycle(self, cycle_id: str) -> Optional[TradeCycle]:
        """
        Récupère un cycle par son ID.
        
        Args:
            cycle_id: ID du cycle
            
        Returns:
            Cycle ou None si non trouvé
        """
        with self.cycles_lock:
            return self.active_cycles.get(cycle_id)
    
    def get_active_cycles(self, symbol: Optional[str] = None, strategy: Optional[str] = None) -> List[TradeCycle]:
        """
        Récupère les cycles actifs, avec filtrage optionnel.
        
        Args:
            symbol: Filtrer par symbole (optionnel)
            strategy: Filtrer par stratégie (optionnel)
            
        Returns:
            Liste des cycles actifs filtrés
        """
        with self.cycles_lock:
            cycles = list(self.active_cycles.values())
        
        # Filtrer par symbole
        if symbol:
            cycles = [c for c in cycles if c.symbol == symbol]
        
        # Filtrer par stratégie
        if strategy:
            cycles = [c for c in cycles if c.strategy == strategy]
        
        return cycles
    
    def process_price_update(self, symbol: str, price: float) -> None:
        """
        Traite une mise à jour de prix pour un symbole.
        Délègue au StopManager pour gérer les stops/targets.
        
        Args:
            symbol: Symbole mis à jour
            price: Nouveau prix
        """
        # Déléguer au StopManager
        self.stop_manager.process_price_update(symbol, price, self.close_cycle)
    
    def close(self) -> None:
        """
        Ferme proprement le gestionnaire de cycles.
        """
        logger.info("Fermeture du gestionnaire de cycles...")
        logger.info("✅ Gestionnaire de cycles fermé")