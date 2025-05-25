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
from shared.src.enums import OrderSide, OrderStatus, CycleStatus, OrderType
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

            # Vérifier le solde avant d'acheter (seulement pour les ordres BUY)
            if side == OrderSide.BUY and not self.demo_mode:
                # Extraire la quote currency (USDC, BTC, etc.)
                quote_currency = symbol.replace("BTC", "").replace("ETH", "").replace("BNB", "").replace("SUI", "")
                if not quote_currency:  # Pour les paires comme ETHBTC
                    quote_currency = "BTC" if "BTC" in symbol and symbol != "BTCUSDC" else "USDC"
                
                # Récupérer les soldes actuels
                balances = self.binance_executor.utils.fetch_account_balances(self.binance_executor.time_offset)
                available_balance = balances.get(quote_currency, {}).get('free', 0)
                
                # Calculer le coût total de l'ordre
                total_cost = price * quantity * 1.001  # Ajouter 0.1% pour les frais
                
                # Vérifier si le solde est suffisant
                if available_balance < total_cost:
                    logger.error(f"❌ Solde {quote_currency} insuffisant: {available_balance:.2f} < {total_cost:.2f} requis")
                    
                    # Créer le cycle avec un statut FAILED pour la traçabilité
                    cycle.status = CycleStatus.FAILED
                    cycle.updated_at = datetime.now()
                    if not hasattr(cycle, 'metadata'):
                        cycle.metadata = {}
                    cycle.metadata['fail_reason'] = f"Solde {quote_currency} insuffisant: {available_balance:.2f} < {total_cost:.2f}"
                    
                    # Sauvegarder le cycle échoué pour la traçabilité
                    self.repository.save_cycle(cycle)
                    
                    # Publier l'événement d'échec
                    self._publish_cycle_event(cycle, "failed")
                    
                    # Proposer une quantité ajustée si possible
                    adjusted_quantity = (available_balance * 0.99) / price  # 99% du solde pour garder une marge
                    min_quantity = self.binance_executor.symbol_constraints.get_min_qty(symbol)
                    
                    if adjusted_quantity >= min_quantity:
                        logger.info(f"💡 Quantité ajustée suggérée: {adjusted_quantity:.8f} {symbol.replace(quote_currency, '')}")
                    
                    return None
            
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
            
            try:
                execution = self.binance_executor.execute_order(entry_order)
                
                # Vérifier si l'exécution a réussi
                if not execution or not execution.order_id:
                    logger.error(f"❌ L'ordre d'entrée pour le cycle {cycle_id} a échoué - pas d'ID d'ordre")
                    
                    # Créer le cycle avec un statut FAILED pour la traçabilité
                    cycle.status = CycleStatus.FAILED
                    cycle.updated_at = datetime.now()
                    if not hasattr(cycle, 'metadata'):
                        cycle.metadata = {}
                    cycle.metadata['fail_reason'] = "Ordre d'entrée échoué - pas d'ID"
                    
                    # Sauvegarder le cycle échoué pour la traçabilité
                    self.repository.save_cycle(cycle)
                    
                    # Publier l'événement d'échec
                    self._publish_cycle_event(cycle, "failed")
                    
                    return None
                
                # Pour les ordres LIMIT, le statut initial est NEW, pas FILLED
                # On accepte NEW, PARTIALLY_FILLED et FILLED comme statuts valides
                valid_statuses = [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED]
                if execution.status not in valid_statuses:
                    logger.error(f"❌ L'ordre d'entrée pour le cycle {cycle_id} a un statut invalide: {execution.status}")
                    
                    # Créer le cycle avec un statut FAILED pour la traçabilité
                    cycle.status = CycleStatus.FAILED
                    cycle.updated_at = datetime.now()
                    if not hasattr(cycle, 'metadata'):
                        cycle.metadata = {}
                    cycle.metadata['fail_reason'] = f"Statut d'ordre invalide: {execution.status}"
                    
                    # Sauvegarder le cycle échoué pour la traçabilité
                    self.repository.save_cycle(cycle)
                    
                    # Publier l'événement d'échec
                    self._publish_cycle_event(cycle, "failed")
                    
                    return None
                    
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'exécution de l'ordre: {str(e)}")
                
                # Créer le cycle avec un statut FAILED pour la traçabilité
                cycle.status = CycleStatus.FAILED
                cycle.updated_at = datetime.now()
                if not hasattr(cycle, 'metadata'):
                    cycle.metadata = {}
                    
                # Analyser le message d'erreur pour identifier les problèmes de fonds
                error_msg = str(e).lower()
                
                # Vérifier spécifiquement les erreurs de solde insuffisant
                if "insufficient balance" in error_msg or "account has insufficient balance" in error_msg:
                    cycle.metadata['fail_reason'] = "Solde insuffisant"
                    logger.warning(f"⚠️ Solde insuffisant pour {symbol}: {error_msg}")
                else:
                    cycle.metadata['fail_reason'] = f"Erreur d'exécution: {str(e)}"
                if "insufficient balance" in error_msg or "insufficient funds" in error_msg:
                    cycle.metadata['fail_reason'] = "Fonds insuffisants"
                    logger.error(f"💰 Fonds insuffisants pour créer le cycle {cycle_id}")
                else:
                    cycle.metadata['fail_reason'] = str(e)
                
                # Sauvegarder le cycle échoué
                self.repository.save_cycle(cycle)
                
                # Publier l'événement d'échec
                self._publish_cycle_event(cycle, "failed")
                
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
            
            # Créer automatiquement l'ordre de sortie si un prix cible existe
            # On ne attend plus que l'ordre soit FILLED
            if cycle.target_price:
                logger.info(f"🎯 Création immédiate de l'ordre de sortie (target: {cycle.target_price})")
                self._create_exit_order_for_cycle(cycle, initial_side=side)
            else:
                logger.warning(f"⚠️ Pas de prix cible défini pour le cycle {cycle_id}, ordre de sortie non créé")
            
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
    
    def close_cycle(self, cycle_id: str, exit_price: Optional[float] = None, is_stop_loss: bool = False) -> bool:
        """
        Ferme un cycle de trading en exécutant l'ordre de sortie.
        
        Args:
            cycle_id: ID du cycle à fermer
            exit_price: Prix de sortie (optionnel, sinon au marché)
            is_stop_loss: Si True, indique que c'est un stop loss qui se déclenche
            
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
            
            # Si c'est un stop loss et qu'il y a un ordre de sortie existant, l'annuler
            if is_stop_loss and cycle.exit_order_id:
                try:
                    logger.info(f"🛑 Stop loss déclenché - Annulation de l'ordre limite {cycle.exit_order_id}")
                    cancel_result = self.binance_executor.cancel_order(cycle.symbol, cycle.exit_order_id)
                    if cancel_result:
                        logger.info(f"✅ Ordre limite {cycle.exit_order_id} annulé avec succès")
                    else:
                        logger.warning(f"⚠️ L'ordre {cycle.exit_order_id} n'a pas pu être annulé")
                except Exception as e:
                    logger.warning(f"⚠️ Erreur lors de l'annulation de l'ordre {cycle.exit_order_id}: {str(e)}")
                    # Continuer même si l'annulation échoue
            
            # Créer l'ordre de sortie
            exit_order = TradeOrder(
                symbol=cycle.symbol,
                side=exit_side,
                quantity=cycle.quantity,
                price=exit_price,  # None pour un ordre au marché si stop loss
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
            
            # Annuler TOUS les ordres associés au cycle
            orders_to_cancel = []
            
            # Ajouter l'ordre d'entrée s'il existe et n'est pas FILLED
            if cycle.entry_order_id:
                entry_status = self.binance_executor.get_order_status(cycle.symbol, cycle.entry_order_id)
                if entry_status and entry_status.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                    orders_to_cancel.append(('entrée', cycle.entry_order_id))
            
            # Ajouter l'ordre de sortie s'il existe
            if cycle.exit_order_id:
                exit_status = self.binance_executor.get_order_status(cycle.symbol, cycle.exit_order_id)
                if exit_status and exit_status.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                    orders_to_cancel.append(('sortie', cycle.exit_order_id))
            
            # Annuler tous les ordres trouvés
            for order_type, order_id in orders_to_cancel:
                try:
                    logger.info(f"🔄 Annulation de l'ordre de {order_type} {order_id} pour le cycle {cycle_id}")
                    cancel_result = self.binance_executor.cancel_order(cycle.symbol, order_id)
                    if cancel_result:
                        logger.info(f"✅ Ordre {order_id} annulé avec succès")
                    else:
                        logger.warning(f"⚠️ Impossible d'annuler l'ordre {order_id}")
                except Exception as e:
                    logger.warning(f"⚠️ Erreur lors de l'annulation de l'ordre {order_id}: {str(e)}")
            
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
        # Créer un wrapper pour close_cycle qui indique que c'est un stop
        def close_cycle_by_stop(cycle_id: str, exit_price: Optional[float] = None) -> bool:
            return self.close_cycle(cycle_id, exit_price, is_stop_loss=True)
        
        # Déléguer au StopManager avec le wrapper
        self.stop_manager.process_price_update(symbol, price, close_cycle_by_stop)
    
    def _create_exit_order_for_cycle(self, cycle: TradeCycle, initial_side: Optional[OrderSide] = None) -> bool:
        """
        Crée automatiquement un ordre de sortie pour un cycle.
        
        Args:
            cycle: Le cycle pour lequel créer l'ordre de sortie
            initial_side: Le côté initial de l'ordre d'entrée (optionnel)
            
        Returns:
            True si l'ordre de sortie a été créé avec succès
        """
        try:
            # Déterminer le side de l'ordre de sortie (inverse de l'entrée)
            if initial_side:
                # Si on connait le côté initial, on prend son inverse
                exit_side = OrderSide.SELL if initial_side == OrderSide.BUY else OrderSide.BUY
            else:
                # Sinon, on se base sur le statut du cycle
                if cycle.status in [CycleStatus.ACTIVE_BUY, CycleStatus.WAITING_BUY]:
                    exit_side = OrderSide.SELL
                elif cycle.status in [CycleStatus.ACTIVE_SELL, CycleStatus.WAITING_SELL]:
                    exit_side = OrderSide.BUY
                else:
                    logger.warning(f"⚠️ Statut inattendu pour créer un ordre de sortie: {cycle.status}")
                    return False
            
            # Utiliser le prix target s'il existe, sinon créer un ordre au marché
            exit_price = cycle.target_price if cycle.target_price else None
            
            logger.info(f"🎯 Création de l'ordre de sortie pour le cycle {cycle.id}")
            logger.info(f"   Side: {exit_side.value}, Prix: {exit_price or 'MARKET'}, Quantité: {cycle.quantity}")
            
            # Créer l'ordre de sortie
            from shared.src.schemas import TradeOrder
            
            exit_order = TradeOrder(
                symbol=cycle.symbol,
                side=exit_side,
                price=exit_price,
                quantity=cycle.quantity,
                order_type=OrderType.LIMIT if exit_price else OrderType.MARKET,
                client_order_id=f"exit_{cycle.id}"
            )
            
            execution = self.binance_executor.execute_order(exit_order)
            
            if execution:
                # Mettre à jour le cycle avec l'ID de l'ordre de sortie
                with self.cycles_lock:
                    cycle.exit_order_id = execution.order_id
                    cycle.status = CycleStatus.WAITING_SELL if exit_side == OrderSide.SELL else CycleStatus.WAITING_BUY
                    cycle.updated_at = datetime.now()
                
                # Sauvegarder les changements
                self.repository.save_execution(execution, cycle.id)
                self.repository.save_cycle(cycle)
                
                logger.info(f"✅ Ordre de sortie créé pour le cycle {cycle.id}: {execution.order_id}")
                return True
            else:
                logger.error(f"❌ Échec de création de l'ordre de sortie pour le cycle {cycle.id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création de l'ordre de sortie: {str(e)}")
            return False
    
    def close(self) -> None:
        """
        Ferme proprement le gestionnaire de cycles.
        """
        logger.info("Fermeture du gestionnaire de cycles...")
        logger.info("✅ Gestionnaire de cycles fermé")