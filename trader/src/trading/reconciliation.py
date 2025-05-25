"""
Module de réconciliation pour synchroniser l'état des cycles avec Binance.
Vérifie périodiquement l'état des ordres sur Binance et met à jour les cycles.
"""
import logging
import time
import threading
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from shared.src.enums import CycleStatus, OrderStatus, OrderSide

# Helper pour la conversion robuste des statuts de cycle
def parse_cycle_status(status_str):
    """Convertit une chaîne de statut de cycle en énumération CycleStatus de manière robuste."""
    if isinstance(status_str, str):
        # Tenter de convertir directement via l'énumération
        try:
            return CycleStatus(status_str)
        except (KeyError, ValueError):
            # Mapping de fallback pour gérer les différences de casse
            mapping = {s.value.lower(): s for s in CycleStatus}
            return mapping.get(status_str.lower(), CycleStatus.FAILED)
    return status_str  # Si c'est déjà une énumération
from shared.src.schemas import TradeCycle
from trader.src.exchange.binance_executor import BinanceExecutor
from trader.src.trading.cycle_repository import CycleRepository

# Configuration du logging
logger = logging.getLogger(__name__)

class ExchangeReconciliation:
    """
    Classe qui gère la réconciliation périodique avec l'exchange.
    Compare l'état des cycles en base de données avec l'état réel sur Binance.
    """
    
    def __init__(self, cycle_repository: CycleRepository, binance_executor: BinanceExecutor, 
                 reconciliation_interval: int = 600, cycle_manager=None):
        """
        Initialise le service de réconciliation.
        
        Args:
            cycle_repository: Référentiel de cycles de trading
            binance_executor: Exécuteur Binance pour vérifier l'état des ordres
            reconciliation_interval: Intervalle entre les réconciliations en secondes (défaut: 10 minutes)
            cycle_manager: Gestionnaire de cycles pour mettre à jour le cache mémoire
        """
        self.repository = cycle_repository
        self.binance_executor = binance_executor
        self.cycle_manager = cycle_manager
        self.reconciliation_interval = reconciliation_interval
        self.running = False
        self.reconciliation_thread = None
        
        # Statistiques
        self.last_reconciliation = None
        self.stats = {
            "cycles_checked": 0,
            "cycles_reconciled": 0,
            "cycles_failed": 0,
            "last_run_duration": 0
        }
        
        logger.info(f"✅ Service de réconciliation initialisé (intervalle: {reconciliation_interval}s)")
    
    def start(self):
        """
        Démarre le thread de réconciliation périodique.
        """
        if self.running:
            logger.warning("Le service de réconciliation est déjà en cours d'exécution")
            return
            
        self.running = True
        self.reconciliation_thread = threading.Thread(
            target=self._reconciliation_loop,
            daemon=True,
            name="ExchangeReconciliation"
        )
        self.reconciliation_thread.start()
        
        logger.info("✅ Service de réconciliation démarré")
    
    def stop(self):
        """
        Arrête le thread de réconciliation.
        """
        if not self.running:
            return
            
        self.running = False
        
        if self.reconciliation_thread and self.reconciliation_thread.is_alive():
            self.reconciliation_thread.join(timeout=5.0)
            
        logger.info("✅ Service de réconciliation arrêté")
    
    def _reconciliation_loop(self):
        """
        Boucle principale de réconciliation périodique.
        """
        while self.running:
            try:
                # Effectuer une réconciliation
                self.reconcile_all_cycles()
                
                # Mettre à jour le timestamp de dernière réconciliation
                self.last_reconciliation = datetime.now()
                
                # Attendre l'intervalle configuré
                time.sleep(self.reconciliation_interval)
                
            except Exception as e:
                logger.error(f"❌ Erreur dans la boucle de réconciliation: {str(e)}")
                # Pause pour éviter une boucle d'erreur infinie
                time.sleep(60)
    
    def reconcile_all_cycles(self, force: bool = False):
        """
        Réconcilie tous les cycles actifs avec l'état sur Binance.
        
        Args:
            force: Si True, force la réconciliation même pour les cycles récemment mis à jour
        """
        start_time = time.time()
        
        try:
            # Récupérer tous les cycles actifs
            active_cycles = self.repository.get_active_cycles()
            logger.info(f"🔄 Début de la réconciliation pour {len(active_cycles)} cycles actifs")
            
            # Réinitialiser les statistiques pour cette exécution
            cycles_checked = 0
            cycles_reconciled = 0
            cycles_failed = 0
            
            # Traiter chaque cycle actif
            for cycle in active_cycles:
                try:
                    cycles_checked += 1
                    
                    # Vérifier si la réconciliation est nécessaire
                    if not force and self._is_recent_update(cycle):
                        continue
                    
                    # Réconcilier ce cycle
                    if self.reconcile_cycle(cycle):
                        cycles_reconciled += 1
                except Exception as e:
                    import traceback
                    logger.error(f"❌ Erreur lors de la réconciliation du cycle {cycle.id}: {str(e)}")
                    logger.error(f"📋 Stack trace: {traceback.format_exc()}")
                    logger.error(f"🔍 Cycle status type: {type(cycle.status)}, value: {cycle.status}")
                    cycles_failed += 1
            
            # Mettre à jour les statistiques
            self.stats["cycles_checked"] = cycles_checked
            self.stats["cycles_reconciled"] = cycles_reconciled
            self.stats["cycles_failed"] = cycles_failed
            self.stats["last_run_duration"] = time.time() - start_time
            
            logger.info(f"✅ Réconciliation terminée: {cycles_reconciled}/{cycles_checked} cycles mis à jour ({cycles_failed} échecs)")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la réconciliation des cycles: {str(e)}")
            self.stats["last_run_duration"] = time.time() - start_time
    
    def _is_recent_update(self, cycle: TradeCycle, threshold_minutes: int = 5) -> bool:
        """
        Vérifie si un cycle a été mis à jour récemment.
        
        Args:
            cycle: Cycle à vérifier
            threshold_minutes: Seuil en minutes considéré comme "récent"
            
        Returns:
            True si le cycle a été mis à jour récemment, False sinon
        """
        if not cycle.updated_at:
            return False
            
        threshold = datetime.now() - timedelta(minutes=threshold_minutes)
        return cycle.updated_at > threshold
    
    def reconcile_cycle(self, cycle: TradeCycle) -> bool:
        """
        Réconcilie un cycle avec son état sur Binance.
        
        Args:
            cycle: Cycle à réconcilier
            
        Returns:
            True si le cycle a été mis à jour, False sinon
        """
        # Si le cycle n'a pas d'ordre d'entrée, le marquer comme annulé
        if not cycle.entry_order_id:
            logger.warning(f"⚠️ Cycle {cycle.id} sans ordre d'entrée, marqué comme annulé")
            cycle.status = CycleStatus.CANCELED
            cycle.updated_at = datetime.now()
            self.repository.save_cycle(cycle)
            return True
        
        # Vérifier l'état de l'ordre d'entrée sur Binance
        entry_execution = self.binance_executor.get_order_status(cycle.symbol, cycle.entry_order_id)
        
        # Si l'ordre d'entrée n'existe pas sur Binance, le marquer comme annulé
        if not entry_execution and not self.binance_executor.demo_mode:
            logger.warning(f"⚠️ Ordre d'entrée {cycle.entry_order_id} non trouvé sur Binance, cycle {cycle.id} marqué comme annulé")
            cycle.status = CycleStatus.CANCELED
            cycle.updated_at = datetime.now()
            self.repository.save_cycle(cycle)
            return True
        
        # Si le cycle est en attente de vente mais n'a pas d'ordre de sortie, créer l'ordre
        if cycle.status in [CycleStatus.WAITING_SELL, CycleStatus.ACTIVE_SELL, CycleStatus.WAITING_BUY, CycleStatus.ACTIVE_BUY] and not cycle.exit_order_id:
            # Conversion robuste du statut pour l'affichage
            status_display = cycle.status.value if hasattr(cycle.status, 'value') else str(cycle.status)
            logger.warning(f"⚠️ Cycle {cycle.id} en statut {status_display} sans ordre de sortie")
            
            # Vérifier que l'ordre d'entrée est bien FILLED avant de créer l'ordre de sortie
            if entry_execution and entry_execution.status == OrderStatus.FILLED:
                logger.info(f"🔧 Création de l'ordre de sortie manquant pour le cycle {cycle.id}")
                
                # Déterminer le side de l'ordre de sortie
                if cycle.status in [CycleStatus.ACTIVE_BUY, CycleStatus.WAITING_SELL]:
                    exit_side = OrderSide.SELL
                else:
                    exit_side = OrderSide.BUY
                
                # Créer l'ordre de sortie
                try:
                    from shared.src.enums import OrderType
                    from shared.src.schemas import TradeOrder
                    
                    exit_price = cycle.target_price if cycle.target_price else None
                    
                    # Créer un objet TradeOrder
                    exit_order = TradeOrder(
                        symbol=cycle.symbol,
                        side=exit_side,
                        price=exit_price,
                        quantity=cycle.quantity,
                        order_type=OrderType.LIMIT if exit_price else OrderType.MARKET,
                        client_order_id=f"exit_{cycle.id}"
                    )
                    
                    exit_execution = self.binance_executor.execute_order(exit_order)
                    
                    if exit_execution:
                        # Mettre à jour le cycle
                        cycle.exit_order_id = exit_execution.order_id
                        cycle.status = CycleStatus.WAITING_SELL if exit_side == OrderSide.SELL else CycleStatus.WAITING_BUY
                        cycle.updated_at = datetime.now()
                        self.repository.save_cycle(cycle)
                        self.repository.save_execution(exit_execution, cycle.id)
                        
                        logger.info(f"✅ Ordre de sortie {exit_execution.order_id} créé pour le cycle {cycle.id}")
                        return True
                    else:
                        logger.error(f"❌ Échec de création de l'ordre de sortie pour le cycle {cycle.id}")
                        
                except Exception as e:
                    logger.error(f"❌ Erreur lors de la création de l'ordre de sortie: {str(e)}")
            
            # Vérifier depuis combien de temps le cycle est dans cet état
            time_since_update = datetime.now() - cycle.updated_at
            
            # Si le cycle est bloqué depuis plus de 30 minutes, le marquer comme échoué
            if time_since_update.total_seconds() > 1800:  # 30 minutes
                logger.error(f"❌ Cycle {cycle.id} bloqué en {status_display} depuis {time_since_update.total_seconds()/60:.1f} minutes, marqué comme échoué")
                cycle.status = CycleStatus.FAILED
                cycle.updated_at = datetime.now()
                if not hasattr(cycle, 'metadata') or cycle.metadata is None:
                    cycle.metadata = {}
                cycle.metadata['cancel_reason'] = f"Bloqué en {status_display} sans ordre de sortie pendant {time_since_update.total_seconds()/60:.1f} minutes"
                self.repository.save_cycle(cycle)
                
                # Retirer le cycle du cache mémoire
                if self.cycle_manager and hasattr(self.cycle_manager, 'active_cycles'):
                    with self.cycle_manager.cycles_lock:
                        if cycle.id in self.cycle_manager.active_cycles:
                            del self.cycle_manager.active_cycles[cycle.id]
                            logger.info(f"♻️ Cycle {cycle.id} retiré du cache mémoire (FAILED)")
                
                return True
            
            logger.info(f"ℹ️ Cycle {cycle.id} en {status_display} depuis {time_since_update.total_seconds()/60:.1f} minutes")
            return False
        
        # Si le cycle a un ordre de sortie, vérifier son état
        if cycle.exit_order_id:
            exit_execution = self.binance_executor.get_order_status(cycle.symbol, cycle.exit_order_id)
            
            # Si l'ordre de sortie n'existe pas, marquer comme échoué
            if not exit_execution and not self.binance_executor.demo_mode:
                logger.warning(f"⚠️ Ordre de sortie {cycle.exit_order_id} non trouvé sur Binance, cycle {cycle.id} marqué comme échoué")
                cycle.status = CycleStatus.FAILED
                cycle.updated_at = datetime.now()
                self.repository.save_cycle(cycle)
                return True
            
            # Si l'ordre de sortie est rempli, marquer comme terminé
            if exit_execution and exit_execution.status == OrderStatus.FILLED:
                # Conversion robuste du statut pour la comparaison
                status_lower = cycle.status.value.lower() if hasattr(cycle.status, 'value') else str(cycle.status).lower()
                if cycle.status != CycleStatus.COMPLETED and status_lower != 'completed':
                    logger.info(f"✅ Ordre de sortie {cycle.exit_order_id} rempli, cycle {cycle.id} marqué comme terminé")
                    # Utiliser CycleStatus.COMPLETED directement comme enum
                    cycle.status = CycleStatus.COMPLETED
                    cycle.exit_price = exit_execution.price
                    cycle.completed_at = exit_execution.timestamp
                    cycle.updated_at = datetime.now()
                    
                    # Calculer P&L si possible
                    if cycle.entry_price and cycle.exit_price and cycle.quantity:
                        # Déterminer si c'était un cycle d'achat ou de vente initial
                        # On vérifie l'ordre d'entrée pour savoir la direction initiale
                        if entry_execution:
                            entry_side = entry_execution.side
                        else:
                            # Fallback: si l'ordre d'entrée n'est pas trouvé, on suppose BUY
                            # (la plupart des stratégies commencent par acheter)
                            entry_side = OrderSide.BUY
                            logger.warning(f"⚠️ Ordre d'entrée non trouvé pour le cycle {cycle.id}, assumé BUY")
                        
                        # Si entrée = BUY, alors sortie = SELL : profit = (prix_sortie - prix_entrée) * quantité
                        if entry_side == OrderSide.BUY:
                            cycle.profit_loss = (cycle.exit_price - cycle.entry_price) * cycle.quantity
                        # Si entrée = SELL, alors sortie = BUY : profit = (prix_entrée - prix_sortie) * quantité
                        else:
                            cycle.profit_loss = (cycle.entry_price - cycle.exit_price) * cycle.quantity
                        
                        # Calculer le pourcentage de P&L
                        entry_value = cycle.entry_price * cycle.quantity
                        if entry_value > 0:
                            cycle.profit_loss_percent = (cycle.profit_loss / entry_value) * 100
                    
                    self.repository.save_cycle(cycle)
                    
                    # Retirer le cycle du cache mémoire du cycle_manager
                    if self.cycle_manager and hasattr(self.cycle_manager, 'active_cycles'):
                        with self.cycle_manager.cycles_lock:
                            if cycle.id in self.cycle_manager.active_cycles:
                                del self.cycle_manager.active_cycles[cycle.id]
                                logger.info(f"♻️ Cycle {cycle.id} retiré du cache mémoire")
                    
                    return True
        
        # Vérifier si le cycle est actif depuis trop longtemps (> 7 jours)
        cycle_age = datetime.now() - cycle.created_at
        # Convertir le statut en chaîne minuscule pour comparaison insensible à la casse
        status_lower = cycle.status.value.lower() if hasattr(cycle.status, 'value') else str(cycle.status).lower()
        terminal_statuses = ['completed', 'canceled', 'failed']
        if cycle_age.days > 7 and status_lower not in terminal_statuses:
            logger.warning(f"⚠️ Cycle {cycle.id} actif depuis plus de 7 jours ({cycle_age.days}), marqué comme échoué")
            cycle.status = CycleStatus.FAILED
            cycle.updated_at = datetime.now()
            if not hasattr(cycle, 'metadata') or cycle.metadata is None:
                cycle.metadata = {}
            cycle.metadata['cancel_reason'] = "Timeout automatique après 7 jours"
            self.repository.save_cycle(cycle)
            
            # Retirer le cycle du cache mémoire
            if self.cycle_manager and hasattr(self.cycle_manager, 'active_cycles'):
                with self.cycle_manager.cycles_lock:
                    if cycle.id in self.cycle_manager.active_cycles:
                        del self.cycle_manager.active_cycles[cycle.id]
                        logger.info(f"♻️ Cycle {cycle.id} retiré du cache mémoire (timeout 7 jours)")
            
            return True
        
        # Aucune mise à jour nécessaire
        return False
    
    def force_reconciliation(self):
        """
        Force une réconciliation immédiate de tous les cycles.
        """
        logger.info("Forçage de la réconciliation de tous les cycles")
        self.reconcile_all_cycles(force=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Récupère les statistiques de réconciliation.
        
        Returns:
            Statistiques du service de réconciliation
        """
        stats = self.stats.copy()
        stats['last_reconciliation'] = self.last_reconciliation.isoformat() if self.last_reconciliation else None
        return stats