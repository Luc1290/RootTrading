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
                 reconciliation_interval: int = 300, cycle_manager=None):
        """
        Initialise le service de réconciliation.
        
        Args:
            cycle_repository: Référentiel de cycles de trading
            binance_executor: Exécuteur Binance pour vérifier l'état des ordres
            reconciliation_interval: Intervalle entre les réconciliations en secondes (défaut: 5 minutes)
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
            # Log uniquement si debug activé ou s'il y a beaucoup de cycles
            if logger.isEnabledFor(logging.DEBUG) or len(active_cycles) > 5:
                logger.debug(f"🔄 Début de la réconciliation pour {len(active_cycles)} cycles actifs")
            
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
            
            # Nettoyer les ordres orphelins après la réconciliation des cycles
            orphans_cleaned = self._clean_orphan_orders()
            
            # Mettre à jour les statistiques
            self.stats["cycles_checked"] = cycles_checked
            self.stats["cycles_reconciled"] = cycles_reconciled
            self.stats["cycles_failed"] = cycles_failed
            self.stats["orphan_orders_cleaned"] = orphans_cleaned
            self.stats["last_run_duration"] = time.time() - start_time
            
            # Ne logger en INFO que s'il y a eu des changements significatifs
            if cycles_reconciled > 0 or cycles_failed > 0 or orphans_cleaned > 0:
                logger.info(f"✅ Réconciliation terminée: {cycles_reconciled}/{cycles_checked} cycles mis à jour ({cycles_failed} échecs, {orphans_cleaned} ordres orphelins nettoyés)")
            # else:
            #     logger.debug(f"✅ Réconciliation terminée: {cycles_reconciled}/{cycles_checked} cycles mis à jour")
            
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
        # Si le cycle n'a pas d'ordre d'entrée, le marquer comme échoué
        if not cycle.entry_order_id:
            logger.error(f"❌ Cycle {cycle.id} sans ordre d'entrée, marqué comme FAILED")
            cycle.status = CycleStatus.FAILED
            cycle.updated_at = datetime.now()
            if not hasattr(cycle, 'metadata'):
                cycle.metadata = {}
            cycle.metadata['fail_reason'] = "Pas d'ordre d'entrée"
            self.repository.save_cycle(cycle)
            return True
        
        # Ignorer uniquement les cycles marqués comme démo
        if cycle.demo:
            logger.debug(f"⏭️ Ignorer la réconciliation du cycle démo {cycle.id}")
            return False
        
        # Vérifier l'état de l'ordre d'entrée sur Binance
        entry_execution = self.binance_executor.get_order_status(cycle.symbol, cycle.entry_order_id)
        
        # CORRECTION: Ne pas marquer FAILED si l'ordre d'entrée a déjà été exécuté avec succès
        if not entry_execution and not self.binance_executor.demo_mode:
            # Vérifier si le cycle a déjà un prix d'entrée (ordre exécuté avec succès)
            if cycle.entry_price and cycle.entry_price > 0:
                logger.info(f"✅ Ordre d'entrée {cycle.entry_order_id} déjà exécuté (prix: {cycle.entry_price}), cycle {cycle.id} valide")
                # L'ordre a été exécuté avec succès, continuer normalement
                return False
            
            # Vérifier l'historique des trades pour confirmer l'exécution
            try:
                trades = self.binance_executor.utils.get_my_trades(cycle.symbol, limit=100)
                order_executed = any(trade.get('orderId') == int(cycle.entry_order_id) for trade in trades if trade.get('orderId'))
                if order_executed:
                    logger.info(f"✅ Ordre d'entrée {cycle.entry_order_id} trouvé dans l'historique des trades, cycle {cycle.id} valide")
                    return False
            except Exception as e:
                logger.warning(f"⚠️ Impossible de vérifier l'historique des trades: {e}")
            
            # Seulement maintenant, marquer comme échoué si aucune preuve d'exécution
            logger.error(f"❌ Ordre d'entrée {cycle.entry_order_id} non trouvé sur Binance et pas de preuve d'exécution, cycle {cycle.id} marqué comme FAILED")
            cycle.status = CycleStatus.FAILED
            cycle.updated_at = datetime.now()
            if not hasattr(cycle, 'metadata'):
                cycle.metadata = {}
            cycle.metadata['fail_reason'] = "Ordre d'entrée non trouvé sur Binance et pas de preuve d'exécution"
            self.repository.save_cycle(cycle)
            return True
        
        # Dans le nouveau mode, les cycles n'ont plus d'exit orders sur Binance (gérés par StopManager)
        # Les cycles actifs sont normaux sans exit_order_id
        
        # Si le cycle a un ordre de sortie et n'est pas déjà terminé, vérifier son état
        if cycle.exit_order_id and cycle.status not in [CycleStatus.COMPLETED, CycleStatus.CANCELED, CycleStatus.FAILED]:
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
                        # On vérifie l'ordre d'entrée pour savoir le side initiale
                        if entry_execution:
                            entry_side = entry_execution.side
                        else:
                            # Fallback: utiliser le side du cycle si l'ordre d'entrée n'est pas trouvé
                            entry_side = cycle.side
                            logger.warning(f"⚠️ Ordre d'entrée non trouvé pour le cycle {cycle.id}, utilisant le side du cycle: {cycle.side.value}")

                        # IMPORTANT: Utiliser la quantité réellement exécutée
                        actual_quantity = cycle.metadata.get('executed_quantity', cycle.quantity) if cycle.metadata else cycle.quantity

                        # Si entrée = BUY, alors sortie = SELL : profit = (prix_sortie - prix_entrée) * quantité
                        if entry_side == OrderSide.BUY:
                            cycle.profit_loss = (cycle.exit_price - cycle.entry_price) * actual_quantity
                        # Si entrée = SELL, alors sortie = BUY : profit = (prix_entrée - prix_sortie) * quantité
                        else:
                            cycle.profit_loss = (cycle.entry_price - cycle.exit_price) * actual_quantity
                        
                        # Calculer le pourcentage de P&L
                        entry_value = cycle.entry_price * actual_quantity
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
    
    def _clean_orphan_orders(self) -> int:
        """
        Détecte et annule les ordres orphelins sur Binance.
        Un ordre est considéré orphelin si:
        - Il existe sur Binance mais le cycle correspondant est terminé/annulé
        - C'est un ordre de sortie d'un cycle qui n'existe plus
        
        Returns:
            Nombre d'ordres orphelins nettoyés
        """
        orphans_cleaned = 0
        
        try:
            # Recherche silencieuse des ordres orphelins
            
            # Récupérer tous les cycles récents (dernières 48h) qui sont terminés/annulés
            # Exclure les cycles démo pour éviter de vérifier leurs ordres sur Binance
            query = """
                SELECT id, symbol, entry_order_id, exit_order_id, status
                FROM trade_cycles 
                WHERE status IN ('completed', 'canceled', 'failed')
                AND created_at > NOW() - INTERVAL '48 hours'
                AND exit_order_id IS NOT NULL
                AND demo = false
            """
            
            completed_cycles = []
            # Utiliser DBContextManager pour accéder à la DB
            from shared.src.db_pool import DBContextManager
            
            with DBContextManager() as cursor:
                cursor.execute(query)
                for row in cursor.fetchall():
                    completed_cycles.append({
                        'id': row[0],
                        'symbol': row[1],
                        'entry_order_id': row[2],
                        'exit_order_id': row[3],
                        'status': row[4]
                    })
            
            # Ne logger que s'il y a potentiellement des orphelins
            if len(completed_cycles) > 0 and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"📊 {len(completed_cycles)} cycles terminés avec ordre de sortie trouvés")
            
            # Pour chaque cycle terminé, vérifier si l'ordre de sortie est toujours ouvert
            for cycle_data in completed_cycles:
                try:
                    if not cycle_data['exit_order_id']:
                        continue
                    
                    # Ne plus vérifier l'ID numérique car les vrais ordres Binance peuvent avoir des IDs élevés
                    
                    # Vérifier l'état de l'ordre de sortie sur Binance
                    exit_status = self.binance_executor.get_order_status(
                        cycle_data['symbol'], 
                        cycle_data['exit_order_id']
                    )
                    
                    # Si get_order_status retourne None (ordre démo ou erreur), passer au suivant
                    if not exit_status:
                        continue
                    
                    # Si l'ordre est toujours ouvert (NEW ou PARTIALLY_FILLED)
                    if exit_status and exit_status.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                        logger.warning(f"🚨 Ordre orphelin détecté: {cycle_data['exit_order_id']} "
                                     f"pour le cycle {cycle_data['id']} ({cycle_data['status']})")
                        
                        # Annuler l'ordre
                        try:
                            cancel_result = self.binance_executor.cancel_order(
                                cycle_data['symbol'], 
                                cycle_data['exit_order_id']
                            )
                            
                            if cancel_result:
                                logger.info(f"✅ Ordre orphelin {cycle_data['exit_order_id']} annulé avec succès")
                                orphans_cleaned += 1
                            else:
                                logger.warning(f"⚠️ Impossible d'annuler l'ordre {cycle_data['exit_order_id']}")
                        
                        except Exception as e:
                            logger.error(f"❌ Erreur lors de l'annulation de l'ordre orphelin {cycle_data['exit_order_id']}: {str(e)}")
                
                except Exception as e:
                    logger.error(f"❌ Erreur lors de la vérification du cycle {cycle_data['id']}: {str(e)}")
                    continue
            
            if orphans_cleaned > 0:
                logger.info(f"🧹 {orphans_cleaned} ordres orphelins nettoyés")
            else:
                # Pas de log pour les cas normaux sans orphelins
                pass
                
        except Exception as e:
            logger.error(f"❌ Erreur lors du nettoyage des ordres orphelins: {str(e)}")
        
        return orphans_cleaned