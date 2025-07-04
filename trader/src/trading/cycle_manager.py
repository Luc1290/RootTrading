# trader/src/trading/cycle_manager.py
"""
Gestionnaire des cycles de trading.
Version simplifiée qui délègue à d'autres modules.
"""
import logging
import time
import uuid
import os
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timezone
import threading
from threading import RLock

from shared.src.config import get_db_url, TRADING_MODE
from shared.src.enums import OrderSide, OrderStatus, CycleStatus, OrderType
from shared.src.schemas import TradeOrder, TradeExecution, TradeCycle
from shared.src.db_pool import DBContextManager, DBConnectionPool, transaction

from trader.src.exchange.binance_executor import BinanceExecutor
from trader.src.trading.cycle_repository import CycleRepository
from trader.src.trading.stop_manager_pure import StopManagerPure

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
        self.stop_manager = StopManagerPure(self.repository)
        
        # Dictionnaire des cycles actifs {id_cycle: cycle}
        self.active_cycles: Dict[str, TradeCycle] = {}
        
        # Mutex pour l'accès concurrent aux cycles
        self.cycles_lock = RLock()
        
        # Initialiser le pool de connexions DB
        try:
            self.db_pool = DBConnectionPool.get_instance()
            self._load_active_cycles_from_db()
            # Vérifier et nettoyer les cycles au démarrage
            logger.info("🧼 Vérification des cycles au démarrage...")
            self._verify_cycles_on_startup()
            # Démarrer le thread de nettoyage périodique
            self._start_cleanup_thread()
            # Démarrer le thread de synchronisation DB périodique
            self._start_sync_thread()
            # Démarrer le thread de nettoyage des ordres orphelins
            self._start_orphan_cleanup_thread()
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

    def _start_sync_thread(self):
        """Démarre un thread de synchronisation périodique avec la DB."""
        def sync_routine():
            while True:
                try:
                    # Synchroniser toutes les 30 secondes
                    time.sleep(30)
                    self._sync_cycles_with_db()
                except Exception as e:
                    logger.error(f"❌ Erreur dans le thread de synchronisation: {str(e)}")
        
        sync_thread = threading.Thread(target=sync_routine, daemon=True, name="CycleSyncThread")
        sync_thread.start()
        logger.info("🔄 Thread de synchronisation DB démarré (30s)")

    def _sync_cycles_with_db(self):
        """Synchronise les cycles en mémoire avec la base de données."""
        try:
            # Récupérer tous les cycles actifs depuis la DB
            db_cycles = self.repository.get_active_cycles()
            db_cycle_ids = {cycle.id for cycle in db_cycles}
            
            with self.cycles_lock:
                # 1. Identifier les cycles à supprimer de la mémoire (n'existent plus en DB ou sont terminés)
                memory_cycle_ids = set(self.active_cycles.keys())
                cycles_to_remove = memory_cycle_ids - db_cycle_ids
                
                # 2. Identifier les nouveaux cycles à ajouter (existent en DB mais pas en mémoire)
                cycles_to_add = []
                for cycle in db_cycles:
                    if cycle.id not in self.active_cycles:
                        cycles_to_add.append(cycle)
                
                # 3. Supprimer les cycles obsolètes
                if cycles_to_remove:
                    for cycle_id in cycles_to_remove:
                        del self.active_cycles[cycle_id]
                    logger.info(f"🗑️ {len(cycles_to_remove)} cycles supprimés de la mémoire (plus en DB)")
                
                # 4. Ajouter les nouveaux cycles
                if cycles_to_add:
                    for cycle in cycles_to_add:
                        self.active_cycles[cycle.id] = cycle
                    logger.info(f"➕ {len(cycles_to_add)} nouveaux cycles ajoutés depuis la DB")
                
                # 5. Mettre à jour les statuts des cycles existants
                updated_count = 0
                for cycle in db_cycles:
                    if cycle.id in self.active_cycles:
                        mem_cycle = self.active_cycles[cycle.id]
                        if mem_cycle.status != cycle.status:
                            # Préserver l'attribut confirmed du cycle en mémoire si il est True
                            if hasattr(mem_cycle, 'confirmed') and mem_cycle.confirmed:
                                cycle.confirmed = mem_cycle.confirmed
                            self.active_cycles[cycle.id] = cycle
                            updated_count += 1
                
                if updated_count > 0:
                    logger.debug(f"🔄 {updated_count} cycles mis à jour depuis la DB")
                
                # Log final
                total_cycles = len(self.active_cycles)
                if cycles_to_remove or cycles_to_add or updated_count > 0:
                    logger.info(f"✅ Synchronisation DB terminée: {total_cycles} cycles actifs en mémoire")
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la synchronisation avec la DB: {str(e)}")

    def _start_orphan_cleanup_thread(self):
        """Démarre un thread de nettoyage périodique des ordres orphelins."""
        def orphan_cleanup_routine():
            # Attendre 2 minutes au démarrage pour laisser le système se stabiliser
            time.sleep(120)
            
            while True:
                try:
                    # Nettoyer les ordres orphelins toutes les 5 minutes
                    self._cleanup_orphan_orders()
                    time.sleep(300)  # 5 minutes
                except Exception as e:
                    logger.error(f"❌ Erreur dans le thread de nettoyage des ordres orphelins: {str(e)}")
                    time.sleep(60)  # En cas d'erreur, attendre 1 minute avant de réessayer
        
        orphan_thread = threading.Thread(target=orphan_cleanup_routine, daemon=True, name="OrphanCleanupThread")
        orphan_thread.start()
        logger.info("🧹 Thread de nettoyage des ordres orphelins démarré (toutes les 5 minutes)")

    def _cleanup_orphan_orders(self):
        """Nettoie les ordres orphelins sur Binance en distinguant les 3 cas selon votre analyse."""
        try:
            logger.info("🧹 Début du nettoyage intelligent des ordres orphelins")
            
            # 1. Construire le mapping cycle => orderId pour les cycles actifs
            with self.cycles_lock:
                active_cycles = list(self.active_cycles.values())
            
            # Récupérer les cycles actifs depuis la DB
            db_cycles_active = self.repository.get_active_cycles()
            
            # Créer le mapping en utilisant SEULEMENT les cycles en mémoire et DB actifs
            # Pas besoin de récupérer les cycles terminés car s'ils sont terminés, 
            # leurs ordres ne devraient plus être sur Binance
            all_cycles = {}
            for cycle in db_cycles_active:
                all_cycles[cycle.id] = cycle
            for cycle in active_cycles:
                all_cycles[cycle.id] = cycle
            
            # Construire le mapping orderId => (type, cycle) pour les ordres légitimes
            # Gérer les conversions string/int pour éviter les erreurs de type
            cycle_orders = {}
            for cycle in all_cycles.values():
                if hasattr(cycle, 'entry_order_id') and cycle.entry_order_id:
                    # Ajouter les deux formats (string et int) pour être sûr
                    entry_id = cycle.entry_order_id
                    cycle_orders[entry_id] = ("entry", cycle)
                    try:
                        cycle_orders[int(entry_id)] = ("entry", cycle)
                    except (ValueError, TypeError):
                        pass
                        
                if hasattr(cycle, 'exit_order_id') and cycle.exit_order_id:
                    exit_id = cycle.exit_order_id  
                    cycle_orders[exit_id] = ("exit", cycle)
                    try:
                        cycle_orders[int(exit_id)] = ("exit", cycle)
                    except (ValueError, TypeError):
                        pass
            
            # 2. Récupérer tous les ordres ouverts sur Binance
            open_orders = self.binance_executor.utils.fetch_open_orders()
            binance_order_ids = {order['orderId'] for order in open_orders}
            
            # 3. Parcourir les ordres Binance et appliquer la logique des 3 cas
            orphan_count = 0
            cleaned_count = 0
            
            for order in open_orders:
                order_id = order['orderId']
                symbol = order['symbol']
                client_order_id = order.get('clientOrderId', '')
                
                if order_id in cycle_orders:
                    # CAS A: Ordre légitime avec cycle correspondant
                    order_type, cycle = cycle_orders[order_id]
                    logger.debug(f"✅ Ordre légitime trouvé: {symbol} {order['side']} (cycle {cycle.id}, {order_type})")
                    continue
                
                # CAS C: Vrai orphelin - Annuler l'ordre
                orphan_count += 1
                logger.warning(f"🚨 Ordre orphelin détecté: {symbol} {order['side']} {order['origQty']}@{order['price']} (ID: {order_id}, ClientID: {client_order_id})")
                
                try:
                    self.binance_executor.utils.cancel_order(symbol, order_id)
                    cleaned_count += 1
                    logger.info(f"✅ Ordre orphelin {order_id} annulé sur Binance")
                except Exception as e:
                    logger.error(f"❌ Impossible d'annuler l'ordre orphelin {order_id}: {str(e)}")
            
            # 4. CAS B: Détecter les cycles fantômes (waiting_* sans ordre sur Binance)
            phantom_cycles = []
            
            for cycle in all_cycles.values():
                status_str = cycle.status.value if hasattr(cycle.status, 'value') else str(cycle.status)
                # Vérifier tous les statuts actifs qui devraient avoir des ordres sur Binance
                if status_str in ['waiting_buy', 'waiting_sell', 'active_buy', 'active_sell']:
                    # Cas spécial : cycles en waiting_buy/waiting_sell avec entry_order_id (trailing stop)
                    if status_str in ['waiting_buy', 'waiting_sell'] and cycle.entry_order_id and not cycle.exit_price:
                        logger.debug(f"✅ Cycle {cycle.id} en {status_str} avec trailing stop")
                        continue
                    # Pour ces cycles, vérifier le bon ordre selon le statut
                    # NOUVEAU: Avec le système no-exit-order, distinguer entrée vs sortie
                    expected_order_id = None
                    is_entry_phase = False
                    
                    # Détecter si c'est la phase d'entrée ou de sortie
                    if cycle.entry_order_id and not cycle.exit_price:
                        # Cycle en cours : entrée non terminée ou sortie sans exit order
                        try:
                            entry_execution = self.binance_executor.get_order_status(cycle.symbol, cycle.entry_order_id)
                            if entry_execution and entry_execution.status != OrderStatus.FILLED:
                                # Phase d'entrée : ordre d'entrée pas encore rempli
                                is_entry_phase = True
                                expected_order_id = cycle.entry_order_id
                            else:
                                # Phase de sortie : ordre d'entrée rempli, pas d'exit order
                                logger.debug(f"✅ Cycle {cycle.id} en phase de sortie {status_str} sans exit order")
                                continue  # Skip la vérification, c'est normal
                        except Exception as e:
                            # Si on ne peut pas vérifier le statut de l'ordre d'entrée, 
                            # on assume que c'est un cycle en phase de sortie
                            logger.debug(f"⚠️ Impossible de vérifier l'ordre d'entrée {cycle.entry_order_id} pour {cycle.id}: {str(e)} - Assumé comme phase de sortie")
                            continue  # Skip la vérification, on assume que c'est le nouveau système
                    else:
                        # Cycle sans entry_order_id valide
                        expected_order_id = cycle.entry_order_id
                        is_entry_phase = True
                    
                    has_active_order = (
                        expected_order_id and 
                        (expected_order_id in binance_order_ids or 
                         str(expected_order_id) in {str(oid) for oid in binance_order_ids} or
                         int(expected_order_id) in binance_order_ids)
                    )
                    
                    if not has_active_order and expected_order_id:
                        phantom_cycles.append(cycle)
                        phase_desc = "entrée" if is_entry_phase else "sortie"
                        logger.warning(f"👻 Cycle fantôme détecté: {cycle.id} en statut {status_str} sans ordre de {phase_desc} sur Binance")
            
            # 5. Traiter les cycles fantômes
            for cycle in phantom_cycles:
                try:
                    # Vérifier depuis combien de temps le cycle est en waiting
                    
                    # Calculer l'âge du cycle
                    if cycle.updated_at:
                        if cycle.updated_at.tzinfo is None:
                            # Si pas de timezone, on assume UTC
                            cycle_time = cycle.updated_at.replace(tzinfo=timezone.utc)
                        else:
                            cycle_time = cycle.updated_at
                        now = datetime.now(timezone.utc)
                        age_minutes = (now - cycle_time).total_seconds() / 60
                    else:
                        age_minutes = 999  # Très vieux si pas de timestamp
                    
                    # NOUVEAU: Délai minimum avant de considérer un ordre comme fantôme
                    MIN_AGE_BEFORE_PHANTOM = 3.0  # 3 minutes minimum
                    if age_minutes < MIN_AGE_BEFORE_PHANTOM:
                        logger.debug(f"⏳ Cycle {cycle.id} trop récent ({age_minutes:.1f}min < {MIN_AGE_BEFORE_PHANTOM}min), skip")
                        continue
                    
                    # NOUVEAU: Recharger le cycle depuis la DB pour avoir les dernières infos
                    fresh_cycle = self.repository.get_cycle(cycle.id)
                    if fresh_cycle:
                        # NOUVEAU: Vérifier si le cycle a été exécuté (exit_price existe)
                        if fresh_cycle.exit_price is not None:
                            logger.info(f"✅ Cycle {cycle.id} a un exit_price ({fresh_cycle.exit_price}), pas un fantôme")
                            # Mettre à jour le cycle en mémoire
                            with self.cycles_lock:
                                self.active_cycles[cycle.id] = fresh_cycle
                            continue
                        
                        # NOUVEAU: Vérifier si le cycle est déjà en statut terminal
                        if fresh_cycle.status in [CycleStatus.COMPLETED, CycleStatus.CANCELED, CycleStatus.FAILED]:
                            logger.info(f"✅ Cycle {cycle.id} déjà en statut {fresh_cycle.status}, pas un fantôme")
                            # Mettre à jour le cycle en mémoire
                            with self.cycles_lock:
                                self.active_cycles[cycle.id] = fresh_cycle
                            continue
                        
                        # Utiliser le cycle rechargé pour la suite
                        cycle = fresh_cycle
                    
                    # Marquer comme failed et libérer les fonds
                    with self.cycles_lock:
                        cycle.status = CycleStatus.FAILED
                        cycle.updated_at = datetime.now()
                        if not hasattr(cycle, 'metadata'):
                            cycle.metadata = {}
                        cycle.metadata['fail_reason'] = f"Ordre manquant sur Binance (âge: {age_minutes:.1f}min)"
                    
                    # Sauvegarder en DB
                    self.repository.save_cycle(cycle)
                    
                    # Publier l'événement
                    self._publish_cycle_event(cycle, "failed")
                    
                    # Supprimer de la mémoire
                    with self.cycles_lock:
                        self.active_cycles.pop(cycle.id, None)
                    
                    logger.info(f"🔧 Cycle fantôme {cycle.id} fermé et nettoyé (âge: {age_minutes:.1f}min)")
                    
                except Exception as e:
                    logger.error(f"❌ Erreur lors du traitement du cycle fantôme {cycle.id}: {str(e)}")
            
            # 7. Résumé du nettoyage
            total_issues = orphan_count + len(phantom_cycles)
            if total_issues > 0:
                logger.warning(f"🎯 Nettoyage terminé: {cleaned_count}/{orphan_count} ordres orphelins annulés, {len(phantom_cycles)} cycles fantômes fermés")
            else:
                logger.debug("✨ Aucun problème détecté - système propre")
                
        except Exception as e:
            logger.error(f"❌ Erreur lors du nettoyage intelligent des ordres orphelins: {str(e)}")

    def _cleanup_inactive_cycles(self):
        """Nettoie les cycles inactifs qui sont restés en mémoire trop BUYtemps."""
        now = datetime.now()
        cycles_to_remove = []
        
        with self.cycles_lock:
            for cycle_id, cycle in self.active_cycles.items():
                # Si le cycle est en état terminal depuis plus de 5 minutes, le supprimer de la mémoire
                if (cycle.status in [CycleStatus.COMPLETED, CycleStatus.CANCELED, CycleStatus.FAILED] and
                    (now - cycle.updated_at).total_seconds() > 5 * 60):
                    cycles_to_remove.append(cycle_id)
        
        # Supprimer les cycles identifiés
        if cycles_to_remove:
            with self.cycles_lock:
                for cycle_id in cycles_to_remove:
                    self.active_cycles.pop(cycle_id, None)
            
            logger.debug(f"🧹 {len(cycles_to_remove)} cycles inactifs nettoyés de la mémoire")
    
    def _remove_failed_cycle(self, cycle_id: str) -> None:
        """
        Supprime immédiatement un cycle failed de la mémoire.
        Cette méthode est appelée dès qu'un cycle passe en statut FAILED.
        """
        with self.cycles_lock:
            if cycle_id in self.active_cycles:
                self.active_cycles.pop(cycle_id)
                logger.debug(f"🗑️ Cycle {cycle_id} supprimé de la mémoire (FAILED)")
    
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
    
    def _verify_cycles_on_startup(self) -> None:
        """
        Vérifie et nettoie les cycles au démarrage.
        Marque comme failed les cycles qui n'ont pas d'ordres correspondants sur Binance.
        """
        try:
            # Attendre un peu pour laisser Binance se stabiliser
            time.sleep(2)
            
            # Récupérer tous les ordres ouverts sur Binance
            open_orders = self.binance_executor.utils.fetch_open_orders()
            binance_order_ids = {str(order['orderId']) for order in open_orders}
            
            logger.info(f"🔍 Vérification de {len(self.active_cycles)} cycles actifs contre {len(binance_order_ids)} ordres Binance")
            
            cycles_to_fail = []
            
            with self.cycles_lock:
                for cycle_id, cycle in self.active_cycles.items():
                    status_str = cycle.status.value if hasattr(cycle.status, 'value') else str(cycle.status)
                    
                    # Vérifier si le cycle devrait avoir des ordres actifs
                    if status_str in ['waiting_buy', 'waiting_sell', 'active_buy', 'active_sell']:
                        # NOUVEAU: Avec no-exit-order, vérifier seulement les cycles en phase d'entrée
                        if cycle.entry_order_id and not cycle.exit_price:
                            # Vérifier si ordre d'entrée existe et n'est pas FILLED
                            entry_execution = self.binance_executor.get_order_status(cycle.symbol, cycle.entry_order_id)
                            if entry_execution and entry_execution.status != OrderStatus.FILLED:
                                # Phase d'entrée : vérifier présence ordre d'entrée
                                expected_order_id = cycle.entry_order_id
                                has_order = (
                                    expected_order_id and 
                                    (str(expected_order_id) in binance_order_ids or
                                     expected_order_id in binance_order_ids or
                                     str(expected_order_id) in {str(oid) for oid in binance_order_ids})
                                )
                            else:
                                # Phase de sortie : normal sans exit order
                                logger.debug(f"✅ Cycle {cycle_id} en phase de sortie au startup")
                                has_order = True  # Considérer comme OK
                        else:
                            # Cycle terminé ou sans entry_order_id
                            has_order = True  # Pas de vérification nécessaire
                        
                        if not has_order:
                            # NOUVEAU: Recharger le cycle depuis la DB pour avoir les dernières infos
                            fresh_cycle = self.repository.get_cycle(cycle_id)
                            if fresh_cycle and fresh_cycle.exit_price is not None:
                                logger.info(f"✅ Cycle {cycle_id} a un exit_price ({fresh_cycle.exit_price}), pas un fantôme au démarrage")
                                # Mettre à jour en mémoire avec le statut correct
                                self.active_cycles[cycle_id] = fresh_cycle
                                continue
                            
                            # Utiliser le cycle rechargé si disponible pour avoir les dernières infos
                            if fresh_cycle:
                                cycle = fresh_cycle
                            
                            # CORRECTION: Vérifier si l'ordre d'entrée a été exécuté avec succès avant de marquer comme fantôme
                            if cycle.entry_price and cycle.entry_price > 0:
                                logger.info(f"✅ Cycle {cycle_id} a un prix d'entrée ({cycle.entry_price}), ordre exécuté - pas un fantôme")
                                continue
                            
                            # Vérifier dans l'historique des trades
                            try:
                                trades = self.binance_executor.utils.get_my_trades(cycle.symbol, limit=100)
                                order_executed = any(trade.get('orderId') == int(cycle.entry_order_id) for trade in trades if trade.get('orderId'))
                                if order_executed:
                                    logger.info(f"✅ Cycle {cycle_id} ordre d'entrée trouvé dans historique - pas un fantôme")
                                    continue
                            except Exception as e:
                                logger.warning(f"⚠️ Impossible de vérifier l'historique des trades pour {cycle_id}: {e}")
                            
                            logger.warning(f"👻 Cycle {cycle_id} en statut {status_str} sans ordre Binance correspondant ni preuve d'exécution")
                            cycles_to_fail.append(cycle)
            
            # Marquer les cycles fantômes comme failed
            for cycle in cycles_to_fail:
                try:
                    logger.info(f"🔧 Nettoyage du cycle fantôme {cycle.id}")
                    
                    # Mettre à jour le statut
                    with self.cycles_lock:
                        cycle.status = CycleStatus.FAILED
                        cycle.updated_at = datetime.now()
                        if not hasattr(cycle, 'metadata'):
                            cycle.metadata = {}
                        cycle.metadata['fail_reason'] = "Cycle fantôme détecté au démarrage - ordres manquants sur Binance"
                    
                    # Sauvegarder en base
                    self.repository.save_cycle(cycle)
                    
                    # Publier l'événement
                    self._publish_cycle_event(cycle, "failed")
                    
                    # Retirer de la mémoire
                    self._remove_failed_cycle(cycle.id)
                    
                except Exception as e:
                    logger.error(f"❌ Erreur lors du nettoyage du cycle {cycle.id}: {str(e)}")
            
            if cycles_to_fail:
                logger.info(f"✅ {len(cycles_to_fail)} cycles fantômes nettoyés au démarrage")
            else:
                logger.info("✅ Aucun cycle fantôme détecté au démarrage")
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la vérification des cycles au démarrage: {str(e)}")
    
    def create_cycle(self, symbol: str, strategy: str, side: Union[OrderSide, str], 
                    price: float, quantity: float,
                    stop_price: Optional[float] = None,
                    trailing_delta: Optional[float] = None) -> Optional[TradeCycle]:
        """
        Crée un nouveau cycle de trading et exécute l'ordre d'entrée.
        
        L'ordre d'entrée est exécuté en MARKET pour garantir l'exécution immédiate.
        Le paramètre price est utilisé comme référence pour les calculs mais pas pour l'ordre.
        
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
            
            # Avec StopManagerPure, seul stop_price est nécessaire
            # Le stop à 3% suffit pour gérer la sortie automatiquement

            cycle_id = f"cycle_{uuid.uuid4().hex[:16]}"
            now = datetime.now()

            # Créer l'objet cycle
            cycle = TradeCycle(
                id=cycle_id,
                symbol=symbol,
                strategy=strategy,
                status=CycleStatus.INITIATING,
                side=side,  # Ajouter le side au cycle
                entry_price=None,
                quantity=quantity,
                stop_price=stop_price,
                trailing_delta=trailing_delta,
                created_at=now,
                updated_at=now,
                demo=self.demo_mode,
                metadata={}  # IMPORTANT: Toujours initialiser metadata
            )

            # Garder le prix de référence pour les calculs (validation des fonds, target price, etc.)
            reference_price = price

            # Vérifier le solde avant d'exécuter l'ordre (pour BUY et SELL)
            if not self.demo_mode:
                # Extraire la base currency et quote currency
                if symbol.endswith('USDC'):
                    base_currency = symbol.replace('USDC', '')
                    quote_currency = 'USDC'
                elif symbol.endswith('BTC'):
                    base_currency = symbol.replace('BTC', '') if symbol != 'BTCUSDC' else 'BTC'
                    quote_currency = 'BTC' if symbol != 'BTCUSDC' else 'USDC'
                else:
                    # Fallback pour autres paires
                    base_currency = symbol[:3]
                    quote_currency = symbol[3:]
                
                # Récupérer les soldes actuels
                balances = self.binance_executor.utils.fetch_account_balances(self.binance_executor.time_offset)
                logger.info(f"🔍 Balances Binance récupérées: {balances}")
                
                if side == OrderSide.BUY:
                    # Pour BUY: vérifier qu'on a assez de quote currency
                    available_balance = balances.get(quote_currency, {}).get('free', 0)
                    logger.info(f"💰 Balance {quote_currency}: {available_balance}")
                    
                    # Calculer le coût total de l'ordre
                    slippage_margin = 1.005  # 0.5% de marge pour le slippage
                    fee_margin = 1.001       # 0.1% pour les frais taker
                    total_cost = reference_price * quantity * slippage_margin * fee_margin
                    
                    if available_balance < total_cost:
                        logger.warning(f"⚠️ Solde {quote_currency} insuffisant pour BUY: {available_balance:.8f} < {total_cost:.8f}")
                        
                        # Si le solde n'est pas complètement à zéro, ajuster la quantité
                        if available_balance > 0:
                            safe_margin = 0.98  # 98% du solde pour couvrir slippage + frais
                            adjusted_quantity = (available_balance * safe_margin) / reference_price
                            min_quantity = self.binance_executor.symbol_constraints.get_min_qty(symbol)
                            
                            if adjusted_quantity >= min_quantity:
                                logger.info(f"💡 Ajustement automatique de la quantité: {quantity:.8f} → {adjusted_quantity:.8f} {base_currency}")
                                quantity = adjusted_quantity
                                cycle.quantity = adjusted_quantity
                                cycle.metadata['quantity_adjusted'] = True
                                cycle.metadata['original_quantity'] = quantity
                                cycle.metadata['adjustment_reason'] = f"Solde insuffisant: {available_balance:.8f} {quote_currency}"
                            else:
                                # Quantité ajustée toujours trop petite
                                logger.error(f"❌ Solde {quote_currency} insuffisant même après ajustement: {adjusted_quantity:.8f} < {min_quantity:.8f}")
                                
                                # Créer le cycle avec un statut FAILED pour la traçabilité
                                cycle.status = CycleStatus.FAILED
                                cycle.updated_at = datetime.now()
                                if not hasattr(cycle, 'metadata'):
                                    cycle.metadata = {}
                                cycle.metadata['fail_reason'] = f"Solde {quote_currency} insuffisant même après ajustement: {adjusted_quantity:.8f} < {min_quantity:.8f}"
                                
                                # Sauvegarder le cycle échoué pour la traçabilité
                                self.repository.save_cycle(cycle)
                                
                                # Publier l'événement d'échec
                                self._publish_cycle_event(cycle, "failed")
                                
                                # Nettoyer les ordres potentiels sur Binance
                                self._cleanup_cycle_orders(cycle)
                                
                                return None
                        else:
                            # Solde complètement à zéro
                            logger.error(f"❌ Aucun solde {quote_currency} disponible pour BUY")
                            
                            # Créer le cycle avec un statut FAILED pour la traçabilité
                            cycle.status = CycleStatus.FAILED
                            cycle.updated_at = datetime.now()
                            if not hasattr(cycle, 'metadata'):
                                cycle.metadata = {}
                            cycle.metadata['fail_reason'] = f"Aucun solde {quote_currency} disponible"
                            
                            # Sauvegarder le cycle échoué pour la traçabilité
                            self.repository.save_cycle(cycle)
                            
                            # Publier l'événement d'échec
                            self._publish_cycle_event(cycle, "failed")
                            
                            # Nettoyer les ordres potentiels sur Binance
                            self._cleanup_cycle_orders(cycle)
                            
                            return None
                        
                elif side == OrderSide.SELL:
                    # Pour SELL: vérifier qu'on a assez de base currency à vendre
                    available_balance = balances.get(base_currency, {}).get('free', 0)
                    logger.info(f"💰 Balance {base_currency}: {available_balance}")
                    
                    # Ajouter une petite marge pour les frais
                    required_quantity = quantity * 1.001  # 0.1% de marge pour les frais
                    
                    if available_balance < required_quantity:
                        logger.warning(f"⚠️ Solde {base_currency} insuffisant pour SELL: {available_balance:.8f} < {required_quantity:.8f}")
                        
                        # Si le solde n'est pas complètement à zéro, ajuster la quantité
                        if available_balance > 0:
                            safe_margin = 0.99  # 99% du solde disponible
                            adjusted_quantity = available_balance * safe_margin
                            min_quantity = self.binance_executor.symbol_constraints.get_min_qty(symbol)
                            
                            if adjusted_quantity >= min_quantity:
                                logger.info(f"💡 Ajustement automatique de la quantité: {quantity:.8f} → {adjusted_quantity:.8f} {base_currency}")
                                quantity = adjusted_quantity
                                cycle.quantity = adjusted_quantity
                                cycle.metadata['quantity_adjusted'] = True
                                cycle.metadata['original_quantity'] = quantity
                                cycle.metadata['adjustment_reason'] = f"Solde insuffisant: {available_balance:.8f} {base_currency}"
                            else:
                                # Quantité ajustée toujours trop petite
                                logger.error(f"❌ Solde {base_currency} insuffisant même après ajustement: {adjusted_quantity:.8f} < {min_quantity:.8f}")
                                
                                # Créer le cycle avec un statut FAILED pour la traçabilité
                                cycle.status = CycleStatus.FAILED
                                cycle.updated_at = datetime.now()
                                if not hasattr(cycle, 'metadata'):
                                    cycle.metadata = {}
                                cycle.metadata['fail_reason'] = f"Solde {base_currency} insuffisant même après ajustement: {adjusted_quantity:.8f} < {min_quantity:.8f}"
                                
                                # Sauvegarder le cycle échoué pour la traçabilité
                                self.repository.save_cycle(cycle)
                                
                                # Publier l'événement d'échec
                                self._publish_cycle_event(cycle, "failed")
                                
                                # Nettoyer les ordres potentiels sur Binance
                                self._cleanup_cycle_orders(cycle)
                                
                                return None
                        else:
                            # Solde complètement à zéro
                            logger.error(f"❌ Aucun solde {base_currency} disponible pour SELL")
                            
                            # Créer le cycle avec un statut FAILED pour la traçabilité
                            cycle.status = CycleStatus.FAILED
                            cycle.updated_at = datetime.now()
                            if not hasattr(cycle, 'metadata'):
                                cycle.metadata = {}
                            cycle.metadata['fail_reason'] = f"Aucun solde {base_currency} disponible"
                            
                            # Sauvegarder le cycle échoué pour la traçabilité
                            self.repository.save_cycle(cycle)
                            
                            # Publier l'événement d'échec
                            self._publish_cycle_event(cycle, "failed")
                            
                            # Nettoyer les ordres potentiels sur Binance
                            self._cleanup_cycle_orders(cycle)
                            
                            return None
            
            # Créer l'ordre d'entrée - utiliser MARKET pour exécution immédiate
            # On ne passe pas de prix à l'ordre pour forcer MARKET
            entry_order = TradeOrder(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=None,  # Pas de prix = ordre MARKET pour exécution immédiate
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
                
                # Pour les ordres MARKET, le statut devrait être FILLED immédiatement
                # On accepte aussi PARTIALLY_FILLED au cas où
                valid_statuses = [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]
                if execution.status not in valid_statuses:
                    logger.error(f"❌ L'ordre MARKET d'entrée pour le cycle {cycle_id} n'a pas été exécuté immédiatement: {execution.status}")
                    
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
                # Initialiser min_price et max_price avec le prix d'entrée
                cycle.min_price = execution.price
                cycle.max_price = execution.price
                # Si la quantité exécutée diffère, la stocker dans metadata
                if execution.quantity != cycle.quantity:
                    logger.info(f"📊 Quantité ajustée: {cycle.quantity} → {execution.quantity}")
                    cycle.metadata['executed_quantity'] = float(execution.quantity)
                # Après un ordre MARKET d'entrée, on attend l'ordre de sortie
                # BUY -> on a acheté, on attend de vendre -> waiting_sell
                # SELL -> on a vendu, on attend de racheter -> WAITING_BUY
                cycle.status = CycleStatus.WAITING_SELL if side == OrderSide.BUY else CycleStatus.WAITING_BUY
                cycle.confirmed = True
                cycle.updated_at = datetime.now()
                self.active_cycles[cycle_id] = cycle

            # Enregistrer l'exécution et le cycle
            self.repository.save_execution(execution, cycle_id)
            try:
                self.repository.save_cycle(cycle)
            except Exception as e:
                import traceback
                logger.error(f"❌ Erreur détaillée save_cycle: {str(e)}")
                logger.error(f"❌ Stack trace save_cycle: {traceback.format_exc()}")
                # Ne pas faire échouer le cycle pour cette erreur de sauvegarde
                logger.warning("⚠️ Cycle créé mais non sauvegardé - continuons")

            # Publier sur Redis
            self._publish_cycle_event(cycle, "created")

            logger.info(f"✅ Cycle {cycle_id} créé avec succès: {side.value} {quantity} {symbol} @ {execution.price}")
            
            # Initialiser immédiatement le trailing stop
            try:
                self.stop_manager.initialize_trailing_stop(cycle)
                logger.info(f"🎯 TrailingStop initialisé immédiatement pour le cycle {cycle_id}")
            except Exception as e:
                logger.warning(f"⚠️ Échec d'initialisation immédiate du trailing stop pour {cycle_id}: {str(e)}")
                logger.info(f"🎯 Cycle créé - StopManagerPure gère le trailing stop à 8% (initialisation différée)")
            
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
                "status": cycle.status.value if hasattr(cycle.status, 'value') else str(cycle.status),
                "quantity": float(cycle.quantity) if hasattr(cycle.quantity, 'dtype') else cycle.quantity,
                "entry_price": float(cycle.entry_price) if hasattr(cycle.entry_price, 'dtype') else cycle.entry_price,
                "timestamp": int(cycle.created_at.timestamp() * 1000),
            }
            
            # Ajouter des infos supplémentaires selon le type d'événement
            if event_type == "completed" and cycle.exit_price:
                cycle_data["exit_price"] = float(cycle.exit_price) if hasattr(cycle.exit_price, 'dtype') else cycle.exit_price
                cycle_data["profit_loss"] = float(cycle.profit_loss) if hasattr(cycle.profit_loss, 'dtype') else cycle.profit_loss
                cycle_data["profit_loss_percent"] = float(cycle.profit_loss_percent) if hasattr(cycle.profit_loss_percent, 'dtype') else cycle.profit_loss_percent
            
            redis.publish(f"roottrading:cycle:{event_type}", cycle_data)
            logger.info(f"📢 Événement {event_type} publié pour le cycle {cycle.id}")
        except Exception as e:
            logger.warning(f"⚠️ Impossible de publier l'événement de cycle sur Redis: {str(e)}")
    
    def close_cycle_accounting(self, cycle_id: str, exit_price: float, reason: str = "Fermeture comptable") -> bool:
        """
        Ferme un cycle de manière comptable sans passer d'ordre réel.
        Utilisé pour les retournements où on calcule la position nette.
        
        Args:
            cycle_id: ID du cycle à fermer
            exit_price: Prix de sortie pour le calcul du P&L
            reason: Raison de la fermeture
            
        Returns:
            True si la fermeture a réussi, False sinon
        """
        try:
            # Récupérer le cycle
            with self.cycles_lock:
                cycle = self.active_cycles.get(cycle_id)
                if not cycle:
                    logger.warning(f"⚠️ Cycle {cycle_id} non trouvé pour fermeture comptable")
                    return False
                
                # Vérifier que le cycle peut être fermé
                if cycle.status not in [CycleStatus.WAITING_BUY, CycleStatus.ACTIVE_BUY, 
                                       CycleStatus.WAITING_SELL, CycleStatus.ACTIVE_SELL]:
                    logger.warning(f"⚠️ Impossible de fermer comptablement le cycle {cycle_id} avec le statut {cycle.status}")
                    return False
            
            # Calculer le P&L théorique
            actual_quantity = cycle.metadata.get('executed_quantity', cycle.quantity) if cycle.metadata else cycle.quantity
            entry_value = cycle.entry_price * actual_quantity
            exit_value = exit_price * actual_quantity
            
            # Déterminer le côté pour le calcul du profit
            if cycle.status in [CycleStatus.WAITING_SELL, CycleStatus.ACTIVE_SELL]:
                # Position BUY fermée par SELL théorique
                profit_loss = exit_value - entry_value
            else:
                # Position SELL fermée par BUY théorique  
                profit_loss = entry_value - exit_value
            
            profit_loss_percent = (profit_loss / entry_value) * 100
            
            # Marquer le cycle comme complété
            with self.cycles_lock:
                cycle.status = CycleStatus.COMPLETED
                cycle.exit_price = exit_price
                cycle.profit_loss = profit_loss
                cycle.profit_loss_percent = profit_loss_percent
                cycle.completed_at = datetime.now()
                cycle.updated_at = datetime.now()
                
                # Ajouter la raison dans les métadonnées
                if not hasattr(cycle, 'metadata'):
                    cycle.metadata = {}
                cycle.metadata['closure_type'] = 'accounting'
                cycle.metadata['closure_reason'] = reason
            
            # Sauvegarder en base
            self.repository.save_cycle(cycle)
            
            # Publier l'événement et nettoyer la mémoire
            self._publish_cycle_event(cycle, "completed")
            with self.cycles_lock:
                self.active_cycles.pop(cycle_id, None)
            
            logger.info(f"✅ Cycle {cycle_id} fermé comptablement: P&L = {profit_loss:.4f} ({profit_loss_percent:.2f}%)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur fermeture comptable cycle {cycle_id}: {str(e)}")
            return False

    def create_net_position_cycle(self, new_signal_data: Dict, opposite_cycles: List[Dict]) -> Optional[str]:
        """
        Crée un cycle avec position nette en fermant comptablement les cycles opposés.
        
        Args:
            new_signal_data: Données du nouveau signal/cycle
            opposite_cycles: Cycles opposés à fermer comptablement
            
        Returns:
            ID du nouveau cycle créé ou None
        """
        try:
            new_side = new_signal_data['side']
            new_quantity = new_signal_data['quantity']
            new_price = new_signal_data['price']
            symbol = new_signal_data['symbol']
            
            logger.info(f"🔄 Calcul position nette pour {symbol}: nouveau {new_side} {new_quantity}")
            
            # Calculer la quantité nette
            opposite_quantity = 0.0
            for cycle_data in opposite_cycles:
                cycle_id = cycle_data.get('id')
                if not cycle_id:
                    continue
                    
                # Récupérer le cycle complet
                cycle = self.get_cycle(cycle_id)
                if cycle:
                    # Utiliser la quantité exécutée si disponible
                    actual_quantity = cycle.metadata.get('executed_quantity', cycle.quantity) if cycle.metadata else cycle.quantity
                    opposite_quantity += actual_quantity
                    
                    # Fermer comptablement
                    success = self.close_cycle_accounting(
                        cycle_id, 
                        new_price, 
                        f"Retournement vers {new_side} - fermeture comptable"
                    )
                    
                    if success:
                        logger.info(f"📝 Cycle opposé {cycle_id} fermé comptablement ({actual_quantity} unités)")
                    else:
                        logger.error(f"❌ Échec fermeture comptable {cycle_id}")
                        return None
            
            # Calculer la quantité nette finale
            if new_side == 'BUY':
                # BUY nouveau - positions SELL existantes = position nette BUY
                net_quantity = new_quantity + opposite_quantity
            else:
                # SELL nouveau - positions BUY existantes = position nette SELL  
                net_quantity = new_quantity + opposite_quantity
                
            logger.info(f"📊 Position nette calculée: {new_side} {net_quantity} (nouveau: {new_quantity} + opposé: {opposite_quantity})")
            
            # Si la quantité nette est trop faible, ne pas créer de cycle
            min_qty = self.binance_executor.symbol_constraints.get_min_qty(symbol)
            if net_quantity < min_qty:
                logger.warning(f"⚠️ Quantité nette {net_quantity} trop faible (min: {min_qty}), pas de nouveau cycle")
                return "net_position_too_small"
            
            # Créer le nouveau cycle avec la quantité nette
            net_signal_data = new_signal_data.copy()
            net_signal_data['quantity'] = net_quantity
            
            # Ajouter métadonnées pour traçabilité
            if 'metadata' not in net_signal_data:
                net_signal_data['metadata'] = {}
            net_signal_data['metadata']['net_position'] = True
            net_signal_data['metadata']['original_quantity'] = new_quantity
            net_signal_data['metadata']['opposite_quantity'] = opposite_quantity
            net_signal_data['metadata']['closed_cycles'] = [cycle['id'] for cycle in opposite_cycles]
            
            # Créer le cycle normalement
            cycle_id = self.create_cycle_from_signal(net_signal_data)
            
            if cycle_id:
                logger.info(f"✅ Cycle position nette créé: {cycle_id} ({new_side} {net_quantity})")
            
            return cycle_id
            
        except Exception as e:
            logger.error(f"❌ Erreur création position nette: {str(e)}")
            return None

    def close_cycle(self, cycle_id: str, exit_price: Optional[float] = None, is_stop_loss: bool = False, force_market: bool = False) -> bool:
        """
        Ferme un cycle de trading en exécutant l'ordre de sortie.
        
        Args:
            cycle_id: ID du cycle à fermer
            exit_price: Prix de sortie (optionnel, sinon au marché)
            is_stop_loss: Si True, indique que c'est un stop loss qui se déclenche
            force_market: Si True, utilise un ordre MARKET pour fermeture immédiate
            
        Returns:
            True si la fermeture a réussi, False sinon
        """
        try:
            # Récupérer le cycle
            with self.cycles_lock:
                # D'abord vérifier l'état en DB pour éviter les doubles fermetures
                db_cycle = self.repository.get_cycle(cycle_id)
                if db_cycle and db_cycle.status in [CycleStatus.COMPLETED, CycleStatus.CANCELED, CycleStatus.FAILED]:
                    logger.debug(f"✅ Cycle {cycle_id} déjà fermé en DB avec le statut {db_cycle.status}")
                    # Nettoyer le cache mémoire s'il y est encore
                    self.active_cycles.pop(cycle_id, None)
                    return True
                
                if cycle_id not in self.active_cycles:
                    logger.warning(f"⚠️ Cycle {cycle_id} non trouvé dans les cycles actifs")
                    return False
                
                cycle = self.active_cycles[cycle_id]
                
                # PROTECTION SUPPLÉMENTAIRE: Vérifier si le cycle en mémoire est déjà complété
                if cycle.status == CycleStatus.COMPLETED:
                    logger.info(f"⛔ Le cycle {cycle_id} est déjà marqué comme terminé, skip du close.")
                    return True
            
            # Vérifier que le cycle peut être fermé
            if cycle.status not in [CycleStatus.WAITING_BUY, CycleStatus.ACTIVE_BUY, 
                                   CycleStatus.WAITING_SELL, CycleStatus.ACTIVE_SELL]:
                logger.warning(f"⚠️ Impossible de fermer le cycle {cycle_id} avec le statut {cycle.status}")
                return False
            
            # Déterminer le côté de l'ordre de sortie (inverse de l'entrée)
            if cycle.status in [CycleStatus.WAITING_SELL, CycleStatus.ACTIVE_SELL]:
                # Position BUY → fermer par SELL
                exit_side = OrderSide.SELL
            else:  # WAITING_BUY ou ACTIVE_BUY
                # Position SELL → fermer par BUY
                exit_side = OrderSide.BUY

            # Si il y a un ordre de sortie existant, vérifier son statut avant de l'annuler
            # Ne pas essayer d'annuler si le cycle est en WAITING_SELL/BUY car l'ordre n'existe pas sur Binance
            # Si le cycle a un ordre de sortie et est en attente, vérifier s'il est exécuté
            if cycle.exit_order_id and cycle.status in [CycleStatus.WAITING_SELL, CycleStatus.WAITING_BUY]:
                try:
                    # Vérifier d'abord le statut de l'ordre
                    order_status = self.binance_executor.get_order_status(cycle.symbol, cycle.exit_order_id)
                    
                    if order_status:
                        if order_status.status == OrderStatus.FILLED:
                            # L'ordre est déjà exécuté, mettre à jour le cycle et terminer
                            logger.info(f"✅ L'ordre limite {cycle.exit_order_id} est déjà exécuté, fermeture du cycle")
                            
                            # Calculer le P&L
                            # Utiliser la quantité réellement exécutée
                            actual_quantity = order_status.quantity  # Quantité de l'ordre de sortie
                            entry_value = cycle.entry_price * actual_quantity
                            exit_value = order_status.price * actual_quantity
                            
                            if exit_side == OrderSide.SELL:
                                profit_loss = exit_value - entry_value
                            else:
                                profit_loss = entry_value - exit_value
                            
                            profit_loss_percent = (profit_loss / entry_value) * 100
                            
                            # Mettre à jour le cycle
                            with self.cycles_lock:
                                cycle.exit_price = order_status.price
                                cycle.status = CycleStatus.COMPLETED
                                cycle.profit_loss = profit_loss
                                cycle.profit_loss_percent = profit_loss_percent
                                cycle.completed_at = datetime.now()
                                cycle.updated_at = datetime.now()
                            
                            # Sauvegarder le cycle
                            self.repository.save_cycle(cycle)
                            
                            # Publier l'événement et nettoyer
                            self._publish_cycle_event(cycle, "completed")
                            
                            with self.cycles_lock:
                                self.active_cycles.pop(cycle_id, None)
                            
                            logger.info(f"✅ Cycle {cycle_id} fermé avec succès: P&L = {profit_loss:.2f} ({profit_loss_percent:.2f}%)")
                            return True
                            
                        elif order_status.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                            # L'ordre est encore actif, on peut l'annuler
                            if is_stop_loss:
                                logger.info(f"🛑 Stop loss déclenché - Annulation de l'ordre limite {cycle.exit_order_id}")
                            else:
                                logger.info(f"🎯 Prix cible atteint - Annulation de l'ordre limite {cycle.exit_order_id}")
                            
                            cancel_result = self.binance_executor.cancel_order(cycle.symbol, cycle.exit_order_id)
                            if cancel_result:
                                logger.info(f"✅ Ordre limite {cycle.exit_order_id} annulé avec succès")
                            else:
                                logger.warning(f"⚠️ L'ordre {cycle.exit_order_id} n'a pas pu être annulé")
                        else:
                            # L'ordre est dans un état inattendu (CANCELED, REJECTED, etc.)
                            logger.warning(f"⚠️ L'ordre {cycle.exit_order_id} est dans l'état {order_status.status}")
                            
                            # Si l'ordre a été annulé ou rejeté, on doit gérer le cycle
                            if order_status.status in [OrderStatus.CANCELED, OrderStatus.REJECTED]:
                                logger.error(f"❌ L'ordre de sortie a été {order_status.status}, marquage du cycle comme FAILED")
                                with self.cycles_lock:
                                    cycle.status = CycleStatus.FAILED
                                    cycle.updated_at = datetime.now()
                                    if not hasattr(cycle, 'metadata'):
                                        cycle.metadata = {}
                                    cycle.metadata['fail_reason'] = f"Ordre de sortie {order_status.status}"
                                self.repository.save_cycle(cycle)
                                self._publish_cycle_event(cycle, "failed")
                                self._remove_failed_cycle(cycle_id)
                                return False
                    else:
                        # Pas pu récupérer le statut, continuer avec prudence
                        logger.warning(f"⚠️ Impossible de vérifier le statut de l'ordre {cycle.exit_order_id}, tentative d'annulation")
                        self.binance_executor.cancel_order(cycle.symbol, cycle.exit_order_id)
                        
                except Exception as e:
                    logger.warning(f"⚠️ Erreur lors de la gestion de l'ordre existant {cycle.exit_order_id}: {str(e)}")
                    # Continuer même si l'annulation échoue
            
            # Vérifier les soldes disponibles avant de créer l'ordre
            balances = self.binance_executor.get_account_balances()
            
            if exit_side == OrderSide.SELL:
                # Pour vendre, on a besoin de la devise de base (ex: BTC pour BTCUSDC)
                base_asset = cycle.symbol[:-4] if cycle.symbol.endswith('USDC') else cycle.symbol[:-3]
                available = balances.get(base_asset, {}).get('free', 0)
                
                if available < cycle.quantity:
                    logger.error(f"❌ Solde {base_asset} insuffisant pour l'ordre de sortie: {available:.8f} < {cycle.quantity:.8f}")
                    # Marquer le cycle comme échoué
                    with self.cycles_lock:
                        cycle.status = CycleStatus.FAILED
                        cycle.updated_at = datetime.now()
                    self.repository.save_cycle(cycle)
                    # Supprimer le cycle de la mémoire
                    self._remove_failed_cycle(cycle_id)
                    return False
            else:
                # Pour acheter, on a besoin de la devise de cotation (ex: USDC pour BTCUSDC)
                quote_asset = cycle.symbol[-4:] if cycle.symbol.endswith('USDC') else cycle.symbol[-3:]
                # Utiliser la quantité réellement exécutée (après ventes partielles)
                exit_quantity = cycle.metadata.get('executed_quantity', cycle.quantity) if cycle.metadata else cycle.quantity
                required_amount = exit_quantity * (exit_price or cycle.entry_price)
                available = balances.get(quote_asset, {}).get('free', 0)
                
                if available < required_amount:
                    logger.error(f"❌ Solde {quote_asset} insuffisant pour l'ordre de sortie: {available:.8f} < {required_amount:.8f}")
                    # Marquer le cycle comme échoué
                    with self.cycles_lock:
                        cycle.status = CycleStatus.FAILED
                        cycle.updated_at = datetime.now()
                    self.repository.save_cycle(cycle)
                    # Supprimer le cycle de la mémoire
                    self._remove_failed_cycle(cycle_id)
                    return False
            
            # PROTECTION: Vérifier une dernière fois qu'un ordre de sortie n'existe pas déjà
            # Recharger le cycle depuis la DB pour avoir la version la plus récente
            fresh_cycle = self.repository.get_cycle(cycle_id)
            if fresh_cycle and fresh_cycle.exit_order_id and fresh_cycle.exit_order_id != cycle.exit_order_id:
                # Vérifier si l'ordre existant est toujours actif
                existing_order_status = self.binance_executor.get_order_status(fresh_cycle.symbol, fresh_cycle.exit_order_id)
                if existing_order_status and existing_order_status.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                    logger.warning(f"⚠️ Un ordre de sortie actif {fresh_cycle.exit_order_id} existe déjà pour le cycle {cycle_id}")
                    # Si c'est un stop loss et qu'un ordre limite existe, on doit l'annuler
                    if is_stop_loss:
                        logger.info(f"🛑 Stop loss déclenché - Annulation de l'ordre limite existant {fresh_cycle.exit_order_id}")
                        try:
                            self.binance_executor.cancel_order(fresh_cycle.symbol, fresh_cycle.exit_order_id)
                            logger.info(f"✅ Ordre limite annulé avec succès")
                        except Exception as e:
                            logger.error(f"❌ Impossible d'annuler l'ordre limite: {str(e)}")
                    else:
                        # Pas un stop loss, on ne crée pas de nouvel ordre
                        return False
            
            # Créer l'ordre de sortie avec un ID unique pour éviter les duplicatas
            # Si c'est un stop loss, utiliser un ID différent mais plus court
            if is_stop_loss:
                # Utiliser les 6 derniers caractères du cycle_id + "s" + timestamp court (6 chiffres)
                SELL_cycle_id = cycle_id[-6:]
                SELL_timestamp = str(int(time.time()))[-6:]
                client_order_id = f"exit_{SELL_cycle_id}_s{SELL_timestamp}"
            else:
                client_order_id = f"exit_{cycle_id}"
            
            # Utiliser la quantité réellement exécutée à l'entrée (si disponible dans metadata)
            exit_quantity = cycle.metadata.get('executed_quantity', cycle.quantity)
            if exit_quantity != cycle.quantity:
                logger.debug(f"📊 Utilisation de la quantité exécutée pour la sortie: {exit_quantity} (vs théorique: {cycle.quantity})")
            
            # Déterminer le type d'ordre et le prix
            order_price = exit_price
            if force_market or is_stop_loss:
                # Pour les fermetures forcées (retournement) ou stop-loss, utiliser MARKET
                # On garde le prix pour les logs/calculs mais force l'exécution immédiate
                order_price = None
                logger.info(f"🚀 Fermeture MARKET forcée pour le cycle {cycle_id} (prix cible: {exit_price})")
            
            exit_order = TradeOrder(
                symbol=cycle.symbol,
                side=exit_side,
                quantity=exit_quantity,
                price=order_price,  # None pour un ordre au marché
                client_order_id=client_order_id,
                strategy=cycle.strategy,
                demo=cycle.demo
            )
            
            # Exécuter l'ordre de sortie
            logger.info(f"🔄 Exécution de l'ordre de sortie pour le cycle {cycle_id}")
            
            try:
                execution = self.binance_executor.execute_order(exit_order)
            except Exception as e:
                logger.error(f"❌ Échec de l'exécution de l'ordre de sortie: {str(e)}")
                
                # Si c'est une erreur de solde insuffisant, marquer le cycle comme échoué
                if "insufficient balance" in str(e).lower():
                    with self.cycles_lock:
                        cycle.status = CycleStatus.FAILED
                        cycle.updated_at = datetime.now()
                    self.repository.save_cycle(cycle)
                    logger.error(f"❌ Cycle {cycle_id} marqué comme FAILED suite à un solde insuffisant")
                    # Supprimer le cycle de la mémoire
                    self._remove_failed_cycle(cycle_id)
                
                return False
            
            # Vérifier le statut de l'exécution
            if execution.status == OrderStatus.NEW:
                # L'ordre est créé mais pas encore exécuté (ordre LIMIT)
                if force_market:
                    # Si c'était censé être MARKET mais est en NEW, c'est anormal
                    logger.warning(f"⚠️ Ordre MARKET pour {cycle_id} en statut NEW: {execution.order_id}")
                else:
                    logger.info(f"⏳ Ordre de sortie créé pour le cycle {cycle_id}: {execution.order_id} (en attente d'exécution)")
                
                # Mettre à jour le cycle avec l'ordre de sortie en attente
                with self.cycles_lock:
                    cycle.exit_order_id = execution.order_id
                    cycle.exit_price = execution.price  # Prix cible, pas le prix d'exécution
                    
                    if force_market:
                        # Pour les fermetures forcées, marquer comme COMPLETED immédiatement
                        cycle.status = CycleStatus.COMPLETED
                        cycle.completed_at = datetime.now()
                        logger.info(f"✅ Cycle {cycle_id} marqué COMPLETED pour fermeture forcée")
                    # Sinon le statut reste waiting_sell ou WAITING_BUY car on attend que l'ordre LIMIT soit exécuté
                    cycle.updated_at = datetime.now()
                
                # Sauvegarder le cycle mis à jour
                self.repository.save_cycle(cycle)
                self.repository.save_execution(execution, cycle_id)
                
                if force_market:
                    # Publier l'événement completed et nettoyer la mémoire
                    self._publish_cycle_event(cycle, "completed")
                    with self.cycles_lock:
                        self.active_cycles.pop(cycle_id, None)
                    logger.info(f"✅ Cycle {cycle_id} fermé immédiatement (fermeture forcée)")
                else:
                    logger.info(f"✅ Ordre de sortie créé pour le cycle {cycle_id}: {execution.order_id}")
                
                return True
                
            elif execution.status != OrderStatus.FILLED:
                # Autre statut (PARTIALLY_FILLED, REJECTED, etc.)
                logger.warning(f"⚠️ Ordre de sortie pour le cycle {cycle_id} dans un état inattendu: {execution.status}")
                return False
            
            # Si l'ordre est FILLED, calculer le P&L et marquer comme complété
            # Calculer le profit/perte
            # IMPORTANT: Utiliser la quantité réellement exécutée à l'entrée
            actual_entry_quantity = cycle.metadata.get('executed_quantity', cycle.quantity) if cycle.metadata else cycle.quantity
            entry_value = cycle.entry_price * actual_entry_quantity
            exit_value = execution.price * execution.quantity

            if exit_side == OrderSide.SELL:
                # Position BUY fermée par SELL, profit = sortie - entrée
                profit_loss = exit_value - entry_value
            else:
                # Position SELL fermée par BUY, profit = entrée - sortie
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
            self._publish_cycle_event(cycle, "completed")
            
            # Nettoyer les ordres restants sur Binance
            self._cleanup_cycle_orders(cycle)
            
            # Supprimer le cycle des cycles actifs
            with self.cycles_lock:
                self.active_cycles.pop(cycle_id, None)
                        
            logger.info(f"✅ Cycle {cycle_id} fermé avec succès: P&L = {profit_loss:.2f} ({profit_loss_percent:.2f}%)")
            return True
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la fermeture du cycle {cycle_id}: {str(e)}")
            return False
    
    def partial_SELL_cycle(self, cycle_id: str, percentage: float, price: float, reason: str = "take_profit") -> bool:
        """
        Effectue une fermeture partielle d'un cycle (vente partielle BUY ou rachat partiel SELL).
        
        Args:
            cycle_id: ID du cycle
            percentage: Pourcentage à fermer (ex: 30.0 pour 30%)
            price: Prix de fermeture partielle
            reason: Raison de la fermeture partielle
            
        Returns:
            True si la fermeture partielle a réussi, False sinon
        """
        try:
            # Récupérer le cycle
            with self.cycles_lock:
                if cycle_id not in self.active_cycles:
                    logger.warning(f"⚠️ Cycle {cycle_id} non trouvé pour fermeture partielle")
                    return False
                
                cycle = self.active_cycles[cycle_id]
                
                # Vérifier que le cycle est dans un état fermable partiellement
                if cycle.status not in [CycleStatus.WAITING_SELL, CycleStatus.ACTIVE_SELL, 
                                       CycleStatus.WAITING_BUY, CycleStatus.ACTIVE_BUY]:
                    logger.warning(f"⚠️ Impossible de fermer partiellement le cycle {cycle_id} avec statut {cycle.status}")
                    return False
            
            # Déterminer le côté de la fermeture partielle selon la position
            if cycle.status in [CycleStatus.WAITING_SELL, CycleStatus.ACTIVE_SELL]:
                # Position BUY ouverte → fermeture partielle par VENTE (SELL)
                partial_side = OrderSide.SELL
                position_type = "BUY"
            else:  # WAITING_BUY ou ACTIVE_BUY
                # Position SELL ouverte → fermeture partielle par RACHAT (BUY)
                partial_side = OrderSide.BUY
                position_type = "SELL"
            
            # Calculer la quantité à fermer
            total_quantity = cycle.metadata.get('executed_quantity', cycle.quantity)
            partial_quantity = total_quantity * (percentage / 100.0)
            
            logger.info(f"💰 Fermeture partielle {percentage}% de la position {position_type} du cycle {cycle_id}: "
                       f"{partial_quantity:.8f} {cycle.symbol} à {price}")
            
            # Créer l'ordre de fermeture partielle
            client_order_id = f"partial_{cycle_id}_{int(time.time())}"
            
            order = TradeOrder(
                id=client_order_id,
                cycle_id=cycle_id,
                symbol=cycle.symbol,
                side=partial_side,
                type=OrderType.MARKET,
                quantity=partial_quantity,
                price=price,
                status=OrderStatus.NEW,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # Exécuter l'ordre sur Binance
            execution = self.binance_executor.execute_order(order)
            
            if not execution:
                logger.error(f"❌ Échec de l'exécution de la fermeture partielle pour cycle {cycle_id}")
                return False
            
            # Mettre à jour la quantité restante du cycle
            remaining_quantity = total_quantity - execution.quantity
            
            with self.cycles_lock:
                # Mettre à jour les métadonnées du cycle
                if 'partial_SELLs' not in cycle.metadata:
                    cycle.metadata['partial_SELLs'] = []
                
                cycle.metadata['partial_SELLs'].append({
                    'timestamp': datetime.now().isoformat(),
                    'percentage': percentage,
                    'quantity': execution.quantity,
                    'price': execution.price,
                    'reason': reason,
                    'order_id': execution.order_id,
                    'side': partial_side.value,
                    'position_type': position_type
                })
                
                # Mettre à jour la quantité exécutée (maintenant réduite)
                cycle.metadata['executed_quantity'] = remaining_quantity
                cycle.updated_at = datetime.now()
            
            # Sauvegarder l'exécution et le cycle
            self.repository.save_execution(execution, cycle_id)
            self.repository.save_cycle(cycle)
            
            logger.info(f"✅ Fermeture partielle {position_type} réussie: {execution.quantity:.8f} à {execution.price:.6f}, "
                       f"quantité restante: {remaining_quantity:.8f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la fermeture partielle du cycle {cycle_id}: {str(e)}")
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
                    # Vérifier si le cycle est déjà fermé dans la DB
                    db_cycle = self.repository.get_cycle(cycle_id)
                    if db_cycle and db_cycle.status in [CycleStatus.COMPLETED, CycleStatus.CANCELED, CycleStatus.FAILED]:
                        logger.debug(f"✅ Cycle {cycle_id} déjà fermé avec le statut {db_cycle.status}")
                        return True  # Le cycle est déjà fermé, pas d'erreur
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
            
            # Nettoyer les ordres restants sur Binance
            self._cleanup_cycle_orders(cycle)
            
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
    
    def reload_active_cycles(self) -> None:
        """
        Recharge les cycles actifs depuis la base de données.
        Utile pour resynchroniser la mémoire avec la DB.
        """
        logger.info("🔄 Rechargement des cycles actifs depuis la DB...")
        self._load_active_cycles_from_db()
    
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
            # Filtrer les cycles FAILED qui ne devraient pas être là
            cycles = [cycle for cycle in self.active_cycles.values() 
                     if cycle.status not in [CycleStatus.FAILED, CycleStatus.COMPLETED, CycleStatus.CANCELED]]
        
        # Filtrer par symbole
        if symbol:
            cycles = [c for c in cycles if c.symbol == symbol]
        
        # Filtrer par stratégie
        if strategy:
            cycles = [c for c in cycles if c.strategy == strategy]
        
        return cycles
    
    def update_cycle_reinforcement(self, cycle_id: str, additional_quantity: float, 
                                 new_avg_price: float, reinforce_order_id: str, 
                                 metadata: Dict[str, Any] = None) -> bool:
        """
        Met à jour un cycle après un renforcement (DCA).
        
        Args:
            cycle_id: ID du cycle à mettre à jour
            additional_quantity: Quantité supplémentaire ajoutée
            new_avg_price: Nouveau prix moyen après renforcement
            reinforce_order_id: ID de l'ordre de renforcement
            metadata: Métadonnées supplémentaires
            
        Returns:
            True si succès, False sinon
        """
        try:
            with self.cycles_lock:
                cycle = self.active_cycles.get(cycle_id)
                if not cycle:
                    logger.error(f"❌ Cycle {cycle_id} non trouvé dans les cycles actifs")
                    return False
                
                # Mettre à jour les métadonnées
                if not cycle.metadata:
                    cycle.metadata = {}
                
                # Ajouter l'historique de renforcement
                if 'reinforcements' not in cycle.metadata:
                    cycle.metadata['reinforcements'] = []
                
                cycle.metadata['reinforcements'].append({
                    'order_id': reinforce_order_id,
                    'quantity': additional_quantity,
                    'price': new_avg_price,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'metadata': metadata
                })
                
                # Mettre à jour la quantité et le prix moyen
                old_quantity = cycle.quantity
                cycle.quantity += additional_quantity
                cycle.entry_price = new_avg_price
                
                # Sauvegarder en base de données
                with transaction() as cursor:
                    cursor.execute("""
                        UPDATE trade_cycles 
                        SET quantity = %s, 
                            entry_price = %s, 
                            metadata = %s,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (cycle.quantity, cycle.entry_price, 
                         json.dumps(cycle.metadata) if cycle.metadata else '{}', 
                         cycle_id))
                    
                    # Ajouter une entrée dans trade_executions pour tracer le renforcement
                    cursor.execute("""
                        INSERT INTO trade_executions 
                        (cycle_id, order_id, side, symbol, price, quantity, 
                         quote_quantity, fee, fee_asset, status, timestamp, demo, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, 0, 'USDC', %s, %s, %s, %s)
                    """, (
                        cycle_id,
                        reinforce_order_id,
                        cycle.side.value if hasattr(cycle.side, 'value') else str(cycle.side),
                        cycle.symbol,
                        new_avg_price,
                        additional_quantity,
                        new_avg_price * additional_quantity,  # quote_quantity
                        'FILLED',
                        datetime.now(timezone.utc),
                        False,  # demo
                        json.dumps({'type': 'reinforcement', 'original_metadata': metadata})
                    ))
                
                logger.info(f"✅ Cycle {cycle_id} renforcé: {old_quantity:.8f} + {additional_quantity:.8f} = {cycle.quantity:.8f} @ {new_avg_price:.8f}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la mise à jour du renforcement: {str(e)}")
            return False
    
    def process_price_update(self, symbol: str, price: float) -> None:
        """
        Traite une mise à jour de prix pour un symbole.
        Délègue au StopManagerPure pour gérer le trailing stop uniquement.
        
        Args:
            symbol: Symbole mis à jour
            price: Nouveau prix
        """
        # Créer un wrapper pour close_cycle qui indique que c'est un stop
        def close_cycle_by_stop(cycle_id: str, exit_price: Optional[float] = None) -> bool:
            return self.close_cycle(cycle_id, exit_price, is_stop_loss=True)
        
        # Créer un wrapper pour partial_SELL_cycle
        def partial_SELL_by_protection(cycle_id: str, percentage: float, price: float, reason: str) -> bool:
            return self.partial_SELL_cycle(cycle_id, percentage, price, reason)
        
        # Déléguer au StopManagerPure avec les deux wrappers
        self.stop_manager.process_price_update(symbol, price, close_cycle_by_stop, partial_SELL_by_protection)
        
    def _cleanup_cycle_orders(self, cycle: TradeCycle) -> None:
        """
        Nettoie les ordres restants d'un cycle sur Binance.
        
        Args:
            cycle: Le cycle dont les ordres doivent être nettoyés
        """
        try:
            # Vérifier et annuler l'ordre d'entrée s'il existe et n'est pas FILLED
            if cycle.entry_order_id:
                try:
                    entry_status = self.binance_executor.get_order_status(cycle.symbol, cycle.entry_order_id)
                    if entry_status and entry_status.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                        self.binance_executor.cancel_order(cycle.symbol, cycle.entry_order_id)
                        logger.info(f"✅ Ordre d'entrée {cycle.entry_order_id} annulé")
                except Exception as e:
                    logger.debug(f"Ordre d'entrée {cycle.entry_order_id} déjà fermé ou non trouvé: {str(e)}")
            
            # Vérifier et annuler l'ordre de sortie s'il existe et n'est pas FILLED
            if cycle.exit_order_id:
                try:
                    exit_status = self.binance_executor.get_order_status(cycle.symbol, cycle.exit_order_id)
                    if exit_status and exit_status.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                        self.binance_executor.cancel_order(cycle.symbol, cycle.exit_order_id)
                        logger.info(f"✅ Ordre de sortie {cycle.exit_order_id} annulé")
                except Exception as e:
                    logger.debug(f"Ordre de sortie {cycle.exit_order_id} déjà fermé ou non trouvé: {str(e)}")
                    
        except Exception as e:
            logger.warning(f"⚠️ Erreur lors du nettoyage des ordres du cycle {cycle.id}: {str(e)}")       

    def close(self) -> None:
        """
        Ferme proprement le gestionnaire de cycles.
        """
        logger.info("Fermeture du gestionnaire de cycles...")
        logger.info("✅ Gestionnaire de cycles fermé")