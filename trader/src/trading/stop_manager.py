# trader/src/trading/stop_manager.py
"""
Gestionnaire des stops et targets pour les cycles de trading.
S'occupe de surveiller les prix et de déclencher les fermetures de cycles.
"""
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from threading import RLock

from shared.src.enums import CycleStatus
from shared.src.schemas import TradeCycle
from trader.src.trading.cycle_repository import CycleRepository

# Configuration du logging
logger = logging.getLogger(__name__)

class StopManager:
    """
    Gestionnaire des stops et targets.
    Surveille les prix et déclenche les fermetures de cycles.
    """
    
    def __init__(self, cycle_repository: CycleRepository):
        """
        Initialise le gestionnaire de stops.
        
        Args:
            cycle_repository: Repository pour les cycles
        """
        self.repository = cycle_repository
        self.price_locks = RLock()  # Verrou pour les mises à jour de prix
        self.cycles_cache = {}  # Cache des cycles par symbole
        logger.info("✅ StopManager initialisé")
    
    def process_price_update(self, symbol: str, price: float, close_cycle_callback: Callable[[str, Optional[float]], bool]) -> None:
        """
        Traite une mise à jour de prix pour un symbole.
        Vérifie les stops, targets et met à jour les trailing stops.
        
        Args:
            symbol: Symbole mis à jour
            price: Nouveau prix
            close_cycle_callback: Fonction de rappel pour fermer un cycle
        """
        # Récupérer les cycles pertinents pour ce symbole
        cycles = self.repository.get_active_cycles(symbol=symbol)
        
        # Mettre à jour le cache
        with self.price_locks:
            self.cycles_cache[symbol] = cycles
            
        # Analyser les cycles sans verrou
        stops_to_trigger = []
        targets_to_trigger = []
        trailing_updates = []
        
        for cycle in cycles:
            # PROTECTION AJOUTÉE: Skip si le cycle est déjà terminé
            if cycle.status == CycleStatus.COMPLETED:
                logger.debug(f"⛔ Skip du stop/target check, cycle {cycle.id} déjà terminé.")
                continue
                
            # Vérifier les stop-loss
            if cycle.stop_price is not None:
                if ((cycle.status in [CycleStatus.ACTIVE_BUY, CycleStatus.WAITING_SELL] and 
                    price <= cycle.stop_price) or
                    (cycle.status in [CycleStatus.ACTIVE_SELL, CycleStatus.WAITING_BUY] and 
                    price >= cycle.stop_price)):
                    stops_to_trigger.append(cycle.id)
            
            # Vérifier les target-price
            if cycle.target_price is not None:
                if ((cycle.status in [CycleStatus.ACTIVE_BUY, CycleStatus.WAITING_SELL] and 
                    price >= cycle.target_price) or
                    (cycle.status in [CycleStatus.ACTIVE_SELL, CycleStatus.WAITING_BUY] and 
                    price <= cycle.target_price)):
                    targets_to_trigger.append(cycle.id)
            
            # Préparer les updates de trailing stop
            if cycle.trailing_delta is not None:
                if (cycle.status in [CycleStatus.ACTIVE_BUY, CycleStatus.WAITING_SELL] and
                    (not hasattr(cycle, 'max_price') or price > cycle.max_price)):
                    trailing_updates.append({
                        'id': cycle.id,
                        'type': 'max',
                        'value': price
                    })
                elif (cycle.status in [CycleStatus.ACTIVE_SELL, CycleStatus.WAITING_BUY] and
                      (not hasattr(cycle, 'min_price') or price < cycle.min_price)):
                    trailing_updates.append({
                        'id': cycle.id,
                        'type': 'min',
                        'value': price
                    })
        
        # Appliquer les trailing updates (nécessite moins de verrous que fermer des cycles)
        for update in trailing_updates:
            self._update_trailing_stop(update['id'], update['type'], update['value'])
        
        # Déclencher les stops et targets (en évitant les doublons)
        cycles_to_close = set()
        close_reasons = {}
        
        for cycle_id in stops_to_trigger:
            cycles_to_close.add(cycle_id)
            close_reasons[cycle_id] = "stop-loss"
            
        for cycle_id in targets_to_trigger:
            if cycle_id not in cycles_to_close:
                cycles_to_close.add(cycle_id)
                close_reasons[cycle_id] = "target"
            else:
                # Le cycle est déjà marqué pour fermeture (stop-loss a priorité)
                logger.debug(f"⚠️ Cycle {cycle_id} déclenche à la fois stop-loss et target, stop-loss prioritaire")
        
        # Fermer chaque cycle une seule fois
        for cycle_id in cycles_to_close:
            reason = close_reasons[cycle_id]
            if reason == "stop-loss":
                logger.info(f"🔴 Stop-loss déclenché pour le cycle {cycle_id} au prix {price}")
            else:
                logger.info(f"🎯 Prix cible atteint pour le cycle {cycle_id} au prix {price}")
            close_cycle_callback(cycle_id, price)
    
    def _update_trailing_stop(self, cycle_id: str, update_type: str, price_value: float) -> None:
        """
        Met à jour un trailing stop pour un cycle spécifique.
        
        Args:
            cycle_id: ID du cycle
            update_type: Type de mise à jour ('max' ou 'min')
            price_value: Nouvelle valeur de prix
        """
        # Récupérer le cycle
        cycle = self.repository.get_cycle(cycle_id)
        if not cycle or cycle.trailing_delta is None:
            return
        
        if update_type == 'max':
            # Mise à jour du prix maximum
            if not hasattr(cycle, 'max_price'):
                cycle.max_price = price_value
            else:
                cycle.max_price = max(cycle.max_price, price_value)
            
            # Calcul du nouveau stop-loss trailing
            new_stop = cycle.max_price * (1 - cycle.trailing_delta / 100)
            
            # Mise à jour du stop-loss si plus haut que l'ancien
            if cycle.stop_price is None or new_stop > cycle.stop_price:
                old_stop = cycle.stop_price
                cycle.stop_price = new_stop
                
                # Enregistrer la mise à jour en DB
                if self.repository.save_cycle(cycle):
                    logger.info(f"🔄 Trailing stop mis à jour pour le cycle {cycle_id}: {old_stop} → {new_stop}")
        
        elif update_type == 'min':
            # Mise à jour du prix minimum
            if not hasattr(cycle, 'min_price'):
                cycle.min_price = price_value
            else:
                cycle.min_price = min(cycle.min_price, price_value)
            
            # Calcul du nouveau stop-loss trailing
            new_stop = cycle.min_price * (1 + cycle.trailing_delta / 100)
            
            # Mise à jour du stop-loss si plus bas que l'ancien
            if cycle.stop_price is None or new_stop < cycle.stop_price:
                old_stop = cycle.stop_price
                cycle.stop_price = new_stop
                
                # Enregistrer la mise à jour en DB
                if self.repository.save_cycle(cycle):
                    logger.info(f"🔄 Trailing stop mis à jour pour le cycle {cycle_id}: {old_stop} → {new_stop}")
    
    def get_cycle_stops(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Récupère les informations de stop pour un symbole.
        Utile pour le monitoring et le débogage.
        
        Args:
            symbol: Symbole à vérifier
            
        Returns:
            Liste des informations de stop pour les cycles actifs
        """
        # Récupérer les cycles actifs pour ce symbole
        with self.price_locks:
            if symbol in self.cycles_cache:
                cycles = self.cycles_cache[symbol]
            else:
                cycles = self.repository.get_active_cycles(symbol=symbol)
                self.cycles_cache[symbol] = cycles
        
        # Collecter les informations de stop
        stops_info = []
        for cycle in cycles:
            # Ne prendre que les cycles avec des stops définis
            if cycle.stop_price is not None or cycle.target_price is not None:
                info = {
                    'cycle_id': cycle.id,
                    'symbol': cycle.symbol,
                    'status': cycle.status.value if hasattr(cycle.status, 'value') else str(cycle.status),
                    'entry_price': cycle.entry_price,
                    'stop_price': cycle.stop_price,
                    'target_price': cycle.target_price,
                    'trailing_delta': cycle.trailing_delta,
                }
                
                # Ajouter les infos de trailing si disponibles
                if hasattr(cycle, 'max_price'):
                    info['max_price'] = cycle.max_price
                if hasattr(cycle, 'min_price'):
                    info['min_price'] = cycle.min_price
                
                stops_info.append(info)
        
        return stops_info