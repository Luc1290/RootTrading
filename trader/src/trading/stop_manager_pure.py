# trader/src/trading/stop_manager_pure.py
"""
Gestionnaire de stops PURE - Trailing Stop seulement, pas de target adaptatif.
Utilise la classe TrailingStop pour une logique simple et robuste.
Intègre le GainProtector pour la sécurisation intelligente des gains.
"""
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from threading import RLock

from shared.src.enums import CycleStatus, OrderSide
from shared.src.schemas import TradeCycle
from trader.src.trading.cycle_repository import CycleRepository
from trader.src.utils.trailing_stop import TrailingStop, Side
from trader.src.utils.gain_protector import GainProtector

# Configuration du logging
logger = logging.getLogger(__name__)

class StopManagerPure:
    """
    Gestionnaire de stops PURE.
    Utilise uniquement un trailing stop à 1.5% - pas de target adaptatif. Mode SCALPING.
    """
    
    def __init__(self, cycle_repository: CycleRepository, default_stop_pct: float = 1.5):
        """
        Initialise le gestionnaire de stops pure.
        
        Args:
            cycle_repository: Repository pour les cycles
            default_stop_pct: Pourcentage de stop par défaut (1.5% en mode scalping)
        """
        self.repository = cycle_repository
        self.price_locks = RLock()
        self.default_stop_pct = default_stop_pct
        
        # Cache des TrailingStop par cycle_id
        self.trailing_stops: Dict[str, TrailingStop] = {}
        
        # Intégration du GainProtector
        self.gain_protector = GainProtector()
        
        logger.info(f"✅ StopManagerPure initialisé (stop par défaut: {default_stop_pct}% - MODE SCALPING) avec GainProtector")
    
    def initialize_trailing_stop(self, cycle: TradeCycle) -> TrailingStop:
        """
        Initialise un TrailingStop pour un cycle.
        
        Args:
            cycle: Cycle de trading
            
        Returns:
            Instance TrailingStop
        """
        # Déterminer le side basé sur la logique correcte
        # waiting_SELL = position BUY ouverte (a acheté, attend de vendre)
        # waiting_buy = position SELL ouverte (a vendu, attend de racheter)
        if cycle.status == CycleStatus.WAITING_SELL:
            side = Side.BUY  # Position BUYue ouverte
        elif cycle.status == CycleStatus.WAITING_BUY:
            side = Side.SELL  # Position courte ouverte
        elif cycle.status == CycleStatus.ACTIVE_BUY:
            side = Side.BUY   # En cours d'achat pour position BUYue
        elif cycle.status == CycleStatus.ACTIVE_SELL:
            side = Side.SELL  # En cours de vente pour position courte
        else:
            # Fallback : essayer de déduire du contexte
            logger.warning(f"⚠️ Statut {cycle.status} non reconnu pour {cycle.id}, assume BUY par défaut")
            side = Side.BUY
        
        # Créer le trailing stop
        ts = TrailingStop(
            side=side,
            entry_price=cycle.entry_price,
            stop_pct=self.default_stop_pct
        )
        
        # CORRECTION: Restaurer l'état du TrailingStop après redémarrage
        # Si le cycle a des min_price/max_price sauvegardés différents du prix d'entrée,
        # cela signifie que le trailing stop avait déjà évolué avant le redémarrage.
        # Il faut restaurer cet état pour ne pas perdre les gains du trailing stop.
        
        restored = False
        
        # Debug : afficher les valeurs pour diagnostiquer
        logger.info(f"🔍 Debug restauration {cycle.id}: side={side.name}, "
                   f"entry_price={cycle.entry_price:.6f}, max_price={cycle.max_price:.6f}, "
                   f"min_price={cycle.min_price:.6f}")
        
        if side == Side.BUY and cycle.max_price and cycle.max_price > cycle.entry_price:
            # Position BUY : restaurer le max_price et recalculer le stop
            ts.max_price = cycle.max_price
            new_stop = ts._calc_stop(cycle.max_price)
            if new_stop > ts.stop_price:
                ts.stop_price = new_stop
                restored = True
                logger.info(f"🔄 État TrailingStop BUY restauré pour {cycle.id}: "
                           f"max_price={cycle.max_price:.6f}, stop={ts.stop_price:.6f}")
                
        elif side == Side.SELL and cycle.min_price and cycle.min_price < cycle.entry_price:
            # Position SELL : restaurer le min_price et recalculer le stop  
            ts.min_price = cycle.min_price
            new_stop = ts._calc_stop(cycle.min_price)
            if new_stop < ts.stop_price:
                ts.stop_price = new_stop
                restored = True
                logger.info(f"🔄 État TrailingStop SELL restauré pour {cycle.id}: "
                           f"min_price={cycle.min_price:.6f}, stop={ts.stop_price:.6f}")
        
        if not restored:
            logger.info(f"🎯 Nouveau TrailingStop initialisé pour {cycle.id}: "
                       f"{side.name} @ {cycle.entry_price:.6f}, stop @ {ts.stop_price:.6f}")
        
        # Stocker dans le cache
        self.trailing_stops[cycle.id] = ts
        
        # Initialiser le GainProtector pour ce cycle
        side_str = "BUY" if side == Side.BUY else "SELL"
        self.gain_protector.initialize_cycle(
            cycle_id=cycle.id,
            entry_price=cycle.entry_price,
            side=side_str,
            quantity=cycle.quantity or 0.0
        )
        
        # Mettre à jour le cycle avec le stop (initial ou restauré)
        cycle.stop_price = ts.stop_price
        cycle.trailing_delta = self.default_stop_pct
        
        # Pour les restaurations, aussi mettre à jour les extremes
        if restored:
            if side == Side.BUY:
                cycle.max_price = ts.max_price
            else:
                cycle.min_price = ts.min_price
        
        self.repository.save_cycle(cycle)
        
        return ts

    def process_price_update(self, symbol: str, price: float, close_cycle_callback: Callable[[str, Optional[float]], bool], partial_SELL_callback: Optional[Callable[[str, float, float, str], bool]] = None) -> None:
        """
        Traite une mise à jour de prix - VERSION PURE avec GainProtector.
        Utilise le TrailingStop + protection intelligente des gains.
        
        Args:
            symbol: Symbole mis à jour
            price: Nouveau prix
            close_cycle_callback: Fonction de rappel pour fermer un cycle
            partial_SELL_callback: Fonction de rappel pour les ventes partielles (cycle_id, percentage, price, reason)
        """
        # Récupérer les cycles actifs pour ce symbole
        cycles = self.repository.get_active_cycles(symbol=symbol)
        
        with self.price_locks:
            stops_to_trigger = []
            partial_SELLs_to_execute = []
            
            for cycle in cycles:
                # Vérifier si on a un TrailingStop pour ce cycle
                if cycle.id not in self.trailing_stops:
                    # Créer le TrailingStop s'il n'existe pas (inclut l'initialisation du GainProtector)
                    logger.debug(f"🔧 Initialisation trailing stop manquant pour cycle {cycle.id}")
                    self.initialize_trailing_stop(cycle)
                
                ts = self.trailing_stops[cycle.id]
                
                # 1. Vérifier les protections de gains AVANT le trailing stop
                protection_actions = self.gain_protector.update_and_check_protections(cycle.id, price)
                
                for action in protection_actions:
                    if action['action'] == 'SELL_partial':
                        partial_SELLs_to_execute.append({
                            'cycle_id': cycle.id,
                            'action': action,
                            'price': price
                        })
                        logger.info(f"💰 Take profit partiel {action['percentage']}% pour cycle {cycle.id} à {price}")
                    
                    elif action['action'] in ['SELL_all', 'cover_all']:
                        stops_to_trigger.append(cycle.id)
                        logger.info(f"🚨 {action['reason']} pour cycle {cycle.id}: fermeture complète à {price}")
                        continue  # Pas besoin de vérifier le trailing stop
                    
                    elif action['action'] == 'update_stop':
                        # Mettre à jour le stop loss via le TrailingStop
                        new_stop = action['new_stop_price']
                        ts.stop_price = new_stop
                        cycle.stop_price = new_stop
                        self.repository.save_cycle(cycle)
                        logger.info(f"🛡️ Stop déplacé à {new_stop} pour cycle {cycle.id} ({action['reason']})")
                
                # 2. Si pas d'action de fermeture complète, vérifier le trailing stop normal
                if cycle.id not in [item['cycle_id'] for item in partial_SELLs_to_execute if item['action']['action'] in ['SELL_all', 'cover_all']]:
                    stop_hit = ts.update(price)
                    
                    if stop_hit:
                        # Stop déclenché !
                        stops_to_trigger.append(cycle.id)
                        profit = ts.get_profit_if_exit_now(price)
                        logger.info(f"🔴 Stop trailing déclenché pour cycle {cycle.id}: "
                                   f"prix {price:.6f} ≤ stop {ts.stop_price:.6f}, "
                                   f"profit: {profit:+.6f}%")
                    else:
                        # Mettre à jour le cycle avec le nouveau stop_price
                        if ts.stop_price != cycle.stop_price:
                            old_stop = cycle.stop_price
                            cycle.stop_price = ts.stop_price
                            cycle.max_price = ts.max_price if ts.side == Side.BUY else cycle.max_price
                            cycle.min_price = ts.min_price if ts.side == Side.SELL else cycle.min_price
                            
                            self.repository.save_cycle(cycle)
                            
                            logger.debug(f"📈 Stop trailing mis à jour pour cycle {cycle.id}: "
                                       f"{old_stop:.6f} → {ts.stop_price:.6f}")
            
            # Exécuter les ventes partielles
            for partial_SELL in partial_SELLs_to_execute:
                if partial_SELL_callback:
                    action = partial_SELL['action']
                    success = partial_SELL_callback(
                        partial_SELL['cycle_id'],
                        action['percentage'],
                        partial_SELL['price'],
                        action['reason']
                    )
                    if success:
                        logger.info(f"✅ Vente partielle {action['percentage']}% exécutée pour cycle {partial_SELL['cycle_id']}")
                    else:
                        logger.error(f"❌ Échec vente partielle pour cycle {partial_SELL['cycle_id']}")
                else:
                    logger.warning(f"⚠️ Vente partielle demandée mais callback non fourni: {partial_SELL}")
            
            # Déclencher les stops en dehors du verrou
            for cycle_id in stops_to_trigger:
                success = close_cycle_callback(cycle_id, price)
                if success:
                    # Nettoyer les caches
                    if cycle_id in self.trailing_stops:
                        del self.trailing_stops[cycle_id]
                    self.gain_protector.cleanup_cycle(cycle_id)
                    logger.info(f"✅ Cycle {cycle_id} fermé par protection/stop-loss")
                else:
                    logger.error(f"❌ Échec fermeture cycle {cycle_id} par protection/stop-loss")

    def get_cycle_status(self, cycle_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère le statut d'un TrailingStop.
        
        Args:
            cycle_id: ID du cycle
            
        Returns:
            Statut du trailing stop ou None
        """
        if cycle_id in self.trailing_stops:
            ts = self.trailing_stops[cycle_id]
            return ts.get_status()
        return None

    def get_cycle_stops(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Récupère les informations de stop pour un symbole.
        
        Args:
            symbol: Symbole à analyser
            
        Returns:
            Liste des stops actifs
        """
        cycles = self.repository.get_active_cycles(symbol=symbol)
        stops_info = []
        
        for cycle in cycles:
            if cycle.id in self.trailing_stops:
                ts = self.trailing_stops[cycle.id]
                status = ts.get_status()
                status.update({
                    'cycle_id': cycle.id,
                    'symbol': cycle.symbol,
                    'strategy': cycle.strategy,
                    'status': cycle.status.value if hasattr(cycle.status, 'value') else str(cycle.status)
                })
                stops_info.append(status)
        
        return stops_info

    def cleanup_cycle(self, cycle_id: str) -> None:
        """
        Nettoie les ressources pour un cycle fermé.
        
        Args:
            cycle_id: ID du cycle à nettoyer
        """
        if cycle_id in self.trailing_stops:
            del self.trailing_stops[cycle_id]
            logger.debug(f"🧹 TrailingStop nettoyé pour cycle {cycle_id}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Récupère les statistiques du gestionnaire.
        
        Returns:
            Statistiques du stop manager
        """
        return {
            'active_trailing_stops': len(self.trailing_stops),
            'default_stop_pct': self.default_stop_pct,
            'cycles': list(self.trailing_stops.keys())
        }