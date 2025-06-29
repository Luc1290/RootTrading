# trader/src/trading/stop_manager_pure.py
"""
Gestionnaire de stops PURE - Trailing Stop seulement, pas de target adaptatif.
Utilise la classe TrailingStop pour une logique simple et robuste.
Intègre le GainProtector pour la sécurisation intelligente des gains.
Version adaptative basée sur l'ATR pour des stops plus intelligents.
"""
import logging
import requests
import time
from datetime import datetime
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
    Utilise uniquement un trailing stop à 2.5% - pas de target adaptatif. MODIFIÉ pour éviter fausses sorties.
    """
    
    def __init__(self, cycle_repository: CycleRepository, default_stop_pct: float = 2.5, atr_multiplier: float = 1.5, min_stop_pct: float = 2.0, analyzer_url: str = "http://analyzer:5012"):
        """
        Initialise le gestionnaire de stops pure.
        
        Args:
            cycle_repository: Repository pour les cycles
            default_stop_pct: Pourcentage de stop par défaut (2.5% - fallback si pas d'ATR)
            atr_multiplier: Multiplicateur ATR pour calcul adaptatif (défaut: 1.5)
            min_stop_pct: Pourcentage minimum de stop (défaut: 2.0%)
            analyzer_url: URL du service analyzer pour récupérer l'ATR
        """
        self.repository = cycle_repository
        self.price_locks = RLock()
        self.default_stop_pct = default_stop_pct
        self.atr_multiplier = atr_multiplier
        self.min_stop_pct = min_stop_pct
        self.analyzer_url = analyzer_url
        
        # Cache des TrailingStop par cycle_id
        self.trailing_stops: Dict[str, TrailingStop] = {}
        
        # Cache ATR pour éviter les appels répétés
        self.atr_cache: Dict[str, Dict[str, Any]] = {}  # {symbol: {atr: float, timestamp: float, price_moves: int}}
        self.atr_cache_ttl = 30  # 30 secondes pour scalping
        self.atr_force_update_moves = 20  # Force update après 20 mouvements de prix
        
        # Intégration du GainProtector
        self.gain_protector = GainProtector()
        
        logger.info(f"✅ StopManagerPure initialisé (stop défaut: {default_stop_pct}%, ATR: {atr_multiplier}x, min: {min_stop_pct}%) avec GainProtector")
    
    def _get_current_atr(self, symbol: str, force_update: bool = False) -> Optional[float]:
        """
        Récupère l'ATR actuel pour un symbole depuis l'analyzer.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            force_update: Force la mise à jour même si en cache
            
        Returns:
            Valeur ATR ou None si indisponible
        """
        current_time = time.time()
        
        # Vérifier le cache d'abord (sauf si force_update)
        if not force_update and symbol in self.atr_cache:
            cache_entry = self.atr_cache[symbol]
            time_since_cache = current_time - cache_entry['timestamp']
            price_moves = cache_entry.get('price_moves', 0)
            
            # Utiliser le cache si dans les temps ET pas trop de mouvements
            if (time_since_cache < self.atr_cache_ttl and 
                price_moves < self.atr_force_update_moves):
                logger.debug(f"📊 ATR cache hit pour {symbol}: {cache_entry['atr']:.6f} (moves: {price_moves})")
                return cache_entry['atr']
            elif price_moves >= self.atr_force_update_moves:
                logger.debug(f"📊 ATR force update pour {symbol} après {price_moves} mouvements")
        
        # Cache expiré ou inexistant, récupérer depuis l'analyzer
        try:
            # Construire l'URL pour récupérer l'ATR
            url = f"{self.analyzer_url}/api/indicators/{symbol}"
            
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                
                # Chercher l'ATR dans la réponse
                atr_value = None
                if 'atr' in data:
                    atr_value = data['atr']
                elif 'indicators' in data and 'atr' in data['indicators']:
                    atr_value = data['indicators']['atr']
                elif 'technical_analysis' in data:
                    # Peut-être dans une structure plus complexe
                    ta_data = data['technical_analysis']
                    if 'atr' in ta_data:
                        atr_value = ta_data['atr']
                
                if atr_value is not None:
                    # Mettre en cache avec compteur de mouvements remis à zéro
                    self.atr_cache[symbol] = {
                        'atr': atr_value,
                        'timestamp': current_time,
                        'price_moves': 0
                    }
                    logger.debug(f"📊 ATR récupéré depuis analyzer pour {symbol}: {atr_value:.6f}")
                    return atr_value
                else:
                    logger.warning(f"⚠️ ATR non trouvé dans la réponse analyzer pour {symbol}: {data}")
            else:
                logger.warning(f"⚠️ Erreur HTTP {response.status_code} lors de la récupération ATR pour {symbol}")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"⚠️ Erreur réseau lors de la récupération ATR pour {symbol}: {e}")
        except Exception as e:
            logger.warning(f"⚠️ Erreur inattendue lors de la récupération ATR pour {symbol}: {e}")
        
        return None
    
    def initialize_trailing_stop(self, cycle: TradeCycle) -> TrailingStop:
        """
        Initialise un TrailingStop pour un cycle.
        
        Args:
            cycle: Cycle de trading
            
        Returns:
            Instance TrailingStop
        """
        # Utiliser le side directement depuis le cycle (plus fiable)
        if hasattr(cycle, 'side') and cycle.side:
            # Convertir OrderSide vers Side
            side = Side.BUY if cycle.side == OrderSide.BUY else Side.SELL
        else:
            # Fallback : déduire depuis le statut (pour compatibilité avec anciens cycles)
            if cycle.status == CycleStatus.WAITING_SELL:
                side = Side.BUY  # Position BUY ouverte (a acheté, attend de vendre)
            elif cycle.status == CycleStatus.WAITING_BUY:
                side = Side.SELL  # Position courte ouverte (a vendu, attend de racheter)
            elif cycle.status == CycleStatus.ACTIVE_BUY:
                side = Side.BUY   # En cours d'achat pour position longue
            elif cycle.status == CycleStatus.ACTIVE_SELL:
                side = Side.SELL  # En cours de vente pour position courte
            else:
                # Fallback : essayer de déduire du contexte
                logger.warning(f"⚠️ Statut {cycle.status} non reconnu pour {cycle.id}, assume BUY par défaut")
                side = Side.BUY
        
        # Restaurer les paramètres ATR depuis les métadonnées si disponibles
        atr_config = None
        if cycle.metadata and 'atr_config' in cycle.metadata:
            atr_config = cycle.metadata['atr_config']
            logger.debug(f"🔄 Configuration ATR restaurée depuis DB pour {cycle.id}: {atr_config}")
        
        # Récupérer l'ATR actuel pour ce symbole (priorité sur celui en cache)
        current_atr = self._get_current_atr(cycle.symbol)
        
        # Utiliser ATR restauré si pas de nouveau disponible
        if current_atr is None and atr_config and atr_config.get('atr_value'):
            current_atr = atr_config['atr_value']
            logger.info(f"📊 ATR restauré depuis DB pour {cycle.symbol}: {current_atr:.6f}")
        
        # Créer le trailing stop avec paramètres ATR (restaurés ou par défaut)
        atr_multiplier = atr_config.get('atr_multiplier', self.atr_multiplier) if atr_config else self.atr_multiplier
        min_stop_pct = atr_config.get('min_stop_pct', self.min_stop_pct) if atr_config else self.min_stop_pct
        
        ts = TrailingStop(
            side=side,
            entry_price=cycle.entry_price,
            stop_pct=self.default_stop_pct,
            atr_multiplier=atr_multiplier,
            min_stop_pct=min_stop_pct
        )
        
        # Appliquer l'ATR si disponible
        if current_atr is not None:
            ts.update_atr(current_atr)
            logger.info(f"🧮 ATR appliqué pour {cycle.symbol}: {current_atr:.6f} (multiplier: {atr_multiplier}x)")
        else:
            logger.info(f"⚠️ ATR indisponible pour {cycle.symbol}, utilisation stop fixe: {self.default_stop_pct}%")
        
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
        
        # Sauvegarder les métadonnées ATR dans le cycle
        if cycle.metadata is None:
            cycle.metadata = {}
        
        cycle.metadata['atr_config'] = {
            'atr_value': ts.current_atr,
            'atr_multiplier': ts.atr_multiplier,
            'min_stop_pct': ts.min_stop_pct,
            'is_atr_based': ts.current_atr is not None,
            'effective_stop_pct': ts._get_effective_stop_percentage(ts.stop_price) if hasattr(ts, '_get_effective_stop_percentage') else None,
            'stop_calculation_method': 'atr_adaptive' if ts.current_atr is not None else 'fixed_percentage',
            'last_updated': datetime.now().isoformat()
        }
        
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
                
                # Incrémenter le compteur de mouvements pour ce symbole
                if cycle.symbol in self.atr_cache:
                    self.atr_cache[cycle.symbol]['price_moves'] += 1
                
                # Déterminer si on doit forcer la mise à jour ATR
                force_atr_update = False
                if cycle.symbol in self.atr_cache:
                    moves = self.atr_cache[cycle.symbol]['price_moves']
                    if moves >= self.atr_force_update_moves:
                        force_atr_update = True
                
                # Mettre à jour l'ATR (30s max OU après 20 mouvements)
                current_atr = self._get_current_atr(cycle.symbol, force_update=force_atr_update)
                if current_atr is not None and current_atr != ts.current_atr:
                    ts.update_atr(current_atr)
                    logger.debug(f"📊 ATR mis à jour pour cycle {cycle.id}: {current_atr:.6f}")
                
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
                        # CORRECTION: Toujours mettre à jour les extremes de prix
                        extremes_updated = False
                        
                        # Mettre à jour max_price pour les positions BUY
                        if ts.side == Side.BUY:
                            if cycle.max_price is None or ts.max_price > cycle.max_price:
                                old_max = cycle.max_price
                                cycle.max_price = ts.max_price
                                extremes_updated = True
                                logger.debug(f"📈 Nouveau max_price pour cycle {cycle.id}: {old_max} → {ts.max_price:.6f}")
                        
                        # Mettre à jour min_price pour les positions SELL  
                        if ts.side == Side.SELL:
                            if cycle.min_price is None or ts.min_price < cycle.min_price:
                                old_min = cycle.min_price
                                cycle.min_price = ts.min_price
                                extremes_updated = True
                                logger.debug(f"📉 Nouveau min_price pour cycle {cycle.id}: {old_min} → {ts.min_price:.6f}")
                        
                        # Mettre à jour le cycle avec le nouveau stop_price
                        stop_price_changed = ts.stop_price != cycle.stop_price
                        atr_changed = current_atr != ts.current_atr
                        
                        # Calculer si le changement de stop est significatif (> 0.01%)
                        significant_stop_change = False
                        if stop_price_changed and cycle.stop_price and cycle.stop_price > 0:
                            change_pct = abs(ts.stop_price - cycle.stop_price) / cycle.stop_price * 100
                            significant_stop_change = change_pct > 0.01  # Plus de 0.01%
                        
                        if stop_price_changed or atr_changed or extremes_updated:
                            old_stop = cycle.stop_price
                            cycle.stop_price = ts.stop_price
                            
                            # Mettre à jour les métadonnées ATR si changement
                            if atr_changed or cycle.metadata is None or 'atr_config' not in cycle.metadata:
                                if cycle.metadata is None:
                                    cycle.metadata = {}
                                
                                cycle.metadata['atr_config'] = {
                                    'atr_value': ts.current_atr,
                                    'atr_multiplier': ts.atr_multiplier,
                                    'min_stop_pct': ts.min_stop_pct,
                                    'is_atr_based': ts.current_atr is not None,
                                    'effective_stop_pct': ts._get_effective_stop_percentage(price),
                                    'stop_calculation_method': 'atr_adaptive' if ts.current_atr is not None else 'fixed_percentage',
                                    'last_updated': datetime.now().isoformat()
                                }
                            
                            # CORRECTION: Sauvegarder en DB si les extremes ou stops changent significativement
                            if significant_stop_change or atr_changed or extremes_updated:
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
        # Compter les stops basés sur ATR vs fixes
        atr_based_count = sum(1 for ts in self.trailing_stops.values() if ts.current_atr is not None)
        fixed_count = len(self.trailing_stops) - atr_based_count
        
        # Stats par symbole
        atr_stats = {}
        for symbol, cache_data in self.atr_cache.items():
            current_time = time.time()
            age_seconds = current_time - cache_data['timestamp']
            atr_stats[symbol] = {
                'atr': cache_data['atr'],
                'age_seconds': age_seconds,
                'price_moves': cache_data.get('price_moves', 0),
                'needs_update': age_seconds > self.atr_cache_ttl or cache_data.get('price_moves', 0) >= self.atr_force_update_moves
            }
        
        return {
            'active_trailing_stops': len(self.trailing_stops),
            'default_stop_pct': self.default_stop_pct,
            'atr_multiplier': self.atr_multiplier,
            'min_stop_pct': self.min_stop_pct,
            'atr_based_stops': atr_based_count,
            'fixed_stops': fixed_count,
            'atr_cache_ttl': self.atr_cache_ttl,
            'atr_force_update_moves': self.atr_force_update_moves,
            'atr_cache_entries': len(self.atr_cache),
            'atr_stats_by_symbol': atr_stats,
            'cycles': list(self.trailing_stops.keys())
        }