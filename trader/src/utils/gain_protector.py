"""
Système de protection des gains inspiré des recommandations de ChatGPT.
Implémente le take profit partiel et le trailing stop intelligent.
"""

import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


class GainProtectionLevel(Enum):
    """Niveaux de protection des gains"""
    NONE = "none"
    BREAKEVEN = "breakeven"
    PARTIAL_SECURED = "partial_secured"
    TRAILING_ACTIVE = "trailing_active"
    FINAL_PROTECTION = "final_protection"


@dataclass
class ProtectionTarget:
    """Configuration d'un target de protection"""
    gain_threshold: float  # Seuil de gain pour déclencher (en %)
    take_profit_percentage: float  # % de la position à fermer
    new_stop_percentage: float  # Nouveau stop loss (en % du prix d'entrée)
    trailing_distance: Optional[float] = None  # Distance de trailing (en %)


class GainProtector:
    """
    Gestionnaire intelligent de protection des gains.
    
    Stratégie basée sur les recommandations de ChatGPT :
    1. Take profit partiel à certains seuils
    2. Trailing stop intelligent avec activation conditionnelle
    3. Protection break-even automatique
    4. Sécurisation progressive des gains
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configuration des niveaux de protection
        self.protection_targets = {
            # Niveau 1: Sécurisation break-even dès +1%
            1: ProtectionTarget(
                gain_threshold=1.0,
                take_profit_percentage=0.0,  # Pas de vente, juste déplacer le stop
                new_stop_percentage=0.0,     # Stop au break-even
                trailing_distance=None
            ),
            
            # Niveau 2: Premier take profit partiel à +2%
            2: ProtectionTarget(
                gain_threshold=2.0,
                take_profit_percentage=30.0,  # Vendre 30% de la position
                new_stop_percentage=0.5,      # Stop à +0.5% (sécurisé)
                trailing_distance=1.0         # Activer trailing à 1%
            ),
            
            # Niveau 3: Deuxième take profit à +4%
            3: ProtectionTarget(
                gain_threshold=4.0,
                take_profit_percentage=30.0,  # Vendre 30% supplémentaire
                new_stop_percentage=1.5,      # Stop à +1.5%
                trailing_distance=0.8         # Trailing plus serré
            ),
            
            # Niveau 4: Protection finale à +6%
            4: ProtectionTarget(
                gain_threshold=6.0,
                take_profit_percentage=0.0,   # Garder le reste
                new_stop_percentage=3.0,      # Stop à +3%
                trailing_distance=0.5         # Trailing très serré
            )
        }
        
        # État de chaque cycle
        self.cycle_states: Dict[str, Dict] = {}
    
    def initialize_cycle(self, cycle_id: str, entry_price: float, side: str, quantity: float):
        """
        Initialise la protection pour un nouveau cycle.
        
        Args:
            cycle_id: ID unique du cycle
            entry_price: Prix d'entrée
            side: LONG ou SHORT
            quantity: Quantité initiale
        """
        self.cycle_states[cycle_id] = {
            'entry_price': entry_price,
            'side': side,
            'initial_quantity': quantity,
            'remaining_quantity': quantity,
            'current_level': 0,
            'max_gain_reached': 0.0,
            'levels_triggered': set(),
            'trailing_active': False,
            'trailing_high': entry_price if side == 'LONG' else entry_price,
            'break_even_set': False,
            'created_at': time.time()
        }

        self.logger.info(f"🛡️ Protection initialisée pour cycle {cycle_id}: {side} @ {entry_price}")

    def update_and_check_protections(self, cycle_id: str, current_price: float) -> List[Dict]:
        """
        Met à jour les protections et retourne les actions à exécuter.
        
        Args:
            cycle_id: ID du cycle
            current_price: Prix actuel
            
        Returns:
            Liste des actions à exécuter: [{'action': 'sell_partial', 'percentage': 30, ...}, ...]
        """
        if cycle_id not in self.cycle_states:
            self.logger.warning(f"Cycle {cycle_id} non initialisé dans GainProtector")
            return []
        
        state = self.cycle_states[cycle_id]
        actions = []
        
        # Calculer le gain actuel
        if state['side'] == 'LONG':
            gain_pct = ((current_price - state['entry_price']) / state['entry_price']) * 100
        else:  # SHORT
            gain_pct = ((state['entry_price'] - current_price) / state['entry_price']) * 100
        
        # Mettre à jour le gain maximum
        if gain_pct > state['max_gain_reached']:
            state['max_gain_reached'] = gain_pct
            
            # Mettre à jour le trailing high/low
            if state['side'] == 'LONG':
                state['trailing_high'] = max(state['trailing_high'], current_price)
            else: # SHORT
                # Pour un SHORT, on suit le plus bas
                state['trailing_high'] = min(state['trailing_high'], current_price)
        
        # Vérifier chaque niveau de protection
        for level, target in self.protection_targets.items():
            if level not in state['levels_triggered'] and gain_pct >= target.gain_threshold:
                actions.extend(self._trigger_protection_level(cycle_id, level, target, current_price))
                state['levels_triggered'].add(level)
        
        # Vérifier le trailing stop si actif
        if state['trailing_active']:
            trailing_action = self._check_trailing_stop(cycle_id, current_price)
            if trailing_action:
                actions.append(trailing_action)
        
        return actions
    
    def _trigger_protection_level(self, cycle_id: str, level: int, target: ProtectionTarget, current_price: float) -> List[Dict]:
        """
        Déclenche un niveau de protection spécifique.
        
        Args:
            cycle_id: ID du cycle
            level: Niveau de protection
            target: Configuration du target
            current_price: Prix actuel
            
        Returns:
            Liste des actions à exécuter
        """
        state = self.cycle_states[cycle_id]
        actions = []
        
        self.logger.info(f"🎯 Déclenchement niveau {level} pour cycle {cycle_id} (gain: {state['max_gain_reached']:.2f}%)")
        
        # Take profit partiel si configuré
        if target.take_profit_percentage > 0:
            sell_quantity = (state['remaining_quantity'] * target.take_profit_percentage) / 100
            actions.append({
                'action': 'sell_partial',
                'quantity': sell_quantity,
                'percentage': target.take_profit_percentage,
                'reason': f'take_profit_level_{level}',
                'price': current_price
            })
            
            # Mettre à jour la quantité restante
            state['remaining_quantity'] -= sell_quantity
            
            self.logger.info(f"💰 Take profit {target.take_profit_percentage}% au niveau {level} - Reste: {state['remaining_quantity']:.6f}")
        
        # Déplacer le stop loss
        if target.new_stop_percentage is not None:
            new_stop_price = self._calculate_stop_price(state['entry_price'], target.new_stop_percentage, state['side'])
            actions.append({
                'action': 'update_stop',
                'new_stop_price': new_stop_price,
                'stop_percentage': target.new_stop_percentage,
                'reason': f'protection_level_{level}'
            })
            
            self.logger.info(f"🛡️ Stop déplacé à {new_stop_price} (+{target.new_stop_percentage}%) - Niveau {level}")
        
        # Activer le trailing stop si configuré
        if target.trailing_distance is not None:
            state['trailing_active'] = True
            state['trailing_distance'] = target.trailing_distance
            
            self.logger.info(f"📈 Trailing stop activé avec distance {target.trailing_distance}% - Niveau {level}")
        
        return actions
    
    def _check_trailing_stop(self, cycle_id: str, current_price: float) -> Optional[Dict]:
        """
        Vérifie si le trailing stop doit être déclenché.
        
        Args:
            cycle_id: ID du cycle
            current_price: Prix actuel
            
        Returns:
            Action de stop si nécessaire
        """
        state = self.cycle_states[cycle_id]
        
        if not state['trailing_active']:
            return None
        
        trailing_distance = state['trailing_distance']

        if state['side'] == 'LONG':
            # Pour un LONG, vérifier si le prix a baissé de X% depuis le high
            price_drop_pct = ((state['trailing_high'] - current_price) / state['trailing_high']) * 100
            
            if price_drop_pct >= trailing_distance:
                self.logger.info(f"🚨 Trailing stop déclenché pour cycle {cycle_id}: baisse de {price_drop_pct:.2f}% depuis {state['trailing_high']}")
                
                return {
                    'action': 'sell_all',
                    'reason': 'trailing_stop_triggered',
                    'price': current_price,
                    'trailing_high': state['trailing_high'],
                    'drop_percentage': price_drop_pct
                }
        else:  # SHORT
            # Pour un SHORT, vérifier si le prix a monté de X% depuis le low
            price_rise_pct = ((current_price - state['trailing_high']) / state['trailing_high']) * 100
            
            if price_rise_pct >= trailing_distance:
                self.logger.info(f"🚨 Trailing stop déclenché pour cycle {cycle_id}: hausse de {price_rise_pct:.2f}% depuis {state['trailing_high']}")
                
                return {
                    'action': 'cover_all',
                    'reason': 'trailing_stop_triggered',
                    'price': current_price,
                    'trailing_low': state['trailing_high'],
                    'rise_percentage': price_rise_pct
                }
        
        return None
    
    def _calculate_stop_price(self, entry_price: float, stop_percentage: float, side: str) -> float:
        """
        Calcule le prix de stop basé sur un pourcentage.
        
        Args:
            entry_price: Prix d'entrée
            stop_percentage: Pourcentage de stop (positif = profit, négatif = perte)
            side: LONG ou SHORT

        Returns:
            Prix de stop calculé
        """
        if side == 'LONG':
            # Pour un LONG, stop_percentage positif = stop plus haut que l'entrée
            return entry_price * (1 + stop_percentage / 100)
        else:  # SHORT
            # Pour un SHORT, stop_percentage positif = stop plus bas que l'entrée
            return entry_price * (1 - stop_percentage / 100)
    
    def get_cycle_status(self, cycle_id: str) -> Optional[Dict]:
        """
        Retourne le statut de protection d'un cycle.
        
        Args:
            cycle_id: ID du cycle
            
        Returns:
            Dictionnaire avec le statut de protection
        """
        if cycle_id not in self.cycle_states:
            return None
        
        state = self.cycle_states[cycle_id]
        
        return {
            'cycle_id': cycle_id,
            'entry_price': state['entry_price'],
            'side': state['side'],
            'initial_quantity': state['initial_quantity'],
            'remaining_quantity': state['remaining_quantity'],
            'quantity_sold_pct': ((state['initial_quantity'] - state['remaining_quantity']) / state['initial_quantity']) * 100,
            'max_gain_reached': state['max_gain_reached'],
            'levels_triggered': list(state['levels_triggered']),
            'trailing_active': state['trailing_active'],
            'trailing_high': state['trailing_high'],
            'protection_level': self._get_current_protection_level(state)
        }
    
    
    def _get_current_protection_level(self, state: Dict) -> GainProtectionLevel:
        """
        Détermine le niveau de protection actuel.
        
        Args:
            state: État du cycle
            
        Returns:
            Niveau de protection actuel
        """
        if not state['levels_triggered']:
            return GainProtectionLevel.NONE
        
        max_level = max(state['levels_triggered'])
        
        if max_level >= 4:
            return GainProtectionLevel.FINAL_PROTECTION
        elif state['trailing_active']:
            return GainProtectionLevel.TRAILING_ACTIVE
        elif max_level >= 2:
            return GainProtectionLevel.PARTIAL_SECURED
        elif max_level >= 1:
            return GainProtectionLevel.BREAKEVEN
        
        return GainProtectionLevel.NONE
    
    def cleanup_cycle(self, cycle_id: str):
        """
        Nettoie les données d'un cycle terminé.
        
        Args:
            cycle_id: ID du cycle à nettoyer
        """
        if cycle_id in self.cycle_states:
            del self.cycle_states[cycle_id]
            self.logger.info(f"🧹 Données de protection nettoyées pour cycle {cycle_id}")