"""
SmartCycleManager - Gestionnaire intelligent unifié des cycles.
Remplace la logique de micro-cycles par des cycles renforcables et évolutifs.
"""

import logging
import time
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

from shared.src.enums import SignalStrength
from shared.src.schemas import StrategySignal

logger = logging.getLogger(__name__)


class CycleAction(Enum):
    """Actions possibles sur un cycle"""
    CREATE_NEW = "create_new"
    REINFORCE = "reinforce"
    WAIT = "wait"
    REDUCE = "reduce"
    CLOSE = "close"


@dataclass
class SmartCycleDecision:
    """Décision du SmartCycleManager"""
    action: CycleAction
    symbol: str
    amount: float
    currency: str
    reason: str
    confidence: float
    price_target: Optional[float] = None
    existing_cycle_id: Optional[str] = None
    signal: Optional['StrategySignal'] = None
    
    def to_dict(self) -> Dict:
        return {
            'action': self.action.value,
            'symbol': self.symbol,
            'amount': self.amount,
            'currency': self.currency,
            'reason': self.reason,
            'confidence': self.confidence,
            'price_target': self.price_target,
            'existing_cycle_id': self.existing_cycle_id
        }


class SmartCycleManager:
    """
    Gestionnaire intelligent des cycles de trading.
    
    Principe :
    - 1 seul cycle actif par symbole/direction
    - Renforcement dynamique basé sur les signaux
    - Allocation intelligente du capital
    - DCA automatique si opportunité
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configuration du renforcement
        self.min_reinforcement_gap = 1.5  # % minimum de baisse pour renforcer
        self.max_reinforcements = 3       # Max 3 renforcements par cycle
        self.reinforcement_multiplier = 1.5  # Multiplier la position de 1.5x
        
        # Configuration de la pyramide
        self.pyramid_levels = {
            1: {'allocation_pct': 30, 'confidence_min': 0.6},  # Premier niveau : 30% du capital
            2: {'allocation_pct': 50, 'confidence_min': 0.7},  # Deuxième niveau : 50% supplémentaire
            3: {'allocation_pct': 20, 'confidence_min': 0.8}   # Troisième niveau : 20% final
        }
        
        # État des cycles actifs
        self.active_cycles: Dict[str, Dict] = {}  # {symbol_direction: cycle_info}
        
        self.logger.info("🧠 SmartCycleManager initialisé - Cycles intelligents activés")
    
    def analyze_signal(self, 
                      signal: StrategySignal, 
                      current_price: float, 
                      available_balance: float,
                      existing_cycles: List[Dict]) -> SmartCycleDecision:
        """
        Analyse un signal et décide de l'action à prendre.
        
        Args:
            signal: Signal de trading reçu
            current_price: Prix actuel du marché
            available_balance: Capital disponible
            existing_cycles: Cycles existants pour ce symbole
            
        Returns:
            Décision d'action SmartCycleDecision
        """
        # Gérer le cas où signal.side peut être une chaîne ou un enum
        side_str = signal.side.value if hasattr(signal.side, 'value') else str(signal.side)
        symbol_side = f"{signal.symbol}_{side_str}"
        
        # Calculer la confiance globale du signal
        confidence = self._calculate_signal_confidence(signal)
        
        # Vérifier s'il y a déjà un cycle actif
        active_cycle = self._find_active_cycle(signal.symbol, side_str, existing_cycles)
        
        if not active_cycle:
            # Pas de cycle actif → Créer un nouveau cycle
            return self._decide_new_cycle(signal, current_price, available_balance, confidence)
        else:
            # Cycle actif → Analyser si on doit renforcer, attendre ou fermer
            return self._decide_cycle_action(signal, current_price, available_balance, active_cycle, confidence)
    
    def _calculate_signal_confidence(self, signal: StrategySignal) -> float:
        """
        Calcule la confiance globale dans un signal.
        
        Args:
            signal: Signal à analyser
            
        Returns:
            Score de confiance entre 0 et 1
        """
        # Base sur la force du signal
        strength_scores = {
            SignalStrength.WEAK: 0.3,
            SignalStrength.MODERATE: 0.5,
            SignalStrength.STRONG: 0.7,
            SignalStrength.VERY_STRONG: 0.9
        }
        
        base_confidence = strength_scores.get(signal.strength, 0.5)
        
        # Bonus si multiple indicateurs alignés (utiliser metadata si disponible)
        if signal.metadata and 'indicators' in signal.metadata:
            indicators = signal.metadata.get('indicators', [])
            if isinstance(indicators, list) and len(indicators) > 1:
                alignment_bonus = min(0.2, len(indicators) * 0.05)
                base_confidence += alignment_bonus
        
        # Malus si signal récent (moins de 2 minutes)
        signal_age = time.time() - signal.timestamp.timestamp()
        if signal_age < 120:  # Moins de 2 minutes
            freshness_malus = 0.1
            base_confidence -= freshness_malus
        
        return min(1.0, max(0.1, base_confidence))
    
    def _find_active_cycle(self, symbol: str, direction: str, existing_cycles: List[Dict]) -> Optional[Dict]:
        """
        Trouve le cycle actif pour un symbole/direction.
        
        Args:
            symbol: Symbole à chercher
            direction: Direction du signal (LONG/SHORT)
            existing_cycles: Liste des cycles existants
            
        Returns:
            Cycle actif ou None
        """
        for cycle in existing_cycles:
            # LOGIQUE CORRECTE basée sur l'analyse du code :
            # waiting_sell = position LONG ouverte (a acheté, attend de vendre)
            # waiting_buy = position SHORT ouverte (a vendu, attend de racheter)
            cycle_status = cycle.get('status')
            
            if cycle_status == 'waiting_sell':
                cycle_direction = "LONG"  # Position LONG ouverte
            elif cycle_status == 'waiting_buy': 
                cycle_direction = "SHORT"  # Position SHORT ouverte
            else:
                continue  # Ignorer les autres statuts
            
            if cycle.get('symbol') == symbol and cycle_direction == direction:
                return cycle
        
        return None
    
    def _decide_new_cycle(self, 
                         signal: StrategySignal, 
                         current_price: float, 
                         available_balance: float, 
                         confidence: float) -> SmartCycleDecision:
        """
        Décide de la création d'un nouveau cycle.
        
        Args:
            signal: Signal reçu
            current_price: Prix actuel
            available_balance: Capital disponible  
            confidence: Confiance dans le signal
            
        Returns:
            Décision de création de cycle
        """
        # Calculer l'allocation basée sur la confiance et la force du signal
        allocation_pct = self._calculate_allocation_percentage(signal.strength, confidence)
        amount = available_balance * (allocation_pct / 100.0)
        
        # Appliquer les limites min/max
        currency = self._get_quote_asset(signal.symbol)
        amount = self._apply_amount_limits(amount, currency)
        
        # Vérifier si on a assez de capital
        if amount < self._get_min_amount(currency):
            return SmartCycleDecision(
                action=CycleAction.WAIT,
                symbol=signal.symbol,
                amount=0,
                currency=currency,
                reason=f"Capital insuffisant: {available_balance:.4f} {currency}",
                confidence=confidence
            )
        
        # Calculer le prix cible intelligent
        price_target = self._calculate_smart_entry_price(signal, current_price, confidence)
        
        return SmartCycleDecision(
            action=CycleAction.CREATE_NEW,
            symbol=signal.symbol,
            amount=amount,
            currency=currency,
            reason=f"Nouveau cycle {signal.strength.value if hasattr(signal.strength, 'value') else str(signal.strength)} (confiance: {confidence:.1%})",
            confidence=confidence,
            price_target=price_target,
            signal=signal
        )
    
    def _decide_cycle_action(self, 
                           signal: StrategySignal, 
                           current_price: float, 
                           available_balance: float, 
                           active_cycle: Dict, 
                           confidence: float) -> SmartCycleDecision:
        """
        Décide de l'action sur un cycle existant.
        
        Args:
            signal: Nouveau signal reçu
            current_price: Prix actuel
            available_balance: Capital disponible
            active_cycle: Cycle actif existant
            confidence: Confiance dans le nouveau signal
            
        Returns:
            Décision d'action sur le cycle
        """
        cycle_id = active_cycle.get('id')
        entry_price = active_cycle.get('entry_price', current_price)
        current_quantity = active_cycle.get('quantity', 0)
        
        # Calculer la performance actuelle
        side_str = signal.side.value if hasattr(signal.side, 'value') else str(signal.side)
        if side_str == "LONG":
            price_change_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            price_change_pct = ((entry_price - current_price) / entry_price) * 100
        
        # Règles de décision
        
        # 1. Si signal très fort et prix a baissé → RENFORCER
        if (confidence >= 0.7 and 
            price_change_pct <= -self.min_reinforcement_gap and
            available_balance > self._get_min_amount(self._get_quote_asset(signal.symbol))):
            
            reinforcement_amount = current_quantity * self.reinforcement_multiplier
            reinforcement_amount = min(reinforcement_amount, available_balance * 0.3)  # Max 30% du capital
            
            return SmartCycleDecision(
                action=CycleAction.REINFORCE,
                symbol=signal.symbol,
                amount=reinforcement_amount,
                currency=self._get_quote_asset(signal.symbol),
                reason=f"DCA: Prix baissé de {abs(price_change_pct):.1f}%, signal fort (confiance: {confidence:.1%})",
                confidence=confidence,
                existing_cycle_id=cycle_id
            )
        
        # 2. Si gain > 2% et signal faible → RÉDUIRE partiellement
        elif price_change_pct > 2.0 and confidence < 0.5:
            return SmartCycleDecision(
                action=CycleAction.REDUCE,
                symbol=signal.symbol,
                amount=current_quantity * 0.3,  # Vendre 30%
                currency=self._get_quote_asset(signal.symbol),
                reason=f"Take profit partiel: +{price_change_pct:.1f}%, signal faible",
                confidence=confidence,
                existing_cycle_id=cycle_id
            )
        
        # 3. Si perte > 5% et signal très faible → FERMER
        elif price_change_pct < -5.0 and confidence < 0.3:
            return SmartCycleDecision(
                action=CycleAction.CLOSE,
                symbol=signal.symbol,
                amount=current_quantity,
                currency=self._get_quote_asset(signal.symbol),
                reason=f"Stop loss: {price_change_pct:.1f}%, signal très faible",
                confidence=confidence,
                existing_cycle_id=cycle_id
            )
        
        # 4. Sinon → ATTENDRE
        else:
            return SmartCycleDecision(
                action=CycleAction.WAIT,
                symbol=signal.symbol,
                amount=0,
                currency=self._get_quote_asset(signal.symbol),
                reason=f"Position stable: {price_change_pct:+.1f}%, confiance: {confidence:.1%}",
                confidence=confidence,
                existing_cycle_id=cycle_id
            )
    
    def _calculate_allocation_percentage(self, strength: SignalStrength, confidence: float) -> float:
        """
        Calcule le pourcentage d'allocation basé sur la force et la confiance.
        
        Args:
            strength: Force du signal
            confidence: Confiance calculée
            
        Returns:
            Pourcentage d'allocation (1-20%)
        """
        base_allocations = {
            SignalStrength.WEAK: 3.0,
            SignalStrength.MODERATE: 6.0,
            SignalStrength.STRONG: 10.0,
            SignalStrength.VERY_STRONG: 15.0
        }
        
        base_pct = base_allocations.get(strength, 5.0)
        
        # Ajuster selon la confiance
        confidence_multiplier = 0.5 + (confidence * 1.0)  # Entre 0.5x et 1.5x
        
        final_pct = base_pct * confidence_multiplier
        
        # Limiter entre 1% et 20%
        return min(20.0, max(1.0, final_pct))
    
    def _calculate_smart_entry_price(self, signal: StrategySignal, current_price: float, confidence: float) -> float:
        """
        Calcule un prix d'entrée intelligent basé sur le signal et la confiance.
        
        Args:
            signal: Signal de trading
            current_price: Prix actuel
            confidence: Confiance dans le signal
            
        Returns:
            Prix d'entrée optimal
        """
        # Si confiance élevée → Prix proche du marché (plus agressif)
        # Si confiance faible → Prix plus conservateur
        
        side_str = signal.side.value if hasattr(signal.side, 'value') else str(signal.side)
        if side_str == "BUY" or side_str == "LONG":
            # Pour un LONG, plus la confiance est haute, plus on accepte d'acheter proche du prix actuel
            discount_pct = (1.0 - confidence) * 2.0  # Entre 0% et 2% de discount
            entry_price = current_price * (1 - discount_pct / 100)
        else:
            # Pour un SHORT, plus la confiance est haute, plus on accepte de vendre proche du prix actuel  
            premium_pct = (1.0 - confidence) * 2.0   # Entre 0% et 2% de premium
            entry_price = current_price * (1 + premium_pct / 100)
        
        return entry_price
    
    def _get_quote_asset(self, symbol: str) -> str:
        """Extrait l'asset de cotation d'un symbole"""
        if symbol.endswith('USDC'):
            return 'USDC'
        elif symbol.endswith('BTC'):
            return 'BTC'
        elif symbol.endswith('ETH'):
            return 'ETH'
        else:
            return 'USDC'
    
    def _get_min_amount(self, currency: str) -> float:
        """Retourne le montant minimum pour une devise"""
        minimums = {'USDC': 10.0, 'BTC': 0.0001, 'ETH': 0.003}
        return minimums.get(currency, 10.0)
    
    def _apply_amount_limits(self, amount: float, currency: str) -> float:
        """Applique les limites min/max sur un montant"""
        limits = {
            'USDC': {'min': 10.0, 'max': 500.0},
            'BTC': {'min': 0.0001, 'max': 0.01},
            'ETH': {'min': 0.003, 'max': 0.2}
        }
        
        currency_limits = limits.get(currency, {'min': 10.0, 'max': 100.0})
        return min(currency_limits['max'], max(currency_limits['min'], amount))
    
    def get_active_cycles_summary(self) -> Dict:
        """
        Retourne un résumé des cycles actifs gérés.
        
        Returns:
            Dictionnaire avec le résumé
        """
        return {
            'total_active_cycles': len(self.active_cycles),
            'cycles_by_symbol': {k: v for k, v in self.active_cycles.items()},
            'total_exposure': sum(cycle.get('total_amount', 0) for cycle in self.active_cycles.values()),
            'avg_confidence': sum(cycle.get('confidence', 0) for cycle in self.active_cycles.values()) / len(self.active_cycles) if self.active_cycles else 0
        }