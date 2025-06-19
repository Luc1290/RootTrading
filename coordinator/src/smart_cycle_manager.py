"""
SmartCycleManager - Gestionnaire intelligent unifié des cycles.
Remplace la logique de micro-cycles par des cycles renforcables et évolutifs.
"""

import logging
import time
import requests
import json
from typing import Dict, Optional, Tuple, List, Any
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
    - 1 seul cycle actif par symbole/side
    - Renforcement dynamique basé sur les signaux
    - Allocation intelligente du capital
    - DCA automatique si opportunité
    """
    
    def __init__(self, trader_api_url: str = "http://trader:5002", redis_client=None):
        self.logger = logging.getLogger(__name__)
        
        # Configuration API
        self.trader_api_url = trader_api_url
        self.redis_client = redis_client
        
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
        self.active_cycles: Dict[str, Dict] = {}  # {side: cycle_info}
        
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

    def _find_active_cycle(self, symbol: str, side: str, existing_cycles: List[Dict]) -> Optional[Dict]:
        """
        Trouve le cycle actif pour un symbole/side.
        
        Args:
            symbol: Symbole à chercher
            side: Side du signal (LONG/SHORT)
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
                cycle_side = "LONG"  # Position LONG ouverte
            elif cycle_status == 'waiting_buy': 
                cycle_side = "SHORT"  # Position SHORT ouverte
            else:
                continue  # Ignorer les autres statuts

            if cycle.get('symbol') == symbol and cycle_side == side:
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
    
    def _make_request_with_retry(self, url: str, method: str = "GET", json_data: Optional[Dict] = None, 
                                params: Optional[Dict] = None, timeout: float = 10.0, max_retries: int = 3) -> Optional[Dict]:
        """
        Effectue une requête HTTP avec retry et gestion d'erreurs.
        
        Args:
            url: URL de la requête
            method: Méthode HTTP (GET, POST, etc.)
            json_data: Données JSON à envoyer
            params: Paramètres URL
            timeout: Timeout en secondes
            max_retries: Nombre max de tentatives
            
        Returns:
            Réponse JSON ou None en cas d'erreur
        """
        for attempt in range(max_retries):
            try:
                if method.upper() == "GET":
                    response = requests.get(url, params=params, timeout=timeout)
                elif method.upper() == "POST":
                    response = requests.post(url, json=json_data, params=params, timeout=timeout)
                else:
                    self.logger.error(f"Méthode HTTP non supportée: {method}")
                    return None
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Tentative {attempt + 1}/{max_retries} échouée pour {url}: {str(e)}")
                if attempt == max_retries - 1:
                    self.logger.error(f"Échec définitif de la requête vers {url}")
                    return None
                time.sleep(0.5 * (attempt + 1))  # Backoff progressif
        
        return None

    def _reinforce_existing_cycle(self, decision: SmartCycleDecision) -> bool:
        """
        Renforce un cycle existant (DCA - Dollar Cost Averaging).
        
        Args:
            decision: Décision de renforcement avec montant et raison
            
        Returns:
            True si succès, False sinon
        """
        try:
            # 1. Récupérer les détails du cycle existant
            cycle_response = self._make_request_with_retry(
                f"{self.trader_api_url}/cycles/{decision.existing_cycle_id}",
                method="GET",
                timeout=5.0
            )
            
            if not cycle_response or not cycle_response.get('success'):
                self.logger.error(f"Impossible de récupérer le cycle {decision.existing_cycle_id}")
                return False
            
            cycle = cycle_response.get('cycle', {})
            
            # 2. Calculer la nouvelle quantité à ajouter
            current_quantity = float(cycle.get('quantity', 0))
            current_avg_price = float(cycle.get('entry_price', decision.price_target))
            
            # Nouvelle quantité basée sur le montant de renforcement
            additional_quantity = decision.amount / decision.price_target
            
            # 3. Calculer le nouveau prix moyen (moyenne pondérée)
            total_cost = (current_quantity * current_avg_price) + (additional_quantity * decision.price_target)
            new_total_quantity = current_quantity + additional_quantity
            new_avg_price = total_cost / new_total_quantity
            
            # 4. Déterminer le côté correct pour le renforcement
            # Si waiting_sell → position LONG → besoin d'un ordre LONG pour renforcer
            # Si waiting_buy → position SHORT → besoin d'un ordre SHORT pour renforcer
            cycle_status = cycle.get('status', '')
            if cycle_status == 'waiting_sell':
                reinforce_side = "LONG"  # Acheter plus pour une position longue
            elif cycle_status == 'waiting_buy':
                reinforce_side = "SHORT"  # Vendre plus pour une position courte
            else:
                self.logger.error(f"Status de cycle invalide pour renforcement: {cycle_status}")
                return False
            
            # 5. Créer l'ordre de renforcement
            reinforce_data = {
                "symbol": decision.symbol,
                "side": reinforce_side,
                "quantity": additional_quantity,
                "price": decision.price_target,
                "strategy": f"SmartCycle_DCA_{decision.confidence:.0%}",
                "parent_cycle_id": decision.existing_cycle_id,  # Lier au cycle parent
                "metadata": {
                    "smart_cycle": True,
                    "action": "reinforce",
                    "reason": decision.reason,
                    "confidence": decision.confidence,
                    "new_avg_price": new_avg_price,
                    "total_quantity": new_total_quantity
                }
            }
            
            # 6. Envoyer l'ordre de renforcement
            result = self._make_request_with_retry(
                f"{self.trader_api_url}/order",
                method="POST",
                json_data=reinforce_data,
                timeout=10.0
            )
            
            if result and result.get('order_id'):
                self.logger.info(f"✅ Cycle {decision.existing_cycle_id} renforcé: "
                               f"+{additional_quantity:.6f} @ {decision.price_target:.2f} "
                               f"(nouveau prix moyen: {new_avg_price:.2f})")
                
                # 7. Publier un événement pour tracking
                if self.redis_client:
                    self.redis_client.publish("roottrading:cycle:reinforced", json.dumps({
                        "cycle_id": decision.existing_cycle_id,
                        "reinforcement_order_id": result['order_id'],
                        "additional_quantity": additional_quantity,
                        "price": decision.price_target,
                        "new_avg_price": new_avg_price,
                        "reason": decision.reason
                    }))
                
                return True
            else:
                self.logger.error(f"❌ Échec du renforcement du cycle: {result}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Erreur lors du renforcement du cycle: {str(e)}")
            return False

    def _reduce_cycle_position(self, decision: SmartCycleDecision) -> bool:
        """
        Réduit partiellement une position (take profit partiel).
        
        Args:
            decision: Décision de réduction avec montant/quantité
            
        Returns:
            True si succès, False sinon
        """
        try:
            # 1. Récupérer les détails du cycle
            cycle_response = self._make_request_with_retry(
                f"{self.trader_api_url}/cycles/{decision.existing_cycle_id}",
                method="GET",
                timeout=5.0
            )
            
            if not cycle_response or not cycle_response.get('success'):
                self.logger.error(f"Impossible de récupérer le cycle {decision.existing_cycle_id}")
                return False
            
            cycle = cycle_response.get('cycle', {})
            current_quantity = float(cycle.get('quantity', 0))
            
            # 2. Calculer la quantité à vendre (30% par défaut)
            reduction_quantity = min(decision.amount, current_quantity * 0.3)
            
            if reduction_quantity <= 0:
                self.logger.warning(f"Quantité de réduction invalide: {reduction_quantity}")
                return False
            
            # 3. Créer un ordre de vente partielle
            partial_close_data = {
                "cycle_id": decision.existing_cycle_id,
                "quantity": reduction_quantity,
                "reason": decision.reason,
                "partial": True  # Indique une fermeture partielle
            }
            
            # 4. Envoyer la demande de fermeture partielle
            result = self._make_request_with_retry(
                f"{self.trader_api_url}/close/{decision.existing_cycle_id}/partial",
                method="POST",
                json_data=partial_close_data,
                timeout=10.0
            )
            
            if result and result.get('success'):
                remaining_quantity = current_quantity - reduction_quantity
                self.logger.info(f"✅ Position réduite: -{reduction_quantity:.6f} "
                               f"(reste: {remaining_quantity:.6f}) - {decision.reason}")
                
                # 5. Publier un événement
                if self.redis_client:
                    self.redis_client.publish("roottrading:cycle:reduced", json.dumps({
                        "cycle_id": decision.existing_cycle_id,
                        "reduced_quantity": reduction_quantity,
                        "remaining_quantity": remaining_quantity,
                        "reason": decision.reason
                    }))
                
                return True
            else:
                self.logger.error(f"❌ Échec de la réduction de position: {result}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Erreur lors de la réduction de position: {str(e)}")
            return False

    def _analyze_portfolio_exposure(self) -> Dict[str, Any]:
        """
        Analyse l'exposition totale du portfolio pour éviter la sur-concentration.
        
        Returns:
            Dict avec les métriques d'exposition
        """
        try:
            # Récupérer tous les cycles actifs
            cycles_response = self._make_request_with_retry(
                f"{self.trader_api_url}/cycles",
                params={"confirmed": "true", "include_completed": "false"},
                timeout=5.0
            )
            
            if not cycles_response or not cycles_response.get('success'):
                return {}
            
            cycles = cycles_response.get('cycles', [])
            
            # Calculer l'exposition par symbole et globale
            exposure = {
                'total_positions': len(cycles),
                'by_symbol': {},
                'by_side': {'LONG': 0, 'SHORT': 0},
                'total_value_usd': 0
            }
            
            for cycle in cycles:
                symbol = cycle.get('symbol', '')
                quantity = float(cycle.get('quantity', 0))
                entry_price = float(cycle.get('entry_price', 0))
                status = cycle.get('status', '')
                
                # Déterminer le côté de la position
                if status in ['waiting_sell', 'active_sell']:
                    side = 'LONG'
                else:
                    side = 'SHORT'
                
                # Calculer la valeur en USD (approximative)
                position_value = quantity * entry_price
                
                # Mettre à jour les métriques
                if symbol not in exposure['by_symbol']:
                    exposure['by_symbol'][symbol] = {
                        'count': 0,
                        'total_value': 0,
                        'LONG': 0,
                        'SHORT': 0
                    }
                
                exposure['by_symbol'][symbol]['count'] += 1
                exposure['by_symbol'][symbol]['total_value'] += position_value
                exposure['by_symbol'][symbol][side] += 1
                exposure['by_side'][side] += 1
                exposure['total_value_usd'] += position_value
            
            # Calculer les pourcentages de concentration
            if exposure['total_value_usd'] > 0:
                for symbol in exposure['by_symbol']:
                    symbol_value = exposure['by_symbol'][symbol]['total_value']
                    exposure['by_symbol'][symbol]['concentration_pct'] = (
                        symbol_value / exposure['total_value_usd'] * 100
                    )
            
            return exposure
            
        except Exception as e:
            self.logger.error(f"Erreur analyse exposition portfolio: {str(e)}")
            return {}

    def should_allow_new_position(self, symbol: str, side: str) -> Tuple[bool, str]:
        """
        Vérifie si une nouvelle position devrait être autorisée selon les règles de risque.
        
        Args:
            symbol: Symbole concerné
            side: Côté de la position (LONG/SHORT)
            
        Returns:
            Tuple (allowed, reason)
        """
        # Analyser l'exposition actuelle
        exposure = self._analyze_portfolio_exposure()
        
        # Règle 1: Limite globale de positions
        max_total_positions = 20  # Maximum 20 positions ouvertes
        if exposure.get('total_positions', 0) >= max_total_positions:
            return False, f"Limite globale atteinte: {max_total_positions} positions"
        
        # Règle 2: Limite par symbole
        max_per_symbol = 5  # Maximum 5 positions par symbole
        symbol_data = exposure.get('by_symbol', {}).get(symbol, {})
        if symbol_data.get('count', 0) >= max_per_symbol:
            return False, f"Limite par symbole atteinte: {max_per_symbol} positions sur {symbol}"
        
        # Règle 3: Concentration maximale
        max_concentration = 30  # Maximum 30% du portfolio sur un symbole
        if symbol_data.get('concentration_pct', 0) >= max_concentration:
            return False, f"Concentration excessive: {symbol_data['concentration_pct']:.1f}% sur {symbol}"
        
        # Règle 4: Équilibre LONG/SHORT
        long_count = exposure.get('by_side', {}).get('LONG', 0)
        short_count = exposure.get('by_side', {}).get('SHORT', 0)
        
        # Si déséquilibre important, favoriser le côté opposé
        if side == 'LONG' and long_count > short_count * 2:
            return False, f"Déséquilibre LONG/SHORT: {long_count} LONG vs {short_count} SHORT"
        elif side == 'SHORT' and short_count > long_count * 2:
            return False, f"Déséquilibre SHORT/LONG: {short_count} SHORT vs {long_count} LONG"
        
        return True, "Position autorisée"
    
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