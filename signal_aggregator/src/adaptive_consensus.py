"""
Module de consensus adaptatif basé sur les familles de stratégies et le régime de marché.

Au lieu d'exiger un nombre fixe de stratégies, le consensus s'adapte selon :
- Le régime de marché actuel
- Les familles de stratégies qui ont émis des signaux
- La cohérence entre familles adaptées au régime
"""

import logging
from typing import Dict, List, Any, Tuple
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../config'))
from strategy_classification import (
    get_strategy_family,
    is_strategy_optimal_for_regime,
    is_strategy_acceptable_for_regime,
    STRATEGY_FAMILIES
)

logger = logging.getLogger(__name__)


class AdaptiveConsensusAnalyzer:
    """
    Analyse le consensus de manière adaptative selon le régime de marché.
    
    Au lieu d'un seuil fixe (6 stratégies), utilise une approche intelligente :
    - En trending : privilégier les stratégies trend-following et breakout
    - En ranging : privilégier les mean-reversion
    - En volatile : privilégier les breakout et volume-based
    """
    
    def __init__(self):
        # Consensus minimum par famille selon le régime
        self.regime_family_requirements = {
            'TRENDING_BULL': {
                'trend_following': 2,  # Au moins 2 trend-following
                'breakout': 1,         # Au moins 1 breakout
                'volume_based': 1,     # Au moins 1 volume
                'total_min': 4         # Minimum 4 stratégies au total
            },
            'TRENDING_BEAR': {
                'trend_following': 2,
                'breakout': 1,
                'volume_based': 1,
                'total_min': 4
            },
            'RANGING': {
                'mean_reversion': 2,   # Au moins 2 mean-reversion
                'structure_based': 1,  # Au moins 1 structure
                'total_min': 3         # Seulement 3 en ranging (plus difficile)
            },
            'VOLATILE': {
                'breakout': 2,         # Au moins 2 breakout
                'volume_based': 1,     # Au moins 1 volume
                'total_min': 4
            },
            'BREAKOUT_BULL': {
                'breakout': 2,
                'trend_following': 1,
                'volume_based': 1,
                'total_min': 4
            },
            'BREAKOUT_BEAR': {
                'breakout': 2,
                'trend_following': 1,
                'volume_based': 1,
                'total_min': 4
            },
            'TRANSITION': {
                'total_min': 5         # Plus strict en transition (incertitude)
            },
            'UNKNOWN': {
                'total_min': 6         # Conservateur si régime inconnu
            }
        }
        
        # Poids des familles pour le calcul de consensus
        self.family_weights = {
            'trend_following': 1.0,
            'mean_reversion': 1.0,
            'breakout': 1.2,       # Breakout légèrement plus important
            'volume_based': 1.3,   # Volume très important pour confirmation
            'structure_based': 1.1 # Structure importante
        }
        
    def analyze_adaptive_consensus(self, signals: List[Dict[str, Any]], 
                                  market_regime: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyse si un groupe de signaux forme un consensus adapté au régime.
        
        Args:
            signals: Liste des signaux du même symbole/direction
            market_regime: Régime de marché actuel
            
        Returns:
            Tuple (has_consensus, analysis_details)
        """
        if not signals:
            return False, {'reason': 'Aucun signal'}
            
        # Classifier les signaux par famille
        families_count = {}
        families_signals = {}
        adaptability_scores = []
        
        for signal in signals:
            strategy = signal.get('strategy', 'Unknown')
            family = get_strategy_family(strategy)
            
            if family not in families_count:
                families_count[family] = 0
                families_signals[family] = []
                
            families_count[family] += 1
            families_signals[family].append(signal)
            
            # Calculer le score d'adaptabilité au régime
            is_optimal = is_strategy_optimal_for_regime(strategy, market_regime)
            is_acceptable = is_strategy_acceptable_for_regime(strategy, market_regime)
            
            if is_optimal:
                adaptability_scores.append(1.0)
            elif is_acceptable:
                adaptability_scores.append(0.7)
            else:
                adaptability_scores.append(0.3)
                
        # Calculer les métriques
        total_strategies = len(signals)
        avg_adaptability = sum(adaptability_scores) / len(adaptability_scores) if adaptability_scores else 0
        
        # Obtenir les requirements pour ce régime
        regime = market_regime.upper() if market_regime else 'UNKNOWN'
        if regime not in self.regime_family_requirements:
            regime = 'UNKNOWN'
            
        requirements = self.regime_family_requirements[regime]
        
        # Vérifier le minimum total
        total_min = requirements.get('total_min', 6)
        if total_strategies < total_min:
            return False, {
                'reason': f'Pas assez de stratégies: {total_strategies} < {total_min}',
                'families_count': families_count,
                'total_strategies': total_strategies,
                'required_min': total_min,
                'avg_adaptability': avg_adaptability
            }
            
        # Vérifier les requirements par famille (si spécifiés)
        missing_families = []
        for family, required_count in requirements.items():
            if family == 'total_min':
                continue
                
            actual_count = families_count.get(family, 0)
            if actual_count < required_count:
                missing_families.append(f"{family}: {actual_count}/{required_count}")
                
        # Si des familles critiques manquent mais qu'on a une forte adaptabilité, assouplir
        if missing_families and avg_adaptability < 0.7:
            return False, {
                'reason': f'Familles manquantes: {", ".join(missing_families)}',
                'families_count': families_count,
                'missing_families': missing_families,
                'avg_adaptability': avg_adaptability
            }
            
        # Calculer le score de consensus pondéré
        weighted_score = 0
        total_weight = 0
        
        for family, count in families_count.items():
            if family == 'unknown':
                continue
                
            weight = self.family_weights.get(family, 1.0)
            # Bonus si la famille est optimale pour ce régime
            family_config = STRATEGY_FAMILIES.get(family, {})
            if regime in family_config.get('best_regimes', []):
                weight *= 1.5
            elif regime in family_config.get('poor_regimes', []):
                weight *= 0.5
                
            weighted_score += count * weight
            total_weight += weight
            
        consensus_strength = weighted_score / max(1, total_weight)
        
        # Décision finale basée sur la force du consensus
        # Plus permissif si les stratégies sont bien adaptées
        min_consensus_strength = 2.5 if avg_adaptability > 0.8 else 3.0
        
        has_consensus = consensus_strength >= min_consensus_strength
        
        return has_consensus, {
            'has_consensus': has_consensus,
            'families_count': families_count,
            'total_strategies': total_strategies,
            'avg_adaptability': avg_adaptability,
            'consensus_strength': consensus_strength,
            'min_required_strength': min_consensus_strength,
            'regime': regime,
            'missing_families': missing_families if missing_families else None
        }
        
    def get_adjusted_min_strategies(self, market_regime: str, 
                                   available_families: List[str]) -> int:
        """
        Retourne le nombre minimum de stratégies ajusté selon le régime et les familles disponibles.
        
        Args:
            market_regime: Régime de marché actuel
            available_families: Familles de stratégies disponibles
            
        Returns:
            Nombre minimum de stratégies requis
        """
        regime = market_regime.upper() if market_regime else 'UNKNOWN'
        if regime not in self.regime_family_requirements:
            regime = 'UNKNOWN'
            
        base_min = self.regime_family_requirements[regime].get('total_min', 6)
        
        # Ajuster selon les familles disponibles
        optimal_families = 0
        for family in available_families:
            family_config = STRATEGY_FAMILIES.get(family, {})
            if regime in family_config.get('best_regimes', []):
                optimal_families += 1
                
        # Si on a beaucoup de familles optimales, on peut réduire le minimum
        if optimal_families >= 3:
            return max(3, base_min - 1)
        elif optimal_families >= 2:
            return base_min
        else:
            # Peu de familles optimales, on augmente le minimum pour compenser
            return base_min + 1