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
        # ÉQUILIBRAGE: Strict en baissier, permissif en haussier pour rapidité d'entrée
        self.regime_family_requirements = {
            'TRENDING_BULL': {
                'trend_following': 1,  # RÉDUIT: 1 au lieu de 2 pour entrée rapide
                'breakout': 1,         # Au moins 1 breakout
                'volume_based': 1,     # Au moins 1 volume
                'total_min': 4         # RÉDUIT: 4 au lieu de 5 pour entrée haussière rapide
            },
            'TRENDING_BEAR': {
                'trend_following': 2,
                'breakout': 1,
                'volume_based': 1,
                'total_min': 6         # MAINTENU: strict en baisse pour éviter pertes
            },
            'RANGING': {
                'mean_reversion': 1,   # Au moins 1 stratégie mean_reversion
                #'structure_based': 1,  # REMIS: important pour structure de marché en ranging
                'total_min': 2         # 2 minimum avec diversité de familles
            },
            'VOLATILE': {
                'breakout': 1,         # RÉDUIT: 1 au lieu de 2 pour volatilité
                'volume_based': 1,     # Au moins 1 volume
                'total_min': 4         # RÉDUIT: 4 au lieu de 5
            },
            'BREAKOUT_BULL': {
                'breakout': 1,         # RÉDUIT: 1 au lieu de 2 pour breakout haussier rapide
                'trend_following': 1,
                'volume_based': 1,
                'total_min': 3         # RÉDUIT: 3 au lieu de 5 pour breakout bull rapide
            },
            'BREAKOUT_BEAR': {
                'breakout': 2,
                'trend_following': 1,
                'volume_based': 2,     # Volume critique en bear
                'total_min': 7         # MAINTENU: très strict en breakout bear
            },
            'TRANSITION': {
                'total_min': 5         # RÉDUIT: 5 au lieu de 6
            },
            'UNKNOWN': {
                'total_min': 4         # RÉDUIT: était trop restrictif à 7
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
        logger.info(f"🔍 Analyse consensus: {len(signals)} signaux, régime: {market_regime}")
        
        if not signals:
            logger.info("🔍 Consensus: Aucun signal")
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
        
        logger.info(f"🔍 Familles détectées: {families_count}")
        logger.info(f"🔍 Scores adaptabilité: {adaptability_scores}")
        
        # Obtenir les requirements pour ce régime
        regime = market_regime.upper() if market_regime else 'UNKNOWN'
        if regime not in self.regime_family_requirements:
            regime = 'UNKNOWN'
            
        requirements = self.regime_family_requirements[regime]
        logger.info(f"🔍 Requirements pour {regime}: {requirements}")
        
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
                
        # Si des familles critiques manquent, vérifier conditions d'assouplissement
        if missing_families:
            # Assouplir si forte adaptabilité OU consensus très fort
            consensus_strength_preview = self._calculate_preview_consensus_strength(families_count, regime)
            
            can_bypass = (avg_adaptability >= 0.7 or 
                         consensus_strength_preview >= 2.5 or 
                         (regime == 'RANGING' and 'structure_based' in [fam.split(':')[0] for fam in missing_families]))
                         
            if not can_bypass:
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
        # Plus permissif si les stratégies sont bien adaptées OU en régime UNKNOWN
        if avg_adaptability > 0.8:
            min_consensus_strength = 2.5
        elif regime == 'UNKNOWN':
            min_consensus_strength = 1.6  # Revenu au niveau correct
        else:
            min_consensus_strength = 3.0
        
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
        
    def _calculate_preview_consensus_strength(self, families_count: Dict[str, int], regime: str) -> float:
        """Calcule rapidement le consensus_strength pour décision d'assouplissement."""
        weighted_score = 0
        total_weight = 0
        
        for family, count in families_count.items():
            if family == 'unknown':
                continue
                
            weight = self.family_weights.get(family, 1.0)
            family_config = STRATEGY_FAMILIES.get(family, {})
            
            if regime in family_config.get('best_regimes', []):
                weight *= 1.5
            elif regime in family_config.get('poor_regimes', []):
                weight *= 0.5
                
            weighted_score += count * weight
            total_weight += weight
            
        return weighted_score / max(1, total_weight)
        
    def analyze_adaptive_consensus_mtf(self, signals: List[Dict[str, Any]], 
                                      market_regime: str, 
                                      original_signal_count: int) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyse le consensus pour les signaux MTF post-conflit avec des critères assouplis.
        
        Quand le buffer MTF a résolu un conflit et filtré des signaux, on doit adapter
        notre logique car on avait initialement plus de stratégies qui étaient d'accord.
        
        Args:
            signals: Liste des signaux restants après résolution de conflit
            market_regime: Régime de marché actuel
            original_signal_count: Nombre de signaux avant la résolution de conflit
            
        Returns:
            Tuple (has_consensus, analysis_details)
        """
        if not signals:
            return False, {'reason': 'Aucun signal'}
            
        # Utiliser le nombre original pour la validation du consensus
        # car ces stratégies étaient toutes d'accord avant la résolution de conflit
        effective_strategy_count = max(len(signals), original_signal_count)
        
        # Classifier les signaux par famille (sur les signaux restants)
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
            
            # Score d'adaptabilité
            is_optimal = is_strategy_optimal_for_regime(strategy, market_regime)
            is_acceptable = is_strategy_acceptable_for_regime(strategy, market_regime)
            
            if is_optimal:
                adaptability_scores.append(1.0)
            elif is_acceptable:
                adaptability_scores.append(0.7)
            else:
                adaptability_scores.append(0.3)
                
        avg_adaptability = sum(adaptability_scores) / len(adaptability_scores) if adaptability_scores else 0
        
        # Pour MTF post-conflit, on assouplit les critères
        regime = market_regime.upper() if market_regime else 'UNKNOWN'
        if regime not in self.regime_family_requirements:
            regime = 'UNKNOWN'
            
        requirements = self.regime_family_requirements[regime]
        
        # Réduire le minimum requis pour MTF post-conflit
        # Car on sait qu'on avait plus de stratégies au départ
        total_min = max(3, requirements.get('total_min', 6) - 2)  # -2 pour MTF post-conflit
        
        # Vérifier avec le nombre effectif (original)
        if effective_strategy_count < total_min:
            return False, {
                'reason': f'Pas assez de stratégies effectives: {effective_strategy_count} < {total_min}',
                'families_count': families_count,
                'total_strategies': len(signals),
                'original_strategies': original_signal_count,
                'effective_strategies': effective_strategy_count,
                'required_min': total_min,
                'avg_adaptability': avg_adaptability,
                'is_mtf_post_conflict': True
            }
            
        # Pour MTF post-conflit, on est plus permissif sur les familles manquantes
        # car le filtrage a pu éliminer certaines familles
        
        # Calculer le score de consensus pondéré
        weighted_score = 0
        total_weight = 0
        
        for family, count in families_count.items():
            if family == 'unknown':
                continue
                
            weight = self.family_weights.get(family, 1.0)
            family_config = STRATEGY_FAMILIES.get(family, {})
            
            if regime in family_config.get('best_regimes', []):
                weight *= 1.5
            elif regime in family_config.get('poor_regimes', []):
                weight *= 0.5
                
            # Bonus pour MTF post-conflit car on sait que d'autres stratégies étaient d'accord
            weight *= 1.2
            
            weighted_score += count * weight
            total_weight += weight
            
        consensus_strength = weighted_score / max(1, total_weight)
        
        # Plus permissif pour MTF post-conflit
        min_consensus_strength = 2.0 if avg_adaptability > 0.7 else 2.5
        
        has_consensus = consensus_strength >= min_consensus_strength or effective_strategy_count >= total_min + 2
        
        return has_consensus, {
            'has_consensus': has_consensus,
            'families_count': families_count,
            'total_strategies': len(signals),
            'original_strategies': original_signal_count,
            'effective_strategies': effective_strategy_count,
            'avg_adaptability': avg_adaptability,
            'consensus_strength': consensus_strength,
            'min_required_strength': min_consensus_strength,
            'regime': regime,
            'is_mtf_post_conflict': True,
            'mtf_consensus_bonus': 'Applied - reduced thresholds for post-conflict MTF signals'
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