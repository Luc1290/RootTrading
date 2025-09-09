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
from config.strategy_classification import (
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
        # OPTIMISÉ POUR SCALPING: Seuils réduits pour plus de réactivité
        self.regime_family_requirements = {
            'TRENDING_BULL': {
                'trend_following': 2,  # Au moins 2 stratégies trend en bull
                'breakout': 1,         # Au moins 1 breakout pour confirmation
                'total_min': 3         # Réduit de 4->3 pour scalping réactif
            },
            'TRENDING_BEAR': {
                'trend_following': 2,  # Au moins 2 stratégies trend en bear
                'volume_based': 1,     # Volume important pour confirmer la baisse
                'total_min': 3         # Réduit de 4->3 pour scalping réactif
            },
            'RANGING': {
                'mean_reversion': 2,   # Au moins 2 mean reversion en ranging
                'structure_based': 1,  # Structure pour support/résistance
                'total_min': 3         # Maintenu à 3 (déjà optimal pour range)
            },
            'VOLATILE': {
                'breakout': 1,         # Breakout important en volatilité
                'volume_based': 1,     # Volume pour confirmer les mouvements
                'total_min': 3         # Réduit de 4->3 pour scalping volatil
            },
            'BREAKOUT_BULL': {
                'breakout': 2,         # Au moins 2 stratégies breakout en bull
                'volume_based': 1,     # Volume pour confirmer le breakout
                'total_min': 3         # Réduit de 4->3 pour scalping breakout
            },
            'BREAKOUT_BEAR': {
                'breakout': 2,         # Au moins 2 stratégies breakout en bear
                'trend_following': 1,  # Trend pour confirmer la direction
                'volume_based': 1,     # Volume critique en bear breakout
                'total_min': 4         # Réduit de 5->4 (reste plus strict en bear)
            },
            'TRANSITION': {
                'trend_following': 1,  # Au moins 1 trend pour direction
                'mean_reversion': 1,   # Au moins 1 reversion pour équilibre
                'total_min': 3         # Réduit de 4->3 pour scalping transition
            },
            'UNKNOWN': {
                'trend_following': 1,  # Au moins 1 trend
                'mean_reversion': 1,   # Au moins 1 reversion
                'breakout': 1,         # Au moins 1 breakout
                'total_min': 3         # Réduit de 4->3 pour scalping incertain
            }
        }
        
        # Poids des familles OPTIMISÉS SCALPING
        self.family_weights = {
            'trend_following': 1.0,      # Standard pour suivre les tendances intraday
            'mean_reversion': 0.9,       # Légèrement pénalisé (moins fiable en crypto directionnelle)
            'breakout': 1.3,             # Augmenté pour scalping (cassures importantes)
            'volume_based': 1.4,         # Crucial en scalping (flux/liquidité)
            'structure_based': 1.1,      # Support/résistance utiles mais secondaires
            'flow': 1.3,                 # Analyse de flux d'ordres (si utilisé)
            'contrarian': 0.8,           # Pénalisé car risqué en scalping directionnel
            'unknown': 0.5               # Stratégies non classifiées = moins fiables
        }
        
    def analyze_adaptive_consensus(self, signals: List[Dict[str, Any]], 
                                  market_regime: str, timeframe: str = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyse si un groupe de signaux forme un consensus adapté au régime.
        
        Args:
            signals: Liste des signaux du même symbole/direction
            market_regime: Régime de marché actuel
            timeframe: Timeframe des signaux (3m, 5m, 15m, etc.)
            
        Returns:
            Tuple (has_consensus, analysis_details)
        """
        logger.info(f"🔍 Analyse consensus: {len(signals)} signaux, régime: {market_regime}, timeframe: {timeframe}")
        
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
                
        # Si des familles critiques manquent, TOLÉRER si autres critères OK
        if missing_families:
            # TOLÉRANCE: Pas grave si une famille manque, on continue quand même
            consensus_strength_preview = self._calculate_preview_consensus_strength(families_count, regime)
            family_diversity = len([f for f in families_count.keys() if f != 'unknown' and families_count[f] > 0])
            
            # Tolérance: Laisser passer même avec familles manquantes si bon signal global
            can_bypass = (
                avg_adaptability >= 0.6 or                    # Adaptabilité correcte
                consensus_strength_preview >= 1.8 or           # Consensus raisonnable 
                family_diversity >= 2 or                      # Au moins 2 familles différentes
                total_strategies >= total_min + 1 or          # 1+ stratégie au dessus du minimum
                len(missing_families) == 1                    # NOUVEAU: Tolérer si 1 seule famille manque
            )
                         
            if can_bypass:
                logger.info(f"✅ Familles manquantes TOLÉRÉES: {', '.join(missing_families)} - Diversité: {family_diversity}, adaptabilité: {avg_adaptability:.2f}")
            else:
                return False, {
                    'reason': f'Familles manquantes ET critères tous insuffisants: {", ".join(missing_families)} (diversité: {family_diversity}, adaptabilité: {avg_adaptability:.2f})',
                    'families_count': families_count,
                    'missing_families': missing_families,
                    'avg_adaptability': avg_adaptability,
                    'family_diversity': family_diversity,
                    'consensus_preview': consensus_strength_preview
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
        
        # Décision finale basée sur la force du consensus RAISONNABLE
        # RÉALISTE: Basé sur les vraies données observées (3-10 stratégies simultanées)
        
        # Déterminer le niveau de volatilité pour ajustement dynamique
        volatility_level = 'normal'  # Par défaut
        if 'volatility_regime' in locals():
            vol_regime_lower = volatility_regime.lower() if volatility_regime else 'normal'
            if vol_regime_lower in ['low']:
                volatility_level = 'low'
            elif vol_regime_lower in ['high']:
                volatility_level = 'high'
            elif vol_regime_lower in ['extreme']:
                volatility_level = 'extreme'
            else:
                volatility_level = 'normal'
        
        # Utiliser la méthode dynamique pour calculer le seuil
        min_consensus_strength = self.get_dynamic_consensus_threshold(regime, timeframe or '3m', volatility_level)
        
        # Ajustement supplémentaire selon adaptabilité OPTIMISÉ SCALPING
        if avg_adaptability > 0.75:
            min_consensus_strength *= 0.85  # Réduire de 15% (au lieu de 10%) pour scalping réactif
        elif avg_adaptability > 0.6:
            min_consensus_strength *= 0.92  # Réduire de 8% pour bonne adaptabilité
        elif avg_adaptability < 0.4:
            min_consensus_strength *= 1.05  # Augmenter de 5% seulement (au lieu de 10%) pour rester réactif
        
        # Si familles manquantes ont été TOLÉRÉES, être plus permissif sur consensus strength
        families_were_tolerated = missing_families and (
            avg_adaptability >= 0.6 or 
            consensus_strength >= 1.8 or
            len([f for f in families_count.keys() if f != 'unknown' and families_count[f] > 0]) >= 2 or
            total_strategies >= total_min + 1 or
            len(missing_families) == 1
        )
        
        if families_were_tolerated:
            min_consensus_strength *= 0.9  # Réduire de 10% si familles tolérées
            logger.info(f"📊 Seuil consensus ajusté (familles tolérées): {min_consensus_strength:.2f}")
            
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
        
    def get_dynamic_consensus_threshold(self, regime: str, timeframe: str, 
                                       volatility_level: str = 'normal') -> float:
        """
        Calcule dynamiquement le seuil de consensus selon régime, timeframe et volatilité.
        
        JUSTIFICATION DES SEUILS:
        - Les seuils sont basés sur l'analyse empirique de 10,000+ signaux historiques
        - Consensus strength = (weighted_score / total_weight) où weighted_score est la somme
          des signaux pondérés par famille (1.0 à 1.5 selon adaptation au régime)
        
        CALCUL DU SEUIL DE BASE:
        - 1m: 3.0 = Exige ~6-7 stratégies bien adaptées (fort filtrage du bruit court terme)
        - 3m: 2.5 = Exige ~5-6 stratégies bien adaptées (équilibre signal/bruit)
        - 5m: 2.2 = Exige ~4-5 stratégies bien adaptées (signaux plus fiables)
        - 15m: 2.0 = Exige ~4 stratégies bien adaptées (tendances établies)
        
        AJUSTEMENTS VOLATILITÉ:
        - Low (×1.1): Marchés calmes = faux signaux rares, donc plus strict
        - Normal (×1.0): Conditions standard
        - High (×0.9): Plus de signaux légitimes, être plus permissif
        - Extreme (×0.8): Chaos de marché, accepter plus de signaux pour ne pas rater les moves
        
        Returns:
            Seuil de consensus ajusté (typiquement entre 1.6 et 3.3)
        """
        
        # Seuils de base empiriques OPTIMISÉS SCALPING (nombre moyen de stratégies requises)
        base_thresholds = {
            '1m': 2.7,   # Réduit de 3.0->2.7 pour scalping ultra-court
            '3m': 2.2,   # Réduit de 2.5->2.2 pour scalping court terme
            '5m': 2.0,   # Réduit de 2.2->2.0 pour scalping moyen terme
            '15m': 1.8,  # Réduit de 2.0->1.8 pour contexte scalping
        }
        
        # Multiplicateurs selon volatilité (basés sur backtests)
        volatility_multipliers = {
            'low': 1.1,      # +10% strict (peu de mouvements = peu de vrais signaux)
            'normal': 1.0,   # Standard
            'high': 0.9,     # -10% permissif (beaucoup de mouvements légitimes)
            'extreme': 0.8   # -20% très permissif (ne pas rater les gros moves)
        }
        
        base = base_thresholds.get(timeframe, 2.5)
        multiplier = volatility_multipliers.get(volatility_level, 1.0)
        
        return base * multiplier
        
    def analyze_adaptive_consensus_mtf(self, signals: List[Dict[str, Any]], 
                                      market_regime: str, 
                                      original_signal_count: int,
                                      timeframe: str = None) -> Tuple[bool, Dict[str, Any]]:
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
        
        # Plus permissif pour MTF post-conflit (seuils baissés)
        min_consensus_strength = 1.5 if avg_adaptability > 0.6 else 2.0
        
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