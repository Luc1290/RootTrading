#!/usr/bin/env python3
"""
Module pour le calcul de métriques et l'application de boosts aux signaux.
Contient toute la logique de scoring et d'amélioration des signaux.
"""

import logging
from typing import Dict, List, Any
from .shared.technical_utils import VolumeAnalyzer

logger = logging.getLogger(__name__)


class SignalMetrics:
    """Classe pour calculer les métriques et appliquer des boosts aux signaux"""
    
    def __init__(self, performance_tracker=None):
        self.performance_tracker = performance_tracker
    
    async def apply_performance_boost(self, confidence: float, contributing_strategies: List[str]) -> float:
        """Applique un boost adaptatif basé sur la performance des stratégies"""
        if not self.performance_tracker:
            return confidence
        
        try:
            boost_factor = 1.0
            
            for strategy in contributing_strategies:
                # Obtenir le poids de performance (1.0 = neutre, >1.0 = surperformance)
                performance_weight = await self.performance_tracker.get_strategy_weight(strategy)
                
                if performance_weight > 1.1:  # Plus de 10% au-dessus du benchmark
                    # Boost progressif selon la surperformance
                    individual_boost = 1.0 + (performance_weight - 1.0) * 0.2  # Max +20% pour 2x performance
                    boost_factor *= individual_boost
                    logger.debug(f"🚀 Boost performance pour {strategy}: {performance_weight:.2f} -> boost {individual_boost:.2f}")
                
                elif performance_weight < 0.9:  # Plus de 10% en-dessous du benchmark
                    # Malus modéré pour sous-performance
                    individual_malus = max(0.95, 1.0 - (1.0 - performance_weight) * 0.1)  # Max -5%
                    boost_factor *= individual_malus
                    logger.debug(f"📉 Malus performance pour {strategy}: {performance_weight:.2f} -> malus {individual_malus:.2f}")
            
            # Limiter le boost total
            boost_factor = min(1.15, max(0.95, boost_factor))  # Entre -5% et +15%
            
            boosted_confidence = confidence * boost_factor
            final_confidence = min(1.0, boosted_confidence)
            
            if boost_factor != 1.0:
                logger.info(f"📊 Boost performance: {confidence:.3f} * {boost_factor:.2f} = {boosted_confidence:.3f} -> {final_confidence:.3f} (strategies: {contributing_strategies})")
            
            return final_confidence
            
        except Exception as e:
            logger.error(f"Erreur dans boost performance: {e}")
            return confidence
    
    def apply_volume_boost(self, confidence: float, signals: List[Dict[str, Any]]) -> float:
        """
        Applique un boost de confiance basé sur le volume_ratio des signaux
        
        Args:
            confidence: Confiance actuelle du signal agrégé
            signals: Liste des signaux contributeurs
            
        Returns:
            Confiance boostée par le volume
        """
        try:
            volume_ratios = []
            volume_scores = []
            
            # Extraire les ratios de volume et scores des métadonnées
            for signal in signals:
                metadata = signal.get('metadata', {})
                
                # Chercher volume_ratio directement ou dans les sous-données
                volume_ratio = metadata.get('volume_ratio')
                if volume_ratio is None:
                    # Peut-être dans volume_spike ou autre champ volume
                    volume_ratio = metadata.get('volume_spike', 1.0)
                
                volume_score = metadata.get('volume_score', 0.5)
                
                if volume_ratio and isinstance(volume_ratio, (int, float)):
                    volume_ratios.append(float(volume_ratio))
                
                if volume_score and isinstance(volume_score, (int, float)):
                    volume_scores.append(float(volume_score))
            
            if not volume_ratios and not volume_scores:
                return confidence  # Pas de données volume, pas de boost
            
            # Calculer le boost basé sur volume_ratio avec utilitaire partagé
            volume_boost = 1.0
            if volume_ratios:
                avg_volume_ratio = sum(volume_ratios) / len(volume_ratios)
                volume_boost = VolumeAnalyzer.calculate_volume_boost(avg_volume_ratio)
                
                # Log avec description
                _, description = VolumeAnalyzer.analyze_volume_strength({'volume_ratio': avg_volume_ratio})
                logger.debug(f"📊 {description} -> boost {volume_boost:.2f}x")
            
            # Boost supplémentaire basé sur volume_score des stratégies
            if volume_scores:
                avg_volume_score = sum(volume_scores) / len(volume_scores)
                
                if avg_volume_score >= 0.8:
                    # Score volume excellent: boost additionnel (+5%)
                    volume_boost *= 1.05
                    logger.debug(f"⭐ Score volume excellent: {avg_volume_score:.2f} -> boost additionnel +5%")
                elif avg_volume_score <= 0.3:
                    # Score volume faible: pénalité (-3%)
                    volume_boost *= 0.97
                    logger.debug(f"⚠️ Score volume faible: {avg_volume_score:.2f} -> malus -3%")
            
            # Appliquer le boost final
            boosted_confidence = confidence * volume_boost
            
            if volume_boost != 1.0:
                logger.info(f"🎚️ Boost volume global: {confidence:.3f} -> {boosted_confidence:.3f} "
                           f"(facteur: {volume_boost:.3f})")
            
            return min(1.0, boosted_confidence)  # Cap à 1.0
            
        except Exception as e:
            logger.error(f"Erreur dans boost volume: {e}")
            return confidence  # En cas d'erreur, retourner confiance originale
    
    def apply_multi_strategy_bonus(self, confidence: float, contributing_strategies: List[str]) -> float:
        """
        Applique un bonus de confiance si plusieurs stratégies convergent
        
        Args:
            confidence: Confiance actuelle
            contributing_strategies: Liste des stratégies contributrices
            
        Returns:
            Confiance boostée par la convergence multi-stratégies
        """
        try:
            strategy_count = len(contributing_strategies)
            
            if strategy_count >= 2:
                # Bonus +0.05 pour 2+ stratégies alignées
                bonus = 0.05
                boosted_confidence = confidence + bonus
                
                logger.info(f"🤝 Bonus multi-stratégies: {strategy_count} stratégies -> "
                           f"{confidence:.3f} + {bonus:.2f} = {boosted_confidence:.3f}")
                
                return min(1.0, boosted_confidence)  # Cap à 1.0
            
            return confidence
            
        except Exception as e:
            logger.error(f"Erreur dans bonus multi-stratégies: {e}")
            return confidence
    
    def extract_volume_summary(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extrait un résumé des données de volume des signaux pour les métadonnées
        
        Args:
            signals: Liste des signaux contributeurs
            
        Returns:
            Dictionnaire avec le résumé des données volume
        """
        try:
            volume_ratios = []
            volume_scores = []
            
            for signal in signals:
                metadata = signal.get('metadata', {})
                
                volume_ratio = metadata.get('volume_ratio')
                if volume_ratio is None:
                    volume_ratio = metadata.get('volume_spike', 1.0)
                
                volume_score = metadata.get('volume_score', 0.5)
                
                if volume_ratio and isinstance(volume_ratio, (int, float)):
                    volume_ratios.append(float(volume_ratio))
                
                if volume_score and isinstance(volume_score, (int, float)):
                    volume_scores.append(float(volume_score))
            
            summary: Dict[str, Any] = {
                'signals_with_volume': len(volume_ratios),
                'total_signals': len(signals)
            }
            
            if volume_ratios:
                summary.update({
                    'avg_volume_ratio': round(sum(volume_ratios) / len(volume_ratios), 2),
                    'max_volume_ratio': round(max(volume_ratios), 2),
                    'min_volume_ratio': round(min(volume_ratios), 2)
                })
            
            if volume_scores:
                summary.update({
                    'avg_volume_score': round(sum(volume_scores) / len(volume_scores), 3),
                    'max_volume_score': round(max(volume_scores), 3)
                })
            
            return summary
            
        except Exception as e:
            logger.error(f"Erreur extraction résumé volume: {e}")
            return {'error': 'extraction_failed'}
    
    def calculate_soft_cap_confidence(self, confidence: float) -> float:
        """
        Applique un soft-cap sophistiqué avec tanh() pour préserver les nuances
        
        Args:
            confidence: Confiance à capper
            
        Returns:
            Confiance avec soft-cap appliqué
        """
        import math
        
        if confidence > 0.95:
            # Appliquer tanh() pour un soft-cap au-dessus de 0.95
            # tanh(x) mapping: 0.95 -> 0.95, 1.0 -> 0.96, 1.1 -> 0.97, etc.
            raw_confidence = confidence
            confidence = 0.95 + 0.05 * math.tanh((confidence - 0.95) * 4)
            logger.debug(f"🧠 Soft-cap tanh appliqué: {raw_confidence:.3f} -> {confidence:.3f}")
        
        return confidence
    
    def strength_to_normalized_force(self, strength: str) -> float:
        """Convertit la force textuelle en valeur normalisée 0-1"""
        strength_mapping = {
            'weak': 0.25,
            'moderate': 0.5,
            'strong': 0.75,
            'very_strong': 1.0
        }
        return strength_mapping.get(strength, 0.5)