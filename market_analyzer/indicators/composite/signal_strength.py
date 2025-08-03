"""
Signal Strength Calculator

Ce module calcule un signal_strength basé sur la confluence des indicateurs
pour un timeframe donné (mono-timeframe).
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def calculate_signal_strength(indicators: Dict) -> str:
    """
    Calcule la force du signal basée sur la confluence des indicateurs.
    
    Args:
        indicators: Dictionnaire contenant les valeurs des indicateurs
        
    Returns:
        String: "VERY_WEAK", "WEAK", "MODERATE", "STRONG", "VERY_STRONG"
    """
    try:
        # Score basé sur confluence_score et d'autres indicateurs clés
        confluence_score = indicators.get('confluence_score', 50.0)
        
        # Facteurs additionnels
        score_factors = []
        
        # 1. Confluence score (poids principal)
        if confluence_score is not None:
            # Convertir de 0-100 vers score 0-1
            confluence_factor = confluence_score / 100.0
            score_factors.append(confluence_factor * 0.4)  # 40% du poids
        
        # 2. Trend strength (ADX)
        trend_strength = indicators.get('trend_strength')
        if trend_strength:
            trend_factor = _trend_strength_to_score(trend_strength)
            score_factors.append(trend_factor * 0.2)  # 20% du poids
        
        # 3. Market regime confidence
        regime_confidence = indicators.get('regime_confidence')
        if regime_confidence is not None:
            regime_factor = regime_confidence / 100.0
            score_factors.append(regime_factor * 0.15)  # 15% du poids
        
        # 4. Volume quality
        volume_quality = indicators.get('volume_quality_score')
        if volume_quality is not None:
            volume_factor = volume_quality / 100.0
            score_factors.append(volume_factor * 0.15)  # 15% du poids
        
        # 5. Pattern confidence
        pattern_confidence = indicators.get('pattern_confidence')
        if pattern_confidence is not None and pattern_confidence > 0:
            pattern_factor = pattern_confidence / 100.0
            score_factors.append(pattern_factor * 0.1)  # 10% du poids
        
        # Calculer le score final
        if not score_factors:
            return "WEAK"
        
        final_score = sum(score_factors)
        
        # Convertir en catégories
        if final_score >= 0.8:
            return "VERY_STRONG"
        elif final_score >= 0.65:
            return "STRONG"
        elif final_score >= 0.45:
            return "MODERATE"
        elif final_score >= 0.25:
            return "WEAK"
        else:
            return "VERY_WEAK"
            
    except Exception as e:
        logger.warning(f"❌ Erreur calcul signal_strength: {e}")
        return "WEAK"


def _trend_strength_to_score(trend_strength: str) -> float:
    """Convertit trend_strength string en score 0-1."""
    strength_map = {
        'very_strong': 1.0,
        'strong': 0.8,
        'moderate': 0.6,
        'weak': 0.3,
        'very_weak': 0.1,
        'absent': 0.1  # Valeur spéciale pour trend_strength absent
    }
    return strength_map.get(str(trend_strength).lower(), 0.5)