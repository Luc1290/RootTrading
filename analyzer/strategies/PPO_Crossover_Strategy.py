"""
PPO_Crossover_Strategy - Stratégie basée sur le PPO (Percentage Price Oscillator).
Le PPO est similaire au MACD mais normalisé en pourcentage, permettant des comparaisons entre actifs.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class PPO_Crossover_Strategy(BaseStrategy):
    """
    Stratégie utilisant le PPO (Percentage Price Oscillator) et indicateurs pré-calculés.
    
    Le PPO est un MACD normalisé en pourcentage (PPO = (EMA12 - EMA26) / EMA26 * 100).
    
    Signaux générés:
    - BUY: PPO croisant au-dessus de 0 avec momentum favorable
    - SELL: PPO croisant en-dessous de 0 avec momentum défavorable
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Seuils PPO
        self.bullish_threshold = 0.0     # Croisement au-dessus de 0
        self.bearish_threshold = 0.0     # Croisement en-dessous de 0
        self.strong_signal_threshold = 0.5  # PPO > 0.5% = signal fort
        self.extreme_threshold = 1.0     # PPO > 1% = signal très fort
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs pré-calculés."""
        return {
            'ppo': self.indicators.get('ppo'),
            'macd_line': self.indicators.get('macd_line'),
            'macd_signal': self.indicators.get('macd_signal'), 
            'macd_histogram': self.indicators.get('macd_histogram'),
            'momentum_score': self.indicators.get('momentum_score'),
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'confluence_score': self.indicators.get('confluence_score'),
            'signal_strength': self.indicators.get('signal_strength'),
            'pattern_confidence': self.indicators.get('pattern_confidence')
        }
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur PPO et indicateurs pré-calculés.
        """
        if not self.validate_data():
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Données insuffisantes",
                "metadata": {}
            }
            
        values = self._get_current_values()
        
        # Vérification des données essentielles
        ppo = values['ppo']
        if ppo is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "PPO non disponible",
                "metadata": {"strategy": self.name}
            }
            
        try:
            ppo_val = float(ppo)
        except (ValueError, TypeError):
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Erreur conversion PPO",
                "metadata": {"strategy": self.name}
            }
            
        signal_side = None
        reason = ""
        confidence_boost = 0.0
        
        # Logique de signal basée sur PPO crossover
        if ppo_val > self.bullish_threshold:
            # PPO au-dessus de 0 - signal haussier
            signal_side = "BUY"
            
            if ppo_val >= self.extreme_threshold:
                reason = f"PPO très fort ({ppo_val:.3f}%) - momentum haussier extrême"
                confidence_boost += 0.3
            elif ppo_val >= self.strong_signal_threshold:
                reason = f"PPO fort ({ppo_val:.3f}%) - momentum haussier fort"
                confidence_boost += 0.2
            else:
                reason = f"PPO haussier ({ppo_val:.3f}%) - momentum positif"
                confidence_boost += 0.1
                
        elif ppo_val < self.bearish_threshold:
            # PPO en-dessous de 0 - signal baissier
            signal_side = "SELL"
            
            if ppo_val <= -self.extreme_threshold:
                reason = f"PPO très faible ({ppo_val:.3f}%) - momentum baissier extrême"
                confidence_boost += 0.3
            elif ppo_val <= -self.strong_signal_threshold:
                reason = f"PPO faible ({ppo_val:.3f}%) - momentum baissier fort"
                confidence_boost += 0.2
            else:
                reason = f"PPO baissier ({ppo_val:.3f}%) - momentum négatif"
                confidence_boost += 0.1
                
        if signal_side:
            # Utilisation des indicateurs pré-calculés pour renforcer le signal
            base_confidence = 0.5
            
            # Confirmation avec MACD histogram (dérivée du momentum)
            macd_histogram = values.get('macd_histogram')
            if macd_histogram is not None:
                try:
                    histogram_val = float(macd_histogram)
                    if (signal_side == "BUY" and histogram_val > 0) or \
                       (signal_side == "SELL" and histogram_val < 0):
                        confidence_boost += 0.15
                        reason += " + histogram MACD confirmé"
                except (ValueError, TypeError):
                    pass
                    
            # Ajustement avec momentum_score
            momentum_score = values.get('momentum_score')
            if momentum_score is not None:
                try:
                    momentum_val = float(momentum_score)
                    if (signal_side == "BUY" and momentum_val > 0) or \
                       (signal_side == "SELL" and momentum_val < 0):
                        confidence_boost += 0.1
                        reason += " avec momentum favorable"
                except (ValueError, TypeError):
                    pass
                    
            # Ajustement avec trend_strength
            trend_strength = values.get('trend_strength')
            if trend_strength is not None:
                if trend_strength in ['STRONG', 'VERY_STRONG']:  # Tendance forte selon schéma
                    confidence_boost += 0.1
                    reason += " et tendance forte"
                    
            # Ajustement avec directional_bias
            directional_bias = values.get('directional_bias')
            if directional_bias:
                if (signal_side == "BUY" and directional_bias == "BULLISH") or \
                   (signal_side == "SELL" and directional_bias == "BEARISH"):
                    confidence_boost += 0.1
                    reason += " confirmé par bias directionnel"
                    
            # Ajustement avec confluence_score
            confluence_score = values.get('confluence_score')
            if confluence_score is not None:
                try:
                    confluence_val = float(confluence_score)
                    if confluence_val > 0.7:
                        confidence_boost += 0.15
                        reason += " avec haute confluence"
                except (ValueError, TypeError):
                    pass
                    
            # Ajustement avec signal_strength pré-calculé
            signal_strength_calc = values.get('signal_strength')
            if signal_strength_calc is not None:
                if signal_strength_calc in ['STRONG', 'VERY_STRONG']:  # Signal fort selon schéma
                    confidence_boost += 0.1
                    
            # Confirmation avec MACD line trend
            macd_line = values.get('macd_line')
            if macd_line is not None:
                try:
                    macd_val = float(macd_line)
                    if (signal_side == "BUY" and macd_val > 0) or \
                       (signal_side == "SELL" and macd_val < 0):
                        confidence_boost += 0.05
                        reason += " et MACD aligné"
                except (ValueError, TypeError):
                    pass
                    
            confidence = self.calculate_confidence(base_confidence, 1.0 + confidence_boost)
            strength = self.get_strength_from_confidence(confidence)
            
            return {
                "side": signal_side,
                "confidence": confidence,
                "strength": strength,
                "reason": reason,
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "ppo": ppo_val,
                    "macd_line": macd_line,
                    "macd_histogram": macd_histogram,
                    "momentum_score": momentum_score,
                    "trend_strength": trend_strength,
                    "directional_bias": directional_bias,
                    "confluence_score": confluence_score,
                    "signal_strength_calc": signal_strength_calc
                }
            }
            
        return {
            "side": None,
            "confidence": 0.0,
            "strength": "weak",
            "reason": f"PPO neutre ({ppo_val:.3f}%) - pas de crossover significatif",
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "ppo": ppo_val
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que tous les indicateurs requis sont présents."""
        required_indicators = ['ppo']
        
        if 'indicators' not in self.data and not self.indicators:
            logger.warning(f"{self.name}: Aucun indicateur disponible")
            return False
            
        # Utilisation de self.indicators directement (pattern du système)
        for indicator in required_indicators:
            if indicator not in self.indicators:
                logger.warning(f"{self.name}: Indicateur manquant: {indicator}")
                return False
            if self.indicators[indicator] is None:
                logger.warning(f"{self.name}: Indicateur null: {indicator}")
                return False
                
        return True
