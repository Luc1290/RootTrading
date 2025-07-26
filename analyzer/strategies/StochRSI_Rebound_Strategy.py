"""
StochRSI_Rebound_Strategy - Stratégie basée sur les signaux StochRSI pré-calculés.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class StochRSI_Rebound_Strategy(BaseStrategy):
    """
    Stratégie utilisant les signaux StochRSI et indicateurs pré-calculés.
    
    Signaux générés:
    - BUY: StochRSI en zone de survente avec signaux favorables
    - SELL: StochRSI en zone de surachat avec signaux favorables
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Seuils StochRSI
        self.oversold_zone = 20
        self.overbought_zone = 80
        self.extreme_oversold = 10
        self.extreme_overbought = 90
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs pré-calculés."""
        return {
            'stoch_rsi': self.indicators.get('stoch_rsi'),
            'stoch_k': self.indicators.get('stoch_k'),
            'stoch_d': self.indicators.get('stoch_d'),
            'stoch_signal': self.indicators.get('stoch_signal'),
            'stoch_divergence': self.indicators.get('stoch_divergence'),
            'rsi_14': self.indicators.get('rsi_14'),
            'momentum_score': self.indicators.get('momentum_score'),
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'confluence_score': self.indicators.get('confluence_score'),
            'signal_strength': self.indicators.get('signal_strength')
        }
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur les indicateurs StochRSI pré-calculés.
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
        stoch_rsi = values['stoch_rsi']
        if stoch_rsi is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "StochRSI non disponible",
                "metadata": {"strategy": self.name}
            }
            
        signal_side = None
        reason = ""
        confidence_boost = 0.0
        
        # Utilisation du signal StochRSI pré-calculé si disponible
        stoch_signal = values.get('stoch_signal')
        if stoch_signal:
            if stoch_signal in ['BUY', 'SELL']:
                signal_side = stoch_signal
                reason = f"Signal StochRSI pré-calculé: {stoch_signal}"
                confidence_boost += 0.2
        
        # Si pas de signal pré-calculé, analyse manuelle des zones
        if not signal_side:
            if stoch_rsi <= self.oversold_zone:
                signal_side = "BUY"
                zone = "survente extrême" if stoch_rsi <= self.extreme_oversold else "survente"
                reason = f"StochRSI ({stoch_rsi:.1f}) en zone de {zone}"
                confidence_boost += 0.15 if stoch_rsi <= self.extreme_oversold else 0.1
                
            elif stoch_rsi >= self.overbought_zone:
                signal_side = "SELL"
                zone = "surachat extrême" if stoch_rsi >= self.extreme_overbought else "surachat"
                reason = f"StochRSI ({stoch_rsi:.1f}) en zone de {zone}"
                confidence_boost += 0.15 if stoch_rsi >= self.extreme_overbought else 0.1
                
        if signal_side:
            base_confidence = 0.55
            
            # Bonus pour divergence détectée
            stoch_divergence = values.get('stoch_divergence')
            if stoch_divergence:
                confidence_boost += 0.2
                reason += " avec divergence détectée"
                
            # Confirmation avec croisement K/D
            stoch_k = values.get('stoch_k')
            stoch_d = values.get('stoch_d')
            if stoch_k is not None and stoch_d is not None:
                if (signal_side == "BUY" and stoch_k > stoch_d) or \
                   (signal_side == "SELL" and stoch_k < stoch_d):
                    confidence_boost += 0.1
                    reason += " avec croisement K/D favorable"
                    
            # Confirmation avec RSI
            rsi_14 = values.get('rsi_14')
            if rsi_14:
                if (signal_side == "BUY" and rsi_14 <= 35) or \
                   (signal_side == "SELL" and rsi_14 >= 65):
                    confidence_boost += 0.1
                    reason += " confirmé par RSI"
                    
            # Utilisation du momentum_score
            momentum_score = values.get('momentum_score', 0)
            if momentum_score:
                if (signal_side == "BUY" and momentum_score > 0) or \
                   (signal_side == "SELL" and momentum_score < 0):
                    confidence_boost += 0.1
                    reason += " avec momentum favorable"
                    
            # Utilisation du trend_strength
            trend_strength = values.get('trend_strength', 0)
            if trend_strength and trend_strength > 0.5:
                confidence_boost += 0.1
                reason += " et tendance forte"
                
            # Utilisation du directional_bias
            directional_bias = values.get('directional_bias')
            if directional_bias:
                if (signal_side == "BUY" and directional_bias == "bullish") or \
                   (signal_side == "SELL" and directional_bias == "bearish"):
                    confidence_boost += 0.1
                    reason += " aligné avec bias directionnel"
                    
            # Utilisation du confluence_score
            confluence_score = values.get('confluence_score', 0)
            if confluence_score and confluence_score > 0.7:
                confidence_boost += 0.15
                reason += " avec haute confluence"
                
            # Utilisation du signal_strength pré-calculé
            signal_strength_calc = values.get('signal_strength', 0)
            if signal_strength_calc and signal_strength_calc > 0.6:
                confidence_boost += 0.1
                
            confidence = self.calculate_confidence(base_confidence, confidence_boost)
            strength = self.get_strength_from_confidence(confidence)
            
            return {
                "side": signal_side,
                "confidence": confidence,
                "strength": strength,
                "reason": reason,
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "stoch_rsi": stoch_rsi,
                    "stoch_k": stoch_k,
                    "stoch_d": stoch_d,
                    "stoch_signal": stoch_signal,
                    "stoch_divergence": stoch_divergence,
                    "rsi_14": rsi_14,
                    "momentum_score": momentum_score,
                    "trend_strength": trend_strength,
                    "confluence_score": confluence_score
                }
            }
            
        return {
            "side": None,
            "confidence": 0.0,
            "strength": "weak",
            "reason": f"StochRSI neutre ({stoch_rsi:.1f}) - pas de zone extrême",
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "stoch_rsi": stoch_rsi
            }
        }
