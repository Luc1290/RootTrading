"""
RSI_Cross_Strategy - Stratégie basée sur les positions du RSI dans les zones de survente/surachat.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class RSI_Cross_Strategy(BaseStrategy):
    """
    Stratégie utilisant le RSI et les indicateurs pré-calculés.
    
    Signaux générés:
    - BUY: RSI en zone de survente avec momentum et tendance favorables
    - SELL: RSI en zone de surachat avec momentum et tendance favorables
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Seuils RSI
        self.oversold_level = 30
        self.overbought_level = 70
        self.extreme_oversold = 20
        self.extreme_overbought = 80
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs pré-calculés."""
        return {
            'rsi_14': self.indicators.get('rsi_14'),
            'rsi_21': self.indicators.get('rsi_21'),
            'signal_strength': self.indicators.get('signal_strength'),
            'momentum_score': self.indicators.get('momentum_score'),
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'confluence_score': self.indicators.get('confluence_score'),
            'pattern_confidence': self.indicators.get('pattern_confidence')
        }
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur le RSI et les indicateurs pré-calculés.
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
        rsi_14 = values['rsi_14']
        if rsi_14 is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "RSI non disponible",
                "metadata": {"strategy": self.name}
            }
            
        signal_side = None
        reason = ""
        confidence_boost = 0.0
        
        # Logique de signal basée sur RSI et indicateurs pré-calculés
        if rsi_14 <= self.oversold_level:
            # Zone de survente - chercher signal BUY
            signal_side = "BUY"
            zone = "survente extrême" if rsi_14 <= self.extreme_oversold else "survente"
            reason = f"RSI ({rsi_14:.1f}) en zone de {zone}"
            
            # Bonus pour survente extrême
            if rsi_14 <= self.extreme_oversold:
                confidence_boost += 0.2
            else:
                confidence_boost += 0.1
                
        elif rsi_14 >= self.overbought_level:
            # Zone de surachat - chercher signal SELL
            signal_side = "SELL"
            zone = "surachat extrême" if rsi_14 >= self.extreme_overbought else "surachat"
            reason = f"RSI ({rsi_14:.1f}) en zone de {zone}"
            
            # Bonus pour surachat extrême
            if rsi_14 >= self.extreme_overbought:
                confidence_boost += 0.2
            else:
                confidence_boost += 0.1
                
        if signal_side:
            # Utilisation des indicateurs pré-calculés pour ajuster la confiance
            base_confidence = 0.5
            
            # Ajustement avec momentum_score
            momentum_score = values.get('momentum_score', 0)
            if momentum_score:
                if (signal_side == "BUY" and momentum_score > 0) or \
                   (signal_side == "SELL" and momentum_score < 0):
                    confidence_boost += 0.15
                    reason += " avec momentum favorable"
                elif (signal_side == "BUY" and momentum_score < -0.3) or \
                     (signal_side == "SELL" and momentum_score > 30):
                    confidence_boost -= 0.1
                    
            # Ajustement avec trend_strength
            trend_strength = values.get('trend_strength')
            if trend_strength and isinstance(trend_strength, str) and trend_strength in ['STRONG', 'VERY_STRONG']:
                confidence_boost += 0.1
                reason += f" et tendance {trend_strength.lower()}"
                
            # Ajustement avec directional_bias
            directional_bias = values.get('directional_bias')
            if directional_bias:
                if (signal_side == "BUY" and directional_bias == "bullish") or \
                   (signal_side == "SELL" and directional_bias == "bearish"):
                    confidence_boost += 0.1
                    reason += " confirmé par bias directionnel"
                    
            # Ajustement avec confluence_score
            confluence_score = values.get('confluence_score', 0)
            if confluence_score and confluence_score > 60:
                confidence_boost += 0.15
                reason += " avec haute confluence"
                
            # Ajustement avec signal_strength pré-calculé
            signal_strength_calc = values.get('signal_strength')
            if signal_strength_calc and signal_strength_calc in ['STRONG', 'VERY_STRONG']:
                confidence_boost += 0.1
                
            # Confirmation avec RSI 21 pour multi-timeframe
            rsi_21 = values.get('rsi_21')
            if rsi_21:
                if (signal_side == "BUY" and rsi_21 <= self.oversold_level) or \
                   (signal_side == "SELL" and rsi_21 >= self.overbought_level):
                    confidence_boost += 0.1
                    reason += " confirmé sur RSI 21"
                    
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
                    "rsi_14": rsi_14,
                    "rsi_21": rsi_21,
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
            "reason": f"RSI neutre ({rsi_14:.1f}) - pas de zone extrême",
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "rsi_14": rsi_14
            }
        }
