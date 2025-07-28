"""
CCI_Reversal_Strategy - Stratégie basée sur le CCI et les indicateurs pré-calculés.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class CCI_Reversal_Strategy(BaseStrategy):
    """
    Stratégie utilisant le CCI et les indicateurs pré-calculés pour détecter les retournements.
    
    Signaux générés:
    - BUY: CCI en zone de survente avec conditions favorables
    - SELL: CCI en zone de surachat avec conditions favorables
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres CCI
        self.oversold_level = -100
        self.overbought_level = 100
        self.extreme_oversold = -200
        self.extreme_overbought = 200
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs pré-calculés."""
        return {
            'cci_20': self.indicators.get('cci_20'),
            'momentum_score': self.indicators.get('momentum_score'),
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'confluence_score': self.indicators.get('confluence_score'),
            'signal_strength': self.indicators.get('signal_strength'),
            'pattern_detected': self.indicators.get('pattern_detected'),
            'pattern_confidence': self.indicators.get('pattern_confidence'),
            'market_regime': self.indicators.get('market_regime'),
            'regime_strength': self.indicators.get('regime_strength'),
            'volatility_regime': self.indicators.get('volatility_regime')
        }
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur le CCI et les indicateurs pré-calculés.
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
        cci_20_raw = values['cci_20']
        if cci_20_raw is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "CCI non disponible",
                "metadata": {"strategy": self.name}
            }
            
        # Conversion robuste en float
        try:
            cci_20 = float(cci_20_raw)
        except (ValueError, TypeError):
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"CCI invalide: {cci_20_raw}",
                "metadata": {"strategy": self.name}
            }
            
        signal_side = None
        reason = ""
        confidence_boost = 0.0
        
        # Logique de signal basée sur CCI et indicateurs pré-calculés
        if cci_20 <= self.oversold_level:
            # Zone de survente - chercher signal BUY
            signal_side = "BUY"
            if cci_20 <= self.extreme_oversold:
                zone = "survente extrême"
                confidence_boost += 0.25
            else:
                zone = "survente"
                confidence_boost += 0.15
            reason = f"CCI ({cci_20:.1f}) en zone de {zone}"
            
        elif cci_20 >= self.overbought_level:
            # Zone de surachat - chercher signal SELL
            signal_side = "SELL"
            if cci_20 >= self.extreme_overbought:
                zone = "surachat extrême"
                confidence_boost += 0.25
            else:
                zone = "surachat"
                confidence_boost += 0.15
            reason = f"CCI ({cci_20:.1f}) en zone de {zone}"
            
        if signal_side:
            base_confidence = 0.6
            
            # Utilisation du momentum_score pré-calculé avec conversion sécurisée
            momentum_score_raw = values.get('momentum_score')
            momentum_score = 0
            if momentum_score_raw is not None:
                try:
                    momentum_score = float(momentum_score_raw)
                except (ValueError, TypeError):
                    momentum_score = 0
            
            if momentum_score != 0:
                if (signal_side == "BUY" and momentum_score > 0) or \
                   (signal_side == "SELL" and momentum_score < 0):
                    confidence_boost += 0.15
                    reason += " avec momentum favorable"
                elif abs(momentum_score) < 0.1:
                    confidence_boost -= 0.05  # Momentum faible
                    
            # Utilisation du trend_strength
            trend_strength_raw = values.get('trend_strength')
            if trend_strength_raw and trend_strength_raw in ['STRONG', 'VERY_STRONG']:
                confidence_boost += 0.1
                reason += f" et tendance {trend_strength_raw.lower()}"
                
            # Utilisation du directional_bias
            directional_bias = values.get('directional_bias')
            if directional_bias:
                if (signal_side == "BUY" and directional_bias == "bullish") or \
                   (signal_side == "SELL" and directional_bias == "bearish"):
                    confidence_boost += 0.1
                    reason += " confirmé par bias directionnel"
                elif (signal_side == "BUY" and directional_bias == "bearish") or \
                     (signal_side == "SELL" and directional_bias == "bullish"):
                    confidence_boost -= 0.1  # Contradictoire
                    
            # Utilisation du confluence_score avec conversion sécurisée
            confluence_score_raw = values.get('confluence_score')
            confluence_score = 0
            if confluence_score_raw is not None:
                try:
                    confluence_score = float(confluence_score_raw)
                except (ValueError, TypeError):
                    confluence_score = 0
                    
            if confluence_score > 0.7:
                confidence_boost += 0.15
                reason += " avec haute confluence"
                
            # Utilisation du pattern_detected et pattern_confidence avec conversion sécurisée
            pattern_detected = values.get('pattern_detected')
            pattern_confidence_raw = values.get('pattern_confidence')
            pattern_confidence = 0
            if pattern_confidence_raw is not None:
                try:
                    pattern_confidence = float(pattern_confidence_raw)
                except (ValueError, TypeError):
                    pattern_confidence = 0
                    
            if pattern_detected and pattern_confidence > 0.6:
                confidence_boost += 0.1
                reason += f" avec pattern {pattern_detected}"
                
            # Utilisation du market_regime
            market_regime = values.get('market_regime')
            regime_strength_raw = values.get('regime_strength')
            
            if market_regime and regime_strength_raw and regime_strength_raw in ['MODERATE', 'STRONG', 'EXTREME']:
                if (signal_side == "BUY" and market_regime in ["TRENDING_BULL", "BREAKOUT_BULL"]) or \
                   (signal_side == "SELL" and market_regime in ["TRENDING_BEAR", "BREAKOUT_BEAR"]):
                    confidence_boost += 0.1
                    reason += f" en régime {market_regime}"
                    
            # Utilisation du volatility_regime
            volatility_regime = values.get('volatility_regime')
            if volatility_regime:
                if volatility_regime == "low":
                    confidence_boost += 0.05  # Meilleur pour les retournements
                elif volatility_regime == "high":
                    confidence_boost -= 0.05  # Plus risqué
                    
            # Utilisation du signal_strength pré-calculé
            signal_strength_calc_raw = values.get('signal_strength')
            if signal_strength_calc_raw and signal_strength_calc_raw in ['STRONG', 'VERY_STRONG']:
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
                    "cci_20": cci_20,
                    "zone": zone,
                    "momentum_score": momentum_score,
                    "trend_strength": trend_strength_raw,
                    "directional_bias": directional_bias,
                    "confluence_score": confluence_score,
                    "pattern_detected": pattern_detected,
                    "pattern_confidence": pattern_confidence,
                    "market_regime": market_regime,
                    "volatility_regime": volatility_regime
                }
            }
            
        return {
            "side": None,
            "confidence": 0.0,
            "strength": "weak",
            "reason": f"CCI neutre ({cci_20:.1f}) - pas de zone extrême",
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "cci_20": cci_20
            }
        }
