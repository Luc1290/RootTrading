"""
Supertrend_Reversal_Strategy - Version ULTRA SIMPLIFIÉE pour crypto.
Détecte les reversals de tendance avec logique adaptée spot crypto intraday.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class Supertrend_Reversal_Strategy(BaseStrategy):
    """
    Stratégie ULTRA SIMPLIFIÉE de détection reversal adaptée crypto.
    
    Principe simplifié :
    - Direction basée sur directional_bias + momentum_score
    - Confirmations volume, trend_strength, confluence
    - Bonus adaptés crypto (volatilité, regime, RSI)
    - Pas de simulation Supertrend complexe
    
    Signaux générés:
    - BUY: Reversal haussier détecté + confirmations
    - SELL: Reversal baissier détecté + confirmations
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        
        # Paramètres DURCIS pour vrais reversals
        self.min_momentum_threshold = 55  # BUY si momentum > 55 (plus strict)
        self.max_momentum_threshold = 45  # SELL si momentum < 45 (plus strict)
        self.base_confidence = 0.65       # Base élevée maintenue
        
    def generate_signal(self) -> Dict[str, Any]:
        """Version ULTRA SIMPLIFIÉE pour crypto spot intraday."""
        
        # Validation minimale
        if not self.validate_data():
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Données insuffisantes",
                "metadata": {"strategy": self.name}
            }
            
        values = self._get_current_values()
        confidence_boost = 0.0
        
        # Direction simple basée sur bias + momentum
        directional_bias = values.get('directional_bias')
        momentum_score = values.get('momentum_score', 50)
        
        try:
            momentum_val = float(momentum_score)
        except (ValueError, TypeError):
            momentum_val = 50
        
        # REJETS CRITIQUES avant signal
        
        # Déterminer signal avec cohérence bias/momentum
        if directional_bias == 'BULLISH' and momentum_val > self.min_momentum_threshold:
            signal_side = "BUY"
            reason = "Reversal haussier cohérent (bias + momentum)"
            confidence_boost += 0.20  # Double confirmation
            
        elif directional_bias == 'BULLISH' and momentum_val <= self.min_momentum_threshold:
            # Bias positif mais momentum faible = incohérent
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Rejet BUY: bias positif mais momentum faible ({momentum_val:.0f})",
                "metadata": {"strategy": self.name}
            }
            
        elif directional_bias == 'BEARISH' and momentum_val < self.max_momentum_threshold:
            signal_side = "SELL"
            reason = "Reversal baissier cohérent (bias + momentum)"
            confidence_boost += 0.20  # Double confirmation
            
        elif directional_bias == 'BEARISH' and momentum_val >= self.max_momentum_threshold:
            # Bias négatif mais momentum élevé = incohérent
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Rejet SELL: bias négatif mais momentum élevé ({momentum_val:.0f})",
                "metadata": {"strategy": self.name}
            }
            
        elif momentum_val > self.min_momentum_threshold:
            signal_side = "BUY"
            reason = f"Reversal haussier momentum ({momentum_val:.0f})"
            confidence_boost += 0.08
            
        elif momentum_val < self.max_momentum_threshold:
            signal_side = "SELL"
            reason = f"Reversal baissier momentum ({momentum_val:.0f})"
            confidence_boost += 0.08
            
        else:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak", 
                "reason": f"Pas de direction claire (momentum {momentum_val:.0f})",
                "metadata": {"strategy": self.name}
            }
        
        # Bonus simples crypto
        
        # Volume durci
        volume_ratio = values.get('volume_ratio', 1.0)
        try:
            vol_ratio = float(volume_ratio)
            if vol_ratio >= 2.0:
                confidence_boost += 0.15
                reason += f" + volume exceptionnel ({vol_ratio:.1f}x)"
            elif vol_ratio >= 1.5:
                confidence_boost += 0.10
                reason += f" + volume fort ({vol_ratio:.1f}x)"
            # Volume <1.5x = neutre, pas de bonus
        except (ValueError, TypeError):
            pass
        
        # Trend strength avec condition confluence
        trend_strength = values.get('trend_strength')
        confluence_score = values.get('confluence_score', 0)
        try:
            conf_val = float(confluence_score)
        except (ValueError, TypeError):
            conf_val = 0
            
        if trend_strength:
            trend_str = str(trend_strength).lower()
            if trend_str in ['weak', 'absent'] and conf_val >= 50:
                confidence_boost += 0.10  # Réduit et conditionné
                reason += " + trend faible"
            elif trend_str == 'moderate':
                confidence_boost += 0.05  # Réduit
                reason += " + trend modéré"
        
        # Confluence score avec rejet
        if conf_val < 40:  # Rejet direct si confluence trop faible
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Rejet Supertrend: confluence insuffisante ({conf_val})",
                "metadata": {"strategy": self.name, "confluence_score": conf_val}
            }
        elif conf_val >= 70:
            confidence_boost += 0.15
            reason += f" + confluence {conf_val:.0f}"
        elif conf_val >= 60:
            confidence_boost += 0.10
        
        # Market regime avec rejets contradictoires
        market_regime = values.get('market_regime')
        if (signal_side == "BUY" and market_regime == "TRENDING_BEAR") or \
           (signal_side == "SELL" and market_regime == "TRENDING_BULL"):
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Rejet {signal_side}: régime contradictoire ({market_regime})",
                "metadata": {"strategy": self.name, "market_regime": market_regime}
            }
        elif market_regime == "RANGING":
            confidence_boost += 0.15  # Reversals excellents en ranging
            reason += " + ranging"
        elif market_regime in ["TRENDING_BULL", "TRENDING_BEAR"]:
            confidence_boost += 0.05  # Bonus réduit (après rejet contradictions)
            reason += " + trending"
        
        # RSI pour timing reversal - plus sélectif
        rsi_14 = values.get('rsi_14')
        if rsi_14:
            try:
                rsi_val = float(rsi_14)
                if signal_side == "BUY" and rsi_val <= 30:  # Plus strict
                    confidence_boost += 0.12  # Oversold = bon timing BUY
                    reason += f" + RSI {rsi_val:.0f}"
                elif signal_side == "SELL" and rsi_val >= 70:  # Plus strict
                    confidence_boost += 0.12  # Overbought = bon timing SELL
                    reason += f" + RSI {rsi_val:.0f}"
            except (ValueError, TypeError):
                pass
        
        # Volatilité crypto bonus
        volatility_regime = values.get('volatility_regime')
        if volatility_regime in ['high', 'extreme']:
            confidence_boost += 0.08  # Haute volatilité = reversals plus forts
            reason += " + volatilité"
        
        # PÉNALITÉ VOLUME - Empêcher les boosts faciles sans volume
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio < 0.8:
                    # Malus pour volume très faible
                    confidence_boost -= 0.20
                    reason += f" - volume très faible ({vol_ratio:.2f}x)"
                elif vol_ratio < 1.1:
                    # Limiter les boosts si volume insuffisant
                    confidence_boost = min(confidence_boost, 0.10)
                    reason += f" - boost limité par volume faible ({vol_ratio:.2f}x)"
            except (ValueError, TypeError):
                pass
        
        # Calcul final
        confidence = min(1.0, self.base_confidence * (1 + confidence_boost))
        strength = self.get_strength_from_confidence(confidence)
        
        return {
            "side": signal_side,
            "confidence": confidence,
            "strength": strength,
            "reason": reason,
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "momentum_score": momentum_val,
                "directional_bias": directional_bias,
                "volume_ratio": volume_ratio,
                "trend_strength": trend_strength,
                "market_regime": market_regime,
                "confluence_score": confluence_score,
                "base_confidence": self.base_confidence,
                "confidence_boost": confidence_boost
            }
        }
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère seulement les indicateurs essentiels."""
        return {
            'directional_bias': self.indicators.get('directional_bias'),
            'momentum_score': self.indicators.get('momentum_score'),
            'trend_strength': self.indicators.get('trend_strength'),
            'volume_ratio': self.indicators.get('volume_ratio'),
            'confluence_score': self.indicators.get('confluence_score'),
            'market_regime': self.indicators.get('market_regime'),
            'rsi_14': self.indicators.get('rsi_14'),
            'volatility_regime': self.indicators.get('volatility_regime')
        }
        
    def validate_data(self) -> bool:
        """Validation ULTRA SIMPLIFIÉE - seulement essentiels."""
        if not super().validate_data():
            return False
            
        # Seulement 1 indicateur requis : trend_strength OU momentum_score
        has_trend = 'trend_strength' in self.indicators and self.indicators['trend_strength'] is not None
        has_momentum = 'momentum_score' in self.indicators and self.indicators['momentum_score'] is not None
        
        if not (has_trend or has_momentum):
            logger.warning(f"{self.name}: Ni trend_strength ni momentum_score disponible")
            return False
                
        return True