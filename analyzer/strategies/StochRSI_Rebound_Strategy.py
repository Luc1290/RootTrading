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
        # Seuils StochRSI - AJUSTÉS (moins stricts)
        self.oversold_zone = 25       # Moins strict: 25 au lieu de 15
        self.overbought_zone = 75     # Moins strict: 75 au lieu de 85
        self.extreme_oversold = 10    # Moins strict: 10 au lieu de 5
        self.extreme_overbought = 90  # Moins strict: 90 au lieu de 95
        self.neutral_low = 35         # Zone neutre basse
        self.neutral_high = 65        # Zone neutre haute
        
    def _safe_float(self, value) -> Optional[float]:
        """Convertit en float de manière sécurisée."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _get_current_values(self) -> Dict[str, Any]:
        """Récupère les valeurs actuelles des indicateurs pré-calculés."""
        return {
            'stoch_rsi': self._safe_float(self.indicators.get('stoch_rsi')),
            'stoch_k': self._safe_float(self.indicators.get('stoch_k')),
            'stoch_d': self._safe_float(self.indicators.get('stoch_d')),
            'stoch_signal': self.indicators.get('stoch_signal'),
            'stoch_divergence': self.indicators.get('stoch_divergence'),
            'rsi_14': self._safe_float(self.indicators.get('rsi_14')),
            'momentum_score': self._safe_float(self.indicators.get('momentum_score')),
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'confluence_score': self._safe_float(self.indicators.get('confluence_score')),
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
            # Conversion des signaux DB vers signaux stratégie
            if stoch_signal == 'OVERSOLD':
                signal_side = "BUY"
                reason = f"Signal StochRSI pré-calculé: {stoch_signal} -> BUY"
                confidence_boost += 0.25  # Augmenté - signal pré-calculé = plus fiable
            elif stoch_signal == 'OVERBOUGHT':
                signal_side = "SELL"
                reason = f"Signal StochRSI pré-calculé: {stoch_signal} -> SELL"
                confidence_boost += 0.25  # Augmenté - signal pré-calculé = plus fiable
        
        # Si pas de signal pré-calculé, analyse manuelle des zones
        if not signal_side:
            if stoch_rsi <= self.oversold_zone:
                signal_side = "BUY"
                zone = "survente extrême" if stoch_rsi <= self.extreme_oversold else "survente"
                reason = f"StochRSI ({stoch_rsi:.1f}) en zone de {zone}"
                confidence_boost += 0.20 if stoch_rsi <= self.extreme_oversold else 0.12  # Augmenté
                
            elif stoch_rsi >= self.overbought_zone:
                signal_side = "SELL"
                zone = "surachat extrême" if stoch_rsi >= self.extreme_overbought else "surachat"
                reason = f"StochRSI ({stoch_rsi:.1f}) en zone de {zone}"
                confidence_boost += 0.20 if stoch_rsi >= self.extreme_overbought else 0.12  # Augmenté
                
        if signal_side:
            base_confidence = 0.50  # Standardisé à 0.50 pour équité avec autres stratégies
            
            # Bonus pour divergence détectée - AUGMENTÉ
            stoch_divergence = values.get('stoch_divergence')
            if stoch_divergence:
                confidence_boost += 0.25  # Augmenté - divergence = signal très fort
                reason += " avec divergence FORTE détectée"
                
            # Confirmation avec croisement K/D
            stoch_k = values.get('stoch_k')
            stoch_d = values.get('stoch_d')
            if stoch_k is not None and stoch_d is not None:
                # Croisement K/D PLUS STRICT
                k_d_diff = abs(stoch_k - stoch_d)
                if (signal_side == "BUY" and stoch_k > stoch_d and k_d_diff > 2) or \
                   (signal_side == "SELL" and stoch_k < stoch_d and k_d_diff > 2):
                    confidence_boost += 0.12  # Augmenté avec séparation minimum
                    reason += f" avec croisement K/D fort ({k_d_diff:.1f})"
                elif (signal_side == "BUY" and stoch_k <= stoch_d) or \
                     (signal_side == "SELL" and stoch_k >= stoch_d):
                    confidence_boost -= 0.10  # Pénalité si croisement défavorable
                    reason += " MAIS K/D défavorable"
                    
            # Confirmation avec RSI
            rsi_14 = values.get('rsi_14')
            if rsi_14:
                # RSI PLUS STRICT pour confirmation
                if (signal_side == "BUY" and rsi_14 <= 30) or \
                   (signal_side == "SELL" and rsi_14 >= 70):
                    confidence_boost += 0.15  # Augmenté avec seuils plus stricts
                    reason += f" CONFIRMÉ par RSI ({rsi_14:.1f})"
                elif (signal_side == "BUY" and rsi_14 <= 35) or \
                     (signal_side == "SELL" and rsi_14 >= 65):
                    confidence_boost += 0.08
                    reason += f" confirmé par RSI ({rsi_14:.1f})"
                elif (signal_side == "BUY" and rsi_14 > 60) or \
                     (signal_side == "SELL" and rsi_14 < 40):
                    confidence_boost -= 0.15  # Pénalité RSI contradictoire
                    reason += f" MAIS RSI contradictoire ({rsi_14:.1f})"
                    
            # Utilisation du momentum_score (format 0-100, 50=neutre)
            momentum_score = values.get('momentum_score', 50)
            if momentum_score:
                # MOMENTUM PLUS STRICT
                if (signal_side == "BUY" and momentum_score > 60) or \
                   (signal_side == "SELL" and momentum_score < 40):
                    confidence_boost += 0.15  # Augmenté avec seuils plus stricts
                    reason += f" avec momentum FORT ({momentum_score:.0f})"
                elif (signal_side == "BUY" and momentum_score > 52) or \
                     (signal_side == "SELL" and momentum_score < 48):
                    confidence_boost += 0.08
                    reason += f" avec momentum favorable ({momentum_score:.0f})"
                elif (signal_side == "BUY" and momentum_score < 45) or \
                     (signal_side == "SELL" and momentum_score > 55):
                    confidence_boost -= 0.12  # Pénalité momentum contraire
                    reason += f" MAIS momentum CONTRAIRE ({momentum_score:.0f})"
                    
            # Utilisation du trend_strength (VARCHAR: weak/moderate/strong/very_strong/extreme)
            trend_strength = values.get('trend_strength')
            if trend_strength:
                trend_str = str(trend_strength).lower()
                # TREND STRENGTH PLUS NUANCÉ
                if trend_str in ['extreme', 'very_strong']:
                    if signal_side == "BUY":
                        confidence_boost += 0.20  # AUGMENTÉ - rebond + trend fort = excellent
                        reason += f" et tendance {trend_str} haussière"
                    else:  # SELL
                        confidence_boost += 0.15  # AUGMENTÉ
                        reason += f" et tendance {trend_str} baissière"
                elif trend_str == 'strong':
                    confidence_boost += 0.15 if signal_side == "BUY" else 0.12  # AUGMENTÉ
                    reason += f" et tendance {trend_str}"
                elif trend_str == 'moderate':
                    confidence_boost += 0.10 if signal_side == "BUY" else 0.08  # AUGMENTÉ
                    reason += f" et tendance {trend_str}"
                elif trend_str in ['weak', 'absent']:  # NOUVEAU: pénalité
                    confidence_boost -= 0.08
                    reason += f" MAIS tendance {trend_str}"
                
            # Utilisation du directional_bias
            directional_bias = values.get('directional_bias')
            if directional_bias:
                bias_upper = directional_bias.upper()
                if (signal_side == "BUY" and bias_upper == "BULLISH") or \
                   (signal_side == "SELL" and bias_upper == "BEARISH"):
                    confidence_boost += 0.12  # AUGMENTÉ
                    reason += f" aligné avec bias {bias_upper}"
                elif (signal_side == "BUY" and bias_upper == "BEARISH") or \
                     (signal_side == "SELL" and bias_upper == "BULLISH"):
                    confidence_boost -= 0.12  # Pénalité bias contraire
                    reason += f" MAIS bias CONTRAIRE ({bias_upper})"
                    
            # Utilisation du confluence_score (format 0-100)
            confluence_score = values.get('confluence_score', 0)
            if confluence_score:
                # CONFLUENCE PLUS STRICTE
                if confluence_score > 80:  # Seuil augmenté
                    confidence_boost += 0.20
                    reason += f" avec confluence EXCELLENTE ({confluence_score:.0f})"
                elif confluence_score > 70:  # Seuil augmenté
                    confidence_boost += 0.15
                    reason += f" avec haute confluence ({confluence_score:.0f})"
                elif confluence_score > 60:  # Seuil augmenté
                    confidence_boost += 0.08
                    reason += f" avec confluence correcte ({confluence_score:.0f})"
                elif confluence_score < 50:  # Pénalité
                    confidence_boost -= 0.10
                    reason += f" mais confluence FAIBLE ({confluence_score:.0f})"
                
            # Utilisation du signal_strength pré-calculé (VARCHAR: WEAK/MODERATE/STRONG)
            signal_strength_calc = values.get('signal_strength')
            if signal_strength_calc:
                sig_str = str(signal_strength_calc).upper()
                if sig_str == 'STRONG':
                    confidence_boost += 0.1
                    reason += " + signal fort"
                elif sig_str == 'MODERATE':
                    confidence_boost += 0.05
                    reason += " + signal modéré"
                
            # NOUVEAU: Filtre final - rejeter si confidence trop faible
            raw_confidence = base_confidence * (1 + confidence_boost)
            if raw_confidence < 0.50:  # Seuil minimum 50% pour StochRSI
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Signal StochRSI rejeté - confiance insuffisante ({raw_confidence:.2f} < 0.50)",
                    "metadata": {
                        "strategy": self.name,
                        "symbol": self.symbol,
                        "rejected_signal": signal_side,
                        "raw_confidence": raw_confidence,
                        "stoch_rsi": stoch_rsi
                    }
                }
            
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
            "reason": f"StochRSI en zone neutre ({stoch_rsi:.1f}) - seuils: {self.oversold_zone}/{self.overbought_zone}",
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "stoch_rsi": stoch_rsi
            }
        }
