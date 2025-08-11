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
        # Seuils PPO OPTIMISÉS - Zone neutre ajoutée
        self.bullish_threshold = 0.08     # PPO > 0.08% = signal haussier
        self.bearish_threshold = -0.08    # PPO < -0.08% = signal baissier
        self.neutral_zone = 0.08          # Zone neutre ±0.08%
        self.strong_signal_threshold = 0.4  # PPO > 0.4% = signal fort (relevé)
        self.extreme_threshold = 0.8     # PPO > 0.8% = signal très fort (relevé)
        
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
        
        # NOUVELLE LOGIQUE avec zone neutre - Évite les faux signaux
        if abs(ppo_val) < self.neutral_zone:
            # Zone neutre - pas de signal
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"PPO en zone neutre ({ppo_val:.3f}%) - pas de momentum clair",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "ppo": ppo_val,
                    "neutral_zone": self.neutral_zone
                }
            }
            
        elif ppo_val >= self.bullish_threshold:
            # PPO haussier significatif
            signal_side = "BUY"
            
            if ppo_val >= self.extreme_threshold:
                reason = f"PPO très fort ({ppo_val:.3f}%) - momentum haussier extrême"
                confidence_boost += 0.25  # Réduit de 0.3
            elif ppo_val >= self.strong_signal_threshold:
                reason = f"PPO fort ({ppo_val:.3f}%) - momentum haussier fort"
                confidence_boost += 0.18  # Réduit de 0.2
            else:
                reason = f"PPO haussier ({ppo_val:.3f}%) - momentum positif"
                confidence_boost += 0.12  # Augmenté de 0.1
                
        elif ppo_val <= self.bearish_threshold:
            # PPO baissier significatif
            signal_side = "SELL"
            
            if ppo_val <= -self.extreme_threshold:
                reason = f"PPO très faible ({ppo_val:.3f}%) - momentum baissier extrême"
                confidence_boost += 0.25  # Réduit de 0.3
            elif ppo_val <= -self.strong_signal_threshold:
                reason = f"PPO faible ({ppo_val:.3f}%) - momentum baissier fort"
                confidence_boost += 0.18  # Réduit de 0.2
            else:
                reason = f"PPO baissier ({ppo_val:.3f}%) - momentum négatif"
                confidence_boost += 0.12  # Augmenté de 0.1
                
        if signal_side:
            # Base confidence RÉDUITE pour stratégie momentum
            base_confidence = 0.50  # Standardisé à 0.50 pour équité avec autres stratégies
            
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
                    
            # Ajustement avec momentum_score (format 0-100, 50=neutre)
            momentum_score = values.get('momentum_score')
            if momentum_score is not None:
                try:
                    momentum_val = float(momentum_score)
                    # MOMENTUM plus strict pour PPO
                    if (signal_side == "BUY" and momentum_val > 60) or \
                       (signal_side == "SELL" and momentum_val < 40):
                        confidence_boost += 0.12
                        reason += " avec momentum FORT"
                    elif (signal_side == "BUY" and momentum_val > 52) or \
                         (signal_side == "SELL" and momentum_val < 48):
                        confidence_boost += 0.06
                        reason += " avec momentum favorable"
                    elif (signal_side == "BUY" and momentum_val < 40) or \
                         (signal_side == "SELL" and momentum_val > 60):
                        confidence_boost -= 0.15  # Pénalité si momentum contraire
                        reason += " MAIS momentum contraire"
                except (ValueError, TypeError):
                    pass
                    
            # Ajustement avec trend_strength (VARCHAR: weak/moderate/strong/very_strong/extreme)
            trend_strength = values.get('trend_strength')
            if trend_strength is not None:
                trend_str = str(trend_strength).lower()
                if trend_str in ['extreme', 'very_strong']:
                    confidence_boost += 0.15
                    reason += f" et tendance {trend_str}"
                elif trend_str == 'strong':
                    confidence_boost += 0.1
                    reason += f" et tendance {trend_str}"
                elif trend_str == 'moderate':
                    confidence_boost += 0.05
                    reason += f" et tendance {trend_str}"
                    
            # Ajustement avec directional_bias
            directional_bias = values.get('directional_bias')
            if directional_bias:
                if (signal_side == "BUY" and directional_bias == "BULLISH") or \
                   (signal_side == "SELL" and directional_bias == "BEARISH"):
                    confidence_boost += 0.1
                    reason += " confirmé par bias directionnel"
                    
            # Ajustement avec confluence_score (format 0-100)
            confluence_score = values.get('confluence_score')
            if confluence_score is not None:
                try:
                    confluence_val = float(confluence_score)
                    # CONFLUENCE plus stricte
                    if confluence_val > 75:  # Seuil relevé
                        confidence_boost += 0.18
                        reason += f" avec confluence EXCELLENTE ({confluence_val:.0f})"
                    elif confluence_val > 65:  # Seuil relevé
                        confidence_boost += 0.12
                        reason += f" avec confluence élevée ({confluence_val:.0f})"
                    elif confluence_val > 55:
                        confidence_boost += 0.06
                        reason += f" avec confluence modérée ({confluence_val:.0f})"
                    elif confluence_val < 50:  # Pénalité si faible
                        confidence_boost -= 0.08
                        reason += f" mais confluence FAIBLE ({confluence_val:.0f})"
                except (ValueError, TypeError):
                    pass
                    
            # Ajustement avec signal_strength pré-calculé (VARCHAR: WEAK/MODERATE/STRONG)
            signal_strength_calc = values.get('signal_strength')
            if signal_strength_calc is not None:
                sig_str = str(signal_strength_calc).upper()
                if sig_str == 'STRONG':
                    confidence_boost += 0.1
                    reason += " + signal fort"
                elif sig_str == 'MODERATE':
                    confidence_boost += 0.05
                    reason += " + signal modéré"
                    
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
                    
            # NOUVEAU: Filtre final - rejeter si confidence trop faible
            raw_confidence = base_confidence * (1.0 + confidence_boost)
            if raw_confidence < 0.35:  # Seuil minimum 35% pour PPO
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Signal PPO rejeté - confiance insuffisante ({raw_confidence:.2f} < 0.35)",
                    "metadata": {
                        "strategy": self.name,
                        "symbol": self.symbol,
                        "rejected_signal": signal_side,
                        "raw_confidence": raw_confidence,
                        "ppo": ppo_val
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
