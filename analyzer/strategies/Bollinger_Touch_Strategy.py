"""
Bollinger_Touch_Strategy - Stratégie basée sur les touches des bandes de Bollinger.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class Bollinger_Touch_Strategy(BaseStrategy):
    """
    Stratégie utilisant les touches des bandes de Bollinger pour détections de retournements.
    
    Signaux générés:
    - BUY: Prix touche la bande basse + indicateurs de retournement haussier
    - SELL: Prix touche la bande haute + indicateurs de retournement baissier
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres Bollinger Bands
        self.touch_threshold = 0.01    # 1% de proximité pour considérer une "touche"
        self.bb_position_extreme = 0.95  # Position extrême dans les bandes
        self.bb_width_min = 0.02       # Largeur minimum des bandes (éviter squeeze)
        self.reversion_zone = 0.05     # Zone de retournement (5%)
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs Bollinger."""
        return {
            'bb_upper': self.indicators.get('bb_upper'),
            'bb_middle': self.indicators.get('bb_middle'),
            'bb_lower': self.indicators.get('bb_lower'),
            'bb_position': self.indicators.get('bb_position'),
            'bb_width': self.indicators.get('bb_width'),
            'bb_squeeze': self.indicators.get('bb_squeeze'),
            'bb_expansion': self.indicators.get('bb_expansion'),
            'bb_breakout_direction': self.indicators.get('bb_breakout_direction'),
            'rsi_14': self.indicators.get('rsi_14'),
            'stoch_k': self.indicators.get('stoch_k'),
            'stoch_d': self.indicators.get('stoch_d'),
            'williams_r': self.indicators.get('williams_r'),
            'momentum_score': self.indicators.get('momentum_score'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            'signal_strength': self.indicators.get('signal_strength'),
            'confluence_score': self.indicators.get('confluence_score')
        }
        
    def _get_current_price(self) -> Optional[float]:
        """Récupère le prix actuel depuis les données OHLCV."""
        try:
            if self.data and 'close' in self.data and self.data['close']:
                return float(self.data['close'][-1])
        except (IndexError, ValueError, TypeError):
            pass
        return None
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur les touches des bandes de Bollinger.
        """
        if not self.validate_data():
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Données insuffisantes",
                "metadata": {"strategy": self.name}
            }
            
        values = self._get_current_values()
        current_price = self._get_current_price()
        
        # Vérification des indicateurs essentiels
        try:
            bb_upper = float(values['bb_upper']) if values['bb_upper'] is not None else None
            bb_lower = float(values['bb_lower']) if values['bb_lower'] is not None else None
            bb_middle = float(values['bb_middle']) if values['bb_middle'] is not None else None
            bb_position = float(values['bb_position']) if values['bb_position'] is not None else None
            bb_width = float(values['bb_width']) if values['bb_width'] is not None else None
        except (ValueError, TypeError) as e:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Erreur conversion Bollinger: {e}",
                "metadata": {"strategy": self.name}
            }
            
        if bb_upper is None or bb_lower is None or current_price is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Bollinger Bands ou prix non disponibles",
                "metadata": {"strategy": self.name}
            }
            
        # Vérification que les bandes ne sont pas trop compressées (squeeze)
        if bb_width is not None and bb_width < self.bb_width_min:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Bollinger Bands trop compressées ({bb_width:.3f}) - squeeze actif",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "bb_width": bb_width,
                    "current_price": current_price
                }
            }
            
        # Calcul des distances aux bandes
        distance_to_upper = abs(current_price - bb_upper) / bb_upper
        distance_to_lower = abs(current_price - bb_lower) / bb_lower
        
        signal_side = None
        reason = ""
        base_confidence = 0.5
        confidence_boost = 0.0
        touch_type = None
        
        # Détection des touches de bandes
        is_touching_upper = distance_to_upper <= self.touch_threshold
        is_touching_lower = distance_to_lower <= self.touch_threshold
        
        # Position dans les bandes pour confirmation
        is_extreme_high = bb_position is not None and bb_position >= self.bb_position_extreme
        is_extreme_low = bb_position is not None and bb_position <= (1 - self.bb_position_extreme)
        
        # Logique de signal - Touche bande basse = BUY (retournement haussier)
        if is_touching_lower or is_extreme_low:
            signal_side = "BUY"
            touch_type = "lower_band"
            
            if is_touching_lower:
                reason = f"Touche bande basse {bb_lower:.2f} (prix: {current_price:.2f})"
                confidence_boost += 0.15
            else:
                reason = f"Position extrême basse BB ({bb_position:.3f})"
                confidence_boost += 0.10
                
        # Logique de signal - Touche bande haute = SELL (retournement baissier)
        elif is_touching_upper or is_extreme_high:
            signal_side = "SELL"
            touch_type = "upper_band"
            
            if is_touching_upper:
                reason = f"Touche bande haute {bb_upper:.2f} (prix: {current_price:.2f})"
                confidence_boost += 0.15
            else:
                reason = f"Position extrême haute BB ({bb_position:.3f})"
                confidence_boost += 0.10
                
        # Pas de touche détectée
        if signal_side is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Prix au milieu des bandes - pas de setup touch (pos: {bb_position:.3f})",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "current_price": current_price,
                    "bb_upper": bb_upper,
                    "bb_lower": bb_lower,
                    "bb_position": bb_position
                }
            }
            
        # Confirmation avec oscillateurs pour les retournements
        
        # RSI pour confirmation de survente/surachat
        rsi_14 = values.get('rsi_14')
        if rsi_14 is not None:
            try:
                rsi = float(rsi_14)
                if signal_side == "BUY" and rsi <= 30:
                    confidence_boost += 0.15
                    reason += f" + RSI survente ({rsi:.1f})"
                elif signal_side == "SELL" and rsi >= 70:
                    confidence_boost += 0.15
                    reason += f" + RSI surachat ({rsi:.1f})"
                elif signal_side == "BUY" and rsi <= 40:
                    confidence_boost += 0.10
                    reason += f" + RSI favorable ({rsi:.1f})"
                elif signal_side == "SELL" and rsi >= 60:
                    confidence_boost += 0.10
                    reason += f" + RSI favorable ({rsi:.1f})"
            except (ValueError, TypeError):
                pass
                
        # Stochastic pour confirmation
        stoch_k = values.get('stoch_k')
        stoch_d = values.get('stoch_d')
        if stoch_k is not None and stoch_d is not None:
            try:
                k = float(stoch_k)
                d = float(stoch_d)
                if signal_side == "BUY" and k <= 20 and d <= 20:
                    confidence_boost += 0.10
                    reason += f" + Stoch survente ({k:.1f},{d:.1f})"
                elif signal_side == "SELL" and k >= 80 and d >= 80:
                    confidence_boost += 0.10
                    reason += f" + Stoch surachat ({k:.1f},{d:.1f})"
            except (ValueError, TypeError):
                pass
                
        # Williams %R pour confirmation
        williams_r = values.get('williams_r')
        if williams_r is not None:
            try:
                wr = float(williams_r)
                if signal_side == "BUY" and wr <= -80:
                    confidence_boost += 0.08
                    reason += f" + Williams%R survente ({wr:.1f})"
                elif signal_side == "SELL" and wr >= -20:
                    confidence_boost += 0.08
                    reason += f" + Williams%R surachat ({wr:.1f})"
            except (ValueError, TypeError):
                pass
                
        # Momentum score pour éviter les faux signaux
        momentum_score = values.get('momentum_score')
        if momentum_score is not None:
            try:
                momentum = float(momentum_score)
                # Pour touch lower (BUY), on veut un momentum qui commence à remonter (format 0-100)
                if signal_side == "BUY" and momentum > 35:
                    confidence_boost += 0.08
                    reason += " avec momentum stabilisé"
                # Pour touch upper (SELL), on veut un momentum qui commence à baisser  
                elif signal_side == "SELL" and momentum < 65:
                    confidence_boost += 0.08
                    reason += " avec momentum affaibli"
                # Pénalité si momentum trop fort dans la direction opposée
                elif (signal_side == "BUY" and momentum < 25) or \
                     (signal_side == "SELL" and momentum > 75):
                    confidence_boost -= 0.10
                    reason += " mais momentum défavorable"
            except (ValueError, TypeError):
                pass
                
        # BB expansion/squeeze pour contexte (boolean)
        bb_expansion = values.get('bb_expansion')
        bb_squeeze = values.get('bb_squeeze')
        
        if bb_expansion is not None:
            if bb_expansion:  # Boolean: True = expansion
                confidence_boost += 0.05
                reason += " (BB expansion)"
                
        if bb_squeeze is not None:
            if not bb_squeeze:  # Boolean: False = sortie de squeeze
                confidence_boost += 0.10
                reason += " (sortie squeeze)"
            elif bb_squeeze:  # True = encore en squeeze
                confidence_boost -= 0.05  # Pénalité légère
                reason += " (squeeze actif)"
                
        # Volatility regime pour contexte
        volatility_regime = values.get('volatility_regime')
        if volatility_regime == "high":
            confidence_boost += 0.05
            reason += " (volatilité élevée)"
        elif volatility_regime == "normal":
            confidence_boost += 0.03
            
        # Signal strength (varchar: WEAK/MODERATE/STRONG)
        signal_strength_calc = values.get('signal_strength')
        if signal_strength_calc is not None:
            sig_str = str(signal_strength_calc).upper()
            if sig_str == 'STRONG':
                confidence_boost += 0.10
                reason += " + signal fort"
            elif sig_str == 'MODERATE':
                confidence_boost += 0.05
                reason += " + signal modéré"
                
        # CORRECTION: Confluence score (format 0-100)
        confluence_score = values.get('confluence_score')
        if confluence_score is not None:
            try:
                confluence = float(confluence_score)
                if signal_side == "BUY":
                    # BUY (lower band touch) : confluence modérée suffisante (oversold confluence)
                    if confluence > 75:
                        confidence_boost += 0.15
                        reason += " + confluence très élevée BUY"
                    elif confluence > 65:
                        confidence_boost += 0.12
                        reason += " + confluence élevée"
                    elif confluence > 55:
                        confidence_boost += 0.08
                        reason += " + confluence modérée"
                elif signal_side == "SELL":
                    # SELL (upper band touch) : confluence élevée requise (overbought confirmation)
                    if confluence > 80:
                        confidence_boost += 0.18  # Bonus supérieur pour SELL
                        reason += " + confluence très élevée SELL"
                    elif confluence > 70:  
                        confidence_boost += 0.14
                        reason += " + confluence élevée"
                    elif confluence > 60:
                        confidence_boost += 0.09
                        reason += " + confluence modérée"
            except (ValueError, TypeError):
                pass
                
        confidence = self.calculate_confidence(base_confidence, 1 + confidence_boost)
        strength = self.get_strength_from_confidence(confidence)
        
        return {
            "side": signal_side,
            "confidence": confidence,
            "strength": strength,
            "reason": reason,
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "current_price": current_price,
                "bb_upper": bb_upper,
                "bb_middle": bb_middle,
                "bb_lower": bb_lower,
                "bb_position": bb_position,
                "bb_width": bb_width,
                "touch_type": touch_type,
                "distance_to_upper": distance_to_upper,
                "distance_to_lower": distance_to_lower,
                "rsi_14": rsi_14,
                "stoch_k": stoch_k,
                "stoch_d": stoch_d,
                "williams_r": williams_r,
                "momentum_score": momentum_score,
                "bb_expansion": bb_expansion,
                "bb_squeeze": bb_squeeze,
                "confluence_score": confluence_score
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que tous les indicateurs Bollinger requis sont présents."""
        if not super().validate_data():
            return False
            
        required = ['bb_upper', 'bb_lower']
        
        for indicator in required:
            if indicator not in self.indicators:
                logger.warning(f"{self.name}: Indicateur manquant: {indicator}")
                return False
            if self.indicators[indicator] is None:
                logger.warning(f"{self.name}: Indicateur null: {indicator}")
                return False
                
        # Vérifier aussi qu'on a des données de prix
        if not self.data or 'close' not in self.data or not self.data['close']:
            logger.warning(f"{self.name}: Données de prix manquantes")
            return False
            
        return True
