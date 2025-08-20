"""
Bollinger_Touch_Strategy - Stratégie basée sur les touches des bandes de Bollinger.
OPTIMISÉE POUR CRYPTO SPOT INTRADAY
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
        # Paramètres Bollinger Bands - OPTIMISÉS CRYPTO SPOT
        self.touch_threshold = 0.006   # 0.6% de proximité (plus sensible pour crypto)
        self.bb_position_extreme_buy = 0.08  # Position extrême basse (8% depuis le bas)
        self.bb_position_extreme_sell = 0.92  # Position extrême haute (92% depuis le bas)
        self.bb_width_min = 0.008       # Largeur minimum 0.8% (éviter les squeezes trop serrés)
        self.min_bb_width_for_trade = 0.012  # Ne pas trader si bandes < 1.2%
        self.max_bb_width_for_trade = 0.15   # Ne pas trader si bandes > 15% (trop volatile)
        
        # Paramètres volume (utilisant volume_ratio disponible)
        self.min_volume_ratio = 0.7  # Volume minimum 70% de la moyenne
        self.high_volume_ratio = 1.5  # Volume élevé pour confirmation
        
        # Paramètres RSI adaptés crypto
        self.rsi_oversold_strong = 22  # Survente forte
        self.rsi_oversold = 30         # Survente standard
        self.rsi_overbought = 70       # Surachat standard  
        self.rsi_overbought_strong = 78  # Surachat fort
        
        # Paramètres Stochastic adaptés crypto
        self.stoch_oversold_strong = 12  # Survente forte
        self.stoch_oversold = 20         # Survente standard
        self.stoch_overbought = 80       # Surachat standard
        self.stoch_overbought_strong = 88  # Surachat fort
        
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
            'confluence_score': self.indicators.get('confluence_score'),
            'volume_ratio': self.indicators.get('volume_ratio'),
            'trend_alignment': self.indicators.get('trend_alignment'),
            'ema_7': self.indicators.get('ema_7'),
            'ema_26': self.indicators.get('ema_26'),
            'macd_histogram': self.indicators.get('macd_histogram')
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
            
        # Vérification du volume minimum
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio < self.min_volume_ratio:
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Volume insuffisant ({vol_ratio:.2f}x < {self.min_volume_ratio}x)",
                        "metadata": {"strategy": self.name, "volume_ratio": vol_ratio}
                    }
            except (ValueError, TypeError):
                pass
                
        # Vérification que les bandes sont dans la plage tradable
        if bb_width is not None:
            if bb_width < self.bb_width_min:
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Bollinger squeeze trop serré ({bb_width:.4f} < {self.bb_width_min})",
                    "metadata": {"strategy": self.name, "bb_width": bb_width}
                }
            elif bb_width > self.max_bb_width_for_trade:
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Volatilité excessive ({bb_width:.4f} > {self.max_bb_width_for_trade})",
                    "metadata": {"strategy": self.name, "bb_width": bb_width}
                }
                
        # Calcul des distances aux bandes
        distance_to_upper = abs(current_price - bb_upper) / bb_upper
        distance_to_lower = abs(current_price - bb_lower) / bb_lower
        
        signal_side = None
        reason = ""
        base_confidence = 0.45  # Base plus conservative
        confidence_boost = 0.0
        touch_type = None
        
        # Détection des touches de bandes avec position
        is_touching_upper = distance_to_upper <= self.touch_threshold
        is_touching_lower = distance_to_lower <= self.touch_threshold
        
        # Position dans les bandes pour confirmation
        is_extreme_high = bb_position is not None and bb_position >= self.bb_position_extreme_sell
        is_extreme_low = bb_position is not None and bb_position <= self.bb_position_extreme_buy
        
        # SIGNAL BUY - Touche bande basse ou position extrême basse
        if (is_touching_lower or is_extreme_low) and bb_width > self.min_bb_width_for_trade:
            signal_side = "BUY"
            touch_type = "lower_band"
            
            if is_touching_lower and is_extreme_low:
                reason = f"Touche bande basse confirmée {bb_lower:.2f} (pos: {bb_position:.3f})"
                confidence_boost += 0.15
            elif is_touching_lower:
                reason = f"Touche bande basse {bb_lower:.2f}"
                confidence_boost += 0.08
            else:
                reason = f"Position extrême basse ({bb_position:.3f})"
                confidence_boost += 0.06
                
        # SIGNAL SELL - Touche bande haute ou position extrême haute
        elif (is_touching_upper or is_extreme_high) and bb_width > self.min_bb_width_for_trade:
            signal_side = "SELL"
            touch_type = "upper_band"
            
            if is_touching_upper and is_extreme_high:
                reason = f"Touche bande haute confirmée {bb_upper:.2f} (pos: {bb_position:.3f})"
                confidence_boost += 0.15
            elif is_touching_upper:
                reason = f"Touche bande haute {bb_upper:.2f}"
                confidence_boost += 0.08
            else:
                reason = f"Position extrême haute ({bb_position:.3f})"
                confidence_boost += 0.06
                
        # Pas de touche détectée
        if signal_side is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Pas de setup (pos: {bb_position:.3f}, dist_up: {distance_to_upper:.4f}, dist_low: {distance_to_lower:.4f})",
                "metadata": {
                    "strategy": self.name,
                    "bb_position": bb_position,
                    "distance_to_upper": distance_to_upper,
                    "distance_to_lower": distance_to_lower
                }
            }
            
        # === CONFIRMATIONS AVEC OSCILLATEURS ===
        
        # RSI - Seuils adaptés crypto
        rsi_14 = values.get('rsi_14')
        if rsi_14 is not None:
            try:
                rsi = float(rsi_14)
                if signal_side == "BUY":
                    if rsi <= self.rsi_oversold_strong:
                        confidence_boost += 0.25
                        reason += f" + RSI survente forte ({rsi:.1f})"
                    elif rsi <= self.rsi_oversold:
                        confidence_boost += 0.12
                        reason += f" + RSI survente ({rsi:.1f})"
                    elif rsi <= 40:
                        confidence_boost += 0.05
                        reason += f" + RSI favorable ({rsi:.1f})"
                    elif rsi > 65:  # Pénalité si RSI contradictoire
                        confidence_boost -= 0.20
                        reason += f" MAIS RSI élevé ({rsi:.1f})"
                        
                elif signal_side == "SELL":
                    if rsi >= self.rsi_overbought_strong:
                        confidence_boost += 0.25
                        reason += f" + RSI surachat fort ({rsi:.1f})"
                    elif rsi >= self.rsi_overbought:
                        confidence_boost += 0.12
                        reason += f" + RSI surachat ({rsi:.1f})"
                    elif rsi >= 60:
                        confidence_boost += 0.05
                        reason += f" + RSI favorable ({rsi:.1f})"
                    elif rsi < 35:  # Pénalité si RSI contradictoire
                        confidence_boost -= 0.20
                        reason += f" MAIS RSI faible ({rsi:.1f})"
            except (ValueError, TypeError):
                pass
                
        # Stochastic - Seuils adaptés crypto
        stoch_k = values.get('stoch_k')
        stoch_d = values.get('stoch_d')
        if stoch_k is not None and stoch_d is not None:
            try:
                k = float(stoch_k)
                d = float(stoch_d)
                if signal_side == "BUY":
                    if k <= self.stoch_oversold_strong and d <= self.stoch_oversold_strong:
                        confidence_boost += 0.18
                        reason += f" + Stoch survente forte ({k:.1f}/{d:.1f})"
                    elif k <= self.stoch_oversold and d <= self.stoch_oversold:
                        confidence_boost += 0.10
                        reason += f" + Stoch survente ({k:.1f}/{d:.1f})"
                    elif k > 70:  # Pénalité si contradictoire
                        confidence_boost -= 0.15
                        reason += f" MAIS Stoch élevé ({k:.1f})"
                        
                elif signal_side == "SELL":
                    if k >= self.stoch_overbought_strong and d >= self.stoch_overbought_strong:
                        confidence_boost += 0.18
                        reason += f" + Stoch surachat fort ({k:.1f}/{d:.1f})"
                    elif k >= self.stoch_overbought and d >= self.stoch_overbought:
                        confidence_boost += 0.10
                        reason += f" + Stoch surachat ({k:.1f}/{d:.1f})"
                    elif k < 30:  # Pénalité si contradictoire
                        confidence_boost -= 0.15
                        reason += f" MAIS Stoch faible ({k:.1f})"
            except (ValueError, TypeError):
                pass
                
        # Williams %R
        williams_r = values.get('williams_r')
        if williams_r is not None:
            try:
                wr = float(williams_r)
                if signal_side == "BUY" and wr <= -85:
                    confidence_boost += 0.08
                    reason += f" + Williams%R survente ({wr:.1f})"
                elif signal_side == "SELL" and wr >= -15:
                    confidence_boost += 0.08
                    reason += f" + Williams%R surachat ({wr:.1f})"
            except (ValueError, TypeError):
                pass
                
        # MACD Histogram pour momentum
        macd_hist = values.get('macd_histogram')
        if macd_hist is not None:
            try:
                hist = float(macd_hist)
                if signal_side == "BUY" and hist > 0:
                    confidence_boost += 0.06
                    reason += " + MACD haussier"
                elif signal_side == "BUY" and hist < -0.001:
                    confidence_boost -= 0.10
                    reason += " MAIS MACD baissier"
                elif signal_side == "SELL" and hist < 0:
                    confidence_boost += 0.06
                    reason += " + MACD baissier"
                elif signal_side == "SELL" and hist > 0.001:
                    confidence_boost -= 0.10
                    reason += " MAIS MACD haussier"
            except (ValueError, TypeError):
                pass
                
        # Momentum score (0-100)
        momentum_score = values.get('momentum_score')
        if momentum_score is not None:
            try:
                momentum = float(momentum_score)
                if signal_side == "BUY":
                    if 35 <= momentum <= 50:  # Zone de retournement haussier
                        confidence_boost += 0.12
                        reason += f" + momentum retournement ({momentum:.0f})"
                    elif momentum < 25:  # Très survendu
                        confidence_boost += 0.08
                        reason += f" + momentum extrême bas ({momentum:.0f})"
                    elif momentum > 65:  # Contradictoire
                        confidence_boost -= 0.15
                        reason += f" MAIS momentum élevé ({momentum:.0f})"
                        
                elif signal_side == "SELL":
                    if 50 <= momentum <= 65:  # Zone de retournement baissier
                        confidence_boost += 0.12
                        reason += f" + momentum retournement ({momentum:.0f})"
                    elif momentum > 75:  # Très suracheté
                        confidence_boost += 0.08
                        reason += f" + momentum extrême haut ({momentum:.0f})"
                    elif momentum < 35:  # Contradictoire
                        confidence_boost -= 0.15
                        reason += f" MAIS momentum faible ({momentum:.0f})"
            except (ValueError, TypeError):
                pass
                
        # Trend alignment (0-100) - Nouveau filtre important
        trend_alignment = values.get('trend_alignment')
        if trend_alignment is not None:
            try:
                alignment = float(trend_alignment)
                if signal_side == "BUY":
                    if alignment >= 60:  # Tendance haussière
                        confidence_boost += 0.10
                        reason += f" + tendance alignée ({alignment:.0f})"
                    elif alignment <= 30:  # Tendance baissière forte
                        confidence_boost -= 0.20
                        reason += f" ATTENTION: contre-tendance ({alignment:.0f})"
                        
                elif signal_side == "SELL":
                    if alignment <= 40:  # Tendance baissière
                        confidence_boost += 0.10
                        reason += f" + tendance alignée ({alignment:.0f})"
                    elif alignment >= 70:  # Tendance haussière forte
                        confidence_boost -= 0.20
                        reason += f" ATTENTION: contre-tendance ({alignment:.0f})"
            except (ValueError, TypeError):
                pass
                
        # BB expansion/squeeze context
        bb_expansion = values.get('bb_expansion')
        bb_squeeze = values.get('bb_squeeze')
        
        if bb_expansion is True:
            confidence_boost += 0.08
            reason += " (BB expansion)"
        elif bb_squeeze is False:  # Sortie de squeeze
            confidence_boost += 0.12
            reason += " (sortie squeeze)"
        elif bb_squeeze is True:  # En squeeze
            confidence_boost -= 0.08
            reason += " (squeeze actif)"
            
        # Volatility regime
        volatility_regime = values.get('volatility_regime')
        if volatility_regime == "high":
            confidence_boost += 0.05
            reason += " + volatilité haute"
        elif volatility_regime == "extreme":
            confidence_boost -= 0.10
            reason += " MAIS volatilité extrême"
            
        # Volume confirmation
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= self.high_volume_ratio:
                    confidence_boost += 0.12
                    reason += f" + volume élevé ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.0:
                    confidence_boost += 0.05
                    reason += f" + volume normal ({vol_ratio:.1f}x)"
            except (ValueError, TypeError):
                pass
                
        # Signal strength (WEAK/MODERATE/STRONG)
        signal_strength_calc = values.get('signal_strength')
        if signal_strength_calc is not None:
            sig_str = str(signal_strength_calc).upper()
            if sig_str == 'STRONG':
                confidence_boost += 0.15
                reason += " + signal FORT"
            elif sig_str == 'MODERATE':
                confidence_boost += 0.06
                
        # Confluence score (0-100) - Seuils adaptés crypto
        confluence_score = values.get('confluence_score')
        if confluence_score is not None:
            try:
                confluence = float(confluence_score)
                if confluence >= 70:  # Excellent
                    confidence_boost += 0.20
                    reason += f" + confluence excellente ({confluence:.0f})"
                elif confluence >= 55:  # Bon
                    confidence_boost += 0.10
                    reason += f" + confluence bonne ({confluence:.0f})"
                elif confluence >= 40:  # Acceptable
                    confidence_boost += 0.04
                elif confluence < 30:  # Faible
                    confidence_boost -= 0.12
                    reason += f" mais confluence faible ({confluence:.0f})"
            except (ValueError, TypeError):
                pass
                
        # Calcul final de la confiance
        raw_confidence = base_confidence * (1 + confidence_boost)
        
        # Filtre final - rejeter si confiance trop faible
        if raw_confidence < 0.40:  # Seuil minimum
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Signal rejeté - confiance insuffisante ({raw_confidence:.2f})",
                "metadata": {
                    "strategy": self.name,
                    "rejected_signal": signal_side,
                    "raw_confidence": raw_confidence
                }
            }
        
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
                "trend_alignment": trend_alignment,
                "volume_ratio": volume_ratio,
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