"""
Donchian_Breakout_Strategy - Stratégie basée sur les breakouts des canaux de Donchian.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class Donchian_Breakout_Strategy(BaseStrategy):
    """
    Stratégie utilisant les canaux de Donchian pour détecter les breakouts de tendance.
    
    Signaux générés:
    - BUY: Prix casse au-dessus du canal haut + confirmation tendance haussière
    - SELL: Prix casse en-dessous du canal bas + confirmation tendance baissière
    
    Note: Les canaux de Donchian sont construits avec les plus hauts/plus bas sur N périodes.
    Cette stratégie utilise les niveaux de support/résistance comme proxy des canaux.
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres Donchian (via support/résistance)
        self.breakout_threshold = 0.005   # 0.5% au-dessus/en-dessous pour confirmer breakout
        self.min_distance_threshold = 0.01  # Distance minimum aux niveaux (1%)
        self.volume_confirmation = 1.2    # Volume 20% au-dessus de la moyenne
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs."""
        return {
            # Support/Résistance (proxy pour canaux Donchian)
            'nearest_support': self.indicators.get('nearest_support'),
            'nearest_resistance': self.indicators.get('nearest_resistance'),
            'support_strength': self.indicators.get('support_strength'),
            'resistance_strength': self.indicators.get('resistance_strength'),
            'break_probability': self.indicators.get('break_probability'),
            'pivot_count': self.indicators.get('pivot_count'),
            # Tendance et momentum
            'adx_14': self.indicators.get('adx_14'),
            'plus_di': self.indicators.get('plus_di'),
            'minus_di': self.indicators.get('minus_di'),
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'momentum_score': self.indicators.get('momentum_score'),
            # Volume
            'volume_ratio': self.indicators.get('volume_ratio'),
            'relative_volume': self.indicators.get('relative_volume'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            # ATR pour contexte volatilité
            'atr_14': self.indicators.get('atr_14'),
            'atr_percentile': self.indicators.get('atr_percentile'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            # Confluence
            'signal_strength': self.indicators.get('signal_strength'),
            'confluence_score': self.indicators.get('confluence_score')
        }
        
    def _get_current_price_data(self) -> Dict[str, Optional[float]]:
        """Récupère les données de prix actuelles."""
        try:
            if self.data and 'close' in self.data and self.data['close'] and \
               'high' in self.data and self.data['high'] and \
               'low' in self.data and self.data['low'] and \
               'volume' in self.data and self.data['volume']:
                return {
                    'current_price': float(self.data['close'][-1]),
                    'current_high': float(self.data['high'][-1]),
                    'current_low': float(self.data['low'][-1]),
                    'current_volume': float(self.data['volume'][-1])
                }
        except (IndexError, ValueError, TypeError):
            pass
        return {'current_price': None, 'current_high': None, 'current_low': None, 'current_volume': None}
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur les breakouts de canaux de Donchian.
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
        price_data = self._get_current_price_data()
        
        # Vérification des données essentielles
        current_price = price_data['current_price']
        current_high = price_data['current_high']
        current_low = price_data['current_low']
        current_volume = price_data['current_volume']
        
        if current_price is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Prix non disponible",
                "metadata": {"strategy": self.name}
            }
            
        # Récupération des niveaux (proxy pour canaux Donchian)
        try:
            nearest_resistance = float(values['nearest_resistance']) if values['nearest_resistance'] is not None else None
            nearest_support = float(values['nearest_support']) if values['nearest_support'] is not None else None
            resistance_strength = float(values['resistance_strength']) if values['resistance_strength'] is not None else None
            support_strength = float(values['support_strength']) if values['support_strength'] is not None else None
        except (ValueError, TypeError) as e:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Erreur conversion niveaux: {e}",
                "metadata": {"strategy": self.name}
            }
            
        signal_side = None
        reason = ""
        base_confidence = 0.5
        confidence_boost = 0.0
        breakout_type = None
        
        # Analyse breakout au-dessus de la résistance (canal haut)
        if nearest_resistance is not None and nearest_resistance > 0:
            breakout_distance = (current_price - nearest_resistance) / nearest_resistance
            
            # Breakout haussier confirmé
            if breakout_distance >= self.breakout_threshold:
                signal_side = "BUY"
                breakout_type = "resistance_breakout"
                reason = f"Breakout résistance {nearest_resistance:.2f} (+{breakout_distance*100:.1f}%)"
                confidence_boost += 0.20
                
                # Bonus si résistance était forte = breakout plus significatif
                if resistance_strength is not None and resistance_strength > 0.7:
                    confidence_boost += 0.15
                    reason += " - résistance forte cassée"
                elif resistance_strength is not None and resistance_strength > 0.5:
                    confidence_boost += 0.10
                    reason += " - résistance modérée cassée"
                    
        # Analyse breakdown en-dessous du support (canal bas)
        if signal_side is None and nearest_support is not None and nearest_support > 0:
            breakdown_distance = (nearest_support - current_price) / nearest_support
            
            # Breakdown baissier confirmé
            if breakdown_distance >= self.breakout_threshold:
                signal_side = "SELL"
                breakout_type = "support_breakdown"
                reason = f"Breakdown support {nearest_support:.2f} (-{breakdown_distance*100:.1f}%)"
                confidence_boost += 0.20
                
                # Bonus si support était fort = breakdown plus significatif
                if support_strength is not None and support_strength > 0.7:
                    confidence_boost += 0.15
                    reason += " - support fort cassé"
                elif support_strength is not None and support_strength > 0.5:
                    confidence_boost += 0.10
                    reason += " - support modéré cassé"
                    
        # Pas de breakout détecté
        if signal_side is None:
            proximity_info = ""
            if nearest_resistance is not None and nearest_resistance > 0:
                dist_res = abs(current_price - nearest_resistance) / nearest_resistance
                proximity_info += f"rés: {dist_res*100:.1f}%"
            if nearest_support is not None and nearest_support > 0:
                dist_sup = abs(current_price - nearest_support) / nearest_support
                if proximity_info:
                    proximity_info += ", "
                proximity_info += f"sup: {dist_sup*100:.1f}%"
                
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Pas de breakout Donchian détecté ({proximity_info})",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "current_price": current_price,
                    "nearest_resistance": nearest_resistance,
                    "nearest_support": nearest_support
                }
            }
            
        # Confirmation avec tendance (ADX et DI)
        adx = values.get('adx_14')
        plus_di = values.get('plus_di') 
        minus_di = values.get('minus_di')
        
        if adx is not None:
            try:
                adx_val = float(adx)
                if adx_val > 25:  # Tendance forte
                    confidence_boost += 0.10
                    reason += f" avec ADX fort ({adx_val:.1f})"
                    
                    # Confirmation directionnelle
                    if plus_di is not None and minus_di is not None:
                        try:
                            plus_di_val = float(plus_di)
                            minus_di_val = float(minus_di)
                            
                            if signal_side == "BUY" and plus_di_val > minus_di_val:
                                confidence_boost += 0.10
                                reason += " + DI haussier"
                            elif signal_side == "SELL" and minus_di_val > plus_di_val:
                                confidence_boost += 0.10
                                reason += " + DI baissier"
                        except (ValueError, TypeError):
                            pass
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec trend_strength et directional_bias
        trend_strength = values.get('trend_strength')
        if trend_strength is not None:
            try:
                trend_str = float(trend_strength)
                if trend_str > 0.6:
                    confidence_boost += 0.08
                    reason += f" (trend: {trend_str:.2f})"
            except (ValueError, TypeError):
                pass
                
        directional_bias = values.get('directional_bias')
        if directional_bias:
            if (signal_side == "BUY" and directional_bias == "bullish") or \
               (signal_side == "SELL" and directional_bias == "bearish"):
                confidence_boost += 0.10
                reason += f" + bias {directional_bias}"
                
        # Confirmation avec momentum
        momentum_score = values.get('momentum_score')
        if momentum_score is not None:
            try:
                momentum = float(momentum_score)
                # Format 0-100, 50=neutre
                if (signal_side == "BUY" and momentum > 60) or \
                   (signal_side == "SELL" and momentum < 40):
                    confidence_boost += 0.10
                    reason += " + momentum favorable"
            except (ValueError, TypeError):
                pass
                
        # CORRECTION: Volume confirmation directionnelle asymétrique pour breakouts
        volume_ratio = values.get('volume_ratio')
        relative_volume = values.get('relative_volume')
        
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                
                if signal_side == "BUY":
                    # BUY (resistance breakout) : volume élevé crucial (buying pressure)
                    if vol_ratio >= 2.0:
                        confidence_boost += 0.20  # Volume très élevé = breakout fort
                        reason += f" + volume très élevé BUY ({vol_ratio:.1f}x)"
                    elif vol_ratio >= 1.5:
                        confidence_boost += 0.16
                        reason += f" + volume élevé BUY ({vol_ratio:.1f}x)"
                    elif vol_ratio >= 1.2:
                        confidence_boost += 0.10
                        reason += f" + volume modéré ({vol_ratio:.1f}x)"
                    else:
                        confidence_boost -= 0.10  # Pénalité forte pour breakout sans volume
                        reason += f" mais volume insuffisant BUY ({vol_ratio:.1f}x)"
                        
                elif signal_side == "SELL":
                    # SELL (support breakdown) : volume fort requis mais naturellement plus bas
                    if vol_ratio >= 1.8:
                        confidence_boost += 0.18  # Volume élevé confirme breakdown
                        reason += f" + volume élevé SELL ({vol_ratio:.1f}x)"
                    elif vol_ratio >= 1.3:
                        confidence_boost += 0.14
                        reason += f" + volume confirmé SELL ({vol_ratio:.1f}x)"
                    elif vol_ratio >= 1.0:
                        confidence_boost += 0.08  # Volume normal acceptable pour SELL
                        reason += f" + volume normal ({vol_ratio:.1f}x)"
                    else:
                        confidence_boost -= 0.08  # Pénalité modérée pour SELL
                        reason += f" mais volume faible ({vol_ratio:.1f}x)"
            except (ValueError, TypeError):
                pass
                
        # Volume quality score
        volume_quality_score = values.get('volume_quality_score')
        if volume_quality_score is not None:
            try:
                vol_quality = float(volume_quality_score)
                if vol_quality > 0.7:
                    confidence_boost += 0.08
                    reason += " + volume qualité"
            except (ValueError, TypeError):
                pass
                
        # Break probability
        break_probability = values.get('break_probability')
        if break_probability is not None:
            try:
                break_prob = float(break_probability)
                if break_prob > 0.6:
                    confidence_boost += 0.10
                    reason += f" (prob: {break_prob:.2f})"
            except (ValueError, TypeError):
                pass
                
        # Contexte volatilité
        atr_percentile = values.get('atr_percentile')
        if atr_percentile is not None:
            try:
                atr_perc = float(atr_percentile)
                if atr_perc > 0.7:  # Volatilité élevée favorable aux breakouts
                    confidence_boost += 0.08
                    reason += " + volatilité élevée"
            except (ValueError, TypeError):
                pass
                
        volatility_regime = values.get('volatility_regime')
        if volatility_regime == "expanding":
            confidence_boost += 0.10
            reason += " (vol. expansion)"
        elif volatility_regime == "high":
            confidence_boost += 0.05
            
        # Signal strength et confluence
        signal_strength_calc = values.get('signal_strength')
        if signal_strength_calc is not None:
            try:
                sig_str = float(signal_strength_calc)
                if sig_str > 0.7:
                    confidence_boost += 0.05
            except (ValueError, TypeError):
                pass
                
        # CORRECTION: Confluence score avec logique directionnelle asymétrique
        confluence_score = values.get('confluence_score')
        if confluence_score is not None:
            try:
                confluence = float(confluence_score)
                if signal_side == "BUY":
                    # BUY (resistance breakout) : confluence élevée = multiples confirmations haussières
                    if confluence > 0.8:
                        confidence_boost += 0.16
                        reason += " + confluence très élevée BUY"
                    elif confluence > 0.7:
                        confidence_boost += 0.13
                        reason += " + confluence élevée"
                    elif confluence > 0.6:
                        confidence_boost += 0.10
                        reason += " + confluence modérée"
                elif signal_side == "SELL":
                    # SELL (support breakdown) : confluence forte = confirmations baissières multiples
                    if confluence > 0.75:
                        confidence_boost += 0.18  # Bonus supérieur pour breakdown
                        reason += " + confluence très élevée SELL"
                    elif confluence > 0.65:
                        confidence_boost += 0.14
                        reason += " + confluence élevée"
                    elif confluence > 0.55:
                        confidence_boost += 0.11
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
                "current_high": current_high,
                "current_low": current_low,
                "breakout_type": breakout_type,
                "nearest_resistance": nearest_resistance,
                "nearest_support": nearest_support,
                "resistance_strength": resistance_strength,
                "support_strength": support_strength,
                "adx_14": adx,
                "plus_di": plus_di,
                "minus_di": minus_di,
                "trend_strength": trend_strength,
                "directional_bias": directional_bias,
                "momentum_score": momentum_score,
                "volume_ratio": volume_ratio,
                "relative_volume": relative_volume,
                "break_probability": break_probability,
                "atr_percentile": atr_percentile,
                "confluence_score": confluence_score
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que toutes les données requises sont présentes."""
        if not super().validate_data():
            return False
            
        # Pour cette stratégie, on a besoin au minimum des niveaux de support/résistance
        required = ['nearest_support', 'nearest_resistance']
        has_level = False
        
        for indicator in required:
            if indicator in self.indicators and self.indicators[indicator] is not None:
                has_level = True
                break
                
        if not has_level:
            logger.warning(f"{self.name}: Aucun niveau support/résistance disponible")
            return False
            
        # Vérifier aussi qu'on a des données de prix
        if not self.data or 'close' not in self.data or not self.data['close']:
            logger.warning(f"{self.name}: Données de prix manquantes")
            return False
            
        return True
