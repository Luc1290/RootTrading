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
        # Paramètres Donchian optimisés pour crypto
        self.breakout_threshold = 0.008   # 0.8% au-dessus/en-dessous pour confirmer breakout (évite faux signaux)
        self.min_distance_threshold = 0.01  # Distance minimum aux niveaux (1%) pour éviter le bruit
        self.volume_confirmation = 1.2    # Volume 20% au-dessus de la moyenne pour valider
        self.min_adx_strength = 15        # ADX minimum pour confirmer tendance
        self.momentum_threshold = 45      # Seuil momentum pour confirmation directionnelle
        
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
            # Support/resistance strength sont des VARCHAR (MAJOR, STRONG, MODERATE, WEAK)
            resistance_strength = values.get('resistance_strength')
            support_strength = values.get('support_strength')
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
        base_confidence = 0.50  # Standardisé à 0.50 pour équité avec autres stratégies
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
                if resistance_strength is not None:
                    res_str = str(resistance_strength).upper()
                    if res_str == 'MAJOR':
                        confidence_boost += 0.20
                        reason += " - résistance majeure cassée"
                    elif res_str == 'STRONG':
                        confidence_boost += 0.15
                        reason += " - résistance forte cassée"
                    elif res_str == 'MODERATE':
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
                if support_strength is not None:
                    sup_str = str(support_strength).upper()
                    if sup_str == 'MAJOR':
                        confidence_boost += 0.20
                        reason += " - support majeur cassé"
                    elif sup_str == 'STRONG':
                        confidence_boost += 0.15
                        reason += " - support fort cassé"
                    elif sup_str == 'MODERATE':
                        confidence_boost += 0.10
                        reason += " - support modéré cassé"
                    
        # Filtre anti-faux signaux : vérifier que le breakout est net
        if signal_side is not None:
            # Vérification volume minimum OBLIGATOIRE
            volume_ratio = values.get('volume_ratio')
            if volume_ratio is None or float(volume_ratio) < 1.2:
                signal_side = None
                reason = f"Breakout rejeté : volume insuffisant ({float(volume_ratio):.2f}x < 1.2x)"
                
        # Pas de breakout détecté ou filtré
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
                if adx_val > 30:  # Tendance très forte requise pour breakout
                    confidence_boost += 0.12
                    reason += f" avec ADX très fort ({adx_val:.1f})"
                elif adx_val > 25:
                    confidence_boost += 0.08
                    reason += f" avec ADX fort ({adx_val:.1f})"
                elif adx_val > 20:
                    confidence_boost += 0.04
                    reason += f" avec ADX modéré ({adx_val:.1f})"
                else:
                    # ADX faible = pas de tendance = breakout suspect
                    confidence_boost -= 0.05
                    reason += f" mais ADX faible ({adx_val:.1f})"
                    
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
                
        # Confirmation avec trend_strength (VARCHAR: weak/absent/strong/very_strong/extreme)
        trend_strength = values.get('trend_strength')
        if trend_strength is not None:
            trend_str = str(trend_strength).lower()
            if trend_str in ['extreme', 'very_strong']:
                confidence_boost += 0.15
                reason += f" + trend {trend_str}"
            elif trend_str == 'strong':
                confidence_boost += 0.10
                reason += f" + trend {trend_str}"
            elif trend_str in ['moderate', 'present']:
                confidence_boost += 0.05
                reason += f" + trend {trend_str}"
                
        directional_bias = values.get('directional_bias')
        if directional_bias:
            bias_str = str(directional_bias).upper()
            if (signal_side == "BUY" and bias_str == "BULLISH") or \
               (signal_side == "SELL" and bias_str == "BEARISH"):
                confidence_boost += 0.10
                reason += f" + bias {bias_str.lower()}"
                
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
                    # BUY : confirmation volume progressive et réaliste
                    if vol_ratio >= 1.8:
                        confidence_boost += 0.15  # Volume fort = breakout confirmé
                        reason += f" + volume fort BUY ({vol_ratio:.1f}x)"
                    elif vol_ratio >= 1.5:
                        confidence_boost += 0.12
                        reason += f" + volume élevé ({vol_ratio:.1f}x)"
                    elif vol_ratio >= 1.2:
                        confidence_boost += 0.08
                        reason += f" + volume confirmé ({vol_ratio:.1f}x)"
                    # Pas de pénalité si > 1.2x (déjà filtré avant)
                        
                elif signal_side == "SELL":
                    # SELL : seuils adaptés pour breakdown
                    if vol_ratio >= 1.6:
                        confidence_boost += 0.14  # Volume élevé confirme breakdown
                        reason += f" + volume fort SELL ({vol_ratio:.1f}x)"
                    elif vol_ratio >= 1.3:
                        confidence_boost += 0.11
                        reason += f" + volume élevé ({vol_ratio:.1f}x)"
                    elif vol_ratio >= 1.2:
                        confidence_boost += 0.07  # Volume minimum acceptable
                        reason += f" + volume confirmé ({vol_ratio:.1f}x)"
                    # Pas de pénalité si > 1.2x (déjà filtré avant)
            except (ValueError, TypeError):
                pass
                
        # Volume quality score
        volume_quality_score = values.get('volume_quality_score')
        if volume_quality_score is not None:
            try:
                volume_quality = float(volume_quality_score)
                if volume_quality > 70:  # Format 0-100
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
                
        # Contexte volatilité (format 0-100)
        atr_percentile = values.get('atr_percentile')
        if atr_percentile is not None:
            try:
                atr_perc = float(atr_percentile)
                if atr_perc > 70:  # Volatilité élevée favorable aux breakouts (format 0-100)
                    confidence_boost += 0.08
                    reason += f" + volatilité élevée ({atr_perc:.0f}%)"
                elif atr_perc > 50:
                    confidence_boost += 0.05
                    reason += f" + volatilité modérée ({atr_perc:.0f}%)"
            except (ValueError, TypeError):
                pass
                
        volatility_regime = values.get('volatility_regime')
        if volatility_regime == "extreme":
            confidence_boost += 0.10
            reason += " (vol. extreme)"
        elif volatility_regime == "high":
            confidence_boost += 0.05
            reason += " (vol. high)"
            
        # Signal strength (VARCHAR: WEAK/MODERATE/STRONG/VERY_WEAK)
        signal_strength_calc = values.get('signal_strength')
        if signal_strength_calc is not None:
            sig_str = str(signal_strength_calc).upper()
            if sig_str == 'STRONG':
                confidence_boost += 0.10
                reason += " + signal fort"
            elif sig_str == 'MODERATE':
                confidence_boost += 0.05
                reason += " + signal modéré"
                
        # CORRECTION: Confluence score avec logique directionnelle asymétrique (format 0-100)
        confluence_score = values.get('confluence_score')
        if confluence_score is not None:
            try:
                confluence = float(confluence_score)
                if signal_side == "BUY":
                    # BUY (resistance breakout) : confluence élevée = multiples confirmations haussières
                    if confluence > 80:
                        confidence_boost += 0.16
                        reason += f" + confluence très élevée BUY ({confluence:.0f})"
                    elif confluence > 70:
                        confidence_boost += 0.13
                        reason += f" + confluence élevée ({confluence:.0f})"
                    elif confluence > 60:
                        confidence_boost += 0.10
                        reason += f" + confluence modérée ({confluence:.0f})"
                elif signal_side == "SELL":
                    # SELL (support breakdown) : confluence forte = confirmations baissières multiples
                    if confluence > 75:
                        confidence_boost += 0.18  # Bonus supérieur pour breakdown
                        reason += f" + confluence très élevée SELL ({confluence:.0f})"
                    elif confluence > 65:
                        confidence_boost += 0.14
                        reason += f" + confluence élevée ({confluence:.0f})"
                    elif confluence > 55:
                        confidence_boost += 0.11
                        reason += f" + confluence modérée ({confluence:.0f})"
            except (ValueError, TypeError):
                pass
                
        # Calcul optimisé : additionnel au lieu de multiplicatif pour éviter l'écrasement
        confidence = min(base_confidence + confidence_boost, 0.95)  # Cap à 95% max
        
        # Ajustement final basé sur la force du breakout
        if breakout_type == "resistance_breakout":
            breakout_strength = (current_price - nearest_resistance) / nearest_resistance
            if breakout_strength > 0.03:  # Breakout > 3%
                confidence = min(confidence * 1.15, 0.95)
        elif breakout_type == "support_breakdown":
            breakdown_strength = (nearest_support - current_price) / nearest_support  
            if breakdown_strength > 0.03:  # Breakdown > 3%
                confidence = min(confidence * 1.15, 0.95)
                
        # Vérification finale : au moins 40% de confidence pour émettre un signal
        if confidence < 0.40:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Signal trop faible (conf: {confidence:.2f} < 0.40)",
                "metadata": {"strategy": self.name, "symbol": self.symbol}
            }
            
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
