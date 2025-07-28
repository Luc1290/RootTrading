"""
Range_Breakout_Confirmation_Strategy - Stratégie basée sur les breakouts de range avec confirmation.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class Range_Breakout_Confirmation_Strategy(BaseStrategy):
    """
    Stratégie utilisant les breakouts de ranges (consolidation) avec confirmations multiples.
    
    Un range breakout se produit quand le prix sort d'une zone de consolidation horizontale :
    - Range = zone entre support et résistance bien définis
    - Breakout = prix casse avec volume et momentum
    - Confirmation = reteste et confirme la cassure
    
    Signaux générés:
    - BUY: Breakout au-dessus résistance + volume + momentum + confirmations
    - SELL: Breakout en-dessous support + volume + momentum + confirmations
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres Range Breakout
        self.min_range_width = 0.01  # Largeur minimum du range (1%)
        self.max_range_width = 0.08  # Largeur maximum du range (8%)
        self.breakout_threshold = 0.002  # Distance minimum pour considérer un breakout (0.2%)
        self.volume_breakout_threshold = 1.5  # Volume minimum pour breakout valide
        self.retest_tolerance = 0.005  # Tolérance pour retest du niveau cassé (0.5%)
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs pour range breakout."""
        return {
            # Support/Résistance pour définir le range
            'support_levels': self.indicators.get('support_levels'),
            'resistance_levels': self.indicators.get('resistance_levels'),
            'nearest_support': self.indicators.get('nearest_support'),
            'nearest_resistance': self.indicators.get('nearest_resistance'),
            'support_strength': self.indicators.get('support_strength'),
            'resistance_strength': self.indicators.get('resistance_strength'),
            'break_probability': self.indicators.get('break_probability'),
            # Bollinger Bands pour range identification
            'bb_upper': self.indicators.get('bb_upper'),
            'bb_lower': self.indicators.get('bb_lower'),
            'bb_width': self.indicators.get('bb_width'),
            'bb_position': self.indicators.get('bb_position'),
            'bb_squeeze': self.indicators.get('bb_squeeze'),
            'bb_expansion': self.indicators.get('bb_expansion'),
            'bb_breakout_direction': self.indicators.get('bb_breakout_direction'),
            # Volume pour confirmation breakout
            'volume_ratio': self.indicators.get('volume_ratio'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            'trade_intensity': self.indicators.get('trade_intensity'),
            'relative_volume': self.indicators.get('relative_volume'),
            'volume_buildup_periods': self.indicators.get('volume_buildup_periods'),
            'volume_spike_multiplier': self.indicators.get('volume_spike_multiplier'),
            # Momentum pour confirmer la direction
            'momentum_score': self.indicators.get('momentum_score'),
            'rsi_14': self.indicators.get('rsi_14'),
            'stoch_k': self.indicators.get('stoch_k'),
            'stoch_d': self.indicators.get('stoch_d'),
            'williams_r': self.indicators.get('williams_r'),
            'cci_20': self.indicators.get('cci_20'),
            # ADX pour force du breakout
            'adx_14': self.indicators.get('adx_14'),
            'plus_di': self.indicators.get('plus_di'),
            'minus_di': self.indicators.get('minus_di'),
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            # ATR pour volatilité et targets
            'atr_14': self.indicators.get('atr_14'),
            'atr_percentile': self.indicators.get('atr_percentile'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            # VWAP levels
            'vwap_10': self.indicators.get('vwap_10'),
            'vwap_upper_band': self.indicators.get('vwap_upper_band'),
            'vwap_lower_band': self.indicators.get('vwap_lower_band'),
            'anchored_vwap': self.indicators.get('anchored_vwap'),
            # Market context
            'market_regime': self.indicators.get('market_regime'),
            'regime_strength': self.indicators.get('regime_strength'),
            'regime_confidence': self.indicators.get('regime_confidence'),
            # Pattern recognition
            'pattern_detected': self.indicators.get('pattern_detected'),
            'pattern_confidence': self.indicators.get('pattern_confidence'),
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
        
    def _get_recent_highs_lows(self) -> Optional[Dict[str, float]]:
        """Récupère les highs/lows récents pour analyser le range."""
        try:
            if self.data and all(k in self.data for k in ['high', 'low', 'close']):
                if len(self.data['high']) >= 10 and len(self.data['low']) >= 10:
                    recent_highs = [float(h) for h in self.data['high'][-10:]]
                    recent_lows = [float(l) for l in self.data['low'][-10:]]
                    recent_closes = [float(c) for c in self.data['close'][-5:]]
                    
                    return {
                        'recent_high': max(recent_highs),
                        'recent_low': min(recent_lows),
                        'avg_recent_high': sum(recent_highs[-5:]) / 5,
                        'avg_recent_low': sum(recent_lows[-5:]) / 5,
                        'price_trend': recent_closes[-1] - recent_closes[0]  # Direction sur 5 périodes
                    }
        except (IndexError, ValueError, TypeError):
            pass
        return None
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur les breakouts de range confirmés.
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
        price_data = self._get_recent_highs_lows()
        
        if current_price is None or price_data is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Prix ou données historiques non disponibles",
                "metadata": {"strategy": self.name}
            }
            
        # Identifier le range principal
        range_info = self._identify_range(values, current_price, price_data)
        if range_info is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Aucun range clair identifié",
                "metadata": {"strategy": self.name}
            }
            
        # Analyser le breakout
        breakout_analysis = self._analyze_breakout(values, current_price, range_info, price_data)
        if breakout_analysis is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Aucun breakout détecté",
                "metadata": {"strategy": self.name}
            }
            
        return self._create_breakout_signal(values, current_price, range_info, breakout_analysis, price_data)
        
    def _identify_range(self, values: Dict[str, Any], current_price: float, price_data: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Identifie un range trading valide."""
        range_info = None
        
        # Méthode 1: Support/Résistance explicites
        nearest_support = values.get('nearest_support')
        nearest_resistance = values.get('nearest_resistance')
        
        if nearest_support is not None and nearest_resistance is not None:
            try:
                support_val = float(nearest_support)
                resistance_val = float(nearest_resistance)
                
                if support_val < resistance_val:  # Vérification logique
                    range_width = (resistance_val - support_val) / support_val
                    
                    if self.min_range_width <= range_width <= self.max_range_width:
                        # Vérifier que le prix est dans le range ou près des bords
                        price_in_range = support_val <= current_price <= resistance_val
                        near_edges = (abs(current_price - support_val) / current_price <= 0.01 or
                                    abs(current_price - resistance_val) / current_price <= 0.01)
                        
                        if price_in_range or near_edges:
                            range_info = {
                                'method': 'support_resistance',
                                'support': support_val,
                                'resistance': resistance_val,
                                'width': range_width,
                                'mid_point': (support_val + resistance_val) / 2
                            }
            except (ValueError, TypeError):
                pass
                
        # Méthode 2: Bollinger Bands squeeze (range étroit)
        if range_info is None:
            bb_upper = values.get('bb_upper')
            bb_lower = values.get('bb_lower')
            bb_squeeze = values.get('bb_squeeze')
            
            if all(x is not None for x in [bb_upper, bb_lower]) and bb_squeeze:
                try:
                    upper_val = float(bb_upper)
                    lower_val = float(bb_lower)
                    range_width = (upper_val - lower_val) / lower_val
                    
                    if self.min_range_width <= range_width <= self.max_range_width:
                        range_info = {
                            'method': 'bollinger_squeeze',
                            'support': lower_val,
                            'resistance': upper_val,
                            'width': range_width,
                            'mid_point': (upper_val + lower_val) / 2
                        }
                except (ValueError, TypeError):
                    pass
                    
        # Méthode 3: Highs/Lows récents
        if range_info is None:
            recent_high = price_data.get('recent_high')
            recent_low = price_data.get('recent_low')
            
            if recent_high is not None and recent_low is not None:
                range_width = (recent_high - recent_low) / recent_low
                
                if self.min_range_width <= range_width <= self.max_range_width:
                    # Vérifier que les prix récents restent dans ce range
                    price_stability = abs(price_data.get('price_trend', 0)) < (recent_high - recent_low) * 0.3
                    
                    if price_stability:
                        range_info = {
                            'method': 'recent_highs_lows',
                            'support': recent_low,
                            'resistance': recent_high,
                            'width': range_width,
                            'mid_point': (recent_high + recent_low) / 2
                        }
                        
        return range_info
        
    def _analyze_breakout(self, values: Dict[str, Any], current_price: float, 
                         range_info: Dict[str, Any], price_data: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Analyse si un breakout valide est en cours."""
        support = range_info['support']
        resistance = range_info['resistance']
        
        # Déterminer le type de breakout
        breakout_type = None
        breakout_level = None
        breakout_distance = 0
        
        # Breakout haussier (au-dessus résistance)
        if current_price > resistance:
            breakout_distance = (current_price - resistance) / resistance
            if breakout_distance >= self.breakout_threshold:
                breakout_type = "bullish"
                breakout_level = resistance
                
        # Breakout baissier (en-dessous support)  
        elif current_price < support:
            breakout_distance = (support - current_price) / current_price
            if breakout_distance >= self.breakout_threshold:
                breakout_type = "bearish"
                breakout_level = support
                
        if breakout_type is None:
            return None
            
        # Vérifier les confirmations du breakout
        confirmations = self._check_breakout_confirmations(values, breakout_type, current_price)
        
        if confirmations['volume_confirmed'] or confirmations['momentum_confirmed']:
            return {
                'type': breakout_type,
                'level': breakout_level,
                'distance': breakout_distance,
                'confirmations': confirmations
            }
            
        return None
        
    def _check_breakout_confirmations(self, values: Dict[str, Any], breakout_type: str, current_price: float) -> Dict[str, bool]:
        """Vérifie les confirmations du breakout."""
        confirmations = {
            'volume_confirmed': False,
            'momentum_confirmed': False,
            'trend_confirmed': False,
            'pattern_confirmed': False
        }
        
        # Confirmation Volume
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= self.volume_breakout_threshold:
                    confirmations['volume_confirmed'] = True
            except (ValueError, TypeError):
                pass
                
        # Confirmation Momentum
        momentum_score = values.get('momentum_score')
        rsi_14 = values.get('rsi_14')
        
        if momentum_score is not None:
            try:
                momentum = float(momentum_score)
                if breakout_type == "bullish" and momentum > 0.2:
                    confirmations['momentum_confirmed'] = True
                elif breakout_type == "bearish" and momentum < -0.2:
                    confirmations['momentum_confirmed'] = True
            except (ValueError, TypeError):
                pass
                
        # RSI comme confirmation momentum additionnelle
        if rsi_14 is not None and not confirmations['momentum_confirmed']:
            try:
                rsi = float(rsi_14)
                if breakout_type == "bullish" and rsi > 55:
                    confirmations['momentum_confirmed'] = True
                elif breakout_type == "bearish" and rsi < 45:
                    confirmations['momentum_confirmed'] = True
            except (ValueError, TypeError):
                pass
                
        # Confirmation Trend (ADX)
        adx_14 = values.get('adx_14')
        plus_di = values.get('plus_di')
        minus_di = values.get('minus_di')
        
        if all(x is not None for x in [adx_14, plus_di, minus_di]):
            try:
                adx = float(adx_14)
                plus_val = float(plus_di)
                minus_val = float(minus_di)
                
                if adx > 20:  # Tendance en formation
                    if breakout_type == "bullish" and plus_val > minus_val:
                        confirmations['trend_confirmed'] = True
                    elif breakout_type == "bearish" and minus_val > plus_val:
                        confirmations['trend_confirmed'] = True
            except (ValueError, TypeError):
                pass
                
        # Pattern confirmation
        bb_breakout_direction = values.get('bb_breakout_direction')
        if bb_breakout_direction is not None:
            if (breakout_type == "bullish" and bb_breakout_direction == "up") or \
               (breakout_type == "bearish" and bb_breakout_direction == "down"):
                confirmations['pattern_confirmed'] = True
                
        return confirmations
        
    def _create_breakout_signal(self, values: Dict[str, Any], current_price: float,
                               range_info: Dict[str, Any], breakout_analysis: Dict[str, Any],
                               price_data: Dict[str, float]) -> Dict[str, Any]:
        """Crée le signal final pour le breakout."""
        breakout_type = breakout_analysis['type']
        breakout_level = breakout_analysis['level']
        confirmations = breakout_analysis['confirmations']
        
        signal_side = "BUY" if breakout_type == "bullish" else "SELL"
        base_confidence = 0.6  # Base plus élevée pour breakouts
        confidence_boost = 0.0
        
        # Construction de la raison
        direction = "au-dessus résistance" if breakout_type == "bullish" else "en-dessous support"
        reason = f"Breakout {direction} {breakout_level:.4f} - distance {breakout_analysis['distance']:.3f}"
        
        # Bonus selon les confirmations
        confirmed_count = sum(confirmations.values())
        confidence_boost += confirmed_count * 0.1  # +10% par confirmation
        
        if confirmations['volume_confirmed']:
            reason += " + volume confirmé"
        if confirmations['momentum_confirmed']:
            reason += " + momentum confirmé"
        if confirmations['trend_confirmed']:
            reason += " + tendance confirmée"
        if confirmations['pattern_confirmed']:
            reason += " + pattern confirmé"
            
        # Bonus selon largeur du range (ranges plus larges = breakouts plus significatifs)
        range_width = range_info['width']
        if range_width > 0.04:  # Range large (>4%)
            confidence_boost += 0.15
            reason += " - range large"
        elif range_width > 0.02:  # Range moyen (>2%)
            confidence_boost += 0.10
            reason += " - range moyen"
        else:
            confidence_boost += 0.05
            reason += " - range étroit"
            
        # Confirmation additionnelle avec VWAP
        vwap_10 = values.get('vwap_10')
        if vwap_10 is not None:
            try:
                vwap = float(vwap_10)
                if (signal_side == "BUY" and current_price > vwap) or \
                   (signal_side == "SELL" and current_price < vwap):
                    confidence_boost += 0.08
                    reason += " + VWAP aligné"
            except (ValueError, TypeError):
                pass
                
        # ATR pour contexte volatilité
        atr_percentile = values.get('atr_percentile')
        if atr_percentile is not None:
            try:
                atr_pct = float(atr_percentile)
                if 30 <= atr_pct <= 70:  # Volatilité normale
                    confidence_boost += 0.05
                    reason += " + volatilité normale"
                elif atr_pct > 80:  # Volatilité très élevée
                    confidence_boost -= 0.05
                    reason += " mais volatilité élevée"
            except (ValueError, TypeError):
                pass
                
        # Market regime
        market_regime = values.get('market_regime')
        if market_regime == "ranging":
            confidence_boost += 0.12  # Breakouts plus significatifs après ranging
            reason += " (sortie ranging)"
        elif market_regime == "trending":
            confidence_boost += 0.08  # Continuation de tendance
            reason += " (continuation trend)"
            
        # Confluence score
        confluence_score = values.get('confluence_score')
        if confluence_score is not None:
            try:
                confluence = float(confluence_score)
                if confluence > 0.7:
                    confidence_boost += 0.10
                    reason += " + confluence élevée"
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
                "breakout_type": breakout_type,
                "breakout_level": breakout_level,
                "breakout_distance": breakout_analysis['distance'],
                "range_method": range_info['method'],
                "range_support": range_info['support'],
                "range_resistance": range_info['resistance'],
                "range_width": range_info['width'],
                "range_mid_point": range_info['mid_point'],
                "confirmations": confirmations,
                "confirmed_count": confirmed_count,
                "volume_ratio": values.get('volume_ratio'),
                "momentum_score": values.get('momentum_score'),
                "adx_14": values.get('adx_14'),
                "rsi_14": values.get('rsi_14'),
                "bb_breakout_direction": values.get('bb_breakout_direction'),
                "market_regime": values.get('market_regime'),
                "confluence_score": values.get('confluence_score')
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que les données nécessaires pour range breakout sont présentes."""
        if not super().validate_data():
            return False
            
        # Au minimum, il faut des niveaux ou des Bollinger Bands ou des données OHLC
        required_any = [
            ['nearest_support', 'nearest_resistance'],
            ['bb_upper', 'bb_lower'],
            # Les données OHLC sont vérifiées dans super().validate_data()
        ]
        
        for group in required_any:
            if all(indicator in self.indicators and self.indicators[indicator] is not None 
                   for indicator in group):
                return True
                
        # Si on a les données OHLC, on peut construire le range
        if self.data and all(k in self.data for k in ['high', 'low', 'close']):
            if all(len(self.data[k]) >= 10 for k in ['high', 'low', 'close']):
                return True
                
        logger.warning(f"{self.name}: Aucun indicateur de range disponible")
        return False
