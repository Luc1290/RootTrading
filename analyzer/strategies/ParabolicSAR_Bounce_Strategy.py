"""
ParabolicSAR_Bounce_Strategy - Stratégie basée sur les rebonds du prix sur le Parabolic SAR.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class ParabolicSAR_Bounce_Strategy(BaseStrategy):
    """
    Stratégie utilisant les rebonds du prix sur le Parabolic SAR pour détecter des retournements.
    
    Le Parabolic SAR (Stop and Reverse) suit la tendance et s'inverse quand le prix croise le SAR :
    - SAR sous le prix = tendance haussière
    - SAR au-dessus du prix = tendance baissière
    - Rebond = prix s'approche du SAR puis repart dans la direction de la tendance
    
    Signaux générés:
    - BUY: Prix rebondit sur SAR en tendance haussière (SAR < prix) + confirmations
    - SELL: Prix rebondit sur SAR en tendance baissière (SAR > prix) + confirmations
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres Parabolic SAR
        self.min_sar_distance = 0.005  # Distance minimum prix/SAR pour éviter faux rebonds
        self.max_sar_distance = 0.05   # Distance maximum pour considérer un rebond
        self.trend_confirmation_bonus = 0.15  # Bonus si aligné avec tendance
        self.volume_confirmation_threshold = 1.1  # Seuil volume pour confirmation rebond
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs SAR et contexte."""
        return {
            # Parabolic SAR (à implémenter si disponible)
            # Note: Le SAR n'est pas listé dans les 108 indicateurs, utilisons d'autres
            # On utilisera les niveaux de support/résistance comme proxy pour rebonds
            'support_levels': self.indicators.get('support_levels'),
            'resistance_levels': self.indicators.get('resistance_levels'),
            'nearest_support': self.indicators.get('nearest_support'),
            'nearest_resistance': self.indicators.get('nearest_resistance'),
            'support_strength': self.indicators.get('support_strength'),
            'resistance_strength': self.indicators.get('resistance_strength'),
            'break_probability': self.indicators.get('break_probability'),
            # Tendance pour déterminer la direction attendue du rebond
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'trend_alignment': self.indicators.get('trend_alignment'),
            'trend_angle': self.indicators.get('trend_angle'),
            # EMA pour contexte de tendance
            'ema_12': self.indicators.get('ema_12'),
            'ema_26': self.indicators.get('ema_26'),
            'ema_50': self.indicators.get('ema_50'),
            # ADX pour force de tendance
            'adx_14': self.indicators.get('adx_14'),
            'plus_di': self.indicators.get('plus_di'),
            'minus_di': self.indicators.get('minus_di'),
            # Volume pour confirmation rebond
            'volume_ratio': self.indicators.get('volume_ratio'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            'trade_intensity': self.indicators.get('trade_intensity'),
            # Oscillateurs pour timing
            'rsi_14': self.indicators.get('rsi_14'),
            'stoch_k': self.indicators.get('stoch_k'),
            'stoch_d': self.indicators.get('stoch_d'),
            'williams_r': self.indicators.get('williams_r'),
            # ATR pour volatilité et stops
            'atr_14': self.indicators.get('atr_14'),
            'atr_stop_long': self.indicators.get('atr_stop_long'),
            'atr_stop_short': self.indicators.get('atr_stop_short'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            # VWAP institutional levels
            'vwap_10': self.indicators.get('vwap_10'),
            'anchored_vwap': self.indicators.get('anchored_vwap'),
            # Market structure
            'market_regime': self.indicators.get('market_regime'),
            'regime_strength': self.indicators.get('regime_strength'),
            'momentum_score': self.indicators.get('momentum_score'),
            # Pattern et confluence
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
        
    def _get_recent_prices(self) -> Optional[list]:
        """Récupère les derniers prix pour analyser le mouvement."""
        try:
            if self.data and 'close' in self.data and len(self.data['close']) >= 3:
                return [float(p) for p in self.data['close'][-3:]]
        except (IndexError, ValueError, TypeError):
            pass
        return None
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur les rebonds sur niveaux SAR/Support/Résistance.
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
        recent_prices = self._get_recent_prices()
        
        if current_price is None or recent_prices is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Prix non disponibles",
                "metadata": {"strategy": self.name}
            }
            
        # Déterminer la tendance principale
        trend_direction = self._get_trend_direction(values, current_price)
        if trend_direction is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Tendance non claire",
                "metadata": {"strategy": self.name}
            }
            
        # Analyser les rebonds selon la tendance
        if trend_direction == "bullish":
            return self._analyze_bullish_bounce(values, current_price, recent_prices)
        else:
            return self._analyze_bearish_bounce(values, current_price, recent_prices)
            
    def _get_trend_direction(self, values: Dict[str, Any], current_price: float) -> Optional[str]:
        """Détermine la direction de la tendance principale."""
        try:
            # Vérifier directional_bias d'abord
            directional_bias = values.get('directional_bias')
            if directional_bias in ["bullish", "bearish"]:
                return directional_bias
                
            # Utiliser les EMA comme fallback
            ema_12 = values.get('ema_12')
            ema_26 = values.get('ema_26')
            ema_50 = values.get('ema_50')
            
            if all(x is not None for x in [ema_12, ema_26, ema_50]):
                ema12_val = float(ema_12)
                ema26_val = float(ema_26)
                ema50_val = float(ema_50)
                
                # Tendance haussière: prix > EMA12 > EMA26 > EMA50
                if current_price > ema12_val > ema26_val > ema50_val:
                    return "bullish"
                # Tendance baissière: prix < EMA12 < EMA26 < EMA50
                elif current_price < ema12_val < ema26_val < ema50_val:
                    return "bearish"
                    
            # Utiliser ADX/DI comme dernier recours
            plus_di = values.get('plus_di')
            minus_di = values.get('minus_di')
            if plus_di is not None and minus_di is not None:
                plus_val = float(plus_di)
                minus_val = float(minus_di)
                if plus_val > minus_val:
                    return "bullish"
                elif minus_val > plus_val:
                    return "bearish"
                    
        except (ValueError, TypeError):
            pass
            
        return None
        
    def _analyze_bullish_bounce(self, values: Dict[str, Any], current_price: float, recent_prices: list) -> Dict[str, Any]:
        """Analyse un rebond haussier sur support."""
        signal_side = "BUY"
        reason = ""
        base_confidence = 0.5
        confidence_boost = 0.0
        bounce_level = None
        
        # Chercher le niveau de rebond (support ou EMA)
        nearest_support = values.get('nearest_support')
        ema_50 = values.get('ema_50')
        
        if nearest_support is not None:
            try:
                support_val = float(nearest_support)
                distance_to_support = abs(current_price - support_val) / current_price
                
                if distance_to_support <= self.max_sar_distance:
                    bounce_level = support_val
                    level_type = "support"
                    reason = f"Rebond sur support {support_val:.4f}"
                    confidence_boost += 0.15
                    
                    # Vérifier que le prix a effectivement rebondi
                    if self._detect_bounce_pattern(recent_prices, support_val, "bullish"):
                        confidence_boost += 0.20
                        reason += " - rebond confirmé"
                    else:
                        confidence_boost += 0.05
                        reason += " - proche support"
            except (ValueError, TypeError):
                pass
                
        # Si pas de support clair, utiliser EMA50 comme niveau SAR
        if bounce_level is None and ema_50 is not None:
            try:
                ema50_val = float(ema_50)
                if current_price > ema50_val:  # Prix au-dessus de EMA50 (tendance haussière)
                    distance_to_ema = abs(current_price - ema50_val) / current_price
                    
                    if distance_to_ema <= self.max_sar_distance:
                        bounce_level = ema50_val
                        level_type = "ema50"
                        reason = f"Rebond sur EMA50 {ema50_val:.4f}"
                        confidence_boost += 0.12
                        
                        if self._detect_bounce_pattern(recent_prices, ema50_val, "bullish"):
                            confidence_boost += 0.15
                            reason += " - rebond confirmé"
            except (ValueError, TypeError):
                pass
                
        if bounce_level is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Aucun niveau de rebond détecté",
                "metadata": {"strategy": self.name}
            }
            
        # Confirmations additionnelles
        confidence_boost += self._add_confirmations(values, signal_side, current_price)
        
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
                "bounce_level": bounce_level,
                "level_type": level_type,
                "distance_to_level": abs(current_price - bounce_level) / current_price,
                "trend_direction": "bullish",
                "recent_prices": recent_prices,
                **self._get_metadata_values(values)
            }
        }
        
    def _analyze_bearish_bounce(self, values: Dict[str, Any], current_price: float, recent_prices: list) -> Dict[str, Any]:
        """Analyse un rebond baissier sur résistance."""
        signal_side = "SELL"
        reason = ""
        base_confidence = 0.5
        confidence_boost = 0.0
        bounce_level = None
        
        # Chercher le niveau de rebond (résistance ou EMA)
        nearest_resistance = values.get('nearest_resistance')
        ema_50 = values.get('ema_50')
        
        if nearest_resistance is not None:
            try:
                resistance_val = float(nearest_resistance)
                distance_to_resistance = abs(current_price - resistance_val) / current_price
                
                if distance_to_resistance <= self.max_sar_distance:
                    bounce_level = resistance_val
                    level_type = "resistance"
                    reason = f"Rebond sur résistance {resistance_val:.4f}"
                    confidence_boost += 0.15
                    
                    if self._detect_bounce_pattern(recent_prices, resistance_val, "bearish"):
                        confidence_boost += 0.20
                        reason += " - rebond confirmé"
                    else:
                        confidence_boost += 0.05
                        reason += " - proche résistance"
            except (ValueError, TypeError):
                pass
                
        # Si pas de résistance claire, utiliser EMA50 comme niveau SAR
        if bounce_level is None and ema_50 is not None:
            try:
                ema50_val = float(ema_50)
                if current_price < ema50_val:  # Prix en-dessous de EMA50 (tendance baissière)
                    distance_to_ema = abs(current_price - ema50_val) / current_price
                    
                    if distance_to_ema <= self.max_sar_distance:
                        bounce_level = ema50_val
                        level_type = "ema50"
                        reason = f"Rebond sur EMA50 {ema50_val:.4f}"
                        confidence_boost += 0.12
                        
                        if self._detect_bounce_pattern(recent_prices, ema50_val, "bearish"):
                            confidence_boost += 0.15
                            reason += " - rebond confirmé"
            except (ValueError, TypeError):
                pass
                
        if bounce_level is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Aucun niveau de rebond détecté",
                "metadata": {"strategy": self.name}
            }
            
        # Confirmations additionnelles  
        confidence_boost += self._add_confirmations(values, signal_side, current_price)
        
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
                "bounce_level": bounce_level,
                "level_type": level_type,
                "distance_to_level": abs(current_price - bounce_level) / current_price,
                "trend_direction": "bearish",
                "recent_prices": recent_prices,
                **self._get_metadata_values(values)
            }
        }
        
    def _detect_bounce_pattern(self, recent_prices: list, level: float, direction: str) -> bool:
        """Détecte si les prix récents montrent un pattern de rebond."""
        if len(recent_prices) < 3:
            return False
            
        try:
            price_1, price_2, price_3 = recent_prices[-3:]
            
            if direction == "bullish":
                # Pour rebond haussier: prix s'approche du niveau puis remonte
                # price_1 > level, price_2 proche de level, price_3 > price_2
                return (price_1 > level and 
                        abs(price_2 - level) / level < 0.02 and  # Prix 2 proche du niveau
                        price_3 > price_2)  # Prix 3 remonte
            else:  # bearish
                # Pour rebond baissier: prix s'approche du niveau puis redescend
                # price_1 < level, price_2 proche de level, price_3 < price_2
                return (price_1 < level and
                        abs(price_2 - level) / level < 0.02 and  # Prix 2 proche du niveau
                        price_3 < price_2)  # Prix 3 redescend
        except (ValueError, TypeError, ZeroDivisionError):
            return False
            
    def _add_confirmations(self, values: Dict[str, Any], signal_side: str, current_price: float) -> float:
        """Ajoute les confirmations pour renforcer le signal."""
        boost = 0.0
        
        # Confirmation avec trend_strength
        trend_strength = values.get('trend_strength')
        if trend_strength is not None:
            try:
                strength = float(trend_strength)
                if strength > 0.6:
                    boost += 0.15
                elif strength > 0.4:
                    boost += 0.10
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec ADX (force de tendance)
        adx_14 = values.get('adx_14')
        if adx_14 is not None:
            try:
                adx = float(adx_14)
                if adx > 25:  # Tendance forte
                    boost += 0.12
                elif adx > 20:  # Tendance modérée
                    boost += 0.08
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec volume
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= self.volume_confirmation_threshold:
                    boost += 0.15
                elif vol_ratio >= 1.05:
                    boost += 0.08
                else:
                    boost -= 0.05  # Volume faible = moins fiable
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec RSI (éviter zones extrêmes)
        rsi_14 = values.get('rsi_14')
        if rsi_14 is not None:
            try:
                rsi = float(rsi_14)
                if signal_side == "BUY" and 30 <= rsi <= 65:
                    boost += 0.10
                elif signal_side == "SELL" and 35 <= rsi <= 70:
                    boost += 0.10
                elif (signal_side == "BUY" and rsi >= 80) or (signal_side == "SELL" and rsi <= 20):
                    boost -= 0.15  # Zones extrêmes moins favorables pour rebonds
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec Williams %R
        williams_r = values.get('williams_r')
        if williams_r is not None:
            try:
                wr = float(williams_r)
                if signal_side == "BUY" and -80 <= wr <= -20:
                    boost += 0.08
                elif signal_side == "SELL" and -80 <= wr <= -20:
                    boost += 0.08
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec VWAP
        vwap_10 = values.get('vwap_10')
        if vwap_10 is not None:
            try:
                vwap = float(vwap_10)
                if signal_side == "BUY" and current_price > vwap:
                    boost += 0.08
                elif signal_side == "SELL" and current_price < vwap:
                    boost += 0.08
            except (ValueError, TypeError):
                pass
                
        # Market regime
        market_regime = values.get('market_regime')
        if market_regime == "trending":
            boost += 0.10
        elif market_regime == "ranging":
            boost += 0.05  # Rebonds plus fréquents en ranging
            
        # Volatilité (rebonds plus fiables en volatilité modérée)
        volatility_regime = values.get('volatility_regime')
        if volatility_regime == "normal":
            boost += 0.05
        elif volatility_regime == "high":
            boost -= 0.05  # Volatilité élevée = rebonds moins fiables
            
        # Pattern detected
        pattern_detected = values.get('pattern_detected')
        pattern_confidence = values.get('pattern_confidence')
        if pattern_detected and pattern_confidence is not None:
            try:
                confidence = float(pattern_confidence)
                if confidence > 0.7:
                    boost += 0.08
            except (ValueError, TypeError):
                pass
                
        # Confluence score
        confluence_score = values.get('confluence_score')
        if confluence_score is not None:
            try:
                confluence = float(confluence_score)
                if confluence > 0.6:
                    boost += 0.12
            except (ValueError, TypeError):
                pass
                
        return boost
        
    def _get_metadata_values(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Récupère les valeurs importantes pour les métadonnées."""
        metadata = {}
        important_keys = [
            'trend_strength', 'directional_bias', 'adx_14', 'volume_ratio',
            'rsi_14', 'williams_r', 'vwap_10', 'market_regime', 'volatility_regime',
            'confluence_score', 'pattern_detected', 'pattern_confidence'
        ]
        
        for key in important_keys:
            if values.get(key) is not None:
                metadata[key] = values[key]
                
        return metadata
        
    def validate_data(self) -> bool:
        """Valide que les données nécessaires pour les rebonds sont présentes."""
        if not super().validate_data():
            return False
            
        # Au minimum, il faut des niveaux de support/résistance ou des EMA
        required_any = [
            ['nearest_support', 'nearest_resistance'],
            ['ema_50'],
            ['directional_bias']
        ]
        
        for group in required_any:
            if any(indicator in self.indicators and self.indicators[indicator] is not None 
                   for indicator in group):
                return True
                
        logger.warning(f"{self.name}: Aucun indicateur de niveau/tendance disponible")
        return False
