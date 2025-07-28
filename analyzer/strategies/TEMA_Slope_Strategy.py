"""
TEMA_Slope_Strategy - Stratégie basée sur la pente du TEMA (Triple Exponential Moving Average).
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class TEMA_Slope_Strategy(BaseStrategy):
    """
    Stratégie utilisant la pente du TEMA pour détecter les changements de momentum et tendance.
    
    Le TEMA (Triple Exponential Moving Average) est une moyenne mobile très réactive qui réduit le lag :
    - TEMA = 3*EMA1 - 3*EMA2 + EMA3
    - EMA1 = EMA du prix, EMA2 = EMA de EMA1, EMA3 = EMA de EMA2
    - Pente positive = momentum haussier
    - Pente négative = momentum baissier
    
    Signaux générés:
    - BUY: TEMA pente positive forte + prix au-dessus TEMA + confirmations
    - SELL: TEMA pente négative forte + prix en-dessous TEMA + confirmations
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres TEMA Slope
        self.min_slope_threshold = 0.0001  # Pente minimum pour considérer un signal
        self.strong_slope_threshold = 0.001  # Pente forte
        self.very_strong_slope_threshold = 0.005  # Pente très forte
        self.price_tema_alignment_bonus = 0.15  # Bonus si prix aligné avec TEMA
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs TEMA et contexte."""
        return {
            # TEMA principal
            'tema_12': self.indicators.get('tema_12'),
            # Autres moyennes mobiles pour comparaison
            'ema_12': self.indicators.get('ema_12'),
            'ema_26': self.indicators.get('ema_26'),
            'ema_50': self.indicators.get('ema_50'),
            'dema_12': self.indicators.get('dema_12'),  # Double EMA pour comparaison
            'hull_20': self.indicators.get('hull_20'),  # Hull MA (aussi réactive)
            # Tendance et momentum
            'trend_strength': self.indicators.get('trend_strength'),
            'trend_angle': self.indicators.get('trend_angle'),
            'directional_bias': self.indicators.get('directional_bias'),
            'momentum_score': self.indicators.get('momentum_score'),
            # ADX pour force de tendance
            'adx_14': self.indicators.get('adx_14'),
            'plus_di': self.indicators.get('plus_di'),
            'minus_di': self.indicators.get('minus_di'),
            # MACD pour confirmation momentum
            'macd_line': self.indicators.get('macd_line'),
            'macd_signal': self.indicators.get('macd_signal'),
            'macd_histogram': self.indicators.get('macd_histogram'),
            'macd_trend': self.indicators.get('macd_trend'),
            # ROC pour momentum
            'roc_10': self.indicators.get('roc_10'),
            'momentum_10': self.indicators.get('momentum_10'),
            # RSI pour confluence
            'rsi_14': self.indicators.get('rsi_14'),
            'rsi_21': self.indicators.get('rsi_21'),
            # Volume pour confirmation
            'volume_ratio': self.indicators.get('volume_ratio'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            'trade_intensity': self.indicators.get('trade_intensity'),
            # ATR pour volatilité
            'atr_14': self.indicators.get('atr_14'),
            'atr_percentile': self.indicators.get('atr_percentile'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            # VWAP pour contexte
            'vwap_10': self.indicators.get('vwap_10'),
            'anchored_vwap': self.indicators.get('anchored_vwap'),
            # Support/Résistance
            'nearest_support': self.indicators.get('nearest_support'),
            'nearest_resistance': self.indicators.get('nearest_resistance'),
            'support_strength': self.indicators.get('support_strength'),
            'resistance_strength': self.indicators.get('resistance_strength'),
            # Market structure
            'market_regime': self.indicators.get('market_regime'),
            'regime_strength': self.indicators.get('regime_strength'),
            'trend_alignment': self.indicators.get('trend_alignment'),
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
        """Récupère les derniers prix pour calculer la pente TEMA."""
        try:
            if self.data and 'close' in self.data and len(self.data['close']) >= 3:
                return [float(p) for p in self.data['close'][-3:]]
        except (IndexError, ValueError, TypeError):
            pass
        return None
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur la pente du TEMA.
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
        
        # Analyser le TEMA et sa pente
        tema_analysis = self._analyze_tema_slope(values, current_price, recent_prices)
        if tema_analysis is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Données TEMA non disponibles",
                "metadata": {"strategy": self.name}
            }
            
        # Vérifier les conditions de signal
        signal_condition = self._check_tema_signal_conditions(tema_analysis)
        if signal_condition is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": tema_analysis.get('rejection_reason', "Conditions TEMA pente non remplies"),
                "metadata": {"strategy": self.name}
            }
            
        # Créer le signal avec confirmations
        return self._create_tema_slope_signal(values, current_price, tema_analysis, signal_condition)
        
    def _analyze_tema_slope(self, values: Dict[str, Any], current_price: float, recent_prices: list) -> Optional[Dict[str, Any]]:
        """Analyse la pente du TEMA et sa relation avec le prix."""
        tema_12 = values.get('tema_12')
        
        if tema_12 is None:
            return None
            
        try:
            tema_val = float(tema_12)
        except (ValueError, TypeError):
            return None
            
        # Calculer une approximation de la pente TEMA
        # En l'absence de valeurs TEMA historiques, utiliser trend_angle ou approximer
        tema_slope = None
        slope_strength = "unknown"
        
        # Méthode 1: Utiliser trend_angle si disponible
        trend_angle = values.get('trend_angle')
        if trend_angle is not None:
            try:
                angle = float(trend_angle)
                tema_slope = angle  # Approximation
            except (ValueError, TypeError):
                pass
                
        # Méthode 2: Approximer avec la relation prix/TEMA et prix récents
        if tema_slope is None and recent_prices is not None and len(recent_prices) >= 3:
            try:
                # Calculer une pente approximative basée sur les prix récents
                price_change = recent_prices[-1] - recent_prices[0]
                price_slope = price_change / recent_prices[0] if recent_prices[0] != 0 else 0
                
                # Ajuster selon la position par rapport au TEMA
                price_tema_ratio = current_price / tema_val if tema_val != 0 else 1
                tema_slope = price_slope * price_tema_ratio  # Approximation
            except (ValueError, TypeError, ZeroDivisionError):
                tema_slope = 0
                
        if tema_slope is None:
            tema_slope = 0
            
        # Classifier la force de la pente
        abs_slope = abs(tema_slope)
        if abs_slope >= self.very_strong_slope_threshold:
            slope_strength = "very_strong"
        elif abs_slope >= self.strong_slope_threshold:
            slope_strength = "strong"
        elif abs_slope >= self.min_slope_threshold:
            slope_strength = "moderate"
        else:
            slope_strength = "weak"
            
        # Direction de la pente
        slope_direction = "bullish" if tema_slope > 0 else "bearish" if tema_slope < 0 else "neutral"
        
        # Relation prix/TEMA
        price_above_tema = current_price > tema_val
        price_tema_distance = abs(current_price - tema_val) / tema_val if tema_val != 0 else 0
        
        # Alignement prix/TEMA/pente
        alignment = None
        if slope_direction == "bullish" and price_above_tema:
            alignment = "bullish_aligned"
        elif slope_direction == "bearish" and not price_above_tema:
            alignment = "bearish_aligned"
        elif slope_direction == "bullish" and not price_above_tema:
            alignment = "bullish_divergent"
        elif slope_direction == "bearish" and price_above_tema:
            alignment = "bearish_divergent"
        else:
            alignment = "neutral"
            
        # Raisons de rejet potentielles
        rejection_reasons = []
        if slope_strength == "weak":
            rejection_reasons.append(f"Pente TEMA trop faible ({abs_slope:.6f})")
        if alignment in ["bullish_divergent", "bearish_divergent"]:
            rejection_reasons.append(f"Prix et pente TEMA divergents ({alignment})")
            
        return {
            'tema_value': tema_val,
            'tema_slope': tema_slope,
            'slope_strength': slope_strength,
            'slope_direction': slope_direction,
            'price_above_tema': price_above_tema,
            'price_tema_distance': price_tema_distance,
            'alignment': alignment,
            'rejection_reason': "; ".join(rejection_reasons) if rejection_reasons else None
        }
        
    def _check_tema_signal_conditions(self, tema_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Vérifie si les conditions de signal TEMA sont remplies."""
        slope_strength = tema_analysis['slope_strength']
        slope_direction = tema_analysis['slope_direction']
        alignment = tema_analysis['alignment']
        
        # Rejeter si pente trop faible
        if slope_strength == "weak":
            return None
            
        # Rejeter si prix/pente divergents
        if alignment in ["bullish_divergent", "bearish_divergent"]:
            return None
            
        # Rejeter si pente neutre
        if slope_direction == "neutral":
            return None
            
        # Déterminer le type de signal
        if alignment == "bullish_aligned":
            signal_side = "BUY"
        elif alignment == "bearish_aligned":
            signal_side = "SELL"
        else:
            return None
            
        return {
            'signal_side': signal_side,
            'slope_strength': slope_strength,
            'slope_direction': slope_direction,
            'alignment': alignment
        }
        
    def _create_tema_slope_signal(self, values: Dict[str, Any], current_price: float,
                                 tema_analysis: Dict[str, Any], signal_condition: Dict[str, Any]) -> Dict[str, Any]:
        """Crée le signal TEMA slope avec confirmations."""
        signal_side = signal_condition['signal_side']
        slope_strength = signal_condition['slope_strength']
        slope_direction = signal_condition['slope_direction']
        alignment = signal_condition['alignment']
        
        base_confidence = 0.55  # Base modérée pour pente
        confidence_boost = 0.0
        
        tema_val = tema_analysis['tema_value']
        tema_slope = tema_analysis['tema_slope']
        price_tema_distance = tema_analysis['price_tema_distance']
        
        # Construction de la raison
        reason = f"TEMA pente {slope_direction}: {tema_slope:.6f} ({slope_strength})"
        
        # Bonus selon la force de la pente
        if slope_strength == "very_strong":
            confidence_boost += 0.25
            reason += " - momentum très fort"
        elif slope_strength == "strong":
            confidence_boost += 0.20
            reason += " - momentum fort"
        else:  # moderate
            confidence_boost += 0.15
            reason += " - momentum modéré"
            
        # Bonus selon l'alignement prix/TEMA
        if alignment in ["bullish_aligned", "bearish_aligned"]:
            confidence_boost += self.price_tema_alignment_bonus
            distance_text = f"distance {price_tema_distance:.3f}"
            reason += f" + prix/TEMA alignés ({distance_text})"
            
        # Confirmation avec autres moyennes mobiles
        ema_12 = values.get('ema_12')
        ema_26 = values.get('ema_26')
        if ema_12 is not None and ema_26 is not None:
            try:
                ema12_val = float(ema_12)
                ema26_val = float(ema_26)
                ema_cross_aligned = (signal_side == "BUY" and ema12_val > ema26_val) or \
                                   (signal_side == "SELL" and ema12_val < ema26_val)
                
                if ema_cross_aligned:
                    confidence_boost += 0.12
                    reason += " + EMA confirme"
                else:
                    confidence_boost -= 0.05
                    reason += " mais EMA diverge"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec Hull MA (autre MA réactive)
        hull_20 = values.get('hull_20')
        if hull_20 is not None:
            try:
                hull_val = float(hull_20)
                hull_aligned = (signal_side == "BUY" and current_price > hull_val) or \
                              (signal_side == "SELL" and current_price < hull_val)
                
                if hull_aligned:
                    confidence_boost += 0.10
                    reason += " + Hull MA confirme"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec DEMA (Double EMA)
        dema_12 = values.get('dema_12')
        if dema_12 is not None:
            try:
                dema_val = float(dema_12)
                dema_aligned = (signal_side == "BUY" and current_price > dema_val) or \
                              (signal_side == "SELL" and current_price < dema_val)
                
                if dema_aligned:
                    confidence_boost += 0.08
                    reason += " + DEMA confirme"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec momentum indicators
        momentum_score = values.get('momentum_score')
        if momentum_score is not None:
            try:
                momentum = float(momentum_score)
                momentum_aligned = (signal_side == "BUY" and momentum > 0.2) or \
                                  (signal_side == "SELL" and momentum < -0.2)
                
                if momentum_aligned:
                    confidence_boost += 0.15
                    reason += " + momentum confirmé"
                elif abs(momentum) > 0.1:
                    confidence_boost += 0.08
                    reason += " + momentum aligné"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec ROC
        roc_10 = values.get('roc_10')
        if roc_10 is not None:
            try:
                roc = float(roc_10)
                roc_aligned = (signal_side == "BUY" and roc > 1.0) or \
                             (signal_side == "SELL" and roc < -1.0)
                
                if roc_aligned:
                    confidence_boost += 0.12
                    reason += f" + ROC confirmé ({roc:.2f}%)"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec MACD
        macd_line = values.get('macd_line')
        macd_signal = values.get('macd_signal')
        if macd_line is not None and macd_signal is not None:
            try:
                macd_val = float(macd_line)
                macd_sig = float(macd_signal)
                macd_aligned = (signal_side == "BUY" and macd_val > macd_sig) or \
                              (signal_side == "SELL" and macd_val < macd_sig)
                
                if macd_aligned:
                    confidence_boost += 0.10
                    reason += " + MACD confirme"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec directional bias
        directional_bias = values.get('directional_bias')
        if directional_bias is not None:
            if (signal_side == "BUY" and directional_bias == "bullish") or \
               (signal_side == "SELL" and directional_bias == "bearish"):
                confidence_boost += 0.12
                reason += f" + bias {directional_bias}"
                
        # Confirmation avec trend strength
        trend_strength = values.get('trend_strength')
        if trend_strength is not None:
            try:
                strength = float(trend_strength)
                if strength > 0.6:
                    confidence_boost += 0.12
                    reason += f" + tendance forte ({strength:.2f})"
                elif strength > 0.4:
                    confidence_boost += 0.08
                    reason += f" + tendance modérée ({strength:.2f})"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec ADX
        adx_14 = values.get('adx_14')
        if adx_14 is not None:
            try:
                adx = float(adx_14)
                if adx > 25:  # Tendance forte
                    confidence_boost += 0.12
                    reason += " + ADX fort"
                elif adx > 20:
                    confidence_boost += 0.08
                    reason += " + ADX modéré"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec RSI (éviter extrêmes)
        rsi_14 = values.get('rsi_14')
        if rsi_14 is not None:
            try:
                rsi = float(rsi_14)
                if signal_side == "BUY" and 40 <= rsi <= 70:
                    confidence_boost += 0.08
                    reason += " + RSI favorable"
                elif signal_side == "SELL" and 30 <= rsi <= 60:
                    confidence_boost += 0.08
                    reason += " + RSI favorable"
                elif (signal_side == "BUY" and rsi >= 80) or (signal_side == "SELL" and rsi <= 20):
                    confidence_boost -= 0.10
                    reason += " mais RSI extrême"
            except (ValueError, TypeError):
                pass
                
        # Volume confirmation
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= 1.3:
                    confidence_boost += 0.12
                    reason += f" + volume élevé ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.1:
                    confidence_boost += 0.08
                    reason += f" + volume modéré ({vol_ratio:.1f}x)"
            except (ValueError, TypeError):
                pass
                
        # VWAP context
        vwap_10 = values.get('vwap_10')
        if vwap_10 is not None:
            try:
                vwap = float(vwap_10)
                vwap_aligned = (signal_side == "BUY" and current_price > vwap) or \
                              (signal_side == "SELL" and current_price < vwap)
                
                if vwap_aligned:
                    confidence_boost += 0.08
                    reason += " + VWAP aligné"
            except (ValueError, TypeError):
                pass
                
        # Support/Resistance context
        if signal_side == "BUY":
            nearest_support = values.get('nearest_support')
            if nearest_support is not None:
                try:
                    support = float(nearest_support)
                    distance_to_support = abs(current_price - support) / current_price
                    if distance_to_support <= 0.02:
                        confidence_boost += 0.10
                        reason += " + près support"
                except (ValueError, TypeError):
                    pass
        else:  # SELL
            nearest_resistance = values.get('nearest_resistance')
            if nearest_resistance is not None:
                try:
                    resistance = float(nearest_resistance)
                    distance_to_resistance = abs(current_price - resistance) / current_price
                    if distance_to_resistance <= 0.02:
                        confidence_boost += 0.10
                        reason += " + près résistance"
                except (ValueError, TypeError):
                    pass
                    
        # Market regime
        market_regime = values.get('market_regime')
        if market_regime == "trending":
            confidence_boost += 0.10
            reason += " (marché trending)"
        elif market_regime == "ranging":
            confidence_boost -= 0.05  # TEMA slope moins fiable en ranging
            reason += " (marché ranging)"
            
        # Volatility context
        volatility_regime = values.get('volatility_regime')
        if volatility_regime == "normal":
            confidence_boost += 0.05
            reason += " + volatilité normale"
        elif volatility_regime == "high":
            if slope_strength in ["strong", "very_strong"]:
                confidence_boost += 0.05  # Forte pente en haute volatilité = signal plus fiable
                reason += " + volatilité élevée favorable"
            else:
                confidence_boost -= 0.05
                reason += " mais volatilité élevée"
                
        # Confluence score
        confluence_score = values.get('confluence_score')
        if confluence_score is not None:
            try:
                confluence = float(confluence_score)
                if confluence > 0.7:
                    confidence_boost += 0.10
                    reason += " + confluence élevée"
                elif confluence > 0.5:
                    confidence_boost += 0.05
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
                "tema_value": tema_val,
                "tema_slope": tema_slope,
                "slope_strength": slope_strength,
                "slope_direction": slope_direction,
                "alignment": alignment,
                "price_tema_distance": price_tema_distance,
                "ema_12": values.get('ema_12'),
                "ema_26": values.get('ema_26'),
                "hull_20": values.get('hull_20'),
                "dema_12": values.get('dema_12'),
                "momentum_score": values.get('momentum_score'),
                "roc_10": values.get('roc_10'),
                "macd_line": values.get('macd_line'),
                "macd_signal": values.get('macd_signal'),
                "directional_bias": values.get('directional_bias'),
                "trend_strength": values.get('trend_strength'),
                "adx_14": values.get('adx_14'),
                "rsi_14": values.get('rsi_14'),
                "volume_ratio": values.get('volume_ratio'),
                "vwap_10": values.get('vwap_10'),
                "market_regime": values.get('market_regime'),
                "volatility_regime": values.get('volatility_regime'),
                "confluence_score": values.get('confluence_score')
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que les données TEMA nécessaires sont présentes."""
        if not super().validate_data():
            return False
            
        # Au minimum, il faut TEMA_12
        if 'tema_12' not in self.indicators:
            logger.warning(f"{self.name}: Indicateur manquant: tema_12")
            return False
        if self.indicators['tema_12'] is None:
            logger.warning(f"{self.name}: Indicateur null: tema_12")
            return False
                
        return True
