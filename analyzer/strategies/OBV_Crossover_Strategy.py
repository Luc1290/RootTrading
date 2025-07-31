"""
OBV_Crossover_Strategy - Stratégie basée sur les croisements OBV avec sa moyenne mobile.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class OBV_Crossover_Strategy(BaseStrategy):
    """
    Stratégie utilisant les croisements de l'On-Balance Volume (OBV) avec sa moyenne mobile.
    
    L'OBV accumule le volume selon la direction du prix :
    - Volume ajouté si clôture > clôture précédente
    - Volume soustrait si clôture < clôture précédente
    - Volume neutre si clôture = clôture précédente
    
    Signaux générés:
    - BUY: OBV croise au-dessus de sa MA + confirmations haussières
    - SELL: OBV croise en-dessous de sa MA + confirmations baissières
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres OBV
        self.min_obv_ma_distance = 0.001  # Distance minimum OBV/MA pour éviter bruit
        self.volume_confirmation_threshold = 1.2  # Seuil volume ratio pour confirmation
        self.trend_alignment_bonus = 15  # Bonus si aligné avec tendance prix
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs OBV et volume."""
        return {
            # OBV et sa moyenne mobile
            'obv': self.indicators.get('obv'),
            'obv_ma_10': self.indicators.get('obv_ma_10'),
            'obv_oscillator': self.indicators.get('obv_oscillator'),
            'ad_line': self.indicators.get('ad_line'),  # Accumulation/Distribution Line
            # Volume pour confirmation
            'volume_ratio': self.indicators.get('volume_ratio'),
            'avg_volume_20': self.indicators.get('avg_volume_20'),
            'quote_volume_ratio': self.indicators.get('quote_volume_ratio'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            'trade_intensity': self.indicators.get('trade_intensity'),
            'relative_volume': self.indicators.get('relative_volume'),
            # Contexte prix pour confirmation tendance
            'ema_12': self.indicators.get('ema_12'),
            'ema_26': self.indicators.get('ema_26'),
            'ema_50': self.indicators.get('ema_50'),
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            # VWAP pour contexte institutional
            'vwap_10': self.indicators.get('vwap_10'),
            'anchored_vwap': self.indicators.get('anchored_vwap'),
            # Momentum pour confluence
            'rsi_14': self.indicators.get('rsi_14'),
            'momentum_score': self.indicators.get('momentum_score'),
            # Structure de marché
            'market_regime': self.indicators.get('market_regime'),
            'support_levels': self.indicators.get('support_levels'),
            'resistance_levels': self.indicators.get('resistance_levels'),
            # Confluence
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
        Génère un signal basé sur les croisements OBV/MA.
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
        
        # Vérification des indicateurs OBV essentiels
        try:
            obv = float(values['obv']) if values['obv'] is not None else None
            obv_ma = float(values['obv_ma_10']) if values['obv_ma_10'] is not None else None
        except (ValueError, TypeError) as e:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Erreur conversion OBV: {e}",
                "metadata": {"strategy": self.name}
            }
            
        if obv is None or obv_ma is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "OBV ou OBV MA non disponibles",
                "metadata": {"strategy": self.name}
            }
            
        # Analyse du croisement OBV/MA
        obv_above_ma = obv > obv_ma
        obv_distance = abs(obv - obv_ma) / abs(obv_ma) if obv_ma != 0 else 0
        
        # Vérification que les lignes ne sont pas trop proches (éviter faux signaux)
        if obv_distance < self.min_obv_ma_distance:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"OBV trop proche MA ({obv_distance:.4f}) - pas de signal clair",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "obv": obv,
                    "obv_ma": obv_ma,
                    "distance": obv_distance
                }
            }
            
        signal_side = None
        reason = ""
        base_confidence = 0.5
        confidence_boost = 0.0
        cross_type = None
        
        # Logique de croisement OBV
        if obv_above_ma:
            signal_side = "BUY"
            cross_type = "bullish_cross"
            reason = f"OBV ({obv:.0f}) > MA ({obv_ma:.0f})"
            confidence_boost += 0.15
        else:
            signal_side = "SELL"
            cross_type = "bearish_cross"
            reason = f"OBV ({obv:.0f}) < MA ({obv_ma:.0f})"
            confidence_boost += 0.15
            
        # Bonus selon la force de la séparation
        if obv_distance >= 0.05:
            confidence_boost += 0.15
            reason += f" - séparation forte ({obv_distance:.3f})"
        elif obv_distance >= 0.02:
            confidence_boost += 0.10
            reason += f" - séparation modérée ({obv_distance:.3f})"
        else:
            confidence_boost += 0.05
            reason += f" - séparation faible ({obv_distance:.3f})"
            
        # Confirmation avec OBV Oscillator
        obv_oscillator = values.get('obv_oscillator')
        if obv_oscillator is not None:
            try:
                obv_osc = float(obv_oscillator)
                if signal_side == "BUY" and obv_osc > 0:
                    confidence_boost += 0.15
                    reason += f" + OBV osc positif ({obv_osc:.3f})"
                elif signal_side == "SELL" and obv_osc < 0:
                    confidence_boost += 0.15
                    reason += f" + OBV osc négatif ({obv_osc:.3f})"
                else:
                    confidence_boost -= 0.05
                    reason += f" mais OBV osc défavorable ({obv_osc:.3f})"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec Volume Ratio (force du mouvement)
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= self.volume_confirmation_threshold:
                    confidence_boost += 0.15
                    reason += f" + volume élevé ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.1:
                    confidence_boost += 0.10
                    reason += f" + volume modéré ({vol_ratio:.1f}x)"
                else:
                    confidence_boost -= 0.05
                    reason += f" mais volume faible ({vol_ratio:.1f}x)"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec A/D Line (Accumulation/Distribution)
        ad_line = values.get('ad_line')
        if ad_line is not None and obv is not None:
            try:
                ad_val = float(ad_line)
                # Vérifier si OBV et A/D Line s'alignent
                obv_direction = 1 if obv > 0 else -1
                ad_direction = 1 if ad_val > 0 else -1
                
                if (signal_side == "BUY" and obv_direction == ad_direction == 1) or \
                   (signal_side == "SELL" and obv_direction == ad_direction == -1):
                    confidence_boost += 0.12
                    reason += " + A/D Line confirme"
                elif obv_direction != ad_direction:
                    confidence_boost -= 0.08
                    reason += " mais A/D Line diverge"
            except (ValueError, TypeError):
                pass
                
        # Alignement avec tendance prix pour confirmation
        ema_12 = values.get('ema_12')
        ema_26 = values.get('ema_26')
        if ema_12 is not None and ema_26 is not None and current_price is not None:
            try:
                ema12_val = float(ema_12)
                ema26_val = float(ema_26)
                price_trend_bullish = current_price > ema12_val > ema26_val
                price_trend_bearish = current_price < ema12_val < ema26_val
                
                if (signal_side == "BUY" and price_trend_bullish) or \
                   (signal_side == "SELL" and price_trend_bearish):
                    confidence_boost += self.trend_alignment_bonus
                    reason += " + tendance prix alignée"
                elif (signal_side == "BUY" and price_trend_bearish) or \
                     (signal_side == "SELL" and price_trend_bullish):
                    confidence_boost -= 0.10
                    reason += " mais tendance prix diverge"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec EMA 50 pour filtre de tendance
        ema_50 = values.get('ema_50')
        if ema_50 is not None and current_price is not None:
            try:
                ema50_val = float(ema_50)
                if signal_side == "BUY" and current_price > ema50_val:
                    confidence_boost += 0.08
                    reason += " + prix > EMA50"
                elif signal_side == "SELL" and current_price < ema50_val:
                    confidence_boost += 0.08
                    reason += " + prix < EMA50"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec VWAP (institutional level)
        vwap_10 = values.get('vwap_10')
        if vwap_10 is not None and current_price is not None:
            try:
                vwap_val = float(vwap_10)
                if signal_side == "BUY" and current_price > vwap_val:
                    confidence_boost += 0.10
                    reason += " + prix > VWAP"
                elif signal_side == "SELL" and current_price < vwap_val:
                    confidence_boost += 0.10
                    reason += " + prix < VWAP"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec trend_strength
        trend_strength = values.get('trend_strength')
        if trend_strength is not None:
            try:
                trend_str = float(trend_strength)
                if trend_str > 0.6:
                    confidence_boost += 0.12
                    reason += f" + tendance forte ({trend_str:.2f})"
                elif trend_str > 0.4:
                    confidence_boost += 0.08
                    reason += f" + tendance modérée ({trend_str:.2f})"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec directional_bias
        directional_bias = values.get('directional_bias')
        if directional_bias:
            if (signal_side == "BUY" and directional_bias == "bullish") or \
               (signal_side == "SELL" and directional_bias == "bearish"):
                confidence_boost += 0.10
                reason += f" + bias {directional_bias}"
                
        # Confirmation avec qualité du volume
        volume_quality_score = values.get('volume_quality_score')
        if volume_quality_score is not None:
            try:
                vol_quality = float(volume_quality_score)
                if vol_quality > 0.7:
                    confidence_boost += 0.10
                    reason += " + volume de qualité"
                elif vol_quality < 0.3:
                    confidence_boost -= 0.05
                    reason += " mais volume de faible qualité"
            except (ValueError, TypeError):
                pass
                
        # Trade intensity pour confirmation
        trade_intensity = values.get('trade_intensity')
        if trade_intensity is not None:
            try:
                intensity = float(trade_intensity)
                if intensity > 1.5:
                    confidence_boost += 0.08
                    reason += " + intensité élevée"
                elif intensity < 0.8:
                    confidence_boost -= 0.05
                    reason += " mais intensité faible"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec RSI (éviter zones extrêmes)
        rsi_14 = values.get('rsi_14')
        if rsi_14 is not None:
            try:
                rsi = float(rsi_14)
                if signal_side == "BUY" and 30 <= rsi <= 70:
                    confidence_boost += 0.08
                    reason += " + RSI favorable"
                elif signal_side == "SELL" and 30 <= rsi <= 70:
                    confidence_boost += 0.08
                    reason += " + RSI favorable"
                elif signal_side == "BUY" and rsi >= 80:
                    confidence_boost -= 0.10
                    reason += " mais RSI surachat"
                elif signal_side == "SELL" and rsi <= 20:
                    confidence_boost -= 0.10
                    reason += " mais RSI survente"
            except (ValueError, TypeError):
                pass
                
        # Market regime
        market_regime = values.get('market_regime')
        if market_regime == "trending":
            confidence_boost += 0.10
            reason += " (marché trending)"
        elif market_regime == "ranging":
            confidence_boost -= 0.05
            reason += " (marché ranging)"
            
        # Support/Resistance pour contexte
        if signal_side == "BUY":
            support_levels = values.get('support_levels')
            if support_levels is not None and current_price is not None:
                try:
                    # Supposer que support_levels est une liste ou valeur proche
                    confidence_boost += 0.05
                    reason += " + près support"
                except (ValueError, TypeError):
                    pass
        elif signal_side == "SELL":
            resistance_levels = values.get('resistance_levels')
            if resistance_levels is not None and current_price is not None:
                try:
                    # Supposer que resistance_levels est une liste ou valeur proche
                    confidence_boost += 0.05
                    reason += " + près résistance"
                except (ValueError, TypeError):
                    pass
                    
        # Signal strength et confluence
        signal_strength_calc = values.get('signal_strength')
        if signal_strength_calc is not None:
            try:
                sig_str = float(signal_strength_calc)
                if sig_str > 0.7:
                    confidence_boost += 0.08
                    reason += " + signal fort"
            except (ValueError, TypeError):
                pass
                
        confluence_score = values.get('confluence_score')
        if confluence_score is not None:
            try:
                confluence = float(confluence_score)
                if confluence > 0.6:
                    confidence_boost += 0.12
                    reason += " + confluence"
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
                "obv": obv,
                "obv_ma_10": obv_ma,
                "obv_distance": obv_distance,
                "cross_type": cross_type,
                "obv_oscillator": obv_oscillator,
                "ad_line": ad_line,
                "volume_ratio": volume_ratio,
                "volume_quality_score": volume_quality_score,
                "trade_intensity": trade_intensity,
                "ema_12": ema_12,
                "ema_26": ema_26,
                "ema_50": ema_50,
                "vwap_10": vwap_10,
                "trend_strength": trend_strength,
                "directional_bias": directional_bias,
                "rsi_14": rsi_14,
                "market_regime": market_regime,
                "confluence_score": confluence_score
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que tous les indicateurs OBV requis sont présents."""
        if not super().validate_data():
            return False
            
        required = ['obv', 'obv_ma_10']
        
        for indicator in required:
            if indicator not in self.indicators:
                logger.warning(f"{self.name}: Indicateur manquant: {indicator}")
                return False
            if self.indicators[indicator] is None:
                logger.warning(f"{self.name}: Indicateur null: {indicator}")
                return False
                
        return True
