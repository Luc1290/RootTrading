"""
EMA_Cross_Strategy - Stratégie basée sur les croisements d'EMA.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class EMA_Cross_Strategy(BaseStrategy):
    """
    Stratégie utilisant les croisements d'EMA pour détecter les changements de tendance.
    
    Signaux générés:
    - BUY: EMA rapide croise au-dessus EMA lente + confirmations haussières
    - SELL: EMA rapide croise en-dessous EMA lente + confirmations baissières
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Configuration des EMA
        self.ema_fast_period = 12      # EMA rapide
        self.ema_slow_period = 26      # EMA lente  
        self.ema_filter_period = 50    # EMA filtre pour tendance générale
        self.cross_confirmation = 3    # Barres de confirmation du croisement
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs EMA."""
        return {
            # EMA disponibles
            'ema_7': self.indicators.get('ema_7'),
            'ema_12': self.indicators.get('ema_12'),
            'ema_26': self.indicators.get('ema_26'),
            'ema_50': self.indicators.get('ema_50'),
            'ema_99': self.indicators.get('ema_99'),
            # SMA pour comparaison
            'sma_20': self.indicators.get('sma_20'),
            'sma_50': self.indicators.get('sma_50'),
            # MACD (basé sur EMA 12/26)
            'macd_line': self.indicators.get('macd_line'),
            'macd_signal': self.indicators.get('macd_signal'),
            'macd_histogram': self.indicators.get('macd_histogram'),
            'macd_zero_cross': self.indicators.get('macd_zero_cross'),
            'macd_signal_cross': self.indicators.get('macd_signal_cross'),
            'macd_trend': self.indicators.get('macd_trend'),
            # Contexte tendance
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'trend_alignment': self.indicators.get('trend_alignment'),
            'momentum_score': self.indicators.get('momentum_score'),
            # Volume pour confirmation
            'volume_ratio': self.indicators.get('volume_ratio'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
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
        Génère un signal basé sur les croisements d'EMA.
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
        
        # Vérification des EMA essentielles (12 et 26 pour logique classique)
        try:
            ema_12 = float(values['ema_12']) if values['ema_12'] is not None else None
            ema_26 = float(values['ema_26']) if values['ema_26'] is not None else None
            ema_50 = float(values['ema_50']) if values['ema_50'] is not None else None
        except (ValueError, TypeError) as e:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Erreur conversion EMA: {e}",
                "metadata": {"strategy": self.name}
            }
            
        if ema_12 is None or ema_26 is None or current_price is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "EMA 12/26 ou prix non disponibles",
                "metadata": {"strategy": self.name}
            }
            
        # Analyse du croisement EMA 12/26
        ema_fast_above_slow = ema_12 > ema_26
        ema_distance_pct = abs(ema_12 - ema_26) / ema_26 * 100
        
        # Vérification que les EMA ne sont pas trop proches (éviter faux signaux)
        min_separation = 0.1  # 0.1% minimum de séparation
        if ema_distance_pct < min_separation:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"EMA trop proches ({ema_distance_pct:.2f}%) - pas de signal clair",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "ema_12": ema_12,
                    "ema_26": ema_26,
                    "separation_pct": ema_distance_pct
                }
            }
            
        signal_side = None
        reason = ""
        base_confidence = 0.5
        confidence_boost = 0.0
        cross_type = None
        
        # Logique de croisement
        if ema_fast_above_slow:
            signal_side = "BUY"
            cross_type = "golden_cross"
            reason = f"EMA12 ({ema_12:.2f}) > EMA26 ({ema_26:.2f})"
            confidence_boost += 0.15
        else:
            signal_side = "SELL"
            cross_type = "death_cross"
            reason = f"EMA12 ({ema_12:.2f}) < EMA26 ({ema_26:.2f})"
            confidence_boost += 0.15
            
        # Bonus selon la force de la séparation
        if ema_distance_pct >= 1.0:
            confidence_boost += 0.15
            reason += f" - séparation forte ({ema_distance_pct:.2f}%)"
        elif ema_distance_pct >= 0.5:
            confidence_boost += 0.10
            reason += f" - séparation modérée ({ema_distance_pct:.2f}%)"
        else:
            confidence_boost += 0.05
            reason += f" - séparation faible ({ema_distance_pct:.2f}%)"
            
        # Confirmation avec EMA 50 (filtre de tendance)
        if ema_50 is not None:
            price_vs_ema50 = current_price > ema_50
            
            if signal_side == "BUY" and price_vs_ema50:
                confidence_boost += 0.10
                reason += f" + prix > EMA50 ({ema_50:.2f})"
            elif signal_side == "SELL" and not price_vs_ema50:
                confidence_boost += 0.10
                reason += f" + prix < EMA50 ({ema_50:.2f})"
            elif signal_side == "BUY" and not price_vs_ema50:
                confidence_boost -= 0.05
                reason += f" mais prix < EMA50"
            elif signal_side == "SELL" and price_vs_ema50:
                confidence_boost -= 0.05
                reason += f" mais prix > EMA50"
                
        # Confirmation avec MACD (basé sur EMA 12/26)
        macd_line = values.get('macd_line')
        macd_signal = values.get('macd_signal')
        macd_histogram = values.get('macd_histogram')
        
        if macd_line is not None and macd_signal is not None:
            try:
                macd = float(macd_line)
                macd_sig = float(macd_signal)
                macd_cross = macd > macd_sig
                
                if (signal_side == "BUY" and macd_cross) or \
                   (signal_side == "SELL" and not macd_cross):
                    confidence_boost += 0.15
                    reason += " + MACD confirme"
                else:
                    confidence_boost -= 0.05
                    reason += " mais MACD diverge"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec histogramme MACD
        if macd_histogram is not None:
            try:
                histogram = float(macd_histogram)
                if (signal_side == "BUY" and histogram > 0) or \
                   (signal_side == "SELL" and histogram < 0):
                    confidence_boost += 0.08
                    reason += " + histogram MACD"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec trend_strength (VARCHAR: absent/weak/moderate/strong/very_strong)
        trend_strength = values.get('trend_strength')
        if trend_strength is not None:
            trend_str = str(trend_strength).lower()
            if trend_str in ['strong', 'very_strong']:
                confidence_boost += 0.12
                reason += f" + tendance {trend_str}"
            elif trend_str == 'moderate':
                confidence_boost += 0.08
                reason += f" + tendance {trend_str}"
                
        # Confirmation avec directional_bias
        directional_bias = values.get('directional_bias')
        if directional_bias:
            if (signal_side == "BUY" and directional_bias == "BULLISH") or \
               (signal_side == "SELL" and directional_bias == "BEARISH"):
                confidence_boost += 0.10
                reason += f" + bias {directional_bias}"
                
        # Confirmation avec trend_alignment (toutes les EMA alignées) - format décimal
        trend_alignment = values.get('trend_alignment')
        if trend_alignment is not None:
            try:
                alignment = float(trend_alignment)
                if abs(alignment) > 0.3:  # Format décimal : 0.3 = strong alignment (était 0.7 dans l'ancien format)
                    confidence_boost += 0.10
                    reason += " + EMA alignées"
            except (ValueError, TypeError):
                pass
                
        # Momentum pour éviter signaux contre-tendance (format 0-100, 50=neutre)
        momentum_score = values.get('momentum_score')
        if momentum_score is not None:
            try:
                momentum = float(momentum_score)
                if (signal_side == "BUY" and momentum > 55) or \
                   (signal_side == "SELL" and momentum < 45):
                    confidence_boost += 0.08
                    reason += " + momentum favorable"
                elif (signal_side == "BUY" and momentum < 35) or \
                     (signal_side == "SELL" and momentum > 65):
                    confidence_boost -= 0.10
                    reason += " mais momentum défavorable"
            except (ValueError, TypeError):
                pass
                
        # Volume pour confirmation
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= 1.2:
                    confidence_boost += 0.08
                    reason += f" + volume élevé ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.1:
                    confidence_boost += 0.05
                    reason += f" + volume modéré ({vol_ratio:.1f}x)"
            except (ValueError, TypeError):
                pass
                
        # Volume quality (champ DB: volume_quality_score - format 0-100)
        volume_quality_score = values.get('volume_quality_score')
        if volume_quality_score is not None:
            try:
                vol_quality = float(volume_quality_score)
                if vol_quality > 70:
                    confidence_boost += 0.08
                    reason += f" + volume qualité ({vol_quality:.0f})"
                elif vol_quality > 50:
                    confidence_boost += 0.05
                    reason += f" + volume correct ({vol_quality:.0f})"
            except (ValueError, TypeError):
                pass
                
        # Signal strength (VARCHAR: WEAK/MODERATE/STRONG)
        signal_strength_calc = values.get('signal_strength')
        if signal_strength_calc is not None:
            sig_str = str(signal_strength_calc).upper()
            if sig_str == 'STRONG':
                confidence_boost += 0.10
                reason += " + signal fort"
            elif sig_str == 'MODERATE':
                confidence_boost += 0.05
                reason += " + signal modéré"
                
        confluence_score = values.get('confluence_score')
        if confluence_score is not None:
            try:
                confluence = float(confluence_score)
                if confluence > 60:
                    confidence_boost += 0.12
                    reason += f" + confluence élevée ({confluence:.0f})"
                elif confluence > 45:
                    confidence_boost += 0.08
                    reason += f" + confluence modérée ({confluence:.0f})"
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
                "ema_12": ema_12,
                "ema_26": ema_26,
                "ema_50": ema_50,
                "cross_type": cross_type,
                "ema_separation_pct": ema_distance_pct,
                "macd_line": macd_line,
                "macd_signal": macd_signal,
                "macd_histogram": macd_histogram,
                "trend_strength": trend_strength,
                "directional_bias": directional_bias,
                "trend_alignment": trend_alignment,
                "momentum_score": momentum_score,
                "volume_ratio": volume_ratio,
                "confluence_score": confluence_score
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que tous les indicateurs EMA requis sont présents."""
        if not super().validate_data():
            return False
            
        required = ['ema_12', 'ema_26']
        
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
