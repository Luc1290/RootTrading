"""
MultiTF_ConfluentEntry_Strategy - Stratégie basée sur la confluence multi-timeframes.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class MultiTF_ConfluentEntry_Strategy(BaseStrategy):
    """
    Stratégie de confluence multi-timeframes pour des entrées précises.
    
    Utilise la confluence de plusieurs éléments techniques sur différents timeframes :
    - Trend alignment (toutes les moyennes mobiles alignées)
    - Signal strength élevé
    - Confluence score élevé
    - Support/résistance respectés
    - Volume et momentum favorables
    
    Signaux générés:
    - BUY: Confluence haussière sur multiple timeframes + confirmations
    - SELL: Confluence baissière sur multiple timeframes + confirmations
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres de confluence multi-TF
        self.min_confluence_score = 55      # Score confluence minimum
        self.min_signal_strength = 0.45      # Force signal minimum
        self.min_trend_alignment = 55      # Alignement tendance minimum
        self.max_regime_conflicts = 2       # Max conflits entre régimes
        self.volume_confirmation_min = 1.05  # Volume minimum requis
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs multi-TF."""
        return {
            # Confluence et force du signal
            'confluence_score': self.indicators.get('confluence_score'),
            'signal_strength': self.indicators.get('signal_strength'),
            'pattern_confidence': self.indicators.get('pattern_confidence'),
            'pattern_detected': self.indicators.get('pattern_detected'),
            # Alignement des timeframes
            'trend_alignment': self.indicators.get('trend_alignment'),
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'trend_angle': self.indicators.get('trend_angle'),
            # Régimes de marché
            'market_regime': self.indicators.get('market_regime'),
            'regime_strength': self.indicators.get('regime_strength'),
            'regime_confidence': self.indicators.get('regime_confidence'),
            'regime_duration': self.indicators.get('regime_duration'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            # Support/Résistance (multi-TF)
            'nearest_support': self.indicators.get('nearest_support'),
            'nearest_resistance': self.indicators.get('nearest_resistance'),
            'support_strength': self.indicators.get('support_strength'),
            'resistance_strength': self.indicators.get('resistance_strength'),
            'break_probability': self.indicators.get('break_probability'),
            # Moyennes mobiles (alignement)
            'ema_7': self.indicators.get('ema_7'),
            'ema_12': self.indicators.get('ema_12'),
            'ema_26': self.indicators.get('ema_26'),
            'ema_50': self.indicators.get('ema_50'),
            'ema_99': self.indicators.get('ema_99'),
            'sma_20': self.indicators.get('sma_20'),
            'sma_50': self.indicators.get('sma_50'),
            'hull_20': self.indicators.get('hull_20'),
            # MACD multi-TF
            'macd_line': self.indicators.get('macd_line'),
            'macd_signal': self.indicators.get('macd_signal'),
            'macd_histogram': self.indicators.get('macd_histogram'),
            'macd_trend': self.indicators.get('macd_trend'),
            # Oscillateurs convergents
            'rsi_14': self.indicators.get('rsi_14'),
            'rsi_21': self.indicators.get('rsi_21'),
            'stoch_k': self.indicators.get('stoch_k'),
            'stoch_d': self.indicators.get('stoch_d'),
            'cci_20': self.indicators.get('cci_20'),
            'williams_r': self.indicators.get('williams_r'),
            # Volume multi-TF
            'volume_ratio': self.indicators.get('volume_ratio'),
            'relative_volume': self.indicators.get('relative_volume'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            'volume_context': self.indicators.get('volume_context'),
            # ADX pour confirmation tendance
            'adx_14': self.indicators.get('adx_14'),
            'plus_di': self.indicators.get('plus_di'),
            'minus_di': self.indicators.get('minus_di'),
            # Momentum général
            'momentum_score': self.indicators.get('momentum_score'),
            'momentum_10': self.indicators.get('momentum_10')
        }
        
    def _get_current_price(self) -> Optional[float]:
        """Récupère le prix actuel depuis les données OHLCV."""
        try:
            if self.data and 'close' in self.data and self.data['close']:
                return float(self.data['close'][-1])
        except (IndexError, ValueError, TypeError):
            pass
        return None
        
    def _analyze_ma_alignment(self, values: Dict[str, Optional[float]], 
                             current_price: float) -> Dict[str, Any]:
        """Analyse l'alignement des moyennes mobiles pour détecter la tendance."""
        mas = {}
        ma_keys = ['ema_7', 'ema_12', 'ema_26', 'ema_50', 'ema_99', 'sma_20', 'sma_50', 'hull_20']
        
        # Récupération des MAs disponibles
        for key in ma_keys:
            value = values.get(key)
            if value is not None:
                try:
                    mas[key] = float(value)
                except (ValueError, TypeError):
                    continue
                    
        if len(mas) < 3:
            return {'alignment_score': 0.0, 'direction': None, 'reason': 'Pas assez de MAs'}
            
        # Tri des MAs par valeur
        sorted_mas = sorted(mas.items(), key=lambda x: x[1])
        
        # Analyse pour direction haussière (prix > MA rapides > MA lentes)
        bullish_alignment = 0
        bearish_alignment = 0
        total_checks = 0
        
        # Vérifier ordre croissant pour haussier
        ma_values = [ma[1] for ma in sorted_mas]
        for i in range(len(ma_values) - 1):
            total_checks += 1
            if ma_values[i] < ma_values[i + 1]:
                bullish_alignment += 1
            else:
                bearish_alignment += 1
                
        # Position du prix par rapport aux MAs
        price_above_count = sum(1 for _, ma_val in mas.items() if current_price > ma_val)
        price_ratio = price_above_count / len(mas)
        
        if price_ratio >= 0.8 and bullish_alignment / total_checks >= 0.7:
            direction = "bullish"
            alignment_score = min(0.9, (bullish_alignment / total_checks) * price_ratio)
        elif price_ratio <= 0.2 and bearish_alignment / total_checks >= 0.7:
            direction = "bearish"  
            alignment_score = min(0.9, (bearish_alignment / total_checks) * (1 - price_ratio))
        else:
            direction = "neutral"
            alignment_score = 0.3
            
        return {
            'alignment_score': alignment_score,
            'direction': direction,
            'price_above_ratio': price_ratio,
            'ma_count': len(mas)
        }
        
    def _analyze_oscillator_confluence(self, values: Dict[str, Optional[float]]) -> Dict[str, Any]:
        """Analyse la confluence des oscillateurs."""
        oscillators = {}
        
        # RSI
        if values.get('rsi_14') is not None:
            try:
                rsi = float(values['rsi_14']) if values['rsi_14'] is not None else 0.0
                if rsi <= 30:
                    oscillators['rsi_14'] = 'oversold'
                elif rsi >= 70:
                    oscillators['rsi_14'] = 'overbought'
                else:
                    oscillators['rsi_14'] = 'neutral'
            except (ValueError, TypeError):
                pass
                
        # Stochastic
        if values.get('stoch_k') is not None and values.get('stoch_d') is not None:
            try:
                k = float(values['stoch_k']) if values['stoch_k'] is not None else 0.0
                d = float(values['stoch_d']) if values['stoch_d'] is not None else 0.0
                if k <= 20 and d <= 20:
                    oscillators['stoch'] = 'oversold'
                elif k >= 80 and d >= 80:
                    oscillators['stoch'] = 'overbought'
                else:
                    oscillators['stoch'] = 'neutral'
            except (ValueError, TypeError):
                pass
                
        # CCI
        if values.get('cci_20') is not None:
            try:
                cci = float(values['cci_20']) if values['cci_20'] is not None else 0.0
                if cci <= -100:
                    oscillators['cci'] = 'oversold'
                elif cci >= 100:
                    oscillators['cci'] = 'overbought'
                else:
                    oscillators['cci'] = 'neutral'
            except (ValueError, TypeError):
                pass
                
        # Williams %R
        if values.get('williams_r') is not None:
            try:
                wr = float(values['williams_r']) if values['williams_r'] is not None else 0.0
                if wr <= -80:
                    oscillators['williams'] = 'oversold'
                elif wr >= -20:
                    oscillators['williams'] = 'overbought'
                else:
                    oscillators['williams'] = 'neutral'
            except (ValueError, TypeError):
                pass
                
        if not oscillators:
            return {'confluence': 'none', 'strength': 0.0, 'count': 0}
            
        # Calcul de la confluence
        oversold_count = sum(1 for v in oscillators.values() if v == 'oversold')
        overbought_count = sum(1 for v in oscillators.values() if v == 'overbought')
        total_count = len(oscillators)
        
        if oversold_count >= total_count * 0.7:
            return {'confluence': 'oversold', 'strength': oversold_count / total_count, 'count': total_count}
        elif overbought_count >= total_count * 0.7:
            return {'confluence': 'overbought', 'strength': overbought_count / total_count, 'count': total_count}
        else:
            return {'confluence': 'mixed', 'strength': 0.3, 'count': total_count}
            
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur la confluence multi-timeframes.
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
        
        if current_price is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Prix non disponible",
                "metadata": {"strategy": self.name}
            }
            
        # Vérification des scores de confluence principaux
        try:
            confluence_score = float(values['confluence_score']) if values['confluence_score'] is not None else None
            signal_strength = values['signal_strength']  # STRING: WEAK/MODERATE/STRONG/VERY_STRONG
            trend_alignment = float(values['trend_alignment']) if values['trend_alignment'] is not None else None
        except (ValueError, TypeError) as e:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Erreur conversion scores: {e}",
                "metadata": {"strategy": self.name}
            }
            
        # Vérification des scores minimums
        if confluence_score is None or confluence_score < self.min_confluence_score:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Confluence insuffisante ({confluence_score:.2f} < {self.min_confluence_score})",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "confluence_score": confluence_score
                }
            }
            
        if signal_strength is None or signal_strength not in ['STRONG', 'VERY_STRONG']:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Signal trop faible ({signal_strength})",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "signal_strength": signal_strength
                }
            }
            
        # Analyse de l'alignement des moyennes mobiles
        ma_analysis = self._analyze_ma_alignment(values, current_price)
        
        if ma_analysis['alignment_score'] < (self.min_trend_alignment / 100):
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Alignement MA insuffisant ({ma_analysis['alignment_score']:.2f})",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "ma_alignment": ma_analysis
                }
            }
            
        # Analyse des oscillateurs
        osc_analysis = self._analyze_oscillator_confluence(values)
        
        signal_side = None
        reason = ""
        base_confidence = 0.55  # Base réduite pour équilibrage avec autres stratégies
        confidence_boost = 0.0
        
        # Détermination du signal selon l'alignement MA et oscillateurs
        if ma_analysis['direction'] == "bullish":
            # Setup haussier
            if osc_analysis['confluence'] == 'oversold':
                signal_side = "BUY"
                reason = f"Confluence haussière forte (score: {confluence_score:.2f}) + oscillateurs survente"
                confidence_boost += 0.20
            elif osc_analysis['confluence'] == 'neutral' or osc_analysis['confluence'] == 'mixed':
                signal_side = "BUY"
                reason = f"Confluence haussière forte (score: {confluence_score:.2f}) + MA alignées"
                confidence_boost += 0.10
            # Si oscillateurs en surachat, pas de signal (éviter tops)
            
        elif ma_analysis['direction'] == "bearish":
            # Setup baissier
            if osc_analysis['confluence'] == 'overbought':
                signal_side = "SELL"
                reason = f"Confluence baissière forte (score: {confluence_score:.2f}) + oscillateurs surachat"
                confidence_boost += 0.20
            elif osc_analysis['confluence'] == 'neutral' or osc_analysis['confluence'] == 'mixed':
                signal_side = "SELL"
                reason = f"Confluence baissière forte (score: {confluence_score:.2f}) + MA alignées"
                confidence_boost += 0.10
            # Si oscillateurs en survente, pas de signal (éviter bottoms)
            
        # Pas d'alignement clair
        if signal_side is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"MA direction {ma_analysis['direction']} + oscillateurs {osc_analysis['confluence']} - pas de setup clair",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "ma_analysis": ma_analysis,
                    "osc_analysis": osc_analysis
                }
            }
            
        # Bonus selon la force des scores
        if confluence_score >= 90:
            confidence_boost += 0.15
            reason += " - confluence exceptionnelle"
        elif confluence_score >= 80:
            confidence_boost += 0.10
            reason += " - confluence très forte"
            
        if signal_strength == 'VERY_STRONG':
            confidence_boost += 0.10
            reason += f" + signal très fort ({signal_strength})"
        elif signal_strength == 'STRONG':
            confidence_boost += 0.05
            reason += f" + signal fort ({signal_strength})"
            
        # Bonus alignement MA
        if ma_analysis['alignment_score'] >= 0.9:
            confidence_boost += 0.15
            reason += " + MA parfaitement alignées"
        elif ma_analysis['alignment_score'] >= 0.8:
            confidence_boost += 0.10
            reason += " + MA bien alignées"
            
        # Confirmation avec directional bias
        directional_bias = values.get('directional_bias')
        if directional_bias:
            if (signal_side == "BUY" and directional_bias == "BULLISH") or \
               (signal_side == "SELL" and directional_bias == "BEARISH"):
                confidence_boost += 0.10
                reason += f" + bias {directional_bias}"
                
        # Régime de marché
        market_regime = values.get('market_regime')
        regime_strength = values.get('regime_strength')
        
        if market_regime in ["TRENDING_BULL", "TRENDING_BEAR"]:
            confidence_boost += 0.10
            reason += " (marché trending)"
            
            if regime_strength in ['STRONG', 'EXTREME']:
                confidence_boost += 0.08
                reason += f" avec régime {str(regime_strength).lower()}"
                    
        # Volume pour confirmation
        volume_ratio = values.get('volume_ratio')
        volume_quality_score = values.get('volume_quality_score')
        
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= self.volume_confirmation_min * 1.5:
                    confidence_boost += 0.15
                    reason += f" + volume exceptionnel ({vol_ratio:.1f}x)"
                elif vol_ratio >= self.volume_confirmation_min:
                    confidence_boost += 0.10
                    reason += f" + volume confirmé ({vol_ratio:.1f}x)"
                else:
                    confidence_boost -= 0.05
                    reason += f" mais volume faible ({vol_ratio:.1f}x)"
            except (ValueError, TypeError):
                pass
                
        if volume_quality_score is not None:
            try:
                volume_quality_score = float(volume_quality_score)
                if volume_quality_score > 80:
                    confidence_boost += 0.08
                    reason += " + volume de qualité"
            except (ValueError, TypeError):
                pass
                
        # ADX pour confirmation de tendance
        adx_14 = values.get('adx_14')
        if adx_14 is not None:
            try:
                adx = float(adx_14)
                if adx > 25:
                    confidence_boost += 0.08
                    reason += f" + ADX fort ({adx:.1f})"
            except (ValueError, TypeError):
                pass
                
        # Pattern recognition
        pattern_detected = values.get('pattern_detected')
        pattern_confidence = values.get('pattern_confidence')
        
        if pattern_detected and pattern_confidence is not None:
            try:
                pattern_conf = float(pattern_confidence)
                if pattern_conf > 70:
                    confidence_boost += 0.10
                    reason += f" + pattern {pattern_detected}"
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
                "confluence_score": confluence_score,
                "signal_strength": signal_strength,
                "trend_alignment": trend_alignment,
                "ma_analysis": ma_analysis,
                "osc_analysis": osc_analysis,
                "directional_bias": directional_bias,
                "market_regime": market_regime,
                "regime_strength": regime_strength,
                "volume_ratio": volume_ratio,
                "volume_quality_score": volume_quality_score,
                "adx_14": adx_14,
                "pattern_detected": pattern_detected,
                "pattern_confidence": pattern_confidence
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que tous les indicateurs de confluence requis sont présents."""
        if not super().validate_data():
            return False
            
        required = ['confluence_score', 'signal_strength']
        
        for indicator in required:
            if indicator not in self.indicators:
                logger.warning(f"{self.name}: Indicateur manquant: {indicator}")
                return False
            if self.indicators[indicator] is None:
                logger.warning(f"{self.name}: Indicateur null: {indicator}")
                return False
                
        # Vérifier qu'on a au moins quelques moyennes mobiles
        ma_indicators = ['ema_12', 'ema_26', 'ema_50', 'sma_20', 'sma_50']
        ma_available = sum(1 for ma in ma_indicators if ma in self.indicators and self.indicators[ma] is not None)
        
        if ma_available < 3:
            logger.warning(f"{self.name}: Pas assez de moyennes mobiles ({ma_available}/5)")
            return False
            
        return True
