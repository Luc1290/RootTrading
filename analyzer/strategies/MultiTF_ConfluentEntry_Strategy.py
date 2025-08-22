"""
MultiTF_ConfluentEntry_Strategy - Stratégie basée sur la confluence multi-timeframes.
OPTIMISÉE POUR CRYPTO SPOT INTRADAY
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
        # Paramètres de confluence multi-TF - AJUSTÉS CRYPTO RÉALISTES
        self.min_confluence_score = 35      # Score confluence minimum (accessible)
        self.strong_confluence_score = 55   # Score pour boost fort (réajusté)
        self.min_trend_alignment = 0.15     # Alignement tendance minimum (format 0-1)
        self.strong_trend_alignment = 0.45  # Alignement fort (format 0-1)
        self.volume_confirmation_min = 0.8  # Volume minimum requis
        self.volume_strong = 1.5            # Volume fort pour boost
        
        # Seuils oscillateurs adaptés crypto
        self.rsi_oversold = 28              # RSI survente crypto
        self.rsi_overbought = 72            # RSI surachat crypto
        self.stoch_oversold = 18            # Stoch survente crypto
        self.stoch_overbought = 82          # Stoch surachat crypto
        self.cci_oversold = -120            # CCI survente crypto
        self.cci_overbought = 120           # CCI surachat crypto
        self.williams_oversold = -85        # Williams survente crypto
        self.williams_overbought = -15      # Williams surachat crypto
        
        # ADX pour filtrer les tendances faibles - ASSOUPLI CRYPTO
        self.min_adx_trend = 15             # ADX minimum pour tendance (crypto volatile)
        self.strong_adx_trend = 25          # ADX fort (crypto réaliste)
        
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
            
        # Analyse spécifique pour crypto : importance des EMA courtes
        ema_7 = mas.get('ema_7')
        ema_12 = mas.get('ema_12')
        ema_26 = mas.get('ema_26')
        ema_50 = mas.get('ema_50')
        
        # Score d'alignement plus sophistiqué
        alignment_score = 0.0
        direction = None
        
        # Position du prix par rapport aux MAs
        price_above_count = sum(1 for _, ma_val in mas.items() if current_price > ma_val)
        price_ratio = price_above_count / len(mas)
        
        # Analyse hiérarchique des EMAs (plus important en crypto)
        if ema_7 and ema_12 and ema_26:
            # Configuration haussière parfaite : Prix > EMA7 > EMA12 > EMA26
            if current_price > ema_7 > ema_12 > ema_26:
                alignment_score += 0.4
                direction = "bullish"
                
                # Bonus si EMA50 aussi alignée
                if ema_50 and ema_26 > ema_50:
                    alignment_score += 0.2
                    
            # Configuration baissière parfaite : Prix < EMA7 < EMA12 < EMA26
            elif current_price < ema_7 < ema_12 < ema_26:
                alignment_score += 0.4
                direction = "bearish"
                
                # Bonus si EMA50 aussi alignée
                if ema_50 and ema_26 < ema_50:
                    alignment_score += 0.2
                    
            # Configurations partielles
            elif current_price > ema_7 and ema_7 > ema_26:
                alignment_score += 0.2
                direction = "bullish_weak"
            elif current_price < ema_7 and ema_7 < ema_26:
                alignment_score += 0.2
                direction = "bearish_weak"
                
        # Ajustement selon position globale du prix
        if price_ratio >= 0.75:
            if direction in ["bullish", "bullish_weak"] or direction is None:
                direction = "bullish"
                alignment_score = max(alignment_score + 0.2, 0.6)
        elif price_ratio <= 0.25:
            if direction in ["bearish", "bearish_weak"] or direction is None:
                direction = "bearish"
                alignment_score = max(alignment_score + 0.2, 0.6)
        else:
            if direction is None:
                direction = "neutral"
                alignment_score = 0.3
                
        # Hull MA pour confirmation (très réactive)
        hull_20 = mas.get('hull_20')
        if hull_20:
            if direction == "bullish" and current_price > hull_20:
                alignment_score = min(alignment_score + 0.1, 0.95)
            elif direction == "bearish" and current_price < hull_20:
                alignment_score = min(alignment_score + 0.1, 0.95)
                
        return {
            'alignment_score': alignment_score,
            'direction': direction,
            'price_above_ratio': price_ratio,
            'ma_count': len(mas),
            'ema_aligned': ema_7 is not None and ema_12 is not None and ema_26 is not None
        }
        
    def _analyze_oscillator_confluence(self, values: Dict[str, Optional[float]]) -> Dict[str, Any]:
        """Analyse la confluence des oscillateurs avec seuils crypto."""
        oscillators = {}
        
        # RSI avec seuils crypto
        rsi_14 = values.get('rsi_14')
        if rsi_14 is not None:
            try:
                rsi = float(rsi_14)
                if rsi <= self.rsi_oversold:
                    oscillators['rsi_14'] = 'oversold'
                elif rsi >= self.rsi_overbought:
                    oscillators['rsi_14'] = 'overbought'
                else:
                    oscillators['rsi_14'] = 'neutral'
            except (ValueError, TypeError):
                pass
                
        # RSI 21 pour confirmation
        rsi_21 = values.get('rsi_21')
        if rsi_21 is not None:
            try:
                rsi = float(rsi_21)
                if rsi <= self.rsi_oversold + 2:  # Légèrement plus tolérant
                    oscillators['rsi_21'] = 'oversold'
                elif rsi >= self.rsi_overbought - 2:
                    oscillators['rsi_21'] = 'overbought'
                else:
                    oscillators['rsi_21'] = 'neutral'
            except (ValueError, TypeError):
                pass
                
        # Stochastic avec seuils crypto
        stoch_k = values.get('stoch_k')
        stoch_d = values.get('stoch_d')
        if stoch_k is not None and stoch_d is not None:
            try:
                k = float(stoch_k)
                d = float(stoch_d)
                if k <= self.stoch_oversold and d <= self.stoch_oversold:
                    oscillators['stoch'] = 'oversold'
                elif k >= self.stoch_overbought and d >= self.stoch_overbought:
                    oscillators['stoch'] = 'overbought'
                else:
                    oscillators['stoch'] = 'neutral'
            except (ValueError, TypeError):
                pass
                
        # CCI avec seuils crypto
        cci_20 = values.get('cci_20')
        if cci_20 is not None:
            try:
                cci = float(cci_20)
                if cci <= self.cci_oversold:
                    oscillators['cci'] = 'oversold'
                elif cci >= self.cci_overbought:
                    oscillators['cci'] = 'overbought'
                else:
                    oscillators['cci'] = 'neutral'
            except (ValueError, TypeError):
                pass
                
        # Williams %R avec seuils crypto
        williams_r = values.get('williams_r')
        if williams_r is not None:
            try:
                wr = float(williams_r)
                if wr <= self.williams_oversold:
                    oscillators['williams'] = 'oversold'
                elif wr >= self.williams_overbought:
                    oscillators['williams'] = 'overbought'
                else:
                    oscillators['williams'] = 'neutral'
            except (ValueError, TypeError):
                pass
                
        if not oscillators:
            return {'confluence': 'none', 'strength': 0.0, 'count': 0}
            
        # Calcul de la confluence avec pondération
        oversold_count = sum(1 for v in oscillators.values() if v == 'oversold')
        overbought_count = sum(1 for v in oscillators.values() if v == 'overbought')
        total_count = len(oscillators)
        
        # Seuils ajustés pour crypto (plus stricts)
        if oversold_count >= total_count * 0.7:  # 70% au lieu de 60%
            return {'confluence': 'oversold', 'strength': oversold_count / total_count, 'count': total_count}
        elif overbought_count >= total_count * 0.7:
            return {'confluence': 'overbought', 'strength': overbought_count / total_count, 'count': total_count}
        elif oversold_count >= total_count * 0.5:  # Confluence modérée
            return {'confluence': 'oversold_moderate', 'strength': oversold_count / total_count * 0.7, 'count': total_count}
        elif overbought_count >= total_count * 0.5:
            return {'confluence': 'overbought_moderate', 'strength': overbought_count / total_count * 0.7, 'count': total_count}
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
                "reason": f"Confluence insuffisante ({confluence_score:.1f} < {self.min_confluence_score})",
                "metadata": {"strategy": self.name, "confluence_score": confluence_score}
            }
            
        # Vérification signal_strength valide
        valid_strengths = ['WEAK', 'MODERATE', 'STRONG', 'VERY_STRONG']
        if signal_strength not in valid_strengths:
            signal_strength = 'WEAK'  # Default si invalide
            
        # Vérification ADX pour tendance suffisante (déplacée ici)
        adx_14 = values.get('adx_14')
        adx_value = None
        if adx_14 is not None:
            try:
                adx_value = float(adx_14)
            except (ValueError, TypeError):
                pass
        
        # Filtre signal_strength - ASSOUPLI pour crypto
        # On accepte maintenant WEAK si autres conditions très bonnes
        signal_strength_penalty = 0.0
        if signal_strength == 'WEAK':
            # Vérifier si conditions exceptionnelles compensent
            exceptional_conditions = (
                confluence_score >= 55 and 
                trend_alignment and trend_alignment >= 0.3 and
                adx_value is not None and adx_value >= 20
            )
            if not exceptional_conditions:
                signal_strength_penalty = -0.15  # Pénalité au lieu de rejet
            
        # Vérification trend_alignment
        if trend_alignment and trend_alignment < self.min_trend_alignment:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Alignement insuffisant ({trend_alignment:.1f} < {self.min_trend_alignment})",
                "metadata": {"strategy": self.name, "trend_alignment": trend_alignment}
            }
            
        # Vérification ADX pour tendance suffisante (utilise adx_value déjà calculé)
        if adx_value is not None and adx_value < self.min_adx_trend:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Tendance trop faible (ADX: {adx_value:.1f} < {self.min_adx_trend})",
                "metadata": {"strategy": self.name, "adx": adx_value}
            }
                
        # Analyse de l'alignement des moyennes mobiles
        ma_analysis = self._analyze_ma_alignment(values, current_price)
        
        # Analyse des oscillateurs
        osc_analysis = self._analyze_oscillator_confluence(values)
        
        signal_side = None
        reason = ""
        base_confidence = 0.40  # Base plus accessible
        confidence_boost = signal_strength_penalty  # Inclut pénalité WEAK
        
        # Détermination du signal selon l'alignement MA et oscillateurs
        if ma_analysis['direction'] in ["bullish", "bullish_weak"]:
            # Setup haussier
            if osc_analysis['confluence'] == 'oversold':
                signal_side = "BUY"
                reason = f"Confluence haussière forte (score: {confluence_score:.1f}) + oscillateurs survente parfaite"
                confidence_boost += 0.25
            elif osc_analysis['confluence'] == 'oversold_moderate':
                signal_side = "BUY"
                reason = f"Confluence haussière (score: {confluence_score:.1f}) + oscillateurs survente modérée"
                confidence_boost += 0.15
            elif osc_analysis['confluence'] in ['neutral', 'mixed'] and ma_analysis['alignment_score'] >= 0.7:
                signal_side = "BUY"
                reason = f"Confluence haussière (score: {confluence_score:.1f}) + MA fortement alignées"
                confidence_boost += 0.12
                
        elif ma_analysis['direction'] in ["bearish", "bearish_weak"]:
            # Setup baissier
            if osc_analysis['confluence'] == 'overbought':
                signal_side = "SELL"
                reason = f"Confluence baissière forte (score: {confluence_score:.1f}) + oscillateurs surachat parfait"
                confidence_boost += 0.25
            elif osc_analysis['confluence'] == 'overbought_moderate':
                signal_side = "SELL"
                reason = f"Confluence baissière (score: {confluence_score:.1f}) + oscillateurs surachat modéré"
                confidence_boost += 0.15
            elif osc_analysis['confluence'] in ['neutral', 'mixed'] and ma_analysis['alignment_score'] >= 0.7:
                signal_side = "SELL"
                reason = f"Confluence baissière (score: {confluence_score:.1f}) + MA fortement alignées"
                confidence_boost += 0.12
                
        # Pas d'alignement clair ou contradictoire
        if signal_side is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Pas de confluence claire - MA: {ma_analysis['direction']}, Osc: {osc_analysis['confluence']}",
                "metadata": {
                    "strategy": self.name,
                    "ma_analysis": ma_analysis,
                    "osc_analysis": osc_analysis
                }
            }
            
        # === BOOSTS DE CONFIANCE ===
        
        # Boost selon confluence_score
        if confluence_score >= 75:
            confidence_boost += 0.20
            reason += " [confluence EXCELLENTE]"
        elif confluence_score >= self.strong_confluence_score:
            confidence_boost += 0.12
            reason += " [confluence forte]"
        else:
            confidence_boost += 0.05
            
        # Boost selon signal_strength
        if signal_strength == 'VERY_STRONG':
            confidence_boost += 0.15
            reason += f" + signal {signal_strength}"
        elif signal_strength == 'STRONG':
            confidence_boost += 0.08
            reason += f" + signal {signal_strength}"
        elif signal_strength == 'MODERATE':
            confidence_boost += 0.03
            
        # Boost alignement MA
        if ma_analysis['alignment_score'] >= 0.85:
            confidence_boost += 0.18
            reason += " + MA parfaitement alignées"
        elif ma_analysis['alignment_score'] >= 0.70:
            confidence_boost += 0.10
            reason += " + MA bien alignées"
        elif ma_analysis['alignment_score'] >= 0.50:
            confidence_boost += 0.05
            reason += " + MA alignées"
            
        # Boost trend_alignment global (format 0-1)
        if trend_alignment and trend_alignment >= self.strong_trend_alignment:
            confidence_boost += 0.12
            reason += f" + tendance forte ({trend_alignment:.2f})"
        elif trend_alignment and trend_alignment >= self.min_trend_alignment:
            confidence_boost += 0.05
            reason += f" + tendance ok ({trend_alignment:.2f})"
            
        # Confirmation directional bias
        directional_bias = values.get('directional_bias')
        if directional_bias:
            if (signal_side == "BUY" and directional_bias == "BULLISH") or \
               (signal_side == "SELL" and directional_bias == "BEARISH"):
                confidence_boost += 0.08
                reason += f" + bias confirmé"
            elif (signal_side == "BUY" and directional_bias == "BEARISH") or \
                 (signal_side == "SELL" and directional_bias == "BULLISH"):
                confidence_boost -= 0.15  # Pénalité si contradictoire
                reason += " MAIS bias opposé!"
                
        # Régime de marché
        market_regime = values.get('market_regime')
        regime_strength = values.get('regime_strength')
        regime_confidence = values.get('regime_confidence')
        
        if market_regime:
            if signal_side == "BUY" and market_regime in ["TRENDING_BULL", "BREAKOUT_BULL"]:
                confidence_boost += 0.12
                reason += f" ({market_regime})"
            elif signal_side == "SELL" and market_regime in ["TRENDING_BEAR", "BREAKOUT_BEAR"]:
                confidence_boost += 0.12
                reason += f" ({market_regime})"
            elif market_regime == "VOLATILE":
                confidence_boost -= 0.08  # Pénalité en marché volatile
                reason += " (marché volatile)"
                
        if regime_strength in ['STRONG', 'EXTREME']:
            confidence_boost += 0.06
            
        if regime_confidence is not None:
            try:
                reg_conf = float(regime_confidence)
                if reg_conf >= 80:
                    confidence_boost += 0.08
            except (ValueError, TypeError):
                pass
                
        # Volume confirmation IMPORTANT en crypto - ASSOUPLI
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio < 0.5:  # Seuil très bas pour pénalité forte
                    confidence_boost -= 0.15  # Forte pénalité si volume très faible
                    reason += f" MAIS volume très faible ({vol_ratio:.1f}x)"
                elif vol_ratio < self.volume_confirmation_min:
                    confidence_boost -= 0.08  # Pénalité réduite
                    reason += f" + volume modéré ({vol_ratio:.1f}x)"
                elif vol_ratio >= self.volume_strong * 1.5:  # Volume très élevé
                    confidence_boost += 0.20
                    reason += f" + volume EXCEPTIONNEL ({vol_ratio:.1f}x)"
                elif vol_ratio >= self.volume_strong:
                    confidence_boost += 0.12
                    reason += f" + volume élevé ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.0:
                    confidence_boost += 0.05
                    reason += f" + volume ok ({vol_ratio:.1f}x)"
            except (ValueError, TypeError):
                pass
                
        # Volume quality
        volume_quality_score = values.get('volume_quality_score')
        if volume_quality_score is not None:
            try:
                vq_score = float(volume_quality_score)
                if vq_score >= 75:
                    confidence_boost += 0.10
                    reason += " + volume HQ"
                elif vq_score >= 60:
                    confidence_boost += 0.05
            except (ValueError, TypeError):
                pass
                
        # ADX pour force de tendance - ASSOUPLI (utilise adx_value déjà calculé)
        if adx_value is not None:
            if adx_value >= 35:  # ADX très fort
                confidence_boost += 0.15
                reason += f" + ADX très fort ({adx_value:.1f})"
            elif adx_value >= self.strong_adx_trend:
                confidence_boost += 0.10
                reason += f" + ADX fort ({adx_value:.1f})"
            elif adx_value >= self.min_adx_trend:
                confidence_boost += 0.04
                reason += f" + ADX ok ({adx_value:.1f})"
            else:
                confidence_boost -= 0.05  # Légère pénalité si ADX très faible
                
        # DI+ / DI- pour direction
        plus_di = values.get('plus_di')
        minus_di = values.get('minus_di')
        if plus_di is not None and minus_di is not None:
            try:
                pdi = float(plus_di)
                mdi = float(minus_di)
                if signal_side == "BUY" and pdi > mdi * 1.3:  # DI+ dominant
                    confidence_boost += 0.08
                    reason += " + DI+ dominant"
                elif signal_side == "SELL" and mdi > pdi * 1.3:  # DI- dominant
                    confidence_boost += 0.08
                    reason += " + DI- dominant"
            except (ValueError, TypeError):
                pass
                
        # MACD confirmation
        macd_histogram = values.get('macd_histogram')
        macd_trend = values.get('macd_trend')
        if macd_histogram is not None:
            try:
                hist = float(macd_histogram)
                if signal_side == "BUY" and hist > 0:
                    confidence_boost += 0.06
                    reason += " + MACD+"
                elif signal_side == "SELL" and hist < 0:
                    confidence_boost += 0.06
                    reason += " + MACD-"
                elif (signal_side == "BUY" and hist < -0.001) or (signal_side == "SELL" and hist > 0.001):
                    confidence_boost -= 0.10  # Pénalité si MACD contradictoire
            except (ValueError, TypeError):
                pass
                
        # Pattern detection
        pattern_detected = values.get('pattern_detected')
        pattern_confidence = values.get('pattern_confidence')
        if pattern_detected and pattern_confidence is not None:
            try:
                pattern_conf = float(pattern_confidence)
                if pattern_conf >= 75:
                    confidence_boost += 0.12
                    reason += f" + pattern {pattern_detected}"
                elif pattern_conf >= 60:
                    confidence_boost += 0.06
                    reason += f" + pattern {pattern_detected}"
            except (ValueError, TypeError):
                pass
                
        # Momentum score
        momentum_score = values.get('momentum_score')
        if momentum_score is not None:
            try:
                momentum = float(momentum_score)
                if signal_side == "BUY" and 40 <= momentum <= 60:
                    confidence_boost += 0.08
                    reason += f" + momentum équilibré ({momentum:.0f})"
                elif signal_side == "SELL" and 40 <= momentum <= 60:
                    confidence_boost += 0.08
                    reason += f" + momentum équilibré ({momentum:.0f})"
                elif (signal_side == "BUY" and momentum < 25) or (signal_side == "SELL" and momentum > 75):
                    confidence_boost += 0.12  # Extrême favorable
                    reason += f" + momentum extrême ({momentum:.0f})"
            except (ValueError, TypeError):
                pass
                
        # Support/Resistance proximity
        nearest_support = values.get('nearest_support')
        nearest_resistance = values.get('nearest_resistance')
        if nearest_support is not None and nearest_resistance is not None and current_price:
            try:
                support = float(nearest_support)
                resistance = float(nearest_resistance)
                
                # Distance en pourcentage
                dist_to_support = abs(current_price - support) / current_price
                dist_to_resistance = abs(resistance - current_price) / current_price
                
                if signal_side == "BUY" and dist_to_support < 0.02:  # Proche support (2%)
                    confidence_boost += 0.10
                    reason += " + proche support"
                elif signal_side == "SELL" and dist_to_resistance < 0.02:  # Proche résistance
                    confidence_boost += 0.10
                    reason += " + proche résistance"
            except (ValueError, TypeError):
                pass
                
        # Calcul final
        raw_confidence = base_confidence * (1 + confidence_boost)
        
        # Filtre final - seuil minimum ASSOUPLI pour crypto
        if raw_confidence < 0.30:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Signal rejeté - confiance insuffisante ({raw_confidence:.2f} < 0.30)",
                "metadata": {
                    "strategy": self.name,
                    "rejected_signal": signal_side,
                    "raw_confidence": raw_confidence,
                    "confluence_score": confluence_score,
                    "signal_strength": signal_strength,
                    "trend_alignment": trend_alignment
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
                "adx_14": adx_value,
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
        ma_indicators = ['ema_7', 'ema_12', 'ema_26', 'ema_50', 'sma_20', 'sma_50']
        ma_available = sum(1 for ma in ma_indicators if ma in self.indicators and self.indicators[ma] is not None)
        
        if ma_available < 3:
            logger.warning(f"{self.name}: Pas assez de moyennes mobiles ({ma_available}/6)")
            return False
            
        return True