"""
VWAP_Support_Resistance_Strategy - Stratégie utilisant VWAP comme niveau dynamique de support/résistance.
Le VWAP (Volume-Weighted Average Price) agit comme une référence importante pour les institutions.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class VWAP_Support_Resistance_Strategy(BaseStrategy):
    """
    Stratégie combinant VWAP avec les niveaux de support/résistance statiques.
    
    Principe VWAP :
    - VWAP = support dynamique quand prix au-dessus
    - VWAP = résistance dynamique quand prix en-dessous
    - Confluence VWAP + support/résistance statique = signal fort
    
    Signaux générés:
    - BUY: Prix rebondit sur VWAP support + confluence avec support statique
    - SELL: Prix rejette VWAP résistance + confluence avec résistance statique
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        
        # Paramètres VWAP
        self.vwap_distance_threshold = 0.003      # 0.3% distance pour considérer près VWAP
        self.vwap_confluence_threshold = 0.01     # 1% pour confluence avec S/R statique
        self.strong_vwap_volume_threshold = 1.5   # Volume 50% au-dessus pour VWAP fort
        
        # Paramètres rebond/rejet
        self.min_bounce_strength = 0.002          # Rebond minimum 0.2%
        self.max_bounce_distance = 0.015          # Distance max 1.5% du niveau
        self.rejection_confirmation_bars = 2      # Barres pour confirmer rejet
        
        # Paramètres support/résistance
        self.min_sr_strength = 0.4               # Force minimum niveau S/R
        self.confluence_bonus_multiplier = 1.5   # Multiplicateur bonus confluence
        
        # Paramètres volume et momentum
        self.min_volume_confirmation = 1.2       # Volume minimum pour confirmation
        self.momentum_alignment_required = True  # Momentum doit être aligné
        self.min_momentum_threshold = 0.15       # Momentum minimum requis
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs pré-calculés."""
        return {
            # VWAP principal (support/résistance dynamique)
            'vwap_10': self.indicators.get('vwap_10'),
            'anchored_vwap': self.indicators.get('anchored_vwap'),
            'vwap_upper_band': self.indicators.get('vwap_upper_band'),
            'vwap_lower_band': self.indicators.get('vwap_lower_band'),
            
            # Support/Résistance statiques
            'nearest_support': self.indicators.get('nearest_support'),
            'nearest_resistance': self.indicators.get('nearest_resistance'),
            'support_strength': self.indicators.get('support_strength'),
            'resistance_strength': self.indicators.get('resistance_strength'),
            'support_levels': self.indicators.get('support_levels'),
            'resistance_levels': self.indicators.get('resistance_levels'),
            'break_probability': self.indicators.get('break_probability'),
            'pivot_count': self.indicators.get('pivot_count'),
            
            # Volume analysis (crucial pour VWAP)
            'volume_ratio': self.indicators.get('volume_ratio'),
            'relative_volume': self.indicators.get('relative_volume'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            'trade_intensity': self.indicators.get('trade_intensity'),
            'volume_spike_multiplier': self.indicators.get('volume_spike_multiplier'),
            'avg_volume_20': self.indicators.get('avg_volume_20'),
            
            # Momentum et tendance
            'momentum_score': self.indicators.get('momentum_score'),
            'directional_bias': self.indicators.get('directional_bias'),
            'trend_strength': self.indicators.get('trend_strength'),
            'trend_alignment': self.indicators.get('trend_alignment'),
            
            # RSI pour timing
            'rsi_14': self.indicators.get('rsi_14'),
            'rsi_21': self.indicators.get('rsi_21'),
            'williams_r': self.indicators.get('williams_r'),
            
            # MACD pour confirmation momentum
            'macd_line': self.indicators.get('macd_line'),
            'macd_signal': self.indicators.get('macd_signal'),
            'macd_histogram': self.indicators.get('macd_histogram'),
            'macd_trend': self.indicators.get('macd_trend'),
            
            # ADX pour force tendance
            'adx_14': self.indicators.get('adx_14'),
            'plus_di': self.indicators.get('plus_di'),
            'minus_di': self.indicators.get('minus_di'),
            
            # EMA pour contexte tendance
            'ema_12': self.indicators.get('ema_12'),
            'ema_26': self.indicators.get('ema_26'),
            'ema_50': self.indicators.get('ema_50'),
            
            # ATR pour volatilité
            'atr_14': self.indicators.get('atr_14'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            'atr_percentile': self.indicators.get('atr_percentile'),
            
            # OBV pour validation volume
            'obv': self.indicators.get('obv'),
            'obv_ma_10': self.indicators.get('obv_ma_10'),
            'obv_oscillator': self.indicators.get('obv_oscillator'),
            
            # Contexte marché
            'market_regime': self.indicators.get('market_regime'),
            'regime_strength': self.indicators.get('regime_strength'),
            'confluence_score': self.indicators.get('confluence_score'),
            'signal_strength': self.indicators.get('signal_strength'),
            'pattern_detected': self.indicators.get('pattern_detected'),
            'pattern_confidence': self.indicators.get('pattern_confidence')
        }
        
    def _detect_vwap_support_bounce(self, values: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Détecte un rebond sur VWAP agissant comme support."""
        bounce_score = 0.0
        bounce_indicators = []
        
        # VWAP principal
        vwap_10 = values.get('vwap_10')
        if vwap_10 is None:
            return {'is_bounce': False, 'score': 0.0, 'indicators': []}
            
        try:
            vwap_val = float(vwap_10)
        except (ValueError, TypeError):
            return {'is_bounce': False, 'score': 0.0, 'indicators': []}
            
        # Vérifier si prix est près du VWAP (potentiel support)
        if current_price <= vwap_val:
            return {
                'is_bounce': False,
                'score': 0.0, 
                'indicators': ['Prix sous VWAP - pas de support'],
                'vwap_level': vwap_val
            }
            
        # Distance au VWAP
        vwap_distance = (current_price - vwap_val) / vwap_val
        
        if vwap_distance > self.max_bounce_distance:
            return {
                'is_bounce': False,
                'score': 0.0,
                'indicators': [f'Prix trop loin VWAP ({vwap_distance*100:.1f}%)'],
                'vwap_level': vwap_val
            }
            
        # Bounce scoring selon la proximité
        if vwap_distance <= self.vwap_distance_threshold:
            bounce_score += 0.3
            bounce_indicators.append(f"Prix très près VWAP ({vwap_distance*100:.2f}%)")
        elif vwap_distance <= self.vwap_distance_threshold * 2:
            bounce_score += 0.2
            bounce_indicators.append(f"Prix près VWAP ({vwap_distance*100:.2f}%)")
        else:
            bounce_score += 0.1
            bounce_indicators.append(f"Prix proche VWAP ({vwap_distance*100:.2f}%)")
            
        # Confluence avec support statique
        nearest_support = values.get('nearest_support')
        if nearest_support is not None:
            try:
                support_level = float(nearest_support)
                support_vwap_distance = abs(support_level - vwap_val) / vwap_val
                
                if support_vwap_distance <= self.vwap_confluence_threshold:
                    bounce_score += 0.25
                    bounce_indicators.append(f"Confluence VWAP/Support ({support_vwap_distance*100:.2f}%)")
                    
                    # Bonus si support fort
                    support_strength = values.get('support_strength')
                    if support_strength is not None:
                        try:
                            if isinstance(support_strength, str):
                                strength_map = {'WEAK': 0.2, 'MODERATE': 0.5, 'STRONG': 0.8, 'MAJOR': 1.0}
                                strength_val = strength_map.get(support_strength.upper(), 0.5)
                            else:
                                strength_val = float(support_strength)
                                
                            if strength_val >= 0.8:
                                bounce_score += 0.2
                                bounce_indicators.append(f"Support très fort ({strength_val:.2f})")
                            elif strength_val >= self.min_sr_strength:
                                bounce_score += 0.15
                                bounce_indicators.append(f"Support fort ({strength_val:.2f})")
                        except (ValueError, TypeError):
                            pass
            except (ValueError, TypeError):
                pass
                
        # Volume confirmation (crucial pour VWAP)
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= self.strong_vwap_volume_threshold:
                    bounce_score += 0.2
                    bounce_indicators.append(f"Volume très fort ({vol_ratio:.1f}x)")
                elif vol_ratio >= self.min_volume_confirmation:
                    bounce_score += 0.15
                    bounce_indicators.append(f"Volume fort ({vol_ratio:.1f}x)")
            except (ValueError, TypeError):
                pass
                
        return {
            'is_bounce': bounce_score >= 0.4,
            'score': bounce_score,
            'indicators': bounce_indicators,
            'vwap_level': vwap_val,
            'vwap_distance_pct': vwap_distance * 100
        }
        
    def _detect_vwap_resistance_rejection(self, values: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Détecte un rejet sur VWAP agissant comme résistance."""
        rejection_score = 0.0
        rejection_indicators = []
        
        # VWAP principal  
        vwap_10 = values.get('vwap_10')
        if vwap_10 is None:
            return {'is_rejection': False, 'score': 0.0, 'indicators': []}
            
        try:
            vwap_val = float(vwap_10)
        except (ValueError, TypeError):
            return {'is_rejection': False, 'score': 0.0, 'indicators': []}
            
        # Vérifier si prix est près du VWAP (potentiel résistance)
        if current_price >= vwap_val:
            return {
                'is_rejection': False,
                'score': 0.0,
                'indicators': ['Prix au-dessus VWAP - pas de résistance'],
                'vwap_level': vwap_val
            }
            
        # Distance au VWAP
        vwap_distance = (vwap_val - current_price) / current_price
        
        if vwap_distance > self.max_bounce_distance:
            return {
                'is_rejection': False,
                'score': 0.0,
                'indicators': [f'Prix trop loin VWAP ({vwap_distance*100:.1f}%)'],
                'vwap_level': vwap_val
            }
            
        # Rejection scoring selon la proximité
        if vwap_distance <= self.vwap_distance_threshold:
            rejection_score += 0.3
            rejection_indicators.append(f"Prix très près VWAP ({vwap_distance*100:.2f}%)")
        elif vwap_distance <= self.vwap_distance_threshold * 2:
            rejection_score += 0.2
            rejection_indicators.append(f"Prix près VWAP ({vwap_distance*100:.2f}%)")
        else:
            rejection_score += 0.1
            rejection_indicators.append(f"Prix proche VWAP ({vwap_distance*100:.2f}%)")
            
        # Confluence avec résistance statique
        nearest_resistance = values.get('nearest_resistance')
        if nearest_resistance is not None:
            try:
                resistance_level = float(nearest_resistance)
                resistance_vwap_distance = abs(resistance_level - vwap_val) / current_price
                
                if resistance_vwap_distance <= self.vwap_confluence_threshold:
                    rejection_score += 0.25
                    rejection_indicators.append(f"Confluence VWAP/Résistance ({resistance_vwap_distance*100:.2f}%)")
                    
                    # Bonus si résistance forte
                    resistance_strength = values.get('resistance_strength')
                    if resistance_strength is not None:
                        try:
                            if isinstance(resistance_strength, str):
                                strength_map = {'WEAK': 0.2, 'MODERATE': 0.5, 'STRONG': 0.8, 'MAJOR': 1.0}
                                strength_val = strength_map.get(resistance_strength.upper(), 0.5)
                            else:
                                strength_val = float(resistance_strength)
                                
                            if strength_val >= 0.8:
                                rejection_score += 0.2
                                rejection_indicators.append(f"Résistance très forte ({strength_val:.2f})")
                            elif strength_val >= self.min_sr_strength:
                                rejection_score += 0.15
                                rejection_indicators.append(f"Résistance forte ({strength_val:.2f})")
                        except (ValueError, TypeError):
                            pass
            except (ValueError, TypeError):
                pass
                
        # Volume confirmation (crucial pour VWAP)
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= self.strong_vwap_volume_threshold:
                    rejection_score += 0.2
                    rejection_indicators.append(f"Volume très fort ({vol_ratio:.1f}x)")
                elif vol_ratio >= self.min_volume_confirmation:
                    rejection_score += 0.15
                    rejection_indicators.append(f"Volume fort ({vol_ratio:.1f}x)")
            except (ValueError, TypeError):
                pass
                
        return {
            'is_rejection': rejection_score >= 0.4,
            'score': rejection_score,
            'indicators': rejection_indicators,
            'vwap_level': vwap_val,
            'vwap_distance_pct': vwap_distance * 100
        }
        
    def _detect_momentum_alignment(self, values: Dict[str, Any], signal_direction: str) -> Dict[str, Any]:
        """Détecte l'alignement du momentum avec la direction du signal."""
        momentum_score = 0
        momentum_indicators = []
        
        # Momentum score général
        momentum_val = values.get('momentum_score')
        if momentum_val is not None:
            try:
                momentum = float(momentum_val)
                
                if signal_direction == "BUY" and momentum >= self.min_momentum_threshold:
                    momentum_score += 25
                    momentum_indicators.append(f"Momentum haussier ({momentum:.2f})")
                elif signal_direction == "SELL" and momentum <= -self.min_momentum_threshold:
                    momentum_score += 25
                    momentum_indicators.append(f"Momentum baissier ({momentum:.2f})")
            except (ValueError, TypeError):
                pass
                
        # Directional bias alignment
        directional_bias = values.get('directional_bias')
        if directional_bias:
            if (signal_direction == "BUY" and directional_bias == 'bullish') or \
               (signal_direction == "SELL" and directional_bias == 'bearish'):
                momentum_score += 20
                momentum_indicators.append(f"Bias directionnel {directional_bias}")
                
        # MACD confirmation
        macd_line = values.get('macd_line')
        macd_signal = values.get('macd_signal')
        if macd_line is not None and macd_signal is not None:
            try:
                macd_val = float(macd_line)
                macd_sig = float(macd_signal)
                
                if signal_direction == "BUY" and macd_val > macd_sig:
                    momentum_score += 15
                    momentum_indicators.append("MACD haussier")
                elif signal_direction == "SELL" and macd_val < macd_sig:
                    momentum_score += 15
                    momentum_indicators.append("MACD baissier")
            except (ValueError, TypeError):
                pass
                
        # ADX force tendance
        adx_14 = values.get('adx_14')
        plus_di = values.get('plus_di')
        minus_di = values.get('minus_di')
        
        if adx_14 is not None and plus_di is not None and minus_di is not None:
            try:
                adx_val = float(adx_14)
                plus_di_val = float(plus_di)
                minus_di_val = float(minus_di)
                
                if adx_val > 25:  # Tendance forte
                    if signal_direction == "BUY" and plus_di_val > minus_di_val:
                        momentum_score += 15
                        momentum_indicators.append(f"ADX fort + DI+ ({adx_val:.1f})")
                    elif signal_direction == "SELL" and minus_di_val > plus_di_val:
                        momentum_score += 15
                        momentum_indicators.append(f"ADX fort + DI- ({adx_val:.1f})")
            except (ValueError, TypeError):
                pass
                
        # RSI pour timing (éviter extrêmes)
        rsi_14 = values.get('rsi_14')
        if rsi_14 is not None:
            try:
                rsi_val = float(rsi_14)
                if signal_direction == "BUY" and 30 <= rsi_val <= 70:
                    momentum_score += 10
                    momentum_indicators.append(f"RSI favorable BUY ({rsi_val:.1f})")
                elif signal_direction == "SELL" and 30 <= rsi_val <= 70:
                    momentum_score += 10
                    momentum_indicators.append(f"RSI favorable SELL ({rsi_val:.1f})")
                elif (signal_direction == "BUY" and rsi_val >= 80) or \
                     (signal_direction == "SELL" and rsi_val <= 20):
                    momentum_score -= 10
                    momentum_indicators.append(f"RSI extrême ({rsi_val:.1f})")
            except (ValueError, TypeError):
                pass
                
        return {
            'is_aligned': momentum_score >= 30,
            'score': momentum_score,
            'indicators': momentum_indicators
        }
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur VWAP Support/Resistance.
        """
        if not self.validate_data():
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Données insuffisantes",
                "metadata": {}
            }
            
        values = self._get_current_values()
        
        # Récupérer prix actuel
        current_price = None
        if 'close' in self.data and self.data['close']:
            try:
                current_price = float(self.data['close'][-1])
            except (IndexError, ValueError, TypeError):
                pass
            
        if current_price is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Prix actuel non disponible",
                "metadata": {"strategy": self.name}
            }
            
        # Analyse VWAP support (signal BUY)
        support_analysis = self._detect_vwap_support_bounce(values, current_price)
        
        # Analyse VWAP résistance (signal SELL)  
        resistance_analysis = self._detect_vwap_resistance_rejection(values, current_price)
        
        # Déterminer signal principal
        signal_side = None
        primary_analysis = None
        
        if support_analysis['is_bounce'] and resistance_analysis['is_rejection']:
            # Conflit - prendre le score le plus élevé
            if support_analysis['score'] > resistance_analysis['score']:
                signal_side = "BUY"
                primary_analysis = support_analysis
            else:
                signal_side = "SELL"
                primary_analysis = resistance_analysis
        elif support_analysis['is_bounce']:
            signal_side = "BUY"
            primary_analysis = support_analysis
        elif resistance_analysis['is_rejection']:
            signal_side = "SELL"
            primary_analysis = resistance_analysis
            
        if signal_side is None:
            # Diagnostic des conditions manquées
            missing_conditions = []
            if support_analysis['score'] < 0.4:
                missing_conditions.append(f"Support VWAP faible (score: {support_analysis['score']:.2f})")
            if resistance_analysis['score'] < 0.4:
                missing_conditions.append(f"Résistance VWAP faible (score: {resistance_analysis['score']:.2f})")
                
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Conditions VWAP insuffisantes: {'; '.join(missing_conditions[:2])}",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "current_price": current_price,
                    "support_score": support_analysis['score'],
                    "resistance_score": resistance_analysis['score']
                }
            }
            
        # Vérifier alignement momentum si requis
        momentum_analysis = self._detect_momentum_alignment(values, signal_side)
        
        if self.momentum_alignment_required and momentum_analysis is not None and not momentum_analysis['is_aligned']:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"VWAP {signal_side} détecté mais momentum pas aligné (score: {momentum_analysis['score']:.2f})",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "vwap_score": primary_analysis['score'] if primary_analysis else 0.0,
                    "momentum_score": momentum_analysis['score'] if momentum_analysis else 0.0
                }
            }
            
        # Construire signal final
        base_confidence = 0.55
        confidence_boost = 0.0
        
        # Score VWAP principal
        if primary_analysis is not None:
            confidence_boost += primary_analysis['score'] * 0.4
        
        # Score momentum
        if momentum_analysis is not None:
            confidence_boost += momentum_analysis['score'] * 0.3
        
        # Construire raison
        vwap_level = primary_analysis['vwap_level'] if primary_analysis else 0.0
        vwap_distance = primary_analysis['vwap_distance_pct'] if primary_analysis else 0.0
        
        if signal_side == "BUY":
            reason = f"VWAP support {vwap_level:.2f} (distance: {vwap_distance:.2f}%)"
        else:
            reason = f"VWAP résistance {vwap_level:.2f} (distance: {vwap_distance:.2f}%)"
            
        if primary_analysis and primary_analysis['indicators']:
            reason += f" - {primary_analysis['indicators'][0]}"
            
        if momentum_analysis and momentum_analysis['indicators']:
            reason += f" + {momentum_analysis['indicators'][0]}"
            
        # Bonus confluences et confirmations supplémentaires
        
        # Trend alignment
        trend_alignment = values.get('trend_alignment')
        if trend_alignment is not None:
            try:
                trend_align = float(trend_alignment)
                if signal_side == "BUY" and trend_align >= 20:
                    confidence_boost += 0.1
                    reason += " + trend haussier"
                elif signal_side == "SELL" and trend_align <= -20:
                    confidence_boost += 0.1
                    reason += " + trend baissier"
            except (ValueError, TypeError):
                pass
                
        # OBV confirmation
        obv_oscillator = values.get('obv_oscillator')
        if obv_oscillator is not None:
            try:
                obv_osc = float(obv_oscillator)
                if signal_side == "BUY" and obv_osc > 0:
                    confidence_boost += 0.08
                    reason += " + OBV positif"
                elif signal_side == "SELL" and obv_osc < 0:
                    confidence_boost += 0.08
                    reason += " + OBV négatif"
            except (ValueError, TypeError):
                pass
                
        # EMA context
        ema_12 = values.get('ema_12')
        ema_26 = values.get('ema_26')
        if ema_12 is not None and ema_26 is not None:
            try:
                ema12_val = float(ema_12)
                ema26_val = float(ema_26)
                
                if signal_side == "BUY" and ema12_val > ema26_val:
                    confidence_boost += 0.08
                    reason += " + EMA haussier"
                elif signal_side == "SELL" and ema12_val < ema26_val:
                    confidence_boost += 0.08
                    reason += " + EMA baissier"
            except (ValueError, TypeError):
                pass
                
        # CORRECTION MAGISTRALE: Volatility context avec seuils adaptatifs
        volatility_regime = values.get('volatility_regime')
        atr_percentile = values.get('atr_percentile')
        
        if volatility_regime is not None:
            try:
                atr_pct = float(atr_percentile) if atr_percentile is not None else 50
                
                if signal_side == "BUY":
                    # BUY sur VWAP support nécessite volatilité contrôlée pour éviter faux rebonds
                    if volatility_regime == "low" and atr_pct < 30:
                        confidence_boost += 0.12
                        reason += f" + volatilité très faible idéale support ({atr_pct:.0f}%)"
                    elif volatility_regime == "normal" and 30 <= atr_pct <= 60:
                        confidence_boost += 0.08
                        reason += f" + volatilité normale support ({atr_pct:.0f}%)"
                    elif volatility_regime == "high" and atr_pct > 70:
                        if atr_pct > 90:  # Volatilité extrême
                            confidence_boost -= 0.12
                            reason += f" mais volatilité extrême support ({atr_pct:.0f}%)"
                        else:  # Volatilité élevée mais gérable
                            confidence_boost -= 0.06
                            reason += f" mais volatilité élevée support ({atr_pct:.0f}%)"
                    elif volatility_regime == "expanding":
                        confidence_boost -= 0.08  # Expansion défavorable aux supports
                        reason += " mais volatilité en expansion défavorable"
                        
                else:  # SELL
                    # SELL sur VWAP résistance peut bénéficier de volatilité modérée à élevée
                    if volatility_regime == "low" and atr_pct < 25:
                        confidence_boost += 0.06  # Résistance solide en low vol
                        reason += f" + volatilité faible résistance solide ({atr_pct:.0f}%)"
                    elif volatility_regime == "normal" and 25 <= atr_pct <= 70:
                        confidence_boost += 0.10
                        reason += f" + volatilité optimale résistance ({atr_pct:.0f}%)"
                    elif volatility_regime == "high" and atr_pct > 70:
                        if atr_pct > 85:  # Volatilité très élevée = continuation baissière
                            confidence_boost += 0.15
                            reason += f" + volatilité très élevée continuation ({atr_pct:.0f}%)"
                        else:  # Volatilité élevée favorable
                            confidence_boost += 0.12
                            reason += f" + volatilité élevée résistance ({atr_pct:.0f}%)"
                    elif volatility_regime == "expanding":
                        confidence_boost += 0.08  # Expansion favorable aux résistances
                        reason += " + volatilité expansion favorable résistance"
                        
            except (ValueError, TypeError):
                pass
            
        # CORRECTION MAGISTRALE: Market regime avec logique institutionnelle VWAP
        market_regime = values.get('market_regime')
        regime_strength = values.get('regime_strength')
        
        if market_regime is not None:
            try:
                regime_str = float(regime_strength) if regime_strength is not None else 0.5
                
                if signal_side == "BUY":
                    # BUY sur VWAP support : trending haussier > ranging > trending baissier
                    if market_regime == "trending":
                        if regime_str > 0.7:  # Trend très fort
                            confidence_boost += 0.18
                            reason += f" + trend très fort support VWAP ({regime_str:.2f})"
                        elif regime_str > 0.5:  # Trend modéré
                            confidence_boost += 0.14
                            reason += f" + trend fort support VWAP ({regime_str:.2f})"
                        else:  # Trend faible
                            confidence_boost += 0.08
                            reason += f" + trend faible support VWAP ({regime_str:.2f})"
                    elif market_regime == "ranging":
                        if regime_str > 0.6:  # Range bien défini
                            confidence_boost += 0.15
                            reason += f" + range fort rebond support ({regime_str:.2f})"
                        else:  # Range faible
                            confidence_boost += 0.10
                            reason += f" + range modéré support ({regime_str:.2f})"
                    elif market_regime == "choppy":
                        confidence_boost -= 0.05  # Marché chaotique défavorable
                        reason += " mais marché chaotique"
                        
                else:  # SELL
                    # SELL sur VWAP résistance : ranging > trending baissier > trending haussier
                    if market_regime == "ranging":
                        if regime_str > 0.7:  # Range très défini
                            confidence_boost += 0.20  # VWAP résistance excellent en ranging
                            reason += f" + range très fort rejet résistance ({regime_str:.2f})"
                        elif regime_str > 0.5:  # Range modéré
                            confidence_boost += 0.16
                            reason += f" + range fort résistance VWAP ({regime_str:.2f})"
                        else:  # Range faible
                            confidence_boost += 0.12
                            reason += f" + range modéré résistance ({regime_str:.2f})"
                    elif market_regime == "trending":
                        if regime_str > 0.6:  # Trend fort - résistance peut tenir
                            confidence_boost += 0.12
                            reason += f" + trend fort résistance VWAP ({regime_str:.2f})"
                        else:  # Trend faible
                            confidence_boost += 0.06
                            reason += f" + trend faible résistance ({regime_str:.2f})"
                    elif market_regime == "choppy":
                        confidence_boost += 0.08  # Chaos favorable aux résistances
                        reason += " + marché chaotique favorable résistance"
                        
            except (ValueError, TypeError):
                pass
            
        # Pattern detection
        pattern_detected = values.get('pattern_detected')
        pattern_confidence = values.get('pattern_confidence')
        if pattern_detected and pattern_confidence is not None:
            try:
                pattern_conf = float(pattern_confidence)
                if pattern_conf > 70:
                    confidence_boost += 0.08
                    reason += " + pattern détecté"
            except (ValueError, TypeError):
                pass
                
        # Confluence score global
        confluence_score = values.get('confluence_score')
        if confluence_score is not None:
            try:
                conf_val = float(confluence_score)
                if conf_val > 80:
                    confidence_boost += 0.12
                    reason += " + très haute confluence"
                elif conf_val > 60:
                    confidence_boost += 0.08
                    reason += " + haute confluence"
            except (ValueError, TypeError):
                pass
                
        confidence = self.calculate_confidence(base_confidence, 1.0 + confidence_boost)
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
                "vwap_level": vwap_level,
                "vwap_distance_pct": vwap_distance,
                "vwap_score": primary_analysis['score'] if primary_analysis else 0.0,
                "momentum_score": momentum_analysis['score'] if momentum_analysis else 0.0,
                "vwap_indicators": primary_analysis['indicators'] if primary_analysis else [],
                "momentum_indicators": momentum_analysis['indicators'] if momentum_analysis else [],
                "support_analysis": support_analysis if signal_side == "BUY" else None,
                "resistance_analysis": resistance_analysis if signal_side == "SELL" else None,
                "volume_ratio": values.get('volume_ratio'),
                "trend_alignment": values.get('trend_alignment'),
                "market_regime": values.get('market_regime'),
                "volatility_regime": values.get('volatility_regime'),
                "confluence_score": values.get('confluence_score'),
                "pattern_detected": values.get('pattern_detected'),
                "pattern_confidence": values.get('pattern_confidence')
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que tous les indicateurs requis sont présents."""
        required_indicators = [
            'vwap_10', 'volume_ratio', 'momentum_score'
        ]
        
        if not self.indicators:
            logger.warning(f"{self.name}: Aucun indicateur disponible")
            return False
            
        for indicator in required_indicators:
            if indicator not in self.indicators:
                logger.warning(f"{self.name}: Indicateur manquant: {indicator}")
                return False
            if self.indicators[indicator] is None:
                logger.warning(f"{self.name}: Indicateur null: {indicator}")
                return False
                
        # Vérifier données OHLCV
        if 'close' not in self.data or not self.data['close']:
            logger.warning(f"{self.name}: Données OHLCV manquantes")
            return False
            
        return True
