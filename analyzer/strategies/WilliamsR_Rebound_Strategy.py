"""
WilliamsR_Rebound_Strategy - Stratégie de rebound basée sur Williams %R.
Williams %R est un oscillateur de momentum qui mesure les niveaux de surachat/survente
et génère des signaux de rebound depuis les extrêmes.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class WilliamsR_Rebound_Strategy(BaseStrategy):
    """
    Stratégie utilisant Williams %R pour détecter les opportunités de rebound.
    
    Principe Williams %R :
    - Williams %R = (Plus Haut N - Close) / (Plus Haut N - Plus Bas N) * -100
    - Valeurs entre -100 et 0
    - Williams %R < -80 = survente (signal BUY potentiel)
    - Williams %R > -20 = surachat (signal SELL potentiel)
    - Plus sensible que RSI, excellent pour les rebonds courts
    
    Signaux générés:
    - BUY: Williams %R sort de zone survente (-80) vers le haut + confirmations
    - SELL: Williams %R sort de zone surachat (-20) vers le bas + confirmations
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        
        # Paramètres Williams %R
        self.oversold_threshold = -80.0        # Seuil survente
        self.overbought_threshold = -20.0      # Seuil surachat
        self.extreme_oversold_threshold = -90.0  # Survente extrême
        self.extreme_overbought_threshold = -10.0 # Surachat extrême
        
        # Paramètres rebond
        self.min_rebound_strength = 5.0        # Williams %R doit bouger ≥5 points
        self.rebound_confirmation_threshold = 10.0  # 10 points pour confirmation
        self.max_time_in_extreme = 5           # Max barres en zone extrême
        
        # Paramètres momentum et volume
        self.momentum_alignment_required = True  # Momentum doit confirmer
        self.min_momentum_threshold = 0.1        # Momentum minimum
        self.min_volume_confirmation = 1.2       # Volume ≥20% au-dessus normal
        
        # Paramètres confluence
        self.support_resistance_confluence = True  # Confluence S/R requise
        self.confluence_distance_threshold = 0.02  # 2% max du S/R pour confluence
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs pré-calculés."""
        return {
            # Williams %R principal
            'williams_r': self.indicators.get('williams_r'),
            
            # Autres oscillateurs (confluence)
            'rsi_14': self.indicators.get('rsi_14'),
            'rsi_21': self.indicators.get('rsi_21'),
            'stoch_k': self.indicators.get('stoch_k'),
            'stoch_d': self.indicators.get('stoch_d'),
            'stoch_rsi': self.indicators.get('stoch_rsi'),
            'stoch_fast_k': self.indicators.get('stoch_fast_k'),
            'stoch_fast_d': self.indicators.get('stoch_fast_d'),
            
            # Momentum indicators
            'momentum_score': self.indicators.get('momentum_score'),
            'momentum_10': self.indicators.get('momentum_10'),
            'roc_10': self.indicators.get('roc_10'),
            'roc_20': self.indicators.get('roc_20'),
            
            # MACD pour confirmation trend
            'macd_line': self.indicators.get('macd_line'),
            'macd_signal': self.indicators.get('macd_signal'),
            'macd_histogram': self.indicators.get('macd_histogram'),
            'macd_trend': self.indicators.get('macd_trend'),
            'macd_zero_cross': self.indicators.get('macd_zero_cross'),
            
            # Moyennes mobiles (support/résistance dynamique)
            'ema_12': self.indicators.get('ema_12'),
            'ema_26': self.indicators.get('ema_26'),
            'ema_50': self.indicators.get('ema_50'),
            'sma_20': self.indicators.get('sma_20'),
            'hull_20': self.indicators.get('hull_20'),
            
            # Support/Résistance statiques
            'nearest_support': self.indicators.get('nearest_support'),
            'nearest_resistance': self.indicators.get('nearest_resistance'),
            'support_strength': self.indicators.get('support_strength'),
            'resistance_strength': self.indicators.get('resistance_strength'),
            'support_levels': self.indicators.get('support_levels'),
            'resistance_levels': self.indicators.get('resistance_levels'),
            
            # Volume analysis
            'volume_ratio': self.indicators.get('volume_ratio'),
            'relative_volume': self.indicators.get('relative_volume'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            'volume_spike_multiplier': self.indicators.get('volume_spike_multiplier'),
            'trade_intensity': self.indicators.get('trade_intensity'),
            
            # Trend et direction
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'trend_alignment': self.indicators.get('trend_alignment'),
            'trend_angle': self.indicators.get('trend_angle'),
            
            # ADX pour force tendance
            'adx_14': self.indicators.get('adx_14'),
            'plus_di': self.indicators.get('plus_di'),
            'minus_di': self.indicators.get('minus_di'),
            
            # Bollinger Bands (contexte volatilité)
            'bb_upper': self.indicators.get('bb_upper'),
            'bb_middle': self.indicators.get('bb_middle'),
            'bb_lower': self.indicators.get('bb_lower'),
            'bb_position': self.indicators.get('bb_position'),
            'bb_width': self.indicators.get('bb_width'),
            
            # ATR et volatilité
            'atr_14': self.indicators.get('atr_14'),
            'atr_percentile': self.indicators.get('atr_percentile'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            
            # Market context
            'market_regime': self.indicators.get('market_regime'),
            'regime_strength': self.indicators.get('regime_strength'),
            'confluence_score': self.indicators.get('confluence_score'),
            'signal_strength': self.indicators.get('signal_strength'),
            'pattern_detected': self.indicators.get('pattern_detected'),
            'pattern_confidence': self.indicators.get('pattern_confidence')
        }
        
    def _detect_williamsR_rebound_buy(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Détecte un rebound haussier depuis zone survente Williams %R."""
        rebound_score = 0.0
        rebound_indicators = []
        
        williams_r = values.get('williams_r')
        if williams_r is None:
            return {'is_rebound': False, 'score': 0.0, 'indicators': []}
            
        try:
            williams_val = float(williams_r)
        except (ValueError, TypeError):
            return {'is_rebound': False, 'score': 0.0, 'indicators': []}
            
        # Vérifier si Williams %R est en phase de rebound depuis survente
        if williams_val <= self.oversold_threshold:
            # Encore en survente - pas de rebound
            return {
                'is_rebound': False,
                'score': 0.0,
                'indicators': [f'Williams %R encore en survente ({williams_val:.1f})'],
                'williams_value': williams_val
            }
            
        # Vérifier rebound depuis survente (Williams %R > -80 après avoir été < -80)
        if williams_val > self.oversold_threshold and williams_val < -50:
            # Zone de rebound depuis survente (entre -80 et -50)
            rebound_distance = abs(williams_val - self.oversold_threshold)
            
            if rebound_distance >= self.rebound_confirmation_threshold:
                rebound_score += 0.3
                rebound_indicators.append(f"Rebound fort depuis survente ({rebound_distance:.1f} points)")
            elif rebound_distance >= self.min_rebound_strength:
                rebound_score += 0.2
                rebound_indicators.append(f"Rebound modéré depuis survente ({rebound_distance:.1f} points)")
            else:
                rebound_score += 0.1
                rebound_indicators.append(f"Rebound faible depuis survente ({rebound_distance:.1f} points)")
                
            # Bonus si venait de zone survente extrême
            if williams_val > self.extreme_oversold_threshold:
                rebound_score += 0.15
                rebound_indicators.append("Rebound depuis survente extrême")
                
        else:
            # Williams %R pas dans zone de rebound appropriée
            return {
                'is_rebound': False,
                'score': 0.0,
                'indicators': [f'Williams %R pas en zone rebound BUY ({williams_val:.1f})'],
                'williams_value': williams_val
            }
            
        return {
            'is_rebound': rebound_score >= 0.2,
            'score': rebound_score,
            'indicators': rebound_indicators,
            'williams_value': williams_val,
            'rebound_type': 'bullish_from_oversold'
        }
        
    def _detect_williamsR_rebound_sell(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Détecte un rebound baissier depuis zone surachat Williams %R."""
        rebound_score = 0.0
        rebound_indicators = []
        
        williams_r = values.get('williams_r')
        if williams_r is None:
            return {'is_rebound': False, 'score': 0.0, 'indicators': []}
            
        try:
            williams_val = float(williams_r)
        except (ValueError, TypeError):
            return {'is_rebound': False, 'score': 0.0, 'indicators': []}
            
        # Vérifier si Williams %R est en phase de rebound depuis surachat
        if williams_val >= self.overbought_threshold:
            # Encore en surachat - pas de rebound
            return {
                'is_rebound': False,
                'score': 0.0,
                'indicators': [f'Williams %R encore en surachat ({williams_val:.1f})'],
                'williams_value': williams_val
            }
            
        # Vérifier rebound depuis surachat (Williams %R < -20 après avoir été > -20)  
        if williams_val < self.overbought_threshold and williams_val > -50:
            # Zone de rebound depuis surachat (entre -50 et -20)
            rebound_distance = abs(self.overbought_threshold - williams_val)
            
            if rebound_distance >= self.rebound_confirmation_threshold:
                rebound_score += 0.3
                rebound_indicators.append(f"Rebound fort depuis surachat ({rebound_distance:.1f} points)")
            elif rebound_distance >= self.min_rebound_strength:
                rebound_score += 0.2
                rebound_indicators.append(f"Rebound modéré depuis surachat ({rebound_distance:.1f} points)")
            else:
                rebound_score += 0.1
                rebound_indicators.append(f"Rebound faible depuis surachat ({rebound_distance:.1f} points)")
                
            # Bonus si venait de zone surachat extrême
            if williams_val < self.extreme_overbought_threshold:
                rebound_score += 0.15
                rebound_indicators.append("Rebound depuis surachat extrême")
                
        else:
            # Williams %R pas dans zone de rebound appropriée
            return {
                'is_rebound': False,
                'score': 0.0,
                'indicators': [f'Williams %R pas en zone rebound SELL ({williams_val:.1f})'],
                'williams_value': williams_val
            }
            
        return {
            'is_rebound': rebound_score >= 0.2,
            'score': rebound_score,
            'indicators': rebound_indicators,
            'williams_value': williams_val,
            'rebound_type': 'bearish_from_overbought'
        }
        
    def _detect_oscillator_confluence(self, values: Dict[str, Any], signal_direction: str) -> Dict[str, Any]:
        """Détecte la confluence avec autres oscillateurs."""
        confluence_score = 0.0
        confluence_indicators = []
        
        # RSI confluence
        rsi_14 = values.get('rsi_14')
        if rsi_14 is not None:
            try:
                rsi_val = float(rsi_14)
                
                if signal_direction == "BUY":
                    if 30 <= rsi_val <= 50:  # RSI sortant de survente
                        confluence_score += 0.15
                        confluence_indicators.append(f"RSI sortant survente ({rsi_val:.1f})")
                    elif 20 <= rsi_val <= 40:  # RSI encore en zone basse
                        confluence_score += 0.1
                        confluence_indicators.append(f"RSI zone basse ({rsi_val:.1f})")
                elif signal_direction == "SELL":
                    if 50 <= rsi_val <= 70:  # RSI sortant de surachat
                        confluence_score += 0.15
                        confluence_indicators.append(f"RSI sortant surachat ({rsi_val:.1f})")
                    elif 60 <= rsi_val <= 80:  # RSI encore en zone haute
                        confluence_score += 0.1
                        confluence_indicators.append(f"RSI zone haute ({rsi_val:.1f})")
                        
            except (ValueError, TypeError):
                pass
                
        # Stochastic confluence
        stoch_k = values.get('stoch_k')
        stoch_d = values.get('stoch_d')
        if stoch_k is not None and stoch_d is not None:
            try:
                stoch_k_val = float(stoch_k)
                stoch_d_val = float(stoch_d)
                
                if signal_direction == "BUY":
                    if stoch_k_val < 30 and stoch_k_val > stoch_d_val:  # Stoch cross up from oversold
                        confluence_score += 0.12
                        confluence_indicators.append("Stochastic cross haussier")
                    elif stoch_k_val < 40:
                        confluence_score += 0.08
                        confluence_indicators.append(f"Stochastic bas ({stoch_k_val:.1f})")
                elif signal_direction == "SELL":
                    if stoch_k_val > 70 and stoch_k_val < stoch_d_val:  # Stoch cross down from overbought
                        confluence_score += 0.12
                        confluence_indicators.append("Stochastic cross baissier")
                    elif stoch_k_val > 60:
                        confluence_score += 0.08
                        confluence_indicators.append(f"Stochastic haut ({stoch_k_val:.1f})")
                        
            except (ValueError, TypeError):
                pass
                
        # CCI approximation (non disponible direct, utiliser momentum)
        momentum_10 = values.get('momentum_10')
        if momentum_10 is not None:
            try:
                momentum_val = float(momentum_10)
                
                if signal_direction == "BUY" and momentum_val > 100:  # Momentum positif après baisse
                    confluence_score += 0.1
                    confluence_indicators.append("Momentum rebond haussier")
                elif signal_direction == "SELL" and momentum_val < 100:  # Momentum négatif après hausse
                    confluence_score += 0.1
                    confluence_indicators.append("Momentum rebond baissier")
                    
            except (ValueError, TypeError):
                pass
                
        return {
            'is_confluent': confluence_score >= 0.15,
            'score': confluence_score,
            'indicators': confluence_indicators
        }
        
    def _detect_support_resistance_confluence(self, values: Dict[str, Any], current_price: float, signal_direction: str) -> Dict[str, Any]:
        """Détecte la confluence avec niveaux de support/résistance."""
        sr_score = 0.0
        sr_indicators = []
        
        if signal_direction == "BUY":
            # Rechercher confluence avec support
            nearest_support = values.get('nearest_support')
            if nearest_support is not None:
                try:
                    support_level = float(nearest_support)
                    distance_to_support = abs(current_price - support_level) / current_price
                    
                    if distance_to_support <= self.confluence_distance_threshold:
                        sr_score += 0.2
                        sr_indicators.append(f"Proche support {support_level:.2f} ({distance_to_support*100:.1f}%)")
                        
                        # Bonus selon force support
                        support_strength = values.get('support_strength')
                        if support_strength is not None:
                            try:
                                if isinstance(support_strength, str):
                                    strength_map = {'WEAK': 0.2, 'MODERATE': 0.5, 'STRONG': 0.8, 'MAJOR': 1.0}
                                    strength_val = strength_map.get(support_strength.upper(), 0.5)
                                else:
                                    strength_val = float(support_strength)
                                    
                                if strength_val >= 0.8:
                                    sr_score += 0.15
                                    sr_indicators.append(f"Support très fort ({strength_val:.2f})")
                                elif strength_val >= 0.6:
                                    sr_score += 0.1
                                    sr_indicators.append(f"Support fort ({strength_val:.2f})")
                            except (ValueError, TypeError):
                                pass
                except (ValueError, TypeError):
                    pass
                    
        elif signal_direction == "SELL":
            # Rechercher confluence avec résistance
            nearest_resistance = values.get('nearest_resistance')
            if nearest_resistance is not None:
                try:
                    resistance_level = float(nearest_resistance)
                    distance_to_resistance = abs(current_price - resistance_level) / current_price
                    
                    if distance_to_resistance <= self.confluence_distance_threshold:
                        sr_score += 0.2
                        sr_indicators.append(f"Proche résistance {resistance_level:.2f} ({distance_to_resistance*100:.1f}%)")
                        
                        # Bonus selon force résistance
                        resistance_strength = values.get('resistance_strength')
                        if resistance_strength is not None:
                            try:
                                if isinstance(resistance_strength, str):
                                    strength_map = {'WEAK': 0.2, 'MODERATE': 0.5, 'STRONG': 0.8, 'MAJOR': 1.0}
                                    strength_val = strength_map.get(resistance_strength.upper(), 0.5)
                                else:
                                    strength_val = float(resistance_strength)
                                    
                                if strength_val >= 0.8:
                                    sr_score += 0.15
                                    sr_indicators.append(f"Résistance très forte ({strength_val:.2f})")
                                elif strength_val >= 0.6:
                                    sr_score += 0.1
                                    sr_indicators.append(f"Résistance forte ({strength_val:.2f})")
                            except (ValueError, TypeError):
                                pass
                except (ValueError, TypeError):
                    pass
                    
        # EMA confluence comme support/résistance dynamique
        ema_50 = values.get('ema_50')
        if ema_50 is not None:
            try:
                ema_val = float(ema_50)
                distance_to_ema = abs(current_price - ema_val) / current_price
                
                if distance_to_ema <= 0.01:  # 1% de l'EMA50
                    if signal_direction == "BUY" and current_price >= ema_val:
                        sr_score += 0.1
                        sr_indicators.append("EMA50 support dynamique")
                    elif signal_direction == "SELL" and current_price <= ema_val:
                        sr_score += 0.1
                        sr_indicators.append("EMA50 résistance dynamique")
            except (ValueError, TypeError):
                pass
                
        return {
            'is_confluent': sr_score >= 0.15,
            'score': sr_score,
            'indicators': sr_indicators
        }
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur Williams %R rebound.
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
        if 'ohlcv' in self.data and self.data['ohlcv']:
            current_price = float(self.data['ohlcv'][-1]['close'])
            
        if current_price is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Prix actuel non disponible",
                "metadata": {"strategy": self.name}
            }
            
        # Analyser rebound haussier (BUY)
        buy_rebound = self._detect_williamsR_rebound_buy(values)
        
        # Analyser rebound baissier (SELL)
        sell_rebound = self._detect_williamsR_rebound_sell(values)
        
        # Déterminer signal principal
        signal_side = None
        primary_rebound = None
        
        if buy_rebound['is_rebound'] and sell_rebound['is_rebound']:
            # Conflit - prendre le score le plus élevé
            if buy_rebound['score'] > sell_rebound['score']:
                signal_side = "BUY"
                primary_rebound = buy_rebound
            else:
                signal_side = "SELL"
                primary_rebound = sell_rebound
        elif buy_rebound['is_rebound']:
            signal_side = "BUY"
            primary_rebound = buy_rebound
        elif sell_rebound['is_rebound']:
            signal_side = "SELL"
            primary_rebound = sell_rebound
            
        if signal_side is None:
            # Diagnostic conditions non remplies
            williams_val = values.get('williams_r')
            missing_conditions = []
            
            if buy_rebound['score'] < 0.2:
                missing_conditions.append(f"Rebound BUY faible (score: {buy_rebound['score']:.2f})")
            if sell_rebound['score'] < 0.2:
                missing_conditions.append(f"Rebound SELL faible (score: {sell_rebound['score']:.2f})")
                
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Williams %R ({williams_val:.1f}) - {'; '.join(missing_conditions[:2])}",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "williams_r": williams_val,
                    "buy_rebound_score": buy_rebound['score'],
                    "sell_rebound_score": sell_rebound['score']
                }
            }
            
        # Vérifier confluences
        oscillator_confluence = self._detect_oscillator_confluence(values, signal_side)
        sr_confluence = self._detect_support_resistance_confluence(values, current_price, signal_side)
        
        # Construire signal final
        base_confidence = 0.5
        confidence_boost = 0.0
        
        # Score rebound Williams %R
        confidence_boost += primary_rebound['score'] * 0.4
        
        # Score confluences oscillateurs
        confidence_boost += oscillator_confluence['score'] * 0.3
        
        # Score confluence support/résistance
        confidence_boost += sr_confluence['score'] * 0.3
        
        # Construire raison
        williams_val = primary_rebound['williams_value']
        rebound_type = primary_rebound['rebound_type']
        
        reason = f"Williams %R {williams_val:.1f} - {primary_rebound['indicators'][0]}"
        
        if oscillator_confluence['indicators']:
            reason += f" + {oscillator_confluence['indicators'][0]}"
            
        if sr_confluence['indicators']:
            reason += f" + {sr_confluence['indicators'][0]}"
            
        # Confirmations supplémentaires
        
        # Momentum alignment
        momentum_score_val = values.get('momentum_score')
        if momentum_score_val is not None:
            try:
                momentum = float(momentum_score_val)
                
                if signal_side == "BUY" and momentum >= self.min_momentum_threshold:
                    confidence_boost += 0.1
                    reason += " + momentum haussier"
                elif signal_side == "SELL" and momentum <= -self.min_momentum_threshold:
                    confidence_boost += 0.1
                    reason += " + momentum baissier"
            except (ValueError, TypeError):
                pass
                
        # Volume confirmation
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= self.min_volume_confirmation:
                    confidence_boost += 0.1
                    reason += f" + volume ({vol_ratio:.1f}x)"
            except (ValueError, TypeError):
                pass
                
        # MACD confirmation
        macd_line = values.get('macd_line')
        macd_signal = values.get('macd_signal')
        if macd_line is not None and macd_signal is not None:
            try:
                macd_val = float(macd_line)
                macd_sig = float(macd_signal)
                
                if signal_side == "BUY" and macd_val > macd_sig:
                    confidence_boost += 0.08
                    reason += " + MACD haussier"
                elif signal_side == "SELL" and macd_val < macd_sig:
                    confidence_boost += 0.08
                    reason += " + MACD baissier"
            except (ValueError, TypeError):
                pass
                
        # Trend alignment
        directional_bias = values.get('directional_bias')
        if directional_bias:
            if (signal_side == "BUY" and directional_bias == 'bullish') or \
               (signal_side == "SELL" and directional_bias == 'bearish'):
                confidence_boost += 0.08
                reason += f" + bias {directional_bias}"
                
        # ADX context (rebonds meilleurs avec ADX modéré)
        adx_14 = values.get('adx_14')
        if adx_14 is not None:
            try:
                adx_val = float(adx_14)
                if 20 <= adx_val <= 35:  # ADX modéré = rebonds plus probables
                    confidence_boost += 0.05
                    reason += " + ADX modéré"
                elif adx_val > 50:  # Trend très fort = rebonds difficiles
                    confidence_boost -= 0.08
                    reason += " mais ADX très fort"
            except (ValueError, TypeError):
                pass
                
        # Bollinger Bands context
        bb_position = values.get('bb_position')
        if bb_position is not None:
            try:
                bb_pos = float(bb_position)
                if signal_side == "BUY" and bb_pos <= 0.2:  # Prix près BB basse
                    confidence_boost += 0.05
                    reason += " + BB position basse"
                elif signal_side == "SELL" and bb_pos >= 0.8:  # Prix près BB haute
                    confidence_boost += 0.05
                    reason += " + BB position haute"
            except (ValueError, TypeError):
                pass
                
        # Market regime context
        market_regime = values.get('market_regime')
        if market_regime == "ranging":
            confidence_boost += 0.08  # Williams %R excellent en ranging
            reason += " (marché ranging)"
        elif market_regime == "trending":
            confidence_boost += 0.03  # Rebonds possibles mais plus risqués
            reason += " (marché trending)"
            
        # Volatility context
        volatility_regime = values.get('volatility_regime')
        if volatility_regime == "normal":
            confidence_boost += 0.05
            reason += " + volatilité normale"
        elif volatility_regime == "high":
            confidence_boost += 0.03  # Haute volatilité = rebonds plus forts mais plus risqués
            reason += " (volatilité élevée)"
            
        # Confluence score global
        confluence_score_global = values.get('confluence_score')
        if confluence_score_global is not None:
            try:
                conf_val = float(confluence_score_global)
                if conf_val > 0.8:
                    confidence_boost += 0.1
                    reason += " + très haute confluence"
                elif conf_val > 0.6:
                    confidence_boost += 0.05
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
                "williams_r": williams_val,
                "rebound_type": rebound_type,
                "rebound_score": primary_rebound['score'],
                "rebound_indicators": primary_rebound['indicators'],
                "oscillator_confluence_score": oscillator_confluence['score'],
                "oscillator_confluence_indicators": oscillator_confluence['indicators'],
                "sr_confluence_score": sr_confluence['score'],
                "sr_confluence_indicators": sr_confluence['indicators'],
                "buy_rebound_analysis": buy_rebound if signal_side == "BUY" else None,
                "sell_rebound_analysis": sell_rebound if signal_side == "SELL" else None,
                "volume_ratio": values.get('volume_ratio'),
                "momentum_score": values.get('momentum_score'),
                "directional_bias": values.get('directional_bias'),
                "market_regime": values.get('market_regime'),
                "volatility_regime": values.get('volatility_regime'),
                "confluence_score": values.get('confluence_score')
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que tous les indicateurs requis sont présents."""
        required_indicators = [
            'williams_r'
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
        if 'ohlcv' not in self.data or not self.data['ohlcv']:
            logger.warning(f"{self.name}: Données OHLCV manquantes")
            return False
            
        return True
