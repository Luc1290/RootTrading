"""
ZScore_Extreme_Reversal_Strategy - Stratégie de reversal basée sur les extrêmes Z-Score.
Le Z-Score mesure combien d'écarts-types le prix s'éloigne de sa moyenne - les valeurs extrêmes 
indiquent des surachat/survente propices aux reversals.
"""

from typing import Dict, Any, Optional, List, cast
from .base_strategy import BaseStrategy
import logging
import math

logger = logging.getLogger(__name__)


class ZScore_Extreme_Reversal_Strategy(BaseStrategy):
    """
    Stratégie utilisant les Z-Scores extrêmes pour détecter les opportunités de reversal.
    
    Principe Z-Score :
    - Z-Score = (Prix - Moyenne) / Écart-type
    - Z > +2 = surachat potentiel (signal SELL)  
    - Z < -2 = survente potentielle (signal BUY)
    - Plus le Z-Score est extrême, plus le reversal est probable
    
    Le Z-Score n'étant pas directement disponible, nous le simulons avec :
    - Bollinger Bands position (proxy Z-Score)
    - RSI extrêmes (surachat/survente)
    - Williams %R (momentum reversal)
    - ATR percentile (volatilité context)
    
    Signaux générés:
    - BUY: Z-Score extrême négatif (survente) + confirmations reversal haussier
    - SELL: Z-Score extrême positif (surachat) + confirmations reversal baissier
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        
        # Paramètres Z-Score simulé
        self.extreme_zscore_threshold = 2.0      # Z-Score > 2 pour extrême
        self.very_extreme_zscore_threshold = 2.5  # Z-Score > 2.5 pour très extrême
        self.ultra_extreme_zscore_threshold = 3.0 # Z-Score > 3.0 pour ultra extrême
        
        # Paramètres Bollinger Bands (proxy Z-Score)
        self.bb_extreme_position_threshold = 95   # Position BB > 95% = extrême
        self.bb_very_extreme_threshold = 98       # Position BB > 98% = très extrême
        self.bb_squeeze_bonus = 0.15                # Bonus si BB squeeze avant expansion
        
        # Paramètres RSI reversal
        self.rsi_oversold_threshold = 25            # RSI < 25 = survente extrême
        self.rsi_overbought_threshold = 75          # RSI > 75 = surachat extrême
        self.rsi_very_extreme_threshold = 15        # RSI < 15 ou > 85 = très extrême
        
        # Paramètres Williams %R
        self.williams_oversold_threshold = -80      # Williams %R < -80 = survente
        self.williams_overbought_threshold = -20    # Williams %R > -20 = surachat
        
        # Paramètres volume et momentum
        self.min_volume_confirmation = 1.1          # Volume minimum pour reversal
        self.momentum_divergence_required = True    # Divergence momentum/prix requise
        self.min_volatility_percentile = 60       # Volatilité minimum pour signal fort
        
        # Paramètres temporels
        self.max_extreme_duration = 5               # Max barres en zone extrême
        self.reversal_confirmation_bars = 2         # Barres confirmation reversal
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs pré-calculés."""
        return {
            # Bollinger Bands (proxy Z-Score principal)
            'bb_upper': self.indicators.get('bb_upper'),
            'bb_middle': self.indicators.get('bb_middle'),
            'bb_lower': self.indicators.get('bb_lower'),
            'bb_position': self.indicators.get('bb_position'),
            'bb_width': self.indicators.get('bb_width'),
            'bb_squeeze': self.indicators.get('bb_squeeze'),
            'bb_expansion': self.indicators.get('bb_expansion'),
            'bb_breakout_direction': self.indicators.get('bb_breakout_direction'),
            
            # Oscillateurs extremes (surachat/survente)
            'rsi_14': self.indicators.get('rsi_14'),
            'rsi_21': self.indicators.get('rsi_21'),
            'williams_r': self.indicators.get('williams_r'),
            'stoch_k': self.indicators.get('stoch_k'),
            'stoch_d': self.indicators.get('stoch_d'),
            'stoch_rsi': self.indicators.get('stoch_rsi'),
            
            # Momentum et divergences
            'momentum_score': self.indicators.get('momentum_score'),
            'momentum_10': self.indicators.get('momentum_10'),
            'roc_10': self.indicators.get('roc_10'),
            'roc_20': self.indicators.get('roc_20'),
            
            # MACD pour confirmation reversal
            'macd_line': self.indicators.get('macd_line'),
            'macd_signal': self.indicators.get('macd_signal'),
            'macd_histogram': self.indicators.get('macd_histogram'),
            'macd_trend': self.indicators.get('macd_trend'),
            'macd_zero_cross': self.indicators.get('macd_zero_cross'),
            
            # Volatilité context (crucial pour Z-Score)
            'atr_14': self.indicators.get('atr_14'),
            'atr_percentile': self.indicators.get('atr_percentile'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            'natr': self.indicators.get('natr'),
            
            # Moyennes mobiles (support/résistance dynamique)
            'ema_12': self.indicators.get('ema_12'),
            'ema_26': self.indicators.get('ema_26'),
            'ema_50': self.indicators.get('ema_50'),
            'sma_20': self.indicators.get('sma_20'),
            'sma_50': self.indicators.get('sma_50'),
            'hull_20': self.indicators.get('hull_20'),
            
            # Volume analysis
            'volume_ratio': self.indicators.get('volume_ratio'),
            'relative_volume': self.indicators.get('relative_volume'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            'volume_spike_multiplier': self.indicators.get('volume_spike_multiplier'),
            'trade_intensity': self.indicators.get('trade_intensity'),
            
            # ADX pour force tendance
            'adx_14': self.indicators.get('adx_14'),
            'plus_di': self.indicators.get('plus_di'),
            'minus_di': self.indicators.get('minus_di'),
            
            # Support/Résistance
            'nearest_support': self.indicators.get('nearest_support'),
            'nearest_resistance': self.indicators.get('nearest_resistance'),
            'support_strength': self.indicators.get('support_strength'),
            'resistance_strength': self.indicators.get('resistance_strength'),
            
            # Market context
            'market_regime': self.indicators.get('market_regime'),
            'regime_strength': self.indicators.get('regime_strength'),
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'trend_alignment': self.indicators.get('trend_alignment'),
            
            # Pattern et confluence
            'pattern_detected': self.indicators.get('pattern_detected'),
            'pattern_confidence': self.indicators.get('pattern_confidence'),
            'confluence_score': self.indicators.get('confluence_score'),
            'signal_strength': self.indicators.get('signal_strength'),
            'anomaly_detected': self.indicators.get('anomaly_detected')
        }
        
    def _calculate_zscore_proxy(self, values: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Calcule une approximation du Z-Score basée sur Bollinger Bands et oscillateurs."""
        zscore_data: Dict[str, Any] = {
            'zscore_value': None,
            'zscore_direction': None,
            'zscore_strength': 'weak',
            'zscore_components': []
        }
        
        # Méthode 1: Bollinger Bands position (meilleur proxy Z-Score)
        bb_position = values.get('bb_position')
        if bb_position is not None:
            try:
                bb_pos = float(bb_position)
                # Convertir position BB (0-1) vers approximation Z-Score (-3 à +3)
                # BB position 0.5 = Z-Score 0, BB position 1.0 = Z-Score ~3
                zscore_from_bb = (bb_pos - 0.5) * 6  # Approximation linéaire
                
                zscore_data['zscore_value'] = zscore_from_bb
                cast(List[str], zscore_data['zscore_components']).append(f"BB position ({bb_pos:.3f})")
                
                # Direction et force selon valeur
                abs_zscore = abs(zscore_from_bb)
                if abs_zscore >= self.ultra_extreme_zscore_threshold:
                    zscore_data['zscore_strength'] = 'ultra_extreme'
                elif abs_zscore >= self.very_extreme_zscore_threshold:
                    zscore_data['zscore_strength'] = 'very_extreme'
                elif abs_zscore >= self.extreme_zscore_threshold:
                    zscore_data['zscore_strength'] = 'extreme'
                else:
                    zscore_data['zscore_strength'] = 'normal'
                    
                # Direction
                if zscore_from_bb > self.extreme_zscore_threshold:
                    zscore_data['zscore_direction'] = 'extreme_positive'  # Surachat
                elif zscore_from_bb < -self.extreme_zscore_threshold:
                    zscore_data['zscore_direction'] = 'extreme_negative'  # Survente
                else:
                    zscore_data['zscore_direction'] = 'normal'
                    
            except (ValueError, TypeError):
                pass
                
        # Méthode 2: RSI comme confirmation Z-Score
        rsi_14 = values.get('rsi_14')
        if rsi_14 is not None:
            try:
                rsi_val = float(rsi_14)
                
                # RSI extrêmes renforcent le Z-Score
                if rsi_val <= self.rsi_very_extreme_threshold:
                    cast(List[str], zscore_data['zscore_components']).append(f"RSI très extrême ({rsi_val:.1f})")
                    if zscore_data['zscore_direction'] == 'extreme_negative':
                        zscore_data['zscore_strength'] = 'ultra_extreme'
                elif rsi_val <= self.rsi_oversold_threshold:
                    cast(List[str], zscore_data['zscore_components']).append(f"RSI survente ({rsi_val:.1f})")
                elif rsi_val >= 100 - self.rsi_very_extreme_threshold:
                    cast(List[str], zscore_data['zscore_components']).append(f"RSI très extrême ({rsi_val:.1f})")
                    if zscore_data['zscore_direction'] == 'extreme_positive':
                        zscore_data['zscore_strength'] = 'ultra_extreme'
                elif rsi_val >= self.rsi_overbought_threshold:
                    cast(List[str], zscore_data['zscore_components']).append(f"RSI surachat ({rsi_val:.1f})")
                    
            except (ValueError, TypeError):
                pass
                
        # Méthode 3: Williams %R confirmation
        williams_r = values.get('williams_r')
        if williams_r is not None:
            try:
                williams_val = float(williams_r)
                
                if williams_val <= -90:  # Très extrême
                    cast(List[str], zscore_data['zscore_components']).append(f"Williams %R très extrême ({williams_val:.1f})")
                elif williams_val <= self.williams_oversold_threshold:
                    cast(List[str], zscore_data['zscore_components']).append(f"Williams %R survente ({williams_val:.1f})")
                elif williams_val >= -10:  # Très extrême haut
                    cast(List[str], zscore_data['zscore_components']).append(f"Williams %R très extrême ({williams_val:.1f})")
                elif williams_val >= self.williams_overbought_threshold:
                    cast(List[str], zscore_data['zscore_components']).append(f"Williams %R surachat ({williams_val:.1f})")
                    
            except (ValueError, TypeError):
                pass
                
        return zscore_data
        
    def _detect_reversal_signals(self, values: Dict[str, Any], zscore_data: Dict[str, Any]) -> Dict[str, Any]:
        """Détecte les signaux de reversal selon la direction Z-Score."""
        reversal_score = 0.0
        reversal_indicators = []
        reversal_direction = None
        
        zscore_direction = zscore_data['zscore_direction']
        zscore_strength = zscore_data['zscore_strength']
        
        if zscore_direction == 'extreme_negative':
            # Survente extrême -> signal BUY potentiel
            reversal_direction = 'bullish'
            
            # Score base selon force Z-Score
            if zscore_strength == 'ultra_extreme':
                reversal_score += 0.4
                reversal_indicators.append("Z-Score ultra extrême négatif")
            elif zscore_strength == 'very_extreme':
                reversal_score += 0.35
                reversal_indicators.append("Z-Score très extrême négatif")
            elif zscore_strength == 'extreme':
                reversal_score += 0.3
                reversal_indicators.append("Z-Score extrême négatif")
                
        elif zscore_direction == 'extreme_positive':
            # Surachat extrême -> signal SELL potentiel
            reversal_direction = 'bearish'
            
            # Score base selon force Z-Score
            if zscore_strength == 'ultra_extreme':
                reversal_score += 0.4
                reversal_indicators.append("Z-Score ultra extrême positif")
            elif zscore_strength == 'very_extreme':
                reversal_score += 0.35
                reversal_indicators.append("Z-Score très extrême positif")
            elif zscore_strength == 'extreme':
                reversal_score += 0.3
                reversal_indicators.append("Z-Score extrême positif")
                
        if reversal_direction is None:
            return {'is_reversal': False, 'direction': None, 'score': 0.0, 'indicators': []}
            
        # Confirmation avec momentum divergence
        momentum_score = values.get('momentum_score')
        if momentum_score is not None:
            try:
                momentum_val = float(momentum_score)
                
                # Divergence momentum/prix favorable au reversal (format 0-100, 50=neutre)
                # Reversal bullish : prix en survente mais momentum pas trop négatif (divergence)
                if reversal_direction == 'bullish' and momentum_val > 40:  # Momentum > 40 = moins bearish que prix
                    reversal_score += 0.15
                    reversal_indicators.append(f"Divergence momentum haussière ({momentum_val:.1f})")
                # Reversal bearish : prix en surachat mais momentum pas trop positif (divergence)
                elif reversal_direction == 'bearish' and momentum_val < 60:  # Momentum < 60 = moins bullish que prix
                    reversal_score += 0.15
                    reversal_indicators.append(f"Divergence momentum baissière ({momentum_val:.1f})")
                    
            except (ValueError, TypeError):
                pass
                
        # MACD confirmation reversal
        macd_histogram = values.get('macd_histogram')
        if macd_histogram is not None:
            try:
                macd_hist = float(macd_histogram)
                
                # MACD histogram changing direction = early reversal signal
                if reversal_direction == 'bullish' and macd_hist > -0.001:  # MACD hist turning positive
                    reversal_score += 0.12
                    reversal_indicators.append("MACD histogram tournant positif")
                elif reversal_direction == 'bearish' and macd_hist < 0.001:  # MACD hist turning negative
                    reversal_score += 0.12
                    reversal_indicators.append("MACD histogram tournant négatif")
                    
            except (ValueError, TypeError):
                pass
                
        # Stochastic confirmation
        stoch_k = values.get('stoch_k')
        stoch_d = values.get('stoch_d')
        if stoch_k is not None and stoch_d is not None:
            try:
                stoch_k_val = float(stoch_k)
                stoch_d_val = float(stoch_d)
                
                if reversal_direction == 'bullish' and stoch_k_val < 25 and stoch_k_val > stoch_d_val:
                    reversal_score += 0.1
                    reversal_indicators.append("Stochastic reversal haussier")
                elif reversal_direction == 'bearish' and stoch_k_val > 75 and stoch_k_val < stoch_d_val:
                    reversal_score += 0.1
                    reversal_indicators.append("Stochastic reversal baissier")
                    
            except (ValueError, TypeError):
                pass
                
        return {
            'is_reversal': reversal_score >= 0.4,
            'direction': reversal_direction,
            'score': reversal_score,
            'indicators': reversal_indicators
        }
        
    def _detect_volatility_context(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Détecte le contexte de volatilité favorable aux reversals Z-Score."""
        volatility_score = 0.0
        volatility_indicators = []
        
        # ATR percentile (volatilité élevée = Z-Score plus significatif)
        atr_percentile = values.get('atr_percentile')
        if atr_percentile is not None:
            try:
                atr_perc = float(atr_percentile)
                if atr_perc >= 0.8:  # 80ème percentile et plus
                    volatility_score += 0.2
                    volatility_indicators.append(f"Volatilité très élevée ({atr_perc:.2f})")
                elif atr_perc >= self.min_volatility_percentile:
                    volatility_score += 0.15
                    volatility_indicators.append(f"Volatilité élevée ({atr_perc:.2f})")
                elif atr_perc < 0.3:  # Volatilité trop faible
                    volatility_score -= 0.1
                    volatility_indicators.append(f"Volatilité faible ({atr_perc:.2f})")
            except (ValueError, TypeError):
                pass
                
        # BB squeeze/expansion cycle
        bb_squeeze = values.get('bb_squeeze')
        bb_expansion = values.get('bb_expansion')
        
        if bb_squeeze is not None:
            try:
                if bool(bb_squeeze):  # Squeeze en cours
                    volatility_score += self.bb_squeeze_bonus
                    volatility_indicators.append("BB squeeze - expansion attendue")
            except (ValueError, TypeError):
                pass
                
        if bb_expansion is not None:
            try:
                if bool(bb_expansion):  # Expansion en cours
                    volatility_score += 0.1
                    volatility_indicators.append("BB expansion - volatilité élevée")
            except (ValueError, TypeError):
                pass
                
        # Volatility regime
        volatility_regime = values.get('volatility_regime')
        if volatility_regime == "high":
            volatility_score += 0.1
            volatility_indicators.append("Régime haute volatilité")
        elif volatility_regime == "expanding":
            volatility_score += 0.15
            volatility_indicators.append("Volatilité en expansion")
        elif volatility_regime == "low":
            volatility_score -= 0.05
            volatility_indicators.append("Régime faible volatilité")
            
        return {
            'is_favorable': volatility_score >= 0.1,
            'score': volatility_score,
            'indicators': volatility_indicators
        }
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur les Z-Scores extrêmes et reversals.
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
            
        # Calculer Z-Score proxy
        zscore_data = self._calculate_zscore_proxy(values, current_price)
        
        if zscore_data['zscore_direction'] == 'normal':
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Z-Score non extrême ({zscore_data['zscore_value']:.2f})" if zscore_data['zscore_value'] else "Z-Score non extrême",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "zscore_value": zscore_data['zscore_value'],
                    "zscore_strength": zscore_data['zscore_strength']
                }
            }
            
        # Détecter signaux de reversal
        reversal_analysis = self._detect_reversal_signals(values, zscore_data)
        
        if not reversal_analysis['is_reversal']:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Z-Score extrême mais reversal non confirmé (score: {reversal_analysis['score']:.2f})",
                "metadata": {
                    "strategy": self.name,
                    "zscore_direction": zscore_data['zscore_direction'],
                    "reversal_score": reversal_analysis['score']
                }
            }
            
        # Vérifier contexte volatilité
        volatility_analysis = self._detect_volatility_context(values)
        
        # Déterminer signal side
        reversal_direction = reversal_analysis['direction']
        if reversal_direction == 'bullish':
            signal_side = "BUY"
        elif reversal_direction == 'bearish':
            signal_side = "SELL"
        else:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Direction reversal indéterminée",
                "metadata": {"strategy": self.name}
            }
            
        # Construire signal final
        base_confidence = 0.6  # Base élevée pour Z-Score extrême
        confidence_boost = 0.0
        
        # Score Z-Score strength
        if zscore_data['zscore_strength'] == 'ultra_extreme':
            confidence_boost += 0.3
        elif zscore_data['zscore_strength'] == 'very_extreme':
            confidence_boost += 0.25
        elif zscore_data['zscore_strength'] == 'extreme':
            confidence_boost += 0.2
            
        # Score reversal confirmations
        confidence_boost += reversal_analysis['score'] * 0.4
        
        # Score volatilité
        confidence_boost += volatility_analysis['score'] * 0.3
        
        # Construire raison
        zscore_val = zscore_data['zscore_value']
        reason = f"Z-Score {zscore_data['zscore_strength']} ({zscore_val:.2f})" if zscore_val else f"Z-Score {zscore_data['zscore_strength']}"
        reason += f" reversal {reversal_direction}"
        
        if reversal_analysis['indicators']:
            reason += f": {reversal_analysis['indicators'][0]}"
            
        if volatility_analysis['indicators']:
            reason += f" + {volatility_analysis['indicators'][0]}"
            
        # Confirmations supplémentaires
        
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
                
        # Support/Resistance confluence
        if signal_side == "BUY":
            nearest_support = values.get('nearest_support')
            if nearest_support is not None:
                try:
                    support_level = float(nearest_support)
                    distance_to_support = abs(current_price - support_level) / current_price
                    if distance_to_support <= 0.02:  # Près support
                        confidence_boost += 0.1
                        reason += " + près support"
                except (ValueError, TypeError):
                    pass
        else:  # SELL
            nearest_resistance = values.get('nearest_resistance')
            if nearest_resistance is not None:
                try:
                    resistance_level = float(nearest_resistance)
                    distance_to_resistance = abs(current_price - resistance_level) / current_price
                    if distance_to_resistance <= 0.02:  # Près résistance
                        confidence_boost += 0.1
                        reason += " + près résistance"
                except (ValueError, TypeError):
                    pass
                    
        # ADX pour force (reversal meilleur avec ADX faible)
        adx_14 = values.get('adx_14')
        if adx_14 is not None:
            try:
                adx_val = float(adx_14)
                if adx_val < 25:  # Trend faible = reversal plus probable
                    confidence_boost += 0.08
                    reason += " + ADX faible favorable"
                elif adx_val > 40:  # Trend très forte = reversal difficile
                    confidence_boost -= 0.1
                    reason += " mais ADX très fort"
            except (ValueError, TypeError):
                pass
                
        # Market regime context
        market_regime = values.get('market_regime')
        if market_regime == "RANGING":
            confidence_boost += 0.08  # Z-Score excellent en ranging
            reason += " (marché ranging)"
        elif market_regime in ["TRENDING_BULL", "TRENDING_BEAR"]:
            confidence_boost -= 0.05  # Z-Score moins fiable en trending fort
            reason += " (marché trending)"
            
        # Anomaly detection
        anomaly_detected = values.get('anomaly_detected')
        if anomaly_detected:
            confidence_boost += 0.05
            reason += " + anomalie détectée"
            
        # Pattern confluence
        pattern_detected = values.get('pattern_detected')
        pattern_confidence = values.get('pattern_confidence')
        if pattern_detected and 'reversal' in str(pattern_detected).lower():
            if pattern_confidence is not None:
                try:
                    pattern_conf = float(pattern_confidence)
                    if pattern_conf > 70:
                        confidence_boost += 0.1
                        reason += " + pattern reversal"
                except (ValueError, TypeError):
                    pass
                    
        # Confluence score global
        confluence_score = values.get('confluence_score')
        if confluence_score is not None:
            try:
                conf_val = float(confluence_score)
                if conf_val > 80:
                    confidence_boost += 0.1
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
                "zscore_value": zscore_data['zscore_value'],
                "zscore_direction": zscore_data['zscore_direction'],
                "zscore_strength": zscore_data['zscore_strength'],
                "zscore_components": zscore_data['zscore_components'],
                "reversal_direction": reversal_direction,
                "reversal_score": reversal_analysis['score'],
                "reversal_indicators": reversal_analysis['indicators'],
                "volatility_score": volatility_analysis['score'],
                "volatility_indicators": volatility_analysis['indicators'],
                "bb_position": values.get('bb_position'),
                "rsi_14": values.get('rsi_14'),
                "williams_r": values.get('williams_r'),
                "volume_ratio": values.get('volume_ratio'),
                "atr_percentile": values.get('atr_percentile'),
                "market_regime": values.get('market_regime'),
                "confluence_score": values.get('confluence_score'),
                "anomaly_detected": values.get('anomaly_detected')
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que tous les indicateurs requis sont présents."""
        required_indicators = [
            'bb_position', 'rsi_14', 'atr_percentile'
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
