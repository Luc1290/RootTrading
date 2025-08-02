"""
Stochastic_Oversold_Buy_Strategy - Stratégie basée sur les conditions oversold du Stochastic.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class Stochastic_Oversold_Buy_Strategy(BaseStrategy):
    """
    Stratégie utilisant les conditions de survente du Stochastic pour identifier les opportunités d'achat.
    
    Le Stochastic mesure la position du prix de clôture par rapport aux highs/lows récents :
    - %K = (Close - Low_n) / (High_n - Low_n) * 100
    - %D = moyenne mobile de %K
    - Oversold = %K et %D < 20 (survente)
    - Signal d'achat = sortie de survente + croisement %K > %D + confirmations
    
    Signaux générés:
    - BUY: Stochastic sort de zone oversold + croisement %K > %D + confirmations
    - Pas de signaux SELL (stratégie focalisée sur les achats en survente)
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres Stochastic
        self.oversold_threshold = 20  # Seuil de survente
        self.exit_oversold_threshold = 25  # Seuil de sortie de survente
        self.overbought_threshold = 80  # Seuil de surachat (pour éviter les entrées)
        self.min_crossover_separation = 2  # Distance minimum entre %K et %D pour croisement valide
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs Stochastic et confirmation."""
        return {
            # Stochastic principal
            'stoch_k': self.indicators.get('stoch_k'),
            'stoch_d': self.indicators.get('stoch_d'),
            'stoch_fast_k': self.indicators.get('stoch_fast_k'),
            'stoch_fast_d': self.indicators.get('stoch_fast_d'),
            'stoch_rsi': self.indicators.get('stoch_rsi'),
            # RSI pour confluence
            'rsi_14': self.indicators.get('rsi_14'),
            'rsi_21': self.indicators.get('rsi_21'),
            # Williams %R (similaire au Stochastic)
            'williams_r': self.indicators.get('williams_r'),
            # Moyennes mobiles pour contexte de tendance
            'ema_12': self.indicators.get('ema_12'),
            'ema_26': self.indicators.get('ema_26'),
            'ema_50': self.indicators.get('ema_50'),
            'hull_20': self.indicators.get('hull_20'),
            # MACD pour confirmation momentum
            'macd_line': self.indicators.get('macd_line'),
            'macd_signal': self.indicators.get('macd_signal'),
            'macd_histogram': self.indicators.get('macd_histogram'),
            'macd_trend': self.indicators.get('macd_trend'),
            # Tendance et direction
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'momentum_score': self.indicators.get('momentum_score'),
            # ADX pour force de tendance
            'adx_14': self.indicators.get('adx_14'),
            'plus_di': self.indicators.get('plus_di'),
            'minus_di': self.indicators.get('minus_di'),
            # Volume pour confirmation
            'volume_ratio': self.indicators.get('volume_ratio'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            'trade_intensity': self.indicators.get('trade_intensity'),
            # Support/Résistance pour contexte
            'nearest_support': self.indicators.get('nearest_support'),
            'support_strength': self.indicators.get('support_strength'),
            'nearest_resistance': self.indicators.get('nearest_resistance'),
            # VWAP pour niveaux institutionnels
            'vwap_10': self.indicators.get('vwap_10'),
            'anchored_vwap': self.indicators.get('anchored_vwap'),
            # Bollinger Bands pour contexte volatilité
            'bb_lower': self.indicators.get('bb_lower'),
            'bb_position': self.indicators.get('bb_position'),
            'bb_width': self.indicators.get('bb_width'),
            # Market structure
            'market_regime': self.indicators.get('market_regime'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            'regime_strength': self.indicators.get('regime_strength'),
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
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal d'achat basé sur les conditions oversold du Stochastic.
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
        
        # Analyser les conditions Stochastic
        stoch_analysis = self._analyze_stochastic_conditions(values)
        if stoch_analysis is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Données Stochastic non disponibles",
                "metadata": {"strategy": self.name}
            }
            
        # Vérifier les conditions d'achat oversold
        buy_condition = self._check_oversold_buy_conditions(stoch_analysis)
        if buy_condition is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": stoch_analysis.get('rejection_reason', "Conditions oversold non remplies"),
                "metadata": {"strategy": self.name}
            }
            
        # Créer le signal d'achat avec confirmations
        return self._create_oversold_buy_signal(values, current_price or 0.0, stoch_analysis, buy_condition)
        
    def _analyze_stochastic_conditions(self, values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyse les conditions actuelles du Stochastic."""
        stoch_k = values.get('stoch_k')
        stoch_d = values.get('stoch_d')
        
        if stoch_k is None or stoch_d is None:
            return None
            
        try:
            k_val = float(stoch_k)
            d_val = float(stoch_d)
        except (ValueError, TypeError):
            return None
            
        # États du Stochastic
        is_oversold = k_val <= self.oversold_threshold and d_val <= self.oversold_threshold
        is_exiting_oversold = (k_val > self.oversold_threshold or d_val > self.oversold_threshold) and \
                             (k_val <= self.exit_oversold_threshold or d_val <= self.exit_oversold_threshold)
        is_overbought = k_val >= self.overbought_threshold or d_val >= self.overbought_threshold
        
        # Croisement %K > %D (signal haussier)
        k_above_d = k_val > d_val
        crossover_strength = abs(k_val - d_val)
        
        # Analyser les Stochastic Fast si disponibles
        stoch_fast_k = values.get('stoch_fast_k')
        stoch_fast_d = values.get('stoch_fast_d')
        fast_analysis = None
        
        if stoch_fast_k is not None and stoch_fast_d is not None:
            try:
                fast_k = float(stoch_fast_k)
                fast_d = float(stoch_fast_d)
                fast_analysis = {
                    'fast_k': fast_k,
                    'fast_d': fast_d,
                    'fast_oversold': fast_k <= self.oversold_threshold and fast_d <= self.oversold_threshold,
                    'fast_crossover': fast_k > fast_d
                }
            except (ValueError, TypeError):
                fast_analysis = None
                
        # Déterminer les raisons de rejet potentielles
        rejection_reasons = []
        if is_overbought:
            rejection_reasons.append(f"Stochastic en surachat (K:{k_val:.1f}, D:{d_val:.1f})")
        if not (is_oversold or is_exiting_oversold):
            rejection_reasons.append(f"Stochastic pas en survente (K:{k_val:.1f}, D:{d_val:.1f})")
        if not k_above_d:
            rejection_reasons.append(f"Pas de croisement haussier K<D ({k_val:.1f}<{d_val:.1f})")
        if crossover_strength < self.min_crossover_separation:
            rejection_reasons.append(f"Croisement trop faible ({crossover_strength:.1f})")
            
        return {
            'stoch_k': k_val,
            'stoch_d': d_val,
            'is_oversold': is_oversold,
            'is_exiting_oversold': is_exiting_oversold,
            'is_overbought': is_overbought,
            'k_above_d': k_above_d,
            'crossover_strength': crossover_strength,
            'fast_analysis': fast_analysis,
            'rejection_reason': "; ".join(rejection_reasons) if rejection_reasons else None
        }
        
    def _check_oversold_buy_conditions(self, stoch_analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Vérifie si les conditions d'achat oversold sont remplies."""
        # Rejeter si en surachat
        if stoch_analysis['is_overbought']:
            return None
            
        # Condition principale: oversold OU sortie d'oversold
        if not (stoch_analysis['is_oversold'] or stoch_analysis['is_exiting_oversold']):
            return None
            
        # Condition croisement: %K > %D
        if not stoch_analysis['k_above_d']:
            return None
            
        # Condition force du croisement
        if stoch_analysis['crossover_strength'] < self.min_crossover_separation:
            return None
            
        # Déterminer le type de signal
        signal_quality = "strong" if stoch_analysis['is_oversold'] else "moderate"
        signal_type = "oversold_bounce" if stoch_analysis['is_oversold'] else "oversold_exit"
        
        return {
            'signal_type': signal_type,
            'signal_quality': signal_quality,
            'crossover_strength': stoch_analysis['crossover_strength']
        }
        
    def _create_oversold_buy_signal(self, values: Dict[str, Any], current_price: float,
                                   stoch_analysis: Dict[str, Any], buy_condition: Dict[str, Any]) -> Dict[str, Any]:
        """Crée le signal d'achat oversold avec confirmations."""
        signal_side = "BUY"  # Stratégie uniquement orientée achat
        base_confidence = 0.6  # Base élevée pour signaux oversold
        confidence_boost = 0.0
        
        stoch_k = stoch_analysis['stoch_k']
        stoch_d = stoch_analysis['stoch_d']
        signal_type = buy_condition['signal_type']
        signal_quality = buy_condition['signal_quality']
        
        # Construction de la raison
        reason = f"Stochastic oversold: K={stoch_k:.1f}, D={stoch_d:.1f}"
        
        # Bonus selon le type de signal
        if signal_quality == "strong":
            confidence_boost += 0.20  # Signal fort en survente
            reason += " (survente profonde)"
        else:
            confidence_boost += 0.15  # Signal modéré en sortie de survente
            reason += " (sortie survente)"
            
        # Bonus selon la force du croisement
        crossover_strength = buy_condition['crossover_strength']
        if crossover_strength > 10:
            confidence_boost += 0.15
            reason += f" + croisement fort (Δ{crossover_strength:.1f})"
        elif crossover_strength > 5:
            confidence_boost += 0.10
            reason += f" + croisement modéré (Δ{crossover_strength:.1f})"
        else:
            confidence_boost += 0.05
            reason += f" + croisement faible (Δ{crossover_strength:.1f})"
            
        # Confirmation avec Stochastic Fast
        fast_analysis = stoch_analysis.get('fast_analysis')
        if fast_analysis is not None:
            if fast_analysis['fast_oversold'] and fast_analysis['fast_crossover']:
                confidence_boost += 0.15
                reason += " + Stoch Fast confirme"
            elif fast_analysis['fast_crossover']:
                confidence_boost += 0.08
                reason += " + Stoch Fast aligné"
                
        # Confirmation avec RSI
        rsi_14 = values.get('rsi_14')
        if rsi_14 is not None:
            try:
                rsi = float(rsi_14)
                if rsi <= 30:  # RSI aussi en survente
                    confidence_boost += 0.15
                    reason += f" + RSI survente ({rsi:.1f})"
                elif rsi <= 40:  # RSI supportive
                    confidence_boost += 0.10
                    reason += f" + RSI favorable ({rsi:.1f})"
                elif rsi >= 70:  # RSI diverge
                    confidence_boost -= 0.10
                    reason += f" mais RSI élevé ({rsi:.1f})"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec Williams %R
        williams_r = values.get('williams_r')
        if williams_r is not None:
            try:
                wr = float(williams_r)
                if wr <= -80:  # Williams R aussi en survente
                    confidence_boost += 0.12
                    reason += f" + Williams R survente ({wr:.1f})"
                elif wr <= -60:
                    confidence_boost += 0.08
                    reason += f" + Williams R favorable ({wr:.1f})"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec MACD
        macd_line = values.get('macd_line')
        macd_signal = values.get('macd_signal')
        if macd_line is not None and macd_signal is not None:
            try:
                macd_val = float(macd_line)
                macd_sig = float(macd_signal)
                macd_bullish = macd_val > macd_sig
                
                if macd_bullish:
                    confidence_boost += 0.12
                    reason += " + MACD haussier"
                elif macd_val > macd_sig - 0.0001:  # MACD proche du croisement
                    confidence_boost += 0.05
                    reason += " + MACD proche croisement"
            except (ValueError, TypeError):
                pass
                
        # Contexte de tendance
        directional_bias = values.get('directional_bias')
        trend_strength = values.get('trend_strength')
        
        if directional_bias == "BULLISH":
            confidence_boost += 0.15
            reason += " + bias haussier"
        elif directional_bias == "BEARISH":
            confidence_boost -= 0.08  # Contre-tendance mais oversold peut rebondir
            reason += " mais bias baissier"
            
        if trend_strength is not None:
            # trend_strength selon schéma: WEAK, MODERATE, STRONG, VERY_STRONG
            if trend_strength in ['VERY_STRONG']:
                confidence_boost += 0.12
                reason += f" + tendance très forte ({trend_strength})"
            elif trend_strength == 'STRONG':
                confidence_boost += 0.08
                reason += f" + tendance forte ({trend_strength})"
            elif trend_strength in ['MODERATE', 'WEAK']:
                confidence_boost += 0.03
                reason += f" + tendance modérée ({trend_strength})"
                
        # Support proche pour confluence
        nearest_support = values.get('nearest_support')
        if nearest_support is not None and current_price is not None:
            try:
                support = float(nearest_support)
                distance_to_support = abs(current_price - support) / current_price
                
                if distance_to_support <= 0.02:  # Proche du support (2%)
                    confidence_boost += 0.15
                    reason += " + près support"
                elif distance_to_support <= 0.05:  # Support modéré (5%)
                    confidence_boost += 0.08
                    reason += " + support proche"
            except (ValueError, TypeError):
                pass
                
        # Bollinger Bands pour contexte
        bb_lower = values.get('bb_lower')
        bb_position = values.get('bb_position')
        if bb_lower is not None and current_price is not None:
            try:
                bb_low = float(bb_lower)
                if current_price <= bb_low * 1.02:  # Prix près BB lower
                    confidence_boost += 0.10
                    reason += " + près BB inférieure"
            except (ValueError, TypeError):
                pass
                
        if bb_position is not None:
            try:
                pos = float(bb_position)
                if pos <= 0.2:  # Position basse dans les BB
                    confidence_boost += 0.08
                    reason += " + position BB basse"
            except (ValueError, TypeError):
                pass
                
        # VWAP pour contexte institutionnel
        vwap_10 = values.get('vwap_10')
        if vwap_10 is not None and current_price is not None:
            try:
                vwap = float(vwap_10)
                if current_price < vwap:  # Prix sous VWAP = potentiel rebond
                    confidence_boost += 0.08
                    reason += " + prix < VWAP"
            except (ValueError, TypeError):
                pass
                
        # Volume pour confirmation
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= 1.3:  # Volume élevé confirme l'intérêt
                    confidence_boost += 0.12
                    reason += f" + volume élevé ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.1:
                    confidence_boost += 0.08
                    reason += f" + volume modéré ({vol_ratio:.1f}x)"
            except (ValueError, TypeError):
                pass
                
        # Market regime
        market_regime = values.get('market_regime')
        if market_regime == "RANGING":
            confidence_boost += 0.10  # Oversold plus fiable en ranging
            reason += " (marché ranging)"
        elif market_regime in ["TRENDING_BULL", "TRENDING_BEAR"]:
            confidence_boost += 0.05  # Oversold peut fonctionner en trending
            reason += " (marché trending)"
            
        # Volatilité
        volatility_regime = values.get('volatility_regime')
        if volatility_regime in ["HIGH", "EXTREME"]:
            confidence_boost += 0.05  # Haute volatilité = rebonds plus forts
            reason += " + volatilité élevée"
        elif volatility_regime == "NORMAL":
            confidence_boost += 0.03
            reason += " + volatilité normale"
            
        # Pattern detection
        pattern_detected = values.get('pattern_detected')
        pattern_confidence = values.get('pattern_confidence')
        if pattern_detected and pattern_confidence is not None:
            try:
                confidence = float(pattern_confidence)
                if confidence > 0.7:
                    confidence_boost += 0.08
                    reason += " + pattern détecté"
            except (ValueError, TypeError):
                pass
                
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
        strength: str = self.get_strength_from_confidence(confidence)
        
        return {
            "side": signal_side,
            "confidence": confidence,
            "strength": strength,
            "reason": reason,
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "current_price": current_price,
                "stoch_k": stoch_k,
                "stoch_d": stoch_d,
                "signal_type": signal_type,
                "signal_quality": signal_quality,
                "crossover_strength": crossover_strength,
                "is_oversold": stoch_analysis['is_oversold'],
                "is_exiting_oversold": stoch_analysis['is_exiting_oversold'],
                "fast_analysis": fast_analysis,
                "rsi_14": values.get('rsi_14'),
                "williams_r": values.get('williams_r'),
                "macd_line": values.get('macd_line'),
                "macd_signal": values.get('macd_signal'),
                "directional_bias": values.get('directional_bias'),
                "trend_strength": values.get('trend_strength'),
                "nearest_support": values.get('nearest_support'),
                "bb_position": values.get('bb_position'),
                "volume_ratio": values.get('volume_ratio'),
                "market_regime": values.get('market_regime'),
                "volatility_regime": values.get('volatility_regime'),
                "confluence_score": values.get('confluence_score')
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que les données Stochastic nécessaires sont présentes."""
        if not super().validate_data():
            return False
            
        # Au minimum, il faut stoch_k et stoch_d
        required = ['stoch_k', 'stoch_d']
        
        for indicator in required:
            if indicator not in self.indicators:
                logger.warning(f"{self.name}: Indicateur manquant: {indicator}")
                return False
            if self.indicators[indicator] is None:
                logger.warning(f"{self.name}: Indicateur null: {indicator}")
                return False
                
        return True
