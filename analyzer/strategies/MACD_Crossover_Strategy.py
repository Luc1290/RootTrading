"""
MACD_Crossover_Strategy - Stratégie basée sur les croisements MACD.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class MACD_Crossover_Strategy(BaseStrategy):
    """
    Stratégie utilisant les croisements MACD pour détecter les changements de momentum.
    
    MACD = EMA12 - EMA26, Signal = EMA9 du MACD, Histogram = MACD - Signal
    
    Signaux générés:
    - BUY: MACD croise au-dessus Signal + confirmations haussières
    - SELL: MACD croise en-dessous Signal + confirmations baissières
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Paramètres MACD
        self.min_macd_distance = 0.001   # Distance minimum MACD/Signal pour éviter bruit
        self.histogram_threshold = 0.0005  # Seuil histogram pour confirmation
        self.zero_line_bonus = 0.1       # Bonus si MACD au-dessus/dessous zéro
        # Paramètres de filtre de tendance
        self.trend_filter_enabled = True  # Activer le filtre de tendance globale
        self.contra_trend_penalty = 0.3   # Pénalité pour signaux contra-trend
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs MACD."""
        return {
            # MACD complet
            'macd_line': self.indicators.get('macd_line'),
            'macd_signal': self.indicators.get('macd_signal'),
            'macd_histogram': self.indicators.get('macd_histogram'),
            'macd_zero_cross': self.indicators.get('macd_zero_cross'),
            'macd_signal_cross': self.indicators.get('macd_signal_cross'),
            'macd_trend': self.indicators.get('macd_trend'),
            # PPO (Percentage Price Oscillator - MACD normalisé)
            'ppo': self.indicators.get('ppo'),
            # EMA pour contexte (MACD = EMA12 - EMA26)
            'ema_12': self.indicators.get('ema_12'),
            'ema_26': self.indicators.get('ema_26'),
            'ema_50': self.indicators.get('ema_50'),
            # Trend et momentum pour confirmation
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'momentum_score': self.indicators.get('momentum_score'),
            # Oscillateurs pour confluence
            'rsi_14': self.indicators.get('rsi_14'),
            'stoch_k': self.indicators.get('stoch_k'),
            'stoch_d': self.indicators.get('stoch_d'),
            # Volume pour confirmation
            'volume_ratio': self.indicators.get('volume_ratio'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            # Contexte marché
            'market_regime': self.indicators.get('market_regime'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            # Confluence
            'signal_strength': self.indicators.get('signal_strength'),
            'confluence_score': self.indicators.get('confluence_score'),
            # Indicateurs de tendance globale
            'regime_strength': self.indicators.get('regime_strength'),
            'trend_alignment': self.indicators.get('trend_alignment'),
            'adx_14': self.indicators.get('adx_14')
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
        Génère un signal basé sur les croisements MACD.
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
        
        # Vérification des indicateurs MACD essentiels
        try:
            macd_line = float(values['macd_line']) if values['macd_line'] is not None else None
            macd_signal = float(values['macd_signal']) if values['macd_signal'] is not None else None
            macd_histogram = float(values['macd_histogram']) if values['macd_histogram'] is not None else None
        except (ValueError, TypeError) as e:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Erreur conversion MACD: {e}",
                "metadata": {"strategy": self.name}
            }
            
        if macd_line is None or macd_signal is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "MACD line ou signal non disponibles",
                "metadata": {"strategy": self.name}
            }
            
        # Analyse du croisement MACD
        macd_above_signal = macd_line > macd_signal
        macd_distance = abs(macd_line - macd_signal)
        
        # Vérification que les lignes ne sont pas trop proches (éviter faux signaux)
        if macd_distance < self.min_macd_distance:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"MACD trop proche Signal ({macd_distance:.4f}) - pas de signal clair",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "macd_line": macd_line,
                    "macd_signal": macd_signal,
                    "distance": macd_distance
                }
            }
            
        signal_side = None
        reason = ""
        base_confidence = 0.5
        confidence_boost = 0.0
        cross_type = None
        
        # Filtre de tendance global AVANT de décider du signal
        market_regime = values.get('market_regime')
        trend_alignment = values.get('trend_alignment')
        regime_strength = values.get('regime_strength')
        adx_value = values.get('adx_14')
        
        # Déterminer la tendance principale
        is_strong_uptrend = False
        is_strong_downtrend = False
        trend_confirmed = False
        
        if self.trend_filter_enabled and market_regime:
            if market_regime == 'BULLISH':
                is_strong_uptrend = True
                trend_confirmed = True
            elif market_regime == 'BEARISH':
                is_strong_downtrend = True
                trend_confirmed = True
        
        # Vérification supplémentaire avec trend_alignment
        if trend_alignment is not None:
            if trend_alignment > 30:  # Forte tendance haussière
                is_strong_uptrend = True
            elif trend_alignment < -30:  # Forte tendance baissière
                is_strong_downtrend = True
        
        # Logique de croisement MACD adaptée à la tendance
        if macd_above_signal:
            # MACD au-dessus du signal
            if not self.trend_filter_enabled or not is_strong_downtrend:
                # OK pour BUY si pas en forte tendance baissière
                signal_side = "BUY"
                cross_type = "bullish_cross"
                reason = f"MACD ({macd_line:.4f}) > Signal ({macd_signal:.4f})"
                confidence_boost += 0.15
                
                if is_strong_uptrend:
                    confidence_boost += 0.20  # Bonus si aligné avec tendance
                    reason += " - aligné tendance haussière"
            else:
                # En forte tendance baissière, ignorer les signaux BUY
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": "Signal BUY ignoré en forte tendance baissière",
                    "metadata": {
                        "strategy": self.name,
                        "symbol": self.symbol,
                        "market_regime": market_regime,
                        "trend_alignment": trend_alignment
                    }
                }
        else:
            # MACD en-dessous du signal
            if not self.trend_filter_enabled or not is_strong_uptrend:
                # OK pour SELL si pas en forte tendance haussière
                signal_side = "SELL"
                cross_type = "bearish_cross"
                reason = f"MACD ({macd_line:.4f}) < Signal ({macd_signal:.4f})"
                confidence_boost += 0.15
                
                if is_strong_downtrend:
                    confidence_boost += 0.20  # Bonus si aligné avec tendance
                    reason += " - aligné tendance baissière"
            else:
                # En forte tendance haussière, ignorer les signaux SELL
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": "Signal SELL ignoré en forte tendance haussière",
                    "metadata": {
                        "strategy": self.name,
                        "symbol": self.symbol,
                        "market_regime": market_regime,
                        "trend_alignment": trend_alignment
                    }
                }
            
        # Bonus selon la force de la séparation
        separation_strength = abs(macd_line - macd_signal)
        if separation_strength >= 0.01:
            confidence_boost += 0.15
            reason += f" - séparation forte ({separation_strength:.4f})"
        elif separation_strength >= 0.005:
            confidence_boost += 0.10
            reason += f" - séparation modérée ({separation_strength:.4f})"
        else:
            confidence_boost += 0.05
            reason += f" - séparation faible ({separation_strength:.4f})"
            
        # Confirmation avec Histogram MACD
        if macd_histogram is not None:
            if signal_side == "BUY" and macd_histogram > self.histogram_threshold:
                confidence_boost += 0.15
                reason += f" + histogram positif ({macd_histogram:.4f})"
            elif signal_side == "SELL" and macd_histogram < -self.histogram_threshold:
                confidence_boost += 0.15
                reason += f" + histogram négatif ({macd_histogram:.4f})"
            elif signal_side == "BUY" and macd_histogram > 0:
                confidence_boost += 0.10
                reason += f" + histogram favorable ({macd_histogram:.4f})"
            elif signal_side == "SELL" and macd_histogram < 0:
                confidence_boost += 0.10
                reason += f" + histogram favorable ({macd_histogram:.4f})"
            else:
                confidence_boost -= 0.05
                reason += f" mais histogram défavorable ({macd_histogram:.4f})"
                
        # Bonus si MACD dans la bonne zone par rapport à zéro
        if signal_side == "BUY" and macd_line > 0:
            confidence_boost += self.zero_line_bonus
            reason += " + MACD au-dessus zéro"
        elif signal_side == "SELL" and macd_line < 0:
            confidence_boost += self.zero_line_bonus
            reason += " + MACD en-dessous zéro"
        elif signal_side == "BUY" and macd_line < -0.01:
            confidence_boost -= 0.05
            reason += " mais MACD très négatif"
        elif signal_side == "SELL" and macd_line > 0.01:
            confidence_boost -= 0.05
            reason += " mais MACD très positif"
            
        # Confirmation avec macd_trend pré-calculé
        macd_trend = values.get('macd_trend')
        if macd_trend:
            if (signal_side == "BUY" and macd_trend == "bullish") or \
               (signal_side == "SELL" and macd_trend == "bearish"):
                confidence_boost += 0.10
                reason += f" + trend MACD {macd_trend}"
                
        # Confirmation avec EMA (base du MACD)
        ema_12 = values.get('ema_12')
        ema_26 = values.get('ema_26')
        ema_50 = values.get('ema_50')
        
        if ema_12 is not None and ema_26 is not None:
            try:
                ema12_val = float(ema_12)
                ema26_val = float(ema_26)
                ema_cross_matches = (signal_side == "BUY" and ema12_val > ema26_val) or \
                                  (signal_side == "SELL" and ema12_val < ema26_val)
                
                if ema_cross_matches:
                    confidence_boost += 0.10
                    reason += " + EMA confirme"
                else:
                    confidence_boost -= 0.05
                    reason += " mais EMA diverge"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec EMA 50 pour filtre de tendance
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
                
        # Confirmation avec trend_strength
        trend_strength = values.get('trend_strength')
        if trend_strength is not None:
            try:
                trend_str = float(trend_strength)
                if trend_str > 0.6:
                    confidence_boost += 0.10
                    reason += f" + tendance forte ({trend_str:.2f})"
                elif trend_str > 0.4:
                    confidence_boost += 0.05
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
                
        # Momentum score pour confluence
        momentum_score = values.get('momentum_score')
        if momentum_score is not None:
            try:
                momentum = float(momentum_score)
                # Format 0-100, 50=neutre
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
                
        # Confirmation avec RSI (éviter zones extrêmes)
        rsi_14 = values.get('rsi_14')
        if rsi_14 is not None:
            try:
                rsi = float(rsi_14)
                if signal_side == "BUY" and rsi < 70:
                    confidence_boost += 0.05
                elif signal_side == "SELL" and rsi > 30:
                    confidence_boost += 0.05
                elif signal_side == "BUY" and rsi >= 80:
                    confidence_boost -= 0.10
                    reason += " mais RSI surachat"
                elif signal_side == "SELL" and rsi <= 20:
                    confidence_boost -= 0.10
                    reason += " mais RSI survente"
            except (ValueError, TypeError):
                pass
                
        # Stochastic pour confluence
        stoch_k = values.get('stoch_k')
        stoch_d = values.get('stoch_d')
        if stoch_k is not None and stoch_d is not None:
            try:
                k = float(stoch_k)
                d = float(stoch_d)
                stoch_cross = k > d
                
                if (signal_side == "BUY" and stoch_cross) or \
                   (signal_side == "SELL" and not stoch_cross):
                    confidence_boost += 0.08
                    reason += " + Stoch confirme"
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
                
        # Market regime
        market_regime = values.get('market_regime')
        if market_regime == "trending":
            confidence_boost += 0.08
            reason += " (marché trending)"
        elif market_regime == "ranging":
            confidence_boost -= 0.05
            reason += " (marché ranging)"
            
        # PPO pour confirmation (MACD normalisé)
        ppo = values.get('ppo')
        if ppo is not None:
            try:
                ppo_val = float(ppo)
                if (signal_side == "BUY" and ppo_val > 0) or \
                   (signal_side == "SELL" and ppo_val < 0):
                    confidence_boost += 0.05
                    reason += f" + PPO confirme ({ppo_val:.3f})"
            except (ValueError, TypeError):
                pass
                
        # Signal strength et confluence
        signal_strength_calc = values.get('signal_strength')
        if signal_strength_calc is not None:
            try:
                sig_str = float(signal_strength_calc)
                if sig_str > 0.7:
                    confidence_boost += 0.05
            except (ValueError, TypeError):
                pass
                
        confluence_score = values.get('confluence_score')
        if confluence_score is not None:
            try:
                confluence = float(confluence_score)
                if confluence > 0.6:
                    confidence_boost += 0.10
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
                "macd_line": macd_line,
                "macd_signal": macd_signal,
                "macd_histogram": macd_histogram,
                "cross_type": cross_type,
                "macd_distance": macd_distance,
                "macd_trend": macd_trend,
                "ema_12": ema_12,
                "ema_26": ema_26,
                "ema_50": ema_50,
                "trend_strength": trend_strength,
                "directional_bias": directional_bias,
                "momentum_score": momentum_score,
                "rsi_14": rsi_14,
                "stoch_k": stoch_k,
                "stoch_d": stoch_d,
                "volume_ratio": volume_ratio,
                "market_regime": market_regime,
                "ppo": ppo,
                "confluence_score": confluence_score,
                "trend_alignment": trend_alignment,
                "regime_strength": regime_strength,
                "adx_14": adx_value
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que tous les indicateurs MACD requis sont présents."""
        if not super().validate_data():
            return False
            
        required = ['macd_line', 'macd_signal']
        
        for indicator in required:
            if indicator not in self.indicators:
                logger.warning(f"{self.name}: Indicateur manquant: {indicator}")
                return False
            if self.indicators[indicator] is None:
                logger.warning(f"{self.name}: Indicateur null: {indicator}")
                return False
                
        return True
