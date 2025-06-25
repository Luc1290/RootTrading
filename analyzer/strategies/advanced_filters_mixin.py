"""
Mixin contenant les filtres sophistiqués réutilisables pour toutes les stratégies.
Évite la duplication de code et assure la cohérence des filtres.
"""
import logging
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import talib
from shared.src.enums import OrderSide

logger = logging.getLogger(__name__)


class AdvancedFiltersMixin:
    """
    Mixin fournissant des filtres sophistiqués réutilisables pour toutes les stratégies.
    """
    
    def _analyze_volume_confirmation_common(self, volumes: Optional[np.ndarray]) -> float:
        """
        Filtre volume uniforme pour toutes les stratégies.
        """
        if volumes is None or len(volumes) < 10:
            return 0.7
        
        current_volume = volumes[-1]
        avg_volume_10 = np.mean(volumes[-10:])
        avg_volume_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else avg_volume_10
        
        volume_ratio_10 = current_volume / avg_volume_10 if avg_volume_10 > 0 else 1.0
        volume_ratio_20 = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
        
        if volume_ratio_10 > 1.5 and volume_ratio_20 > 1.3:
            return 0.95  # Très forte expansion
        elif volume_ratio_10 > 1.2 and volume_ratio_20 > 1.1:
            return 0.85  # Bonne expansion
        elif volume_ratio_10 > 0.8:
            return 0.75  # Volume acceptable
        else:
            return 0.5   # Volume faible
    
    def _analyze_atr_environment_common(self, df: pd.DataFrame) -> float:
        """
        Analyse l'environnement de volatilité via ATR.
        """
        try:
            if len(df) < 20:
                return 0.7
            
            # Calculer ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr_series = true_range.rolling(14).mean()
            
            if len(atr_series) < 20:
                return 0.7
            
            current_atr = atr_series.iloc[-1]
            avg_atr = atr_series.iloc[-20:].mean()
            
            atr_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
            
            if atr_ratio > 1.3:
                return 0.9   # Volatilité en expansion
            elif atr_ratio > 1.1:
                return 0.8   # Légère expansion
            elif atr_ratio > 0.7:
                return 0.75  # Volatilité normale
            else:
                return 0.6   # Faible volatilité
                
        except Exception as e:
            logger.warning(f"Erreur analyse ATR: {e}")
            return 0.7
    
    def _analyze_adx_trend_strength_common(self, df: pd.DataFrame, min_adx_threshold: float = 20.0) -> Dict[str, Any]:
        """
        Analyse la force de tendance via l'ADX et détermine si les stratégies de tendance doivent être actives.
        
        Args:
            df: DataFrame avec OHLCV
            min_adx_threshold: Seuil minimum ADX pour considérer une tendance (défaut: 20)
            
        Returns:
            Dict avec adx_value, is_trending, confidence_score
        """
        try:
            if len(df) < 30:
                return {
                    'adx_value': None,
                    'is_trending': False,
                    'confidence_score': 0.5,
                    'disable_trend_strategies': True,
                    'reason': 'insufficient_data'
                }
            
            # Calculer ADX
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            adx = talib.ADX(high, low, close, timeperiod=14)
            
            if np.isnan(adx[-1]):
                return {
                    'adx_value': None,
                    'is_trending': False,
                    'confidence_score': 0.5,
                    'disable_trend_strategies': True,
                    'reason': 'adx_calculation_failed'
                }
            
            current_adx = adx[-1]
            
            # Déterminer si on est en tendance
            is_trending = current_adx >= min_adx_threshold
            disable_trend_strategies = current_adx < min_adx_threshold
            
            # Calculer un score de confiance basé sur l'ADX
            if current_adx >= 40:
                confidence_score = 0.95  # Tendance très forte
            elif current_adx >= 30:
                confidence_score = 0.85  # Tendance forte
            elif current_adx >= min_adx_threshold:
                confidence_score = 0.75  # Tendance modérée
            else:
                confidence_score = 0.3   # Pas de tendance (range)
            
            reason = f"adx_{current_adx:.1f}_{'trending' if is_trending else 'ranging'}"
            
            logger.debug(f"📊 ADX Analysis: {current_adx:.1f} - {'Trending' if is_trending else 'Ranging'} (threshold: {min_adx_threshold})")
            
            return {
                'adx_value': current_adx,
                'is_trending': is_trending,
                'confidence_score': confidence_score,
                'disable_trend_strategies': disable_trend_strategies,
                'reason': reason
            }
            
        except Exception as e:
            logger.warning(f"Erreur analyse ADX: {e}")
            return {
                'adx_value': None,
                'is_trending': False,
                'confidence_score': 0.5,
                'disable_trend_strategies': True,
                'reason': f'error_{str(e)[:20]}'
            }
    
    def _analyze_trend_alignment_common(self, df: pd.DataFrame, signal_side: OrderSide) -> float:
        """
        Analyse de tendance uniforme via EMA.
        """
        try:
            if len(df) < 50:
                return 0.7
            
            prices = df['close'].values
            
            # EMAs pour tendance
            ema_21 = talib.EMA(prices, timeperiod=21)
            ema_50 = talib.EMA(prices, timeperiod=50)
            
            if np.isnan(ema_21[-1]) or np.isnan(ema_50[-1]):
                return 0.7
            
            current_price = prices[-1]
            trend_21 = ema_21[-1]
            trend_50 = ema_50[-1]
            
            # HARMONISATION: Ajouter vérification de seuil comme dans le signal_aggregator
            ema_trend_bullish = trend_21 > trend_50 * 1.005  # Seuil 0.5%
            ema_trend_bearish = trend_21 < trend_50 * 0.995  # Seuil 0.5%
            
            if signal_side == OrderSide.BUY:
                if current_price > trend_21 and ema_trend_bullish:
                    return 0.9   # Tendance alignée avec seuil
                elif current_price > trend_50:
                    return 0.8   # Tendance modérée
                else:
                    return 0.5   # Contre tendance
            
            else:  # SELL
                if current_price < trend_21 and ema_trend_bearish:
                    return 0.9   # Tendance alignée avec seuil
                elif current_price < trend_50:
                    return 0.8   # Tendance modérée
                else:
                    return 0.5   # Contre tendance
                    
        except Exception as e:
            logger.warning(f"Erreur analyse tendance: {e}")
            return 0.7
    
    def _detect_support_resistance_common(self, df: pd.DataFrame, current_price: float, signal_side: OrderSide) -> float:
        """
        Détection support/résistance uniforme.
        """
        try:
            if len(df) < 30:
                return 0.7
            
            highs = df['high'].values
            lows = df['low'].values
            
            # Chercher les pivots sur 30 périodes
            lookback = min(30, len(df))
            recent_highs = highs[-lookback:]
            recent_lows = lows[-lookback:]
            
            # Détecter les niveaux pivots simples
            pivot_highs = []
            pivot_lows = []
            
            for i in range(2, len(recent_highs) - 2):
                # Pivot haut
                if (recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i-2] and
                    recent_highs[i] > recent_highs[i+1] and recent_highs[i] > recent_highs[i+2]):
                    pivot_highs.append(recent_highs[i])
                
                # Pivot bas
                if (recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i-2] and
                    recent_lows[i] < recent_lows[i+1] and recent_lows[i] < recent_lows[i+2]):
                    pivot_lows.append(recent_lows[i])
            
            if signal_side == OrderSide.BUY:
                if not pivot_lows:
                    return 0.6
                
                # Trouver le support le plus proche
                supports_below = [s for s in pivot_lows if s <= current_price * 1.02]
                if not supports_below:
                    return 0.5
                
                nearest_support = max(supports_below)
                distance_pct = abs(current_price - nearest_support) / current_price * 100
                
                if distance_pct < 0.5:
                    return 0.95  # Très proche du support
                elif distance_pct < 1.5:
                    return 0.8   # Proche du support
                else:
                    return 0.7   # Support distant
            
            else:  # SELL
                if not pivot_highs:
                    return 0.6
                
                resistances_above = [r for r in pivot_highs if r >= current_price * 0.98]
                if not resistances_above:
                    return 0.5
                
                nearest_resistance = min(resistances_above)
                distance_pct = abs(nearest_resistance - current_price) / current_price * 100
                
                if distance_pct < 0.5:
                    return 0.95  # Très proche de la résistance
                elif distance_pct < 1.5:
                    return 0.8   # Proche de la résistance
                else:
                    return 0.7   # Résistance distante
                    
        except Exception as e:
            logger.warning(f"Erreur S/R: {e}")
            return 0.7
    
    def _calculate_rsi_confirmation_common(self, df: pd.DataFrame, signal_side: OrderSide) -> float:
        """
        Confirmation RSI uniforme.
        """
        try:
            if len(df) < 30:
                return 0.7
            
            prices = df['close'].values
            rsi = talib.RSI(prices, timeperiod=14)
            
            if np.isnan(rsi[-1]):
                return 0.7
            
            current_rsi = rsi[-1]
            
            if signal_side == OrderSide.BUY:
                if current_rsi < 35:  # Zone survente
                    return 0.9
                elif current_rsi < 50:  # Zone neutre basse
                    return 0.8
                elif current_rsi < 65:  # Zone neutre haute
                    return 0.7
                else:  # Zone surachat
                    return 0.5
            
            else:  # SELL
                if current_rsi > 65:  # Zone surachat
                    return 0.9
                elif current_rsi > 50:  # Zone neutre haute
                    return 0.8
                elif current_rsi > 35:  # Zone neutre basse
                    return 0.7
                else:  # Zone survente
                    return 0.5
                    
        except Exception as e:
            logger.warning(f"Erreur RSI: {e}")
            return 0.7
    
    def _calculate_composite_confidence_common(self, scores: Dict[str, float], 
                                              weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calcul de confiance composite uniforme.
        """
        if weights is None:
            # Poids par défaut équilibrés
            weights = {key: 1.0/len(scores) for key in scores.keys()}
        
        # Normaliser les poids
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        composite = sum(scores[key] * normalized_weights.get(key, 0) for key in scores.keys())
        
        return max(0.0, min(1.0, composite))