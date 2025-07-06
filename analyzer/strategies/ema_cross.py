"""
Stratégie EMA Cross Ultra-Précise
Utilise les EMAs pré-calculées de la DB avec validation multi-critères
pour générer des signaux de crossover de très haute qualité.
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from shared.src.enums import OrderSide, SignalStrength

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class EMACrossStrategy(BaseStrategy):
    """
    Stratégie EMA Cross Ultra-Précise qui utilise les EMAs de la DB
    avec des filtres sophistiqués pour des signaux de momentum fiables.
    
    Critères ultra-stricts :
    - Croisements EMA 12/26 avec validation de force
    - Confirmation EMA 50 pour tendance
    - Analyse momentum et volume
    - Filtres volatilité et anti-bruit
    - Détection breakout vs pullback
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        super().__init__(symbol, params)
        
        # Paramètres configurables par symbole depuis la DB
        symbol_params = self.params.get(symbol, {}) if self.params else {}
        
        # Paramètres EMA ultra-précis (harmonisés avec MACD)
        self.ema_fast = 12
        self.ema_slow = 26
        self.ema_trend = 50
        
        # Filtres de qualité configurables
        self.min_crossover_strength = symbol_params.get('ema_gap_min', 0.003)
        self.min_momentum_alignment = 0.7     # Score momentum minimum
        self.min_volume_confirmation = symbol_params.get('volume_ratio_min', 1.4)
        self.max_volatility = 0.08            # Volatilité maximum 8%
        
        # Filtres ultra-précis
        self.min_confidence = 0.78            # Confiance minimum 78%
        self.trend_confirmation_periods = 5   # Périodes pour confirmation
        
        logger.info(f"🎯 EMA Cross Ultra-Précis initialisé pour {symbol}")

    @property
    def name(self) -> str:
        return "EMA_Cross_Ultra_Strategy"
    
    def get_min_data_points(self) -> int:
        return 80  # Minimum pour EMA 50 fiable
    
    def analyze(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Optional[Dict]:
        """
        Analyse EMA Cross ultra-précise avec validation complète
        """
        try:
            if len(df) < self.get_min_data_points():
                return None
            
            # 1. Récupérer les EMAs pré-calculées de la DB
            ema_12 = self._get_current_indicator(indicators, 'ema_12')
            ema_26 = self._get_current_indicator(indicators, 'ema_26')
            ema_50 = self._get_current_indicator(indicators, 'ema_50')
            
            if None in [ema_12, ema_26, ema_50]:
                logger.debug(f"❌ {symbol}: EMAs non disponibles")
                return None
            
            current_price = df['close'].iloc[-1]
            
            # 2. Analyser les croisements EMA
            crossover_analysis = self._analyze_ema_crossover(indicators)
            
            # 3. Analyser l'alignement de tendance
            trend_analysis = self._analyze_trend_alignment(ema_12, ema_26, ema_50, current_price)
            
            # 4. Analyser le momentum et la force
            momentum_analysis = self._analyze_momentum_strength(df, indicators)
            
            # 5. Analyser le contexte de marché
            market_context = self._analyze_market_context(df, indicators)
            
            # 6. Appliquer les filtres ultra-stricts
            if not self._passes_ultra_filters(market_context, momentum_analysis):
                return None
            
            # 7. Logique de signal ultra-sélective
            signal = None
            
            # SIGNAL D'ACHAT EMA - Conditions ultra-strictes
            if (crossover_analysis.get('bullish_crossover') and
                trend_analysis.get('bullish_alignment') and
                momentum_analysis.get('strong_bullish_momentum')):
                
                confidence = self._calculate_buy_confidence(
                    crossover_analysis, trend_analysis, momentum_analysis, market_context
                )
                
                if confidence >= self.min_confidence:
                    signal = self._create_signal(
                        symbol, OrderSide.BUY, current_price, confidence,
                        ema_12, ema_26, ema_50, market_context
                    )
            
            # SIGNAL DE VENTE EMA - Conditions ultra-strictes
            elif (crossover_analysis.get('bearish_crossover') and
                  trend_analysis.get('bearish_alignment') and
                  momentum_analysis.get('strong_bearish_momentum')):
                
                confidence = self._calculate_sell_confidence(
                    crossover_analysis, trend_analysis, momentum_analysis, market_context
                )
                
                if confidence >= self.min_confidence:
                    signal = self._create_signal(
                        symbol, OrderSide.SELL, current_price, confidence,
                        ema_12, ema_26, ema_50, market_context
                    )
            
            if signal:
                logger.info(f"🎯 EMA Cross {symbol}: {signal['side'].value} @ {current_price:.4f} "
                          f"(EMAs: {ema_12:.4f}/{ema_26:.4f}/{ema_50:.4f}, conf: {signal['confidence']:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Erreur EMA Cross Strategy {symbol}: {e}")
            return None
    
    def _get_current_indicator(self, indicators: Dict, key: str) -> Optional[float]:
        """Récupère valeur actuelle d'un indicateur"""
        value = indicators.get(key)
        if value is None:
            return None
        
        if isinstance(value, (list, np.ndarray)) and len(value) > 0:
            return float(value[-1])
        elif isinstance(value, (int, float)):
            return float(value)
        
        return None
    
    def _get_indicator_series(self, indicators: Dict, key: str, length: int) -> Optional[np.ndarray]:
        """Récupère série d'indicateur"""
        value = indicators.get(key)
        if value is None:
            return None
            
        if isinstance(value, (list, np.ndarray)) and len(value) >= length:
            return np.array(value[-length:])
        
        return None
    
    def _analyze_ema_crossover(self, indicators: Dict) -> Dict:
        """Analyse les croisements EMA avec validation de force"""
        try:
            # Récupérer les séries EMA
            ema_12_series = self._get_indicator_series(indicators, 'ema_12', 3)
            ema_26_series = self._get_indicator_series(indicators, 'ema_26', 3)
            
            if ema_12_series is None or ema_26_series is None or len(ema_12_series) < 3:
                return {'bullish_crossover': False, 'bearish_crossover': False}
            
            # Valeurs actuelles et précédentes
            current_fast, prev_fast = ema_12_series[-1], ema_12_series[-2]
            current_slow, prev_slow = ema_26_series[-1], ema_26_series[-2]
            
            # Détecter croisements
            bullish_crossover = (prev_fast <= prev_slow and current_fast > current_slow)
            bearish_crossover = (prev_fast >= prev_slow and current_fast < current_slow)
            
            # Mesurer la force du croisement
            crossover_gap = abs(current_fast - current_slow)
            reference_value = max(abs(current_fast), abs(current_slow), 0.001)
            crossover_strength = crossover_gap / reference_value
            
            # Valider la force minimum
            strong_enough = crossover_strength >= self.min_crossover_strength
            
            # Vérifier la persistance (confirmer direction)
            if len(ema_12_series) >= 3:
                prev2_fast, prev2_slow = ema_12_series[-3], ema_26_series[-3]
                direction_persistent = (
                    (bullish_crossover and current_fast > prev2_fast) or
                    (bearish_crossover and current_fast < prev2_fast)
                )
            else:
                direction_persistent = True
            
            return {
                'bullish_crossover': bullish_crossover and strong_enough and direction_persistent,
                'bearish_crossover': bearish_crossover and strong_enough and direction_persistent,
                'crossover_strength': crossover_strength,
                'direction_persistent': direction_persistent
            }
            
        except Exception as e:
            logger.debug(f"Erreur analyse croisement EMA: {e}")
            return {'bullish_crossover': False, 'bearish_crossover': False}
    
    def _analyze_trend_alignment(self, ema_12: float, ema_26: float, ema_50: float, price: float) -> Dict:
        """Analyse l'alignement des tendances EMA"""
        try:
            # Alignement EMA (ordre croissant = haussier)
            bullish_alignment = (ema_12 > ema_26 > ema_50 and price > ema_12)
            bearish_alignment = (ema_12 < ema_26 < ema_50 and price < ema_12)
            
            # Force de l'alignement
            fast_slow_gap = abs(ema_12 - ema_26) / ema_26
            slow_trend_gap = abs(ema_26 - ema_50) / ema_50
            price_fast_gap = abs(price - ema_12) / ema_12
            
            # Score d'alignement
            alignment_score = 0.5
            if bullish_alignment:
                alignment_score += min(0.3, fast_slow_gap * 50)
                alignment_score += min(0.2, slow_trend_gap * 50)
            elif bearish_alignment:
                alignment_score += min(0.3, fast_slow_gap * 50)
                alignment_score += min(0.2, slow_trend_gap * 50)
            
            return {
                'bullish_alignment': bullish_alignment,
                'bearish_alignment': bearish_alignment,
                'alignment_score': min(1.0, alignment_score),
                'fast_slow_gap': fast_slow_gap,
                'price_position': 'above' if price > ema_12 else 'below'
            }
            
        except Exception as e:
            logger.debug(f"Erreur alignement tendance: {e}")
            return {'bullish_alignment': False, 'bearish_alignment': False}
    
    def _analyze_momentum_strength(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyse la force du momentum pour EMA"""
        try:
            if len(df) < 20:
                return {'strong_bullish_momentum': False, 'strong_bearish_momentum': False}
            
            # RSI pour momentum
            rsi = self._get_current_indicator(indicators, 'rsi_14')
            
            # ADX pour force de tendance
            adx = self._get_current_indicator(indicators, 'adx')
            
            # MACD pour confirmation
            macd_line = self._get_current_indicator(indicators, 'macd_line')
            macd_signal = self._get_current_indicator(indicators, 'macd_signal')
            
            # Analyse momentum
            momentum_score = 0.5
            
            # RSI momentum
            if rsi is not None:
                if 40 <= rsi <= 60:  # Zone neutre favorable
                    momentum_score += 0.15
                elif rsi > 60:  # Momentum haussier
                    momentum_score += 0.1
                elif rsi < 40:  # Momentum baissier
                    momentum_score += 0.1
            
            # ADX force
            if adx is not None and adx > 20:
                momentum_score += min(0.2, adx / 100)
            
            # MACD confirmation
            if macd_line is not None and macd_signal is not None:
                macd_bullish = macd_line > macd_signal
                macd_bearish = macd_line < macd_signal
                momentum_score += 0.1 if (macd_bullish or macd_bearish) else 0
            
            # Conditions momentum fort
            strong_bullish = (
                momentum_score >= self.min_momentum_alignment and
                (rsi is None or rsi >= 45) and
                (macd_line is None or macd_signal is None or macd_line > macd_signal)
            )
            
            strong_bearish = (
                momentum_score >= self.min_momentum_alignment and
                (rsi is None or rsi <= 55) and
                (macd_line is None or macd_signal is None or macd_line < macd_signal)
            )
            
            return {
                'strong_bullish_momentum': strong_bullish,
                'strong_bearish_momentum': strong_bearish,
                'momentum_score': min(1.0, momentum_score),
                'rsi': rsi,
                'adx': adx
            }
            
        except Exception as e:
            logger.debug(f"Erreur analyse momentum: {e}")
            return {'strong_bullish_momentum': False, 'strong_bearish_momentum': False}
    
    def _analyze_market_context(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyse contexte marché pour EMA Cross"""
        try:
            recent_data = df.tail(20)
            
            # Volume
            current_volume = recent_data['volume'].iloc[-1]
            avg_volume = recent_data['volume'].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Volatilité
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std()
            
            # ATR
            atr = self._get_current_indicator(indicators, 'atr_14') or 0
            
            return {
                'volume_ratio': volume_ratio,
                'volatility': volatility,
                'atr': atr,
                'is_high_volume': volume_ratio >= self.min_volume_confirmation,
                'is_stable_volatility': volatility <= self.max_volatility
            }
            
        except Exception:
            return {'volume_ratio': 1.0, 'volatility': 0.03, 'is_high_volume': False}
    
    def _passes_ultra_filters(self, market_context: Dict, momentum_analysis: Dict) -> bool:
        """Filtres ultra-stricts pour la qualité du signal"""
        return (
            # Volume suffisant
            market_context.get('is_high_volume', False) and
            # Volatilité contrôlée
            market_context.get('is_stable_volatility', False) and
            # Momentum validé
            (momentum_analysis.get('strong_bullish_momentum') or 
             momentum_analysis.get('strong_bearish_momentum'))
        )
    
    def _calculate_buy_confidence(self, crossover: Dict, trend: Dict,
                                momentum: Dict, market: Dict) -> float:
        """Confiance pour signal d'achat EMA"""
        confidence = 0.65  # Base élevée
        
        # Force du croisement
        crossover_strength = crossover.get('crossover_strength', 0)
        confidence += min(0.1, crossover_strength * 30)
        
        # Alignement tendance
        if trend.get('bullish_alignment'):
            confidence += min(0.1, trend.get('alignment_score', 0) * 0.1)
        
        # Momentum
        momentum_score = momentum.get('momentum_score', 0)
        confidence += min(0.1, momentum_score * 0.1)
        
        # Volume exceptionnel
        vol_ratio = market.get('volume_ratio', 1)
        if vol_ratio >= 2.0:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _calculate_sell_confidence(self, crossover: Dict, trend: Dict,
                                 momentum: Dict, market: Dict) -> float:
        """Confiance pour signal de vente EMA"""
        confidence = 0.65  # Base élevée
        
        # Force du croisement
        crossover_strength = crossover.get('crossover_strength', 0)
        confidence += min(0.1, crossover_strength * 30)
        
        # Alignement tendance
        if trend.get('bearish_alignment'):
            confidence += min(0.1, trend.get('alignment_score', 0) * 0.1)
        
        # Momentum
        momentum_score = momentum.get('momentum_score', 0)
        confidence += min(0.1, momentum_score * 0.1)
        
        # Volume exceptionnel
        vol_ratio = market.get('volume_ratio', 1)
        if vol_ratio >= 2.0:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _create_signal(self, symbol: str, side: OrderSide, price: float,
                      confidence: float, ema_12: float, ema_26: float,
                      ema_50: float, market: Dict) -> Dict:
        """Crée signal EMA Cross structuré"""
        
        if confidence >= 0.85:
            strength = SignalStrength.VERY_STRONG
        elif confidence >= 0.82:
            strength = SignalStrength.STRONG
        elif confidence >= 0.78:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK
        
        return {
            'strategy': self.name,
            'symbol': symbol,
            'side': side,
            'price': price,
            'confidence': confidence,
            'strength': strength,
            'timestamp': datetime.now(),
            'metadata': {
                'ema_12': ema_12,
                'ema_26': ema_26,
                'ema_50': ema_50,
                'volume_ratio': market.get('volume_ratio', 1),
                'volatility': market.get('volatility', 0),
                'signal_type': 'ema_cross_ultra_precise'
            }
        }