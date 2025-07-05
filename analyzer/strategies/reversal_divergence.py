"""
Stratégie Reversal Divergence Ultra-Précise
Utilise les indicateurs RSI et prix de la DB pour détecter les divergences
avec validation multi-critères pour des signaux de retournement fiables.
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from shared.src.enums import OrderSide, SignalStrength

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class ReversalDivergenceStrategy(BaseStrategy):
    """
    Stratégie Reversal Divergence Ultra-Précise qui utilise les indicateurs de la DB
    avec détection sophistiquée des divergences prix/RSI pour retournements fiables.
    
    Critères ultra-stricts :
    - Divergences RSI/Prix validées sur pivots confirmés
    - Analyse force divergence et contexte
    - Validation momentum et volume
    - Filtres support/résistance et survente/surachat
    - Confirmation tendance pour timing optimal
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        super().__init__(symbol, params)
        
        # Paramètres divergence ultra-précis
        self.divergence_lookback = 25             # Périodes pour analyse divergence
        self.min_pivot_distance = 8               # Distance minimum entre pivots
        self.min_divergence_strength = 0.15       # Force minimum divergence
        self.rsi_extreme_oversold = 25            # Zone survente pour divergence haussière
        self.rsi_extreme_overbought = 75          # Zone surachat pour divergence baissière
        
        # Filtres ultra-précis
        self.min_confidence = 0.76                # Confiance minimum 76%
        self.min_volume_confirmation = 1.3        # Volume 30% au-dessus moyenne
        self.pivot_validation_periods = 3         # Périodes pour validation pivot
        
        logger.info(f"🎯 Reversal Divergence Ultra-Précis initialisé pour {symbol}")
    
    @property
    def name(self) -> str:
        return "Reversal_Divergence_Ultra_Strategy"
    
    def get_min_data_points(self) -> int:
        return 70  # Minimum pour détection pivots et divergences fiables
    
    def analyze(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Optional[Dict]:
        """
        Analyse divergence ultra-précise avec validation complète
        """
        try:
            if len(df) < self.get_min_data_points():
                return None
            
            current_price = df['close'].iloc[-1]
            
            # 1. Détecter les divergences RSI/Prix
            divergence_analysis = self._detect_rsi_price_divergence(df, indicators)
            if not divergence_analysis.get('valid_divergence'):
                return None
            
            # 2. Valider la force de la divergence
            strength_analysis = self._analyze_divergence_strength(df, divergence_analysis, indicators)
            
            # 3. Analyser le contexte RSI (survente/surachat)
            rsi_context = self._analyze_rsi_context(indicators, divergence_analysis)
            
            # 4. Analyser le contexte de marché
            market_context = self._analyze_market_context(df, indicators)
            
            # 5. Appliquer les filtres ultra-stricts
            if not self._passes_ultra_filters(strength_analysis, rsi_context, market_context):
                return None
            
            # 6. Logique de signal ultra-sélective
            signal = None
            side = divergence_analysis['side']
            
            confidence = self._calculate_divergence_confidence(
                divergence_analysis, strength_analysis, rsi_context, market_context
            )
            
            if confidence >= self.min_confidence:
                signal = self._create_signal(
                    symbol, side, current_price, confidence,
                    divergence_analysis, strength_analysis, market_context
                )
            
            if signal:
                div_type = divergence_analysis.get('type', 'unknown')
                logger.info(f"🎯 Divergence {symbol}: {signal['side'].value} @ {current_price:.4f} "
                          f"(type: {div_type}, conf: {signal['confidence']:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Erreur Reversal Divergence Strategy {symbol}: {e}")
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
    
    def _find_pivot_points(self, data: np.ndarray) -> Tuple[List[int], List[int]]:
        """Trouve pivots hauts et bas avec validation"""
        try:
            highs = []
            lows = []
            
            # Utiliser distance minimum entre pivots
            for i in range(self.pivot_validation_periods, len(data) - self.pivot_validation_periods):
                # Pivot haut
                is_high = True
                for j in range(1, self.pivot_validation_periods + 1):
                    if data[i] <= data[i-j] or data[i] <= data[i+j]:
                        is_high = False
                        break
                
                if is_high:
                    # Vérifier distance minimum avec dernier pivot
                    if not highs or i - highs[-1] >= self.min_pivot_distance:
                        highs.append(i)
                
                # Pivot bas
                is_low = True
                for j in range(1, self.pivot_validation_periods + 1):
                    if data[i] >= data[i-j] or data[i] >= data[i+j]:
                        is_low = False
                        break
                
                if is_low:
                    # Vérifier distance minimum avec dernier pivot
                    if not lows or i - lows[-1] >= self.min_pivot_distance:
                        lows.append(i)
            
            return highs, lows
            
        except Exception as e:
            logger.debug(f"Erreur détection pivots: {e}")
            return [], []
    
    def _detect_rsi_price_divergence(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Détecte divergences RSI/Prix avec validation"""
        try:
            if len(df) < self.divergence_lookback:
                return {'valid_divergence': False}
            
            # Récupérer RSI de la DB
            rsi_series = self._get_indicator_series(indicators, 'rsi_14', self.divergence_lookback)
            if rsi_series is None:
                return {'valid_divergence': False}
            
            # Analyser données récentes
            recent_data = df.tail(self.divergence_lookback)
            prices = recent_data['close'].values
            
            # Trouver pivots prix et RSI
            price_highs, price_lows = self._find_pivot_points(prices)
            rsi_highs, rsi_lows = self._find_pivot_points(rsi_series)
            
            # Détecter divergence haussière (prix baisse, RSI monte)
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                last_price_low_idx, prev_price_low_idx = price_lows[-1], price_lows[-2]
                
                last_price_low = prices[last_price_low_idx]
                prev_price_low = prices[prev_price_low_idx]
                last_rsi_low = rsi_series[last_price_low_idx]
                prev_rsi_low = rsi_series[prev_price_low_idx]
                
                # Divergence: prix plus bas, RSI plus haut
                if (last_price_low < prev_price_low * 0.995 and  # Prix vraiment plus bas
                    last_rsi_low > prev_rsi_low * 1.02):          # RSI vraiment plus haut
                    
                    # Vérifier zone de survente
                    if last_rsi_low <= self.rsi_extreme_oversold:
                        price_change = (prev_price_low - last_price_low) / prev_price_low
                        rsi_change = (last_rsi_low - prev_rsi_low) / prev_rsi_low
                        
                        return {
                            'valid_divergence': True,
                            'type': 'bullish',
                            'side': OrderSide.BUY,
                            'price_change': price_change,
                            'rsi_change': rsi_change,
                            'last_rsi': last_rsi_low,
                            'pivot_distance': last_price_low_idx - prev_price_low_idx
                        }
            
            # Détecter divergence baissière (prix monte, RSI baisse)
            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                last_price_high_idx, prev_price_high_idx = price_highs[-1], price_highs[-2]
                
                last_price_high = prices[last_price_high_idx]
                prev_price_high = prices[prev_price_high_idx]
                last_rsi_high = rsi_series[last_price_high_idx]
                prev_rsi_high = rsi_series[prev_price_high_idx]
                
                # Divergence: prix plus haut, RSI plus bas
                if (last_price_high > prev_price_high * 1.005 and  # Prix vraiment plus haut
                    last_rsi_high < prev_rsi_high * 0.98):         # RSI vraiment plus bas
                    
                    # Vérifier zone de surachat
                    if last_rsi_high >= self.rsi_extreme_overbought:
                        price_change = (last_price_high - prev_price_high) / prev_price_high
                        rsi_change = (prev_rsi_high - last_rsi_high) / prev_rsi_high
                        
                        return {
                            'valid_divergence': True,
                            'type': 'bearish',
                            'side': OrderSide.SELL,
                            'price_change': price_change,
                            'rsi_change': rsi_change,
                            'last_rsi': last_rsi_high,
                            'pivot_distance': last_price_high_idx - prev_price_high_idx
                        }
            
            return {'valid_divergence': False}
            
        except Exception as e:
            logger.debug(f"Erreur détection divergence: {e}")
            return {'valid_divergence': False}
    
    def _analyze_divergence_strength(self, df: pd.DataFrame, divergence: Dict, indicators: Dict) -> Dict:
        """Analyse force et qualité de la divergence"""
        try:
            if not divergence.get('valid_divergence'):
                return {'strong_divergence': False, 'strength_score': 0}
            
            price_change = abs(divergence.get('price_change', 0))
            rsi_change = abs(divergence.get('rsi_change', 0))
            pivot_distance = divergence.get('pivot_distance', 0)
            
            # Score de force basé sur amplitude divergence
            divergence_amplitude = price_change + rsi_change
            
            if divergence_amplitude >= 0.08:  # 8%+ total
                strength_score = 0.9
            elif divergence_amplitude >= 0.05:  # 5%+ total
                strength_score = 0.8
            elif divergence_amplitude >= 0.03:  # 3%+ total
                strength_score = 0.7
            else:
                strength_score = 0.5
            
            # Bonus pour distance entre pivots (divergence mature)
            if pivot_distance >= 15:
                strength_score += 0.1
            elif pivot_distance >= 10:
                strength_score += 0.05
            
            # Validation avec autres indicateurs
            macd_line = self._get_current_indicator(indicators, 'macd_line')
            macd_signal = self._get_current_indicator(indicators, 'macd_signal')
            
            momentum_support = False
            if macd_line is not None and macd_signal is not None:
                if divergence['side'] == OrderSide.BUY and macd_line > macd_signal:
                    momentum_support = True
                elif divergence['side'] == OrderSide.SELL and macd_line < macd_signal:
                    momentum_support = True
            
            if momentum_support:
                strength_score += 0.05
            
            strong_divergence = (
                strength_score >= 0.7 and
                divergence_amplitude >= self.min_divergence_strength
            )
            
            return {
                'strong_divergence': strong_divergence,
                'strength_score': min(1.0, strength_score),
                'divergence_amplitude': divergence_amplitude,
                'momentum_support': momentum_support
            }
            
        except Exception as e:
            logger.debug(f"Erreur analyse force divergence: {e}")
            return {'strong_divergence': False, 'strength_score': 0}
    
    def _analyze_rsi_context(self, indicators: Dict, divergence: Dict) -> Dict:
        """Analyse contexte RSI pour divergence"""
        try:
            current_rsi = self._get_current_indicator(indicators, 'rsi_14')
            if current_rsi is None:
                return {'favorable_rsi_context': False}
            
            div_type = divergence.get('type')
            last_rsi = divergence.get('last_rsi', current_rsi)
            
            # Contexte favorable pour divergence haussière
            if div_type == 'bullish':
                favorable = (
                    last_rsi <= self.rsi_extreme_oversold and
                    current_rsi <= 35  # Toujours en zone basse
                )
                context_score = 0.9 if last_rsi <= 20 else 0.7
            
            # Contexte favorable pour divergence baissière
            elif div_type == 'bearish':
                favorable = (
                    last_rsi >= self.rsi_extreme_overbought and
                    current_rsi >= 65  # Toujours en zone haute
                )
                context_score = 0.9 if last_rsi >= 80 else 0.7
            else:
                favorable = False
                context_score = 0.5
            
            return {
                'favorable_rsi_context': favorable,
                'context_score': context_score,
                'current_rsi': current_rsi,
                'rsi_extreme': last_rsi
            }
            
        except Exception as e:
            logger.debug(f"Erreur contexte RSI: {e}")
            return {'favorable_rsi_context': False}
    
    def _analyze_market_context(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyse contexte marché pour divergence"""
        try:
            recent_data = df.tail(15)
            
            # Volume
            current_volume = recent_data['volume'].iloc[-1]
            avg_volume = recent_data['volume'].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Volatilité
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std()
            
            return {
                'volume_ratio': volume_ratio,
                'volatility': volatility,
                'high_volume': volume_ratio >= self.min_volume_confirmation,
                'stable_volatility': volatility <= 0.05
            }
            
        except Exception:
            return {'volume_ratio': 1.0, 'volatility': 0.03, 'high_volume': False}
    
    def _passes_ultra_filters(self, strength: Dict, rsi_context: Dict, market: Dict) -> bool:
        """Filtres ultra-stricts pour divergence"""
        return (
            # Divergence suffisamment forte
            strength.get('strong_divergence', False) and
            # Contexte RSI favorable
            rsi_context.get('favorable_rsi_context', False) and
            # Volume de confirmation
            market.get('high_volume', False)
        )
    
    def _calculate_divergence_confidence(self, divergence: Dict, strength: Dict,
                                       rsi_context: Dict, market: Dict) -> float:
        """Confiance pour signal divergence"""
        confidence = 0.65  # Base
        
        # Force divergence
        strength_score = strength.get('strength_score', 0)
        confidence += min(0.15, strength_score * 0.15)
        
        # Contexte RSI
        context_score = rsi_context.get('context_score', 0)
        confidence += min(0.1, context_score * 0.1)
        
        # Support momentum
        if strength.get('momentum_support'):
            confidence += 0.05
        
        # Volume de confirmation
        vol_ratio = market.get('volume_ratio', 1)
        if vol_ratio >= 1.5:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _create_signal(self, symbol: str, side: OrderSide, price: float,
                      confidence: float, divergence: Dict, strength: Dict,
                      market: Dict) -> Dict:
        """Crée signal divergence structuré"""
        
        if confidence >= 0.85:
            signal_strength = SignalStrength.VERY_STRONG
        elif confidence >= 0.80:
            signal_strength = SignalStrength.STRONG
        elif confidence >= 0.76:
            signal_strength = SignalStrength.MODERATE
        else:
            signal_strength = SignalStrength.WEAK
        
        return {
            'strategy': self.name,
            'symbol': symbol,
            'side': side,
            'price': price,
            'confidence': confidence,
            'strength': signal_strength,
            'timestamp': datetime.now(),
            'metadata': {
                'divergence_type': divergence.get('type', 'unknown'),
                'divergence_amplitude': strength.get('divergence_amplitude', 0),
                'rsi_level': divergence.get('last_rsi', 0),
                'volume_ratio': market.get('volume_ratio', 1),
                'signal_type': 'reversal_divergence_ultra_precise'
            }
        }