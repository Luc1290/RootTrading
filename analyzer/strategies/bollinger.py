"""
Stratégie Bollinger Bands Ultra-Précise
Utilise les bandes de Bollinger pré-calculées avec squeeze detection,
breakouts validés et filtres de volatilité pour des signaux fiables.
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from shared.src.enums import OrderSide, SignalStrength

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class BollingerStrategy(BaseStrategy):
    """
    Stratégie Bollinger Bands Ultra-Précise avec :
    - Détection de squeeze (compression des bandes)
    - Breakouts validés avec volume
    - Mean reversion près des bandes
    - Filtres de volatilité adaptative
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        super().__init__(symbol, params)
        
        # Paramètres configurables par symbole depuis la DB
        symbol_params = self.params.get(symbol, {}) if self.params else {}
        
        # Paramètres Bollinger ultra-précis configurables
        self.squeeze_threshold = symbol_params.get('squeeze_max', 0.02)
        self.breakout_strength_min = symbol_params.get('breakout_min', 0.15)
        self.min_volume_confirmation = symbol_params.get('volume_ratio_min', 1.6)
        self.mean_reversion_zone = 0.05       # 5% de la bande pour reversion
        
        # Filtres de qualité
        self.min_confidence = 0.75
        self.volatility_range = (0.01, 0.08)  # Plage volatilité optimale
        
        logger.info(f"🎯 Bollinger Ultra-Précis initialisé pour {symbol}")
    
    @property
    def name(self) -> str:
        return "Bollinger_Ultra_Strategy"
    
    def get_min_data_points(self) -> int:
        return 40  # Minimum pour BB stable
    
    def analyze(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Optional[Dict]:
        """Analyse Bollinger Bands ultra-précise"""
        try:
            if len(df) < self.get_min_data_points():
                return None
            
            # Récupérer indicateurs BB de la DB
            bb_upper = self._get_current_indicator(indicators, 'bb_upper')
            bb_middle = self._get_current_indicator(indicators, 'bb_middle')
            bb_lower = self._get_current_indicator(indicators, 'bb_lower')
            bb_width = self._get_current_indicator(indicators, 'bb_width')
            bb_position = self._get_current_indicator(indicators, 'bb_position')
            
            if None in [bb_upper, bb_middle, bb_lower, bb_width]:
                return None
            
            current_price = df['close'].iloc[-1]
            
            # Analyses spécialisées
            squeeze_analysis = self._analyze_squeeze(indicators)
            breakout_analysis = self._analyze_breakout(df, indicators)
            reversion_analysis = self._analyze_mean_reversion(current_price, bb_upper, bb_middle, bb_lower, bb_position)
            market_context = self._analyze_market_context(df, indicators)
            
            # Filtres de qualité
            if not self._passes_quality_filters(market_context):
                return None
            
            signal = None
            
            # BREAKOUT HAUSSIER après squeeze
            if (squeeze_analysis.get('squeeze_ending') and
                breakout_analysis.get('bullish_breakout') and
                market_context['volume_ratio'] >= self.min_volume_confirmation):
                
                confidence = self._calculate_breakout_confidence(
                    squeeze_analysis, breakout_analysis, market_context, "bullish"
                )
                
                if confidence >= self.min_confidence:
                    signal = self._create_signal(
                        symbol, OrderSide.BUY, current_price, confidence,
                        "squeeze_breakout", bb_width, market_context
                    )
            
            # BREAKOUT BAISSIER après squeeze
            elif (squeeze_analysis.get('squeeze_ending') and
                  breakout_analysis.get('bearish_breakout') and
                  market_context['volume_ratio'] >= self.min_volume_confirmation):
                
                confidence = self._calculate_breakout_confidence(
                    squeeze_analysis, breakout_analysis, market_context, "bearish"
                )
                
                if confidence >= self.min_confidence:
                    signal = self._create_signal(
                        symbol, OrderSide.SELL, current_price, confidence,
                        "squeeze_breakout", bb_width, market_context
                    )
            
            # MEAN REVERSION (rebond sur bandes)
            elif reversion_analysis.get('oversold_reversion'):
                confidence = self._calculate_reversion_confidence(reversion_analysis, market_context, "bullish")
                if confidence >= self.min_confidence:
                    signal = self._create_signal(
                        symbol, OrderSide.BUY, current_price, confidence,
                        "mean_reversion", bb_width, market_context
                    )
            
            elif reversion_analysis.get('overbought_reversion'):
                confidence = self._calculate_reversion_confidence(reversion_analysis, market_context, "bearish")
                if confidence >= self.min_confidence:
                    signal = self._create_signal(
                        symbol, OrderSide.SELL, current_price, confidence,
                        "mean_reversion", bb_width, market_context
                    )
            
            if signal:
                logger.info(f"🎯 Bollinger {symbol}: {signal['side'].value} @ {current_price:.4f} "
                          f"(type: {signal['metadata']['signal_type']}, conf: {signal['confidence']:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Erreur Bollinger Strategy {symbol}: {e}")
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
    
    def _analyze_squeeze(self, indicators: Dict) -> Dict:
        """Détecte les phases de squeeze (compression des bandes)"""
        try:
            bb_width_series = self._get_indicator_series(indicators, 'bb_width', 10)
            if bb_width_series is None:
                return {'squeeze_active': False, 'squeeze_ending': False}
            
            current_width = bb_width_series[-1]
            avg_width = np.mean(bb_width_series[:-1])
            
            # Squeeze = largeur actuelle très faible vs moyenne
            squeeze_active = current_width <= self.squeeze_threshold
            
            # Squeeze ending = largeur commence à s'élargir après compression
            squeeze_ending = False
            if len(bb_width_series) >= 3:
                width_trend = bb_width_series[-1] - bb_width_series[-3]
                squeeze_ending = squeeze_active and width_trend > 0
            
            return {
                'squeeze_active': squeeze_active,
                'squeeze_ending': squeeze_ending,
                'width_ratio': current_width / avg_width if avg_width > 0 else 1,
                'width_expansion': width_trend if 'width_trend' in locals() else 0
            }
            
        except Exception as e:
            logger.debug(f"Erreur analyse squeeze: {e}")
            return {'squeeze_active': False, 'squeeze_ending': False}
    
    def _analyze_breakout(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyse les breakouts des bandes de Bollinger"""
        try:
            if len(df) < 5:
                return {'bullish_breakout': False, 'bearish_breakout': False}
            
            recent_prices = df['close'].tail(3).values
            bb_upper_series = self._get_indicator_series(indicators, 'bb_upper', 3)
            bb_lower_series = self._get_indicator_series(indicators, 'bb_lower', 3)
            
            if bb_upper_series is None or bb_lower_series is None:
                return {'bullish_breakout': False, 'bearish_breakout': False}
            
            current_price = recent_prices[-1]
            prev_price = recent_prices[-2]
            current_upper = bb_upper_series[-1]
            current_lower = bb_lower_series[-1]
            prev_upper = bb_upper_series[-2]
            prev_lower = bb_lower_series[-2]
            
            # Breakout haussier = prix passe au-dessus de la bande supérieure
            bullish_breakout = (prev_price <= prev_upper and current_price > current_upper)
            
            # Breakout baissier = prix passe en-dessous de la bande inférieure
            bearish_breakout = (prev_price >= prev_lower and current_price < current_lower)
            
            # Force du breakout
            if bullish_breakout:
                breakout_strength = (current_price - current_upper) / current_upper
            elif bearish_breakout:
                breakout_strength = (current_lower - current_price) / current_lower
            else:
                breakout_strength = 0
            
            return {
                'bullish_breakout': bullish_breakout and breakout_strength >= self.breakout_strength_min,
                'bearish_breakout': bearish_breakout and breakout_strength >= self.breakout_strength_min,
                'breakout_strength': breakout_strength
            }
            
        except Exception as e:
            logger.debug(f"Erreur analyse breakout: {e}")
            return {'bullish_breakout': False, 'bearish_breakout': False}
    
    def _analyze_mean_reversion(self, price: float, bb_upper: float, bb_middle: float, 
                              bb_lower: float, bb_position: Optional[float]) -> Dict:
        """Analyse les opportunités de mean reversion"""
        try:
            if bb_position is None:
                # Calculer position manuellement
                bb_range = bb_upper - bb_lower
                if bb_range > 0:
                    bb_position = (price - bb_lower) / bb_range
                else:
                    bb_position = 0.5
            
            # Zones de reversion
            oversold_zone = bb_position <= self.mean_reversion_zone  # Près bande basse
            overbought_zone = bb_position >= (1 - self.mean_reversion_zone)  # Près bande haute
            
            # Conditions de reversion (prix s'éloigne des bandes vers la moyenne)
            distance_to_middle = abs(price - bb_middle) / bb_middle
            
            # Reversion haussière = prix très bas qui remonte vers moyenne
            oversold_reversion = (oversold_zone and distance_to_middle > 0.01)
            
            # Reversion baissière = prix très haut qui redescend vers moyenne  
            overbought_reversion = (overbought_zone and distance_to_middle > 0.01)
            
            return {
                'oversold_reversion': oversold_reversion,
                'overbought_reversion': overbought_reversion,
                'bb_position': bb_position,
                'distance_to_middle': distance_to_middle
            }
            
        except Exception as e:
            logger.debug(f"Erreur analyse reversion: {e}")
            return {'oversold_reversion': False, 'overbought_reversion': False}
    
    def _analyze_market_context(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyse contexte marché pour Bollinger"""
        try:
            recent_data = df.tail(15)
            
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
                'is_volatile_period': volatility > self.volatility_range[1],
                'is_stable_period': volatility < self.volatility_range[0]
            }
            
        except Exception:
            return {'volume_ratio': 1.0, 'volatility': 0.03}
    
    def _passes_quality_filters(self, market_context: Dict) -> bool:
        """Filtres de qualité du marché"""
        volatility = market_context.get('volatility', 0)
        return (
            self.volatility_range[0] <= volatility <= self.volatility_range[1] and
            not market_context.get('is_stable_period', False)  # Éviter marchés trop calmes
        )
    
    def _calculate_breakout_confidence(self, squeeze: Dict, breakout: Dict, 
                                     market: Dict, direction: str) -> float:
        """Confiance pour signaux de breakout"""
        confidence = 0.7  # Base élevée
        
        # Force du squeeze précédent
        if squeeze.get('squeeze_ending'):
            confidence += 0.1
        
        # Force du breakout
        breakout_strength = breakout.get('breakout_strength', 0)
        confidence += min(0.1, breakout_strength * 2)
        
        # Volume exceptionnel
        vol_ratio = market.get('volume_ratio', 1)
        if vol_ratio >= 2.0:
            confidence += 0.1
        elif vol_ratio >= 1.6:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _calculate_reversion_confidence(self, reversion: Dict, market: Dict, direction: str) -> float:
        """Confiance pour signaux de mean reversion"""
        confidence = 0.6  # Base modérée
        
        # Distance des bandes
        bb_position = reversion.get('bb_position', 0.5)
        if direction == "bullish" and bb_position <= 0.1:
            confidence += 0.15
        elif direction == "bearish" and bb_position >= 0.9:
            confidence += 0.15
        
        # Distance de la moyenne
        distance = reversion.get('distance_to_middle', 0)
        confidence += min(0.1, distance * 5)
        
        return min(1.0, confidence)
    
    def _create_signal(self, symbol: str, side: OrderSide, price: float,
                      confidence: float, signal_type: str, bb_width: float,
                      market: Dict) -> Dict:
        """Crée signal Bollinger structuré"""
        
        if confidence >= 0.85:
            strength = SignalStrength.VERY_STRONG
        elif confidence >= 0.8:
            strength = SignalStrength.STRONG
        elif confidence >= 0.75:
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
                'signal_type': signal_type,
                'bb_width': bb_width,
                'volume_ratio': market.get('volume_ratio', 1),
                'volatility': market.get('volatility', 0)
            }
        }