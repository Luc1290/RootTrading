"""
Stratégie Reversal Divergence Simple
Divergence pure : Prix monte + RSI baisse = SELL, Prix baisse + RSI monte = BUY
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
    Stratégie Reversal Divergence Simple - Divergence pure sans filtres complexes
    BUY: Divergence bullish (prix baisse + RSI monte)
    SELL: Divergence bearish (prix monte + RSI baisse)
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        super().__init__(symbol, params)
        
        # Paramètres divergence depuis la DB
        symbol_params = self.params.get(symbol, {}) if self.params else {}
        self.lookback_periods = 15        # Périodes pour calculer tendances
        self.min_price_change = 1.0       # 1% minimum de mouvement prix
        self.min_rsi_change = 5.0         # 5 points minimum de mouvement RSI
        self.div_strength_min = symbol_params.get('div_strength_min', 0.15)
        
        logger.info(f"🎯 Reversal Divergence Simple initialisé pour {symbol}")
    
    @property
    def name(self) -> str:
        return "Reversal_Divergence_Ultra_Strategy"
    
    def get_min_data_points(self) -> int:
        return 30  # Minimum pour calcul divergence
    
    def analyze(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Optional[Dict]:
        """
        Analyse Divergence simple - divergences pures
        """
        try:
            if len(df) < self.get_min_data_points():
                return None
            
            # Récupérer RSI actuel et historique
            rsi_values = self._get_indicator_history(indicators, 'rsi_14', self.lookback_periods)
            if rsi_values is None or len(rsi_values) < self.lookback_periods:
                logger.debug(f"❌ {symbol}: RSI historique insuffisant")
                return None
            
            current_price = df['close'].iloc[-1]
            recent_prices = df['close'].tail(self.lookback_periods).values
            
            # Calculer tendances sur la période
            price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100
            rsi_trend = rsi_values[-1] - rsi_values[0]
            
            signal = None
            
            # SIGNAL D'ACHAT - Divergence bullish (prix baisse, RSI monte)
            if (price_trend <= -self.min_price_change and rsi_trend >= self.min_rsi_change):
                confidence = min(0.9, (abs(price_trend) + abs(rsi_trend)) / 20 + 0.6)
                signal = {
                    'symbol': symbol,
                    'side': OrderSide.BUY,
                    'price': current_price,
                    'confidence': confidence,
                    'strength': SignalStrength.MODERATE,
                    'strategy': self.name,
                    'timestamp': datetime.now(),
                    'metadata': {
                        'price_trend': price_trend,
                        'rsi_trend': rsi_trend,
                        'current_rsi': rsi_values[-1],
                        'reason': f'Divergence bullish (prix: {price_trend:.1f}%, RSI: +{rsi_trend:.1f})'
                    }
                }
            
            # SIGNAL DE VENTE - Divergence bearish (prix monte, RSI baisse)
            elif (price_trend >= self.min_price_change and rsi_trend <= -self.min_rsi_change):
                confidence = min(0.9, (abs(price_trend) + abs(rsi_trend)) / 20 + 0.6)
                signal = {
                    'symbol': symbol,
                    'side': OrderSide.SELL,
                    'price': current_price,
                    'confidence': confidence,
                    'strength': SignalStrength.MODERATE,
                    'strategy': self.name,
                    'timestamp': datetime.now(),
                    'metadata': {
                        'price_trend': price_trend,
                        'rsi_trend': rsi_trend,
                        'current_rsi': rsi_values[-1],
                        'reason': f'Divergence bearish (prix: +{price_trend:.1f}%, RSI: {rsi_trend:.1f})'
                    }
                }
            
            if signal:
                logger.info(f"🎯 Divergence {symbol}: {signal['side'].value} @ {current_price:.4f} "
                          f"(prix: {price_trend:.1f}%, RSI: {rsi_trend:.1f}, conf: {signal['confidence']:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Erreur Reversal Divergence Strategy {symbol}: {e}")
            return None
    
    def _get_indicator_history(self, indicators: Dict, key: str, periods: int) -> Optional[List[float]]:
        """Récupère l'historique d'un indicateur"""
        value = indicators.get(key)
        if value is None:
            return None
        
        if isinstance(value, (list, np.ndarray)) and len(value) >= periods:
            return [float(v) for v in value[-periods:]]
        
        return None