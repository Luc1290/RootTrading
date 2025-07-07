"""
Stratégie Bollinger Bands Simple
Bollinger pur : Prix touche bande inférieure = BUY, Prix touche bande supérieure = SELL
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
    Stratégie Bollinger Bands Simple - Bandes pures sans filtres complexes
    BUY: Prix touche ou dépasse bande inférieure (oversold)
    SELL: Prix touche ou dépasse bande supérieure (overbought)
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        super().__init__(symbol, params)
        
        # Paramètres Bollinger depuis la DB
        symbol_params = self.params.get(symbol, {}) if self.params else {}
        self.lower_band_threshold = 0.1   # 10% dans la zone basse  
        self.upper_band_threshold = 0.9   # 90% dans la zone haute
        self.breakout_min = symbol_params.get('breakout_min', 0.08)
        
        logger.info(f"🎯 Bollinger Simple initialisé pour {symbol}")
    
    @property
    def name(self) -> str:
        return "Bollinger_Ultra_Strategy"
    
    def get_min_data_points(self) -> int:
        return 25  # Minimum pour Bollinger stable
    
    def analyze(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Optional[Dict]:
        """Analyse Bollinger Bands simple - bandes pures"""
        try:
            if len(df) < self.get_min_data_points():
                return None
            
            # Récupérer Bollinger position pré-calculée (0 = bande inf, 1 = bande sup)
            bb_position = self._get_current_indicator(indicators, 'bb_position')
            if bb_position is None:
                logger.debug(f"❌ {symbol}: Bollinger position non disponible")
                return None
            
            # Récupérer les bandes pour context
            bb_upper = self._get_current_indicator(indicators, 'bb_upper')
            bb_lower = self._get_current_indicator(indicators, 'bb_lower')
            bb_middle = self._get_current_indicator(indicators, 'bb_middle')
            
            current_price = df['close'].iloc[-1]
            
            signal = None
            
            # SIGNAL D'ACHAT - Prix près de la bande inférieure (oversold)
            if bb_position <= self.lower_band_threshold:
                confidence = min(0.95, (self.lower_band_threshold - bb_position) * 6 + 0.6)
                signal = self.create_signal(
                    side=OrderSide.BUY,
                    price=current_price,
                    confidence=confidence,
                    metadata={
                        'bb_position': bb_position,
                        'bb_upper': bb_upper,
                        'bb_lower': bb_lower,
                        'bb_middle': bb_middle,
                        'reason': f'Bollinger oversold (position: {bb_position:.2f} <= {self.lower_band_threshold})'
                    }
                )
            
            # SIGNAL DE VENTE - Prix près de la bande supérieure (overbought)
            elif bb_position >= self.upper_band_threshold:
                confidence = min(0.95, (bb_position - self.upper_band_threshold) * 6 + 0.6)
                signal = self.create_signal(
                    side=OrderSide.SELL,
                    price=current_price,
                    confidence=confidence,
                    metadata={
                        'bb_position': bb_position,
                        'bb_upper': bb_upper,
                        'bb_lower': bb_lower,
                        'bb_middle': bb_middle,
                        'reason': f'Bollinger overbought (position: {bb_position:.2f} >= {self.upper_band_threshold})'
                    }
                )
            
            if signal:
                logger.info(f"🎯 Bollinger {symbol}: {signal.side} @ {current_price:.4f} "
                          f"(position: {bb_position:.2f}, conf: {signal.confidence:.2f}, strength: {signal.strength})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Erreur Bollinger Strategy {symbol}: {e}")
            return None
    
    def _get_current_indicator(self, indicators: Dict, key: str) -> Optional[float]:
        """Récupère la valeur actuelle d'un indicateur"""
        value = indicators.get(key)
        if value is None:
            return None
        
        if isinstance(value, (list, np.ndarray)) and len(value) > 0:
            return float(value[-1])
        elif isinstance(value, (int, float)):
            return float(value)
        
        return None