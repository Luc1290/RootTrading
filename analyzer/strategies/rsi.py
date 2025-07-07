"""
Stratégie RSI Simple
RSI pur : RSI < 30 = BUY, RSI > 70 = SELL
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from shared.src.config import get_strategy_param
from shared.src.enums import OrderSide, SignalStrength
from shared.src.schemas import StrategySignal

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class RSIStrategy(BaseStrategy):
    """
    Stratégie RSI Simple - RSI pur sans filtres complexes
    BUY: RSI < 30 (oversold)
    SELL: RSI > 70 (overbought)
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        super().__init__(symbol, params)
        
        # Paramètres RSI depuis la DB
        symbol_params = self.params.get(symbol, {}) if self.params else {}
        self.oversold_threshold = symbol_params.get('rsi_oversold', 30)
        self.overbought_threshold = symbol_params.get('rsi_overbought', 70)
        
        logger.info(f"🎯 RSI Simple initialisé pour {symbol} (seuils: {self.oversold_threshold}/{self.overbought_threshold})")
    
    @property
    def name(self) -> str:
        return "RSI_Ultra_Strategy"
    
    def get_min_data_points(self) -> int:
        return 15  # Minimum pour RSI
    
    def analyze(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Optional[Dict]:
        """
        Analyse RSI simple - seuils purs
        """
        try:
            if len(df) < self.get_min_data_points():
                return None
            
            # Récupérer RSI pré-calculé
            current_rsi = self._get_current_indicator(indicators, 'rsi_14')
            if current_rsi is None:
                logger.debug(f"❌ {symbol}: RSI non disponible")
                return None
            
            current_price = df['close'].iloc[-1]
            
            signal = None
            
            # SIGNAL D'ACHAT - RSI oversold
            if current_rsi <= self.oversold_threshold:
                confidence = min(0.95, (self.oversold_threshold - current_rsi) / self.oversold_threshold * 1.2 + 0.5)
                signal = self.create_signal(
                    side=OrderSide.BUY,
                    price=current_price,
                    confidence=confidence,
                    metadata={
                        'rsi': current_rsi,
                        'reason': f'RSI oversold ({current_rsi:.1f} <= {self.oversold_threshold})'
                    }
                )
            
            # SIGNAL DE VENTE - RSI overbought  
            elif current_rsi >= self.overbought_threshold:
                confidence = min(0.95, (current_rsi - self.overbought_threshold) / (100 - self.overbought_threshold) * 1.2 + 0.5)
                signal = self.create_signal(
                    side=OrderSide.SELL,
                    price=current_price,
                    confidence=confidence,
                    metadata={
                        'rsi': current_rsi,
                        'reason': f'RSI overbought ({current_rsi:.1f} >= {self.overbought_threshold})'
                    }
                )
            
            if signal:
                logger.info(f"🎯 RSI {symbol}: {signal.side.value} @ {current_price:.4f} "
                          f"(RSI: {current_rsi:.1f}, conf: {signal.confidence:.2f}, strength: {signal.strength.value})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Erreur RSI Strategy {symbol}: {e}")
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