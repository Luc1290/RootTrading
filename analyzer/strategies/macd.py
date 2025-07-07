"""
Stratégie MACD Simple
MACD pur : histogram > 0 = BUY, histogram < 0 = SELL (changement de direction)
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
from shared.src.schemas import StrategySignal

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MACDStrategy(BaseStrategy):
    """
    Stratégie MACD Simple - Crossover pur sans filtres complexes
    BUY: MACD histogram devient positif (momentum haussier)
    SELL: MACD histogram devient négatif (momentum baissier)
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        super().__init__(symbol, params)
        
        # Paramètres MACD depuis la DB
        symbol_params = self.params.get(symbol, {}) if self.params else {}
        self.min_histogram_threshold = symbol_params.get('cross_force', 0.0001)
        
        logger.info(f"🎯 MACD Simple initialisé pour {symbol}")
    
    @property
    def name(self) -> str:
        return "MACD_Ultra_Strategy"
    
    def get_min_data_points(self) -> int:
        return 30  # Minimum pour MACD stable
    
    def analyze(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Optional[Dict]:
        """
        Analyse MACD simple - crossover pur
        """
        try:
            if len(df) < self.get_min_data_points():
                return None
            
            # Récupérer MACD histogram pré-calculé
            current_histogram = self._get_current_indicator(indicators, 'macd_histogram')
            if current_histogram is None:
                logger.debug(f"❌ {symbol}: MACD histogram non disponible")
                return None
            
            # Récupérer histogram précédent pour détecter changement
            previous_histogram = self._get_previous_indicator(indicators, 'macd_histogram')
            if previous_histogram is None:
                return None
            
            current_price = df['close'].iloc[-1]
            
            signal = None
            
            # SIGNAL D'ACHAT - Histogram devient positif (crossover bullish)
            if (previous_histogram <= 0 and current_histogram > self.min_histogram_threshold):
                confidence = min(0.9, abs(current_histogram) * 1000 + 0.5)
                signal = {
                    'symbol': symbol,
                    'side': OrderSide.BUY,
                    'price': current_price,
                    'confidence': confidence,
                    'strength': SignalStrength.MODERATE,
                    'strategy': self.name,
                    'timestamp': datetime.now(),
                    'metadata': {
                        'macd_histogram': current_histogram,
                        'previous_histogram': previous_histogram,
                        'reason': f'MACD bullish crossover (hist: {previous_histogram:.5f} → {current_histogram:.5f})'
                    }
                }
            
            # SIGNAL DE VENTE - Histogram devient négatif (crossover bearish)
            elif (previous_histogram >= 0 and current_histogram < -self.min_histogram_threshold):
                confidence = min(0.9, abs(current_histogram) * 1000 + 0.5)
                signal = {
                    'symbol': symbol,
                    'side': OrderSide.SELL,
                    'price': current_price,
                    'confidence': confidence,
                    'strength': SignalStrength.MODERATE,
                    'strategy': self.name,
                    'timestamp': datetime.now(),
                    'metadata': {
                        'macd_histogram': current_histogram,
                        'previous_histogram': previous_histogram,
                        'reason': f'MACD bearish crossover (hist: {previous_histogram:.5f} → {current_histogram:.5f})'
                    }
                }
            
            if signal:
                logger.info(f"🎯 MACD {symbol}: {signal['side'].value} @ {current_price:.4f} "
                          f"(hist: {current_histogram:.5f}, conf: {signal['confidence']:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Erreur MACD Strategy {symbol}: {e}")
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
    
    def _get_previous_indicator(self, indicators: Dict, key: str) -> Optional[float]:
        """Récupère la valeur précédente d'un indicateur"""
        value = indicators.get(key)
        if value is None:
            return None
        
        if isinstance(value, (list, np.ndarray)) and len(value) > 1:
            return float(value[-2])
        
        return None