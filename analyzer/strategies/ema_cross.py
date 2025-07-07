"""
Stratégie EMA Cross Simple
EMA pur : EMA12 croise au-dessus EMA26 = BUY, EMA12 croise en-dessous EMA26 = SELL
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
    Stratégie EMA Cross Simple - Croisement pur sans filtres complexes
    BUY: EMA12 croise au-dessus EMA26 (golden cross)
    SELL: EMA12 croise en-dessous EMA26 (death cross)
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        super().__init__(symbol, params)
        
        # Paramètres EMA depuis la DB
        symbol_params = self.params.get(symbol, {}) if self.params else {}
        self.min_gap_percent = symbol_params.get('ema_gap_min', 0.0015)
        
        logger.info(f"🎯 EMA Cross Simple initialisé pour {symbol}")

    @property
    def name(self) -> str:
        return "EMA_Cross_Ultra_Strategy"
    
    def get_min_data_points(self) -> int:
        return 30  # Minimum pour EMA stable
    
    def analyze(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Optional[Dict]:
        """
        Analyse EMA Cross simple - croisements purs
        """
        try:
            if len(df) < self.get_min_data_points():
                return None
            
            # Récupérer EMAs pré-calculées
            current_ema12 = self._get_current_indicator(indicators, 'ema_12')
            current_ema26 = self._get_current_indicator(indicators, 'ema_26')
            
            if current_ema12 is None or current_ema26 is None:
                logger.debug(f"❌ {symbol}: EMAs non disponibles")
                return None
            
            # Récupérer EMAs précédentes pour détecter croisement
            previous_ema12 = self._get_previous_indicator(indicators, 'ema_12')
            previous_ema26 = self._get_previous_indicator(indicators, 'ema_26')
            
            if previous_ema12 is None or previous_ema26 is None:
                return None
            
            current_price = df['close'].iloc[-1]
            
            signal = None
            
            # SIGNAL D'ACHAT - Golden Cross (EMA12 croise au-dessus EMA26)
            if (previous_ema12 <= previous_ema26 and current_ema12 > current_ema26):
                gap_percent = abs(current_ema12 - current_ema26) / current_ema26
                confidence = min(0.95, gap_percent * 120 + 0.6)
                signal = self.create_signal(
                    side=OrderSide.BUY,
                    price=current_price,
                    confidence=confidence,
                    metadata={
                        'ema12': current_ema12,
                        'ema26': current_ema26,
                        'gap_percent': gap_percent * 100,
                        'reason': f'EMA Golden Cross (12: {current_ema12:.4f} > 26: {current_ema26:.4f})'
                    }
                )
            
            # SIGNAL DE VENTE - Death Cross (EMA12 croise en-dessous EMA26)
            elif (previous_ema12 >= previous_ema26 and current_ema12 < current_ema26):
                gap_percent = abs(current_ema26 - current_ema12) / current_ema26
                confidence = min(0.95, gap_percent * 120 + 0.6)
                signal = self.create_signal(
                    side=OrderSide.SELL,
                    price=current_price,
                    confidence=confidence,
                    metadata={
                        'ema12': current_ema12,
                        'ema26': current_ema26,
                        'gap_percent': gap_percent * 100,
                        'reason': f'EMA Death Cross (12: {current_ema12:.4f} < 26: {current_ema26:.4f})'
                    }
                )
            
            if signal:
                logger.info(f"🎯 EMA Cross {symbol}: {signal.side} @ {current_price:.4f} "
                          f"(12: {current_ema12:.4f}, 26: {current_ema26:.4f}, conf: {signal.confidence:.2f}, strength: {signal.strength})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Erreur EMA Cross Strategy {symbol}: {e}")
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