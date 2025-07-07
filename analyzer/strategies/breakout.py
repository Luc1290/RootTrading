"""
Stratégie Breakout Simple
Breakout pur : Prix casse résistance = BUY, Prix casse support = SELL
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

class BreakoutStrategy(BaseStrategy):
    """
    Stratégie Breakout Simple - Cassure pure sans filtres complexes
    BUY: Prix casse au-dessus de la résistance (high récent)
    SELL: Prix casse en-dessous du support (low récent)
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        super().__init__(symbol, params)
        
        # Paramètres breakout depuis la DB
        symbol_params = self.params.get(symbol, {}) if self.params else {}
        self.lookback_periods = 20        # Périodes pour trouver support/résistance
        self.min_breakout_percent = symbol_params.get('breakout_min', 0.2)
        
        logger.info(f"🎯 Breakout Simple initialisé pour {symbol}")
    
    @property
    def name(self) -> str:
        return "Breakout_Ultra_Strategy"
    
    def get_min_data_points(self) -> int:
        return 30  # Minimum pour détection breakout
    
    def analyze(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Optional[Dict]:
        """
        Analyse Breakout simple - cassures pures
        """
        try:
            if len(df) < self.get_min_data_points():
                return None
            
            current_price = df['close'].iloc[-1]
            recent_data = df.tail(self.lookback_periods)
            
            # Calculer support et résistance récents
            resistance = recent_data['high'].max()
            support = recent_data['low'].min()
            
            signal = None
            
            # SIGNAL D'ACHAT - Breakout au-dessus de la résistance
            breakout_above_percent = (current_price - resistance) / resistance * 100
            if breakout_above_percent >= self.min_breakout_percent:
                confidence = min(0.95, breakout_above_percent / 1.8 + 0.6)
                signal = self.create_signal(
                    side=OrderSide.BUY,
                    price=current_price,
                    confidence=confidence,
                    metadata={
                        'resistance': resistance,
                        'support': support,
                        'breakout_percent': breakout_above_percent,
                        'reason': f'Breakout résistance ({current_price:.4f} > {resistance:.4f}, +{breakout_above_percent:.2f}%)'
                    }
                )
            
            # SIGNAL DE VENTE - Breakdown en-dessous du support
            breakdown_below_percent = (support - current_price) / support * 100
            if breakdown_below_percent >= self.min_breakout_percent:
                confidence = min(0.95, breakdown_below_percent / 1.8 + 0.6)
                signal = self.create_signal(
                    side=OrderSide.SELL,
                    price=current_price,
                    confidence=confidence,
                    metadata={
                        'resistance': resistance,
                        'support': support,
                        'breakdown_percent': breakdown_below_percent,
                        'reason': f'Breakdown support ({current_price:.4f} < {support:.4f}, -{breakdown_below_percent:.2f}%)'
                    }
                )
            
            if signal:
                logger.info(f"🎯 Breakout {symbol}: {signal.side} @ {current_price:.4f} "
                          f"(S: {support:.4f}, R: {resistance:.4f}, conf: {signal.confidence:.2f}, strength: {signal.strength})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Erreur Breakout Strategy {symbol}: {e}")
            return None