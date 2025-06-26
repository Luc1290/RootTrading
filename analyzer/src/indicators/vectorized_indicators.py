"""
Calculs vectorisés et optimisés des indicateurs techniques
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import talib
import pandas_ta as ta
from .indicator_cache import indicator_cache
import logging

logger = logging.getLogger(__name__)

class VectorizedIndicators:
    """Calculs optimisés et vectorisés des indicateurs techniques"""
    
    @staticmethod
    def compute_all_indicators(df: pd.DataFrame, symbol: str) -> Dict[str, np.ndarray]:
        """
        Calcule tous les indicateurs en une seule passe vectorisée
        
        Returns:
            Dict avec tous les indicateurs calculés
        """
        start_time = pd.Timestamp.now()
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        indicators = {}
        
        # RSI avec cache
        indicators['rsi_14'] = indicator_cache.get(
            'RSI', close, 
            lambda x: talib.RSI(x, timeperiod=14),
            timeperiod=14
        )
        
        indicators['rsi_21'] = indicator_cache.get(
            'RSI', close,
            lambda x: talib.RSI(x, timeperiod=21),
            timeperiod=21
        )
        
        # ATR avec cache (nécessite high, low, close)
        atr_key = f"ATR_{symbol}"
        indicators['atr_14'] = indicator_cache.get(
            atr_key, close,
            lambda _: talib.ATR(high, low, close, timeperiod=14),
            timeperiod=14, high=high[-14:], low=low[-14:]
        )
        
        # Bollinger Bands - calcul une seule fois
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            close, timeperiod=20, nbdevup=2, nbdevdn=2
        )
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        indicators['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # MACD - calcul une seule fois
        macd, macd_signal, macd_hist = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        indicators['macd'] = macd
        indicators['macd_signal'] = macd_signal
        indicators['macd_hist'] = macd_hist
        
        # EMAs multiples - calculées en une passe
        ema_periods = [9, 21, 50, 200]
        for period in ema_periods:
            indicators[f'ema_{period}'] = talib.EMA(close, timeperiod=period)
        
        # SMAs pour certaines stratégies
        sma_periods = [20, 50]
        for period in sma_periods:
            indicators[f'sma_{period}'] = talib.SMA(close, timeperiod=period)
        
        # Volume indicators
        indicators['volume_sma_20'] = talib.SMA(volume, timeperiod=20)
        indicators['obv'] = talib.OBV(close, volume)
        
        # Stochastic
        slowk, slowd = talib.STOCH(
            high, low, close,
            fastk_period=14, slowk_period=3, slowd_period=3
        )
        indicators['stoch_k'] = slowk
        indicators['stoch_d'] = slowd
        
        # ADX pour la force de tendance
        indicators['adx'] = indicator_cache.get(
            'ADX', close,
            lambda _: talib.ADX(high, low, close, timeperiod=14),
            timeperiod=14, high=high[-14:], low=low[-14:]
        )
        
        # Supertrend avec pandas_ta
        try:
            st_df = pd.DataFrame({'high': high, 'low': low, 'close': close})
            supertrend = ta.supertrend(
                st_df['high'], st_df['low'], st_df['close'],
                length=10, multiplier=3.0
            )
            if supertrend is not None:
                indicators['supertrend'] = supertrend['SUPERT_10_3.0'].values
                indicators['supertrend_direction'] = supertrend['SUPERTd_10_3.0'].values
        except Exception as e:
            logger.debug(f"Supertrend calculation skipped: {e}")
        
        # Statistiques de performance
        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
        logger.debug(f"Calcul vectorisé de {len(indicators)} indicateurs en {elapsed:.3f}s")
        
        return indicators
    
    @staticmethod
    def compute_momentum_indicators(df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calcule spécifiquement les indicateurs de momentum"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        return {
            'rsi_14': talib.RSI(close, timeperiod=14),
            'rsi_21': talib.RSI(close, timeperiod=21),
            'mfi': talib.MFI(high, low, close, df['volume'].values, timeperiod=14),
            'cci': talib.CCI(high, low, close, timeperiod=20),
            'roc': talib.ROC(close, timeperiod=10),
            'williams_r': talib.WILLR(high, low, close, timeperiod=14)
        }
    
    @staticmethod
    def compute_volatility_indicators(df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calcule spécifiquement les indicateurs de volatilité"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        # ATR multiple periods
        atr_periods = [14, 21, 50]
        indicators = {}
        
        for period in atr_periods:
            indicators[f'atr_{period}'] = indicator_cache.get(
                f'ATR_{period}', close,
                lambda _: talib.ATR(high, low, close, timeperiod=period),
                timeperiod=period
            )
        
        # Volatilité historique
        returns = np.diff(np.log(close))
        indicators['historical_volatility'] = pd.Series(returns).rolling(20).std().values * np.sqrt(252)
        
        return indicators
    
    @staticmethod
    def compute_trend_indicators(df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calcule spécifiquement les indicateurs de tendance"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        indicators = {}
        
        # EMAs multiples
        ema_periods = [9, 21, 50, 100, 200]
        for period in ema_periods:
            indicators[f'ema_{period}'] = talib.EMA(close, timeperiod=period)
        
        # ADX et composants
        indicators['adx'] = talib.ADX(high, low, close, timeperiod=14)
        indicators['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        indicators['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        
        # Aroon
        indicators['aroon_up'], indicators['aroon_down'] = talib.AROON(high, low, timeperiod=25)
        indicators['aroon_osc'] = talib.AROONOSC(high, low, timeperiod=25)
        
        return indicators