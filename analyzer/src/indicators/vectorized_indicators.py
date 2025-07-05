"""
Calculs vectorisés et optimisés des indicateurs techniques
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from shared.src.technical_indicators import TechnicalIndicators
from .indicator_cache import indicator_cache
import logging

logger = logging.getLogger(__name__)

class VectorizedIndicators:
    """Calculs optimisés et vectorisés des indicateurs techniques"""
    
    def __init__(self):
        self.tech_indicators = TechnicalIndicators()
    
    def compute_all_indicators(self, df: pd.DataFrame, symbol: str) -> Dict[str, np.ndarray]:
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
            lambda x: self._safe_indicator_array(self.tech_indicators.calculate_rsi(x, 14), len(x)),
            timeperiod=14
        )
        
        indicators['rsi_21'] = indicator_cache.get(
            'RSI', close,
            lambda x: self._safe_indicator_array(self.tech_indicators.calculate_rsi(x, 21), len(x)),
            timeperiod=21
        )
        
        # ATR avec cache (nécessite high, low, close)
        atr_key = f"ATR_{symbol}"
        indicators['atr_14'] = indicator_cache.get(
            atr_key, close,
            lambda _: self._safe_indicator_array(self.tech_indicators.calculate_atr(high, low, close, 14), len(close)),
            timeperiod=14, high=high[-14:], low=low[-14:]
        )
        
        # Bollinger Bands - calcul une seule fois
        bb_data = self.tech_indicators.calculate_bollinger_bands(close, 20, 2.0)
        indicators['bb_upper'] = self._safe_indicator_array(bb_data.get('bb_upper'), len(close))
        indicators['bb_middle'] = self._safe_indicator_array(bb_data.get('bb_middle'), len(close))
        indicators['bb_lower'] = self._safe_indicator_array(bb_data.get('bb_lower'), len(close))
        indicators['bb_width'] = self._safe_indicator_array(bb_data.get('bb_width'), len(close))
        
        # MACD - calcul une seule fois
        macd_data = self.tech_indicators.calculate_macd(close)
        indicators['macd'] = self._safe_indicator_array(macd_data.get('macd_line'), len(close))
        indicators['macd_signal'] = self._safe_indicator_array(macd_data.get('macd_signal'), len(close))
        indicators['macd_hist'] = self._safe_indicator_array(macd_data.get('macd_histogram'), len(close))
        
        # EMAs multiples - calculées en une passe
        ema_periods = [9, 21, 50, 200]
        for period in ema_periods:
            indicators[f'ema_{period}'] = self._safe_indicator_array(
                self.tech_indicators.calculate_ema(close, period), len(close)
            )
        
        # SMAs pour certaines stratégies
        sma_periods = [20, 50]
        for period in sma_periods:
            indicators[f'sma_{period}'] = self._safe_indicator_array(
                self.tech_indicators.calculate_sma(close, period), len(close)
            )
        
        # Volume indicators
        indicators['volume_sma_20'] = self._safe_indicator_array(
            self.tech_indicators.calculate_sma(volume, 20), len(volume)
        )
        indicators['obv'] = self._safe_indicator_array(
            self.tech_indicators.calculate_obv(close, volume), len(close)
        )
        
        # Stochastic
        stoch_k, stoch_d = self.tech_indicators.calculate_stochastic(
            high, low, close, 14, 3, 3
        )
        indicators['stoch_k'] = self._safe_indicator_array(stoch_k, len(close))
        indicators['stoch_d'] = self._safe_indicator_array(stoch_d, len(close))
        
        # ADX pour la force de tendance
        adx, _, _ = self.tech_indicators.calculate_adx(high, low, close, 14)
        indicators['adx'] = indicator_cache.get(
            'ADX', close,
            lambda _: self._safe_indicator_array(adx, len(close)),
            timeperiod=14, high=high[-14:], low=low[-14:]
        )
        
        # Supertrend - maintenant implémenté dans shared technical_indicators
        supertrend_value, supertrend_direction = self.tech_indicators.calculate_supertrend(high, low, close, 7, 3.0)
        if supertrend_value is not None:
            indicators['supertrend'] = self._safe_indicator_array(supertrend_value, len(close))
            indicators['supertrend_direction'] = np.full(len(close), supertrend_direction)
        else:
            indicators['supertrend'] = np.full(len(close), np.nan)
            indicators['supertrend_direction'] = np.full(len(close), 0)
        
        # Statistiques de performance
        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
        logger.debug(f"Calcul vectorisé de {len(indicators)} indicateurs en {elapsed:.3f}s")
        
        return indicators
    
    def _safe_indicator_array(self, value: Optional[float], length: int) -> np.ndarray:
        """Convertit une valeur scalaire en array numpy de la bonne taille"""
        if value is None:
            return np.full(length, np.nan)
        # Si c'est déjà un array, le retourner
        if isinstance(value, np.ndarray):
            return value
        # Sinon créer un array avec la dernière valeur valide et NaN avant
        result = np.full(length, np.nan)
        result[-1] = value
        return result
    
    def compute_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calcule spécifiquement les indicateurs de momentum"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        return {
            'rsi_14': self._safe_indicator_array(self.tech_indicators.calculate_rsi(close, 14), len(close)),
            'rsi_21': self._safe_indicator_array(self.tech_indicators.calculate_rsi(close, 21), len(close)),
            'mfi': np.full(len(close), np.nan),  # MFI not yet in shared module
            'cci': np.full(len(close), np.nan),  # CCI not yet in shared module
            'roc': self._safe_indicator_array(self.tech_indicators.calculate_roc(close, 10), len(close)),
            'williams_r': np.full(len(close), np.nan)  # Williams %R not yet in shared module
        }
    
    def compute_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
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
                lambda _: self._safe_indicator_array(
                    self.tech_indicators.calculate_atr(high, low, close, period), len(close)
                ),
                timeperiod=period
            )
        
        # Volatilité historique
        returns = np.diff(np.log(close))
        indicators['historical_volatility'] = pd.Series(returns).rolling(20).std().values * np.sqrt(252)
        
        return indicators
    
    def compute_trend_indicators(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calcule spécifiquement les indicateurs de tendance"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        indicators = {}
        
        # EMAs multiples
        ema_periods = [9, 21, 50, 100, 200]
        for period in ema_periods:
            indicators[f'ema_{period}'] = self._safe_indicator_array(
                self.tech_indicators.calculate_ema(close, period), len(close)
            )
        
        # ADX et composants
        adx, plus_di, minus_di = self.tech_indicators.calculate_adx(high, low, close, 14)
        indicators['adx'] = self._safe_indicator_array(adx, len(close))
        indicators['plus_di'] = self._safe_indicator_array(plus_di, len(close))
        indicators['minus_di'] = self._safe_indicator_array(minus_di, len(close))
        
        # Aroon - not yet in shared module
        indicators['aroon_up'] = np.full(len(close), np.nan)
        indicators['aroon_down'] = np.full(len(close), np.nan)
        indicators['aroon_osc'] = np.full(len(close), np.nan)
        
        return indicators