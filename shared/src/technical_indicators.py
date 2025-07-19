"""
Module centralisé pour le calcul des indicateurs techniques.
Utilise talib pour les calculs optimisés et assure la cohérence entre tous les services.

Ce module remplace toutes les implémentations manuelles dispersées dans:
- gateway/src/binance_ws.py
- gateway/src/ultra_data_fetcher.py  
- analyzer/strategies/*.py
- visualization/src/chart_service.py
- signal_aggregator/src/*.py
"""
import json
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from .config import STRATEGY_PARAMS

logger = logging.getLogger(__name__)

try:
    import talib
    TALIB_AVAILABLE = True
    logger.info("✅ TA-Lib disponible pour calculs optimisés")
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("⚠️ TA-Lib non disponible, utilisation des calculs manuels de fallback")

class TechnicalIndicators:
    """
    Calculateur centralisé d'indicateurs techniques.
    Utilise talib quand disponible, sinon fallback vers calculs manuels.
    """
    
    def __init__(self):
        self.talib_available = TALIB_AVAILABLE
        
        # Configuration des périodes depuis config.py (pour compatibilité)
        self.rsi_period = STRATEGY_PARAMS["rsi"]["window"]
        self.bb_period = STRATEGY_PARAMS["bollinger"]["window"]
        self.bb_std = STRATEGY_PARAMS["bollinger"]["num_std"]
        self.macd_fast = STRATEGY_PARAMS["macd"]["fast_period"]
        self.macd_slow = STRATEGY_PARAMS["macd"]["slow_period"] 
        self.macd_signal = STRATEGY_PARAMS["macd"]["signal_period"]
        
        logger.info(f"🔧 TechnicalIndicators initialisé - talib: {self.talib_available}")
    
    # =================== RSI ===================
    
    def calculate_rsi(self, prices: Union[List[float], np.ndarray, pd.Series], 
                     period: Optional[int] = None) -> Optional[float]:
        """
        Calcule le RSI (Relative Strength Index).
        
        Args:
            prices: Prix de clôture
            period: Période RSI (défaut: depuis config)
            
        Returns:
            Valeur RSI ou None si impossible
        """
        if period is None:
            period = self.rsi_period
            
        prices_array = self._to_numpy_array(prices)
        if len(prices_array) < period + 1:
            return None
            
        if self.talib_available:
            try:
                rsi_values = talib.RSI(prices_array, timeperiod=period)
                return float(rsi_values[-1]) if not np.isnan(rsi_values[-1]) else None
            except Exception as e:
                logger.warning(f"Erreur talib RSI: {e}, utilisation fallback")
                
        return self._calculate_rsi_manual(prices_array, period)
    
    def _calculate_rsi_manual(self, prices: np.ndarray, period: int) -> Optional[float]:
        """Calcul RSI manuel (fallback)"""
        if len(prices) < period + 1:
            return None
            
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        if len(gains) < period:
            return None
            
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return round(float(rsi), 2)
    
    def _ema_smooth(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Calcule une EMA (Exponential Moving Average) pour le lissage.
        Utilisé en interne pour divers calculs.
        OPTIMISÉ : Utilise le calcul incrémental.
        """
        if len(data) < period:
            return data
        
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(data, dtype=float)
        
        # Initialiser avec la première valeur
        ema[0] = float(data[0])
        
        # Calcul incrémental optimisé
        for i in range(1, len(data)):
            ema[i] = alpha * float(data[i]) + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _sma_rolling(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Calcule une SMA (Simple Moving Average) roulante optimisée.
        Alternative à _ema_smooth pour certains indicateurs.
        """
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        
        sma = np.full_like(data, np.nan, dtype=float)
        
        # Première valeur SMA
        sma[period-1] = np.mean(data[:period])
        
        # Calcul roulant optimisé : SMA_new = SMA_old + (new - old) / period
        for i in range(period, len(data)):
            sma[i] = sma[i-1] + (data[i] - data[i-period]) / period
            
        return sma
    
    # =================== EMA ===================
    
    def calculate_ema(self, prices: Union[List[float], np.ndarray, pd.Series], 
                     period: int) -> Optional[float]:
        """
        Calcule l'EMA (Exponential Moving Average) - dernière valeur seulement.
        
        Args:
            prices: Prix de clôture
            period: Période EMA
            
        Returns:
            Valeur EMA ou None si impossible
        """
        prices_array = self._to_numpy_array(prices)
        if len(prices_array) < period:
            return None
            
        if self.talib_available:
            try:
                ema_values = talib.EMA(prices_array, timeperiod=period)
                return float(ema_values[-1]) if not np.isnan(ema_values[-1]) else None
            except Exception as e:
                logger.warning(f"Erreur talib EMA: {e}, utilisation fallback")
                
        return self._calculate_ema_manual(prices_array, period)
    
    def calculate_ema_incremental(self, current_price: float, previous_ema: Optional[float], period: int) -> float:
        """
        Calcule EMA de manière incrémentale pour éviter les dents de scie.
        
        Args:
            current_price: Prix actuel
            previous_ema: EMA précédente (None si première valeur)
            period: Période EMA
            
        Returns:
            Nouvelle valeur EMA
        """
        alpha = 2.0 / (period + 1)
        
        if previous_ema is None:
            # Première valeur : utiliser le prix lui-même
            return float(current_price)
        
        # Formule EMA incrémentale : EMA = α × Prix + (1-α) × EMA_prev
        return alpha * float(current_price) + (1 - alpha) * float(previous_ema)
    
    def calculate_ema_series(self, prices: Union[List[float], np.ndarray, pd.Series], 
                            period: int) -> List[Optional[float]]:
        """
        Calcule une série complète d'EMA de manière incrémentale.
        
        Args:
            prices: Prix de clôture
            period: Période EMA
            
        Returns:
            Liste des valeurs EMA (None pour les premières valeurs insuffisantes)
        """
        prices_array = self._to_numpy_array(prices)
        if len(prices_array) < period:
            return [None] * len(prices_array)
            
        if self.talib_available:
            try:
                ema_values = talib.EMA(prices_array, timeperiod=period)
                return [float(val) if not np.isnan(val) else None for val in ema_values]
            except Exception as e:
                logger.warning(f"Erreur talib EMA series: {e}, utilisation fallback")
        
        # Calcul manuel incrémental
        ema_series: list[float | None] = [None] * len(prices_array)
        alpha = 2.0 / (period + 1)
        
        # Première valeur EMA = première valeur de prix (à l'index period-1)
        if len(prices_array) >= period:
            # Initialiser avec SMA des premières valeurs
            first_ema = float(np.mean(prices_array[:period]))
            ema_series[period - 1] = first_ema
            
            # Calcul incrémental pour le reste
            for i in range(period, len(prices_array)):
                prev_ema = ema_series[i - 1]
                current_price = float(prices_array[i])
                if prev_ema is not None:
                    ema_series[i] = alpha * current_price + (1 - alpha) * prev_ema
                
        return ema_series
    
    def _calculate_ema_manual(self, prices: np.ndarray, period: int) -> Optional[float]:
        """Calcul EMA manuel (fallback)"""
        if len(prices) < period:
            return None
            
        multiplier = 2 / (period + 1)
        ema = float(prices[0])
        
        for price in prices[1:]:
            ema = (float(price) * multiplier) + (ema * (1 - multiplier))
            
        return round(ema, 6)
    
    # =================== MACD ===================
    
    def calculate_macd(self, prices: Union[List[float], np.ndarray, pd.Series]) -> Dict[str, Optional[float]]:
        """
        Calcule le MACD complet (line, signal, histogram) - dernière valeur seulement.
        
        Args:
            prices: Prix de clôture
            
        Returns:
            Dict avec macd_line, macd_signal, macd_histogram
        """
        prices_array = self._to_numpy_array(prices)
        min_required = max(self.macd_slow, self.macd_fast) + self.macd_signal
        
        if len(prices_array) < min_required:
            return {'macd_line': None, 'macd_signal': None, 'macd_histogram': None}
            
        if self.talib_available:
            try:
                macd_line, macd_signal, macd_hist = talib.MACD(
                    prices_array, 
                    fastperiod=self.macd_fast,
                    slowperiod=self.macd_slow, 
                    signalperiod=self.macd_signal
                )
                
                return {
                    'macd_line': float(macd_line[-1]) if not np.isnan(macd_line[-1]) else None,
                    'macd_signal': float(macd_signal[-1]) if not np.isnan(macd_signal[-1]) else None,
                    'macd_histogram': float(macd_hist[-1]) if not np.isnan(macd_hist[-1]) else None
                }
            except Exception as e:
                logger.warning(f"Erreur talib MACD: {e}, utilisation fallback")
                
        return self._calculate_macd_manual(prices_array)
    
    def calculate_macd_incremental(self, current_price: float, 
                                  prev_ema_fast: Optional[float], 
                                  prev_ema_slow: Optional[float],
                                  prev_macd_signal: Optional[float]) -> Dict[str, Optional[float]]:
        """
        Calcule MACD de manière incrémentale pour éviter les dents de scie.
        
        Args:
            current_price: Prix actuel
            prev_ema_fast: EMA rapide précédente
            prev_ema_slow: EMA lente précédente  
            prev_macd_signal: Signal MACD précédent
            
        Returns:
            Dict avec macd_line, macd_signal, macd_histogram
        """
        # Calculer les nouvelles EMA de manière incrémentale
        new_ema_fast = self.calculate_ema_incremental(current_price, prev_ema_fast, self.macd_fast)
        new_ema_slow = self.calculate_ema_incremental(current_price, prev_ema_slow, self.macd_slow)
        
        # MACD Line = EMA_fast - EMA_slow
        macd_line = new_ema_fast - new_ema_slow
        
        # Signal Line = EMA du MACD Line
        macd_signal = self.calculate_ema_incremental(macd_line, prev_macd_signal, self.macd_signal)
        
        # Histogram = MACD Line - Signal Line
        macd_histogram = macd_line - macd_signal
        
        return {
            'macd_line': round(macd_line, 6),
            'macd_signal': round(macd_signal, 6),
            'macd_histogram': round(macd_histogram, 6),
            'ema_fast': round(new_ema_fast, 6),  # Pour cache
            'ema_slow': round(new_ema_slow, 6)   # Pour cache
        }
    
    def calculate_macd_series(self, prices: Union[List[float], np.ndarray, pd.Series]) -> Dict[str, List[Optional[float]]]:
        """
        Calcule une série complète de MACD de manière incrémentale.
        
        Args:
            prices: Prix de clôture
            
        Returns:
            Dict avec séries macd_line, macd_signal, macd_histogram
        """
        prices_array = self._to_numpy_array(prices)
        min_required = max(self.macd_slow, self.macd_fast) + self.macd_signal
        
        if len(prices_array) < min_required:
            empty_series: List[Optional[float]] = [None] * len(prices_array)
            return {
                'macd_line': empty_series,
                'macd_signal': empty_series, 
                'macd_histogram': empty_series
            }
            
        if self.talib_available:
            try:
                macd_line, macd_signal, macd_hist = talib.MACD(
                    prices_array,
                    fastperiod=self.macd_fast,
                    slowperiod=self.macd_slow,
                    signalperiod=self.macd_signal
                )
                
                return {
                    'macd_line': [float(val) if not np.isnan(val) else None for val in macd_line],
                    'macd_signal': [float(val) if not np.isnan(val) else None for val in macd_signal],
                    'macd_histogram': [float(val) if not np.isnan(val) else None for val in macd_hist]
                }
            except Exception as e:
                logger.warning(f"Erreur talib MACD series: {e}, utilisation fallback")
        
        # Calcul manuel incrémental
        ema_fast_series = self.calculate_ema_series(prices_array, self.macd_fast)
        ema_slow_series = self.calculate_ema_series(prices_array, self.macd_slow)
        
        macd_line_series: List[Optional[float]] = []
        for i in range(len(prices_array)):
            if ema_fast_series[i] is not None and ema_slow_series[i] is not None:
                # Type assertions for mypy
                fast_val = ema_fast_series[i]
                slow_val = ema_slow_series[i]
                assert fast_val is not None
                assert slow_val is not None
                macd_line_series.append(fast_val - slow_val)
            else:
                macd_line_series.append(None)
        
        # Signal = EMA du MACD Line
        macd_signal_series = self.calculate_ema_series(macd_line_series, self.macd_signal)
        
        # Histogram = MACD - Signal
        macd_histogram_series: List[Optional[float]] = []
        for i in range(len(prices_array)):
            if macd_line_series[i] is not None and macd_signal_series[i] is not None:
                # Type assertions for mypy
                line_val = macd_line_series[i]
                signal_val = macd_signal_series[i]
                assert line_val is not None
                assert signal_val is not None
                macd_histogram_series.append(line_val - signal_val)
            else:
                macd_histogram_series.append(None)
        
        return {
            'macd_line': macd_line_series,
            'macd_signal': macd_signal_series,
            'macd_histogram': macd_histogram_series
        }
    
    def _calculate_macd_manual(self, prices: np.ndarray) -> Dict[str, Optional[float]]:
        """Calcul MACD manuel (fallback)"""
        try:
            ema_fast = self._calculate_ema_manual(prices, self.macd_fast)
            ema_slow = self._calculate_ema_manual(prices, self.macd_slow)
            
            if ema_fast is None or ema_slow is None:
                return {'macd_line': None, 'macd_signal': None, 'macd_histogram': None}
                
            macd_line = ema_fast - ema_slow
            
            # Calculer signal line (EMA du MACD)
            min_points = max(self.macd_slow, self.macd_fast) + self.macd_signal
            if len(prices) >= min_points:
                # Calculer MACD historique pour signal
                macd_values = []
                for i in range(self.macd_slow, len(prices)):
                    subset = prices[:i+1]
                    fast = self._calculate_ema_manual(subset, self.macd_fast)
                    slow = self._calculate_ema_manual(subset, self.macd_slow)
                    if fast is not None and slow is not None:
                        macd_values.append(fast - slow)
                
                if len(macd_values) >= self.macd_signal:
                    macd_signal = self._calculate_ema_manual(np.array(macd_values), self.macd_signal)
                    macd_histogram = macd_line - (macd_signal or 0)
                else:
                    macd_signal = macd_line
                    macd_histogram = 0
            else:
                macd_signal = macd_line
                macd_histogram = 0
                
            return {
                'macd_line': round(macd_line, 6),
                'macd_signal': round(macd_signal, 6) if macd_signal else None,
                'macd_histogram': round(macd_histogram, 6)
            }
        except Exception as e:
            logger.error(f"Erreur calcul MACD manuel: {e}")
            return {'macd_line': None, 'macd_signal': None, 'macd_histogram': None}
    
    # =================== BOLLINGER BANDS ===================
    
    def calculate_bollinger_bands(self, prices: Union[List[float], np.ndarray, pd.Series],
                                 period: Optional[int] = None,
                                 std_dev: Optional[float] = None) -> Dict[str, Optional[float]]:
        """
        Calcule les Bollinger Bands.
        
        Args:
            prices: Prix de clôture
            period: Période (défaut: depuis config)
            std_dev: Écart-type (défaut: depuis config)
            
        Returns:
            Dict avec bb_upper, bb_middle, bb_lower, bb_position, bb_width
        """
        if period is None:
            period = self.bb_period
        if std_dev is None:
            std_dev = self.bb_std
            
        prices_array = self._to_numpy_array(prices)
        if len(prices_array) < period:
            return {'bb_upper': None, 'bb_middle': None, 'bb_lower': None, 
                   'bb_position': None, 'bb_width': None}
            
        if self.talib_available:
            try:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(
                    prices_array, timeperiod=period, nbdevup=std_dev, 
                    nbdevdn=std_dev, matype=talib.MA_Type.SMA
                )
                
                upper = float(bb_upper[-1]) if not np.isnan(bb_upper[-1]) else None
                middle = float(bb_middle[-1]) if not np.isnan(bb_middle[-1]) else None
                lower = float(bb_lower[-1]) if not np.isnan(bb_lower[-1]) else None
                
                if upper and middle and lower:
                    current_price = float(prices_array[-1])
                    bb_position = (current_price - lower) / (upper - lower) if upper != lower else 0.5
                    bb_width = ((upper - lower) / middle) * 100 if middle != 0 else 0
                    
                    return {
                        'bb_upper': round(upper, 6),
                        'bb_middle': round(middle, 6),
                        'bb_lower': round(lower, 6),
                        'bb_position': round(bb_position, 3),
                        'bb_width': round(bb_width, 2)
                    }
            except Exception as e:
                logger.warning(f"Erreur talib Bollinger: {e}, utilisation fallback")
                
        return self._calculate_bollinger_manual(prices_array, period, std_dev)
    
    def _calculate_bollinger_manual(self, prices: np.ndarray, period: int, std_dev: float) -> Dict[str, Optional[float]]:
        """Calcul Bollinger manuel (fallback)"""
        if len(prices) < period:
            return {'bb_upper': None, 'bb_middle': None, 'bb_lower': None,
                   'bb_position': None, 'bb_width': None}
            
        recent_prices = prices[-period:]
        sma = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        current_price = float(prices[-1])
        bb_position = (current_price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5
        bb_width = ((upper_band - lower_band) / sma) * 100 if sma != 0 else 0
        
        return {
            'bb_upper': round(float(upper_band), 6),
            'bb_middle': round(float(sma), 6),
            'bb_lower': round(float(lower_band), 6),
            'bb_position': round(float(bb_position), 3),
            'bb_width': round(float(bb_width), 2)
        }
    
    # =================== ATR ===================
    
    def calculate_atr(self, highs: Union[List[float], np.ndarray, pd.Series],
                     lows: Union[List[float], np.ndarray, pd.Series],
                     closes: Union[List[float], np.ndarray, pd.Series],
                     period: int = 14) -> Optional[float]:
        """
        Calcule l'ATR (Average True Range).
        
        Args:
            highs: Prix hauts
            lows: Prix bas
            closes: Prix de clôture
            period: Période ATR
            
        Returns:
            Valeur ATR ou None si impossible
        """
        # Validation et alignement des arrays
        try:
            highs_array, lows_array, closes_array = self._validate_and_align_arrays(highs, lows, closes)
        except Exception as e:
            logger.error(f"Erreur validation arrays ATR: {e}")
            return None
        
        if len(highs_array) < period or len(lows_array) < period or len(closes_array) < period:
            return None
            
        if self.talib_available:
            try:
                atr_values = talib.ATR(highs_array, lows_array, closes_array, timeperiod=period)
                if atr_values is not None and len(atr_values) > 0:
                    last_atr = atr_values[-1]
                    if not np.isnan(last_atr) and not np.isinf(last_atr):
                        return float(last_atr)
                    else:
                        logger.warning(f"Talib ATR retourne NaN/Inf pour {period} périodes")
                return None
            except Exception as e:
                logger.warning(f"Erreur talib ATR: {e}, utilisation fallback")
                
        return self._calculate_atr_manual(highs_array, lows_array, closes_array, period)
    
    def _calculate_atr_manual(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> Optional[float]:
        """Calcul ATR manuel (fallback)"""
        if len(highs) < period + 1:
            return None
            
        true_ranges = []
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            
            # Vérifier que les valeurs sont valides
            tr = max(tr1, tr2, tr3)
            if not np.isnan(tr) and not np.isinf(tr):
                true_ranges.append(tr)
            
        if len(true_ranges) < period:
            logger.warning(f"Pas assez de True Range valides: {len(true_ranges)} < {period}")
            return None
            
        atr = np.mean(true_ranges[-period:])
        
        # Vérifier que le résultat est valide
        if np.isnan(atr) or np.isinf(atr):
            logger.warning(f"ATR manuel calculé invalide: {atr}")
            return None
            
        return round(float(atr), 6)
    
    # =================== SMA ===================
    
    def calculate_sma(self, prices: Union[List[float], np.ndarray, pd.Series], 
                     period: int) -> Optional[float]:
        """
        Calcule la SMA (Simple Moving Average) - dernière valeur seulement.
        
        Args:
            prices: Prix de clôture
            period: Période SMA
            
        Returns:
            Valeur SMA ou None si impossible
        """
        prices_array = self._to_numpy_array(prices)
        if len(prices_array) < period:
            return None
            
        if self.talib_available:
            try:
                sma_values = talib.SMA(prices_array, timeperiod=period)
                return float(sma_values[-1]) if not np.isnan(sma_values[-1]) else None
            except Exception as e:
                logger.warning(f"Erreur talib SMA: {e}, utilisation fallback")
                
        return round(float(np.mean(prices_array[-period:])), 6)
    
    def calculate_sma_incremental(self, current_price: float, previous_sma: Optional[float], 
                                 period: int, oldest_price: Optional[float] = None) -> float:
        """
        Calcule SMA de manière incrémentale (rolling average).
        
        Args:
            current_price: Prix actuel
            previous_sma: SMA précédente
            period: Période SMA
            oldest_price: Prix le plus ancien qui sort de la fenêtre (optionnel)
            
        Returns:
            Nouvelle valeur SMA
        """
        if previous_sma is None:
            return float(current_price)
        
        if oldest_price is not None:
            # Calcul rolling : SMA_new = SMA_old + (new - old) / period
            return float(previous_sma + (current_price - oldest_price) / period)
        else:
            # Calcul expanding (pour l'initialisation)
            return float(current_price)  # Simplifié
    
    def calculate_sma_series(self, prices: Union[List[float], np.ndarray, pd.Series], 
                            period: int) -> List[Optional[float]]:
        """
        Calcule une série complète de SMA de manière optimisée.
        
        Args:
            prices: Prix de clôture
            period: Période SMA
            
        Returns:
            Liste des valeurs SMA (None pour les premières valeurs insuffisantes)
        """
        prices_array = self._to_numpy_array(prices)
        if len(prices_array) < period:
            return [None] * len(prices_array)
            
        if self.talib_available:
            try:
                sma_values = talib.SMA(prices_array, timeperiod=period)
                return [float(val) if not np.isnan(val) else None for val in sma_values]
            except Exception as e:
                logger.warning(f"Erreur talib SMA series: {e}, utilisation fallback")
        
        # Calcul manuel optimisé
        sma_series: list[float | None] = [None] * len(prices_array)
        
        if len(prices_array) >= period:
            # Première valeur SMA
            sma_series[period - 1] = float(np.mean(prices_array[:period]))
            
            # Calcul rolling optimisé
            for i in range(period, len(prices_array)):
                prev_sma = sma_series[i - 1]
                new_price = float(prices_array[i])
                old_price = float(prices_array[i - period])
                if prev_sma is not None:
                    sma_series[i] = prev_sma + (new_price - old_price) / period
                
        return sma_series
    
    # =================== ADX ===================
    
    def calculate_adx(self, highs: Union[List[float], np.ndarray, pd.Series],
                     lows: Union[List[float], np.ndarray, pd.Series],
                     closes: Union[List[float], np.ndarray, pd.Series],
                     period: int = 14) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calcule l'ADX et les DI+/DI-.
        
        Args:
            highs: Prix hauts
            lows: Prix bas
            closes: Prix de clôture
            period: Période ADX
            
        Returns:
            Tuple (ADX, DI+, DI-) ou (None, None, None) si impossible
        """
        # Validation et alignement des arrays
        try:
            highs_array, lows_array, closes_array = self._validate_and_align_arrays(highs, lows, closes)
        except Exception as e:
            logger.error(f"Erreur validation arrays ADX: {e}")
            return None, None, None
        
        if len(highs_array) < period * 2:
            return None, None, None
            
        if self.talib_available:
            try:
                adx = talib.ADX(highs_array, lows_array, closes_array, timeperiod=period)
                plus_di = talib.PLUS_DI(highs_array, lows_array, closes_array, timeperiod=period)
                minus_di = talib.MINUS_DI(highs_array, lows_array, closes_array, timeperiod=period)
                
                adx_val = float(adx[-1]) if not np.isnan(adx[-1]) else None
                plus_di_val = float(plus_di[-1]) if not np.isnan(plus_di[-1]) else None
                minus_di_val = float(minus_di[-1]) if not np.isnan(minus_di[-1]) else None
                
                return adx_val, plus_di_val, minus_di_val
            except Exception as e:
                logger.warning(f"Erreur talib ADX: {e}")
        
        # Fallback: calcul manuel de l'ADX
        try:
            # Calcul du True Range (TR)
            high_low = highs_array - lows_array
            high_close = np.abs(highs_array[1:] - closes_array[:-1])
            low_close = np.abs(lows_array[1:] - closes_array[:-1])
            
            # Combine les arrays pour le TR
            tr = np.zeros(len(highs_array))
            tr[0] = high_low[0]
            for i in range(1, len(highs_array)):
                tr[i] = max(high_low[i], 
                           abs(highs_array[i] - closes_array[i-1]),
                           abs(lows_array[i] - closes_array[i-1]))
            
            # Calcul du Directional Movement
            dm_plus = np.zeros(len(highs_array))
            dm_minus = np.zeros(len(highs_array))
            
            for i in range(1, len(highs_array)):
                up_move = highs_array[i] - highs_array[i-1]
                down_move = lows_array[i-1] - lows_array[i]
                
                if up_move > down_move and up_move > 0:
                    dm_plus[i] = up_move
                elif down_move > up_move and down_move > 0:
                    dm_minus[i] = down_move
            
            # Lissage avec EMA
            atr = self._ema_smooth(tr, period)
            dm_plus_smooth = self._ema_smooth(dm_plus, period)
            dm_minus_smooth = self._ema_smooth(dm_minus, period)
            
            # Calcul des DI
            di_plus = np.zeros(len(atr))
            di_minus = np.zeros(len(atr))
            
            for i in range(len(atr)):
                if atr[i] > 0:
                    di_plus[i] = (dm_plus_smooth[i] / atr[i]) * 100
                    di_minus[i] = (dm_minus_smooth[i] / atr[i]) * 100
            
            # Calcul du DX
            dx = np.zeros(len(di_plus))
            for i in range(len(di_plus)):
                di_sum = di_plus[i] + di_minus[i]
                if di_sum > 0:
                    dx[i] = abs(di_plus[i] - di_minus[i]) / di_sum * 100
            
            # Calcul de l'ADX (moyenne lissée du DX)
            adx = self._ema_smooth(dx, period)
            
            # Retourner les dernières valeurs valides
            if len(adx) > 0 and not np.isnan(adx[-1]):
                return float(adx[-1]), float(di_plus[-1]), float(di_minus[-1])
            
        except Exception as e:
            logger.warning(f"Erreur calcul manuel ADX: {e}")
                
        return None, None, None
    
    def calculate_adx_smoothed(self, highs: Union[List[float], np.ndarray, pd.Series],
                             lows: Union[List[float], np.ndarray, pd.Series],
                             closes: Union[List[float], np.ndarray, pd.Series],
                             period: int = 14,
                             smooth_period: int = 3) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calcule ADX avec lissage EMA pour réduire la volatilité.
        
        Args:
            highs: Prix hauts
            lows: Prix bas
            closes: Prix de clôture
            period: Période pour ADX (défaut: 14)
            smooth_period: Période de lissage EMA (défaut: 3)
            
        Returns:
            Tuple (ADX lissé, DI+, DI-)
        """
        try:
            # Importer la configuration
            from shared.src.config import ADX_HYBRID_MODE
            
            # Calculer ADX standard
            raw_adx, plus_di, minus_di = self.calculate_adx(highs, lows, closes, period)
            
            if not ADX_HYBRID_MODE or raw_adx is None:
                return raw_adx, plus_di, minus_di
            
            # Convertir en numpy arrays pour le lissage
            highs_array = np.array(highs, dtype=np.float64)
            lows_array = np.array(lows, dtype=np.float64)
            closes_array = np.array(closes, dtype=np.float64)
            
            # Besoin de plus de données pour le lissage
            min_required = period + smooth_period + 5
            if len(closes_array) < min_required:
                return raw_adx, plus_di, minus_di
                
            # Calculer plusieurs valeurs ADX pour le lissage
            adx_values = []
            for i in range(smooth_period + 2):
                end_idx = len(closes_array) - i
                if end_idx >= period + 5:
                    adx_val, _, _ = self.calculate_adx(
                        highs_array[:end_idx], 
                        lows_array[:end_idx], 
                        closes_array[:end_idx], 
                        period
                    )
                    if adx_val is not None:
                        adx_values.append(adx_val)
            
            # Appliquer lissage EMA si on a assez de valeurs
            if len(adx_values) >= smooth_period:
                # Inverser pour avoir l'ordre chronologique
                adx_values.reverse()
                
                # Calculer EMA avec TA-Lib si disponible
                if self.talib_available:
                    try:
                        import talib
                        smoothed_values = talib.EMA(np.array(adx_values), timeperiod=smooth_period)
                        smoothed_adx = float(smoothed_values[-1]) if not np.isnan(smoothed_values[-1]) else raw_adx
                        
                        logger.debug(f"ADX lissé: {raw_adx:.2f} → {smoothed_adx:.2f} (EMA{smooth_period})")
                        return smoothed_adx, plus_di, minus_di
                    except Exception as e:
                        logger.warning(f"Erreur lissage EMA TA-Lib: {e}")
                
                # Fallback: EMA manuel
                alpha = 2.0 / (smooth_period + 1)
                ema_value = adx_values[0]
                
                for value in adx_values[1:]:
                    ema_value = alpha * value + (1 - alpha) * ema_value
                
                logger.debug(f"ADX lissé (manuel): {raw_adx:.2f} → {ema_value:.2f} (EMA{smooth_period})")
                return round(ema_value, 2), plus_di, minus_di
            
            # Pas assez de données pour lisser
            return raw_adx, plus_di, minus_di
            
        except Exception as e:
            logger.error(f"Erreur calcul ADX lissé: {e}")
            # Fallback sur ADX standard
            return self.calculate_adx(highs, lows, closes, period)
    
    # =================== STOCHASTIC ===================
    
    def calculate_stochastic(self, highs: Union[List[float], np.ndarray, pd.Series],
                           lows: Union[List[float], np.ndarray, pd.Series],
                           closes: Union[List[float], np.ndarray, pd.Series],
                           fastk_period: int = 14,
                           slowk_period: int = 3,
                           slowd_period: int = 3) -> Tuple[Optional[float], Optional[float]]:
        """
        Calcule le Stochastic Oscillator.
        
        Returns:
            Tuple (%K, %D) ou (None, None) si impossible
        """
        # Validation et alignement des arrays
        try:
            highs_array, lows_array, closes_array = self._validate_and_align_arrays(highs, lows, closes)
        except Exception as e:
            logger.error(f"Erreur validation arrays STOCH: {e}")
            return None, None
        
        if len(highs_array) < fastk_period + slowk_period:
            return None, None
            
        if self.talib_available:
            try:
                slowk, slowd = talib.STOCH(highs_array, lows_array, closes_array,
                                          fastk_period=fastk_period,
                                          slowk_period=slowk_period,
                                          slowd_period=slowd_period)
                
                k_val = float(slowk[-1]) if not np.isnan(slowk[-1]) else None
                d_val = float(slowd[-1]) if not np.isnan(slowd[-1]) else None
                
                return k_val, d_val
            except Exception as e:
                logger.warning(f"Erreur talib STOCH: {e}")
                
        return None, None
    
    # =================== ROC ===================
    
    def calculate_roc(self, prices: Union[List[float], np.ndarray, pd.Series], 
                     period: int = 10) -> Optional[float]:
        """
        Calcule le ROC (Rate of Change).
        
        Returns:
            ROC en pourcentage ou None si impossible
        """
        prices_array = self._to_numpy_array(prices)
        if len(prices_array) < period + 1:
            return None
            
        if self.talib_available:
            try:
                roc_values = talib.ROC(prices_array, timeperiod=period)
                return float(roc_values[-1]) if not np.isnan(roc_values[-1]) else None
            except Exception as e:
                logger.warning(f"Erreur talib ROC: {e}")
                
        # Fallback manuel
        past_price = prices_array[-(period+1)]
        current_price = prices_array[-1]
        if past_price == 0:
            return None
            
        return round(((current_price - past_price) / past_price) * 100, 4)
    
    # =================== SUPERTREND ===================
    
    def calculate_supertrend(self, highs: Union[List[float], np.ndarray, pd.Series],
                           lows: Union[List[float], np.ndarray, pd.Series],
                           closes: Union[List[float], np.ndarray, pd.Series],
                           period: int = 7,
                           multiplier: float = 3.0) -> Tuple[Optional[float], Optional[int]]:
        """
        Calcule l'indicateur Supertrend avec historique complet.
        
        Args:
            highs: Prix hauts
            lows: Prix bas
            closes: Prix de clôture
            period: Période pour l'ATR (défaut: 7)
            multiplier: Multiplicateur pour les bandes (défaut: 3.0)
            
        Returns:
            Tuple (supertrend_value, direction)
            direction: 1 pour tendance haussière, -1 pour tendance baissière
        """
        try:
            highs_array = self._to_numpy_array(highs)
            lows_array = self._to_numpy_array(lows)
            closes_array = self._to_numpy_array(closes)
            
            if len(closes_array) < period + 2:
                return None, None
            
            # Calculer HL2 pour tout l'historique
            hl2 = (highs_array + lows_array) / 2
            
            # Calculer ATR pour chaque point
            atr_series = []
            for i in range(period, len(closes_array)):
                atr = self.calculate_atr(
                    highs_array[i-period+1:i+1], 
                    lows_array[i-period+1:i+1], 
                    closes_array[i-period+1:i+1], 
                    period
                )
                atr_series.append(atr if atr is not None else 0)
            
            if len(atr_series) < 2:
                return None, None
            
            # Calculer les bandes supérieures et inférieures
            upper_bands: list[float] = []
            lower_bands: list[float] = []
            supertrend_values: list[float] = []
            directions: list[int] = []
            
            for i, atr in enumerate(atr_series):
                current_idx = period + i
                current_hl2 = hl2[current_idx]
                current_close = closes_array[current_idx]
                
                # Bandes de base
                basic_upper = current_hl2 + (multiplier * atr)
                basic_lower = current_hl2 - (multiplier * atr)
                
                # Bandes ajustées (règles Supertrend)
                if i == 0:
                    # Premier calcul
                    final_upper = basic_upper
                    final_lower = basic_lower
                else:
                    # Ajustement des bandes basé sur l'historique
                    prev_close = closes_array[current_idx - 1]
                    prev_final_upper = upper_bands[-1]
                    prev_final_lower = lower_bands[-1]
                    
                    # Règle upper band: ne pas descendre si prix était dessous
                    if basic_upper < prev_final_upper or prev_close > prev_final_upper:
                        final_upper = basic_upper
                    else:
                        final_upper = prev_final_upper
                    
                    # Règle lower band: ne pas monter si prix était dessus  
                    if basic_lower > prev_final_lower or prev_close < prev_final_lower:
                        final_lower = basic_lower
                    else:
                        final_lower = prev_final_lower
                
                upper_bands.append(final_upper)
                lower_bands.append(final_lower)
                
                # Déterminer Supertrend et direction
                if i == 0:
                    # Premier calcul - basé sur position prix vs bandes
                    if current_close <= final_lower:
                        supertrend = final_upper
                        direction = -1
                    else:
                        supertrend = final_lower
                        direction = 1
                else:
                    # Calcul basé sur direction précédente et règles de changement
                    prev_direction = directions[-1]
                    prev_supertrend = supertrend_values[-1]
                    
                    if prev_direction == 1:  # Était en tendance haussière
                        if current_close <= final_lower:
                            # Changement vers tendance baissière
                            supertrend = final_upper
                            direction = -1
                        else:
                            # Continue en tendance haussière
                            supertrend = final_lower
                            direction = 1
                    else:  # Était en tendance baissière (-1)
                        if current_close >= final_upper:
                            # Changement vers tendance haussière
                            supertrend = final_lower
                            direction = 1
                        else:
                            # Continue en tendance baissière
                            supertrend = final_upper
                            direction = -1
                
                supertrend_values.append(supertrend)
                directions.append(direction)
            
            # Retourner la dernière valeur calculée
            if supertrend_values and directions:
                return round(supertrend_values[-1], 4), directions[-1]
            else:
                return None, None
                
        except Exception as e:
            logger.error(f"Erreur calcul Supertrend: {e}")
            return None, None
    
    # =================== OBV ===================
    
    def calculate_obv(self, prices: Union[List[float], np.ndarray, pd.Series],
                     volumes: Union[List[float], np.ndarray, pd.Series]) -> Optional[float]:
        """
        Calcule l'OBV (On Balance Volume).
        
        Returns:
            Valeur OBV ou None si impossible
        """
        # Validation et alignement des arrays
        try:
            prices_array, volumes_array = self._validate_and_align_arrays(prices, volumes)
        except Exception as e:
            logger.error(f"Erreur validation arrays OBV: {e}")
            return None
        
        if len(prices_array) < 2 or len(volumes_array) < 2:
            return None
            
        if self.talib_available:
            try:
                obv_values = talib.OBV(prices_array, volumes_array)
                return float(obv_values[-1]) if not np.isnan(obv_values[-1]) else None
            except Exception as e:
                logger.warning(f"Erreur talib OBV: {e}")
                
        return None
    
    # =================== CALCUL COMPLET (ARRAYS) ===================
    
    def calculate_all_indicators_array(self, highs: List[float], lows: List[float], 
                                     closes: List[float], volumes: List[float]) -> Dict[str, np.ndarray]:
        """
        Calcule tous les indicateurs techniques et retourne des ARRAYS complets.
        Chaque indicateur contient une valeur pour chaque point de données.
        
        Args:
            highs: Prix hauts
            lows: Prix bas  
            closes: Prix de clôture
            volumes: Volumes
            
        Returns:
            Dict avec arrays numpy pour chaque indicateur
        """
        indicators = {}
        
        try:
            # Validation et alignement des arrays
            highs_array, lows_array, closes_array, volumes_array = self._validate_and_align_arrays(highs, lows, closes, volumes)
            
            n_points = len(closes_array)
            
            if self.talib_available:
                # RSI
                rsi_array = talib.RSI(closes_array, timeperiod=14)
                indicators['rsi_14'] = rsi_array
                
                # EMAs - Binance standard (7/26/99)
                indicators['ema_7'] = talib.EMA(closes_array, timeperiod=7)
                indicators['ema_26'] = talib.EMA(closes_array, timeperiod=26)
                indicators['ema_99'] = talib.EMA(closes_array, timeperiod=99)
                
                # SMAs
                indicators['sma_20'] = talib.SMA(closes_array, timeperiod=20)
                indicators['sma_50'] = talib.SMA(closes_array, timeperiod=50)
                
                # MACD
                macd_line, macd_signal, macd_hist = talib.MACD(closes_array, 
                                                               fastperiod=7, 
                                                               slowperiod=26, 
                                                               signalperiod=9)
                indicators['macd_line'] = macd_line
                indicators['macd_signal'] = macd_signal
                indicators['macd_histogram'] = macd_hist
                
                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = talib.BBANDS(closes_array, 
                                                            timeperiod=20, 
                                                            nbdevup=2, 
                                                            nbdevdn=2,
                                                            matype=talib.MA_Type.SMA)
                indicators['bb_upper'] = bb_upper
                indicators['bb_middle'] = bb_middle
                indicators['bb_lower'] = bb_lower
                
                # BB position et width
                bb_position = np.full(n_points, np.nan)
                bb_width = np.full(n_points, np.nan)
                mask = ~(np.isnan(bb_upper) | np.isnan(bb_lower) | np.isnan(bb_middle))
                width_diff = bb_upper[mask] - bb_lower[mask]
                bb_position[mask] = np.where(width_diff != 0, 
                                           (closes_array[mask] - bb_lower[mask]) / width_diff, 
                                           0.5)
                bb_width[mask] = np.where(bb_middle[mask] != 0,
                                        (width_diff / bb_middle[mask]) * 100,
                                        0)
                indicators['bb_position'] = bb_position
                indicators['bb_width'] = bb_width
                
                # ATR
                indicators['atr_14'] = talib.ATR(highs_array, lows_array, closes_array, timeperiod=14)
                
                # ADX
                indicators['adx_14'] = talib.ADX(highs_array, lows_array, closes_array, timeperiod=14)
                indicators['plus_di'] = talib.PLUS_DI(highs_array, lows_array, closes_array, timeperiod=14)
                indicators['minus_di'] = talib.MINUS_DI(highs_array, lows_array, closes_array, timeperiod=14)
                
                # Stochastic
                stoch_k, stoch_d = talib.STOCH(highs_array, lows_array, closes_array,
                                              fastk_period=14, slowk_period=3, slowd_period=3)
                indicators['stoch_k'] = stoch_k
                indicators['stoch_d'] = stoch_d
                
                # Stochastic RSI
                indicators['stoch_rsi'] = talib.STOCHRSI(closes_array, timeperiod=14, 
                                                        fastk_period=3, fastd_period=3)[0]
                
                # Williams %R
                indicators['williams_r'] = talib.WILLR(highs_array, lows_array, closes_array, timeperiod=14)
                
                # CCI
                indicators['cci_20'] = talib.CCI(highs_array, lows_array, closes_array, timeperiod=20)
                
                # MFI
                indicators['mfi_14'] = talib.MFI(highs_array, lows_array, closes_array, volumes_array, timeperiod=14)
                
                # Momentum et ROC
                indicators['momentum_10'] = talib.MOM(closes_array, timeperiod=10)
                indicators['roc_10'] = talib.ROC(closes_array, timeperiod=10)
                indicators['roc_20'] = talib.ROC(closes_array, timeperiod=20)
                
                # OBV
                indicators['obv'] = talib.OBV(closes_array, volumes_array)
                
                # Volume analysis
                volume_sma = talib.SMA(volumes_array, timeperiod=20)
                volume_ratio = np.full(n_points, np.nan)
                mask = volume_sma > 0
                volume_ratio[mask] = volumes_array[mask] / volume_sma[mask]
                indicators['avg_volume_20'] = volume_sma
                indicators['volume_ratio'] = volume_ratio
                
                # VWAP approximation (10 périodes)
                vwap = np.full(n_points, np.nan)
                for i in range(9, n_points):
                    price_volume = closes_array[i-9:i+1] * volumes_array[i-9:i+1]
                    total_volume = np.sum(volumes_array[i-9:i+1])
                    if total_volume > 0:
                        vwap[i] = np.sum(price_volume) / total_volume
                indicators['vwap_10'] = vwap
                
                # Trend angle (sur 5 périodes)
                trend_angle = np.full(n_points, np.nan)
                for i in range(4, n_points):
                    price_change = (closes_array[i] - closes_array[i-4]) / closes_array[i-4] * 100
                    trend_angle[i] = np.arctan(price_change) * 180 / np.pi
                indicators['trend_angle'] = trend_angle
                
                # Pivot count (simplifié)
                pivot_count = np.zeros(n_points)
                for i in range(2, n_points-2):
                    # High pivot
                    if highs_array[i] > max(highs_array[i-2:i]) and highs_array[i] > max(highs_array[i+1:i+3]):
                        pivot_count[i] += 1
                    # Low pivot
                    if lows_array[i] < min(lows_array[i-2:i]) and lows_array[i] < min(lows_array[i+1:i+3]):
                        pivot_count[i] += 1
                indicators['pivot_count'] = pivot_count
                
            else:
                # Fallback manuel si talib non disponible
                logger.warning("TA-Lib non disponible, utilisation limitée des indicateurs manuels")
                # Implémenter au minimum RSI et EMAs manuellement
                # TODO: Ajouter calculs manuels si nécessaire
                
            return indicators
            
        except Exception as e:
            logger.error(f"Erreur calculate_all_indicators_array: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    # =================== CALCUL COMPLET (SCALAIRE) ===================
    
    def calculate_all_indicators(self, highs: List[float], lows: List[float], 
                               closes: List[float], volumes: List[float]) -> Dict[str, Any]:
        """
        Calcule tous les indicateurs techniques en une fois.
        Optimisé pour le Gateway qui a besoin de tout.
        
        Args:
            highs: Prix hauts
            lows: Prix bas  
            closes: Prix de clôture
            volumes: Volumes
            
        Returns:
            Dict avec tous les indicateurs calculés
        """
        indicators = {}
        
        try:
            # RSI
            try:
                rsi_value = self.calculate_rsi(closes, 14)
                if rsi_value is not None:
                    indicators['rsi_14'] = rsi_value
            except Exception as e:
                logger.error(f"❌ Erreur RSI: {e}")
            
            # EMAs multiples (garde l'ancien comportement pour compatibilité)
            try:
                for period in [7, 26, 99]:
                    ema_value = self.calculate_ema(closes, period)
                    if ema_value is not None:
                        indicators[f'ema_{period}'] = ema_value
            except Exception as e:
                logger.error(f"❌ Erreur EMAs: {e}")
            
            # SMAs
            for period in [20, 50]:
                sma_value = self.calculate_sma(closes, period)
                if sma_value is not None:
                    indicators[f'sma_{period}'] = sma_value
                
            # MACD
            macd_data = self.calculate_macd(closes)
            for key, value in macd_data.items():
                if value is not None:
                    indicators[key] = value
            
            # Bollinger Bands
            bb_data = self.calculate_bollinger_bands(closes)
            for key, value in bb_data.items():
                if value is not None:
                    indicators[key] = value
            
            # ATR
            atr_value = self.calculate_atr(highs, lows, closes, 14)
            if atr_value is not None:
                indicators['atr_14'] = atr_value
            
            # ADX et Directional Indicators
            adx, plus_di, minus_di = self.calculate_adx_smoothed(highs, lows, closes, 14)
            if adx is not None:
                indicators['adx_14'] = adx
            if plus_di is not None:
                indicators['plus_di'] = plus_di
            if minus_di is not None:
                indicators['minus_di'] = minus_di
            
            # Stochastic
            stoch_k, stoch_d = self.calculate_stochastic(highs, lows, closes)
            if stoch_k is not None:
                indicators['stoch_k'] = stoch_k
            if stoch_d is not None:
                indicators['stoch_d'] = stoch_d
            
            # ROC multiple périodes
            roc_10 = self.calculate_roc(closes, 10)
            if roc_10 is not None:
                indicators['roc_10'] = roc_10
            roc_20 = self.calculate_roc(closes, 20)
            if roc_20 is not None:
                indicators['roc_20'] = roc_20
            
            # OBV
            obv_value = self.calculate_obv(closes, volumes)
            if obv_value is not None:
                indicators['obv'] = obv_value
            
            # Stochastic RSI
            stoch_rsi_value = self.calculate_stoch_rsi(closes, 14)
            if stoch_rsi_value is not None:
                indicators['stoch_rsi'] = stoch_rsi_value
            
            # MFI (Money Flow Index)
            mfi_value = self.calculate_mfi(highs, lows, closes, volumes, 14)
            if mfi_value is not None:
                indicators['mfi_14'] = mfi_value
            
            # Williams %R
            williams_r_value = self.calculate_williams_r(highs, lows, closes, 14)
            if williams_r_value is not None:
                indicators['williams_r'] = williams_r_value
            
            # CCI (Commodity Channel Index)
            cci_value = self.calculate_cci(highs, lows, closes, 20)
            if cci_value is not None:
                indicators['cci_20'] = cci_value
            
            # VWAP
            vwap_value = self.calculate_vwap(highs, lows, closes, volumes, 10)
            if vwap_value is not None:
                indicators['vwap_10'] = vwap_value
            
            # Indicateurs supplémentaires
            trend_angle = self.calculate_trend_angle(closes, 10)
            if trend_angle is not None:
                indicators['trend_angle'] = trend_angle
            pivot_count = self.calculate_pivot_count(highs, lows, 5)
            if pivot_count is not None:
                indicators['pivot_count'] = pivot_count
            
            # Métriques additionnelles
            if len(closes) >= 10:
                momentum_val = self._calculate_momentum(closes, 10)
                if momentum_val is not None:
                    indicators['momentum_10'] = momentum_val
                
            if len(volumes) >= 20:
                vol_analysis = self._analyze_volume(volumes)
                indicators.update(vol_analysis)
                
            logger.debug(f"✅ Calculé {len(indicators)} indicateurs finaux")
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul indicateurs: {e}")
            logger.error(f"❌ Détails erreur: {str(e)}")
            import traceback
            logger.error(f"❌ Stack trace: {traceback.format_exc()}")
            
        return indicators
    
    def calculate_williams_r(self, highs: Union[List[float], np.ndarray], 
                            lows: Union[List[float], np.ndarray], 
                            closes: Union[List[float], np.ndarray], 
                            period: int = 14) -> Optional[float]:
        """
        Calcule Williams %R.
        
        Args:
            highs: Prix hauts
            lows: Prix bas  
            closes: Prix de clôture
            period: Période de calcul
            
        Returns:
            Valeur Williams %R (entre -100 et 0)
        """
        try:
            if len(highs) < period or len(lows) < period or len(closes) < period:
                return None
                
            highs = np.array(highs)
            lows = np.array(lows)
            closes = np.array(closes)
            
            # Prendre les derniers points pour le calcul
            recent_highs = highs[-period:]
            recent_lows = lows[-period:]
            current_close = closes[-1]
            
            highest_high = np.max(recent_highs)
            lowest_low = np.min(recent_lows)
            
            if highest_high == lowest_low:
                return -50.0  # Valeur par défaut
                
            williams_r = ((highest_high - current_close) / (highest_high - lowest_low)) * -100
            
            return float(williams_r)
            
        except Exception as e:
            logger.debug(f"Erreur calcul Williams %R: {e}")
            return None

    def calculate_cci(self, highs: Union[List[float], np.ndarray], 
                     lows: Union[List[float], np.ndarray], 
                     closes: Union[List[float], np.ndarray], 
                     period: int = 20) -> Optional[float]:
        """
        Calcule le Commodity Channel Index (CCI).
        
        Args:
            highs: Prix hauts
            lows: Prix bas
            closes: Prix de clôture
            period: Période de calcul
            
        Returns:
            Valeur CCI
        """
        try:
            if len(highs) < period or len(lows) < period or len(closes) < period:
                return None
                
            highs = np.array(highs)
            lows = np.array(lows)
            closes = np.array(closes)
            
            # Calcul du prix typique (TP)
            typical_price = (highs + lows + closes) / 3
            
            # Prendre les derniers points
            recent_tp = typical_price[-period:]
            
            # SMA du prix typique
            sma_tp = np.mean(recent_tp)
            
            # Écart absolu moyen
            mean_deviation = np.mean(np.abs(recent_tp - sma_tp))
            
            if mean_deviation == 0:
                return 0.0
                
            # CCI = (Prix Typique - SMA) / (0.015 * Écart Moyen)
            current_tp = typical_price[-1]
            cci = (current_tp - sma_tp) / (0.015 * mean_deviation)
            
            return float(cci)
            
        except Exception as e:
            logger.debug(f"Erreur calcul CCI: {e}")
            return None

    def calculate_vwap(self, highs: Union[List[float], np.ndarray], 
                      lows: Union[List[float], np.ndarray], 
                      closes: Union[List[float], np.ndarray], 
                      volumes: Union[List[float], np.ndarray], 
                      period: int = 10) -> Optional[float]:
        """
        Calcule le Volume Weighted Average Price (VWAP).
        
        Args:
            highs: Prix hauts
            lows: Prix bas
            closes: Prix de clôture
            volumes: Volumes
            period: Période de calcul
            
        Returns:
            Valeur VWAP
        """
        try:
            min_len = min(len(highs), len(lows), len(closes), len(volumes))
            if min_len < period:
                return None
                
            highs = np.array(highs)
            lows = np.array(lows)
            closes = np.array(closes)
            volumes = np.array(volumes)
            
            # Prix typique
            typical_price = (highs + lows + closes) / 3
            
            # Prendre les derniers points
            recent_tp = typical_price[-period:]
            recent_vol = volumes[-period:]
            
            # VWAP = Somme(Prix Typique * Volume) / Somme(Volume)
            total_vol = np.sum(recent_vol)
            if total_vol == 0:
                return float(np.mean(recent_tp))
                
            vwap = np.sum(recent_tp * recent_vol) / total_vol
            
            return float(vwap)
            
        except Exception as e:
            logger.debug(f"Erreur calcul VWAP: {e}")
            return None

    # =================== INDICATEURS ADDITIONNELS ===================
    
    def calculate_stoch_rsi(self, prices: Union[List[float], np.ndarray, pd.Series], 
                           period: Optional[int] = None) -> Optional[float]:
        """Calcule le Stochastic RSI"""
        if period is None:
            period = 14
            
        try:
            if len(prices) < period * 2:
                return None
                
            if self.talib_available:
                prices_array = self._to_numpy_array(prices)
                rsi_values = talib.RSI(prices_array, timeperiod=period)
                if len(rsi_values) < period:
                    return None
                stoch_rsi = talib.STOCH(rsi_values, rsi_values, rsi_values, 
                                     fastk_period=period, slowk_period=3, slowd_period=3)
                return round(float(stoch_rsi[0][-1]), 4) if not np.isnan(stoch_rsi[0][-1]) else None
            else:
                # Fallback manuel
                rsi_values = []
                for i in range(period, len(prices)):
                    rsi = self.calculate_rsi(prices[i-period:i+1], period)
                    if rsi is not None:
                        rsi_values.append(rsi)
                        
                if len(rsi_values) < period:
                    return None
                    
                recent_rsi = rsi_values[-period:]
                min_rsi = min(recent_rsi)
                max_rsi = max(recent_rsi)
                
                if max_rsi == min_rsi:
                    return 50.0
                    
                stoch_rsi = ((rsi_values[-1] - min_rsi) / (max_rsi - min_rsi)) * 100
                return round(stoch_rsi, 4)
                
        except Exception as e:
            logger.error(f"Erreur calcul Stochastic RSI: {e}")
            return None
    
    def calculate_mfi(self, highs: List[float], lows: List[float], 
                     closes: List[float], volumes: List[float], 
                     period: Optional[int] = None) -> Optional[float]:
        """Calcule le Money Flow Index"""
        if period is None:
            period = 14
            
        try:
            if len(closes) < period + 1:
                return None
                
            if self.talib_available:
                # Validation et alignement des arrays
                try:
                    highs_array, lows_array, closes_array, volumes_array = self._validate_and_align_arrays(highs, lows, closes, volumes)
                except Exception as e:
                    logger.error(f"Erreur validation arrays MFI: {e}")
                    return None
                
                mfi = talib.MFI(highs_array, lows_array, closes_array, volumes_array, timeperiod=period)
                return round(float(mfi[-1]), 4) if not np.isnan(mfi[-1]) else None
            else:
                # Fallback manuel
                money_flows = []
                for i in range(1, len(closes)):
                    typical_price = (highs[i] + lows[i] + closes[i]) / 3
                    prev_typical = (highs[i-1] + lows[i-1] + closes[i-1]) / 3
                    
                    money_flow = typical_price * volumes[i]
                    if typical_price > prev_typical:
                        money_flows.append((money_flow, 0.0))  # positive
                    else:
                        money_flows.append((0.0, money_flow))  # negative
                
                if len(money_flows) < period:
                    return None
                    
                recent_flows = money_flows[-period:]
                positive_flow = sum(flow[0] for flow in recent_flows)
                negative_flow = sum(flow[1] for flow in recent_flows)
                
                if negative_flow == 0:
                    return 100.0
                    
                money_ratio = positive_flow / negative_flow
                mfi = 100 - (100 / (1 + money_ratio))
                return round(mfi, 4)
                
        except Exception as e:
            logger.error(f"Erreur calcul MFI: {e}")
            return None
    
    def calculate_trend_angle(self, prices: List[float], period: Optional[int] = None) -> Optional[float]:
        """Calcule l'angle de tendance (régression linéaire)"""
        if period is None:
            period = 10
            
        try:
            if len(prices) < period:
                return None
                
            # Régression linéaire sur les derniers points
            y = np.array(prices[-period:], dtype=float)
            x = np.arange(len(y))
            
            # Calcul de la pente
            n = len(x)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x * x) - np.sum(x) ** 2)
            
            # Conversion en angle (en degrés)
            angle = np.degrees(np.arctan(slope))
            return round(float(angle), 4)
            
        except Exception as e:
            logger.error(f"Erreur calcul trend angle: {e}")
            return None
    
    def calculate_pivot_count(self, highs: List[float], lows: List[float], 
                             period: Optional[int] = None) -> Optional[int]:
        """Compte les points pivots (hauts/bas locaux)"""
        if period is None:
            period = 5
            
        try:
            if len(highs) < period * 2 + 1:
                return 0
                
            pivot_count = 0
            
            # Chercher les pivots hauts
            for i in range(period, len(highs) - period):
                is_pivot_high = True
                for j in range(i - period, i + period + 1):
                    if j != i and highs[j] >= highs[i]:
                        is_pivot_high = False
                        break
                if is_pivot_high:
                    pivot_count += 1
            
            # Chercher les pivots bas
            for i in range(period, len(lows) - period):
                is_pivot_low = True
                for j in range(i - period, i + period + 1):
                    if j != i and lows[j] <= lows[i]:
                        is_pivot_low = False
                        break
                if is_pivot_low:
                    pivot_count += 1
                    
            return pivot_count
            
        except Exception as e:
            logger.error(f"Erreur calcul pivot count: {e}")
            return 0
    
    # =================== MÉTHODES UTILITAIRES ===================
    
    def _to_numpy_array(self, data: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
        """Convertit les données en numpy array"""
        if isinstance(data, pd.Series):
            return data.values
        elif isinstance(data, list):
            return np.array(data, dtype=float)
        elif isinstance(data, np.ndarray):
            return data.astype(float)
        else:
            raise ValueError(f"Type de données non supporté: {type(data)}")
    
    def _validate_and_align_arrays(self, *arrays) -> Tuple[np.ndarray, ...]:
        """
        Valide et aligne les arrays pour s'assurer qu'ils ont la même longueur.
        Troncature à la longueur minimum pour éviter les erreurs talib.
        
        Args:
            *arrays: Arrays à valider et aligner
            
        Returns:
            Tuple d'arrays alignés
        """
        if not arrays:
            return ()
        
        # Convertir tous en numpy arrays
        np_arrays = [self._to_numpy_array(arr) for arr in arrays]
        
        # Vérifier les longueurs
        lengths = [len(arr) for arr in np_arrays]
        min_length = min(lengths)
        max_length = max(lengths)
        
        # Log détaillé si différences détectées
        if min_length != max_length:
            data_loss = max_length - min_length
            data_loss_pct = (data_loss / max_length) * 100 if max_length > 0 else 0
            
            if data_loss_pct > 10:  # Perte de plus de 10% des données
                logger.error(f"❌ PERTE DE DONNÉES CRITIQUE: Arrays désalignés {lengths}")
                logger.error(f"❌ Perte: {data_loss} points ({data_loss_pct:.1f}%) - INDICATEURS COMPROMIS")
            else:
                logger.warning(f"⚠️ Arrays de longueurs différentes: {lengths}")
            
            logger.warning(f"🔧 Troncature forcée à {min_length} éléments pour alignement")
            
            # Tronquer tous les arrays à la longueur minimum (garder les plus récents)
            aligned_arrays = [arr[-min_length:] for arr in np_arrays]
            
            # Vérification post-alignement
            post_lengths = [len(arr) for arr in aligned_arrays]
            if all(l == min_length for l in post_lengths):
                logger.info(f"✅ Post-alignement réussi: {post_lengths}")
            else:
                logger.error(f"❌ Échec post-alignement: {post_lengths}")
            
            return tuple(aligned_arrays)
        
        return tuple(np_arrays)
    
    def _calculate_momentum(self, prices: List[float], period: int) -> Optional[float]:
        """Calcule le momentum (changement de prix sur période)"""
        if len(prices) < period + 1:
            return None
        return round(((prices[-1] / prices[-period-1]) - 1) * 100, 2)
    
    def _analyze_volume(self, volumes: List[float]) -> Dict[str, float]:
        """Analyse du volume"""
        if len(volumes) < 20:
            return {}
            
        recent_vol = volumes[-20:]
        avg_vol = sum(recent_vol) / len(recent_vol)
        current_vol = volumes[-1]
        
        return {
            'volume_ratio': round(current_vol / avg_vol if avg_vol > 0 else 1.0, 2),
            'avg_volume_20': round(avg_vol, 2)
        }


# Instance globale pour utilisation dans tous les services
indicators = TechnicalIndicators()

# Cache global pour les indicateurs incrémentaux avec persistence Redis
class IndicatorCache:
    """Cache persistant pour les valeurs précédentes des indicateurs incrémentaux"""
    def __init__(self):
        self.cache = {}  # Cache local pour performance
        self.redis_client = None
        self._init_redis()
        
        # TTL pour la persistence (48h au lieu de 1h)
        self.cache_ttl = 48 * 3600  # 48 heures
        self.auto_save_interval = 300  # Sauvegarde auto toutes les 5 minutes
        self.last_save_time = 0
    
    def _init_redis(self):
        """Initialise la connexion Redis pour la persistence"""
        try:
            from shared.src.redis_client import RedisClient
            self.redis_client = RedisClient()
            logger.info("✅ IndicatorCache connecté à Redis pour persistence")
        except Exception as e:
            logger.warning(f"⚠️ IndicatorCache sans Redis: {e}")
            self.redis_client = None
    
    def get_key(self, symbol: str, timeframe: str, indicator: str) -> str:
        return f"indicators:{symbol}:{timeframe}:{indicator}"
    
    def get(self, symbol: str, timeframe: str, indicator: str, default=None):
        key = self.get_key(symbol, timeframe, indicator)
        
        # Vérifier le cache local d'abord
        if key in self.cache:
            return self.cache[key]
        
        # Fallback vers Redis pour restoration
        if self.redis_client:
            try:
                redis_value = self.redis_client.get(key)
                if redis_value is not None:
                    # RedisClientPool fait déjà le parsing JSON automatiquement
                    # Si c'est encore une string, parser manuellement
                    if isinstance(redis_value, str):
                        try:
                            parsed_value = json.loads(redis_value)
                        except json.JSONDecodeError:
                            parsed_value = redis_value
                    else:
                        parsed_value = redis_value
                    
                    self.cache[key] = parsed_value
                    logger.debug(f"🔄 Indicateur restauré depuis Redis: {key}")
                    return parsed_value
            except Exception as e:
                logger.warning(f"Erreur lecture Redis pour {key}: {e}")
        
        return default
    
    def set(self, symbol: str, timeframe: str, indicator: str, value):
        key = self.get_key(symbol, timeframe, indicator)
        
        # Mettre à jour le cache local
        self.cache[key] = value
        
        # Sauvegarde automatique périodique (non bloquante)
        current_time = time.time()
        if current_time - self.last_save_time > self.auto_save_interval:
            self._save_to_redis_async(key, value)
            self.last_save_time = current_time
        else:
            # Sauvegarde immédiate pour les valeurs critiques
            self._save_to_redis_async(key, value)
    
    def _save_to_redis_async(self, key: str, value):
        """Sauvegarde asynchrone vers Redis (non bloquante)"""
        if self.redis_client:
            try:
                # Sérialiser la valeur
                serialized_value = json.dumps(value) if not isinstance(value, (str, int, float)) else value
                self.redis_client.set(key, serialized_value, expiration=self.cache_ttl)
                logger.debug(f"💾 Indicateur sauvegardé: {key}")
            except Exception as e:
                logger.warning(f"Erreur sauvegarde Redis {key}: {e}")
    
    def clear_symbol(self, symbol: str, force_clear: bool = False):
        """
        Efface le cache pour un symbole
        
        Args:
            symbol: Symbole à effacer
            force_clear: Si True, efface même les données persistées (utiliser avec prudence)
        """
        if force_clear:
            # ATTENTION: Utilisé seulement si corruption de données
            logger.warning(f"🗑️ EFFACEMENT FORCÉ du cache pour {symbol}")
            
            # Effacer du cache local
            keys_to_remove = [k for k in self.cache.keys() if f":{symbol}:" in k]
            for key in keys_to_remove:
                del self.cache[key]
            
            # Effacer de Redis
            if self.redis_client:
                try:
                    redis_pattern = f"indicators:{symbol}:*"
                    redis_keys = list(self.redis_client.redis.scan_iter(match=redis_pattern))
                    for redis_key in redis_keys:
                        self.redis_client.delete(redis_key)
                    logger.warning(f"Redis cache effacé pour {symbol}")
                except Exception as e:
                    logger.error(f"Erreur effacement Redis pour {symbol}: {e}")
        else:
            # Mode normal: NE PAS effacer, juste log
            logger.info(f"📊 Cache préservé pour {symbol} (continuité des indicateurs)")
    
    def restore_from_redis(self, symbol: Optional[str] = None):
        """
        Restaure le cache depuis Redis au démarrage
        
        Args:
            symbol: Si spécifié, restore seulement ce symbole, sinon tous
        """
        if not self.redis_client:
            logger.warning("Pas de Redis, impossible de restaurer le cache")
            return 0
        
        restored_count = 0
        try:
            pattern = f"indicators:{symbol}:*" if symbol else "indicators:*"
            redis_keys = list(self.redis_client.redis.scan_iter(match=pattern))
            
            for redis_key in redis_keys:
                try:
                    redis_value = self.redis_client.get(redis_key)
                    if redis_value is not None:
                        # RedisClientPool fait déjà le parsing JSON automatiquement
                        if isinstance(redis_value, str):
                            try:
                                parsed_value = json.loads(redis_value)
                            except json.JSONDecodeError:
                                parsed_value = redis_value
                        else:
                            parsed_value = redis_value
                        
                        self.cache[redis_key] = parsed_value
                        restored_count += 1
                except Exception as e:
                    logger.warning(f"Erreur restoration {redis_key}: {e}")
            
            logger.info(f"🔄 {restored_count} indicateurs restaurés depuis Redis")
            return restored_count
            
        except Exception as e:
            logger.error(f"Erreur restoration globale: {e}")
            return 0
    
    def force_save_all(self):
        """Force la sauvegarde de tout le cache vers Redis"""
        if not self.redis_client:
            return
        
        saved_count = 0
        for key, value in self.cache.items():
            try:
                serialized_value = json.dumps(value) if not isinstance(value, (str, int, float)) else value
                self.redis_client.set(key, serialized_value, expiration=self.cache_ttl)
                saved_count += 1
            except Exception as e:
                logger.warning(f"Erreur sauvegarde {key}: {e}")
        
        logger.info(f"💾 {saved_count} indicateurs sauvegardés vers Redis")
        return saved_count
    
    def get_cache_stats(self) -> dict:
        """Retourne les statistiques du cache"""
        return {
            "local_entries": len(self.cache),
            "redis_connected": self.redis_client is not None,
            "cache_ttl_hours": self.cache_ttl / 3600,
            "auto_save_interval_minutes": self.auto_save_interval / 60
        }
    
    def get_all_indicators(self, symbol: str, timeframe: str) -> dict:
        """
        Récupère tous les indicateurs stockés dans le cache pour un symbole/timeframe
        Priorise le cache local, puis Redis
        """
        prefix = f"indicators:{symbol}:{timeframe}:"
        indicators = {}
        
        # Chercher dans le cache local d'abord
        for key, value in self.cache.items():
            if key.startswith(prefix):
                indicator_name = key[len(prefix):]
                indicators[indicator_name] = value
        
        # Compléter avec Redis si nécessaire
        if self.redis_client and len(indicators) == 0:
            try:
                redis_keys = list(self.redis_client.redis.scan_iter(match=f"{prefix}*"))
                for redis_key in redis_keys:
                    try:
                        redis_value = self.redis_client.get(redis_key)
                        if redis_value is not None:
                            indicator_name = redis_key[len(prefix):]
                            # RedisClientPool fait déjà le parsing JSON automatiquement
                            if isinstance(redis_value, str):
                                try:
                                    parsed_value = json.loads(redis_value)
                                except json.JSONDecodeError:
                                    parsed_value = redis_value
                            else:
                                parsed_value = redis_value
                            
                            indicators[indicator_name] = parsed_value
                            # Restaurer dans le cache local
                            self.cache[redis_key] = parsed_value
                    except Exception as e:
                        logger.warning(f"Erreur lecture indicateur {redis_key}: {e}")
            except Exception as e:
                logger.warning(f"Erreur scan Redis pour {prefix}: {e}")
        
        return indicators

# Instance globale du cache
indicator_cache = IndicatorCache()

# Fonctions de convenance pour compatibilité
def calculate_rsi(prices: Union[List[float], np.ndarray, pd.Series], period: int = 14) -> Optional[float]:
    """Fonction de convenance pour calculer RSI"""
    return indicators.calculate_rsi(prices, period)

def calculate_ema(prices: Union[List[float], np.ndarray, pd.Series], period: int) -> Optional[float]:
    """Fonction de convenance pour calculer EMA"""
    return indicators.calculate_ema(prices, period)

def calculate_ema_incremental(current_price: float, previous_ema: Optional[float], period: int) -> float:
    """Fonction de convenance pour EMA incrémentale"""
    return indicators.calculate_ema_incremental(current_price, previous_ema, period)

def calculate_macd(prices: Union[List[float], np.ndarray, pd.Series]) -> Dict[str, Optional[float]]:
    """Fonction de convenance pour calculer MACD"""
    return indicators.calculate_macd(prices)

def calculate_macd_incremental(current_price: float, prev_ema_fast: Optional[float], 
                              prev_ema_slow: Optional[float], prev_macd_signal: Optional[float]) -> Dict[str, Optional[float]]:
    """Fonction de convenance pour MACD incrémental"""
    return indicators.calculate_macd_incremental(current_price, prev_ema_fast, prev_ema_slow, prev_macd_signal)

def calculate_bollinger_bands(prices: Union[List[float], np.ndarray, pd.Series], 
                            period: int = 20, std_dev: float = 2.0) -> Dict[str, Optional[float]]:
    """Fonction de convenance pour calculer Bollinger Bands"""
    return indicators.calculate_bollinger_bands(prices, period, std_dev)

def calculate_atr(highs: Union[List[float], np.ndarray, pd.Series],
                 lows: Union[List[float], np.ndarray, pd.Series],
                 closes: Union[List[float], np.ndarray, pd.Series],
                 period: int = 14) -> Optional[float]:
    """Fonction de convenance pour calculer ATR"""
    return indicators.calculate_atr(highs, lows, closes, period)

def calculate_adx(highs: Union[List[float], np.ndarray, pd.Series],
                 lows: Union[List[float], np.ndarray, pd.Series],
                 closes: Union[List[float], np.ndarray, pd.Series],
                 period: int = 14) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Fonction de convenance pour calculer ADX et DI+/-"""
    return indicators.calculate_adx(highs, lows, closes, period)

def calculate_stochastic(highs: Union[List[float], np.ndarray, pd.Series],
                        lows: Union[List[float], np.ndarray, pd.Series],
                        closes: Union[List[float], np.ndarray, pd.Series],
                        fastk_period: int = 14,
                        slowk_period: int = 3,
                        slowd_period: int = 3) -> Tuple[Optional[float], Optional[float]]:
    """Fonction de convenance pour calculer Stochastic"""
    return indicators.calculate_stochastic(highs, lows, closes, fastk_period, slowk_period, slowd_period)

def calculate_roc(prices: Union[List[float], np.ndarray, pd.Series], period: int = 10) -> Optional[float]:
    """Fonction de convenance pour calculer ROC"""
    return indicators.calculate_roc(prices, period)

def calculate_obv(prices: Union[List[float], np.ndarray, pd.Series],
                 volumes: Union[List[float], np.ndarray, pd.Series]) -> Optional[float]:
    """Fonction de convenance pour calculer OBV"""
    return indicators.calculate_obv(prices, volumes)

def calculate_all_indicators(highs: List[float], lows: List[float], 
                           closes: List[float], volumes: List[float]) -> Dict[str, Any]:
    """Fonction de convenance pour calculer tous les indicateurs"""
    return indicators.calculate_all_indicators(highs, lows, closes, volumes)

def calculate_indicators_incremental(symbol: str, timeframe: str, current_candle: Dict) -> Dict[str, Any]:
    """
    Calcule les indicateurs de manière incrémentale pour éviter les dents de scie.
    
    Args:
        symbol: Symbole tradé
        timeframe: Intervalle de temps
        current_candle: Bougie actuelle avec keys: open, high, low, close, volume
        
    Returns:
        Dict avec tous les indicateurs calculés de manière incrémentale
    """
    result: Dict[str, Any] = {}
    current_price = current_candle['close']
    
    if current_price is None:
        return result
    
    # EMA 7, 26, 99
    for period in [7, 26, 99]:
        prev_ema = indicator_cache.get(symbol, timeframe, f'ema_{period}')
        new_ema = indicators.calculate_ema_incremental(current_price, prev_ema, period)
        if new_ema is not None:
            result[f'ema_{period}'] = new_ema
            indicator_cache.set(symbol, timeframe, f'ema_{period}', new_ema)
    
    # MACD incrémental
    prev_ema_fast = indicator_cache.get(symbol, timeframe, 'macd_ema_fast')
    prev_ema_slow = indicator_cache.get(symbol, timeframe, 'macd_ema_slow')
    prev_macd_signal = indicator_cache.get(symbol, timeframe, 'macd_signal')
    
    macd_result = indicators.calculate_macd_incremental(
        current_price, prev_ema_fast, prev_ema_slow, prev_macd_signal
    )
    
    # Ajouter seulement les valeurs non-None
    for key in ['macd_line', 'macd_signal', 'macd_histogram']:
        if macd_result[key] is not None:
            result[key] = macd_result[key]
    
    # Mettre à jour le cache MACD
    indicator_cache.set(symbol, timeframe, 'macd_ema_fast', macd_result['ema_fast'])
    indicator_cache.set(symbol, timeframe, 'macd_ema_slow', macd_result['ema_slow'])
    indicator_cache.set(symbol, timeframe, 'macd_signal', macd_result['macd_signal'])
    
    # SMA 20, 50 (pour comparaison)
    for period in [20, 50]:
        prev_sma = indicator_cache.get(symbol, timeframe, f'sma_{period}')
        # Pour SMA, on a besoin du prix le plus ancien, simplification ici
        sma_value = indicators.calculate_sma_incremental(current_price, prev_sma, period)
        if sma_value is not None:
            result[f'sma_{period}'] = sma_value
    
    # Autres indicateurs qui ne nécessitent pas de cache (recalcul acceptable)
    rsi_value = indicators.calculate_rsi([current_price], 14)  # Simplifié
    if rsi_value is not None:
        result['rsi_14'] = rsi_value
    
    return result