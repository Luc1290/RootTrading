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
import logging
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
        
        # Configuration des périodes depuis config.py
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
        """
        if len(data) < period:
            return data
        
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    # =================== EMA ===================
    
    def calculate_ema(self, prices: Union[List[float], np.ndarray, pd.Series], 
                     period: int) -> Optional[float]:
        """
        Calcule l'EMA (Exponential Moving Average).
        
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
        Calcule le MACD complet (line, signal, histogram).
        
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
                    nbdevdn=std_dev, matype=0
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
        highs_array = self._to_numpy_array(highs)
        lows_array = self._to_numpy_array(lows)
        closes_array = self._to_numpy_array(closes)
        
        if len(highs_array) < period or len(lows_array) < period or len(closes_array) < period:
            return None
            
        if self.talib_available:
            try:
                atr_values = talib.ATR(highs_array, lows_array, closes_array, timeperiod=period)
                return float(atr_values[-1]) if not np.isnan(atr_values[-1]) else None
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
            true_ranges.append(max(tr1, tr2, tr3))
            
        if len(true_ranges) < period:
            return None
            
        atr = np.mean(true_ranges[-period:])
        return round(float(atr), 6)
    
    # =================== SMA ===================
    
    def calculate_sma(self, prices: Union[List[float], np.ndarray, pd.Series], 
                     period: int) -> Optional[float]:
        """
        Calcule la SMA (Simple Moving Average).
        
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
        highs_array = self._to_numpy_array(highs)
        lows_array = self._to_numpy_array(lows)
        closes_array = self._to_numpy_array(closes)
        
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
        highs_array = self._to_numpy_array(highs)
        lows_array = self._to_numpy_array(lows)
        closes_array = self._to_numpy_array(closes)
        
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
    
    # =================== OBV ===================
    
    def calculate_obv(self, prices: Union[List[float], np.ndarray, pd.Series],
                     volumes: Union[List[float], np.ndarray, pd.Series]) -> Optional[float]:
        """
        Calcule l'OBV (On Balance Volume).
        
        Returns:
            Valeur OBV ou None si impossible
        """
        prices_array = self._to_numpy_array(prices)
        volumes_array = self._to_numpy_array(volumes)
        
        if len(prices_array) < 2 or len(volumes_array) < 2:
            return None
            
        if self.talib_available:
            try:
                obv_values = talib.OBV(prices_array, volumes_array)
                return float(obv_values[-1]) if not np.isnan(obv_values[-1]) else None
            except Exception as e:
                logger.warning(f"Erreur talib OBV: {e}")
                
        return None
    
    # =================== CALCUL COMPLET ===================
    
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
            indicators['rsi_14'] = self.calculate_rsi(closes, 14)
            
            # EMAs multiples
            for period in [12, 26, 50]:
                indicators[f'ema_{period}'] = self.calculate_ema(closes, period)
            
            # SMAs
            for period in [20, 50]:
                indicators[f'sma_{period}'] = self.calculate_sma(closes, period)
                
            # MACD
            macd_data = self.calculate_macd(closes)
            indicators.update(macd_data)
            
            # Bollinger Bands
            bb_data = self.calculate_bollinger_bands(closes)
            indicators.update(bb_data)
            
            # ATR
            indicators['atr_14'] = self.calculate_atr(highs, lows, closes, 14)
            
            # ADX et Directional Indicators
            adx, plus_di, minus_di = self.calculate_adx(highs, lows, closes, 14)
            if adx is not None:
                indicators['adx_14'] = adx
                indicators['plus_di'] = plus_di
                indicators['minus_di'] = minus_di
            
            # Stochastic
            stoch_k, stoch_d = self.calculate_stochastic(highs, lows, closes)
            if stoch_k is not None:
                indicators['stoch_k'] = stoch_k
                indicators['stoch_d'] = stoch_d
            
            # ROC
            indicators['roc_10'] = self.calculate_roc(closes, 10)
            
            # OBV
            indicators['obv'] = self.calculate_obv(closes, volumes)
            
            # Métriques additionnelles
            if len(closes) >= 10:
                indicators['momentum_10'] = self._calculate_momentum(closes, 10)
                
            if len(volumes) >= 20:
                vol_analysis = self._analyze_volume(volumes)
                indicators.update(vol_analysis)
                
            logger.debug(f"✅ Calculé {len(indicators)} indicateurs")
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul indicateurs: {e}")
            
        return indicators
    
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

# Fonctions de convenance pour compatibilité
def calculate_rsi(prices: Union[List[float], np.ndarray, pd.Series], period: int = 14) -> Optional[float]:
    """Fonction de convenance pour calculer RSI"""
    return indicators.calculate_rsi(prices, period)

def calculate_ema(prices: Union[List[float], np.ndarray, pd.Series], period: int) -> Optional[float]:
    """Fonction de convenance pour calculer EMA"""
    return indicators.calculate_ema(prices, period)

def calculate_macd(prices: Union[List[float], np.ndarray, pd.Series]) -> Dict[str, Optional[float]]:
    """Fonction de convenance pour calculer MACD"""
    return indicators.calculate_macd(prices)

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