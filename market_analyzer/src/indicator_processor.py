"""
Market Analyzer Indicator Processor
Appelle DIRECTEMENT tous les modules indicator/detector existants et sauvegarde en DB.
Architecture simple : récupère données → appelle modules → sauvegarde résultats.
"""

import logging
import asyncio
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import asyncpg
import sys
import os

# Ajouter les chemins pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import get_db_config

# Import DIRECT de tous vos modules existants
from market_analyzer.indicators import (
    calculate_rsi, calculate_ema, calculate_sma, calculate_macd_series,
    calculate_bollinger_bands_series, calculate_atr, 
    calculate_stochastic_series, calculate_williams_r, calculate_cci,
    calculate_obv_series, calculate_vwap_series
)

# Import ADX complet
from market_analyzer.indicators.trend.adx import calculate_adx_full

# Import des moyennes avancées
from market_analyzer.indicators.trend.moving_averages import (
    calculate_wma, calculate_dema, calculate_tema, calculate_hull_ma, calculate_adaptive_ma
)

# Import des indicateurs de momentum
from market_analyzer.indicators.momentum.momentum import (
    calculate_momentum, calculate_roc, calculate_price_oscillator
)
from market_analyzer.indicators.momentum.rsi import calculate_stoch_rsi

# Import des indicateurs de volatilité  
from market_analyzer.indicators.volatility.atr import calculate_natr, volatility_regime
from market_analyzer.indicators.volatility.bollinger import calculate_bollinger_squeeze

# Import des indicateurs de volume
from market_analyzer.indicators.volume.obv import calculate_obv_ma, calculate_obv_oscillator
from market_analyzer.indicators.volume.vwap import calculate_vwap_bands, calculate_vwap_quote_series
from market_analyzer.indicators.volume.advanced_metrics import (
    calculate_quote_volume_ratio, calculate_avg_trade_size, calculate_trade_intensity
)

# Import des détecteurs
from market_analyzer.detectors.regime_detector import RegimeDetector
from market_analyzer.detectors.support_resistance_detector import SupportResistanceDetector
from market_analyzer.detectors.volume_context_analyzer import VolumeContextAnalyzer
from market_analyzer.detectors.spike_detector import SpikeDetector

logger = logging.getLogger(__name__)

class IndicatorProcessor:
    """
    Processeur simple qui appelle vos modules existants et sauvegarde en DB.
    """
    
    def __init__(self):
        self.db_pool = None
        self.running = False
        
        # Initialiser les détecteurs
        self.regime_detector = RegimeDetector()
        self.sr_detector = SupportResistanceDetector()
        self.volume_analyzer = VolumeContextAnalyzer()
        self.spike_detector = SpikeDetector()
        
        logger.info("🧮 IndicatorProcessor initialisé")

    async def initialize(self):
        """Initialise la connexion à la base de données."""
        try:
            db_config = get_db_config()
            self.db_pool = await asyncpg.create_pool(
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                user=db_config['user'],
                password=db_config['password'],
                min_size=2,
                max_size=10
            )
            self.running = True
            logger.info("✅ IndicatorProcessor connecté à la base de données")
        except Exception as e:
            logger.error(f"❌ Erreur connexion DB: {e}")
            raise

    async def process_new_data(self, symbol: str, timeframe: str, timestamp: datetime):
        """
        Traite une nouvelle donnée OHLCV : appelle tous vos modules et sauvegarde.
        
        Args:
            symbol: Symbole (ex: BTCUSDC)
            timeframe: Timeframe (ex: 1m, 5m, 1h)
            timestamp: Timestamp de la nouvelle donnée
        """
        try:
            logger.debug(f"🔄 Traitement {symbol} {timeframe} @ {timestamp}")
            
            # Récupérer les données historiques nécessaires
            ohlcv_data = await self._get_historical_data(symbol, timeframe, limit=500)
            
            if len(ohlcv_data) < 20:
                logger.warning(f"⚠️ Pas assez de données pour {symbol} {timeframe}: {len(ohlcv_data)} < 20")
                return
            
            # Appeler TOUS vos modules et collecter les résultats
            indicators_data = await self._call_all_indicator_modules(symbol, timeframe, ohlcv_data, timestamp)
            
            # Sauvegarder en DB
            await self._save_indicators_to_db(indicators_data)
            
            logger.info(f"✅ Indicateurs sauvegardés: {symbol} {timeframe}")
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement {symbol} {timeframe}: {e}")

    async def _get_historical_data(self, symbol: str, timeframe: str, limit: int = 500) -> List[Dict]:
        """Récupère les données OHLCV historiques depuis market_data."""
        query = """
            SELECT time, open, high, low, close, volume,
                   quote_asset_volume, number_of_trades
            FROM market_data 
            WHERE symbol = $1 AND timeframe = $2
            ORDER BY time DESC
            LIMIT $3
        """
        
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, timeframe, limit)
            
            # Inverser pour avoir l'ordre chronologique
            data = []
            for row in reversed(rows):
                data.append({
                    'time': row['time'],
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']),
                    'quote_asset_volume': float(row['quote_asset_volume']) if row['quote_asset_volume'] else 0,
                    'number_of_trades': row['number_of_trades'] if row['number_of_trades'] else 0
                })
            
            return data

    async def _call_all_indicator_modules(self, symbol: str, timeframe: str, ohlcv_data: List[Dict], timestamp: datetime) -> Dict:
        """
        Appelle DIRECTEMENT tous vos modules indicator/detector existants.
        """
        start_time = time.time()
        
        # Extraire les arrays
        opens = [d['open'] for d in ohlcv_data]
        highs = [d['high'] for d in ohlcv_data]
        lows = [d['low'] for d in ohlcv_data]
        closes = [d['close'] for d in ohlcv_data]
        volumes = [d['volume'] for d in ohlcv_data]
        
        # Résultat final
        indicators = {
            'time': timestamp,
            'symbol': symbol,
            'timeframe': timeframe,
            'analysis_timestamp': datetime.now(),
            'analyzer_version': '1.0'
        }
        
        try:
            # === APPEL DIRECT DE VOS MODULES INDICATORS ===
            
            # RSI (plusieurs périodes, sans cache pour corriger le problème)
            if len(closes) >= 14:
                indicators['rsi_14'] = self._safe_call(lambda: calculate_rsi(closes, 14))
            if len(closes) >= 21:
                indicators['rsi_21'] = self._safe_call(lambda: calculate_rsi(closes, 21))
            
            # Stochastic RSI
            if len(closes) >= 14:
                indicators['stoch_rsi'] = self._safe_call(lambda: calculate_stoch_rsi(closes, 14, 14))
            
            # EMAs (sans cache pour corriger le problème)
            if len(closes) >= 7:
                indicators['ema_7'] = self._safe_call(lambda: calculate_ema(closes, 7))
            if len(closes) >= 12:
                indicators['ema_12'] = self._safe_call(lambda: calculate_ema(closes, 12))
            if len(closes) >= 26:
                indicators['ema_26'] = self._safe_call(lambda: calculate_ema(closes, 26))
            if len(closes) >= 50:
                indicators['ema_50'] = self._safe_call(lambda: calculate_ema(closes, 50))
            if len(closes) >= 99:
                indicators['ema_99'] = self._safe_call(lambda: calculate_ema(closes, 99))
            
            # SMAs
            if len(closes) >= 20:
                indicators['sma_20'] = self._safe_call(lambda: calculate_sma(closes, 20))
            if len(closes) >= 50:
                indicators['sma_50'] = self._safe_call(lambda: calculate_sma(closes, 50))
            
            # Moyennes avancées
            if len(closes) >= 20:
                indicators['wma_20'] = self._safe_call(lambda: calculate_wma(closes, 20))
                indicators['hull_20'] = self._safe_call(lambda: calculate_hull_ma(closes, 20))
            if len(closes) >= 12:
                indicators['dema_12'] = self._safe_call(lambda: calculate_dema(closes, 12))
                indicators['tema_12'] = self._safe_call(lambda: calculate_tema(closes, 12))
            if len(closes) >= 14:
                indicators['kama_14'] = self._safe_call(lambda: calculate_adaptive_ma(closes, 14))
            
            # MACD
            if len(closes) >= 26:
                macd_series = self._safe_call(lambda: calculate_macd_series(closes))
                if macd_series and isinstance(macd_series, dict):
                    # macd_series is a dict with 'macd_line', 'macd_signal', 'macd_histogram' keys
                    # Each key contains a list, so we get the last value
                    macd_line_series = macd_series.get('macd_line', [])
                    macd_signal_series = macd_series.get('macd_signal', [])
                    macd_histogram_series = macd_series.get('macd_histogram', [])
                    
                    indicators.update({
                        'macd_line': macd_line_series[-1] if macd_line_series else None,
                        'macd_signal': macd_signal_series[-1] if macd_signal_series else None,
                        'macd_histogram': macd_histogram_series[-1] if macd_histogram_series else None
                    })
                
                # PPO
                indicators['ppo'] = self._safe_call(lambda: calculate_price_oscillator(closes, 12, 26))
            
            # ADX
            if len(closes) >= 14:
                adx_full = self._safe_call(lambda: calculate_adx_full(highs, lows, closes, 14))
                if adx_full and isinstance(adx_full, dict):
                    indicators.update({
                        'adx_14': adx_full.get('adx'),
                        'plus_di': adx_full.get('plus_di'),
                        'minus_di': adx_full.get('minus_di'),
                        'dx': adx_full.get('dx'),
                        'adxr': adx_full.get('adxr')
                    })
            
            # ATR et volatilité (calculer d'abord car utilisé par Keltner)
            atr = None
            if len(closes) >= 14:
                atr = self._safe_call(lambda: calculate_atr(highs, lows, closes, 14))
                indicators['atr_14'] = atr
                
                if atr and closes[-1] > 0:
                    indicators['natr'] = self._safe_call(lambda: calculate_natr(highs, lows, closes, 14))
                    
                    # Régime de volatilité
                    volatility_reg = self._safe_call(lambda: volatility_regime(highs, lows, closes, 14, 20))  # Réduire lookback de 50 à 20
                    indicators['volatility_regime'] = volatility_reg
                    
                    # ATR stop loss calculé manuellement
                    atr_multiplier = 2.0
                    indicators.update({
                        'atr_stop_long': closes[-1] - (atr * atr_multiplier),
                        'atr_stop_short': closes[-1] + (atr * atr_multiplier)
                    })

            # Bollinger Bands
            if len(closes) >= 20:
                bb_series = self._safe_call(lambda: calculate_bollinger_bands_series(closes, 20, 2.0))
                if bb_series and isinstance(bb_series, dict):
                    # bb_series is a dict with lists for each key
                    bb_upper_series = bb_series.get('upper', [])
                    bb_middle_series = bb_series.get('middle', [])
                    bb_lower_series = bb_series.get('lower', [])
                    bb_percent_b_series = bb_series.get('percent_b', [])
                    bb_bandwidth_series = bb_series.get('bandwidth', [])
                    
                    indicators.update({
                        'bb_upper': bb_upper_series[-1] if bb_upper_series else None,
                        'bb_middle': bb_middle_series[-1] if bb_middle_series else None,
                        'bb_lower': bb_lower_series[-1] if bb_lower_series else None,
                        'bb_position': bb_percent_b_series[-1] if bb_percent_b_series else None,
                        'bb_width': bb_bandwidth_series[-1] if bb_bandwidth_series else None
                    })
                
                # Keltner Channels (calculé manuellement avec ATR)
                if atr:
                    ema_20 = self._safe_call(lambda: calculate_ema(closes, 20))
                    if ema_20:
                        keltner_multiplier = 2.0
                        indicators.update({
                            'keltner_upper': ema_20 + (atr * keltner_multiplier),
                            'keltner_lower': ema_20 - (atr * keltner_multiplier)
                        })
            
            # Stochastic
            if len(closes) >= 14:
                stoch_series = self._safe_call(lambda: calculate_stochastic_series(highs, lows, closes, 14, 3))
                if stoch_series and isinstance(stoch_series, dict):
                    # stoch_series is a dict with 'k' and 'd' keys containing lists
                    stoch_k_series = stoch_series.get('k', [])
                    stoch_d_series = stoch_series.get('d', [])
                    
                    indicators.update({
                        'stoch_k': stoch_k_series[-1] if stoch_k_series else None,
                        'stoch_d': stoch_d_series[-1] if stoch_d_series else None
                    })
                
                # Fast Stochastic
                fast_stoch_series = self._safe_call(lambda: calculate_stochastic_series(highs, lows, closes, 14, 1))
                if fast_stoch_series and isinstance(fast_stoch_series, dict):
                    # fast_stoch_series is a dict with 'k' and 'd' keys containing lists
                    fast_k_series = fast_stoch_series.get('k', [])
                    fast_d_series = fast_stoch_series.get('d', [])
                    
                    indicators.update({
                        'stoch_fast_k': fast_k_series[-1] if fast_k_series else None,
                        'stoch_fast_d': fast_d_series[-1] if fast_d_series else None
                    })
            
            # Williams %R
            if len(closes) >= 14:
                indicators['williams_r'] = self._safe_call(lambda: calculate_williams_r(highs, lows, closes, 14))
            
            # CCI
            if len(closes) >= 20:
                indicators['cci_20'] = self._safe_call(lambda: calculate_cci(highs, lows, closes, 20))
            
            # Momentum et ROC
            if len(closes) >= 10:
                indicators['momentum_10'] = self._safe_call(lambda: calculate_momentum(closes, 10))
                indicators['roc_10'] = self._safe_call(lambda: calculate_roc(closes, 10))
            if len(closes) >= 20:
                indicators['roc_20'] = self._safe_call(lambda: calculate_roc(closes, 20))
            
            # Volume (OBV et VWAP)
            if len(volumes) >= 10:
                obv_series = self._safe_call(lambda: calculate_obv_series(closes, volumes))
                if obv_series and isinstance(obv_series, list) and len(obv_series) > 0:
                    indicators['obv'] = obv_series[-1]
                    
                    # OBV MA et oscillateur
                    if len(obv_series) >= 10:
                        indicators['obv_ma_10'] = self._safe_call(lambda: calculate_obv_ma(closes, volumes, 10))
                        indicators['obv_oscillator'] = self._safe_call(lambda: calculate_obv_oscillator(closes, volumes, 10))
                
                vwap_series = self._safe_call(lambda: calculate_vwap_series(highs, lows, closes, volumes))
                if vwap_series and isinstance(vwap_series, list) and len(vwap_series) > 0:
                    indicators['vwap_10'] = vwap_series[-1]
                
                # VWAP Quote (plus précis avec quote_asset_volume)
                quote_volumes = [d.get('quote_asset_volume', 0) for d in ohlcv_data]
                vwap_quote_series = self._safe_call(lambda: calculate_vwap_quote_series(highs, lows, closes, quote_volumes))
                if vwap_quote_series and isinstance(vwap_quote_series, list) and len(vwap_quote_series) > 0:
                    indicators['vwap_quote_10'] = vwap_quote_series[-1]
                
                # Volume context avec métriques avancées
                if len(volumes) >= 20:
                    avg_volume = sum(volumes[-20:]) / 20
                    
                    # Extraire les données pour les métriques avancées
                    quote_volumes = [d.get('quote_asset_volume', 0) for d in ohlcv_data]
                    trades_counts = [d.get('number_of_trades', 0) for d in ohlcv_data]
                    
                    indicators.update({
                        'avg_volume_20': avg_volume,
                        'volume_ratio': volumes[-1] / avg_volume if avg_volume > 0 else 1,
                        'quote_volume_ratio': self._safe_call(lambda: calculate_quote_volume_ratio(quote_volumes, 20)),
                        'avg_trade_size': self._safe_call(lambda: calculate_avg_trade_size(volumes[-1], trades_counts[-1])),
                        'trade_intensity': self._safe_call(lambda: calculate_trade_intensity(trades_counts, 20))
                    })
            
            # === APPEL DIRECT DE VOS MODULES DETECTORS ===
            if len(closes) >= 100:  # Assez de données pour les détecteurs
                
                # RegimeDetector
                try:
                    regime_result = self.regime_detector.detect_regime(
                        highs=highs,
                        lows=lows,
                        closes=closes,
                        volumes=volumes,
                        symbol=symbol,
                        include_analysis=True,
                        enable_cache=True
                    )
                    
                    if regime_result:
                        indicators.update({
                            'market_regime': str(regime_result.regime_type.value if hasattr(regime_result.regime_type, 'value') else regime_result.regime_type).upper(),
                            'regime_strength': str(regime_result.strength.value if hasattr(regime_result.strength, 'value') else regime_result.strength).upper(),
                            'regime_confidence': float(regime_result.confidence),
                            'regime_duration': int(regime_result.duration),
                            'trend_alignment': float(regime_result.trend_slope),  # Déjà en pourcentage
                            'momentum_score': float(regime_result.support_resistance_strength * 100)
                        })
                except Exception as e:
                    logger.warning(f"RegimeDetector error: {e}")
                
                # SupportResistanceDetector
                try:
                    sr_levels = self.sr_detector.detect_levels(
                        highs=highs,
                        lows=lows,
                        closes=closes,
                        volumes=volumes,
                        current_price=closes[-1],
                        timeframe=timeframe
                    )
                    
                    if sr_levels:
                        # Séparer supports et résistances
                        current_price = closes[-1]
                        supports = [level for level in sr_levels if level.price < current_price]
                        resistances = [level for level in sr_levels if level.price > current_price]
                        
                        # Extraire les prix pour JSONB
                        support_prices = [level.price for level in supports[:5]]  # Top 5
                        resistance_prices = [level.price for level in resistances[:5]]  # Top 5
                        
                        indicators.update({
                            'support_levels': support_prices,
                            'resistance_levels': resistance_prices,
                            'nearest_support': supports[0].price if supports else None,
                            'nearest_resistance': resistances[0].price if resistances else None,
                            'support_strength': str(supports[0].strength.value).upper() if supports else 'MODERATE',
                            'resistance_strength': str(resistances[0].strength.value).upper() if resistances else 'MODERATE',
                            'break_probability': float(supports[0].break_probability if supports else 50.0),
                            'pivot_count': len(sr_levels)
                        })
                except Exception as e:
                    logger.warning(f"SupportResistanceDetector error: {e}")
                
                # VolumeContextAnalyzer
                try:
                    volume_result = self.volume_analyzer.analyze_volume_context(
                        volumes=volumes,
                        closes=closes,
                        highs=highs,
                        lows=lows,
                        symbol=symbol
                    )
                    
                    if volume_result:
                        indicators.update({
                            'volume_context': str(volume_result.context.context_type.value).upper(),
                            'volume_pattern': str(volume_result.context.pattern_detected.value).upper(),
                            'volume_quality_score': float(volume_result.quality_score),
                            'relative_volume': float(volume_result.current_volume_ratio),
                            'volume_buildup_periods': int(self.volume_analyzer.buildup_lookback if volume_result.buildup_detected else 0),
                            'volume_spike_multiplier': float(volume_result.current_volume_ratio if volume_result.spike_detected else 1.0)
                        })
                except Exception as e:
                    logger.debug(f"VolumeContextAnalyzer error: {e}")
                
                # SpikeDetector avec vérification de fraîcheur
                try:
                    spike_events = self.spike_detector.detect_spikes(
                        highs=highs,
                        lows=lows,
                        closes=closes,
                        volumes=volumes,
                        timestamps=None  # Will be auto-generated
                    )
                    
                    if spike_events:
                        # Prendre le spike le plus récent
                        latest_spike = spike_events[0] if spike_events else None
                        
                        # Vérifier la fraîcheur du pattern pour éviter la persistance
                        if latest_spike and self._is_pattern_fresh(latest_spike, closes, volumes):
                            indicators.update({
                                'pattern_detected': str(latest_spike.spike_type.value).upper(),
                                'pattern_confidence': float(latest_spike.confidence)
                            })
                        else:
                            # Pattern trop ancien ou non significatif
                            indicators.update({
                                'pattern_detected': 'NORMAL',
                                'pattern_confidence': 0.0
                            })
                    else:
                        indicators.update({
                            'pattern_detected': 'NORMAL',
                            'pattern_confidence': 0.0
                        })
                except Exception as e:
                    logger.warning(f"SpikeDetector error: {e}")
                    indicators.update({
                        'pattern_detected': 'NORMAL',
                        'pattern_confidence': 0.0
                    })
            
            # Métadonnées
            calculation_time = int((time.time() - start_time) * 1000)
            indicators['calculation_time_ms'] = calculation_time
            indicators['data_quality'] = 'EXCELLENT' if len(ohlcv_data) >= 100 else 'GOOD'
            indicators['anomaly_detected'] = False
            
            logger.debug(f"🧮 {len([k for k, v in indicators.items() if v is not None])} indicateurs calculés en {calculation_time}ms")
            
        except Exception as e:
            import traceback
            logger.error(f"❌ Erreur lors des calculs: {e}")
            logger.error(f"❌ Type d'erreur: {type(e).__name__}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            indicators['data_quality'] = 'POOR'
            indicators['anomaly_detected'] = True
        
        return indicators

    def _safe_call(self, func):
        """Exécute un appel de fonction de manière sécurisée."""
        try:
            result = func()
            # Ne convertir en float que les types numériques, pas les strings
            if result is not None and not isinstance(result, (list, dict, str)):
                return float(result)
            return result
        except Exception as e:
            import traceback
            logger.debug(f"Appel échoué: {e}")
            logger.debug(f"Type d'erreur: {type(e).__name__}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _sanitize_numeric_value(self, value, max_abs_value=1e11):
        """Sanitise une valeur numérique pour éviter les débordements DB (precision 20, scale 8)."""
        if value is None:
            return None
        
        try:
            value = float(value)
            
            # Vérifier si la valeur est finie
            if not (isinstance(value, (int, float)) and abs(value) < float('inf')):
                return None
            
            # Limiter la valeur absolue pour éviter les débordements DB
            if abs(value) > max_abs_value:
                logger.warning(f"Valeur trop grande pour la DB: {value}, limitée à {max_abs_value}")
                return max_abs_value if value > 0 else -max_abs_value
            
            return value
            
        except (ValueError, TypeError, OverflowError):
            return None

    def _sanitize_percentage_value(self, value):
        """Sanitise une valeur pourcentage pour les colonnes precision(5,2) - max 999.99."""
        if value is None:
            return None
        
        try:
            value = float(value)
            
            # Vérifier si la valeur est finie
            if not (isinstance(value, (int, float)) and abs(value) < float('inf')):
                return None
            
            # Limiter à la précision (5,2) : max 999.99
            if abs(value) > 999.99:
                logger.warning(f"Valeur pourcentage trop grande pour la DB: {value}, limitée à 999.99")
                return 999.99 if value > 0 else -999.99
            
            return value
            
        except (ValueError, TypeError, OverflowError):
            return None

    async def _save_indicators_to_db(self, indicators: Dict):
        """Sauvegarde tous les indicateurs dans analyzer_data."""
        
        query = """
            INSERT INTO analyzer_data (
                time, symbol, timeframe, analysis_timestamp, analyzer_version,
                -- ORDRE EXACT DU SCHEMA.SQL
                -- Moyennes mobiles avancées
                wma_20, dema_12, tema_12, hull_20, kama_14,
                -- Indicateurs de base
                rsi_14, rsi_21, ema_7, ema_12, ema_26, ema_50, ema_99, sma_20, sma_50,
                -- MACD
                macd_line, macd_signal, macd_histogram, ppo,
                -- Bollinger Bands
                bb_upper, bb_middle, bb_lower, bb_position, bb_width, keltner_upper, keltner_lower,
                -- Stochastic
                stoch_k, stoch_d, stoch_rsi, stoch_fast_k, stoch_fast_d,
                -- ATR & Volatilité
                atr_14, natr, atr_stop_long, atr_stop_short, volatility_regime,
                -- ADX
                adx_14, plus_di, minus_di, dx, adxr,
                -- Oscillateurs
                williams_r, cci_20, momentum_10, roc_10, roc_20,
                -- Volume avancé
                vwap_10, vwap_quote_10, volume_ratio, avg_volume_20, quote_volume_ratio, avg_trade_size, trade_intensity,
                obv, obv_ma_10, obv_oscillator,
                -- Régime de marché
                market_regime, regime_strength, regime_confidence, regime_duration, trend_alignment, momentum_score,
                -- Support/Résistance
                support_levels, resistance_levels, nearest_support, nearest_resistance, 
                support_strength, resistance_strength, break_probability, pivot_count,
                -- Volume contexte
                volume_context, volume_pattern, volume_quality_score, relative_volume, 
                volume_buildup_periods, volume_spike_multiplier,
                -- Patterns
                pattern_detected, pattern_confidence,
                -- Métadonnées
                calculation_time_ms, data_quality, anomaly_detected
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19,
                $20, $21, $22, $23,
                $24, $25, $26, $27, $28, $29, $30,
                $31, $32, $33, $34, $35,
                $36, $37, $38, $39, $40,
                $41, $42, $43, $44, $45,
                $46, $47, $48, $49, $50,
                $51, $52, $53, $54, $55, $56, $57,
                $58, $59, $60,
                $61, $62, $63, $64, $65, $66,
                $67, $68, $69, $70, $71, $72, $73, $74,
                $75, $76, $77, $78, $79, $80,
                $81, $82,
                $83, $84, $85
            )
            ON CONFLICT (time, symbol, timeframe) DO UPDATE SET
                analysis_timestamp = EXCLUDED.analysis_timestamp,
                rsi_14 = EXCLUDED.rsi_14,
                rsi_21 = EXCLUDED.rsi_21,
                ema_7 = EXCLUDED.ema_7,
                macd_line = EXCLUDED.macd_line,
                bb_upper = EXCLUDED.bb_upper,
                calculation_time_ms = EXCLUDED.calculation_time_ms
        """
        
        try:
            # Sanitize all numeric values to prevent DB overflow - ORDRE EXACT DU SCHEMA
            sanitized_params = [
                indicators['time'], indicators['symbol'], indicators['timeframe'],
                indicators['analysis_timestamp'], indicators['analyzer_version'],
                # Moyennes mobiles avancées
                self._sanitize_numeric_value(indicators.get('wma_20')),
                self._sanitize_numeric_value(indicators.get('dema_12')), 
                self._sanitize_numeric_value(indicators.get('tema_12')), 
                self._sanitize_numeric_value(indicators.get('hull_20')), 
                self._sanitize_numeric_value(indicators.get('kama_14')),
                # Indicateurs de base
                self._sanitize_numeric_value(indicators.get('rsi_14')), 
                self._sanitize_numeric_value(indicators.get('rsi_21')), 
                self._sanitize_numeric_value(indicators.get('ema_7')), 
                self._sanitize_numeric_value(indicators.get('ema_12')), 
                self._sanitize_numeric_value(indicators.get('ema_26')), 
                self._sanitize_numeric_value(indicators.get('ema_50')), 
                self._sanitize_numeric_value(indicators.get('ema_99')),
                self._sanitize_numeric_value(indicators.get('sma_20')), 
                self._sanitize_numeric_value(indicators.get('sma_50')), 
                # MACD
                self._sanitize_numeric_value(indicators.get('macd_line')), 
                self._sanitize_numeric_value(indicators.get('macd_signal')), 
                self._sanitize_numeric_value(indicators.get('macd_histogram')), 
                self._sanitize_numeric_value(indicators.get('ppo')),
                # Bollinger Bands
                self._sanitize_numeric_value(indicators.get('bb_upper')), 
                self._sanitize_numeric_value(indicators.get('bb_middle')), 
                self._sanitize_numeric_value(indicators.get('bb_lower')),
                self._sanitize_numeric_value(indicators.get('bb_position')), 
                self._sanitize_numeric_value(indicators.get('bb_width')),
                self._sanitize_numeric_value(indicators.get('keltner_upper')), 
                self._sanitize_numeric_value(indicators.get('keltner_lower')),
                # Stochastic
                self._sanitize_numeric_value(indicators.get('stoch_k')), 
                self._sanitize_numeric_value(indicators.get('stoch_d')), 
                self._sanitize_numeric_value(indicators.get('stoch_rsi')),
                self._sanitize_numeric_value(indicators.get('stoch_fast_k')), 
                self._sanitize_numeric_value(indicators.get('stoch_fast_d')),
                # ATR & Volatilité
                self._sanitize_numeric_value(indicators.get('atr_14')), 
                self._sanitize_numeric_value(indicators.get('natr')), 
                self._sanitize_numeric_value(indicators.get('atr_stop_long')), 
                self._sanitize_numeric_value(indicators.get('atr_stop_short')),
                indicators.get('volatility_regime'),
                # ADX
                self._sanitize_numeric_value(indicators.get('adx_14')),
                self._sanitize_numeric_value(indicators.get('plus_di')),
                self._sanitize_numeric_value(indicators.get('minus_di')),
                self._sanitize_numeric_value(indicators.get('dx')),
                self._sanitize_numeric_value(indicators.get('adxr')),
                # Oscillateurs
                self._sanitize_numeric_value(indicators.get('williams_r')), 
                self._sanitize_numeric_value(indicators.get('cci_20')), 
                self._sanitize_numeric_value(indicators.get('momentum_10')),
                self._sanitize_numeric_value(indicators.get('roc_10')), 
                self._sanitize_numeric_value(indicators.get('roc_20')),
                # Volume avancé
                self._sanitize_numeric_value(indicators.get('vwap_10')), 
                self._sanitize_numeric_value(indicators.get('vwap_quote_10')),
                self._sanitize_numeric_value(indicators.get('volume_ratio')),
                self._sanitize_numeric_value(indicators.get('avg_volume_20')), 
                self._sanitize_numeric_value(indicators.get('quote_volume_ratio')), 
                self._sanitize_numeric_value(indicators.get('avg_trade_size')), 
                self._sanitize_numeric_value(indicators.get('trade_intensity')),
                self._sanitize_numeric_value(indicators.get('obv')), 
                self._sanitize_numeric_value(indicators.get('obv_ma_10')), 
                self._sanitize_numeric_value(indicators.get('obv_oscillator')),
                # Régime de marché
                indicators.get('market_regime'), indicators.get('regime_strength'), 
                self._sanitize_numeric_value(indicators.get('regime_confidence')),
                indicators.get('regime_duration'), 
                self._sanitize_numeric_value(indicators.get('trend_alignment')), 
                self._sanitize_numeric_value(indicators.get('momentum_score')),
                # Support/Résistance (JSONB)
                json.dumps(indicators.get('support_levels', [])) if indicators.get('support_levels') else None, 
                json.dumps(indicators.get('resistance_levels', [])) if indicators.get('resistance_levels') else None,
                self._sanitize_numeric_value(indicators.get('nearest_support')), 
                self._sanitize_numeric_value(indicators.get('nearest_resistance')),
                indicators.get('support_strength'), indicators.get('resistance_strength'),
                self._sanitize_numeric_value(indicators.get('break_probability')), 
                indicators.get('pivot_count'),
                # Volume contexte
                indicators.get('volume_context'), indicators.get('volume_pattern'),
                self._sanitize_numeric_value(indicators.get('volume_quality_score')), 
                self._sanitize_numeric_value(indicators.get('relative_volume')),
                indicators.get('volume_buildup_periods'), 
                self._sanitize_numeric_value(indicators.get('volume_spike_multiplier')),
                # Patterns
                indicators.get('pattern_detected'), 
                self._sanitize_numeric_value(indicators.get('pattern_confidence')),
                # Métadonnées
                indicators.get('calculation_time_ms'), indicators.get('data_quality'), indicators.get('anomaly_detected')
            ]
            
            async with self.db_pool.acquire() as conn:
                await conn.execute(query, *sanitized_params)
                
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde analyzer_data: {e}")
            raise

    def _is_pattern_fresh(self, latest_spike, closes, volumes):
        """Vérifie si le pattern détecté est récent et significatif."""
        if not latest_spike:
            return False
            
        try:
            # Vérifier la fraîcheur temporelle (pattern ne doit pas être trop ancien)
            # En l'absence de timestamps réels, on utilise la position dans la série
            max_age_periods = 3  # Maximum 3 périodes d'âge
            
            # Vérifier si les conditions de marché ont significativement changé
            # depuis la détection du pattern
            if len(closes) >= 5:
                recent_closes = closes[-5:]
                volatility_current = self._safe_call(lambda: float(np.std(recent_closes[-3:]))) if len(recent_closes) >= 3 else 0
                volatility_previous = self._safe_call(lambda: float(np.std(recent_closes[:3]))) if len(recent_closes) >= 3 else 0
                
                # Si la volatilité a significativement changé, le pattern n'est plus frais
                if volatility_previous and volatility_previous > 0 and abs(volatility_current - volatility_previous) / volatility_previous > 0.5:
                    return False
            
            # Vérifier la cohérence du volume avec le pattern détecté
            if len(volumes) >= 3:
                recent_volume_avg = self._safe_call(lambda: float(np.mean(volumes[-3:])))
                older_volume_avg = self._safe_call(lambda: float(np.mean(volumes[-6:-3]))) if len(volumes) >= 6 else recent_volume_avg
                
                # Si le volume a drastiquement diminué, le pattern perd sa validité
                if older_volume_avg and older_volume_avg > 0 and recent_volume_avg and recent_volume_avg / older_volume_avg < 0.5:
                    return False
            
            # Vérifier la cohérence du prix avec le type de pattern
            if len(closes) >= 2:
                price_change = (closes[-1] - closes[-2]) / closes[-2]
                
                # Pour un PRICE_SPIKE_UP, le prix ne devrait pas chuter drastiquement
                if (hasattr(latest_spike.spike_type, 'value') and latest_spike.spike_type.value == 'price_spike_up' and price_change < -0.02):
                    return False
                # Pour un PRICE_SPIKE_DOWN, le prix ne devrait pas monter drastiquement  
                elif (hasattr(latest_spike.spike_type, 'value') and latest_spike.spike_type.value == 'price_spike_down' and price_change > 0.02):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur vérification fraîcheur pattern: {e}")
            return False

    async def close(self):
        """Ferme les connexions."""
        self.running = False
        if self.db_pool:
            await self.db_pool.close()
            logger.info("🔌 IndicatorProcessor fermé")