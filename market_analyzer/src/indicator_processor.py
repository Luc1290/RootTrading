"""
Market Analyzer Indicator Processor
Appelle DIRECTEMENT tous les modules indicator/detector existants et sauvegarde en DB.
Architecture simple : récupère données → appelle modules → sauvegarde résultats.
"""

import logging
import asyncio
import time
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
    calculate_bollinger_bands_series, calculate_atr, calculate_adx,
    calculate_stochastic_series, calculate_williams_r, calculate_cci,
    calculate_obv_series, calculate_vwap_series
)

# Import des moyennes avancées
from market_analyzer.indicators.trend.moving_averages import (
    calculate_wma, calculate_dema, calculate_tema, calculate_hull_ma, calculate_kama
)

# Import des indicateurs de momentum
from market_analyzer.indicators.momentum.momentum import (
    calculate_momentum, calculate_roc, calculate_price_oscillator
)
from market_analyzer.indicators.momentum.rsi import calculate_stochastic_rsi

# Import des indicateurs de volatilité
from market_analyzer.indicators.volatility.atr import calculate_natr, calculate_atr_stop_loss
from market_analyzer.indicators.volatility.bollinger import calculate_keltner_channels

# Import des indicateurs de volume
from market_analyzer.indicators.volume.obv import calculate_obv_ma, calculate_obv_oscillator
from market_analyzer.indicators.volume.vwap import calculate_vwap_bands

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
            
            # RSI (plusieurs périodes)
            if len(closes) >= 14:
                indicators['rsi_14'] = self._safe_call(lambda: calculate_rsi(closes, 14, symbol))
            if len(closes) >= 21:
                indicators['rsi_21'] = self._safe_call(lambda: calculate_rsi(closes, 21, symbol))
            
            # Stochastic RSI
            if len(closes) >= 14:
                indicators['stoch_rsi'] = self._safe_call(lambda: calculate_stochastic_rsi(closes, 14, 14))
            
            # EMAs
            if len(closes) >= 7:
                indicators['ema_7'] = self._safe_call(lambda: calculate_ema(closes, 7, symbol))
            if len(closes) >= 12:
                indicators['ema_12'] = self._safe_call(lambda: calculate_ema(closes, 12, symbol))
            if len(closes) >= 26:
                indicators['ema_26'] = self._safe_call(lambda: calculate_ema(closes, 26, symbol))
            if len(closes) >= 50:
                indicators['ema_50'] = self._safe_call(lambda: calculate_ema(closes, 50, symbol))
            if len(closes) >= 99:
                indicators['ema_99'] = self._safe_call(lambda: calculate_ema(closes, 99, symbol))
            
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
                indicators['kama_14'] = self._safe_call(lambda: calculate_kama(closes, 14))
            
            # MACD
            if len(closes) >= 26:
                macd_series = self._safe_call(lambda: calculate_macd_series(closes))
                if macd_series and len(macd_series) > 0:
                    latest_macd = macd_series[-1]
                    indicators.update({
                        'macd_line': latest_macd.get('macd_line'),
                        'macd_signal': latest_macd.get('macd_signal'),
                        'macd_histogram': latest_macd.get('macd_histogram')
                    })
                
                # PPO
                indicators['ppo'] = self._safe_call(lambda: calculate_price_oscillator(closes, 12, 26))
            
            # ADX
            if len(closes) >= 14:
                indicators['adx_14'] = self._safe_call(lambda: calculate_adx(highs, lows, closes, 14, symbol))
            
            # Bollinger Bands
            if len(closes) >= 20:
                bb_series = self._safe_call(lambda: calculate_bollinger_bands_series(closes, 20, 2.0))
                if bb_series and len(bb_series) > 0:
                    latest_bb = bb_series[-1]
                    indicators.update({
                        'bb_upper': latest_bb.get('upper'),
                        'bb_middle': latest_bb.get('middle'),
                        'bb_lower': latest_bb.get('lower'),
                        'bb_position': latest_bb.get('position'),
                        'bb_width': latest_bb.get('width')
                    })
                
                # Keltner Channels
                keltner = self._safe_call(lambda: calculate_keltner_channels(closes, 20))
                if keltner:
                    indicators.update({
                        'keltner_upper': keltner.get('upper'),
                        'keltner_lower': keltner.get('lower')
                    })
            
            # ATR et volatilité
            if len(closes) >= 14:
                atr = self._safe_call(lambda: calculate_atr(highs, lows, closes, 14, symbol))
                indicators['atr_14'] = atr
                
                if atr and closes[-1] > 0:
                    indicators['natr'] = self._safe_call(lambda: calculate_natr(highs, lows, closes, 14))
                    
                    # ATR stop loss avec votre fonction
                    indicators.update({
                        'atr_stop_long': self._safe_call(lambda: calculate_atr_stop_loss(closes[-1], atr, 2.0, is_long=True)),
                        'atr_stop_short': self._safe_call(lambda: calculate_atr_stop_loss(closes[-1], atr, 2.0, is_long=False))
                    })
            
            # Stochastic
            if len(closes) >= 14:
                stoch_series = self._safe_call(lambda: calculate_stochastic_series(highs, lows, closes, 14, 3))
                if stoch_series and len(stoch_series) > 0:
                    latest_stoch = stoch_series[-1]
                    indicators.update({
                        'stoch_k': latest_stoch.get('k'),
                        'stoch_d': latest_stoch.get('d')
                    })
                
                # Fast Stochastic
                fast_stoch_series = self._safe_call(lambda: calculate_stochastic_series(highs, lows, closes, 14, 1))
                if fast_stoch_series and len(fast_stoch_series) > 0:
                    latest_fast = fast_stoch_series[-1]
                    indicators.update({
                        'stoch_fast_k': latest_fast.get('k'),
                        'stoch_fast_d': latest_fast.get('d')
                    })
            
            # Williams %R
            if len(closes) >= 14:
                indicators['williams_r'] = self._safe_call(lambda: calculate_williams_r(highs, lows, closes, 14, symbol))
            
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
                if obv_series and len(obv_series) > 0:
                    indicators['obv'] = obv_series[-1]
                    
                    # OBV MA et oscillateur
                    if len(obv_series) >= 10:
                        indicators['obv_ma_10'] = self._safe_call(lambda: calculate_obv_ma(closes, volumes, 10))
                        indicators['obv_oscillator'] = self._safe_call(lambda: calculate_obv_oscillator(closes, volumes, 10))
                
                vwap_series = self._safe_call(lambda: calculate_vwap_series(highs, lows, closes, volumes))
                if vwap_series and len(vwap_series) > 0:
                    indicators['vwap_10'] = vwap_series[-1]
                
                # Volume context
                if len(volumes) >= 20:
                    avg_volume = sum(volumes[-20:]) / 20
                    indicators.update({
                        'avg_volume_20': avg_volume,
                        'volume_ratio': volumes[-1] / avg_volume if avg_volume > 0 else 1
                    })
            
            # === APPEL DIRECT DE VOS MODULES DETECTORS ===
            if len(closes) >= 100:  # Assez de données pour les détecteurs
                
                # RegimeDetector
                try:
                    regime_result = self.regime_detector.analyze_regime({
                        'prices': closes,
                        'volumes': volumes,
                        'highs': highs,
                        'lows': lows
                    })
                    
                    if regime_result:
                        indicators.update({
                            'market_regime': str(regime_result.get('regime', 'UNKNOWN')),
                            'regime_strength': str(regime_result.get('strength', 'MODERATE')),
                            'regime_confidence': float(regime_result.get('confidence', 50.0)),
                            'regime_duration': int(regime_result.get('duration', 1)),
                            'trend_alignment': float(regime_result.get('trend_alignment', 50.0)),
                            'momentum_score': float(regime_result.get('momentum_score', 50.0))
                        })
                except Exception as e:
                    logger.debug(f"RegimeDetector error: {e}")
                
                # SupportResistanceDetector
                try:
                    sr_result = self.sr_detector.find_levels(highs, lows, closes)
                    
                    if sr_result:
                        indicators.update({
                            'support_levels': sr_result.get('support_levels', []),
                            'resistance_levels': sr_result.get('resistance_levels', []),
                            'nearest_support': sr_result.get('nearest_support'),
                            'nearest_resistance': sr_result.get('nearest_resistance'),
                            'support_strength': str(sr_result.get('support_strength', 'MODERATE')),
                            'resistance_strength': str(sr_result.get('resistance_strength', 'MODERATE')),
                            'break_probability': float(sr_result.get('break_probability', 50.0)),
                            'pivot_count': int(sr_result.get('pivot_count', 0))
                        })
                except Exception as e:
                    logger.debug(f"SupportResistanceDetector error: {e}")
                
                # VolumeContextAnalyzer
                try:
                    volume_result = self.volume_analyzer.analyze_context({
                        'volumes': volumes,
                        'prices': closes,
                        'highs': highs,
                        'lows': lows
                    })
                    
                    if volume_result:
                        indicators.update({
                            'volume_context': str(volume_result.get('context', 'NORMAL')),
                            'volume_pattern': str(volume_result.get('pattern', 'STEADY')),
                            'volume_quality_score': float(volume_result.get('quality_score', 50.0)),
                            'relative_volume': float(volume_result.get('relative_volume', 1.0)),
                            'volume_buildup_periods': int(volume_result.get('buildup_periods', 0)),
                            'volume_spike_multiplier': float(volume_result.get('spike_multiplier', 1.0))
                        })
                except Exception as e:
                    logger.debug(f"VolumeContextAnalyzer error: {e}")
                
                # SpikeDetector
                try:
                    spike_result = self.spike_detector.detect_spikes(volumes, closes)
                    if spike_result:
                        indicators.update({
                            'pattern_detected': spike_result.get('pattern'),
                            'pattern_confidence': float(spike_result.get('confidence', 0.0))
                        })
                except Exception as e:
                    logger.debug(f"SpikeDetector error: {e}")
            
            # Métadonnées
            calculation_time = int((time.time() - start_time) * 1000)
            indicators['calculation_time_ms'] = calculation_time
            indicators['data_quality'] = 'EXCELLENT' if len(ohlcv_data) >= 100 else 'GOOD'
            indicators['anomaly_detected'] = False
            
            logger.debug(f"🧮 {len([k for k, v in indicators.items() if v is not None])} indicateurs calculés en {calculation_time}ms")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors des calculs: {e}")
            indicators['data_quality'] = 'POOR'
            indicators['anomaly_detected'] = True
        
        return indicators

    def _safe_call(self, func):
        """Exécute un appel de fonction de manière sécurisée."""
        try:
            result = func()
            return float(result) if result is not None and not isinstance(result, (list, dict)) else result
        except Exception as e:
            logger.debug(f"Appel échoué: {e}")
            return None

    async def _save_indicators_to_db(self, indicators: Dict):
        """Sauvegarde tous les indicateurs dans analyzer_data."""
        
        query = """
            INSERT INTO analyzer_data (
                time, symbol, timeframe, analysis_timestamp, analyzer_version,
                -- RSI et momentum
                rsi_14, rsi_21, stoch_rsi,
                -- EMAs
                ema_7, ema_12, ema_26, ema_50, ema_99,
                -- SMAs et moyennes avancées
                sma_20, sma_50, wma_20, dema_12, tema_12, hull_20, kama_14,
                -- MACD
                macd_line, macd_signal, macd_histogram, ppo,
                -- ADX
                adx_14,
                -- Bollinger Bands
                bb_upper, bb_middle, bb_lower, bb_position, bb_width,
                keltner_upper, keltner_lower,
                -- ATR
                atr_14, natr, atr_stop_long, atr_stop_short,
                -- Stochastic
                stoch_k, stoch_d, stoch_fast_k, stoch_fast_d,
                -- Oscillateurs
                williams_r, cci_20, momentum_10, roc_10, roc_20,
                -- Volume
                obv, obv_ma_10, obv_oscillator, vwap_10, avg_volume_20, volume_ratio,
                -- Régime
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
                $6, $7, $8,
                $9, $10, $11, $12, $13,
                $14, $15, $16, $17, $18, $19, $20,
                $21, $22, $23, $24,
                $25,
                $26, $27, $28, $29, $30, $31, $32,
                $33, $34, $35, $36,
                $37, $38, $39, $40, $41,
                $42, $43, $44, $45, $46, $47, $48,
                $49, $50, $51, $52, $53, $54, $55,
                $56, $57, $58, $59, $60, $61, $62, $63,
                $64, $65, $66, $67, $68, $69,
                $70, $71,
                $72, $73, $74
            )
            ON CONFLICT (time, symbol, timeframe) DO UPDATE SET
                analysis_timestamp = EXCLUDED.analysis_timestamp,
                rsi_14 = EXCLUDED.rsi_14,
                ema_7 = EXCLUDED.ema_7,
                macd_line = EXCLUDED.macd_line,
                bb_upper = EXCLUDED.bb_upper,
                calculation_time_ms = EXCLUDED.calculation_time_ms
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    query,
                    indicators['time'], indicators['symbol'], indicators['timeframe'],
                    indicators['analysis_timestamp'], indicators['analyzer_version'],
                    # RSI et momentum
                    indicators.get('rsi_14'), indicators.get('rsi_21'), indicators.get('stoch_rsi'),
                    # EMAs
                    indicators.get('ema_7'), indicators.get('ema_12'), indicators.get('ema_26'), 
                    indicators.get('ema_50'), indicators.get('ema_99'),
                    # SMAs et moyennes avancées
                    indicators.get('sma_20'), indicators.get('sma_50'), indicators.get('wma_20'),
                    indicators.get('dema_12'), indicators.get('tema_12'), indicators.get('hull_20'), indicators.get('kama_14'),
                    # MACD
                    indicators.get('macd_line'), indicators.get('macd_signal'), indicators.get('macd_histogram'), indicators.get('ppo'),
                    # ADX
                    indicators.get('adx_14'),
                    # Bollinger Bands
                    indicators.get('bb_upper'), indicators.get('bb_middle'), indicators.get('bb_lower'),
                    indicators.get('bb_position'), indicators.get('bb_width'),
                    indicators.get('keltner_upper'), indicators.get('keltner_lower'),
                    # ATR
                    indicators.get('atr_14'), indicators.get('natr'), 
                    indicators.get('atr_stop_long'), indicators.get('atr_stop_short'),
                    # Stochastic
                    indicators.get('stoch_k'), indicators.get('stoch_d'), 
                    indicators.get('stoch_fast_k'), indicators.get('stoch_fast_d'),
                    # Oscillateurs
                    indicators.get('williams_r'), indicators.get('cci_20'), indicators.get('momentum_10'),
                    indicators.get('roc_10'), indicators.get('roc_20'),
                    # Volume
                    indicators.get('obv'), indicators.get('obv_ma_10'), indicators.get('obv_oscillator'),
                    indicators.get('vwap_10'), indicators.get('avg_volume_20'), indicators.get('volume_ratio'),
                    # Régime
                    indicators.get('market_regime'), indicators.get('regime_strength'), indicators.get('regime_confidence'),
                    indicators.get('regime_duration'), indicators.get('trend_alignment'), indicators.get('momentum_score'),
                    # Support/Résistance (JSONB)
                    indicators.get('support_levels'), indicators.get('resistance_levels'),
                    indicators.get('nearest_support'), indicators.get('nearest_resistance'),
                    indicators.get('support_strength'), indicators.get('resistance_strength'),
                    indicators.get('break_probability'), indicators.get('pivot_count'),
                    # Volume contexte
                    indicators.get('volume_context'), indicators.get('volume_pattern'),
                    indicators.get('volume_quality_score'), indicators.get('relative_volume'),
                    indicators.get('volume_buildup_periods'), indicators.get('volume_spike_multiplier'),
                    # Patterns
                    indicators.get('pattern_detected'), indicators.get('pattern_confidence'),
                    # Métadonnées
                    indicators.get('calculation_time_ms'), indicators.get('data_quality'), indicators.get('anomaly_detected')
                )
                
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde analyzer_data: {e}")
            raise

    async def close(self):
        """Ferme les connexions."""
        self.running = False
        if self.db_pool:
            await self.db_pool.close()
            logger.info("🔌 IndicatorProcessor fermé")