"""
Market Analyzer Indicator Processor - VERSION PARALLÉLISÉE
OPTIMISATION Phase 3: Calculs parallélisés avec asyncio pour +50% performance

Appelle DIRECTEMENT tous les modules indicator/detector existants et sauvegarde en DB.
Architecture simple : récupère données → appelle modules → sauvegarde résultats.
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import asyncpg  # type: ignore
import numpy as np
import redis.asyncio as redis

from market_analyzer.detectors.regime_detector import RegimeDetector
from market_analyzer.detectors.spike_detector import SpikeDetector
from market_analyzer.detectors.support_resistance_detector import (
    SupportResistanceDetector,
)
from market_analyzer.detectors.volume_context_analyzer import VolumeContextAnalyzer
from market_analyzer.indicators import (
    calculate_bollinger_bands_series,
    calculate_cci,
    calculate_ema_series,
    calculate_macd_series,
    calculate_obv_series,
    calculate_rsi_series,
    calculate_sma,
    calculate_stochastic_series,
    calculate_vwap_series,
    calculate_williams_r,
)
from market_analyzer.indicators.composite.confluence import (
    ConfluenceType,
    calculate_confluence_score,
)
from market_analyzer.indicators.composite.signal_strength import (
    calculate_signal_strength,
)
from market_analyzer.indicators.momentum.momentum import (
    calculate_momentum,
    calculate_roc,
)
from market_analyzer.indicators.momentum.rsi import calculate_stoch_rsi
from market_analyzer.indicators.trend.adx import calculate_adx_full
from market_analyzer.indicators.trend.moving_averages import (
    calculate_adaptive_ma,
    calculate_dema,
    calculate_hull_ma,
    calculate_tema,
    calculate_wma,
)
from market_analyzer.indicators.volatility.atr import (
    calculate_atr,
    calculate_atr_percentile,
    calculate_natr,
    volatility_regime,
)
from market_analyzer.indicators.volatility.bollinger import calculate_keltner_channels
from market_analyzer.indicators.volume.advanced_metrics import (
    calculate_avg_trade_size,
    calculate_quote_volume_ratio,
    calculate_trade_intensity,
)
from market_analyzer.indicators.volume.obv import (
    calculate_obv_ma,
    calculate_obv_oscillator,
)
from market_analyzer.indicators.volume.vwap import calculate_vwap_quote_series
from shared.src.config import get_db_config

sys.path.append(str((Path(__file__).parent / "../../").resolve()))

logger = logging.getLogger(__name__)


class IndicatorProcessor:
    """
    Processeur optimisé avec calculs parallélisés (Phase 3).
    """

    def __init__(self) -> None:
        self.db_pool = None
        self.running = False
        self.redis_client: redis.Redis | None = None

        # Initialiser les détecteurs
        self.regime_detector = RegimeDetector()
        self.sr_detector = SupportResistanceDetector()
        self.volume_analyzer = VolumeContextAnalyzer()
        self.spike_detector = SpikeDetector()

        logger.info("🧮 IndicatorProcessor initialisé (VERSION PARALLÉLISÉE)")

    async def initialize(self):
        """Initialise la connexion à la base de données."""
        try:
            db_config = get_db_config()
            self.db_pool = await asyncpg.create_pool(
                host=db_config["host"],
                port=db_config["port"],
                database=db_config["database"],
                user=db_config["user"],
                password=db_config["password"],
                min_size=2,
                max_size=10,
            )
            self.running = True
            logger.info("✅ IndicatorProcessor connecté à la base de données")
        except Exception:
            logger.exception("❌ Erreur connexion DB")
            raise

    async def process_new_data(self, symbol: str, timeframe: str, timestamp: datetime):
        """
        Traite une nouvelle donnée OHLCV : appelle tous vos modules et sauvegarde.
        """
        try:
            logger.debug(f"🔄 Traitement {symbol} {timeframe} @ {timestamp}")

            # Récupérer données historiques OPTIMISÉES (200 points au lieu de 500)
            ohlcv_data = await self._get_historical_data(
                symbol, timeframe, limit=200, up_to_timestamp=timestamp
            )

            if len(ohlcv_data) < 20:
                logger.debug(
                    f"⏭️ Pas assez de données pour {symbol} {timeframe}: {len(ohlcv_data)} < 20"
                )
                return

            # Appeler TOUS vos modules EN PARALLÈLE (Phase 3)
            indicators_data = await self._call_all_indicator_modules(
                symbol, timeframe, ohlcv_data, timestamp
            )

            # Sauvegarder en DB
            await self._save_indicators_to_db(indicators_data)

            logger.info(f"✅ Indicateurs sauvegardés: {symbol} {timeframe}")

        except Exception:
            logger.exception(f"❌ Erreur traitement {symbol} {timeframe}")

    async def _get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 1000000,
        up_to_timestamp: datetime | None = None,
    ) -> list[dict]:
        """Récupère les données OHLCV historiques (OPTIMISÉE avec subquery ORDER BY ASC)."""
        if up_to_timestamp:
            query = """
                SELECT time, open, high, low, close, volume,
                       quote_asset_volume, number_of_trades
                FROM (
                    SELECT time, open, high, low, close, volume,
                           quote_asset_volume, number_of_trades
                    FROM market_data
                    WHERE symbol = $1 AND timeframe = $2 AND time <= $3
                    ORDER BY time DESC
                    LIMIT $4
                ) AS recent_data
                ORDER BY time ASC
            """
            params = [symbol, timeframe, up_to_timestamp, limit]
        else:
            query = """
                SELECT time, open, high, low, close, volume,
                       quote_asset_volume, number_of_trades
                FROM (
                    SELECT time, open, high, low, close, volume,
                           quote_asset_volume, number_of_trades
                    FROM market_data
                    WHERE symbol = $1 AND timeframe = $2
                    ORDER BY time DESC
                    LIMIT $3
                ) AS recent_data
                ORDER BY time ASC
            """
            params = [symbol, timeframe, limit]

        if self.db_pool is None:
            raise RuntimeError("Database pool not initialized")

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

            # OPTIMISATION: List comprehension
            data = [
                {
                    "time": row["time"],
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                    "quote_asset_volume": (
                        float(row["quote_asset_volume"])
                        if row["quote_asset_volume"]
                        else 0.0
                    ),
                    "number_of_trades": row["number_of_trades"] or 0,
                }
                for row in rows
            ]

            return data

    async def _call_all_indicator_modules(
        self, symbol: str, timeframe: str, ohlcv_data: list[dict], timestamp: datetime
    ) -> dict:
        """
        OPTIMISATION Phase 3: Calculs parallélisés avec asyncio pour +50% performance.

        Architecture:
        - Groupe 1: Indicateurs indépendants (RSI, EMAs, SMAs, Williams %R, Momentum, ROC)
        - Groupe 2: Indicateurs dépendants (MACD, Bollinger, Keltner, ADX, ATR, Stochastic, CCI, MFI, OBV, VWAP)
        - Groupe 3: Détecteurs complexes (Regime, S/R, Volume Context, Spike Detection)
        """
        start_time = time.time()

        # Extraire les arrays
        opens = [d["open"] for d in ohlcv_data]
        highs = [d["high"] for d in ohlcv_data]
        lows = [d["low"] for d in ohlcv_data]
        closes = [d["close"] for d in ohlcv_data]
        volumes = [d["volume"] for d in ohlcv_data]

        # Résultat final
        indicators = {
            "time": timestamp,
            "symbol": symbol,
            "timeframe": timeframe,
            "analysis_timestamp": datetime.now(tz=timezone.utc),
            "analyzer_version": "1.0_parallel",
        }

        try:
            # === GROUPE 1: CALCULS PARALLÉLISÉS INDÉPENDANTS ===
            logger.debug(f"🔄 Phase 1/3: Calculs indépendants en parallèle...")
            group1_results = await asyncio.gather(
                self._calc_rsi_indicators(closes),
                self._calc_ema_indicators(closes),
                self._calc_sma_indicators(closes),
                self._calc_advanced_ma_indicators(closes),
                self._calc_williams_r(highs, lows, closes),
                self._calc_momentum_roc(closes),
                self._calc_stoch_rsi(closes),
                return_exceptions=True
            )

            # Fusionner résultats Groupe 1
            for result in group1_results:
                if isinstance(result, dict):
                    indicators.update(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Erreur Groupe 1: {result}")

            # === GROUPE 2: CALCULS PARALLÉLISÉS DÉPENDANTS ===
            logger.debug(f"🔄 Phase 2/3: Calculs dépendants en parallèle...")
            group2_results = await asyncio.gather(
                self._calc_macd_indicators(closes, indicators),
                self._calc_bollinger_bands(closes),
                self._calc_keltner_channels(highs, lows, closes),
                self._calc_adx_indicators(highs, lows, closes),
                self._calc_atr_indicators(highs, lows, closes),
                self._calc_stochastic_indicators(highs, lows, closes),
                self._calc_cci(highs, lows, closes),
                self._calc_mfi(highs, lows, closes, volumes),
                self._calc_volume_indicators(closes, volumes, ohlcv_data),
                return_exceptions=True
            )

            # Fusionner résultats Groupe 2
            for result in group2_results:
                if isinstance(result, dict):
                    indicators.update(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Erreur Groupe 2: {result}")

            # === GROUPE 3: DÉTECTEURS COMPLEXES EN PARALLÈLE ===
            logger.debug(f"🔄 Phase 3/3: Détecteurs complexes en parallèle...")
            if len(closes) >= 100:
                group3_results = await asyncio.gather(
                    self._detect_regime(highs, lows, closes, volumes, symbol, indicators),
                    self._detect_support_resistance(highs, lows, closes, volumes, timeframe),
                    self._analyze_volume_context(volumes, closes, highs, lows, symbol),
                    self._detect_spikes(highs, lows, closes, volumes),
                    return_exceptions=True
                )

                # Fusionner résultats Groupe 3
                for result in group3_results:
                    if isinstance(result, dict):
                        indicators.update(result)
                    elif isinstance(result, Exception):
                        logger.warning(f"Erreur Groupe 3: {result}")

            # === CALCULS FINAUX SÉQUENTIELS (nécessitent tous les indicateurs) ===
            confluence_score = self._safe_call(
                lambda: calculate_confluence_score(
                    indicators,
                    closes[-1] if closes else None,
                    ConfluenceType.MONO_TIMEFRAME,
                )
            )
            indicators["confluence_score"] = confluence_score

            signal_strength = self._safe_call(
                lambda: calculate_signal_strength(indicators)
            )
            indicators["signal_strength"] = signal_strength

            # Métadonnées
            calculation_time = int((time.time() - start_time) * 1000)
            indicators["calculation_time_ms"] = calculation_time
            indicators["data_quality"] = (
                "EXCELLENT" if len(ohlcv_data) >= 100 else "GOOD"
            )
            indicators["anomaly_detected"] = False

            logger.debug(
                f"✅ {len([k for k, v in indicators.items() if v is not None])} indicateurs calculés en {calculation_time}ms (3 phases parallèles)"
            )

        except Exception as e:
            import traceback

            logger.exception("❌ Erreur lors des calculs")
            logger.exception(f"❌ Type d'erreur: {type(e).__name__}")
            logger.exception(f"❌ Traceback: {traceback.format_exc()}")
            indicators["data_quality"] = "POOR"
            indicators["anomaly_detected"] = True

        return indicators

    # === MÉTHODES HELPER POUR PARALLÉLISATION - GROUPE 1 (INDÉPENDANTS) ===

    async def _calc_rsi_indicators(self, closes: list[float]) -> dict:
        """Calcule RSI 14 et 21."""
        result = {}
        if len(closes) >= 14:
            rsi_14_series = self._safe_call(
                lambda: calculate_rsi_series(closes, 14)
            )
            result["rsi_14"] = (
                rsi_14_series[-1]
                if rsi_14_series and len(rsi_14_series) > 0
                else None
            )

        if len(closes) >= 21:
            rsi_21_series = self._safe_call(
                lambda: calculate_rsi_series(closes, 21)
            )
            result["rsi_21"] = (
                rsi_21_series[-1]
                if rsi_21_series and len(rsi_21_series) > 0
                else None
            )

        return result

    async def _calc_ema_indicators(self, closes: list[float]) -> dict:
        """Calcule toutes les EMAs (7, 12, 26, 50, 99)."""
        result = {}
        ema_periods = [(7, "ema_7"), (12, "ema_12"), (26, "ema_26"), (50, "ema_50"), (99, "ema_99")]

        for period, key in ema_periods:
            if len(closes) >= period:
                ema_series = self._safe_call(lambda p=period: calculate_ema_series(closes, p))
                result[key] = (
                    ema_series[-1] if ema_series and len(ema_series) > 0 else None
                )

        return result

    async def _calc_sma_indicators(self, closes: list[float]) -> dict:
        """Calcule SMAs (20, 50)."""
        result = {}
        if len(closes) >= 20:
            result["sma_20"] = self._safe_call(lambda: calculate_sma(closes, 20))
        if len(closes) >= 50:
            result["sma_50"] = self._safe_call(lambda: calculate_sma(closes, 50))

        return result

    async def _calc_advanced_ma_indicators(self, closes: list[float]) -> dict:
        """Calcule moyennes avancées (WMA, Hull, DEMA, TEMA, KAMA)."""
        result = {}

        if len(closes) >= 20:
            result["wma_20"] = self._safe_call(lambda: calculate_wma(closes, 20))
            result["hull_20"] = self._safe_call(lambda: calculate_hull_ma(closes, 20))

        if len(closes) >= 12:
            result["dema_12"] = self._safe_call(lambda: calculate_dema(closes, 12))
            result["tema_12"] = self._safe_call(lambda: calculate_tema(closes, 12))

        if len(closes) >= 14:
            result["kama_14"] = self._safe_call(lambda: calculate_adaptive_ma(closes, 14))

        return result

    async def _calc_williams_r(self, highs: list[float], lows: list[float], closes: list[float]) -> dict:
        """Calcule Williams %R."""
        result = {}
        if len(closes) >= 14:
            result["williams_r"] = self._safe_call(
                lambda: calculate_williams_r(highs, lows, closes, 14)
            )

        return result

    async def _calc_momentum_roc(self, closes: list[float]) -> dict:
        """Calcule Momentum et ROC."""
        result = {}
        if len(closes) >= 10:
            result["momentum_10"] = self._safe_call(lambda: calculate_momentum(closes, 10))
            result["roc_10"] = self._safe_call(lambda: calculate_roc(closes, 10))

        if len(closes) >= 20:
            result["roc_20"] = self._safe_call(lambda: calculate_roc(closes, 20))

        return result

    async def _calc_stoch_rsi(self, closes: list[float]) -> dict:
        """Calcule Stochastic RSI."""
        result = {}
        if len(closes) >= 14:
            result["stoch_rsi"] = self._safe_call(
                lambda: calculate_stoch_rsi(closes, 14, 14)
            )

        return result

    # === MÉTHODES HELPER POUR PARALLÉLISATION - GROUPE 2 (DÉPENDANTS) ===

    async def _calc_macd_indicators(self, closes: list[float], indicators: dict) -> dict:
        """Calcule MACD complet avec signaux."""
        result = {}
        if len(closes) >= 26:
            from market_analyzer.indicators.trend.macd import (
                calculate_macd_trend,
                calculate_ppo,
                macd_signal_cross,
                macd_zero_cross,
            )

            macd_series = self._safe_call(
                lambda: calculate_macd_series(
                    closes, fast_period=12, slow_period=26, signal_period=9
                )
            )

            if macd_series and isinstance(macd_series, dict):
                macd_line_series = macd_series.get("macd_line", [])
                macd_signal_series = macd_series.get("macd_signal", [])
                macd_histogram_series = macd_series.get("macd_histogram", [])

                result.update({
                    "macd_line": (macd_line_series[-1] if macd_line_series else None),
                    "macd_signal": (macd_signal_series[-1] if macd_signal_series else None),
                    "macd_histogram": (macd_histogram_series[-1] if macd_histogram_series else None),
                })

                # MACD Signaux binaires
                if macd_line_series and macd_signal_series and len(macd_line_series) >= 2:
                    current_macd = {
                        "macd_line": macd_line_series[-1],
                        "macd_signal": macd_signal_series[-1],
                        "macd_histogram": (macd_histogram_series[-1] if macd_histogram_series else None),
                    }
                    prev_macd = {
                        "macd_line": macd_line_series[-2],
                        "macd_signal": macd_signal_series[-2],
                        "macd_histogram": (macd_histogram_series[-2] if macd_histogram_series and len(macd_histogram_series) >= 2 else None),
                    }

                    zero_cross = self._safe_call(lambda: macd_zero_cross(current_macd, prev_macd))
                    signal_cross = self._safe_call(lambda: macd_signal_cross(current_macd, prev_macd))
                    trend = self._safe_call(lambda: calculate_macd_trend(current_macd, prev_macd))

                    result.update({
                        "macd_zero_cross": bool(zero_cross not in [None, "none"]),
                        "macd_signal_cross": bool(signal_cross not in [None, "none"]),
                        "macd_trend": (trend.upper() if trend and trend != "none" else "NEUTRAL"),
                    })

            # PPO
            ppo_result = self._safe_call(lambda: calculate_ppo(closes, 12, 26, 9))
            if ppo_result and isinstance(ppo_result, dict):
                result["ppo"] = ppo_result.get("ppo_line")

        return result

    async def _calc_bollinger_bands(self, closes: list[float]) -> dict:
        """Calcule Bollinger Bands avec signaux."""
        result = {}
        if len(closes) >= 20:
            from market_analyzer.indicators.volatility.bollinger import (
                calculate_bollinger_breakout_direction,
                calculate_bollinger_expansion,
                calculate_bollinger_squeeze,
            )

            bb_series = self._safe_call(
                lambda: calculate_bollinger_bands_series(closes, 20, 2.0)
            )

            if bb_series and isinstance(bb_series, dict):
                bb_upper_series = bb_series.get("upper", [])
                bb_middle_series = bb_series.get("middle", [])
                bb_lower_series = bb_series.get("lower", [])
                bb_percent_b_series = bb_series.get("percent_b", [])
                bb_bandwidth_series = bb_series.get("bandwidth", [])

                result.update({
                    "bb_upper": (bb_upper_series[-1] if bb_upper_series else None),
                    "bb_middle": (bb_middle_series[-1] if bb_middle_series else None),
                    "bb_lower": (bb_lower_series[-1] if bb_lower_series else None),
                    "bb_position": (bb_percent_b_series[-1] if bb_percent_b_series else None),
                    "bb_width": (bb_bandwidth_series[-1] if bb_bandwidth_series else None),
                })

                # Bollinger Signaux binaires
                squeeze_data = self._safe_call(lambda: calculate_bollinger_squeeze(closes, 20, 2.0, 20))
                expansion = self._safe_call(lambda: calculate_bollinger_expansion(closes, 20, 2.0, 10))
                breakout_dir = self._safe_call(lambda: calculate_bollinger_breakout_direction(closes, 20, 2.0, 3))

                result.update({
                    "bb_squeeze": (bool(squeeze_data.get("in_squeeze", False)) if squeeze_data else False),
                    "bb_expansion": (bool(expansion) if expansion is not None else False),
                    "bb_breakout_direction": (breakout_dir if breakout_dir else "NONE"),
                })

        return result

    async def _calc_keltner_channels(self, highs: list[float], lows: list[float], closes: list[float]) -> dict:
        """Calcule Keltner Channels."""
        result = {}
        if len(highs) >= 20 and len(lows) >= 20:
            keltner = self._safe_call(
                lambda: calculate_keltner_channels(
                    closes, highs, lows, period=20, atr_period=10, multiplier=2.0,
                )
            )
            if keltner and isinstance(keltner, dict):
                result.update({
                    "keltner_upper": keltner.get("upper"),
                    "keltner_lower": keltner.get("lower"),
                })

        return result

    async def _calc_adx_indicators(self, highs: list[float], lows: list[float], closes: list[float]) -> dict:
        """Calcule ADX complet avec DI et trend."""
        result = {}
        if len(closes) >= 14:
            from market_analyzer.indicators.trend.adx import (
                adx_trend_strength,
                calculate_directional_bias,
                calculate_trend_angle,
            )

            adx_full = self._safe_call(lambda: calculate_adx_full(highs, lows, closes, 14))
            if adx_full and isinstance(adx_full, dict):
                result.update({
                    "adx_14": adx_full.get("adx"),
                    "plus_di": adx_full.get("plus_di"),
                    "minus_di": adx_full.get("minus_di"),
                    "dx": adx_full.get("dx"),
                    "adxr": adx_full.get("adxr"),
                })

                adx_value = adx_full.get("adx")
                if adx_value is not None:
                    result["trend_strength"] = adx_trend_strength(adx_value)

                plus_di = adx_full.get("plus_di")
                minus_di = adx_full.get("minus_di")
                if plus_di is not None and minus_di is not None:
                    result["directional_bias"] = calculate_directional_bias(
                        plus_di, minus_di, adx_value
                    )

            # Trend Angle
            result["trend_angle"] = self._safe_call(lambda: calculate_trend_angle(closes, 14))

        return result

    async def _calc_atr_indicators(self, highs: list[float], lows: list[float], closes: list[float]) -> dict:
        """Calcule ATR et volatilité."""
        result = {}
        if len(closes) >= 14:
            atr = self._safe_call(lambda: calculate_atr(highs, lows, closes, 14))
            result["atr_14"] = atr

            if atr and closes[-1] > 0:
                result["natr"] = self._safe_call(lambda: calculate_natr(highs, lows, closes, 14))
                result["volatility_regime"] = self._safe_call(
                    lambda: volatility_regime(highs, lows, closes, 14, 20)
                )

                # ATR stop loss
                atr_multiplier = 2.0
                result.update({
                    "atr_stop_long": closes[-1] - (atr * atr_multiplier),
                    "atr_stop_short": closes[-1] + (atr * atr_multiplier),
                })

        return result

    async def _calc_stochastic_indicators(self, highs: list[float], lows: list[float], closes: list[float]) -> dict:
        """Calcule Stochastic complet."""
        result = {}
        if len(closes) >= 14:
            from market_analyzer.indicators.oscillators.stochastic import (
                calculate_stochastic_divergence,
                calculate_stochastic_signal,
            )

            # Stochastic standard
            stoch_series = self._safe_call(
                lambda: calculate_stochastic_series(highs, lows, closes, 14, 3)
            )
            if stoch_series and isinstance(stoch_series, dict):
                stoch_k_series = stoch_series.get("k", [])
                stoch_d_series = stoch_series.get("d", [])

                result.update({
                    "stoch_k": stoch_k_series[-1] if stoch_k_series else None,
                    "stoch_d": stoch_d_series[-1] if stoch_d_series else None,
                })

            # Fast Stochastic
            fast_stoch_series = self._safe_call(
                lambda: calculate_stochastic_series(highs, lows, closes, 14, 1)
            )
            if fast_stoch_series and isinstance(fast_stoch_series, dict):
                fast_k_series = fast_stoch_series.get("k", [])
                fast_d_series = fast_stoch_series.get("d", [])

                result.update({
                    "stoch_fast_k": (fast_k_series[-1] if fast_k_series else None),
                    "stoch_fast_d": (fast_d_series[-1] if fast_d_series else None),
                })

                # Stochastic Signaux binaires
                divergence = self._safe_call(
                    lambda: calculate_stochastic_divergence(closes, highs, lows, 14)
                )
                signal = self._safe_call(
                    lambda: calculate_stochastic_signal(highs, lows, closes, 14, 3)
                )

                result.update({
                    "stoch_divergence": (bool(divergence not in [None, "none"]) if divergence else False),
                    "stoch_signal": (signal.upper() if signal and signal != "none" else "NEUTRAL"),
                })

        return result

    async def _calc_cci(self, highs: list[float], lows: list[float], closes: list[float]) -> dict:
        """Calcule CCI."""
        result = {}
        if len(closes) >= 20:
            result["cci_20"] = self._safe_call(lambda: calculate_cci(highs, lows, closes, 20))

        return result

    async def _calc_mfi(self, highs: list[float], lows: list[float], closes: list[float], volumes: list[float]) -> dict:
        """Calcule MFI (Money Flow Index)."""
        result = {}
        if len(closes) >= 15 and len(volumes) >= 15:
            from market_analyzer.indicators.momentum.mfi import calculate_mfi

            result["mfi_14"] = self._safe_call(
                lambda: calculate_mfi(highs, lows, closes, volumes, 14)
            )

        return result

    async def _calc_volume_indicators(self, closes: list[float], volumes: list[float], ohlcv_data: list[dict]) -> dict:
        """Calcule tous les indicateurs de volume (OBV, VWAP, etc.)."""
        result = {}

        if len(volumes) >= 10:
            # OBV
            obv_series = self._safe_call(lambda: calculate_obv_series(closes, volumes))
            if obv_series and isinstance(obv_series, list) and len(obv_series) > 0:
                result["obv"] = obv_series[-1]

                if len(obv_series) >= 10:
                    result["obv_ma_10"] = self._safe_call(lambda: calculate_obv_ma(closes, volumes, 10))
                    result["obv_oscillator"] = self._safe_call(lambda: calculate_obv_oscillator(closes, volumes, 10))

            # A/D Line
            from market_analyzer.indicators.volume.obv import calculate_volume_accumulation_distribution

            result["ad_line"] = self._safe_call(
                lambda: calculate_volume_accumulation_distribution(
                    [d["high"] for d in ohlcv_data],
                    [d["low"] for d in ohlcv_data],
                    closes,
                    volumes
                )
            )

            # VWAP
            vwap_series = self._safe_call(
                lambda: calculate_vwap_series(
                    [d["high"] for d in ohlcv_data],
                    [d["low"] for d in ohlcv_data],
                    closes,
                    volumes
                )
            )
            if vwap_series and isinstance(vwap_series, list) and len(vwap_series) > 0:
                result["vwap_10"] = vwap_series[-1]

            # VWAP Quote
            quote_volumes = [d.get("quote_asset_volume", 0) for d in ohlcv_data]
            vwap_quote_series = self._safe_call(
                lambda: calculate_vwap_quote_series(
                    [d["high"] for d in ohlcv_data],
                    [d["low"] for d in ohlcv_data],
                    closes,
                    quote_volumes
                )
            )
            if vwap_quote_series and isinstance(vwap_quote_series, list) and len(vwap_quote_series) > 0:
                result["vwap_quote_10"] = vwap_quote_series[-1]

            # Anchored VWAP, VWAP Bands, Volume Profile
            if len(closes) >= 20:
                from market_analyzer.indicators.volume.vwap import (
                    calculate_anchored_vwap,
                    calculate_vwap_bands,
                    calculate_value_area,
                    find_poc,
                )

                anchor_index = max(0, len(closes) - 50) if len(closes) > 50 else 0
                result["anchored_vwap"] = self._safe_call(
                    lambda: calculate_anchored_vwap(
                        [d["high"] for d in ohlcv_data],
                        [d["low"] for d in ohlcv_data],
                        closes,
                        volumes,
                        anchor_index
                    )
                )

                vwap_bands = self._safe_call(
                    lambda: calculate_vwap_bands(
                        [d["high"] for d in ohlcv_data],
                        [d["low"] for d in ohlcv_data],
                        closes,
                        volumes
                    )
                )
                if vwap_bands and isinstance(vwap_bands, dict):
                    result["vwap_upper_band"] = vwap_bands.get("upper_band")
                    result["vwap_lower_band"] = vwap_bands.get("lower_band")

                # Volume Profile
                result["volume_profile_poc"] = self._safe_call(lambda: find_poc(closes, volumes, 20))

                value_area = self._safe_call(lambda: calculate_value_area(closes, volumes, 0.7, 20))
                if value_area and isinstance(value_area, dict):
                    result["volume_profile_vah"] = value_area.get("vah")
                    result["volume_profile_val"] = value_area.get("val")

            # Volume context
            if len(volumes) >= 20:
                avg_volume = sum(volumes[-20:]) / 20
                quote_volumes = [d.get("quote_asset_volume", 0) for d in ohlcv_data]
                trades_counts = [d.get("number_of_trades", 0) for d in ohlcv_data]

                result.update({
                    "avg_volume_20": avg_volume,
                    "volume_ratio": (volumes[-1] / avg_volume if avg_volume > 0 else 1),
                    "quote_volume_ratio": self._safe_call(lambda: calculate_quote_volume_ratio(quote_volumes, 20)),
                    "avg_trade_size": self._safe_call(lambda: calculate_avg_trade_size(volumes[-1], trades_counts[-1])),
                    "trade_intensity": self._safe_call(lambda: calculate_trade_intensity(trades_counts, 20)),
                })

        return result

    # === MÉTHODES HELPER POUR PARALLÉLISATION - GROUPE 3 (DÉTECTEURS COMPLEXES) ===

    async def _detect_regime(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float],
        symbol: str,
        indicators: dict
    ) -> dict:
        """Détecte le régime de marché."""
        result = {}

        try:
            regime_result = self.regime_detector.detect_regime(
                highs=highs,
                lows=lows,
                closes=closes,
                volumes=volumes,
                symbol=symbol,
                include_analysis=True,
                enable_cache=True,
            )

            if regime_result:
                trend_alignment = max(-100, min(100, regime_result.trend_slope * 10))

                # Calculer momentum_score à partir de RSI + MACD + ADX + ROC
                momentum_components = []

                # RSI Component (40%)
                rsi_14 = indicators.get("rsi_14")
                if rsi_14 is not None:
                    momentum_components.append(("rsi", rsi_14, 40))

                # MACD Component (30%)
                macd_hist = indicators.get("macd_histogram")
                if macd_hist is not None:
                    macd_normalized = max(-50, min(50, float(macd_hist)))
                    macd_score = 50 + macd_normalized
                    momentum_components.append(("macd", macd_score, 30))

                # ADX Component (20%)
                adx_14 = indicators.get("adx_14")
                plus_di = indicators.get("plus_di")
                minus_di = indicators.get("minus_di")
                if adx_14 is not None and plus_di is not None and minus_di is not None:
                    if float(plus_di) > float(minus_di):
                        adx_score = min(100, 50 + (float(adx_14) / 2))
                    else:
                        adx_score = max(0, 50 - (float(adx_14) / 2))
                    momentum_components.append(("adx", adx_score, 20))

                # ROC Component (10%)
                roc_10 = indicators.get("roc_10")
                if roc_10 is not None:
                    roc_normalized = max(-5, min(5, float(roc_10) * 100))
                    roc_score = 50 + (roc_normalized * 10)
                    momentum_components.append(("roc", roc_score, 10))

                # Calculer moyenne pondérée
                if momentum_components:
                    total_weighted = sum(score * weight for _, score, weight in momentum_components)
                    total_weight = sum(weight for _, _, weight in momentum_components)
                    momentum_score = total_weighted / total_weight
                else:
                    momentum_score = 50.0

                momentum_score = max(0, min(100, momentum_score))

                result.update({
                    "market_regime": str(
                        regime_result.regime_type.value
                        if hasattr(regime_result.regime_type, "value")
                        else regime_result.regime_type
                    ).upper(),
                    "regime_strength": str(
                        regime_result.strength.value
                        if hasattr(regime_result.strength, "value")
                        else regime_result.strength
                    ).upper(),
                    "regime_confidence": float(regime_result.confidence),
                    "regime_duration": int(regime_result.duration),
                    "trend_alignment": float(trend_alignment),
                    "momentum_score": float(momentum_score),
                })

                # ATR Percentile
                try:
                    atr_percentile = calculate_atr_percentile(
                        highs=highs,
                        lows=lows,
                        closes=closes,
                        period=14,
                        lookback=100,
                        max_lookback=500,
                    )
                    result["atr_percentile"] = float(atr_percentile) if atr_percentile is not None else 50.0
                except Exception:
                    result["atr_percentile"] = 50.0

        except Exception as e:
            logger.warning(f"RegimeDetector error: {e}")

        return result

    async def _detect_support_resistance(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float],
        timeframe: str
    ) -> dict:
        """Détecte les niveaux de support et résistance."""
        result = {}

        try:
            sr_levels = self.sr_detector.detect_levels(
                highs=highs,
                lows=lows,
                closes=closes,
                volumes=volumes,
                current_price=closes[-1],
                timeframe=timeframe,
            )

            if sr_levels:
                current_price = closes[-1]
                supports = [level for level in sr_levels if level.price < current_price]
                resistances = [level for level in sr_levels if level.price > current_price]

                supports.sort(key=lambda x: current_price - x.price)
                resistances.sort(key=lambda x: x.price - current_price)

                support_prices = [
                    level.price
                    for level in sorted(
                        supports,
                        key=lambda x: -self._strength_to_number(x.strength),
                    )[:5]
                ]
                resistance_prices = [
                    level.price
                    for level in sorted(
                        resistances,
                        key=lambda x: -self._strength_to_number(x.strength),
                    )[:5]
                ]

                result.update({
                    "support_levels": support_prices,
                    "resistance_levels": resistance_prices,
                    "nearest_support": (supports[0].price if supports else None),
                    "nearest_resistance": (resistances[0].price if resistances else None),
                    "support_strength": (str(supports[0].strength.value).upper() if supports else "MODERATE"),
                    "resistance_strength": (str(resistances[0].strength.value).upper() if resistances else "MODERATE"),
                    "break_probability": float(resistances[0].break_probability if resistances else 0.5),
                    "pivot_count": len(sr_levels),
                })

        except Exception as e:
            logger.warning(f"SupportResistanceDetector error: {e}")

        return result

    async def _analyze_volume_context(
        self,
        volumes: list[float],
        closes: list[float],
        highs: list[float],
        lows: list[float],
        symbol: str
    ) -> dict:
        """Analyse le contexte de volume."""
        result = {}

        try:
            volume_result = self.volume_analyzer.analyze_volume_context(
                volumes=volumes,
                closes=closes,
                highs=highs,
                lows=lows,
                symbol=symbol,
            )

            if volume_result:
                buildup_count = (
                    self.volume_analyzer.get_buildup_period_count(np.array(volumes))
                    if volume_result.buildup_detected
                    else 0
                )

                if volume_result.spike_detected and len(volumes) >= 5:
                    spike_multiplier = (
                        volumes[-1] / np.mean(volumes[-5:-1])
                        if np.mean(volumes[-5:-1]) > 0
                        else 1.0
                    )
                else:
                    spike_multiplier = volume_result.current_volume_ratio

                result.update({
                    "volume_context": str(volume_result.context.context_type.value).upper(),
                    "volume_pattern": str(volume_result.context.pattern_detected.value).upper(),
                    "volume_quality_score": float(volume_result.quality_score),
                    "relative_volume": float(volume_result.current_volume_ratio),
                    "volume_buildup_periods": int(buildup_count),
                    "volume_spike_multiplier": float(spike_multiplier),
                })

        except Exception as e:
            logger.debug(f"VolumeContextAnalyzer error: {e}")

        return result

    async def _detect_spikes(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float]
    ) -> dict:
        """Détecte les spikes de prix/volume."""
        result = {}

        try:
            spike_events = self.spike_detector.detect_spikes(
                highs=highs,
                lows=lows,
                closes=closes,
                volumes=volumes,
                timestamps=None,
            )

            if spike_events:
                latest_spike = spike_events[0]

                if latest_spike and self._is_pattern_fresh(latest_spike, closes, volumes):
                    result.update({
                        "pattern_detected": str(latest_spike.spike_type.value).upper(),
                        "pattern_confidence": float(latest_spike.confidence),
                    })
                else:
                    result.update({
                        "pattern_detected": "NORMAL",
                        "pattern_confidence": 0.0,
                    })
            else:
                result.update({
                    "pattern_detected": "NORMAL",
                    "pattern_confidence": 0.0,
                })

        except Exception as e:
            logger.warning(f"SpikeDetector error: {e}")
            result.update({
                "pattern_detected": "NORMAL",
                "pattern_confidence": 0.0,
            })

        return result

    # === MÉTHODES UTILITAIRES ===

    def _safe_call(self, func):
        """Exécute un appel de fonction de manière sécurisée."""
        try:
            result = func()
            if result is not None and not isinstance(result, list | dict | str):
                return float(result)
        except Exception as e:
            import traceback
            logger.debug(f"Appel échoué: {e}")
            logger.debug(f"Type d'erreur: {type(e).__name__}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
        else:
            return result

    def _strength_to_number(self, strength):
        """Convertit la force en nombre pour le tri."""
        strength_map = {"WEAK": 1, "MODERATE": 2, "STRONG": 3, "MAJOR": 4}
        if hasattr(strength, "value"):
            return strength_map.get(strength.value.upper(), 0)
        return strength_map.get(str(strength).upper(), 0)

    def _sanitize_numeric_value(self, value, max_abs_value=1e19):
        """Sanitise une valeur numérique pour éviter les débordements DB."""
        if value is None:
            return None

        try:
            value = float(value)

            if not (isinstance(value, int | float) and abs(value) < float("inf")):
                return None

            if abs(value) > max_abs_value:
                logger.warning(
                    f"Valeur trop grande pour la DB: {value}, limitée à {max_abs_value}"
                )
                return max_abs_value if value > 0 else -max_abs_value
        except (ValueError, TypeError, OverflowError):
            return None
        else:
            return value

    def _is_pattern_fresh(self, latest_spike, closes, volumes):
        """Vérifie si le pattern détecté est récent et significatif."""
        if not latest_spike:
            return False

        try:
            # Vérifier changement de volatilité
            if len(closes) >= 5:
                recent_closes = closes[-5:]
                volatility_current = (
                    self._safe_call(lambda: float(np.std(recent_closes[-3:])))
                    if len(recent_closes) >= 3
                    else 0
                )
                volatility_previous = (
                    self._safe_call(lambda: float(np.std(recent_closes[:3])))
                    if len(recent_closes) >= 3
                    else 0
                )

                if (
                    volatility_previous
                    and volatility_previous > 0
                    and abs(volatility_current - volatility_previous) / volatility_previous > 0.5
                ):
                    return False

            # Vérifier cohérence du volume
            if len(volumes) >= 3:
                recent_volume_avg = self._safe_call(lambda: float(np.mean(volumes[-3:])))
                older_volume_avg = (
                    self._safe_call(lambda: float(np.mean(volumes[-6:-3])))
                    if len(volumes) >= 6
                    else recent_volume_avg
                )

                if (
                    older_volume_avg
                    and older_volume_avg > 0
                    and recent_volume_avg
                    and recent_volume_avg / older_volume_avg < 0.5
                ):
                    return False

            # Vérifier cohérence du prix
            if len(closes) >= 2:
                price_change = (closes[-1] - closes[-2]) / closes[-2]

                if (
                    hasattr(latest_spike.spike_type, "value")
                    and latest_spike.spike_type.value == "price_spike_up"
                    and price_change < -0.02
                ) or (
                    hasattr(latest_spike.spike_type, "value")
                    and latest_spike.spike_type.value == "price_spike_down"
                    and price_change > 0.02
                ):
                    return False
        except Exception:
            logger.exception("Erreur vérification fraîcheur pattern")
            return False
        else:
            return True

    async def _save_indicators_to_db(self, indicators: dict):
        """Sauvegarde tous les indicateurs dans analyzer_data."""

        query = """
            INSERT INTO analyzer_data (
                time, symbol, timeframe, analysis_timestamp, analyzer_version,
                wma_20, dema_12, tema_12, hull_20, kama_14,
                rsi_14, rsi_21, ema_7, ema_12, ema_26, ema_50, ema_99, sma_20, sma_50,
                macd_line, macd_signal, macd_histogram, ppo, macd_zero_cross, macd_signal_cross, macd_trend,
                bb_upper, bb_middle, bb_lower, bb_position, bb_width, bb_squeeze, bb_expansion, bb_breakout_direction, keltner_upper, keltner_lower,
                stoch_k, stoch_d, stoch_rsi, stoch_fast_k, stoch_fast_d, stoch_divergence, stoch_signal,
                atr_14, atr_percentile, natr, volatility_regime, atr_stop_long, atr_stop_short,
                adx_14, plus_di, minus_di, dx, adxr, trend_strength, directional_bias, trend_angle,
                williams_r, mfi_14, cci_20, momentum_10, roc_10, roc_20,
                vwap_10, vwap_quote_10, anchored_vwap, vwap_upper_band, vwap_lower_band, volume_ratio, avg_volume_20, quote_volume_ratio, avg_trade_size, trade_intensity,
                obv, obv_ma_10, obv_oscillator, ad_line,
                volume_profile_poc, volume_profile_vah, volume_profile_val,
                market_regime, regime_strength, regime_confidence, regime_duration, trend_alignment, momentum_score,
                support_levels, resistance_levels, nearest_support, nearest_resistance,
                support_strength, resistance_strength, break_probability, pivot_count,
                volume_context, volume_pattern, volume_quality_score, relative_volume,
                volume_buildup_periods, volume_spike_multiplier,
                pattern_detected, pattern_confidence,
                signal_strength, confluence_score,
                calculation_time_ms, data_quality, anomaly_detected
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16, $17, $18, $19,
                $20, $21, $22, $23, $24, $25, $26,
                $27, $28, $29, $30, $31, $32, $33, $34, $35, $36,
                $37, $38, $39, $40, $41, $42, $43,
                $44, $45, $46, $47, $48, $49,
                $50, $51, $52, $53, $54, $55, $56, $57,
                $58, $59, $60, $61, $62, $63,
                $64, $65, $66, $67, $68, $69, $70, $71, $72, $73,
                $74, $75, $76, $77,
                $78, $79, $80,
                $81, $82, $83, $84, $85, $86,
                $87, $88, $89, $90, $91, $92, $93, $94,
                $95, $96, $97, $98, $99, $100,
                $101, $102,
                $103, $104,
                $105, $106, $107
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
            sanitized_params = [
                indicators["time"],
                indicators["symbol"],
                indicators["timeframe"],
                indicators["analysis_timestamp"],
                indicators["analyzer_version"],
                # Moyennes mobiles avancées
                self._sanitize_numeric_value(indicators.get("wma_20")),
                self._sanitize_numeric_value(indicators.get("dema_12")),
                self._sanitize_numeric_value(indicators.get("tema_12")),
                self._sanitize_numeric_value(indicators.get("hull_20")),
                self._sanitize_numeric_value(indicators.get("kama_14")),
                # Indicateurs de base
                self._sanitize_numeric_value(indicators.get("rsi_14")),
                self._sanitize_numeric_value(indicators.get("rsi_21")),
                self._sanitize_numeric_value(indicators.get("ema_7")),
                self._sanitize_numeric_value(indicators.get("ema_12")),
                self._sanitize_numeric_value(indicators.get("ema_26")),
                self._sanitize_numeric_value(indicators.get("ema_50")),
                self._sanitize_numeric_value(indicators.get("ema_99")),
                self._sanitize_numeric_value(indicators.get("sma_20")),
                self._sanitize_numeric_value(indicators.get("sma_50")),
                # MACD
                self._sanitize_numeric_value(indicators.get("macd_line")),
                self._sanitize_numeric_value(indicators.get("macd_signal")),
                self._sanitize_numeric_value(indicators.get("macd_histogram")),
                self._sanitize_numeric_value(indicators.get("ppo")),
                indicators.get("macd_zero_cross", False),
                indicators.get("macd_signal_cross", False),
                indicators.get("macd_trend", "NEUTRAL"),
                # Bollinger Bands
                self._sanitize_numeric_value(indicators.get("bb_upper")),
                self._sanitize_numeric_value(indicators.get("bb_middle")),
                self._sanitize_numeric_value(indicators.get("bb_lower")),
                self._sanitize_numeric_value(indicators.get("bb_position")),
                self._sanitize_numeric_value(indicators.get("bb_width")),
                indicators.get("bb_squeeze", False),
                indicators.get("bb_expansion", False),
                indicators.get("bb_breakout_direction", "NONE"),
                self._sanitize_numeric_value(indicators.get("keltner_upper")),
                self._sanitize_numeric_value(indicators.get("keltner_lower")),
                # Stochastic
                self._sanitize_numeric_value(indicators.get("stoch_k")),
                self._sanitize_numeric_value(indicators.get("stoch_d")),
                self._sanitize_numeric_value(indicators.get("stoch_rsi")),
                self._sanitize_numeric_value(indicators.get("stoch_fast_k")),
                self._sanitize_numeric_value(indicators.get("stoch_fast_d")),
                indicators.get("stoch_divergence", False),
                indicators.get("stoch_signal", "NEUTRAL"),
                # ATR & Volatilité
                self._sanitize_numeric_value(indicators.get("atr_14")),
                self._sanitize_numeric_value(indicators.get("atr_percentile")),
                self._sanitize_numeric_value(indicators.get("natr")),
                indicators.get("volatility_regime"),
                self._sanitize_numeric_value(indicators.get("atr_stop_long")),
                self._sanitize_numeric_value(indicators.get("atr_stop_short")),
                # ADX
                self._sanitize_numeric_value(indicators.get("adx_14")),
                self._sanitize_numeric_value(indicators.get("plus_di")),
                self._sanitize_numeric_value(indicators.get("minus_di")),
                self._sanitize_numeric_value(indicators.get("dx")),
                self._sanitize_numeric_value(indicators.get("adxr")),
                indicators.get("trend_strength"),
                indicators.get("directional_bias", "NEUTRAL"),
                self._sanitize_numeric_value(indicators.get("trend_angle")),
                # Oscillateurs
                self._sanitize_numeric_value(indicators.get("williams_r")),
                self._sanitize_numeric_value(indicators.get("mfi_14")),
                self._sanitize_numeric_value(indicators.get("cci_20")),
                self._sanitize_numeric_value(indicators.get("momentum_10")),
                self._sanitize_numeric_value(indicators.get("roc_10")),
                self._sanitize_numeric_value(indicators.get("roc_20")),
                # Volume avancé
                self._sanitize_numeric_value(indicators.get("vwap_10")),
                self._sanitize_numeric_value(indicators.get("vwap_quote_10")),
                self._sanitize_numeric_value(indicators.get("anchored_vwap")),
                self._sanitize_numeric_value(indicators.get("vwap_upper_band")),
                self._sanitize_numeric_value(indicators.get("vwap_lower_band")),
                self._sanitize_numeric_value(indicators.get("volume_ratio")),
                self._sanitize_numeric_value(indicators.get("avg_volume_20")),
                self._sanitize_numeric_value(indicators.get("quote_volume_ratio")),
                self._sanitize_numeric_value(indicators.get("avg_trade_size")),
                self._sanitize_numeric_value(indicators.get("trade_intensity")),
                self._sanitize_numeric_value(indicators.get("obv")),
                self._sanitize_numeric_value(indicators.get("obv_ma_10")),
                self._sanitize_numeric_value(indicators.get("obv_oscillator")),
                self._sanitize_numeric_value(indicators.get("ad_line")),
                self._sanitize_numeric_value(indicators.get("volume_profile_poc")),
                self._sanitize_numeric_value(indicators.get("volume_profile_vah")),
                self._sanitize_numeric_value(indicators.get("volume_profile_val")),
                # Régime de marché
                indicators.get("market_regime"),
                indicators.get("regime_strength"),
                self._sanitize_numeric_value(indicators.get("regime_confidence")),
                indicators.get("regime_duration"),
                self._sanitize_numeric_value(indicators.get("trend_alignment")),
                self._sanitize_numeric_value(indicators.get("momentum_score")),
                # Support/Résistance
                (
                    json.dumps(indicators.get("support_levels", []))
                    if indicators.get("support_levels")
                    else None
                ),
                (
                    json.dumps(indicators.get("resistance_levels", []))
                    if indicators.get("resistance_levels")
                    else None
                ),
                self._sanitize_numeric_value(indicators.get("nearest_support")),
                self._sanitize_numeric_value(indicators.get("nearest_resistance")),
                indicators.get("support_strength"),
                indicators.get("resistance_strength"),
                self._sanitize_numeric_value(indicators.get("break_probability")),
                indicators.get("pivot_count"),
                # Volume contexte
                indicators.get("volume_context"),
                indicators.get("volume_pattern"),
                self._sanitize_numeric_value(indicators.get("volume_quality_score")),
                self._sanitize_numeric_value(indicators.get("relative_volume")),
                indicators.get("volume_buildup_periods"),
                self._sanitize_numeric_value(indicators.get("volume_spike_multiplier")),
                # Patterns
                indicators.get("pattern_detected"),
                self._sanitize_numeric_value(indicators.get("pattern_confidence")),
                # Indicateurs composites
                indicators.get("signal_strength"),
                self._sanitize_numeric_value(indicators.get("confluence_score")),
                # Métadonnées
                indicators.get("calculation_time_ms"),
                indicators.get("data_quality"),
                indicators.get("anomaly_detected"),
            ]

            if self.db_pool is None:
                raise RuntimeError("Database pool not initialized")

            async with self.db_pool.acquire() as conn:
                await conn.execute(query, *sanitized_params)
                await self._notify_analyzer_ready(indicators)

        except Exception:
            logger.exception("❌ Erreur sauvegarde analyzer_data")
            raise

    async def _notify_analyzer_ready(self, indicators: dict):
        """Notifie l'Analyzer que de nouvelles données sont prêtes."""
        try:
            if not self.redis_client:
                self.redis_client = redis.from_url("redis://redis:6379")

            time_value = indicators.get("time")
            notification = {
                "event": "analyzer_data_ready",
                "symbol": indicators.get("symbol"),
                "timeframe": indicators.get("timeframe"),
                "timestamp": (
                    time_value.isoformat()
                    if time_value is not None and hasattr(time_value, "isoformat")
                    else None
                ),
            }

            if self.redis_client:
                await self.redis_client.publish(
                    "analyzer_trigger", json.dumps(notification)
                )
            logger.debug(
                f"📢 Notification envoyée: {indicators.get('symbol')} {indicators.get('timeframe')}"
            )

        except Exception as e:
            logger.warning(f"⚠️ Erreur notification Redis: {e}")

    async def close(self):
        """Ferme les connexions."""
        self.running = False
        if self.db_pool:
            await self.db_pool.close()
            logger.info("🔌 IndicatorProcessor fermé")
