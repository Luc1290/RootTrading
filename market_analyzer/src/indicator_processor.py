"""
Market Analyzer Indicator Processor
Appelle DIRECTEMENT tous les modules indicator/detector existants et sauvegarde en DB.
Architecture simple : récupère données → appelle modules → sauvegarde résultats.
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone

# Ajouter les chemins pour les imports
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

# Import des détecteurs
# Import DIRECT de tous vos modules existants
# Import des modules composites
# Import des indicateurs de momentum
# Import ADX complet
# Import des moyennes avancées
# Import des indicateurs de volatilité
# Import des indicateurs de volume

logger = logging.getLogger(__name__)


class IndicatorProcessor:
    """
    Processeur simple qui appelle vos modules existants et sauvegarde en DB.
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

        logger.info("🧮 IndicatorProcessor initialisé")

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

        Args:
            symbol: Symbole (ex: BTCUSDC)
            timeframe: Timeframe (ex: 1m, 5m, 1h)
            timestamp: Timestamp de la nouvelle donnée
        """
        try:
            logger.debug(f"🔄 Traitement {symbol} {timeframe} @ {timestamp}")

            # Récupérer les données historiques nécessaires JUSQU'AU timestamp en cours
            # Optimisé: 500 points suffisent largement pour tous les
            # indicateurs (EMA99, ADX, etc.)
            ohlcv_data = await self._get_historical_data(
                symbol, timeframe, limit=500, up_to_timestamp=timestamp
            )

            if len(ohlcv_data) < 20:
                logger.debug(
                    f"⏭️ Pas assez de données pour {symbol} {timeframe}: {len(ohlcv_data)} < 20"
                )
                return

            # Appeler TOUS vos modules et collecter les résultats
            indicators_data = await self._call_all_indicator_modules(
                symbol, timeframe, ohlcv_data, timestamp
            )

            # Sauvegarder en DB
            await self._save_indicators_to_db(indicators_data)

            logger.info(f"✅ Indicateurs sauvegardés: {symbol} {timeframe}")

        except Exception:
            logger.exception("❌ Erreur traitement {symbol} {timeframe}")

    async def _get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 1000000,
        up_to_timestamp: datetime | None = None,
    ) -> list[dict]:
        """Récupère les données OHLCV historiques depuis market_data jusqu'à un timestamp donné."""
        if up_to_timestamp:
            query = """
                SELECT time, open, high, low, close, volume,
                       quote_asset_volume, number_of_trades
                FROM market_data
                WHERE symbol = $1 AND timeframe = $2 AND time <= $3
                ORDER BY time DESC
                LIMIT $4
            """
            params = [symbol, timeframe, up_to_timestamp, limit]
        else:
            query = """
                SELECT time, open, high, low, close, volume,
                       quote_asset_volume, number_of_trades
                FROM market_data
                WHERE symbol = $1 AND timeframe = $2
                ORDER BY time DESC
                LIMIT $3
            """
            params = [symbol, timeframe, limit]

        if self.db_pool is None:
            raise RuntimeError("Database pool not initialized")

        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

            # Inverser pour ordre chronologique ASC (nécessaire pour les
            # calculs d'indicateurs)
            data = []
            for row in reversed(rows):
                data.append(
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
                            else 0
                        ),
                        "number_of_trades": (
                            row["number_of_trades"] if row["number_of_trades"] else 0
                        ),
                    }
                )

            return data

    async def _call_all_indicator_modules(
        self, symbol: str, timeframe: str, ohlcv_data: list[dict], timestamp: datetime
    ) -> dict:
        """
        Appelle DIRECTEMENT tous vos modules indicator/detector existants.
        """
        start_time = time.time()

        # Extraire les arrays
        [d["open"] for d in ohlcv_data]
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
            "analyzer_version": "1.0",
        }

        try:
            # === APPEL DIRECT DE VOS MODULES INDICATORS ===

            # RSI - Utiliser VOS modules directement
            if len(closes) >= 14:
                rsi_14_series = self._safe_call(
                    lambda: calculate_rsi_series(closes, 14)
                )
                indicators["rsi_14"] = (
                    rsi_14_series[-1]
                    if rsi_14_series and len(rsi_14_series) > 0
                    else None
                )

            if len(closes) >= 21:
                rsi_21_series = self._safe_call(
                    lambda: calculate_rsi_series(closes, 21)
                )
                indicators["rsi_21"] = (
                    rsi_21_series[-1]
                    if rsi_21_series and len(rsi_21_series) > 0
                    else None
                )

            # Stochastic RSI
            if len(closes) >= 14:
                indicators["stoch_rsi"] = self._safe_call(
                    lambda: calculate_stoch_rsi(closes, 14, 14)
                )

            # EMAs - Utiliser les modules existants avec series pour obtenir la
            # dernière valeur
            if len(closes) >= 7:
                ema_7_series = self._safe_call(lambda: calculate_ema_series(closes, 7))
                indicators["ema_7"] = (
                    ema_7_series[-1] if ema_7_series and len(ema_7_series) > 0 else None
                )
            if len(closes) >= 12:
                ema_12_series = self._safe_call(
                    lambda: calculate_ema_series(closes, 12)
                )
                indicators["ema_12"] = (
                    ema_12_series[-1]
                    if ema_12_series and len(ema_12_series) > 0
                    else None
                )
            if len(closes) >= 26:
                ema_26_series = self._safe_call(
                    lambda: calculate_ema_series(closes, 26)
                )
                indicators["ema_26"] = (
                    ema_26_series[-1]
                    if ema_26_series and len(ema_26_series) > 0
                    else None
                )
            if len(closes) >= 50:
                ema_50_series = self._safe_call(
                    lambda: calculate_ema_series(closes, 50)
                )
                indicators["ema_50"] = (
                    ema_50_series[-1]
                    if ema_50_series and len(ema_50_series) > 0
                    else None
                )
            if len(closes) >= 99:
                ema_99_series = self._safe_call(
                    lambda: calculate_ema_series(closes, 99)
                )
                indicators["ema_99"] = (
                    ema_99_series[-1]
                    if ema_99_series and len(ema_99_series) > 0
                    else None
                )

            # SMAs
            if len(closes) >= 20:
                indicators["sma_20"] = self._safe_call(
                    lambda: calculate_sma(closes, 20)
                )
            if len(closes) >= 50:
                indicators["sma_50"] = self._safe_call(
                    lambda: calculate_sma(closes, 50)
                )

            # Moyennes avancées
            if len(closes) >= 20:
                indicators["wma_20"] = self._safe_call(
                    lambda: calculate_wma(closes, 20)
                )
                indicators["hull_20"] = self._safe_call(
                    lambda: calculate_hull_ma(closes, 20)
                )
            if len(closes) >= 12:
                indicators["dema_12"] = self._safe_call(
                    lambda: calculate_dema(closes, 12)
                )
                indicators["tema_12"] = self._safe_call(
                    lambda: calculate_tema(closes, 12)
                )
            if len(closes) >= 14:
                indicators["kama_14"] = self._safe_call(
                    lambda: calculate_adaptive_ma(closes, 14)
                )

            # MACD - Utiliser VOS modules directement
            if len(closes) >= 26:
                macd_series = self._safe_call(
                    lambda: calculate_macd_series(
                        closes, fast_period=12, slow_period=26, signal_period=9
                    )
                )
                if macd_series and isinstance(macd_series, dict):
                    # Vos modules retournent déjà le bon format
                    macd_line_series = macd_series.get("macd_line", [])
                    macd_signal_series = macd_series.get("macd_signal", [])
                    macd_histogram_series = macd_series.get("macd_histogram", [])

                    indicators.update(
                        {
                            "macd_line": (
                                macd_line_series[-1] if macd_line_series else None
                            ),
                            "macd_signal": (
                                macd_signal_series[-1] if macd_signal_series else None
                            ),
                            "macd_histogram": (
                                macd_histogram_series[-1]
                                if macd_histogram_series
                                else None
                            ),
                        }
                    )

                    # MACD Signaux binaires
                    from ..indicators.trend.macd import (
                        calculate_macd_trend,
                        calculate_ppo,
                        macd_signal_cross,
                        macd_zero_cross,
                    )

                    if (
                        macd_line_series
                        and macd_signal_series
                        and len(macd_line_series) >= 2
                    ):
                        current_macd = {
                            "macd_line": macd_line_series[-1],
                            "macd_signal": macd_signal_series[-1],
                            "macd_histogram": (
                                macd_histogram_series[-1]
                                if macd_histogram_series
                                else None
                            ),
                        }
                        prev_macd = {
                            "macd_line": macd_line_series[-2],
                            "macd_signal": macd_signal_series[-2],
                            "macd_histogram": (
                                macd_histogram_series[-2]
                                if macd_histogram_series
                                and len(macd_histogram_series) >= 2
                                else None
                            ),
                        }

                        # Croisements
                        zero_cross = self._safe_call(
                            lambda: macd_zero_cross(current_macd, prev_macd)
                        )
                        signal_cross = self._safe_call(
                            lambda: macd_signal_cross(current_macd, prev_macd)
                        )
                        trend = self._safe_call(
                            lambda: calculate_macd_trend(current_macd, prev_macd)
                        )

                        indicators.update(
                            {
                                "macd_zero_cross": bool(
                                    zero_cross not in [None, "none"]
                                ),
                                "macd_signal_cross": bool(
                                    signal_cross not in [None, "none"]
                                ),
                                "macd_trend": (
                                    trend.upper()
                                    if trend and trend != "none"
                                    else "NEUTRAL"
                                ),
                            }
                        )

                # PPO (Percentage Price Oscillator) - pas Price Oscillator !
                ppo_result = self._safe_call(lambda: calculate_ppo(closes, 12, 26, 9))
                if ppo_result and isinstance(ppo_result, dict):
                    indicators["ppo"] = ppo_result.get("ppo_line")

            # ADX
            if len(closes) >= 14:
                adx_full = self._safe_call(
                    lambda: calculate_adx_full(highs, lows, closes, 14)
                )
                if adx_full and isinstance(adx_full, dict):
                    indicators.update(
                        {
                            "adx_14": adx_full.get("adx"),
                            "plus_di": adx_full.get("plus_di"),
                            "minus_di": adx_full.get("minus_di"),
                            "dx": adx_full.get("dx"),
                            "adxr": adx_full.get("adxr"),
                        }
                    )

                    # Calculer trend_strength depuis ADX (string: "weak",
                    # "strong", etc.)
                    from ..indicators.trend.adx import (
                        adx_trend_strength,
                        calculate_directional_bias,
                    )

                    adx_value = adx_full.get("adx")
                    if adx_value is not None:
                        indicators["trend_strength"] = adx_trend_strength(adx_value)

                    # Calculer directional_bias depuis +DI et -DI
                    plus_di = adx_full.get("plus_di")
                    minus_di = adx_full.get("minus_di")
                    if plus_di is not None and minus_di is not None:
                        indicators["directional_bias"] = calculate_directional_bias(
                            plus_di, minus_di, adx_value
                        )

            # Trend Angle
            if len(closes) >= 14:
                from ..indicators.trend.adx import calculate_trend_angle

                indicators["trend_angle"] = self._safe_call(
                    lambda: calculate_trend_angle(closes, 14)
                )

            # ATR et volatilité (calculer d'abord car utilisé par Keltner)
            atr = None
            if len(closes) >= 14:
                atr = self._safe_call(lambda: calculate_atr(highs, lows, closes, 14))
                indicators["atr_14"] = atr

                if atr and closes[-1] > 0:
                    indicators["natr"] = self._safe_call(
                        lambda: calculate_natr(highs, lows, closes, 14)
                    )

                    # Régime de volatilité
                    volatility_reg = self._safe_call(
                        lambda: volatility_regime(highs, lows, closes, 14, 20)
                    )  # Réduire lookback de 50 à 20
                    indicators["volatility_regime"] = volatility_reg

                    # ATR stop loss calculé manuellement
                    atr_multiplier = 2.0
                    indicators.update(
                        {
                            "atr_stop_long": closes[-1] - (atr * atr_multiplier),
                            "atr_stop_short": closes[-1] + (atr * atr_multiplier),
                        }
                    )

            # Bollinger Bands
            if len(closes) >= 20:
                bb_series = self._safe_call(
                    lambda: calculate_bollinger_bands_series(closes, 20, 2.0)
                )
                if bb_series and isinstance(bb_series, dict):
                    # bb_series is a dict with lists for each key
                    bb_upper_series = bb_series.get("upper", [])
                    bb_middle_series = bb_series.get("middle", [])
                    bb_lower_series = bb_series.get("lower", [])
                    bb_percent_b_series = bb_series.get("percent_b", [])
                    bb_bandwidth_series = bb_series.get("bandwidth", [])

                    indicators.update(
                        {
                            "bb_upper": (
                                bb_upper_series[-1] if bb_upper_series else None
                            ),
                            "bb_middle": (
                                bb_middle_series[-1] if bb_middle_series else None
                            ),
                            "bb_lower": (
                                bb_lower_series[-1] if bb_lower_series else None
                            ),
                            "bb_position": (
                                bb_percent_b_series[-1] if bb_percent_b_series else None
                            ),
                            "bb_width": (
                                bb_bandwidth_series[-1] if bb_bandwidth_series else None
                            ),
                        }
                    )

                    # Bollinger Signaux binaires
                    from ..indicators.volatility.bollinger import (
                        calculate_bollinger_breakout_direction,
                        calculate_bollinger_expansion,
                        calculate_bollinger_squeeze,
                    )

                    squeeze_data = self._safe_call(
                        lambda: calculate_bollinger_squeeze(closes, 20, 2.0, 20)
                    )
                    expansion = self._safe_call(
                        lambda: calculate_bollinger_expansion(closes, 20, 2.0, 10)
                    )
                    breakout_dir = self._safe_call(
                        lambda: calculate_bollinger_breakout_direction(
                            closes, 20, 2.0, 3
                        )
                    )

                    indicators.update(
                        {
                            "bb_squeeze": (
                                bool(squeeze_data.get("in_squeeze", False))
                                if squeeze_data
                                else False
                            ),
                            "bb_expansion": (
                                bool(expansion) if expansion is not None else False
                            ),
                            "bb_breakout_direction": (
                                breakout_dir if breakout_dir else "NONE"
                            ),
                        }
                    )

                # Keltner Channels (utilise la fonction dédiée)
                if len(highs) >= 20 and len(lows) >= 20:
                    keltner = self._safe_call(
                        lambda: calculate_keltner_channels(
                            closes,
                            highs,
                            lows,
                            period=20,
                            atr_period=10,
                            multiplier=2.0,
                        )
                    )
                    if keltner and isinstance(keltner, dict):
                        indicators.update(
                            {
                                "keltner_upper": keltner.get("upper"),
                                "keltner_lower": keltner.get("lower"),
                            }
                        )

            # Stochastic
            if len(closes) >= 14:
                stoch_series = self._safe_call(
                    lambda: calculate_stochastic_series(highs, lows, closes, 14, 3)
                )
                if stoch_series and isinstance(stoch_series, dict):
                    # stoch_series is a dict with 'k' and 'd' keys containing
                    # lists
                    stoch_k_series = stoch_series.get("k", [])
                    stoch_d_series = stoch_series.get("d", [])

                    indicators.update(
                        {
                            "stoch_k": stoch_k_series[-1] if stoch_k_series else None,
                            "stoch_d": stoch_d_series[-1] if stoch_d_series else None,
                        }
                    )

                # Fast Stochastic
                fast_stoch_series = self._safe_call(
                    lambda: calculate_stochastic_series(highs, lows, closes, 14, 1)
                )
                if fast_stoch_series and isinstance(fast_stoch_series, dict):
                    # fast_stoch_series is a dict with 'k' and 'd' keys
                    # containing lists
                    fast_k_series = fast_stoch_series.get("k", [])
                    fast_d_series = fast_stoch_series.get("d", [])

                    indicators.update(
                        {
                            "stoch_fast_k": (
                                fast_k_series[-1] if fast_k_series else None
                            ),
                            "stoch_fast_d": (
                                fast_d_series[-1] if fast_d_series else None
                            ),
                        }
                    )

                    # Stochastic Signaux binaires
                    from ..indicators.oscillators.stochastic import (
                        calculate_stochastic_divergence,
                        calculate_stochastic_signal,
                    )

                    divergence = self._safe_call(
                        lambda: calculate_stochastic_divergence(closes, highs, lows, 14)
                    )
                    signal = self._safe_call(
                        lambda: calculate_stochastic_signal(highs, lows, closes, 14, 3)
                    )

                    indicators.update(
                        {
                            "stoch_divergence": (
                                bool(divergence not in [None, "none"])
                                if divergence
                                else False
                            ),
                            "stoch_signal": (
                                signal.upper()
                                if signal and signal != "none"
                                else "NEUTRAL"
                            ),
                        }
                    )

            # Williams %R
            if len(closes) >= 14:
                indicators["williams_r"] = self._safe_call(
                    lambda: calculate_williams_r(highs, lows, closes, 14)
                )

            # MFI (Money Flow Index)
            if len(closes) >= 15 and len(volumes) >= 15:  # Need period + 1
                from ..indicators.momentum.mfi import calculate_mfi

                indicators["mfi_14"] = self._safe_call(
                    lambda: calculate_mfi(highs, lows, closes, volumes, 14)
                )

            # CCI
            if len(closes) >= 20:
                indicators["cci_20"] = self._safe_call(
                    lambda: calculate_cci(highs, lows, closes, 20)
                )

            # Momentum et ROC
            if len(closes) >= 10:
                indicators["momentum_10"] = self._safe_call(
                    lambda: calculate_momentum(closes, 10)
                )
                indicators["roc_10"] = self._safe_call(
                    lambda: calculate_roc(closes, 10)
                )
            if len(closes) >= 20:
                indicators["roc_20"] = self._safe_call(
                    lambda: calculate_roc(closes, 20)
                )

            # Volume (OBV et VWAP)
            if len(volumes) >= 10:
                obv_series = self._safe_call(
                    lambda: calculate_obv_series(closes, volumes)
                )
                if obv_series and isinstance(obv_series, list) and len(obv_series) > 0:
                    indicators["obv"] = obv_series[-1]

                    # OBV MA et oscillateur
                    if len(obv_series) >= 10:
                        indicators["obv_ma_10"] = self._safe_call(
                            lambda: calculate_obv_ma(closes, volumes, 10)
                        )
                        indicators["obv_oscillator"] = self._safe_call(
                            lambda: calculate_obv_oscillator(closes, volumes, 10)
                        )

                # A/D Line (Accumulation/Distribution Line)
                from ..indicators.volume.obv import (
                    calculate_volume_accumulation_distribution,
                )

                indicators["ad_line"] = self._safe_call(
                    lambda: calculate_volume_accumulation_distribution(
                        highs, lows, closes, volumes
                    )
                )

                vwap_series = self._safe_call(
                    lambda: calculate_vwap_series(highs, lows, closes, volumes)
                )
                if (
                    vwap_series
                    and isinstance(vwap_series, list)
                    and len(vwap_series) > 0
                ):
                    indicators["vwap_10"] = vwap_series[-1]

                # VWAP Quote (plus précis avec quote_asset_volume)
                quote_volumes = [d.get("quote_asset_volume", 0) for d in ohlcv_data]
                vwap_quote_series = self._safe_call(
                    lambda: calculate_vwap_quote_series(
                        highs, lows, closes, quote_volumes
                    )
                )
                if (
                    vwap_quote_series
                    and isinstance(vwap_quote_series, list)
                    and len(vwap_quote_series) > 0
                ):
                    indicators["vwap_quote_10"] = vwap_quote_series[-1]

                # Anchored VWAP (utilise un point d'ancrage significatif)
                from ..indicators.volume.vwap import calculate_anchored_vwap

                if len(closes) >= 20:  # Minimum de données nécessaires
                    # Utilise les 50 derniers points comme ancrage ou début des
                    # données si moins de 50 points
                    anchor_index = max(0, len(closes) - 50) if len(closes) > 50 else 0
                    indicators["anchored_vwap"] = self._safe_call(
                        lambda: calculate_anchored_vwap(
                            highs, lows, closes, volumes, anchor_index
                        )
                    )

                # VWAP Bands (upper/lower bands pour support/résistance)
                from ..indicators.volume.vwap import calculate_vwap_bands

                if len(closes) >= 20:
                    vwap_bands = self._safe_call(
                        lambda: calculate_vwap_bands(highs, lows, closes, volumes)
                    )
                    if vwap_bands and isinstance(vwap_bands, dict):
                        indicators["vwap_upper_band"] = vwap_bands.get("upper_band")
                        indicators["vwap_lower_band"] = vwap_bands.get("lower_band")

                # Volume context avec métriques avancées
                if len(volumes) >= 20:
                    avg_volume = sum(volumes[-20:]) / 20

                    # Extraire les données pour les métriques avancées
                    quote_volumes = [d.get("quote_asset_volume", 0) for d in ohlcv_data]
                    trades_counts = [d.get("number_of_trades", 0) for d in ohlcv_data]

                    indicators.update(
                        {
                            "avg_volume_20": avg_volume,
                            "volume_ratio": (
                                volumes[-1] / avg_volume if avg_volume > 0 else 1
                            ),
                            "quote_volume_ratio": self._safe_call(
                                lambda: calculate_quote_volume_ratio(quote_volumes, 20)
                            ),
                            "avg_trade_size": self._safe_call(
                                lambda: calculate_avg_trade_size(
                                    volumes[-1], trades_counts[-1]
                                )
                            ),
                            "trade_intensity": self._safe_call(
                                lambda: calculate_trade_intensity(trades_counts, 20)
                            ),
                        }
                    )

                # Volume Profile (POC, VAH, VAL)
                if len(volumes) >= 20:
                    from ..indicators.volume.vwap import calculate_value_area, find_poc

                    indicators["volume_profile_poc"] = self._safe_call(
                        lambda: find_poc(closes, volumes, 20)
                    )

                    value_area = self._safe_call(
                        lambda: calculate_value_area(closes, volumes, 0.7, 20)
                    )
                    if value_area and isinstance(value_area, dict):
                        indicators["volume_profile_vah"] = value_area.get("vah")
                        indicators["volume_profile_val"] = value_area.get("val")

            # === APPEL DIRECT DE VOS MODULES DETECTORS ===
            logger.debug(
                f"🔍 {symbol} {timeframe}: {len(closes)} closes disponibles (min 100 requis pour régime)"
            )
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
                        enable_cache=True,
                    )

                    if regime_result:
                        logger.debug(
                            f"✅ Régime détecté pour {symbol} {timeframe}: {regime_result.regime_type.value}"
                        )
                        # Calculer trend_alignment à partir du trend_slope
                        # (normaliser entre -100 et 100)
                        trend_alignment = max(
                            -100, min(100, regime_result.trend_slope * 10)
                        )

                        # Calculer momentum_score (0-100, où 50 = neutre)
                        # FIXÉ: Utiliser RSI + MACD + ADX au lieu de
                        # trend_slope qui varie peu

                        # Composantes du momentum (chacune 0-100, moyenne =
                        # score final)
                        momentum_components = []

                        # 1. RSI Component (40 pts max)
                        rsi_14 = indicators.get("rsi_14")
                        if rsi_14 is not None:
                            # RSI 50 = neutre (50pts), RSI 70 = bullish
                            # (80pts), RSI 30 = bearish (20pts)
                            rsi_score = rsi_14  # RSI est déjà 0-100
                            momentum_components.append(
                                ("rsi", rsi_score, 40)
                            )  # Poids 40%

                        # 2. MACD Histogram Component (30 pts max)
                        macd_hist = indicators.get("macd_histogram")
                        if macd_hist is not None:
                            # Normaliser MACD histogram (typiquement -50 à +50 pour crypto)
                            # Positif = bullish, négatif = bearish
                            macd_normalized = max(
                                -50, min(50, float(macd_hist))
                            )  # Limiter à ±50
                            macd_score = 50 + macd_normalized  # Convertir en 0-100
                            momentum_components.append(
                                ("macd", macd_score, 30)
                            )  # Poids 30%

                        # 3. ADX Component (20 pts max)
                        adx_14 = indicators.get("adx_14")
                        plus_di = indicators.get("plus_di")
                        minus_di = indicators.get("minus_di")
                        if (
                            adx_14 is not None
                            and plus_di is not None
                            and minus_di is not None
                        ):
                            # ADX mesure la force de la tendance (0-100)
                            # +DI vs -DI donne la direction
                            if float(plus_di) > float(minus_di):
                                # Tendance haussière: score = 50 + (ADX/2)
                                # ADX 25 → 62.5, ADX 50 → 75
                                adx_score = min(100, 50 + (float(adx_14) / 2))
                            else:
                                # Tendance baissière: score = 50 - (ADX/2)
                                adx_score = max(0, 50 - (float(adx_14) / 2))
                            momentum_components.append(
                                ("adx", adx_score, 20)
                            )  # Poids 20%

                        # 4. ROC Component (10 pts max)
                        roc_10 = indicators.get("roc_10")
                        if roc_10 is not None:
                            # ROC typiquement -5% à +5% pour 1m crypto
                            # Normaliser autour de 50
                            roc_normalized = max(
                                -5, min(5, float(roc_10) * 100)
                            )  # Convertir en %
                            # ±50 points max
                            roc_score = 50 + (roc_normalized * 10)
                            momentum_components.append(
                                ("roc", roc_score, 10)
                            )  # Poids 10%

                        # Calculer moyenne pondérée
                        if momentum_components:
                            total_weighted = sum(
                                score * weight
                                for _, score, weight in momentum_components
                            )
                            total_weight = sum(
                                weight for _, _, weight in momentum_components
                            )
                            momentum_score = total_weighted / total_weight
                        else:
                            # Fallback si aucun composant disponible
                            momentum_score = 50.0

                        # Limiter entre 0-100
                        momentum_score = max(0, min(100, momentum_score))

                        indicators.update(
                            {
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
                            }
                        )

                        # Calculer atr_percentile en utilisant la fonction
                        # dédiée
                        try:
                            atr_percentile = calculate_atr_percentile(
                                highs=highs,
                                lows=lows,
                                closes=closes,
                                period=14,
                                lookback=100,
                                max_lookback=500,
                            )
                            if atr_percentile is not None:
                                indicators["atr_percentile"] = float(atr_percentile)
                            else:
                                indicators["atr_percentile"] = (
                                    50.0  # Valeur neutre par défaut
                                )
                        except Exception as e:
                            logger.debug(f"Erreur calcul atr_percentile: {e}")
                            indicators["atr_percentile"] = 50.0
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
                        timeframe=timeframe,
                    )

                    if sr_levels:
                        # Séparer supports et résistances
                        current_price = closes[-1]
                        supports = [
                            level for level in sr_levels if level.price < current_price
                        ]
                        resistances = [
                            level for level in sr_levels if level.price > current_price
                        ]

                        # Trier par proximité pour trouver les plus proches
                        supports.sort(
                            key=lambda x: current_price - x.price
                        )  # Plus proche en premier
                        resistances.sort(
                            key=lambda x: x.price - current_price
                        )  # Plus proche en premier

                        # Extraire les prix pour JSONB (garder l'ordre par
                        # force pour la liste)
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

                        indicators.update(
                            {
                                "support_levels": support_prices,
                                "resistance_levels": resistance_prices,
                                "nearest_support": (
                                    supports[0].price if supports else None
                                ),
                                "nearest_resistance": (
                                    resistances[0].price if resistances else None
                                ),
                                "support_strength": (
                                    str(supports[0].strength.value).upper()
                                    if supports
                                    else "MODERATE"
                                ),
                                "resistance_strength": (
                                    str(resistances[0].strength.value).upper()
                                    if resistances
                                    else "MODERATE"
                                ),
                                "break_probability": float(
                                    resistances[0].break_probability
                                    if resistances
                                    else 0.5
                                ),  # FIXÉ: resistances au lieu de supports
                                "pivot_count": len(sr_levels),
                            }
                        )
                except Exception as e:
                    logger.warning(f"SupportResistanceDetector error: {e}")

                # VolumeContextAnalyzer
                try:
                    volume_result = self.volume_analyzer.analyze_volume_context(
                        volumes=volumes,
                        closes=closes,
                        highs=highs,
                        lows=lows,
                        symbol=symbol,
                    )

                    if volume_result:
                        # Calculer le nombre RÉEL de périodes de buildup (pas
                        # juste la config)
                        buildup_count = (
                            self.volume_analyzer.get_buildup_period_count(
                                np.array(volumes)
                            )
                            if volume_result.buildup_detected
                            else 0
                        )

                        # volume_spike_multiplier: Calculer le multiplicateur réel du spike
                        # Si spike détecté, c'est le ratio vs moyenne des 4 périodes précédentes
                        # Sinon, c'est toujours le ratio actuel (cohérent avec
                        # relative_volume)
                        if volume_result.spike_detected and len(volumes) >= 5:
                            # Spike = volume actuel vs moyenne des 4 dernières
                            # (excluant actuelle)
                            spike_multiplier = (
                                volumes[-1] / np.mean(volumes[-5:-1])
                                if np.mean(volumes[-5:-1]) > 0
                                else 1.0
                            )
                        else:
                            # Pas de spike, mais on garde le ratio pour
                            # cohérence
                            spike_multiplier = volume_result.current_volume_ratio

                        indicators.update(
                            {
                                "volume_context": str(
                                    volume_result.context.context_type.value
                                ).upper(),
                                "volume_pattern": str(
                                    volume_result.context.pattern_detected.value
                                ).upper(),
                                "volume_quality_score": float(
                                    volume_result.quality_score
                                ),
                                "relative_volume": float(
                                    volume_result.current_volume_ratio
                                ),
                                "volume_buildup_periods": int(
                                    buildup_count
                                ),  # FIXÉ: nombre réel au lieu de constante
                                "volume_spike_multiplier": float(
                                    spike_multiplier
                                ),  # FIXÉ: multiplicateur réel du spike
                            }
                        )
                except Exception as e:
                    logger.debug(f"VolumeContextAnalyzer error: {e}")

                # SpikeDetector avec vérification de fraîcheur
                try:
                    spike_events = self.spike_detector.detect_spikes(
                        highs=highs,
                        lows=lows,
                        closes=closes,
                        volumes=volumes,
                        timestamps=None,  # Will be auto-generated
                    )

                    if spike_events:
                        # Prendre le spike le plus récent
                        latest_spike = spike_events[0] if spike_events else None

                        # Vérifier la fraîcheur du pattern pour éviter la
                        # persistance
                        if latest_spike and self._is_pattern_fresh(
                            latest_spike, closes, volumes
                        ):
                            indicators.update(
                                {
                                    "pattern_detected": str(
                                        latest_spike.spike_type.value
                                    ).upper(),
                                    "pattern_confidence": float(
                                        latest_spike.confidence
                                    ),
                                }
                            )
                        else:
                            # Pattern trop ancien ou non significatif
                            indicators.update(
                                {
                                    "pattern_detected": "NORMAL",
                                    "pattern_confidence": 0.0,
                                }
                            )
                    else:
                        indicators.update(
                            {"pattern_detected": "NORMAL", "pattern_confidence": 0.0}
                        )
                except Exception as e:
                    logger.warning(f"SpikeDetector error: {e}")
                    indicators.update(
                        {"pattern_detected": "NORMAL", "pattern_confidence": 0.0}
                    )

            # === CALCUL CONFLUENCE SCORE ===
            confluence_score = self._safe_call(
                lambda: calculate_confluence_score(
                    indicators,
                    closes[-1] if closes else None,
                    ConfluenceType.MONO_TIMEFRAME,
                )
            )
            indicators["confluence_score"] = confluence_score

            # === CALCUL SIGNAL STRENGTH ===
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
                f"🧮 {len([k for k, v in indicators.items() if v is not None])} indicateurs calculés en {calculation_time}ms"
            )

        except Exception as e:
            import traceback

            logger.exception("❌ Erreur lors des calculs")
            logger.exception(f"❌ Type d'erreur: {type(e).__name__}")
            logger.exception(f"❌ Traceback: {traceback.format_exc()}")
            indicators["data_quality"] = "POOR"
            indicators["anomaly_detected"] = True

        return indicators

    def _safe_call(self, func):
        """Exécute un appel de fonction de manière sécurisée."""
        try:
            result = func()
            # Ne convertir en float que les types numériques, pas les strings
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
        """Sanitise une valeur numérique pour éviter les débordements DB (DECIMAL(28,8) -> max ~1e19)."""
        if value is None:
            return None

        try:
            value = float(value)

            # Vérifier si la valeur est finie
            if not (isinstance(value, int | float) and abs(value) < float("inf")):
                return None

            # Limiter la valeur absolue pour éviter les débordements DB
            if abs(value) > max_abs_value:
                logger.warning(
                    f"Valeur trop grande pour la DB: {value}, limitée à {max_abs_value}"
                )
                return max_abs_value if value > 0 else -max_abs_value
        except (ValueError, TypeError, OverflowError):
            return None
        else:
            return value

    async def _save_indicators_to_db(self, indicators: dict):
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
                macd_line, macd_signal, macd_histogram, ppo, macd_zero_cross, macd_signal_cross, macd_trend,
                -- Bollinger Bands
                bb_upper, bb_middle, bb_lower, bb_position, bb_width, bb_squeeze, bb_expansion, bb_breakout_direction, keltner_upper, keltner_lower,
                -- Stochastic
                stoch_k, stoch_d, stoch_rsi, stoch_fast_k, stoch_fast_d, stoch_divergence, stoch_signal,
                -- ATR & Volatilité
                atr_14, atr_percentile, natr, volatility_regime, atr_stop_long, atr_stop_short,
                -- ADX
                adx_14, plus_di, minus_di, dx, adxr, trend_strength, directional_bias, trend_angle,
                -- Oscillateurs
                williams_r, mfi_14, cci_20, momentum_10, roc_10, roc_20,
                -- Volume avancé
                vwap_10, vwap_quote_10, anchored_vwap, vwap_upper_band, vwap_lower_band, volume_ratio, avg_volume_20, quote_volume_ratio, avg_trade_size, trade_intensity,
                obv, obv_ma_10, obv_oscillator, ad_line,
                volume_profile_poc, volume_profile_vah, volume_profile_val,
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
                -- Indicateurs composites
                signal_strength, confluence_score,
                -- Métadonnées
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
            # Sanitize all numeric values to prevent DB overflow - ORDRE EXACT
            # DU SCHEMA
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
                # Support/Résistance (JSONB)
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

                # Notification Redis simple : nouvelles données analysées
                # disponibles
                await self._notify_analyzer_ready(indicators)

        except Exception:
            logger.exception("❌ Erreur sauvegarde analyzer_data")
            raise

    async def _notify_analyzer_ready(self, indicators: dict):
        """Notifie l'Analyzer que de nouvelles données sont prêtes."""
        try:
            if not self.redis_client:
                # Connexion Redis paresseuse
                self.redis_client = redis.from_url("redis://redis:6379")

            time_value = indicators.get("time")
            notification = {
                "event": "analyzer_data_ready",
                "symbol": indicators.get("symbol"),
                "timeframe": indicators.get("timeframe"),
                "timestamp": (
                    time_value.isoformat()  # type: ignore
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
            # Ne pas faire échouer la sauvegarde si Redis fail
            logger.warning(f"⚠️ Erreur notification Redis: {e}")

    def _is_pattern_fresh(self, latest_spike, closes, volumes):
        """Vérifie si le pattern détecté est récent et significatif."""
        if not latest_spike:
            return False

        try:
            # Vérifier la fraîcheur temporelle (pattern ne doit pas être trop ancien)
            # En l'absence de timestamps réels, on utilise la position dans la
            # série

            # Vérifier si les conditions de marché ont significativement changé
            # depuis la détection du pattern
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

                # Si la volatilité a significativement changé, le pattern n'est
                # plus frais
                if (
                    volatility_previous
                    and volatility_previous > 0
                    and abs(volatility_current - volatility_previous)
                    / volatility_previous
                    > 0.5
                ):
                    return False

            # Vérifier la cohérence du volume avec le pattern détecté
            if len(volumes) >= 3:
                recent_volume_avg = self._safe_call(
                    lambda: float(np.mean(volumes[-3:]))
                )
                older_volume_avg = (
                    self._safe_call(lambda: float(np.mean(volumes[-6:-3])))
                    if len(volumes) >= 6
                    else recent_volume_avg
                )

                # Si le volume a drastiquement diminué, le pattern perd sa
                # validité
                if (
                    older_volume_avg
                    and older_volume_avg > 0
                    and recent_volume_avg
                    and recent_volume_avg / older_volume_avg < 0.5
                ):
                    return False

            # Vérifier la cohérence du prix avec le type de pattern
            if len(closes) >= 2:
                price_change = (closes[-1] - closes[-2]) / closes[-2]

                # Pour un PRICE_SPIKE_UP, le prix ne devrait pas chuter
                # drastiquement
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

    async def close(self):
        """Ferme les connexions."""
        self.running = False
        if self.db_pool:
            await self.db_pool.close()
            logger.info("🔌 IndicatorProcessor fermé")
