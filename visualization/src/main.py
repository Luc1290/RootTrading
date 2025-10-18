import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import psycopg2
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from notifications.telegram_service import get_notifier
from visualization.src.chart_service import ChartService
from visualization.src.data_manager import DataManager
from visualization.src.opportunity_calculator_pro import OpportunityCalculatorPro
from visualization.src.statistics_service import StatisticsService
from visualization.src.websocket_hub import WebSocketHub
from shared.src.config import SYMBOLS

# Ajouter le path pour accéder au module notifications
sys.path.insert(0, str((Path(__file__).parent / "../..").resolve()))

# Configuration du logging centralisée
from shared.logging_config import setup_logging

logger = setup_logging("visualization", log_level="INFO")


# Service container to hold all services
class ServiceContainer:
    """Container to hold all application services"""

    def __init__(self):
        self.data_manager: DataManager | None = None
        self.chart_service: ChartService | None = None
        self.websocket_hub: WebSocketHub | None = None
        self.statistics_service: StatisticsService | None = None
        self.opportunity_calculator: OpportunityCalculatorPro | None = None
        self._websocket_task: asyncio.Task | None = None


# Global service container instance
services = ServiceContainer()


def check_service_availability(
    data_mgr: bool = False,
    chart_svc: bool = False,
    stats_svc: bool = False,
    opp_calc: bool = False,
) -> None:
    """Check if required services are available, raise HTTPException if not"""
    if data_mgr and (
        services.data_manager is None or not services.data_manager.postgres_pool
    ):
        raise HTTPException(status_code=503, detail="Data manager not available")
    if chart_svc and services.chart_service is None:
        raise HTTPException(status_code=503, detail="Chart service not available")
    if stats_svc and services.statistics_service is None:
        raise HTTPException(status_code=503, detail="Statistics service not available")
    if opp_calc and services.opportunity_calculator is None:
        raise HTTPException(
            status_code=503, detail="Opportunity calculator not available"
        )


def validate_parameters(period: str | None = None, interval: str | None = None) -> None:
    """Validate period and interval parameters"""
    if period is not None:
        valid_periods = ["1d", "7d", "30d", "90d", "1y"]
        if period not in valid_periods:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid period. Must be one of: {valid_periods}",
            )
    if interval is not None:
        valid_intervals = ["1h", "1d"]
        if interval not in valid_intervals:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid interval. Must be one of: {valid_intervals}",
            )


def check_trading_symbols_config() -> list[str]:
    """Check and return configured trading symbols from environment"""
    trading_symbols_env = os.getenv("TRADING_SYMBOLS", "")
    if not trading_symbols_env:
        raise HTTPException(
            status_code=500, detail="TRADING_SYMBOLS not configured in .env"
        )
    return [s.strip() for s in trading_symbols_env.split(",")]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Starting visualization service...")

    services.data_manager = DataManager()
    await services.data_manager.initialize()

    services.chart_service = ChartService(services.data_manager)
    services.websocket_hub = WebSocketHub(services.data_manager)
    services.statistics_service = StatisticsService(services.data_manager)
    services.opportunity_calculator = (
        OpportunityCalculatorPro()
    )  # PRO: Scoring 7 catégories + Validation 4 niveaux + 108 indicateurs

    services._websocket_task = asyncio.create_task(services.websocket_hub.start())

    yield

    logger.info("Shutting down visualization service...")
    await services.websocket_hub.stop()
    await services.data_manager.close()


app = FastAPI(
    title="RootTrading Visualization Service", version="1.0.0", lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration des templates et fichiers statiques
templates = Jinja2Templates(directory="src/templates")
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Configuration des templates et fichiers statiques (temporaire)
frontend_build_path = "frontend/dist"
serve_react_app_flag = Path(frontend_build_path).exists()

if serve_react_app_flag:
    # Serve static assets first
    app.mount(
        "/assets", StaticFiles(directory=f"{frontend_build_path}/assets"), name="assets"
    )
    logger.info(f"Serving React app from {frontend_build_path}")
else:
    logger.warning(
        f"Frontend build path {frontend_build_path} not found, serving legacy template"
    )

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        """Interface web principale (legacy)"""
        return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(tz=timezone.utc).isoformat() + "Z",
        "redis_connected": (
            services.data_manager.is_redis_connected()
            if services.data_manager
            else False
        ),
        "postgres_connected": (
            services.data_manager.is_postgres_connected()
            if services.data_manager
            else False
        ),
    }


@app.get("/api/system/alerts")
async def get_system_alerts():
    """Get system health alerts from all services"""

    async def fetch_service_health(url: str, service_name: str):
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response,
            ):
                if response.status == 200:
                    return await response.json()
                return {
                    "status": "error",
                    "service": service_name,
                    "error": f"HTTP {response.status}",
                }
        except Exception as e:
            return {"status": "offline", "service": service_name, "error": str(e)}

    # Appels vers les services Docker
    portfolio_health, trader_health = await asyncio.gather(
        fetch_service_health("http://portfolio:8000/health", "portfolio"),
        fetch_service_health("http://trader:5002/health", "trader"),
        return_exceptions=True,
    )

    # Gérer les exceptions
    if isinstance(portfolio_health, Exception):
        portfolio_health = {
            "status": "offline",
            "service": "portfolio",
            "error": str(portfolio_health),
        }
    if isinstance(trader_health, Exception):
        trader_health = {
            "status": "offline",
            "service": "trader",
            "error": str(trader_health),
        }

    return {
        "portfolio": portfolio_health,
        "trader": trader_health,
        "timestamp": datetime.now(tz=timezone.utc).isoformat() + "Z",
    }


@app.get("/api/charts/market/{symbol}")
async def get_market_chart(
    symbol: str,
    interval: str = "1m",
    limit: int | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
):
    """Get market data chart for a specific symbol"""
    try:
        # Adapter automatiquement la limite selon le timeframe si non spécifiée
        if limit is None:
            timeframe_limits = {
                "1m": 2880,  # 48 heures de données pour le dézoom
                "3m": 960,  # 48 heures de données
                "5m": 2016,  # 7 jours de données
                "15m": 1344,  # 14 jours de données
                "1h": 720,  # 30 jours de données
                "1d": 365,  # 1 an de données
            }
            limit = timeframe_limits.get(interval, 2880)

        check_service_availability(chart_svc=True)

        return await services.chart_service.get_market_chart(
            symbol=symbol,
            interval=interval,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
        )
    except Exception as e:
        logger.exception("Error getting market chart")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/charts/signals/{symbol}")
async def get_signals_chart(
    symbol: str,
    strategy: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
):
    """Get trading signals overlaid on price chart"""
    try:
        check_service_availability(chart_svc=True)

        return await services.chart_service.get_signals_chart(
            symbol=symbol, strategy=strategy, start_time=start_time, end_time=end_time
        )
    except Exception as e:
        logger.exception("Error getting signals chart")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/charts/telegram-signals/{symbol}")
async def get_telegram_signals(symbol: str, hours: int = 24, limit: int = 100):
    """Get Telegram signals for a specific symbol"""
    try:
        check_service_availability(data_mgr=True)

        async with services.data_manager.postgres_pool.acquire() as conn:
            query = """
                SELECT
                    id,
                    symbol,
                    side,
                    timestamp,
                    score,
                    price,
                    action,
                    tp1, tp2, tp3,
                    stop_loss,
                    reason,
                    momentum,
                    volume_ratio,
                    regime,
                    estimated_hold_time,
                    grade,
                    rr_ratio,
                    risk_level,
                    telegram_message_id,
                    metadata,
                    created_at
                FROM telegram_signals
                WHERE symbol = $1
                  AND timestamp > NOW() - INTERVAL '1 hour' * $2
                ORDER BY timestamp DESC
                LIMIT $3
            """

            rows = await conn.fetch(query, symbol, hours, limit)

            signals = []
            for row in rows:
                signal = {
                    "id": row["id"],
                    "symbol": row["symbol"],
                    "side": row["side"],
                    "timestamp": row["timestamp"].isoformat(),
                    "score": row["score"],
                    "price": float(row["price"]),
                    "action": row["action"],
                    "targets": {
                        "tp1": float(row["tp1"]) if row["tp1"] else None,
                        "tp2": float(row["tp2"]) if row["tp2"] else None,
                        "tp3": float(row["tp3"]) if row["tp3"] else None,
                    },
                    "stop_loss": float(row["stop_loss"]) if row["stop_loss"] else None,
                    "reason": row["reason"],
                    "momentum": float(row["momentum"]) if row["momentum"] else None,
                    "volume_ratio": (
                        float(row["volume_ratio"]) if row["volume_ratio"] else None
                    ),
                    "regime": row["regime"],
                    "estimated_hold_time": row["estimated_hold_time"],
                    "grade": row["grade"],
                    "rr_ratio": float(row["rr_ratio"]) if row["rr_ratio"] else None,
                    "risk_level": row["risk_level"],
                    "telegram_message_id": row["telegram_message_id"],
                    "metadata": row["metadata"],
                    "created_at": row["created_at"].isoformat(),
                }
                signals.append(signal)

            return {"signals": signals}

    except Exception as e:
        logger.exception("Error getting Telegram signals")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/charts/performance")
async def get_performance_chart(period: str = "24h", metric: str = "pnl"):
    """Get portfolio performance chart"""
    try:
        check_service_availability(chart_svc=True)

        return await services.chart_service.get_performance_chart(
            period=period, metric=metric
        )
    except Exception as e:
        logger.exception("Error getting performance chart")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/charts/indicators/{symbol}")
async def get_indicators_chart(
    symbol: str,
    indicators: str,  # comma-separated list
    interval: str = "1m",
    limit: int | None = None,
):
    """Get technical indicators chart"""
    try:
        # Adapter automatiquement la limite selon le timeframe si non spécifiée
        if limit is None:
            timeframe_limits = {
                "1m": 2880,  # 48 heures de données pour le dézoom
                "3m": 960,  # 48 heures de données
                "5m": 2016,  # 7 jours de données
                "15m": 1344,  # 14 jours de données
                "1h": 720,  # 30 jours de données
                "1d": 365,  # 1 an de données
            }
            limit = timeframe_limits.get(interval, 2880)

        indicator_list = indicators.split(",")
        check_service_availability(chart_svc=True)

        return await services.chart_service.get_indicators_chart(
            symbol=symbol, indicators=indicator_list, interval=interval, limit=limit
        )
    except Exception as e:
        logger.exception("Error getting indicators chart")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.websocket("/ws/charts/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time chart updates"""
    if services.websocket_hub is None:
        await websocket.close(code=1011, reason="WebSocket service not available")
        return

    await services.websocket_hub.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_json()

            if data.get("action") == "subscribe":
                if services.websocket_hub is not None:
                    await services.websocket_hub.subscribe_client(
                        client_id, data.get("channel"), data.get("params", {})
                    )
            elif (
                data.get("action") == "unsubscribe"
                and services.websocket_hub is not None
            ):
                await services.websocket_hub.unsubscribe_client(
                    client_id, data.get("channel")
                )

    except WebSocketDisconnect:
        if services.websocket_hub is not None:
            await services.websocket_hub.disconnect(client_id)
    except Exception:
        logger.exception("WebSocket error")
        if services.websocket_hub is not None:
            await services.websocket_hub.disconnect(client_id)


@app.get("/api/available-symbols")
async def get_available_symbols():
    """Get list of available trading symbols"""
    try:
        symbols = await services.data_manager.get_available_symbols()
    except Exception as e:
        logger.exception("Error getting available symbols")
        raise HTTPException(status_code=500, detail=str(e)) from e
    else:
        return {"symbols": symbols}


@app.get("/api/configured-symbols")
async def get_configured_symbols():
    """Get list of configured trading symbols from shared config"""
    try:
        # Vérifier que SYMBOLS est accessible et valide
        if not SYMBOLS:
            raise ValueError("SYMBOLS configuration is empty")
    except Exception:
        logger.exception("Error getting configured symbols")
        # Fallback avec symboles par défaut en cas d'erreur
        default_symbols = [
            "BTCUSDC",
            "ETHUSDC",
            "SOLUSDC",
            "XRPUSDC",
            "ADAUSDC",
            "AVAXUSDC",
            "LINKUSDC",
            "AAVEUSDC",
            "SUIUSDC",
            "LDOUSDC",
        ]
        logger.info(f"Using fallback symbols: {default_symbols}")
        return {"symbols": default_symbols}
    else:
        return {"symbols": SYMBOLS}


@app.get("/api/trading-opportunities/{symbol}")
async def get_trading_opportunity(symbol: str):
    """Get manual trading opportunity analysis for a specific symbol"""
    try:
        check_service_availability(data_mgr=True, opp_calc=True)

        async with services.data_manager.postgres_pool.acquire() as conn:
            # Récupérer données de la dernière heure (scalping rapide sur 1m)
            signals_query = """
                SELECT COUNT(*) as count, AVG(confidence) as avg_conf
                FROM trading_signals
                WHERE symbol = $1
                AND side = 'BUY'
                AND created_at > NOW() - INTERVAL '1 hour'
            """
            signals_data = await conn.fetchrow(signals_query, symbol)

            # Récupérer TOUTES les données techniques pertinentes (1m pour
            # signal, 5m pour contexte)
            analyzer_query = """
                SELECT
                    -- Data Quality & Performance
                    data_quality, anomaly_detected, cache_hit_ratio,

                    -- Regime & Trend
                    market_regime, regime_confidence, regime_strength, trend_alignment,

                    -- ADX & Directional Movement
                    adx_14, plus_di, minus_di, trend_strength, directional_bias,

                    -- RSI & Oscillators
                    rsi_14, rsi_21, williams_r, cci_20, momentum_10, roc_10,

                    -- Stochastic
                    stoch_k, stoch_d, stoch_rsi, stoch_divergence, stoch_signal,

                    -- MACD
                    macd_line, macd_signal, macd_histogram, ppo, macd_trend, macd_signal_cross,

                    -- Bollinger & Keltner
                    bb_position, bb_width, bb_squeeze, bb_expansion, bb_breakout_direction,
                    bb_lower,

                    -- Money Flow
                    mfi_14,

                    -- Volume Analysis
                    volume_ratio, volume_context, volume_quality_score, volume_pattern,
                    relative_volume, volume_spike_multiplier, trade_intensity, volume_buildup_periods,

                    -- OBV & Accumulation
                    obv, obv_oscillator, ad_line,

                    -- VWAP
                    vwap_10, vwap_quote_10, vwap_upper_band, vwap_lower_band,

                    -- Support/Resistance
                    nearest_support, nearest_resistance, support_strength, resistance_strength,
                    break_probability, pivot_count,

                    -- Pattern & Signal
                    pattern_detected, pattern_confidence, signal_strength, confluence_score,

                    -- Volatility
                    atr_14, natr, volatility_regime, atr_percentile,

                    -- Moving Averages
                    ema_7, ema_12, ema_26, ema_50, ema_99, sma_20, sma_50,
                    hull_20, kama_14,

                    -- Score composite
                    momentum_score

                FROM analyzer_data
                WHERE symbol = $1
                AND timeframe = $2
                ORDER BY time DESC
                LIMIT 1
            """
            analyzer_data_1m = await conn.fetchrow(analyzer_query, symbol, "1m")
            analyzer_data_5m = await conn.fetchrow(analyzer_query, symbol, "5m")

            # Récupérer prix actuel
            price_query = """
                SELECT close
                FROM market_data
                WHERE symbol = $1
                ORDER BY time DESC
                LIMIT 1
            """
            price_data = await conn.fetchrow(price_query, symbol)

            if not price_data:
                return {
                    "symbol": symbol,
                    "score": 0,
                    "action": "AVOID",
                    "reason": "Pas de données de prix disponibles",
                }

            current_price = float(price_data["close"])

            # Convertir les données DB en dict
            analyzer_dict_1m = dict(analyzer_data_1m) if analyzer_data_1m else {}
            analyzer_dict_5m = dict(analyzer_data_5m) if analyzer_data_5m else None
            signals_dict = dict(signals_data) if signals_data else {}

            # Récupérer les 10 dernières périodes 1m pour early detector
            historical_query = """
                SELECT *
                FROM analyzer_data
                WHERE symbol = $1
                  AND timeframe = '1m'
                ORDER BY time DESC
                LIMIT 10
                OFFSET 1
            """
            historical_rows = await conn.fetch(historical_query, symbol)
            historical_data = (
                [dict(row) for row in reversed(historical_rows)]
                if historical_rows
                else None
            )

            # Fallback: Si nearest_resistance/support manquant, récupérer
            # dernière valeur connue
            if (
                analyzer_dict_1m.get("nearest_resistance") is None
                or analyzer_dict_1m.get("nearest_support") is None
            ):
                fallback_query = """
                    SELECT nearest_resistance, nearest_support
                    FROM analyzer_data
                    WHERE symbol = $1
                    AND timeframe = '1m'
                    AND nearest_resistance IS NOT NULL
                    AND nearest_support IS NOT NULL
                    ORDER BY time DESC
                    LIMIT 1
                """
                fallback_data = await conn.fetchrow(fallback_query, symbol)
                if fallback_data:
                    if analyzer_dict_1m.get("nearest_resistance") is None:
                        analyzer_dict_1m["nearest_resistance"] = float(
                            fallback_data["nearest_resistance"]
                        )
                    if analyzer_dict_1m.get("nearest_support") is None:
                        analyzer_dict_1m["nearest_support"] = float(
                            fallback_data["nearest_support"]
                        )

            # Calculer l'opportunité via le calculateur PRO (avec contexte 5m +
            # early detection)
            opportunity = services.opportunity_calculator.calculate_opportunity(
                symbol=symbol,
                current_price=current_price,
                analyzer_data=analyzer_dict_1m,
                higher_tf_data=analyzer_dict_5m,
                signals_data=signals_dict,
                historical_data=historical_data,  # NOUVEAU: Pour early detector
            )

            # Convertir en dict pour l'API
            response_data = services.opportunity_calculator.to_dict(opportunity)

            # Envoyer notification Telegram pour BUY_NOW, BUY_DCA et
            # EARLY_ENTRY
            if opportunity.action in ["BUY_NOW", "BUY_DCA", "EARLY_ENTRY"]:
                try:
                    # Créer une connexion psycopg2 synchrone pour
                    # TelegramNotifier
                    db_config = {
                        "host": os.getenv("DB_HOST", "db"),
                        "port": int(os.getenv("DB_PORT", "5432")),
                        "database": os.getenv("DB_NAME", "trading"),
                        "user": os.getenv("DB_USER", "postgres"),
                        "password": os.getenv("DB_PASSWORD", "postgres"),
                    }
                    sync_conn = psycopg2.connect(**db_config)

                    notifier = get_notifier(db_connection=sync_conn)

                    # Score PRO (0-100 direct)
                    score = opportunity.score.total_score

                    # Préparer early_signal pour Telegram (dict serializable)
                    early_signal_dict = None
                    if opportunity.early_signal and opportunity.is_early_entry:
                        early_signal_dict = {
                            "level": opportunity.early_signal.level.value,
                            "score": opportunity.early_signal.score,
                            "estimated_entry_window_seconds": opportunity.early_signal.estimated_entry_window_seconds,
                            "velocity_score": opportunity.early_signal.velocity_score,
                            "volume_buildup_score": opportunity.early_signal.volume_buildup_score,
                        }

                    success = notifier.send_buy_signal(
                        symbol=symbol,
                        score=score,
                        price=current_price,
                        action=opportunity.action,
                        targets={
                            "tp1": opportunity.tp1,
                            "tp2": opportunity.tp2,
                            "tp3": opportunity.tp3 if opportunity.tp3 else None,
                        },
                        stop_loss=opportunity.stop_loss,
                        reason="\n".join(opportunity.reasons),
                        momentum=analyzer_dict_1m.get("adx_14", 0),
                        volume_ratio=analyzer_dict_1m.get("relative_volume", 1.0),
                        regime=opportunity.market_regime,
                        estimated_hold_time=opportunity.estimated_hold_time,
                        grade=opportunity.score.grade,
                        rr_ratio=opportunity.rr_ratio,
                        risk_level=opportunity.risk_level,
                        early_signal=early_signal_dict,  # NOUVEAU
                    )

                    # Fermer la connexion synchrone
                    sync_conn.close()

                    if success:
                        logger.info(
                            f"✅ Signal Telegram envoyé et stocké pour {symbol}"
                        )
                    else:
                        logger.warning(
                            f"⚠️ Signal Telegram non envoyé pour {symbol} (cooldown ou erreur)"
                        )

                except Exception as e:
                    logger.error(
                        f"❌ Erreur envoi notification Telegram pour {symbol}: {e}",
                        exc_info=True,
                    )

            return response_data

    except Exception as e:
        logger.exception("Error getting trading opportunity for {symbol}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/automatic-signals/{symbol}")
async def get_automatic_signals(symbol: str):
    """
    Get automatic trading signals from signal_aggregator for comparison.
    Returns the most recent validated signal and aggregated stats.
    """
    try:
        check_service_availability(data_mgr=True)

        async with services.data_manager.postgres_pool.acquire() as conn:
            # Récupérer les signaux validés récents (dernières 15 minutes)
            signals_query = """
                SELECT
                    symbol,
                    side,
                    confidence,
                    strategy,
                    metadata,
                    created_at
                FROM trading_signals
                WHERE symbol = $1
                  AND created_at > NOW() - INTERVAL '15 minutes'
                ORDER BY created_at DESC
            """
            signals = await conn.fetch(signals_query, symbol)

            if not signals:
                return {
                    "symbol": symbol,
                    "has_signal": False,
                    "validated": False,
                    "message": "Aucun signal validé dans les 15 dernières minutes",
                }

            # Compter les stratégies uniques et calculer consensus
            strategies = set()
            total_confidence = 0
            buy_count = 0
            sell_count = 0
            consensus_strength = 0
            metadata_latest = {}

            for signal in signals:
                strategies.add(signal["strategy"])
                total_confidence += float(signal["confidence"])

                if signal["side"] == "BUY":
                    buy_count += 1
                else:
                    sell_count += 1

                # Récupérer métadonnées du signal le plus récent
                if signal == signals[0] and signal["metadata"]:
                    # Parser JSON si c'est une string, sinon utiliser
                    # directement
                    try:
                        if isinstance(signal["metadata"], str):
                            metadata_latest = json.loads(signal["metadata"])
                        else:
                            metadata_latest = signal["metadata"]
                        consensus_strength = metadata_latest.get(
                            "consensus_strength", 0
                        )
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        logger.warning(
                            f"Could not parse metadata for {symbol}: {signal['metadata']}"
                        )
                        metadata_latest = {}

            avg_confidence = total_confidence / len(signals) if signals else 0
            strategies_count = len(strategies)

            # Déterminer le side dominant
            dominant_side = (
                "BUY"
                if buy_count > sell_count
                else "SELL" if sell_count > buy_count else "NEUTRAL"
            )

            # Status validé si consensus fort (au moins 3 stratégies)
            validated = strategies_count >= 3 and abs(buy_count - sell_count) >= 2

            return {
                "symbol": symbol,
                "has_signal": True,
                "validated": validated,
                "side": dominant_side,
                "confidence": round(avg_confidence, 2),
                "consensus_strength": (
                    round(consensus_strength, 2) if consensus_strength else None
                ),
                "strategies_count": strategies_count,
                "strategies": list(strategies),
                "signals_count": len(signals),
                "buy_signals": buy_count,
                "sell_signals": sell_count,
                "last_signal_time": (
                    signals[0]["created_at"].isoformat() if signals else None
                ),
                "metadata": metadata_latest,
                "rejection_reason": (
                    None if validated else "Consensus faible ou signaux contradictoires"
                ),
            }

    except Exception as e:
        logger.exception("Error getting automatic signals for {symbol}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/available-indicators")
async def get_available_indicators():
    """Get list of available technical indicators"""
    return {
        "indicators": [
            "sma",
            "ema",
            "rsi",
            "macd",
            "bollinger_bands",
            "volume",
            "atr",
            "stochastic",
            "adx",
            "obv",
        ]
    }


# ============================================================================
# Routes API Statistiques
# ============================================================================


@app.get("/api/statistics/global")
async def get_global_statistics():
    """Get global trading statistics across all symbols and strategies"""
    try:
        check_service_availability(stats_svc=True)

        return await services.statistics_service.get_global_statistics()
    except Exception as e:
        logger.exception("Error getting global statistics")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/statistics/symbols")
async def get_all_symbols_statistics():
    """Get detailed statistics for all trading symbols"""
    try:
        check_service_availability(stats_svc=True)

        return await services.statistics_service.get_all_symbols_statistics()
    except Exception as e:
        logger.exception("Error getting all symbols statistics")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/statistics/symbol/{symbol}")
async def get_symbol_statistics(symbol: str):
    """Get detailed statistics for a specific trading symbol"""
    try:
        check_service_availability(stats_svc=True)

        return await services.statistics_service.get_symbol_statistics(symbol)
    except Exception as e:
        logger.exception("Error getting symbol statistics")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/statistics/performance-history")
async def get_performance_history(period: str = "7d", interval: str = "1h"):
    """Get historical performance data with configurable period and interval"""
    try:
        check_service_availability(stats_svc=True)
        validate_parameters(period=period, interval=interval)

        return await services.statistics_service.get_performance_history(
            period, interval
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting performance history")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/statistics/strategies")
async def get_strategy_comparison():
    """Compare performance metrics across different trading strategies"""
    try:
        check_service_availability(stats_svc=True)

        return await services.statistics_service.get_strategy_comparison()
    except Exception as e:
        logger.exception("Error getting strategy comparison")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/ema-sentiment")
async def get_ema_sentiment(timeframe: str = "1m", limit: int = 100):
    """Get EMA alignment sentiment across all symbols"""
    try:
        check_service_availability(data_mgr=True)
        active_symbols = check_trading_symbols_config()

        async with services.data_manager.postgres_pool.acquire() as conn:
            # Récupérer uniquement les symboles actifs
            query = """
                WITH latest_data AS (
                    SELECT DISTINCT ON (symbol)
                        symbol,
                        ema_7,
                        ema_26,
                        ema_99,
                        time
                    FROM analyzer_data
                    WHERE timeframe = $1
                      AND symbol = ANY($3)
                      AND ema_7 IS NOT NULL
                      AND ema_26 IS NOT NULL
                      AND ema_99 IS NOT NULL
                    ORDER BY symbol, time DESC
                ),
                current_prices AS (
                    SELECT DISTINCT ON (symbol)
                        symbol,
                        close as price
                    FROM market_data
                    WHERE timeframe = $1
                      AND symbol = ANY($3)
                    ORDER BY symbol, time DESC
                )
                SELECT
                    l.symbol,
                    l.ema_7,
                    l.ema_26,
                    l.ema_99,
                    p.price,
                    l.time
                FROM latest_data l
                JOIN current_prices p ON l.symbol = p.symbol
                LIMIT $2
            """

            rows = await conn.fetch(query, timeframe, limit, active_symbols)

            bullish_count = 0
            bearish_count = 0
            neutral_count = 0
            mixed_count = 0

            symbols_detail = []

            for row in rows:
                symbol = row["symbol"]
                price = float(row["price"])
                ema7 = float(row["ema_7"])
                ema26 = float(row["ema_26"])
                ema99 = float(row["ema_99"])

                # Déterminer l'alignement EMA
                # BULLISH: Prix > EMA7 > EMA26 > EMA99
                # BEARISH: Prix < EMA7 < EMA26 < EMA99
                # MIXED: Alignement partiel ou croisements

                if price > ema7 > ema26 > ema99:
                    sentiment = "BULLISH"
                    bullish_count += 1
                    score = 100  # Alignement parfait
                elif price < ema7 < ema26 < ema99:
                    sentiment = "BEARISH"
                    bearish_count += 1
                    score = 0  # Alignement baissier parfait
                else:
                    # Calculer un score basé sur les croisements partiels
                    alignment_score = 0
                    if price > ema7:
                        alignment_score += 25
                    if ema7 > ema26:
                        alignment_score += 25
                    if ema26 > ema99:
                        alignment_score += 25
                    if price > ema99:
                        alignment_score += 25

                    score = alignment_score

                    if 40 <= score <= 60:
                        sentiment = "NEUTRAL"
                        neutral_count += 1
                    else:
                        sentiment = "MIXED"
                        mixed_count += 1

                # Distance du prix par rapport aux EMAs (en %)
                distance_ema7 = ((price - ema7) / ema7) * 100
                distance_ema26 = ((price - ema26) / ema26) * 100
                distance_ema99 = ((price - ema99) / ema99) * 100

                symbols_detail.append(
                    {
                        "symbol": symbol,
                        "sentiment": sentiment,
                        "score": score,
                        "price": price,
                        "ema_7": ema7,
                        "ema_26": ema26,
                        "ema_99": ema99,
                        "distance_ema7_percent": round(distance_ema7, 2),
                        "distance_ema26_percent": round(distance_ema26, 2),
                        "distance_ema99_percent": round(distance_ema99, 2),
                        "timestamp": row["time"].isoformat(),
                    }
                )

            # Calculer le sentiment global
            total_symbols = len(symbols_detail)
            if total_symbols == 0:
                return {
                    "sentiment": "NEUTRAL",
                    "bullish_count": 0,
                    "bearish_count": 0,
                    "neutral_count": 0,
                    "mixed_count": 0,
                    "total_symbols": 0,
                    "bullish_percent": 0,
                    "bearish_percent": 0,
                    "symbols": [],
                    "timeframe": timeframe,
                }

            bullish_percent = (bullish_count / total_symbols) * 100
            bearish_percent = (bearish_count / total_symbols) * 100

            # Déterminer le sentiment global
            if bullish_percent >= 60:
                global_sentiment = "BULLISH"
            elif bearish_percent >= 60:
                global_sentiment = "BEARISH"
            elif abs(bullish_percent - bearish_percent) <= 20:
                global_sentiment = "NEUTRAL"
            else:
                global_sentiment = "MIXED"

            # Trier par score (meilleurs alignements en premier)
            symbols_detail.sort(key=lambda x: x["score"], reverse=True)

            return {
                "sentiment": global_sentiment,
                "bullish_count": bullish_count,
                "bearish_count": bearish_count,
                "neutral_count": neutral_count,
                "mixed_count": mixed_count,
                "total_symbols": total_symbols,
                "bullish_percent": round(bullish_percent, 1),
                "bearish_percent": round(bearish_percent, 1),
                "top_bullish": [
                    s for s in symbols_detail if s["sentiment"] == "BULLISH"
                ][:5],
                "top_bearish": [
                    s for s in symbols_detail if s["sentiment"] == "BEARISH"
                ][:5],
                "symbols": symbols_detail,
                "timeframe": timeframe,
            }

    except Exception as e:
        logger.exception("Error getting EMA sentiment")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/top-signals")
async def get_top_signals(
    # MODIFIÉ: 180min (3h) au lieu de 15min pour capter tous signaux actifs
    timeframe_minutes: int = 180,
    limit: int = 50,
):
    """Get all trading signals ranked by net signal strength (BUY - SELL)

    Note: timeframe_minutes=180 (3h) pour capter tous les signaux actifs depuis plusieurs heures
    """
    try:
        check_service_availability(data_mgr=True)

        async with services.data_manager.postgres_pool.acquire() as conn:
            query = """
                WITH signal_counts AS (
                    SELECT
                        symbol,
                        side,
                        COUNT(*) as consensus_count,
                        AVG(confidence) as avg_confidence,
                        MAX(timestamp) as last_signal_time
                    FROM trading_signals
                    WHERE timestamp > NOW() - INTERVAL '1 minute' * $1
                      AND strategy = 'CONSENSUS'
                    GROUP BY symbol, side
                ),
                buy_signals AS (
                    SELECT
                        symbol,
                        consensus_count as buy_count,
                        avg_confidence as buy_confidence,
                        last_signal_time as buy_time
                    FROM signal_counts WHERE side = 'BUY'
                ),
                sell_signals AS (
                    SELECT
                        symbol,
                        consensus_count as sell_count,
                        avg_confidence as sell_confidence,
                        last_signal_time as sell_time
                    FROM signal_counts WHERE side = 'SELL'
                )
                SELECT
                    COALESCE(b.symbol, s.symbol) as symbol,
                    COALESCE(b.buy_count, 0) as buy_count,
                    COALESCE(s.sell_count, 0) as sell_count,
                    COALESCE(b.buy_confidence, 0) as buy_confidence,
                    COALESCE(s.sell_confidence, 0) as sell_confidence,
                    GREATEST(COALESCE(b.buy_time, '1970-01-01'), COALESCE(s.sell_time, '1970-01-01')) as last_signal_time,
                    (COALESCE(b.buy_count, 0) - COALESCE(s.sell_count, 0)) as net_signal
                FROM buy_signals b
                FULL OUTER JOIN sell_signals s ON b.symbol = s.symbol
                ORDER BY net_signal DESC
                LIMIT $2
            """

            rows = await conn.fetch(query, timeframe_minutes, limit)

            top_signals = []
            for row in rows:
                net = row["net_signal"]
                dominant_side = "BUY" if net > 0 else ("SELL" if net < 0 else "NEUTRAL")

                top_signals.append(
                    {
                        "symbol": row["symbol"],
                        "buy_count": row["buy_count"],
                        "sell_count": row["sell_count"],
                        "buy_confidence": float(row["buy_confidence"]),
                        "sell_confidence": float(row["sell_confidence"]),
                        "net_signal": net,
                        "dominant_side": dominant_side,
                        "last_signal_time": (
                            row["last_signal_time"].isoformat()
                            if row["last_signal_time"]
                            else None
                        ),
                    }
                )

            return {"signals": top_signals, "timeframe_minutes": timeframe_minutes}
    except Exception as e:
        logger.exception("Error getting top signals")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/trade-cycles")
async def get_trade_cycles(
    symbol: str | None = None, status: str | None = None, limit: int = 100
):
    """Get trade cycles from database"""
    try:
        check_service_availability(data_mgr=True)

        cycles = await services.data_manager.get_trade_cycles(
            symbol=symbol, status=status, limit=limit
        )
    except Exception as e:
        logger.exception("Error getting trade cycles")
        raise HTTPException(status_code=500, detail=str(e)) from e
    else:
        return {"cycles": cycles}


# ============================================================================
# Routes Proxy pour les autres services
# ============================================================================


@app.get("/api/portfolio/{path:path}")
async def proxy_portfolio(path: str, request: Request):
    """Proxy vers le service portfolio"""
    # Construire l'URL complète avec les query parameters
    query_string = str(request.url.query)
    portfolio_url = f"http://portfolio:8000/{path}"
    if query_string:
        portfolio_url += f"?{query_string}"

    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(
                portfolio_url, timeout=aiohttp.ClientTimeout(total=10)
            ) as response,
        ):
            if response.content_type == "application/json":
                return await response.json()
            text = await response.text()
            return {"data": text}
    except Exception as e:
        logger.exception("Error proxying to portfolio service")
        raise HTTPException(
            status_code=503, detail=f"Portfolio service unavailable: {e!s}"
        ) from e


@app.get("/api/trader/{path:path}")
async def proxy_trader(path: str, request: Request):
    """Proxy vers le service trader"""
    # Construire l'URL complète avec les query parameters
    query_string = str(request.url.query)
    trader_url = f"http://trader:5002/{path}"
    if query_string:
        trader_url += f"?{query_string}"

    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get(
                trader_url, timeout=aiohttp.ClientTimeout(total=10)
            ) as response,
        ):
            if response.content_type == "application/json":
                return await response.json()
            text = await response.text()
            return {"data": text}
    except Exception as e:
        logger.exception("Error proxying to trader service")
        raise HTTPException(
            status_code=503, detail=f"Trader service unavailable: {e!s}"
        ) from e


@app.post("/api/trader/{path:path}")
async def proxy_trader_post(path: str, request: Request):
    """Proxy POST vers le service trader"""
    # Récupérer le body de la requête
    body = await request.body()
    headers = dict(request.headers)

    # Construire l'URL complète
    trader_url = f"http://trader:5002/{path}"

    try:
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                trader_url,
                data=body,
                headers={
                    k: v
                    for k, v in headers.items()
                    if k.lower() not in ["host", "content-length"]
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response,
        ):
            if response.content_type == "application/json":
                return await response.json()
            text = await response.text()
            return {"data": text}
    except Exception as e:
        logger.exception("Error proxying POST to trader service")
        raise HTTPException(
            status_code=503, detail=f"Trader service unavailable: {e!s}"
        ) from e


@app.post("/api/portfolio/{path:path}")
async def proxy_portfolio_post(path: str, request: Request):
    """Proxy POST vers le service portfolio"""
    # Récupérer le body de la requête
    body = await request.body()
    headers = dict(request.headers)

    # Construire l'URL complète
    portfolio_url = f"http://portfolio:8000/{path}"

    try:
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                portfolio_url,
                data=body,
                headers={
                    k: v
                    for k, v in headers.items()
                    if k.lower() not in ["host", "content-length"]
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response,
        ):
            if response.content_type == "application/json":
                return await response.json()
            text = await response.text()
            return {"data": text}
    except Exception as e:
        logger.exception("Error proxying POST to portfolio service")
        raise HTTPException(
            status_code=503, detail=f"Portfolio service unavailable: {e!s}"
        ) from e


# Routes React (à la fin pour ne pas intercepter les routes API)
if serve_react_app_flag:

    @app.get("/", response_class=HTMLResponse)
    async def serve_react_app(_request: Request):
        """Serve React app"""
        index_path = Path(frontend_build_path) / "index.html"
        return HTMLResponse(content=index_path.read_text())

    @app.get("/{path:path}", response_class=HTMLResponse)
    async def serve_react_app_routes(_request: Request, path: str):
        """Serve React app for all routes (SPA routing)"""
        # Skip API routes and WebSocket routes
        if path.startswith(("api/", "ws/", "health")):
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="Not found")
        index_path = Path(frontend_build_path) / "index.html"
        return HTMLResponse(content=index_path.read_text())


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("VISUALIZATION_PORT", "5009"))
    uvicorn.run(app, host="0.0.0.0", port=port)
