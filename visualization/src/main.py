from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from typing import Optional
from datetime import datetime
import asyncio
import logging
import os
import sys

from data_manager import DataManager
from chart_service import ChartService
from websocket_hub import WebSocketHub
from statistics_service import StatisticsService
from opportunity_calculator import OpportunityCalculator

# Ajouter le path pour accéder au module notifications
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data_manager: Optional[DataManager] = None
chart_service: Optional[ChartService] = None
websocket_hub: Optional[WebSocketHub] = None
statistics_service: Optional[StatisticsService] = None
opportunity_calculator: Optional[OpportunityCalculator] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global data_manager, chart_service, websocket_hub, statistics_service, opportunity_calculator

    logger.info("Starting visualization service...")

    data_manager = DataManager()
    await data_manager.initialize()

    chart_service = ChartService(data_manager)
    websocket_hub = WebSocketHub(data_manager)
    statistics_service = StatisticsService(data_manager)
    opportunity_calculator = OpportunityCalculator()

    asyncio.create_task(websocket_hub.start())

    yield

    logger.info("Shutting down visualization service...")
    await websocket_hub.stop()
    await data_manager.close()

app = FastAPI(
    title="RootTrading Visualization Service",
    version="1.0.0",
    lifespan=lifespan
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
serve_react_app_flag = os.path.exists(frontend_build_path)

if serve_react_app_flag:
    # Serve static assets first
    app.mount("/assets", StaticFiles(directory=f"{frontend_build_path}/assets"), name="assets")
    logger.info(f"Serving React app from {frontend_build_path}")
else:
    logger.warning(f"Frontend build path {frontend_build_path} not found, serving legacy template")
    
    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        """Interface web principale (legacy)"""
        return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + 'Z',
        "redis_connected": data_manager.is_redis_connected() if data_manager else False,
        "postgres_connected": data_manager.is_postgres_connected() if data_manager else False
    }

@app.get("/api/system/alerts")
async def get_system_alerts():
    """Get system health alerts from all services"""
    import aiohttp
    
    async def fetch_service_health(url: str, service_name: str):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data
                    else:
                        return {"status": "error", "service": service_name, "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"status": "offline", "service": service_name, "error": str(e)}
    
    # Appels vers les services Docker
    portfolio_health, trader_health = await asyncio.gather(
        fetch_service_health("http://portfolio:8000/health", "portfolio"),
        fetch_service_health("http://trader:5002/health", "trader"),
        return_exceptions=True
    )
    
    # Gérer les exceptions
    if isinstance(portfolio_health, Exception):
        portfolio_health = {"status": "offline", "service": "portfolio", "error": str(portfolio_health)}
    if isinstance(trader_health, Exception):
        trader_health = {"status": "offline", "service": "trader", "error": str(trader_health)}
    
    return {
        "portfolio": portfolio_health,
        "trader": trader_health,
        "timestamp": datetime.utcnow().isoformat() + 'Z'
    }

@app.get("/api/charts/market/{symbol}")
async def get_market_chart(
    symbol: str,
    interval: str = "1m",
    limit: Optional[int] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
):
    """Get market data chart for a specific symbol"""
    try:
        # Adapter automatiquement la limite selon le timeframe si non spécifiée
        if limit is None:
            timeframe_limits = {
                "1m": 2880,    # 48 heures de données pour le dézoom
                "3m": 960,     # 48 heures de données
                "5m": 2016,    # 7 jours de données
                "15m": 1344,   # 14 jours de données  
                "1h": 720,     # 30 jours de données
                "1d": 365      # 1 an de données
            }
            limit = timeframe_limits.get(interval, 2880)
        
        if chart_service is None:
            raise HTTPException(status_code=503, detail="Chart service not available")
            
        data = await chart_service.get_market_chart(
            symbol=symbol,
            interval=interval,
            limit=limit,
            start_time=start_time,
            end_time=end_time
        )
        return data
    except Exception as e:
        logger.error(f"Error getting market chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/charts/signals/{symbol}")
async def get_signals_chart(
    symbol: str,
    strategy: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
):
    """Get trading signals overlaid on price chart"""
    try:
        if chart_service is None:
            raise HTTPException(status_code=503, detail="Chart service not available")
            
        data = await chart_service.get_signals_chart(
            symbol=symbol,
            strategy=strategy,
            start_time=start_time,
            end_time=end_time
        )
        return data
    except Exception as e:
        logger.error(f"Error getting signals chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/charts/performance")
async def get_performance_chart(
    period: str = "24h",
    metric: str = "pnl"
):
    """Get portfolio performance chart"""
    try:
        if chart_service is None:
            raise HTTPException(status_code=503, detail="Chart service not available")
            
        data = await chart_service.get_performance_chart(
            period=period,
            metric=metric
        )
        return data
    except Exception as e:
        logger.error(f"Error getting performance chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/charts/indicators/{symbol}")
async def get_indicators_chart(
    symbol: str,
    indicators: str,  # comma-separated list
    interval: str = "1m",
    limit: Optional[int] = None
):
    """Get technical indicators chart"""
    try:
        # Adapter automatiquement la limite selon le timeframe si non spécifiée
        if limit is None:
            timeframe_limits = {
                "1m": 2880,    # 48 heures de données pour le dézoom
                "3m": 960,     # 48 heures de données
                "5m": 2016,    # 7 jours de données
                "15m": 1344,   # 14 jours de données  
                "1h": 720,     # 30 jours de données
                "1d": 365      # 1 an de données
            }
            limit = timeframe_limits.get(interval, 2880)
            
        indicator_list = indicators.split(",")
        if chart_service is None:
            raise HTTPException(status_code=503, detail="Chart service not available")
            
        data = await chart_service.get_indicators_chart(
            symbol=symbol,
            indicators=indicator_list,
            interval=interval,
            limit=limit
        )
        return data
    except Exception as e:
        logger.error(f"Error getting indicators chart: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/charts/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time chart updates"""
    if websocket_hub is None:
        await websocket.close(code=1011, reason="WebSocket service not available")
        return
        
    await websocket_hub.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("action") == "subscribe":
                if websocket_hub is not None:
                    await websocket_hub.subscribe_client(
                        client_id,
                        data.get("channel"),
                        data.get("params", {})
                    )
            elif data.get("action") == "unsubscribe":
                if websocket_hub is not None:
                    await websocket_hub.unsubscribe_client(
                        client_id,
                        data.get("channel")
                    )
                
    except WebSocketDisconnect:
        if websocket_hub is not None:
            await websocket_hub.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket_hub is not None:
            await websocket_hub.disconnect(client_id)

@app.get("/api/available-symbols")
async def get_available_symbols():
    """Get list of available trading symbols"""
    try:
        symbols = await data_manager.get_available_symbols()
        return {"symbols": symbols}
    except Exception as e:
        logger.error(f"Error getting available symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/configured-symbols")
async def get_configured_symbols():
    """Get list of configured trading symbols from shared config"""
    try:
        # Import standard depuis shared.src.config
        from shared.src.config import SYMBOLS
        return {"symbols": SYMBOLS}

    except Exception as e:
        logger.error(f"Error getting configured symbols: {e}")
        # Fallback avec symboles par défaut en cas d'erreur
        default_symbols = [
            "BTCUSDC", "ETHUSDC", "SOLUSDC", "XRPUSDC", "ADAUSDC",
            "AVAXUSDC", "LINKUSDC", "AAVEUSDC", "SUIUSDC", "LDOUSDC"
        ]
        logger.info(f"Using fallback symbols: {default_symbols}")
        return {"symbols": default_symbols}

@app.get("/api/trading-opportunities/{symbol}")
async def get_trading_opportunity(symbol: str):
    """Get manual trading opportunity analysis for a specific symbol"""
    try:
        if data_manager is None or not data_manager.postgres_pool:
            raise HTTPException(status_code=503, detail="Data manager not available")

        if opportunity_calculator is None:
            raise HTTPException(status_code=503, detail="Opportunity calculator not available")

        async with data_manager.postgres_pool.acquire() as conn:
            # Récupérer données de la dernière heure (scalping rapide sur 1m)
            signals_query = """
                SELECT COUNT(*) as count, AVG(confidence) as avg_conf
                FROM trading_signals
                WHERE symbol = $1
                AND side = 'BUY'
                AND created_at > NOW() - INTERVAL '1 hour'
            """
            signals_data = await conn.fetchrow(signals_query, symbol)

            # Récupérer TOUTES les données techniques pertinentes
            analyzer_query = """
                SELECT
                    -- Regime & Trend
                    market_regime, regime_confidence, regime_strength, trend_alignment,

                    -- ADX & Directional Movement
                    adx_14, plus_di, minus_di, trend_strength, directional_bias,

                    -- RSI & Oscillators
                    rsi_14, rsi_21, williams_r, cci_20, momentum_10, roc_10,

                    -- Stochastic
                    stoch_k, stoch_d, stoch_rsi, stoch_divergence, stoch_signal,

                    -- MACD
                    macd_line, macd_signal, macd_histogram, ppo, macd_trend,

                    -- Bollinger & Keltner
                    bb_position, bb_width, bb_squeeze, bb_expansion, bb_breakout_direction,

                    -- Money Flow
                    mfi_14,

                    -- Volume Analysis
                    volume_ratio, volume_context, volume_quality_score, volume_pattern,
                    relative_volume, volume_spike_multiplier, trade_intensity,

                    -- OBV & Accumulation
                    obv, obv_oscillator, ad_line,

                    -- VWAP
                    vwap_10, vwap_upper_band, vwap_lower_band,

                    -- Support/Resistance
                    nearest_support, nearest_resistance, support_strength, resistance_strength,
                    break_probability,

                    -- Pattern & Signal
                    pattern_detected, pattern_confidence, signal_strength, confluence_score,

                    -- Volatility
                    atr_14, volatility_regime, atr_percentile,

                    -- Moving Averages
                    ema_7, ema_12, ema_26, ema_50, ema_99, sma_20, sma_50,
                    hull_20, kama_14,

                    -- Score composite
                    momentum_score

                FROM analyzer_data
                WHERE symbol = $1
                AND timeframe = '1m'
                ORDER BY time DESC
                LIMIT 1
            """
            analyzer_data = await conn.fetchrow(analyzer_query, symbol)

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
                    "reason": "Pas de données de prix disponibles"
                }

            current_price = float(price_data['close'])

            # Convertir les données DB en dict
            analyzer_dict = dict(analyzer_data) if analyzer_data else {}
            signals_dict = dict(signals_data) if signals_data else {}

            # Calculer l'opportunité via le calculateur
            response_data = opportunity_calculator.calculate_opportunity(
                symbol=symbol,
                current_price=current_price,
                analyzer_data=analyzer_dict,
                signals_data=signals_dict
            )

            # Envoyer notification Telegram uniquement pour les signaux BUY
            if response_data.get("action") == "BUY_NOW":
                try:
                    from notifications.telegram_service import get_notifier
                    notifier = get_notifier()

                    # Récupérer le vrai score momentum depuis score_details
                    score_details = response_data.get("score_details", {})
                    momentum_score = score_details.get("momentum", 0)

                    notifier.send_buy_signal(
                        symbol=symbol,
                        score=response_data["score"],
                        price=current_price,
                        action=response_data["action"],
                        targets=response_data["targets"],
                        stop_loss=response_data["stop_loss"],
                        reason=response_data["reason"],
                        momentum=momentum_score,  # Vrai score momentum (/35)
                        volume_ratio=response_data.get("volume_ratio", 1.0),
                        regime=response_data.get("market_regime", "UNKNOWN"),
                        estimated_hold_time=response_data.get("estimated_hold_time", "N/A")
                    )
                except Exception as e:
                    logger.warning(f"Erreur envoi notification Telegram pour {symbol}: {e}")

            return response_data

    except Exception as e:
        logger.error(f"Error getting trading opportunity for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/available-indicators")
async def get_available_indicators():
    """Get list of available technical indicators"""
    return {
        "indicators": [
            "sma", "ema", "rsi", "macd", "bollinger_bands",
            "volume", "atr", "stochastic", "adx", "obv"
        ]
    }

# ============================================================================
# Routes API Statistiques
# ============================================================================

@app.get("/api/statistics/global")
async def get_global_statistics():
    """Get global trading statistics across all symbols and strategies"""
    try:
        if statistics_service is None:
            raise HTTPException(status_code=503, detail="Statistics service not available")
        
        stats = await statistics_service.get_global_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting global statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/statistics/symbols")
async def get_all_symbols_statistics():
    """Get detailed statistics for all trading symbols"""
    try:
        if statistics_service is None:
            raise HTTPException(status_code=503, detail="Statistics service not available")
        
        stats = await statistics_service.get_all_symbols_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting all symbols statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/statistics/symbol/{symbol}")
async def get_symbol_statistics(symbol: str):
    """Get detailed statistics for a specific trading symbol"""
    try:
        if statistics_service is None:
            raise HTTPException(status_code=503, detail="Statistics service not available")
        
        stats = await statistics_service.get_symbol_statistics(symbol)
        return stats
    except Exception as e:
        logger.error(f"Error getting symbol statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/statistics/performance-history")
async def get_performance_history(
    period: str = "7d",
    interval: str = "1h"
):
    """Get historical performance data with configurable period and interval"""
    try:
        if statistics_service is None:
            raise HTTPException(status_code=503, detail="Statistics service not available")
        
        # Validation des paramètres
        valid_periods = ["1d", "7d", "30d", "90d", "1y"]
        valid_intervals = ["1h", "1d"]
        
        if period not in valid_periods:
            raise HTTPException(status_code=400, detail=f"Invalid period. Must be one of: {valid_periods}")
        if interval not in valid_intervals:
            raise HTTPException(status_code=400, detail=f"Invalid interval. Must be one of: {valid_intervals}")
        
        history = await statistics_service.get_performance_history(period, interval)
        return history
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting performance history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/statistics/strategies")
async def get_strategy_comparison():
    """Compare performance metrics across different trading strategies"""
    try:
        if statistics_service is None:
            raise HTTPException(status_code=503, detail="Statistics service not available")
        
        comparison = await statistics_service.get_strategy_comparison()
        return comparison
    except Exception as e:
        logger.error(f"Error getting strategy comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trade-cycles")
async def get_trade_cycles(
    symbol: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
):
    """Get trade cycles from database"""
    try:
        if data_manager is None:
            raise HTTPException(status_code=503, detail="Data service not available")
        
        cycles = await data_manager.get_trade_cycles(
            symbol=symbol,
            status=status,
            limit=limit
        )
        return {"cycles": cycles}
    except Exception as e:
        logger.error(f"Error getting trade cycles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Routes Proxy pour les autres services
# ============================================================================

@app.get("/api/portfolio/{path:path}")
async def proxy_portfolio(path: str, request: Request):
    """Proxy vers le service portfolio"""
    import aiohttp
    
    # Construire l'URL complète avec les query parameters
    query_string = str(request.url.query)
    portfolio_url = f"http://portfolio:8000/{path}"
    if query_string:
        portfolio_url += f"?{query_string}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(portfolio_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.content_type == 'application/json':
                    data = await response.json()
                    return data
                else:
                    text = await response.text()
                    return {"data": text}
    except Exception as e:
        logger.error(f"Error proxying to portfolio service: {e}")
        raise HTTPException(status_code=503, detail=f"Portfolio service unavailable: {str(e)}")

@app.get("/api/trader/{path:path}")
async def proxy_trader(path: str, request: Request):
    """Proxy vers le service trader"""
    import aiohttp
    
    # Construire l'URL complète avec les query parameters
    query_string = str(request.url.query)
    trader_url = f"http://trader:5002/{path}"
    if query_string:
        trader_url += f"?{query_string}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(trader_url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.content_type == 'application/json':
                    data = await response.json()
                    return data
                else:
                    text = await response.text()
                    return {"data": text}
    except Exception as e:
        logger.error(f"Error proxying to trader service: {e}")
        raise HTTPException(status_code=503, detail=f"Trader service unavailable: {str(e)}")

@app.post("/api/trader/{path:path}")
async def proxy_trader_post(path: str, request: Request):
    """Proxy POST vers le service trader"""
    import aiohttp
    
    # Récupérer le body de la requête
    body = await request.body()
    headers = dict(request.headers)
    
    # Construire l'URL complète
    trader_url = f"http://trader:5002/{path}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                trader_url, 
                data=body,
                headers={k: v for k, v in headers.items() if k.lower() not in ['host', 'content-length']},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.content_type == 'application/json':
                    data = await response.json()
                    return data
                else:
                    text = await response.text()
                    return {"data": text}
    except Exception as e:
        logger.error(f"Error proxying POST to trader service: {e}")
        raise HTTPException(status_code=503, detail=f"Trader service unavailable: {str(e)}")

@app.post("/api/portfolio/{path:path}")
async def proxy_portfolio_post(path: str, request: Request):
    """Proxy POST vers le service portfolio"""
    import aiohttp
    
    # Récupérer le body de la requête
    body = await request.body()
    headers = dict(request.headers)
    
    # Construire l'URL complète
    portfolio_url = f"http://portfolio:8000/{path}"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                portfolio_url, 
                data=body,
                headers={k: v for k, v in headers.items() if k.lower() not in ['host', 'content-length']},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.content_type == 'application/json':
                    data = await response.json()
                    return data
                else:
                    text = await response.text()
                    return {"data": text}
    except Exception as e:
        logger.error(f"Error proxying POST to portfolio service: {e}")
        raise HTTPException(status_code=503, detail=f"Portfolio service unavailable: {str(e)}")

# Routes React (à la fin pour ne pas intercepter les routes API)
if serve_react_app_flag:
    @app.get("/", response_class=HTMLResponse)
    async def serve_react_app(request: Request):
        """Serve React app"""
        with open(f"{frontend_build_path}/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    
    @app.get("/{path:path}", response_class=HTMLResponse)
    async def serve_react_app_routes(request: Request, path: str):
        """Serve React app for all routes (SPA routing)"""
        # Skip API routes and WebSocket routes
        if path.startswith(("api/", "ws/", "health")):
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Not found")
        with open(f"{frontend_build_path}/index.html", "r") as f:
            return HTMLResponse(content=f.read())

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("VISUALIZATION_PORT", "5009"))
    uvicorn.run(app, host="0.0.0.0", port=port)