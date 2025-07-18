from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import logging
import os

from data_manager import DataManager
from chart_service import ChartService
from websocket_hub import WebSocketHub

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data_manager: Optional[DataManager] = None
chart_service: Optional[ChartService] = None
websocket_hub: Optional[WebSocketHub] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global data_manager, chart_service, websocket_hub
    
    logger.info("Starting visualization service...")
    
    data_manager = DataManager()
    await data_manager.initialize()
    
    chart_service = ChartService(data_manager)
    websocket_hub = WebSocketHub(data_manager)
    
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
                "30m": 1440,   # 30 jours de données
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
                "30m": 1440,   # 30 jours de données
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