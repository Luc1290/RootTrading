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

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Interface web principale"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "redis_connected": data_manager.is_redis_connected() if data_manager else False,
        "postgres_connected": data_manager.is_postgres_connected() if data_manager else False
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
                "5m": 2016,    # 7 jours de données
                "15m": 1344,   # 14 jours de données  
                "30m": 1440,   # 30 jours de données
                "1h": 720,     # 30 jours de données
                "4h": 720,     # 120 jours de données
                "1d": 365      # 1 an de données
            }
            limit = timeframe_limits.get(interval, 2880)
        
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
                "5m": 2016,    # 7 jours de données
                "15m": 1344,   # 14 jours de données  
                "30m": 1440,   # 30 jours de données
                "1h": 720,     # 30 jours de données
                "4h": 720,     # 120 jours de données
                "1d": 365      # 1 an de données
            }
            limit = timeframe_limits.get(interval, 2880)
            
        indicator_list = indicators.split(",")
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
    await websocket_hub.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("action") == "subscribe":
                await websocket_hub.subscribe_client(
                    client_id,
                    data.get("channel"),
                    data.get("params", {})
                )
            elif data.get("action") == "unsubscribe":
                await websocket_hub.unsubscribe_client(
                    client_id,
                    data.get("channel")
                )
                
    except WebSocketDisconnect:
        await websocket_hub.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("VISUALIZATION_PORT", "5009"))
    uvicorn.run(app, host="0.0.0.0", port=port)