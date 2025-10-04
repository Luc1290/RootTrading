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

from data_manager import DataManager
from chart_service import ChartService
from websocket_hub import WebSocketHub
from statistics_service import StatisticsService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data_manager: Optional[DataManager] = None
chart_service: Optional[ChartService] = None
websocket_hub: Optional[WebSocketHub] = None
statistics_service: Optional[StatisticsService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global data_manager, chart_service, websocket_hub, statistics_service
    
    logger.info("Starting visualization service...")
    
    data_manager = DataManager()
    await data_manager.initialize()
    
    chart_service = ChartService(data_manager)
    websocket_hub = WebSocketHub(data_manager)
    statistics_service = StatisticsService(data_manager)
    
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

        async with data_manager.postgres_pool.acquire() as conn:
            # Récupérer données des 2 dernières heures (scalping rapide sur 5m)
            signals_query = """
                SELECT COUNT(*) as count, AVG(confidence) as avg_conf
                FROM trading_signals
                WHERE symbol = $1
                AND side = 'BUY'
                AND created_at > NOW() - INTERVAL '2 hours'
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
                AND timeframe = '5m'
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
            signals_count = int(signals_data['count']) if signals_data else 0
            avg_confidence = float(signals_data['avg_conf']) if signals_data and signals_data['avg_conf'] else 0

            # ================================================================
            # SCORING AVANCÉ - 100 points répartis intelligemment
            # ================================================================

            def safe_float(value, default=0.0):
                """Convertir en float avec fallback"""
                return float(value) if value is not None else default

            # Extraction des données
            ad = analyzer_data if analyzer_data else {}

            # Calculer ATR en pourcentage
            atr_value = safe_float(ad.get('atr_14'))
            atr_percent = (atr_value / current_price) if current_price > 0 and atr_value > 0 else 0.015

            # ============================================================
            # 1. TREND QUALITY (25 points)
            # ============================================================
            trend_score = 0

            # ADX - Force de tendance (max 10 pts)
            adx = safe_float(ad.get('adx_14'))
            if adx > 40:  # Tendance extrêmement forte
                trend_score += 10
            elif adx > 30:  # Forte tendance
                trend_score += 8
            elif adx > 25:  # Tendance modérée
                trend_score += 5
            elif adx > 20:  # Tendance faible
                trend_score += 2

            # Directional Movement - Alignement (max 8 pts)
            plus_di = safe_float(ad.get('plus_di'))
            minus_di = safe_float(ad.get('minus_di'))
            if plus_di > minus_di and plus_di > 25:  # Bull fort
                trend_score += 8
            elif plus_di > minus_di and plus_di > 20:  # Bull modéré
                trend_score += 5
            elif plus_di > minus_di:  # Bull faible
                trend_score += 3

            # Regime confidence (max 7 pts)
            regime_conf = safe_float(ad.get('regime_confidence'))
            regime = ad.get('market_regime', 'UNKNOWN')
            if regime in ['TRENDING_BULL', 'BREAKOUT_BULL'] and regime_conf > 80:
                trend_score += 7
            elif regime in ['TRENDING_BULL', 'BREAKOUT_BULL'] and regime_conf > 60:
                trend_score += 4
            elif regime == 'RANGING':
                trend_score += 0  # Pas de tendance

            # ============================================================
            # 2. MOMENTUM CONFLUENCE (25 points)
            # ============================================================
            momentum_score = 0

            # RSI - Position et alignement (max 8 pts)
            rsi_14 = safe_float(ad.get('rsi_14'), 50)
            rsi_21 = safe_float(ad.get('rsi_21'), 50)
            if 50 < rsi_14 < 70 and rsi_14 > rsi_21:  # Zone bull optimal
                momentum_score += 8
            elif 40 < rsi_14 < 60:  # Zone neutre
                momentum_score += 4
            elif rsi_14 < 30:  # Oversold - peut rebondir
                momentum_score += 3

            # Stochastic (max 7 pts)
            stoch_k = safe_float(ad.get('stoch_k'))
            stoch_d = safe_float(ad.get('stoch_d'))
            stoch_signal = ad.get('stoch_signal')
            if stoch_signal == 'BUY' or (stoch_k > stoch_d and stoch_k > 20):
                momentum_score += 7
            elif stoch_k > 50:
                momentum_score += 4

            # MFI - Money Flow (max 5 pts)
            mfi = safe_float(ad.get('mfi_14'))
            if 50 < mfi < 80:  # Argent entre mais pas overbought
                momentum_score += 5
            elif 40 < mfi < 60:
                momentum_score += 3

            # MACD (max 5 pts)
            macd_hist = safe_float(ad.get('macd_histogram'))
            macd_trend = ad.get('macd_trend')
            if macd_trend == 'BULLISH' and macd_hist > 0:
                momentum_score += 5
            elif macd_hist > 0:
                momentum_score += 3

            # ============================================================
            # 3. VOLUME VALIDATION (20 points)
            # ============================================================
            volume_score = 0

            # Volume Quality Score (max 8 pts)
            vol_quality = safe_float(ad.get('volume_quality_score'))
            volume_score += min(8, (vol_quality / 100) * 8)

            # Volume Context (max 7 pts)
            vol_context = ad.get('volume_context')
            if vol_context == 'ACCUMULATION':
                volume_score += 7
            elif vol_context == 'BREAKOUT':
                volume_score += 6
            elif vol_context == 'DISTRIBUTION':
                volume_score += 0  # Mauvais signe

            # Relative Volume (max 5 pts)
            rel_volume = safe_float(ad.get('relative_volume'), 1.0)
            if rel_volume > 2.0:
                volume_score += 5
            elif rel_volume > 1.5:
                volume_score += 3
            elif rel_volume > 1.2:
                volume_score += 2

            # ============================================================
            # 4. PRICE ACTION (20 points)
            # ============================================================
            price_action_score = 0

            # Distance au support/résistance (max 8 pts)
            nearest_support = safe_float(ad.get('nearest_support'))
            nearest_resistance = safe_float(ad.get('nearest_resistance'))
            support_strength = ad.get('support_strength')
            resistance_strength = ad.get('resistance_strength')

            if nearest_support > 0:
                dist_to_support = ((current_price - nearest_support) / nearest_support) * 100
                if 0.5 < dist_to_support < 2 and support_strength == 'STRONG':
                    price_action_score += 8  # Près d'un support fort
                elif 0.5 < dist_to_support < 3:
                    price_action_score += 5

            # Break probability (max 6 pts)
            break_prob = safe_float(ad.get('break_probability'))
            if break_prob > 70:
                price_action_score += 6
            elif break_prob > 50:
                price_action_score += 4

            # Bollinger position & squeeze (max 6 pts)
            bb_position = safe_float(ad.get('bb_position'), 0.5)
            bb_width = safe_float(ad.get('bb_width'), 0.0)
            bb_squeeze = ad.get('bb_squeeze', False)
            bb_expansion = ad.get('bb_expansion', False)

            if bb_expansion and 0.3 < bb_position < 0.7:  # Expansion en cours, milieu de bande
                price_action_score += 6
            elif bb_squeeze:  # Squeeze = volatilité imminente
                price_action_score += 4
            elif 0.2 < bb_position < 0.8:  # Position normale
                price_action_score += 3

            # ============================================================
            # 5. CONSENSUS & SIGNALS (10 points)
            # ============================================================
            consensus_score = 0

            # Confluence score de l'analyzer (max 5 pts)
            confluence = safe_float(ad.get('confluence_score'))
            consensus_score += min(5, (confluence / 100) * 5)

            # Signal strength (max 3 pts)
            signal_strength = ad.get('signal_strength')
            if signal_strength == 'STRONG':
                consensus_score += 3
            elif signal_strength == 'MODERATE':
                consensus_score += 2

            # Pattern confidence (max 2 pts)
            pattern_conf = safe_float(ad.get('pattern_confidence'))
            if pattern_conf > 70:
                consensus_score += 2
            elif pattern_conf > 50:
                consensus_score += 1

            # ============================================================
            # SCORE TOTAL
            # ============================================================
            total_score = trend_score + momentum_score + volume_score + price_action_score + consensus_score

            # ============================================================
            # ESTIMATION DURÉE DE HOLD (scalping 5m)
            # ============================================================
            # Basé sur volatilité, momentum et type de setup
            hold_time_min = 5  # Minimum 5 minutes (1 bougie 5m)
            hold_time_max = 45  # Maximum 45 minutes (9 bougies 5m)

            # Ajuster selon volatilité (ATR)
            if atr_percent > 0.025:  # Haute volatilité (>2.5%)
                hold_time_min = 5
                hold_time_max = 20  # Sortir vite si très volatile
            elif atr_percent > 0.015:  # Volatilité moyenne
                hold_time_min = 10
                hold_time_max = 30
            else:  # Faible volatilité
                hold_time_min = 15
                hold_time_max = 45  # Tenir plus longtemps

            # Ajuster selon momentum
            if momentum_score > 20:  # Fort momentum
                hold_time_max = min(hold_time_max, 30)  # Sortir avant retournement
            elif momentum_score < 15:  # Momentum faible
                hold_time_min = max(hold_time_min, 15)  # Attendre confirmation

            # Ajuster selon régime
            if regime == 'BREAKOUT_BULL':
                hold_time_min = 5  # Sortir vite sur breakout
                hold_time_max = 20
            elif regime == 'TRENDING_BULL':
                hold_time_min = 15  # Tenir la tendance
                hold_time_max = 45

            estimated_hold_time = f"{hold_time_min}-{hold_time_max} min"

            # Détails des composantes
            score_details = {
                "trend": round(trend_score, 1),
                "momentum": round(momentum_score, 1),
                "volume": round(volume_score, 1),
                "price_action": round(price_action_score, 1),
                "consensus": round(consensus_score, 1)
            }

            # Explication détaillée de chaque pilier
            score_explanation = {
                "trend": f"ADX:{adx:.1f} ({'+10pts' if adx>40 else '+8pts' if adx>30 else '+5pts' if adx>25 else '+2pts' if adx>20 else '0pt'}), DI+:{plus_di:.1f} vs DI-:{minus_di:.1f} ({'+8pts' if plus_di>minus_di and plus_di>25 else '+5pts' if plus_di>minus_di and plus_di>20 else '+3pts' if plus_di>minus_di else '0pt'}), Régime:{regime} conf:{regime_conf:.0f}% ({'+7pts' if regime in ['TRENDING_BULL','BREAKOUT_BULL'] and regime_conf>80 else '+4pts' if regime in ['TRENDING_BULL','BREAKOUT_BULL'] and regime_conf>60 else '0pt'})",
                "momentum": f"RSI14:{rsi_14:.1f} vs RSI21:{rsi_21:.1f} ({'+8pts' if 50<rsi_14<70 and rsi_14>rsi_21 else '+4pts' if 40<rsi_14<60 else '+3pts' if rsi_14<30 else '0pt'}), Stoch:{stoch_k:.1f}/{stoch_d:.1f} signal:{stoch_signal or 'N/A'} ({'+7pts' if stoch_signal=='BUY' or (stoch_k>stoch_d and stoch_k>20) else '+4pts' if stoch_k>50 else '0pt'}), MFI:{mfi:.1f} ({'+5pts' if 50<mfi<80 else '+3pts' if 40<mfi<60 else '0pt'}), MACD hist:{macd_hist:.4f} trend:{macd_trend or 'N/A'} ({'+5pts' if macd_trend=='BULLISH' and macd_hist>0 else '+3pts' if macd_hist>0 else '0pt'})",
                "volume": f"Quality:{vol_quality:.0f}/100 (+{min(8, (vol_quality/100)*8):.1f}pts), Context:{vol_context or 'N/A'} ({'+7pts' if vol_context=='ACCUMULATION' else '+6pts' if vol_context=='BREAKOUT' else '0pt'}), Rel.volume:{rel_volume:.2f}x ({'+5pts' if rel_volume>2.0 else '+3pts' if rel_volume>1.5 else '+2pts' if rel_volume>1.2 else '0pt'})",
                "price_action": f"Support:{nearest_support:.4f} ({support_strength or 'N/A'}), Resistance:{nearest_resistance:.4f} ({resistance_strength or 'N/A'}), Break prob:{break_prob:.0f}% ({'+6pts' if break_prob>70 else '+4pts' if break_prob>50 else '0pt'}), BB pos:{bb_position:.2f} squeeze:{bb_squeeze} expansion:{bb_expansion}",
                "consensus": f"Confluence:{confluence:.0f}/100 (+{min(5, (confluence/100)*5):.1f}pts), Signal strength:{signal_strength or 'N/A'} ({'+3pts' if signal_strength=='STRONG' else '+2pts' if signal_strength=='MODERATE' else '0pt'}), Pattern conf:{pattern_conf:.0f}% ({'+2pts' if pattern_conf>70 else '+1pt' if pattern_conf>50 else '0pt'})"
            }

            # ============================================================
            # DÉTERMINATION ACTION - Multi-critères robustes
            # ============================================================
            action = "WAIT"
            reason_parts = []

            # ============================================================
            # 1. VÉRIFIER OVERBOUGHT (priorité absolue)
            # ============================================================
            is_overbought = False
            overbought_reasons = []

            if rsi_14 > 75:
                is_overbought = True
                overbought_reasons.append(f"RSI {rsi_14:.0f}")

            if mfi > 80:
                is_overbought = True
                overbought_reasons.append(f"MFI {mfi:.0f}")

            if stoch_k > 90 and stoch_d > 90:
                is_overbought = True
                overbought_reasons.append(f"Stoch {stoch_k:.0f}")

            if bb_position > 1.0:  # Au-dessus de la bande supérieure
                is_overbought = True
                overbought_reasons.append(f"BB overshoot")

            # Si OVERBOUGHT → SELL signal
            if is_overbought:
                action = "SELL_OVERBOUGHT"
                score_breakdown = f"Trend:{trend_score:.0f}/25 | Momentum:{momentum_score:.0f}/25 | Volume:{volume_score:.0f}/20"
                indicators_detail = f"ADX:{adx:.1f}, RSI:{rsi_14:.1f}, MFI:{mfi:.1f}, Stoch:{stoch_k:.1f}/{stoch_d:.1f}, BB pos:{bb_position:.2f}"
                reason = f"🔴 SURACHETÉ ({score_breakdown}) | {indicators_detail} | " + " + ".join(overbought_reasons) + " → Correction imminente probable, VENDRE ou éviter l'achat"

            # ============================================================
            # 2. VÉRIFIER OVERSOLD (opportunité d'achat au rebond)
            # ============================================================
            elif rsi_14 < 30 and stoch_k < 20:
                action = "WAIT_OVERSOLD"
                score_breakdown = f"Score:{total_score:.0f}/100 (Trend:{trend_score:.0f} | Momentum:{momentum_score:.0f} | Volume:{volume_score:.0f})"
                indicators_detail = f"RSI:{rsi_14:.1f} (seuil:<30), Stoch:{stoch_k:.1f} (seuil:<20), MFI:{mfi:.1f}"
                reason = f"🔵 SURVENDU ({score_breakdown}) | {indicators_detail} | Zone de rebond potentiel → Attendre signal d'inversion (RSI>35, volume en hausse, bougie verte)"

            # ============================================================
            # 3. CRITÈRES BUY NOW (conditions strictes)
            # ============================================================
            else:
                buy_criteria = {
                    "score_high": total_score >= 65,  # Abaissé pour 5m (plus volatil)
                    "trend_strong": trend_score >= 12,  # Sur 25 - moins strict pour scalping
                    "volume_confirmed": volume_score >= 10,  # Sur 20 - volume plus important en 5m
                    "momentum_aligned": momentum_score >= 10,  # Sur 25 - momentum plus réactif
                    "regime_bull": regime in ['TRENDING_BULL', 'BREAKOUT_BULL'],
                    "adx_trending": adx > 20,  # ADX moins strict sur 5m
                    "not_overbought": rsi_14 < 72,  # Légèrement plus tolérant sur 5m
                    "vol_quality": vol_quality > 45  # Volume quality moins strict
                }

                buy_score = sum(buy_criteria.values())

                if buy_score >= 7:  # Au moins 7 critères sur 8
                    action = "BUY_NOW"
                    score_breakdown = f"Trend:{trend_score:.0f}/25 | Momentum:{momentum_score:.0f}/25 | Volume:{volume_score:.0f}/20 | Price:{price_action_score:.0f}/20"

                    # Détails ultra précis
                    detail_parts = [
                        f"ADX:{adx:.1f} (+DI:{plus_di:.1f} vs -DI:{minus_di:.1f})",
                        f"RSI:{rsi_14:.1f}",
                        f"MFI:{mfi:.1f}",
                        f"Stoch:{stoch_k:.1f}/{stoch_d:.1f}"
                    ]
                    if vol_context:
                        detail_parts.append(f"Vol:{vol_context} (qual:{vol_quality:.0f}, ratio:{rel_volume:.1f}x)")
                    if regime:
                        detail_parts.append(f"Régime:{regime} ({regime_conf:.0f}%)")

                    reason = f"💎 EXCELLENT ({buy_score}/8 critères) | {score_breakdown} | " + " | ".join(detail_parts)

                elif buy_score >= 5 and total_score >= 55:  # Abaissé pour scalping 5m
                    action = "BUY_NOW"
                    score_breakdown = f"Trend:{trend_score:.0f}/25 | Momentum:{momentum_score:.0f}/25 | Volume:{volume_score:.0f}/20"

                    # Expliquer ce qui est bon et ce qui manque
                    good_parts = []
                    missing_parts = []

                    if trend_score >= 12:
                        good_parts.append(f"✓Trend (ADX:{adx:.1f}, +DI:{plus_di:.1f})")
                    else:
                        missing_parts.append(f"✗Trend faible ({trend_score:.0f}/25)")

                    if volume_score >= 10:
                        good_parts.append(f"✓Volume (qual:{vol_quality:.0f}, {vol_context or 'N/A'})")
                    else:
                        missing_parts.append(f"✗Volume ({volume_score:.0f}/20)")

                    if momentum_score >= 10:
                        good_parts.append(f"✓Momentum (RSI:{rsi_14:.1f}, MFI:{mfi:.1f})")
                    else:
                        missing_parts.append(f"✗Momentum ({momentum_score:.0f}/25)")

                    all_details = good_parts + missing_parts
                    reason = f"✅ BON ({buy_score}/8 critères) | {score_breakdown} | " + " | ".join(all_details)

                # ============================================================
                # 4. CRITÈRES WAIT (observation)
                # ============================================================
                elif total_score >= 40:  # Abaissé pour 5m - plus tolérant
                    score_breakdown = f"Trend:{trend_score:.0f}/25 | Momentum:{momentum_score:.0f}/25 | Volume:{volume_score:.0f}/20"

                    # Construire explication détaillée
                    indicators_detail = f"ADX:{adx:.1f}, RSI:{rsi_14:.1f}, MFI:{mfi:.1f}, Stoch:{stoch_k:.1f}"

                    # Identifier ce qui manque pour BUY
                    missing_for_buy = []
                    if total_score < 65:
                        missing_for_buy.append(f"Score {total_score:.0f}/65 requis")
                    if trend_score < 12:
                        missing_for_buy.append(f"Trend faible ({trend_score:.0f}/12)")
                    if volume_score < 10:
                        missing_for_buy.append(f"Volume faible ({volume_score:.0f}/10)")
                    if momentum_score < 10:
                        missing_for_buy.append(f"Momentum faible ({momentum_score:.0f}/10)")

                    if bb_squeeze:
                        action = "WAIT_BREAKOUT"
                        reason = f"🟡 {score_breakdown} | {indicators_detail} | BB squeeze (BB width:{bb_width:.4f}) → Volatilité imminente | Manque: " + ", ".join(missing_for_buy if missing_for_buy else ["Attendre direction"])
                    elif vol_context == 'DISTRIBUTION':
                        action = "WAIT"
                        reason = f"🟡 {score_breakdown} | {indicators_detail} | Distribution (vente institutionnelle, vol qual:{vol_quality:.0f}) → Attendre accumulation | Manque: " + ", ".join(missing_for_buy)
                    elif nearest_resistance > 0:
                        dist_to_resistance = ((nearest_resistance - current_price) / current_price) * 100
                        if dist_to_resistance < 1:
                            action = "WAIT"
                            reason = f"🟡 {score_breakdown} | {indicators_detail} | Résistance {nearest_resistance:.4f} à {dist_to_resistance:.1f}% → Attendre cassure | Manque: " + ", ".join(missing_for_buy)
                        else:
                            action = "WAIT"
                            reason = f"🟡 {score_breakdown} | {indicators_detail} | Régime:{regime} | Manque: " + ", ".join(missing_for_buy if missing_for_buy else ["Confluence insuffisante"])
                    else:
                        action = "WAIT"
                        reason = f"🟡 {score_breakdown} | {indicators_detail} | Manque: " + ", ".join(missing_for_buy if missing_for_buy else ["Confirmation volume/tendance"])

                # ============================================================
                # 5. AVOID (conditions vraiment défavorables)
                # ============================================================
                else:
                    action = "AVOID"
                    # Détails du breakdown
                    score_breakdown = f"Trend:{trend_score:.0f}/25 | Momentum:{momentum_score:.0f}/25 | Volume:{volume_score:.0f}/20 | Price:{price_action_score:.0f}/20 | Consensus:{consensus_score:.0f}/10"

                    # Identifier les points faibles
                    weak_points = []
                    if trend_score < 8:
                        weak_points.append(f"Trend faible ({trend_score:.0f}/25, ADX:{adx:.0f})")
                    if momentum_score < 8:
                        weak_points.append(f"Momentum faible ({momentum_score:.0f}/25, RSI:{rsi_14:.0f})")
                    if volume_score < 6:
                        weak_points.append(f"Volume insuffisant ({volume_score:.0f}/20, qual:{vol_quality:.0f}, context:{vol_context or 'N/A'})")
                    if price_action_score < 6:
                        weak_points.append(f"Price action faible ({price_action_score:.0f}/20)")
                    if regime in ['TRENDING_BEAR', 'BREAKOUT_BEAR']:
                        weak_points.append(f"Régime baissier ({regime})")

                    reason = f"⚫ {score_breakdown} | " + " | ".join(weak_points if weak_points else ["Setup défavorable"])

            # Calcul zones et targets
            entry_min = current_price * 0.998
            entry_max = current_price * 1.002

            tp1 = current_price * (1 + max(0.01, atr_percent * 0.7))
            tp2 = current_price * (1 + max(0.015, atr_percent * 1.0))
            tp3 = current_price * (1 + max(0.02, atr_percent * 1.5))

            stop_loss = current_price * (1 - max(0.012, atr_percent * 1.2))

            # Taille position recommandée basée sur volatilité
            if atr_percent < 0.01:
                rec_min, rec_max = 5000, 10000  # Faible volatilité
            elif atr_percent < 0.02:
                rec_min, rec_max = 3000, 7000   # Volatilité moyenne
            else:
                rec_min, rec_max = 2000, 5000   # Haute volatilité

            return {
                "symbol": symbol,
                "score": round(total_score, 1),
                "score_details": score_details,
                "score_explanation": score_explanation,
                "signals_count": signals_count,
                "avg_confidence": avg_confidence,
                "momentum_score": safe_float(ad.get('momentum_score')),
                "volume_ratio": safe_float(ad.get('volume_ratio'), 1.0),
                "market_regime": regime,
                "adx": adx,
                "rsi": rsi_14,
                "mfi": mfi,
                "volume_context": vol_context,
                "volume_quality_score": vol_quality,
                "nearest_support": nearest_support,
                "nearest_resistance": nearest_resistance,
                "break_probability": break_prob,
                "entry_zone": {
                    "min": round(entry_min, 8),
                    "max": round(entry_max, 8)
                },
                "targets": {
                    "tp1": round(tp1, 8),
                    "tp2": round(tp2, 8),
                    "tp3": round(tp3, 8)
                },
                "stop_loss": round(stop_loss, 8),
                "recommended_size": {
                    "min": rec_min,
                    "max": rec_max
                },
                "action": action,
                "reason": reason,
                "estimated_hold_time": estimated_hold_time
            }

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