"""
API REST pour le service Portfolio.
Expose les endpoints pour accéder aux données du portefeuille et des poches.
Version optimisée avec meilleure gestion des erreurs et du cache.
"""
import logging
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import time
from fastapi import FastAPI, Query, Path, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import psutil
import os

# Importer les modules partagés
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.schemas import AssetBalance, PortfolioSummary
from portfolio.src.startup import on_startup
from portfolio.src.models import PortfolioModel, DBManager, SharedCache
from portfolio.src.binance_account_manager import BinanceAccountManager

# Mesure des performances et des ressources
process = psutil.Process(os.getpid())
start_time = time.time()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('portfolio_api.log')
    ]
)
logger = logging.getLogger("portfolio_api")

# Gestionnaire de contexte pour les événements de démarrage/arrêt
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 API Portfolio en démarrage...")
    
    app.state.shutdown_event = threading.Event()

    try:
        db = DBManager()
        result = db.execute_query("SELECT 1 as test", fetch_one=True)
        db.close()
        if result and result.get('test') == 1:
            logger.info("✅ Connexion à la base de données vérifiée")
        else:
            logger.error("❌ Problème de connexion à la base de données")
    except Exception as e:
        logger.error(f"❌ Erreur de connexion à la base de données: {str(e)}")
    
    # ✅ Appelle la synchronisation Binance ici
    try:
        from portfolio.src.startup import initial_sync_binance
        await initial_sync_binance()
    except Exception as e:
        logger.error(f"❌ Erreur pendant initial_sync_binance depuis lifespan: {e}")

    # ✅ Lance les autres tâches ici aussi
    try:
        from portfolio.src.startup import start_sync_tasks, start_redis_subscriptions
        start_sync_tasks()
        start_redis_subscriptions()
    except Exception as e:
        logger.error(f"❌ Erreur lancement des tâches asynchrones: {e}")

    yield

    logger.info("🛑 API Portfolio en arrêt...")
    app.state.shutdown_event.set()


# Créer l'application FastAPI
app = FastAPI(
    title="RootTrading Portfolio API",
    description="API pour la gestion du portefeuille et des poches de capital",
    version="1.0.0",
    lifespan=lifespan
)

# Configurer CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ajouter la compression gzip
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Cache mémoire simple avec TTL
class InMemoryCache:
    """Cache mémoire simple avec TTL pour réduire la charge sur la DB."""
    
    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache si elle est valide."""
        with self.lock:
            if key in self.cache:
                value, expiry = self.cache[key]
                if time.time() < expiry:
                    return value
                else:
                    del self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: float = 2.0):
        """Stocke une valeur dans le cache avec un TTL."""
        with self.lock:
            expiry = time.time() + ttl
            self.cache[key] = (value, expiry)
    
    def clear(self, pattern: Optional[str] = None):
        """Efface le cache (ou les clés correspondant au pattern)."""
        with self.lock:
            if pattern is None:
                self.cache.clear()
            else:
                keys_to_delete = [k for k in self.cache.keys() if pattern in k]
                for k in keys_to_delete:
                    del self.cache[k]

# Instance globale du cache
api_cache = InMemoryCache()

# Classes pour les réponses API
class TradeHistoryResponse(BaseModel):
    trades: List[Dict[str, Any]]
    total_count: int
    page: int
    page_size: int

class PerformanceResponse(BaseModel):
    data: List[Dict[str, Any]]
    period: str

class ManualBalanceUpdateRequest(BaseModel):
    asset: str
    free: float
    locked: float = 0.0
    value_usdc: Optional[float] = None

# Middleware pour la mesure des performances
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Middleware pour la gestion des erreurs globales
@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"❌ Exception non gérée: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return Response(
            content=json.dumps({
                "detail": "Erreur interne du serveur",
                "status": "error"
            }),
            status_code=500,
            media_type="application/json"
        )

# Dépendances
def get_portfolio_model():
    """Fournit une instance du modèle de portefeuille."""
    db = DBManager()
    model = PortfolioModel(db_manager=db)
    try:
        yield model
    finally:
        model.close()

# Routes
@app.get("/")
async def root():
    """Vérification de base de l'API."""
    return {"status": "ok", "service": "Portfolio API", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health_check():
    """Vérification de l'état de santé de l'API."""
    uptime_seconds = time.time() - start_time
    
    # Formater l'uptime de manière lisible
    days, remainder = divmod(uptime_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if days > 0:
        uptime = f"{int(days)}d {int(hours)}h {int(minutes)}m"
    elif hours > 0:
        uptime = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    else:
        uptime = f"{int(minutes)}m {int(seconds)}s"
    
    # Vérifier l'état de la base de données
    db_status = "ok"
    try:
        db = DBManager()
        db.execute_query("SELECT 1")
        db.close()
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "uptime": uptime,
        "uptime_seconds": uptime_seconds,
        "database": db_status,
        "memory_mb": round(process.memory_info().rss / (1024 * 1024), 2)
    }

@app.get("/healthcheck/binance")
async def binance_healthcheck():
    """Health check simple de Binance"""
    from portfolio.src.binance_account_manager import BinanceAccountManager
    from shared.src.config import BINANCE_API_KEY, BINANCE_SECRET_KEY

    try:
        account = BinanceAccountManager(BINANCE_API_KEY, BINANCE_SECRET_KEY)
        info = account.get_account_info()
        return {"binance_status": "online", "balances": len(info.get('balances', []))}
    except Exception as e:
        return {"binance_status": "offline", "error": str(e)}


@app.get("/diagnostic")
async def diagnostic():
    """
    Endpoint de diagnostic fournissant des informations complètes sur l'état du service.
    """
    try:
        # Obtenir les stats DB
        db_stats = {}
        try:
            db = DBManager()
            result = db.execute_query("SELECT COUNT(*) as count FROM portfolio_balances", fetch_one=True)
            db_stats["balance_count"] = result["count"] if result else 0
                        
            result = db.execute_query("SELECT COUNT(*) as count FROM trade_cycles", fetch_one=True)
            db_stats["cycle_count"] = result["count"] if result else 0
            
            # Informations sur les connexions DB
            result = db.execute_query(
                """
                SELECT 
                    count(*) as connections,
                    sum(case when state = 'active' then 1 else 0 end) as active
                FROM 
                    pg_stat_activity
                """, 
                fetch_one=True
            )
            if result:
                db_stats["connections"] = result["connections"]
                db_stats["active_connections"] = result["active"]
            
            db.close()
        except Exception as e:
            db_stats["error"] = str(e)
        
        # Informations sur le cache
        cache_info = {
            "entries": len(SharedCache._cache) if hasattr(SharedCache, "_cache") else 0,
            "keys": list(SharedCache._cache.keys()) if hasattr(SharedCache, "_cache") else []
        }
        
        # Informations sur les threads
        thread_info = {
            "active_count": threading.active_count(),
            "threads": [t.name for t in threading.enumerate()]
        }
        
        # Informations système
        system_info = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_usage_mb": round(process.memory_info().rss / (1024 * 1024), 2),
            "process_cpu_percent": process.cpu_percent(interval=0.1)
        }
        
        # Construire la réponse
        diagnostic_info = {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - start_time,
            "version": "1.0.0",
            "database": db_stats,
            "cache": cache_info,
            "threads": thread_info,
            "system": system_info
        }
        
        return diagnostic_info
        
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": f"Erreur lors du diagnostic: {str(e)}",
            "traceback": traceback.format_exc()
        }

@app.get("/summary", response_model=PortfolioSummary)
async def get_portfolio_summary(
    portfolio: PortfolioModel = Depends(get_portfolio_model),
    response: Response = None
):
    """
    Récupère un résumé du portefeuille avec cache mémoire.
    
    Returns:
        Résumé du portefeuille
    """
    # Vérifier le cache
    cache_key = "portfolio_summary"
    cached_summary = api_cache.get(cache_key)
    
    if cached_summary:
        logger.debug("📦 Résumé du portfolio servi depuis le cache")
        if response:
            response.headers["X-Cache"] = "HIT"
            response.headers["Cache-Control"] = "public, max-age=5"
        return cached_summary
    
    # Si pas en cache, récupérer depuis la DB
    summary = portfolio.get_portfolio_summary()
    
    if not summary:
        raise HTTPException(status_code=404, detail="Aucune donnée de portefeuille trouvée")
    
    # Mettre en cache pour 2 secondes
    api_cache.set(cache_key, summary, ttl=2.0)
    
    # Ajouter des en-têtes de cache
    if response:
        response.headers["X-Cache"] = "MISS"
        response.headers["Cache-Control"] = "public, max-age=5"
    
    return summary

@app.get("/balances", response_model=List[AssetBalance])
async def get_balances(
    portfolio: PortfolioModel = Depends(get_portfolio_model),
    response: Response = None
):
    """
    Récupère les soldes actuels du portefeuille.
    
    Returns:
        Liste des soldes par actif
    """
    balances = portfolio.get_latest_balances()
    
    if not balances:
        raise HTTPException(status_code=404, detail="Aucun solde trouvé")
    
    # Ajouter des en-têtes de cache
    if response:
        response.headers["Cache-Control"] = "public, max-age=5"
    
    return balances

@app.get("/balance/{asset}")
async def get_balance_by_asset(
    asset: str = Path(..., description="Actif à récupérer (BTC, ETH, USDC, etc.)"),
    portfolio: PortfolioModel = Depends(get_portfolio_model),
    response: Response = None
):
    """
    Récupère le solde pour un actif spécifique.
    
    Args:
        asset: Symbole de l'actif (BTC, ETH, USDC, etc.)
    
    Returns:
        Solde de l'actif avec free, locked et total
    """
    # Récupérer tous les soldes
    balances = portfolio.get_latest_balances()
    
    # Chercher l'actif demandé
    for balance in balances:
        if balance.asset == asset:
            # Ajouter des en-têtes de cache
            if response:
                response.headers["Cache-Control"] = "public, max-age=5"
            
            return {
                "asset": balance.asset,
                "free": balance.free,
                "locked": balance.locked,
                "total": balance.total,
                "value_usdc": balance.value_usdc,
                "available": balance.free  # Alias pour la compatibilité
            }
    
    # Si l'actif n'est pas trouvé, retourner 0
    return {
        "asset": asset,
        "free": 0.0,
        "locked": 0.0,
        "total": 0.0,
        "value_usdc": 0.0,
        "available": 0.0
    }

@app.get("/trades", response_model=TradeHistoryResponse)
async def get_trade_history(
    page: int = Query(1, ge=1, description="Numéro de page"),
    page_size: int = Query(50, ge=1, le=200, description="Taille de la page"),
    symbol: Optional[str] = Query(None, description="Filtrer par symbole"),
    strategy: Optional[str] = Query(None, description="Filtrer par stratégie"),
    start_date: Optional[str] = Query(None, description="Date de début (format ISO)"),
    end_date: Optional[str] = Query(None, description="Date de fin (format ISO)"),
    portfolio: PortfolioModel = Depends(get_portfolio_model),
    response: Response = None
):
    """
    Récupère l'historique des trades avec pagination et filtrage.
    
    Args:
        page: Numéro de page
        page_size: Taille de la page
        symbol: Filtrer par symbole
        strategy: Filtrer par stratégie
        start_date: Date de début
        end_date: Date de fin
    
    Returns:
        Historique des trades
    """
    # Convertir les dates si fournies
    start_date_obj = None
    end_date_obj = None
    
    if start_date:
        try:
            start_date_obj = datetime.fromisoformat(start_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Format de date de début invalide")
    
    if end_date:
        try:
            end_date_obj = datetime.fromisoformat(end_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Format de date de fin invalide")
    
    # Calculer l'offset pour la pagination
    offset = (page - 1) * page_size
    
    # Récupérer les trades avec les filtres
    trades = portfolio.get_trades_history(
        limit=page_size,
        offset=offset,
        symbol=symbol,
        strategy=strategy,
        start_date=start_date_obj,
        end_date=end_date_obj
    )
    
    # Compter le nombre total pour la pagination
    # Amélioration: effectuer une requête COUNT(*) spécifique
    count_query = """
    SELECT COUNT(*) as total FROM trade_cycles tc WHERE 1=1
    """
    
    params = []
    
    if symbol:
        count_query += " AND tc.symbol = %s"
        params.append(symbol)
    
    if strategy:
        count_query += " AND tc.strategy = %s"
        params.append(strategy)
    
    if start_date_obj:
        count_query += " AND tc.created_at >= %s"
        params.append(start_date_obj)
    
    if end_date_obj:
        count_query += " AND tc.created_at <= %s"
        params.append(end_date_obj)
    
    count_result = portfolio.db.execute_query(count_query, tuple(params), fetch_one=True)
    total_count = count_result.get('total', 0) if count_result else len(trades)
    
    # Configurer les entêtes de cache si les données sont historiques
    if response and (not start_date_obj or start_date_obj < datetime.now() - timedelta(days=1)):
        response.headers["Cache-Control"] = "public, max-age=60"  # 5 minutes
    
    return TradeHistoryResponse(
        trades=trades,
        total_count=total_count,
        page=page,
        page_size=page_size
    )

@app.get("/performance/{period}", response_model=PerformanceResponse)
async def get_performance(
    period: str = Path(..., description="Période ('daily', 'weekly', 'monthly')"),
    limit: int = Query(30, ge=1, le=365, description="Nombre de périodes à récupérer"),
    portfolio: PortfolioModel = Depends(get_portfolio_model),
    response: Response = None
):
    """
    Récupère les statistiques de performance.
    
    Args:
        period: Période ('daily', 'weekly', 'monthly')
        limit: Nombre de périodes à récupérer
    
    Returns:
        Statistiques de performance
    """
    if period not in ['daily', 'weekly', 'monthly']:
        raise HTTPException(status_code=400, detail="Période invalide")
    
    stats = portfolio.get_performance_stats(period=period, limit=limit)
    
    # Les données de performance peuvent être mises en cache plus longtemps
    if response:
        response.headers["Cache-Control"] = "public, max-age=60"  # 5 minutes
    
    return PerformanceResponse(data=stats, period=period)

@app.get("/performance/strategy", response_model=Dict[str, Any])
async def get_strategy_performance(
    portfolio: PortfolioModel = Depends(get_portfolio_model),
    response: Response = None
):
    """
    Récupère les performances par stratégie.
    
    Returns:
        Performances par stratégie
    """
    performance = portfolio.get_strategy_performance()
    
    # Les données de performance peuvent être mises en cache plus longtemps
    if response:
        response.headers["Cache-Control"] = "public, max-age=60"  # 5 minutes
    
    return {"data": performance}

@app.get("/performance/symbol", response_model=Dict[str, Any])
async def get_symbol_performance(
    portfolio: PortfolioModel = Depends(get_portfolio_model),
    response: Response = None
):
    """
    Récupère les performances par symbole.
    
    Returns:
        Performances par symbole
    """
    performance = portfolio.get_symbol_performance()
    
    # Les données de performance peuvent être mises en cache plus longtemps
    if response:
        response.headers["Cache-Control"] = "public, max-age=60"  # 5 minutes
    
    return {"data": performance}

@app.post("/balances/update", status_code=200)
async def update_balance_manually(
    balances: List[ManualBalanceUpdateRequest],
    portfolio: PortfolioModel = Depends(get_portfolio_model)
):
    """
    Met à jour les soldes manuellement.
    Utile pour les tests ou les corrections.
    
    Args:
        balances: Liste des soldes à mettre à jour
    
    Returns:
        Statut de la mise à jour
    """
    if not balances:
        raise HTTPException(status_code=400, detail="Liste des soldes vide")
    
    # Convertir en AssetBalance
    asset_balances = []
    for balance in balances:
        asset_balance = AssetBalance(
            asset=balance.asset,
            free=balance.free,
            locked=balance.locked,
            total=balance.free + balance.locked,
            value_usdc=balance.value_usdc
        )
        asset_balances.append(asset_balance)
    
    success = portfolio.update_balances(asset_balances)
    
    if not success:
        raise HTTPException(status_code=500, detail="Échec de la mise à jour des soldes")
    
    # Invalider le cache explicitement
    SharedCache.clear('latest_balances')
    SharedCache.clear('portfolio_summary')
    
    return {"status": "success", "message": f"{len(balances)} soldes mis à jour"}

@app.get("/_debug/db_test")
async def test_db_connection():
    """
    Endpoint de diagnostic pour tester la connexion à la base de données.
    """
    try:
        db = DBManager()
        
        # Mesurer le temps de réponse
        start = time.time()
        result = db.execute_query("SELECT 1 as test", fetch_one=True)
        query_time = time.time() - start
        
        # Récupérer le nombre de connexions
        connections_query = """
        SELECT 
            count(*) as total,
            sum(case when state = 'active' then 1 else 0 end) as active
        FROM 
            pg_stat_activity
        """
        conn_result = db.execute_query(connections_query, fetch_one=True)
        
        db.close()
        
        if result and result.get('test') == 1:
            return {
                "status": "ok", 
                "message": "Connexion à la base de données réussie",
                "query_time_ms": round(query_time * 1000, 2),
                "connections": conn_result
            }
        else:
            return {"status": "error", "message": "Problème de connexion à la base de données"}
    except Exception as e:
        import traceback
        return {
            "status": "error", 
            "message": f"Exception: {str(e)}", 
            "traceback": traceback.format_exc()
        }

@app.get("/_debug/cache_info")
async def get_cache_info():
    """
    Endpoint de diagnostic pour inspecter l'état du cache.
    """
    try:
        if not hasattr(SharedCache, "_cache"):
            return {
                "status": "ok",
                "message": "Cache non initialisé",
                "entries": 0
            }
        
        cache_entries = {}
        total_size = 0
        
        for key, (timestamp, data) in SharedCache._cache.items():
            age = time.time() - timestamp
            
            # Estimer la taille des données
            try:
                import sys
                data_size = sys.getsizeof(data)
                if hasattr(data, "__len__"):
                    data_size += sum(sys.getsizeof(i) for i in data[:10]) * (len(data) / 10) if len(data) > 10 else sum(sys.getsizeof(i) for i in data)
            except:
                data_size = 0
            
            cache_entries[key] = {
                "age_seconds": round(age, 2),
                "type": type(data).__name__,
                "size_bytes": data_size,
                "items": len(data) if hasattr(data, "__len__") else 1
            }
            
            total_size += data_size
        
        return {
            "status": "ok",
            "entries": len(SharedCache._cache),
            "total_size_kb": round(total_size / 1024, 2),
            "cache": cache_entries
        }
    except Exception as e:
        import traceback
        return {
            "status": "error", 
            "message": f"Exception: {str(e)}", 
            "traceback": traceback.format_exc()
        }

@app.delete("/_debug/cache_clear")
async def clear_cache(prefix: Optional[str] = Query(None, description="Préfixe des clés à effacer")):
    """
    Endpoint de diagnostic pour effacer le cache.
    """
    try:
        before_count = len(SharedCache._cache) if hasattr(SharedCache, "_cache") else 0
        
        SharedCache.clear(prefix)
        
        after_count = len(SharedCache._cache) if hasattr(SharedCache, "_cache") else 0
        
        return {
            "status": "ok",
            "message": f"Cache {'partiellement ' if prefix else ''}effacé",
            "prefix": prefix or "all",
            "entries_before": before_count,
            "entries_after": after_count,
            "cleared": before_count - after_count
        }
    except Exception as e:
        import traceback
        return {
            "status": "error", 
            "message": f"Exception: {str(e)}", 
            "traceback": traceback.format_exc()
        }

# Point d'entrée principal
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)