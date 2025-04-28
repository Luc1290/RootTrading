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

from shared.src.schemas import AssetBalance, PortfolioSummary, PocketSummary

from portfolio.src.models import PortfolioModel, DBManager, SharedCache
from portfolio.src.pockets import PocketManager
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
    # Code exécuté au démarrage
    logger.info("🚀 API Portfolio en démarrage...")
    
    # Fournir un événement global
    app.state.shutdown_event = threading.Event()
    
    # Vérifier l'état de la base de données
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
    
    yield
    
    # Code exécuté à l'arrêt
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

# Classes pour les réponses API
class TradeHistoryResponse(BaseModel):
    trades: List[Dict[str, Any]]
    total_count: int
    page: int
    page_size: int

class PerformanceResponse(BaseModel):
    data: List[Dict[str, Any]]
    period: str

class PocketUpdateRequest(BaseModel):
    pocket_type: str
    allocation_percent: float

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

def get_pocket_manager():
    """Fournit une instance du gestionnaire de poches."""
    db = DBManager()
    manager = PocketManager(db_manager=db)
    try:
        yield manager
    finally:
        manager.close()

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
            
            result = db.execute_query("SELECT COUNT(*) as count FROM capital_pockets", fetch_one=True)
            db_stats["pocket_count"] = result["count"] if result else 0
            
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
        
        # Obtenir les info des poches
        pockets_info = {}
        try:
            pocket_manager = PocketManager()
            pockets = pocket_manager.get_pockets()
            for pocket in pockets:
                pockets_info[pocket.pocket_type] = {
                    "current_value": float(pocket.current_value),
                    "used_value": float(pocket.used_value),
                    "available_value": float(pocket.available_value),
                    "active_cycles": pocket.active_cycles
                }
            pocket_manager.close()
        except Exception as e:
            pockets_info["error"] = str(e)
        
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
            "pockets": pockets_info,
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
    Récupère un résumé du portefeuille.
    
    Returns:
        Résumé du portefeuille
    """
    summary = portfolio.get_portfolio_summary()
    
    if not summary:
        raise HTTPException(status_code=404, detail="Aucune donnée de portefeuille trouvée")
    
    # Ajouter des en-têtes de cache
    if response:
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

@app.get("/pockets", response_model=List[PocketSummary])
async def get_pockets(
    pocket_manager: PocketManager = Depends(get_pocket_manager),
    response: Response = None
):
    """
    Récupère l'état actuel des poches de capital.
    
    Returns:
        Liste des poches de capital
    """
    pockets = pocket_manager.get_pockets()
    
    if not pockets:
        raise HTTPException(status_code=404, detail="Aucune poche trouvée")
    
    # Ajouter des en-têtes de cache
    if response:
        response.headers["Cache-Control"] = "public, max-age=5"
    
    return pockets

@app.put("/pockets/sync", status_code=200)
async def sync_pockets(
    background_tasks: BackgroundTasks,
    pocket_manager: PocketManager = Depends(get_pocket_manager)
):
    """
    Synchronise les poches avec les trades actifs.
    Exécuté en arrière-plan pour ne pas bloquer la réponse.
    
    Returns:
        Statut de la synchronisation
    """
    # Fonction d'arrière-plan
    def sync_task():
        try:
            success = pocket_manager.sync_with_trades()
            if not success:
                logger.error("❌ Échec de la synchronisation des poches en arrière-plan")
            else:
                logger.info("✅ Synchronisation des poches réussie en arrière-plan")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la synchronisation en arrière-plan: {str(e)}")
    
    # Ajouter la tâche en arrière-plan
    background_tasks.add_task(sync_task)
    
    return {
        "status": "accepted", 
        "message": "Synchronisation des poches lancée en arrière-plan"
    }

@app.put("/pockets/allocation", status_code=200)
async def update_pocket_allocation(
    total_value: float = Query(..., description="Valeur totale du portefeuille"),
    pocket_manager: PocketManager = Depends(get_pocket_manager)
):
    """
    Met à jour l'allocation des poches en fonction de la valeur totale.
    """
    try:
        # Validation des paramètres
        if total_value < 0:
            raise HTTPException(status_code=400, detail=f"Valeur totale invalide: {total_value}")
        
        # Modification: utiliser une valeur minimale si total_value est 0
        if total_value == 0:
            logger.info(f"Valeur totale reçue: {total_value}, utilisation d'une valeur minimale de 100.0")
            total_value = 100.0  # Valeur par défaut pour l'initialisation
        
        success = pocket_manager.update_pockets_allocation(total_value)
        
        if not success:
            raise HTTPException(status_code=500, detail="Échec de la mise à jour de l'allocation")
        
        return {"status": "success", "message": "Allocation mise à jour avec succès"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur dans update_pocket_allocation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.post("/pockets/{pocket_type}/reserve", status_code=200)
async def reserve_funds(
    pocket_type: str = Path(..., description="Type de poche (active, buffer, safety)"),
    amount: float = Query(..., description="Montant à réserver"),
    cycle_id: str = Query(..., description="ID du cycle de trading"),
    pocket_manager: PocketManager = Depends(get_pocket_manager)
):
    """
    Réserve des fonds dans une poche pour un cycle de trading.
    
    Args:
        pocket_type: Type de poche
        amount: Montant à réserver
        cycle_id: ID du cycle de trading
    
    Returns:
        Statut de la réservation
    """
    if amount <= 0:
        raise HTTPException(status_code=400, detail="Montant invalide")
    
    # Valider le type de poche
    if pocket_type not in ['active', 'buffer', 'safety']:
        raise HTTPException(status_code=400, detail=f"Type de poche invalide: {pocket_type}")
    
    success = pocket_manager.reserve_funds(pocket_type, amount, cycle_id)
    
    if not success:
        raise HTTPException(status_code=400, detail="Échec de la réservation des fonds")
    
    return {"status": "success", "message": f"{amount} réservés dans la poche {pocket_type}"}

@app.post("/pockets/{pocket_type}/release", status_code=200)
async def release_funds(
    pocket_type: str = Path(..., description="Type de poche (active, buffer, safety)"),
    amount: float = Query(..., description="Montant à libérer"),
    cycle_id: str = Query(..., description="ID du cycle de trading"),
    pocket_manager: PocketManager = Depends(get_pocket_manager)
):
    """
    Libère des fonds réservés dans une poche.
    
    Args:
        pocket_type: Type de poche
        amount: Montant à libérer
        cycle_id: ID du cycle de trading
    
    Returns:
        Statut de la libération
    """
    if amount <= 0:
        raise HTTPException(status_code=400, detail="Montant invalide")
    
    # Valider le type de poche
    if pocket_type not in ['active', 'buffer', 'safety']:
        raise HTTPException(status_code=400, detail=f"Type de poche invalide: {pocket_type}")
    
    success = pocket_manager.release_funds(pocket_type, amount, cycle_id)
    
    if not success:
        raise HTTPException(status_code=400, detail="Échec de la libération des fonds")
    
    return {"status": "success", "message": f"{amount} libérés dans la poche {pocket_type}"}

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
        response.headers["Cache-Control"] = "public, max-age=300"  # 5 minutes
    
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
        response.headers["Cache-Control"] = "public, max-age=300"  # 5 minutes
    
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
        response.headers["Cache-Control"] = "public, max-age=300"  # 5 minutes
    
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
        response.headers["Cache-Control"] = "public, max-age=300"  # 5 minutes
    
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

@app.post("/pockets/reconcile", status_code=200)
async def reconcile_pockets(
    background_tasks: BackgroundTasks,
    pocket_manager: PocketManager = Depends(get_pocket_manager)
):
    """
    Force la réconciliation complète des poches avec les trades actifs.
    Recalcule les valeurs utilisées, disponibles et le nombre de cycles actifs.
    Exécuté en arrière-plan pour ne pas bloquer la réponse.
    """
    # Fonction d'arrière-plan
    def reconcile_task():
        try:
            # Récupérer la valeur totale du portefeuille
            portfolio = PortfolioModel()
            summary = portfolio.get_portfolio_summary()
            total_value = summary.total_value
            
            # Synchroniser les poches avec les trades actifs
            success1 = pocket_manager.sync_with_trades()
            
            # Réallouer les poches selon les pourcentages configurés
            success2 = pocket_manager.update_pockets_allocation(total_value)
            
            # Mettre à jour le nombre de cycles actifs
            success3 = pocket_manager.recalculate_active_cycles()
            
            logger.info(f"✅ Réconciliation des poches terminée: sync={success1}, allocation={success2}, recalculation={success3}")
            
            portfolio.close()
        except Exception as e:
            logger.error(f"❌ Erreur lors de la réconciliation des poches: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Ajouter la tâche en arrière-plan
    background_tasks.add_task(reconcile_task)
    
    return {
        "status": "accepted", 
        "message": "Réconciliation des poches lancée en arrière-plan"
    }

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