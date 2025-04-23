"""
API REST pour le service Portfolio.
Expose les endpoints pour accéder aux données du portefeuille et des poches.
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

from fastapi import FastAPI, Query, Path, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Importer les modules partagés
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.schemas import AssetBalance, PortfolioSummary, PocketSummary

from portfolio.src.models import PortfolioModel, DBManager
from portfolio.src.pockets import PocketManager

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

# Créer l'application FastAPI
app = FastAPI(
    title="RootTrading Portfolio API",
    description="API pour la gestion du portefeuille et des poches de capital",
    version="1.0.0"
)

# Configurer CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "uptime": "unknown"  # TODO: Ajouter l'uptime réel
    }

@app.get("/summary", response_model=PortfolioSummary)
async def get_portfolio_summary(portfolio: PortfolioModel = Depends(get_portfolio_model)):
    """
    Récupère un résumé du portefeuille.
    
    Returns:
        Résumé du portefeuille
    """
    summary = portfolio.get_portfolio_summary()
    
    if not summary:
        raise HTTPException(status_code=404, detail="Aucune donnée de portefeuille trouvée")
    
    return summary

@app.get("/balances", response_model=List[AssetBalance])
async def get_balances(portfolio: PortfolioModel = Depends(get_portfolio_model)):
    """
    Récupère les soldes actuels du portefeuille.
    
    Returns:
        Liste des soldes par actif
    """
    balances = portfolio.get_latest_balances()
    
    if not balances:
        raise HTTPException(status_code=404, detail="Aucun solde trouvé")
    
    return balances

@app.get("/pockets", response_model=List[PocketSummary])
async def get_pockets(pocket_manager: PocketManager = Depends(get_pocket_manager)):
    """
    Récupère l'état actuel des poches de capital.
    
    Returns:
        Liste des poches de capital
    """
    pockets = pocket_manager.get_pockets()
    
    if not pockets:
        raise HTTPException(status_code=404, detail="Aucune poche trouvée")
    
    return pockets

@app.put("/pockets/sync")
async def sync_pockets(pocket_manager: PocketManager = Depends(get_pocket_manager)):
    """
    Synchronise les poches avec les trades actifs.
    
    Returns:
        Statut de la synchronisation
    """
    success = pocket_manager.sync_with_trades()
    
    if not success:
        raise HTTPException(status_code=500, detail="Échec de la synchronisation des poches")
    
    return {"status": "success", "message": "Poches synchronisées avec succès"}

@app.put("/pockets/allocation")
async def update_pocket_allocation(
    total_value: float = Query(..., description="Valeur totale du portefeuille"),
    pocket_manager: PocketManager = Depends(get_pocket_manager)
):
    """
    Met à jour l'allocation des poches en fonction de la valeur totale.
    """
    try:
        # Modification ici: utiliser une valeur minimale si total_value est 0
        if total_value <= 0:
            logger.info(f"Valeur totale reçue: {total_value}, utilisation d'une valeur minimale de 100.0")
            total_value = 100.0  # Valeur par défaut pour l'initialisation
        
        success = pocket_manager.update_pockets_allocation(total_value)
        
        if not success:
            raise HTTPException(status_code=500, detail="Échec de la mise à jour de l'allocation")
        
        return {"status": "success", "message": "Allocation mise à jour avec succès"}
    except Exception as e:
        logger.error(f"Erreur dans update_pocket_allocation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Erreur interne: {str(e)}")

@app.post("/pockets/{pocket_type}/reserve")
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
    
    success = pocket_manager.reserve_funds(pocket_type, amount, cycle_id)
    
    if not success:
        raise HTTPException(status_code=400, detail="Échec de la réservation des fonds")
    
    return {"status": "success", "message": f"{amount} réservés dans la poche {pocket_type}"}

@app.post("/pockets/{pocket_type}/release")
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
    
    success = pocket_manager.release_funds(pocket_type, amount, cycle_id)
    
    if not success:
        raise HTTPException(status_code=400, detail="Échec de la libération des fonds")
    
    return {"status": "success", "message": f"{amount} libérés dans la poche {pocket_type}"}

@app.get("/trades", response_model=TradeHistoryResponse)
async def get_trade_history(
    page: int = Query(1, description="Numéro de page"),
    page_size: int = Query(50, description="Taille de la page"),
    symbol: Optional[str] = Query(None, description="Filtrer par symbole"),
    strategy: Optional[str] = Query(None, description="Filtrer par stratégie"),
    start_date: Optional[str] = Query(None, description="Date de début (format ISO)"),
    end_date: Optional[str] = Query(None, description="Date de fin (format ISO)"),
    portfolio: PortfolioModel = Depends(get_portfolio_model)
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
    
    # TODO: Ajouter le comptage total pour la pagination
    total_count = len(trades)  # Approximation, à remplacer par un count réel
    
    return TradeHistoryResponse(
        trades=trades,
        total_count=total_count,
        page=page,
        page_size=page_size
    )

@app.get("/performance/{period}", response_model=PerformanceResponse)
async def get_performance(
    period: str = Path(..., description="Période ('daily', 'weekly', 'monthly')"),
    limit: int = Query(30, description="Nombre de périodes à récupérer"),
    portfolio: PortfolioModel = Depends(get_portfolio_model)
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
    
    return PerformanceResponse(data=stats, period=period)

@app.get("/performance/strategy", response_model=Dict[str, Any])
async def get_strategy_performance(portfolio: PortfolioModel = Depends(get_portfolio_model)):
    """
    Récupère les performances par stratégie.
    
    Returns:
        Performances par stratégie
    """
    performance = portfolio.get_strategy_performance()
    return {"data": performance}

@app.get("/performance/symbol", response_model=Dict[str, Any])
async def get_symbol_performance(portfolio: PortfolioModel = Depends(get_portfolio_model)):
    """
    Récupère les performances par symbole.
    
    Returns:
        Performances par symbole
    """
    performance = portfolio.get_symbol_performance()
    return {"data": performance}

@app.post("/balances/update")
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
    
    return {"status": "success", "message": f"{len(balances)} soldes mis à jour"}

@app.get("/_debug/db_test")
async def test_db_connection():
    """
    Endpoint de diagnostic pour tester la connexion à la base de données.
    """
    try:
        db = DBManager()
        result = db.execute_query("SELECT 1 as test", fetch_one=True)
        db.close()
        
        if result and result.get('test') == 1:
            return {"status": "ok", "message": "Connexion à la base de données réussie"}
        else:
            return {"status": "error", "message": "Problème de connexion à la base de données"}
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