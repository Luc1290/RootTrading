import asyncio
import logging
from portfolio.src.sync import start_sync_tasks
from portfolio.src.redis_subscriber import start_redis_subscriptions
from portfolio.src.models import DBManager
from portfolio.src.binance_account_manager import BinanceAccountManager
from shared.src.config import BINANCE_API_KEY, BINANCE_SECRET_KEY

logger = logging.getLogger(__name__)

async def on_startup():
    """Tâches de démarrage"""
    logger.info("🚀 Initialisation du Portfolio...")
    await initial_sync_binance()
    start_sync_tasks()
    start_redis_subscriptions()

async def initial_sync_binance():
    """Force une synchronisation avec Binance à l'initialisation"""
    try:
        if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
            logger.warning("Clés API Binance manquantes.")
            return

        account_manager = BinanceAccountManager(BINANCE_API_KEY, BINANCE_SECRET_KEY)
        balances = account_manager.calculate_asset_values()
        if not balances:
            logger.warning("Aucune balance reçue depuis Binance.")
            return

        db = DBManager()
        from portfolio.src.models import PortfolioModel
        portfolio = PortfolioModel(db)
        portfolio.update_balances(balances)
        portfolio.close()
        db.close()
        logger.info(f"✅ Synchronisation initiale Binance réussie ({len(balances)} actifs)")
    except Exception as e:
        logger.error(f"❌ Erreur de synchronisation initiale Binance: {e}")
