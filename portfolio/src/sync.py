import asyncio
import threading
import logging
from portfolio.src.models import PortfolioModel, DBManager, SharedCache
from portfolio.src.pockets import PocketManager
from portfolio.src.binance_account_manager import BinanceAccountManager
from shared.src.config import BINANCE_API_KEY, BINANCE_SECRET_KEY

logger = logging.getLogger(__name__)

def start_sync_tasks():
    threading.Thread(target=binance_sync_task, daemon=True).start()
    threading.Thread(target=database_sync_task, daemon=True).start()

def binance_sync_task():
    """Synchronisation Binance en continu"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(sync_binance_forever())

async def sync_binance_forever():
    """Task async qui sync les balances Binance"""
    while True:
        try:
            account_manager = BinanceAccountManager(BINANCE_API_KEY, BINANCE_SECRET_KEY)
            balances = account_manager.calculate_asset_values()
            db = DBManager()
            portfolio = PortfolioModel(db)
            success = portfolio.update_balances(balances)
            if success:
                SharedCache.clear('latest_balances')
                SharedCache.clear('portfolio_summary')
                logger.info(f"✅ {len(balances)} balances sauvegardées et cache invalidé.")
            else:
                logger.error("❌ Échec de la sauvegarde des balances Binance.")
            portfolio.close()
            db.close()
            logger.info(f"🔄 Balances Binance synchronisées ({len(balances)} actifs)")
        except Exception as e:
            logger.error(f"❌ Erreur sync Binance: {e}")
        await asyncio.sleep(60)

def database_sync_task():
    """Synchronisation poches et DB"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(sync_db_forever())

async def sync_db_forever():
    """Task async pour synchroniser les poches"""
    while True:
        try:
            db = DBManager()
            pockets = PocketManager(db)
            pockets.sync_with_trades()
            db.close()
            logger.info("🔄 Synchronisation des poches DB")
        except Exception as e:
            logger.error(f"❌ Erreur sync DB: {e}")
        await asyncio.sleep(60)
