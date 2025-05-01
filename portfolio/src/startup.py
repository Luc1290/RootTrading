import asyncio
import logging

from portfolio.src.sync import start_sync_tasks
from portfolio.src.redis_subscriber import start_redis_subscriptions
from portfolio.src.models import DBManager, SharedCache
from portfolio.src.binance_account_manager import BinanceAccountManager, BinanceApiError
from portfolio.src.models import PortfolioModel
from shared.src.config import BINANCE_API_KEY, BINANCE_SECRET_KEY

logger = logging.getLogger(__name__)

async def on_startup():
    """Tâches de démarrage"""
    logger.info("🚀 Initialisation du Portfolio...")

    await initial_sync_binance()

    logger.info("▶️ Démarrage des tâches de synchronisation continue")
    start_sync_tasks()

    logger.info("▶️ Abonnement aux canaux Redis")
    start_redis_subscriptions()

async def initial_sync_binance():
    """Force une synchronisation avec Binance à l'initialisation"""
    try:
        if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
            logger.warning("Clés API Binance manquantes.")
            return

        logger.info("⏳ Synchronisation initiale des balances Binance...")
        account_manager = BinanceAccountManager(BINANCE_API_KEY, BINANCE_SECRET_KEY)
        balances = account_manager.calculate_asset_values()

        if not balances:
            logger.warning("Aucune balance reçue depuis Binance.")
            return

        db = DBManager()
        portfolio = PortfolioModel(db)
        success = portfolio.update_balances(balances)

        if success:
            logger.info(f"✅ {len(balances)} balances sauvegardées avec succès")
        else:
            logger.error("❌ update_balances() a échoué")

        portfolio.close()
        db.close()

        SharedCache.clear('latest_balances')
        SharedCache.clear('portfolio_summary')
        logger.info("♻️ Cache invalidé après synchronisation")
    except BinanceApiError as e:
        logger.error(f"❌ Erreur API Binance: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Erreur de synchronisation initiale Binance: {e}")
