"""
Module de démarrage pour le service Portfolio.
Contient la logique de synchronisation initiale avec Binance.
"""

import logging

from shared.src.config import BINANCE_API_KEY, BINANCE_SECRET_KEY

from .binance_account_manager import BinanceAccountManager, BinanceApiError
from .models import DBManager, PortfolioModel, SharedCache

logger = logging.getLogger(__name__)


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

        SharedCache.clear("latest_balances")
        SharedCache.clear("portfolio_summary")
        logger.info("♻️ Cache invalidé après synchronisation")
    except BinanceApiError:
        logger.exception("❌ Erreur API Binance")
    except Exception:
        logger.exception("❌ Erreur de synchronisation initiale Binance")
