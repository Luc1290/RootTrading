"""
Module de synchronisation pour le service Portfolio.
Gère les tâches de synchronisation continue avec Binance et la base de données.
"""

import asyncio
import threading
import logging
from models import PortfolioModel, DBManager, SharedCache
from binance_account_manager import BinanceAccountManager
from shared.src.config import BINANCE_API_KEY, BINANCE_SECRET_KEY

logger = logging.getLogger(__name__)


def start_sync_tasks():
    """
    Démarre les tâches de synchronisation en arrière-plan.
    """
    logger.info("▶️ Démarrage des tâches de synchronisation continue")
    threading.Thread(target=binance_sync_task, daemon=True).start()
    threading.Thread(target=database_sync_task, daemon=True).start()


def binance_sync_task():
    """
    Tâche de synchronisation Binance en continu.
    Exécute dans un thread séparé.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(sync_binance_forever())


async def sync_binance_forever():
    """
    Tâche asynchrone de synchronisation continue des balances Binance.
    Met à jour les balances toutes les 60 secondes.
    """
    while True:
        try:
            account_manager = BinanceAccountManager(BINANCE_API_KEY, BINANCE_SECRET_KEY)
            balances = account_manager.calculate_asset_values()

            db = DBManager()
            portfolio = PortfolioModel(db)
            success = portfolio.update_balances(balances)

            if success:
                SharedCache.clear("latest_balances")
                SharedCache.clear("portfolio_summary")
                logger.info(
                    f"✅ {len(balances)} balances sauvegardées et cache invalidé."
                )
            else:
                logger.error("❌ Échec de la sauvegarde des balances Binance.")

            portfolio.close()
            db.close()
            logger.debug(f"🔄 Balances Binance synchronisées ({len(balances)} actifs)")
        except Exception as e:
            logger.error(f"❌ Erreur sync Binance: {e}")
        await asyncio.sleep(60)


def database_sync_task():
    """
    Tâche de synchronisation de la base de données.
    Exécute dans un thread séparé.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(sync_db_forever())


async def sync_db_forever():
    """
    Tâche asynchrone de synchronisation continue de la base de données.
    Met à jour les statistiques du portfolio toutes les 60 secondes.
    """
    sync_count = 0
    while True:
        try:
            db = DBManager()
            portfolio = PortfolioModel(db)

            # Récupérer la valeur totale du portfolio
            summary = portfolio.get_portfolio_summary()
            if summary:
                # Afficher le portfolio complet toutes les 10 sync (10 minutes)
                if sync_count % 10 == 0:
                    print("\n" + "=" * 60)
                    print("📊 PORTFOLIO ROOTTRADING - Mise à jour")
                    print("=" * 60)
                    print(f"💎 Total Portfolio: {summary.total_value:.2f} USDC")
                    print(f"📊 Trades Actifs: {summary.active_trades}")
                    print("-" * 60)
                    print("📈 Balances par Crypto:")

                    # Trier les balances par valeur décroissante
                    balances_sorted = sorted(
                        summary.balances, key=lambda x: x.value_usdc or 0, reverse=True
                    )

                    for balance in balances_sorted:
                        if balance.total > 0:
                            percentage = (
                                (balance.value_usdc / summary.total_value) * 100
                                if summary.total_value > 0
                                else 0
                            )
                            print(
                                f"  {balance.asset:6} | {balance.total:>12.8f} | {balance.value_usdc:>8.2f} USDC ({percentage:>5.1f}%)"
                            )

                    print("=" * 60)
                else:
                    # Affichage simplifié pour les autres sync
                    logger.info(
                        f"🔄 Portfolio: {summary.total_value:.2f} USDC | Trades: {summary.active_trades}"
                    )

                sync_count += 1
            else:
                logger.warning("⚠️ Impossible de récupérer le résumé du portfolio")

            portfolio.close()
            db.close()
            logger.debug("🔄 Synchronisation des poches DB terminée")
        except Exception as e:
            logger.error(f"❌ Erreur sync DB: {e}")
        await asyncio.sleep(60)
