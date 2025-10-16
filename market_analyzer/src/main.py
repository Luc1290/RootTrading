"""
Market Analyzer Service - Point d'entrée principal
Service qui calcule TOUS les indicateurs techniques à partir des données OHLCV.
Architecture propre : Gateway → market_data → Market Analyzer → analyzer_data
"""

import asyncio
import signal
import sys
import time
from pathlib import Path

from aiohttp import web

from market_analyzer.src.data_listener import DataListener
from shared.src.config import SYMBOLS

# Ajouter les chemins pour les imports
sys.path.append(str(Path(__file__).parent / "../../"))


# Configuration du logging centralisée
from shared.logging_config import setup_logging

logger = setup_logging("market_analyzer", log_level="INFO")

# Variables globales
data_listener = None
running = True
start_time = time.time()

# Routes pour l'API de diagnostic
routes = web.RouteTableDef()


@routes.get("/health")
async def health_check(_request):
    """Point de terminaison santé du service."""
    uptime = time.time() - start_time
    return web.json_response(
        {
            "status": "ok",
            "timestamp": time.time(),
            "uptime": uptime,
            "service": "market_analyzer",
            "mode": "active" if running else "stopping",
            "symbols": SYMBOLS,
            "architecture": "clean_calculation_engine",
        }
    )


@routes.get("/stats")
async def get_stats(_request):
    """Statistiques du Market Analyzer."""
    if data_listener:
        stats = await data_listener.get_stats()
        return web.json_response(stats)

    return web.json_response(
        {"error": "DataListener not initialized", "status": "not_running"},
        status=503,
    )


@routes.post("/process-historical")
async def process_historical(request):
    """API pour traiter les données historiques."""
    if not data_listener:
        return web.json_response({"error": "DataListener not initialized"}, status=503)

    try:
        # Parser les paramètres
        data = (
            await request.json() if request.content_type == "application/json" else {}
        )
        symbol = data.get("symbol")
        timeframe = data.get("timeframe")
        limit = data.get("limit", 1000)

        # Lancer le traitement optimisé en arrière-plan
        _task = asyncio.create_task(
            data_listener.process_historical_optimized(symbol, timeframe, limit)
        )

        return web.json_response(
            {
                "status": "started",
                "message": f"Traitement historique lancé (limit: {limit})",
                "symbol": symbol,
                "timeframe": timeframe,
            }
        )

    except Exception as e:
        return web.json_response(
            {"error": f"Erreur lancement traitement: {e}"}, status=500
        )


@routes.get("/coverage")
async def get_coverage(_request):
    """Analyse de couverture des données."""
    if not data_listener:
        return web.json_response({"error": "DataListener not initialized"}, status=503)

    try:
        # Statistiques par symbole/timeframe
        coverage_query = """
            SELECT
                md.symbol,
                md.timeframe,
                COUNT(md.*) as market_data_count,
                COUNT(ad.*) as analyzer_data_count,
                ROUND((COUNT(ad.*)::FLOAT / COUNT(md.*)) * 100, 2) as coverage_percent
            FROM market_data md
            LEFT JOIN analyzer_data ad ON (
                md.symbol = ad.symbol AND
                md.timeframe = ad.timeframe AND
                md.time = ad.time
            )
            GROUP BY md.symbol, md.timeframe
            ORDER BY md.symbol, md.timeframe
        """

        async with data_listener.db_pool.acquire() as conn:
            rows = await conn.fetch(coverage_query)

        coverage_data = []
        for row in rows:
            coverage_data.append(
                {
                    "symbol": row["symbol"],
                    "timeframe": row["timeframe"],
                    "market_data_count": row["market_data_count"],
                    "analyzer_data_count": row["analyzer_data_count"],
                    "coverage_percent": float(row["coverage_percent"]),
                    "missing_count": row["market_data_count"]
                    - row["analyzer_data_count"],
                }
            )

        return web.json_response(
            {
                "coverage_by_asset": coverage_data,
                "total_assets": len({row["symbol"] for row in coverage_data}),
                "total_timeframes": len({row["timeframe"] for row in coverage_data}),
            }
        )

    except Exception as e:
        return web.json_response(
            {"error": f"Erreur analyse couverture: {e}"}, status=500
        )


async def start_http_server():
    """Démarre le serveur HTTP pour l'API."""
    app = web.Application()
    app.add_routes(routes)

    runner = web.AppRunner(app)
    await runner.setup()

    # Port 5020 pour le Market Analyzer
    site = web.TCPSite(runner, "0.0.0.0", 5020)
    await site.start()

    logger.info("✅ API HTTP démarrée sur le port 5020")
    return runner


async def shutdown(signal_type, loop):
    """Gère l'arrêt propre du service."""
    global running

    logger.info(f"Signal {signal_type.name} reçu, arrêt en cours...")
    running = False

    # Arrêter le DataListener AVANT de fermer la boucle
    if data_listener:
        try:
            await data_listener.stop()
            logger.info("✅ DataListener arrêté proprement")
        except Exception:
            logger.exception("❌ Erreur arrêt DataListener")

    # Attendre plus longtemps pour que toutes les connexions DB se ferment
    await asyncio.sleep(1.0)

    # Annuler toutes les tâches pendantes
    try:
        pending_tasks = [
            task
            for task in asyncio.all_tasks(loop)
            if not task.done() and task != asyncio.current_task()
        ]
        if pending_tasks:
            logger.info(f"Annulation de {len(pending_tasks)} tâches pendantes...")
            for task in pending_tasks:
                task.cancel()
            # Attendre que les tâches se terminent
            await asyncio.gather(*pending_tasks, return_exceptions=True)
    except Exception as e:
        logger.warning(f"Erreur lors de l'annulation des tâches: {e}")

    logger.info("Market Analyzer terminé")

    # NE PAS appeler loop.stop() ici - laisser le main() se terminer
    # naturellement


async def main():
    """Fonction principale du Market Analyzer."""
    global data_listener

    logger.info("🚀 Démarrage du Market Analyzer Service")
    logger.info("🎯 Architecture: Calcul TOUS les indicateurs depuis market_data")
    logger.info(f"📊 Symboles surveillés: {', '.join(SYMBOLS)}")

    # Démarrer le serveur HTTP
    http_runner = await start_http_server()

    try:
        # Initialiser le DataListener
        data_listener = DataListener()
        await data_listener.initialize()

        logger.info("✅ Market Analyzer initialisé")

        # Démarrer l'écoute temps réel IMMÉDIATEMENT (non-bloquant)
        logger.info("🎧 Démarrage de l'écoute temps réel...")
        _listening_task = asyncio.create_task(data_listener.start_listening())

        # Proposer de traiter l'historique en parallèle
        logger.info("🔍 Vérification de la couverture des données...")
        stats = await data_listener.get_stats()

        if stats["missing_analyses"] > 0:
            logger.info(
                f"⚠️ {stats['missing_analyses']} données non analysées détectées"
            )
            logger.info("💡 Démarrage du traitement historique automatique...")

            # LIMITÉ à 100k données max au démarrage pour éviter saturation DB
            await data_listener.process_historical_optimized(limit=100000)
        else:
            logger.info("✅ Toutes les données sont analysées")

        # Afficher les statistiques finales
        final_stats = await data_listener.get_stats()
        logger.info(
            f"📊 Couverture: {final_stats['coverage_percent']}% ({final_stats['total_analyzer_data']}/{final_stats['total_market_data']})"
        )

        # Attendre que l'écoute temps réel continue ou que l'arrêt soit demandé
        logger.info("✅ Traitement historique terminé - écoute temps réel active")

        # Boucle d'attente au lieu d'attendre le listening_task qui peut ne
        # jamais se terminer
        while running:
            await asyncio.sleep(1)

        logger.info("📟 Arrêt demandé - terminaison en cours...")

    except Exception:
        logger.exception("❌ Erreur critique")
    finally:
        if data_listener:
            await data_listener.stop()

        # Arrêter le serveur HTTP
        await http_runner.cleanup()

        logger.info("Market Analyzer terminé")


if __name__ == "__main__":
    # Obtenir la boucle asyncio
    loop = asyncio.get_event_loop()

    # Configurer les gestionnaires de signaux
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig, lambda s=sig: asyncio.create_task(shutdown(s, loop))  # type: ignore
        )

    try:
        # Exécuter la fonction principale
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Interruption clavier détectée")
    finally:
        # Fermer la boucle asyncio
        loop.close()
        logger.info("Boucle asyncio fermée")
