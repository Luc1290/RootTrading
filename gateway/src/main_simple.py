"""
Point d'entrée principal SIMPLIFIÉ pour le microservice Gateway.
ARCHITECTURE PROPRE : Récupération de données OHLCV brutes uniquement.
AUCUN calcul d'indicateur - transmission pure vers le dispatcher.
"""

import asyncio
import logging
import signal
import time
from collections.abc import Callable

from aiohttp import web

from shared.src.config import SYMBOLS

from .gap_detector import GapDetector
from .simple_binance_ws import SimpleBinanceWebSocket
from .simple_data_fetcher import SimpleDataFetcher

# Configuration du logging centralisée
from shared.logging_config import setup_logging
logger = setup_logging("gateway", log_level="INFO")


class SmartDataFetcher:
    """
    Fetcher intelligent qui utilise GapDetector pour ne charger que les données manquantes.
    """

    def __init__(self):
        self.gap_detector = GapDetector()
        self.simple_fetcher = SimpleDataFetcher()
        self.running = False

    async def start(self):
        """Démarre le fetcher intelligent avec détection de gaps."""
        self.running = True
        logger.info("🧠 Démarrage du SmartDataFetcher...")

        try:
            # 1. Initialiser le gap detector
            await self.gap_detector.initialize()

            # 2. Détecter les gaps pour tous les symboles (24h lookback)
            logger.info("🔍 Détection des gaps en cours...")
            all_gaps = await self.gap_detector.detect_all_gaps(
                symbols=SYMBOLS, lookback_hours=24
            )

            # 3. Analyser les résultats
            total_gaps = sum(
                len(timeframe_gaps)
                for symbol_gaps in all_gaps.values()
                for timeframe_gaps in symbol_gaps.values()
            )

            if total_gaps == 0:
                logger.info(
                    "✅ Aucun gap détecté - Base de données synchronisée")
                logger.info("🎯 Mode: WebSocket temps réel uniquement")
                return  # Pas besoin de fetch, on est sync

            logger.warning(
                f"📊 {total_gaps} gaps détectés - Remplissage nécessaire")

            # 4. Générer un plan de remplissage optimisé
            filling_plan = self.gap_detector.generate_gap_filling_plan(
                all_gaps)
            estimated_time = self.gap_detector.estimate_fill_time(filling_plan)

            logger.info(
                f"⏱️ Temps estimé pour synchronisation: {estimated_time:.1f}s")

            # 5. Exécuter le remplissage intelligent
            await self._execute_smart_fill(filling_plan)

            logger.info("✅ Synchronisation terminée - Passage en mode live")

        except Exception:
            logger.exception("❌ Erreur SmartDataFetcher")
            # Fallback: utiliser le fetcher classique
            logger.info("🔄 Fallback vers fetcher classique")
            await self.simple_fetcher.start()

    async def _execute_smart_fill(self, filling_plan):
        """Exécute le plan de remplissage de façon optimisée."""
        total_requests = sum(
            len(periods)
            for symbol_plan in filling_plan.values()
            for periods in symbol_plan.values()
        )
        completed = 0

        logger.info(
            f"🚀 Début du remplissage intelligent ({total_requests} requêtes)")

        for symbol, timeframe_plan in filling_plan.items():
            for timeframe, periods in timeframe_plan.items():
                for start_time, end_time in periods:
                    try:
                        # Récupérer les données pour cette période spécifique
                        await self.simple_fetcher._fetch_period_data(
                            symbol, timeframe, start_time, end_time
                        )
                        completed += 1

                        if completed % 10 == 0:
                            progress = (completed / total_requests) * 100
                            logger.info(
                                f"📈 Progression: {completed}/{total_requests} ({progress:.1f}%)"
                            )

                        # Respecter les rate limits
                        await asyncio.sleep(0.1)

                    except Exception:
                        logger.exception(
                            "❌ Erreur remplissage {symbol} {timeframe}")

        logger.info(
            f"✅ Remplissage terminé: {completed}/{total_requests} requêtes réussies"
        )

    async def stop(self):
        """Arrête le fetcher intelligent."""
        self.running = False
        if self.gap_detector:
            await self.gap_detector.close()
        await self.simple_fetcher.stop()
        logger.info("🛑 SmartDataFetcher arrêté")


# Variables globales
data_fetcher = None
ws_client = None
gap_detector = None
running = True
start_time = time.time()

# Routes pour le serveur HTTP
routes = web.RouteTableDef()


@routes.get("/health")
async def health_check(request):
    """Point de terminaison pour vérifier l'état du service."""
    uptime = time.time() - start_time
    return web.json_response(
        {
            "status": "ok",
            "timestamp": time.time(),
            "uptime": uptime,
            "mode": "active" if running else "stopping",
            "symbols": SYMBOLS,
            "intervals": ["1m", "3m", "5m", "15m", "1h", "1d"],
            "architecture": "multi_timeframe_clean_data",
        }
    )


@routes.get("/diagnostic")
async def diagnostic(request):
    """Point de terminaison pour le diagnostic du service."""
    global data_fetcher

    # État du fetcher
    fetcher_status = {
        "running": data_fetcher.running if data_fetcher else False,
        "symbols_count": len(SYMBOLS),
        "timeframes": ["1m", "3m", "5m", "15m", "1h", "1d"],
        "data_type": "raw_ohlcv_only",
    }

    is_operational = data_fetcher is not None and data_fetcher.running

    diagnostic_info = {
        "status": "operational" if is_operational else "stopped",
        "timestamp": time.time(),
        "uptime": time.time() - start_time,
        "data_fetcher": fetcher_status,
        "symbols": SYMBOLS,
        "intervals": ["1m", "3m", "5m", "15m", "1h", "1d"],
        "architecture": "multi_timeframe_clean_gateway",
    }

    return web.json_response(diagnostic_info)


async def start_http_server():
    """Démarre le serveur HTTP pour les endpoints de santé."""
    app = web.Application()
    app.add_routes(routes)

    runner = web.AppRunner(app)
    await runner.setup()

    # Utiliser le port 5010 pour l'API REST
    site = web.TCPSite(runner, "0.0.0.0", 5010)
    await site.start()

    logger.info("✅ Serveur HTTP démarré sur le port 5010")
    return runner


async def shutdown(signal_type, loop):
    """
    Gère l'arrêt propre du service en cas de signal (SIGINT, SIGTERM).
    """
    global running, data_fetcher, ws_client

    logger.info(f"Signal {signal_type.name} reçu, arrêt en cours...")
    running = False

    try:
        # Arrêter les services avec timeout
        shutdown_tasks: list[asyncio.Task] = []

        if ws_client is not None:
            logger.info("Arrêt WebSocket client...")
            shutdown_tasks.append(
                asyncio.wait_for(
                    ws_client.stop(),
                    timeout=5.0))

        if data_fetcher is not None:
            logger.info("Arrêt Data fetcher...")
            shutdown_tasks.append(
                asyncio.wait_for(
                    data_fetcher.stop(),
                    timeout=5.0))

        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        logger.info("Services arrêtés proprement")

    except asyncio.TimeoutError:
        logger.warning("Timeout lors de l'arrêt des services - arrêt forcé")
    except Exception:
        logger.exception("Erreur lors de l'arrêt des services")
    finally:
        # Ne pas appeler loop.stop() ici - laissons le main loop se terminer
        # naturellement
        logger.info("Shutdown terminé")


async def main():
    """
    Fonction principale qui démarre le Gateway SIMPLIFIÉ.
    """
    global data_fetcher, ws_client

    logger.info(
        "🚀 Démarrage du service Gateway SIMPLE (données brutes uniquement)...")
    logger.info(
        f"Configuration: {', '.join(SYMBOLS)} @ ['1m', '3m', '5m', '15m', '1h', '1d']"
    )
    logger.info("🎯 Architecture: AUCUN calcul d'indicateur - transmission pure")

    # Démarrer le serveur HTTP
    http_runner = await start_http_server()

    try:
        # Créer les services intelligents
        data_fetcher = SmartDataFetcher()
        ws_client = SimpleBinanceWebSocket(
            symbols=SYMBOLS, intervals=["1m", "3m", "5m", "15m", "1h", "1d"]
        )

        logger.info("📡 Initialisation des services intelligents...")
        logger.info("🧠 Mode: Détection de gaps + WebSocket temps réel")

        # 1. D'abord synchroniser les données manquantes
        logger.info("🔍 Phase 1: Synchronisation intelligente...")
        await data_fetcher.start()

        # 2. Ensuite démarrer le WebSocket temps réel en mode continu
        logger.info("🚀 Phase 2: Démarrage WebSocket temps réel...")
        # Note: WebSocket fonctionne en continu, pas de synchronisation
        # nécessaire
        await ws_client.start()

        # 3. Attendre le signal d'arrêt
        logger.info("✅ Gateway démarré - en attente du signal d'arrêt...")
        while running:
            await asyncio.sleep(1)  # Vérifier le statut toutes les secondes

        logger.info("📟 Signal d'arrêt reçu - fermeture en cours...")

    except Exception:
        logger.exception("❌ Erreur critique dans le Gateway")
    finally:
        if ws_client:
            await ws_client.stop()
        if data_fetcher:
            await data_fetcher.stop()

        # Arrêter le serveur HTTP
        await http_runner.cleanup()

        logger.info("Service Gateway (simple) terminé")


if __name__ == "__main__":
    # Obtenir la boucle asyncio
    loop = asyncio.get_event_loop()

    # Configurer les gestionnaires de signaux pour l'arrêt propre
    for sig in (signal.SIGINT, signal.SIGTERM):

        def make_signal_handler(
                signal_type: signal.Signals) -> Callable[[], None]:
            def handler() -> None:
                asyncio.create_task(shutdown(signal_type, loop))

            return handler

        loop.add_signal_handler(sig, make_signal_handler(sig))

    try:
        # Exécuter la fonction principale
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Interruption clavier détectée")
    except Exception:
        logger.exception("Erreur lors de l'exécution")
    finally:
        # Arrêt propre de toutes les tâches
        try:
            # Annuler toutes les tâches en cours
            pending_tasks = [
                task for task in asyncio.all_tasks(loop) if not task.done()
            ]
            if pending_tasks:
                logger.info(
                    f"Annulation de {len(pending_tasks)} tâches en cours...")
                for task in pending_tasks:
                    task.cancel()

                # Attendre que toutes les tâches se terminent proprement
                loop.run_until_complete(
                    asyncio.gather(*pending_tasks, return_exceptions=True)
                )
        except Exception as e:
            logger.warning(f"Erreur lors de l'annulation des tâches: {e}")
        finally:
            # Fermer la boucle asyncio
            if not loop.is_closed():
                loop.close()
            logger.info("Boucle asyncio fermée proprement")
