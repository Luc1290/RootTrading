"""
Point d'entrée principal SIMPLIFIÉ pour le microservice Gateway.
ARCHITECTURE PROPRE : Récupération de données OHLCV brutes uniquement.
AUCUN calcul d'indicateur - transmission pure vers le dispatcher.
"""
import asyncio
import logging
import signal
import sys
import os
import time
from aiohttp import web

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from simple_data_fetcher import SimpleDataFetcher
from simple_binance_ws import SimpleBinanceWebSocket
from shared.src.config import SYMBOLS, INTERVAL

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gateway.log')
    ]
)
logger = logging.getLogger("gateway")

# Variables globales
data_fetcher = None
ws_client = None
running = True
start_time = time.time()

# Routes pour le serveur HTTP
routes = web.RouteTableDef()

@routes.get('/health')
async def health_check(request):
    """Point de terminaison pour vérifier l'état du service."""
    uptime = time.time() - start_time
    return web.json_response({
        "status": "ok",
        "timestamp": time.time(),
        "uptime": uptime,
        "mode": "active" if running else "stopping",
        "symbols": SYMBOLS,
        "intervals": ['1m', '3m', '5m', '15m', '1d'],
        "architecture": "multi_timeframe_clean_data"
    })

@routes.get('/diagnostic')
async def diagnostic(request):
    """Point de terminaison pour le diagnostic du service."""
    global data_fetcher
    
    # État du fetcher
    fetcher_status = {
        "running": data_fetcher.running if data_fetcher else False,
        "symbols_count": len(SYMBOLS),
        "timeframes": ['1m', '3m', '5m', '15m', '1d'],
        "data_type": "raw_ohlcv_only"
    }
    
    diagnostic_info = {
        "status": "operational" if (data_fetcher and data_fetcher.running) else "stopped",
        "timestamp": time.time(),
        "uptime": time.time() - start_time,
        "data_fetcher": fetcher_status,
        "symbols": SYMBOLS,
        "intervals": ['1m', '3m', '5m', '15m', '1d'],
        "architecture": "multi_timeframe_clean_gateway"
    }
    
    return web.json_response(diagnostic_info)

async def start_http_server():
    """Démarre le serveur HTTP pour les endpoints de santé."""
    app = web.Application()
    app.add_routes(routes)
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    # Utiliser le port 5010 pour l'API REST
    site = web.TCPSite(runner, '0.0.0.0', 5010)
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
    
    # Arrêter les services
    if ws_client:
        await ws_client.stop()
    if data_fetcher:
        await data_fetcher.stop()
    
    # Arrêter la boucle asyncio
    loop.stop()
    
    logger.info("Service Gateway (simple) arrêté proprement")

async def main():
    """
    Fonction principale qui démarre le Gateway SIMPLIFIÉ.
    """
    global data_fetcher, ws_client
    
    logger.info("🚀 Démarrage du service Gateway SIMPLE (données brutes uniquement)...")
    logger.info(f"Configuration: {', '.join(SYMBOLS)} @ ['1m', '3m', '5m', '15m', '1d']")
    logger.info("🎯 Architecture: AUCUN calcul d'indicateur - transmission pure")
    
    # Démarrer le serveur HTTP
    http_runner = await start_http_server()
    
    try:
        # Créer les services simples (pas d'indicateurs)
        data_fetcher = SimpleDataFetcher()
        ws_client = SimpleBinanceWebSocket(symbols=SYMBOLS, intervals=['1m', '3m', '5m', '15m', '1d'])
        
        logger.info("📡 Initialisation des services de données brutes...")
        logger.info("🔄 Mode: WebSocket temps réel + récupération historique")
        
        # Démarrer les services en parallèle
        logger.info("🚀 Démarrage WebSocket + DataFetcher...")
        await asyncio.gather(
            ws_client.start(),
            data_fetcher.start()
        )
        
    except Exception as e:
        logger.error(f"❌ Erreur critique dans le Gateway: {str(e)}")
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
        loop.add_signal_handler(
            sig, 
            lambda s=sig: asyncio.create_task(shutdown(s, loop))
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