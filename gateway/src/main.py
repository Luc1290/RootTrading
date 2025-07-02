"""
Point d'entrée principal pour le microservice Gateway.
Gère les connexions WebSocket à Binance et la publication des données vers Kafka.
"""
import asyncio
import logging
import signal
import sys
import os
import time
from aiohttp import web
import argparse

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from binance_ws import BinanceWebSocket
from kafka_producer import get_producer
from ultra_data_fetcher import UltraDataFetcher
from shared.src.config import BINANCE_API_KEY, BINANCE_SECRET_KEY, SYMBOLS, INTERVAL

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


if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
    logger.warning("⚠️ Clés API Binance non configurées ou incomplètes!")

# Variables globales
ws_client = None
running = True
start_time = time.time()

# Routes pour le serveur HTTP
routes = web.RouteTableDef()

@routes.get('/health')
async def health_check(request):
    """
    Point de terminaison pour vérifier l'état du service.
    """
    uptime = time.time() - start_time
    return web.json_response({
        "status": "ok",
        "timestamp": time.time(),
        "uptime": uptime,
        "mode": "active" if running else "stopping",
        "symbols": SYMBOLS,
        "interval": INTERVAL
    })

@routes.get('/diagnostic')
async def diagnostic(request):
    """
    Point de terminaison pour le diagnostic du service.
    """
    global ws_client
    
    # Récupérer l'état du client WebSocket
    ws_status = {
        "connected": ws_client.ws is not None,
        "running": ws_client.running,
        "reconnect_delay": ws_client.reconnect_delay,
        "last_message_time": ws_client.last_message_time,
        "stream_paths": ws_client.stream_paths,
    }
    
    # Construire la réponse
    diagnostic_info = {
        "status": "operational" if ws_client.running else "stopped",
        "timestamp": time.time(),
        "uptime": time.time() - start_time,
        "websocket": ws_status,
        "symbols": SYMBOLS,
        "interval": INTERVAL,
        "kafka_connected": True,  # Simplification, à améliorer avec un vrai check
    }
    
    return web.json_response(diagnostic_info)

async def start_http_server():
    """
    Démarre le serveur HTTP pour les endpoints de santé.
    """
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
    
    Args:
        signal_type: Type de signal reçu
        loop: Boucle asyncio en cours
    """
    global running, ws_client
    
    logger.info(f"Signal {signal_type.name} reçu, arrêt en cours...")
    running = False
    
    # Arrêter le WebSocket
    if ws_client:
        await ws_client.stop()
    
    # Fermer le producteur Kafka
    producer = get_producer()
    producer.close()
    
    # Arrêter la boucle asyncio
    loop.stop()
    
    logger.info("Service Gateway arrêté proprement")

def parse_arguments():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description='Gateway RootTrading Ultra-Enrichi')
    parser.add_argument(
        '--skip-init', 
        action='store_true', 
        help='Ignorer l\'initialisation des données ultra-enrichies'
    )
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Activer le mode debug pour plus de logs'
    )
    return parser.parse_args()

async def main():
    """
    Fonction principale qui démarre le Gateway.
    """
    global ws_client
    validation_fetcher = None
    
    # Parser les arguments
    args = parse_arguments()
    
    logger.info("🚀 Démarrage du service Gateway RootTrading...")
    logger.info(f"Configuration: {', '.join(SYMBOLS)} @ {INTERVAL}")
    
    # Démarrer le serveur HTTP
    http_runner = await start_http_server()
    
    # Obtenir le producteur Kafka
    producer = get_producer()
    
    try:
        # Initialiser les données ultra-enrichies au démarrage si demandé
        if not args.skip_init:
            logger.info(f"🔥 Initialisation des données ultra-enrichies multi-timeframes...")
            
            # Créer l'UltraDataFetcher pour l'initialisation
            init_fetcher = UltraDataFetcher()
            
            # **NOUVEAU**: Charger 5 jours de données historiques pour tous les timeframes
            try:
                logger.info(f"📚 Chargement de 5 jours de données historiques...")
                await init_fetcher.load_historical_data(days=5)
                logger.info(f"✅ Données historiques chargées avec succès")
            except Exception as e:
                logger.error(f"❌ Erreur chargement données historiques: {e}")
                logger.warning(f"⚠️ Poursuite sans données historiques complètes")
            
            # Exécuter un cycle d'initialisation pour remplir les caches Redis
            try:
                await init_fetcher._fetch_initialization_data()
                logger.info(f"✅ Données ultra-enrichies initialisées avec succès")
            except Exception as e:
                logger.warning(f"⚠️ Erreur lors de l'initialisation: {e}")
        else:
            logger.info("⏩ Initialisation des données ultra-enrichies ignorée")
        
        # Créer le client WebSocket Binance
        ws_client = BinanceWebSocket(symbols=SYMBOLS, interval=INTERVAL)
        
        # Créer le service ultra-enrichi multi-timeframes
        ultra_fetcher = UltraDataFetcher()
        
        # Démarrer les services en parallèle
        logger.info("🚀 Démarrage WebSocket + UltraDataFetcher")
        await asyncio.gather(
            ws_client.start(),
            ultra_fetcher.start()
        )
    except Exception as e:
        logger.error(f"❌ Erreur critique dans le Gateway: {str(e)}")
    finally:
        if ws_client:
            await ws_client.stop()
        if ultra_fetcher:
            await ultra_fetcher.stop()
        
        # Arrêter le serveur HTTP
        await http_runner.cleanup()
        
        logger.info("Service Gateway terminé")

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