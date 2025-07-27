"""
Market Analyzer Service - Point d'entrée principal
Service qui calcule TOUS les indicateurs techniques à partir des données OHLCV.
Architecture propre : Gateway → market_data → Market Analyzer → analyzer_data
"""

import asyncio
import logging
import signal
import sys
import os
import time
from aiohttp import web

# Ajouter les chemins pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from market_analyzer.src.data_listener import DataListener
from shared.src.config import SYMBOLS

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('market_analyzer.log')
    ]
)
logger = logging.getLogger("market_analyzer")

# Variables globales
data_listener = None
running = True
start_time = time.time()

# Routes pour l'API de diagnostic
routes = web.RouteTableDef()

@routes.get('/health')
async def health_check(request):
    """Point de terminaison santé du service."""
    uptime = time.time() - start_time
    return web.json_response({
        "status": "ok",
        "timestamp": time.time(),
        "uptime": uptime,
        "service": "market_analyzer",
        "mode": "active" if running else "stopping",
        "symbols": SYMBOLS,
        "architecture": "clean_calculation_engine"
    })

@routes.get('/stats')
async def get_stats(request):
    """Statistiques du Market Analyzer."""
    global data_listener
    
    if data_listener:
        stats = await data_listener.get_stats()
        return web.json_response(stats)
    else:
        return web.json_response({
            "error": "DataListener not initialized",
            "status": "not_running"
        }, status=503)

@routes.post('/process-historical')
async def process_historical(request):
    """API pour traiter les données historiques."""
    global data_listener
    
    if not data_listener:
        return web.json_response({
            "error": "DataListener not initialized"
        }, status=503)
    
    try:
        # Parser les paramètres
        data = await request.json() if request.content_type == 'application/json' else {}
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        limit = data.get('limit', 1000)
        
        # Lancer le traitement optimisé en arrière-plan
        asyncio.create_task(
            data_listener.process_historical_optimized(symbol, timeframe, limit)
        )
        
        return web.json_response({
            "status": "started",
            "message": f"Traitement historique lancé (limit: {limit})",
            "symbol": symbol,
            "timeframe": timeframe
        })
        
    except Exception as e:
        return web.json_response({
            "error": f"Erreur lancement traitement: {e}"
        }, status=500)

@routes.get('/coverage')
async def get_coverage(request):
    """Analyse de couverture des données."""
    global data_listener
    
    if not data_listener:
        return web.json_response({
            "error": "DataListener not initialized"
        }, status=503)
    
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
            coverage_data.append({
                'symbol': row['symbol'],
                'timeframe': row['timeframe'],
                'market_data_count': row['market_data_count'],
                'analyzer_data_count': row['analyzer_data_count'],
                'coverage_percent': float(row['coverage_percent']),
                'missing_count': row['market_data_count'] - row['analyzer_data_count']
            })
        
        return web.json_response({
            "coverage_by_asset": coverage_data,
            "total_assets": len(set(row['symbol'] for row in coverage_data)),
            "total_timeframes": len(set(row['timeframe'] for row in coverage_data))
        })
        
    except Exception as e:
        return web.json_response({
            "error": f"Erreur analyse couverture: {e}"
        }, status=500)

async def start_http_server():
    """Démarre le serveur HTTP pour l'API."""
    app = web.Application()
    app.add_routes(routes)
    
    runner = web.AppRunner(app)
    await runner.setup()
    
    # Port 5020 pour le Market Analyzer
    site = web.TCPSite(runner, '0.0.0.0', 5020)
    await site.start()
    
    logger.info("✅ API HTTP démarrée sur le port 5020")
    return runner

async def shutdown(signal_type, loop):
    """Gère l'arrêt propre du service."""
    global running, data_listener
    
    logger.info(f"Signal {signal_type.name} reçu, arrêt en cours...")
    running = False
    
    # Arrêter le DataListener
    if data_listener:
        await data_listener.stop()
    
    # Arrêter la boucle asyncio
    loop.stop()
    
    logger.info("Market Analyzer arrêté proprement")

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
        listening_task = asyncio.create_task(data_listener.start_listening())
        
        # Proposer de traiter l'historique en parallèle
        logger.info("🔍 Vérification de la couverture des données...")
        stats = await data_listener.get_stats()
        
        if stats['missing_analyses'] > 0:
            logger.info(f"⚠️ {stats['missing_analyses']} données non analysées détectées")
            logger.info("💡 Démarrage du traitement historique automatique...")
            
            # Utiliser la méthode optimisée qui traite par symbole et dans l'ordre
            await data_listener.process_historical_optimized(limit=1000000)
        else:
            logger.info("✅ Toutes les données sont analysées")
        
        # Afficher les statistiques finales
        final_stats = await data_listener.get_stats()
        logger.info(f"📊 Couverture: {final_stats['coverage_percent']}% ({final_stats['total_analyzer_data']}/{final_stats['total_market_data']})")
        
        # Attendre que l'écoute temps réel continue
        logger.info("✅ Traitement historique terminé - écoute temps réel active")
        await listening_task
        
    except Exception as e:
        logger.error(f"❌ Erreur critique: {e}")
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