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

async def sync_historical_to_realtime_cache(ws_client):
    """
    Synchronise TOUS les indicateurs calculés historiquement avec le cache temps réel du WebSocket.
    CRITIQUE pour maintenir la continuité de TOUS les indicateurs (extensible pour futurs ajouts).
    """
    from shared.src.technical_indicators import indicator_cache
    
    try:
        timeframes = ['1m', '5m', '15m', '1h', '4h']
        synced_count = 0
        indicator_types = set()
        
        for symbol in SYMBOLS:
            for timeframe in timeframes:
                # **NOUVEAU**: Récupérer TOUS les indicateurs calculés historiquement
                all_historical_indicators = indicator_cache.get_all_indicators(symbol, timeframe)
                
                # Vérifier que le symbole/timeframe existe dans le cache WebSocket
                if symbol in ws_client.incremental_cache and timeframe in ws_client.incremental_cache[symbol]:
                    cache_ref = ws_client.incremental_cache[symbol][timeframe]
                    
                    # **EXTENSIBLE**: Transférer TOUS les indicateurs disponibles
                    for indicator_name, indicator_value in all_historical_indicators.items():
                        if indicator_value is not None:
                            cache_ref[indicator_name] = indicator_value
                            synced_count += 1
                            indicator_types.add(indicator_name)
                    
                    if all_historical_indicators:
                        key_indicators = {k: v for k, v in all_historical_indicators.items() 
                                        if k in ['ema_12', 'ema_26', 'macd_line', 'rsi_14'] and v is not None}
                        logger.debug(f"Sync {symbol} {timeframe}: {len(all_historical_indicators)} indicateurs → " +
                                   ', '.join([f"{k}={v:.4f}" for k, v in key_indicators.items()]))
                else:
                    logger.warning(f"Cache WebSocket non initialisé pour {symbol} {timeframe}")
        
        logger.info(f"📊 Synchronisation terminée: {synced_count} indicateurs transférés")
        logger.info(f"🔧 Types d'indicateurs synchronisés: {', '.join(sorted(indicator_types))}")
        
    except Exception as e:
        logger.error(f"❌ Erreur synchronisation cache: {e}")
        # Ne pas faire échouer le démarrage pour un problème de sync

async def restore_indicators_from_db():
    """
    Restaure les indicateurs techniques depuis la base de données PostgreSQL.
    Charge les dernières valeurs d'indicateurs pour chaque symbole/timeframe pour maintenir la continuité.
    
    Returns:
        int: Nombre d'indicateurs restaurés
    """
    from shared.src.technical_indicators import indicator_cache
    from shared.src.config import get_db_config
    import asyncpg
    
    restored_count = 0
    
    try:
        # Connexion à la base de données
        db_config = get_db_config()
        conn = await asyncpg.connect(
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['database'],
            user=db_config['user'],
            password=db_config['password']
        )
        
        # Requête pour récupérer les derniers indicateurs pour chaque symbole/timeframe
        query = """
            WITH latest_data AS (
                SELECT DISTINCT ON (symbol, timeframe) 
                    symbol, timeframe, time,
                    rsi_14, ema_12, ema_26, ema_50, sma_20, sma_50,
                    macd_line, macd_signal, macd_histogram,
                    bb_upper, bb_middle, bb_lower, bb_position, bb_width,
                    atr_14, momentum_10, volume_ratio, avg_volume_20,
                    stoch_k, stoch_d, stoch_rsi, adx_14, plus_di, minus_di,
                    williams_r, cci_20, mfi_14, vwap_10, roc_10, roc_20,
                    obv, trend_angle, pivot_count
                FROM market_data 
                WHERE enhanced = true 
                    AND symbol = ANY($1)
                    AND timeframe = ANY($2)
                ORDER BY symbol, timeframe, time DESC
            )
            SELECT * FROM latest_data
        """
        
        timeframes = ['1m', '5m', '15m', '1h', '4h']
        result = await conn.fetch(query, SYMBOLS, timeframes)
        
        # Restaurer les indicateurs dans le cache
        for row in result:
            symbol = row['symbol']
            timeframe = row['timeframe']
            
            # Construire le dictionnaire des indicateurs
            indicators = {}
            
            # Parcourir toutes les colonnes d'indicateurs
            for field_name in row.keys():
                if field_name not in ['symbol', 'timeframe', 'time'] and row[field_name] is not None:
                    indicators[field_name] = float(row[field_name])
            
            # Restaurer dans le cache si des indicateurs sont trouvés
            if indicators:
                for indicator_name, value in indicators.items():
                    indicator_cache.set(symbol, timeframe, indicator_name, value)
                    restored_count += 1
                
                logger.debug(f"🔄 Restauré {symbol} {timeframe}: {len(indicators)} indicateurs depuis {row['time']}")
        
        await conn.close()
        
        if restored_count > 0:
            logger.info(f"✅ {restored_count} indicateurs restaurés depuis la DB pour {len(result)} symbole/timeframe")
        
        return restored_count
        
    except Exception as e:
        logger.error(f"❌ Erreur restoration indicateurs depuis DB: {e}")
        return 0


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
        
        # **NOUVELLE LOGIQUE**: Restoration automatique du cache d'indicateurs depuis la DB
        from shared.src.technical_indicators import indicator_cache
        
        logger.info("🔄 Restoration automatique du cache d'indicateurs depuis la DB...")
        restored_indicators = await restore_indicators_from_db()
        
        if restored_indicators > 0:
            logger.info(f"✅ {restored_indicators} indicateurs restaurés depuis la DB - CONTINUITÉ PRÉSERVÉE")
            
            # Afficher les statistiques de restoration
            cache_stats = indicator_cache.get_cache_stats()
            logger.info(f"📊 Cache Statistics: {cache_stats['local_entries']} entrées locales")
        else:
            logger.warning("⚠️ Aucun indicateur restauré - Premier démarrage ou DB vide")
            
            # Seulement si pas de restoration, faire l'initialisation complète
            if not args.skip_init:
                logger.info("🔧 Recalcul complet des indicateurs (pas de données persistées)...")
                
                # Forcer un calcul point par point pour remplir le cache
                init_fetcher = UltraDataFetcher()
                for symbol in SYMBOLS:
                    for timeframe in ['1m', '5m', '15m', '1h', '4h']:
                        try:
                            # Utiliser UltraDataFetcher pour récupérer et calculer
                            await init_fetcher._fetch_ultra_enriched_data(symbol, timeframe)
                            logger.debug(f"✅ Cache recalculé pour {symbol} {timeframe}")
                        except Exception as e:
                            logger.warning(f"Erreur calcul initial {symbol} {timeframe}: {e}")
        
        # **FIX CRITIQUE**: Synchroniser les indicateurs restaurés/calculés avec le cache temps réel WebSocket
        logger.info("🔄 Synchronisation cache → WebSocket temps réel...")
        await sync_historical_to_realtime_cache(ws_client)
        
        # Vérifier la synchronisation
        final_cache_stats = {"total": 0, "symbols": set()}
        for symbol in SYMBOLS:
            for timeframe in ['1m', '5m', '15m', '1h', '4h']:
                indicators = indicator_cache.get_all_indicators(symbol, timeframe)
                final_cache_stats["total"] += len(indicators)
                if indicators:
                    final_cache_stats["symbols"].add(symbol)
        
        logger.info(f"✅ Synchronisation terminée: {final_cache_stats['total']} indicateurs pour {len(final_cache_stats['symbols'])} symboles")
        
        # Force sauvegarde pour s'assurer que les données sont persistées
        if final_cache_stats["total"] > 0:
            saved_count = indicator_cache.force_save_all()
            logger.info(f"💾 {saved_count} indicateurs sauvegardés pour la prochaine restoration")
        
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