#!/usr/bin/env python3
import asyncio
import logging
import sys
import os
import json
from aiohttp import web # Import aiohttp

# Ajouter le chemin vers les modules partagés et src
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')

# Imports directs des modules locaux
from signal_aggregator import EnhancedSignalAggregator  # type: ignore[attr-defined]

from performance_tracker import PerformanceTracker
from db_manager import DatabaseManager
from shared.src.kafka_client import KafkaClient
from shared.src.redis_client import RedisClient

def get_config():
    """Wrapper pour la config"""
    return {
        'KAFKA_BROKER': os.getenv('KAFKA_BROKER', 'kafka:9092'),
        'REDIS_HOST': os.getenv('REDIS_HOST', 'redis'),
        'REDIS_PORT': int(os.getenv('REDIS_PORT', 6379)),
        'HEALTH_CHECK_PORT': int(os.getenv('SIGNAL_AGGREGATOR_PORT', 5013)) # Health check port
    }

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SignalAggregatorService:
    def __init__(self):
        self.config = get_config()
        self.kafka = None
        self.redis = None
        self.aggregator = None
        self.regime_detector = None
        self.performance_tracker = None
        self.db_manager = None
        self.running = False
        self.consumer_id = None
        self.main_loop = None  # Event loop principal qui reste ouvert
        
    async def start(self):
        """Initialize and start the service"""
        try:
            # Capturer l'event loop principal
            self.main_loop = asyncio.get_running_loop()
            logger.info("🔄 Event loop principal capturé")
            logger.info("Starting Signal Aggregator Service...")
            
            # Initialize connections
            self.kafka = KafkaClient()
            
            self.redis = RedisClient()
            
            # Initialize database manager
            self.db_manager = DatabaseManager()
            await self.db_manager.initialize()
            
            # Initialize components
            from enhanced_regime_detector import EnhancedRegimeDetector
            self.regime_detector = EnhancedRegimeDetector(self.redis)
            logger.info("✅ Utilisation d'Enhanced Regime Detector comme détecteur principal")
                
            self.performance_tracker = PerformanceTracker(self.redis, db_pool=self.db_manager._connection_pool)
            # Utiliser la version améliorée avec filtres intelligents et DB
            self.aggregator = EnhancedSignalAggregator(
                self.redis,
                self.regime_detector,
                self.performance_tracker,
                db_pool=self.db_manager._connection_pool
            )
            
            # Charger les données historiques pour initialiser l'accumulateur
            await self._load_historical_market_data()
            
            # Ensure output topic exists with proper configuration
            self.kafka.ensure_topics_exist(
                topics=['signals.filtered', 'analyzer.signals'],
                num_partitions=3,
                replication_factor=1
            )
            
            # Subscribe to raw signals
            self.consumer_id = self.kafka.consume(
                topics=['analyzer.signals'],
                callback=self._process_kafka_message,
                group_id='signal-aggregator'
            )
            
            self.running = True
            
            # Start processing
            # Note: Enhanced Regime Detector calcule à la demande, pas besoin de tâche périodique
            if hasattr(self.regime_detector, 'update_all_regimes'):
                # Seulement pour le détecteur classique
                await asyncio.gather(
                    self.update_regimes(),
                    self.update_performance_metrics()
                )
            else:
                # Enhanced detector: seulement les métriques de performance
                await self.update_performance_metrics()
            
        except Exception as e:
            logger.error(f"Failed to start service: {e}")
            raise
            
    def _process_kafka_message(self, topic: str, message: dict):
        """Process incoming Kafka message (synchronous callback)"""
        try:
            logger.info(f"📨 Received message from {topic}: {message}")
            
            # Utiliser le loop principal au lieu de créer des threads/loops séparés
            import asyncio
            import concurrent.futures
            
            try:
                if self.main_loop and not self.main_loop.is_closed():
                    # Envoyer la coroutine vers le loop principal thread-safe
                    future = asyncio.run_coroutine_threadsafe(
                        self.aggregator.process_signal(message), 
                        self.main_loop
                    )
                    aggregated = future.result(timeout=30)  # Timeout de 30s
                else:
                    logger.error("❌ Event loop principal non disponible")
                    return
                
                if aggregated:
                    # Publish filtered signal on Kafka
                    self.kafka.produce('signals.filtered', aggregated)
                    
                    # AUSSI publier sur Redis pour le coordinator
                    self.redis.set(f"signal_filtered:{aggregated['symbol']}", str(aggregated), expiration=300)
                    # Et publier sur le canal Redis que le coordinator écoute
                    import json
                    self.redis.publish('roottrading:signals:filtered', json.dumps(aggregated))
                    
                    # Sauvegarder le signal dans la DB
                    if self.db_manager and self.main_loop and not self.main_loop.is_closed():
                        save_future = asyncio.run_coroutine_threadsafe(
                            self.db_manager.save_signal(aggregated),
                            self.main_loop
                        )
                        saved = save_future.result(timeout=5)
                        if not saved:
                            logger.warning("⚠️ Échec de la sauvegarde du signal dans la DB")
                    
                    logger.info(f"✅ Published aggregated signal on Kafka and Redis: {aggregated}")
            except concurrent.futures.TimeoutError:
                logger.error(f"⏰ Timeout lors du traitement du signal: {message.get('symbol', 'unknown')}")
            except Exception as e:
                logger.error(f"❌ Erreur lors du traitement du signal: {e}")
                
        except Exception as e:
            logger.error(f"❌ Error processing signal: {e}")
    
    async def update_regimes(self):
        """Periodically update market regime detection"""
        while self.running:
            try:
                await self.regime_detector.update_all_regimes()
                await asyncio.sleep(60)  # Update every minute
            except Exception as e:
                logger.error(f"Error updating regimes: {e}")
                await asyncio.sleep(5)
                
    async def update_performance_metrics(self):
        """Periodically update strategy performance metrics"""
        while self.running:
            try:
                await self.performance_tracker.update_all_metrics()
                await asyncio.sleep(300)  # Update every 5 minutes
            except Exception as e:
                logger.error(f"Error updating performance: {e}")
                await asyncio.sleep(30)
    
    async def _load_historical_market_data(self):
        """Charge les vraies données historiques depuis la DB pour initialiser l'accumulateur"""
        try:
            logger.info("🔄 Chargement des vraies données historiques depuis la DB...")
            
            # Utiliser les symboles depuis la config partagée
            from shared.src.config import SYMBOLS
            symbols = SYMBOLS
            
            for symbol in symbols:
                try:
                    # Récupérer les vraies données historiques depuis la DB (250 derniers points 5m)
                    query = """
                        SELECT time, symbol, timeframe, open, high, low, close, volume,
                               rsi_14, ema_7, ema_26, macd_line, macd_signal, bb_upper, bb_lower, atr_14  -- MIGRATION BINANCE
                        FROM market_data 
                        WHERE symbol = $1 AND timeframe = '5m' AND enhanced = true
                        ORDER BY time DESC 
                        LIMIT 250
                    """
                    
                    # Exécuter la requête via le pool de connexions du db_manager
                    if self.db_manager and self.db_manager._connection_pool:
                        async with self.db_manager._connection_pool.acquire() as conn:
                            rows = await conn.fetch(query, symbol)
                            
                            if rows:
                                # Inverser l'ordre pour avoir du plus ancien au plus récent
                                rows = list(reversed(rows))
                                
                                for row in rows:
                                    # Convertir la row PostgreSQL en dict pour l'accumulateur
                                    historical_data = {
                                        'timestamp': row['time'].timestamp(),
                                        'symbol': row['symbol'],
                                        'timeframe': row['timeframe'],
                                        'open': float(row['open']),
                                        'high': float(row['high']),
                                        'low': float(row['low']),
                                        'close': float(row['close']),
                                        'volume': float(row['volume']),
                                        'rsi_14': float(row['rsi_14']) if row['rsi_14'] else None,
                                        'ema_7': float(row['ema_7']) if row['ema_7'] else None,  # MIGRATION BINANCE
                                        'ema_26': float(row['ema_26']) if row['ema_26'] else None,
                                        'macd_line': float(row['macd_line']) if row['macd_line'] else None,
                                        'macd_signal': float(row['macd_signal']) if row['macd_signal'] else None,
                                        'bb_upper': float(row['bb_upper']) if row['bb_upper'] else None,
                                        'bb_lower': float(row['bb_lower']) if row['bb_lower'] else None,
                                        'atr_14': float(row['atr_14']) if row['atr_14'] else None,
                                        'ultra_enriched': True  # Marquer comme vraies données enrichies
                                    }
                                    
                                    # Ajouter à l'accumulateur
                                    self.aggregator.market_data_accumulator.add_market_data(symbol, historical_data)
                                
                                count = self.aggregator.market_data_accumulator.get_history_count(symbol)
                                logger.info(f"✅ Vraies données historiques DB chargées pour {symbol}: {count} points")
                            else:
                                logger.warning(f"⚠️ Aucune donnée historique trouvée en DB pour {symbol}")
                    else:
                        logger.warning("⚠️ Pas de pool de connexion DB disponible")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Erreur chargement données DB {symbol}: {e}")
                    
                    # Fallback: essayer Redis comme secours (sans simulation)
                    try:
                        redis_key = f"market_data:{symbol}:5m_history"
                        raw_data = self.redis.get(redis_key)
                        if raw_data:
                            historical_list = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
                            if isinstance(historical_list, list):
                                for data in historical_list[-100:]:  # 100 derniers points
                                    if isinstance(data, dict):
                                        self.aggregator.market_data_accumulator.add_market_data(symbol, data)
                                
                                count = self.aggregator.market_data_accumulator.get_history_count(symbol)
                                logger.info(f"✅ Données historiques Redis chargées pour {symbol}: {count} points")
                        else:
                            logger.warning(f"⚠️ Pas de données historiques Redis pour {symbol}")
                    except Exception as redis_e:
                        logger.error(f"❌ Erreur fallback Redis pour {symbol}: {redis_e}")
            
            logger.info("✅ Chargement des vraies données historiques terminé")
            
        except Exception as e:
            logger.error(f"❌ Erreur générale lors du chargement historique: {e}")
                
    async def stop(self):
        """Stop the service gracefully"""
        logger.info("Stopping Signal Aggregator Service...")
        self.running = False
        
        if self.kafka and self.consumer_id:
            self.kafka.stop_consuming(self.consumer_id)
        if self.redis:
            self.redis.close()
        if self.db_manager:
            await self.db_manager.close()


async def health_check(request):
    return web.Response(text="Signal Aggregator is healthy")

async def performance_summary(request):
    """Endpoint pour récupérer le résumé des performances bayésiennes et seuils dynamiques"""
    try:
        service = request.app.get('service')
        if service and hasattr(service, 'aggregator'):
            summary = service.aggregator.get_performance_summary()
            return web.json_response(summary)
        else:
            return web.json_response({'error': 'Service not available'}, status=503)
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)

async def update_performance(request):
    """Endpoint pour mettre à jour les performances d'une stratégie"""
    try:
        data = await request.json()
        service = request.app.get('service')
        
        if not service or not hasattr(service, 'aggregator'):
            return web.json_response({'error': 'Service not available'}, status=503)
        
        strategy = data.get('strategy')
        is_win = data.get('is_win')
        return_pct = data.get('return_pct', 0.0)
        
        if not strategy or is_win is None:
            return web.json_response({'error': 'Missing required fields: strategy, is_win'}, status=400)
        
        service.aggregator.update_strategy_performance(strategy, is_win, return_pct)
        return web.json_response({'status': 'updated', 'strategy': strategy})
        
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)

async def main():
    service = SignalAggregatorService()
    
    # Setup aiohttp for health checks et API
    app = web.Application()
    app['service'] = service  # Stocker la référence du service
    app.router.add_get('/health', health_check)
    app.router.add_get('/performance', performance_summary)
    app.router.add_post('/performance/update', update_performance)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', service.config['HEALTH_CHECK_PORT'])
    await site.start()
    logger.info(f"Health check server started on port {service.config['HEALTH_CHECK_PORT']}")

    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Service error: {e}")
    finally:
        await service.stop()
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())