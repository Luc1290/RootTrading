#!/usr/bin/env python3
import asyncio
import logging
import sys
import os
import json
import time
from typing import Dict, List, Optional
from datetime import datetime, timezone

# Ajouter le chemin vers les modules partagés et src
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/src')

from signal_aggregator import SignalAggregator, EnhancedSignalAggregator
from regime_detector import RegimeDetector
from performance_tracker import PerformanceTracker
from shared.src.kafka_client import KafkaClient
from shared.src.redis_client import RedisClient

def get_config():
    """Wrapper pour la config"""
    return {
        'KAFKA_BROKER': os.getenv('KAFKA_BROKER', 'kafka:9092'),
        'REDIS_HOST': os.getenv('REDIS_HOST', 'redis'),
        'REDIS_PORT': int(os.getenv('REDIS_PORT', 6379))
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
        self.running = False
        self.consumer_id = None
        
    async def start(self):
        """Initialize and start the service"""
        try:
            logger.info("Starting Signal Aggregator Service...")
            
            # Initialize connections
            self.kafka = KafkaClient()
            
            self.redis = RedisClient()
            
            # Initialize components
            try:
                from enhanced_regime_detector import EnhancedRegimeDetector
                self.regime_detector = EnhancedRegimeDetector(self.redis)
                logger.info("✅ Utilisation d'Enhanced Regime Detector comme détecteur principal")
            except ImportError:
                from regime_detector import RegimeDetector
                self.regime_detector = RegimeDetector(self.redis)
                logger.warning("⚠️ Enhanced Regime Detector non disponible, utilisation du classique")
                
            self.performance_tracker = PerformanceTracker(self.redis)
            # Utiliser la version améliorée avec filtres intelligents
            self.aggregator = EnhancedSignalAggregator(
                self.redis,
                self.regime_detector,
                self.performance_tracker
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
            
            # Use asyncio.run for async processing in sync callback
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Utiliser la méthode améliorée avec filtres intelligents
                aggregated = loop.run_until_complete(self.aggregator.process_signal(message))
                
                if aggregated:
                    # Publish filtered signal on Kafka
                    self.kafka.produce('signals.filtered', aggregated)
                    
                    # AUSSI publier sur Redis pour le coordinator
                    self.redis.set(f"signal_filtered:{aggregated['symbol']}", str(aggregated), expiration=300)
                    # Et publier sur le canal Redis que le coordinator écoute
                    import json
                    self.redis.publish('roottrading:signals:filtered', json.dumps(aggregated))
                    
                    logger.info(f"✅ Published aggregated signal on Kafka and Redis: {aggregated}")
            finally:
                loop.close()
                
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
        """Charge les données historiques pour initialiser l'accumulateur"""
        try:
            logger.info("🔄 Chargement des données historiques pour signal_aggregator...")
            
            # Symboles de trading (peut-être à récupérer depuis config)
            symbols = ['BTCUSDC', 'ETHUSDC', 'SOLUSDC', 'XRPUSDC']
            
            for symbol in symbols:
                # Charger les données de marché 1m (timeframe principal)
                redis_key = f"market_data:{symbol}:1m"
                
                try:
                    raw_data = self.redis.get(redis_key)
                    if raw_data:
                        data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
                        
                        if isinstance(data, dict) and 'ultra_enriched' in data:
                            # Simuler plusieurs points de données historiques
                            base_time = time.time() - (100 * 60)  # 100 minutes ago
                            
                            for i in range(100):
                                historical_data = data.copy()
                                historical_data['timestamp'] = base_time + (i * 60)  # 1 minute intervals
                                
                                # Légère variation des prix pour simuler l'historique
                                price_variation = (i % 10 - 5) * 0.001  # ±0.5% variation
                                close_price = data['close'] * (1 + price_variation)
                                
                                # Ajouter les valeurs OHLC manquantes
                                historical_data['close'] = close_price
                                historical_data['open'] = close_price  # Approximation
                                historical_data['high'] = close_price * 1.002  # +0.2%
                                historical_data['low'] = close_price * 0.998   # -0.2%
                                
                                # Ajouter à l'accumulateur
                                self.aggregator.market_data_accumulator.add_market_data(symbol, historical_data)
                            
                            count = self.aggregator.market_data_accumulator.get_history_count(symbol)
                            logger.info(f"✅ Données historiques chargées pour {symbol}: {count} points")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Erreur chargement données historiques {symbol}: {e}")
            
            logger.info("✅ Chargement des données historiques terminé")
            
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


async def main():
    service = SignalAggregatorService()
    
    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Service error: {e}")
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(main())