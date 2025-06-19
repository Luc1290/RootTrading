#!/usr/bin/env python3
import asyncio
import logging
import sys
import os
from typing import Dict, List, Optional
from datetime import datetime, timezone

# Ajouter le chemin vers les modules partagés
sys.path.insert(0, '/app')

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
            self.regime_detector = RegimeDetector(self.redis)
            self.performance_tracker = PerformanceTracker(self.redis)
            # Utiliser la version améliorée avec filtres intelligents
            self.aggregator = EnhancedSignalAggregator(
                self.redis,
                self.regime_detector,
                self.performance_tracker
            )
            
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
            await asyncio.gather(
                self.update_regimes(),
                self.update_performance_metrics()
            )
            
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
                aggregated = loop.run_until_complete(self.aggregator.process_signal_enhanced(message))
                
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