#!/usr/bin/env python3
import asyncio
import logging
import sys
from typing import Dict, List, Optional
from datetime import datetime, timezone

from signal_aggregator import SignalAggregator
from regime_detector import RegimeDetector
from performance_tracker import PerformanceTracker
from kafka_manager import KafkaManager

sys.path.append('/app/shared')
from config import get_config
from redis_client import RedisClient

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
        
    async def start(self):
        """Initialize and start the service"""
        try:
            logger.info("Starting Signal Aggregator Service...")
            
            # Initialize connections
            self.kafka = KafkaManager()
            await self.kafka.start()
            
            self.redis = RedisClient()
            await self.redis.connect()
            
            # Initialize components
            self.regime_detector = RegimeDetector(self.redis)
            self.performance_tracker = PerformanceTracker(self.redis)
            self.aggregator = SignalAggregator(
                self.redis,
                self.regime_detector,
                self.performance_tracker
            )
            
            # Ensure output topic exists with proper configuration
            await self.kafka.ensure_topics_exist(
                topics=['signals.filtered'],
                num_partitions=3,
                replication_factor=1,
                config={
                    'retention.ms': '86400000',  # 24 hours
                    'compression.type': 'lz4'
                }
            )
            
            # Subscribe to raw signals
            await self.kafka.subscribe(['analyzer.signals'])
            
            self.running = True
            
            # Start processing
            await asyncio.gather(
                self.process_signals(),
                self.update_regimes(),
                self.update_performance_metrics()
            )
            
        except Exception as e:
            logger.error(f"Failed to start service: {e}")
            raise
            
    async def process_signals(self):
        """Process incoming signals and aggregate them"""
        while self.running:
            try:
                messages = await self.kafka.consume(timeout=1.0)
                
                for message in messages:
                    signal = message.value
                    
                    # Aggregate signal
                    aggregated = await self.aggregator.process_signal(signal)
                    
                    if aggregated:
                        # Publish filtered signal
                        await self.kafka.produce(
                            'signals.filtered',
                            aggregated
                        )
                        logger.info(f"Published aggregated signal: {aggregated}")
                        
            except Exception as e:
                logger.error(f"Error processing signals: {e}")
                await asyncio.sleep(1)
                
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
        
        if self.kafka:
            await self.kafka.stop()
        if self.redis:
            await self.redis.close()


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