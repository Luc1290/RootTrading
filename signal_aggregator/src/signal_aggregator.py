#!/usr/bin/env python3
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class SignalAggregator:
    """Aggregates multiple strategy signals and resolves conflicts"""
    
    # Strategy groupings by market condition
    STRATEGY_GROUPS = {
        'trend': ['EMA_Cross', 'MACD', 'Breakout'],
        'mean_reversion': ['Bollinger', 'RSI', 'Divergence'],
        'adaptive': ['Ride_or_React']
    }
    
    def __init__(self, redis_client, regime_detector, performance_tracker):
        self.redis = redis_client
        self.regime_detector = regime_detector
        self.performance_tracker = performance_tracker
        
        # Signal buffer for aggregation
        self.signal_buffer = defaultdict(list)
        self.last_signal_time = {}
        self.cooldown_period = timedelta(minutes=3)  # 3 candles for 1m timeframe
        
        # Voting thresholds
        self.min_vote_threshold = 0.5
        self.min_confidence_threshold = 0.7  # Augmenté de 0.6 à 0.7 pour filtrer les signaux faibles
        
    async def process_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a raw signal and return aggregated decision"""
        try:
            symbol = signal['symbol']
            strategy = signal['strategy']
            
            # Convert 'side' to 'direction' for compatibility
            if 'side' in signal and 'direction' not in signal:
                side = signal['side']
                if side in ['BUY', 'buy']:
                    signal['direction'] = 'BUY'
                elif side in ['SELL', 'sell']:
                    signal['direction'] = 'SELL'
                else:
                    logger.warning(f"Unknown side value: {side}")
                    return None
            
            # Handle timestamp conversion
            timestamp_str = signal.get('timestamp', signal.get('created_at'))
            if timestamp_str:
                if isinstance(timestamp_str, str):
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                else:
                    timestamp = datetime.fromtimestamp(timestamp_str / 1000 if timestamp_str > 1e10 else timestamp_str, tz=timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)
            
            # Check cooldown
            if await self._is_in_cooldown(symbol):
                logger.debug(f"Symbol {symbol} in cooldown, ignoring signal")
                return None
                
            # Add to buffer
            self.signal_buffer[symbol].append(signal)
            
            # Clean old signals (keep only last 30 seconds)
            cutoff_time = timestamp - timedelta(seconds=30)
            self.signal_buffer[symbol] = [
                s for s in self.signal_buffer[symbol]
                if self._get_signal_timestamp(s) > cutoff_time
            ]
            
            # Check if we have enough signals to make a decision
            if len(self.signal_buffer[symbol]) < 2:
                return None  # Wait for more signals
                
            # Get market regime
            regime = await self.regime_detector.get_regime(symbol)
            
            # Calculate aggregated signal
            aggregated = await self._aggregate_signals(
                symbol, 
                self.signal_buffer[symbol],
                regime
            )
            
            if aggregated:
                # Clear buffer after successful aggregation
                self.signal_buffer[symbol].clear()
                self.last_signal_time[symbol] = timestamp
                
                # Add metadata
                aggregated.update({
                    'aggregation_method': 'weighted_vote',
                    'regime': regime,
                    'timestamp': timestamp.isoformat(),
                    'source_signals': len(self.signal_buffer[symbol])
                })
                
                return aggregated
                
            return None
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return None
            
    def _get_signal_timestamp(self, signal: Dict[str, Any]) -> datetime:
        """Extract timestamp from signal with multiple format support"""
        timestamp_str = signal.get('timestamp', signal.get('created_at'))
        if timestamp_str:
            if isinstance(timestamp_str, str):
                timestamp = datetime.fromisoformat(timestamp_str)
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                return timestamp
            else:
                return datetime.fromtimestamp(timestamp_str / 1000 if timestamp_str > 1e10 else timestamp_str, tz=timezone.utc)
        return datetime.now(timezone.utc)
            
    async def _aggregate_signals(self, symbol: str, signals: List[Dict], 
                               regime: str) -> Optional[Dict[str, Any]]:
        """Aggregate multiple signals into a single decision"""
        
        # Group signals by direction
        buy_signals = []
        sell_signals = []
        
        for signal in signals:
            strategy = signal['strategy']
            
            # Check if strategy is appropriate for current regime
            if not self._is_strategy_active(strategy, regime):
                logger.debug(f"Strategy {strategy} not active in {regime} regime")
                continue
                
            # Get strategy weight based on performance
            weight = await self.performance_tracker.get_strategy_weight(strategy)
            
            # Apply confidence threshold
            confidence = signal.get('confidence', 0.5)
            if confidence < self.min_confidence_threshold:
                continue
                
            # Get direction (handle both 'direction' and 'side' keys)
            direction = signal.get('direction', signal.get('side'))
            if direction in ['BUY', 'buy']:
                direction = 'BUY'
            elif direction in ['SELL', 'sell']:
                direction = 'SELL'
            
            # Weighted signal
            weighted_signal = {
                'strategy': strategy,
                'direction': direction,
                'confidence': confidence,
                'weight': weight,
                'score': confidence * weight
            }
            
            if direction == 'BUY':
                buy_signals.append(weighted_signal)
            elif direction == 'SELL':
                sell_signals.append(weighted_signal)
                
        # Calculate total scores
        buy_score = sum(s['score'] for s in buy_signals)
        sell_score = sum(s['score'] for s in sell_signals)
        
        # Determine direction
        if buy_score > sell_score and buy_score >= self.min_vote_threshold:
            direction = 'BUY'
            confidence = buy_score / (buy_score + sell_score)
            contributing_strategies = [s['strategy'] for s in buy_signals]
        elif sell_score > buy_score and sell_score >= self.min_vote_threshold:
            direction = 'SELL'
            confidence = sell_score / (buy_score + sell_score)
            contributing_strategies = [s['strategy'] for s in sell_signals]
        else:
            # No clear signal
            logger.debug(f"No clear signal for {symbol}: buy={buy_score:.2f}, sell={sell_score:.2f}")
            return None
            
        # Calculate averaged take profit and stop loss
        relevant_signals = buy_signals if direction == 'BUY' else sell_signals
        
        total_weight = sum(s['weight'] for s in relevant_signals)
        if total_weight == 0:
            return None
            
        # Weighted average of targets
        take_profit_sum = 0
        stop_loss_sum = 0
        
        for signal in signals:
            signal_direction = signal.get('direction', signal.get('side'))
            if signal_direction == direction and signal['strategy'] in contributing_strategies:
                weight = await self.performance_tracker.get_strategy_weight(signal['strategy'])
                
                # Extract target_price and stop_price from metadata
                metadata = signal.get('metadata', {})
                target_price = metadata.get('target_price', signal.get('take_profit', signal['price'] * 1.002))
                stop_price = metadata.get('stop_price', signal.get('stop_loss', signal['price'] * 0.998))
                
                take_profit_sum += target_price * weight
                stop_loss_sum += stop_price * weight
                
        take_profit = take_profit_sum / total_weight
        stop_loss = stop_loss_sum / total_weight
        
        # Get the latest price from one of the signals
        current_price = signals[0]['price'] if signals else 0.0
        
        # Create main strategy name from contributing strategies
        main_strategy = contributing_strategies[0] if contributing_strategies else 'SignalAggregator'
        
        # Déterminer la force du signal basée sur la confiance
        if confidence >= 0.8:
            strength = 'very_strong'
        elif confidence >= 0.6:
            strength = 'strong'
        elif confidence >= 0.4:
            strength = 'moderate'
        else:
            strength = 'weak'
            
        # Calculer la distance de trailing stop basée sur la distance target/stop
        price_distance = abs(take_profit - current_price) / current_price
        # Trailing stop à 50% de la distance target pour laisser de la marge
        trailing_delta = price_distance * 50  # En pourcentage
        
        # Minimum 0.15% pour éviter les trails trop serrés
        trailing_delta = max(trailing_delta, 0.15)
        # Maximum 1.0% pour éviter les trails trop larges
        trailing_delta = min(trailing_delta, 1.0)
        
        return {
            'symbol': symbol,
            'side': direction,  # Use 'side' instead of 'direction' for coordinator compatibility
            'direction': direction,  # Keep for backward compatibility
            'price': current_price,  # Add price field required by coordinator
            'strategy': f"Aggregated_{len(contributing_strategies)}",  # Create strategy name
            'confidence': confidence,
            'strength': strength,  # Ajouter la force du signal
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'trailing_delta': trailing_delta,  # NOUVEAU: Trailing stop activé
            'contributing_strategies': contributing_strategies,
            'buy_score': buy_score,
            'sell_score': sell_score,
            'metadata': {
                'aggregated': True,
                'contributing_strategies': contributing_strategies,
                'strategy_count': len(contributing_strategies),
                'target_price': take_profit,
                'stop_price': stop_loss,
                'trailing_delta': trailing_delta  # NOUVEAU: Ajouter au metadata
            }
        }
        
    def _is_strategy_active(self, strategy: str, regime: str) -> bool:
        """Check if a strategy should be active in current regime"""
        
        # Adaptive strategies are always active
        if strategy in self.STRATEGY_GROUPS['adaptive']:
            return True
            
        # In trending markets, use trend strategies
        if regime == 'TREND':
            return strategy in self.STRATEGY_GROUPS['trend']
            
        # In ranging markets, use mean reversion strategies
        elif regime == 'RANGE':
            return strategy in self.STRATEGY_GROUPS['mean_reversion']
            
        # In undefined regime, all strategies are active
        return True
        
    async def _is_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown period"""
        
        # Check local cooldown
        if symbol in self.last_signal_time:
            time_since_last = datetime.now(timezone.utc) - self.last_signal_time[symbol]
            if time_since_last < self.cooldown_period:
                return True
                
        # Check Redis for distributed cooldown
        cooldown_key = f"signal_cooldown:{symbol}"
        cooldown = await self.redis.get(cooldown_key)
        
        return cooldown is not None
        
    async def set_cooldown(self, symbol: str, duration_seconds: int = 180):
        """Set cooldown for a symbol"""
        cooldown_key = f"signal_cooldown:{symbol}"
        await self.redis.setex(cooldown_key, duration_seconds, "1")