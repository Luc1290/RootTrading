#!/usr/bin/env python3
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import json
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Tracks strategy performance and calculates dynamic weights"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        
        # Performance window
        self.lookback_days = 30
        self.min_trades_for_weight = 10
        
        # Weight bounds
        self.min_weight = 0.1
        self.max_weight = 2.0
        self.default_weight = 1.0
        
    async def get_strategy_weight(self, strategy: str) -> float:
        """Get dynamic weight for a strategy based on recent performance"""
        try:
            # Try cached weight first
            cache_key = f"strategy_weight:{strategy}"
            cached = self.redis.get(cache_key)
            
            if cached:
                return float(cached)
                
            # Calculate weight if not cached
            weight = await self._calculate_strategy_weight(strategy)
            
            # Cache for 5 minutes
            # Sauvegarder de façon permanente + cache court terme
            self.redis.set(cache_key, str(weight), expiration=300)
            self.redis.set(f"permanent:{cache_key}", str(weight))  # Sauvegarde permanente
            
            return weight
            
        except Exception as e:
            logger.error(f"Error getting weight for {strategy}: {e}")
            return self.default_weight
            
    async def _calculate_strategy_weight(self, strategy: str) -> float:
        """Calculate strategy weight based on Sharpe ratio and win rate"""
        try:
            # Get performance metrics
            metrics = await self._get_strategy_metrics(strategy)
            
            if not metrics or metrics['trade_count'] < self.min_trades_for_weight:
                return self.default_weight
                
            # Calculate weight components
            sharpe_ratio = metrics.get('sharpe_ratio', 0)
            win_rate = metrics.get('win_rate', 0.5)
            avg_profit = metrics.get('avg_profit', 0)
            
            # Weight formula:
            # Base weight from Sharpe ratio (most important)
            if sharpe_ratio > 0:
                sharpe_weight = 1 + (sharpe_ratio / 2)  # Sharpe of 2 = weight of 2
            else:
                sharpe_weight = max(0.5, 1 + sharpe_ratio)  # Negative Sharpe reduces weight
                
            # Win rate adjustment
            win_rate_multiplier = win_rate / 0.5  # 50% win rate = neutral
            
            # Profit adjustment
            profit_multiplier = 1.0
            if avg_profit > 0:
                profit_multiplier = 1 + (avg_profit / 0.005)  # 0.5% avg profit = 2x
            else:
                profit_multiplier = max(0.5, 1 + (avg_profit / 0.002))
                
            # Combine factors
            weight = sharpe_weight * win_rate_multiplier * profit_multiplier
            
            # Apply bounds
            weight = max(self.min_weight, min(self.max_weight, weight))
            
            logger.info(f"Strategy {strategy} weight: {weight:.2f} "
                       f"(Sharpe={sharpe_ratio:.2f}, WR={win_rate:.2%}, "
                       f"AvgProfit={avg_profit:.4f})")
                       
            return weight
            
        except Exception as e:
            logger.error(f"Error calculating weight: {e}")
            return self.default_weight
            
    async def _get_strategy_metrics(self, strategy: str) -> Dict:
        """Get performance metrics for a strategy"""
        try:
            # Get recent trades
            trades_key = f"strategy_trades:{strategy}"
            recent_trades = self.redis.smembers(trades_key)  # Fallback to set
            
            if not recent_trades:
                return {}
                
            # Parse trades
            trades = []
            cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
            
            for trade_data in recent_trades:
                trade = json.loads(trade_data)
                trade_date = datetime.fromisoformat(trade['timestamp'])
                
                if trade_date > cutoff_date:
                    trades.append(trade)
                    
            if not trades:
                return {}
                
            # Calculate metrics
            profits = [t['profit'] for t in trades]
            wins = [p > 0 for p in profits]
            
            metrics = {
                'trade_count': len(trades),
                'win_rate': sum(wins) / len(wins) if wins else 0,
                'avg_profit': np.mean(profits),
                'total_profit': sum(profits),
                'sharpe_ratio': self._calculate_sharpe(profits),
                'max_drawdown': self._calculate_max_drawdown(profits)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {}
            
    def _calculate_sharpe(self, returns: list, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
            
        returns_array = np.array(returns)
        
        # Annualized return (assuming daily returns)
        mean_return = np.mean(returns_array) * 365
        
        # Annualized volatility
        std_return = np.std(returns_array) * np.sqrt(365)
        
        if std_return == 0:
            return 0.0
            
        sharpe = (mean_return - risk_free_rate) / std_return
        
        return sharpe
        
    def _calculate_max_drawdown(self, returns: list) -> float:
        """Calculate maximum drawdown"""
        if not returns:
            return 0.0
            
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (running_max + 1e-10)
        
        return float(np.min(drawdown))
        
    async def update_all_metrics(self):
        """Update metrics for all strategies"""
        try:
            # Get all strategies
            strategies = [
                'Bollinger', 'Breakout', 'EMA_Cross', 'MACD',
                'Divergence', 'Ride_or_React', 'RSI'
            ]
            
            for strategy in strategies:
                # Force recalculation
                weight = await self._calculate_strategy_weight(strategy)
                
                # Store in Redis
                cache_key = f"strategy_weight:{strategy}"
                # Sauvegarder de façon permanente + cache court terme
            self.redis.set(cache_key, str(weight), expiration=300)
            self.redis.set(f"permanent:{cache_key}", str(weight))  # Sauvegarde permanente
                
            logger.info(f"Updated metrics for {len(strategies)} strategies")
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            
    async def record_trade_result(self, strategy: str, trade_result: Dict):
        """Record a trade result for performance tracking"""
        try:
            trades_key = f"strategy_trades:{strategy}"
            
            trade_data = {
                'strategy': strategy,
                'symbol': trade_result['symbol'],
                'side': trade_result['side'],
                'profit': trade_result['profit'],
                'duration': trade_result['duration'],
                'timestamp': datetime.now().isoformat()
            }
            
            # Store in Redis set (simplified)
            self.redis.sadd(trades_key, json.dumps(trade_data))
            
            # Update metrics cache
            await self.get_strategy_weight(strategy)
            
        except Exception as e:
            logger.error(f"Error recording trade result: {e}")