"""
Service de statistiques avancées pour RootTrading.
Calcule et agrège les métriques de trading, performance et activité.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
import logging
import asyncpg
from collections import defaultdict

logger = logging.getLogger(__name__)

class StatisticsService:
    """
    Service principal pour calculer et servir les statistiques de trading.
    """
    
    def __init__(self, data_manager):
        """
        Initialise le service de statistiques.
        
        Args:
            data_manager: Instance de DataManager pour accès DB
        """
        self.data_manager = data_manager
        self.cache = {}
        self.cache_ttl = 60  # Cache TTL en secondes
        
        # Facteur de correction pour ajuster aux profits réels (inclut frais, etc.)
        self.pnl_correction_factor = 0.525  # Basé sur données réelles: 42€/80€
        
    async def get_global_statistics(self) -> Dict[str, Any]:
        """
        Récupère les statistiques globales du système.
        
        Returns:
            Dict contenant toutes les métriques globales
        """
        try:
            # Exécuter toutes les requêtes en parallèle
            results = await asyncio.gather(
                self._get_portfolio_summary(),
                self._get_trading_activity_24h(),
                self._get_performance_metrics(),
                self._get_cycle_statistics(),
                self._get_signal_statistics(),
                return_exceptions=True
            )
            
            portfolio = results[0] if not isinstance(results[0], Exception) else {}
            activity = results[1] if not isinstance(results[1], Exception) else {}
            performance = results[2] if not isinstance(results[2], Exception) else {}
            cycles = results[3] if not isinstance(results[3], Exception) else {}
            signals = results[4] if not isinstance(results[4], Exception) else {}
            
            # Appliquer la correction aux profits pour refléter les frais et coûts réels
            raw_pnl = performance.get('pnl_30d', 0)
            corrected_pnl = raw_pnl * self.pnl_correction_factor
            
            # Formater selon le format attendu par le frontend
            return {
                'totalTrades': activity.get('trades_24h', 0),
                'totalVolume': activity.get('volume_24h', 0),
                'totalPnl': corrected_pnl,
                'totalPnlRaw': raw_pnl,  # Garder la valeur brute pour référence
                'winRate': performance.get('win_rate', 0),
                'profitFactor': performance.get('profit_factor', 0),
                'avgTradeSize': activity.get('volume_24h', 0) / max(activity.get('trades_24h', 1), 1),
                'totalFees': 0,  # À calculer si nécessaire
                'activePositions': cycles.get('cycles_24h', {}).get('active_buy', 0),
                'availableBalance': portfolio.get('total_balance', 0),
                'totalBalance': portfolio.get('total_value_usdc', 0),
                'unrealizedPnl': 0,  # À calculer si nécessaire
                'realizedPnl': corrected_pnl,
                'totalWins': performance.get('total_wins', 0) * self.pnl_correction_factor,
                'totalLosses': performance.get('total_losses', 0) * self.pnl_correction_factor,
                'correctionFactor': self.pnl_correction_factor,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting global statistics: {e}")
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}
    
    async def get_all_symbols_statistics(self) -> Dict[str, Any]:
        """
        Récupère les statistiques détaillées pour TOUS les symboles.
        
        Returns:
            Dict contenant les métriques de tous les symboles
        """
        try:
            # Requête pour récupérer tous les symboles actifs avec des cycles complétés
            symbols_query = """
                SELECT DISTINCT symbol
                FROM trade_cycles
                WHERE status = 'completed'
                    AND completed_at >= NOW() - INTERVAL '30 days'
                ORDER BY symbol;
            """
            
            symbols_result = await self.data_manager.execute_query(symbols_query)
            symbols = [row['symbol'] for row in symbols_result] if symbols_result else []
            
            if not symbols:
                return {
                    'symbols': [],
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            # Requête principale pour récupérer toutes les statistiques en une fois
            main_query = f"""
                WITH symbol_performance AS (
                    SELECT 
                        symbol,
                        COUNT(*) as trades_count,
                        SUM(quantity * price) as total_volume,
                        SUM(profit_loss) as total_pnl,
                        AVG(profit_loss_percent) as avg_pnl_percent,
                        SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
                        AVG(quantity * price) as avg_trade_size
                    FROM (
                        SELECT 
                            tc.symbol,
                            tc.profit_loss,
                            tc.profit_loss_percent,
                            COALESCE(te.quantity, 100) as quantity,
                            COALESCE(te.price, 1) as price
                        FROM trade_cycles tc
                        LEFT JOIN trade_executions te ON tc.symbol = te.symbol 
                            AND te.timestamp BETWEEN tc.created_at - INTERVAL '5 minutes' AND tc.completed_at + INTERVAL '5 minutes'
                        WHERE tc.status = 'completed'
                            AND tc.completed_at >= NOW() - INTERVAL '30 days'
                            AND tc.symbol IN ({','.join(["'" + s + "'" for s in symbols])})
                    ) combined_data
                    GROUP BY symbol
                ),
                price_data AS (
                    SELECT DISTINCT ON (symbol)
                        symbol,
                        close as current_price,
                        LAG(close, 24) OVER (PARTITION BY symbol ORDER BY time) as price_24h_ago
                    FROM market_data
                    WHERE symbol IN ({','.join(["'" + s + "'" for s in symbols])})
                        AND time >= NOW() - INTERVAL '25 hours'
                    ORDER BY symbol, time DESC
                )
                SELECT 
                    sp.symbol,
                    sp.trades_count,
                    sp.total_volume,
                    sp.total_pnl,
                    CASE 
                        WHEN sp.trades_count > 0 THEN (sp.winning_trades::float / sp.trades_count * 100)
                        ELSE 0 
                    END as win_rate,
                    sp.avg_trade_size,
                    COALESCE(pd.current_price, 0) as current_price,
                    CASE 
                        WHEN pd.price_24h_ago > 0 THEN ((pd.current_price - pd.price_24h_ago) / pd.price_24h_ago * 100)
                        ELSE 0 
                    END as price_change_24h
                FROM symbol_performance sp
                LEFT JOIN price_data pd ON sp.symbol = pd.symbol
                ORDER BY sp.total_pnl DESC;
            """
            
            rows = await self.data_manager.execute_query(main_query)
            
            # Formater les résultats
            symbol_stats = []
            for row in rows:
                symbol_stats.append({
                    'symbol': row['symbol'],
                    'trades': row['trades_count'] or 0,
                    'volume': float(row['total_volume'] or 0),
                    'pnl': float(row['total_pnl'] or 0),
                    'winRate': round(float(row['win_rate'] or 0), 2),
                    'avgTradeSize': float(row['avg_trade_size'] or 0),
                    'fees': 0,  # À calculer si nécessaire
                    'lastPrice': float(row['current_price'] or 0),
                    'priceChange24h': round(float(row['price_change_24h'] or 0), 2)
                })
            
            return {
                'symbols': symbol_stats,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting all symbols statistics: {e}")
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}

    async def get_symbol_statistics(self, symbol: str) -> Dict[str, Any]:
        """
        Récupère les statistiques détaillées pour un symbole.
        
        Args:
            symbol: Le symbole à analyser (ex: BTCUSDC)
            
        Returns:
            Dict contenant les métriques du symbole
        """
        try:
            # Requêtes parallèles pour le symbole
            results = await asyncio.gather(
                self._get_symbol_performance(symbol),
                self._get_symbol_trading_activity(symbol),
                self._get_symbol_signal_accuracy(symbol),
                self._get_symbol_cycle_analysis(symbol),
                self._get_symbol_market_metrics(symbol),
                return_exceptions=True
            )
            
            performance = results[0] if not isinstance(results[0], Exception) else {}
            activity = results[1] if not isinstance(results[1], Exception) else {}
            accuracy = results[2] if not isinstance(results[2], Exception) else {}
            cycles = results[3] if not isinstance(results[3], Exception) else {}
            market = results[4] if not isinstance(results[4], Exception) else {}
            
            # Récupérer les vrais prix depuis market_data
            price_query = f"""
                WITH price_data AS (
                    SELECT 
                        close,
                        time,
                        ROW_NUMBER() OVER (ORDER BY time DESC) as rn
                    FROM market_data
                    WHERE symbol = '{symbol}'
                    ORDER BY time DESC
                    LIMIT 25
                ),
                latest_price AS (
                    SELECT close as current_price FROM price_data WHERE rn = 1
                ),
                price_24h_ago AS (
                    SELECT close as old_price FROM price_data WHERE rn = 24
                )
                SELECT 
                    lp.current_price,
                    COALESCE(p24.old_price, lp.current_price) as price_24h_ago,
                    CASE 
                        WHEN p24.old_price > 0 THEN ((lp.current_price - p24.old_price) / p24.old_price * 100)
                        ELSE 0.0
                    END as price_change_24h
                FROM latest_price lp
                LEFT JOIN price_24h_ago p24 ON true
            """
            
            try:
                price_result = await self.data_manager.execute_query(price_query)
                if price_result and len(price_result) > 0:
                    price_row = price_result[0]
                    current_price = float(price_row['current_price'] or 0)
                    price_change_24h = float(price_row['price_change_24h'] or 0)
                else:
                    current_price = 0
                    price_change_24h = 0
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {e}")
                current_price = 0
                price_change_24h = 0
            
            # Formater selon le format attendu par le frontend
            symbol_stat = {
                'symbol': symbol,
                'trades': activity.get('trades_24h', 0),
                'volume': activity.get('volume_24h', 0),
                'pnl': performance.get('total_pnl', 0),
                'winRate': performance.get('win_rate', 0),
                'avgTradeSize': activity.get('volume_24h', 0) / max(activity.get('trades_24h', 1), 1),
                'fees': 0,  # À calculer si nécessaire
                'lastPrice': current_price,
                'priceChange24h': price_change_24h
            }
            
            return {
                'symbols': [symbol_stat],
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting symbol statistics for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol, 'timestamp': datetime.utcnow().isoformat()}
    
    async def get_performance_history(
        self, 
        period: str = '7d',
        interval: str = '1h'
    ) -> Dict[str, Any]:
        """
        Récupère l'historique de performance.
        
        Args:
            period: Période (1d, 7d, 30d, 90d, 1y)
            interval: Intervalle d'agrégation (1h, 1d)
            
        Returns:
            Dict avec historique de performance
        """
        try:
            # Calculer les dates
            period_map = {
                '1d': timedelta(days=1),
                '7d': timedelta(days=7),
                '30d': timedelta(days=30),
                '90d': timedelta(days=90),
                '1y': timedelta(days=365)
            }
            
            interval_map = {
                '1h': 'hour',
                '1d': 'day'
            }
            
            start_time = datetime.utcnow() - period_map.get(period, timedelta(days=7))
            pg_interval = interval_map.get(interval, 'hour')
            
            query = f"""
                WITH time_series AS (
                    SELECT 
                        date_trunc('{pg_interval}', completed_at) as period,
                        symbol,
                        SUM(profit_loss) as period_pnl,
                        SUM(profit_loss_percent) as period_pnl_percent,
                        COUNT(*) as trades_count,
                        SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades
                    FROM trade_cycles
                    WHERE completed_at >= '{start_time.isoformat()}'
                        AND status = 'completed'
                    GROUP BY date_trunc('{pg_interval}', completed_at), symbol
                ),
                aggregated AS (
                    SELECT 
                        period,
                        SUM(period_pnl) as total_pnl,
                        AVG(period_pnl_percent) as avg_pnl_percent,
                        SUM(trades_count) as total_trades,
                        SUM(winning_trades) as total_winning_trades,
                        COUNT(DISTINCT symbol) as active_symbols
                    FROM time_series
                    GROUP BY period
                    ORDER BY period
                )
                SELECT * FROM aggregated;
            """
            
            rows = await self.data_manager.execute_query(query)
            
            # Formatter les résultats
            history = []
            cumulative_pnl = Decimal('0')
            
            for row in rows:
                raw_pnl = row['total_pnl'] or Decimal('0')
                corrected_pnl = raw_pnl * Decimal(str(self.pnl_correction_factor))
                cumulative_pnl += corrected_pnl
                
                win_rate = 0
                if row['total_trades'] > 0:
                    win_rate = (row['total_winning_trades'] / row['total_trades']) * 100
                
                history.append({
                    'timestamp': row['period'].isoformat(),
                    'pnl': float(corrected_pnl),
                    'pnl_raw': float(raw_pnl),
                    'cumulative_pnl': float(cumulative_pnl),
                    'avg_pnl_percent': float(row['avg_pnl_percent'] or 0),
                    'trades': row['total_trades'],
                    'win_rate': round(win_rate, 2),
                    'active_symbols': row['active_symbols']
                })
            
            return {
                'period': period,
                'interval': interval,
                'timestamps': [h['timestamp'] for h in history],
                'pnl': [h['pnl'] for h in history],
                'balance': [h['cumulative_pnl'] for h in history],
                'winRate': [h['win_rate'] for h in history],
                'volume': [h['trades'] * 60 for h in history],  # Approximation du volume
                'data': history,
                'summary': {
                    'total_pnl': float(cumulative_pnl),
                    'total_trades': sum(h['trades'] for h in history),
                    'avg_win_rate': round(sum(h['win_rate'] for h in history) / len(history), 2) if history else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting performance history: {e}")
            return {'error': str(e), 'period': period, 'interval': interval}
    
    async def get_strategy_comparison(self) -> Dict[str, Any]:
        """
        Compare les performances des différentes stratégies individuelles.
        CONSENSUS SUPPRIMÉ - Plus utilisé depuis l'implémentation du consensus adaptatif.
        
        Returns:
            Dict avec comparaison des stratégies individuelles optimisée
        """
        try:
            
            # Liste complète de toutes les stratégies disponibles
            all_available_strategies = [
                'ADX_Direction_Strategy',
                'ATR_Breakout_Strategy', 
                'Bollinger_Touch_Strategy',
                'CCI_Reversal_Strategy',
                'Donchian_Breakout_Strategy',
                'EMA_Cross_Strategy',
                'HullMA_Slope_Strategy',
                'Liquidity_Sweep_Buy_Strategy',
                'MACD_Crossover_Strategy',
                'MultiTF_ConfluentEntry_Strategy',
                'OBV_Crossover_Strategy',
                'PPO_Crossover_Strategy',
                'ParabolicSAR_Bounce_Strategy',
                'Pump_Dump_Pattern_Strategy',
                'ROC_Threshold_Strategy',
                'RSI_Cross_Strategy',
                'Range_Breakout_Confirmation_Strategy',
                'Resistance_Rejection_Strategy',
                'Spike_Reaction_Buy_Strategy',
                'StochRSI_Rebound_Strategy',
                'Stochastic_Oversold_Buy_Strategy',
                'Supertrend_Reversal_Strategy',
                'Support_Breakout_Strategy',
                'TEMA_Slope_Strategy',
                'TRIX_Crossover_Strategy',
                'VWAP_Support_Resistance_Strategy',
                'WilliamsR_Rebound_Strategy',
                'ZScore_Extreme_Reversal_Strategy'
            ]
            
            # 2. Récupérer les stratégies individuelles avec séparation signaux émis vs trades effectués
            individual_query = """
                WITH strategy_trades AS (
                    -- Trouver tous les trades où chaque stratégie a participé
                    -- En analysant les signaux individuels qui ont contribué aux consensus
                    SELECT DISTINCT
                        ts.strategy,
                        tc.id as cycle_id,
                        tc.profit_loss,
                        tc.profit_loss_percent,
                        tc.status,
                        tc.completed_at,
                        ts.side
                    FROM trading_signals ts
                    -- Joindre avec les cycles basés sur la proximité temporelle
                    JOIN trade_cycles tc ON 
                        tc.symbol = ts.symbol 
                        AND tc.side = ts.side
                        AND tc.created_at >= ts.timestamp - INTERVAL '1 minute'
                        AND tc.created_at <= ts.timestamp + INTERVAL '1 minute'
                        AND tc.status = 'completed'
                    WHERE ts.timestamp >= NOW() - INTERVAL '30 days'
                        AND ts.strategy NOT LIKE 'CONSENSUS_%'
                ),
                strategy_performance AS (
                    SELECT 
                        strategy,
                        COUNT(DISTINCT cycle_id) as total_trades_participated,
                        SUM(profit_loss) as total_pnl,
                        AVG(profit_loss) as avg_pnl,
                        AVG(profit_loss_percent) as avg_pnl_percent,
                        SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
                        MAX(profit_loss) as max_gain,
                        MIN(profit_loss) as max_loss,
                        MAX(profit_loss_percent) as max_gain_percent,
                        MIN(profit_loss_percent) as max_loss_percent,
                        -- Séparer les trades BUY et SELL
                        COUNT(DISTINCT CASE WHEN side = 'BUY' THEN cycle_id END) as buy_trades_executed,
                        COUNT(DISTINCT CASE WHEN side = 'SELL' THEN cycle_id END) as sell_trades_executed
                    FROM strategy_trades
                    GROUP BY strategy
                ),
                signal_counts AS (
                    -- Compter le nombre total de signaux émis par chaque stratégie (séparés BUY/SELL)
                    SELECT 
                        strategy,
                        COUNT(*) as total_signals_emitted,
                        COUNT(CASE WHEN side = 'BUY' THEN 1 END) as buy_signals_emitted,
                        COUNT(CASE WHEN side = 'SELL' THEN 1 END) as sell_signals_emitted,
                        COUNT(CASE WHEN metadata->>'part_of_consensus' = 'true' THEN 1 END) as signals_in_consensus
                    FROM trading_signals
                    WHERE timestamp >= NOW() - INTERVAL '30 days'
                        AND strategy NOT LIKE 'CONSENSUS_%'
                    GROUP BY strategy
                )
                SELECT 
                    sc.strategy,
                    -- Signaux émis (tous)
                    COALESCE(sc.total_signals_emitted, 0) as total_signals_emitted,
                    COALESCE(sc.buy_signals_emitted, 0) as buy_signals_emitted,
                    COALESCE(sc.sell_signals_emitted, 0) as sell_signals_emitted,
                    COALESCE(sc.signals_in_consensus, 0) as signals_in_consensus,
                    
                    -- Trades effectués (qui ont vraiment eu lieu)
                    COALESCE(sp.total_trades_participated, 0) as trades_executed,
                    COALESCE(sp.buy_trades_executed, 0) as buy_trades_executed,
                    COALESCE(sp.sell_trades_executed, 0) as sell_trades_executed,
                    
                    -- Performance des trades effectués
                    COALESCE(sp.total_pnl, 0) as total_pnl,
                    COALESCE(sp.avg_pnl, 0) as avg_pnl,
                    COALESCE(sp.avg_pnl_percent, 0) as avg_pnl_percent,
                    COALESCE(sp.winning_trades, 0) as winning_trades,
                    COALESCE(sp.max_gain, 0) as max_gain,
                    COALESCE(sp.max_loss, 0) as max_loss,
                    COALESCE(sp.max_gain_percent, 0) as max_gain_percent,
                    COALESCE(sp.max_loss_percent, 0) as max_loss_percent,
                    
                    -- Taux de conversion signal -> trade
                    CASE 
                        WHEN sc.total_signals_emitted > 0 
                        THEN (COALESCE(sp.total_trades_participated, 0)::float / sc.total_signals_emitted * 100)
                        ELSE 0 
                    END as signal_to_trade_rate,
                    
                    -- Taux de conversion BUY
                    CASE 
                        WHEN sc.buy_signals_emitted > 0 
                        THEN (COALESCE(sp.buy_trades_executed, 0)::float / sc.buy_signals_emitted * 100)
                        ELSE 0 
                    END as buy_conversion_rate,
                    
                    -- Taux de conversion SELL
                    CASE 
                        WHEN sc.sell_signals_emitted > 0 
                        THEN (COALESCE(sp.sell_trades_executed, 0)::float / sc.sell_signals_emitted * 100)
                        ELSE 0 
                    END as sell_conversion_rate
                    
                FROM signal_counts sc
                LEFT JOIN strategy_performance sp ON sc.strategy = sp.strategy
                ORDER BY COALESCE(sp.total_pnl, 0) DESC;
            """
            
            # Exécuter seulement la requête des stratégies individuelles
            individual_rows = await self.data_manager.execute_query(individual_query)
            
            # CONSENSUS SUPPRIMÉ - Section obsolète
            consensus_strategies = []  # Liste vide - plus utilisée
            
            # Créer un dictionnaire avec les stratégies qui ont des données
            strategies_with_data = {}
            for row in individual_rows:
                strategies_with_data[row['strategy']] = row
            
            # Formatter les résultats INDIVIDUELS (inclure TOUTES les stratégies avec données RÉELLES)
            individual_strategies = []
            for strategy_name in all_available_strategies:
                if strategy_name in strategies_with_data:
                    # Stratégie avec des données RÉELLES
                    row = strategies_with_data[strategy_name]
                    
                    # Calculer le win rate basé sur les trades réels où la stratégie a participé
                    win_rate = 0
                    if row['trades_executed'] > 0:
                        win_rate = (row['winning_trades'] / row['trades_executed']) * 100
                    
                    individual_strategies.append({
                        'strategy': strategy_name,
                        'type': 'INDIVIDUAL',
                        
                        # Signaux émis
                        'total_signals_emitted': row['total_signals_emitted'],
                        'buy_signals_emitted': row['buy_signals_emitted'],
                        'sell_signals_emitted': row['sell_signals_emitted'],
                        'signals_in_consensus': row['signals_in_consensus'],
                        
                        # Trades effectués
                        'trades_executed': row['trades_executed'],
                        'buy_trades_executed': row['buy_trades_executed'],
                        'sell_trades_executed': row['sell_trades_executed'],
                        
                        # Taux de conversion
                        'signal_to_trade_rate': float(row['signal_to_trade_rate'] or 0),
                        'buy_conversion_rate': float(row['buy_conversion_rate'] or 0),
                        'sell_conversion_rate': float(row['sell_conversion_rate'] or 0),
                        
                        # Performance
                        'total_pnl': float(row['total_pnl'] or 0),
                        'avg_pnl': float(row['avg_pnl'] or 0),
                        'avg_pnl_percent': float(row['avg_pnl_percent'] or 0),
                        'win_rate': round(win_rate, 2),
                        'max_gain': float(row['max_gain'] or 0),
                        'max_loss': float(row['max_loss'] or 0),
                        'max_gain_percent': float(row['max_gain_percent'] or 0),
                        'max_loss_percent': float(row['max_loss_percent'] or 0),
                        
                        # Champs optimisés pour le nouveau frontend
                        'total_signals_emitted': row['total_signals_emitted'],
                        'buy_signals_emitted': row['buy_signals_emitted'], 
                        'sell_signals_emitted': row['sell_signals_emitted'],
                        'signal_to_trade_rate': float(row['signal_to_trade_rate'] or 0),
                        'buy_conversion_rate': float(row['buy_conversion_rate'] or 0),
                        'sell_conversion_rate': float(row['sell_conversion_rate'] or 0),
                        'trades_executed': row['trades_executed'],
                        'avgPnl': float(row['avg_pnl'] or 0),
                        'avgDuration': 0,
                        'maxDrawdown': float(row['max_loss'] or 0),
                        'sharpeRatio': 0,
                        'trades': row['trades_executed'],
                        'winRate': round(win_rate, 2),
                        'totalPnl': float(row['total_pnl'] or 0)
                    })
                else:
                    # Stratégie sans données (0 signaux émis)
                    individual_strategies.append({
                        'strategy': strategy_name,
                        'type': 'INDIVIDUAL',
                        
                        # Signaux émis
                        'total_signals_emitted': 0,
                        'buy_signals_emitted': 0,
                        'sell_signals_emitted': 0,
                        'signals_in_consensus': 0,
                        
                        # Trades effectués  
                        'trades_executed': 0,
                        'buy_trades_executed': 0,
                        'sell_trades_executed': 0,
                        
                        # Taux de conversion
                        'signal_to_trade_rate': 0,
                        'buy_conversion_rate': 0,
                        'sell_conversion_rate': 0,
                        
                        # Performance
                        'total_pnl': 0,
                        'avg_pnl': 0,
                        'avg_pnl_percent': 0,
                        'win_rate': 0,
                        'max_gain': 0,
                        'max_loss': 0,
                        'max_gain_percent': 0,
                        'max_loss_percent': 0,
                        
                        # Champs pour stratégies sans données
                        'total_signals_emitted': 0,
                        'buy_signals_emitted': 0,
                        'sell_signals_emitted': 0,
                        'signal_to_trade_rate': 0,
                        'buy_conversion_rate': 0,
                        'sell_conversion_rate': 0,
                        'trades_executed': 0,
                        'avgPnl': 0,
                        'avgDuration': 0,
                        'maxDrawdown': 0,
                        'sharpeRatio': 0,
                        'trades': 0,
                        'winRate': 0,
                        'totalPnl': 0
                    })
            
            # Trier par performance (P&L Total décroissant, puis par taux de conversion)
            individual_strategies.sort(key=lambda x: (x.get('totalPnl', 0), x.get('signal_to_trade_rate', 0)), reverse=True)
            
            # Statistiques résumées optimisées
            active_strategies = [s for s in individual_strategies if s.get('total_signals_emitted', 0) > 0]
            profitable_strategies = [s for s in active_strategies if s.get('totalPnl', 0) > 0]
            
            return {
                'strategies': individual_strategies,  # Compatibilité frontend
                'individual_strategies': individual_strategies,
                'active_strategies': active_strategies,
                'profitable_strategies': profitable_strategies,
                'best_strategy': individual_strategies[0]['strategy'] if individual_strategies else None,
                'total_strategies': len(individual_strategies),
                'active_count': len(active_strategies),
                'profitable_count': len(profitable_strategies),
                'avg_conversion_rate': sum(s.get('signal_to_trade_rate', 0) for s in active_strategies) / len(active_strategies) if active_strategies else 0,
                'total_signals_emitted': sum(s.get('total_signals_emitted', 0) for s in individual_strategies),
                'total_trades_executed': sum(s.get('trades_executed', 0) for s in individual_strategies)
            }
            
        except Exception as e:
            logger.error(f"Error getting strategy comparison: {e}")
            return {'error': str(e)}
    
    # Méthodes privées pour requêtes spécifiques
    
    async def _get_portfolio_summary(self) -> Dict[str, Any]:
        """Récupère le résumé du portfolio."""
        query = """
            WITH latest_balances AS (
                SELECT DISTINCT ON (asset) 
                    asset,
                    total,
                    value_usdc,
                    timestamp
                FROM portfolio_balances
                WHERE total > 0
                ORDER BY asset, timestamp DESC
            )
            SELECT 
                SUM(total) as total_balance,
                SUM(value_usdc) as total_value_usdc,
                COUNT(DISTINCT asset) as active_assets
            FROM latest_balances;
        """
        result = await self.data_manager.execute_query(query)
        
        if result:
            row = result[0]
            return {
                'total_balance': float(row['total_balance'] or 0),
                'total_value_usdc': float(row['total_value_usdc'] or 0),
                'active_assets': row['active_assets'] or 0
            }
        return {}
    
    async def _get_trading_activity_24h(self) -> Dict[str, Any]:
        """Récupère l'activité de trading des 24 dernières heures."""
        query = """
            SELECT 
                COUNT(*) as total_trades,
                SUM(quantity * price) as total_volume_usdc,
                SUM(quantity) as total_quantity,
                COUNT(DISTINCT symbol) as active_pairs,
                AVG(price) as avg_price
            FROM trade_executions
            WHERE timestamp >= NOW() - INTERVAL '24 hours';
        """
        result = await self.data_manager.execute_query(query)
        
        if result:
            row = result[0]
            return {
                'trades_24h': row['total_trades'] or 0,
                'volume_24h': float(row['total_volume_usdc'] or 0),
                'volume_quantity': float(row['total_quantity'] or 0),
                'active_pairs': row['active_pairs'] or 0,
                'avg_execution_price': float(row['avg_price'] or 0)
            }
        return {}
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Calcule les métriques de performance globales."""
        query = """
            WITH recent_cycles AS (
                SELECT 
                    profit_loss,
                    profit_loss_percent,
                    status
                FROM trade_cycles
                WHERE completed_at >= NOW() - INTERVAL '30 days'
                    AND status = 'completed'
            )
            SELECT 
                SUM(profit_loss) as total_pnl_30d,
                AVG(profit_loss_percent) as avg_pnl_percent,
                COUNT(*) as total_cycles,
                SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_cycles,
                SUM(CASE WHEN profit_loss > 0 THEN profit_loss ELSE 0 END) as total_wins,
                SUM(CASE WHEN profit_loss < 0 THEN ABS(profit_loss) ELSE 0 END) as total_losses,
                MAX(profit_loss) as best_trade,
                MIN(profit_loss) as worst_trade
            FROM recent_cycles;
        """
        result = await self.data_manager.execute_query(query)
        
        if result:
            row = result[0]
            win_rate = 0
            if row['total_cycles'] and row['total_cycles'] > 0:
                win_rate = (row['winning_cycles'] / row['total_cycles']) * 100
            
            # Calculer le Profit Factor
            profit_factor = 0
            if row['total_losses'] and float(row['total_losses']) > 0:
                profit_factor = float(row['total_wins'] or 0) / float(row['total_losses'])
            
            return {
                'pnl_30d': float(row['total_pnl_30d'] or 0),
                'avg_pnl_percent': float(row['avg_pnl_percent'] or 0),
                'win_rate': round(win_rate, 2),
                'profit_factor': round(profit_factor, 2),
                'total_cycles_30d': row['total_cycles'] or 0,
                'best_trade': float(row['best_trade'] or 0),
                'worst_trade': float(row['worst_trade'] or 0),
                'total_wins': float(row['total_wins'] or 0),
                'total_losses': float(row['total_losses'] or 0)
            }
        return {}
    
    async def _get_cycle_statistics(self) -> Dict[str, Any]:
        """Récupère les statistiques sur les cycles de trading."""
        query = """
            SELECT 
                status,
                COUNT(*) as count
            FROM trade_cycles
            WHERE created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY status;
        """
        result = await self.data_manager.execute_query(query)
        
        stats = {
            'active_buy': 0,
            'completed': 0,
            'cancelled': 0,
            'failed': 0
        }
        
        for row in result:
            stats[row['status']] = row['count']
        
        return {
            'cycles_24h': stats,
            'total_24h': sum(stats.values()),
            'completion_rate': round((stats['completed'] / sum(stats.values()) * 100), 2) if sum(stats.values()) > 0 else 0
        }
    
    async def _get_signal_statistics(self) -> Dict[str, Any]:
        """Récupère les statistiques sur les signaux."""
        query = """
            SELECT 
                side,
                COUNT(*) as count,
                COUNT(DISTINCT symbol) as symbols,
                COUNT(DISTINCT strategy) as strategies
            FROM trading_signals
            WHERE timestamp >= NOW() - INTERVAL '24 hours'
            GROUP BY side;
        """
        result = await self.data_manager.execute_query(query)
        
        buy_signals = 0
        sell_signals = 0
        total_symbols = set()
        total_strategies = set()
        
        for row in result:
            if row['side'] == 'BUY':
                buy_signals = row['count']
            elif row['side'] == 'SELL':
                sell_signals = row['count']
            total_symbols.add(row['symbols'])
            total_strategies.add(row['strategies'])
        
        return {
            'signals_24h': {
                'buy': buy_signals,
                'sell': sell_signals,
                'total': buy_signals + sell_signals
            },
            'active_symbols': len(total_symbols),
            'active_strategies': len(total_strategies)
        }
    
    async def _get_symbol_performance(self, symbol: str) -> Dict[str, Any]:
        """Récupère la performance d'un symbole spécifique."""
        query = f"""
            SELECT 
                SUM(profit_loss) as total_pnl,
                AVG(profit_loss_percent) as avg_pnl_percent,
                COUNT(*) as total_cycles,
                SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_cycles,
                MAX(profit_loss) as best_trade,
                MIN(profit_loss) as worst_trade,
                AVG(EXTRACT(EPOCH FROM (completed_at - created_at))/3600) as avg_duration_hours
            FROM trade_cycles
            WHERE symbol = '{symbol}'
                AND status = 'completed'
                AND completed_at >= NOW() - INTERVAL '30 days';
        """
        result = await self.data_manager.execute_query(query)
        
        if result:
            row = result[0]
            win_rate = 0
            if row['total_cycles'] and row['total_cycles'] > 0:
                win_rate = (row['winning_cycles'] / row['total_cycles']) * 100
            
            return {
                'total_pnl': float(row['total_pnl'] or 0),
                'avg_pnl_percent': float(row['avg_pnl_percent'] or 0),
                'win_rate': round(win_rate, 2),
                'total_cycles': row['total_cycles'] or 0,
                'best_trade': float(row['best_trade'] or 0),
                'worst_trade': float(row['worst_trade'] or 0),
                'avg_duration_hours': round(row['avg_duration_hours'] or 0, 2)
            }
        return {}
    
    async def _get_symbol_trading_activity(self, symbol: str) -> Dict[str, Any]:
        """Récupère l'activité de trading pour un symbole."""
        query = f"""
            SELECT 
                COUNT(*) as trades_24h,
                SUM(quantity) as volume_24h,
                AVG(price) as avg_price,
                MIN(price) as min_price,
                MAX(price) as max_price
            FROM trade_executions
            WHERE symbol = '{symbol}'
                AND timestamp >= NOW() - INTERVAL '24 hours';
        """
        result = await self.data_manager.execute_query(query)
        
        if result:
            row = result[0]
            return {
                'trades_24h': row['trades_24h'] or 0,
                'volume_24h': float(row['volume_24h'] or 0),
                'avg_price': float(row['avg_price'] or 0),
                'price_range': {
                    'min': float(row['min_price'] or 0),
                    'max': float(row['max_price'] or 0)
                }
            }
        return {}
    
    async def _get_symbol_signal_accuracy(self, symbol: str) -> Dict[str, Any]:
        """Calcule la précision des signaux pour un symbole."""
        query = f"""
            WITH signal_results AS (
                SELECT 
                    ts.side,
                    ts.strategy,
                    CASE 
                        WHEN tc.profit_loss > 0 THEN 'profitable'
                        ELSE 'unprofitable'
                    END as result
                FROM trading_signals ts
                LEFT JOIN trade_cycles tc ON tc.symbol = ts.symbol
                    AND tc.created_at >= ts.timestamp - INTERVAL '1 minute'
                    AND tc.created_at <= ts.timestamp + INTERVAL '1 minute'
                WHERE ts.symbol = '{symbol}'
                    AND ts.timestamp >= NOW() - INTERVAL '7 days'
                    AND tc.status = 'completed'
            )
            SELECT 
                strategy,
                COUNT(*) as total_signals,
                SUM(CASE WHEN result = 'profitable' THEN 1 ELSE 0 END) as profitable_signals
            FROM signal_results
            GROUP BY strategy;
        """
        result = await self.data_manager.execute_query(query)
        
        strategies = []
        for row in result:
            accuracy = 0
            if row['total_signals'] > 0:
                accuracy = (row['profitable_signals'] / row['total_signals']) * 100
            
            strategies.append({
                'strategy': row['strategy'],
                'total_signals': row['total_signals'],
                'accuracy': round(accuracy, 2)
            })
        
        return {
            'strategy_accuracy': strategies,
            'best_strategy': max(strategies, key=lambda x: x['accuracy'])['strategy'] if strategies else None
        }
    
    async def _get_symbol_cycle_analysis(self, symbol: str) -> Dict[str, Any]:
        """Analyse détaillée des cycles pour un symbole."""
        query = f"""
            SELECT 
                DATE(completed_at) as date,
                COUNT(*) as cycles,
                SUM(profit_loss) as daily_pnl,
                AVG(profit_loss_percent) as avg_pnl_percent
            FROM trade_cycles
            WHERE symbol = '{symbol}'
                AND status = 'completed'
                AND completed_at >= NOW() - INTERVAL '7 days'
            GROUP BY DATE(completed_at)
            ORDER BY date DESC;
        """
        result = await self.data_manager.execute_query(query)
        
        daily_stats = []
        for row in result:
            daily_stats.append({
                'date': row['date'].isoformat(),
                'cycles': row['cycles'],
                'pnl': float(row['daily_pnl'] or 0),
                'avg_pnl_percent': float(row['avg_pnl_percent'] or 0)
            })
        
        return {
            'daily_performance': daily_stats,
            'best_day': max(daily_stats, key=lambda x: x['pnl'])['date'] if daily_stats else None,
            'worst_day': min(daily_stats, key=lambda x: x['pnl'])['date'] if daily_stats else None
        }
    
    async def _get_symbol_market_metrics(self, symbol: str) -> Dict[str, Any]:
        """Récupère les métriques de marché pour un symbole."""
        query = f"""
            SELECT 
                AVG(close) as avg_price,
                STDDEV(close) as price_volatility,
                AVG(volume) as avg_volume,
                MAX(high) as period_high,
                MIN(low) as period_low
            FROM market_data
            WHERE symbol = '{symbol}'
                AND time >= NOW() - INTERVAL '24 hours';
        """
        result = await self.data_manager.execute_query(query)
        
        if result:
            row = result[0]
            price_range_percent = 0
            if row['avg_price'] and row['avg_price'] > 0:
                price_range_percent = ((row['period_high'] - row['period_low']) / row['avg_price']) * 100
            
            return {
                'avg_price_24h': float(row['avg_price'] or 0),
                'volatility_24h': float(row['price_volatility'] or 0),
                'avg_volume_24h': float(row['avg_volume'] or 0),
                'price_range': {
                    'high': float(row['period_high'] or 0),
                    'low': float(row['period_low'] or 0),
                    'range_percent': round(price_range_percent, 2)
                }
            }
        return {}