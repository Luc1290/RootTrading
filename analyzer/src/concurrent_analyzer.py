"""
Analyseur concurrent optimisé avec asyncio.gather()
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from .indicators.vectorized_indicators import VectorizedIndicators
from .strategy_loader import StrategyLoader

logger = logging.getLogger(__name__)

class ConcurrentAnalyzer:
    """Analyse concurrente des stratégies sur plusieurs symboles"""
    
    def __init__(self, strategy_loader: StrategyLoader, max_workers: int = 4):
        self.strategy_loader = strategy_loader
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def analyze_symbols_concurrent(self, symbols_data: Dict[str, pd.DataFrame]) -> Dict[str, List[Any]]:
        """
        Analyse plusieurs symboles en parallèle
        
        Args:
            symbols_data: Dict {symbol: dataframe}
            
        Returns:
            Dict {symbol: [signals]}
        """
        start_time = time.time()
        
        # Créer les tâches d'analyse pour chaque symbole
        tasks = []
        for symbol, df in symbols_data.items():
            task = self._analyze_symbol_async(symbol, df)
            tasks.append(task)
        
        # Exécuter toutes les analyses en parallèle
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Construire le dictionnaire de résultats
        all_signals = {}
        for i, symbol in enumerate(symbols_data.keys()):
            if isinstance(results[i], Exception):
                logger.error(f"Erreur lors de l'analyse de {symbol}: {results[i]}")
                all_signals[symbol] = []
            else:
                all_signals[symbol] = results[i]
        
        elapsed = time.time() - start_time
        total_signals = sum(len(signals) for signals in all_signals.values())
        logger.info(f"⚡ Analyse concurrente de {len(symbols_data)} symboles en {elapsed:.2f}s - {total_signals} signaux générés")
        
        return all_signals
    
    async def _analyze_symbol_async(self, symbol: str, df: pd.DataFrame) -> List[Any]:
        """Analyse asynchrone d'un symbole"""
        # Calcul vectorisé des indicateurs (CPU-bound, donc dans executor)
        loop = asyncio.get_event_loop()
        indicators = await loop.run_in_executor(
            self.executor,
            VectorizedIndicators.compute_all_indicators,
            df, symbol
        )
        
        # Préparer les données enrichies
        enriched_df = df.copy()
        for name, values in indicators.items():
            if len(values) == len(enriched_df):
                enriched_df[name] = values
            else:
                # Pad with NaN for indicators that need warmup
                enriched_df[name] = pd.Series(values).reindex(enriched_df.index)
        
        # Analyser avec chaque stratégie en parallèle
        strategy_tasks = []
        for strategy_name, strategy in self.strategy_loader.strategies.items():
            task = self._analyze_with_strategy_async(
                strategy, symbol, enriched_df, indicators
            )
            strategy_tasks.append(task)
        
        # Exécuter toutes les stratégies en parallèle pour ce symbole
        strategy_results = await asyncio.gather(*strategy_tasks, return_exceptions=True)
        
        # Filtrer les signaux valides
        signals = []
        for result in strategy_results:
            if not isinstance(result, Exception) and result is not None:
                signals.append(result)
            elif isinstance(result, Exception):
                logger.debug(f"Stratégie error sur {symbol}: {result}")
        
        return signals
    
    async def _analyze_with_strategy_async(self, strategy: Any, symbol: str, 
                                         df: pd.DataFrame, indicators: Dict[str, np.ndarray]) -> Optional[Any]:
        """Exécute une stratégie de manière asynchrone"""
        try:
            # Si la stratégie est async
            if asyncio.iscoroutinefunction(strategy.analyze):
                signal = await strategy.analyze(symbol, df, precomputed_indicators=indicators)
            else:
                # Sinon, l'exécuter dans l'executor
                loop = asyncio.get_event_loop()
                signal = await loop.run_in_executor(
                    self.executor,
                    strategy.analyze,
                    symbol, df
                )
            
            return signal
            
        except Exception as e:
            logger.debug(f"Erreur stratégie {strategy.name} sur {symbol}: {e}")
            return None
    
    async def analyze_timeframes_concurrent(self, symbol: str, timeframes_data: Dict[str, pd.DataFrame]) -> List[Any]:
        """
        Analyse plusieurs timeframes en parallèle pour un symbole
        
        Args:
            symbol: Symbole à analyser
            timeframes_data: Dict {timeframe: dataframe}
            
        Returns:
            List de signaux multi-timeframe
        """
        tasks = []
        for timeframe, df in timeframes_data.items():
            task = self._analyze_symbol_async(f"{symbol}_{timeframe}", df)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Agréger les signaux multi-timeframe
        all_signals = []
        for i, (timeframe, _) in enumerate(timeframes_data.items()):
            if not isinstance(results[i], Exception):
                for signal in results[i]:
                    signal['timeframe'] = timeframe
                    all_signals.append(signal)
        
        return all_signals
    
    def close(self):
        """Ferme l'executor"""
        self.executor.shutdown(wait=True)