"""
Analyseur optimisé qui utilise les indicateurs de la base de données
au lieu de les recalculer. Plus rapide et sans duplication.
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from .indicators.db_indicators import db_indicators
from .strategy_loader import StrategyLoader

logger = logging.getLogger(__name__)

class OptimizedAnalyzer:
    """
    Analyseur optimisé qui récupère les indicateurs de la DB
    au lieu de les recalculer
    """
    
    def __init__(self, strategy_loader: StrategyLoader, max_workers: int = 4):
        self.strategy_loader = strategy_loader
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def analyze_symbols_optimized(self, symbols: List[str]) -> Dict[str, List[Any]]:
        """
        Analyse plusieurs symboles en utilisant les données enrichies de la DB
        
        Args:
            symbols: Liste des symboles à analyser
            
        Returns:
            Dict {symbol: [signals]}
        """
        start_time = time.time()
        
        # Créer les tâches d'analyse pour chaque symbole
        tasks = []
        for symbol in symbols:
            task = self._analyze_symbol_from_db(symbol)
            tasks.append(task)
        
        # Exécuter toutes les analyses en parallèle
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Construire le dictionnaire de résultats
        all_signals = {}
        for i, symbol in enumerate(symbols):
            if isinstance(results[i], Exception):
                logger.error(f"❌ Erreur lors de l'analyse de {symbol}: {results[i]}")
                all_signals[symbol] = []
            else:
                all_signals[symbol] = results[i] or []
        
        elapsed = time.time() - start_time
        total_signals = sum(len(signals) for signals in all_signals.values())
        
        logger.info(f"⚡ Analyse optimisée de {len(symbols)} symboles en {elapsed:.2f}s - {total_signals} signaux générés")
        
        return all_signals
    
    async def _analyze_symbol_from_db(self, symbol: str) -> List[Any]:
        """
        Analyse un symbole en utilisant les données enrichies de la DB
        
        Args:
            symbol: Symbole à analyser
            
        Returns:
            Liste des signaux générés
        """
        try:
            logger.debug(f"🔍 Starting analysis for {symbol}")
            
            # 1. Récupérer les données enrichies de la DB (OHLCV + indicateurs)
            enriched_df = db_indicators.get_optimized_indicators(symbol, limit=200)
            
            if enriched_df is None or enriched_df.empty:
                logger.warning(f"⚠️ Pas de données enrichies pour {symbol}")
                return []
            
            # 2. Vérifier la qualité des données
            if len(enriched_df) < 50:
                logger.warning(f"⚠️ Pas assez de données pour {symbol}: {len(enriched_df)} chandelles")
                return []
            
            # 3. Convertir en format compatible avec les stratégies existantes
            indicators = self._dataframe_to_indicators_dict(enriched_df)
            
            # 4. Analyser avec chaque stratégie en parallèle pour ce symbole
            symbol_strategies = self.strategy_loader.strategies.get(symbol, {})
            if not symbol_strategies:
                logger.warning(f"⚠️ Aucune stratégie pour {symbol}")
                return []
            
            logger.debug(f"🔍 Found {len(symbol_strategies)} strategies for {symbol}")
                
            strategy_tasks = []
            for strategy_name, strategy in symbol_strategies.items():
                task = self._analyze_with_strategy_async(
                    strategy, symbol, enriched_df, indicators
                )
                strategy_tasks.append(task)
            
            # 5. Exécuter toutes les stratégies en parallèle
            strategy_results = await asyncio.gather(*strategy_tasks, return_exceptions=True)
            
            # 6. Filtrer et valider les signaux
            signals = []
            strategy_names = list(symbol_strategies.keys())
            for i, result in enumerate(strategy_results):
                strategy_name = strategy_names[i]
                
                if isinstance(result, Exception):
                    logger.debug(f"⚠️ Erreur stratégie {strategy_name} sur {symbol}: {result}")
                elif result is not None:
                    # Valider et enrichir le signal
                    validated_signal = self._validate_and_enrich_signal(result, symbol, enriched_df)
                    if validated_signal:
                        signals.append(validated_signal)
            
            if signals:
                logger.info(f"📊 {symbol}: {len(signals)} signaux générés")
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse {symbol}: {e}")
            return []
    
    def _dataframe_to_indicators_dict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Convertit le DataFrame enrichi en dictionnaire d'indicateurs
        compatible avec les stratégies existantes
        """
        indicators = {}
        
        # Indicateurs de base - Binance EMAs (7/26/99)
        indicator_columns = [
            'rsi_14', 'ema_7', 'ema_26', 'ema_99', 'sma_20', 'sma_50',
            'macd_line', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_position', 'bb_width',
            'atr_14', 'adx_14', 'plus_di', 'minus_di',
            'stoch_k', 'stoch_d', 'stoch_rsi',
            'williams_r', 'cci_20', 'vwap_10',
            'roc_10', 'roc_20', 'obv',
            'momentum_10', 'volume_ratio'
        ]
        
        for col in indicator_columns:
            if col in df.columns:
                values = df[col].values
                # Remplacer les NaN par des valeurs moyennes pour éviter les erreurs
                # Utiliser pd.isna() au lieu de np.isnan() pour compatibilité avec Decimal
                if pd.isna(values).any():
                    values = pd.Series(values).fillna(method='ffill').fillna(method='bfill').values
                indicators[col] = values
                
        # Ajouter des alias pour compatibilité avec les stratégies existantes
        if 'macd_line' in indicators:
            indicators['macd'] = indicators['macd_line']
        if 'macd_histogram' in indicators:
            indicators['macd_hist'] = indicators['macd_histogram']
            
        # SUPPRIMÉ: Plus de recalcul d'indicateurs - tout vient de la DB
        # Les EMAs manquantes doivent être ajoutées dans le gateway/technical_indicators
        
        return indicators
    
    async def _analyze_with_strategy_async(self, strategy, symbol: str, 
                                         df: pd.DataFrame, indicators: Dict) -> Optional[Dict]:
        """Analyse avec une stratégie spécifique de manière asynchrone"""
        try:
            loop = asyncio.get_event_loop()
            
            # Exécuter l'analyse dans l'executor pour éviter de bloquer
            result = await loop.run_in_executor(
                self.executor,
                self._run_strategy_analysis,
                strategy, symbol, df, indicators
            )
            
            return result
            
        except Exception as e:
            logger.debug(f"Erreur stratégie {strategy.__class__.__name__} sur {symbol}: {e}")
            return None
    
    def _run_strategy_analysis(self, strategy, symbol: str, 
                             df: pd.DataFrame, indicators: Dict) -> Optional[Dict]:
        """Exécute l'analyse d'une stratégie de manière synchrone"""
        try:
            logger.debug(f"🔍 Running {strategy.__class__.__name__} for {symbol}")
            
            # La plupart des stratégies ont une méthode analyze()
            if hasattr(strategy, 'analyze'):
                result = strategy.analyze(symbol, df, indicators)
                logger.debug(f"🔍 {strategy.__class__.__name__} returned: {result is not None}")
                return result
            elif hasattr(strategy, 'generate_signal'):
                return strategy.generate_signal(symbol, df, indicators)
            else:
                logger.warning(f"Stratégie {strategy.__class__.__name__} sans méthode d'analyse")
                return None
                
        except Exception as e:
            logger.debug(f"Erreur exécution stratégie {strategy.__class__.__name__}: {e}")
            return None
    
    def _validate_and_enrich_signal(self, signal, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """
        Valide et enrichit un signal avec des métriques de qualité
        """
        if not signal:
            return None
        
        # Convertir StrategySignal en dict si nécessaire
        if hasattr(signal, 'dict'):
            signal_dict = signal.dict()
        elif isinstance(signal, dict):
            signal_dict = signal
        else:
            logger.debug(f"Signal invalide pour {symbol}: type non supporté {type(signal)}")
            return None
            
        # Vérifications de base
        required_fields = ['strategy', 'symbol', 'side', 'price', 'confidence']
        if not all(field in signal_dict for field in required_fields):
            logger.debug(f"Signal invalide pour {symbol}: champs manquants")
            return None
            
        # Enrichir avec des métriques de contexte
        try:
            current_price = df['close'].iloc[-1]
            
            # Ajouter des métriques de contexte du marché
            signal_dict.update({
                'current_price': current_price,
                'timestamp': df.index[-1].isoformat(),
                'market_context': self._get_market_context(df),
                'signal_quality': self._calculate_signal_quality(signal_dict, df)
            })
            
            # Valider la cohérence du prix
            if abs(signal_dict['price'] - current_price) / current_price > 0.05:  # 5% de différence max
                logger.warning(f"Signal {symbol}: prix incohérent {signal_dict['price']} vs {current_price}")
                return None
                
            return signal_dict
            
        except Exception as e:
            logger.debug(f"Erreur validation signal {symbol}: {e}")
            return None
    
    def _get_market_context(self, df: pd.DataFrame) -> Dict:
        """Calcule le contexte de marché actuel"""
        try:
            recent_data = df.tail(20)
            
            # Volatilité récente
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(24 * 60)  # Annualisée pour 1min
            
            # Tendance générale
            price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            
            # Volume moyen
            avg_volume = recent_data['volume'].mean()
            current_volume = recent_data['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            return {
                'volatility': round(volatility, 4),
                'trend_pct': round(price_change * 100, 2),
                'volume_ratio': round(volume_ratio, 2),
                'is_high_volume': volume_ratio > 1.5  # STANDARDISÉ: Très bon volume
            }
            
        except Exception:
            return {}
    
    def _calculate_signal_quality(self, signal: Dict, df: pd.DataFrame) -> float:
        """
        Calcule un score de qualité du signal (0-1)
        Basé sur la cohérence des indicateurs et le contexte de marché
        """
        try:
            quality_score = signal.get('confidence', 0.5)
            
            # Bonus pour les signaux avec du volume
            market_context = self._get_market_context(df)
            if market_context.get('is_high_volume', False):
                quality_score += 0.1
                
            # Bonus pour les tendances claires
            if abs(market_context.get('trend_pct', 0)) > 2:  # Tendance > 2%
                quality_score += 0.1
                
            # Pénalité pour la volatilité excessive
            if market_context.get('volatility', 0) > 0.5:
                quality_score -= 0.1
                
            return max(0.0, min(1.0, quality_score))
            
        except Exception:
            return 0.5
    
    async def close(self):
        """Ferme les ressources"""
        if self.executor:
            self.executor.shutdown(wait=True)
        db_indicators.close()