#!/usr/bin/env python3
"""
Signal Aggregator principal - Version refactorée et modulaire.
Agrège les signaux de plusieurs stratégies et résout les conflits.
"""

import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING, Union
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import json
import numpy as np
import time

# Import des modules séparés
from market_data_accumulator import MarketDataAccumulator
from signal_validator import SignalValidator
from signal_processor import SignalProcessor  
from technical_analysis import TechnicalAnalysis
from regime_filtering import RegimeFiltering
from signal_metrics import SignalMetrics

logger = logging.getLogger(__name__)

# Type alias pour le regime de marché
if TYPE_CHECKING:
    from enhanced_regime_detector import MarketRegime
    MarketRegimeType = Union[MarketRegime, Any]
else:
    MarketRegimeType = Any

from enhanced_regime_detector import EnhancedRegimeDetector, MarketRegime


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
        
        # Accumulateur de données de marché pour construire l'historique
        self.market_data_accumulator = MarketDataAccumulator(max_history=200)
        
        # Nouveau détecteur de régime amélioré
        self.enhanced_regime_detector = EnhancedRegimeDetector(redis_client)
        # Connecter l'accumulateur au détecteur
        self.enhanced_regime_detector.set_market_data_accumulator(self.market_data_accumulator)
        logger.info("✅ Enhanced Regime Detector activé avec accumulateur historique")
        
        # Nouveaux modules d'analyse avancée
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from multi_timeframe_confluence import MultiTimeframeConfluence
            self.confluence_analyzer = MultiTimeframeConfluence(redis_client)
            logger.info("✅ Analyseur de confluence multi-timeframes activé")
        except ImportError as e:
            logger.error(f"❌ Erreur import confluence analyzer: {e}")
            self.confluence_analyzer = None
        
        try:
            from market_structure_detector import MarketStructureDetector
            self.structure_detector = MarketStructureDetector(redis_client)
            logger.info("✅ Détecteur de structure de marché activé")
        except ImportError as e:
            logger.error(f"❌ Erreur import structure detector: {e}")
            self.structure_detector = None
        
        try:
            from momentum_cross_timeframe import MomentumCrossTimeframe
            self.momentum_analyzer = MomentumCrossTimeframe(redis_client)
            logger.info("✅ Analyseur de momentum cross-timeframe activé")
        except ImportError as e:
            logger.error(f"❌ Erreur import momentum analyzer: {e}")
            self.momentum_analyzer = None
        
        try:
            from adaptive_regime_enhanced import AdaptiveRegimeEnhanced
            self.adaptive_regime = AdaptiveRegimeEnhanced(redis_client)
            logger.info("✅ Système de régime adaptatif amélioré activé")
        except ImportError as e:
            logger.error(f"❌ Erreur import adaptive regime: {e}")
            self.adaptive_regime = None
        
        # Signal buffer for aggregation
        self.signal_buffer = defaultdict(list)
        self.last_signal_time = {}
        
        # Cache incrémental pour EMAs lisses (comme dans Gateway WebSocket)
        self.ema_incremental_cache = defaultdict(lambda: defaultdict(dict))
        
        # Hybrid approach: load thresholds from config
        from shared.src.config import (SIGNAL_COOLDOWN_MINUTES, VOTE_THRESHOLD, 
                                     CONFIDENCE_THRESHOLD)
        self.cooldown_period = timedelta(minutes=SIGNAL_COOLDOWN_MINUTES)
        
        # Voting thresholds adaptatifs - HYBRID OPTIMIZED (plus réactif)
        self.min_vote_threshold = VOTE_THRESHOLD
        self.min_confidence_threshold = CONFIDENCE_THRESHOLD
        
        # Seuils spéciaux pour RANGE_TIGHT - HYBRID (plus permissif)
        self.range_tight_vote_threshold = max(0.30, VOTE_THRESHOLD - 0.05)
        self.range_tight_confidence_threshold = max(0.55, CONFIDENCE_THRESHOLD - 0.05)
        
        # Initialiser les modules
        self._init_modules()
        
        # Monitoring stats
        try:
            from .monitoring_stats import SignalMonitoringStats
            self.monitoring_stats = SignalMonitoringStats(redis_client)
        except ImportError:
            # Fallback pour import relatif
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from monitoring_stats import SignalMonitoringStats
            self.monitoring_stats = SignalMonitoringStats(redis_client)
        
        # Bayesian strategy weights avec DB
        try:
            from .bayesian_weights import BayesianStrategyWeights
            # Utiliser le db_pool passé au constructeur
            db_pool = getattr(self, 'db_pool', None)
            self.bayesian_weights = BayesianStrategyWeights(redis_client, db_pool)
            if db_pool:
                logger.info("✅ Pondération bayésienne avec sauvegarde PostgreSQL activée")
            else:
                logger.info("✅ Pondération bayésienne avec cache Redis activée")
        except ImportError:
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from bayesian_weights import BayesianStrategyWeights
            db_pool = getattr(self, 'db_pool', None)
            self.bayesian_weights = BayesianStrategyWeights(redis_client, db_pool)
            if db_pool:
                logger.info("✅ Pondération bayésienne avec sauvegarde PostgreSQL activée")
            else:
                logger.info("✅ Pondération bayésienne avec cache Redis activée")
        
        # Dynamic thresholds
        try:
            from .dynamic_thresholds import DynamicThresholdManager
            self.dynamic_thresholds = DynamicThresholdManager(
                redis_client,
                target_signal_rate=0.08  # 8% des signaux devraient passer (sélectif)
            )
            logger.info("✅ Seuils dynamiques adaptatifs activés")
        except ImportError:
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from dynamic_thresholds import DynamicThresholdManager
            self.dynamic_thresholds = DynamicThresholdManager(
                redis_client,
                target_signal_rate=0.08
            )
            logger.info("✅ Seuils dynamiques adaptatifs activés")
    
    def _init_modules(self):
        """Initialise les modules séparés"""
        # Analyse technique
        self.technical_analysis = TechnicalAnalysis(self.redis, self.ema_incremental_cache)
        
        # Validation des signaux
        self.signal_validator = SignalValidator(self.redis, self.ema_incremental_cache)
        
        # Traitement des signaux
        self.signal_processor = SignalProcessor(self.redis)
        
        # Filtrage par régime
        self.regime_filtering = RegimeFiltering(self.technical_analysis)
        
        # Métriques et boost
        self.signal_metrics = SignalMetrics(self.performance_tracker)
        
    async def _update_market_data_history(self, symbol: str) -> None:
        """Met à jour l'historique des données de marché pour un symbole"""
        try:
            # Récupérer les données actuelles depuis Redis
            key = f"market_data:{symbol}:15m"
            data = self.redis.get(key)
            if data:
                parsed = json.loads(data) if isinstance(data, str) else data
                if isinstance(parsed, dict) and 'ultra_enriched' in parsed:
                    # Ajouter les valeurs OHLC manquantes si nécessaires
                    if 'open' not in parsed:
                        close_price = parsed.get('close', 0)
                        parsed['open'] = close_price
                        parsed['high'] = close_price * 1.001  # +0.1%
                        parsed['low'] = close_price * 0.999   # -0.1%
                    
                    # Ajouter à l'accumulateur
                    self.market_data_accumulator.add_market_data(symbol, parsed)
                    
                    count = self.market_data_accumulator.get_history_count(symbol)
                    if count % 10 == 0:  # Log tous les 10 points
                        logger.info(f"📈 Historique {symbol}: {count} points accumulés")
                        
        except Exception as e:
            logger.error(f"Erreur mise à jour historique {symbol}: {e}")

    async def process_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a raw signal and return aggregated decision with ultra-confluent validation"""
        try:
            symbol = signal['symbol']
            strategy = signal['strategy']
            
            # Normalize strategy name by removing '_Strategy' suffix
            strategy = strategy.replace('_Strategy', '')
            
            # Mettre à jour l'historique des données de marché
            await self._update_market_data_history(symbol)
            
            # NOUVEAU: Gestion des signaux ultra-confluents avec scoring
            is_ultra_confluent = signal.get('metadata', {}).get('ultra_confluence', False)
            signal_score = signal.get('metadata', {}).get('total_score')
            
            # Normalize side value
            if 'side' in signal:
                side = signal['side'].upper()
                if side in ('BUY', 'LONG'):
                    signal['side'] = 'BUY'
                elif side in ('SELL', 'SHORT'):
                    signal['side'] = 'SELL'
                else:
                    logger.warning(f"Unknown side value: {side}")
                    return None
                    
            # NOUVEAU: Traitement prioritaire pour signaux ultra-confluents de haute qualité
            if is_ultra_confluent and signal_score:
                logger.info(f"🔥 Signal ULTRA-CONFLUENT {strategy} {signal['side']} {symbol}: score={signal_score:.1f}")
                
                # Signaux de qualité institutionnelle (95+) passent avec traitement express
                if signal_score >= 95:
                    logger.info(f"⭐ SIGNAL INSTITUTIONNEL accepté directement: {symbol} score={signal_score:.1f}")
                    return await self.signal_processor.process_institutional_signal(signal)
                # Signaux excellents (85+) ont priorité mais validation allégée
                elif signal_score >= 85:
                    logger.info(f"✨ SIGNAL EXCELLENT priorité haute: {symbol} score={signal_score:.1f}")
                    return await self.signal_processor.process_excellent_signal(signal, self.set_cooldown)
                # Signaux faibles (<50) sont rejetés immédiatement
                elif signal_score < 50:
                    logger.info(f"❌ Signal ultra-confluent rejeté (score faible): {symbol} score={signal_score:.1f}")
                    return None
            
            # NOUVEAU: Validation multi-timeframe avec 5m (SWING CRYPTO)
            # Validation 5m pour swing trading, filtrage plus strict
            if not await self.signal_validator.validate_signal_with_higher_timeframe(signal):
                logger.info(f"Signal {strategy} {signal['side']} sur {symbol} rejeté par validation 5m swing")
                return None
            
            # Handle timestamp conversion
            timestamp = self.signal_processor.get_signal_timestamp(signal)
            
            # Check cooldown
            if await self._is_in_cooldown(symbol):
                logger.debug(f"Symbol {symbol} in cooldown, ignoring signal")
                return None
                
            # Add to buffer
            self.signal_buffer[symbol].append(signal)
            
            # Clean old signals (keep only last 300 seconds for confluence)
            cutoff_time = timestamp - timedelta(seconds=300)
            self.signal_buffer[symbol] = [
                s for s in self.signal_buffer[symbol]
                if self.signal_processor.get_signal_timestamp(s) > cutoff_time
            ]
            
            # Check if we have enough signals to make a decision - MODE CONFLUENCE
            buffer_size = len(self.signal_buffer[symbol])
            
            # Logique d'attente intelligente : attendre plus de signaux ou un délai
            if buffer_size < 2:
                # Si on a seulement 1 signal, attendre 300 secondes pour voir si d'autres arrivent
                first_signal_time = self.signal_processor.get_signal_timestamp(self.signal_buffer[symbol][0])
                time_since_first = timestamp - first_signal_time

                if time_since_first.total_seconds() < 300:
                    logger.info(f"🕐 Signal unique pour {symbol}, attente {300 - time_since_first.total_seconds():.0f}s pour confluence")
                    return None  # Attendre plus de signaux
                else:
                    logger.info(f"⏰ Délai d'attente écoulé pour {symbol}, traitement du signal unique")
                    # Continuer avec le signal unique après délai
            else:
                # Analyser rapidement si les signaux sont dans la même direction
                sides = [s.get('side', s.get('side', '')).upper() for s in self.signal_buffer[symbol]]
                unique_sides = set(sides)
                
                if len(unique_sides) == 1:
                    logger.info(f"🎯 Confluence détectée pour {symbol}: {buffer_size} signaux {list(unique_sides)[0]}, traitement immédiat")
                else:
                    logger.info(f"⚡ Signaux multiples détectés pour {symbol}: {buffer_size} signaux mixtes, analyse en cours")
                
            # Get market regime FIRST pour filtrage intelligent (enhanced if available, sinon fallback)
            if self.enhanced_regime_detector:
                # Utiliser la version async - le Signal Aggregator s'exécute déjà dans un contexte async
                regime, regime_metrics = await self.enhanced_regime_detector.get_detailed_regime(symbol)
                
                # NOUVEAU: Filtrage intelligent basé sur les régimes Enhanced
                signal_filtered = await self.regime_filtering.apply_enhanced_regime_filtering(
                    signal, regime, regime_metrics, is_ultra_confluent, signal_score, len(self.signal_buffer[symbol])
                )
                if not signal_filtered:
                    return None  # Signal rejeté par le filtrage intelligent
                
                # Calculate aggregated signal with regime-adaptive weights
                aggregated = await self._aggregate_signals_enhanced(
                    symbol, 
                    self.signal_buffer[symbol],
                    regime,
                    regime_metrics
                )
            else:
                # Fallback vers l'ancien système
                regime = await self.regime_detector.get_regime(symbol)
                aggregated = await self._aggregate_signals(
                    symbol, 
                    self.signal_buffer[symbol],
                    regime
                )
            
            if aggregated:
                # Store source signals count before clearing buffer
                source_signals_count = len(self.signal_buffer[symbol])
                
                # Clear buffer after successful aggregation
                self.signal_buffer[symbol].clear()
                self.last_signal_time[symbol] = timestamp
                
                # Add metadata
                if self.enhanced_regime_detector and hasattr(regime, 'value'):
                    aggregated.update({
                        'aggregation_method': 'enhanced_weighted_vote',
                        'regime': regime.value,
                        'regime_metrics': regime_metrics,
                        'timestamp': timestamp.isoformat(),
                        'source_signals': source_signals_count
                    })
                else:
                    aggregated.update({
                        'aggregation_method': 'weighted_vote',
                        'regime': regime,
                        'timestamp': timestamp.isoformat(),
                        'source_signals': source_signals_count
                    })
                
                return aggregated
                
            return None
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return None
    
    async def _aggregate_signals(self, symbol: str, signals: List[Dict], 
                               regime: str) -> Optional[Dict[str, Any]]:
        """Aggregate multiple signals into a single decision"""

        # Group signals by side
        BUY_signals = []
        SELL_signals = []

        for signal in signals:
            strategy = signal['strategy']
            
            # Check if strategy is appropriate for current regime
            if not self.regime_filtering.is_strategy_active(strategy, regime):
                logger.debug(f"Strategy {strategy} not active in {regime} regime")
                continue
                
            # Get strategy weight based on performance
            weight = await self.performance_tracker.get_strategy_weight(strategy)
            
            # Apply confidence threshold with enhanced filtering for mixed signals
            confidence = signal.get('confidence', 0.5)
            signal_is_ultra_confluent = signal.get('metadata', {}).get('ultra_confluence', False)
            signal_score = signal.get('metadata', {}).get('total_score')
            
            # NOUVEAU: Seuils adaptatifs selon le régime ET le type de signal
            if regime in ["RANGE_TIGHT", "RANGE_VOLATILE", "CHOPPY"]:
                # Régimes difficiles : seuils plus stricts
                if signal_is_ultra_confluent and signal_score:
                    min_threshold = 0.75  # Plus strict pour ultra-confluent en range
                else:
                    min_threshold = max(0.65, self.min_confidence_threshold)
            elif regime in ["WEAK_TREND_UP", "WEAK_TREND_DOWN"]:
                # Tendances faibles : seuils modérés
                if signal_is_ultra_confluent and signal_score:
                    min_threshold = 0.70
                else:
                    min_threshold = max(0.60, self.min_confidence_threshold)
            else:  # STRONG_TREND_*, TREND_*
                # Tendances fortes : seuils standards
                if signal_is_ultra_confluent and signal_score:
                    min_threshold = 0.65
                else:
                    min_threshold = self.min_confidence_threshold
                
            if confidence < min_threshold:
                logger.debug(f"Signal {strategy} filtré: confidence {confidence:.2f} < {min_threshold:.2f} (régime: {regime})")
                continue

            # Get side (handle both 'side' and 'side' keys)
            side = signal.get('side', signal.get('side'))
            if side in ['BUY', 'BUY']:
                side = 'BUY'
            elif side in ['SELL', 'SELL']:
                side = 'SELL'

            # Weighted signal
            weighted_signal = {
                'strategy': strategy,
                'side': side,
                'confidence': confidence,
                'weight': weight,
                'score': confidence * weight
            }

            if side == 'BUY':
                BUY_signals.append(weighted_signal)
                # Enregistrer l'acceptation dans les stats
                self.monitoring_stats.record_signal_decision(
                    strategy=strategy,
                    regime=regime.name if hasattr(regime, 'name') else str(regime),
                    symbol=symbol,
                    accepted=True,
                    confidence=confidence
                )
            elif side == 'SELL':
                SELL_signals.append(weighted_signal)
                # Enregistrer l'acceptation dans les stats
                self.monitoring_stats.record_signal_decision(
                    strategy=strategy,
                    regime=regime.name if hasattr(regime, 'name') else str(regime),
                    symbol=symbol,
                    accepted=True,
                    confidence=confidence
                )

        # Calculate total scores with quality tracking
        BUY_score = sum(s['score'] for s in BUY_signals)
        SELL_score = sum(s['score'] for s in SELL_signals)
        
        # Log signal quality breakdown for debugging
        if BUY_signals or SELL_signals:
            ultra_buy = [s for s in BUY_signals if s.get('signal_type') == 'ultra_confluent']
            classic_buy = [s for s in BUY_signals if s.get('signal_type') == 'classic']
            ultra_sell = [s for s in SELL_signals if s.get('signal_type') == 'ultra_confluent']
            classic_sell = [s for s in SELL_signals if s.get('signal_type') == 'classic']
            
            logger.debug(f"📊 {symbol} signaux: "
                        f"BUY ultra={len(ultra_buy)} classic={len(classic_buy)} "
                        f"SELL ultra={len(ultra_sell)} classic={len(classic_sell)}")

        # NOUVEAU: Détection de signaux contradictoires
        if BUY_signals and SELL_signals:
            # Les signaux sont opposés - c'est un conflit, pas une confluence!
            total_signals = len(BUY_signals) + len(SELL_signals)
            buy_ratio = len(BUY_signals) / total_signals
            sell_ratio = len(SELL_signals) / total_signals
            
            # Si les signaux sont trop équilibrés (40-60%), rejeter
            if 0.4 <= buy_ratio <= 0.6:
                logger.warning(f"⚠️ Signaux contradictoires pour {symbol}: "
                             f"{len(BUY_signals)} BUY vs {len(SELL_signals)} SELL - REJET")
                return None
            
            # Si un côté domine fortement (>70%), l'accepter mais réduire la confiance
            confidence_penalty = 0.2  # Pénalité pour signaux opposés
        else:
            confidence_penalty = 0.0
        
        # Determine side
        if BUY_score > SELL_score and BUY_score >= self.min_vote_threshold:
            side = 'BUY'
            confidence = max(0.5, (BUY_score / (BUY_score + SELL_score)) - confidence_penalty)
            contributing_strategies = [s['strategy'] for s in BUY_signals]
        elif SELL_score > BUY_score and SELL_score >= self.min_vote_threshold:
            side = 'SELL'
            confidence = max(0.5, (SELL_score / (BUY_score + SELL_score)) - confidence_penalty)
            contributing_strategies = [s['strategy'] for s in SELL_signals]
        else:
            # No clear signal
            logger.debug(f"No clear signal for {symbol}: BUY={BUY_score:.2f}, SELL={SELL_score:.2f}")
            return None
            
        # Calculate averaged stop loss (plus de take profit avec TrailingStop pur)
        relevant_signals = BUY_signals if side == 'BUY' else SELL_signals

        total_weight = sum(s['weight'] for s in relevant_signals)
        if total_weight == 0:
            return None
            
        # NOUVEAU: Calcul de stop-loss adaptatif avec ATR dynamique
        stop_loss_sum = 0
        atr_stop_loss = await self.technical_analysis.calculate_atr_based_stop_loss(symbol, signals[0]['price'], side)
        
        for signal in signals:
            signal_side = signal.get('side', signal.get('side'))
            if signal_side == side and signal['strategy'] in contributing_strategies:
                weight = await self.performance_tracker.get_strategy_weight(signal['strategy'])
                
                # Prioriser stop ATR si disponible, sinon fallback
                if atr_stop_loss is not None:
                    stop_price = atr_stop_loss
                    logger.info(f"🎯 Stop ATR adaptatif utilisé pour {symbol}: {stop_price:.4f}")
                else:
                    # Extract stop_price from metadata (fallback)
                    metadata = signal.get('metadata', {})
                    # Stop-loss correct selon le side: BUY stop en dessous, SELL stop au dessus - CRYPTO OPTIMIZED
                    default_stop = signal['price'] * (1.08 if side == 'SELL' else 0.92)  # 8% crypto stops (était 0.2%!)
                    stop_price = metadata.get('stop_price', signal.get('stop_loss', default_stop))
                    logger.debug(f"📊 Stop fixe utilisé pour {symbol}: {stop_price:.4f}")
                
                stop_loss_sum += stop_price * weight
                
        stop_loss = stop_loss_sum / total_weight
        
        # Get the latest price from one of the signals
        current_price = signals[0]['price'] if signals else 0.0
        
        # Create main strategy name from contributing strategies
        main_strategy = contributing_strategies[0] if contributing_strategies else 'SignalAggregator'
        
        # NOUVEAU: Volume-based confidence boost (classique)
        confidence = self.signal_metrics.apply_volume_boost(confidence, signals)
        
        # Bonus multi-stratégies
        confidence = self.signal_metrics.apply_multi_strategy_bonus(confidence, contributing_strategies)
        
        # Déterminer la force du signal basée sur la confiance - CONFLUENCE CRYPTO (très strict)
        if confidence >= 0.90:  # CONFLUENCE: Très strict pour very_strong (90%+)
            strength = 'very_strong'
        elif confidence >= 0.80:  # CONFLUENCE: Strict pour strong (80%+)
            strength = 'strong'
        elif confidence >= 0.70:  # CONFLUENCE: Plus strict pour moderate (70%+)
            strength = 'moderate'
        else:
            strength = 'weak'
            
        # Trailing stop adaptatif : plus serré si stop ATR raisonnable
        if stop_price is not None:
            stop_distance_percent = abs(stop_price - current_price) / current_price * 100
            if stop_distance_percent <= 8:  # Stop ATR raisonnable (≤8%)
                trailing_delta = 2.0  # Trailing plus serré pour stops corrects
                logger.debug(f"🎯 Trailing serré: stop {stop_distance_percent:.1f}% -> trailing {trailing_delta:.1f}%")
            else:
                trailing_delta = 8.0  # Trailing large pour stops aberrants
                logger.warning(f"🚨 Trailing large: stop aberrant {stop_distance_percent:.1f}% -> trailing {trailing_delta:.1f}%")
        else:
            trailing_delta = 3.0  # Défaut si pas de stop calculé
        
        # NOUVEAU: Validation stricte minimum 2 stratégies pour confluence
        if len(contributing_strategies) < 2:
            if len(contributing_strategies) == 1:
                logger.info(f"❌ Signal unique rejeté (confluence requise): {contributing_strategies[0]} pour {symbol}")
                return None
            else:
                logger.info(f"❌ Signal rejeté: aucune stratégie valide pour {symbol}")
                return None
        
        return {
            'symbol': symbol,
            'side': side,  # Use 'side' instead of 'side' for coordinator compatibility
            'price': current_price,  # Add price field required by coordinator
            'strategy': f"Aggregated_{len(contributing_strategies)}",  # Create strategy name
            'confidence': confidence,
            'strength': strength,  # Ajouter la force du signal
            'stop_loss': stop_loss,
            'trailing_delta': trailing_delta,  # NOUVEAU: Trailing stop activé
            'contributing_strategies': contributing_strategies,
            'BUY_score': BUY_score,
            'SELL_score': SELL_score,
            'metadata': {
                'aggregated': True,
                'contributing_strategies': contributing_strategies,
                'strategy_count': len(contributing_strategies),
                'stop_price': stop_loss,
                'trailing_delta': trailing_delta  # NOUVEAU: Ajouter au metadata
            }
        }
    
    async def _aggregate_signals_enhanced(self, symbol: str, signals: List[Dict], 
                                        regime: Any, regime_metrics: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Version ultra-améliorée de l'agrégation avec analyse multi-timeframe complète
        """
        try:
            # ÉTAPE 1: Analyse multi-timeframe complète (avec fallbacks)
            confluence_analysis = None
            structure_analysis = None
            momentum_analysis = None
            adaptive_regime = None
            adaptive_metrics = None
            adaptive_thresholds = None
            
            if self.confluence_analyzer:
                try:
                    confluence_analysis = await self.confluence_analyzer.analyze_confluence(symbol)
                except Exception as e:
                    logger.warning(f"⚠️ Erreur analyse confluence pour {symbol}: {e}")
            
            if self.structure_detector:
                try:
                    structure_analysis = await self.structure_detector.analyze_market_structure(symbol)
                except Exception as e:
                    logger.warning(f"⚠️ Erreur analyse structure pour {symbol}: {e}")
            
            if self.momentum_analyzer:
                try:
                    momentum_analysis = await self.momentum_analyzer.analyze_momentum_cross_timeframe(symbol)
                except Exception as e:
                    logger.warning(f"⚠️ Erreur analyse momentum pour {symbol}: {e}")
            
            # ÉTAPE 2: Régime adaptatif amélioré (avec fallback)
            if self.adaptive_regime:
                try:
                    adaptive_regime, adaptive_metrics, adaptive_thresholds = await self.adaptive_regime.get_adaptive_regime(symbol)
                except Exception as e:
                    logger.warning(f"⚠️ Erreur régime adaptatif pour {symbol}: {e}")
            
            # ÉTAPE 3: Utiliser le meilleur régime (adaptatif vs standard)
            if adaptive_thresholds and adaptive_thresholds.confidence > 0.7:
                final_regime = adaptive_regime
                final_metrics = adaptive_metrics
                logger.info(f"🧠 Utilisation régime adaptatif pour {symbol}: {adaptive_regime.value} (conf={adaptive_thresholds.confidence:.2f})")
            else:
                final_regime = regime
                final_metrics = regime_metrics
                logger.info(f"📊 Utilisation régime standard pour {symbol}: {regime.value}")
            
            # ÉTAPE 4: Calculer le score global de qualité du signal
            global_quality_score = self._calculate_global_quality_score(
                confluence_analysis, structure_analysis, momentum_analysis, final_regime, adaptive_thresholds
            )
            
            # ÉTAPE 5: Filtrage selon la qualité globale
            if global_quality_score < 30:  # Seuil minimum
                logger.info(f"❌ Signal rejeté pour {symbol}: qualité globale trop faible ({global_quality_score:.1f})")
                return None
            
            # ÉTAPE 6: Obtenir les poids des stratégies pour ce régime
            regime_weights = self.enhanced_regime_detector.get_strategy_weights_for_regime(final_regime)
            
            # ÉTAPE 7: Ajuster les poids selon les analyses multi-timeframe
            confluence_weight_modifier = self._get_confluence_weight_modifier(confluence_analysis)
            structure_weight_modifier = self._get_structure_weight_modifier(structure_analysis)
            momentum_weight_modifier = self._get_momentum_weight_modifier(momentum_analysis)
            
            logger.info(f"🎯 Modificateurs de poids pour {symbol}: "
                       f"confluence={confluence_weight_modifier:.2f}, "
                       f"structure={structure_weight_modifier:.2f}, "
                       f"momentum={momentum_weight_modifier:.2f}")
        
        except Exception as e:
            logger.error(f"❌ Erreur dans analyse multi-timeframe pour {symbol}: {e}")
            # Fallback vers analyse standard
            final_regime = regime
            final_metrics = regime_metrics
            regime_weights = self.enhanced_regime_detector.get_strategy_weights_for_regime(final_regime)
            confluence_weight_modifier = 1.0
            structure_weight_modifier = 1.0
            momentum_weight_modifier = 1.0
            global_quality_score = 50  # Score neutre en cas d'erreur
            confluence_analysis = None
            structure_analysis = None
            momentum_analysis = None
            adaptive_regime = None
            adaptive_thresholds = None
        
        # Group signals by side
        BUY_signals = []
        SELL_signals = []

        for signal in signals:
            strategy = signal['strategy']
            
            # Get strategy weight based on performance
            performance_weight = await self.performance_tracker.get_strategy_weight(strategy)
            
            # Get regime-specific weight
            regime_weight = regime_weights.get(strategy, 1.0)
            
            # NOUVEAU: Pondération bayésienne des stratégies
            bayesian_weight = self.bayesian_weights.get_bayesian_weight(strategy)
            
            # Combined weight (performance * regime * bayesian * multi-timeframe modifiers)
            combined_weight = (performance_weight * regime_weight * bayesian_weight * 
                             confluence_weight_modifier * structure_weight_modifier * momentum_weight_modifier)
            
            # Apply adaptive confidence threshold based on regime
            confidence = signal.get('confidence', 0.5)
            confidence_threshold = self.min_confidence_threshold
            
            # Seuils adaptatifs pour certains régimes
            if hasattr(regime, 'name') and regime.name == 'RANGE_TIGHT':
                confidence_threshold = self.range_tight_confidence_threshold
                logger.debug(f"📊 Seuil RANGE_TIGHT adaptatif: {confidence_threshold} pour {strategy}")
            
            # NOUVEAU: Appliquer les seuils dynamiques
            dynamic_thresholds = self.dynamic_thresholds.get_current_thresholds()
            confidence_threshold = max(confidence_threshold, dynamic_thresholds['confidence_threshold'])
            
            if confidence < confidence_threshold:
                logger.debug(f"Signal {strategy} rejeté: confiance {confidence:.2f} < {confidence_threshold:.2f}")
                # Enregistrer le rejet dans les stats
                self.monitoring_stats.record_signal_decision(
                    strategy=strategy,
                    regime=regime.name if hasattr(regime, 'name') else str(regime),
                    symbol=symbol,
                    accepted=False,
                    confidence=confidence,
                    reason=f"confiance {confidence:.2f} < {confidence_threshold:.2f}"
                )
                continue

            # Get side (handle both 'side' and 'side' keys)
            side = signal.get('side', signal.get('side'))
            if side in ['BUY', 'BUY']:
                side = 'BUY'
            elif side in ['SELL', 'SELL']:
                side = 'SELL'

            # Enhanced weighted signal with quality boost for ultra-confluent signals
            quality_boost = 1.0
            signal_is_ultra_confluent = signal.get('metadata', {}).get('ultra_confluence', False)
            signal_score = signal.get('metadata', {}).get('total_score')
            
            if signal_is_ultra_confluent and signal_score:
                # Boost basé sur le score ultra-confluent
                if signal_score >= 90:
                    quality_boost = 1.5  # +50% de poids
                elif signal_score >= 80:
                    quality_boost = 1.3  # +30% de poids
                elif signal_score >= 70:
                    quality_boost = 1.2  # +20% de poids
                    
            # Appliquer le modificateur ADX si présent
            adx_modifier = signal.get('metadata', {}).get('adx_weight_modifier', 1.0)
            final_combined_weight = combined_weight * quality_boost * adx_modifier
            
            weighted_signal = {
                'strategy': strategy,
                'side': side,
                'confidence': confidence,
                'performance_weight': performance_weight,
                'regime_weight': regime_weight,
                'quality_boost': quality_boost,
                'combined_weight': final_combined_weight,
                'score': confidence * final_combined_weight,
                'signal_type': 'ultra_confluent' if signal_is_ultra_confluent else 'classic',
                'signal_score': signal_score
            }

            if side == 'BUY':
                BUY_signals.append(weighted_signal)
                # Enregistrer l'acceptation dans les stats
                self.monitoring_stats.record_signal_decision(
                    strategy=strategy,
                    regime=regime.name if hasattr(regime, 'name') else str(regime),
                    symbol=symbol,
                    accepted=True,
                    confidence=confidence
                )
            elif side == 'SELL':
                SELL_signals.append(weighted_signal)
                # Enregistrer l'acceptation dans les stats
                self.monitoring_stats.record_signal_decision(
                    strategy=strategy,
                    regime=regime.name if hasattr(regime, 'name') else str(regime),
                    symbol=symbol,
                    accepted=True,
                    confidence=confidence
                )

        # NOUVEAU: Vérifier la cohérence entre stratégies trend/reversal
        if not self._check_strategy_coherence(BUY_signals + SELL_signals, regime):
            logger.info(f"Signal rejeté pour {symbol}: incohérence entre stratégies trend/reversal")
            return None

        # Calculate total scores
        BUY_score = sum(s['score'] for s in BUY_signals)
        SELL_score = sum(s['score'] for s in SELL_signals)

        # Enhanced decision logic based on regime avec seuils dynamiques
        min_threshold = self.regime_filtering.get_regime_threshold(regime)
        
        # Adapter le seuil de vote pour RANGE_TIGHT
        if hasattr(regime, 'name') and regime.name == 'RANGE_TIGHT':
            min_threshold = self.range_tight_vote_threshold
            logger.debug(f"📊 Seuil de vote RANGE_TIGHT adaptatif: {min_threshold}")
        
        # NOUVEAU: Appliquer les seuils dynamiques
        dynamic_thresholds = self.dynamic_thresholds.get_current_thresholds()
        min_threshold = max(min_threshold, dynamic_thresholds['vote_threshold'])
        logger.debug(f"🎯 Seuil vote dynamique appliqué: {min_threshold}")
        
        # Determine side
        if BUY_score > SELL_score and BUY_score >= min_threshold:
            side = 'BUY'
            confidence = BUY_score / (BUY_score + SELL_score)
            contributing_strategies = list(set(s['strategy'] for s in BUY_signals))  # Dé-dupliquer
            relevant_signals = BUY_signals
        elif SELL_score > BUY_score and SELL_score >= min_threshold:
            side = 'SELL'
            confidence = SELL_score / (BUY_score + SELL_score)
            contributing_strategies = list(set(s['strategy'] for s in SELL_signals))  # Dé-dupliquer
            relevant_signals = SELL_signals
        else:
            # No clear signal
            logger.debug(f"No clear signal for {symbol} in {regime.value}: BUY={BUY_score:.2f}, SELL={SELL_score:.2f}")
            return None
            
        # Calculate averaged stop loss
        total_weight = sum(s['combined_weight'] for s in relevant_signals)
        if total_weight == 0:
            return None
            
        # NOUVEAU: Calcul de stop-loss adaptatif Enhanced avec ATR dynamique
        stop_loss_sum = 0
        atr_stop_loss = await self.technical_analysis.calculate_atr_based_stop_loss(symbol, signals[0]['price'], side)
        
        for signal in signals:
            signal_side = signal.get('side', signal.get('side'))
            if signal_side == side and signal['strategy'] in contributing_strategies:
                # Find the corresponding weighted signal
                weighted_sig = next((s for s in relevant_signals if s['strategy'] == signal['strategy']), None)
                if weighted_sig:
                    weight = weighted_sig['combined_weight']
                    
                    # Prioriser stop ATR adaptatif si disponible
                    if atr_stop_loss is not None:
                        stop_price = atr_stop_loss
                        logger.info(f"🎯 Stop ATR Enhanced utilisé pour {symbol}: {stop_price:.4f}")
                    else:
                        # Fallback: Extract stop_price from metadata
                        metadata = signal.get('metadata', {})
                        # Stop-loss correct selon le side: BUY stop en dessous, SELL stop au dessus - CRYPTO OPTIMIZED
                        default_stop = signal['price'] * (1.08 if side == 'SELL' else 0.92)  # 8% crypto stops (était 0.2%!)
                        stop_price = metadata.get('stop_price', signal.get('stop_loss', default_stop))
                        logger.debug(f"📊 Stop fixe Enhanced utilisé pour {symbol}: {stop_price:.4f}")
                    
                    stop_loss_sum += stop_price * weight
                
        stop_loss = stop_loss_sum / total_weight
        
        # Get the latest price from one of the signals
        current_price = signals[0]['price'] if signals else 0.0
        
        # Create main strategy name from contributing strategies
        main_strategy = contributing_strategies[0] if contributing_strategies else 'SignalAggregator'
        
        # Performance-based adaptive boost
        confidence = await self.signal_metrics.apply_performance_boost(confidence, contributing_strategies)
        
        # Regime-adaptive confidence boost
        confidence = self.regime_filtering.apply_regime_confidence_boost(confidence, regime, regime_metrics)
        
        # NOUVEAU: Volume-based confidence boost
        confidence = self.signal_metrics.apply_volume_boost(confidence, signals)
        
        # Bonus multi-stratégies
        confidence = self.signal_metrics.apply_multi_strategy_bonus(confidence, contributing_strategies)
        
        # SOFT-CAP sophistiqué avec tanh() pour préserver les nuances
        confidence = self.signal_metrics.calculate_soft_cap_confidence(confidence)
        
        # Déterminer la force du signal basée sur la confiance et le régime
        strength = self.regime_filtering.determine_signal_strength(confidence, regime)
        
        # VRAIE logique pour 'moderate' avec ≥2 stratégies
        # Assouplir la force si multiple strategies en régime strict
        if (strength == 'moderate' and len(contributing_strategies) >= 2 and 
            hasattr(regime, 'name') and regime.name in ['RANGE_TIGHT', 'RANGE_VOLATILE']):
            logger.info(f"✅ Force 'moderate' VRAIMENT acceptée: {len(contributing_strategies)} stratégies convergent "
                       f"en {regime.name} pour {symbol}")
            # Force sera validée comme acceptable plus tard
        
        # Trailing stop fixe à 8% pour système crypto pur
        trailing_delta = 8.0  # Crypto optimized (était 3.0%)
        
        # NOUVEAU: Validation stricte minimum 2 stratégies pour confluence
        if len(contributing_strategies) < 2:
            if len(contributing_strategies) == 1:
                logger.info(f"❌ Signal unique rejeté (confluence requise): {contributing_strategies[0]} pour {symbol}")
                return None
            else:
                logger.info(f"❌ Signal rejeté: aucune stratégie valide pour {symbol}")
                return None
        
        # VALIDATION FINALE: Override pour 'moderate' avec ≥2 stratégies
        final_strength = strength
        if (strength == 'moderate' and len(contributing_strategies) >= 2 and 
            hasattr(regime, 'name') and regime.name in ['RANGE_TIGHT', 'RANGE_VOLATILE']):
            # Force acceptée malgré les règles strictes du régime
            logger.info(f"🚀 Override 'moderate' appliqué: {len(contributing_strategies)} stratégies "
                       f"en {regime.name} pour {symbol}")
        
        # NOUVEAU: Validation finale avec seuils dynamiques
        vote_weight = max(BUY_score, SELL_score)
        if not self.dynamic_thresholds.should_accept_signal(confidence, vote_weight):
            logger.info(f"Signal {side} {symbol} rejeté par seuils dynamiques - confiance: {confidence:.3f}, vote: {vote_weight:.3f}")
            return None
        
        # NOUVEAU: Vérifier le debounce pour éviter les signaux groupés
        if not await self._check_signal_debounce(symbol, side):
            logger.info(f"Signal {side} {symbol} rejeté par filtre debounce")
            return None
        
        return {
            'symbol': symbol,
            'side': side,
            'price': current_price,
            'strategy': f"Aggregated_{len(contributing_strategies)}",
            'confidence': confidence,
            'strength': final_strength,
            'stop_loss': stop_loss,
            'trailing_delta': trailing_delta,
            'contributing_strategies': contributing_strategies,
            'BUY_score': BUY_score,
            'SELL_score': SELL_score,
            'regime_analysis': {
                'regime': regime.value,
                'metrics': regime_metrics,
                'applied_weights': {s['strategy']: s['regime_weight'] for s in relevant_signals}
            },
            'metadata': {
                'aggregated': True,
                'contributing_strategies': contributing_strategies,
                'strategy_count': len(contributing_strategies),
                'stop_price': stop_loss,
                'trailing_delta': trailing_delta,
                'regime_adaptive': True,
                'regime': regime.value,
                'volume_boosted': True,  # Indicateur que le volume a été pris en compte
                'volume_analysis': self.signal_metrics.extract_volume_summary(signals),
                'multi_timeframe_analysis': {
                    'global_quality_score': global_quality_score,
                    'confluence_analysis': {
                        'overall_signal': confluence_analysis.overall_signal if confluence_analysis else 0,
                        'confluence_score': confluence_analysis.confluence_score if confluence_analysis else 0,
                        'strength_rating': confluence_analysis.strength_rating if confluence_analysis else 'UNDEFINED',
                        'recommended_action': confluence_analysis.recommended_action if confluence_analysis else 'HOLD'
                    },
                    'structure_analysis': {
                        'structure_type': structure_analysis.structure_type.value if structure_analysis else 'UNDEFINED',
                        'structure_score': structure_analysis.structure_score if structure_analysis else 0,
                        'bias': structure_analysis.bias if structure_analysis else 'neutral',
                        'trend_strength': structure_analysis.trend_strength if structure_analysis else 0
                    },
                    'momentum_analysis': {
                        'overall_momentum': momentum_analysis.overall_momentum if momentum_analysis else 0,
                        'momentum_direction': momentum_analysis.momentum_direction.value if momentum_analysis else 'NEUTRAL',
                        'momentum_score': momentum_analysis.momentum_score if momentum_analysis else 0,
                        'entry_quality': momentum_analysis.entry_quality if momentum_analysis else 'POOR',
                        'momentum_alignment': momentum_analysis.momentum_alignment if momentum_analysis else 0
                    },
                    'adaptive_regime': {
                        'regime': adaptive_regime.value if 'adaptive_regime' in locals() else regime.value,
                        'confidence': adaptive_thresholds.confidence if 'adaptive_thresholds' in locals() else 0,
                        'used_adaptive': adaptive_thresholds.confidence > 0.7 if 'adaptive_thresholds' in locals() else False
                    },
                    'weight_modifiers': {
                        'confluence_modifier': confluence_weight_modifier,
                        'structure_modifier': structure_weight_modifier,
                        'momentum_modifier': momentum_weight_modifier
                    }
                }
            }
        }
    
    async def get_dynamic_cooldown(self, symbol: str) -> timedelta:
        """
        Calcule un cooldown adaptatif basé sur la volatilité (ATR).
        Plus la volatilité est faible, plus le cooldown est long.
        
        Args:
            symbol: Symbole de trading
            
        Returns:
            Période de cooldown adaptée
        """
        try:
            # Récupérer l'ATR depuis les données techniques
            atr_data = await self.technical_analysis.get_atr(symbol)
            atr_percent = atr_data.get('atr_percent', 1.0) if atr_data else 1.0
            
            # Cooldown inversement proportionnel à la volatilité
            base_minutes = self.cooldown_period.total_seconds() / 60
            
            if atr_percent < 0.3:  # Très faible volatilité
                cooldown_minutes = base_minutes * 3
                logger.debug(f"🐢 Cooldown étendu pour {symbol}: {cooldown_minutes:.0f}min (ATR={atr_percent:.2f}%)")
            elif atr_percent < 0.5:  # Faible volatilité
                cooldown_minutes = base_minutes * 2
                logger.debug(f"🐌 Cooldown augmenté pour {symbol}: {cooldown_minutes:.0f}min (ATR={atr_percent:.2f}%)")
            elif atr_percent > 2.0:  # Haute volatilité
                cooldown_minutes = base_minutes * 0.5
                logger.debug(f"🚀 Cooldown réduit pour {symbol}: {cooldown_minutes:.0f}min (ATR={atr_percent:.2f}%)")
            else:  # Volatilité normale
                cooldown_minutes = base_minutes
                
            return timedelta(minutes=cooldown_minutes)
            
        except Exception as e:
            logger.error(f"Erreur calcul cooldown dynamique pour {symbol}: {e}")
            return self.cooldown_period  # Fallback au cooldown standard

    def _check_strategy_coherence(self, signals: List[Dict], regime: str) -> bool:
        """
        Vérifie la cohérence entre stratégies trend-following et mean-reversion.
        Évite les conflits où des stratégies opposées donnent des signaux contradictoires.
        
        Args:
            signals: Liste des signaux pondérés
            regime: Régime de marché actuel
            
        Returns:
            True si les signaux sont cohérents, False sinon
        """
        if len(signals) < 2:
            return True  # Pas de conflit possible avec un seul signal
        
        # Classifier les stratégies
        trend_strategies = ['EMA_Cross', 'MACD', 'Breakout']
        reversal_strategies = ['RSI', 'Bollinger', 'Divergence']
        adaptive_strategies = ['Ride_or_React']  # S'adapte au contexte
        
        # Séparer les signaux par type
        trend_signals = [s for s in signals if s['strategy'] in trend_strategies]
        reversal_signals = [s for s in signals if s['strategy'] in reversal_strategies]
        
        # Si pas de mélange, c'est cohérent
        if not trend_signals or not reversal_signals:
            return True
        
        # Vérifier la direction des signaux
        trend_sides = set(s['side'] for s in trend_signals)
        reversal_sides = set(s['side'] for s in reversal_signals)
        
        # Si les directions sont opposées, vérifier le régime
        if trend_sides != reversal_sides:
            # En tendance forte, privilégier les stratégies de tendance
            if regime in ['STRONG_TREND_UP', 'STRONG_TREND_DOWN', 'TREND_UP', 'TREND_DOWN']:
                logger.debug(f"Conflit trend/reversal en régime {regime}: privilégier trend")
                # OK si les stratégies de tendance dominent
                return len(trend_signals) >= len(reversal_signals)
            # En range, privilégier les stratégies de retournement
            elif regime in ['RANGE_TIGHT', 'RANGE_VOLATILE']:
                logger.debug(f"Conflit trend/reversal en régime {regime}: privilégier reversal")
                # OK si les stratégies de retournement dominent
                return len(reversal_signals) >= len(trend_signals)
            else:
                # Régime mixte : exiger consensus plus fort
                total_trend_score = sum(s['score'] for s in trend_signals)
                total_reversal_score = sum(s['score'] for s in reversal_signals)
                # Le côté avec le score le plus élevé doit avoir 50% de plus
                if total_trend_score > total_reversal_score:
                    return total_trend_score > total_reversal_score * 1.5
                else:
                    return total_reversal_score > total_trend_score * 1.5
        
        return True  # Directions cohérentes

    async def _is_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown period with dynamic adjustment"""
        
        # Get dynamic cooldown period
        cooldown_period = await self.get_dynamic_cooldown(symbol)
        
        # Check local cooldown
        if symbol in self.last_signal_time:
            time_since_last = datetime.now(timezone.utc) - self.last_signal_time[symbol]
            if time_since_last < cooldown_period:
                return True
                
        # Check Redis for distributed cooldown
        cooldown_key = f"signal_cooldown:{symbol}"
        cooldown = self.redis.get(cooldown_key)
        
        return cooldown is not None
        
    async def set_cooldown(self, symbol: str, duration_seconds: int = 180):
        """Set cooldown for a symbol"""
        cooldown_key = f"signal_cooldown:{symbol}"
        self.redis.set(cooldown_key, "1", expiration=duration_seconds)
    
    async def _check_signal_debounce(self, symbol: str, side: str) -> bool:
        """
        Vérifie si un signal respecte le délai de debounce pour éviter les signaux groupés
        Délégué à EnhancedSignalAggregator pour éviter la duplication
        """
        # Cette méthode est maintenant dans EnhancedSignalAggregator
        # Retourner True par défaut pour SignalAggregator de base
        return True


class EnhancedSignalAggregator(SignalAggregator):
    """Version améliorée avec plus de filtres et validations"""
    
    def __init__(self, redis_client, regime_detector, performance_tracker, db_pool=None):
        super().__init__(redis_client, regime_detector, performance_tracker)
        self.db_pool = db_pool  # Stocker le db_pool pour les modules bayésiens
        
        # Vérifier si les modules améliorés sont disponibles
        if not EnhancedRegimeDetector:
            logger.warning("EnhancedSignalAggregator initialisé en mode dégradé (modules améliorés non disponibles)")
            self.enhanced_mode = False
        else:
            self.enhanced_mode = True
        
        # Nouveaux paramètres
        self.correlation_threshold = 0.7  # Corrélation minimale entre signaux
        self.divergence_penalty = 0.5  # Pénalité pour signaux divergents
        self.regime_transition_cooldown = timedelta(minutes=5)
        self.last_regime_change = {}
        
        # Suivi des faux signaux
        self.false_signal_tracker = defaultdict(int)
        self.false_signal_threshold = 3  # Max faux signaux avant désactivation temporaire
        
        # NOUVEAU: Debounce pour éviter les signaux groupés
        self.signal_debounce = defaultdict(lambda: {'last_buy': None, 'last_sell': None})
        self.debounce_periods = {
            'same_side': 3,  # Nombre de bougies minimum entre signaux du même côté (BUY-BUY ou SELL-SELL)
            'opposite_side': 1  # Nombre de bougies minimum entre signaux opposés (BUY-SELL)
        }
        self.candle_duration = timedelta(minutes=1)  # Durée d'une bougie (à adapter selon l'intervalle)
        
    def update_strategy_performance(self, strategy: str, is_win: bool, return_pct: float = 0.0):
        """
        Met à jour les performances bayésiennes d'une stratégie
        À appeler quand un trade se termine
        """
        try:
            self.bayesian_weights.update_performance(strategy, is_win, return_pct)
            logger.info(f"📈 Performance mise à jour pour {strategy}: {'WIN' if is_win else 'LOSS'} ({return_pct:+.2%})")
        except Exception as e:
            logger.error(f"Erreur mise à jour performance {strategy}: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Retourne un résumé des performances et seuils"""
        try:
            return {
                'bayesian_weights': self.bayesian_weights.get_performance_summary(),
                'dynamic_thresholds': self.dynamic_thresholds.get_statistics()
            }
        except Exception as e:
            logger.error(f"Erreur récupération résumé performances: {e}")
            return {}

    async def _check_signal_debounce(self, symbol: str, side: str, interval: str = None) -> bool:
        """
        Vérifie si un signal respecte le délai de debounce pour éviter les signaux groupés
        
        Args:
            symbol: Symbole du signal
            side: Côté du signal (BUY ou SELL)
            interval: Intervalle temporel (optionnel, détecté automatiquement si non fourni)
            
        Returns:
            True si le signal est autorisé, False s'il doit être filtré
        """
        try:
            current_time = datetime.now(timezone.utc)
            debounce_info = self.signal_debounce[symbol]
            
            # Déterminer la durée d'une bougie selon l'intervalle
            interval_map = {
                '1m': timedelta(minutes=1),
                '3m': timedelta(minutes=3),
                '5m': timedelta(minutes=5),
                '15m': timedelta(minutes=15),
                '30m': timedelta(minutes=30),
                '1h': timedelta(hours=1),
                '4h': timedelta(hours=4)
            }
            
            # Utiliser l'intervalle fourni ou détecter depuis Redis
            if not interval:
                # Essayer de récupérer l'intervalle depuis les données de marché
                market_key = f"market_interval:{symbol}"
                interval = self.redis.get(market_key) or '15m'  # Par défaut 15m pour swing trading
            
            candle_duration = interval_map.get(interval, timedelta(minutes=15))
            
            # Adapter les périodes de debounce selon l'intervalle
            # Pour 15m: 3 bougies = 45 min entre signaux même côté
            # Debounce adaptatif basé sur l'ADX (Hybrid approach)
            current_adx = await self.technical_analysis._get_current_adx(symbol)
            base_same = self.debounce_periods['same_side']
            base_opposite = self.debounce_periods['opposite_side']
            
            # Calculer multiplicateur ADX pour debounce adaptatif
            if current_adx is not None:
                if current_adx >= 42:  # Tendance très forte
                    adx_multiplier = 0.5  # Debounce réduit de moitié
                    trend_strength = "très forte"
                elif current_adx >= 32:  # Tendance forte
                    adx_multiplier = 0.7  # Debounce réduit
                    trend_strength = "forte" 
                elif current_adx >= 23:  # Tendance modérée
                    adx_multiplier = 1.0  # Debounce normal
                    trend_strength = "modérée"
                else:  # Range/tendance faible
                    adx_multiplier = 1.8  # Debounce augmenté
                    trend_strength = "faible/range"
            else:
                adx_multiplier = 1.0  # Fallback si ADX non disponible
                trend_strength = "inconnue"
            
            # Appliquer multiplicateur
            debounce_same = int(base_same * adx_multiplier)
            debounce_opposite = int(base_opposite * adx_multiplier)
            
            logger.debug(f"📊 Debounce adaptatif {symbol}: ADX={current_adx:.1f} (tendance {trend_strength}) → même={debounce_same}, opposé={debounce_opposite} bougies")
            
            # Déterminer le dernier signal du même côté et du côté opposé
            if side == 'BUY':
                last_same_side = debounce_info['last_buy']
                last_opposite_side = debounce_info['last_sell']
            else:  # SELL
                last_same_side = debounce_info['last_sell']
                last_opposite_side = debounce_info['last_buy']
            
            # Vérifier le debounce pour le même côté
            if last_same_side:
                time_since_same = (current_time - last_same_side).total_seconds()
                min_time_same = debounce_same * candle_duration.total_seconds()
                
                if time_since_same < min_time_same:
                    logger.info(f"❌ Signal {side} {symbol} filtré par debounce même côté: "
                              f"{time_since_same:.0f}s < {min_time_same:.0f}s requis ({debounce_same} bougies {interval})")
                    return False
            
            # Vérifier le debounce pour le côté opposé
            if last_opposite_side:
                time_since_opposite = (current_time - last_opposite_side).total_seconds()
                min_time_opposite = debounce_opposite * candle_duration.total_seconds()
                
                if time_since_opposite < min_time_opposite:
                    logger.info(f"⚠️ Signal {side} {symbol} filtré par debounce côté opposé: "
                              f"{time_since_opposite:.0f}s < {min_time_opposite:.0f}s requis ({debounce_opposite} bougies {interval})")
                    return False
            
            # Signal autorisé - mettre à jour le tracking
            if side == 'BUY':
                debounce_info['last_buy'] = current_time
            else:
                debounce_info['last_sell'] = current_time
            
            logger.debug(f"✅ Signal {side} {symbol} passe le filtre debounce (intervalle: {interval})")
            return True
            
        except Exception as e:
            logger.error(f"Erreur dans check_signal_debounce: {e}")
            return True  # En cas d'erreur, laisser passer le signal
    
    def _calculate_global_quality_score(self, confluence_analysis, structure_analysis, 
                                       momentum_analysis, regime, adaptive_thresholds) -> float:
        """Calcule le score global de qualité du signal"""
        try:
            quality_components = []
            
            # 1. Score de confluence (0-100)
            confluence_score = confluence_analysis.confluence_score if confluence_analysis else 50
            quality_components.append(confluence_score * 0.35)  # 35% du score
            
            # 2. Score de structure (0-100)  
            structure_score = structure_analysis.structure_score if structure_analysis else 50
            quality_components.append(structure_score * 0.25)  # 25% du score
            
            # 3. Score de momentum (0-100)
            momentum_score = momentum_analysis.momentum_score if momentum_analysis else 50
            quality_components.append(momentum_score * 0.25)  # 25% du score
            
            # 4. Score de régime adaptatif (0-100)
            if adaptive_thresholds:
                regime_score = adaptive_thresholds.confidence * 100
                quality_components.append(regime_score * 0.15)  # 15% du score
            else:
                quality_components.append(50 * 0.15)  # Score neutre
            
            # Score global final
            global_score = sum(quality_components)
            
            # Bonus pour alignement exceptionnel
            if (confluence_analysis and confluence_analysis.strength_rating == 'VERY_STRONG' and
                momentum_analysis and momentum_analysis.entry_quality == 'EXCELLENT' and
                structure_analysis and structure_analysis.structure_score > 80):
                global_score += 10  # Bonus d'alignement
            
            return min(100.0, max(0.0, global_score))
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul score qualité globale: {e}")
            return 50.0  # Score neutre en cas d'erreur
    
    def _get_confluence_weight_modifier(self, confluence_analysis) -> float:
        """Calcule le modificateur de poids basé sur l'analyse de confluence"""
        try:
            if not confluence_analysis:
                return 1.0
            
            # Modificateur basé sur la force de confluence
            if confluence_analysis.strength_rating == 'VERY_STRONG':
                return 1.5
            elif confluence_analysis.strength_rating == 'STRONG':
                return 1.3
            elif confluence_analysis.strength_rating == 'MODERATE':
                return 1.1
            elif confluence_analysis.strength_rating == 'WEAK':
                return 0.9
            else:  # CONFLICTED
                return 0.7
                
        except Exception as e:
            logger.error(f"❌ Erreur calcul modificateur confluence: {e}")
            return 1.0
    
    def _get_structure_weight_modifier(self, structure_analysis) -> float:
        """Calcule le modificateur de poids basé sur l'analyse de structure"""
        try:
            if not structure_analysis:
                return 1.0
            
            # Modificateur basé sur la force de structure et le biais
            structure_strength = structure_analysis.structure_score / 100.0
            
            # Bonus si structure et biais sont alignés
            if structure_analysis.bias in ['bullish', 'bearish']:
                return 1.0 + (structure_strength - 0.5) * 0.4  # Entre 0.8 et 1.2
            else:
                return 1.0 + (structure_strength - 0.5) * 0.2  # Entre 0.9 et 1.1
                
        except Exception as e:
            logger.error(f"❌ Erreur calcul modificateur structure: {e}")
            return 1.0
    
    def _get_momentum_weight_modifier(self, momentum_analysis) -> float:
        """Calcule le modificateur de poids basé sur l'analyse de momentum"""
        try:
            if not momentum_analysis:
                return 1.0
            
            # Modificateur basé sur la qualité d'entrée
            if momentum_analysis.entry_quality == 'EXCELLENT':
                return 1.4
            elif momentum_analysis.entry_quality == 'GOOD':
                return 1.2
            elif momentum_analysis.entry_quality == 'AVERAGE':
                return 1.0
            else:  # POOR
                return 0.8
                
        except Exception as e:
            logger.error(f"❌ Erreur calcul modificateur momentum: {e}")
            return 1.0