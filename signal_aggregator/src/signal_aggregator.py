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
            
            # Clean old signals (keep only last 140 seconds for confluence)
            cutoff_time = timestamp - timedelta(seconds=140)
            self.signal_buffer[symbol] = [
                s for s in self.signal_buffer[symbol]
                if self.signal_processor.get_signal_timestamp(s) > cutoff_time
            ]
            
            # Check if we have enough signals to make a decision - MODE CONFLUENCE
            buffer_size = len(self.signal_buffer[symbol])
            
            # Logique d'attente intelligente : attendre plus de signaux ou un délai
            if buffer_size < 2:
                # Si on a seulement 1 signal, attendre 140 secondes pour voir si d'autres arrivent
                first_signal_time = self.signal_processor.get_signal_timestamp(self.signal_buffer[symbol][0])
                time_since_first = timestamp - first_signal_time
                
                if time_since_first.total_seconds() < 140:
                    logger.debug(f"🕐 Signal unique pour {symbol}, attente {140 - time_since_first.total_seconds():.0f}s pour confluence")
                    return None  # Attendre plus de signaux
                else:
                    logger.info(f"⏰ Délai d'attente écoulé pour {symbol}, traitement du signal unique")
                    # Continuer avec le signal unique après délai
            else:
                logger.info(f"🎯 Confluence détectée pour {symbol}: {buffer_size} signaux, traitement immédiat")
                
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
            
            # NOUVEAU: Seuils adaptatifs selon le type de signal
            if signal_is_ultra_confluent and signal_score:
                # Signaux ultra-confluents : seuil plus strict
                min_threshold = 0.7
            else:
                # Signaux classiques : seuil standard
                min_threshold = self.min_confidence_threshold
                
            if confidence < min_threshold:
                logger.debug(f"Signal {strategy} filtré: confidence {confidence:.2f} < {min_threshold:.2f}")
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

        # Determine side
        if BUY_score > SELL_score and BUY_score >= self.min_vote_threshold:
            side = 'BUY'
            confidence = BUY_score / (BUY_score + SELL_score)
            contributing_strategies = [s['strategy'] for s in BUY_signals]
        elif SELL_score > BUY_score and SELL_score >= self.min_vote_threshold:
            side = 'SELL'
            confidence = SELL_score / (BUY_score + SELL_score)
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
        
        # NOUVEAU: Validation minimum 2 stratégies pour publier un signal (sauf cas particuliers)
        if len(contributing_strategies) < 2:
            # Permettre temporairement les signaux uniques pour débugger la confluence
            if len(contributing_strategies) == 1:
                logger.info(f"⚠️ Signal unique accepté temporairement pour debug: {contributing_strategies[0]} pour {symbol}")
                # Continuer le traitement
            else:
                logger.info(f"❌ Signal rejeté: minimum 1 stratégie requise, seulement {len(contributing_strategies)} trouvée(s) pour {symbol}")
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
        Version améliorée de l'agrégation avec poids adaptatifs selon le régime
        """
        # Obtenir les poids des stratégies pour ce régime
        regime_weights = self.enhanced_regime_detector.get_strategy_weights_for_regime(regime)
        
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
            
            # Combined weight (performance * regime * bayesian)
            combined_weight = performance_weight * regime_weight * bayesian_weight
            
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
        
        # NOUVEAU: Validation minimum 2 stratégies pour publier un signal (sauf cas particuliers)
        if len(contributing_strategies) < 2:
            # Permettre temporairement les signaux uniques pour débugger la confluence
            if len(contributing_strategies) == 1:
                logger.info(f"⚠️ Signal unique accepté temporairement pour debug: {contributing_strategies[0]} pour {symbol}")
                # Continuer le traitement
            else:
                logger.info(f"❌ Signal rejeté: minimum 1 stratégie requise, seulement {len(contributing_strategies)} trouvée(s) pour {symbol}")
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
                'volume_analysis': self.signal_metrics.extract_volume_summary(signals)
            }
        }
    
    async def _is_in_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown period"""
        
        # Check local cooldown
        if symbol in self.last_signal_time:
            time_since_last = datetime.now(timezone.utc) - self.last_signal_time[symbol]
            if time_since_last < self.cooldown_period:
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