#!/usr/bin/env python3
import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING, Union
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
import json
import numpy as np
import time

logger = logging.getLogger(__name__)

# Type alias pour le regime de marché
if TYPE_CHECKING:
    from enhanced_regime_detector import MarketRegime
    MarketRegimeType = Union[MarketRegime, Any]
else:
    MarketRegimeType = Any

from enhanced_regime_detector import EnhancedRegimeDetector, MarketRegime


class MarketDataAccumulator:
    """Accumule les données de marché pour construire un historique"""
    
    def __init__(self, max_history: int = 200):
        self.max_history = max_history
        self.data_history = defaultdict(lambda: deque(maxlen=max_history))
        self.last_update = defaultdict(float)
    
    def add_market_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """Ajoute des données de marché à l'historique"""
        try:
            timestamp = data.get('timestamp', time.time())
            
            # Éviter les doublons (même timestamp)
            if timestamp <= self.last_update[symbol]:
                return
                
            # Enrichir les données avec timestamp normalisé
            enriched_data = data.copy()
            enriched_data['timestamp'] = timestamp
            enriched_data['datetime'] = datetime.fromtimestamp(timestamp)
            
            # Ajouter à l'historique
            self.data_history[symbol].append(enriched_data)
            self.last_update[symbol] = timestamp
            
        except Exception as e:
            logger.error(f"Erreur ajout données historiques {symbol}: {e}")
    
    def get_history(self, symbol: str, limit: int = None) -> List[Dict[str, Any]]:
        """Récupère l'historique des données pour un symbole"""
        history = list(self.data_history[symbol])
        if limit and len(history) > limit:
            return history[-limit:]
        return history
    
    def get_history_count(self, symbol: str) -> int:
        """Retourne le nombre de points historiques disponibles"""
        return len(self.data_history[symbol])


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
        
    async def _update_market_data_history(self, symbol: str) -> None:
        """Met à jour l'historique des données de marché pour un symbole"""
        try:
            # Récupérer les données actuelles depuis Redis
            key = f"market_data:{symbol}:5m"
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
            
            # Note: Removed impossible condition bug
            
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
                    return await self._process_institutional_signal(signal)
                # Signaux excellents (85+) ont priorité mais validation allégée
                elif signal_score >= 85:
                    logger.info(f"✨ SIGNAL EXCELLENT priorité haute: {symbol} score={signal_score:.1f}")
                    return await self._process_excellent_signal(signal)
                # Signaux faibles (<50) sont rejetés immédiatement
                elif signal_score < 50:
                    logger.info(f"❌ Signal ultra-confluent rejeté (score faible): {symbol} score={signal_score:.1f}")
                    return None
            
            # NOUVEAU: Validation multi-timeframe avec 15m (SWING CRYPTO)
            # Validation 15m pour swing trading, filtrage plus strict
            if not await self._validate_signal_with_higher_timeframe(signal):
                logger.info(f"Signal {strategy} {signal['side']} sur {symbol} rejeté par validation 15m swing")
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
            
            # Clean old signals (keep only last 120 seconds for confluence)
            cutoff_time = timestamp - timedelta(seconds=120)
            self.signal_buffer[symbol] = [
                s for s in self.signal_buffer[symbol]
                if self._get_signal_timestamp(s) > cutoff_time
            ]
            
            # Check if we have enough signals to make a decision - MODE CONFLUENCE (1+ signaux temporaire)
            if len(self.signal_buffer[symbol]) < 1:
                return None  # Wait for more signals
                
            # Get market regime FIRST pour filtrage intelligent (enhanced if available, sinon fallback)
            if self.enhanced_regime_detector:
                # Utiliser la version async - le Signal Aggregator s'exécute déjà dans un contexte async
                regime, regime_metrics = await self.enhanced_regime_detector.get_detailed_regime(symbol)
                
                # NOUVEAU: Filtrage intelligent basé sur les régimes Enhanced
                signal_filtered = await self._apply_enhanced_regime_filtering(
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
            
    async def _process_institutional_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Traitement express pour signaux de qualité institutionnelle (95+ points)"""
        try:
            symbol = signal['symbol']
            metadata = signal.get('metadata', {})
            
            # Traitement express - validation minimale
            current_price = signal['price']
            confidence = min(signal.get('confidence', 0.9), 1.0)  # Cap à 1.0
            
            # Force basée sur le score
            score = metadata.get('total_score', 95)
            if score >= 98:
                strength = 'very_strong'
            else:
                strength = 'strong'
                
            # Utiliser les niveaux de prix calculés par ultra-confluence
            price_levels = metadata.get('price_levels', {})
            # Stop-loss correct selon le side: SELL stop au dessus, BUY stop en dessous
            side = signal.get('side', 'BUY')
            default_stop = current_price * (1.025 if side == 'SELL' else 0.975)
            stop_loss = price_levels.get('stop_loss', default_stop)  # Stop plus serré pour signaux premium
            
            # Métadonnées enrichies
            enhanced_metadata = {
                'aggregated': True,
                'institutional_grade': True,
                'ultra_confluence': True,
                'total_score': score,
                'quality': metadata.get('quality', 'institutional'),
                'confirmation_count': metadata.get('confirmation_count', 0),
                'express_processing': True,
                'timeframes_analyzed': metadata.get('timeframes_analyzed', []),
                'stop_price': stop_loss,
                'trailing_delta': 2.0,  # Trailing plus serré pour signaux premium
                'recommended_size_multiplier': 1.2  # Taille légèrement augmentée
            }
            
            # Log pour debug stop-loss
            logger.info(f"🎯 Signal institutionnel {side} {symbol}: entry={current_price:.4f}, stop={stop_loss:.4f}")
            
            result = {
                'symbol': symbol,
                'side': signal['side'],
                'price': current_price,
                'strategy': 'UltraConfluence_Institutional',
                'confidence': confidence,
                'strength': strength,
                'stop_loss': stop_loss,
                'trailing_delta': 2.0,
                'contributing_strategies': ['UltraConfluence'],
                'metadata': enhanced_metadata
            }
            
            logger.info(f"⭐ Signal INSTITUTIONNEL traité: {symbol} {signal['side']} @ {current_price:.4f} (score={score:.1f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement signal institutionnel: {e}")
            return None
            
    async def _process_excellent_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Traitement prioritaire pour signaux excellents (85+ points)"""
        try:
            symbol = signal['symbol']
            metadata = signal.get('metadata', {})
            
            # Validation légère mais présente
            if await self._is_in_cooldown(symbol):
                logger.debug(f"Signal excellent {symbol} en cooldown, ignoré")
                return None
                
            # Vérification ADX allégée pour signaux excellents
            adx_value = await self._get_current_adx(symbol)
            score = metadata.get('total_score', 85)
            
            if adx_value and adx_value < 20 and score < 90:  # Seuil ADX plus strict seulement pour scores < 90
                logger.info(f"Signal excellent rejeté: ADX trop faible ({adx_value:.1f}) pour score {score:.1f}")
                return None
                
            current_price = signal['price']
            confidence = signal.get('confidence', 0.85)
            
            # Ajuster la confiance basée sur le score
            confidence_boost = min((score - 85) / 15 * 0.1, 0.1)  # Max 10% boost
            confidence = min(confidence + confidence_boost, 1.0)
            
            # Force basée sur le score et la confluence
            confirmation_count = metadata.get('confirmation_count', 0)
            if score >= 90 and confirmation_count >= 15:
                strength = 'very_strong'
            elif score >= 85:
                strength = 'strong'
            else:
                strength = 'moderate'
                
            # Prix et stop loss optimisés
            price_levels = metadata.get('price_levels', {})
            # Stop-loss correct selon le side: SELL stop au dessus, BUY stop en dessous
            side = signal.get('side', 'BUY')
            default_stop = current_price * (1.02 if side == 'SELL' else 0.98)
            stop_loss = price_levels.get('stop_loss', default_stop)  # Stop modéré
            
            enhanced_metadata = {
                'aggregated': True,
                'excellent_grade': True,
                'ultra_confluence': True,
                'total_score': score,
                'quality': metadata.get('quality', 'excellent'),
                'confirmation_count': confirmation_count,
                'priority_processing': True,
                'timeframes_analyzed': metadata.get('timeframes_analyzed', []),
                'stop_price': stop_loss,
                'trailing_delta': 2.5,
                'recommended_size_multiplier': 1.1
            }
            
            # Log pour debug stop-loss
            logger.info(f"🎯 Signal excellent {side} {symbol}: entry={current_price:.4f}, stop={stop_loss:.4f}")
            
            result = {
                'symbol': symbol,
                'side': signal['side'],
                'price': current_price,
                'strategy': 'UltraConfluence_Excellent',
                'confidence': confidence,
                'strength': strength,
                'stop_loss': stop_loss,
                'trailing_delta': 2.5,
                'contributing_strategies': ['UltraConfluence'],
                'metadata': enhanced_metadata
            }
            
            # Définir cooldown court pour signaux excellents
            await self.set_cooldown(symbol, 60)  # 1 minute seulement
            
            logger.info(f"✨ Signal EXCELLENT traité: {symbol} {signal['side']} @ {current_price:.4f} (score={score:.1f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement signal excellent: {e}")
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

        # Group signals by side
        BUY_signals = []
        SELL_signals = []

        for signal in signals:
            strategy = signal['strategy']
            
            # Check if strategy is appropriate for current regime
            if not self._is_strategy_active(strategy, regime):
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
        atr_stop_loss = await self._calculate_atr_based_stop_loss(symbol, signals[0]['price'], side)
        
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
        confidence = self._apply_volume_boost(confidence, signals)
        
        # Bonus multi-stratégies
        confidence = self._apply_multi_strategy_bonus(confidence, contributing_strategies)
        
        # Déterminer la force du signal basée sur la confiance - CONFLUENCE CRYPTO (très strict)
        if confidence >= 0.90:  # CONFLUENCE: Très strict pour very_strong (90%+)
            strength = 'very_strong'
        elif confidence >= 0.80:  # CONFLUENCE: Strict pour strong (80%+)
            strength = 'strong'
        elif confidence >= 0.70:  # CONFLUENCE: Plus strict pour moderate (70%+)
            strength = 'moderate'
        else:
            strength = 'weak'
            
        # Trailing stop fixe à 3% pour système pur (TrailingStop gère le reste)
        trailing_delta = 3.0
        
        # NOUVEAU: Validation minimum 2 stratégies pour publier un signal
        if len(contributing_strategies) < 2:
            logger.info(f"❌ Signal rejeté: minimum 2 stratégies requises, seulement {len(contributing_strategies)} trouvée(s) pour {symbol}")
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
        min_threshold = self._get_regime_threshold(regime)
        
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
        atr_stop_loss = await self._calculate_atr_based_stop_loss(symbol, signals[0]['price'], side)
        
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
        confidence = await self._apply_performance_boost(confidence, contributing_strategies)
        
        # Regime-adaptive confidence boost
        confidence = self._apply_regime_confidence_boost(confidence, regime, regime_metrics)
        
        # NOUVEAU: Volume-based confidence boost
        confidence = self._apply_volume_boost(confidence, signals)
        
        # Bonus multi-stratégies
        confidence = self._apply_multi_strategy_bonus(confidence, contributing_strategies)
        
        # Déterminer la force du signal basée sur la confiance et le régime
        strength = self._determine_signal_strength(confidence, regime)
        
        # VRAIE logique pour 'moderate' avec ≥2 stratégies
        # Assouplir la force si multiple strategies en régime strict
        if (strength == 'moderate' and len(contributing_strategies) >= 2 and 
            hasattr(regime, 'name') and regime.name in ['RANGE_TIGHT', 'RANGE_VOLATILE']):
            logger.info(f"✅ Force 'moderate' VRAIMENT acceptée: {len(contributing_strategies)} stratégies convergent "
                       f"en {regime.name} pour {symbol}")
            # Force sera validée comme acceptable plus tard
        
        # Trailing stop fixe à 8% pour système crypto pur
        trailing_delta = 8.0  # Crypto optimized (était 3.0%)
        
        # NOUVEAU: Validation minimum 2 stratégies pour publier un signal
        if len(contributing_strategies) < 2:
            logger.info(f"❌ Signal rejeté: minimum 2 stratégies requises, seulement {len(contributing_strategies)} trouvée(s) pour {symbol}")
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
                'volume_analysis': self._extract_volume_summary(signals)
            }
        }
    
    def _get_regime_threshold(self, regime: Any) -> float:
        """Retourne le seuil de vote minimum selon le régime"""
        if MarketRegime is None:
            return self.min_vote_threshold
            
        thresholds = {
            MarketRegime.STRONG_TREND_UP: 0.6,
            MarketRegime.STRONG_TREND_DOWN: 0.6,
            MarketRegime.TREND_UP: 0.5,
            MarketRegime.TREND_DOWN: 0.5,
            MarketRegime.WEAK_TREND_UP: 0.4,
            MarketRegime.WEAK_TREND_DOWN: 0.4,
            MarketRegime.RANGE_TIGHT: 0.7,  # Plus strict en range serré
            MarketRegime.RANGE_VOLATILE: 0.6,
            MarketRegime.UNDEFINED: 0.8  # Très prudent si indéfini
        }
        return thresholds.get(regime, self.min_vote_threshold)
    
    def _get_single_strategy_threshold(self, regime: Any) -> float:
        """Retourne le seuil de confiance pour les signaux d'une seule stratégie selon le régime"""
        if MarketRegime is None:
            return 0.8
            
        thresholds = {
            MarketRegime.STRONG_TREND_UP: 0.7,
            MarketRegime.STRONG_TREND_DOWN: 0.7,
            MarketRegime.TREND_UP: 0.75,
            MarketRegime.TREND_DOWN: 0.75,
            MarketRegime.WEAK_TREND_UP: 0.8,
            MarketRegime.WEAK_TREND_DOWN: 0.8,
            MarketRegime.RANGE_TIGHT: 0.85,  # Très strict en range serré
            MarketRegime.RANGE_VOLATILE: 0.8,
            MarketRegime.UNDEFINED: 0.9  # Très prudent si indéfini
        }
        return thresholds.get(regime, 0.8)
    
    def _apply_regime_confidence_boost(self, confidence: float, regime: Any, metrics: Dict[str, float]) -> float:
        """Applique un boost de confiance basé sur les métriques du régime"""
        # Boost basé sur la force de la tendance (ADX)
        adx = metrics.get('adx', 20)
        if adx > 40:  # Tendance très forte
            confidence *= 1.1
        elif adx > 30:  # Tendance forte
            confidence *= 1.05
        
        # Boost basé sur le momentum (ROC)
        roc = abs(metrics.get('roc', 0))
        if roc > 5:  # Momentum fort
            confidence *= 1.05
        
        # Penalty pour les régimes indéfinis ou instables
        if MarketRegime is not None:
            if regime == MarketRegime.UNDEFINED:
                confidence *= 0.9
            elif regime in [MarketRegime.RANGE_VOLATILE]:
                # Ne pas pénaliser les stratégies de mean-reversion en range
                # Note: cette logique est maintenant dans _apply_enhanced_regime_filtering
                confidence *= 0.95
        
        return min(1.0, confidence)  # Cap à 1.0
    
    async def _get_technical_context(self, symbol: str) -> Dict[str, Any]:
        """
        Récupère le contexte technique enrichi pour un symbole
        
        Returns:
            Dictionnaire avec les indicateurs techniques actuels
        """
        try:
            # CORRECTION: Import direct avec chemin complet au lieu de manipulation sys.path
            from shared.src.technical_indicators import TechnicalIndicators
            indicators = TechnicalIndicators()
            
            # Récupérer les données 5m depuis Redis
            market_data_key = f"market_data:{symbol}:5m"
            data_5m = self.redis.get(market_data_key)
            
            context = {
                'macd': None,
                'obv': None, 
                'roc': None,
                'available': False
            }
            
            if not data_5m or not isinstance(data_5m, dict):
                return context
                
            # Extraire les prix historiques
            prices = data_5m.get('prices', [])
            volumes = data_5m.get('volumes', [])
            highs = data_5m.get('highs', [])
            lows = data_5m.get('lows', [])
            
            if len(prices) < 30:  # Minimum pour les calculs
                return context
            
            # Calculer MACD
            macd_data = indicators.calculate_macd(prices)
            if macd_data['macd_line'] is not None:
                context['macd'] = macd_data
            
            # Calculer OBV approximatif (si volumes disponibles)
            if len(volumes) >= len(prices):
                try:
                    obv_value = indicators.calculate_obv(prices, volumes)
                    if obv_value is not None:
                        context['obv'] = obv_value
                except:
                    pass  # OBV optionnel
            
            # Calculer ROC
            try:
                roc_value = indicators.calculate_roc(prices, period=10)
                if roc_value is not None:
                    context['roc'] = roc_value
            except:
                pass  # ROC optionnel
            
            context['available'] = True
            return context
            
        except Exception as e:
            logger.error(f"Erreur récupération contexte technique pour {symbol}: {e}")
            return {'macd': None, 'obv': None, 'roc': None, 'available': False}
    
    def _validate_macd_trend(self, technical_context: Dict[str, Any], expected_trend: str) -> Optional[bool]:
        """
        Valide si le MACD confirme la tendance attendue
        
        Args:
            technical_context: Contexte technique
            expected_trend: 'bullish' ou 'bearish'
            
        Returns:
            True/False si MACD confirme, None si pas de données
        """
        try:
            macd_data = technical_context.get('macd')
            if not macd_data or not macd_data.get('macd_line'):
                return None
                
            macd_line = macd_data['macd_line']
            macd_signal = macd_data.get('macd_signal')
            macd_histogram = macd_data.get('macd_histogram')
            
            if macd_signal is None:
                return None
            
            if expected_trend == 'bullish':
                # Tendance haussière: MACD au-dessus signal ET histogram positif
                return macd_line > macd_signal and (macd_histogram is None or macd_histogram > 0)
            elif expected_trend == 'bearish':
                # Tendance baissière: MACD en-dessous signal ET histogram négatif
                return macd_line < macd_signal and (macd_histogram is None or macd_histogram < 0)
                
            return None
            
        except Exception as e:
            logger.error(f"Erreur validation MACD: {e}")
            return None
    
    def _validate_obv_trend(self, technical_context: Dict[str, Any], side: str) -> Optional[bool]:
        """
        Valide si l'OBV confirme le côté du signal avec analyse de tendance historique
        
        Args:
            technical_context: Contexte technique
            side: 'BUY' ou 'SELL'
            
        Returns:
            True si OBV confirme, False sinon, None si pas de données
        """
        try:
            obv_current = technical_context.get('obv')
            if obv_current is None:
                return None
            
            # Essayer de récupérer l'historique OBV récent
            symbol = technical_context.get('symbol', 'UNKNOWN')
            
            try:
                # Récupérer les dernières données depuis Redis pour calculer la tendance OBV
                import json
                recent_key = f"market_data:{symbol}:recent"
                recent_data = self.redis_client.lrange(recent_key, 0, 9)  # 10 dernières valeurs
                
                if len(recent_data) >= 3:  # Besoin d'au moins 3 points pour une tendance
                    obv_history = []
                    
                    for data_str in recent_data:
                        try:
                            data = json.loads(data_str)
                            if 'obv' in data and data['obv'] is not None:
                                obv_history.append(float(data['obv']))
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue
                    
                    # Ajouter l'OBV actuel
                    obv_history.append(float(obv_current))
                    
                    if len(obv_history) >= 3:
                        # Calculer la tendance OBV sur les derniers points
                        recent_slope = self._calculate_obv_slope(obv_history[-3:])
                        overall_slope = self._calculate_obv_slope(obv_history)
                        
                        # Validation basée sur la tendance OBV
                        if side == 'BUY':
                            # Pour BUY: OBV doit être en tendance haussière (volume d'achat croissant)
                            obv_confirms = recent_slope > 0 or (overall_slope > 0 and recent_slope >= -0.1)
                            logger.debug(f"📈 OBV validation BUY {symbol}: recent_slope={recent_slope:.2f}, overall_slope={overall_slope:.2f} → {obv_confirms}")
                            return obv_confirms
                        else:  # SELL
                            # Pour SELL: OBV doit être en tendance baissière (volume de vente croissant) 
                            obv_confirms = recent_slope < 0 or (overall_slope < 0 and recent_slope <= 0.1)
                            logger.debug(f"📉 OBV validation SELL {symbol}: recent_slope={recent_slope:.2f}, overall_slope={overall_slope:.2f} → {obv_confirms}")
                            return obv_confirms
                            
            except Exception as redis_error:
                logger.warning(f"Impossible de récupérer l'historique OBV pour {symbol}: {redis_error}")
            
            # Fallback: validation simplifiée si pas d'historique disponible
            # Comparer l'OBV actuel avec une valeur de référence basique
            volume_ratio = technical_context.get('volume_ratio', 1.0)
            
            if side == 'BUY':
                # BUY: favorable si volume ratio élevé (indique plus d'activité d'achat)
                return volume_ratio >= 1.0
            else:  # SELL  
                # SELL: on accepte même avec volume normal (vente peut se faire avec moins de volume)
                return volume_ratio >= 0.7
            
        except Exception as e:
            logger.error(f"Erreur validation OBV: {e}")
            return None
    
    def _calculate_obv_slope(self, obv_values: List[float]) -> float:
        """
        Calcule la pente de la tendance OBV
        
        Args:
            obv_values: Liste des valeurs OBV chronologiques
            
        Returns:
            Pente (positive = tendance haussière, négative = tendance baissière)
        """
        try:
            if len(obv_values) < 2:
                return 0.0
                
            # Calcul simple de la pente moyenne
            n = len(obv_values)
            x_values = list(range(n))
            
            # Régression linéaire simple: slope = Σ((x-x̄)(y-ȳ)) / Σ((x-x̄)²)
            x_mean = sum(x_values) / n
            y_mean = sum(obv_values) / n
            
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, obv_values))
            denominator = sum((x - x_mean) ** 2 for x in x_values)
            
            if denominator == 0:
                return 0.0
                
            slope = numerator / denominator
            return slope
            
        except Exception as e:
            logger.error(f"Erreur calcul pente OBV: {e}")
            return 0.0
    
    def _check_roc_acceleration(self, technical_context: Dict[str, Any], side: str) -> bool:
        """
        Vérifie si le ROC indique une accélération dans la direction du signal
        
        Args:
            technical_context: Contexte technique
            side: 'BUY' ou 'SELL'
            
        Returns:
            True si accélération détectée, False sinon
        """
        try:
            roc_value = technical_context.get('roc')
            if roc_value is None:
                return False
            
            # ROC positif = accélération haussière, ROC négatif = accélération baissière
            if side == 'BUY':
                # Pour BUY: chercher accélération haussière (ROC > 2%)
                return roc_value > 2.0
            elif side == 'SELL':
                # Pour SELL: chercher accélération baissière (ROC < -2%)
                return roc_value < -2.0
                
            return False
            
        except Exception as e:
            logger.error(f"Erreur vérification ROC: {e}")
            return False
    
    async def _calculate_atr_based_stop_loss(self, symbol: str, entry_price: float, side: str) -> Optional[float]:
        """
        Calcule un stop-loss adaptatif basé sur l'ATR pour optimiser selon la volatilité
        
        Args:
            symbol: Symbole du trading
            entry_price: Prix d'entrée
            side: 'BUY' ou 'SELL'
            
        Returns:
            Prix de stop-loss adaptatif ou None si impossible
        """
        try:
            # CORRECTION: Import direct avec chemin complet au lieu de manipulation sys.path
            from shared.src.technical_indicators import TechnicalIndicators
            indicators = TechnicalIndicators()
            
            # Récupérer les données OHLC depuis Redis
            market_data_key = f"market_data:{symbol}:5m"
            data_5m = self.redis.get(market_data_key)
            
            if not data_5m or not isinstance(data_5m, dict):
                logger.debug(f"Données 5m non disponibles pour ATR {symbol}")
                return None
                
            # Extraire les données OHLC
            highs = data_5m.get('highs', [])
            lows = data_5m.get('lows', [])
            closes = data_5m.get('closes', data_5m.get('prices', []))
            
            if len(highs) < 14 or len(lows) < 14 or len(closes) < 14:
                logger.debug(f"Pas assez de données OHLC pour ATR {symbol}")
                return None
            
            # Calculer ATR(14)
            atr_value = indicators.calculate_atr(highs, lows, closes, period=14)
            if atr_value is None:
                logger.debug(f"Calcul ATR échoué pour {symbol}")
                return None
            
            # Récupérer ADX pour adapter le multiplicateur
            adx_value = await self._get_current_adx(symbol)
            
            # Multiplicateur ATR adaptatif selon l'ADX (force de tendance)
            if adx_value is not None:
                if adx_value > 40:  # Tendance très forte
                    atr_multiplier = 2.0  # Stop plus serré en tendance forte
                    logger.debug(f"ADX forte ({adx_value:.1f}): multiplicateur ATR 2.0x")
                elif adx_value > 25:  # Tendance modérée
                    atr_multiplier = 2.5  # Standard
                    logger.debug(f"ADX modérée ({adx_value:.1f}): multiplicateur ATR 2.5x")
                else:  # Tendance faible ou range
                    atr_multiplier = 3.0  # Stop plus large en range
                    logger.debug(f"ADX faible ({adx_value:.1f}): multiplicateur ATR 3.0x")
            else:
                atr_multiplier = 2.5  # Par défaut
                logger.debug(f"ADX non disponible: multiplicateur ATR par défaut 2.5x")
            
            # Calculer le stop-loss selon le côté
            atr_distance = atr_value * atr_multiplier
            
            if side == 'BUY':
                # BUY: stop en dessous du prix d'entrée
                stop_loss = entry_price - atr_distance
            else:  # SELL
                # SELL: stop au-dessus du prix d'entrée
                stop_loss = entry_price + atr_distance
            
            # Validation: s'assurer que le stop n'est pas trop proche (minimum 0.5%)
            min_distance_percent = 0.005  # 0.5%
            min_distance = entry_price * min_distance_percent
            
            if side == 'BUY':
                if entry_price - stop_loss < min_distance:
                    stop_loss = entry_price - min_distance
                    logger.debug(f"Stop BUY ajusté au minimum 0.5%: {stop_loss:.4f}")
            else:  # SELL
                if stop_loss - entry_price < min_distance:
                    stop_loss = entry_price + min_distance
                    logger.debug(f"Stop SELL ajusté au minimum 0.5%: {stop_loss:.4f}")
            
            # Validation: s'assurer que le stop n'est pas trop loin (maximum 10%)
            max_distance_percent = 0.10  # 10%
            max_distance = entry_price * max_distance_percent
            
            if side == 'BUY':
                if entry_price - stop_loss > max_distance:
                    stop_loss = entry_price - max_distance
                    logger.debug(f"Stop BUY plafonné à 10%: {stop_loss:.4f}")
            else:  # SELL
                if stop_loss - entry_price > max_distance:
                    stop_loss = entry_price + max_distance
                    logger.debug(f"Stop SELL plafonné à 10%: {stop_loss:.4f}")
            
            distance_percent = abs(stop_loss - entry_price) / entry_price * 100
            logger.info(f"🎯 Stop ATR calculé pour {symbol} {side}: {stop_loss:.4f} "
                       f"(distance: {distance_percent:.2f}%, ATR: {atr_value:.4f}, mult: {atr_multiplier}x)")
            
            return round(stop_loss, 6)
            
        except Exception as e:
            logger.error(f"Erreur calcul stop ATR pour {symbol}: {e}")
            return None
    
    async def _apply_performance_boost(self, confidence: float, contributing_strategies: List[str]) -> float:
        """Applique un boost adaptatif basé sur la performance des stratégies"""
        if not hasattr(self, 'performance_tracker') or not self.performance_tracker:
            return confidence
        
        try:
            boost_factor = 1.0
            
            for strategy in contributing_strategies:
                # Obtenir le poids de performance (1.0 = neutre, >1.0 = surperformance)
                performance_weight = await self.performance_tracker.get_strategy_weight(strategy)
                
                if performance_weight > 1.1:  # Plus de 10% au-dessus du benchmark
                    # Boost progressif selon la surperformance
                    individual_boost = 1.0 + (performance_weight - 1.0) * 0.2  # Max +20% pour 2x performance
                    boost_factor *= individual_boost
                    logger.debug(f"🚀 Boost performance pour {strategy}: {performance_weight:.2f} -> boost {individual_boost:.2f}")
                
                elif performance_weight < 0.9:  # Plus de 10% en-dessous du benchmark
                    # Malus modéré pour sous-performance
                    individual_malus = max(0.95, 1.0 - (1.0 - performance_weight) * 0.1)  # Max -5%
                    boost_factor *= individual_malus
                    logger.debug(f"📉 Malus performance pour {strategy}: {performance_weight:.2f} -> malus {individual_malus:.2f}")
            
            # Limiter le boost total
            boost_factor = min(1.15, max(0.95, boost_factor))  # Entre -5% et +15%
            
            if boost_factor != 1.0:
                logger.info(f"📊 Boost performance global: {boost_factor:.2f} pour {contributing_strategies}")
            
            return confidence * boost_factor
            
        except Exception as e:
            logger.error(f"Erreur dans boost performance: {e}")
            return confidence
    
    def _determine_signal_strength(self, confidence: float, regime: Any) -> str:
        """Détermine la force du signal basée sur la confiance et le régime"""
        # Seuils standardisés alignés avec analyzer
        # moderate ≥ 0.55, strong ≥ 0.75, very_strong ≥ 0.9
        if confidence >= 0.9:
            return 'very_strong'
        elif confidence >= 0.75:
            return 'strong'
        elif confidence >= 0.55:
            return 'moderate'
        else:
            return 'weak'
    
    def _strength_to_normalized_force(self, strength: str) -> float:
        """Convertit la force textuelle en valeur normalisée 0-1"""
        strength_mapping = {
            'weak': 0.25,
            'moderate': 0.5,
            'strong': 0.75,
            'very_strong': 1.0
        }
        return strength_mapping.get(strength, 0.5)
        
    def _is_strategy_active(self, strategy: str, regime: str) -> bool:
        """Check if a strategy should be active in current regime"""
        
        # Adaptive strategies are always active
        if strategy in self.STRATEGY_GROUPS['adaptive']:
            return True
            
        # Handle enhanced regime codes
        if regime.startswith('STRONG_TREND') or regime.startswith('TREND'):
            return strategy in self.STRATEGY_GROUPS['trend']
            
        # Handle range regimes (RANGE_TIGHT, RANGE_VOLATILE, etc.)
        elif regime.startswith('RANGE'):
            return strategy in self.STRATEGY_GROUPS['mean_reversion']
            
        # Handle other enhanced regimes
        elif regime in ['BREAKOUT_UP', 'BREAKOUT_DOWN']:
            # Breakout regimes favor trend strategies
            return strategy in self.STRATEGY_GROUPS['trend']
        elif regime == 'VOLATILE':
            # In volatile markets, adaptive strategies work best
            return strategy in self.STRATEGY_GROUPS['adaptive']
            
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
        cooldown = self.redis.get(cooldown_key)
        
        return cooldown is not None
        
    async def set_cooldown(self, symbol: str, duration_seconds: int = 180):
        """Set cooldown for a symbol"""
        cooldown_key = f"signal_cooldown:{symbol}"
        self.redis.set(cooldown_key, "1", expiration=duration_seconds)
    
    async def _validate_signal_with_higher_timeframe(self, signal: Dict[str, Any]) -> bool:
        """
        Valide un signal 1m avec le contexte 15m enrichi pour éviter les faux signaux.
        
        Logique de validation enrichie :
        - Signal BUY : validé si tendance 15m haussière + BB position favorable + Stochastic non surachat
        - Signal SELL : validé si tendance 15m baissière + BB position favorable + Stochastic non survente
        - Utilise ATR pour adapter dynamiquement les seuils

        Args:
            signal: Signal 1m à valider
            
        Returns:
            True si le signal est validé, False sinon
        """
        try:
            symbol = signal['symbol']
            side = signal['side']

            # Récupérer les données 15m récentes depuis Redis (MODE SWING CRYPTO)
            market_data_key = f"market_data:{symbol}:15m"
            data_5m = self.redis.get(market_data_key)
            
            if not data_5m:
                # Si pas de données 15m, on accepte le signal (mode dégradé)
                logger.warning(f"Pas de données 15m pour {symbol}, validation swing en mode dégradé")
                return True
            
            # Le RedisClient parse automatiquement les données JSON
            if not isinstance(data_5m, dict):
                logger.warning(f"Données 5m invalides pour {symbol}: type {type(data_5m)}")
                return True
            
            # CORRECTION: Vérifier la fraîcheur des données 5m
            last_update = data_5m.get('last_update')
            if last_update:
                from datetime import datetime, timezone
                try:
                    if isinstance(last_update, str):
                        update_time = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                    else:
                        update_time = datetime.fromtimestamp(last_update, tz=timezone.utc)
                    
                    age_seconds = (datetime.now(timezone.utc) - update_time).total_seconds()
                    if age_seconds > 120:  # Plus de 2 minutes = données stales
                        logger.warning(f"Données 5m trop anciennes pour {symbol} ({age_seconds:.0f}s), bypass validation")
                        return True
                except Exception as e:
                    logger.warning(f"Erreur parsing timestamp 5m pour {symbol}: {e}")
                    return True
            
            # Calculer la tendance 5m avec une EMA simple (MODE SCALPING)
            prices = data_5m.get('prices', [])
            if len(prices) < 5:
                # Pas assez de données pour une tendance fiable (seuil réduit pour scalping)
                return True
            
            # HARMONISATION: EMA 21 vs EMA 50 pour cohérence avec les stratégies
            if len(prices) < 50:
                return True  # Pas assez de données pour EMA 50
            
            # AMÉLIORATION : Utiliser EMAs incrémentales pour courbes lisses
            current_price = prices[-1] if prices else 0
            ema_21 = self._get_or_calculate_ema_incremental(symbol, current_price, 21)
            ema_50 = self._get_or_calculate_ema_incremental(symbol, current_price, 50)
            
            # LOGIQUE SOPHISTIQUÉE : Analyser la force et le momentum de la tendance
            
            # Calculer la vélocité des EMAs (momentum) - méthode améliorée
            # Récupérer les EMAs précédentes depuis le cache pour vélocité
            timeframe = "1m"
            cache = self.ema_incremental_cache.get(symbol, {}).get(timeframe, {})
            ema_21_prev = cache.get('ema_21_prev', ema_21)
            ema_50_prev = cache.get('ema_50_prev', ema_50)
            
            # Calculer vélocité avec EMAs lisses
            ema_21_velocity = (ema_21 - ema_21_prev) / ema_21_prev if ema_21_prev > 0 else 0
            ema_50_velocity = (ema_50 - ema_50_prev) / ema_50_prev if ema_50_prev > 0 else 0
            
            # Stocker les valeurs actuelles comme "précédentes" pour le prochain calcul
            cache['ema_21_prev'] = ema_21
            cache['ema_50_prev'] = ema_50
            
            # Calculer la force de la tendance
            trend_strength = abs(ema_21 - ema_50) / ema_50 if ema_50 > 0 else 0
            
            # Classification sophistiquée de la tendance
            if ema_21 > ema_50 * 1.015:  # +1.5% = forte haussière
                trend_5m = "STRONG_BULLISH"
            elif ema_21 > ema_50 * 1.005:  # +0.5% = faible haussière
                trend_5m = "WEAK_BULLISH"
            elif ema_21 < ema_50 * 0.985:  # -1.5% = forte baissière  
                trend_5m = "STRONG_BEARISH"
            elif ema_21 < ema_50 * 0.995:  # -0.5% = faible baissière
                trend_5m = "WEAK_BEARISH"
            else:
                trend_5m = "NEUTRAL"
            
            # Détecter si la tendance s'affaiblit (divergence)
            trend_weakening = False
            if trend_5m in ["STRONG_BULLISH", "WEAK_BULLISH"] and ema_21_velocity < 0:
                trend_weakening = True  # Tendance haussière qui ralentit
            elif trend_5m in ["STRONG_BEARISH", "WEAK_BEARISH"] and ema_21_velocity > 0:
                trend_weakening = True  # Tendance baissière qui ralentit
            
            # DEBUG: Log détaillé pour comprendre les rejets
            logger.info(f"🔍 {symbol} | Prix={current_price:.4f} | EMA21={ema_21:.4f} | EMA50={ema_50:.4f} | Tendance={trend_5m} | Signal={side} | Velocity21={ema_21_velocity*100:.2f}% | Weakening={trend_weakening}")
            
            # LOGIQUE SOPHISTIQUÉE DE VALIDATION
            rejection_reason = None
            
            if side == "BUY":
                # Éviter d'acheter dans une forte montée (risque de sommet)
                if trend_5m == "STRONG_BULLISH" and not trend_weakening:
                    rejection_reason = "forte tendance haussière en cours, risque de sommet"
                # Éviter d'acheter un crash violent (couteau qui tombe)
                elif trend_5m == "STRONG_BEARISH" and ema_21_velocity < -0.01:  # Accélération baissière > 1%
                    rejection_reason = "crash violent en cours, éviter le couteau qui tombe"
                    
            elif side == "SELL":
                # Éviter de vendre dans une forte baisse (risque de creux)  
                if trend_5m == "STRONG_BEARISH" and not trend_weakening:
                    rejection_reason = "forte tendance baissière en cours, risque de creux"
                # Éviter de vendre une pump violente (FOMO manqué)
                elif trend_5m == "STRONG_BULLISH" and ema_21_velocity > 0.01:  # Accélération haussière > 1%
                    rejection_reason = "pump violent en cours, éviter de rater la montée"
            
            # Appliquer le rejet si raison trouvée
            if rejection_reason:
                logger.info(f"Signal {side} {symbol} rejeté : {rejection_reason}")
                return False
            
            # NOUVELLE VALIDATION ENRICHIE : Bollinger Bands position pour timing optimal
            bb_position = data_5m.get('bb_position')
            if bb_position is not None:
                if side == "BUY" and bb_position > 0.8:  # Prix proche de la bande haute
                    logger.info(f"Signal BUY {symbol} rejeté : BB position trop haute ({bb_position:.2f})")
                    return False
                elif side == "SELL" and bb_position < 0.2:  # Prix proche de la bande basse
                    logger.info(f"Signal SELL {symbol} rejeté : BB position trop basse ({bb_position:.2f})")
                    return False
            
            # NOUVELLE VALIDATION : Stochastic pour confirmer oversold/overbought
            stoch_k = data_5m.get('stoch_k')
            stoch_d = data_5m.get('stoch_d')
            if stoch_k is not None and stoch_d is not None:
                if side == "BUY" and stoch_k > 85 and stoch_d > 85:  # Surachat confirmé
                    logger.info(f"Signal BUY {symbol} rejeté : Stochastic surachat K={stoch_k:.1f} D={stoch_d:.1f}")
                    return False
                elif side == "SELL" and stoch_k < 15 and stoch_d < 15:  # Survente confirmé
                    logger.info(f"Signal SELL {symbol} rejeté : Stochastic survente K={stoch_k:.1f} D={stoch_d:.1f}")
                    return False
            
            # VALIDATION ADAPTATIVE : Ajuster seuils RSI selon ATR (volatilité)
            atr_15m = data_5m.get('atr_14')
            current_price = data_5m.get('close', prices[-1] if prices else 0)
            atr_percent = (atr_15m / current_price * 100) if atr_15m and current_price > 0 else 2.0
            
            # Seuils RSI adaptatifs selon volatilité
            if atr_percent > 5.0:  # Haute volatilité
                rsi_buy_threshold = 75  # Plus tolérant
                rsi_sell_threshold = 25
            elif atr_percent > 3.0:  # Volatilité moyenne
                rsi_buy_threshold = 80
                rsi_sell_threshold = 20
            else:  # Faible volatilité
                rsi_buy_threshold = 85  # Plus strict
                rsi_sell_threshold = 15
            
            # Validation RSI avec seuils adaptatifs
            rsi_5m = data_5m.get('rsi_14')
            if rsi_5m:
                if side == "BUY" and rsi_5m > rsi_buy_threshold:
                    logger.info(f"Signal BUY {symbol} rejeté : RSI 5m surachat ({rsi_5m:.1f} > {rsi_buy_threshold}) - ATR={atr_percent:.1f}%")
                    return False
                elif side == "SELL" and rsi_5m < rsi_sell_threshold:
                    logger.info(f"Signal SELL {symbol} rejeté : RSI 5m survente ({rsi_5m:.1f} < {rsi_sell_threshold}) - ATR={atr_percent:.1f}%")
                    return False

            logger.debug(f"Signal {side} {symbol} validé par analyse multi-indicateurs 5m - tendance={trend_5m} BB={bb_position} ATR={atr_percent:.1f}%")
            return True
            
        except Exception as e:
            logger.error(f"Erreur validation multi-timeframe: {e}")
            return True  # Mode dégradé : accepter le signal
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calcule une EMA simple (fallback)"""
        if not prices or period <= 0:
            return prices[-1] if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _get_or_calculate_ema_incremental(self, symbol: str, current_price: float, period: int) -> float:
        """
        Récupère ou calcule EMA de manière incrémentale pour éviter les dents de scie.
        Utilise le même principe que le Gateway WebSocket.
        """
        timeframe = "1m"  # Timeframe principal du Signal Aggregator
        cache_key = f'ema_{period}'
        
        # Initialiser le cache si nécessaire
        if symbol not in self.ema_incremental_cache:
            self.ema_incremental_cache[symbol] = {}
        if timeframe not in self.ema_incremental_cache[symbol]:
            self.ema_incremental_cache[symbol][timeframe] = {}
            
        cache = self.ema_incremental_cache[symbol][timeframe]
        
        # Récupérer EMA précédente du cache
        prev_ema = cache.get(cache_key)
        
        if prev_ema is None:
            # Première fois : utiliser le prix actuel comme base
            new_ema = current_price
        else:
            # Calcul incrémental standard
            from shared.src.technical_indicators import calculate_ema_incremental
            new_ema = calculate_ema_incremental(current_price, prev_ema, period)
        
        # Mettre à jour le cache
        cache[cache_key] = new_ema
        
        return new_ema
    
    def _get_or_calculate_indicator_incremental(self, symbol: str, current_candle: Dict, indicator_type: str, **params) -> Optional[float]:
        """
        Méthode générique pour calculer n'importe quel indicateur de manière incrémentale.
        Évite les dents de scie pour MACD, RSI, ATR, Stochastic, etc.
        
        Args:
            symbol: Symbole tradé
            current_candle: Bougie actuelle avec OHLCV
            indicator_type: Type d'indicateur ('macd', 'rsi', 'atr', 'stoch', etc.)
            **params: Paramètres spécifiques (period, etc.)
        """
        timeframe = "1m"
        
        # Initialiser le cache si nécessaire
        if symbol not in self.ema_incremental_cache:
            self.ema_incremental_cache[symbol] = {}
        if timeframe not in self.ema_incremental_cache[symbol]:
            self.ema_incremental_cache[symbol][timeframe] = {}
            
        cache = self.ema_incremental_cache[symbol][timeframe]
        
        try:
            if indicator_type == 'macd':
                # MACD incrémental (line, signal, histogram)
                prev_ema_fast = cache.get('macd_ema_fast')
                prev_ema_slow = cache.get('macd_ema_slow') 
                prev_macd_signal = cache.get('macd_signal')
                
                from shared.src.technical_indicators import calculate_macd_incremental
                result = calculate_macd_incremental(
                    current_candle['close'], prev_ema_fast, prev_ema_slow, prev_macd_signal
                )
                
                # Mettre à jour le cache
                cache['macd_ema_fast'] = result['ema_fast']
                cache['macd_ema_slow'] = result['ema_slow']
                cache['macd_signal'] = result['macd_signal']
                cache['macd_line'] = result['macd_line']
                cache['macd_histogram'] = result['macd_histogram']
                
                return result
                
            elif indicator_type == 'rsi':
                # RSI incrémental (nécessite historique de gains/pertes)
                period = params.get('period', 14)
                prev_rsi = cache.get(f'rsi_{period}')
                prev_avg_gain = cache.get(f'rsi_{period}_avg_gain')
                prev_avg_loss = cache.get(f'rsi_{period}_avg_loss')
                prev_price = cache.get('prev_close')
                
                if prev_price is None:
                    # Première fois : utiliser valeur neutre
                    cache['prev_close'] = current_candle['close']
                    return 50.0
                
                # Calculer gain/perte pour cette période
                price_change = current_candle['close'] - prev_price
                gain = max(price_change, 0)
                loss = max(-price_change, 0)
                
                if prev_avg_gain is None or prev_avg_loss is None:
                    # Initialisation
                    avg_gain = gain
                    avg_loss = loss
                else:
                    # Calcul incrémental des moyennes
                    alpha = 1.0 / period
                    avg_gain = alpha * gain + (1 - alpha) * prev_avg_gain
                    avg_loss = alpha * loss + (1 - alpha) * prev_avg_loss
                
                # Calculer RSI
                if avg_loss == 0:
                    rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100.0 - (100.0 / (1.0 + rs))
                
                # Mettre à jour le cache
                cache[f'rsi_{period}'] = rsi
                cache[f'rsi_{period}_avg_gain'] = avg_gain
                cache[f'rsi_{period}_avg_loss'] = avg_loss
                cache['prev_close'] = current_candle['close']
                
                return rsi
                
            elif indicator_type == 'sma':
                # SMA incrémental
                period = params.get('period', 20)
                prev_sma = cache.get(f'sma_{period}')
                
                from shared.src.technical_indicators import calculate_sma_incremental
                new_sma = calculate_sma_incremental(current_candle['close'], prev_sma, period)
                
                cache[f'sma_{period}'] = new_sma
                return new_sma
                
            elif indicator_type == 'atr':
                # ATR incrémental pour stop-loss plus précis
                period = params.get('period', 14)
                prev_atr = cache.get(f'atr_{period}')
                prev_close = cache.get('atr_prev_close')
                
                if prev_close is None:
                    cache['atr_prev_close'] = current_candle['close']
                    cache[f'atr_{period}'] = current_candle['high'] - current_candle['low']
                    return cache[f'atr_{period}']
                
                # Calcul True Range
                tr1 = current_candle['high'] - current_candle['low']
                tr2 = abs(current_candle['high'] - prev_close)
                tr3 = abs(current_candle['low'] - prev_close)
                true_range = max(tr1, tr2, tr3)
                
                # ATR incrémental (EMA du True Range)
                if prev_atr is None:
                    new_atr = true_range
                else:
                    alpha = 2.0 / (period + 1)
                    new_atr = alpha * true_range + (1 - alpha) * prev_atr
                
                cache[f'atr_{period}'] = new_atr
                cache['atr_prev_close'] = current_candle['close']
                return new_atr
                
            elif indicator_type == 'stoch':
                # Stochastic incrémental (K% et D%)
                period_k = params.get('period_k', 14)
                period_d = params.get('period_d', 3)
                
                # Maintenir historique des highs/lows pour K%
                highs_key = f'stoch_highs_{period_k}'
                lows_key = f'stoch_lows_{period_k}'
                
                if highs_key not in cache:
                    cache[highs_key] = []
                if lows_key not in cache:
                    cache[lows_key] = []
                
                # Ajouter valeurs actuelles
                cache[highs_key].append(current_candle['high'])
                cache[lows_key].append(current_candle['low'])
                
                # Maintenir seulement les dernières 'period_k' valeurs
                if len(cache[highs_key]) > period_k:
                    cache[highs_key] = cache[highs_key][-period_k:]
                if len(cache[lows_key]) > period_k:
                    cache[lows_key] = cache[lows_key][-period_k:]
                
                if len(cache[highs_key]) < period_k:
                    return {'stoch_k': 50.0, 'stoch_d': 50.0}  # Valeurs neutres
                
                # Calcul K%
                highest_high = max(cache[highs_key])
                lowest_low = min(cache[lows_key])
                
                if highest_high == lowest_low:
                    stoch_k = 50.0
                else:
                    stoch_k = ((current_candle['close'] - lowest_low) / (highest_high - lowest_low)) * 100
                
                # Calcul D% (SMA de K%)
                k_history_key = f'stoch_k_history_{period_d}'
                if k_history_key not in cache:
                    cache[k_history_key] = []
                    
                cache[k_history_key].append(stoch_k)
                if len(cache[k_history_key]) > period_d:
                    cache[k_history_key] = cache[k_history_key][-period_d:]
                
                stoch_d = sum(cache[k_history_key]) / len(cache[k_history_key])
                
                result = {'stoch_k': stoch_k, 'stoch_d': stoch_d}
                cache['stoch_k'] = stoch_k
                cache['stoch_d'] = stoch_d
                return result
                
            elif indicator_type == 'bollinger':
                # Bollinger Bands incrémental
                period = params.get('period', 20)
                std_dev = params.get('std_dev', 2.0)
                
                # Maintenir historique des prix pour calcul écart-type
                prices_key = f'bb_prices_{period}'
                if prices_key not in cache:
                    cache[prices_key] = []
                    
                cache[prices_key].append(current_candle['close'])
                if len(cache[prices_key]) > period:
                    cache[prices_key] = cache[prices_key][-period:]
                
                if len(cache[prices_key]) < period:
                    return None  # Pas assez de données
                
                # Calcul SMA (middle band)
                sma = sum(cache[prices_key]) / period
                
                # Calcul écart-type
                variance = sum((price - sma) ** 2 for price in cache[prices_key]) / period
                std = variance ** 0.5
                
                # Calcul des bandes
                bb_upper = sma + (std_dev * std)
                bb_lower = sma - (std_dev * std)
                bb_width = (bb_upper - bb_lower) / sma if sma > 0 else 0
                
                # Position relative du prix (0 = bande basse, 1 = bande haute)
                if bb_upper == bb_lower:
                    bb_position = 0.5
                else:
                    bb_position = (current_candle['close'] - bb_lower) / (bb_upper - bb_lower)
                
                result = {
                    'bb_upper': bb_upper,
                    'bb_middle': sma,
                    'bb_lower': bb_lower,
                    'bb_position': bb_position,
                    'bb_width': bb_width
                }
                
                # Mettre à jour le cache
                for key, value in result.items():
                    cache[key] = value
                    
                return result
                
            else:
                # Fallback pour indicateurs non implémentés
                logger.debug(f"Indicateur incrémental non implémenté: {indicator_type}")
                return None
                
        except Exception as e:
            logger.error(f"Erreur calcul incrémental {indicator_type} pour {symbol}: {e}")
            return None
    
    async def _get_current_adx(self, symbol: str) -> Optional[float]:
        """
        Récupère la valeur ADX actuelle depuis Redis
        
        Args:
            symbol: Symbole concerné
            
        Returns:
            Valeur ADX ou None si non disponible
        """
        try:
            # Essayer d'abord les données 1m (plus récentes)
            market_data_key = f"market_data:{symbol}:1m"
            data_1m = self.redis.get(market_data_key)
            
            if data_1m and isinstance(data_1m, dict):
                adx = data_1m.get('adx_14')
                if adx is not None:
                    return float(adx)
            
            # Fallback sur les données 5m
            market_data_key_5m = f"market_data:{symbol}:5m"
            data_5m = self.redis.get(market_data_key_5m)
            
            if data_5m and isinstance(data_5m, dict):
                adx = data_5m.get('adx_14')
                if adx is not None:
                    return float(adx)
                    
            logger.debug(f"ADX non disponible pour {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Erreur récupération ADX pour {symbol}: {e}")
            return None


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
        
    async def _validate_signal_correlation(self, signals: List[Dict]) -> float:
        """
        Valide la corrélation entre les signaux multiples
        
        Returns:
            Score de corrélation (0-1)
        """
        if len(signals) < 2:
            return 1.0
        
        # Analyser les métadonnées des signaux
        correlation_score = 1.0
        
        # Vérifier que les signaux pointent dans la même direction générale
        price_targets = []
        stop_losses = []
        
        for signal in signals:
            metadata = signal.get('metadata', {})
            
            # Extraire les niveaux clés
            if 'stop_price' in metadata:
                stop_losses.append(metadata['stop_price'])
            
            # Analyser les indicateurs sous-jacents
            if 'rsi' in metadata:
                rsi = metadata['rsi']
                side = signal.get('side', signal.get('side'))
                
                # Pénaliser si RSI contradictoire
                if side == 'BUY' and rsi > 70:
                    correlation_score *= 0.7
                elif side == 'SELL' and rsi < 30:
                    correlation_score *= 0.7
        
        # Vérifier la cohérence des stops
        if len(stop_losses) >= 2:
            stop_std = np.std(stop_losses) / np.mean(stop_losses)
            if stop_std > 0.02:  # Plus de 2% d'écart
                correlation_score *= 0.8
        
        return correlation_score
    
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
            current_adx = await self._get_current_adx(symbol)
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
    
    async def _check_regime_transition(self, symbol: str) -> bool:
        """
        Vérifie si on est en transition de régime (moment délicat)
        
        Returns:
            True si en transition, False sinon
        """
        try:
            # Récupérer l'historique des régimes
            regime_history_key = f"regime_history:{symbol}"
            try:
                # Essayer d'utiliser la méthode Redis standard
                history = self.redis.lrange(regime_history_key, 0, 10)
            except AttributeError:
                # Fallback pour RedisClientPool customisé
                history_str = self.redis.get(regime_history_key)
                if history_str:
                    history = json.loads(history_str) if isinstance(history_str, str) else history_str
                    if isinstance(history, list):
                        history = history[:10]  # Prendre les 10 premiers
                    else:
                        history = []
                else:
                    history = []
            
            if len(history) < 3:
                return False
            
            # Analyser les changements récents
            recent_regimes = []
            for h in history[:3]:
                if isinstance(h, str):
                    regime_data = json.loads(h)
                else:
                    regime_data = h
                recent_regimes.append(regime_data.get('regime'))
            
            unique_regimes = len(set(recent_regimes))
            
            # Si plus de 2 régimes différents dans les 3 derniers = transition
            if unique_regimes > 2:
                return True
            
            # Vérifier le temps depuis le dernier changement
            if symbol in self.last_regime_change:
                time_since_change = datetime.now() - self.last_regime_change[symbol]
                if time_since_change < self.regime_transition_cooldown:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Erreur vérification transition régime: {e}")
            return False
    
    async def _apply_market_context_filters(self, signal: Dict) -> bool:
        """
        Applique des filtres basés sur le contexte de marché global
        
        Returns:
            True si le signal passe les filtres, False sinon
        """
        symbol = signal['symbol']
        
        # 1. Vérifier les heures de trading (éviter les heures creuses)
        current_hour = datetime.now().hour
        if current_hour in [22, 23, 0, 1, 2, 3, 4, 5]:  # Heures creuses crypto
            # Augmenter le seuil de confiance pendant ces heures
            min_confidence = 0.8
        else:
            min_confidence = self.min_confidence_threshold
        
        if signal.get('confidence', 0) < min_confidence:
            logger.debug(f"Signal filtré: confiance insuffisante pendant heures creuses")
            return False
        
        # 2. Vérifier le spread bid/ask
        spread_key = f"spread:{symbol}"
        spread = self.redis.get(spread_key)
        if spread and float(spread) > 0.003:  # Spread > 0.3%
            logger.info(f"Signal filtré: spread trop large ({float(spread):.3%})")
            return False
        
        # 3. Vérifier la liquidité récente
        volume_key = f"volume_1h:{symbol}"
        recent_volume = self.redis.get(volume_key)
        if recent_volume and float(recent_volume) < 100000:  # Volume < 100k
            logger.info(f"Signal filtré: volume insuffisant ({float(recent_volume):.0f})")
            return False
        
        # 4. Vérifier les corrélations avec BTC (pour les altcoins)
        if symbol != "BTCUSDC":
            btc_correlation = await self._check_btc_correlation(symbol)
            if btc_correlation < -0.7:  # Forte corrélation négative
                logger.info(f"Signal filtré: corrélation BTC négative ({btc_correlation:.2f})")
                return False
        
        return True
    
    async def _check_btc_correlation(self, symbol: str) -> float:
        """
        Vérifie la corrélation avec BTC sur les dernières heures
        """
        try:
            # Récupérer les prix récents
            try:
                symbol_prices = self.redis.lrange(f"prices_1h:{symbol}", 0, 24)
                btc_prices = self.redis.lrange(f"prices_1h:BTCUSDC", 0, 24)
            except AttributeError:
                # Fallback pour RedisClientPool customisé
                symbol_prices_str = self.redis.get(f"prices_1h:{symbol}")
                btc_prices_str = self.redis.get(f"prices_1h:BTCUSDC")
                
                symbol_prices = []
                btc_prices = []
                
                if symbol_prices_str:
                    parsed = json.loads(symbol_prices_str) if isinstance(symbol_prices_str, str) else symbol_prices_str
                    symbol_prices = parsed[-24:] if isinstance(parsed, list) else []
                
                if btc_prices_str:
                    parsed = json.loads(btc_prices_str) if isinstance(btc_prices_str, str) else btc_prices_str
                    btc_prices = parsed[-24:] if isinstance(parsed, list) else []
            
            if len(symbol_prices) < 10 or len(btc_prices) < 10:
                return 0.0
            
            # Convertir en float
            symbol_prices = [float(p) for p in symbol_prices]
            btc_prices = [float(p) for p in btc_prices]
            
            # Calculer la corrélation
            symbol_returns = np.diff(np.array(symbol_prices))
            btc_returns = np.diff(np.array(btc_prices))
            
            if len(symbol_returns) > 0 and len(btc_returns) > 0:
                min_length = min(len(symbol_returns), len(btc_returns))
                correlation = np.corrcoef(
                    symbol_returns[:min_length],
                    btc_returns[:min_length]
                )[0, 1]
                
                return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Erreur calcul corrélation BTC: {e}")
            
        return 0.0
    
    async def _track_signal_accuracy(self, signal: Dict):
        """
        Suit la précision des signaux pour ajuster les poids dynamiquement
        """
        # Stocker le signal pour vérification future
        signal_key = f"pending_signal:{signal['symbol']}:{signal['strategy']}"
        signal_data = {
            'entry_price': signal['price'],
            'side': signal['side'],
            'timestamp': datetime.now().isoformat(),
            'stop_loss': signal.get('stop_loss'),
            'confidence': signal.get('confidence')
        }
        
        # Gérer les différents types de clients Redis
        try:
            self.redis.set(signal_key, json.dumps(signal_data), ex=3600)
        except TypeError:
            # Fallback pour RedisClientPool customisé
            self.redis.set(signal_key, json.dumps(signal_data), expiration=3600)
    
    async def _get_recovery_duration(self, symbol: str) -> float:
        """Get the duration since entering recovery period"""
        try:
            # Get danger history to find when we exited danger
            history_key = f"danger_history:{symbol}"
            history = []
            
            # Try to get recent history
            try:
                for i in range(20):  # Check last 20 entries
                    entry = self.redis.lindex(history_key, i)
                    if entry:
                        history.append(json.loads(entry))
            except:
                pass
                
            # Find when we exited danger zone
            exit_time = None
            for i, entry in enumerate(history):
                if entry['level'] < 5.0 and i > 0 and history[i-1]['level'] >= 7.0:
                    exit_time = datetime.fromisoformat(entry['timestamp'])
                    break
                    
            if exit_time:
                duration = (datetime.now() - exit_time).total_seconds()
                return duration
                
        except Exception as e:
            logger.error(f"Error calculating recovery duration: {e}")
            
        return 0  # Default to start of recovery
    
    async def _apply_enhanced_regime_filtering(self, signal: Dict[str, Any], regime, regime_metrics: Dict[str, float], 
                                             is_ultra_confluent: bool, signal_score: Optional[float], 
                                             strategy_count: int = 1) -> bool:
        """
        Applique un filtrage intelligent basé sur les régimes Enhanced.
        
        Args:
            signal: Signal à filtrer
            regime: Régime Enhanced détecté
            regime_metrics: Métriques du régime
            is_ultra_confluent: Si le signal est ultra-confluent
            signal_score: Score du signal (si disponible)
            strategy_count: Nombre de stratégies qui s'accordent sur ce signal
            
        Returns:
            True si le signal doit être accepté, False sinon
        """
        try:
            symbol = signal['symbol']
            signal_strength = signal.get('strength', 'moderate')
            signal_confidence = signal.get('confidence', 0.5)
            strategy = signal.get('strategy', 'Unknown')
            # Normaliser le nom de stratégie (retirer _Strategy)
            strategy = strategy.replace('_Strategy', '')
            side = signal.get('side', 'UNKNOWN')
            
            # NOUVEAU: Récupérer données techniques pour validation Enhanced
            technical_context = await self._get_technical_context(symbol)
            
            # Seuils adaptatifs selon le régime Enhanced + contexte technique
            if regime.name == 'STRONG_TREND_UP':
                # Tendance haussière forte: favoriser les BUY, pénaliser les SELL
                if side == 'SELL':
                    min_confidence = 0.80  # Pénaliser SELL en forte tendance haussière
                    required_strength = ['very_strong']
                    logger.debug(f"💪 {regime.name}: SELL pénalisé, seuils stricts pour {symbol}")
                else:  # BUY
                    # Validation MACD pour confirmer la force de tendance
                    if self._validate_macd_trend(technical_context, 'bullish'):
                        min_confidence = 0.35  # Encore plus permissif si MACD confirme
                        logger.debug(f"💪 {regime.name}: MACD confirme, seuils très assouplis pour {symbol}")
                    else:
                        min_confidence = 0.4
                        logger.debug(f"💪 {regime.name}: seuils assouplis pour {symbol}")
                    required_strength = ['weak', 'moderate', 'strong', 'very_strong']
                
            elif regime.name == 'TREND_UP':
                # Tendance haussière: favoriser les BUY, pénaliser modérément les SELL
                if side == 'SELL':
                    min_confidence = 0.75  # Pénaliser SELL en tendance haussière
                    required_strength = ['strong', 'very_strong']
                    logger.debug(f"📈 {regime.name}: SELL pénalisé, seuils élevés pour {symbol}")
                else:  # BUY
                    # Validation OBV pour confirmer le volume
                    if self._validate_obv_trend(technical_context, side):
                        min_confidence = 0.45  # Bonus si OBV confirme
                        logger.debug(f"📈 {regime.name}: OBV confirme, seuils bonus (0.45) pour {symbol}")
                    else:
                        min_confidence = 0.5  # ASSOUPLI à 0.50 (était 0.7)
                        logger.debug(f"📈 {regime.name}: seuils ASSOUPLIS (0.5) pour {symbol}")
                    required_strength = ['moderate', 'strong', 'very_strong']
                
            elif regime.name == 'WEAK_TREND_UP':
                # Tendance haussière faible: légère pénalisation des SELL
                if side == 'SELL':
                    min_confidence = 0.70  # Pénaliser légèrement SELL en tendance haussière faible
                    required_strength = ['strong', 'very_strong']
                    logger.debug(f"📊 {regime.name}: SELL légèrement pénalisé pour {symbol}")
                else:  # BUY
                    # Validation ROC pour détecter l'accélération
                    roc_boost = self._check_roc_acceleration(technical_context, side)
                    if roc_boost:
                        min_confidence = 0.50  # Bonus si ROC détecte accélération
                        logger.debug(f"📊 {regime.name}: ROC accélération détectée, seuils bonus (0.50) pour {symbol}")
                    else:
                        min_confidence = 0.55  # ASSOUPLI à 0.55 (était 0.65)
                        logger.debug(f"📊 {regime.name}: seuils ASSOUPLIS (0.55) pour {symbol}")
                    required_strength = ['moderate', 'strong', 'very_strong']
                
            elif regime.name == 'RANGE_TIGHT':
                # Gestion spéciale pour ADX très faible (marché plat)
                adx = regime_metrics.get('adx', 0)
                if adx <= 5:  # ADX près de 0
                    # Exiger confirmation volume élevé
                    volume_ratio = signal.get('metadata', {}).get('volume_ratio', 1.0)
                    if volume_ratio < 2.0:
                        logger.info(f"🚫 Signal rejeté en RANGE_TIGHT: ADX={adx:.1f} et volume_ratio={volume_ratio:.1f} < 2.0")
                        return False
                    
                    # Marquer pour réduction de poids 0.5x
                    signal['metadata'] = signal.get('metadata', {})
                    signal['metadata']['adx_weight_modifier'] = 0.5
                    logger.info(f"⚖️ ADX faible ({adx:.1f}): poids réduit à 0.5x pour {symbol}")
                
                # Range serré: ASSOUPLI pour mean-reversion
                if strategy in self.STRATEGY_GROUPS.get('mean_reversion', []):
                    # ASSOUPLI pour stratégies de mean-reversion
                    min_confidence = 0.6  # ASSOUPLI de 0.75 à 0.6
                    required_strength = ['moderate', 'strong', 'very_strong']  # Ajouter moderate
                    logger.debug(f"🔒 {regime.name}: seuils ASSOUPLIS pour mean-reversion {symbol}")
                else:
                    # ASSOUPLI pour autres stratégies aussi
                    min_confidence = 0.7  # ASSOUPLI de 0.8 à 0.7
                    required_strength = ['strong', 'very_strong']
                    logger.debug(f"🔒 {regime.name}: seuils ASSOUPLIS (0.7) pour {symbol}")
                
            elif regime.name == 'RANGE_VOLATILE':
                # Range volatil: sélectif mais moins que tight
                min_confidence = 0.7
                required_strength = ['strong', 'very_strong']
                logger.debug(f"⚡ {regime.name}: seuils stricts pour {symbol}")
                
            elif regime.name in ['WEAK_TREND_DOWN', 'TREND_DOWN', 'STRONG_TREND_DOWN']:
                # Tendances baissières: favoriser les SELL, bloquer les BUY faibles
                if side == 'BUY':
                    min_confidence = 0.80  # Assoupli de 0.85 à 0.80 pour les BUY en downtrend
                    required_strength = ['very_strong']
                else:  # SELL
                    min_confidence = 0.7  # Seuil ajusté pour les SELL (0.7 recommandé)
                    required_strength = ['moderate', 'strong', 'very_strong']
                logger.debug(f"📉 {regime.name}: adaptation BUY/SELL pour {symbol}")
                
            else:
                # Régime inconnu ou UNDEFINED: seuils par défaut
                min_confidence = 0.6
                required_strength = ['strong', 'very_strong']
                logger.debug(f"❓ {regime.name}: seuils par défaut pour {symbol}")
            
            # Exception pour signaux ultra-confluents de haute qualité
            if is_ultra_confluent and signal_score:
                if signal_score >= 85:
                    # Signaux excellents: réduire les seuils
                    min_confidence *= 0.8
                    if 'moderate' not in required_strength:
                        required_strength.append('moderate')
                    logger.info(f"⭐ Signal ultra-confluent excellent (score={signal_score:.1f}): seuils réduits pour {symbol}")
                elif signal_score >= 75:
                    # Signaux très bons: réduire modérément
                    min_confidence *= 0.9
                    logger.info(f"✨ Signal ultra-confluent très bon (score={signal_score:.1f}): seuils ajustés pour {symbol}")
            
            # Appliquer les filtres
            if signal_confidence < min_confidence:
                logger.info(f"🚫 Signal rejeté en {regime.name}: confiance {signal_confidence:.2f} < {min_confidence:.2f} "
                           f"pour {strategy} {side} {symbol}")
                return False
                
            # NOUVEAU: Accepter les signaux 'moderate' si 2+ stratégies s'accordent
            if signal_strength == 'moderate' and strategy_count >= 2:
                logger.info(f"✅ Signal 'moderate' accepté avec {strategy_count} stratégies en {regime.name}: "
                           f"{strategy} {side} {symbol}")
            elif signal_strength not in required_strength:
                logger.info(f"🚫 Signal rejeté en {regime.name}: force '{signal_strength}' insuffisante "
                           f"(requis: {required_strength}) pour {strategy} {side} {symbol}")
                return False
            
            # Signal accepté
            adx = regime_metrics.get('adx', 0)
            logger.info(f"✅ Signal accepté en {regime.name} (ADX={adx:.1f}): "
                       f"{strategy} {side} {symbol} force={signal_strength} confiance={signal_confidence:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur dans le filtrage Enhanced: {e}")
            return True  # En cas d'erreur, laisser passer le signal
    
    def _apply_volume_boost(self, confidence: float, signals: List[Dict[str, Any]]) -> float:
        """
        Applique un boost de confiance basé sur le volume_ratio des signaux
        
        Args:
            confidence: Confiance actuelle du signal agrégé
            signals: Liste des signaux contributeurs
            
        Returns:
            Confiance boostée par le volume
        """
        try:
            volume_ratios = []
            volume_scores = []
            
            # Extraire les ratios de volume et scores des métadonnées
            for signal in signals:
                metadata = signal.get('metadata', {})
                
                # Chercher volume_ratio directement ou dans les sous-données
                volume_ratio = metadata.get('volume_ratio')
                if volume_ratio is None:
                    # Peut-être dans volume_spike ou autre champ volume
                    volume_ratio = metadata.get('volume_spike', 1.0)
                
                volume_score = metadata.get('volume_score', 0.5)
                
                if volume_ratio and isinstance(volume_ratio, (int, float)):
                    volume_ratios.append(float(volume_ratio))
                
                if volume_score and isinstance(volume_score, (int, float)):
                    volume_scores.append(float(volume_score))
            
            if not volume_ratios and not volume_scores:
                return confidence  # Pas de données volume, pas de boost
            
            # Calculer le boost basé sur volume_ratio
            volume_boost = 1.0
            if volume_ratios:
                avg_volume_ratio = sum(volume_ratios) / len(volume_ratios)
                
                if avg_volume_ratio >= 3.0:
                    # Volume très élevé: boost significatif (+15%)
                    volume_boost = 1.15
                    logger.info(f"🔊 Volume très élevé détecté: ratio={avg_volume_ratio:.1f} -> boost +15%")
                elif avg_volume_ratio >= 2.0:
                    # Volume élevé: boost modéré (+10%)
                    volume_boost = 1.10
                    logger.info(f"📢 Volume élevé détecté: ratio={avg_volume_ratio:.1f} -> boost +10%")
                elif avg_volume_ratio >= 1.5:
                    # Volume augmenté: boost léger (+5%)
                    volume_boost = 1.05
                    logger.debug(f"📈 Volume augmenté: ratio={avg_volume_ratio:.1f} -> boost +5%")
                elif avg_volume_ratio <= 0.5:
                    # Volume très faible: pénalité (-5%)
                    volume_boost = 0.95
                    logger.debug(f"📉 Volume faible: ratio={avg_volume_ratio:.1f} -> malus -5%")
            
            # Boost supplémentaire basé sur volume_score des stratégies
            if volume_scores:
                avg_volume_score = sum(volume_scores) / len(volume_scores)
                
                if avg_volume_score >= 0.8:
                    # Score volume excellent: boost additionnel (+5%)
                    volume_boost *= 1.05
                    logger.debug(f"⭐ Score volume excellent: {avg_volume_score:.2f} -> boost additionnel +5%")
                elif avg_volume_score <= 0.3:
                    # Score volume faible: pénalité (-3%)
                    volume_boost *= 0.97
                    logger.debug(f"⚠️ Score volume faible: {avg_volume_score:.2f} -> malus -3%")
            
            # Appliquer le boost final
            boosted_confidence = confidence * volume_boost
            
            if volume_boost != 1.0:
                logger.info(f"🎚️ Boost volume global: {confidence:.3f} -> {boosted_confidence:.3f} "
                           f"(facteur: {volume_boost:.3f})")
            
            return min(1.0, boosted_confidence)  # Cap à 1.0
            
        except Exception as e:
            logger.error(f"Erreur dans boost volume: {e}")
            return confidence  # En cas d'erreur, retourner confiance originale
    
    def _apply_multi_strategy_bonus(self, confidence: float, contributing_strategies: List[str]) -> float:
        """
        Applique un bonus de confiance si plusieurs stratégies convergent
        
        Args:
            confidence: Confiance actuelle
            contributing_strategies: Liste des stratégies contributrices
            
        Returns:
            Confiance boostée par la convergence multi-stratégies
        """
        try:
            strategy_count = len(contributing_strategies)
            
            if strategy_count >= 2:
                # Bonus +0.05 pour 2+ stratégies alignées
                bonus = 0.05
                boosted_confidence = confidence + bonus
                
                logger.info(f"🤝 Bonus multi-stratégies: {strategy_count} stratégies -> "
                           f"{confidence:.3f} + {bonus:.2f} = {boosted_confidence:.3f}")
                
                return min(1.0, boosted_confidence)  # Cap à 1.0
            
            return confidence
            
        except Exception as e:
            logger.error(f"Erreur dans bonus multi-stratégies: {e}")
            return confidence
    
    def _extract_volume_summary(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extrait un résumé des données de volume des signaux pour les métadonnées
        
        Args:
            signals: Liste des signaux contributeurs
            
        Returns:
            Dictionnaire avec le résumé des données volume
        """
        try:
            volume_ratios = []
            volume_scores = []
            
            for signal in signals:
                metadata = signal.get('metadata', {})
                
                volume_ratio = metadata.get('volume_ratio')
                if volume_ratio is None:
                    volume_ratio = metadata.get('volume_spike', 1.0)
                
                volume_score = metadata.get('volume_score', 0.5)
                
                if volume_ratio and isinstance(volume_ratio, (int, float)):
                    volume_ratios.append(float(volume_ratio))
                
                if volume_score and isinstance(volume_score, (int, float)):
                    volume_scores.append(float(volume_score))
            
            summary = {
                'signals_with_volume': len(volume_ratios),
                'total_signals': len(signals)
            }
            
            if volume_ratios:
                summary.update({
                    'avg_volume_ratio': round(sum(volume_ratios) / len(volume_ratios), 2),
                    'max_volume_ratio': round(max(volume_ratios), 2),
                    'min_volume_ratio': round(min(volume_ratios), 2)
                })
            
            if volume_scores:
                summary.update({
                    'avg_volume_score': round(sum(volume_scores) / len(volume_scores), 3),
                    'max_volume_score': round(max(volume_scores), 3)
                })
            
            return summary
            
        except Exception as e:
            logger.error(f"Erreur extraction résumé volume: {e}")
            return {'error': 'extraction_failed'}