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

try:
    from enhanced_regime_detector import EnhancedRegimeDetector, MarketRegime
except ImportError as e:
    # Fallback si import échoue
    logger.warning(f"Enhanced regime detector non disponible ({e}), utilisation du mode standard")
    EnhancedRegimeDetector = None
    MarketRegime = None


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
        
        # Nouveau détecteur de régime amélioré (si disponible)
        if EnhancedRegimeDetector:
            self.enhanced_regime_detector = EnhancedRegimeDetector(redis_client)
            # Connecter l'accumulateur au détecteur
            self.enhanced_regime_detector.set_market_data_accumulator(self.market_data_accumulator)
            logger.info("✅ Enhanced Regime Detector activé avec accumulateur historique")
        else:
            self.enhanced_regime_detector = None
            logger.warning("⚠️ Enhanced Regime Detector non disponible, utilisation du détecteur classique")
        
        # Signal buffer for aggregation
        self.signal_buffer = defaultdict(list)
        self.last_signal_time = {}
        self.cooldown_period = timedelta(minutes=1)  # Réduit à 1 minute pour mode scalping
        
        # Voting thresholds adaptatifs selon le régime
        self.min_vote_threshold = 0.35  # Base pour régimes favorables
        self.min_confidence_threshold = 0.55  # Base pour tous régimes
        
        # Seuils spéciaux pour RANGE_TIGHT (plus permissifs)
        self.range_tight_vote_threshold = 0.25  # Plus bas pour RANGE_TIGHT
        self.range_tight_confidence_threshold = 0.50  # Plus bas pour RANGE_TIGHT
        
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
        
    async def _update_market_data_history(self, symbol: str) -> None:
        """Met à jour l'historique des données de marché pour un symbole"""
        try:
            # Récupérer les données actuelles depuis Redis
            key = f"market_data:{symbol}:1m"
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
            
            # Convert 'side' to 'side' for compatibility
            if 'side' in signal and 'side' not in signal:
                signal['side'] = signal.pop('side')
            
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
            
            # NOUVEAU: Validation multi-timeframe avec 5m (RÉACTIVÉE pour scalping optimisé)
            # Validation 5m plus rapide que 15m, gardant la qualité sans sacrifier la réactivité
            if not await self._validate_signal_with_higher_timeframe(signal):
                logger.info(f"Signal {strategy} {signal['side']} sur {symbol} rejeté par validation 5m")
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
            
            # Check if we have enough signals to make a decision - MODE SCALPING (1 signal suffit)
            if len(self.signal_buffer[symbol]) < 1:
                return None  # Wait for more signals
                
            # Get market regime FIRST pour filtrage intelligent (enhanced if available, sinon fallback)
            if self.enhanced_regime_detector:
                regime, regime_metrics = await self.enhanced_regime_detector.get_detailed_regime(symbol)
                
                # NOUVEAU: Filtrage intelligent basé sur les régimes Enhanced
                signal_filtered = await self._apply_enhanced_regime_filtering(
                    signal, regime, regime_metrics, is_ultra_confluent, signal_score
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
            
        # Weighted average of stop loss seulement (plus de target avec TrailingStop pur)
        stop_loss_sum = 0
        
        for signal in signals:
            signal_side = signal.get('side', signal.get('side'))
            if signal_side == side and signal['strategy'] in contributing_strategies:
                weight = await self.performance_tracker.get_strategy_weight(signal['strategy'])
                
                # Extract stop_price from metadata (plus de target_price avec TrailingStop pur)
                metadata = signal.get('metadata', {})
                # Stop-loss correct selon le side: SELL stop au dessus, BUY stop en dessous
                default_stop = signal['price'] * (1.002 if side == 'SELL' else 0.998)
                stop_price = metadata.get('stop_price', signal.get('stop_loss', default_stop))
                
                stop_loss_sum += stop_price * weight
                
        stop_loss = stop_loss_sum / total_weight
        
        # Get the latest price from one of the signals
        current_price = signals[0]['price'] if signals else 0.0
        
        # Create main strategy name from contributing strategies
        main_strategy = contributing_strategies[0] if contributing_strategies else 'SignalAggregator'
        
        # NOUVEAU: Volume-based confidence boost (classique)
        confidence = self._apply_volume_boost(confidence, signals)
        
        # Déterminer la force du signal basée sur la confiance
        # MODIFIÉ: Seuils ajustés pour éviter l'amplification artificielle
        if confidence >= 0.9:  # Augmenté de 0.8 à 0.9
            strength = 'very_strong'
        elif confidence >= 0.75:  # Augmenté de 0.6 à 0.75
            strength = 'strong'
        elif confidence >= 0.5:  # Augmenté de 0.4 à 0.5
            strength = 'moderate'
        else:
            strength = 'weak'
            
        # Trailing stop fixe à 3% pour système pur (TrailingStop gère le reste)
        trailing_delta = 3.0
        
        # Validation supplémentaire pour Aggregated_1 (une seule stratégie)
        # MODE SCALPING: Seuil réduit pour les signaux uniques pour plus de trades
        if len(contributing_strategies) == 1 and confidence < 0.55:
            logger.info(f"Signal Aggregated_1 rejeté pour {symbol}: confiance {confidence:.2f} < 0.55 (mode scalping)")
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
            
            # Combined weight (performance * regime adaptation)
            combined_weight = performance_weight * regime_weight
            
            # Apply adaptive confidence threshold based on regime
            confidence = signal.get('confidence', 0.5)
            confidence_threshold = self.min_confidence_threshold
            
            # Seuils adaptatifs pour certains régimes
            if hasattr(regime, 'name') and regime.name == 'RANGE_TIGHT':
                confidence_threshold = self.range_tight_confidence_threshold
                logger.debug(f"📊 Seuil RANGE_TIGHT adaptatif: {confidence_threshold} pour {strategy}")
            
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
                    
            final_combined_weight = combined_weight * quality_boost
            
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

        # Enhanced decision logic based on regime
        min_threshold = self._get_regime_threshold(regime)
        
        # Adapter le seuil de vote pour RANGE_TIGHT
        if hasattr(regime, 'name') and regime.name == 'RANGE_TIGHT':
            min_threshold = self.range_tight_vote_threshold
            logger.debug(f"📊 Seuil de vote RANGE_TIGHT adaptatif: {min_threshold}")
        
        # Determine side
        if BUY_score > SELL_score and BUY_score >= min_threshold:
            side = 'BUY'
            confidence = BUY_score / (BUY_score + SELL_score)
            contributing_strategies = [s['strategy'] for s in BUY_signals]
            relevant_signals = BUY_signals
        elif SELL_score > BUY_score and SELL_score >= min_threshold:
            side = 'SELL'
            confidence = SELL_score / (BUY_score + SELL_score)
            contributing_strategies = [s['strategy'] for s in SELL_signals]
            relevant_signals = SELL_signals
        else:
            # No clear signal
            logger.debug(f"No clear signal for {symbol} in {regime.value}: BUY={BUY_score:.2f}, SELL={SELL_score:.2f}")
            return None
            
        # Calculate averaged stop loss
        total_weight = sum(s['combined_weight'] for s in relevant_signals)
        if total_weight == 0:
            return None
            
        # Weighted average of stop loss
        stop_loss_sum = 0
        
        for signal in signals:
            signal_side = signal.get('side', signal.get('side'))
            if signal_side == side and signal['strategy'] in contributing_strategies:
                # Find the corresponding weighted signal
                weighted_sig = next((s for s in relevant_signals if s['strategy'] == signal['strategy']), None)
                if weighted_sig:
                    weight = weighted_sig['combined_weight']
                    
                    # Extract stop_price from metadata
                    metadata = signal.get('metadata', {})
                    # Stop-loss correct selon le side: SELL stop au dessus, BUY stop en dessous
                    default_stop = signal['price'] * (1.002 if side == 'SELL' else 0.998)
                    stop_price = metadata.get('stop_price', signal.get('stop_loss', default_stop))
                    
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
        
        # Déterminer la force du signal basée sur la confiance et le régime
        strength = self._determine_signal_strength(confidence, regime)
        
        # VRAIE logique pour 'moderate' avec ≥2 stratégies
        # Assouplir la force si multiple strategies en régime strict
        if (strength == 'moderate' and len(contributing_strategies) >= 2 and 
            hasattr(regime, 'name') and regime.name in ['RANGE_TIGHT', 'RANGE_VOLATILE']):
            logger.info(f"✅ Force 'moderate' VRAIMENT acceptée: {len(contributing_strategies)} stratégies convergent "
                       f"en {regime.name} pour {symbol}")
            # Force sera validée comme acceptable plus tard
        
        # Trailing stop fixe à 3% pour système pur
        trailing_delta = 3.0
        
        # Validation renforcée pour les signaux uniques selon le régime
        if len(contributing_strategies) == 1:
            min_single_confidence = self._get_single_strategy_threshold(regime)
            if confidence < min_single_confidence:
                logger.info(f"Signal Aggregated_1 rejeté pour {symbol} en régime {regime.value}: "
                           f"confiance {confidence:.2f} < {min_single_confidence:.2f}")
                return None
        
        # VALIDATION FINALE: Override pour 'moderate' avec ≥2 stratégies
        final_strength = strength
        if (strength == 'moderate' and len(contributing_strategies) >= 2 and 
            hasattr(regime, 'name') and regime.name in ['RANGE_TIGHT', 'RANGE_VOLATILE']):
            # Force acceptée malgré les règles strictes du régime
            logger.info(f"🚀 Override 'moderate' appliqué: {len(contributing_strategies)} stratégies "
                       f"en {regime.name} pour {symbol}")
        
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
        Valide un signal 1m avec le contexte 15m pour éviter les faux signaux.
        
        Logique de validation :
        - Signal BUY : validé si la tendance 15m est haussière ou neutre
        - Signal SELL : validé si la tendance 15m est baissière ou neutre

        Args:
            signal: Signal 1m à valider
            
        Returns:
            True si le signal est validé, False sinon
        """
        try:
            symbol = signal['symbol']
            side = signal['side']

            # Récupérer les données 5m récentes depuis Redis (MODE SCALPING)
            market_data_key = f"market_data:{symbol}:5m"
            data_5m = self.redis.get(market_data_key)
            
            if not data_5m:
                # Si pas de données 5m, on accepte le signal (mode dégradé)
                logger.warning(f"Pas de données 5m pour {symbol}, validation en mode dégradé")
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
            
            # Utiliser les mêmes EMAs que les stratégies : 21 vs 50 périodes
            ema_21 = self._calculate_ema(prices[-21:], 21)
            ema_50 = self._calculate_ema(prices[-50:], 50)
            
            # LOGIQUE SOPHISTIQUÉE : Analyser la force et le momentum de la tendance
            current_price = prices[-1] if prices else 0
            
            # Calculer la vélocité des EMAs (momentum)
            if len(prices) >= 55:  # Besoin de données pour calculer la vélocité
                ema_21_prev = self._calculate_ema(prices[-26:-5], 21)  # EMA21 il y a 5 périodes
                ema_50_prev = self._calculate_ema(prices[-55:-5], 50)  # EMA50 il y a 5 périodes
                ema_21_velocity = (ema_21 - ema_21_prev) / ema_21_prev if ema_21_prev > 0 else 0
                ema_50_velocity = (ema_50 - ema_50_prev) / ema_50_prev if ema_50_prev > 0 else 0
            else:
                ema_21_velocity = 0
                ema_50_velocity = 0
            
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
            
            # Validation additionnelle : RSI 5m (MODE SCALPING - seuils ajustés)
            rsi_5m = data_5m.get('rsi_14')
            if rsi_5m:
                if side == "BUY" and rsi_5m > 80:  # Seuil plus élevé pour scalping
                    logger.info(f"Signal BUY {symbol} rejeté : RSI 5m surachat ({rsi_5m})")
                    return False
                elif side == "SELL" and rsi_5m < 20:  # Seuil plus bas pour scalping
                    logger.info(f"Signal SELL {symbol} rejeté : RSI 5m survente ({rsi_5m})")
                    return False

            logger.debug(f"Signal {side} {symbol} validé par tendance 5m {trend_5m}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur validation multi-timeframe: {e}")
            return True  # Mode dégradé : accepter le signal
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calcule une EMA simple"""
        if not prices or period <= 0:
            return prices[-1] if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
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
    
    def __init__(self, redis_client, regime_detector, performance_tracker):
        super().__init__(redis_client, regime_detector, performance_tracker)
        
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
                                             is_ultra_confluent: bool, signal_score: Optional[float]) -> bool:
        """
        Applique un filtrage intelligent basé sur les régimes Enhanced.
        
        Args:
            signal: Signal à filtrer
            regime: Régime Enhanced détecté
            regime_metrics: Métriques du régime
            is_ultra_confluent: Si le signal est ultra-confluent
            signal_score: Score du signal (si disponible)
            
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
            
            # Seuils adaptatifs selon le régime Enhanced
            if regime.name == 'STRONG_TREND_UP':
                # Tendance forte haussière: accepter presque tout
                min_confidence = 0.4
                required_strength = ['weak', 'moderate', 'strong', 'very_strong']
                logger.debug(f"💪 {regime.name}: seuils assouplis pour {symbol}")
                
            elif regime.name == 'TREND_UP':
                # Tendance haussière: ASSOUPLI à 0.50 (était 0.7)
                min_confidence = 0.5
                required_strength = ['moderate', 'strong', 'very_strong']
                logger.debug(f"📈 {regime.name}: seuils ASSOUPLIS (0.5) pour {symbol}")
                
            elif regime.name == 'WEAK_TREND_UP':
                # Tendance faible: ASSOUPLI à 0.55 (était 0.65)
                min_confidence = 0.55
                required_strength = ['moderate', 'strong', 'very_strong']
                logger.debug(f"📊 {regime.name}: seuils ASSOUPLIS (0.55) pour {symbol}")
                
            elif regime.name == 'RANGE_TIGHT':
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
                    min_confidence = 0.85  # Très strict pour les BUY en downtrend
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
                
            if signal_strength not in required_strength:
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