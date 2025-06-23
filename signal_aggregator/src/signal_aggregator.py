#!/usr/bin/env python3
import logging
from typing import Dict, List, Optional, Any, TYPE_CHECKING, Union
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import json
import numpy as np

logger = logging.getLogger(__name__)

# Type alias pour le regime de marché
if TYPE_CHECKING:
    from enhanced_regime_detector import MarketRegime
    MarketRegimeType = Union[MarketRegime, Any]
else:
    MarketRegimeType = Any

try:
    from enhanced_regime_detector import EnhancedRegimeDetector, MarketRegime
except ImportError:
    # Fallback si import échoue
    logger.warning("Enhanced regime detector non disponible, utilisation du mode standard")
    EnhancedRegimeDetector = None
    MarketRegime = None


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
        
        # Nouveau détecteur de régime amélioré (si disponible)
        if EnhancedRegimeDetector:
            self.enhanced_regime_detector = EnhancedRegimeDetector(redis_client)
        else:
            self.enhanced_regime_detector = None
        
        # Signal buffer for aggregation
        self.signal_buffer = defaultdict(list)
        self.last_signal_time = {}
        self.cooldown_period = timedelta(minutes=1)  # Réduit à 1 minute pour mode scalping
        
        # Voting thresholds - MODE SCALPING (équilibré : fréquence vs qualité)
        self.min_vote_threshold = 0.35  # Réduit de 0.5 à 0.35
        self.min_confidence_threshold = 0.55  # Compromis : filtre le bruit mais garde la fréquence
        
    async def process_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a raw signal and return aggregated decision with multi-timeframe validation"""
        try:
            symbol = signal['symbol']
            strategy = signal['strategy']
            
            # Convert 'side' to 'side' for compatibility
            if 'side' in signal and 'side' not in signal:
                side = signal['side']
                if side in ['BUY', 'BUY']:
                    signal['side'] = 'BUY'
                elif side in ['SELL', 'SELL']:
                    signal['side'] = 'SELL'
                else:
                    logger.warning(f"Unknown side value: {side}")
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
                
            # Get market regime (enhanced if available, sinon fallback)
            if self.enhanced_regime_detector:
                regime, regime_metrics = await self.enhanced_regime_detector.get_detailed_regime(symbol)
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
                        'source_signals': len(self.signal_buffer[symbol])
                    })
                else:
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
            
            # Apply confidence threshold
            confidence = signal.get('confidence', 0.5)
            if confidence < self.min_confidence_threshold:
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
            elif side == 'SELL':
                SELL_signals.append(weighted_signal)

        # Calculate total scores
        BUY_score = sum(s['score'] for s in BUY_signals)
        SELL_score = sum(s['score'] for s in SELL_signals)

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
                stop_price = metadata.get('stop_price', signal.get('stop_loss', signal['price'] * 0.998))
                
                stop_loss_sum += stop_price * weight
                
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
            
            # Apply confidence threshold
            confidence = signal.get('confidence', 0.5)
            if confidence < self.min_confidence_threshold:
                continue

            # Get side (handle both 'side' and 'side' keys)
            side = signal.get('side', signal.get('side'))
            if side in ['BUY', 'BUY']:
                side = 'BUY'
            elif side in ['SELL', 'SELL']:
                side = 'SELL'

            # Enhanced weighted signal with regime adaptation
            weighted_signal = {
                'strategy': strategy,
                'side': side,
                'confidence': confidence,
                'performance_weight': performance_weight,
                'regime_weight': regime_weight,
                'combined_weight': combined_weight,
                'score': confidence * combined_weight
            }

            if side == 'BUY':
                BUY_signals.append(weighted_signal)
            elif side == 'SELL':
                SELL_signals.append(weighted_signal)

        # Calculate total scores
        BUY_score = sum(s['score'] for s in BUY_signals)
        SELL_score = sum(s['score'] for s in SELL_signals)

        # Enhanced decision logic based on regime
        min_threshold = self._get_regime_threshold(regime)
        
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
                    stop_price = metadata.get('stop_price', signal.get('stop_loss', signal['price'] * 0.998))
                    
                    stop_loss_sum += stop_price * weight
                
        stop_loss = stop_loss_sum / total_weight
        
        # Get the latest price from one of the signals
        current_price = signals[0]['price'] if signals else 0.0
        
        # Create main strategy name from contributing strategies
        main_strategy = contributing_strategies[0] if contributing_strategies else 'SignalAggregator'
        
        # Regime-adaptive confidence boost
        confidence = self._apply_regime_confidence_boost(confidence, regime, regime_metrics)
        
        # Déterminer la force du signal basée sur la confiance et le régime
        strength = self._determine_signal_strength(confidence, regime)
            
        # Trailing stop fixe à 3% pour système pur
        trailing_delta = 3.0
        
        # Validation renforcée pour les signaux uniques selon le régime
        if len(contributing_strategies) == 1:
            min_single_confidence = self._get_single_strategy_threshold(regime)
            if confidence < min_single_confidence:
                logger.info(f"Signal Aggregated_1 rejeté pour {symbol} en régime {regime.value}: "
                           f"confiance {confidence:.2f} < {min_single_confidence:.2f}")
                return None
        
        return {
            'symbol': symbol,
            'side': side,
            'price': current_price,
            'strategy': f"Aggregated_{len(contributing_strategies)}",
            'confidence': confidence,
            'strength': strength,
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
                'regime': regime.value
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
                confidence *= 0.95
        
        return min(1.0, confidence)  # Cap à 1.0
    
    def _determine_signal_strength(self, confidence: float, regime: Any) -> str:
        """Détermine la force du signal basée sur la confiance et le régime"""
        # Ajustement des seuils selon le régime
        if MarketRegime is not None and regime in [MarketRegime.STRONG_TREND_UP, MarketRegime.STRONG_TREND_DOWN]:
            # En tendance forte, on peut être plus agressif
            if confidence >= 0.75:
                return 'very_strong'
            elif confidence >= 0.6:
                return 'strong'
            elif confidence >= 0.45:
                return 'moderate'
            else:
                return 'weak'
        else:
            # Régimes moins favorables, seuils plus stricts
            if confidence >= 0.8:
                return 'very_strong'
            elif confidence >= 0.65:
                return 'strong'
            elif confidence >= 0.5:
                return 'moderate'
            else:
                return 'weak'
        
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
    
    async def process_signal_enhanced(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Version améliorée du traitement de signal avec plus de validations
        """
        # Si mode dégradé, utiliser le processus standard
        if not self.enhanced_mode:
            logger.debug("Mode dégradé: utilisation du process_signal standard")
            return await self.process_signal(signal)
        
        # Validations de base
        if not await self._apply_market_context_filters(signal):
            return None
            
        # NOUVEAU: Vérifier le niveau de danger du marché
        symbol = signal['symbol']
        danger_level = await self.regime_detector.get_danger_level(symbol)
        
        # Logique anti-piège à rebond
        if danger_level >= 7.0:
            # Marché dangereux: bloquer les nouveaux LONG/BUY
            side = signal.get('side', signal.get('side'))
            if side in ['BUY', 'LONG']:
                logger.info(f"🚫 Signal BUY bloqué pour {symbol}: danger level {danger_level:.1f} (anti-piège à rebond)")
                return None
            # Les SELL/SHORT passent en marché dangereux
            logger.info(f"⚠️ Signal SELL autorisé malgré danger {danger_level:.1f} pour {symbol}")
            
        elif danger_level >= 5.0:
            # Marché en alerte: réduire la confiance et augmenter les seuils
            signal['confidence'] = signal.get('confidence', 0.5) * 0.8
            logger.info(f"⚠️ Confiance réduite pour {symbol}: danger level {danger_level:.1f}")
            
        # NOUVEAU: Vérifier si on est en période de récupération
        if await self.regime_detector.is_in_recovery(symbol):
            side = signal.get('side', signal.get('side'))
            if side in ['BUY', 'LONG']:
                # Exiger une confirmation TRÈS forte pour réentrer après une crise
                min_confidence_recovery = 0.85
                if signal.get('confidence', 0) < min_confidence_recovery:
                    logger.info(f"🛡️ Signal BUY rejeté en période de récupération: "
                               f"confiance {signal.get('confidence', 0):.2f} < {min_confidence_recovery}")
                    return None
                logger.info(f"✅ Signal BUY accepté en récupération avec forte confiance: {signal.get('confidence', 0):.2f}")
        
        # Vérifier si on est en transition de régime
        if await self._check_regime_transition(signal['symbol']):
            # Augmenter le seuil de confiance pendant les transitions
            signal['confidence'] = signal.get('confidence', 0.5) * 0.8
            logger.info(f"Confiance réduite pour {signal['symbol']}: transition de régime")
        
        # Processus normal d'agrégation
        aggregated = await self.process_signal(signal)
        
        if aggregated:
            # Validation supplémentaire de corrélation
            correlation_score = await self._validate_signal_correlation(
                self.signal_buffer[signal['symbol']]
            )
            
            if correlation_score < self.correlation_threshold:
                logger.info(f"Signal rejeté: faible corrélation ({correlation_score:.2f})")
                return None
            
            # Ajuster la confiance finale
            aggregated['confidence'] *= correlation_score
            
            # NOUVEAU: Ajuster la taille de position selon le danger et la récupération
            if 'metadata' not in aggregated:
                aggregated['metadata'] = {}
                
            # Logique d'accumulation progressive
            if await self.regime_detector.is_in_recovery(symbol):
                # Période de récupération: accumulation progressive
                recovery_time = await self._get_recovery_duration(symbol)
                
                if recovery_time < 300:  # < 5 minutes depuis la sortie de danger
                    size_multiplier = 0.3  # Commencer petit (30%)
                    accumulation_phase = "initial"
                elif recovery_time < 600:  # < 10 minutes
                    size_multiplier = 0.6  # Augmenter progressivement (60%)
                    accumulation_phase = "progressive"
                else:
                    size_multiplier = 1.0  # Retour à la normale
                    accumulation_phase = "complete"
                    
                aggregated['metadata']['accumulation_phase'] = accumulation_phase
                aggregated['metadata']['suggested_size_multiplier'] = size_multiplier
                logger.info(f"📈 Accumulation {accumulation_phase}: taille {size_multiplier:.0%} pour {symbol}")
                
            elif danger_level >= 5.0:
                # Réduire la taille suggérée en marché dangereux
                size_multiplier = max(0.3, 1.0 - (danger_level / 10.0))
                aggregated['metadata']['suggested_size_multiplier'] = size_multiplier
                aggregated['metadata']['danger_level'] = danger_level
                logger.info(f"📉 Taille de position réduite à {size_multiplier:.1%} pour danger {danger_level:.1f}")
            
            aggregated['metadata']['correlation_score'] = correlation_score
            aggregated['metadata']['enhanced_filtering'] = True
            
            # Tracker pour analyse future
            await self._track_signal_accuracy(aggregated)
        
        return aggregated
    
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