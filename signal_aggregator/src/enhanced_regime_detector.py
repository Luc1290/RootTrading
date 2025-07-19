#!/usr/bin/env python3
import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime
import json
from enum import Enum
from shared.src.technical_indicators import TechnicalIndicators
from db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Énumération des régimes de marché détaillés"""
    STRONG_TREND_UP = "STRONG_TREND_UP"
    TREND_UP = "TREND_UP"
    WEAK_TREND_UP = "WEAK_TREND_UP"
    RANGE_VOLATILE = "RANGE_VOLATILE"
    RANGE_TIGHT = "RANGE_TIGHT"
    WEAK_TREND_DOWN = "WEAK_TREND_DOWN"
    TREND_DOWN = "TREND_DOWN"
    STRONG_TREND_DOWN = "STRONG_TREND_DOWN"
    UNDEFINED = "UNDEFINED"


class EnhancedRegimeDetector:
    """Version améliorée du détecteur de régime avec plus de nuances"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.indicators = TechnicalIndicators()
        self.db_manager = DatabaseManager()
        self.db_initialized = False
        
        # ADX thresholds (Hybrid optimized for crypto volatility)
        from shared.src.config import (ADX_NO_TREND_THRESHOLD, ADX_WEAK_TREND_THRESHOLD, 
                                     ADX_TREND_THRESHOLD, ADX_STRONG_TREND_THRESHOLD)
        self.adx_no_trend = ADX_NO_TREND_THRESHOLD
        self.adx_weak_trend = ADX_WEAK_TREND_THRESHOLD  
        self.adx_trend = ADX_TREND_THRESHOLD
        self.adx_strong_trend = ADX_STRONG_TREND_THRESHOLD
        
        # Volatility thresholds
        self.bb_squeeze_tight = 0.015  # Très serré
        self.bb_squeeze_normal = 0.025  # Normal
        self.bb_expansion = 0.04  # Expansion
        
        # Momentum thresholds
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.momentum_strong = 10  # ROC %
        
        # Volume thresholds
        self.volume_surge_multiplier = 2.0
        self.volume_decline_multiplier = 0.5
        
    def initialize_db_sync(self):
        """Initialise la connexion à la base de données de manière synchrone"""
        # Ne pas essayer d'initialiser de manière synchrone - laisser l'initialisation async se faire
        if not self.db_initialized:
            logger.info("📋 DB sera initialisée lors du premier accès async")
            # Marquer comme "prêt" pour l'initialisation différée
            return True
        return self.db_initialized
    
    async def initialize_db(self):
        """Version async pour compatibilité"""
        if not self.db_initialized:
            self.db_initialized = await self.db_manager.initialize()
            if self.db_initialized:
                logger.info("✅ Enhanced Regime Detector: DB connectée pour données enrichies")
            else:
                logger.warning("⚠️ Enhanced Regime Detector: Fallback vers Redis")
        return self.db_initialized
        
    def get_detailed_regime_sync(self, symbol: str) -> Tuple[MarketRegime, Dict[str, float]]:
        """
        Version synchrone pour obtenir le régime de marché détaillé
        
        Returns:
            Tuple (regime, metrics_dict)
        """
        try:
            # Vérifier le cache d'abord
            cache_key = f"detailed_regime:{symbol}"
            cached = self.redis.get(cache_key)
            
            if cached:
                # Handle both string and dict cases
                if isinstance(cached, str):
                    regime_data = json.loads(cached)
                else:
                    regime_data = cached
                return MarketRegime(regime_data['regime']), regime_data['metrics']
            
            # Calculer si pas en cache (version sync)
            regime, metrics = self._calculate_detailed_regime_sync(symbol)
            
            # Mettre en cache pour 1 minute
            cache_data = {
                'regime': regime.value,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            self.redis.set(cache_key, json.dumps(cache_data), expiration=60)
            
            return regime, metrics
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération régime détaillé sync pour {symbol}: {e}")
            return MarketRegime.UNDEFINED, {}

    def _calculate_detailed_regime_sync(self, symbol: str) -> Tuple[MarketRegime, Dict[str, float]]:
        """Version synchrone du calcul de régime détaillé (fallback vers Redis)"""
        try:
            # Utiliser la méthode Redis sync existante avec des seuils ADX corrigés
            return self._calculate_regime_from_redis_sync(symbol)
        except Exception as e:
            logger.error(f"❌ Erreur calcul régime sync pour {symbol}: {e}")
            return MarketRegime.UNDEFINED, {}

    async def get_detailed_regime(self, symbol: str) -> Tuple[MarketRegime, Dict[str, float]]:
        """
        Obtient le régime de marché détaillé avec les métriques
        
        Returns:
            Tuple (regime, metrics_dict)
        """
        try:
            # Vérifier le cache d'abord
            cache_key = f"detailed_regime:{symbol}"
            cached = self.redis.get(cache_key)
            
            if cached:
                # Handle both string and dict cases
                if isinstance(cached, str):
                    regime_data = json.loads(cached)
                else:
                    regime_data = cached
                return MarketRegime(regime_data['regime']), regime_data['metrics']
            
            # Calculer si pas en cache
            regime, metrics = await self._calculate_detailed_regime(symbol)
            
            # Mettre en cache pour 1 minute
            cache_data = {
                'regime': regime.value,
                'metrics': metrics
            }
            # Gérer les différents types de clients Redis
            try:
                self.redis.set(cache_key, json.dumps(cache_data), ex=60)
            except TypeError:
                # Fallback pour RedisClientPool customisé
                self.redis.set(cache_key, json.dumps(cache_data), expiration=60)
            
            return regime, metrics
            
        except Exception as e:
            logger.error(f"Erreur détection régime pour {symbol}: {e}")
            return MarketRegime.UNDEFINED, {}
    
    async def _calculate_detailed_regime_async(self, symbol: str) -> Tuple[MarketRegime, Dict[str, float]]:
        """Version async pour calculer le régime détaillé avec données enrichies"""
        try:
            # Essayer d'utiliser la DB d'abord
            if await self.initialize_db():
                return await self._calculate_regime_from_enriched_data_async(symbol)
            else:
                # Fallback vers Redis si DB non disponible
                return self._calculate_regime_from_redis_sync(symbol)
        except Exception as e:
            logger.error(f"❌ Erreur calcul régime détaillé pour {symbol}: {e}")
            return MarketRegime.UNDEFINED, {}
    
    async def _calculate_regime_from_enriched_data_async(self, symbol: str) -> Tuple[MarketRegime, Dict[str, float]]:
        """Calcule le régime en utilisant les données pré-enrichies de la DB (version async)"""
        try:
            # Récupérer les données enrichies de manière asynchrone
            candles = await self.db_manager.get_enriched_market_data(
                symbol=symbol,
                interval="5m",  # Standard crypto pour détection de régime
                limit=500,
                include_indicators=True
            )
            
            if not candles or len(candles) < 20:
                logger.warning(f"⚠️ Pas assez de données enrichies pour {symbol}, fallback Redis")
                return self._calculate_regime_from_redis_sync(symbol)
            
            # Extraire les indicateurs pré-calculés
            latest = candles[-1]
            
            # Vérifier la disponibilité des indicateurs essentiels
            if not all(key in latest for key in ['rsi_14', 'bb_width', 'atr_14']):
                logger.warning(f"⚠️ Indicateurs manquants dans DB pour {symbol}, fallback Redis")
                return self._calculate_regime_from_redis_sync(symbol)
            
            # Extraire les valeurs des indicateurs
            current_rsi = latest.get('rsi_14', 50)
            current_bb_width = latest.get('bb_width', 0.02)
            current_atr = latest.get('atr_14', 0)
            current_close = latest.get('close', 0)
            volume_ratio = latest.get('volume_ratio', 1.0)
            
            # Calculs additionnels si ADX disponible dans DB
            current_adx = latest.get('adx_14')
            if current_adx is None:
                # Calculer ADX si pas disponible - besoin de plus de données
                required_length = 30  # ADX nécessite au moins 2*14 périodes
                if len(candles) >= required_length:
                    close_prices = [c['close'] for c in candles[-required_length:]]
                    high_prices = [c['high'] for c in candles[-required_length:]]
                    low_prices = [c['low'] for c in candles[-required_length:]]
                    current_adx, _, _ = self.indicators.calculate_adx_smoothed(high_prices, low_prices, close_prices, 14)
                
                if current_adx is None:
                    current_adx = 20  # Valeur par défaut
                    logger.debug(f"ADX calculation failed for {symbol}, using default value 20")
            
            # Détection du régime avec données enrichies
            regime_score = 0.0
            confidence = 0.0
            
            # 1. ADX - Force de tendance (poids principal)
            if current_adx > self.adx_strong_trend:  # > 35
                regime_score += 3.0
                confidence += 0.4
            elif current_adx > self.adx_trend:  # > 25
                regime_score += 2.0
                confidence += 0.3
            elif current_adx > self.adx_weak_trend:  # > 20
                regime_score += 1.0
                confidence += 0.2
            elif current_adx >= self.adx_no_trend:  # >= 15 (FIX: cas 15-20 traité)
                regime_score += 0.5  # Tendance très faible mais présente
                confidence += 0.15
            else:  # < 15
                regime_score -= 1.0  # Pas de tendance claire
                confidence += 0.1
            
            # 2. Bollinger Bands - Volatilité et compression
            if current_bb_width < self.bb_squeeze_tight:
                regime_score -= 2.0  # Compression très serrée = range
                confidence += 0.3
            elif current_bb_width < self.bb_squeeze_normal:
                regime_score -= 1.0  # Compression normale = range modéré
                confidence += 0.2
            elif current_bb_width > self.bb_expansion:
                regime_score += 1.5  # Expansion = tendance
                confidence += 0.25
            
            # 3. Volume - Confirmation de mouvement
            if volume_ratio > self.volume_surge_multiplier:
                regime_score += 1.0  # Volume élevé = mouvement significatif
                confidence += 0.15
            elif volume_ratio < self.volume_decline_multiplier:
                regime_score -= 0.5  # Volume faible = range possible
                confidence += 0.1
            
            # 4. ATR/Price ratio - Volatilité normalisée
            if current_close > 0:
                atr_ratio = current_atr / current_close
                if atr_ratio > 0.03:  # 3% volatilité = tendance
                    regime_score += 1.0
                    confidence += 0.15
                elif atr_ratio < 0.01:  # < 1% volatilité = range
                    regime_score -= 0.5
                    confidence += 0.1
            
            # 5. RSI - Momentum et retournements
            if 40 <= current_rsi <= 60:
                # RSI neutre = incertitude ou consolidation
                regime_score -= 0.5
                confidence += 0.1
            elif current_rsi < 30 or current_rsi > 70:
                # RSI extrême = possibilité de retournement
                regime_score += 0.5  # Peut être début de tendance
                confidence += 0.15
            
            # Détermination du régime final - STANDARDISÉ avec FIX ADX 15-20
            if regime_score >= 3.0 and confidence >= 0.7:
                if current_adx > self.adx_strong_trend:
                    regime = MarketRegime.STRONG_TREND_UP if latest.get('bb_position', 0.5) >= 0.55 else MarketRegime.STRONG_TREND_DOWN
                else:
                    regime = MarketRegime.TREND_UP if latest.get('bb_position', 0.5) >= 0.55 else MarketRegime.TREND_DOWN
            elif regime_score >= 1.5:
                regime = MarketRegime.WEAK_TREND_UP if latest.get('bb_position', 0.5) >= 0.55 else MarketRegime.WEAK_TREND_DOWN
            elif regime_score >= 0.3:  # FIX: ajout cas ADX 15-20 (score ~0.5)
                # Tendance très faible détectée (ADX 15-20)
                regime = MarketRegime.WEAK_TREND_UP if latest.get('bb_position', 0.5) >= 0.55 else MarketRegime.WEAK_TREND_DOWN
            elif regime_score <= -1.5:
                if current_bb_width < self.bb_squeeze_tight:
                    regime = MarketRegime.RANGE_TIGHT
                else:
                    regime = MarketRegime.RANGE_VOLATILE
            else:
                # Score neutre (-1.5 < score < 0.3) : analyser selon BB
                if current_bb_width < self.bb_squeeze_tight:
                    regime = MarketRegime.RANGE_TIGHT
                else:
                    regime = MarketRegime.RANGE_VOLATILE
            
            # Métriques détaillées
            metrics = {
                'adx': current_adx,
                'rsi': current_rsi,
                'bb_width': current_bb_width,
                'bb_position': latest.get('bb_position', 0.5),
                'atr': current_atr,
                'volume_ratio': volume_ratio,
                'regime_score': regime_score,
                'confidence': confidence,
                'data_source': 'enriched_db'
            }
            
            logger.info(f"✅ Régime calculé depuis DB enrichie pour {symbol}: {regime.value} "
                       f"(score={regime_score:.1f}, conf={confidence:.1f}, ADX={current_adx:.1f})")
            
            return regime, metrics
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul régime depuis DB enrichie pour {symbol}: {e}")
            return self._calculate_regime_from_redis_sync(symbol)
    
    def _calculate_regime_from_redis_sync(self, symbol: str) -> Tuple[MarketRegime, Dict[str, float]]:
        """Fallback vers Redis pour calculer le régime (version sync)"""
        try:
            # Récupérer les données depuis Redis
            key = f"market_data:{symbol}:15m"
            market_data = self.redis.get(key)
            
            if not market_data:
                logger.warning(f"⚠️ Aucune donnée Redis trouvée pour {symbol}")
                return MarketRegime.UNDEFINED, {}
            
            # Parser les données
            if isinstance(market_data, str):
                data = json.loads(market_data)
            else:
                data = market_data
            
            # Utiliser les indicateurs Redis s'ils existent
            current_rsi = data.get('rsi', 50)
            current_adx = data.get('adx', 25)
            current_bb_width = data.get('bb_width', 0.025)
            
            # Calcul du régime avec seuils ADX corrects - FIX: ADX 17.1 ne doit plus être UNDEFINED
            if current_adx >= self.adx_strong_trend:  # >= 35
                regime = MarketRegime.STRONG_TREND_UP if current_rsi > 50 else MarketRegime.STRONG_TREND_DOWN
            elif current_adx >= self.adx_trend:  # >= 25
                regime = MarketRegime.TREND_UP if current_rsi > 50 else MarketRegime.TREND_DOWN
            elif current_adx >= self.adx_weak_trend:  # >= 20
                regime = MarketRegime.WEAK_TREND_UP if current_rsi > 50 else MarketRegime.WEAK_TREND_DOWN
            elif current_adx >= self.adx_no_trend:  # >= 15 (FIX: cas 15-20 traité)
                # Tendance très faible - utiliser d'autres indicateurs
                if current_bb_width < 0.015:
                    regime = MarketRegime.RANGE_TIGHT
                else:
                    regime = MarketRegime.WEAK_TREND_UP if current_rsi > 50 else MarketRegime.WEAK_TREND_DOWN
            elif current_bb_width < 0.015:
                regime = MarketRegime.RANGE_TIGHT
            else:
                regime = MarketRegime.UNDEFINED
            
            metrics = {
                'adx': float(current_adx),
                'rsi': float(current_rsi), 
                'bb_width': float(current_bb_width)
            }
            
            logger.info(f"📊 Régime calculé depuis Redis pour {symbol}: {regime.value}")
            return regime, metrics
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul régime depuis Redis pour {symbol}: {e}")
            return MarketRegime.UNDEFINED, {}

    async def _calculate_detailed_regime(self, symbol: str) -> Tuple[MarketRegime, Dict[str, float]]:
        """Calcul détaillé du régime avec multiples indicateurs"""
        try:
            # Utiliser la nouvelle logique unifiée
            return await self._calculate_detailed_regime_async(symbol)
        except Exception as e:
            logger.error(f"❌ Erreur calcul régime détaillé pour {symbol}: {e}")
            # Fallback vers l'ancienne méthode Redis
            candles = await self._get_recent_candles(symbol, limit=100)
            
            if not candles or len(candles) < 50:
                return MarketRegime.UNDEFINED, {}
            
            df = pd.DataFrame(candles)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Convertir en listes pour les indicateurs
            highs = df['high'].values.tolist()
            lows = df['low'].values.tolist()
            closes = df['close'].values.tolist()
            
            # 1. ADX pour la force de tendance (avec lissage)
            current_adx, plus_di, minus_di = self.indicators.calculate_adx_smoothed(highs, lows, closes, 14)
            
            # Vérifier si les valeurs sont valides
            if current_adx is None or plus_di is None or minus_di is None:
                logger.warning(f"ADX/DI non valides pour {symbol}: ADX={current_adx}, +DI={plus_di}, -DI={minus_di}")
                return MarketRegime.UNDEFINED, {}
            
            # 2. Bollinger Bands pour la volatilité
            bb_data = self.indicators.calculate_bollinger_bands(closes, 20, 2.0)
            if bb_data['bb_upper'] is None or bb_data['bb_lower'] is None or bb_data['bb_middle'] is None:
                logger.warning(f"Bollinger Bands non valides pour {symbol}")
                return MarketRegime.UNDEFINED, {}
                
            # Calculer la largeur des bandes et la position
            current_bb_width = bb_data['bb_width']
            bb_position = bb_data['bb_position']
            
            # 3. RSI pour le momentum
            current_rsi = self.indicators.calculate_rsi(closes, 14)
            if current_rsi is None:
                logger.warning(f"RSI non valide pour {symbol}")
                return MarketRegime.UNDEFINED, {}
            
            # 4. ROC (Rate of Change) pour le momentum directionnel
            current_roc = self.indicators.calculate_roc(closes, 10)
            if current_roc is None:
                current_roc = 0.0
            
            # 5. Volume analysis
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # 6. Trend angle et pivot count depuis la DB (si disponibles), sinon calcul local
            if 'trend_angle' in df.columns and not pd.isna(df['trend_angle'].iloc[-1]):
                trend_angle = df['trend_angle'].iloc[-1]
            else:
                # Fallback: calcul local
                prices = df['close'].iloc[-20:].values
                x = np.arange(len(prices))
                slope, _ = np.polyfit(x, prices, 1)
                trend_angle = np.degrees(np.arctan(slope / prices.mean() * 100))
                
            if 'pivot_count' in df.columns and not pd.isna(df['pivot_count'].iloc[-1]):
                pivot_count = df['pivot_count'].iloc[-1]
            else:
                # Fallback: calcul local
                pivot_high_count = self._count_pivots_local(df['high'].values[-50:], is_high=True)
                pivot_low_count = self._count_pivots_local(df['low'].values[-50:], is_low=True)
                pivot_count = pivot_high_count + pivot_low_count
            
            # Calculs enrichis avec les bandes Bollinger (méthode raw data)
            current_close = closes[-1]
            bb_upper = bb_data['bb_upper']
            bb_lower = bb_data['bb_lower'] 
            bb_middle = bb_data['bb_middle']
            
            bb_distance_to_upper = abs(current_close - bb_upper) / bb_upper if bb_upper else 0
            bb_distance_to_lower = abs(current_close - bb_lower) / bb_lower if bb_lower else 0
            bb_squeeze_strength = 1 - current_bb_width if current_bb_width else 0
            price_vs_middle = (current_close - bb_middle) / bb_middle if bb_middle else 0
            
            # Validation des breakouts/breakdowns avec volume
            bb_breakout_confirmed = False
            bb_breakdown_confirmed = False
            if bb_upper and bb_lower and volume_ratio > 1.5:
                if current_close > bb_upper:
                    bb_breakout_confirmed = True
                elif current_close < bb_lower:
                    bb_breakdown_confirmed = True
            
            # Compiler les métriques enrichies
            metrics = {
                'adx': current_adx,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'bb_width': current_bb_width,
                'bb_position': bb_position,
                'bb_distance_to_upper': bb_distance_to_upper,
                'bb_distance_to_lower': bb_distance_to_lower,
                'bb_squeeze_strength': bb_squeeze_strength,
                'price_vs_middle': price_vs_middle,
                'bb_breakout_confirmed': bb_breakout_confirmed,
                'bb_breakdown_confirmed': bb_breakdown_confirmed,
                'rsi': current_rsi,
                'roc': current_roc,
                'volume_ratio': volume_ratio,
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'trend_angle': trend_angle,
                'pivot_count': pivot_count
            }
            
            # Déterminer le régime
            regime = self._determine_regime(metrics)
            
            logger.info(f"Régime {symbol} (Raw data): {regime.value} | ADX={current_adx:.1f}, "
                       f"+DI={plus_di:.1f}, -DI={minus_di:.1f}, ROC={current_roc:.1f}%, "
                       f"RSI={current_rsi:.1f}, BB_pos={bb_position:.2f}, Vol={volume_ratio:.1f}x, "
                       f"Squeeze={bb_squeeze_strength:.2f}, Breakout={'✅' if bb_breakout_confirmed else '❌'}")
            
            return regime, metrics
            
        except Exception as e:
            logger.error(f"Erreur calcul régime détaillé: {e}")
            return MarketRegime.UNDEFINED, {}
    
    def _determine_regime(self, metrics: Dict[str, float]) -> MarketRegime:
        """Détermine le régime avec priorité au 'choppiness' sur l'ADX"""
        adx = metrics['adx']
        plus_di = metrics['plus_di']
        minus_di = metrics['minus_di']
        bb_width = metrics['bb_width']
        rsi = metrics['rsi']
        roc = metrics['roc']
        trend_angle = metrics['trend_angle']
        pivot_count = metrics.get('pivot_count', 0)
        
        # NOUVEAU: Calculer l'indice de "choppiness"
        # Plus il y a de pivots, plus le marché est erratique
        choppiness_threshold = 15  # Ajustable selon le timeframe
        high_choppiness_threshold = 25
        
        # Facteurs de choppiness
        is_choppy = pivot_count > choppiness_threshold
        is_very_choppy = pivot_count > high_choppiness_threshold
        
        # Volatilité sans direction claire
        volatility_without_direction = (
            bb_width > self.bb_expansion and  # Bandes larges
            abs(roc) < 5 and  # Faible momentum directionnel
            abs(trend_angle) < 10  # Angle de tendance plat
        )
        
        # NOUVELLE LOGIQUE : Vérifier d'abord le "choppiness"
        if is_very_choppy or (is_choppy and volatility_without_direction):
            logger.info(f"🌊 Marché CHOPPY détecté: pivots={pivot_count}, ADX={adx:.1f} (ignoré)")
            # Classifier selon la volatilité, PAS selon l'ADX
            if bb_width > self.bb_expansion * 1.5:
                return MarketRegime.RANGE_VOLATILE
            elif bb_width < self.bb_squeeze_tight:
                return MarketRegime.RANGE_TIGHT
            else:
                return MarketRegime.RANGE_VOLATILE
        
        # Si le marché n'est PAS choppy, alors on peut faire confiance à l'ADX
        
        # Direction consensus (comme avant)
        di_bullish = plus_di > minus_di
        roc_bullish = roc > 0
        angle_bullish = trend_angle > 0
        bullish_count = sum([di_bullish, roc_bullish, angle_bullish])
        is_bullish = bullish_count >= 2
        
        # NOUVEAU: Exiger une convergence d'indicateurs pour valider une tendance
        # même si ADX > 30
        trend_confirmation = 0
        
        # Confirmation 1: Direction cohérente
        if (is_bullish and roc > 2 and trend_angle > 5) or \
           (not is_bullish and roc < -2 and trend_angle < -5):
            trend_confirmation += 1
        
        # Confirmation 2: DI dominant
        di_spread = abs(plus_di - minus_di)
        if di_spread > 10:  # Un DI domine clairement
            trend_confirmation += 1
        
        # Confirmation 3: RSI cohérent
        if (is_bullish and rsi > 50) or (not is_bullish and rsi < 50):
            trend_confirmation += 1
        
        # Log détaillé pour debug
        if adx > self.adx_trend:
            logger.info(f"Direction consensus: +DI={plus_di:.1f} vs -DI={minus_di:.1f} ({di_bullish}), "
                       f"ROC={roc:.1f}% ({roc_bullish}), Angle={trend_angle:.1f}° ({angle_bullish}) "
                       f"=> Bullish={is_bullish} (score: {bullish_count}/3, confirmations: {trend_confirmation}/3)")
        
        # ADX élevé MAIS avec confirmations requises
        if adx >= self.adx_trend:
            # Exiger au moins 2 confirmations pour valider la tendance
            if trend_confirmation >= 2:
                if adx >= self.adx_strong_trend:
                    if is_bullish and roc > self.momentum_strong:
                        return MarketRegime.STRONG_TREND_UP
                    elif not is_bullish and roc < -self.momentum_strong:
                        return MarketRegime.STRONG_TREND_DOWN
                    elif is_bullish:
                        return MarketRegime.TREND_UP
                    else:
                        return MarketRegime.TREND_DOWN
                elif adx >= self.adx_trend:
                    if is_bullish:
                        return MarketRegime.TREND_UP
                    else:
                        return MarketRegime.TREND_DOWN
                else:  # ADX entre 30 et adx_trend
                    if is_bullish:
                        return MarketRegime.WEAK_TREND_UP
                    else:
                        return MarketRegime.WEAK_TREND_DOWN
            else:
                # ADX élevé mais pas assez de confirmations = fausse tendance
                logger.info(f"⚠️ ADX élevé ({adx:.1f}) mais seulement {trend_confirmation}/3 confirmations => RANGE")
                if bb_width > self.bb_expansion:
                    return MarketRegime.RANGE_VOLATILE
                else:
                    return MarketRegime.RANGE_TIGHT
        
        # ADX modéré à faible : analyser selon les seuils standards
        if adx >= self.adx_weak_trend:  # >= 20
            if trend_angle > 5 or roc > 5:
                return MarketRegime.WEAK_TREND_UP
            elif trend_angle < -5 or roc < -5:
                return MarketRegime.WEAK_TREND_DOWN
        elif adx >= self.adx_no_trend:  # >= 15 (FIX: cas 15-20 traité)
            # ADX entre 15-20 : tendance très faible mais présente
            if trend_angle > 3 or roc > 3:  # Seuils plus bas pour tendance faible
                return MarketRegime.WEAK_TREND_UP
            elif trend_angle < -3 or roc < -3:
                return MarketRegime.WEAK_TREND_DOWN
            # Pas de tendance claire : utiliser BB
            elif bb_width < self.bb_squeeze_tight:
                return MarketRegime.RANGE_TIGHT
            else:
                return MarketRegime.RANGE_VOLATILE
        
        # ADX < 15 : marché en range
        if bb_width < self.bb_squeeze_tight:
            return MarketRegime.RANGE_TIGHT
        elif bb_width > self.bb_expansion:
            return MarketRegime.RANGE_VOLATILE
        else:
            if trend_angle > 5:
                return MarketRegime.WEAK_TREND_UP
            elif trend_angle < -5:
                return MarketRegime.WEAK_TREND_DOWN
            else:
                return MarketRegime.RANGE_TIGHT
    
    def _count_pivots_local(self, data: np.ndarray, window: int = 3, 
                           is_high: bool = False, is_low: bool = False) -> int:
        """Compte le nombre de pivots (hauts/bas locaux) - fallback local"""
        pivot_count = 0
        
        for i in range(window, len(data) - window):
            if is_high:
                if all(data[i] > data[i-j] for j in range(1, window+1)) and \
                   all(data[i] > data[i+j] for j in range(1, window+1)):
                    pivot_count += 1
            elif is_low:
                if all(data[i] < data[i-j] for j in range(1, window+1)) and \
                   all(data[i] < data[i+j] for j in range(1, window+1)):
                    pivot_count += 1
                    
        return pivot_count
    
    def get_strategy_weights_for_regime(self, regime: MarketRegime) -> Dict[str, float]:
        """
        Retourne les poids optimaux des stratégies pour un régime donné
        
        Returns:
            Dict {strategy_name: weight_multiplier}
        """
        weights = {
            MarketRegime.STRONG_TREND_UP: {
                'EMA_Cross': 1.5,
                'MACD': 1.3,
                'Breakout': 1.8,
                'Bollinger': 0.5,
                'RSI': 0.3,
                'Divergence': 0.4,
                'Ride_or_React': 2.0
            },
            MarketRegime.TREND_UP: {
                'EMA_Cross': 1.3,
                'MACD': 1.2,
                'Breakout': 1.4,
                'Bollinger': 0.7,
                'RSI': 0.6,
                'Divergence': 0.6,
                'Ride_or_React': 1.5
            },
            MarketRegime.WEAK_TREND_UP: {
                'EMA_Cross': 1.0,
                'MACD': 1.0,
                'Breakout': 0.9,
                'Bollinger': 1.0,
                'RSI': 0.9,
                'Divergence': 1.1,
                'Ride_or_React': 1.0
            },
            MarketRegime.RANGE_TIGHT: {
                'EMA_Cross': 0.5,
                'MACD': 0.6,
                'Breakout': 0.4,
                'Bollinger': 1.8,
                'RSI': 1.7,
                'Divergence': 1.3,
                'Ride_or_React': 0.8
            },
            MarketRegime.RANGE_VOLATILE: {
                'EMA_Cross': 0.7,
                'MACD': 0.8,
                'Breakout': 1.2,
                'Bollinger': 1.4,
                'RSI': 1.3,
                'Divergence': 1.2,
                'Ride_or_React': 1.0
            },
            # Régimes baissiers (inverser certains poids)
            MarketRegime.WEAK_TREND_DOWN: {
                'EMA_Cross': 1.0,
                'MACD': 1.0,
                'Breakout': 0.9,
                'Bollinger': 1.0,
                'RSI': 0.9,
                'Divergence': 1.1,
                'Ride_or_React': 1.0
            },
            MarketRegime.TREND_DOWN: {
                'EMA_Cross': 1.3,
                'MACD': 1.2,
                'Breakout': 1.4,
                'Bollinger': 0.7,
                'RSI': 0.6,
                'Divergence': 0.6,
                'Ride_or_React': 1.5
            },
            MarketRegime.STRONG_TREND_DOWN: {
                'EMA_Cross': 1.5,
                'MACD': 1.3,
                'Breakout': 1.8,
                'Bollinger': 0.5,
                'RSI': 0.3,
                'Divergence': 0.4,
                'Ride_or_React': 2.0
            }
        }
        
        return weights.get(regime, {strategy: 1.0 for strategy in [
            'EMA_Cross', 'MACD', 'Breakout', 'Bollinger', 
            'RSI', 'Divergence', 'Ride_or_React'
        ]})
    
    async def get_danger_level(self, symbol: str) -> float:
        """
        Calculate market risk/opportunity level from 0 to 10 based on Enhanced metrics
        0-2: Excellent opportunity (strong trends, low volatility)
        3-4: Good opportunity (weak trends, moderate conditions)
        5-6: Neutral (range markets, balanced risk/opportunity)  
        7-8: Risky (high volatility, uncertain direction)
        9-10: Very dangerous (extreme conditions, avoid new positions)
        """
        try:
            # Get cached danger level first
            cache_key = f"enhanced_danger:{symbol}"
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    return float(cached)
            except Exception:
                pass  # Continue without cache if Redis fails
                
            # Calculate danger level based on current regime and metrics
            regime, metrics = await self.get_detailed_regime(symbol)
            danger_level = self._calculate_danger_from_regime(regime, metrics)
            
            # Cache for 1 minute
            try:
                await self.redis.set(cache_key, str(danger_level), expiration=60)
            except Exception:
                pass  # Continue without cache if Redis fails
            
            return danger_level
            
        except Exception as e:
            logger.error(f"Error getting danger level for {symbol}: {e}")
            return 5.0  # Default to medium danger
    
    def _calculate_danger_from_regime(self, regime: MarketRegime, metrics: Dict[str, float]) -> float:
        """Calculate danger level based on Enhanced regime and metrics"""
        
        # Base danger level by regime type
        regime_danger = {
            MarketRegime.STRONG_TREND_UP: 1.0,      # Lowest danger - strong uptrend
            MarketRegime.TREND_UP: 2.0,             # Low danger - clear uptrend  
            MarketRegime.WEAK_TREND_UP: 3.5,        # Moderate danger - weak trend
            MarketRegime.RANGE_TIGHT: 4.0,          # Neutral - tight range
            MarketRegime.RANGE_VOLATILE: 6.5,       # Higher danger - volatile range
            MarketRegime.WEAK_TREND_DOWN: 7.0,      # Risky - weak downtrend
            MarketRegime.TREND_DOWN: 8.0,           # High danger - clear downtrend
            MarketRegime.STRONG_TREND_DOWN: 9.0     # Very dangerous - strong downtrend
        }
        
        base_danger = regime_danger.get(regime, 5.0)
        
        # Adjust based on volatility (BB width)
        bb_width = metrics.get('bb_width_pct', 2.0)
        if bb_width > 5.0:  # Very high volatility
            base_danger += 1.5
        elif bb_width > 3.0:  # High volatility  
            base_danger += 1.0
        elif bb_width < 1.0:  # Very low volatility
            base_danger -= 0.5
            
        # Adjust based on RSI extremes
        rsi = metrics.get('rsi', 50)
        if rsi < 20 or rsi > 80:  # Extreme oversold/overbought
            base_danger += 1.0
        elif rsi < 30 or rsi > 70:  # Moderately oversold/overbought
            base_danger += 0.5
            
        # Adjust based on trend strength (ADX)
        adx = metrics.get('adx', 25)
        # Import des seuils standardisés
        from shared.src.config import ADX_STRONG_TREND_THRESHOLD, ADX_NO_TREND_THRESHOLD
        
        if adx > ADX_STRONG_TREND_THRESHOLD:  # Very strong trend - reduce danger if trend up
            if regime in [MarketRegime.STRONG_TREND_UP, MarketRegime.TREND_UP]:
                base_danger -= 0.5
            else:  # Strong downtrend - increase danger
                base_danger += 0.5
        elif adx < ADX_NO_TREND_THRESHOLD:  # Very weak trend - increase danger (choppy)
            base_danger += 1.0
            
        # Cap between 0 and 10
        return max(0.0, min(10.0, base_danger))

    async def is_in_recovery(self, symbol: str) -> bool:
        """Check if symbol is in recovery period after danger"""
        try:
            recovery_key = f"recovery_period:{symbol}"
            return bool(await self.redis.get(recovery_key))
        except Exception:
            return False
        
    async def is_opportunity_period(self, symbol: str) -> bool:
        """Check if symbol is in excellent opportunity period"""
        try:
            opportunity_key = f"opportunity_period:{symbol}"
            return bool(await self.redis.get(opportunity_key))
        except Exception:
            return False

    def set_market_data_accumulator(self, accumulator) -> None:
        """Définit l'accumulateur de données de marché"""
        self.market_data_accumulator = accumulator

    async def _get_enriched_candles_from_db(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Récupère les données enrichies depuis la base de données"""
        try:
            if not self.db_manager or not self.db_initialized:
                return []
            
            # Récupérer les données enrichies avec tous les indicateurs
            candles = await self.db_manager.get_enriched_market_data(
                symbol=symbol,
                interval="5m",  # Standardisé sur 5m pour cohérence système
                limit=limit,
                include_indicators=True
            )
            
            if not candles:
                logger.debug(f"Aucune donnée enrichie trouvée pour {symbol}")
                return []
            
            # Vérifier que les données sont récentes (moins de 5 minutes)
            is_fresh = await self.db_manager.check_data_freshness(symbol, max_age_minutes=5)
            if not is_fresh:
                logger.warning(f"⚠️ Données enrichies {symbol} trop anciennes, utiliser Redis")
                return []
            
            logger.info(f"📊 Données enrichies DB récupérées pour {symbol}: {len(candles)} points")
            return candles
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération données enrichies {symbol}: {e}")
            return []
    
    async def _calculate_regime_from_enriched_data(self, symbol: str, candles: List[Dict]) -> Tuple[MarketRegime, Dict[str, float]]:
        """Calcule le régime en utilisant les données pré-enrichies de la DB"""
        try:
            if not candles or len(candles) < 50:
                return MarketRegime.UNDEFINED, {}
            
            # Utiliser les indicateurs pré-calculés au lieu de les recalculer
            latest_candle = candles[-1]
            
            # Extraire les indicateurs pré-calculés
            current_rsi = latest_candle.get('rsi_14')
            bb_upper = latest_candle.get('bb_upper')
            bb_lower = latest_candle.get('bb_lower')
            bb_middle = latest_candle.get('bb_middle')
            bb_width = latest_candle.get('bb_width')
            bb_position = latest_candle.get('bb_position')
            current_close = latest_candle['close']
            
            # Calculer les indicateurs manquants (ADX, ROC) avec le module partagé
            # Extraire les séries de prix pour les calculs manquants
            highs = [c['high'] for c in candles]
            lows = [c['low'] for c in candles]
            closes = [c['close'] for c in candles]
            volumes = [c['volume'] for c in candles]
            
            # ADX et DI (avec lissage - maintenant sauvegardé en DB)
            current_adx, plus_di, minus_di = self.indicators.calculate_adx_smoothed(highs, lows, closes, 14)
            
            # ROC (maintenant sauvegardé en DB)
            current_roc = self.indicators.calculate_roc(closes, 10)
            if current_roc is None:
                current_roc = 0.0
            
            # Vérifier que nous avons les indicateurs essentiels
            if current_rsi is None or bb_width is None or current_adx is None:
                logger.warning(f"⚠️ Indicateurs manquants pour {symbol}, fallback calcul complet")
                # Fallback vers calcul complet
                return await self._calculate_regime_from_raw_data(candles, symbol)
            
            # Volume analysis
            avg_volume = sum(volumes[-20:]) / min(20, len(volumes)) if volumes else 1.0
            current_volume = volumes[-1] if volumes else 1.0
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Trend angle et pivot count depuis les données enrichies (si disponibles)
            # Fallback: calcul local si pas disponible
            trend_angle = 0.0
            pivot_count = 0
            if candles and len(candles) > 0:
                last_candle = candles[-1]
                trend_angle = last_candle.get('trend_angle', 0.0)
                pivot_count = last_candle.get('pivot_count', 0)
            
            # Calculs enrichis avec les bandes Bollinger
            bb_distance_to_upper = abs(current_close - bb_upper) / bb_upper if bb_upper else 0
            bb_distance_to_lower = abs(current_close - bb_lower) / bb_lower if bb_lower else 0
            bb_squeeze_strength = 1 - bb_width if bb_width else 0  # Plus proche de 1 = squeeze plus fort
            price_vs_middle = (current_close - bb_middle) / bb_middle if bb_middle else 0
            
            # Validation des breakouts/breakdowns Bollinger
            bb_breakout_confirmed = False
            bb_breakdown_confirmed = False
            if bb_upper and bb_lower and volume_ratio > 1.5:  # Volume confirme le mouvement
                if current_close > bb_upper:
                    bb_breakout_confirmed = True
                elif current_close < bb_lower:
                    bb_breakdown_confirmed = True
            
            # Compiler les métriques enrichies
            metrics = {
                'adx': current_adx,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'bb_width': bb_width,
                'bb_position': bb_position,
                'bb_distance_to_upper': bb_distance_to_upper,
                'bb_distance_to_lower': bb_distance_to_lower,
                'bb_squeeze_strength': bb_squeeze_strength,
                'price_vs_middle': price_vs_middle,
                'bb_breakout_confirmed': bb_breakout_confirmed,
                'bb_breakdown_confirmed': bb_breakdown_confirmed,
                'rsi': current_rsi,
                'roc': current_roc,
                'volume_ratio': volume_ratio,
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'trend_angle': trend_angle,
                'pivot_count': pivot_count
            }
            
            # Déterminer le régime
            regime = self._determine_regime(metrics)
            
            logger.info(f"📊 Régime {symbol} (DB enrichie): {regime.value} | ADX={current_adx:.1f}, "
                       f"+DI={plus_di:.1f}, -DI={minus_di:.1f}, ROC={current_roc:.1f}%, "
                       f"RSI={current_rsi:.1f}, BB_pos={bb_position:.2f}, Vol={volume_ratio:.1f}x, "
                       f"Squeeze={bb_squeeze_strength:.2f}, Breakout={'✅' if bb_breakout_confirmed else '❌'}")
            
            return regime, metrics
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul régime depuis données enrichies: {e}")
            return MarketRegime.UNDEFINED, {}
    
    async def _calculate_regime_from_raw_data(self, candles: List[Dict], symbol: str) -> Tuple[MarketRegime, Dict[str, float]]:
        """Fallback: calcule le régime depuis les données brutes (ancienne méthode)"""
        try:
            df = pd.DataFrame(candles)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Convertir en listes pour les indicateurs
            highs = df['high'].values.tolist()
            lows = df['low'].values.tolist()
            closes = df['close'].values.tolist()
            
            # Calculs complets avec le module partagé (avec lissage ADX)
            current_adx, plus_di, minus_di = self.indicators.calculate_adx_smoothed(highs, lows, closes, 14)
            
            if current_adx is None or plus_di is None or minus_di is None:
                logger.warning("ADX/DI non valides fallback")
                return MarketRegime.UNDEFINED, {}
            
            bb_data = self.indicators.calculate_bollinger_bands(closes, 20, 2.0)
            if bb_data['bb_upper'] is None:
                logger.warning("Bollinger Bands non valides fallback")
                return MarketRegime.UNDEFINED, {}
                
            current_bb_width = bb_data['bb_width']
            bb_position = bb_data['bb_position']
            
            current_rsi = self.indicators.calculate_rsi(closes, 14)
            if current_rsi is None:
                logger.warning("RSI non valide fallback")
                return MarketRegime.UNDEFINED, {}
            
            current_roc = self.indicators.calculate_roc(closes, 10)
            if current_roc is None:
                current_roc = 0.0
            
            # Volume analysis
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Trend angle et pivot count depuis la DB (si disponibles), sinon calcul local
            if 'trend_angle' in df.columns and not pd.isna(df['trend_angle'].iloc[-1]):
                trend_angle = df['trend_angle'].iloc[-1]
            else:
                # Fallback: calcul local
                prices = df['close'].iloc[-20:].values
                x = np.arange(len(prices))
                slope, _ = np.polyfit(x, prices, 1)
                trend_angle = np.degrees(np.arctan(slope / prices.mean() * 100))
                
            if 'pivot_count' in df.columns and not pd.isna(df['pivot_count'].iloc[-1]):
                pivot_count = df['pivot_count'].iloc[-1]
            else:
                # Fallback: calcul local
                pivot_high_count = self._count_pivots_local(df['high'].values[-50:], is_high=True)
                pivot_low_count = self._count_pivots_local(df['low'].values[-50:], is_low=True)
                pivot_count = pivot_high_count + pivot_low_count
            
            metrics = {
                'adx': current_adx,
                'plus_di': plus_di,
                'minus_di': minus_di,
                'bb_width': current_bb_width,
                'bb_position': bb_position,
                'rsi': current_rsi,
                'roc': current_roc,
                'volume_ratio': volume_ratio,
                'trend_angle': trend_angle,
                'pivot_count': pivot_count
            }
            
            regime = self._determine_regime(metrics)
            
            logger.info(f"📊 Régime {symbol} (fallback): {regime.value} | ADX={current_adx:.1f}, "
                       f"+DI={plus_di:.1f}, -DI={minus_di:.1f}, ROC={current_roc:.1f}%, "
                       f"RSI={current_rsi:.1f}, Angle={trend_angle:.1f}°")
            
            return regime, metrics
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul régime fallback: {e}")
            return MarketRegime.UNDEFINED, {}

    async def _get_recent_candles(self, symbol: str, limit: int = 100) -> list:
        """Récupère les données de marché historiques depuis l'accumulateur"""
        try:
            # Utiliser l'accumulateur si disponible
            if hasattr(self, 'market_data_accumulator') and self.market_data_accumulator:
                history = self.market_data_accumulator.get_history(symbol, limit)
                if len(history) >= 50:  # Minimum 50 points pour calculer ADX/DI correctement
                    # Vérifier que nous avons de vraies données OHLCV
                    sample = history[-1] if history else {}
                    if all(key in sample for key in ['open', 'high', 'low', 'close']) and \
                       sample.get('high') != sample.get('low'):  # Vérifier que high != low
                        logger.info(f"📊 Utilisation historique accumulé pour {symbol}: {len(history)} points")
                        return history
                    else:
                        logger.warning(f"⚠️ Données OHLCV incomplètes dans l'historique pour {symbol}")
                else:
                    logger.warning(f"⚠️ Historique insuffisant pour {symbol}: {len(history)} points (min: 50)")
            
            # Fallback : récupérer les données enrichies actuelles de différentes timeframes
            timeframes = ['1m', '3m', '5m', '15m']
            all_data = []
            
            for tf in timeframes:
                key = f"market_data:{symbol}:{tf}"
                data = self.redis.get(key)
                if data:
                    parsed = json.loads(data) if isinstance(data, str) else data
                    if isinstance(parsed, dict) and 'ultra_enriched' in parsed:
                        # Essayer d'obtenir les vraies données OHLCV
                        # Si disponibles dans les données enrichies
                        close_price = parsed.get('close', 0)
                        synthetic_candle = {
                            'timestamp': parsed.get('timestamp', 0),
                            'open': parsed.get('open', close_price),
                            'high': parsed.get('high', close_price * 1.001),  # +0.1% si pas disponible
                            'low': parsed.get('low', close_price * 0.999),    # -0.1% si pas disponible
                            'close': close_price,
                            'volume': parsed.get('volume', 0),
                            'timeframe': tf,
                            # Ajouter tous les indicateurs techniques déjà calculés
                            'rsi_14': parsed.get('rsi_14', 50),
                            'macd_line': parsed.get('macd_line', 0),
                            'macd_signal': parsed.get('macd_signal', 0),
                            'macd_histogram': parsed.get('macd_histogram', 0),
                            'ema_7': parsed.get('ema_7', 0),  # MIGRATION BINANCE directe
                            'ema_26': parsed.get('ema_26', parsed.get('close', 0)),
                            'ema_99': parsed.get('ema_99', 0),
                            'sma_20': parsed.get('sma_20', parsed.get('close', 0)),
                            'sma_50': parsed.get('sma_50', parsed.get('close', 0)),
                            'bb_upper': parsed.get('bb_upper', parsed.get('close', 0)),
                            'bb_middle': parsed.get('bb_middle', parsed.get('close', 0)),
                            'bb_lower': parsed.get('bb_lower', parsed.get('close', 0)),
                            'bb_position': parsed.get('bb_position', 0.5),
                            'adx_14': parsed.get('adx_14', 25),
                            'atr_14': parsed.get('atr_14', 0),
                            'williams_r': parsed.get('williams_r', -50),
                            'cci_20': parsed.get('cci_20', 0),
                            'vwap_10': parsed.get('vwap_10', parsed.get('close', 0)),
                            'momentum_10': parsed.get('momentum_10', 0),
                            'volume_ratio': parsed.get('volume_ratio', 1),
                            'volume_spike': parsed.get('volume_spike', False),
                            'volume_trend': parsed.get('volume_trend', 'stable')
                        }
                        all_data.append(synthetic_candle)
            
            # Si on a des données, les retourner, sinon essayer l'ancienne méthode
            if all_data:
                # Trier par timestamp et prendre les plus récents
                all_data.sort(key=lambda x: x['timestamp'])
                logger.warning(f"⚠️ Utilisation données synthétiques pour {symbol}: {len(all_data)} points")
                return all_data[-limit:] if len(all_data) > limit else all_data
            
            # Fallback vers l'ancienne méthode si pas de données enrichies
            key = f"candles:1m:{symbol}"
            try:
                candles_data = self.redis.zrange(key, -limit, -1)
            except AttributeError:
                candles_data = self.redis.get(key)
                if candles_data:
                    if isinstance(candles_data, str):
                        candles_data = json.loads(candles_data)
                    if isinstance(candles_data, list):
                        candles_data = candles_data[-limit:]
                    else:
                        candles_data = []
                else:
                    candles_data = []
            
            if candles_data:
                candles = []
                for candle_str in candles_data:
                    if isinstance(candle_str, str):
                        candles.append(json.loads(candle_str))
                    else:
                        candles.append(candle_str)
                return candles
            
            return []
            
        except Exception as e:
            logger.error(f"Erreur récupération données de marché: {e}")
            return []