#!/usr/bin/env python3
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import ta
import json
from enum import Enum

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
        
        # ADX thresholds (plus nuancés)
        self.adx_no_trend = 20
        self.adx_weak_trend = 25
        self.adx_trend = 35
        self.adx_strong_trend = 45
        
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
    
    async def _calculate_detailed_regime(self, symbol: str) -> Tuple[MarketRegime, Dict[str, float]]:
        """Calcul détaillé du régime avec multiples indicateurs"""
        try:
            # Récupérer les données
            candles = await self._get_recent_candles(symbol, limit=100)
            
            if not candles or len(candles) < 50:
                return MarketRegime.UNDEFINED, {}
            
            df = pd.DataFrame(candles)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # 1. ADX pour la force de tendance
            adx_indicator = ta.trend.ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            )
            current_adx = adx_indicator.adx().iloc[-1]
            plus_di = adx_indicator.adx_pos().iloc[-1]
            minus_di = adx_indicator.adx_neg().iloc[-1]
            
            # 2. Bollinger Bands pour la volatilité
            bb = ta.volatility.BollingerBands(
                close=df['close'],
                window=20,
                window_dev=2
            )
            bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
            current_bb_width = bb_width.iloc[-1]
            
            # Position du prix dans les bandes
            bb_position = (df['close'].iloc[-1] - bb.bollinger_lband().iloc[-1]) / (
                bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1]
            )
            
            # 3. RSI pour le momentum
            rsi = ta.momentum.RSIIndicator(close=df['close'], window=14)
            current_rsi = rsi.rsi().iloc[-1]
            
            # 4. ROC (Rate of Change) pour le momentum directionnel
            roc = ta.momentum.ROCIndicator(close=df['close'], window=10)
            current_roc = roc.roc().iloc[-1]
            
            # 5. Volume analysis
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # 6. Trend slope (régression linéaire sur 20 périodes)
            prices = df['close'].iloc[-20:].values
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            trend_angle = np.degrees(np.arctan(slope / prices.mean() * 100))
            
            # 7. Analyse des supports/résistances
            pivot_high_count = self._count_pivots(df['high'].values[-50:], is_high=True)
            pivot_low_count = self._count_pivots(df['low'].values[-50:], is_low=True)
            
            # Compiler les métriques
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
                'pivot_count': pivot_high_count + pivot_low_count
            }
            
            # Déterminer le régime
            regime = self._determine_regime(metrics)
            
            logger.info(f"Régime {symbol}: {regime.value} | ADX={current_adx:.1f}, "
                       f"ROC={current_roc:.1f}%, RSI={current_rsi:.1f}, "
                       f"BB_width={current_bb_width:.3f}")
            
            return regime, metrics
            
        except Exception as e:
            logger.error(f"Erreur calcul régime détaillé: {e}")
            return MarketRegime.UNDEFINED, {}
    
    def _determine_regime(self, metrics: Dict[str, float]) -> MarketRegime:
        """Détermine le régime basé sur les métriques"""
        adx = metrics['adx']
        plus_di = metrics['plus_di']
        minus_di = metrics['minus_di']
        bb_width = metrics['bb_width']
        rsi = metrics['rsi']
        roc = metrics['roc']
        trend_angle = metrics['trend_angle']
        
        # Tendance directionnelle
        is_bullish = plus_di > minus_di
        
        # Force de la tendance basée sur ADX et autres facteurs
        if adx >= self.adx_strong_trend:
            # Tendance très forte
            if is_bullish and roc > self.momentum_strong:
                return MarketRegime.STRONG_TREND_UP
            elif not is_bullish and roc < -self.momentum_strong:
                return MarketRegime.STRONG_TREND_DOWN
                
        if adx >= self.adx_trend:
            # Tendance normale
            if is_bullish:
                return MarketRegime.TREND_UP
            else:
                return MarketRegime.TREND_DOWN
                
        if adx >= self.adx_weak_trend:
            # Tendance faible
            if is_bullish:
                return MarketRegime.WEAK_TREND_UP
            else:
                return MarketRegime.WEAK_TREND_DOWN
                
        # Marché en range
        if bb_width < self.bb_squeeze_tight:
            return MarketRegime.RANGE_TIGHT
        elif bb_width > self.bb_expansion:
            return MarketRegime.RANGE_VOLATILE
        else:
            # Range normal, vérifier la direction générale
            if trend_angle > 5:
                return MarketRegime.WEAK_TREND_UP
            elif trend_angle < -5:
                return MarketRegime.WEAK_TREND_DOWN
            else:
                return MarketRegime.RANGE_TIGHT
    
    def _count_pivots(self, data: np.ndarray, window: int = 3, 
                     is_high: bool = False, is_low: bool = False) -> int:
        """Compte le nombre de pivots (hauts/bas locaux)"""
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
    
    async def _get_recent_candles(self, symbol: str, limit: int = 100) -> list:
        """Récupère les chandeliers récents depuis Redis"""
        try:
            # Clé pour les chandeliers 1m
            key = f"candles:1m:{symbol}"
            
            # Récupérer depuis Redis avec fallback pour différents types de clients
            try:
                # Essayer zrange d'abord (sorted sets)
                candles_data = self.redis.zrange(key, -limit, -1)
            except AttributeError:
                # Fallback pour RedisClientPool customisé
                candles_data = self.redis.get(key)
                if candles_data:
                    if isinstance(candles_data, str):
                        candles_data = json.loads(candles_data)
                    if isinstance(candles_data, list):
                        candles_data = candles_data[-limit:]  # Prendre les derniers
                    else:
                        candles_data = []
                else:
                    candles_data = []
            
            if not candles_data:
                # Fallback sur une clé simple
                candles_data = self.redis.get(f"market_data:{symbol}:1m")
                if candles_data:
                    parsed = json.loads(candles_data) if isinstance(candles_data, str) else candles_data
                    return parsed[-limit:] if isinstance(parsed, list) else []
                return []
            
            # Parser les chandeliers
            candles = []
            for candle_str in candles_data:
                if isinstance(candle_str, str):
                    candles.append(json.loads(candle_str))
                else:
                    candles.append(candle_str)
                    
            return candles
            
        except Exception as e:
            logger.error(f"Erreur récupération chandeliers: {e}")
            return []