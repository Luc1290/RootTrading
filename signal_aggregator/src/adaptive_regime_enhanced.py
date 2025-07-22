#!/usr/bin/env python3
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, field

import sys
import os

# Add path to shared modules BEFORE imports
sys.path.append(os.path.dirname(__file__))

try:
    from .shared.technical_utils import TechnicalCalculators
except ImportError:
    # Fallback pour l'exécution dans le conteneur
    sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))
    from technical_utils import TechnicalCalculators
from shared.src.technical_indicators import TechnicalIndicators
from shared.src.config import (
    ADX_NO_TREND_THRESHOLD, ADX_WEAK_TREND_THRESHOLD, 
    ADX_TREND_THRESHOLD, ADX_STRONG_TREND_THRESHOLD
)
from enhanced_regime_detector import MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveThresholds:
    """Seuils adaptatifs pour un symbole"""
    symbol: str
    adx_no_trend: float = field(default=ADX_NO_TREND_THRESHOLD)
    adx_weak_trend: float = field(default=ADX_WEAK_TREND_THRESHOLD)
    adx_trend: float = field(default=ADX_TREND_THRESHOLD)
    adx_strong_trend: float = field(default=ADX_STRONG_TREND_THRESHOLD)
    bb_squeeze_tight: float = field(default=0.015)
    bb_squeeze_normal: float = field(default=0.025)
    bb_expansion: float = field(default=0.04)
    rsi_oversold: float = field(default=30)
    rsi_overbought: float = field(default=70)
    volume_surge_multiplier: float = field(default=2.0)
    last_update: datetime = field(default_factory=datetime.now)
    confidence: float = field(default=0.5)  # Confiance dans les seuils adaptatifs


@dataclass
class MarketCharacteristics:
    """Caractéristiques de marché analysées"""
    symbol: str
    avg_volatility: float        # Volatilité moyenne sur 30 jours
    avg_volume: float           # Volume moyen
    trend_persistence: float    # Persistance des tendances
    mean_reversion_tendency: float  # Tendance mean reversion
    breakout_frequency: float   # Fréquence des breakouts
    market_cap_category: str    # 'large_cap', 'mid_cap', 'small_cap'
    correlation_btc: float      # Corrélation avec BTC
    trading_hours_activity: Dict[str, float]  # Activité par heure
    seasonal_patterns: Dict[str, float]  # Patterns saisonniers


@dataclass
class RegimeTransition:
    """Transition entre régimes"""
    from_regime: MarketRegime
    to_regime: MarketRegime
    transition_time: datetime
    transition_speed: float  # Vitesse de transition
    probability: float       # Probabilité de la transition
    catalyst: str           # Catalyseur de la transition


class AdaptiveRegimeEnhanced:
    """Système de régime adaptatif amélioré avec ML et analyse comportementale"""
    
    def __init__(self, redis_client) -> None:
        self.redis = redis_client
        self.indicators = TechnicalIndicators()
        
        # Cache des seuils adaptatifs par symbole
        self.adaptive_thresholds: Dict[str, AdaptiveThresholds] = {}
        
        # Historique des régimes pour ML
        self.regime_history: Dict[str, List[Tuple[datetime, MarketRegime]]] = {}
        
        # Caractéristiques de marché par symbole
        self.market_characteristics: Dict[str, MarketCharacteristics] = {}
        
        # Configuration d'adaptation
        self.adaptation_config = {
            'min_data_points': 100,  # Minimum de points pour adapter
            'adaptation_speed': 0.1,  # Vitesse d'adaptation (0.1 = 10%)
            'confidence_threshold': 0.7,  # Seuil de confiance pour utiliser les seuils adaptatifs
            'reversion_period': 7,  # Jours pour revenir aux seuils par défaut si pas de données
            'ml_retrain_interval': 1440,  # Minutes entre re-entraînements ML
        }
        
        # Patterns de régime par type de crypto
        self.crypto_regime_patterns = {
            'btc': {
                'trend_persistence': 0.7,
                'volatility_factor': 1.0,
                'volume_importance': 0.8
            },
            'eth': {
                'trend_persistence': 0.6,
                'volatility_factor': 1.2,
                'volume_importance': 0.7
            },
            'altcoin': {
                'trend_persistence': 0.4,
                'volatility_factor': 1.8,
                'volume_importance': 0.6
            },
            'stablecoin': {
                'trend_persistence': 0.2,
                'volatility_factor': 0.3,
                'volume_importance': 0.9
            }
        }
    
    async def get_adaptive_regime(self, symbol: str) -> Tuple[MarketRegime, Dict[str, float], AdaptiveThresholds]:
        """
        Obtient le régime avec seuils adaptatifs
        
        Returns:
            Tuple (regime, metrics, adaptive_thresholds)
        """
        try:
            # Vérifier le cache
            cache_key = f"adaptive_regime:{symbol}"
            cached = self.redis.get(cache_key)
            
            if cached:
                if isinstance(cached, str):
                    cached_data = json.loads(cached)
                else:
                    cached_data = cached
                return self._deserialize_adaptive_regime(cached_data, symbol)
            
            # Obtenir ou calculer les seuils adaptatifs
            adaptive_thresholds = await self._get_adaptive_thresholds(symbol)
            
            # Récupérer les données de marché
            market_data = await self._get_market_data_for_adaptation(symbol)
            
            if not market_data:
                logger.warning(f"⚠️ Pas de données pour régime adaptatif {symbol}")
                return MarketRegime.UNDEFINED, {}, adaptive_thresholds
            
            # Calculer le régime avec seuils adaptatifs
            regime, metrics = self._calculate_adaptive_regime(market_data, adaptive_thresholds)
            
            # Mettre à jour l'historique des régimes
            self._update_regime_history(symbol, regime)
            
            # Analyser les transitions pour améliorer les prédictions
            await self._analyze_regime_transitions(symbol, regime)
            
            # Mettre en cache
            cache_data = self._serialize_adaptive_regime(regime, metrics, adaptive_thresholds)
            self.redis.set(cache_key, json.dumps(cache_data), expiration=60)
            
            logger.info(f"🧠 Régime adaptatif {symbol}: {regime.value} | "
                       f"Seuils: ADX={adaptive_thresholds.adx_trend:.1f} | "
                       f"Confiance: {adaptive_thresholds.confidence:.2f}")
            
            return regime, metrics, adaptive_thresholds
            
        except Exception as e:
            logger.error(f"❌ Erreur régime adaptatif pour {symbol}: {e}")
            default_thresholds = AdaptiveThresholds(symbol=symbol)
            return MarketRegime.UNDEFINED, {}, default_thresholds
    
    async def _get_adaptive_thresholds(self, symbol: str) -> AdaptiveThresholds:
        """Obtient les seuils adaptatifs pour un symbole"""
        try:
            # Vérifier le cache local
            if symbol in self.adaptive_thresholds:
                thresholds = self.adaptive_thresholds[symbol]
                # Vérifier si les seuils ne sont pas trop anciens
                if datetime.now() - thresholds.last_update < timedelta(hours=1):
                    return thresholds
            
            # Calculer de nouveaux seuils adaptatifs
            new_thresholds = await self._calculate_adaptive_thresholds(symbol)
            
            # Mettre en cache
            self.adaptive_thresholds[symbol] = new_thresholds
            
            return new_thresholds
            
        except Exception as e:
            logger.error(f"❌ Erreur obtention seuils adaptatifs {symbol}: {e}")
            return AdaptiveThresholds(symbol=symbol)
    
    async def _calculate_adaptive_thresholds(self, symbol: str) -> AdaptiveThresholds:
        """Calcule les seuils adaptatifs basés sur l'historique"""
        try:
            # Récupérer l'historique des données
            historical_data = await self._get_historical_data_for_adaptation(symbol)
            
            if not historical_data or len(historical_data) < self.adaptation_config['min_data_points']:
                logger.debug(f"Pas assez de données pour adaptation {symbol}, utilisation seuils par défaut")
                return AdaptiveThresholds(symbol=symbol, confidence=0.3)
            
            # Analyser les caractéristiques du marché
            market_chars = self._analyze_market_characteristics(historical_data, symbol)
            self.market_characteristics[symbol] = market_chars
            
            # Calculer les seuils adaptatifs
            thresholds = self._adapt_thresholds_to_market(market_chars, symbol)
            
            # Valider les seuils avec backtesting
            confidence = self._validate_adaptive_thresholds(historical_data, thresholds)
            thresholds.confidence = confidence
            
            logger.info(f"📊 Seuils adaptatifs calculés pour {symbol}: "
                       f"ADX_trend={thresholds.adx_trend:.1f}, "
                       f"BB_expansion={thresholds.bb_expansion:.3f}, "
                       f"Confiance={confidence:.2f}")
            
            return thresholds
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul seuils adaptatifs {symbol}: {e}")
            return AdaptiveThresholds(symbol=symbol, confidence=0.3)
    
    async def _get_market_data_for_adaptation(self, symbol: str) -> Optional[Dict]:
        """Récupère les données de marché pour le calcul adaptatif"""
        try:
            # Essayer les données enrichies d'abord
            key = f"market_data:{symbol}:15m"
            data = self.redis.get(key)
            
            if data:
                if isinstance(data, str):
                    return json.loads(data)
                return data
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération données marché {symbol}: {e}")
            return None
    
    async def _get_historical_data_for_adaptation(self, symbol: str) -> List[Dict]:
        """Récupère l'historique pour l'adaptation"""
        try:
            # Essayer de récupérer depuis Redis (données récentes)
            historical_keys = [
                f"historical:{symbol}:15m",
                f"historical:{symbol}:1h",
                f"market_data:{symbol}:15m"
            ]
            
            all_data = []
            for key in historical_keys:
                data = self.redis.get(key)
                if data:
                    if isinstance(data, str):
                        parsed = json.loads(data)
                    else:
                        parsed = data
                    
                    if isinstance(parsed, list):
                        all_data.extend(parsed)
                    elif isinstance(parsed, dict):
                        all_data.append(parsed)
            
            # ❌ SIMULATION SUPPRIMÉE : utiliser uniquement les vraies données
            # Si données insuffisantes dans Redis, essayer la DB
            if len(all_data) < self.adaptation_config['min_data_points']:
                logger.info(f"💾 Données Redis insuffisantes pour {symbol}: {len(all_data)}/{self.adaptation_config['min_data_points']}, tentative DB...")
                
                # Fallback vers la base de données
                db_data = await self._get_historical_from_db(symbol, '15m', 200)
                if db_data:
                    all_data.extend(db_data)
                    logger.info(f"✅ DB: +{len(db_data)} points ajoutés pour {symbol} (total: {len(all_data)})")
                else:
                    logger.warning(f"⚠️ Aucune donnée disponible en DB pour {symbol}")
            
            # Trier par timestamp et retourner les plus récents
            if all_data:
                all_data.sort(key=lambda x: x.get('timestamp', 0))
                return all_data[-200:]  # Garder les 200 derniers points
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération historique {symbol}: {e}")
            return []
    
    async def _get_historical_from_db(self, symbol: str, timeframe: str, limit: int) -> List[Dict]:
        """Récupère les données historiques depuis PostgreSQL (fallback)"""
        try:
            from shared.src.db_pool import fetch_all
            
            query = """
                SELECT 
                    EXTRACT(epoch FROM time) as timestamp,
                    time, symbol, open, high, low, close, volume,
                    rsi_14, ema_7, ema_26, ema_99, sma_20, sma_50,
                    macd_line, macd_signal, macd_histogram,
                    bb_upper, bb_middle, bb_lower, bb_position, bb_width,
                    atr_14, adx_14, plus_di, minus_di,
                    momentum_10, volume_ratio, avg_volume_20
                FROM market_data 
                WHERE symbol = %s AND timeframe = %s AND enhanced = true
                ORDER BY time DESC 
                LIMIT %s
            """
            
            rows = fetch_all(query, (symbol, timeframe, limit), dict_result=True)
            
            if rows:
                # Convertir en format JSON compatible
                data = []
                for row in rows:
                    row_dict = {}
                    for key, value in row.items():
                        if value is not None and hasattr(value, '__float__'):
                            row_dict[key] = float(value)
                        else:
                            row_dict[key] = value
                    data.append(row_dict)
                
                # Retourner dans l'ordre chronologique (plus ancien en premier)
                return list(reversed(data))
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Erreur DB fallback pour {symbol}: {e}")
            return []
    
    def _calculate_real_btc_correlation(self, symbol: str, historical_data: List[Dict[str, Any]]) -> float:
        """
        Calcule la corrélation réelle avec le marché global (tous les actifs ROOT).
        
        Args:
            symbol: Symbole à analyser
            historical_data: Données historiques du symbole
            
        Returns:
            Corrélation moyenne avec le marché (0.0 à 1.0)
        """
        try:
            if not historical_data or len(historical_data) < 10:
                return self._get_default_market_correlation(symbol)
            
            # Tous les symboles ROOT pour calculer la corrélation marché
            from shared.src.config import SYMBOLS
            market_symbols = [s for s in SYMBOLS if s != symbol]  # Exclure le symbole lui-même
            
            correlations = []
            symbol_returns = self._calculate_returns(historical_data)
            
            if not symbol_returns or len(symbol_returns) < 5:
                return self._get_default_market_correlation(symbol)
            
            # Calculer corrélation avec chaque actif du portefeuille
            for market_symbol in market_symbols[:8]:  # Limiter à 8 actifs pour performance
                try:
                    market_key = f"historical:{market_symbol}:15m"
                    market_data = self.redis.get(market_key)
                    
                    if market_data:
                        if isinstance(market_data, str):
                            market_historical = json.loads(market_data)
                        else:
                            market_historical = market_data
                        
                        market_returns = self._calculate_returns(market_historical)
                        
                        if market_returns and len(market_returns) >= 5:
                            # Corrélation sur période commune
                            min_len = min(len(symbol_returns), len(market_returns))
                            if min_len >= 5:
                                corr = np.corrcoef(
                                    symbol_returns[-min_len:], 
                                    market_returns[-min_len:]
                                )[0, 1]
                                
                                if not np.isnan(corr):
                                    correlations.append(abs(corr))
                
                except Exception as e:
                    logger.debug(f"Erreur corrélation {symbol}-{market_symbol}: {e}")
                    continue
            
            # Retourner corrélation moyenne si on a des données
            if correlations:
                avg_correlation = np.mean(correlations)
                logger.debug(f"Corrélation marché {symbol}: {avg_correlation:.3f} (sur {len(correlations)} actifs)")
                return float(avg_correlation)
            
            # Fallback sur estimation
            return self._get_default_market_correlation(symbol)
                
        except Exception as e:
            logger.warning(f"Erreur calcul corrélation marché pour {symbol}: {e}")
            return self._get_default_market_correlation(symbol)
    
    def _calculate_returns(self, historical_data: List[Dict[str, Any]]) -> List[float]:
        """Calcule les rendements (returns) à partir des données historiques"""
        try:
            returns = []
            for i in range(1, len(historical_data)):
                if 'close' in historical_data[i] and 'close' in historical_data[i-1]:
                    prev_close = float(historical_data[i-1]['close'])
                    curr_close = float(historical_data[i]['close'])
                    if prev_close > 0:
                        return_pct = (curr_close - prev_close) / prev_close
                        returns.append(return_pct)
            return returns
        except Exception as e:
            logger.debug(f"Erreur calcul returns: {e}")
            return []
    
    def _get_default_market_correlation(self, symbol: str) -> float:
        """Valeurs de corrélation par défaut basées sur l'actif"""
        if 'BTC' in symbol:
            return 1.0  # BTC = référence marché
        elif 'ETH' in symbol:
            return 0.85  # ETH très corrélé
        elif symbol in ['SOLUSDC', 'AVAXUSDC', 'LINKUSDC', 'AAVEUSDC']:
            return 0.70  # L1/DeFi modérément corrélés
        elif symbol in ['SUIUSDC', 'ADAUSDC']:
            return 0.65  # Nouveaux L1
        elif symbol in ['PEPEUSDC', 'BONKUSDC', 'DOGEUSDC']:
            return 0.45  # Memecoins moins corrélés
        else:
            return 0.60  # Défaut conservative
    
    def _classify_crypto_type(self, symbol: str) -> str:
        """Classifie le type de crypto pour adapter les paramètres"""
        symbol_upper = symbol.upper()
        
        if 'BTC' in symbol_upper:
            return 'btc'
        elif 'ETH' in symbol_upper:
            return 'eth'
        elif any(stable in symbol_upper for stable in ['USDT', 'USDC', 'BUSD', 'DAI']):
            return 'stablecoin'
        else:
            return 'altcoin'
    
    def _analyze_market_characteristics(self, historical_data: List[Dict], symbol: str) -> MarketCharacteristics:
        """Analyse les caractéristiques du marché"""
        try:
            if not historical_data:
                return self._default_market_characteristics(symbol)
            
            # Calculer la volatilité moyenne
            closes = [candle['close'] for candle in historical_data]
            returns = np.diff(closes) / closes[:-1]
            avg_volatility = np.std(returns) * np.sqrt(24 * 60 / 15)  # Annualisé
            
            # Volume moyen
            volumes = [candle['volume'] for candle in historical_data]
            avg_volume = np.mean(volumes)
            
            # Persistance des tendances
            trend_persistence = self._calculate_trend_persistence(historical_data)
            
            # Tendance mean reversion
            mean_reversion = self._calculate_mean_reversion_tendency(historical_data)
            
            # Fréquence des breakouts
            breakout_frequency = self._calculate_breakout_frequency(historical_data)
            
            # Corrélation avec BTC réelle (calculée sur données historiques)
            correlation_btc = self._calculate_real_btc_correlation(symbol, historical_data)
            
            # Catégorie market cap (basée sur le symbole)
            market_cap_category = self._determine_market_cap_category(symbol)
            
            return MarketCharacteristics(
                symbol=symbol,
                avg_volatility=avg_volatility,
                avg_volume=avg_volume,
                trend_persistence=trend_persistence,
                mean_reversion_tendency=mean_reversion,
                breakout_frequency=breakout_frequency,
                market_cap_category=market_cap_category,
                correlation_btc=correlation_btc,
                trading_hours_activity={},  # À implémenter
                seasonal_patterns={}  # À implémenter
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse caractéristiques marché: {e}")
            return self._default_market_characteristics(symbol)
    
    def _calculate_trend_persistence(self, historical_data: List[Dict]) -> float:
        """Calcule la persistance des tendances en utilisant les EMAs pré-calculées"""
        try:
            if len(historical_data) < 20:
                return 0.5
            
            # Compter les périodes où EMA7 > EMA26 (tendance haussière)
            trend_periods = 0
            trend_changes = 0
            current_trend = None
            
            for candle in historical_data:
                # Utiliser les EMAs déjà calculées dans les données enrichies
                ema_7 = candle.get('ema_7')
                ema_26 = candle.get('ema_26')
                
                # Vérifier que les valeurs existent
                if ema_7 is None or ema_26 is None:
                    continue
                
                # Convertir en float si nécessaire
                ema_7 = float(ema_7)
                ema_26 = float(ema_26)
                
                if ema_7 > ema_26:
                    new_trend = 'up'
                else:
                    new_trend = 'down'
                
                if current_trend is None:
                    current_trend = new_trend
                elif current_trend != new_trend:
                    trend_changes += 1
                    current_trend = new_trend
                
                trend_periods += 1
            
            # Si pas assez de données avec EMAs
            if trend_periods < 10:
                logger.warning(f"⚠️ Pas assez de données avec EMAs: {trend_periods} périodes")
                return 0.5
            
            # Persistance = 1 - (changements / périodes)
            persistence = 1.0 - (trend_changes / trend_periods) if trend_periods > 0 else 0.5
            return max(0.0, min(1.0, persistence))
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul persistance tendance: {e}")
            return 0.5
    
    def _calculate_mean_reversion_tendency(self, historical_data: List[Dict]) -> float:
        """Calcule la tendance au mean reversion en utilisant les SMAs pré-calculées"""
        try:
            if len(historical_data) < 20:
                return 0.5
            
            # Calculer combien de fois le prix revient vers la SMA20
            reversions = 0
            total_opportunities = 0
            
            for i in range(1, len(historical_data)):
                current = historical_data[i]
                prev = historical_data[i-1]
                
                # Utiliser les données pré-calculées
                current_close = current.get('close')
                prev_close = prev.get('close')
                current_sma = current.get('sma_20')
                
                # Vérifier que toutes les valeurs existent
                if all(v is not None for v in [current_close, prev_close, current_sma]):
                    current_close = float(current_close)
                    prev_close = float(prev_close)
                    current_sma = float(current_sma)
                    
                    # Distance relative à la SMA
                    prev_distance = abs(prev_close - current_sma) / current_sma if current_sma > 0 else 0
                    current_distance = abs(current_close - current_sma) / current_sma if current_sma > 0 else 0
                    
                    # Le prix se rapproche de la SMA
                    if prev_distance > current_distance and prev_distance > 0.01:  # Au moins 1% d'écart
                        reversions += 1
                    
                    total_opportunities += 1
            
            return reversions / total_opportunities if total_opportunities > 0 else 0.5
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul mean reversion: {e}")
            return 0.5
    
    def _calculate_breakout_frequency(self, historical_data: List[Dict]) -> float:
        """Calcule la fréquence des breakouts en utilisant les Bollinger Bands pré-calculées"""
        try:
            if len(historical_data) < 50:
                return 0.2
            
            breakouts = 0
            total_periods = 0
            
            for candle in historical_data:
                # Utiliser les Bollinger Bands pré-calculées
                current_close = candle.get('close')
                bb_upper = candle.get('bb_upper')
                bb_lower = candle.get('bb_lower')
                
                # Vérifier que toutes les valeurs existent
                if all(v is not None for v in [current_close, bb_upper, bb_lower]):
                    current_close = float(current_close)
                    bb_upper = float(bb_upper)
                    bb_lower = float(bb_lower)
                    
                    # Vérifier breakout
                    if current_close > bb_upper or current_close < bb_lower:
                        breakouts += 1
                    
                    total_periods += 1
            
            # Si pas assez de données avec BB
            if total_periods < 30:
                logger.warning(f"⚠️ Pas assez de données avec BB: {total_periods} périodes")
                return 0.2
            
            return breakouts / total_periods if total_periods > 0 else 0.2
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul fréquence breakouts: {e}")
            return 0.2
    
    def _determine_market_cap_category(self, symbol: str) -> str:
        """Détermine la catégorie market cap"""
        symbol_upper = symbol.upper()
        
        large_cap = ['BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'SOL', 'DOT']
        if any(coin in symbol_upper for coin in large_cap):
            return 'large_cap'
        
        # Stablecoins sont considérés comme large cap
        if any(stable in symbol_upper for stable in ['USDT', 'USDC', 'BUSD']):
            return 'large_cap'
        
        # Par défaut, considérer comme altcoin (small/mid cap)
        return 'small_cap'
    
    def _default_market_characteristics(self, symbol: str) -> MarketCharacteristics:
        """Retourne des caractéristiques par défaut"""
        crypto_type = self._classify_crypto_type(symbol)
        base_patterns = self.crypto_regime_patterns[crypto_type]
        
        return MarketCharacteristics(
            symbol=symbol,
            avg_volatility=base_patterns['volatility_factor'] * 0.5,
            avg_volume=1000000,
            trend_persistence=base_patterns['trend_persistence'],
            mean_reversion_tendency=0.5,
            breakout_frequency=0.2,
            market_cap_category='small_cap',
            correlation_btc=0.6,
            trading_hours_activity={},
            seasonal_patterns={}
        )
    
    def _adapt_thresholds_to_market(self, market_chars: MarketCharacteristics, symbol: str) -> AdaptiveThresholds:
        """Adapte les seuils aux caractéristiques du marché"""
        try:
            # Seuils par défaut
            thresholds = AdaptiveThresholds(symbol=symbol)
            
            # Adapter ADX selon la volatilité
            volatility_factor = market_chars.avg_volatility / 0.5  # Normaliser
            
            # Plus volatile = seuils ADX plus élevés
            thresholds.adx_weak_trend = ADX_WEAK_TREND_THRESHOLD * (1 + volatility_factor * 0.2)
            thresholds.adx_trend = ADX_TREND_THRESHOLD * (1 + volatility_factor * 0.15)
            thresholds.adx_strong_trend = ADX_STRONG_TREND_THRESHOLD * (1 + volatility_factor * 0.1)
            
            # Adapter BB selon mean reversion
            if market_chars.mean_reversion_tendency > 0.7:
                # Marché mean-reverting = bandes plus larges
                thresholds.bb_squeeze_tight *= 1.3
                thresholds.bb_expansion *= 1.2
            elif market_chars.mean_reversion_tendency < 0.3:
                # Marché trending = bandes plus serrées
                thresholds.bb_squeeze_tight *= 0.8
                thresholds.bb_expansion *= 0.9
            
            # Adapter RSI selon la persistance des tendances
            if market_chars.trend_persistence > 0.7:
                # Tendances persistantes = RSI plus extrême
                thresholds.rsi_oversold = 25
                thresholds.rsi_overbought = 75
            elif market_chars.trend_persistence < 0.3:
                # Tendances faibles = RSI plus conservateur
                thresholds.rsi_oversold = 35
                thresholds.rsi_overbought = 65
            
            # Adapter volume selon le marché
            if market_chars.market_cap_category == 'small_cap':
                thresholds.volume_surge_multiplier = 1.5  # Plus sensible
            else:
                thresholds.volume_surge_multiplier = 2.5  # Moins sensible
            
            thresholds.last_update = datetime.now()
            
            return thresholds
            
        except Exception as e:
            logger.error(f"❌ Erreur adaptation seuils: {e}")
            return AdaptiveThresholds(symbol=symbol)
    
    def _validate_adaptive_thresholds(self, historical_data: List[Dict], thresholds: AdaptiveThresholds) -> float:
        """Valide les seuils adaptatifs par backtesting"""
        try:
            if len(historical_data) < 50:
                return 0.3
            
            # Simuler l'application des seuils sur l'historique
            correct_predictions = 0
            total_predictions = 0
            
            for i in range(20, len(historical_data)):
                # Données pour le calcul
                current_data = historical_data[i]
                
                # Simuler la classification avec les seuils adaptatifs
                predicted_regime = self._classify_regime_with_thresholds(current_data, thresholds)
                
                # Vérifier la "réalité" (mouvement des prix suivants)
                if i < len(historical_data) - 5:
                    future_prices = [historical_data[j]['close'] for j in range(i+1, min(i+6, len(historical_data)))]
                    actual_movement = self._determine_actual_movement(current_data['close'], future_prices)
                    
                    if self._regime_matches_movement(predicted_regime, actual_movement):
                        correct_predictions += 1
                    
                    total_predictions += 1
            
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.5
            
            # Convertir en confiance (0.5 = confiance neutre, 1.0 = confiance maximale)
            confidence = max(0.3, min(1.0, accuracy))
            
            return confidence
            
        except Exception as e:
            logger.error(f"❌ Erreur validation seuils: {e}")
            return 0.5
    
    def _classify_regime_with_thresholds(self, data: Dict, thresholds: AdaptiveThresholds) -> MarketRegime:
        """Classifie le régime avec des seuils donnés"""
        try:
            adx = data.get('adx_14', 25)
            rsi = data.get('rsi_14', 50)
            bb_width = data.get('bb_width', 0.03)
            
            # Classification selon les seuils adaptatifs
            if adx >= thresholds.adx_strong_trend:
                return MarketRegime.STRONG_TREND_UP if rsi > 50 else MarketRegime.STRONG_TREND_DOWN
            elif adx >= thresholds.adx_trend:
                return MarketRegime.TREND_UP if rsi > 50 else MarketRegime.TREND_DOWN
            elif adx >= thresholds.adx_weak_trend:
                return MarketRegime.WEAK_TREND_UP if rsi > 50 else MarketRegime.WEAK_TREND_DOWN
            elif bb_width < thresholds.bb_squeeze_tight:
                return MarketRegime.RANGE_TIGHT
            else:
                return MarketRegime.RANGE_VOLATILE
                
        except Exception as e:
            logger.error(f"❌ Erreur classification régime: {e}")
            return MarketRegime.UNDEFINED
    
    def _determine_actual_movement(self, current_price: float, future_prices: List[float]) -> str:
        """Détermine le mouvement réel des prix"""
        try:
            if not future_prices:
                return 'sideways'
            
            # Calculer le mouvement moyen
            price_changes = [(price - current_price) / current_price for price in future_prices]
            avg_change = np.mean(price_changes)
            
            if avg_change > 0.02:  # +2%
                return 'strong_up'
            elif avg_change > 0.005:  # +0.5%
                return 'up'
            elif avg_change < -0.02:  # -2%
                return 'strong_down'
            elif avg_change < -0.005:  # -0.5%
                return 'down'
            else:
                return 'sideways'
                
        except Exception as e:
            logger.error(f"❌ Erreur détermination mouvement: {e}")
            return 'sideways'
    
    def _regime_matches_movement(self, regime: MarketRegime, movement: str) -> bool:
        """Vérifie si le régime correspond au mouvement réel"""
        trend_up_regimes = [MarketRegime.STRONG_TREND_UP, MarketRegime.TREND_UP, MarketRegime.WEAK_TREND_UP]
        trend_down_regimes = [MarketRegime.STRONG_TREND_DOWN, MarketRegime.TREND_DOWN, MarketRegime.WEAK_TREND_DOWN]
        range_regimes = [MarketRegime.RANGE_TIGHT, MarketRegime.RANGE_VOLATILE]
        
        if regime in trend_up_regimes:
            return movement in ['up', 'strong_up']
        elif regime in trend_down_regimes:
            return movement in ['down', 'strong_down']
        elif regime in range_regimes:
            return movement == 'sideways'
        else:
            return False
    
    def _calculate_adaptive_regime(self, market_data: Dict, thresholds: AdaptiveThresholds) -> Tuple[MarketRegime, Dict[str, float]]:
        """Calcule le régime avec les seuils adaptatifs"""
        try:
            # Utiliser les seuils adaptatifs si la confiance est suffisante
            if thresholds.confidence >= self.adaptation_config['confidence_threshold']:
                regime = self._classify_regime_with_thresholds(market_data, thresholds)
                
                metrics = {
                    'adx': market_data.get('adx_14', 25),
                    'rsi': market_data.get('rsi_14', 50),
                    'bb_width': market_data.get('bb_width', 0.03),
                    'adaptive_confidence': thresholds.confidence,
                    'adx_threshold_used': thresholds.adx_trend,
                    'data_source': 'adaptive'
                }
                
                return regime, metrics
            else:
                # Utiliser les seuils par défaut
                return self._classify_regime_with_default_thresholds(market_data)
                
        except Exception as e:
            logger.error(f"❌ Erreur calcul régime adaptatif: {e}")
            return MarketRegime.UNDEFINED, {}
    
    def _classify_regime_with_default_thresholds(self, market_data: Dict) -> Tuple[MarketRegime, Dict[str, float]]:
        """Classification avec seuils par défaut"""
        try:
            adx = market_data.get('adx_14', 25)
            rsi = market_data.get('rsi_14', 50)
            bb_width = market_data.get('bb_width', 0.03)
            
            # Utiliser les seuils par défaut
            if adx >= ADX_STRONG_TREND_THRESHOLD:
                regime = MarketRegime.STRONG_TREND_UP if rsi > 50 else MarketRegime.STRONG_TREND_DOWN
            elif adx >= ADX_TREND_THRESHOLD:
                regime = MarketRegime.TREND_UP if rsi > 50 else MarketRegime.TREND_DOWN
            elif adx >= ADX_WEAK_TREND_THRESHOLD:
                regime = MarketRegime.WEAK_TREND_UP if rsi > 50 else MarketRegime.WEAK_TREND_DOWN
            elif bb_width < 0.015:
                regime = MarketRegime.RANGE_TIGHT
            else:
                regime = MarketRegime.RANGE_VOLATILE
            
            metrics = {
                'adx': adx,
                'rsi': rsi,
                'bb_width': bb_width,
                'adaptive_confidence': 0.0,
                'adx_threshold_used': ADX_TREND_THRESHOLD,
                'data_source': 'default'
            }
            
            return regime, metrics
            
        except Exception as e:
            logger.error(f"❌ Erreur classification par défaut: {e}")
            return MarketRegime.UNDEFINED, {}
    
    def _update_regime_history(self, symbol: str, regime: MarketRegime):
        """Met à jour l'historique des régimes"""
        try:
            if symbol not in self.regime_history:
                self.regime_history[symbol] = []
            
            # Ajouter le régime actuel
            self.regime_history[symbol].append((datetime.now(), regime))
            
            # Garder seulement les 1000 dernières entrées
            if len(self.regime_history[symbol]) > 1000:
                self.regime_history[symbol] = self.regime_history[symbol][-1000:]
                
        except Exception as e:
            logger.error(f"❌ Erreur mise à jour historique régime: {e}")
    
    async def _analyze_regime_transitions(self, symbol: str, current_regime: MarketRegime):
        """Analyse les transitions de régime pour améliorer les prédictions"""
        try:
            if symbol not in self.regime_history or len(self.regime_history[symbol]) < 2:
                return
            
            # Analyser les transitions récentes
            recent_history = self.regime_history[symbol][-10:]  # 10 dernières
            
            # Détecter les patterns de transition
            transitions = []
            for i in range(1, len(recent_history)):
                prev_regime = recent_history[i-1][1]
                curr_regime = recent_history[i][1]
                
                if prev_regime != curr_regime:
                    transition_time = recent_history[i][0] - recent_history[i-1][0]
                    transitions.append({
                        'from': prev_regime,
                        'to': curr_regime,
                        'duration': transition_time.total_seconds() / 60  # minutes
                    })
            
            # Analyser les patterns (implémentation future pour ML)
            if transitions:
                logger.debug(f"📈 Transitions détectées pour {symbol}: {len(transitions)}")
                
        except Exception as e:
            logger.error(f"❌ Erreur analyse transitions: {e}")
    
    def _serialize_adaptive_regime(self, regime: MarketRegime, metrics: Dict[str, float], 
                                   thresholds: AdaptiveThresholds) -> Dict:
        """Sérialise le régime adaptatif pour le cache"""
        return {
            'regime': regime.value,
            'metrics': metrics,
            'thresholds': {
                'adx_trend': thresholds.adx_trend,
                'bb_expansion': thresholds.bb_expansion,
                'confidence': thresholds.confidence,
                'last_update': thresholds.last_update.isoformat()
            }
        }
    
    def _deserialize_adaptive_regime(self, data: Dict, symbol: str) -> Tuple[MarketRegime, Dict[str, float], AdaptiveThresholds]:
        """Désérialise le régime adaptatif depuis le cache"""
        try:
            regime = MarketRegime(data['regime'])
            metrics = data['metrics']
            
            thresholds = AdaptiveThresholds(symbol=symbol)
            threshold_data = data.get('thresholds', {})
            thresholds.adx_trend = threshold_data.get('adx_trend', ADX_TREND_THRESHOLD)
            thresholds.bb_expansion = threshold_data.get('bb_expansion', 0.04)
            thresholds.confidence = threshold_data.get('confidence', 0.5)
            
            return regime, metrics, thresholds
            
        except Exception as e:
            logger.error(f"❌ Erreur désérialisation régime adaptatif: {e}")
            return MarketRegime.UNDEFINED, {}, AdaptiveThresholds(symbol=symbol)