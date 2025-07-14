#!/usr/bin/env python3
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
from enum import Enum
from dataclasses import dataclass
from shared.src.technical_indicators import TechnicalIndicators
from enhanced_regime_detector import MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class TimeframeSignal:
    """Signal pour un timeframe spécifique"""
    timeframe: str
    signal_strength: float  # -1.0 à 1.0 (-1 = strong sell, 1 = strong buy)
    trend_direction: str   # 'up', 'down', 'sideways'
    momentum: float        # Momentum score 
    volatility: float      # Volatilité normalisée
    volume_confirmation: bool
    regime: MarketRegime
    confidence: float      # 0.0 à 1.0


@dataclass
class ConfluenceResult:
    """Résultat de l'analyse de confluence"""
    overall_signal: float         # Signal global (-1.0 à 1.0)
    confluence_score: float       # Score de confluence (0-100%)
    dominant_timeframes: List[str] # Timeframes qui dominent
    conflicting_timeframes: List[str] # Timeframes en conflit
    strength_rating: str          # 'VERY_STRONG', 'STRONG', 'MODERATE', 'WEAK', 'CONFLICTED'
    risk_level: float            # Niveau de risque (0-10)
    recommended_action: str      # 'BUY', 'SELL', 'HOLD', 'AVOID'
    timeframe_signals: Dict[str, TimeframeSignal]


class MultiTimeframeConfluence:
    """Analyseur de confluence multi-timeframes avancé"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.indicators = TechnicalIndicators()
        
        # Configuration des timeframes avec pondérations
        self.timeframes = {
            '1m': {
                'weight': 0.20,
                'role': 'entry_timing',
                'min_data_points': 20
            },
            '5m': {
                'weight': 0.30,
                'role': 'signal_validation', 
                'min_data_points': 50
            },
            '15m': {
                'weight': 0.25,
                'role': 'trend_confirmation',
                'min_data_points': 100
            },
            '1h': {
                'weight': 0.15,
                'role': 'market_context',
                'min_data_points': 150
            },
            '4h': {
                'weight': 0.10,
                'role': 'macro_trend',
                'min_data_points': 200
            }
        }
        
        # Seuils de confluence
        self.confluence_thresholds = {
            'very_strong': 80.0,
            'strong': 65.0,
            'moderate': 50.0,
            'weak': 35.0
        }
    
    async def analyze_confluence(self, symbol: str) -> ConfluenceResult:
        """
        Analyse la confluence multi-timeframes pour un symbole
        
        Returns:
            ConfluenceResult avec analyse complète
        """
        try:
            # Vérifier le cache d'abord
            cache_key = f"confluence:{symbol}"
            cached = self.redis.get(cache_key)
            
            if cached:
                if isinstance(cached, str):
                    cached_data = json.loads(cached)
                else:
                    cached_data = cached
                return self._deserialize_confluence_result(cached_data)
            
            # Analyser chaque timeframe
            timeframe_signals = {}
            
            for tf, config in self.timeframes.items():
                signal = await self._analyze_timeframe_signal(symbol, tf, config)
                if signal:
                    timeframe_signals[tf] = signal
            
            if not timeframe_signals:
                logger.warning(f"⚠️ Aucun signal timeframe disponible pour {symbol}")
                return self._create_default_result(symbol)
            
            # Calculer la confluence
            confluence_result = self._calculate_confluence(timeframe_signals)
            
            # Mettre en cache pour 30 secondes
            cache_data = self._serialize_confluence_result(confluence_result)
            self.redis.set(cache_key, json.dumps(cache_data), expiration=30)
            
            logger.info(f"🎯 Confluence {symbol}: {confluence_result.strength_rating} "
                       f"({confluence_result.confluence_score:.1f}%) -> {confluence_result.recommended_action}")
            
            return confluence_result
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse confluence pour {symbol}: {e}")
            return self._create_default_result(symbol)
    
    async def _analyze_timeframe_signal(self, symbol: str, timeframe: str, config: Dict) -> Optional[TimeframeSignal]:
        """Analyse le signal pour un timeframe spécifique"""
        try:
            # Récupérer les données de marché pour ce timeframe
            market_data = await self._get_timeframe_data(symbol, timeframe)
            
            if not market_data:
                logger.debug(f"Pas de données {timeframe} pour {symbol}")
                return None
            
            # Extraire les métriques techniques
            metrics = self._extract_technical_metrics(market_data, timeframe)
            
            if not metrics:
                return None
            
            # Calculer la force du signal
            signal_strength = self._calculate_signal_strength(metrics, timeframe)
            
            # Déterminer la direction de tendance
            trend_direction = self._determine_trend_direction(metrics)
            
            # Calculer le momentum
            momentum = self._calculate_momentum_score(metrics)
            
            # Évaluer la volatilité
            volatility = self._calculate_volatility_score(metrics)
            
            # Vérifier la confirmation de volume
            volume_confirmation = self._check_volume_confirmation(metrics)
            
            # Déterminer le régime de marché pour ce timeframe
            regime = self._determine_timeframe_regime(metrics)
            
            # Calculer la confiance
            confidence = self._calculate_confidence(metrics, timeframe)
            
            return TimeframeSignal(
                timeframe=timeframe,
                signal_strength=signal_strength,
                trend_direction=trend_direction,
                momentum=momentum,
                volatility=volatility,
                volume_confirmation=volume_confirmation,
                regime=regime,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse signal {timeframe} pour {symbol}: {e}")
            return None
    
    async def _get_timeframe_data(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Récupère les données de marché pour un timeframe"""
        try:
            key = f"market_data:{symbol}:{timeframe}"
            data = self.redis.get(key)
            
            if not data:
                return None
            
            if isinstance(data, str):
                return json.loads(data)
            return data
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération données {timeframe}: {e}")
            return None
    
    def _extract_technical_metrics(self, market_data: Dict, timeframe: str) -> Optional[Dict]:
        """Extrait les métriques techniques des données de marché"""
        try:
            if not market_data or 'ultra_enriched' not in market_data:
                return None
            
            # Extraire les indicateurs pré-calculés
            metrics = {
                'price': market_data.get('close', 0),
                'rsi': market_data.get('rsi_14', 50),
                'macd_line': market_data.get('macd_line', 0),
                'macd_signal': market_data.get('macd_signal', 0),
                'macd_histogram': market_data.get('macd_histogram', 0),
                'ema_12': market_data.get('ema_12', 0),
                'ema_26': market_data.get('ema_26', 0),
                'ema_50': market_data.get('ema_50', 0),
                'bb_upper': market_data.get('bb_upper', 0),
                'bb_middle': market_data.get('bb_middle', 0),
                'bb_lower': market_data.get('bb_lower', 0),
                'bb_position': market_data.get('bb_position', 0.5),
                'bb_width': market_data.get('bb_width', 0.02),
                'adx': market_data.get('adx_14', 25),
                'atr': market_data.get('atr_14', 0),
                'volume': market_data.get('volume', 0),
                'volume_ratio': market_data.get('volume_ratio', 1.0),
                'volume_spike': market_data.get('volume_spike', False),
                'williams_r': market_data.get('williams_r', -50),
                'cci': market_data.get('cci_20', 0),
                'momentum': market_data.get('momentum_10', 0),
                'vwap': market_data.get('vwap_10', 0),
                'timeframe': timeframe
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Erreur extraction métriques {timeframe}: {e}")
            return None
    
    def _calculate_signal_strength(self, metrics: Dict, timeframe: str) -> float:
        """Calcule la force du signal (-1.0 à 1.0)"""
        try:
            signals = []
            
            # 1. Signal EMA (12/26/50)
            ema_12 = metrics['ema_12']
            ema_26 = metrics['ema_26'] 
            ema_50 = metrics['ema_50']
            price = metrics['price']
            
            if ema_12 > 0 and ema_26 > 0 and ema_50 > 0:
                # Alignement EMA
                if ema_12 > ema_26 > ema_50 and price > ema_12:
                    signals.append(0.8)  # Signal haussier fort
                elif ema_12 > ema_26 and price > ema_26:
                    signals.append(0.5)  # Signal haussier modéré
                elif ema_12 < ema_26 < ema_50 and price < ema_12:
                    signals.append(-0.8) # Signal baissier fort
                elif ema_12 < ema_26 and price < ema_26:
                    signals.append(-0.5) # Signal baissier modéré
                else:
                    signals.append(0.0)  # Neutre
            
            # 2. Signal MACD
            macd_line = metrics['macd_line']
            macd_signal = metrics['macd_signal']
            macd_histogram = metrics['macd_histogram']
            
            if macd_line > macd_signal and macd_histogram > 0:
                signals.append(0.6)
            elif macd_line < macd_signal and macd_histogram < 0:
                signals.append(-0.6)
            else:
                signals.append(0.0)
            
            # 3. Signal RSI
            rsi = metrics['rsi']
            if rsi > 70:
                signals.append(-0.4)  # Survente
            elif rsi < 30:
                signals.append(0.4)   # Survente
            elif 45 <= rsi <= 55:
                signals.append(0.0)   # Neutre
            elif rsi > 55:
                signals.append(0.3)   # Légèrement haussier
            else:
                signals.append(-0.3)  # Légèrement baissier
            
            # 4. Signal Bollinger Bands
            bb_position = metrics['bb_position']
            bb_width = metrics['bb_width']
            
            if bb_position > 0.8 and bb_width > 0.03:
                signals.append(-0.3)  # Proche de la bande haute, expansion
            elif bb_position < 0.2 and bb_width > 0.03:
                signals.append(0.3)   # Proche de la bande basse, expansion
            elif bb_width < 0.015:
                signals.append(0.0)   # Compression, attente breakout
            
            # 5. Signal ADX + Direction
            adx = metrics['adx']
            williams_r = metrics['williams_r']
            
            if adx > 30:  # Tendance confirmée
                if williams_r > -30:  # Surachat
                    signals.append(-0.2)
                elif williams_r < -70:  # Survente
                    signals.append(0.2)
            
            # Calculer la moyenne pondérée
            if signals:
                # Pondération selon le timeframe
                timeframe_weight = self.timeframes.get(timeframe, {}).get('weight', 1.0)
                weighted_signal = sum(signals) / len(signals) * timeframe_weight
                return max(-1.0, min(1.0, weighted_signal))
            
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul force signal: {e}")
            return 0.0
    
    def _determine_trend_direction(self, metrics: Dict) -> str:
        """Détermine la direction de tendance"""
        try:
            ema_12 = metrics['ema_12']
            ema_26 = metrics['ema_26']
            ema_50 = metrics['ema_50']
            macd_line = metrics['macd_line']
            adx = metrics['adx']
            
            # Compter les signaux haussiers/baissiers
            bullish_signals = 0
            bearish_signals = 0
            
            # EMA alignement
            if ema_12 > ema_26 > ema_50:
                bullish_signals += 2
            elif ema_12 < ema_26 < ema_50:
                bearish_signals += 2
            elif ema_12 > ema_26:
                bullish_signals += 1
            elif ema_12 < ema_26:
                bearish_signals += 1
            
            # MACD
            if macd_line > 0:
                bullish_signals += 1
            elif macd_line < 0:
                bearish_signals += 1
            
            # ADX force
            if adx < 25:
                return 'sideways'  # Pas de tendance claire
            
            if bullish_signals > bearish_signals:
                return 'up'
            elif bearish_signals > bullish_signals:
                return 'down'
            else:
                return 'sideways'
                
        except Exception as e:
            logger.error(f"❌ Erreur détermination direction: {e}")
            return 'sideways'
    
    def _calculate_momentum_score(self, metrics: Dict) -> float:
        """Calcule le score de momentum (0.0 à 1.0)"""
        try:
            momentum_signals = []
            
            # 1. MACD Histogram momentum
            macd_histogram = metrics['macd_histogram']
            if abs(macd_histogram) > 0.001:
                momentum_signals.append(min(1.0, abs(macd_histogram) * 1000))
            
            # 2. RSI momentum (distance from 50)
            rsi = metrics['rsi']
            rsi_momentum = abs(rsi - 50) / 50.0
            momentum_signals.append(rsi_momentum)
            
            # 3. CCI momentum
            cci = metrics['cci']
            cci_momentum = min(1.0, abs(cci) / 100.0)
            momentum_signals.append(cci_momentum)
            
            # 4. Williams %R momentum
            williams_r = metrics['williams_r']
            williams_momentum = abs(williams_r + 50) / 50.0
            momentum_signals.append(williams_momentum)
            
            # 5. Volume momentum
            if metrics['volume_spike']:
                momentum_signals.append(0.8)
            else:
                volume_ratio = metrics['volume_ratio']
                volume_momentum = min(1.0, max(0.0, (volume_ratio - 0.5) * 2))
                momentum_signals.append(volume_momentum)
            
            return sum(momentum_signals) / len(momentum_signals) if momentum_signals else 0.0
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul momentum: {e}")
            return 0.0
    
    def _calculate_volatility_score(self, metrics: Dict) -> float:
        """Calcule le score de volatilité normalisé (0.0 à 1.0)"""
        try:
            # Bollinger Bands width comme indicateur principal
            bb_width = metrics['bb_width']
            
            # ATR comme indicateur secondaire
            atr = metrics['atr']
            price = metrics['price']
            
            # Normaliser BB width (crypto: 0.01-0.1 typique)
            bb_volatility = min(1.0, max(0.0, (bb_width - 0.01) / 0.09))
            
            # Normaliser ATR/Price ratio
            if price > 0:
                atr_ratio = atr / price
                atr_volatility = min(1.0, max(0.0, atr_ratio * 100))
            else:
                atr_volatility = 0.0
            
            # Moyenne pondérée
            return (bb_volatility * 0.7 + atr_volatility * 0.3)
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul volatilité: {e}")
            return 0.0
    
    def _check_volume_confirmation(self, metrics: Dict) -> bool:
        """Vérifie si le volume confirme le mouvement"""
        try:
            volume_ratio = metrics['volume_ratio']
            volume_spike = metrics['volume_spike']
            
            # Volume élevé = confirmation
            return volume_spike or volume_ratio > 1.2
            
        except Exception as e:
            logger.error(f"❌ Erreur vérification volume: {e}")
            return False
    
    def _determine_timeframe_regime(self, metrics: Dict) -> MarketRegime:
        """Détermine le régime pour ce timeframe"""
        try:
            adx = metrics['adx']
            bb_width = metrics['bb_width']
            trend_direction = self._determine_trend_direction(metrics)
            
            # Classification basique selon ADX et direction
            if adx > 40:
                if trend_direction == 'up':
                    return MarketRegime.STRONG_TREND_UP
                elif trend_direction == 'down':
                    return MarketRegime.STRONG_TREND_DOWN
                else:
                    return MarketRegime.RANGE_VOLATILE
            elif adx > 25:
                if trend_direction == 'up':
                    return MarketRegime.TREND_UP
                elif trend_direction == 'down':
                    return MarketRegime.TREND_DOWN
                else:
                    return MarketRegime.RANGE_VOLATILE
            elif bb_width < 0.015:
                return MarketRegime.RANGE_TIGHT
            else:
                return MarketRegime.RANGE_VOLATILE
                
        except Exception as e:
            logger.error(f"❌ Erreur détermination régime: {e}")
            return MarketRegime.UNDEFINED
    
    def _calculate_confidence(self, metrics: Dict, timeframe: str) -> float:
        """Calcule la confiance dans le signal (0.0 à 1.0)"""
        try:
            confidence_factors = []
            
            # 1. Force ADX (tendance claire)
            adx = metrics['adx']
            adx_confidence = min(1.0, adx / 50.0)
            confidence_factors.append(adx_confidence * 0.3)
            
            # 2. Volume confirmation
            if self._check_volume_confirmation(metrics):
                confidence_factors.append(0.2)
            
            # 3. Alignement des indicateurs
            signal_strength = abs(self._calculate_signal_strength(metrics, timeframe))
            confidence_factors.append(signal_strength * 0.3)
            
            # 4. Clarté de direction (pas de signaux mixtes)
            trend_direction = self._determine_trend_direction(metrics)
            if trend_direction != 'sideways':
                confidence_factors.append(0.2)
            
            return sum(confidence_factors)
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul confiance: {e}")
            return 0.0
    
    def _calculate_confluence(self, timeframe_signals: Dict[str, TimeframeSignal]) -> ConfluenceResult:
        """Calcule la confluence globale"""
        try:
            # Calculer le signal global pondéré
            weighted_signals = []
            total_weight = 0.0
            
            for tf, signal in timeframe_signals.items():
                weight = self.timeframes.get(tf, {}).get('weight', 1.0)
                confidence_weight = signal.confidence * weight
                weighted_signals.append(signal.signal_strength * confidence_weight)
                total_weight += confidence_weight
            
            overall_signal = sum(weighted_signals) / total_weight if total_weight > 0 else 0.0
            
            # Calculer le score de confluence
            confluence_score = self._calculate_confluence_score(timeframe_signals)
            
            # Identifier les timeframes dominants et conflictuels
            dominant_tf, conflicting_tf = self._identify_timeframe_consensus(timeframe_signals)
            
            # Déterminer la force du rating
            strength_rating = self._determine_strength_rating(confluence_score, overall_signal)
            
            # Calculer le niveau de risque
            risk_level = self._calculate_risk_level(timeframe_signals, confluence_score)
            
            # Recommandation d'action
            recommended_action = self._determine_action(overall_signal, confluence_score, risk_level)
            
            return ConfluenceResult(
                overall_signal=overall_signal,
                confluence_score=confluence_score,
                dominant_timeframes=dominant_tf,
                conflicting_timeframes=conflicting_tf,
                strength_rating=strength_rating,
                risk_level=risk_level,
                recommended_action=recommended_action,
                timeframe_signals=timeframe_signals
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul confluence: {e}")
            return self._create_default_result("ERROR")
    
    def _calculate_confluence_score(self, timeframe_signals: Dict[str, TimeframeSignal]) -> float:
        """Calcule le score de confluence (0-100%)"""
        try:
            if len(timeframe_signals) < 2:
                return 0.0
            
            # Vérifier l'alignement des directions
            directions = [signal.trend_direction for signal in timeframe_signals.values()]
            signal_strengths = [signal.signal_strength for signal in timeframe_signals.values()]
            
            # Compter les directions dominantes
            direction_counts = {'up': 0, 'down': 0, 'sideways': 0}
            for direction in directions:
                direction_counts[direction] += 1
            
            # Confluence directionnelle
            max_direction_count = max(direction_counts.values())
            directional_confluence = (max_direction_count / len(directions)) * 100
            
            # Confluence de force de signal
            avg_signal = sum(signal_strengths) / len(signal_strengths)
            signal_deviations = [abs(signal - avg_signal) for signal in signal_strengths]
            avg_deviation = sum(signal_deviations) / len(signal_deviations)
            signal_confluence = max(0, (1.0 - avg_deviation) * 100)
            
            # Score de confluence pondéré
            confluence_score = (directional_confluence * 0.6 + signal_confluence * 0.4)
            
            return min(100.0, max(0.0, confluence_score))
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul score confluence: {e}")
            return 0.0
    
    def _identify_timeframe_consensus(self, timeframe_signals: Dict[str, TimeframeSignal]) -> Tuple[List[str], List[str]]:
        """Identifie les timeframes en consensus vs conflit"""
        try:
            # Calculer la direction moyenne
            signal_strengths = {tf: signal.signal_strength for tf, signal in timeframe_signals.items()}
            avg_signal = sum(signal_strengths.values()) / len(signal_strengths)
            
            dominant = []
            conflicting = []
            
            for tf, strength in signal_strengths.items():
                # Même direction que la moyenne
                if (avg_signal > 0 and strength > 0) or (avg_signal < 0 and strength < 0):
                    # Force suffisante
                    if abs(strength) > 0.3:
                        dominant.append(tf)
                elif abs(strength) > 0.3:  # Direction opposée avec force
                    conflicting.append(tf)
            
            return dominant, conflicting
            
        except Exception as e:
            logger.error(f"❌ Erreur identification consensus: {e}")
            return [], []
    
    def _determine_strength_rating(self, confluence_score: float, overall_signal: float) -> str:
        """Détermine le rating de force"""
        signal_strength = abs(overall_signal)
        
        if confluence_score >= self.confluence_thresholds['very_strong'] and signal_strength > 0.6:
            return 'VERY_STRONG'
        elif confluence_score >= self.confluence_thresholds['strong'] and signal_strength > 0.4:
            return 'STRONG' 
        elif confluence_score >= self.confluence_thresholds['moderate'] and signal_strength > 0.2:
            return 'MODERATE'
        elif confluence_score >= self.confluence_thresholds['weak']:
            return 'WEAK'
        else:
            return 'CONFLICTED'
    
    def _calculate_risk_level(self, timeframe_signals: Dict[str, TimeframeSignal], confluence_score: float) -> float:
        """Calcule le niveau de risque (0-10)"""
        try:
            risk_factors = []
            
            # 1. Risque de confluence faible
            confluence_risk = max(0, (100 - confluence_score) / 10)
            risk_factors.append(confluence_risk)
            
            # 2. Risque de volatilité
            volatilities = [signal.volatility for signal in timeframe_signals.values()]
            avg_volatility = sum(volatilities) / len(volatilities)
            volatility_risk = avg_volatility * 5  # 0-5 scale
            risk_factors.append(volatility_risk)
            
            # 3. Risque de momentum divergent
            momentums = [signal.momentum for signal in timeframe_signals.values()]
            if momentums:
                momentum_std = np.std(momentums)
                momentum_risk = momentum_std * 10  # Normaliser
                risk_factors.append(min(3.0, momentum_risk))
            
            # 4. Risque de conflit de timeframes
            conflicting_count = len(self._identify_timeframe_consensus(timeframe_signals)[1])
            conflict_risk = conflicting_count * 1.5
            risk_factors.append(conflict_risk)
            
            total_risk = sum(risk_factors)
            return min(10.0, max(0.0, total_risk))
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul risque: {e}")
            return 5.0
    
    def _determine_action(self, overall_signal: float, confluence_score: float, risk_level: float) -> str:
        """Détermine l'action recommandée"""
        try:
            signal_strength = abs(overall_signal)
            
            # Conditions de sécurité
            if risk_level > 7.0 or confluence_score < 40.0:
                return 'AVOID'
            
            # Signal fort avec bonne confluence
            if signal_strength > 0.6 and confluence_score > 70.0:
                return 'BUY' if overall_signal > 0 else 'SELL'
            
            # Signal modéré
            elif signal_strength > 0.3 and confluence_score > 55.0:
                return 'BUY' if overall_signal > 0 else 'SELL'
            
            # Signal faible ou confluence insuffisante
            else:
                return 'HOLD'
                
        except Exception as e:
            logger.error(f"❌ Erreur détermination action: {e}")
            return 'HOLD'
    
    def _create_default_result(self, symbol: str) -> ConfluenceResult:
        """Crée un résultat par défaut"""
        return ConfluenceResult(
            overall_signal=0.0,
            confluence_score=0.0,
            dominant_timeframes=[],
            conflicting_timeframes=[],
            strength_rating='UNDEFINED',
            risk_level=5.0,
            recommended_action='HOLD',
            timeframe_signals={}
        )
    
    def _serialize_confluence_result(self, result: ConfluenceResult) -> Dict:
        """Sérialise le résultat pour le cache"""
        return {
            'overall_signal': result.overall_signal,
            'confluence_score': result.confluence_score,
            'dominant_timeframes': result.dominant_timeframes,
            'conflicting_timeframes': result.conflicting_timeframes,
            'strength_rating': result.strength_rating,
            'risk_level': result.risk_level,
            'recommended_action': result.recommended_action,
            'timeframe_signals': {
                tf: {
                    'timeframe': signal.timeframe,
                    'signal_strength': signal.signal_strength,
                    'trend_direction': signal.trend_direction,
                    'momentum': signal.momentum,
                    'volatility': signal.volatility,
                    'volume_confirmation': signal.volume_confirmation,
                    'regime': signal.regime.value,
                    'confidence': signal.confidence
                } for tf, signal in result.timeframe_signals.items()
            }
        }
    
    def _deserialize_confluence_result(self, data: Dict) -> ConfluenceResult:
        """Désérialise le résultat depuis le cache"""
        timeframe_signals = {}
        for tf, signal_data in data.get('timeframe_signals', {}).items():
            timeframe_signals[tf] = TimeframeSignal(
                timeframe=signal_data['timeframe'],
                signal_strength=signal_data['signal_strength'],
                trend_direction=signal_data['trend_direction'],
                momentum=signal_data['momentum'],
                volatility=signal_data['volatility'],
                volume_confirmation=signal_data['volume_confirmation'],
                regime=MarketRegime(signal_data['regime']),
                confidence=signal_data['confidence']
            )
        
        return ConfluenceResult(
            overall_signal=data['overall_signal'],
            confluence_score=data['confluence_score'],
            dominant_timeframes=data['dominant_timeframes'],
            conflicting_timeframes=data['conflicting_timeframes'],
            strength_rating=data['strength_rating'],
            risk_level=data['risk_level'],
            recommended_action=data['recommended_action'],
            timeframe_signals=timeframe_signals
        )