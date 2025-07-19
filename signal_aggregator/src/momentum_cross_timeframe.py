#!/usr/bin/env python3
import logging
import numpy as np
from typing import Dict, List, Optional
import json
from enum import Enum
from dataclasses import dataclass
from shared.src.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class MomentumDirection(Enum):
    """Direction du momentum"""
    ACCELERATING_UP = "ACCEL_UP"        # Accélération haussière
    ACCELERATING_DOWN = "ACCEL_DOWN"    # Accélération baissière
    DECELERATING_UP = "DECEL_UP"        # Décélération haussière
    DECELERATING_DOWN = "DECEL_DOWN"    # Décélération baissière
    NEUTRAL = "NEUTRAL"                 # Momentum neutre
    DIVERGENT = "DIVERGENT"             # Divergence entre timeframes


@dataclass
class TimeframeMomentum:
    """Momentum pour un timeframe spécifique"""
    timeframe: str
    velocity: float           # Vitesse du momentum (-1 à 1)
    acceleration: float       # Accélération (-1 à 1)
    direction: MomentumDirection
    strength: float          # Force du momentum (0 à 1)
    rsi_momentum: float      # Momentum RSI
    macd_momentum: float     # Momentum MACD
    price_momentum: float    # Momentum prix (ROC)
    volume_momentum: float   # Momentum volume
    confidence: float        # Confiance dans la mesure


@dataclass
class MomentumDivergence:
    """Divergence de momentum détectée"""
    type: str                # 'bullish', 'bearish', 'hidden_bullish', 'hidden_bearish'
    timeframes: List[str]    # Timeframes impliqués
    strength: float          # Force de la divergence
    price_trend: str         # Direction prix
    momentum_trend: str      # Direction momentum
    confirmation_level: float # Niveau de confirmation
    target_reversal: float   # Prix cible de retournement


@dataclass
class CrossTimeframeMomentumAnalysis:
    """Analyse complète du momentum cross-timeframe"""
    overall_momentum: float              # Momentum global (-1 à 1)
    momentum_direction: MomentumDirection
    momentum_strength: float            # Force globale (0 à 1)
    timeframe_momentums: Dict[str, TimeframeMomentum]
    momentum_alignment: float           # Alignement entre timeframes (0 à 1)
    divergences: List[MomentumDivergence]
    momentum_score: float               # Score final (0 à 100)
    trend_continuation_probability: float # Probabilité continuation (0 à 1)
    reversal_signals: List[str]         # Signaux de retournement détectés
    entry_quality: str                  # 'EXCELLENT', 'GOOD', 'AVERAGE', 'POOR'


class MomentumCrossTimeframe:
    """Analyseur de momentum cross-timeframe avancé"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.indicators = TechnicalIndicators()
        
        # Configuration des timeframes avec rôles spécifiques
        self.momentum_timeframes = {
            '1m': {
                'weight': 0.10,
                'role': 'micro_momentum',
                'lookback': 20,
                'sensitivity': 'high'
            },
            '5m': {
                'weight': 0.25,
                'role': 'entry_momentum', 
                'lookback': 50,
                'sensitivity': 'medium-high'
            },
            '15m': {
                'weight': 0.35,
                'role': 'primary_momentum',
                'lookback': 100,
                'sensitivity': 'medium'
            },
            '3m': {
                'weight': 0.25,
                'role': 'trend_momentum',
                'lookback': 80,
                'sensitivity': 'medium'
            }
        }
        
        # Seuils de momentum
        self.momentum_thresholds = {
            'strong_acceleration': 0.7,
            'moderate_acceleration': 0.4,
            'weak_acceleration': 0.2,
            'divergence_threshold': 0.3,
            'alignment_threshold': 0.6
        }
    
    async def analyze_momentum_cross_timeframe(self, symbol: str) -> CrossTimeframeMomentumAnalysis:
        """
        Analyse le momentum cross-timeframe pour un symbole
        
        Returns:
            CrossTimeframeMomentumAnalysis avec analyse complète
        """
        try:
            # Vérifier le cache
            cache_key = f"momentum_cross_tf:{symbol}"
            cached = self.redis.get(cache_key)
            
            if cached:
                if isinstance(cached, str):
                    cached_data = json.loads(cached)
                else:
                    cached_data = cached
                return self._deserialize_momentum_analysis(cached_data)
            
            # Analyser le momentum pour chaque timeframe
            timeframe_momentums = {}
            
            for tf, config in self.momentum_timeframes.items():
                momentum = await self._analyze_timeframe_momentum(symbol, tf, config)
                if momentum:
                    timeframe_momentums[tf] = momentum
            
            if not timeframe_momentums:
                logger.warning(f"⚠️ Aucun momentum timeframe disponible pour {symbol}")
                return self._create_default_momentum_analysis(symbol)
            
            # Calculer le momentum global
            overall_momentum = self._calculate_overall_momentum(timeframe_momentums)
            
            # Déterminer la direction globale
            momentum_direction = self._determine_global_momentum_direction(timeframe_momentums)
            
            # Calculer la force globale
            momentum_strength = self._calculate_global_momentum_strength(timeframe_momentums)
            
            # Calculer l'alignement entre timeframes
            momentum_alignment = self._calculate_momentum_alignment(timeframe_momentums)
            
            # Détecter les divergences
            divergences = self._detect_momentum_divergences(timeframe_momentums, symbol)
            
            # Calculer le score final
            momentum_score = self._calculate_momentum_score(
                overall_momentum, momentum_strength, momentum_alignment, divergences
            )
            
            # Calculer la probabilité de continuation
            trend_continuation_probability = self._calculate_continuation_probability(
                timeframe_momentums, momentum_alignment
            )
            
            # Détecter les signaux de retournement
            reversal_signals = self._detect_reversal_signals(timeframe_momentums, divergences)
            
            # Évaluer la qualité d'entrée
            entry_quality = self._evaluate_entry_quality(
                momentum_score, momentum_alignment, divergences, reversal_signals
            )
            
            analysis = CrossTimeframeMomentumAnalysis(
                overall_momentum=overall_momentum,
                momentum_direction=momentum_direction,
                momentum_strength=momentum_strength,
                timeframe_momentums=timeframe_momentums,
                momentum_alignment=momentum_alignment,
                divergences=divergences,
                momentum_score=momentum_score,
                trend_continuation_probability=trend_continuation_probability,
                reversal_signals=reversal_signals,
                entry_quality=entry_quality
            )
            
            # Mettre en cache pour 30 secondes
            cache_data = self._serialize_momentum_analysis(analysis)
            self.redis.set(cache_key, json.dumps(cache_data), expiration=30)
            
            logger.info(f"⚡ Momentum {symbol}: {momentum_direction.value} | "
                       f"Force: {momentum_strength:.2f} | Alignement: {momentum_alignment:.2f} | "
                       f"Score: {momentum_score:.1f} | Qualité: {entry_quality}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse momentum cross-timeframe pour {symbol}: {e}")
            return self._create_default_momentum_analysis(symbol)
    
    async def _analyze_timeframe_momentum(self, symbol: str, timeframe: str, config: Dict) -> Optional[TimeframeMomentum]:
        """Analyse le momentum pour un timeframe spécifique"""
        try:
            # Récupérer les données pour ce timeframe
            market_data = await self._get_timeframe_data(symbol, timeframe)
            
            if not market_data:
                logger.debug(f"Pas de données momentum {timeframe} pour {symbol}")
                return None
            
            # Calculer les différents types de momentum
            velocity = self._calculate_momentum_velocity(market_data, config)
            acceleration = self._calculate_momentum_acceleration(market_data, config)
            
            # Momentum par indicateur
            rsi_momentum = self._calculate_rsi_momentum(market_data)
            macd_momentum = self._calculate_macd_momentum(market_data)
            price_momentum = self._calculate_price_momentum(market_data, config)
            volume_momentum = self._calculate_volume_momentum(market_data)
            
            # Force globale pour ce timeframe
            strength = self._calculate_timeframe_momentum_strength(
                velocity, acceleration, rsi_momentum, macd_momentum, price_momentum
            )
            
            # Direction du momentum
            direction = self._determine_timeframe_momentum_direction(
                velocity, acceleration, rsi_momentum, macd_momentum
            )
            
            # Confiance dans la mesure
            confidence = self._calculate_momentum_confidence(
                market_data, velocity, acceleration, strength
            )
            
            return TimeframeMomentum(
                timeframe=timeframe,
                velocity=velocity,
                acceleration=acceleration,
                direction=direction,
                strength=strength,
                rsi_momentum=rsi_momentum,
                macd_momentum=macd_momentum,
                price_momentum=price_momentum,
                volume_momentum=volume_momentum,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse momentum {timeframe} pour {symbol}: {e}")
            return None
    
    async def _get_timeframe_data(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Récupère les données enrichies pour un timeframe"""
        try:
            key = f"market_data:{symbol}:{timeframe}"
            data = self.redis.get(key)
            
            if not data:
                return None
            
            if isinstance(data, str):
                return json.loads(data)
            return data
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération données momentum {timeframe}: {e}")
            return None
    
    def _calculate_momentum_velocity(self, market_data: Dict, config: Dict) -> float:
        """Calcule la vélocité du momentum (vitesse de changement)"""
        try:
            # Utiliser le ROC comme base de vélocité
            roc = market_data.get('momentum_10', 0)
            
            # Normaliser selon la sensibilité du timeframe
            sensitivity_multiplier = {
                'high': 2.0,
                'medium-high': 1.5,
                'medium': 1.0,
                'medium-low': 0.7,
                'low': 0.5
            }.get(config.get('sensitivity', 'medium'), 1.0)
            
            # Normaliser le ROC (-1 à 1)
            normalized_velocity = np.tanh(roc * sensitivity_multiplier / 10.0)
            
            return float(normalized_velocity)
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul vélocité momentum: {e}")
            return 0.0
    
    def _calculate_momentum_acceleration(self, market_data: Dict, config: Dict) -> float:
        """Calcule l'accélération du momentum (changement de vélocité)"""
        try:
            # Utiliser l'histogramme MACD comme proxy d'accélération
            macd_histogram = market_data.get('macd_histogram', 0)
            
            # Normaliser selon la sensibilité
            sensitivity_multiplier = {
                'high': 3.0,
                'medium-high': 2.0,
                'medium': 1.0,
                'medium-low': 0.7,
                'low': 0.5
            }.get(config.get('sensitivity', 'medium'), 1.0)
            
            # Normaliser l'histogramme MACD
            normalized_acceleration = np.tanh(macd_histogram * sensitivity_multiplier * 1000)
            
            return float(normalized_acceleration)
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul accélération momentum: {e}")
            return 0.0
    
    def _calculate_rsi_momentum(self, market_data: Dict) -> float:
        """Calcule le momentum basé sur RSI"""
        try:
            rsi = market_data.get('rsi_14', 50)
            
            # Convertir RSI en momentum (-1 à 1)
            # RSI 50 = momentum 0, RSI 0/100 = momentum -1/+1
            rsi_momentum = (rsi - 50) / 50.0
            
            # Appliquer une courbe sigmoïde pour plus de sensibilité
            return float(np.tanh(rsi_momentum * 2))
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul momentum RSI: {e}")
            return 0.0
    
    def _calculate_macd_momentum(self, market_data: Dict) -> float:
        """Calcule le momentum basé sur MACD"""
        try:
            macd_line = market_data.get('macd_line', 0)
            macd_signal = market_data.get('macd_signal', 0)
            
            # Différence MACD/Signal comme momentum
            macd_diff = macd_line - macd_signal
            
            # Normaliser (-1 à 1)
            return float(np.tanh(macd_diff * 1000))
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul momentum MACD: {e}")
            return 0.0
    
    def _calculate_price_momentum(self, market_data: Dict, config: Dict) -> float:
        """Calcule le momentum basé sur le prix"""
        try:
            # Utiliser le momentum_10 déjà calculé
            price_momentum = market_data.get('momentum_10', 0)
            
            # Normaliser selon le timeframe
            timeframe_factor = {
                '1m': 5.0,
                '5m': 2.0,
                '15m': 1.0,
                '3m': 0.5
            }.get(config.get('timeframe', '15m'), 1.0)
            
            return float(np.tanh(price_momentum * timeframe_factor / 100))
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul momentum prix: {e}")
            return 0.0
    
    def _calculate_volume_momentum(self, market_data: Dict) -> float:
        """Calcule le momentum basé sur le volume"""
        try:
            volume_ratio = market_data.get('volume_ratio', 1.0)
            volume_spike = market_data.get('volume_spike', False)
            
            # Momentum volume basé sur le ratio
            volume_momentum = (volume_ratio - 1.0) / 2.0  # Normaliser autour de 1.0
            
            # Boost si spike de volume
            if volume_spike:
                volume_momentum *= 1.5
            
            return float(np.clip(volume_momentum, -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul momentum volume: {e}")
            return 0.0
    
    def _calculate_timeframe_momentum_strength(self, velocity: float, acceleration: float,
                                               rsi_momentum: float, macd_momentum: float,
                                               price_momentum: float) -> float:
        """Calcule la force du momentum pour un timeframe"""
        try:
            # Moyenne pondérée des différents momentums
            weights = {
                'velocity': 0.3,
                'acceleration': 0.25,
                'rsi': 0.2,
                'macd': 0.15,
                'price': 0.1
            }
            
            weighted_momentum = (
                abs(velocity) * weights['velocity'] +
                abs(acceleration) * weights['acceleration'] +
                abs(rsi_momentum) * weights['rsi'] +
                abs(macd_momentum) * weights['macd'] +
                abs(price_momentum) * weights['price']
            )
            
            return float(np.clip(weighted_momentum, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul force momentum timeframe: {e}")
            return 0.0
    
    def _determine_timeframe_momentum_direction(self, velocity: float, acceleration: float,
                                                rsi_momentum: float, macd_momentum: float) -> MomentumDirection:
        """Détermine la direction du momentum pour un timeframe"""
        try:
            # Analyser la combinaison vélocité/accélération
            if velocity > 0.2 and acceleration > 0.2:
                return MomentumDirection.ACCELERATING_UP
            elif velocity > 0.2 and acceleration < -0.2:
                return MomentumDirection.DECELERATING_UP
            elif velocity < -0.2 and acceleration < -0.2:
                return MomentumDirection.ACCELERATING_DOWN
            elif velocity < -0.2 and acceleration > 0.2:
                return MomentumDirection.DECELERATING_DOWN
            else:
                # Vérifier s'il y a divergence entre indicateurs
                momentum_signs = [
                    1 if x > 0 else -1 if x < 0 else 0 
                    for x in [velocity, rsi_momentum, macd_momentum]
                ]
                
                if len(set(momentum_signs)) > 2:  # Signaux mixtes
                    return MomentumDirection.DIVERGENT
                else:
                    return MomentumDirection.NEUTRAL
                    
        except Exception as e:
            logger.error(f"❌ Erreur détermination direction momentum: {e}")
            return MomentumDirection.NEUTRAL
    
    def _calculate_momentum_confidence(self, market_data: Dict, velocity: float,
                                       acceleration: float, strength: float) -> float:
        """Calcule la confiance dans la mesure de momentum"""
        try:
            confidence_factors = []
            
            # 1. Cohérence des signaux
            signal_coherence = 1.0 - abs(abs(velocity) - strength)
            confidence_factors.append(signal_coherence * 0.3)
            
            # 2. Force des signaux
            signal_strength = (abs(velocity) + abs(acceleration)) / 2
            confidence_factors.append(signal_strength * 0.3)
            
            # 3. Volume confirmation
            volume_ratio = market_data.get('volume_ratio', 1.0)
            volume_confidence = min(1.0, volume_ratio / 1.5) if volume_ratio > 1.0 else 0.5  # STANDARDISÉ: 1.5 = Très bon
            confidence_factors.append(volume_confidence * 0.2)
            
            # 4. Qualité des données
            has_enriched = market_data.get('ultra_enriched', False)
            data_quality = 0.9 if has_enriched else 0.6
            confidence_factors.append(data_quality * 0.2)
            
            return sum(confidence_factors)
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul confiance momentum: {e}")
            return 0.5
    
    def _calculate_overall_momentum(self, timeframe_momentums: Dict[str, TimeframeMomentum]) -> float:
        """Calcule le momentum global pondéré"""
        try:
            if not timeframe_momentums:
                return 0.0
            
            weighted_momentum = 0.0
            total_weight = 0.0
            
            for tf, momentum in timeframe_momentums.items():
                weight = self.momentum_timeframes.get(tf, {}).get('weight', 1.0)
                confidence_weight = momentum.confidence * weight
                
                # Utiliser la vélocité comme momentum principal
                weighted_momentum += momentum.velocity * confidence_weight
                total_weight += confidence_weight
            
            return weighted_momentum / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul momentum global: {e}")
            return 0.0
    
    def _determine_global_momentum_direction(self, timeframe_momentums: Dict[str, TimeframeMomentum]) -> MomentumDirection:
        """Détermine la direction globale du momentum"""
        try:
            if not timeframe_momentums:
                return MomentumDirection.NEUTRAL
            
            # Compter les votes par direction
            direction_votes = {}
            for tf, momentum in timeframe_momentums.items():
                weight = self.momentum_timeframes.get(tf, {}).get('weight', 1.0)
                direction = momentum.direction
                
                if direction not in direction_votes:
                    direction_votes[direction] = 0
                direction_votes[direction] += weight * momentum.confidence
            
            # Retourner la direction avec le plus de votes
            if direction_votes:
                return max(direction_votes, key=lambda x: direction_votes[x])
            else:
                return MomentumDirection.NEUTRAL
                
        except Exception as e:
            logger.error(f"❌ Erreur détermination direction globale: {e}")
            return MomentumDirection.NEUTRAL
    
    def _calculate_global_momentum_strength(self, timeframe_momentums: Dict[str, TimeframeMomentum]) -> float:
        """Calcule la force globale du momentum"""
        try:
            if not timeframe_momentums:
                return 0.0
            
            weighted_strength = 0.0
            total_weight = 0.0
            
            for tf, momentum in timeframe_momentums.items():
                weight = self.momentum_timeframes.get(tf, {}).get('weight', 1.0)
                confidence_weight = momentum.confidence * weight
                
                weighted_strength += momentum.strength * confidence_weight
                total_weight += confidence_weight
            
            return weighted_strength / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul force globale momentum: {e}")
            return 0.0
    
    def _calculate_momentum_alignment(self, timeframe_momentums: Dict[str, TimeframeMomentum]) -> float:
        """Calcule l'alignement entre les momentums des différents timeframes"""
        try:
            if len(timeframe_momentums) < 2:
                return 1.0  # Alignement parfait s'il n'y a qu'un timeframe
            
            # Calculer la variance des vélocités
            velocities = [momentum.velocity for momentum in timeframe_momentums.values()]
            velocity_std = float(np.std(velocities))
            
            # Calculer l'alignement directionnel
            directions = [momentum.direction for momentum in timeframe_momentums.values()]
            unique_directions = len(set(directions))
            direction_alignment = 1.0 - (unique_directions - 1) / (len(directions) - 1)
            
            # Score d'alignement combiné
            velocity_alignment = max(0.0, 1.0 - velocity_std * 2)  # Plus la variance est faible, meilleur l'alignement
            
            return (velocity_alignment * 0.6 + direction_alignment * 0.4)
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul alignement momentum: {e}")
            return 0.0
    
    def _detect_momentum_divergences(self, timeframe_momentums: Dict[str, TimeframeMomentum], symbol: str) -> List[MomentumDivergence]:
        """Détecte les divergences de momentum entre timeframes"""
        try:
            divergences: List[MomentumDivergence] = []
            
            # Comparer les timeframes majeurs
            major_timeframes = ['1m', '3m', '5m']
            available_tf = [tf for tf in major_timeframes if tf in timeframe_momentums]
            
            if len(available_tf) < 2:
                return divergences
            
            # Détecter divergences entre timeframes adjacents
            for i in range(len(available_tf) - 1):
                tf1 = available_tf[i]
                tf2 = available_tf[i + 1]
                
                momentum1 = timeframe_momentums[tf1]
                momentum2 = timeframe_momentums[tf2]
                
                # Vérifier divergence directionnelle
                if (momentum1.velocity > 0.3 and momentum2.velocity < -0.3) or \
                   (momentum1.velocity < -0.3 and momentum2.velocity > 0.3):
                    
                    # Déterminer le type de divergence
                    if momentum1.velocity > momentum2.velocity:
                        divergence_type = 'bullish' if momentum1.velocity > 0 else 'hidden_bearish'
                    else:
                        divergence_type = 'bearish' if momentum2.velocity < 0 else 'hidden_bullish'
                    
                    # Calculer la force
                    strength = abs(momentum1.velocity - momentum2.velocity) / 2
                    
                    # Niveau de confirmation
                    confirmation = min(momentum1.confidence, momentum2.confidence)
                    
                    divergence = MomentumDivergence(
                        type=divergence_type,
                        timeframes=[tf1, tf2],
                        strength=strength,
                        price_trend='up' if momentum1.velocity > 0 else 'down',
                        momentum_trend='diverging',
                        confirmation_level=confirmation,
                        target_reversal=0.0  # À calculer selon le contexte
                    )
                    
                    divergences.append(divergence)
            
            return divergences
            
        except Exception as e:
            logger.error(f"❌ Erreur détection divergences momentum: {e}")
            return []
    
    def _calculate_momentum_score(self, overall_momentum: float, momentum_strength: float,
                                  momentum_alignment: float, divergences: List[MomentumDivergence]) -> float:
        """Calcule le score final de momentum (0-100)"""
        try:
            # Score de base
            base_score = (abs(overall_momentum) * 30 + momentum_strength * 40 + momentum_alignment * 30)
            
            # Pénalités pour divergences
            divergence_penalty = len(divergences) * 10
            
            # Score final
            final_score = max(0.0, base_score - divergence_penalty)
            
            return min(100.0, final_score)
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul score momentum: {e}")
            return 0.0
    
    def _calculate_continuation_probability(self, timeframe_momentums: Dict[str, TimeframeMomentum],
                                            momentum_alignment: float) -> float:
        """Calcule la probabilité de continuation de tendance"""
        try:
            # Facteurs de continuation
            continuation_factors = []
            
            # 1. Alignement des timeframes
            continuation_factors.append(momentum_alignment * 0.4)
            
            # 2. Force des momentums principaux
            main_timeframes = ['1m', '5m']
            main_strength = 0.0
            main_count = 0
            
            for tf in main_timeframes:
                if tf in timeframe_momentums:
                    main_strength += timeframe_momentums[tf].strength
                    main_count += 1
            
            if main_count > 0:
                continuation_factors.append((main_strength / main_count) * 0.3)
            
            # 3. Accélération positive
            accelerating_count = 0
            total_count = len(timeframe_momentums)
            
            for momentum in timeframe_momentums.values():
                if momentum.direction in [MomentumDirection.ACCELERATING_UP, MomentumDirection.ACCELERATING_DOWN]:
                    accelerating_count += 1
            
            acceleration_factor = accelerating_count / total_count if total_count > 0 else 0
            continuation_factors.append(acceleration_factor * 0.3)
            
            return sum(continuation_factors)
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul probabilité continuation: {e}")
            return 0.0
    
    def _detect_reversal_signals(self, timeframe_momentums: Dict[str, TimeframeMomentum],
                                 divergences: List[MomentumDivergence]) -> List[str]:
        """Détecte les signaux de retournement"""
        try:
            reversal_signals = []
            
            # 1. Divergences majeures
            for divergence in divergences:
                if divergence.strength > 0.6 and divergence.confirmation_level > 0.7:
                    reversal_signals.append(f"Divergence {divergence.type} forte")
            
            # 2. Décélération sur timeframes majeurs
            major_decelerating = 0
            for tf in ['5m', '15m']:
                if tf in timeframe_momentums:
                    direction = timeframe_momentums[tf].direction
                    if direction in [MomentumDirection.DECELERATING_UP, MomentumDirection.DECELERATING_DOWN]:
                        major_decelerating += 1
            
            if major_decelerating >= 2:
                reversal_signals.append("Décélération sur timeframes majeurs")
            
            # 3. Momentum extrême
            for tf, momentum in timeframe_momentums.items():
                if abs(momentum.velocity) > 0.8 and momentum.strength > 0.7:
                    reversal_signals.append(f"Momentum extrême sur {tf}")
            
            return reversal_signals
            
        except Exception as e:
            logger.error(f"❌ Erreur détection signaux retournement: {e}")
            return []
    
    def _evaluate_entry_quality(self, momentum_score: float, momentum_alignment: float,
                                 divergences: List[MomentumDivergence], reversal_signals: List[str]) -> str:
        """Évalue la qualité d'entrée basée sur le momentum"""
        try:
            # Score composite
            quality_score = momentum_score * 0.5 + momentum_alignment * 30 + (10 if not reversal_signals else 0)
            
            # Pénalités
            quality_score -= len(divergences) * 15
            quality_score -= len(reversal_signals) * 10
            
            # Classification
            if quality_score >= 75:
                return 'EXCELLENT'
            elif quality_score >= 60:
                return 'GOOD'
            elif quality_score >= 40:
                return 'AVERAGE'
            else:
                return 'POOR'
                
        except Exception as e:
            logger.error(f"❌ Erreur évaluation qualité entrée: {e}")
            return 'AVERAGE'
    
    def _create_default_momentum_analysis(self, symbol: str) -> CrossTimeframeMomentumAnalysis:
        """Crée une analyse de momentum par défaut"""
        return CrossTimeframeMomentumAnalysis(
            overall_momentum=0.0,
            momentum_direction=MomentumDirection.NEUTRAL,
            momentum_strength=0.0,
            timeframe_momentums={},
            momentum_alignment=0.0,
            divergences=[],
            momentum_score=0.0,
            trend_continuation_probability=0.0,
            reversal_signals=[],
            entry_quality='POOR'
        )
    
    def _serialize_momentum_analysis(self, analysis: CrossTimeframeMomentumAnalysis) -> Dict:
        """Sérialise l'analyse pour le cache"""
        return {
            'overall_momentum': analysis.overall_momentum,
            'momentum_direction': analysis.momentum_direction.value,
            'momentum_strength': analysis.momentum_strength,
            'momentum_alignment': analysis.momentum_alignment,
            'momentum_score': analysis.momentum_score,
            'trend_continuation_probability': analysis.trend_continuation_probability,
            'reversal_signals': analysis.reversal_signals,
            'entry_quality': analysis.entry_quality,
            'divergences_count': len(analysis.divergences),
            'timeframes_count': len(analysis.timeframe_momentums)
        }
    
    def _deserialize_momentum_analysis(self, data: Dict) -> CrossTimeframeMomentumAnalysis:
        """Désérialise l'analyse depuis le cache"""
        return CrossTimeframeMomentumAnalysis(
            overall_momentum=data.get('overall_momentum', 0.0),
            momentum_direction=MomentumDirection(data.get('momentum_direction', 'NEUTRAL')),
            momentum_strength=data.get('momentum_strength', 0.0),
            timeframe_momentums={},  # Simplifié pour le cache
            momentum_alignment=data.get('momentum_alignment', 0.0),
            divergences=[],  # Simplifié pour le cache
            momentum_score=data.get('momentum_score', 0.0),
            trend_continuation_probability=data.get('trend_continuation_probability', 0.0),
            reversal_signals=data.get('reversal_signals', []),
            entry_quality=data.get('entry_quality', 'POOR')
        )