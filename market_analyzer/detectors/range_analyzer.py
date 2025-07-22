"""
Range Analyzer

This module provides detailed analysis of price action within ranges:
- Price position within range (% of range)
- Range quality assessment
- Breakout detection and confirmation
- Range trading opportunities
- False breakout identification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RangeQuality(Enum):
    """Qualité du range pour le trading."""
    EXCELLENT = "excellent"      # Range très net, respecté
    GOOD = "good"               # Range correct
    AVERAGE = "average"         # Range moyen
    POOR = "poor"              # Range peu fiable
    INVALID = "invalid"        # Pas de range valide


class BreakoutType(Enum):
    """Type de breakout."""
    CONFIRMED_BULL = "confirmed_bull"
    CONFIRMED_BEAR = "confirmed_bear"
    FALSE_BREAKOUT_UP = "false_breakout_up"
    FALSE_BREAKOUT_DOWN = "false_breakout_down"
    PENDING = "pending"
    NONE = "none"


class RangePosition(Enum):
    """Position du prix dans le range."""
    BOTTOM = "bottom"           # 0-25% du range
    LOWER_MIDDLE = "lower_middle"  # 25-50% du range
    UPPER_MIDDLE = "upper_middle"  # 50-75% du range
    TOP = "top"                # 75-100% du range
    OUTSIDE = "outside"        # En dehors du range


@dataclass
class RangeInfo:
    """Information complète sur un range."""
    range_high: float
    range_low: float
    range_size: float  # En prix absolu
    range_size_pct: float  # En pourcentage du prix moyen
    current_position: float  # Position actuelle dans le range (0-1)
    position_category: RangePosition
    quality: RangeQuality
    duration: int  # Nombre de périodes
    tests_high: int  # Nombre de tests de résistance
    tests_low: int  # Nombre de tests de support
    efficiency: float  # % de temps où prix respecte le range
    volume_at_boundaries: float  # Volume moyen aux limites
    breakout_probability: float  # Probabilité de breakout
    preferred_direction: str  # Direction probable du breakout
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire pour export."""
        return {
            'range_high': self.range_high,
            'range_low': self.range_low,
            'range_size': self.range_size,
            'range_size_pct': self.range_size_pct,
            'current_position': self.current_position,
            'position_category': self.position_category.value,
            'quality': self.quality.value,
            'duration': self.duration,
            'tests_high': self.tests_high,
            'tests_low': self.tests_low,
            'efficiency': self.efficiency,
            'volume_at_boundaries': self.volume_at_boundaries,
            'breakout_probability': self.breakout_probability,
            'preferred_direction': self.preferred_direction,
            'timestamp': self.timestamp
        }


@dataclass
class BreakoutAnalysis:
    """Analyse de breakout."""
    breakout_type: BreakoutType
    confidence: float  # 0-100
    price_movement: float  # % de mouvement depuis la limite
    volume_confirmation: bool
    momentum_confirmation: bool
    sustainability_score: float  # Score de durabilité du breakout
    target_projection: Optional[float]  # Objectif projeté
    stop_level: Optional[float]  # Niveau de stop
    time_since_breakout: int  # Périodes depuis le breakout
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire pour export."""
        return {
            'breakout_type': self.breakout_type.value,
            'confidence': self.confidence,
            'price_movement': self.price_movement,
            'volume_confirmation': self.volume_confirmation,
            'momentum_confirmation': self.momentum_confirmation,
            'sustainability_score': self.sustainability_score,
            'target_projection': self.target_projection,
            'stop_level': self.stop_level,
            'time_since_breakout': self.time_since_breakout
        }


class RangeAnalyzer:
    """
    Analyseur de ranges et de position de prix.
    
    Fournit une analyse complète des ranges incluant:
    - Identification des ranges valides
    - Position du prix dans le range
    - Qualité du range pour le trading
    - Détection et confirmation de breakouts
    - Opportunités de trading dans le range
    """
    
    def __init__(self,
                 min_range_size: float = 0.02,  # 2% minimum pour crypto
                 min_duration: int = 10,  # 10 périodes minimum
                 breakout_threshold: float = 0.005,  # 0.5% pour confirmer breakout
                 volume_threshold: float = 1.5):  # 1.5x volume moyen
        """
        Args:
            min_range_size: Taille minimum du range en %
            min_duration: Durée minimum du range
            breakout_threshold: Seuil pour détecter un breakout
            volume_threshold: Multiplicateur de volume pour confirmation
        """
        self.min_range_size = min_range_size
        self.min_duration = min_duration
        self.breakout_threshold = breakout_threshold
        self.volume_threshold = volume_threshold
    
    def analyze_range(self,
                     highs: Union[List[float], np.ndarray],
                     lows: Union[List[float], np.ndarray],
                     closes: Union[List[float], np.ndarray],
                     volumes: Union[List[float], np.ndarray],
                     lookback: int = 50) -> RangeInfo:
        """
        Analyse complète du range actuel.
        
        Args:
            highs: Prix hauts
            lows: Prix bas
            closes: Prix de clôture
            volumes: Volumes
            lookback: Période d'analyse
            
        Returns:
            Information complète sur le range
        """
        try:
            # Conversion en arrays numpy
            highs = np.array(highs, dtype=float)
            lows = np.array(lows, dtype=float)
            closes = np.array(closes, dtype=float)
            volumes = np.array(volumes, dtype=float)
            
            if len(closes) < self.min_duration:
                return self._invalid_range()
            
            # Utiliser les données récentes
            recent_highs = highs[-lookback:]
            recent_lows = lows[-lookback:]
            recent_closes = closes[-lookback:]
            recent_volumes = volumes[-lookback:]
            
            current_price = closes[-1]
            
            # 1. Identifier les limites du range
            range_high, range_low = self._identify_range_boundaries(
                recent_highs, recent_lows, recent_closes
            )
            
            if range_high is None or range_low is None:
                return self._invalid_range()
            
            # 2. Calculer les métriques de base
            range_size = range_high - range_low
            avg_price = (range_high + range_low) / 2
            range_size_pct = range_size / avg_price
            
            # Vérifier si le range est assez grand
            if range_size_pct < self.min_range_size:
                return self._invalid_range()
            
            # 3. Position actuelle dans le range
            current_position = self._calculate_position_in_range(
                current_price, range_high, range_low
            )
            position_category = self._categorize_position(current_position)
            
            # 4. Qualité du range
            quality = self._assess_range_quality(
                recent_highs, recent_lows, recent_closes, range_high, range_low
            )
            
            # 5. Durée du range
            duration = self._estimate_range_duration(
                recent_closes, range_high, range_low
            )
            
            # 6. Tests des limites
            tests_high = self._count_boundary_tests(
                recent_highs, range_high, tolerance=0.01
            )
            tests_low = self._count_boundary_tests(
                recent_lows, range_low, tolerance=0.01, is_low=True
            )
            
            # 7. Efficacité du range
            efficiency = self._calculate_range_efficiency(
                recent_closes, range_high, range_low
            )
            
            # 8. Volume aux limites
            volume_at_boundaries = self._calculate_boundary_volume(
                recent_highs, recent_lows, recent_volumes, range_high, range_low
            )
            
            # 9. Probabilité de breakout
            breakout_probability = self._calculate_breakout_probability(
                duration, tests_high, tests_low, efficiency, volume_at_boundaries
            )
            
            # 10. Direction préférée
            preferred_direction = self._determine_preferred_direction(
                recent_closes, recent_volumes, current_position
            )
            
            return RangeInfo(
                range_high=float(range_high),
                range_low=float(range_low),
                range_size=float(range_size),
                range_size_pct=float(range_size_pct),
                current_position=float(current_position),
                position_category=position_category,
                quality=quality,
                duration=int(duration),
                tests_high=int(tests_high),
                tests_low=int(tests_low),
                efficiency=float(efficiency),
                volume_at_boundaries=float(volume_at_boundaries),
                breakout_probability=float(breakout_probability),
                preferred_direction=preferred_direction
            )
            
        except Exception as e:
            logger.error(f"Erreur analyse range: {e}")
            return self._invalid_range()
    
    def detect_breakout(self,
                       highs: Union[List[float], np.ndarray],
                       lows: Union[List[float], np.ndarray],
                       closes: Union[List[float], np.ndarray],
                       volumes: Union[List[float], np.ndarray],
                       range_info: RangeInfo) -> BreakoutAnalysis:
        """
        Détecte et analyse les breakouts du range.
        
        Args:
            highs: Prix hauts
            lows: Prix bas
            closes: Prix de clôture
            volumes: Volumes
            range_info: Information sur le range
            
        Returns:
            Analyse complète du breakout
        """
        try:
            current_price = closes[-1]
            current_high = highs[-1]
            current_low = lows[-1]
            current_volume = volumes[-1]
            
            # Vérifier si on est en breakout
            breakout_up = current_high > range_info.range_high * (1 + self.breakout_threshold)
            breakout_down = current_low < range_info.range_low * (1 - self.breakout_threshold)
            
            if not breakout_up and not breakout_down:
                return BreakoutAnalysis(
                    breakout_type=BreakoutType.NONE,
                    confidence=0.0,
                    price_movement=0.0,
                    volume_confirmation=False,
                    momentum_confirmation=False,
                    sustainability_score=0.0,
                    target_projection=None,
                    stop_level=None,
                    time_since_breakout=0
                )
            
            # Analyser le breakout détecté
            if breakout_up:
                return self._analyze_bullish_breakout(
                    highs, lows, closes, volumes, range_info
                )
            else:
                return self._analyze_bearish_breakout(
                    highs, lows, closes, volumes, range_info
                )
                
        except Exception as e:
            logger.error(f"Erreur détection breakout: {e}")
            return self._empty_breakout_analysis()
    
    def get_trading_levels(self, range_info: RangeInfo, current_price: float) -> Dict:
        """
        Retourne les niveaux de trading optimaux pour le range.
        
        Args:
            range_info: Information sur le range
            current_price: Prix actuel
            
        Returns:
            Dictionnaire avec les niveaux de trading
        """
        levels = {
            'range_high': range_info.range_high,
            'range_low': range_info.range_low,
            'range_mid': (range_info.range_high + range_info.range_low) / 2,
            'buy_zones': [],
            'sell_zones': [],
            'breakout_levels': {}
        }
        
        range_size = range_info.range_high - range_info.range_low
        
        # Zones d'achat (près du support)
        levels['buy_zones'] = [
            range_info.range_low,
            range_info.range_low + range_size * 0.1,  # 10% du range
            range_info.range_low + range_size * 0.25  # 25% du range
        ]
        
        # Zones de vente (près de la résistance)
        levels['sell_zones'] = [
            range_info.range_high,
            range_info.range_high - range_size * 0.1,  # 10% du range
            range_info.range_high - range_size * 0.25  # 25% du range
        ]
        
        # Niveaux de breakout
        levels['breakout_levels'] = {
            'bull_entry': range_info.range_high * (1 + self.breakout_threshold),
            'bear_entry': range_info.range_low * (1 - self.breakout_threshold),
            'bull_target': range_info.range_high + range_size,  # Projection 1:1
            'bear_target': range_info.range_low - range_size,   # Projection 1:1
            'bull_stop': range_info.range_high * 0.995,        # Juste sous résistance
            'bear_stop': range_info.range_low * 1.005          # Juste sur support
        }
        
        return levels
    
    def _identify_range_boundaries(self, highs: np.ndarray, lows: np.ndarray, 
                                  closes: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """Identifie les limites du range."""
        try:
            # Méthode 1: Percentiles pour identifier les limites principales
            high_percentile = np.percentile(highs, 95)
            low_percentile = np.percentile(lows, 5)
            
            # Méthode 2: Résistances/supports par clustering
            # Tolérance pour grouper les niveaux
            tolerance = 0.01  # 1%
            
            # Trouver les pics de résistance
            resistance_candidates = []
            for i in range(2, len(highs) - 2):
                if (highs[i] > highs[i-1] and highs[i] > highs[i+1] and
                    highs[i] > highs[i-2] and highs[i] > highs[i+2]):
                    resistance_candidates.append(highs[i])
            
            # Trouver les creux de support
            support_candidates = []
            for i in range(2, len(lows) - 2):
                if (lows[i] < lows[i-1] and lows[i] < lows[i+1] and
                    lows[i] < lows[i-2] and lows[i] < lows[i+2]):
                    support_candidates.append(lows[i])
            
            # Choisir les niveaux les plus fréquents
            range_high = None
            range_low = None
            
            if resistance_candidates:
                # Grouper les résistances similaires
                resistance_groups = self._group_similar_levels(resistance_candidates, tolerance)
                if resistance_groups:
                    # Prendre le groupe avec le plus de tests
                    best_resistance = max(resistance_groups, key=len)
                    range_high = np.mean(best_resistance)
            
            if support_candidates:
                # Grouper les supports similaires
                support_groups = self._group_similar_levels(support_candidates, tolerance)
                if support_groups:
                    best_support = max(support_groups, key=len)
                    range_low = np.mean(best_support)
            
            # Fallback aux percentiles si pas assez de pivots
            if range_high is None:
                range_high = high_percentile
            if range_low is None:
                range_low = low_percentile
            
            # Validation finale
            if range_high <= range_low:
                return None, None
            
            return range_high, range_low
            
        except Exception as e:
            logger.warning(f"Erreur identification limites range: {e}")
            return None, None
    
    def _group_similar_levels(self, levels: List[float], tolerance: float) -> List[List[float]]:
        """Groupe les niveaux similaires."""
        if not levels:
            return []
        
        levels = sorted(levels)
        groups = []
        current_group = [levels[0]]
        
        for i in range(1, len(levels)):
            if abs(levels[i] - levels[i-1]) / levels[i-1] <= tolerance:
                current_group.append(levels[i])
            else:
                if len(current_group) >= 2:  # Au moins 2 tests
                    groups.append(current_group)
                current_group = [levels[i]]
        
        if len(current_group) >= 2:
            groups.append(current_group)
        
        return groups
    
    def _calculate_position_in_range(self, price: float, range_high: float, range_low: float) -> float:
        """Calcule la position du prix dans le range (0-1)."""
        if range_high == range_low:
            return 0.5
        
        position = (price - range_low) / (range_high - range_low)
        return max(0, min(1, position))  # Clamp entre 0 et 1
    
    def _categorize_position(self, position: float) -> RangePosition:
        """Catégorise la position dans le range."""
        if position < 0 or position > 1:
            return RangePosition.OUTSIDE
        elif position <= 0.25:
            return RangePosition.BOTTOM
        elif position <= 0.50:
            return RangePosition.LOWER_MIDDLE
        elif position <= 0.75:
            return RangePosition.UPPER_MIDDLE
        else:
            return RangePosition.TOP
    
    def _assess_range_quality(self, highs: np.ndarray, lows: np.ndarray, 
                             closes: np.ndarray, range_high: float, range_low: float) -> RangeQuality:
        """Évalue la qualité du range."""
        try:
            # Critères de qualité
            score = 0
            
            # 1. Respect des limites (% de temps dans le range)
            in_range_count = sum(1 for c in closes if range_low <= c <= range_high)
            respect_ratio = in_range_count / len(closes)
            if respect_ratio > 0.9:
                score += 3
            elif respect_ratio > 0.8:
                score += 2
            elif respect_ratio > 0.7:
                score += 1
            
            # 2. Tests des limites sans cassure
            high_tests = sum(1 for h in highs if h >= range_high * 0.99)
            low_tests = sum(1 for l in lows if l <= range_low * 1.01)
            
            if high_tests >= 3 and low_tests >= 3:
                score += 2
            elif high_tests >= 2 and low_tests >= 2:
                score += 1
            
            # 3. Stabilité du range (pas de fausses cassures)
            false_breakouts = 0
            for i, (h, l, c) in enumerate(zip(highs, lows, closes)):
                if h > range_high and c < range_high:  # Fausse cassure haute
                    false_breakouts += 1
                elif l < range_low and c > range_low:  # Fausse cassure basse
                    false_breakouts += 1
            
            if false_breakouts == 0:
                score += 2
            elif false_breakouts <= 2:
                score += 1
            
            # 4. Taille du range appropriée
            avg_price = (range_high + range_low) / 2
            range_size_pct = (range_high - range_low) / avg_price
            
            if 0.03 <= range_size_pct <= 0.15:  # 3-15% pour crypto
                score += 1
            
            # Classification
            if score >= 6:
                return RangeQuality.EXCELLENT
            elif score >= 4:
                return RangeQuality.GOOD
            elif score >= 2:
                return RangeQuality.AVERAGE
            else:
                return RangeQuality.POOR
                
        except Exception as e:
            logger.warning(f"Erreur évaluation qualité range: {e}")
            return RangeQuality.POOR
    
    def _estimate_range_duration(self, closes: np.ndarray, 
                                range_high: float, range_low: float) -> int:
        """Estime la durée du range actuel."""
        # Chercher depuis quand on est dans ce range
        duration = 0
        for i in range(len(closes) - 1, -1, -1):
            if range_low <= closes[i] <= range_high:
                duration += 1
            else:
                break
        
        return max(duration, 1)
    
    def _count_boundary_tests(self, prices: np.ndarray, boundary: float, 
                             tolerance: float, is_low: bool = False) -> int:
        """Compte les tests d'une limite."""
        tests = 0
        boundary_range = boundary * tolerance
        
        for price in prices:
            if is_low:
                if price <= boundary + boundary_range:
                    tests += 1
            else:
                if price >= boundary - boundary_range:
                    tests += 1
        
        return tests
    
    def _calculate_range_efficiency(self, closes: np.ndarray, 
                                   range_high: float, range_low: float) -> float:
        """Calcule l'efficacité du range (% de temps respecté)."""
        in_range = sum(1 for c in closes if range_low <= c <= range_high)
        return in_range / len(closes)
    
    def _calculate_boundary_volume(self, highs: np.ndarray, lows: np.ndarray,
                                  volumes: np.ndarray, range_high: float, range_low: float) -> float:
        """Calcule le volume moyen aux limites du range."""
        boundary_volumes = []
        tolerance = 0.01
        
        for h, l, v in zip(highs, lows, volumes):
            # Test de la résistance
            if h >= range_high * (1 - tolerance):
                boundary_volumes.append(v)
            # Test du support
            elif l <= range_low * (1 + tolerance):
                boundary_volumes.append(v)
        
        if not boundary_volumes:
            return np.mean(volumes)
        
        return np.mean(boundary_volumes)
    
    def _calculate_breakout_probability(self, duration: int, tests_high: int, 
                                      tests_low: int, efficiency: float, 
                                      boundary_volume: float) -> float:
        """Calcule la probabilité de breakout."""
        # Facteurs augmentant la probabilité de breakout
        prob = 0.1  # Base
        
        # Durée (plus long = plus probable)
        if duration > 20:
            prob += 0.3
        elif duration > 10:
            prob += 0.2
        
        # Tests répétés des limites
        total_tests = tests_high + tests_low
        if total_tests > 10:
            prob += 0.3
        elif total_tests > 5:
            prob += 0.2
        
        # Efficacité décroissante
        if efficiency < 0.8:
            prob += 0.2
        
        return min(prob, 0.9)
    
    def _determine_preferred_direction(self, closes: np.ndarray, volumes: np.ndarray, 
                                     current_position: float) -> str:
        """Détermine la direction préférée du breakout."""
        # Analyse de momentum récent
        if len(closes) < 10:
            return "neutral"
        
        recent_closes = closes[-10:]
        recent_volumes = volumes[-10:]
        
        # Tendance des prix
        price_slope = np.polyfit(range(len(recent_closes)), recent_closes, 1)[0]
        
        # Tendance du volume
        volume_slope = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
        
        # Position dans le range
        if current_position > 0.6 and price_slope > 0:
            return "bullish"
        elif current_position < 0.4 and price_slope < 0:
            return "bearish"
        else:
            return "neutral"
    
    def _analyze_bullish_breakout(self, highs: np.ndarray, lows: np.ndarray,
                                 closes: np.ndarray, volumes: np.ndarray,
                                 range_info: RangeInfo) -> BreakoutAnalysis:
        """Analyse un breakout haussier."""
        current_price = closes[-1]
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-20:])
        
        # Mouvement depuis la résistance
        price_movement = (current_price - range_info.range_high) / range_info.range_high * 100
        
        # Confirmations
        volume_confirmation = current_volume > avg_volume * self.volume_threshold
        momentum_confirmation = self._check_momentum_confirmation(closes, True)
        
        # Type de breakout
        if volume_confirmation and momentum_confirmation and price_movement > 1:
            breakout_type = BreakoutType.CONFIRMED_BULL
            confidence = 80
        elif current_price < range_info.range_high:  # Retour dans le range
            breakout_type = BreakoutType.FALSE_BREAKOUT_UP
            confidence = 20
        else:
            breakout_type = BreakoutType.PENDING
            confidence = 50
        
        # Calcul du score de durabilité
        sustainability = self._calculate_sustainability_score(
            closes, volumes, range_info.range_high, True
        )
        
        # Projection de target
        range_size = range_info.range_high - range_info.range_low
        target = range_info.range_high + range_size  # Projection 1:1
        
        return BreakoutAnalysis(
            breakout_type=breakout_type,
            confidence=confidence,
            price_movement=price_movement,
            volume_confirmation=volume_confirmation,
            momentum_confirmation=momentum_confirmation,
            sustainability_score=sustainability,
            target_projection=target,
            stop_level=range_info.range_high * 0.995,
            time_since_breakout=1
        )
    
    def _analyze_bearish_breakout(self, highs: np.ndarray, lows: np.ndarray,
                                 closes: np.ndarray, volumes: np.ndarray,
                                 range_info: RangeInfo) -> BreakoutAnalysis:
        """Analyse un breakout baissier."""
        current_price = closes[-1]
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-20:])
        
        # Mouvement depuis le support
        price_movement = (range_info.range_low - current_price) / range_info.range_low * 100
        
        # Confirmations
        volume_confirmation = current_volume > avg_volume * self.volume_threshold
        momentum_confirmation = self._check_momentum_confirmation(closes, False)
        
        # Type de breakout
        if volume_confirmation and momentum_confirmation and price_movement > 1:
            breakout_type = BreakoutType.CONFIRMED_BEAR
            confidence = 80
        elif current_price > range_info.range_low:  # Retour dans le range
            breakout_type = BreakoutType.FALSE_BREAKOUT_DOWN
            confidence = 20
        else:
            breakout_type = BreakoutType.PENDING
            confidence = 50
        
        # Calcul du score de durabilité
        sustainability = self._calculate_sustainability_score(
            closes, volumes, range_info.range_low, False
        )
        
        # Projection de target
        range_size = range_info.range_high - range_info.range_low
        target = range_info.range_low - range_size  # Projection 1:1
        
        return BreakoutAnalysis(
            breakout_type=breakout_type,
            confidence=confidence,
            price_movement=price_movement,
            volume_confirmation=volume_confirmation,
            momentum_confirmation=momentum_confirmation,
            sustainability_score=sustainability,
            target_projection=target,
            stop_level=range_info.range_low * 1.005,
            time_since_breakout=1
        )
    
    def _check_momentum_confirmation(self, closes: np.ndarray, is_bullish: bool) -> bool:
        """Vérifie la confirmation de momentum."""
        if len(closes) < 10:
            return False
        
        # Calcul de momentum simple (changement sur 5 périodes)
        momentum = (closes[-1] - closes[-6]) / closes[-6]
        
        if is_bullish:
            return momentum > 0.01  # 1% de momentum haussier
        else:
            return momentum < -0.01  # 1% de momentum baissier
    
    def _calculate_sustainability_score(self, closes: np.ndarray, volumes: np.ndarray,
                                       breakout_level: float, is_bullish: bool) -> float:
        """Calcule le score de durabilité du breakout."""
        score = 50  # Base
        
        current_price = closes[-1]
        
        # Distance depuis le breakout
        if is_bullish:
            distance = (current_price - breakout_level) / breakout_level
            if distance > 0.02:  # 2% au-dessus
                score += 20
            elif distance > 0.01:
                score += 10
        else:
            distance = (breakout_level - current_price) / breakout_level
            if distance > 0.02:
                score += 20
            elif distance > 0.01:
                score += 10
        
        # Consistance du mouvement
        if len(closes) >= 5:
            recent_prices = closes[-5:]
            if is_bullish:
                consistent = all(recent_prices[i] >= recent_prices[i-1] for i in range(1, len(recent_prices)))
            else:
                consistent = all(recent_prices[i] <= recent_prices[i-1] for i in range(1, len(recent_prices)))
            
            if consistent:
                score += 15
        
        return min(score, 100)
    
    def _invalid_range(self) -> RangeInfo:
        """Retourne un range invalide."""
        return RangeInfo(
            range_high=0.0,
            range_low=0.0,
            range_size=0.0,
            range_size_pct=0.0,
            current_position=0.0,
            position_category=RangePosition.OUTSIDE,
            quality=RangeQuality.INVALID,
            duration=0,
            tests_high=0,
            tests_low=0,
            efficiency=0.0,
            volume_at_boundaries=0.0,
            breakout_probability=0.0,
            preferred_direction="neutral"
        )
    
    def _empty_breakout_analysis(self) -> BreakoutAnalysis:
        """Analyse de breakout vide."""
        return BreakoutAnalysis(
            breakout_type=BreakoutType.NONE,
            confidence=0.0,
            price_movement=0.0,
            volume_confirmation=False,
            momentum_confirmation=False,
            sustainability_score=0.0,
            target_projection=None,
            stop_level=None,
            time_since_breakout=0
        )