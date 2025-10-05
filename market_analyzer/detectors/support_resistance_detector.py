"""
Support and Resistance Level Detector

This module identifies key support and resistance levels using multiple methods:
- Pivot Points (traditional and Fibonacci)
- Volume Profile (Point of Control, Value Areas)
- Historical price levels (psychological levels)
- Trend lines and channels
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, NamedTuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats  # type: ignore
from scipy.signal import find_peaks  # type: ignore

logger = logging.getLogger(__name__)


class LevelType(Enum):
    """Types de niveaux de prix."""
    SUPPORT = "support"
    RESISTANCE = "resistance" 
    PIVOT = "pivot"
    PSYCHOLOGICAL = "psychological"
    VOLUME_POC = "volume_poc"  # Point of Control
    VALUE_AREA_HIGH = "value_area_high"
    VALUE_AREA_LOW = "value_area_low"
    TRENDLINE = "trendline"
    CHANNEL = "channel"


class LevelStrength(Enum):
    """Force du niveau."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    MAJOR = "major"


@dataclass
class PriceLevel:
    """Structure représentant un niveau de prix."""
    price: float
    level_type: LevelType
    strength: LevelStrength
    confidence: float  # 0-100
    touches: int  # Nombre de fois testé
    volume_strength: float  # Force basée sur le volume
    distance_from_price: float  # Distance du prix actuel
    last_test_age: int  # Périodes depuis dernier test
    break_probability: float  # Probabilité de cassure
    slope: Optional[float] = None  # Pour trendlines
    timeframe: str = "5m"
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire pour export."""
        return {
            'price': self.price,
            'level_type': self.level_type.value,
            'strength': self.strength.value,
            'confidence': self.confidence,
            'touches': self.touches,
            'volume_strength': self.volume_strength,
            'distance_from_price': self.distance_from_price,
            'last_test_age': self.last_test_age,
            'break_probability': self.break_probability,
            'slope': self.slope,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp
        }


class SupportResistanceDetector:
    """
    Détecteur de niveaux de support et résistance.
    
    Utilise plusieurs méthodes complémentaires:
    1. Points pivots classiques
    2. Analyse de volume (POC, Value Areas)
    3. Niveaux psychologiques (nombres ronds)
    4. Lignes de tendance
    5. Test de rebond/cassure historique
    """
    
    def __init__(self,
                 lookback_period: int = 100,
                 min_touches: int = 2,
                 price_tolerance: float = 0.002,  # 0.2% pour crypto
                 volume_window: int = 50):
        """
        Args:
            lookback_period: Période d'analyse historique
            min_touches: Minimum de touches pour valider un niveau
            price_tolerance: Tolérance de prix pour grouper les niveaux
            volume_window: Fenêtre pour l'analyse de volume
        """
        self.lookback_period = lookback_period
        self.min_touches = min_touches
        self.price_tolerance = price_tolerance
        self.volume_window = volume_window
        
        # Cache pour optimiser les calculs répétés
        self._cache: Dict[str, Any] = {}
    
    def detect_levels(self,
                     highs: Union[List[float], np.ndarray],
                     lows: Union[List[float], np.ndarray],
                     closes: Union[List[float], np.ndarray],
                     volumes: Union[List[float], np.ndarray],
                     current_price: Optional[float] = None,
                     timeframe: str = "5m") -> List[PriceLevel]:
        """
        Détecte tous les niveaux de support et résistance.
        
        Args:
            highs: Prix hauts
            lows: Prix bas
            closes: Prix de clôture
            volumes: Volumes
            current_price: Prix actuel (dernier close si None)
            timeframe: Timeframe des données
            
        Returns:
            Liste des niveaux détectés, triés par force
        """
        try:
            # Conversion en arrays numpy
            highs = np.array(highs, dtype=float)
            lows = np.array(lows, dtype=float)
            closes = np.array(closes, dtype=float)
            volumes = np.array(volumes, dtype=float)
            
            if len(closes) < 20:
                return []
            
            current_price = current_price or closes[-1]
            all_levels = []
            
            # 1. Points pivots (swings)
            pivot_levels = self._detect_pivot_levels(highs, lows, closes, current_price, timeframe)
            all_levels.extend(pivot_levels)
            
            # 2. Niveaux de volume (POC, Value Areas)
            volume_levels = self._detect_volume_levels(closes, volumes, current_price, timeframe)
            all_levels.extend(volume_levels)
            
            # 3. Niveaux psychologiques
            psychological_levels = self._detect_psychological_levels(closes, current_price, timeframe)
            all_levels.extend(psychological_levels)
            
            # 4. Lignes de tendance
            trendline_levels = self._detect_trendlines(highs, lows, closes, current_price, timeframe)
            all_levels.extend(trendline_levels)
            
            # 5. Consolider et nettoyer les niveaux proches
            consolidated_levels = self._consolidate_levels(all_levels, current_price)
            
            # 6. Calculer la force finale et trier
            final_levels = self._calculate_final_strength(consolidated_levels, highs, lows, closes, volumes)
            
            # Si aucun niveau détecté, créer des niveaux basiques de fallback
            if not final_levels:
                logger.debug("Aucun niveau détecté, création de niveaux basiques")
                final_levels = self._create_basic_levels(highs, lows, closes, current_price, timeframe)
            
            # Trier par force (majeurs d'abord) puis par proximité
            final_levels.sort(key=lambda x: (
                -self._strength_to_number(x.strength),
                -x.confidence,
                abs(x.distance_from_price)
            ))
            
            # Limiter à 20 niveaux max pour performance
            return final_levels[:20]
            
        except Exception as e:
            logger.error(f"Erreur détection niveaux S/R: {e}")
            return []
    
    def _detect_pivot_levels(self, highs: np.ndarray, lows: np.ndarray, 
                           closes: np.ndarray, current_price: float, 
                           timeframe: str) -> List[PriceLevel]:
        """Détecte les points pivots (swing highs/lows)."""
        levels = []
        
        try:
            # Paramètres adaptatifs selon le timeframe
            if timeframe in ['1m', '5m']:
                prominence = np.std(closes[-50:]) * 0.5
                distance = 3
            else:
                prominence = np.std(closes[-50:]) * 1.0
                distance = 5
            
            # Détecter les pics (résistances potentielles)
            high_peaks, high_properties = find_peaks(
                highs,
                prominence=prominence,
                distance=distance,
                height=np.percentile(highs, 60)
            )
            
            # Détecter les creux (supports potentiels)
            low_peaks, low_properties = find_peaks(
                -lows,  # Inverser pour trouver les minimums
                prominence=prominence,
                distance=distance,
                height=-np.percentile(lows, 40)
            )
            
            # Traiter les résistances
            for peak_idx in high_peaks[-10:]:  # Derniers 10 pics
                if peak_idx < len(highs):
                    price = highs[peak_idx]
                    touches = self._count_touches(price, highs, lows, self.price_tolerance)
                    
                    if touches >= self.min_touches:
                        levels.append(PriceLevel(
                            price=float(price),
                            level_type=LevelType.RESISTANCE,
                            strength=LevelStrength.MODERATE,
                            confidence=min(touches * 15, 85),
                            touches=touches,
                            volume_strength=0.0,
                            distance_from_price=abs(price - current_price) / current_price,
                            last_test_age=len(closes) - peak_idx,
                            break_probability=self._calculate_break_probability(price, closes, True),
                            timeframe=timeframe
                        ))
            
            # Traiter les supports
            for peak_idx in low_peaks[-10:]:  # Derniers 10 creux
                if peak_idx < len(lows):
                    price = lows[peak_idx]
                    touches = self._count_touches(price, highs, lows, self.price_tolerance)
                    
                    if touches >= self.min_touches:
                        levels.append(PriceLevel(
                            price=float(price),
                            level_type=LevelType.SUPPORT,
                            strength=LevelStrength.MODERATE,
                            confidence=min(touches * 15, 85),
                            touches=touches,
                            volume_strength=0.0,
                            distance_from_price=abs(price - current_price) / current_price,
                            last_test_age=len(closes) - peak_idx,
                            break_probability=self._calculate_break_probability(price, closes, False),
                            timeframe=timeframe
                        ))
            
        except Exception as e:
            logger.warning(f"Erreur détection pivots: {e}")
        
        return levels
    
    def _detect_volume_levels(self, closes: np.ndarray, volumes: np.ndarray,
                            current_price: float, timeframe: str) -> List[PriceLevel]:
        """Détecte les niveaux basés sur le volume (POC, Value Areas)."""
        levels = []
        
        try:
            # Créer un profil de volume
            min_price = np.min(closes[-self.volume_window:])
            max_price = np.max(closes[-self.volume_window:])
            
            # Diviser en bins de prix
            n_bins = min(50, len(closes) // 4)
            price_bins = np.linspace(min_price, max_price, n_bins)
            volume_profile = np.zeros(n_bins - 1)
            
            # Calculer le volume pour chaque bin
            recent_closes = closes[-self.volume_window:]
            recent_volumes = volumes[-self.volume_window:]
            
            for i, (price, vol) in enumerate(zip(recent_closes, recent_volumes)):
                bin_idx = np.digitize(price, price_bins) - 1
                if 0 <= bin_idx < len(volume_profile):
                    volume_profile[bin_idx] += vol
            
            # Trouver le POC (Point of Control)
            poc_idx = np.argmax(volume_profile)
            poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2
            
            levels.append(PriceLevel(
                price=float(poc_price),
                level_type=LevelType.VOLUME_POC,
                strength=LevelStrength.STRONG,
                confidence=80.0,
                touches=1,  # Sera recalculé
                volume_strength=float(volume_profile[poc_idx] / np.sum(volume_profile)),
                distance_from_price=abs(poc_price - current_price) / current_price,
                last_test_age=0,
                break_probability=0.3,
                timeframe=timeframe
            ))
            
            # Calculer les Value Areas (70% du volume autour du POC)
            total_volume = np.sum(volume_profile)
            target_volume = total_volume * 0.7
            
            # Expansion depuis le POC
            va_volume = volume_profile[poc_idx]
            va_low_idx = poc_idx
            va_high_idx = poc_idx
            
            while va_volume < target_volume and (va_low_idx > 0 or va_high_idx < len(volume_profile) - 1):
                # Choisir la direction avec le plus de volume
                low_vol = volume_profile[va_low_idx - 1] if va_low_idx > 0 else 0
                high_vol = volume_profile[va_high_idx + 1] if va_high_idx < len(volume_profile) - 1 else 0
                
                if low_vol >= high_vol and va_low_idx > 0:
                    va_low_idx -= 1
                    va_volume += low_vol
                elif va_high_idx < len(volume_profile) - 1:
                    va_high_idx += 1
                    va_volume += high_vol
                else:
                    break
            
            # Value Area High
            if va_high_idx != poc_idx:
                vah_price = (price_bins[va_high_idx] + price_bins[va_high_idx + 1]) / 2
                levels.append(PriceLevel(
                    price=float(vah_price),
                    level_type=LevelType.VALUE_AREA_HIGH,
                    strength=LevelStrength.MODERATE,
                    confidence=65.0,
                    touches=1,
                    volume_strength=float(volume_profile[va_high_idx] / total_volume),
                    distance_from_price=abs(vah_price - current_price) / current_price,
                    last_test_age=0,
                    break_probability=0.4,
                    timeframe=timeframe
                ))
            
            # Value Area Low
            if va_low_idx != poc_idx:
                val_price = (price_bins[va_low_idx] + price_bins[va_low_idx + 1]) / 2
                levels.append(PriceLevel(
                    price=float(val_price),
                    level_type=LevelType.VALUE_AREA_LOW,
                    strength=LevelStrength.MODERATE,
                    confidence=65.0,
                    touches=1,
                    volume_strength=float(volume_profile[va_low_idx] / total_volume),
                    distance_from_price=abs(val_price - current_price) / current_price,
                    last_test_age=0,
                    break_probability=0.4,
                    timeframe=timeframe
                ))
            
        except Exception as e:
            logger.warning(f"Erreur détection niveaux volume: {e}")
        
        return levels
    
    def _detect_psychological_levels(self, closes: np.ndarray, current_price: float,
                                   timeframe: str) -> List[PriceLevel]:
        """Détecte les niveaux psychologiques (nombres ronds)."""
        levels = []
        
        try:
            price_range = np.max(closes[-50:]) - np.min(closes[-50:])
            avg_price = np.mean(closes[-20:])
            
            # Déterminer les pas selon le prix et la volatilité
            if avg_price < 1:
                steps = [0.01, 0.05, 0.1, 0.5]
            elif avg_price < 10:
                steps = [0.1, 0.5, 1.0, 5.0]
            elif avg_price < 100:
                steps = [1, 5, 10, 50]
            elif avg_price < 1000:
                steps = [10, 50, 100, 500]
            else:
                steps = [100, 500, 1000, 5000]
            
            # Choisir le pas approprié selon la volatilité
            volatility_ratio = price_range / avg_price
            if volatility_ratio > 0.1:  # Très volatil
                chosen_steps = steps[-2:]  # Grands pas
            elif volatility_ratio > 0.05:  # Modérément volatil
                chosen_steps = steps[1:3]  # Pas moyens
            else:  # Peu volatil
                chosen_steps = steps[:2]  # Petits pas
            
            # Générer les niveaux dans une fourchette raisonnable
            min_level = current_price * 0.8
            max_level = current_price * 1.2
            
            for step in chosen_steps:
                # Arrondir aux niveaux psychologiques
                start_level = int(min_level / step) * step
                level = start_level
                
                while level <= max_level:
                    if min_level <= level <= max_level and abs(level - current_price) / current_price > 0.005:
                        # Vérifier si le niveau a de l'historique
                        touches = self._count_psychological_touches(level, closes, step * 0.1)
                        
                        if touches > 0:  # Au moins une interaction historique
                            strength = LevelStrength.WEAK
                            confidence = 30 + min(touches * 10, 40)
                            
                            # Niveaux particulièrement importants
                            if level % (step * 10) == 0:  # Multiples de 10 fois le pas
                                strength = LevelStrength.MODERATE
                                confidence += 15
                            
                            level_type = LevelType.RESISTANCE if level > current_price else LevelType.SUPPORT
                            
                            levels.append(PriceLevel(
                                price=float(level),
                                level_type=level_type,
                                strength=strength,
                                confidence=confidence,
                                touches=touches,
                                volume_strength=0.2,  # Modéré pour psychologique
                                distance_from_price=abs(level - current_price) / current_price,
                                last_test_age=self._days_since_last_touch(level, closes, step * 0.1),
                                break_probability=0.5,  # Neutre
                                timeframe=timeframe
                            ))
                    
                    level += step
                    
        except Exception as e:
            logger.warning(f"Erreur détection niveaux psychologiques: {e}")
        
        return levels
    
    def _detect_trendlines(self, highs: np.ndarray, lows: np.ndarray, 
                          closes: np.ndarray, current_price: float,
                          timeframe: str) -> List[PriceLevel]:
        """Détecte les lignes de tendance dynamiques."""
        levels = []
        
        try:
            # Détecter tendance haussière (support dynamique)
            support_line = self._calculate_trendline(lows, trend_type='support')
            if support_line is not None:
                current_support = support_line['current_level']
                
                levels.append(PriceLevel(
                    price=float(current_support),
                    level_type=LevelType.TRENDLINE,
                    strength=LevelStrength.MODERATE if support_line['r_squared'] > 0.7 else LevelStrength.WEAK,
                    confidence=min(support_line['r_squared'] * 100, 90),
                    touches=support_line['touches'],
                    volume_strength=0.3,
                    distance_from_price=abs(current_support - current_price) / current_price,
                    last_test_age=support_line['last_test_age'],
                    break_probability=1 - support_line['r_squared'],
                    slope=support_line['slope'],
                    timeframe=timeframe
                ))
            
            # Détecter tendance baissière (résistance dynamique)
            resistance_line = self._calculate_trendline(highs, trend_type='resistance')
            if resistance_line is not None:
                current_resistance = resistance_line['current_level']
                
                levels.append(PriceLevel(
                    price=float(current_resistance),
                    level_type=LevelType.TRENDLINE,
                    strength=LevelStrength.MODERATE if resistance_line['r_squared'] > 0.7 else LevelStrength.WEAK,
                    confidence=min(resistance_line['r_squared'] * 100, 90),
                    touches=resistance_line['touches'],
                    volume_strength=0.3,
                    distance_from_price=abs(current_resistance - current_price) / current_price,
                    last_test_age=resistance_line['last_test_age'],
                    break_probability=1 - resistance_line['r_squared'],
                    slope=resistance_line['slope'],
                    timeframe=timeframe
                ))
                
        except Exception as e:
            logger.warning(f"Erreur détection trendlines: {e}")
        
        return levels
    
    def _consolidate_levels(self, levels: List[PriceLevel], current_price: float) -> List[PriceLevel]:
        """Consolide les niveaux proches pour éviter la redondance."""
        if not levels:
            return []
        
        # Trier par prix
        levels.sort(key=lambda x: x.price)
        
        consolidated = []
        i = 0
        
        while i < len(levels):
            level = levels[i]
            similar_levels = [level]
            
            # Trouver tous les niveaux similaires
            j = i + 1
            while j < len(levels):
                if abs(levels[j].price - level.price) / level.price <= self.price_tolerance:
                    similar_levels.append(levels[j])
                    j += 1
                else:
                    break
            
            # Consolider les niveaux similaires
            if len(similar_levels) > 1:
                consolidated_level = self._merge_levels(similar_levels)
                consolidated.append(consolidated_level)
            else:
                consolidated.append(level)
            
            i = j
        
        return consolidated
    
    def _merge_levels(self, levels: List[PriceLevel]) -> PriceLevel:
        """Fusionne plusieurs niveaux similaires en un seul."""
        # Prix pondéré par la confiance
        total_confidence = sum(level.confidence for level in levels)
        if total_confidence > 0:
            weighted_price = sum(level.price * level.confidence for level in levels) / total_confidence
        else:
            weighted_price = float(np.mean([level.price for level in levels]))
        
        # Prendre les meilleures caractéristiques
        max_confidence_level = max(levels, key=lambda x: x.confidence)
        total_touches = sum(level.touches for level in levels)
        max_volume_strength = max(level.volume_strength for level in levels)
        min_last_test_age = min(level.last_test_age for level in levels)
        
        # Force maximale
        strength_values = [self._strength_to_number(level.strength) for level in levels]
        max_strength_idx = np.argmax(strength_values)
        best_strength = levels[max_strength_idx].strength
        
        return PriceLevel(
            price=float(weighted_price),
            level_type=max_confidence_level.level_type,
            strength=best_strength,
            confidence=min(total_confidence / len(levels) + 10, 95),  # Bonus fusion
            touches=min(total_touches, 10),
            volume_strength=max_volume_strength,
            distance_from_price=max_confidence_level.distance_from_price,
            last_test_age=min_last_test_age,
            break_probability=float(np.mean([level.break_probability for level in levels])),
            slope=max_confidence_level.slope,
            timeframe=max_confidence_level.timeframe
        )
    
    def _calculate_final_strength(self, levels: List[PriceLevel],
                                 highs: np.ndarray, lows: np.ndarray,
                                 closes: np.ndarray, volumes: np.ndarray) -> List[PriceLevel]:
        """Calcule la force finale des niveaux."""
        for level in levels:
            # Recalculer les touches avec plus de précision
            level.touches = self._count_touches(level.price, highs, lows, self.price_tolerance)
            
            # Calculer la force du volume autour du niveau
            level.volume_strength = self._calculate_volume_at_level(
                level.price, closes, volumes, self.price_tolerance
            )
            
            # Ajuster la confiance basée sur les touches et le volume
            touch_bonus = min(level.touches * 5, 20)
            volume_bonus = level.volume_strength * 15
            age_penalty = min(level.last_test_age * 0.1, 10)
            
            level.confidence = min(level.confidence + touch_bonus + volume_bonus - age_penalty, 95)
            
            # Déterminer la force finale
            if level.touches >= 5 and level.confidence > 80:
                level.strength = LevelStrength.MAJOR
            elif level.touches >= 3 and level.confidence > 65:
                level.strength = LevelStrength.STRONG
            elif level.touches >= 2 and level.confidence > 45:
                level.strength = LevelStrength.MODERATE
            else:
                level.strength = LevelStrength.WEAK
        
        return levels
    
    # ============ Méthodes utilitaires ============
    
    def _count_touches(self, level: float, highs: np.ndarray, lows: np.ndarray, tolerance: float) -> int:
        """Compte le nombre de fois qu'un niveau a été testé."""
        touches = 0
        level_range = level * tolerance
        
        for high, low in zip(highs, lows):
            # Test du niveau (prix proche du niveau)
            if abs(high - level) <= level_range or abs(low - level) <= level_range:
                touches += 1
            # Crossing du niveau (prix traverse le niveau)
            elif low <= level <= high:
                touches += 1
        
        return touches
    
    def _count_psychological_touches(self, level: float, closes: np.ndarray, tolerance: float) -> int:
        """Compte les interactions avec un niveau psychologique."""
        touches = 0
        for close in closes:
            if abs(close - level) <= tolerance:
                touches += 1
        return touches
    
    def _days_since_last_touch(self, level: float, closes: np.ndarray, tolerance: float) -> int:
        """Calcule les jours depuis le dernier test du niveau."""
        for i in range(len(closes) - 1, -1, -1):
            if abs(closes[i] - level) <= tolerance:
                return len(closes) - 1 - i
        return len(closes)  # Jamais testé dans l'historique visible
    
    def _calculate_break_probability(self, level: float, closes: np.ndarray, is_resistance: bool) -> float:
        """Calcule la probabilité de cassure d'un niveau."""
        # Logique simplifiée - peut être améliorée avec ML
        recent_closes = closes[-10:]
        
        if is_resistance:
            approaches = sum(1 for close in recent_closes if close > level * 0.98)
        else:
            approaches = sum(1 for close in recent_closes if close < level * 1.02)
        
        # Plus d'approches = plus de probabilité de cassure
        base_prob = min(approaches * 0.1, 0.7)
        
        return base_prob
    
    def _calculate_volume_at_level(self, level: float, closes: np.ndarray, 
                                 volumes: np.ndarray, tolerance: float) -> float:
        """Calcule la force du volume autour d'un niveau."""
        level_range = level * tolerance
        total_volume = 0
        count = 0
        
        for close, volume in zip(closes, volumes):
            if abs(close - level) <= level_range:
                total_volume += volume
                count += 1
        
        if count == 0:
            return 0.0
        
        avg_level_volume = total_volume / count
        avg_total_volume = np.mean(volumes)
        
        return min(avg_level_volume / avg_total_volume if avg_total_volume > 0 else 0, 2.0)
    
    def _calculate_trendline(self, prices: np.ndarray, trend_type: str = 'support') -> Optional[Dict]:
        """Calcule une ligne de tendance par régression linéaire."""
        try:
            if len(prices) < 10:
                return None
            
            # Utiliser les 30 derniers points
            y = prices[-30:]
            x = np.arange(len(y))
            
            # Régression linéaire
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # R-squared pour la qualité de la ligne
            r_squared = r_value ** 2
            
            # Niveau actuel de la trendline
            current_level = slope * (len(y) - 1) + intercept
            
            # Compter les touches approximatives
            touches = 0
            tolerance = np.std(y) * 0.1
            
            for i, price in enumerate(y):
                trendline_value = slope * i + intercept
                if abs(price - trendline_value) <= tolerance:
                    touches += 1
            
            # Âge du dernier test
            last_test_age = 0
            for i in range(len(y) - 1, -1, -1):
                trendline_value = slope * i + intercept
                if abs(y[i] - trendline_value) <= tolerance:
                    last_test_age = len(y) - 1 - i
                    break
            
            return {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_squared,
                'current_level': current_level,
                'touches': touches,
                'last_test_age': last_test_age
            }
            
        except Exception as e:
            logger.warning(f"Erreur calcul trendline: {e}")
            return None
    
    def _create_basic_levels(self, highs: np.ndarray, lows: np.ndarray, 
                            closes: np.ndarray, current_price: float, 
                            timeframe: str) -> List[PriceLevel]:
        """
        Crée des niveaux basiques de fallback quand aucun niveau n'est détecté.
        Utilise les high/low récents et les moyennes.
        """
        levels = []
        
        try:
            # Récupérer les dernières périodes selon le timeframe
            lookback = min(100, len(closes))
            recent_highs = highs[-lookback:]
            recent_lows = lows[-lookback:]
            recent_closes = closes[-lookback:]
            
            # Niveau 1: Plus haut récent (résistance)
            recent_high = float(np.max(recent_highs))
            if recent_high > current_price:
                levels.append(PriceLevel(
                    price=recent_high,
                    level_type=LevelType.RESISTANCE,
                    strength=LevelStrength.MODERATE,
                    confidence=60.0,
                    touches=1,
                    volume_strength=0.0,
                    distance_from_price=abs(recent_high - current_price) / current_price,
                    last_test_age=0,
                    break_probability=0.5,
                    timeframe=timeframe
                ))
            
            # Niveau 2: Plus bas récent (support)
            recent_low = float(np.min(recent_lows))
            if recent_low < current_price:
                levels.append(PriceLevel(
                    price=recent_low,
                    level_type=LevelType.SUPPORT,
                    strength=LevelStrength.MODERATE,
                    confidence=60.0,
                    touches=1,
                    volume_strength=0.0,
                    distance_from_price=abs(recent_low - current_price) / current_price,
                    last_test_age=0,
                    break_probability=0.5,
                    timeframe=timeframe
                ))
            
            # Niveau 3: Moyenne mobile 50 périodes
            if len(recent_closes) >= 50:
                ma_50 = float(np.mean(recent_closes[-50:]))
                level_type = LevelType.SUPPORT if ma_50 < current_price else LevelType.RESISTANCE
                levels.append(PriceLevel(
                    price=ma_50,
                    level_type=level_type,
                    strength=LevelStrength.WEAK,
                    confidence=50.0,
                    touches=1,
                    volume_strength=0.0,
                    distance_from_price=abs(ma_50 - current_price) / current_price,
                    last_test_age=0,
                    break_probability=0.6,
                    timeframe=timeframe
                ))
            
            # Niveau 4: Moyenne mobile 20 périodes
            if len(recent_closes) >= 20:
                ma_20 = float(np.mean(recent_closes[-20:]))
                level_type = LevelType.SUPPORT if ma_20 < current_price else LevelType.RESISTANCE
                levels.append(PriceLevel(
                    price=ma_20,
                    level_type=level_type,
                    strength=LevelStrength.WEAK,
                    confidence=45.0,
                    touches=1,
                    volume_strength=0.0,
                    distance_from_price=abs(ma_20 - current_price) / current_price,
                    last_test_age=0,
                    break_probability=0.6,
                    timeframe=timeframe
                ))
                
            # Niveau 5 & 6: High et Low de la dernière période de 24h environ
            if lookback >= 288:  # ~24h en 5m
                day_high = float(np.max(recent_highs[-288:]))
                day_low = float(np.min(recent_lows[-288:]))
                
                if day_high > current_price and day_high != recent_high:
                    levels.append(PriceLevel(
                        price=day_high,
                        level_type=LevelType.RESISTANCE,
                        strength=LevelStrength.MODERATE,
                        confidence=55.0,
                        touches=1,
                        volume_strength=0.0,
                        distance_from_price=abs(day_high - current_price) / current_price,
                        last_test_age=0,
                        break_probability=0.5,
                        timeframe=timeframe
                    ))
                
                if day_low < current_price and day_low != recent_low:
                    levels.append(PriceLevel(
                        price=day_low,
                        level_type=LevelType.SUPPORT,
                        strength=LevelStrength.MODERATE,
                        confidence=55.0,
                        touches=1,
                        volume_strength=0.0,
                        distance_from_price=abs(day_low - current_price) / current_price,
                        last_test_age=0,
                        break_probability=0.5,
                        timeframe=timeframe
                    ))
                    
        except Exception as e:
            logger.warning(f"Erreur création niveaux basiques: {e}")
        
        return levels
    
    def _strength_to_number(self, strength: LevelStrength) -> int:
        """Convertit la force en nombre pour tri."""
        mapping = {
            LevelStrength.WEAK: 1,
            LevelStrength.MODERATE: 2,
            LevelStrength.STRONG: 3,
            LevelStrength.MAJOR: 4
        }
        return mapping.get(strength, 1)
    
    def get_nearest_levels(self, levels: List[PriceLevel], current_price: float,
                          max_distance: float = 0.05, max_count: int = 5) -> List[PriceLevel]:
        """
        Retourne les niveaux les plus proches du prix actuel.
        
        Args:
            levels: Liste des niveaux détectés
            current_price: Prix actuel
            max_distance: Distance maximale (en %)
            max_count: Nombre maximum de niveaux
            
        Returns:
            Niveaux les plus proches, triés par distance
        """
        nearby_levels = []
        
        for level in levels:
            distance = abs(level.price - current_price) / current_price
            if distance <= max_distance:
                level.distance_from_price = distance
                nearby_levels.append(level)
        
        # Trier par distance puis par force
        nearby_levels.sort(key=lambda x: (x.distance_from_price, -self._strength_to_number(x.strength)))
        
        return nearby_levels[:max_count]
    
    def get_key_levels_summary(self, levels: List[PriceLevel], current_price: float) -> Dict:
        """
        Retourne un résumé des niveaux clés.
        
        Returns:
            Dictionnaire avec résistances et supports principaux
        """
        supports = [l for l in levels if l.price < current_price and l.level_type in [LevelType.SUPPORT, LevelType.TRENDLINE]]
        resistances = [l for l in levels if l.price > current_price and l.level_type in [LevelType.RESISTANCE, LevelType.TRENDLINE]]

        # Trier par DISTANCE d'abord (scalping nécessite niveaux les plus proches)
        # Puis par force comme critère secondaire
        supports.sort(key=lambda x: (x.distance_from_price, -self._strength_to_number(x.strength)))
        resistances.sort(key=lambda x: (x.distance_from_price, -self._strength_to_number(x.strength)))
        
        return {
            'nearest_support': supports[0].to_dict() if supports else None,
            'nearest_resistance': resistances[0].to_dict() if resistances else None,
            'strong_supports': [s.to_dict() for s in supports if s.strength in [LevelStrength.STRONG, LevelStrength.MAJOR]][:3],
            'strong_resistances': [r.to_dict() for r in resistances if r.strength in [LevelStrength.STRONG, LevelStrength.MAJOR]][:3],
            'total_levels': len(levels),
            'current_price': current_price
        }