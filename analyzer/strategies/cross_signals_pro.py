"""
Cross Signals Module

This module provides detection of various crossing signals:
- EMA crosses (fast/slow, price/EMA)
- MACD crosses (line/signal, zero line)
- Momentum crosses
- RSI level crosses
- Multi-timeframe cross confirmation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CrossType(Enum):
    """Types de croisements."""
    EMA_CROSS_BULL = "ema_cross_bull"
    EMA_CROSS_BEAR = "ema_cross_bear"
    PRICE_EMA_CROSS_BULL = "price_ema_cross_bull"
    PRICE_EMA_CROSS_BEAR = "price_ema_cross_bear"
    MACD_CROSS_BULL = "macd_cross_bull"
    MACD_CROSS_BEAR = "macd_cross_bear"
    MACD_ZERO_CROSS_BULL = "macd_zero_cross_bull"
    MACD_ZERO_CROSS_BEAR = "macd_zero_cross_bear"
    RSI_OVERSOLD_EXIT = "rsi_oversold_exit"
    RSI_OVERBOUGHT_EXIT = "rsi_overbought_exit"
    MOMENTUM_CROSS_BULL = "momentum_cross_bull"
    MOMENTUM_CROSS_BEAR = "momentum_cross_bear"


class CrossStrength(Enum):
    """Force du signal de cross."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class CrossSignal:
    """Signal de croisement détecté."""
    cross_type: CrossType
    strength: CrossStrength
    confidence: float  # 0-100
    price_at_cross: float
    bars_ago: int  # Nombre de barres depuis le cross
    momentum_confirmation: bool
    volume_confirmation: bool
    trend_alignment: bool  # Aligné avec la tendance principale
    entry_quality: str  # excellent, good, fair, poor
    target_projection: Optional[float] = None
    stop_level: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire pour export."""
        return {
            'cross_type': self.cross_type.value,
            'strength': self.strength.value,
            'confidence': self.confidence,
            'price_at_cross': self.price_at_cross,
            'bars_ago': self.bars_ago,
            'momentum_confirmation': self.momentum_confirmation,
            'volume_confirmation': self.volume_confirmation,
            'trend_alignment': self.trend_alignment,
            'entry_quality': self.entry_quality,
            'target_projection': self.target_projection,
            'stop_level': self.stop_level,
            'risk_reward_ratio': self.risk_reward_ratio,
            'timestamp': self.timestamp
        }


class CrossSignalDetector:
    """
    Détecteur de signaux de croisement.
    
    Détecte et analyse les croisements entre différents indicateurs:
    - Croisements EMA (rapide/lent, prix/EMA)
    - Croisements MACD (ligne/signal, ligne zéro)
    - Croisements de momentum
    - Sorties de niveaux RSI
    - Confirmation multi-indicateurs
    """
    
    def __init__(self,
                 ema_fast: int = 7,
                 ema_slow: int = 26,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70,
                 min_volume_ratio: float = 1.2):
        """
        Args:
            ema_fast: Période EMA rapide
            ema_slow: Période EMA lente
            rsi_oversold: Niveau RSI survendu
            rsi_overbought: Niveau RSI suracheté
            min_volume_ratio: Ratio minimum pour confirmation volume
        """
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.min_volume_ratio = min_volume_ratio
    
    def detect_all_crosses(self,
                          highs: Union[List[float], np.ndarray],
                          lows: Union[List[float], np.ndarray],
                          closes: Union[List[float], np.ndarray],
                          volumes: Union[List[float], np.ndarray],
                          max_bars_back: int = 5) -> List[CrossSignal]:
        """
        Détecte tous les croisements récents.
        
        Args:
            highs: Prix hauts
            lows: Prix bas
            closes: Prix de clôture
            volumes: Volumes
            max_bars_back: Nombre max de barres à analyser
            
        Returns:
            Liste des signaux de croisement détectés
        """
        try:
            closes = np.array(closes, dtype=float)
            volumes = np.array(volumes, dtype=float)
            
            if len(closes) < max(self.ema_fast, self.ema_slow) + 10:
                return []
            
            signals = []
            
            # 1. Croisements EMA
            ema_signals = self._detect_ema_crosses(closes, volumes, max_bars_back)
            signals.extend(ema_signals)
            
            # 2. Croisements Prix/EMA
            price_ema_signals = self._detect_price_ema_crosses(closes, volumes, max_bars_back)
            signals.extend(price_ema_signals)
            
            # 3. Croisements MACD
            macd_signals = self._detect_macd_crosses(closes, volumes, max_bars_back)
            signals.extend(macd_signals)
            
            # 4. Sorties de niveaux RSI
            rsi_signals = self._detect_rsi_level_crosses(closes, volumes, max_bars_back)
            signals.extend(rsi_signals)
            
            # 5. Croisements de momentum
            momentum_signals = self._detect_momentum_crosses(closes, volumes, max_bars_back)
            signals.extend(momentum_signals)
            
            # Trier par force et récence
            signals.sort(key=lambda x: (-self._strength_to_number(x.strength), x.bars_ago))
            
            return signals[:10]  # Top 10 signaux
            
        except Exception as e:
            logger.error(f"Erreur détection croisements: {e}")
            return []
    
    def _detect_ema_crosses(self, closes: np.ndarray, volumes: np.ndarray, 
                           max_bars_back: int) -> List[CrossSignal]:
        """Détecte les croisements entre EMAs."""
        from ...market_analyzer.indicators.trend.moving_averages import calculate_ema_series
        
        signals = []
        
        try:
            # Calculer les EMAs
            ema_fast_series = calculate_ema_series(closes, self.ema_fast)
            ema_slow_series = calculate_ema_series(closes, self.ema_slow)
            
            # Trouver les valeurs valides
            valid_data = []
            for i in range(len(closes)):
                if (ema_fast_series[i] is not None and 
                    ema_slow_series[i] is not None):
                    valid_data.append({
                        'index': i,
                        'price': closes[i],
                        'volume': volumes[i],
                        'ema_fast': ema_fast_series[i],
                        'ema_slow': ema_slow_series[i]
                    })
            
            if len(valid_data) < max_bars_back + 2:
                return signals
            
            # Chercher les croisements récents
            for i in range(len(valid_data) - max_bars_back, len(valid_data) - 1):
                if i <= 0:
                    continue
                
                current = valid_data[i]
                previous = valid_data[i - 1]
                
                # Croisement haussier (EMA rapide croise au-dessus EMA lente)
                if (previous['ema_fast'] <= previous['ema_slow'] and 
                    current['ema_fast'] > current['ema_slow']):
                    
                    signal = self._create_ema_cross_signal(
                        valid_data, i, CrossType.EMA_CROSS_BULL, max_bars_back
                    )
                    signals.append(signal)
                
                # Croisement baissier (EMA rapide croise en-dessous EMA lente)
                elif (previous['ema_fast'] >= previous['ema_slow'] and 
                      current['ema_fast'] < current['ema_slow']):
                    
                    signal = self._create_ema_cross_signal(
                        valid_data, i, CrossType.EMA_CROSS_BEAR, max_bars_back
                    )
                    signals.append(signal)
            
        except Exception as e:
            logger.warning(f"Erreur détection croisements EMA: {e}")
        
        return signals
    
    def _detect_price_ema_crosses(self, closes: np.ndarray, volumes: np.ndarray,
                                 max_bars_back: int) -> List[CrossSignal]:
        """Détecte les croisements Prix/EMA."""
        from ...market_analyzer.indicators.trend.moving_averages import calculate_ema_series
        
        signals = []
        
        try:
            # Utiliser EMA moyenne comme référence
            ema_ref_series = calculate_ema_series(closes, (self.ema_fast + self.ema_slow) // 2)
            
            # Trouver les valeurs valides
            valid_data = []
            for i in range(len(closes)):
                if ema_ref_series[i] is not None:
                    valid_data.append({
                        'index': i,
                        'price': closes[i],
                        'volume': volumes[i],
                        'ema_ref': ema_ref_series[i]
                    })
            
            if len(valid_data) < max_bars_back + 2:
                return signals
            
            # Chercher les croisements
            for i in range(len(valid_data) - max_bars_back, len(valid_data) - 1):
                if i <= 0:
                    continue
                
                current = valid_data[i]
                previous = valid_data[i - 1]
                
                # Prix croise au-dessus EMA
                if (previous['price'] <= previous['ema_ref'] and 
                    current['price'] > current['ema_ref']):
                    
                    signal = self._create_price_ema_cross_signal(
                        valid_data, i, CrossType.PRICE_EMA_CROSS_BULL
                    )
                    signals.append(signal)
                
                # Prix croise en-dessous EMA
                elif (previous['price'] >= previous['ema_ref'] and 
                      current['price'] < current['ema_ref']):
                    
                    signal = self._create_price_ema_cross_signal(
                        valid_data, i, CrossType.PRICE_EMA_CROSS_BEAR
                    )
                    signals.append(signal)
            
        except Exception as e:
            logger.warning(f"Erreur détection croisements Prix/EMA: {e}")
        
        return signals
    
    def _detect_macd_crosses(self, closes: np.ndarray, volumes: np.ndarray,
                            max_bars_back: int) -> List[CrossSignal]:
        """Détecte les croisements MACD."""
        from ...market_analyzer.indicators.trend.macd import calculate_macd_series
        
        signals = []
        
        try:
            macd_data = calculate_macd_series(closes)
            macd_line = macd_data['macd_line']
            macd_signal = macd_data['macd_signal']
            
            # Trouver les valeurs valides
            valid_data = []
            for i in range(len(closes)):
                if (macd_line[i] is not None and 
                    macd_signal[i] is not None):
                    valid_data.append({
                        'index': i,
                        'price': closes[i],
                        'volume': volumes[i],
                        'macd_line': macd_line[i],
                        'macd_signal': macd_signal[i]
                    })
            
            if len(valid_data) < max_bars_back + 2:
                return signals
            
            # Chercher les croisements MACD
            for i in range(len(valid_data) - max_bars_back, len(valid_data) - 1):
                if i <= 0:
                    continue
                
                current = valid_data[i]
                previous = valid_data[i - 1]
                
                # MACD croise au-dessus signal
                if (previous['macd_line'] <= previous['macd_signal'] and 
                    current['macd_line'] > current['macd_signal']):
                    
                    signal = self._create_macd_cross_signal(
                        valid_data, i, CrossType.MACD_CROSS_BULL
                    )
                    signals.append(signal)
                
                # MACD croise en-dessous signal
                elif (previous['macd_line'] >= previous['macd_signal'] and 
                      current['macd_line'] < current['macd_signal']):
                    
                    signal = self._create_macd_cross_signal(
                        valid_data, i, CrossType.MACD_CROSS_BEAR
                    )
                    signals.append(signal)
                
                # Croisements ligne zéro
                if previous['macd_line'] <= 0 and current['macd_line'] > 0:
                    signal = self._create_macd_cross_signal(
                        valid_data, i, CrossType.MACD_ZERO_CROSS_BULL
                    )
                    signals.append(signal)
                elif previous['macd_line'] >= 0 and current['macd_line'] < 0:
                    signal = self._create_macd_cross_signal(
                        valid_data, i, CrossType.MACD_ZERO_CROSS_BEAR
                    )
                    signals.append(signal)
            
        except Exception as e:
            logger.warning(f"Erreur détection croisements MACD: {e}")
        
        return signals
    
    def _detect_rsi_level_crosses(self, closes: np.ndarray, volumes: np.ndarray,
                                 max_bars_back: int) -> List[CrossSignal]:
        """Détecte les sorties de niveaux RSI."""
        from ...market_analyzer.indicators.momentum.rsi import calculate_rsi_series
        
        signals = []
        
        try:
            rsi_series = calculate_rsi_series(closes)
            
            # Trouver les valeurs valides
            valid_data = []
            for i in range(len(closes)):
                if rsi_series[i] is not None:
                    valid_data.append({
                        'index': i,
                        'price': closes[i],
                        'volume': volumes[i],
                        'rsi': rsi_series[i]
                    })
            
            if len(valid_data) < max_bars_back + 2:
                return signals
            
            # Chercher les sorties de niveaux
            for i in range(len(valid_data) - max_bars_back, len(valid_data) - 1):
                if i <= 0:
                    continue
                
                current = valid_data[i]
                previous = valid_data[i - 1]
                
                # Sortie de survendu (RSI > 30)
                if (previous['rsi'] <= self.rsi_oversold and 
                    current['rsi'] > self.rsi_oversold):
                    
                    signal = self._create_rsi_level_cross_signal(
                        valid_data, i, CrossType.RSI_OVERSOLD_EXIT
                    )
                    signals.append(signal)
                
                # Sortie de suracheté (RSI < 70)
                elif (previous['rsi'] >= self.rsi_overbought and 
                      current['rsi'] < self.rsi_overbought):
                    
                    signal = self._create_rsi_level_cross_signal(
                        valid_data, i, CrossType.RSI_OVERBOUGHT_EXIT
                    )
                    signals.append(signal)
            
        except Exception as e:
            logger.warning(f"Erreur détection croisements RSI: {e}")
        
        return signals
    
    def _detect_momentum_crosses(self, closes: np.ndarray, volumes: np.ndarray,
                                max_bars_back: int) -> List[CrossSignal]:
        """Détecte les croisements de momentum."""
        signals = []
        
        try:
            if len(closes) < 20:
                return signals
            
            # Calculer momentum (ROC sur 10 périodes)
            momentum = []
            for i in range(10, len(closes)):
                roc = (closes[i] - closes[i-10]) / closes[i-10] * 100
                momentum.append(roc)
            
            if len(momentum) < max_bars_back + 2:
                return signals
            
            # Chercher les croisements de ligne zéro
            for i in range(len(momentum) - max_bars_back, len(momentum) - 1):
                if i <= 0:
                    continue
                
                current_momentum = momentum[i]
                previous_momentum = momentum[i - 1]
                actual_index = i + 10  # Ajuster pour l'offset
                
                # Momentum croise au-dessus de zéro
                if previous_momentum <= 0 and current_momentum > 0:
                    signal = self._create_momentum_cross_signal(
                        closes, volumes, actual_index, CrossType.MOMENTUM_CROSS_BULL
                    )
                    signals.append(signal)
                
                # Momentum croise en-dessous de zéro
                elif previous_momentum >= 0 and current_momentum < 0:
                    signal = self._create_momentum_cross_signal(
                        closes, volumes, actual_index, CrossType.MOMENTUM_CROSS_BEAR
                    )
                    signals.append(signal)
            
        except Exception as e:
            logger.warning(f"Erreur détection croisements momentum: {e}")
        
        return signals
    
    def _create_ema_cross_signal(self, valid_data: List[Dict], cross_index: int,
                                cross_type: CrossType, max_bars_back: int) -> CrossSignal:
        """Crée un signal de croisement EMA."""
        cross_data = valid_data[cross_index]
        current_data = valid_data[-1]
        
        bars_ago = len(valid_data) - 1 - cross_index
        
        # Évaluer la force du signal
        ema_separation = abs(cross_data['ema_fast'] - cross_data['ema_slow']) / cross_data['ema_slow']
        
        if ema_separation > 0.02:
            strength = CrossStrength.STRONG
        elif ema_separation > 0.01:
            strength = CrossStrength.MODERATE
        else:
            strength = CrossStrength.WEAK
        
        # Confirmations
        volume_confirmation = self._check_volume_confirmation(valid_data, cross_index)
        momentum_confirmation = self._check_momentum_confirmation(valid_data, cross_index, cross_type)
        trend_alignment = self._check_trend_alignment(valid_data, cross_index, cross_type)
        
        # Confiance
        confidence = 50
        if volume_confirmation:
            confidence += 20
        if momentum_confirmation:
            confidence += 15
        if trend_alignment:
            confidence += 15
        
        # Qualité d'entrée
        entry_quality = self._assess_entry_quality(confidence, strength)
        
        # Projections
        target, stop = self._calculate_ema_projections(valid_data, cross_index, cross_type)
        
        return CrossSignal(
            cross_type=cross_type,
            strength=strength,
            confidence=min(confidence, 100),
            price_at_cross=cross_data['price'],
            bars_ago=bars_ago,
            momentum_confirmation=momentum_confirmation,
            volume_confirmation=volume_confirmation,
            trend_alignment=trend_alignment,
            entry_quality=entry_quality,
            target_projection=target,
            stop_level=stop,
            risk_reward_ratio=self._calculate_risk_reward(cross_data['price'], target, stop) if target and stop else None
        )
    
    def _create_price_ema_cross_signal(self, valid_data: List[Dict], cross_index: int,
                                      cross_type: CrossType) -> CrossSignal:
        """Crée un signal de croisement Prix/EMA."""
        cross_data = valid_data[cross_index]
        bars_ago = len(valid_data) - 1 - cross_index
        
        # Force basée sur la distance du prix à l'EMA
        distance = abs(cross_data['price'] - cross_data['ema_ref']) / cross_data['ema_ref']
        
        if distance > 0.015:
            strength = CrossStrength.STRONG
        elif distance > 0.01:
            strength = CrossStrength.MODERATE
        else:
            strength = CrossStrength.WEAK
        
        # Confirmations
        volume_confirmation = self._check_volume_confirmation(valid_data, cross_index)
        momentum_confirmation = True  # Prix/EMA est déjà un signal de momentum
        trend_alignment = self._check_trend_alignment(valid_data, cross_index, cross_type)
        
        confidence = 60  # Base plus élevée pour prix/EMA
        if volume_confirmation:
            confidence += 15
        if trend_alignment:
            confidence += 15
        
        entry_quality = self._assess_entry_quality(confidence, strength)
        
        return CrossSignal(
            cross_type=cross_type,
            strength=strength,
            confidence=min(confidence, 100),
            price_at_cross=cross_data['price'],
            bars_ago=bars_ago,
            momentum_confirmation=momentum_confirmation,
            volume_confirmation=volume_confirmation,
            trend_alignment=trend_alignment,
            entry_quality=entry_quality
        )
    
    def _create_macd_cross_signal(self, valid_data: List[Dict], cross_index: int,
                                 cross_type: CrossType) -> CrossSignal:
        """Crée un signal de croisement MACD."""
        cross_data = valid_data[cross_index]
        bars_ago = len(valid_data) - 1 - cross_index
        
        # Force basée sur la valeur MACD
        macd_strength = abs(cross_data['macd_line'])
        
        if macd_strength > 0.02:
            strength = CrossStrength.STRONG
        elif macd_strength > 0.01:
            strength = CrossStrength.MODERATE
        else:
            strength = CrossStrength.WEAK
        
        # Bonus pour croisements ligne zéro
        if 'ZERO' in cross_type.value:
            confidence = 70
        else:
            confidence = 55
        
        volume_confirmation = self._check_volume_confirmation(valid_data, cross_index)
        momentum_confirmation = True  # MACD est un indicateur de momentum
        trend_alignment = self._check_trend_alignment(valid_data, cross_index, cross_type)
        
        if volume_confirmation:
            confidence += 15
        if trend_alignment:
            confidence += 10
        
        entry_quality = self._assess_entry_quality(confidence, strength)
        
        return CrossSignal(
            cross_type=cross_type,
            strength=strength,
            confidence=min(confidence, 100),
            price_at_cross=cross_data['price'],
            bars_ago=bars_ago,
            momentum_confirmation=momentum_confirmation,
            volume_confirmation=volume_confirmation,
            trend_alignment=trend_alignment,
            entry_quality=entry_quality
        )
    
    def _create_rsi_level_cross_signal(self, valid_data: List[Dict], cross_index: int,
                                      cross_type: CrossType) -> CrossSignal:
        """Crée un signal de sortie de niveau RSI."""
        cross_data = valid_data[cross_index]
        bars_ago = len(valid_data) - 1 - cross_index
        
        # Force basée sur l'extrême RSI précédent
        if cross_type == CrossType.RSI_OVERSOLD_EXIT:
            # Chercher le RSI minimum récent
            min_rsi = min(valid_data[max(0, cross_index-5):cross_index+1], 
                         key=lambda x: x['rsi'])['rsi']
            if min_rsi < 20:
                strength = CrossStrength.STRONG
            elif min_rsi < 25:
                strength = CrossStrength.MODERATE
            else:
                strength = CrossStrength.WEAK
        else:
            # Chercher le RSI maximum récent
            max_rsi = max(valid_data[max(0, cross_index-5):cross_index+1], 
                         key=lambda x: x['rsi'])['rsi']
            if max_rsi > 80:
                strength = CrossStrength.STRONG
            elif max_rsi > 75:
                strength = CrossStrength.MODERATE
            else:
                strength = CrossStrength.WEAK
        
        confidence = 65  # RSI levels ont une bonne fiabilité
        volume_confirmation = self._check_volume_confirmation(valid_data, cross_index)
        momentum_confirmation = True  # RSI est un indicateur de momentum
        trend_alignment = self._check_trend_alignment(valid_data, cross_index, cross_type)
        
        if volume_confirmation:
            confidence += 10
        if trend_alignment:
            confidence += 15
        
        entry_quality = self._assess_entry_quality(confidence, strength)
        
        return CrossSignal(
            cross_type=cross_type,
            strength=strength,
            confidence=min(confidence, 100),
            price_at_cross=cross_data['price'],
            bars_ago=bars_ago,
            momentum_confirmation=momentum_confirmation,
            volume_confirmation=volume_confirmation,
            trend_alignment=trend_alignment,
            entry_quality=entry_quality
        )
    
    def _create_momentum_cross_signal(self, closes: np.ndarray, volumes: np.ndarray,
                                     cross_index: int, cross_type: CrossType) -> CrossSignal:
        """Crée un signal de croisement momentum."""
        bars_ago = len(closes) - 1 - cross_index
        price_at_cross = closes[cross_index]
        
        # Force basée sur la vitesse du changement
        if cross_index >= 5:
            price_change = abs(closes[cross_index] - closes[cross_index-5]) / closes[cross_index-5]
            if price_change > 0.05:
                strength = CrossStrength.STRONG
            elif price_change > 0.03:
                strength = CrossStrength.MODERATE
            else:
                strength = CrossStrength.WEAK
        else:
            strength = CrossStrength.WEAK
        
        confidence = 50
        
        # Vérification volume
        if cross_index < len(volumes):
            avg_volume = np.mean(volumes[max(0, cross_index-10):cross_index])
            volume_confirmation = volumes[cross_index] > avg_volume * self.min_volume_ratio
            if volume_confirmation:
                confidence += 20
        else:
            volume_confirmation = False
        
        momentum_confirmation = True
        trend_alignment = True  # Assume alignment for momentum cross
        
        entry_quality = self._assess_entry_quality(confidence, strength)
        
        return CrossSignal(
            cross_type=cross_type,
            strength=strength,
            confidence=confidence,
            price_at_cross=float(price_at_cross),
            bars_ago=bars_ago,
            momentum_confirmation=momentum_confirmation,
            volume_confirmation=volume_confirmation,
            trend_alignment=trend_alignment,
            entry_quality=entry_quality
        )
    
    def _check_volume_confirmation(self, valid_data: List[Dict], cross_index: int) -> bool:
        """Vérifie la confirmation de volume."""
        if cross_index < 10:
            return False
        
        cross_volume = valid_data[cross_index]['volume']
        avg_volume = np.mean([d['volume'] for d in valid_data[cross_index-10:cross_index]])
        
        return cross_volume > avg_volume * self.min_volume_ratio
    
    def _check_momentum_confirmation(self, valid_data: List[Dict], cross_index: int,
                                    cross_type: CrossType) -> bool:
        """Vérifie la confirmation de momentum."""
        if cross_index < 5:
            return False
        
        current_price = valid_data[cross_index]['price']
        past_price = valid_data[cross_index - 5]['price']
        momentum = (current_price - past_price) / past_price
        
        if 'BULL' in cross_type.value:
            return momentum > 0.01  # 1% momentum haussier
        else:
            return momentum < -0.01  # 1% momentum baissier
    
    def _check_trend_alignment(self, valid_data: List[Dict], cross_index: int,
                              cross_type: CrossType) -> bool:
        """Vérifie l'alignement avec la tendance."""
        if cross_index < 10:
            return False
        
        # Tendance basée sur les 10 dernières périodes
        recent_prices = [d['price'] for d in valid_data[cross_index-10:cross_index+1]]
        trend_slope = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        
        if 'BULL' in cross_type.value or 'OVERSOLD_EXIT' in cross_type.value:
            return trend_slope > 0
        else:
            return trend_slope < 0
    
    def _assess_entry_quality(self, confidence: float, strength: CrossStrength) -> str:
        """Évalue la qualité de l'entrée."""
        if confidence > 80 and strength in [CrossStrength.STRONG, CrossStrength.VERY_STRONG]:
            return "excellent"
        elif confidence > 70 and strength != CrossStrength.WEAK:
            return "good"
        elif confidence > 60:
            return "fair"
        else:
            return "poor"
    
    def _calculate_ema_projections(self, valid_data: List[Dict], cross_index: int,
                                  cross_type: CrossType) -> Tuple[Optional[float], Optional[float]]:
        """Calcule les projections pour croisements EMA."""
        cross_data = valid_data[cross_index]
        price = cross_data['price']
        
        # Distance entre EMAs comme base de projection
        ema_distance = abs(cross_data['ema_fast'] - cross_data['ema_slow'])
        
        if 'BULL' in cross_type.value:
            target = price + ema_distance * 2  # Projection 2x distance EMA
            stop = min(cross_data['ema_fast'], cross_data['ema_slow']) * 0.995  # Sous EMA support
        else:
            target = price - ema_distance * 2
            stop = max(cross_data['ema_fast'], cross_data['ema_slow']) * 1.005  # Au-dessus EMA résistance
        
        return target, stop
    
    def _calculate_risk_reward(self, entry: float, target: Optional[float], 
                              stop: Optional[float]) -> Optional[float]:
        """Calcule le ratio risque/récompense."""
        if target is None or stop is None:
            return None
        
        risk = abs(entry - stop)
        reward = abs(target - entry)
        
        if risk <= 0:
            return None
        
        return reward / risk
    
    def _strength_to_number(self, strength: CrossStrength) -> int:
        """Convertit la force en nombre."""
        mapping = {
            CrossStrength.WEAK: 1,
            CrossStrength.MODERATE: 2,
            CrossStrength.STRONG: 3,
            CrossStrength.VERY_STRONG: 4
        }
        return mapping.get(strength, 1)
    
    def get_best_cross_signal(self, signals: List[CrossSignal]) -> Optional[CrossSignal]:
        """Retourne le meilleur signal de croisement."""
        if not signals:
            return None
        
        # Filtrer les signaux de qualité acceptable
        good_signals = [s for s in signals if s.entry_quality in ['good', 'excellent']]
        
        if good_signals:
            # Trier par confiance et force
            best = max(good_signals, key=lambda x: (x.confidence, self._strength_to_number(x.strength)))
            return best
        
        # Fallback au meilleur signal disponible
        return max(signals, key=lambda x: x.confidence)
    
    def get_cross_summary(self, signals: List[CrossSignal]) -> Dict:
        """Retourne un résumé des signaux de croisement."""
        if not signals:
            return {
                'total_signals': 0,
                'bullish_signals': 0,
                'bearish_signals': 0,
                'best_signal': None,
                'signal_types': {}
            }
        
        bullish_types = [
            'ema_cross_bull', 'price_ema_cross_bull', 'macd_cross_bull',
            'macd_zero_cross_bull', 'rsi_oversold_exit', 'momentum_cross_bull'
        ]
        
        bullish_count = sum(1 for s in signals if s.cross_type.value in bullish_types)
        bearish_count = len(signals) - bullish_count
        
        signal_types = {}
        for signal in signals:
            signal_type = signal.cross_type.value
            signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
        
        best_signal = self.get_best_cross_signal(signals)
        
        return {
            'total_signals': len(signals),
            'bullish_signals': bullish_count,
            'bearish_signals': bearish_count,
            'best_signal': best_signal.to_dict() if best_signal else None,
            'signal_types': signal_types
        }