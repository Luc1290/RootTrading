"""
Multi-Timeframe Analyzer

This module aggregates signals and trends across multiple timeframes to provide:
- Trend alignment across timeframes
- Signal strength confluence
- Multi-timeframe support/resistance levels
- Risk assessment based on timeframe divergence
- Entry/exit timing optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, NamedTuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TimeframeType(Enum):
    """Types de timeframes supportés."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


class SignalStrength(Enum):
    """Force du signal multi-timeframe."""

    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


class TrendAlignment(Enum):
    """Alignement des tendances."""

    FULLY_ALIGNED_BULL = "fully_aligned_bull"
    FULLY_ALIGNED_BEAR = "fully_aligned_bear"
    MOSTLY_ALIGNED_BULL = "mostly_aligned_bull"
    MOSTLY_ALIGNED_BEAR = "mostly_aligned_bear"
    MIXED = "mixed"
    CONFLICTING = "conflicting"


@dataclass
class TimeframeSignal:
    """Signal d'un timeframe spécifique."""

    timeframe: TimeframeType
    regime_type: str
    trend_direction: str  # bullish, bearish, neutral
    trend_strength: float  # 0-100
    momentum_direction: str
    momentum_strength: float
    volatility_level: str  # low, normal, high
    support_levels: List[float]
    resistance_levels: List[float]
    volume_profile: str
    confidence: float
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convertit en dictionnaire pour export."""
        return {
            "timeframe": self.timeframe.value,
            "regime_type": self.regime_type,
            "trend_direction": self.trend_direction,
            "trend_strength": self.trend_strength,
            "momentum_direction": self.momentum_direction,
            "momentum_strength": self.momentum_strength,
            "volatility_level": self.volatility_level,
            "support_levels": self.support_levels,
            "resistance_levels": self.resistance_levels,
            "volume_profile": self.volume_profile,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


@dataclass
class MultiTimeframeAnalysis:
    """Analyse multi-timeframe complète."""

    primary_trend: str  # Direction de tendance dominante
    trend_alignment: TrendAlignment
    signal_strength: SignalStrength
    confidence: float  # 0-100
    risk_level: str  # low, medium, high
    key_support_levels: List[float]  # Niveaux confirmés sur plusieurs TF
    key_resistance_levels: List[float]
    entry_zones: List[Dict]  # Zones d'entrée optimales
    exit_zones: List[Dict]  # Zones de sortie
    timeframe_signals: Dict[str, TimeframeSignal]
    regime_consensus: str  # Régime dominant
    momentum_consensus: str  # Momentum dominant
    next_key_levels: List[float]  # Prochains niveaux importants
    divergence_alerts: List[str]  # Alertes de divergence
    confluence_score: float
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convertit en dictionnaire pour export."""
        return {
            "primary_trend": self.primary_trend,
            "trend_alignment": self.trend_alignment.value,
            "signal_strength": self.signal_strength.value,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "key_support_levels": self.key_support_levels,
            "key_resistance_levels": self.key_resistance_levels,
            "entry_zones": self.entry_zones,
            "exit_zones": self.exit_zones,
            "timeframe_signals": {
                k: v.to_dict() for k, v in self.timeframe_signals.items()
            },
            "regime_consensus": self.regime_consensus,
            "momentum_consensus": self.momentum_consensus,
            "next_key_levels": self.next_key_levels,
            "divergence_alerts": self.divergence_alerts,
            "confluence_score": self.confluence_score,
            "timestamp": self.timestamp,
        }


class MultiTimeframeAnalyzer:
    """
    Analyseur multi-timeframe qui agrège les signaux de plusieurs échelles temporelles.

    Fournit une vue consolidée des conditions de marché en:
    1. Analysant chaque timeframe individuellement
    2. Identifiant les confluences et divergences
    3. Calculant les niveaux clés confirmés
    4. Déterminant la force globale des signaux
    5. Identifiant les zones d'opportunité optimales
    """

    def __init__(
        self,
        timeframes: Optional[List[TimeframeType]] = None,
        primary_timeframe: TimeframeType = TimeframeType.M5,
        min_confluence: int = 2,
    ):
        """
        Args:
            timeframes: Liste des timeframes à analyser
            primary_timeframe: Timeframe principal pour les signaux
            min_confluence: Minimum de timeframes pour confirmer un signal
        """
        if timeframes is None:
            timeframes = [
                TimeframeType.M5,
                TimeframeType.M15,
                TimeframeType.H1,
                TimeframeType.H4,
            ]

        self.timeframes = timeframes
        self.primary_timeframe = primary_timeframe
        self.min_confluence = min_confluence

        # Importation des détecteurs
        from .regime_detector import RegimeDetector
        from .support_resistance_detector import SupportResistanceDetector
        from ..indicators.composite.confluence import (
            calculate_multi_timeframe_confluence,
        )

        self.regime_detector = RegimeDetector()
        self.sr_detector = SupportResistanceDetector()

        # Cache pour optimisation
        self._cache: Dict[str, Any] = {}

    def analyze_multiple_timeframes(
        self,
        timeframe_data: Dict[str, Dict],
        current_price: float,
        symbol: Optional[str] = None,
        enable_cache: bool = True,
    ) -> MultiTimeframeAnalysis:
        """
        Analyse plusieurs timeframes et agrège les signaux.

        Args:
            timeframe_data: Données par timeframe {timeframe: {highs, lows, closes, volumes}}
            current_price: Prix actuel
            symbol: Trading symbol for cached indicators (optional, enables cache if provided)
            enable_cache: Whether to use cached indicators

        Returns:
            Analyse multi-timeframe complète
        """
        try:
            # 1. Analyser chaque timeframe individuellement
            timeframe_signals = {}

            for tf_str, data in timeframe_data.items():
                try:
                    tf = TimeframeType(tf_str)
                    if tf in self.timeframes:
                        signal = self._analyze_single_timeframe(
                            tf, data, current_price, symbol, enable_cache
                        )
                        timeframe_signals[tf_str] = signal
                except ValueError:
                    logger.warning(f"Timeframe non supporté: {tf_str}")
                    continue

            if not timeframe_signals:
                return self._empty_analysis()

            # 2. Analyser l'alignement des tendances
            trend_alignment = self._analyze_trend_alignment(timeframe_signals)

            # 3. Calculer la force globale du signal
            signal_strength = self._calculate_overall_signal_strength(timeframe_signals)

            # 4. Déterminer la tendance primaire
            primary_trend = self._determine_primary_trend(timeframe_signals)

            # 5. Identifier les niveaux clés confirmés
            key_support_levels = self._find_confluent_support_levels(
                timeframe_signals, current_price
            )
            key_resistance_levels = self._find_confluent_resistance_levels(
                timeframe_signals, current_price
            )

            # 6. Calculer les consensus
            regime_consensus = self._calculate_regime_consensus(timeframe_signals)
            momentum_consensus = self._calculate_momentum_consensus(timeframe_signals)

            # 7. Identifier les zones d'entrée/sortie
            entry_zones = self._identify_entry_zones(
                timeframe_signals, current_price, primary_trend
            )
            exit_zones = self._identify_exit_zones(
                timeframe_signals, current_price, primary_trend
            )

            # 8. Détecter les divergences
            divergence_alerts = self._detect_divergences(timeframe_signals)

            # 9. Évaluer le risque
            risk_level = self._assess_risk_level(
                trend_alignment, signal_strength, divergence_alerts
            )

            # 10. Calculer la confiance globale
            confidence = self._calculate_overall_confidence(
                timeframe_signals, trend_alignment
            )

            # 11. Identifier les prochains niveaux clés
            next_key_levels = self._identify_next_key_levels(
                key_support_levels + key_resistance_levels, current_price
            )

            # 12. Calculer le score de confluence
            confluence_score = self._calculate_confluence_score(
                signal_strength, trend_alignment, confidence, risk_level
            )

            return MultiTimeframeAnalysis(
                primary_trend=primary_trend,
                trend_alignment=trend_alignment,
                signal_strength=signal_strength,
                confidence=confidence,
                risk_level=risk_level,
                key_support_levels=key_support_levels,
                key_resistance_levels=key_resistance_levels,
                entry_zones=entry_zones,
                exit_zones=exit_zones,
                timeframe_signals=timeframe_signals,
                regime_consensus=regime_consensus,
                momentum_consensus=momentum_consensus,
                next_key_levels=next_key_levels,
                divergence_alerts=divergence_alerts,
                confluence_score=confluence_score,
            )

        except Exception as e:
            logger.error(f"Erreur analyse multi-timeframe: {e}")
            return self._empty_analysis()

    def _analyze_single_timeframe(
        self,
        timeframe: TimeframeType,
        data: Dict,
        current_price: float,
        symbol: Optional[str] = None,
        enable_cache: bool = True,
    ) -> TimeframeSignal:
        """Analyse un timeframe spécifique."""
        try:
            highs = np.array(data["highs"], dtype=float)
            lows = np.array(data["lows"], dtype=float)
            closes = np.array(data["closes"], dtype=float)
            volumes = np.array(data["volumes"], dtype=float)

            # Régime de marché (with caching if symbol provided)
            regime = self.regime_detector.detect_regime(
                highs, lows, closes, volumes, symbol, True, enable_cache
            )

            # Niveaux de support/résistance
            sr_levels = self.sr_detector.detect_levels(
                highs, lows, closes, volumes, current_price, timeframe.value
            )

            support_levels = [
                l.price
                for l in sr_levels
                if l.price < current_price and "support" in l.level_type.value
            ]
            resistance_levels = [
                l.price
                for l in sr_levels
                if l.price > current_price and "resistance" in l.level_type.value
            ]

            # Analyse de tendance simple (with caching if symbol provided)
            trend_direction, trend_strength = self._analyze_trend_simple(
                closes, symbol, enable_cache
            )
            momentum_direction, momentum_strength = self._analyze_momentum_simple(
                closes, symbol, enable_cache
            )

            return TimeframeSignal(
                timeframe=timeframe,
                regime_type=regime.regime_type.value,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                momentum_direction=momentum_direction,
                momentum_strength=momentum_strength,
                volatility_level=self._classify_volatility(regime.volatility),
                support_levels=support_levels[:3],  # Top 3 supports
                resistance_levels=resistance_levels[:3],  # Top 3 resistances
                volume_profile=regime.volume_profile,
                confidence=regime.confidence,
            )

        except Exception as e:
            logger.warning(f"Erreur analyse timeframe {timeframe}: {e}")
            return self._empty_timeframe_signal(timeframe)

    def _analyze_trend_simple(
        self,
        closes: np.ndarray,
        symbol: Optional[str] = None,
        enable_cache: bool = True,
    ) -> Tuple[str, float]:
        """Analyse simple de tendance."""
        if len(closes) < 20:
            return "neutral", 0.0

        from ..indicators.trend.moving_averages import (
            calculate_ema_series,
            calculate_ema,
        )

        # EMA courte vs EMA longue (with caching if symbol provided)
        if symbol and enable_cache:
            # Use cached individual calculations for current values
            short_val = calculate_ema(closes, 7, symbol, enable_cache)
            long_val = calculate_ema(closes, 21, symbol, enable_cache)
        else:
            # Use series for fallback
            ema_short = calculate_ema_series(closes, 7)
            ema_long = calculate_ema_series(closes, 21)

            # Find last valid values
            short_val = None
            long_val = None

            for i in range(len(ema_short) - 1, -1, -1):
                if ema_short[i] is not None and short_val is None:
                    short_val = ema_short[i]
                if ema_long[i] is not None and long_val is None:
                    long_val = ema_long[i]
                if short_val is not None and long_val is not None:
                    break

        if short_val is None or long_val is None:
            return "neutral", 0.0

        # Direction
        if short_val > long_val:
            direction = "bullish"
        elif short_val < long_val:
            direction = "bearish"
        else:
            direction = "neutral"

        # Force (distance relative entre EMAs)
        strength = abs(short_val - long_val) / long_val * 100
        strength = min(strength, 100)

        return direction, strength

    def _analyze_momentum_simple(
        self,
        closes: np.ndarray,
        symbol: Optional[str] = None,
        enable_cache: bool = True,
    ) -> Tuple[str, float]:
        """Analyse simple de momentum."""
        if len(closes) < 14:
            return "neutral", 0.0

        from ..indicators.momentum.rsi import calculate_rsi_series, calculate_rsi

        # RSI calculation (with caching if symbol provided)
        if symbol and enable_cache:
            # Use cached individual calculation
            current_rsi = calculate_rsi(closes, 14, symbol, enable_cache)
        else:
            # Use series for fallback
            rsi_series = calculate_rsi_series(closes)

            # Find last valid RSI value
            current_rsi = None
            for i in range(len(rsi_series) - 1, -1, -1):
                if rsi_series[i] is not None:
                    current_rsi = rsi_series[i]
                    break

        if current_rsi is None:
            return "neutral", 0.0

        # Direction et force basées sur RSI
        if current_rsi > 60:
            direction = "bullish"
            strength = min((current_rsi - 50) * 2, 100)
        elif current_rsi < 40:
            direction = "bearish"
            strength = min((50 - current_rsi) * 2, 100)
        else:
            direction = "neutral"
            strength = 0.0

        return direction, strength

    def _classify_volatility(self, volatility: float) -> str:
        """Classifie le niveau de volatilité."""
        if volatility > 0.05:
            return "high"
        elif volatility < 0.02:
            return "low"
        else:
            return "normal"

    def _analyze_trend_alignment(
        self, signals: Dict[str, TimeframeSignal]
    ) -> TrendAlignment:
        """Analyse l'alignement des tendances entre timeframes."""
        if not signals:
            return TrendAlignment.MIXED

        bullish_count = sum(
            1 for s in signals.values() if s.trend_direction == "bullish"
        )
        bearish_count = sum(
            1 for s in signals.values() if s.trend_direction == "bearish"
        )
        total_count = len(signals)

        bullish_ratio = bullish_count / total_count
        bearish_ratio = bearish_count / total_count

        if bullish_ratio >= 0.8:
            return TrendAlignment.FULLY_ALIGNED_BULL
        elif bearish_ratio >= 0.8:
            return TrendAlignment.FULLY_ALIGNED_BEAR
        elif bullish_ratio >= 0.6:
            return TrendAlignment.MOSTLY_ALIGNED_BULL
        elif bearish_ratio >= 0.6:
            return TrendAlignment.MOSTLY_ALIGNED_BEAR
        elif abs(bullish_ratio - bearish_ratio) <= 0.2:
            return TrendAlignment.MIXED
        else:
            return TrendAlignment.CONFLICTING

    def _calculate_overall_signal_strength(
        self, signals: Dict[str, TimeframeSignal]
    ) -> SignalStrength:
        """Calcule la force globale du signal."""
        if not signals:
            return SignalStrength.VERY_WEAK

        # Moyenne pondérée des forces de tendance
        total_weight = 0.0
        weighted_strength = 0.0

        timeframe_weights = {"1m": 0.5, "5m": 1.0, "15m": 1.5, "1h": 2.5, "1d": 4.0}

        for tf_str, signal in signals.items():
            weight = timeframe_weights.get(tf_str, 1.0)
            weighted_strength += (
                float(signal.trend_strength) * weight * (signal.confidence / 100)
            )
            total_weight += float(weight)

        if total_weight == 0:
            return SignalStrength.VERY_WEAK

        avg_strength = weighted_strength / total_weight

        if avg_strength >= 80:
            return SignalStrength.VERY_STRONG
        elif avg_strength >= 60:
            return SignalStrength.STRONG
        elif avg_strength >= 40:
            return SignalStrength.MODERATE
        elif avg_strength >= 20:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK

    def _determine_primary_trend(self, signals: Dict[str, TimeframeSignal]) -> str:
        """Détermine la tendance primaire."""
        if not signals:
            return "neutral"

        # Pondération par timeframe
        bullish_weight = 0.0
        bearish_weight = 0.0

        timeframe_weights = {
            "1m": 0.5,
            "5m": 1.0,
            "15m": 1.5,
            "30m": 2.0,
            "1h": 2.5,
            "1d": 4.0,
        }

        for tf_str, signal in signals.items():
            weight = timeframe_weights.get(tf_str, 1.0) * (signal.confidence / 100)

            if signal.trend_direction == "bullish":
                bullish_weight += float(weight)
            elif signal.trend_direction == "bearish":
                bearish_weight += float(weight)

        if bullish_weight > bearish_weight * 1.2:
            return "bullish"
        elif bearish_weight > bullish_weight * 1.2:
            return "bearish"
        else:
            return "neutral"

    def _find_confluent_support_levels(
        self, signals: Dict[str, TimeframeSignal], current_price: float
    ) -> List[float]:
        """Trouve les niveaux de support confirmés sur plusieurs timeframes."""
        all_supports = []

        # Collecter tous les supports
        for signal in signals.values():
            all_supports.extend(signal.support_levels)

        if not all_supports:
            return []

        # Grouper les niveaux proches (tolérance 0.5%)
        confluent_levels = []
        tolerance = 0.005

        all_supports.sort()
        i = 0

        while i < len(all_supports):
            level = all_supports[i]
            similar_levels = [level]

            j = i + 1
            while j < len(all_supports):
                if abs(all_supports[j] - level) / level <= tolerance:
                    similar_levels.append(all_supports[j])
                    j += 1
                else:
                    break

            # Si au moins min_confluence niveaux similaires
            if len(similar_levels) >= self.min_confluence:
                avg_level = np.mean(similar_levels)
                if avg_level < current_price:  # Vérifier que c'est en dessous
                    confluent_levels.append(float(avg_level))

            i = j

        # Retourner les 5 plus proches
        confluent_levels.sort(reverse=True)  # Plus proches d'abord
        return confluent_levels[:5]

    def _find_confluent_resistance_levels(
        self, signals: Dict[str, TimeframeSignal], current_price: float
    ) -> List[float]:
        """Trouve les niveaux de résistance confirmés sur plusieurs timeframes."""
        all_resistances = []

        # Collecter toutes les résistances
        for signal in signals.values():
            all_resistances.extend(signal.resistance_levels)

        if not all_resistances:
            return []

        # Grouper les niveaux proches
        confluent_levels = []
        tolerance = 0.005

        all_resistances.sort()
        i = 0

        while i < len(all_resistances):
            level = all_resistances[i]
            similar_levels = [level]

            j = i + 1
            while j < len(all_resistances):
                if abs(all_resistances[j] - level) / level <= tolerance:
                    similar_levels.append(all_resistances[j])
                    j += 1
                else:
                    break

            if len(similar_levels) >= self.min_confluence:
                avg_level = np.mean(similar_levels)
                if avg_level > current_price:  # Vérifier que c'est au-dessus
                    confluent_levels.append(float(avg_level))

            i = j

        # Retourner les 5 plus proches
        confluent_levels.sort()  # Plus proches d'abord
        return confluent_levels[:5]

    def _calculate_regime_consensus(self, signals: Dict[str, TimeframeSignal]) -> str:
        """Calcule le consensus de régime."""
        if not signals:
            return "unknown"

        regime_counts: Dict[str, int] = {}
        for signal in signals.values():
            regime = signal.regime_type
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # Retourner le régime le plus fréquent
        return max(regime_counts.items(), key=lambda x: x[1])[0]

    def _calculate_momentum_consensus(self, signals: Dict[str, TimeframeSignal]) -> str:
        """Calcule le consensus de momentum."""
        if not signals:
            return "neutral"

        bullish_count = sum(
            1 for s in signals.values() if s.momentum_direction == "bullish"
        )
        bearish_count = sum(
            1 for s in signals.values() if s.momentum_direction == "bearish"
        )

        if bullish_count > bearish_count:
            return "bullish"
        elif bearish_count > bullish_count:
            return "bearish"
        else:
            return "neutral"

    def _identify_entry_zones(
        self,
        signals: Dict[str, TimeframeSignal],
        current_price: float,
        primary_trend: str,
    ) -> List[Dict]:
        """Identifie les zones d'entrée optimales."""
        entry_zones = []

        if primary_trend == "bullish":
            # Chercher des supports pour entrées long
            all_supports = []
            for signal in signals.values():
                all_supports.extend(signal.support_levels)

            if all_supports:
                # Zones autour des supports forts
                for support in sorted(all_supports, reverse=True)[:3]:
                    if support < current_price:
                        entry_zones.append(
                            {
                                "type": "long_entry",
                                "level": support,
                                "zone_low": support * 0.995,
                                "zone_high": support * 1.005,
                                "distance_pct": abs(support - current_price)
                                / current_price
                                * 100,
                                "strength": "moderate",
                            }
                        )

        elif primary_trend == "bearish":
            # Chercher des résistances pour entrées short
            all_resistances = []
            for signal in signals.values():
                all_resistances.extend(signal.resistance_levels)

            if all_resistances:
                for resistance in sorted(all_resistances)[:3]:
                    if resistance > current_price:
                        entry_zones.append(
                            {
                                "type": "short_entry",
                                "level": resistance,
                                "zone_low": resistance * 0.995,
                                "zone_high": resistance * 1.005,
                                "distance_pct": abs(resistance - current_price)
                                / current_price
                                * 100,
                                "strength": "moderate",
                            }
                        )

        return entry_zones

    def _identify_exit_zones(
        self,
        signals: Dict[str, TimeframeSignal],
        current_price: float,
        primary_trend: str,
    ) -> List[Dict]:
        """Identifie les zones de sortie optimales."""
        exit_zones = []

        if primary_trend == "bullish":
            # Résistances pour sorties long
            all_resistances = []
            for signal in signals.values():
                all_resistances.extend(signal.resistance_levels)

            if all_resistances:
                for resistance in sorted(all_resistances)[:3]:
                    if resistance > current_price:
                        exit_zones.append(
                            {
                                "type": "long_exit",
                                "level": resistance,
                                "zone_low": resistance * 0.995,
                                "zone_high": resistance * 1.005,
                                "distance_pct": abs(resistance - current_price)
                                / current_price
                                * 100,
                                "profit_potential": (resistance - current_price)
                                / current_price
                                * 100,
                            }
                        )

        elif primary_trend == "bearish":
            # Supports pour sorties short
            all_supports = []
            for signal in signals.values():
                all_supports.extend(signal.support_levels)

            if all_supports:
                for support in sorted(all_supports, reverse=True)[:3]:
                    if support < current_price:
                        exit_zones.append(
                            {
                                "type": "short_exit",
                                "level": support,
                                "zone_low": support * 0.995,
                                "zone_high": support * 1.005,
                                "distance_pct": abs(support - current_price)
                                / current_price
                                * 100,
                                "profit_potential": (current_price - support)
                                / current_price
                                * 100,
                            }
                        )

        return exit_zones

    def _detect_divergences(self, signals: Dict[str, TimeframeSignal]) -> List[str]:
        """Détecte les divergences entre timeframes."""
        divergences: List[str] = []

        if len(signals) < 2:
            return divergences

        # Tendances contradictoires
        trends = [s.trend_direction for s in signals.values()]
        if "bullish" in trends and "bearish" in trends:
            bullish_count = trends.count("bullish")
            bearish_count = trends.count("bearish")

            if abs(bullish_count - bearish_count) <= 1:
                divergences.append(
                    "Divergence majeure: Tendances contradictoires entre timeframes"
                )

        # Momentum vs trend
        for tf, signal in signals.items():
            if (
                signal.trend_direction != signal.momentum_direction
                and signal.momentum_direction != "neutral"
            ):
                divergences.append(
                    f"Divergence {tf}: Tendance {signal.trend_direction} vs Momentum {signal.momentum_direction}"
                )

        return divergences

    def _assess_risk_level(
        self,
        trend_alignment: TrendAlignment,
        signal_strength: SignalStrength,
        divergences: List[str],
    ) -> str:
        """Évalue le niveau de risque."""
        risk_score = 0

        # Alignement des tendances
        if trend_alignment in [
            TrendAlignment.FULLY_ALIGNED_BULL,
            TrendAlignment.FULLY_ALIGNED_BEAR,
        ]:
            risk_score += 0  # Faible risque
        elif trend_alignment in [
            TrendAlignment.MOSTLY_ALIGNED_BULL,
            TrendAlignment.MOSTLY_ALIGNED_BEAR,
        ]:
            risk_score += 1  # Risque modéré
        else:
            risk_score += 2  # Risque élevé

        # Force du signal
        if signal_strength in [SignalStrength.VERY_STRONG, SignalStrength.STRONG]:
            risk_score += 0
        elif signal_strength == SignalStrength.MODERATE:
            risk_score += 1
        else:
            risk_score += 2

        # Divergences
        risk_score += min(len(divergences), 2)

        if risk_score <= 1:
            return "low"
        elif risk_score <= 3:
            return "medium"
        else:
            return "high"

    def _calculate_overall_confidence(
        self, signals: Dict[str, TimeframeSignal], trend_alignment: TrendAlignment
    ) -> float:
        """Calcule la confiance globale."""
        if not signals:
            return 0.0

        # Confiance moyenne des signaux
        avg_confidence = np.mean([s.confidence for s in signals.values()])

        # Bonus alignement
        alignment_bonus = 0
        if trend_alignment in [
            TrendAlignment.FULLY_ALIGNED_BULL,
            TrendAlignment.FULLY_ALIGNED_BEAR,
        ]:
            alignment_bonus = 20
        elif trend_alignment in [
            TrendAlignment.MOSTLY_ALIGNED_BULL,
            TrendAlignment.MOSTLY_ALIGNED_BEAR,
        ]:
            alignment_bonus = 10

        return min(float(avg_confidence + alignment_bonus), 100.0)

    def _identify_next_key_levels(
        self, all_levels: List[float], current_price: float
    ) -> List[float]:
        """Identifie les prochains niveaux clés."""
        if not all_levels:
            return []

        # Séparer supports et résistances
        supports = [l for l in all_levels if l < current_price]
        resistances = [l for l in all_levels if l > current_price]

        next_levels = []

        # Prochain support
        if supports:
            next_levels.append(max(supports))

        # Prochaine résistance
        if resistances:
            next_levels.append(min(resistances))

        return sorted(next_levels)

    def _calculate_confluence_score(
        self,
        signal_strength: SignalStrength,
        trend_alignment: TrendAlignment,
        confidence: float,
        risk_level: str = "medium",
    ) -> float:
        """
        Calcule un score de confluence entre 0 et 100 basé sur l'analyse multi-timeframe.
        """
        # Calcul simple du score de confluence
        base_score = 50.0

        # Bonus selon la force du signal
        strength_bonus = {
            "very_weak": 0,
            "weak": 10,
            "moderate": 20,
            "strong": 30,
            "very_strong": 40,
        }.get(signal_strength.value, 0)

        # Bonus selon l'alignement
        alignment_bonus = {
            "fully_aligned_bull": 30,
            "fully_aligned_bear": 30,
            "mostly_aligned_bull": 20,
            "mostly_aligned_bear": 20,
            "mixed": 5,
            "conflicting": -10,
        }.get(trend_alignment.value, 0)

        # Ajustement selon le risque
        risk_penalty = {"high": -15, "medium": 0, "low": 10}.get(risk_level, 0)

        final_score = base_score + strength_bonus + alignment_bonus + risk_penalty
        return max(0.0, min(100.0, final_score * (confidence / 100)))

    def _empty_timeframe_signal(self, timeframe: TimeframeType) -> TimeframeSignal:
        """Signal vide pour un timeframe."""
        return TimeframeSignal(
            timeframe=timeframe,
            regime_type="unknown",
            trend_direction="neutral",
            trend_strength=0.0,
            momentum_direction="neutral",
            momentum_strength=0.0,
            volatility_level="normal",
            support_levels=[],
            resistance_levels=[],
            volume_profile="unknown",
            confidence=0.0,
        )

    def _empty_analysis(self) -> MultiTimeframeAnalysis:
        """Analyse vide en cas d'erreur."""
        return MultiTimeframeAnalysis(
            primary_trend="neutral",
            trend_alignment=TrendAlignment.MIXED,
            signal_strength=SignalStrength.VERY_WEAK,
            confidence=0.0,
            risk_level="high",
            key_support_levels=[],
            key_resistance_levels=[],
            entry_zones=[],
            exit_zones=[],
            timeframe_signals={},
            regime_consensus="unknown",
            momentum_consensus="neutral",
            next_key_levels=[],
            divergence_alerts=[],
            confluence_score=0.0,
        )

    def get_trading_recommendation(
        self, analysis: MultiTimeframeAnalysis, current_price: float
    ) -> Dict:
        """
        Génère une recommandation de trading basée sur l'analyse multi-timeframe.

        Returns:
            Dictionnaire avec recommandation, entrée, sortie, stop-loss
        """
        recommendation = {
            "action": "HOLD",
            "confidence": analysis.confidence,
            "risk_level": analysis.risk_level,
            "entry_price": None,
            "target_prices": [],
            "stop_loss": None,
            "position_size": "standard",
            "reasoning": [],
        }

        # Logique de recommandation
        if (
            analysis.signal_strength
            in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]
            and analysis.trend_alignment
            in [TrendAlignment.FULLY_ALIGNED_BULL, TrendAlignment.MOSTLY_ALIGNED_BULL]
            and analysis.confidence > 70
        ):

            recommendation["action"] = "BUY"
            reasoning_list = recommendation.get("reasoning", [])
            if isinstance(reasoning_list, list):
                reasoning_list.append("Forte confluence haussière multi-timeframe")
            else:
                recommendation["reasoning"] = [
                    "Forte confluence haussière multi-timeframe"
                ]

            # Prix d'entrée (niveau de support proche ou prix actuel)
            if analysis.key_support_levels:
                nearest_support = max(
                    [s for s in analysis.key_support_levels if s < current_price * 1.02]
                )
                recommendation["entry_price"] = nearest_support
            else:
                recommendation["entry_price"] = current_price

            # Objectifs (résistances)
            if analysis.key_resistance_levels:
                recommendation["target_prices"] = analysis.key_resistance_levels[:3]

            # Stop loss (support en dessous)
            if analysis.key_support_levels:
                entry_price = recommendation.get("entry_price", current_price)
                try:
                    entry_price_float = (
                        float(entry_price)
                        if isinstance(entry_price, (int, float, str))
                        else current_price
                    )
                    supports_below = [
                        s for s in analysis.key_support_levels if s < entry_price_float
                    ]
                    if supports_below:
                        recommendation["stop_loss"] = max(supports_below) * 0.995
                except (TypeError, ValueError):
                    # Si la conversion échoue, utiliser current_price
                    supports_below = [
                        s for s in analysis.key_support_levels if s < current_price
                    ]
                    if supports_below:
                        recommendation["stop_loss"] = max(supports_below) * 0.995

        elif (
            analysis.signal_strength
            in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]
            and analysis.trend_alignment
            in [TrendAlignment.FULLY_ALIGNED_BEAR, TrendAlignment.MOSTLY_ALIGNED_BEAR]
            and analysis.confidence > 70
        ):

            recommendation["action"] = "SELL"
            reasoning_list = recommendation.get("reasoning", [])
            if isinstance(reasoning_list, list):
                reasoning_list.append("Forte confluence baissière multi-timeframe")
            else:
                recommendation["reasoning"] = [
                    "Forte confluence baissière multi-timeframe"
                ]

            # Prix d'entrée (résistance proche ou prix actuel)
            if analysis.key_resistance_levels:
                nearest_resistance = min(
                    [
                        r
                        for r in analysis.key_resistance_levels
                        if r > current_price * 0.98
                    ]
                )
                recommendation["entry_price"] = nearest_resistance
            else:
                recommendation["entry_price"] = current_price

            # Objectifs (supports)
            if analysis.key_support_levels:
                recommendation["target_prices"] = analysis.key_support_levels[:3]

            # Stop loss (résistance au-dessus)
            if analysis.key_resistance_levels:
                entry_price = recommendation.get("entry_price", current_price)
                try:
                    entry_price_float = (
                        float(entry_price)
                        if isinstance(entry_price, (int, float, str))
                        else current_price
                    )
                    resistances_above = [
                        r
                        for r in analysis.key_resistance_levels
                        if r > entry_price_float
                    ]
                    if resistances_above:
                        recommendation["stop_loss"] = min(resistances_above) * 1.005
                except (TypeError, ValueError):
                    # Si la conversion échoue, utiliser current_price
                    resistances_above = [
                        r for r in analysis.key_resistance_levels if r > current_price
                    ]
                    if resistances_above:
                        recommendation["stop_loss"] = min(resistances_above) * 1.005

        # Ajuster la taille de position selon le risque
        if analysis.risk_level == "high":
            recommendation["position_size"] = "reduced"
        elif analysis.risk_level == "low" and analysis.confidence > 80:
            recommendation["position_size"] = "increased"

        return recommendation
