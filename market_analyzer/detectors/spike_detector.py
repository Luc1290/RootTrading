"""
Spike Detector

This module detects anomalous price and volume movements that may indicate:
- Large institutional orders (whales)
- Market manipulation
- News events impact
- Liquidation cascades
- Pump & dump schemes
- Technical breakouts with high conviction
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SpikeType(Enum):
    """Types de spikes détectés."""

    PRICE_SPIKE_UP = "price_spike_up"
    PRICE_SPIKE_DOWN = "price_spike_down"
    VOLUME_SPIKE = "volume_spike"
    COMBINED_SPIKE = "combined_spike"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    DUMP = "dump"
    PUMP = "pump"
    BREAKOUT_SPIKE = "breakout_spike"
    REVERSAL_SPIKE = "reversal_spike"


class SpikeIntensity(Enum):
    """Intensité du spike."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class SpikeImpact(Enum):
    """Impact sur la tendance."""

    CONTINUATION = "continuation"
    REVERSAL = "reversal"
    EXHAUSTION = "exhaustion"
    BREAKOUT = "breakout"
    UNKNOWN = "unknown"


@dataclass
class SpikeEvent:
    """Structure représentant un événement spike."""

    timestamp: str
    spike_type: SpikeType
    intensity: SpikeIntensity
    impact: SpikeImpact
    price_change: float  # % de variation de prix
    volume_ratio: float  # Ratio vs volume moyen
    duration_bars: int  # Durée du spike en barres
    confidence: float  # 0-100
    z_score_price: float  # Z-score pour le prix
    z_score_volume: float  # Z-score pour le volume
    pre_spike_trend: str  # Tendance avant le spike
    post_spike_behavior: str | None = None  # Comportement après
    related_levels: list[float] | None = None  # Niveaux S/R impliqués
    liquidity_impact: float = 0.0  # Impact sur la liquidité

    def __post_init__(self):
        if self.related_levels is None:
            self.related_levels = []

    def to_dict(self) -> dict:
        """Convertit en dictionnaire pour export."""
        return {
            "timestamp": self.timestamp,
            "spike_type": self.spike_type.value,
            "intensity": self.intensity.value,
            "impact": self.impact.value,
            "price_change": self.price_change,
            "volume_ratio": self.volume_ratio,
            "duration_bars": self.duration_bars,
            "confidence": self.confidence,
            "z_score_price": self.z_score_price,
            "z_score_volume": self.z_score_volume,
            "pre_spike_trend": self.pre_spike_trend,
            "post_spike_behavior": self.post_spike_behavior,
            "related_levels": self.related_levels,
            "liquidity_impact": self.liquidity_impact,
        }


class SpikeDetector:
    """
    Détecteur de spikes (anomalies) de prix et volume.

    Utilise plusieurs méthodes:
    1. Z-score pour détecter les valeurs aberrantes
    2. Analyse des percentiles extrêmes
    3. Détection de patterns de manipulation
    4. Corrélation prix-volume anormale
    5. Identification de sweeps de liquidité
    """

    def __init__(
        self,
        lookback_window: int = 50,
        price_z_threshold: float = 2.0,  # Crypto plus volatil
        volume_z_threshold: float = 2.5,
        extreme_percentile: float = 95,
        min_spike_duration: int = 1,
    ):
        """
        Args:
            lookback_window: Fenêtre d'analyse pour calculs statistiques
            price_z_threshold: Seuil Z-score pour spikes de prix
            volume_z_threshold: Seuil Z-score pour spikes de volume
            extreme_percentile: Percentile pour valeurs extrêmes
            min_spike_duration: Durée minimum d'un spike
        """
        self.lookback_window = lookback_window
        self.price_z_threshold = price_z_threshold
        self.volume_z_threshold = volume_z_threshold
        self.extreme_percentile = extreme_percentile
        self.min_spike_duration = min_spike_duration

        # Cache pour optimisation
        self._cache: dict[str, Any] = {}

    def detect_spikes(
        self,
        highs: list[float] | np.ndarray,
        lows: list[float] | np.ndarray,
        closes: list[float] | np.ndarray,
        volumes: list[float] | np.ndarray,
        timestamps: list[str] | None = None,
    ) -> list[SpikeEvent]:
        """
        Détecte tous les spikes dans les données.

        Args:
            highs: Prix hauts
            lows: Prix bas
            closes: Prix de clôture
            volumes: Volumes
            timestamps: Timestamps (optionnel)

        Returns:
            Liste des événements spike détectés
        """
        try:
            # Conversion en arrays numpy
            highs = np.array(highs, dtype=float)
            lows = np.array(lows, dtype=float)
            closes = np.array(closes, dtype=float)
            volumes = np.array(volumes, dtype=float)

            if len(closes) < self.lookback_window:
                return []

            if timestamps is None:
                timestamps = [f"bar_{i}" for i in range(len(closes))]

            all_spikes = []

            # 1. Détecter les spikes de prix
            price_spikes = self._detect_price_spikes(highs, lows, closes, timestamps)
            all_spikes.extend(price_spikes)

            # 2. Détecter les spikes de volume
            volume_spikes = self._detect_volume_spikes(volumes, closes, timestamps)
            all_spikes.extend(volume_spikes)

            # 3. Détecter les spikes combinés (prix + volume)
            combined_spikes = self._detect_combined_spikes(
                highs, lows, closes, volumes, timestamps
            )
            all_spikes.extend(combined_spikes)

            # 4. Détecter les patterns de manipulation
            manipulation_spikes = self._detect_manipulation_patterns(
                highs, lows, closes, volumes, timestamps
            )
            all_spikes.extend(manipulation_spikes)

            # 5. Détecter les liquidation sweeps
            liquidity_spikes = self._detect_liquidity_sweeps(
                highs, lows, closes, volumes, timestamps
            )
            all_spikes.extend(liquidity_spikes)

            # 6. Consolider et filtrer les doublons
            consolidated_spikes = self._consolidate_spikes(all_spikes)

            # 7. Analyser l'impact post-spike
            final_spikes = self._analyze_post_spike_impact(consolidated_spikes, closes)

            # Trier par timestamp (plus récents d'abord)
            final_spikes.sort(key=lambda x: x.timestamp, reverse=True)

            return final_spikes[:50]  # Limiter pour performance

        except Exception:
            logger.exception("Erreur détection spikes")
            return []

    def _detect_price_spikes(
        self,
        _highs: np.ndarray,
        _lows: np.ndarray,
        closes: np.ndarray,
        timestamps: list[str],
    ) -> list[SpikeEvent]:
        """Détecte les spikes de prix basés sur les variations anormales."""
        spikes = []

        try:
            # Calculer les variations de prix
            price_changes = np.diff(closes) / closes[:-1]  # Returns

            # Rolling statistics pour normaliser
            for i in range(self.lookback_window, len(price_changes)):
                window = price_changes[i - self.lookback_window : i]
                mean_change = np.mean(window)
                std_change = np.std(window)

                if std_change == 0:
                    continue

                current_change = price_changes[i]
                z_score = (current_change - mean_change) / std_change

                # Détecter les spikes significatifs
                if abs(z_score) > self.price_z_threshold:
                    spike_type = (
                        SpikeType.PRICE_SPIKE_UP
                        if current_change > 0
                        else SpikeType.PRICE_SPIKE_DOWN
                    )

                    # Calculer l'intensité
                    intensity = self._calculate_price_intensity(abs(z_score))

                    # Analyser la tendance pré-spike
                    pre_trend = self._analyze_pre_trend(closes[: i + 1])

                    # Impact potentiel
                    impact = self._predict_price_impact(
                        current_change, pre_trend, z_score
                    )

                    spikes.append(
                        SpikeEvent(
                            timestamp=(
                                timestamps[i + 1]
                                if i + 1 < len(timestamps)
                                else f"bar_{i+1}"
                            ),
                            spike_type=spike_type,
                            intensity=intensity,
                            impact=impact,
                            price_change=float(current_change * 100),  # En pourcentage
                            volume_ratio=1.0,  # Sera mis à jour si volume spike aussi
                            duration_bars=1,
                            confidence=min(abs(z_score) * 15, 95),
                            z_score_price=float(z_score),
                            z_score_volume=0.0,
                            pre_spike_trend=pre_trend,
                        )
                    )

        except Exception as e:
            logger.warning(f"Erreur détection spikes prix: {e}")

        return spikes

    def _detect_volume_spikes(
        self, volumes: np.ndarray, closes: np.ndarray, timestamps: list[str]
    ) -> list[SpikeEvent]:
        """Détecte les spikes de volume anormaux."""
        spikes = []

        try:
            for i in range(self.lookback_window, len(volumes)):
                window = volumes[i - self.lookback_window : i]
                mean_volume = np.mean(window)
                std_volume = np.std(window)

                if std_volume == 0 or mean_volume == 0:
                    continue

                current_volume = volumes[i]
                z_score = (current_volume - mean_volume) / std_volume
                volume_ratio = current_volume / mean_volume

                # Détecter les spikes de volume significatifs
                if z_score > self.volume_z_threshold and volume_ratio > 2.0:
                    # Calculer l'intensité basée sur le ratio et z-score
                    intensity = self._calculate_volume_intensity(volume_ratio, z_score)

                    # Analyser la corrélation avec le prix
                    price_change = (
                        (closes[i] - closes[i - 1]) / closes[i - 1] if i > 0 else 0
                    )

                    # Déterminer le type de spike
                    if abs(price_change) > 0.01:  # 1% de mouvement prix
                        spike_type = SpikeType.COMBINED_SPIKE
                    else:
                        spike_type = SpikeType.VOLUME_SPIKE

                    # Impact basé sur la corrélation prix-volume
                    impact = self._predict_volume_impact(volume_ratio, price_change)

                    spikes.append(
                        SpikeEvent(
                            timestamp=(
                                timestamps[i] if i < len(timestamps) else f"bar_{i}"
                            ),
                            spike_type=spike_type,
                            intensity=intensity,
                            impact=impact,
                            price_change=float(price_change * 100),
                            volume_ratio=float(volume_ratio),
                            duration_bars=1,
                            confidence=min(z_score * 10, 90),
                            z_score_price=0.0,
                            z_score_volume=float(z_score),
                            pre_spike_trend=self._analyze_pre_trend(closes[: i + 1]),
                        )
                    )

        except Exception as e:
            logger.warning(f"Erreur détection spikes volume: {e}")

        return spikes

    def _detect_combined_spikes(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        timestamps: list[str],
    ) -> list[SpikeEvent]:
        """Détecte les spikes combinés prix + volume (très significatifs)."""
        spikes = []

        try:
            price_changes = np.diff(closes) / closes[:-1]

            for i in range(self.lookback_window, len(closes) - 1):
                # Fenêtres d'analyse
                price_window = price_changes[i - self.lookback_window : i]
                volume_window = volumes[i - self.lookback_window : i]

                # Statistiques prix
                price_mean = np.mean(price_window)
                price_std = np.std(price_window)

                # Statistiques volume
                volume_mean = np.mean(volume_window)
                volume_std = np.std(volume_window)

                if price_std == 0 or volume_std == 0 or volume_mean == 0:
                    continue

                # Valeurs actuelles
                current_price_change = price_changes[i]
                current_volume = volumes[i]

                # Z-scores
                price_z = (current_price_change - price_mean) / price_std
                volume_z = (current_volume - volume_mean) / volume_std
                volume_ratio = current_volume / volume_mean

                # Critères pour spike combiné
                price_significant = abs(price_z) > self.price_z_threshold
                volume_significant = (
                    volume_z > self.volume_z_threshold and volume_ratio > 2.0
                )

                if price_significant and volume_significant:
                    # Analyser le pattern pour déterminer le type
                    spike_type = self._classify_combined_spike(
                        current_price_change, volume_ratio, highs[i], lows[i], closes[i]
                    )

                    # Intensité combinée
                    intensity = self._calculate_combined_intensity(
                        abs(price_z), volume_z, volume_ratio
                    )

                    # Impact probablement fort avec prix + volume
                    impact = self._predict_combined_impact(
                        current_price_change, volume_ratio, price_z
                    )

                    # Confiance élevée pour spikes combinés
                    confidence = min((abs(price_z) + volume_z) * 8, 95)

                    spikes.append(
                        SpikeEvent(
                            timestamp=(
                                timestamps[i] if i < len(timestamps) else f"bar_{i}"
                            ),
                            spike_type=spike_type,
                            intensity=intensity,
                            impact=impact,
                            price_change=float(current_price_change * 100),
                            volume_ratio=float(volume_ratio),
                            duration_bars=1,
                            confidence=confidence,
                            z_score_price=float(price_z),
                            z_score_volume=float(volume_z),
                            pre_spike_trend=self._analyze_pre_trend(closes[: i + 1]),
                        )
                    )

        except Exception as e:
            logger.warning(f"Erreur détection spikes combinés: {e}")

        return spikes

    def _detect_manipulation_patterns(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        timestamps: list[str],
    ) -> list[SpikeEvent]:
        """Détecte les patterns de manipulation (pump/dump)."""
        spikes = []

        try:
            for i in range(10, len(closes) - 5):  # Besoin de contexte avant/après
                # Analyser une fenêtre de 15 barres (10 avant + 5 après)
                window_closes = closes[i - 10 : i + 5]
                window_volumes = volumes[i - 10 : i + 5]
                highs[i - 10 : i + 5]
                lows[i - 10 : i + 5]

                # Pattern pump: montée rapide puis chute
                pump_pattern = self._detect_pump_pattern(
                    window_closes, window_volumes, i - 10
                )
                if pump_pattern:
                    spikes.append(
                        SpikeEvent(
                            timestamp=(
                                timestamps[i] if i < len(timestamps) else f"bar_{i}"
                            ),
                            spike_type=SpikeType.PUMP,
                            intensity=pump_pattern["intensity"],
                            impact=SpikeImpact.REVERSAL,
                            price_change=pump_pattern["price_change"],
                            volume_ratio=pump_pattern["volume_ratio"],
                            duration_bars=pump_pattern["duration"],
                            confidence=pump_pattern["confidence"],
                            z_score_price=pump_pattern["price_z"],
                            z_score_volume=pump_pattern["volume_z"],
                            pre_spike_trend="neutral",
                            liquidity_impact=pump_pattern["liquidity_impact"],
                        )
                    )

                # Pattern dump: chute rapide avec volume
                dump_pattern = self._detect_dump_pattern(
                    window_closes, window_volumes, i - 10
                )
                if dump_pattern:
                    spikes.append(
                        SpikeEvent(
                            timestamp=(
                                timestamps[i] if i < len(timestamps) else f"bar_{i}"
                            ),
                            spike_type=SpikeType.DUMP,
                            intensity=dump_pattern["intensity"],
                            impact=SpikeImpact.REVERSAL,
                            price_change=dump_pattern["price_change"],
                            volume_ratio=dump_pattern["volume_ratio"],
                            duration_bars=dump_pattern["duration"],
                            confidence=dump_pattern["confidence"],
                            z_score_price=dump_pattern["price_z"],
                            z_score_volume=dump_pattern["volume_z"],
                            pre_spike_trend="bearish",
                            liquidity_impact=dump_pattern["liquidity_impact"],
                        )
                    )

        except Exception as e:
            logger.warning(f"Erreur détection manipulation: {e}")

        return spikes

    def _detect_liquidity_sweeps(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        timestamps: list[str],
    ) -> list[SpikeEvent]:
        """Détecte les sweeps de liquidité (balayage des stops)."""
        spikes = []

        try:
            for i in range(20, len(closes)):
                # Analyser les 20 dernières barres pour trouver des niveaux
                recent_highs = highs[i - 20 : i]
                recent_lows = lows[i - 20 : i]

                # Trouver les niveaux potentiels de liquidité
                resistance_level = np.max(recent_highs)
                support_level = np.min(recent_lows)

                current_high = highs[i]
                current_low = lows[i]
                current_volume = volumes[i]

                # Moyenne du volume récent
                avg_volume = (
                    np.mean(volumes[i - 10 : i]) if i >= 10 else np.mean(volumes[:i])
                )
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

                # Sweep au-dessus de la résistance
                if (
                    current_high > resistance_level * 1.001  # Dépassement de 0.1%
                    # Mais fermeture en dessous
                    and closes[i] < resistance_level
                    and volume_ratio > 1.5
                ):  # Avec volume élevé

                    price_change = (current_high - closes[i - 1]) / closes[i - 1] * 100

                    spikes.append(
                        SpikeEvent(
                            timestamp=(
                                timestamps[i] if i < len(timestamps) else f"bar_{i}"
                            ),
                            spike_type=SpikeType.LIQUIDITY_SWEEP,
                            intensity=SpikeIntensity.MODERATE,
                            impact=SpikeImpact.REVERSAL,
                            price_change=float(price_change),
                            volume_ratio=float(volume_ratio),
                            duration_bars=1,
                            confidence=70.0,
                            z_score_price=2.0,
                            z_score_volume=float(volume_ratio - 1),
                            pre_spike_trend="bullish",
                            related_levels=[float(resistance_level)],
                            liquidity_impact=float(volume_ratio * 0.3),
                        )
                    )

                # Sweep en dessous du support
                elif (
                    current_low < support_level * 0.999  # Cassure de 0.1%
                    and closes[i] > support_level  # Mais fermeture au-dessus
                    and volume_ratio > 1.5
                ):  # Avec volume élevé

                    price_change = (current_low - closes[i - 1]) / closes[i - 1] * 100

                    spikes.append(
                        SpikeEvent(
                            timestamp=(
                                timestamps[i] if i < len(timestamps) else f"bar_{i}"
                            ),
                            spike_type=SpikeType.LIQUIDITY_SWEEP,
                            intensity=SpikeIntensity.MODERATE,
                            impact=SpikeImpact.REVERSAL,
                            price_change=float(price_change),
                            volume_ratio=float(volume_ratio),
                            duration_bars=1,
                            confidence=70.0,
                            z_score_price=-2.0,
                            z_score_volume=float(volume_ratio - 1),
                            pre_spike_trend="bearish",
                            related_levels=[float(support_level)],
                            liquidity_impact=float(volume_ratio * 0.3),
                        )
                    )

        except Exception as e:
            logger.warning(f"Erreur détection liquidity sweeps: {e}")

        return spikes

    # ============ Méthodes utilitaires ============

    def _calculate_price_intensity(self, z_score: float) -> SpikeIntensity:
        """Calcule l'intensité basée sur le Z-score prix."""
        if z_score > 4.0:
            return SpikeIntensity.EXTREME
        if z_score > 3.0:
            return SpikeIntensity.HIGH
        if z_score > 2.0:
            return SpikeIntensity.MODERATE
        return SpikeIntensity.LOW

    def _calculate_volume_intensity(
        self, volume_ratio: float, z_score: float
    ) -> SpikeIntensity:
        """Calcule l'intensité basée sur le ratio de volume et Z-score."""
        composite_score = (volume_ratio - 1) + z_score

        if composite_score > 8.0 or volume_ratio > 10:
            return SpikeIntensity.EXTREME
        if composite_score > 5.0 or volume_ratio > 5:
            return SpikeIntensity.HIGH
        if composite_score > 3.0 or volume_ratio > 3:
            return SpikeIntensity.MODERATE
        return SpikeIntensity.LOW

    def _calculate_combined_intensity(
        self, price_z: float, volume_z: float, volume_ratio: float
    ) -> SpikeIntensity:
        """Calcule l'intensité pour spikes combinés."""
        combined_score = price_z + volume_z + (volume_ratio - 1) * 0.5

        if combined_score > 10.0:
            return SpikeIntensity.EXTREME
        if combined_score > 7.0:
            return SpikeIntensity.HIGH
        if combined_score > 4.0:
            return SpikeIntensity.MODERATE
        return SpikeIntensity.LOW

    def _analyze_pre_trend(self, closes: np.ndarray) -> str:
        """Analyse la tendance précédant le spike."""
        if len(closes) < 10:
            return "unknown"

        recent = closes[-10:]
        slope = np.polyfit(range(len(recent)), recent, 1)[0]

        if slope > closes[-1] * 0.001:  # 0.1% par barre
            return "bullish"
        if slope < -closes[-1] * 0.001:
            return "bearish"
        return "neutral"

    def _predict_price_impact(
        self, price_change: float, pre_trend: str, z_score: float
    ) -> SpikeImpact:
        """Prédit l'impact du spike prix."""
        if abs(z_score) > 3.0:
            if (price_change > 0 and pre_trend == "bearish") or (
                price_change < 0 and pre_trend == "bullish"
            ):
                return SpikeImpact.REVERSAL
            return SpikeImpact.BREAKOUT
        if abs(z_score) > 2.0:
            return SpikeImpact.CONTINUATION
        return SpikeImpact.UNKNOWN

    def _predict_volume_impact(
        self, volume_ratio: float, price_change: float
    ) -> SpikeImpact:
        """Prédit l'impact du spike volume."""
        if (
            volume_ratio > 5 and abs(price_change) > 0.02
        ):  # Volume énorme + mouvement prix
            return SpikeImpact.BREAKOUT
        if volume_ratio > 3:
            return SpikeImpact.CONTINUATION
        return SpikeImpact.UNKNOWN

    def _predict_combined_impact(
        self, _price_change: float, volume_ratio: float, price_z: float
    ) -> SpikeImpact:
        """Prédit l'impact pour spikes combinés."""
        if volume_ratio > 5 and abs(price_z) > 3:
            return SpikeImpact.BREAKOUT
        if volume_ratio > 3 and abs(price_z) > 2:
            return SpikeImpact.CONTINUATION
        return SpikeImpact.UNKNOWN

    def _classify_combined_spike(
        self,
        price_change: float,
        volume_ratio: float,
        _high: float,
        _low: float,
        _close: float,
    ) -> SpikeType:
        """Classifie le type de spike combiné."""
        if price_change > 0.03 and volume_ratio > 5:  # 3% + gros volume
            return SpikeType.PUMP
        if price_change < -0.03 and volume_ratio > 5:
            return SpikeType.DUMP
        if abs(price_change) > 0.02:
            return SpikeType.BREAKOUT_SPIKE
        return SpikeType.COMBINED_SPIKE

    def _detect_pump_pattern(
        self, closes: np.ndarray, volumes: np.ndarray, _offset: int
    ) -> dict | None:
        """Détecte un pattern de pump."""
        if len(closes) < 15:
            return None

        # Chercher une montée rapide suivie d'une chute
        peak_idx = np.argmax(closes)

        if peak_idx < 5 or peak_idx > len(closes) - 3:  # Pas assez de contexte
            return None

        closes[:peak_idx]
        closes[peak_idx:]

        # Critères pump
        pump_rise = (closes[peak_idx] - closes[0]) / closes[0]
        pump_fall = (closes[-1] - closes[peak_idx]) / closes[peak_idx]

        peak_volume = volumes[peak_idx]
        avg_volume = np.mean(volumes[:peak_idx])
        volume_ratio = peak_volume / avg_volume if avg_volume > 0 else 1

        if (
            pump_rise > 0.05 and pump_fall < -0.03 and volume_ratio > 3
        ):  # 5% montée, 3% chute, 3x volume
            return {
                "intensity": SpikeIntensity.HIGH,
                "price_change": pump_rise * 100,
                "volume_ratio": volume_ratio,
                "duration": peak_idx,
                "confidence": min(80 + volume_ratio * 5, 95),
                "price_z": 3.0,
                "volume_z": volume_ratio - 1,
                "liquidity_impact": volume_ratio * 0.4,
            }

        return None

    def _detect_dump_pattern(
        self, closes: np.ndarray, volumes: np.ndarray, _offset: int
    ) -> dict | None:
        """Détecte un pattern de dump."""
        if len(closes) < 10:
            return None

        # Chercher une chute rapide avec volume
        min_idx = np.argmin(closes)

        if min_idx < 2 or min_idx > len(closes) - 2:
            return None

        dump_fall = (closes[min_idx] - closes[0]) / closes[0]
        (
            (closes[-1] - closes[min_idx]) / closes[min_idx]
            if closes[min_idx] != 0
            else 0
        )

        dump_volume = volumes[min_idx]
        avg_volume = np.mean(volumes)
        volume_ratio = dump_volume / avg_volume if avg_volume > 0 else 1

        if dump_fall < -0.04 and volume_ratio > 2:  # 4% chute + 2x volume
            return {
                "intensity": SpikeIntensity.HIGH,
                "price_change": dump_fall * 100,
                "volume_ratio": volume_ratio,
                "duration": min_idx,
                "confidence": min(75 + volume_ratio * 5, 90),
                "price_z": -3.0,
                "volume_z": volume_ratio - 1,
                "liquidity_impact": volume_ratio * 0.5,
            }

        return None

    def _consolidate_spikes(self, spikes: list[SpikeEvent]) -> list[SpikeEvent]:
        """Consolide les spikes proches dans le temps."""
        if not spikes:
            return []

        # Trier par timestamp
        spikes.sort(key=lambda x: x.timestamp)

        consolidated = []
        i = 0

        while i < len(spikes):
            current = spikes[i]
            similar_spikes = [current]

            # Chercher des spikes dans la même période
            j = i + 1
            while j < len(spikes):
                # Si dans la même barre ou très proche
                if spikes[j].timestamp == current.timestamp:
                    similar_spikes.append(spikes[j])
                    j += 1
                else:
                    break

            # Fusionner si multiples spikes
            if len(similar_spikes) > 1:
                merged = self._merge_spikes(similar_spikes)
                consolidated.append(merged)
            else:
                consolidated.append(current)

            i = j

        return consolidated

    def _merge_spikes(self, spikes: list[SpikeEvent]) -> SpikeEvent:
        """Fusionne plusieurs spikes de la même période."""
        # Prendre le plus intense comme base
        primary = max(spikes, key=lambda x: self._intensity_to_number(x.intensity))

        # Combiner les caractéristiques
        combined_confidence = min(
            sum(s.confidence for s in spikes) / len(spikes) + 10, 95
        )
        max_volume_ratio = max(s.volume_ratio for s in spikes)
        max_price_change = max(abs(s.price_change) for s in spikes)

        # Type combiné si différents types
        unique_types = {s.spike_type for s in spikes}
        combined_type = (
            SpikeType.COMBINED_SPIKE if len(unique_types) > 1 else primary.spike_type
        )

        return SpikeEvent(
            timestamp=primary.timestamp,
            spike_type=combined_type,
            intensity=primary.intensity,
            impact=primary.impact,
            price_change=max_price_change,
            volume_ratio=max_volume_ratio,
            duration_bars=primary.duration_bars,
            confidence=combined_confidence,
            z_score_price=primary.z_score_price,
            z_score_volume=primary.z_score_volume,
            pre_spike_trend=primary.pre_spike_trend,
            related_levels=list(
                set().union(
                    *[s.related_levels for s in spikes if s.related_levels is not None]
                )
            ),
            liquidity_impact=max(s.liquidity_impact for s in spikes),
        )

    def _analyze_post_spike_impact(
        self, spikes: list[SpikeEvent], _closes: np.ndarray
    ) -> list[SpikeEvent]:
        """Analyse le comportement post-spike pour ajuster les prédictions."""
        # Implémentation simplifiée - peut être étendue
        for spike in spikes:
            # Logique d'analyse post-spike basée sur les données suivantes
            spike.post_spike_behavior = "continuation"  # Placeholder

        return spikes

    def _intensity_to_number(self, intensity: SpikeIntensity) -> int:
        """Convertit l'intensité en nombre pour comparaisons."""
        mapping = {
            SpikeIntensity.LOW: 1,
            SpikeIntensity.MODERATE: 2,
            SpikeIntensity.HIGH: 3,
            SpikeIntensity.EXTREME: 4,
        }
        return mapping.get(intensity, 1)

    def get_recent_spikes(
        self, spikes: list[SpikeEvent], max_age: int = 10
    ) -> list[SpikeEvent]:
        """Retourne les spikes récents selon l'âge maximal."""
        # Implémentation simplifiée
        return spikes[:max_age]

    def get_spike_summary(self, spikes: list[SpikeEvent]) -> dict:
        """Retourne un résumé des spikes détectés."""
        if not spikes:
            return {"total": 0, "by_type": {}, "by_intensity": {}}

        by_type: dict[str, int] = {}
        by_intensity: dict[str, int] = {}

        for spike in spikes:
            # Compter par type
            type_key = spike.spike_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1

            # Compter par intensité
            intensity_key = spike.intensity.value
            by_intensity[intensity_key] = by_intensity.get(intensity_key, 0) + 1

        return {
            "total": len(spikes),
            "by_type": by_type,
            "by_intensity": by_intensity,
            "most_recent": spikes[0].to_dict() if spikes else None,
            "highest_confidence": (
                max(spikes, key=lambda x: x.confidence).to_dict() if spikes else None
            ),
        }
