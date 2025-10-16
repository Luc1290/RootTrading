"""
Market Regime Detector

This module identifies current market conditions and regimes:
- Trending (bullish/bearish)
- Ranging (sideways)
- Volatile (high uncertainty)
- Breakout (regime change)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    """Types de régimes de marché."""

    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING = "ranging"
    VOLATILE = "volatile"
    BREAKOUT_BULL = "breakout_bull"
    BREAKOUT_BEAR = "breakout_bear"
    TRANSITION = "transition"
    UNKNOWN = "unknown"


class RegimeStrength(Enum):
    """Force du régime."""

    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    EXTREME = "extreme"


@dataclass
class MarketRegime:
    """Structure représentant un régime de marché détecté."""

    regime_type: RegimeType
    strength: RegimeStrength
    confidence: float  # 0-100
    duration: int  # Nombre de périodes
    volatility: float
    trend_slope: float
    support_resistance_strength: float
    volume_profile: str
    key_levels: list[float]
    timestamp: str | None = None

    def to_dict(self) -> dict:
        """Convertit en dictionnaire pour export."""
        return {
            "regime_type": self.regime_type.value,
            "strength": self.strength.value,
            "confidence": self.confidence,
            "duration": self.duration,
            "volatility": self.volatility,
            "trend_slope": self.trend_slope,
            "support_resistance_strength": self.support_resistance_strength,
            "volume_profile": self.volume_profile,
            "key_levels": self.key_levels,
            "timestamp": self.timestamp,
        }


class RegimeDetector:
    """
    Détecteur de régime de marché utilisant plusieurs méthodes:
    - Analyse de volatilité (ATR, Bollinger Width)
    - Analyse de tendance (EMA, ADX, pente)
    - Analyse de momentum (RSI, MACD)
    - Analyse de support/résistance
    - Analyse de volume
    """

    def __init__(
        self,
        lookback_period: int = 50,
        trend_threshold: float = 0.05,  # Augmenté de 0.02 à 0.05 pour crypto
        volatility_threshold: float = 0.08,  # Augmenté de 0.05 à 0.08 pour crypto
        volume_threshold: float = 2.0,
    ):  # Augmenté de 1.5 à 2.0 pour crypto
        """
        Args:
            lookback_period: Période d'analyse (50 points suffisants)
            trend_threshold: Seuil pour détecter une tendance (5% adapté crypto)
            volatility_threshold: Seuil de volatilité (8% adapté crypto)
            volume_threshold: Multiplicateur de volume anormal (2x pour crypto)
        """
        self.lookback_period = lookback_period
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold
        self.volume_threshold = volume_threshold

        # Cache pour optimiser les calculs
        self._cache: dict[str, Any] = {}

    def detect_regime(
        self,
        highs: list[float] | np.ndarray,
        lows: list[float] | np.ndarray,
        closes: list[float] | np.ndarray,
        volumes: list[float] | np.ndarray,
        symbol: str | None = None,
        _include_analysis: bool = True,
        enable_cache: bool = True,
    ) -> MarketRegime:
        """
        Détecte le régime de marché actuel.

        Args:
            highs: Prix hauts
            lows: Prix bas
            closes: Prix de clôture
            volumes: Volumes
            symbol: Trading symbol for cached indicators (enables performance boost)
            include_analysis: Inclure l'analyse détaillée
            enable_cache: Whether to use cached indicators

        Returns:
            MarketRegime détecté

        Notes:
            - When symbol provided, uses cached indicators (5-10x faster)
            - Automatic fallback to non-cached if symbol not provided
        """
        try:
            # Conversion en arrays numpy
            highs = np.array(highs, dtype=float)
            lows = np.array(lows, dtype=float)
            closes = np.array(closes, dtype=float)
            volumes = np.array(volumes, dtype=float)

            if len(closes) < self.lookback_period:
                return self._unknown_regime()

            # Analyses principales
            volatility_analysis = self._analyze_volatility(
                highs, lows, closes, symbol, enable_cache
            )
            trend_analysis = self._analyze_trend(
                highs, lows, closes, symbol, enable_cache
            )  # Passer highs/lows pour ADX
            momentum_analysis = self._analyze_momentum(
                closes, symbol, enable_cache)
            volume_analysis = self._analyze_volume(volumes)
            structure_analysis = self._analyze_market_structure(
                highs, lows, closes)

            # Détection du régime
            regime_type = self._determine_regime_type(
                volatility_analysis,
                trend_analysis,
                momentum_analysis,
                structure_analysis,
            )

            # Calcul de la force et confiance
            strength = self._calculate_regime_strength(
                volatility_analysis, trend_analysis, momentum_analysis
            )

            confidence = self._calculate_confidence(
                volatility_analysis, trend_analysis, momentum_analysis, volume_analysis)

            # Durée du régime (estimation)
            duration = self._estimate_regime_duration(closes, regime_type)

            # Niveaux clés
            key_levels = self._identify_key_levels(highs, lows, closes)

            return MarketRegime(
                regime_type=regime_type,
                strength=strength,
                confidence=confidence,
                duration=duration,
                volatility=volatility_analysis["current_volatility"],
                trend_slope=trend_analysis["slope"],
                support_resistance_strength=structure_analysis["sr_strength"],
                volume_profile=volume_analysis["profile"],
                key_levels=key_levels,
            )

        except Exception:
            logger.exception("Erreur détection régime")
            return self._unknown_regime()

    def _analyze_volatility(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        _symbol: str | None = None,
        _enable_cache: bool = True,
    ) -> dict:
        """Analyse la volatilité du marché."""
        from ..indicators.volatility.atr import calculate_atr_series
        from ..indicators.volatility.bollinger import calculate_bollinger_bands_series

        # ATR pour volatilité absolue
        atr_series = calculate_atr_series(highs, lows, closes)
        atr_values = [x for x in atr_series if x is not None]

        if len(atr_values) < 20:
            return {"current_volatility": 0.0, "volatility_percentile": 50}

        current_atr = atr_values[-1]
        atr_percentile = self._calculate_percentile(
            atr_values, current_atr, 20)

        # Bollinger Bands width pour volatilité relative
        bb_series = calculate_bollinger_bands_series(closes)
        bb_widths = []

        for i in range(len(bb_series["upper"])):
            upper_val = bb_series["upper"][i]
            lower_val = bb_series["lower"][i]
            middle_val = bb_series["middle"][i]
            if (
                upper_val is not None
                and lower_val is not None
                and middle_val is not None
            ):
                width = (float(upper_val) - float(lower_val)) / \
                    float(middle_val)
                bb_widths.append(width)

        bb_width_percentile = 50.0
        if bb_widths:
            current_bb_width = bb_widths[-1]
            bb_width_percentile = self._calculate_percentile(
                bb_widths, current_bb_width, 20
            )

        # Classification volatilité
        volatility_regime = "normal"
        if atr_percentile > 80 or bb_width_percentile > 80:
            volatility_regime = "high"
        elif atr_percentile < 20 or bb_width_percentile < 20:
            volatility_regime = "low"

        return {
            "current_volatility": current_atr
            / np.mean(closes[-20:]),  # Volatilité normalisée
            "volatility_percentile": atr_percentile,
            "bb_width_percentile": bb_width_percentile,
            "regime": volatility_regime,
        }

    def _analyze_trend(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        symbol: str | None = None,
        enable_cache: bool = True,
    ) -> dict:
        """Analyse la tendance."""
        from ..indicators.trend.adx import calculate_adx_full
        from ..indicators.trend.moving_averages import calculate_ema_series

        # === CALCUL ADX (prioritaire pour classification régime) ===
        adx_full = calculate_adx_full(highs, lows, closes, period=14)
        adx_value = adx_full.get("adx", 0) or 0
        plus_di = adx_full.get("plus_di", 0) or 0
        minus_di = adx_full.get("minus_di", 0) or 0

        # Définir les périodes EMA localement
        ema_periods = {"fast": 7, "medium": 26, "slow": 99}

        # EMAs pour direction de tendance (with caching if symbol provided)
        from ..indicators.trend.moving_averages import calculate_ema

        if symbol and enable_cache:
            # Use cached individual EMA calculations
            calculate_ema(
                closes, ema_periods["fast"], symbol, enable_cache
            )
            calculate_ema(
                closes, ema_periods["medium"], symbol, enable_cache
            )
            calculate_ema(
                closes, ema_periods["slow"], symbol, enable_cache
            )

            # For trend analysis, we need recent values - use series
            ema_fast = calculate_ema_series(closes, ema_periods["fast"])
            ema_medium = calculate_ema_series(closes, ema_periods["medium"])
            ema_slow = calculate_ema_series(closes, ema_periods["slow"])
        else:
            # Use non-cached series
            ema_fast = calculate_ema_series(closes, ema_periods["fast"])
            ema_medium = calculate_ema_series(closes, ema_periods["medium"])
            ema_slow = calculate_ema_series(closes, ema_periods["slow"])

        # Enlever les None
        valid_emas = []
        for i in range(len(closes)):
            if (
                ema_fast[i] is not None
                and ema_medium[i] is not None
                and ema_slow[i] is not None
            ):
                valid_emas.append(
                    {
                        "price": closes[i],
                        "fast": ema_fast[i],
                        "medium": ema_medium[i],
                        "slow": ema_slow[i],
                    }
                )

        if len(valid_emas) < 10:
            return {
                "direction": "unknown",
                "slope": 0.0,
                "strength": 0,
                "ema_alignment": False,
            }

        current = valid_emas[-1]

        # Direction basée sur ordre des EMAs - logique plus nuancée
        fast_above_medium = current["fast"] > current["medium"]
        medium_above_slow = current["medium"] > current["slow"]
        fast_above_slow = current["fast"] > current["slow"]

        if fast_above_medium and medium_above_slow and fast_above_slow:
            direction = "bullish"
        elif not fast_above_medium and not medium_above_slow and not fast_above_slow:
            direction = "bearish"
        elif fast_above_slow:  # 2/3 conditions haussières
            direction = "bullish_partial"
        elif not fast_above_slow:  # 2/3 conditions baissières
            direction = "bearish_partial"
        else:
            direction = "mixed"

        # Pente de tendance (EMA rapide) normalisée par le prix moyen
        recent_ema = [x["fast"] for x in valid_emas[-10:]]
        raw_slope = self._calculate_slope(recent_ema)
        avg_price = np.mean(recent_ema)
        # Normaliser la pente : (pente / prix_moyen) * 100 pour obtenir un
        # pourcentage
        slope = (raw_slope / avg_price) * 100 if avg_price > 0 else 0.0

        # Force de tendance (distance entre EMAs)
        ema_spread = abs(current["fast"] - current["slow"]) / current["medium"]

        return {
            "direction": direction,
            "slope": slope,
            "strength": min(ema_spread * 100, 100),  # 0-100
            "ema_alignment": direction
            in [
                "bullish",
                "bearish",
                "bullish_partial",
                "bearish_partial",
            ],  # Accepter les alignements partiels
            "adx": adx_value,  # ADX pour classification régime
            "plus_di": plus_di,
            "minus_di": minus_di,
        }

    def _analyze_momentum(
        self,
        closes: np.ndarray,
        symbol: str | None = None,
        enable_cache: bool = True,
    ) -> dict:
        """Analyse le momentum."""
        from ..indicators.momentum.rsi import calculate_rsi, calculate_rsi_series
        from ..indicators.trend.macd import calculate_macd_series

        # RSI pour momentum (with caching if symbol provided)
        if symbol and enable_cache:
            # Use cached RSI for current value
            current_rsi = calculate_rsi(closes, 14, symbol, enable_cache)
            # For series analysis, still use non-cached version
            rsi_series = calculate_rsi_series(closes)
        else:
            rsi_series = calculate_rsi_series(closes)
            current_rsi = rsi_series[-1] if rsi_series else None

        [x for x in rsi_series if x is not None]

        momentum_direction = "neutral"
        momentum_strength: float = 0.0

        if current_rsi is not None:
            if current_rsi > 60:
                momentum_direction = "bullish"
                momentum_strength = float(min((current_rsi - 50) * 2, 100))
            elif current_rsi < 40:
                momentum_direction = "bearish"
                momentum_strength = float(min((50 - current_rsi) * 2, 100))

        # MACD pour confirmation
        macd_series = calculate_macd_series(closes)
        macd_direction = "neutral"

        macd_line = macd_series["macd_line"]
        macd_signal = macd_series["macd_signal"]

        for i in range(len(macd_line) - 1, -1, -1):
            macd_val = macd_line[i]
            signal_val = macd_signal[i]
            if macd_val is not None and signal_val is not None:
                macd_direction = "bullish" if float(
                    macd_val) > float(signal_val) else "bearish"
                break

        # Cohérence momentum - plus nuancée
        if momentum_direction == macd_direction:
            momentum_coherent = True
        elif momentum_direction == "neutral" or macd_direction == "neutral":
            # Si l'un est neutre, pas d'incohérence
            momentum_coherent = True
        # Vérifier si c'est juste un décalage temporel normal
        # Si la force du momentum est faible, l'incohérence est moins
        # importante
        elif momentum_strength < 40:
            momentum_coherent = True  # Décalage acceptable en momentum faible
        else:
            momentum_coherent = False

        return {
            "direction": momentum_direction,
            "strength": momentum_strength,
            "macd_direction": macd_direction,
            "coherent": momentum_coherent,
        }

    def _analyze_volume(self, volumes: np.ndarray) -> dict:
        """Analyse le profil de volume."""
        if len(volumes) < 20:
            return {"profile": "insufficient_data", "relative_volume": 1.0}

        # Volume relatif
        recent_avg_volume = np.mean(volumes[-20:])
        current_volume = volumes[-1]
        relative_volume = (
            current_volume /
            recent_avg_volume if recent_avg_volume > 0 else 1.0)

        # Classification profil volume
        if relative_volume > 2.0:
            profile = "spike"
        elif relative_volume > 1.5:
            profile = "high"
        elif relative_volume < 0.5:
            profile = "low"
        else:
            profile = "normal"

        # Trend du volume
        volume_trend = "stable"
        if len(volumes) >= 10:
            recent_volumes = volumes[-10:].astype(float)
            volume_list = [float(v) for v in recent_volumes]
            volume_slope = self._calculate_slope(volume_list)
            if volume_slope > 0.1:
                volume_trend = "increasing"
            elif volume_slope < -0.1:
                volume_trend = "decreasing"

        return {
            "profile": profile,
            "relative_volume": relative_volume,
            "trend": volume_trend,
        }

    def _analyze_market_structure(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
    ) -> dict:
        """Analyse la structure de marché."""
        # Détection de niveaux de support/résistance basique
        recent_highs = highs[-20:]
        recent_lows = lows[-20:]

        # Force des niveaux S/R (variance des prix)
        price_variance = np.var(closes[-20:])
        avg_price = np.mean(closes[-20:])
        sr_strength = 1 - (price_variance / (avg_price**2))  # Normalisé

        # Range du marché
        current_range = (np.max(recent_highs) -
                         np.min(recent_lows)) / avg_price

        # Classification structure
        if current_range < 0.02:  # 2%
            structure = "tight_range"
        elif current_range < 0.05:  # 5%
            structure = "normal_range"
        else:
            structure = "wide_range"

        return {
            "sr_strength": sr_strength,
            "range_size": current_range,
            "structure": structure,
        }

    def _determine_regime_type(
        self, volatility: dict, trend: dict, momentum: dict, structure: dict
    ) -> RegimeType:
        """Détermine le type de régime avec logique optimisée pour crypto."""

        # Extraire ADX
        adx = trend.get("adx", 0)
        plus_di = trend.get("plus_di", 0)
        minus_di = trend.get("minus_di", 0)

        # Log pour comprendre la détection de régime
        logger.info(
            f"🔍 Régime - ADX: {adx:.1f}, +DI: {plus_di:.1f}, -DI: {minus_di:.1f}, "
            f"Volatilité: {volatility['regime']}, Trend: {trend['direction']}, "
            f"Slope: {trend['slope']:.3f}, EMA align: {trend['ema_alignment']}, "
            f"Momentum: {momentum['direction']}, cohérent: {momentum['coherent']}, "
            f"force: {momentum['strength']}")

        # === PRIORITÉ 1: VOLATILITÉ EXTRÊME ===
        if volatility["regime"] == "high":
            # Breakout si momentum fort peu importe la cohérence
            if momentum["strength"] > 60:
                if momentum["direction"] == "bullish" or trend["slope"] > 0.2:
                    return RegimeType.BREAKOUT_BULL
                if momentum["direction"] == "bearish" or trend["slope"] < -0.2:
                    return RegimeType.BREAKOUT_BEAR
            # Sinon juste volatile
            if structure["structure"] == "wide_range":
                return RegimeType.VOLATILE

        # === PRIORITÉ 2: RANGING (marché latéral) basé sur ADX ===
        # ADX < 20 = pas de tendance = RANGING
        if adx < 20:
            return RegimeType.RANGING

        # ADX 20-25 = tendance faible = TRANSITION
        if adx < 25:
            return RegimeType.TRANSITION

        # === PRIORITÉ 3: TENDANCES CLAIRES (ADX >= 25) ===

        # ADX >= 25 = Tendance confirmée
        # Utiliser +DI vs -DI pour déterminer direction
        if adx >= 25:
            # ADX >= 40 = Tendance très forte = TRENDING
            if adx >= 40:
                if plus_di > minus_di:
                    return RegimeType.TRENDING_BULL
                return RegimeType.TRENDING_BEAR

            # ADX 25-40 = Tendance confirmée
            # Vérifier alignement EMA + Momentum
            if plus_di > minus_di:
                # Direction bullish selon DI
                if momentum["direction"] == "bullish" or trend["direction"] in [
                        "bullish", "bullish_partial", ]:
                    return RegimeType.TRENDING_BULL
                # Divergence DI/momentum = possiblement début breakout ou
                # transition
                if momentum["strength"] > 50:
                    return RegimeType.TRANSITION
                return RegimeType.TRENDING_BULL  # Suivre ADX/DI
            # Direction bearish selon DI
            if momentum["direction"] == "bearish" or trend["direction"] in [
                "bearish",
                "bearish_partial",
            ]:
                return RegimeType.TRENDING_BEAR
            # Divergence DI/momentum
            if momentum["strength"] > 50:
                return RegimeType.TRANSITION
            return RegimeType.TRENDING_BEAR  # Suivre ADX/DI

        # === FALLBACK (ne devrait presque jamais arriver) ===
        # Si ADX < 25 et pas de condition ci-dessus remplie
        logger.debug(
            f"Régime par défaut TRANSITION - ADX: {adx:.1f}, trend slope: {trend.get('slope'):.4f}, momentum: {momentum.get('strength')}"
        )
        return RegimeType.TRANSITION

    def _calculate_regime_strength(
        self, volatility: dict, trend: dict, momentum: dict
    ) -> RegimeStrength:
        """Calcule la force du régime avec seuils adaptés crypto."""

        # Score composite 0-100 - OPTIMISÉ POUR CRYPTO
        # Crypto : divergences normales, ne pas trop pénaliser
        trend_score = trend["strength"] * (
            1.0 if trend["ema_alignment"] else 0.7
        )  # 70% au lieu de 60%
        momentum_score = momentum["strength"] * (
            1.0 if momentum["coherent"] else 0.7
        )  # 70% au lieu de 60%
        volatility_score = (
            100 - volatility["volatility_percentile"]
        )  # Inverse pour trending

        # Crypto : volatilité moins importante, trend et momentum plus
        # importants
        composite_score = (
            trend_score * 0.5 + momentum_score * 0.4 + volatility_score * 0.1
        )

        # Seuils plus bas pour crypto (mouvement rapide = fort même si
        # divergent)
        if composite_score > 65:  # Baissé de 75 à 65
            return RegimeStrength.EXTREME
        if composite_score > 40:  # Baissé de 50 à 40
            return RegimeStrength.STRONG
        if composite_score > 20:  # Baissé de 25 à 20
            return RegimeStrength.MODERATE
        return RegimeStrength.WEAK

    def _calculate_confidence(
        self, volatility: dict, trend: dict, momentum: dict, volume: dict
    ) -> float:
        """Calcule le niveau de confiance."""

        confidence_factors = []

        # Cohérence trend/momentum
        if trend["ema_alignment"] and momentum["coherent"]:
            confidence_factors.append(30)

        # Confirmation volume
        if volume["profile"] in ["high",
                                 "spike"] and volume["trend"] == "increasing":
            confidence_factors.append(25)

        # Persistance (si données suffisantes)
        confidence_factors.append(20)  # Base

        # Clarté du signal
        if volatility["regime"] != "normal":
            confidence_factors.append(15)

        # Force des signaux
        if trend["strength"] > 50:
            confidence_factors.append(10)

        return min(sum(confidence_factors), 100)

    def _estimate_regime_duration(
        self, closes: np.ndarray, _regime_type: RegimeType
    ) -> int:
        """Estime la durée du régime actuel."""
        # Analyse des changements récents (simplifié)
        if len(closes) < 20:
            return 1

        # Recherche du dernier changement significatif
        recent_prices = closes[-20:]
        price_changes = np.abs(np.diff(recent_prices))
        significant_changes = np.where(
            price_changes > np.std(price_changes) * 2)[0]

        if len(significant_changes) > 0:
            last_change = significant_changes[-1]
            return 20 - last_change

        return 20  # Max observable

    def _identify_key_levels(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
    ) -> list[float]:
        """Identifie les niveaux de prix clés."""
        key_levels: list[float] = []

        if len(closes) < 20:
            return key_levels

        # Niveaux récents significatifs
        recent_highs = highs[-20:]
        recent_lows = lows[-20:]

        # Résistance (max récent)
        resistance = np.max(recent_highs)
        key_levels.append(float(resistance))

        # Support (min récent)
        support = np.min(recent_lows)
        key_levels.append(float(support))

        # Niveau moyen
        avg_level = np.mean(closes[-20:])
        key_levels.append(float(avg_level))

        return sorted(key_levels)

    def _calculate_percentile(
        self, data: list[float], value: float, min_periods: int = 10
    ) -> float:
        """Calcule le percentile d'une valeur dans un dataset."""
        if len(data) < min_periods:
            return 50.0

        recent_data = data[-min_periods:] if len(data) > min_periods else data
        percentile = (np.sum(np.array(recent_data) <= value) /
                      len(recent_data)) * 100
        return float(percentile)

    def _calculate_slope(self, data: list[float]) -> float:
        """Calcule la pente d'une série de données."""
        if len(data) < 2:
            return 0.0

        x = np.arange(len(data))
        slope = np.polyfit(x, data, 1)[0]
        return float(slope)

    def _unknown_regime(self) -> MarketRegime:
        """Retourne un régime inconnu."""
        return MarketRegime(
            regime_type=RegimeType.UNKNOWN,
            strength=RegimeStrength.WEAK,
            confidence=0.0,
            duration=0,
            volatility=0.0,
            trend_slope=0.0,
            support_resistance_strength=0.0,
            volume_profile="unknown",
            key_levels=[],
        )
