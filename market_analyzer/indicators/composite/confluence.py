"""
Confluence Score Calculator

Ce module calcule un score de confluence entre différents indicateurs techniques
pour mesurer la cohérence des signaux haussiers/baissiers.
"""

import logging
from typing import Dict, Optional, Tuple, List
from enum import Enum

logger = logging.getLogger(__name__)


class ConfluenceType(Enum):
    """Types de confluence pour le calcul."""

    MONO_TIMEFRAME = "mono_timeframe"  # Confluence sur un seul timeframe
    MULTI_TIMEFRAME = "multi_timeframe"  # Confluence entre timeframes


def calculate_confluence_score(
    indicators: Dict,
    current_price: Optional[float] = None,
    confluence_type: ConfluenceType = ConfluenceType.MONO_TIMEFRAME,
) -> float:
    """
    Calcule un score de confluence entre les différents indicateurs techniques.

    Args:
        indicators: Dictionnaire contenant les valeurs des indicateurs
        current_price: Prix actuel de l'actif
        confluence_type: Type de confluence à calculer

    Returns:
        Score entre 0.0 et 100.0 (50 = neutre, >50 = haussier, <50 = baissier)
    """
    try:
        signals: List[float] = []
        weights: List[float] = []

        # === MOYENNES MOBILES ===
        if confluence_type == ConfluenceType.MONO_TIMEFRAME:
            _add_ma_signals(indicators, current_price, signals, weights)

        # === OSCILLATEURS ===
        _add_oscillator_signals(indicators, signals, weights)

        # === TENDANCE ===
        _add_trend_signals(indicators, signals, weights)

        # === VOLUME ===
        _add_volume_signals(indicators, signals, weights)

        # === BOLLINGER BANDS ===
        _add_bollinger_signals(indicators, signals, weights)

        # === MOMENTUM ===
        _add_momentum_signals(indicators, signals, weights)

        # === CALCUL CONFLUENCE FINALE ===
        if not signals:
            return 50.0  # Neutre si aucun signal

        # Moyenne pondérée
        weighted_sum = sum(signal * weight for signal, weight in zip(signals, weights))
        total_weight = sum(weights)
        confluence_score = weighted_sum / total_weight

        # Convertir de 0-1 à 0-100
        confluence_score = confluence_score * 100

        # Normaliser entre 0 et 100
        confluence_score = max(0.0, min(100.0, confluence_score))

        logger.debug(
            f"📊 Confluence calculée: {confluence_score:.1f}% ({len(signals)} signaux)"
        )
        return round(confluence_score, 1)

    except Exception as e:
        logger.warning(f"❌ Erreur calcul confluence: {e}")
        return 50.0  # Valeur neutre en cas d'erreur


def _add_ma_signals(
    indicators: Dict, current_price: Optional[float], signals: list, weights: list
) -> None:
    """Ajoute les signaux des moyennes mobiles."""
    # EMA crossover signals
    ema_12 = indicators.get("ema_12")
    ema_26 = indicators.get("ema_26")
    ema_50 = indicators.get("ema_50")

    if ema_12 and ema_26 and current_price:
        if ema_12 > ema_26:
            signals.append(0.65)  # Signal haussier modéré
            weights.append(2.0)  # Poids important
        else:
            signals.append(0.35)  # Signal baissier modéré
            weights.append(2.0)

    # Prix vs EMA50
    if ema_50 and current_price:
        if current_price > ema_50:
            signals.append(0.6)
            weights.append(1.5)
        else:
            signals.append(0.4)
            weights.append(1.5)


def _add_oscillator_signals(indicators: Dict, signals: list, weights: list) -> None:
    """Ajoute les signaux des oscillateurs."""
    # RSI
    rsi_14 = indicators.get("rsi_14")
    if rsi_14:
        if rsi_14 > 70:
            signals.append(0.2)  # Surachat = baissier
            weights.append(1.5)
        elif rsi_14 < 30:
            signals.append(0.8)  # Survente = haussier
            weights.append(1.5)
        elif rsi_14 > 50:
            signals.append(0.6)  # Au-dessus médiane
            weights.append(1.0)
        else:
            signals.append(0.4)  # En-dessous médiane
            weights.append(1.0)

    # MACD
    macd_line = indicators.get("macd_line")
    macd_signal = indicators.get("macd_signal")
    if macd_line and macd_signal:
        if macd_line > macd_signal:
            signals.append(0.65)  # MACD au-dessus signal = haussier
            weights.append(2.0)
        else:
            signals.append(0.35)  # MACD en-dessous = baissier
            weights.append(2.0)

        # MACD vs zéro
        if macd_line > 0:
            signals.append(0.6)
            weights.append(1.0)
        else:
            signals.append(0.4)
            weights.append(1.0)

    # Stochastic
    stoch_k = indicators.get("stoch_k")
    stoch_d = indicators.get("stoch_d")
    if stoch_k and stoch_d:
        if stoch_k > 80:
            signals.append(0.2)  # Surachat = baissier
            weights.append(1.0)
        elif stoch_k < 20:
            signals.append(0.8)  # Survente = haussier
            weights.append(1.0)
        elif stoch_k > stoch_d:
            signals.append(0.6)  # K au-dessus D = haussier
            weights.append(1.2)
        else:
            signals.append(0.4)  # K en-dessous D = baissier
            weights.append(1.2)


def _add_trend_signals(indicators: Dict, signals: list, weights: list) -> None:
    """Ajoute les signaux de tendance."""
    # ADX et directional indicators
    adx_14 = indicators.get("adx_14")
    plus_di = indicators.get("plus_di")
    minus_di = indicators.get("minus_di")

    if adx_14 and plus_di and minus_di:
        if adx_14 > 25:  # Tendance forte
            if plus_di > minus_di:
                signals.append(0.7)  # Tendance haussière forte
                weights.append(2.5)
            else:
                signals.append(0.3)  # Tendance baissière forte
                weights.append(2.5)
        else:  # Tendance faible/range
            signals.append(0.5)  # Neutre
            weights.append(0.5)


def _add_volume_signals(indicators: Dict, signals: list, weights: list) -> None:
    """Ajoute les signaux de volume."""
    volume_ratio = indicators.get("volume_ratio")
    if volume_ratio:
        if volume_ratio > 1.5:
            # Volume élevé renforce le signal dominant
            if len(signals) > 0:
                avg_signal = sum(
                    s * w for s, w in zip(signals[-3:], weights[-3:])
                ) / sum(weights[-3:])
                if avg_signal > 0.5:
                    signals.append(0.65)  # Renforce haussier
                else:
                    signals.append(0.35)  # Renforce baissier
                weights.append(1.5)
            else:
                signals.append(0.5)  # Neutre si pas d'autres signaux
                weights.append(0.5)


def _add_bollinger_signals(indicators: Dict, signals: list, weights: list) -> None:
    """Ajoute les signaux des Bollinger Bands."""
    bb_position = indicators.get("bb_position")
    if bb_position is not None:
        if bb_position > 0.8:
            signals.append(0.25)  # Proche bande haute = baissier
            weights.append(1.2)
        elif bb_position < 0.2:
            signals.append(0.75)  # Proche bande basse = haussier
            weights.append(1.2)
        else:
            # Position normale dans les bandes
            signals.append(0.5)
            weights.append(0.5)


def _add_momentum_signals(indicators: Dict, signals: list, weights: list) -> None:
    """Ajoute les signaux de momentum."""
    momentum_score = indicators.get("momentum_score")
    if momentum_score is not None:
        # Momentum score est déjà entre 0-100, normaliser vers 0-1
        normalized_momentum = momentum_score / 100
        signals.append(normalized_momentum)
        weights.append(1.8)


def calculate_multi_timeframe_confluence(
    signal_strength: str,
    trend_alignment: str,
    confidence: float,
    risk_level: str = "medium",
) -> float:
    """
    Calcule un score de confluence pour l'analyse multi-timeframe.

    Args:
        signal_strength: Force du signal ('very_weak', 'weak', 'moderate', 'strong', 'very_strong')
        trend_alignment: Alignement des tendances ('conflicting', 'mixed', 'mostly_aligned_bull', etc.)
        confidence: Confiance globale (0-100)
        risk_level: Niveau de risque ('low', 'medium', 'high')

    Returns:
        Score entre 0.0 et 100.0
    """
    strength_score = {
        "very_weak": 0.1,
        "weak": 0.3,
        "moderate": 0.5,
        "strong": 0.7,
        "very_strong": 0.9,
    }.get(signal_strength, 0.5)

    alignment_score = {
        "conflicting": 0.2,
        "mixed": 0.4,
        "mostly_aligned_bull": 0.7,
        "mostly_aligned_bear": 0.7,
        "fully_aligned_bull": 0.9,
        "fully_aligned_bear": 0.9,
    }.get(trend_alignment, 0.5)

    confidence_score = confidence / 100
    risk_penalty = {"low": 1.0, "medium": 0.9, "high": 0.7}.get(risk_level.lower(), 0.9)

    raw_score = (
        0.4 * strength_score + 0.3 * alignment_score + 0.3 * confidence_score
    ) * risk_penalty
    return round(min(max(raw_score * 100, 0.0), 100.0), 1)
