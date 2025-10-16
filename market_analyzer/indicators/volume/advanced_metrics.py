"""
Advanced Volume Metrics Module

Calcule des métriques avancées de volume utilisant quote_asset_volume et number_of_trades
pour une analyse plus approfondie de la qualité du volume.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def calculate_quote_volume_ratio(
    quote_volumes: list[float], lookback: int = 20
) -> float | None:
    """
    Calcule le ratio du volume en quote asset (USDC) par rapport à la moyenne.

    Args:
        quote_volumes: Liste des volumes en quote asset
        lookback: Période pour calculer la moyenne

    Returns:
        Ratio du volume actuel vs moyenne ou None
    """
    if len(quote_volumes) < lookback:
        return None

    current_quote_volume = quote_volumes[-1]
    avg_quote_volume = np.mean(quote_volumes[-lookback:])

    if avg_quote_volume == 0:
        return 1.0

    return float(current_quote_volume / avg_quote_volume)


def calculate_avg_trade_size(volume: float,
                             number_of_trades: int) -> float | None:
    """
    Calcule la taille moyenne des trades (volume/nombre de trades).
    Permet de détecter si le volume vient de gros trades (baleines) ou petits trades (retail).

    Args:
        volume: Volume total
        number_of_trades: Nombre de trades

    Returns:
        Taille moyenne d'un trade ou None
    """
    if number_of_trades == 0:
        return volume  # Si aucun trade, on retourne le volume total

    return float(volume / number_of_trades)


def calculate_trade_intensity(
    trades_counts: list[int], lookback: int = 20
) -> float | None:
    """
    Calcule l'intensité du trading (nombre de trades vs moyenne).
    Permet de détecter les périodes d'activité anormale.

    Args:
        trades_counts: Liste du nombre de trades par période
        lookback: Période pour calculer la moyenne

    Returns:
        Ratio d'intensité ou None
    """
    if len(trades_counts) < lookback:
        return None

    current_trades = trades_counts[-1]
    avg_trades = np.mean(trades_counts[-lookback:])

    if avg_trades == 0:
        return 1.0

    return float(current_trades / avg_trades)


def analyze_volume_quality(
    volumes: list[float],
    quote_volumes: list[float],
    trades_counts: list[int],
    lookback: int = 20,
) -> dict[str, float | None]:
    """
    Analyse complète de la qualité du volume.

    Args:
        volumes: Liste des volumes de base
        quote_volumes: Liste des volumes en quote asset
        trades_counts: Liste du nombre de trades
        lookback: Période d'analyse

    Returns:
        Dictionnaire avec toutes les métriques de qualité
    """
    result: dict[str, float | None] = {
        "quote_volume_ratio": None,
        "avg_trade_size": None,
        "trade_intensity": None,
        "volume_quality_score": None,
        "whale_activity_score": None,
        "retail_activity_score": None,
    }

    # Calculs de base
    if quote_volumes:
        result["quote_volume_ratio"] = calculate_quote_volume_ratio(
            quote_volumes, lookback
        )

    if volumes and trades_counts and len(
            volumes) > 0 and len(trades_counts) > 0:
        current_volume = volumes[-1]
        current_trades = trades_counts[-1]

        result["avg_trade_size"] = calculate_avg_trade_size(
            current_volume, current_trades
        )
        result["trade_intensity"] = calculate_trade_intensity(
            trades_counts, lookback)

        # Score de qualité global (0-100)
        if len(volumes) >= lookback and len(trades_counts) >= lookback:
            avg_trade_size_hist = []
            for i in range(lookback):
                idx = -(lookback - i)
                if trades_counts[idx] > 0:
                    avg_trade_size_hist.append(
                        volumes[idx] / trades_counts[idx])

            if avg_trade_size_hist:
                avg_trade_size_mean = np.mean(avg_trade_size_hist)
                current_avg_trade_size = result["avg_trade_size"] or 0

                # Score baleine (gros trades)
                if avg_trade_size_mean > 0:
                    whale_ratio = current_avg_trade_size / avg_trade_size_mean
                    result["whale_activity_score"] = min(
                        100.0, float(whale_ratio * 25)
                    )  # 4x = 100%

                # Score retail (beaucoup de petits trades)
                if result["trade_intensity"]:
                    result["retail_activity_score"] = min(
                        100.0, float(result["trade_intensity"] * 33.33)
                    )  # 3x = 100%

                # Score de qualité combiné
                quality_factors: list[float] = []

                # Volume élevé + peu de trades = Baleines (haute qualité)
                if result["quote_volume_ratio"] and result["quote_volume_ratio"] > 1.5 and result["trade_intensity"] and result["trade_intensity"] < 1.5:
                    quality_factors.append(80)  # Volume de baleines

                # Volume élevé + beaucoup de trades = FOMO retail (qualité
                # moyenne)
                if result["quote_volume_ratio"] and result["quote_volume_ratio"] > 1.5 and result["trade_intensity"] and result["trade_intensity"] > 2.0:
                    quality_factors.append(60)  # FOMO retail

                # Volume normal + trades normaux = Qualité standard
                if not quality_factors:
                    quality_factors.append(50)

                result["volume_quality_score"] = float(
                    np.mean(quality_factors))

    return result


def detect_volume_anomaly(
        avg_trade_sizes: list[float],
        trades_counts: list[int],
        threshold: float = 3.0) -> str:
    """
    Détecte les anomalies dans les patterns de volume.

    Args:
        avg_trade_sizes: Liste des tailles moyennes de trades
        trades_counts: Liste du nombre de trades
        threshold: Seuil de détection (en écarts-types)

    Returns:
        Type d'anomalie détectée ou 'normal'
    """
    if len(avg_trade_sizes) < 20 or len(trades_counts) < 20:
        return "insufficient_data"

    # Stats sur les 20 dernières périodes
    recent_avg_sizes = avg_trade_sizes[-20:]
    recent_trades = trades_counts[-20:]

    avg_size_mean = np.mean(recent_avg_sizes[:-1])  # Exclure la dernière
    avg_size_std = np.std(recent_avg_sizes[:-1])
    trades_mean = np.mean(recent_trades[:-1])
    trades_std = np.std(recent_trades[:-1])

    current_avg_size = recent_avg_sizes[-1]
    current_trades = recent_trades[-1]

    result = "normal"

    # Détection d'anomalies
    if (
        avg_size_std > 0
        and (current_avg_size - avg_size_mean) / avg_size_std > threshold
    ):
        if (
            trades_std > 0
            and (current_trades - trades_mean) / trades_std < -threshold / 2
        ):
            result = "whale_accumulation"  # Gros trades, peu de transactions
        else:
            result = "large_trades_spike"

    elif trades_std > 0 and (current_trades - trades_mean) / \
            trades_std > threshold:
        if (avg_size_std > 0 and (current_avg_size -
                                  avg_size_mean) / avg_size_std < -threshold / 2):
            result = "retail_frenzy"  # Beaucoup de petits trades
        else:
            result = "high_activity_spike"

    elif avg_size_std > 0 and trades_std > 0:
        size_z = abs((current_avg_size - avg_size_mean) / avg_size_std)
        trades_z = abs((current_trades - trades_mean) / trades_std)
        if size_z > threshold / 2 and trades_z > threshold / 2:
            result = "volume_breakout"  # Anomalie dans les deux métriques

    return result
