"""
Module d'upgrade automatique pour les stratégies.
Ajoute des filtres sophistiqués aux generate_signal existants sans tout réécrire.
"""
import logging
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from shared.src.enums import OrderSide
from shared.src.schemas import StrategySignal

logger = logging.getLogger(__name__)


def upgrade_signal_with_filters(strategy_instance, original_signal: Optional[StrategySignal], 
                               df: pd.DataFrame) -> Optional[StrategySignal]:
    """
    Upgrade un signal existant avec des filtres sophistiqués incluant la validation de tendance.
    
    Args:
        strategy_instance: Instance de la stratégie (doit hériter AdvancedFiltersMixin)
        original_signal: Signal original généré par la stratégie
        df: DataFrame des données de marché
        
    Returns:
        Signal upgradé ou None si filtré
    """
    if original_signal is None:
        return None
    
    try:
        # === VALIDATION DE TENDANCE PRIORITAIRE ===
        trend_alignment = _validate_trend_alignment_for_signal(df)
        if trend_alignment is None:
            return original_signal  # Pas assez de données, retourner signal original
        
        # Filtrer selon la tendance
        if original_signal.side == OrderSide.BUY and trend_alignment in ["STRONG_BEARISH", "WEAK_BEARISH"]:
            logger.debug(f"[{strategy_instance.name}] Signal BUY supprimé - tendance {trend_alignment}")
            return None
        elif original_signal.side == OrderSide.SELL and trend_alignment in ["STRONG_BULLISH", "WEAK_BULLISH"]:
            logger.debug(f"[{strategy_instance.name}] Signal SELL supprimé - tendance {trend_alignment}")
            return None
        
        # === APPLICATION DES FILTRES SOPHISTIQUES ===
        
        # 1. FILTRE VOLUME
        volumes = df['volume'].values if 'volume' in df.columns else None
        volume_score = strategy_instance._analyze_volume_confirmation_common(volumes) if volumes is not None else 0.7
        if volume_score < 0.4:
            logger.debug(f"[{strategy_instance.name}] Signal rejeté - volume insuffisant ({volume_score:.2f})")
            return None
        
        # 2. FILTRE TREND ALIGNMENT
        trend_score = strategy_instance._analyze_trend_alignment_common(df, original_signal.side)
        
        # 3. FILTRE RSI CONFIRMATION
        rsi_score = strategy_instance._calculate_rsi_confirmation_common(df, original_signal.side)
        
        # 4. FILTRE SUPPORT/RESISTANCE
        current_price = df['close'].iloc[-1]
        sr_score = strategy_instance._detect_support_resistance_common(df, current_price, original_signal.side)
        
        # 5. FILTRE ATR ENVIRONMENT
        atr_score = strategy_instance._analyze_atr_environment_common(df)
        
        # === CALCUL DE CONFIANCE COMPOSITE ===
        scores = {
            'original': original_signal.confidence,  # Confiance originale
            'volume': volume_score,
            'trend': trend_score,
            'rsi': rsi_score,
            'sr': sr_score,
            'atr': atr_score,
            'trend_alignment': _get_trend_alignment_score(trend_alignment, original_signal.side)
        }
        
        weights = {
            'original': 0.35,          # Poids principal à la stratégie originale
            'volume': 0.20,            # Volume important
            'trend': 0.15,             # Tendance supérieure
            'trend_alignment': 0.15,   # Validation tendance harmoniseée
            'rsi': 0.08,               # RSI confirmation
            'sr': 0.05,                # Support/résistance
            'atr': 0.02               # Environnement ATR
        }
        
        confidence = strategy_instance._calculate_composite_confidence_common(scores, weights)
        
        # Seuil minimum de confiance
        if confidence < 0.65:
            logger.debug(f"[{strategy_instance.name}] Signal rejeté - confiance trop faible ({confidence:.2f})")
            return None
        
        # === UPGRADE DU SIGNAL ===
        # Créer une copie du signal avec confiance upgradée
        upgraded_signal = StrategySignal(
            strategy=original_signal.strategy,
            symbol=original_signal.symbol,
            side=original_signal.side,
            timestamp=original_signal.timestamp,
            price=original_signal.price,
            confidence=confidence,  # Nouvelle confiance composite
            strength=original_signal.strength,
            metadata=original_signal.metadata.copy()
        )
        
        # Ajouter les métadonnées des filtres
        upgraded_signal.metadata.update({
            'upgraded_with_filters': True,
            'original_confidence': original_signal.confidence,
            'volume_score': volume_score,
            'trend_score': trend_score,
            'rsi_score': rsi_score,
            'sr_score': sr_score,
            'atr_score': atr_score,
            'trend_alignment': trend_alignment,
            'trend_alignment_score': _get_trend_alignment_score(trend_alignment, original_signal.side),
            'filter_version': '2.0'
        })
        
        logger.info(f"🔄 [{strategy_instance.name}] Signal upgradé: confiance {original_signal.confidence:.2f} → {confidence:.2f}")
        
        return upgraded_signal
        
    except Exception as e:
        logger.error(f"Erreur upgrade signal: {e}")
        # Retourner le signal original en cas d'erreur
        return original_signal


def wrap_generate_signal(strategy_instance, original_generate_signal_method):
    """
    Wrapper pour la méthode generate_signal qui applique automatiquement les filtres.
    
    Args:
        strategy_instance: Instance de la stratégie
        original_generate_signal_method: Méthode generate_signal originale
        
    Returns:
        Méthode generate_signal wrappée
    """
    def wrapped_generate_signal():
        # Appeler la méthode originale
        original_signal = original_generate_signal_method()
        
        # Si pas de signal, retourner None
        if original_signal is None:
            return None
        
        # Obtenir les données DataFrame
        df = strategy_instance.get_data_as_dataframe()
        if df is None:
            return original_signal  # Retourner l'original si pas de données
        
        # Appliquer les filtres sophistiqués
        return upgrade_signal_with_filters(strategy_instance, original_signal, df)
    
    return wrapped_generate_signal


def _validate_trend_alignment_for_signal(df: pd.DataFrame) -> Optional[str]:
    """
    Valide la tendance actuelle pour déterminer si un signal est approprié.
    Utilise la même logique que le signal_aggregator pour cohérence.
    """
    try:
        if df is None or len(df) < 50:
            return None
        
        prices = df['close'].values
        
        # Calculer EMA 21 vs EMA 50 (harmonisé avec signal_aggregator)
        def ema(data, period):
            """Calcul EMA simple sans dépendance TA-Lib."""
            alpha = 2 / (period + 1)
            ema_values = np.zeros_like(data)
            ema_values[0] = data[0]
            for i in range(1, len(data)):
                ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i-1]
            return ema_values
        
        ema_21 = ema(prices, 21)
        ema_50 = ema(prices, 50)
        
        current_price = prices[-1]
        trend_21 = ema_21[-1]
        trend_50 = ema_50[-1]
        
        # Classification sophistiquée de la tendance (même logique que signal_aggregator)
        if trend_21 > trend_50 * 1.015:  # +1.5% = forte haussière
            return "STRONG_BULLISH"
        elif trend_21 > trend_50 * 1.005:  # +0.5% = faible haussière
            return "WEAK_BULLISH"
        elif trend_21 < trend_50 * 0.985:  # -1.5% = forte baissière
            return "STRONG_BEARISH"
        elif trend_21 < trend_50 * 0.995:  # -0.5% = faible baissière
            return "WEAK_BEARISH"
        else:
            return "NEUTRAL"
            
    except Exception as e:
        logger.warning(f"Erreur validation tendance: {e}")
        return None


def _get_trend_alignment_score(trend_alignment: str, signal_side: OrderSide) -> float:
    """
    Calcule un score d'alignement avec la tendance.
    """
    if trend_alignment is None:
        return 0.7
    
    if signal_side == OrderSide.BUY:
        if trend_alignment == "STRONG_BULLISH":
            return 0.95
        elif trend_alignment == "WEAK_BULLISH":
            return 0.85
        elif trend_alignment == "NEUTRAL":
            return 0.75
        elif trend_alignment == "WEAK_BEARISH":
            return 0.4
        else:  # STRONG_BEARISH
            return 0.2
    else:  # SELL
        if trend_alignment == "STRONG_BEARISH":
            return 0.95
        elif trend_alignment == "WEAK_BEARISH":
            return 0.85
        elif trend_alignment == "NEUTRAL":
            return 0.75
        elif trend_alignment == "WEAK_BULLISH":
            return 0.4
        else:  # STRONG_BULLISH
            return 0.2