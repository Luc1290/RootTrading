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
    Upgrade un signal existant avec des filtres sophistiqués.
    
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
            'atr': atr_score
        }
        
        weights = {
            'original': 0.40,  # Poids principal à la stratégie originale
            'volume': 0.20,    # Volume important
            'trend': 0.15,     # Tendance supérieure
            'rsi': 0.10,       # RSI confirmation
            'sr': 0.10,        # Support/résistance
            'atr': 0.05       # Environnement ATR
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
            'filter_version': '1.0'
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