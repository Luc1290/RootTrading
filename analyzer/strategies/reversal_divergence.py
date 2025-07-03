"""
Stratégie de trading basée sur les divergences de prix et d'indicateurs (RSI).
Détecte les divergences entre le prix et le RSI pour identifier les potentiels retournements.
"""
import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import talib

# Importer les modules partagés
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from shared.src.config import get_strategy_param
from shared.src.enums import OrderSide
from shared.src.schemas import StrategySignal

from .base_strategy import BaseStrategy
from .advanced_filters_mixin import AdvancedFiltersMixin
from .strategy_upgrader import wrap_generate_signal

# Configuration du logging
logger = logging.getLogger(__name__)

class ReversalDivergenceStrategy(BaseStrategy, AdvancedFiltersMixin):
    """
    Stratégie basée sur les divergences entre le prix et le RSI.
    Détecte les moments où le prix fait de nouveaux plus bas (ou plus hauts) 
    mais le RSI ne confirme pas, indiquant un potentiel retournement.
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        """
        Initialise la stratégie de divergence.
        
        Args:
            symbol: Symbole de trading (ex: 'BTCUSDC')
            params: Paramètres spécifiques à la stratégie
        """
        super().__init__(symbol, params)
        
        # Paramètres RSI
        self.rsi_window = self.params.get('rsi_window', 14)
        
        # Paramètres pour la détection des divergences
        self.lookback = self.params.get('lookback', 20)  # Nombre de chandeliers à analyser
        self.min_swing_period = self.params.get('min_swing_period', 5)  # Période minimum entre swings
        self.price_threshold = self.params.get('price_threshold', 0.5)  # % minimum de différence entre prix
        
        logger.info(f"🔧 Stratégie de Divergence initialisée pour {symbol} "
                   f"(rsi_window={self.rsi_window}, lookback={self.lookback})")
        
        # Upgrader automatiquement la méthode generate_signal avec filtres sophistiqués
        self._original_generate_signal = self.generate_signal
        self.generate_signal = wrap_generate_signal(self, self._original_generate_signal)
    
    @property
    def name(self) -> str:
        """Nom unique de la stratégie."""
        return "Divergence_Strategy"
    
    def get_min_data_points(self) -> int:
        """
        Nombre minimum de points de données nécessaires.
        
        Returns:
            Nombre minimum de données requises
        """
        return max(self.lookback * 2, 50)  # Besoin de suffisamment de données pour détecter des patterns
    
    def _calculate_rsi(self, prices: np.ndarray) -> np.ndarray:
        """
        Calcule l'indicateur RSI.
        
        Args:
            prices: Tableau des prix de clôture
            
        Returns:
            Tableau des valeurs RSI
        """
        try:
            # Utiliser le module partagé pour calculer le RSI
            from shared.src.technical_indicators import TechnicalIndicators
            ti = TechnicalIndicators()
            
            # Calculer RSI pour toutes les valeurs
            rsi_series = []
            for i in range(self.rsi_window, len(prices)):
                rsi_val = ti.calculate_rsi(prices[:i+1], period=self.rsi_window)
                rsi_series.append(rsi_val if rsi_val is not None else np.nan)
            
            # Remplir le début avec des NaN
            full_rsi = np.full(len(prices), np.nan)
            full_rsi[self.rsi_window:] = rsi_series
            return full_rsi
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul du RSI: {str(e)}")
            return np.full(len(prices), np.nan)
    
    def _find_swing_points(self, data: np.ndarray, min_points: int = 3) -> Tuple[List[int], List[int]]:
        """
        Trouve les points de swing (pivots hauts et bas).
        
        Args:
            data: Tableau de valeurs (prix ou RSI)
            min_points: Nombre minimal de points pour confirmer un pivot
            
        Returns:
            Tuple de (indices_swing_hauts, indices_swing_bas)
        """
        highs = []
        lows = []
        
        # Parcourir les données, en ignorant les extrémités
        for i in range(min_points, len(data) - min_points):
            # Vérifier si c'est un pivot haut (plus haut que les `min_points` points avant et après)
            if all(data[i] > data[i-j] for j in range(1, min_points+1)) and \
               all(data[i] > data[i+j] for j in range(1, min_points+1)):
                highs.append(i)
            
            # Vérifier si c'est un pivot bas (plus bas que les `min_points` points avant et après)
            if all(data[i] < data[i-j] for j in range(1, min_points+1)) and \
               all(data[i] < data[i+j] for j in range(1, min_points+1)):
                lows.append(i)
        
        return highs, lows
    
    def _check_divergence(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Détecte les divergences entre le prix et le RSI avec validation de tendance.
        
        Args:
            df: DataFrame avec les données de prix
            
        Returns:
            Informations sur la divergence ou None
        """
        if df.shape[0] < self.lookback:
            return None
        
        # NOUVEAU: Validation de tendance avant de générer le signal
        trend_alignment = self._validate_trend_alignment_for_signal(df)
        if trend_alignment is None:
            return None  # Pas assez de données pour valider la tendance
        
        # Ne considérer que les N derniers chandeliers
        recent_df = df.iloc[-self.lookback:]
        
        # Extraire les prix de clôture
        prices = recent_df['close'].values
        
        # Calculer le RSI
        rsi_values = self._calculate_rsi(prices)
        
        # Trouver les points de swing pour le prix et le RSI
        price_highs, price_lows = self._find_swing_points(prices, self.min_swing_period)
        rsi_highs, rsi_lows = self._find_swing_points(rsi_values, self.min_swing_period)
        
        # Convertir les indices relatifs en indices absolus
        price_highs = [i + len(df) - self.lookback for i in price_highs]
        price_lows = [i + len(df) - self.lookback for i in price_lows]
        rsi_highs = [i + len(df) - self.lookback for i in rsi_highs]
        rsi_lows = [i + len(df) - self.lookback for i in rsi_lows]
        
        # Chercher une divergence haussière (prix fait un plus bas plus bas, RSI fait un plus bas plus haut)
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            # Prendre les deux derniers pivots bas
            last_price_low_idx, prev_price_low_idx = price_lows[-1], price_lows[-2]
            last_rsi_low_idx, prev_rsi_low_idx = rsi_lows[-1], rsi_lows[-2]
            
            last_price_low = df.iloc[last_price_low_idx]['close']
            prev_price_low = df.iloc[prev_price_low_idx]['close']
            
            # Convertir les indices de prix en indices relatifs pour accéder au RSI
            last_price_low_rel = last_price_low_idx - (len(df) - self.lookback)
            prev_price_low_rel = prev_price_low_idx - (len(df) - self.lookback)
            
            # Vérifier que les indices sont valides
            if 0 <= last_price_low_rel < len(rsi_values) and 0 <= prev_price_low_rel < len(rsi_values):
                last_rsi_low = rsi_values[last_price_low_rel]
                prev_rsi_low = rsi_values[prev_price_low_rel]
                
                # Divergence haussière: prix fait un plus bas plus bas, RSI fait un plus bas plus haut
                if last_price_low < prev_price_low and last_rsi_low > prev_rsi_low:
                    # NOUVEAU: Ne générer divergence BUY que si tendance n'est pas fortement baissière
                    if trend_alignment in ["STRONG_BEARISH", "WEAK_BEARISH"]:
                        logger.debug(f"[Divergence] {self.symbol}: Divergence BUY supprimée - tendance {trend_alignment}")
                        return None
                    
                    # Calculer le pourcentage de divergence
                    price_percent_change = (last_price_low - prev_price_low) / prev_price_low * 100
                    rsi_percent_change = (last_rsi_low - prev_rsi_low) / prev_rsi_low * 100
                    
                    # Calculer un score de force basé sur la divergence
                    score = min(abs(price_percent_change - rsi_percent_change) / 5, 1.0)
                    
                    current_price = df.iloc[-1]['close']
                    
                    # Utiliser l'ATR pour calculer le stop
                    atr_percent = self.calculate_atr(df)
                    
                    # Calculer le stop basé sur l'ATR
                    atr_stop_distance = atr_percent / 100
                    atr_stop = current_price * (1 - atr_stop_distance)
                    
                    # Le stop peut être ajusté pour être sous le dernier plus bas
                    # si c'est plus proche que l'ATR stop
                    low_stop = last_price_low * 0.995
                    stop_loss = max(low_stop, atr_stop)
                    
                    return {
                        "type": "bullish",
                        "side": OrderSide.BUY,
                        "price": current_price,
                        "confidence": 0.75 + (score / 4),  # Confiance entre 0.75 et 1.0 (augmentée)
                        "last_price_low": float(last_price_low),
                        "prev_price_low": float(prev_price_low),
                        "last_rsi_low": float(last_rsi_low),
                        "prev_rsi_low": float(prev_rsi_low),
                        "stop_price": float(stop_loss)
                    }
        
        # Chercher une divergence baissière (prix fait un plus haut plus haut, RSI fait un plus haut plus bas)
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            # Prendre les deux derniers pivots hauts
            last_price_high_idx, prev_price_high_idx = price_highs[-1], price_highs[-2]
            last_rsi_high_idx, prev_rsi_high_idx = rsi_highs[-1], rsi_highs[-2]
            
            last_price_high = df.iloc[last_price_high_idx]['close']
            prev_price_high = df.iloc[prev_price_high_idx]['close']
            
            # Convertir les indices de prix en indices relatifs pour accéder au RSI
            last_price_high_rel = last_price_high_idx - (len(df) - self.lookback)
            prev_price_high_rel = prev_price_high_idx - (len(df) - self.lookback)
            
            # Vérifier que les indices sont valides
            if 0 <= last_price_high_rel < len(rsi_values) and 0 <= prev_price_high_rel < len(rsi_values):
                last_rsi_high = rsi_values[last_price_high_rel]
                prev_rsi_high = rsi_values[prev_price_high_rel]
                
                # Divergence baissière: prix fait un plus haut plus haut, RSI fait un plus haut plus bas
                if last_price_high > prev_price_high and last_rsi_high < prev_rsi_high:
                    # NOUVEAU: Ne générer divergence SELL que si tendance n'est pas fortement haussière
                    if trend_alignment in ["STRONG_BULLISH", "WEAK_BULLISH"]:
                        logger.debug(f"[Divergence] {self.symbol}: Divergence SELL supprimée - tendance {trend_alignment}")
                        return None
                    
                    # Calculer le pourcentage de divergence
                    price_percent_change = (last_price_high - prev_price_high) / prev_price_high * 100
                    rsi_percent_change = (last_rsi_high - prev_rsi_high) / prev_rsi_high * 100
                    
                    # Calculer un score de force basé sur la divergence
                    score = min(abs(price_percent_change - rsi_percent_change) / 5, 1.0)
                    
                    current_price = df.iloc[-1]['close']
                    
                    # Utiliser l'ATR pour calculer le stop
                    atr_percent = self.calculate_atr(df)
                    
                    # Calculer le stop basé sur l'ATR
                    atr_stop_distance = atr_percent / 100
                    atr_stop = current_price * (1 + atr_stop_distance)
                    
                    # Le stop peut être ajusté pour être au-dessus du dernier plus haut
                    # si c'est plus proche que l'ATR stop
                    high_stop = last_price_high * 1.005
                    stop_loss = min(high_stop, atr_stop)
                    
                    return {
                        "type": "bearish",
                        "side": OrderSide.SELL,
                        "price": current_price,
                        "confidence": 0.75 + (score / 4),  # Confiance entre 0.75 et 1.0 (augmentée)
                        "last_price_high": float(last_price_high),
                        "prev_price_high": float(prev_price_high),
                        "last_rsi_high": float(last_rsi_high),
                        "prev_rsi_high": float(prev_rsi_high),
                        "stop_price": float(stop_loss)
                    }
        
        return None
    
    def generate_signal(self) -> Optional[StrategySignal]:
        """
        Génère un signal de trading basé sur les divergences.
        
        Returns:
            Signal de trading ou None si aucun signal n'est généré
        """
        # Convertir les données en DataFrame
        df = self.get_data_as_dataframe()
        if df is None or len(df) < self.get_min_data_points():
            return None
        
        # Détecter une divergence
        divergence = self._check_divergence(df)
        
        if not divergence:
            return None
        
        # Créer le signal
        current_price = divergence['price']
        side = divergence['side']
        confidence = divergence['confidence']
        
        # Préparer les métadonnées
        metadata = {k: v for k, v in divergence.items() if k not in ['side', 'price', 'confidence']}
        
        # Créer et retourner le signal
        signal = self.create_signal(
            side=side,
            price=current_price,
            confidence=confidence,
            metadata=metadata
        )
        
        # Log du signal
        logger.info(f"🔄 [Divergence] Signal {side.value} sur {self.symbol}: "
                   f"divergence {'haussière' if side == OrderSide.BUY else 'baissière'} détectée (confiance: {confidence:.2f})")

        return signal
    
    def _validate_trend_alignment_for_signal(self, df: pd.DataFrame) -> Optional[str]:
        """
        Valide la tendance actuelle pour déterminer si un signal est approprié.
        Utilise la même logique que le signal_aggregator pour cohérence.
        """
        try:
            if df is None or len(df) < 50:
                return None
            
            prices = df['close'].values
            
            # Calculer EMA 21 vs EMA 50 (harmonisé avec signal_aggregator)
            from shared.src.technical_indicators import calculate_ema
            ema_21_val = calculate_ema(prices, period=21)
            ema_50_val = calculate_ema(prices, period=50)
            
            if ema_21_val is None or ema_50_val is None or np.isnan(ema_21_val) or np.isnan(ema_50_val):
                return None
            
            current_price = prices[-1]
            trend_21 = ema_21_val
            trend_50 = ema_50_val
            
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