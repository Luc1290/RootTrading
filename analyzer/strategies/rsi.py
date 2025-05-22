"""
Stratégie de trading basée sur l'indicateur RSI (Relative Strength Index).
Génère des signaux d'achat quand le RSI est survendu et des signaux de vente quand il est suracheté.
"""
import logging
from typing import Dict, Any, Optional, List
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

# Configuration du logging
logger = logging.getLogger(__name__)

class RSIStrategy(BaseStrategy):
    """
    Stratégie basée sur l'indicateur RSI (Relative Strength Index).
    Génère des signaux d'achat quand le RSI est en zone de survente et des signaux de vente 
    quand il est en zone de surachat.
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        """
        Initialise la stratégie RSI.
        
        Args:
            symbol: Symbole de trading (ex: 'BTCUSDC')
            params: Paramètres spécifiques à la stratégie
        """
        super().__init__(symbol, params)
        
        # Paramètres RSI
        self.rsi_window = self.params.get('window', get_strategy_param('rsi', 'window', 14))
        self.overbought_threshold = self.params.get('overbought', get_strategy_param('rsi', 'overbought', 80))
        self.oversold_threshold = self.params.get('oversold', get_strategy_param('rsi', 'oversold', 20))
        
        # Variables pour suivre les tendances
        self.prev_rsi = None
        self.prev_price = None
        
        logger.info(f"🔧 Stratégie RSI initialisée pour {symbol} "
                   f"(window={self.rsi_window}, overbought={self.overbought_threshold}, "
                   f"oversold={self.oversold_threshold})")
    
    @property
    def name(self) -> str:
        """Nom unique de la stratégie."""
        return "RSI_Strategy"
    
    def get_min_data_points(self) -> int:
        """
        Nombre minimum de points de données nécessaires pour calculer le RSI.
        
        Returns:
            Nombre minimum de données requises
        """
        # Besoin d'au moins 2 * la fenêtre RSI pour avoir un calcul fiable
        return max(self.rsi_window * 2, 15)
    
    def calculate_rsi(self, prices: np.ndarray) -> np.ndarray:
        """
        Calcule l'indicateur RSI sur une série de prix.
        
        Args:
            prices: Tableau numpy des prix de clôture
            
        Returns:
            Tableau numpy des valeurs RSI
        """
        # Utiliser TA-Lib pour calculer le RSI
        try:
            rsi = talib.RSI(prices, timeperiod=self.rsi_window)
            return rsi
        except Exception as e:
            logger.error(f"Erreur lors du calcul du RSI: {str(e)}")
            # Implémenter un calcul manuel de secours en cas d'erreur TA-Lib
            return self._calculate_rsi_manually(prices)
    
    def _calculate_rsi_manually(self, prices: np.ndarray) -> np.ndarray:
        """
        Calcule le RSI manuellement si TA-Lib n'est pas disponible.
        
        Args:
            prices: Tableau numpy des prix de clôture
            
        Returns:
            Tableau numpy des valeurs RSI
        """
        # Calculer les variations de prix
        deltas = np.diff(prices)
        
        # Padding pour maintenir la taille
        deltas = np.append([0], deltas)
        
        # Séparer les variations positives et négatives
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Initialiser les tableaux
        avg_gains = np.zeros_like(prices)
        avg_losses = np.zeros_like(prices)
        
        # Calculer les moyennes mobiles des gains et pertes
        for i in range(len(prices)):
            if i < self.rsi_window:
                # Pas assez de données
                avg_gains[i] = np.nan
                avg_losses[i] = np.nan
            elif i == self.rsi_window:
                # Première moyenne
                avg_gains[i] = np.mean(gains[1:i+1])
                avg_losses[i] = np.mean(losses[1:i+1])
            else:
                # Moyennes suivantes (formule EMA)
                avg_gains[i] = (avg_gains[i-1] * (self.rsi_window-1) + gains[i]) / self.rsi_window
                avg_losses[i] = (avg_losses[i-1] * (self.rsi_window-1) + losses[i]) / self.rsi_window
        
        # Calculer le RS (Relative Strength)
        rs = avg_gains / (avg_losses + 1e-10)  # Éviter la division par zéro
        
        # Calculer le RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signal(self) -> Optional[StrategySignal]:
        """
        Génère un signal de trading basé sur l'indicateur RSI.
        
        Returns:
            Signal de trading ou None si aucun signal n'est généré
        """
        # Convertir les données en DataFrame
        df = self.get_data_as_dataframe()
        if df is None or len(df) < self.get_min_data_points():
            return None
        
        # Extraire les prix de clôture
        prices = df['close'].values
        
        # Calculer le RSI
        rsi_values = self.calculate_rsi(prices)
        
        # Obtenir les dernières valeurs
        current_rsi = rsi_values[-1]
        prev_rsi = rsi_values[-2] if len(rsi_values) > 1 else None
        
        current_price = df['close'].iloc[-1]
        
        # Loguer les valeurs actuelles
        # Utiliser plus de précision pour les paires BTC
        precision = 5 if 'BTC' in self.symbol else 2
        logger.info(f"[RSI] {self.symbol}: RSI={current_rsi:.2f}, Price={current_price:.{precision}f}")
        
        # Vérifier les conditions pour générer un signal
        signal = None
        
        if np.isnan(current_rsi) or prev_rsi is None or np.isnan(prev_rsi):
            # Pas assez de données pour un signal fiable
            return None
        
        # Signal d'achat: RSI passe au-dessus du seuil de survente
        if prev_rsi <= self.oversold_threshold and current_rsi > self.oversold_threshold:
            confidence = self._calculate_confidence(current_rsi, OrderSide.BUY)
            
            metadata = {
                "rsi": current_rsi,
                "rsi_threshold": self.oversold_threshold,
                "previous_rsi": prev_rsi
            }
            
            signal = self.create_signal(
                side=OrderSide.BUY,
                price=current_price,
                confidence=confidence,
                metadata=metadata
            )
        
        # Signal de vente: RSI passe en-dessous du seuil de surachat
        elif prev_rsi >= self.overbought_threshold and current_rsi < self.overbought_threshold:
            confidence = self._calculate_confidence(current_rsi, OrderSide.SELL)
            
            metadata = {
                "rsi": current_rsi,
                "rsi_threshold": self.overbought_threshold,
                "previous_rsi": prev_rsi
            }
            
            signal = self.create_signal(
                side=OrderSide.SELL,
                price=current_price,
                confidence=confidence,
                metadata=metadata
            )
        
        # Mettre à jour les valeurs précédentes
        self.prev_rsi = current_rsi
        self.prev_price = current_price
        
        return signal
    
    def _calculate_confidence(self, rsi_value: float, side: OrderSide) -> float:
        """
        Calcule le niveau de confiance d'un signal basé sur la valeur RSI.
        
        Args:
            rsi_value: Valeur actuelle du RSI
            side: Côté du signal (BUY ou SELL)
            
        Returns:
            Niveau de confiance entre 0.0 et 1.0
        """
        if side == OrderSide.BUY:
            # Plus le RSI est bas sous le seuil de survente, plus la confiance est élevée
            # RSI=10 -> confiance élevée, RSI=29 -> confiance faible
            base_confidence = (self.oversold_threshold - rsi_value) / self.oversold_threshold
            # Limiter entre 0 et 0.5
            base_confidence = min(max(base_confidence, 0), 0.5)
            # Ajuster à l'échelle 0.5-1.0
            return 0.5 + base_confidence
        else:  # SELL
            # Plus le RSI est élevé au-dessus du seuil de surachat, plus la confiance est élevée
            # RSI=90 -> confiance élevée, RSI=71 -> confiance faible
            base_confidence = (rsi_value - self.overbought_threshold) / (100 - self.overbought_threshold)
            # Limiter entre 0 et 0.5
            base_confidence = min(max(base_confidence, 0), 0.5)
            # Ajuster à l'échelle 0.5-1.0
            return 0.5 + base_confidence