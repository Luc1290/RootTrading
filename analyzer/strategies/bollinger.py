"""
Stratégie de trading basée sur les bandes de Bollinger.
Génère des signaux lorsque le prix traverse les bandes de Bollinger.
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

class BollingerStrategy(BaseStrategy):
    """
    Stratégie basée sur les bandes de Bollinger.
    Génère des signaux d'achat quand le prix traverse la bande inférieure vers le haut
    et des signaux de vente quand il traverse la bande supérieure vers le bas.
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        """
        Initialise la stratégie Bollinger Bands.
        
        Args:
            symbol: Symbole de trading (ex: 'BTCUSDC')
            params: Paramètres spécifiques à la stratégie
        """
        super().__init__(symbol, params)
        
        # Paramètres Bollinger
        self.window = self.params.get('window', get_strategy_param('bollinger', 'window', 20))
        self.num_std = self.params.get('num_std', get_strategy_param('bollinger', 'num_std', 2.0))
        
        # Variables pour suivre les tendances
        self.prev_price = None
        self.prev_upper = None
        self.prev_lower = None
        
        logger.info(f"🔧 Stratégie Bollinger initialisée pour {symbol} "
                   f"(window={self.window}, num_std={self.num_std})")
    
    @property
    def name(self) -> str:
        """Nom unique de la stratégie."""
        return "Bollinger_Strategy"
    
    def get_min_data_points(self) -> int:
        """
        Nombre minimum de points de données nécessaires pour calculer les bandes de Bollinger.
        
        Returns:
            Nombre minimum de données requises
        """
        # Besoin d'au moins 2 * la fenêtre Bollinger pour avoir un calcul fiable
        return max(self.window * 2, 30)
    
    def calculate_bollinger_bands(self, prices: np.ndarray) -> tuple:
        """
        Calcule les bandes de Bollinger sur une série de prix.
        
        Args:
            prices: Tableau numpy des prix de clôture
            
        Returns:
            Tuple (upper, middle, lower) des bandes de Bollinger
        """
        # Utiliser TA-Lib pour calculer les bandes de Bollinger
        try:
            upper, middle, lower = talib.BBANDS(
                prices, 
                timeperiod=self.window,
                nbdevup=self.num_std,
                nbdevdn=self.num_std,
                matype=0  # Simple Moving Average
            )
            return upper, middle, lower
        except Exception as e:
            logger.error(f"Erreur lors du calcul des bandes de Bollinger: {str(e)}")
            # Implémenter un calcul manuel de secours en cas d'erreur TA-Lib
            return self._calculate_bollinger_manually(prices)
    
    def _calculate_bollinger_manually(self, prices: np.ndarray) -> tuple:
        """
        Calcule les bandes de Bollinger manuellement si TA-Lib n'est pas disponible.
        
        Args:
            prices: Tableau numpy des prix de clôture
            
        Returns:
            Tuple (upper, middle, lower) des bandes de Bollinger
        """
        # Initialiser les tableaux
        middle = np.zeros_like(prices)
        upper = np.zeros_like(prices)
        lower = np.zeros_like(prices)
        
        # Calculer la moyenne mobile simple (SMA)
        for i in range(len(prices)):
            if i < self.window - 1:
                # Pas assez de données
                middle[i] = np.nan
                upper[i] = np.nan
                lower[i] = np.nan
            else:
                # Calcul de la SMA
                segment = prices[i-(self.window-1):i+1]
                middle[i] = np.mean(segment)
                
                # Calcul de l'écart-type
                std = np.std(segment)
                
                # Calcul des bandes
                upper[i] = middle[i] + (self.num_std * std)
                lower[i] = middle[i] - (self.num_std * std)
        
        return upper, middle, lower
    
    def generate_signal(self) -> Optional[StrategySignal]:
        """
        Génère un signal de trading basé sur les bandes de Bollinger.
        
        Returns:
            Signal de trading ou None si aucun signal n'est généré
        """
        # Convertir les données en DataFrame
        df = self.get_data_as_dataframe()
        if df is None or len(df) < self.get_min_data_points():
            return None
        
        # Extraire les prix de clôture
        prices = df['close'].values
        
        # Calculer les bandes de Bollinger
        upper, middle, lower = self.calculate_bollinger_bands(prices)
        
        # Obtenir les dernières valeurs
        current_price = prices[-1]
        prev_price = prices[-2] if len(prices) > 1 else None
        
        current_upper = upper[-1]
        current_lower = lower[-1]
        
        prev_upper = upper[-2] if len(upper) > 1 else None
        prev_lower = lower[-2] if len(lower) > 1 else None
        
        # Loguer les valeurs actuelles
        logger.debug(f"[Bollinger] {self.symbol}: Price={current_price:.2f}, "
                    f"Upper={current_upper:.2f}, Lower={current_lower:.2f}")
        
        # Vérifier les conditions pour générer un signal
        signal = None
        
        if np.isnan(current_upper) or prev_price is None or np.isnan(prev_upper):
            # Pas assez de données pour un signal fiable
            return None
        
        # Signal d'achat: Prix traverse la bande inférieure de bas en haut
        if prev_price <= prev_lower and current_price > current_lower:
            confidence = self._calculate_confidence(current_price, current_lower, current_upper, OrderSide.BUY)
            
            metadata = {
                "bb_upper": float(current_upper),
                "bb_middle": float(middle[-1]),
                "bb_lower": float(current_lower),
                "target_price": float(middle[-1]),  # Target = middle band
                "stop_price": float(current_lower - (current_lower * 0.01))  # Stop 1% below lower band
            }
            
            signal = self.create_signal(
                side=OrderSide.BUY,
                price=current_price,
                confidence=confidence,
                metadata=metadata
            )
        
        # Signal de vente: Prix traverse la bande supérieure de haut en bas
        elif prev_price >= prev_upper and current_price < current_upper:
            confidence = self._calculate_confidence(current_price, current_lower, current_upper, OrderSide.SELL)
            
            metadata = {
                "bb_upper": float(current_upper),
                "bb_middle": float(middle[-1]),
                "bb_lower": float(current_lower),
                "target_price": float(middle[-1]),  # Target = middle band
                "stop_price": float(current_upper + (current_upper * 0.01))  # Stop 1% above upper band
            }
            
            signal = self.create_signal(
                side=OrderSide.SELL,
                price=current_price,
                confidence=confidence,
                metadata=metadata
            )
        
        # Mettre à jour les valeurs précédentes
        self.prev_price = current_price
        self.prev_upper = current_upper
        self.prev_lower = current_lower
        
        return signal
    
    def _calculate_confidence(self, price: float, lower: float, upper: float, side: OrderSide) -> float:
        """
        Calcule le niveau de confiance d'un signal basé sur la position dans les bandes.
        
        Args:
            price: Prix actuel
            lower: Valeur de la bande inférieure
            upper: Valeur de la bande supérieure
            side: Côté du signal (BUY ou SELL)
            
        Returns:
            Niveau de confiance entre 0.0 et 1.0
        """
        # Calcul de la distance du prix dans les bandes
        band_width = upper - lower
        
        if side == OrderSide.BUY:
            # Plus le prix est bas sous la bande inférieure, plus la confiance est élevée
            penetration = (lower - price) / band_width
            # Normaliser entre 0.5 et 1.0
            confidence = 0.5 + min(max(penetration * 2, 0), 0.5)
            return confidence
        else:  # SELL
            # Plus le prix est haut au-dessus de la bande supérieure, plus la confiance est élevée
            penetration = (price - upper) / band_width
            # Normaliser entre 0.5 et 1.0
            confidence = 0.5 + min(max(penetration * 2, 0), 0.5)
            return confidence