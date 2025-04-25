"""
Stratégie de trading basée sur le croisement de moyennes mobiles exponentielles (EMA).
Génère des signaux lorsque l'EMA courte croise l'EMA longue.
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

class EMACrossStrategy(BaseStrategy):
    """
    Stratégie basée sur le croisement de moyennes mobiles exponentielles (EMA).
    Génère des signaux d'achat quand l'EMA courte croise l'EMA longue vers le haut
    et des signaux de vente quand l'EMA courte croise l'EMA longue vers le bas.
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        """
        Initialise la stratégie EMA Cross.
        
        Args:
            symbol: Symbole de trading (ex: 'BTCUSDC')
            params: Paramètres spécifiques à la stratégie
        """
        super().__init__(symbol, params)
        
        # Paramètres EMA
        self.short_window = self.params.get('short_window', get_strategy_param('ema_cross', 'short_window', 5))
        self.long_window = self.params.get('long_window', get_strategy_param('ema_cross', 'long_window', 20))
        
        # S'assurer que court < long
        if self.short_window >= self.long_window:
            logger.warning(f"⚠️ Configuration incorrecte: EMA court ({self.short_window}) >= EMA long ({self.long_window}). Ajustement automatique.")
            self.short_window = min(self.short_window, self.long_window - 1)
        
        # Variables pour suivre les tendances
        self.prev_short_ema = None
        self.prev_long_ema = None
        
        logger.info(f"🔧 Stratégie EMA Cross initialisée pour {symbol} "
                   f"(short={self.short_window}, long={self.long_window})")
    
    @property
    def name(self) -> str:
        """Nom unique de la stratégie."""
        return "EMA_Cross_Strategy"
    
    def get_min_data_points(self) -> int:
        """
        Nombre minimum de points de données nécessaires pour calculer les EMAs.
        
        Returns:
            Nombre minimum de données requises
        """
        # Besoin d'au moins 3 * la fenêtre longue pour avoir un calcul fiable
        return max(self.long_window * 3, 15)
    
    def calculate_emas(self, prices: np.ndarray) -> tuple:
        """
        Calcule les EMAs courte et longue sur une série de prix.
        
        Args:
            prices: Tableau numpy des prix de clôture
            
        Returns:
            Tuple (short_ema, long_ema) des valeurs EMA
        """
        try:
            # Utiliser TA-Lib pour calculer les EMAs
            short_ema = talib.EMA(prices, timeperiod=self.short_window)
            long_ema = talib.EMA(prices, timeperiod=self.long_window)
            return short_ema, long_ema
        except Exception as e:
            logger.error(f"Erreur lors du calcul des EMAs: {str(e)}")
            # Implémenter un calcul manuel de secours en cas d'erreur TA-Lib
            return self._calculate_emas_manually(prices)
    
    def _calculate_emas_manually(self, prices: np.ndarray) -> tuple:
        """
        Calcule les EMAs manuellement si TA-Lib n'est pas disponible.
        
        Args:
            prices: Tableau numpy des prix de clôture
            
        Returns:
            Tuple (short_ema, long_ema) des valeurs EMA
        """
        # Initialiser les tableaux
        short_ema = np.zeros_like(prices)
        long_ema = np.zeros_like(prices)
        
        # Calculer les EMAs
        # Coefficient de lissage
        short_alpha = 2 / (self.short_window + 1)
        long_alpha = 2 / (self.long_window + 1)
        
        # Pour la première valeur, utiliser la moyenne simple (SMA)
        for i in range(len(prices)):
            if i < self.short_window - 1:
                short_ema[i] = np.nan
            elif i == self.short_window - 1:
                short_ema[i] = np.mean(prices[:self.short_window])
            else:
                short_ema[i] = (prices[i] * short_alpha) + (short_ema[i-1] * (1 - short_alpha))
                
            if i < self.long_window - 1:
                long_ema[i] = np.nan
            elif i == self.long_window - 1:
                long_ema[i] = np.mean(prices[:self.long_window])
            else:
                long_ema[i] = (prices[i] * long_alpha) + (long_ema[i-1] * (1 - long_alpha))
        
        return short_ema, long_ema
    
    def generate_signal(self) -> Optional[StrategySignal]:
        """
        Génère un signal de trading basé sur le croisement d'EMAs.
        
        Returns:
            Signal de trading ou None si aucun signal n'est généré
        """
        # Convertir les données en DataFrame
        df = self.get_data_as_dataframe()
        if df is None or len(df) < self.get_min_data_points():
            return None
        
        # Extraire les prix de clôture
        prices = df['close'].values
        
        # Calculer les EMAs
        short_ema, long_ema = self.calculate_emas(prices)
        
        # Obtenir les dernières valeurs
        current_price = prices[-1]
        current_short_ema = short_ema[-1]
        current_long_ema = long_ema[-1]
        
        prev_short_ema = short_ema[-2] if len(short_ema) > 1 else None
        prev_long_ema = long_ema[-2] if len(long_ema) > 1 else None
        
        # Loguer les valeurs actuelles
        logger.info(f"[EMA Cross] {self.symbol}: Price={current_price:.2f}, "
                    f"Short EMA={current_short_ema:.2f}, Long EMA={current_long_ema:.2f}")
        
        # Vérifier les conditions pour générer un signal
        signal = None
        
        if np.isnan(current_short_ema) or np.isnan(current_long_ema) or prev_short_ema is None or prev_long_ema is None:
            # Pas assez de données pour un signal fiable
            return None
        
        # Signal d'achat: EMA courte croise EMA longue vers le haut
        if prev_short_ema <= prev_long_ema and current_short_ema > current_long_ema:
            # Calculer la distance entre les EMAs pour la confiance
            distance_ratio = abs(current_short_ema - current_long_ema) / current_long_ema
            confidence = min(0.5 + (distance_ratio * 10), 0.95)  # Max confidence of 0.95
            
            # Calculer le stop-loss et target
            stop_loss = current_price * 0.98  # 2% sous le prix actuel
            
            # Le target est calculé comme 2x la distance du stop-loss (risk-reward ratio de 1:2)
            target_price = current_price + (2 * (current_price - stop_loss))
            
            metadata = {
                "short_ema": float(current_short_ema),
                "long_ema": float(current_long_ema),
                "ema_distance": float(distance_ratio),
                "target_price": float(target_price),
                "stop_price": float(stop_loss)
            }
            
            signal = self.create_signal(
                side=OrderSide.BUY,
                price=current_price,
                confidence=confidence,
                metadata=metadata
            )
        
        # Signal de vente: EMA courte croise EMA longue vers le bas
        elif prev_short_ema >= prev_long_ema and current_short_ema < current_long_ema:
            # Calculer la distance entre les EMAs pour la confiance
            distance_ratio = abs(current_short_ema - current_long_ema) / current_long_ema
            confidence = min(0.5 + (distance_ratio * 10), 0.95)  # Max confidence of 0.95
            
            # Calculer le stop-loss et target
            stop_loss = current_price * 1.02  # 2% au-dessus du prix actuel
            
            # Le target est calculé comme 2x la distance du stop-loss (risk-reward ratio de 1:2)
            target_price = current_price - (2 * (stop_loss - current_price))
            
            metadata = {
                "short_ema": float(current_short_ema),
                "long_ema": float(current_long_ema),
                "ema_distance": float(distance_ratio),
                "target_price": float(target_price),
                "stop_price": float(stop_loss)
            }
            
            signal = self.create_signal(
                side=OrderSide.SELL,
                price=current_price,
                confidence=confidence,
                metadata=metadata
            )
        
        # Mettre à jour les valeurs précédentes
        self.prev_short_ema = current_short_ema
        self.prev_long_ema = current_long_ema
        
        return signal