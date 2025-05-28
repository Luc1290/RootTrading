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
        return max(self.long_window * 3, 30)
    
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
        # Utiliser plus de précision pour les paires BTC
        precision = 5 if 'BTC' in self.symbol else 2
        logger.info(f"[EMA Cross] {self.symbol}: Price={current_price:.{precision}f}, "
                    f"Short EMA={current_short_ema:.{precision}f}, Long EMA={current_long_ema:.{precision}f}")
        
        # Vérifier les conditions pour générer un signal
        signal = None
        
        if np.isnan(current_short_ema) or np.isnan(current_long_ema) or prev_short_ema is None or prev_long_ema is None:
            # Pas assez de données pour un signal fiable
            return None
        
        # Nouvelle approche : acheter les pullbacks dans une tendance haussière
        # et vendre les rebonds dans une tendance baissière
        
        # Déterminer la tendance actuelle basée sur les EMAs
        is_uptrend = current_short_ema > current_long_ema
        is_downtrend = current_short_ema < current_long_ema
        
        # Calculer la distance entre le prix et les EMAs
        price_to_short_ema = (current_price - current_short_ema) / current_short_ema * 100
        price_to_long_ema = (current_price - current_long_ema) / current_long_ema * 100
        ema_spread = abs(current_short_ema - current_long_ema) / current_long_ema * 100
        
        # Signal d'ACHAT : Prix revient vers les EMAs dans une tendance haussière (pullback)
        if is_uptrend and price_to_short_ema < -0.5:  # Prix sous l'EMA courte d'au moins 0.5%
            # Plus le prix est proche de l'EMA longue, meilleur est le point d'entrée
            if current_price <= current_long_ema * 1.01:  # Prix proche ou sous l'EMA longue
                confidence = 0.85  # Excellente opportunité
                signal_reason = "deep_pullback"
            elif price_to_short_ema < -1.0:  # Prix bien sous l'EMA courte
                confidence = 0.75  # Bonne opportunité
                signal_reason = "moderate_pullback"
            else:
                confidence = 0.65  # Opportunité correcte
                signal_reason = "light_pullback"
            
            # Ajuster la confiance selon la force de la tendance
            if ema_spread > 2.0:  # Tendance très forte
                confidence *= 1.1
            elif ema_spread < 0.5:  # Tendance faible
                confidence *= 0.8
            
            # Stop sous le récent plus bas ou l'EMA longue
            stop_loss = min(current_price * 0.98, current_long_ema * 0.99)
            
            # Target basé sur le retour vers/au-dessus de l'EMA courte
            # S'assurer que le target est toujours supérieur au prix d'entrée pour un BUY
            target_price = max(current_short_ema * 1.01, current_price * 1.005)  # Au minimum +0.5% du prix actuel
            
            metadata = {
                "short_ema": float(current_short_ema),
                "long_ema": float(current_long_ema),
                "price_to_short_ema": float(price_to_short_ema),
                "price_to_long_ema": float(price_to_long_ema),
                "ema_spread": float(ema_spread),
                "trend": "uptrend",
                "reason": signal_reason,
                "target_price": float(target_price),
                "stop_price": float(stop_loss)
            }
            
            signal = self.create_signal(
                side=OrderSide.BUY,
                price=current_price,
                confidence=min(confidence, 0.95),
                metadata=metadata
            )
        
        # Signal de VENTE : Prix remonte vers les EMAs dans une tendance baissière (rebond)
        elif is_downtrend and price_to_short_ema > 0.5:  # Prix au-dessus de l'EMA courte d'au moins 0.5%
            # Plus le prix est proche de l'EMA longue, meilleur est le point de sortie
            if current_price >= current_long_ema * 0.99:  # Prix proche ou au-dessus de l'EMA longue
                confidence = 0.85  # Excellente opportunité
                signal_reason = "strong_bounce"
            elif price_to_short_ema > 1.0:  # Prix bien au-dessus de l'EMA courte
                confidence = 0.75  # Bonne opportunité
                signal_reason = "moderate_bounce"
            else:
                confidence = 0.65  # Opportunité correcte
                signal_reason = "light_bounce"
            
            # Ajuster la confiance selon la force de la tendance
            if ema_spread > 2.0:  # Tendance très forte
                confidence *= 1.1
            elif ema_spread < 0.5:  # Tendance faible
                confidence *= 0.8
            
            # Stop au-dessus du récent plus haut ou l'EMA longue
            stop_loss = max(current_price * 1.02, current_long_ema * 1.01)
            
            # Target basé sur le retour vers/sous l'EMA courte
            # S'assurer que le target est toujours inférieur au prix d'entrée pour un SELL
            target_price = min(current_short_ema * 0.99, current_price * 0.995)  # Au maximum -0.5% du prix actuel
            
            metadata = {
                "short_ema": float(current_short_ema),
                "long_ema": float(current_long_ema),
                "price_to_short_ema": float(price_to_short_ema),
                "price_to_long_ema": float(price_to_long_ema),
                "ema_spread": float(ema_spread),
                "trend": "downtrend",
                "reason": signal_reason,
                "target_price": float(target_price),
                "stop_price": float(stop_loss)
            }
            
            signal = self.create_signal(
                side=OrderSide.SELL,
                price=current_price,
                confidence=min(confidence, 0.95),
                metadata=metadata
            )
        
        # Mettre à jour les valeurs précédentes
        self.prev_short_ema = current_short_ema
        self.prev_long_ema = current_long_ema
        
        return signal