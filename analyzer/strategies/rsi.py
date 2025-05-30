"""
Stratégie de trading basée sur l'indicateur RSI (Relative Strength Index).
Génère des signaux d'achat quand le RSI est survendu et des signaux de vente quand il est suracheté.
"""
import logging
from datetime import datetime
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
        self.overbought_threshold = self.params.get('overbought', get_strategy_param('rsi', 'overbought', 70))
        self.oversold_threshold = self.params.get('oversold', get_strategy_param('rsi', 'oversold', 30))
        
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
        # Vérifier le cooldown avant de générer un signal
        if not self.can_generate_signal():
            return None
            
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
        
        # Signal d'achat: RSI entre en zone de survente (acheter dans le rouge)
        # Deux conditions possibles:
        # 1. RSI vient de passer sous le seuil de survente
        # 2. RSI est déjà en survente et continue de baisser (plus agressif)
        if current_rsi < self.oversold_threshold:
            # Plus le RSI est bas, plus on est confiant
            confidence = self._calculate_confidence(current_rsi, OrderSide.BUY)
            
            # Ne générer un signal que si:
            # - On vient d'entrer en survente (prev_rsi > oversold et current < oversold)
            # - OU le RSI continue de baisser en zone de survente (pour moyenner à la baisse)
            should_buy = False
            signal_reason = ""
            
            if prev_rsi > self.oversold_threshold:
                # Première entrée en survente
                should_buy = True
                signal_reason = "entry_oversold"
                confidence *= 0.8  # Confiance modérée pour la première entrée
            elif current_rsi < prev_rsi - 2:  # RSI baisse de plus de 2 points
                # RSI continue de chuter, opportunité de moyenner
                should_buy = True
                signal_reason = "deepening_oversold"
                confidence *= 1.2  # Plus confiant quand ça baisse encore
            
            if should_buy:
                # Calculer les niveaux de stop/target basés sur l'ATR
                atr_percent = self.calculate_atr(df)
                
                # Pour RSI, utiliser des multiples plus conservateurs
                # Stop: 1.5x ATR, Target: 2x ATR (ratio 1.33)
                stop_distance = atr_percent * 1.5
                target_distance = atr_percent * 2.0
                
                # Pour BUY: stop en dessous, target au-dessus
                stop_price = current_price * (1 - stop_distance / 100)
                target_price = current_price * (1 + target_distance / 100)
                
                metadata = {
                    "rsi": float(current_rsi),
                    "rsi_threshold": self.oversold_threshold,
                    "previous_rsi": float(prev_rsi),
                    "reason": signal_reason,
                    "rsi_delta": float(current_rsi - prev_rsi),
                    "stop_price": float(stop_price),
                    "target_price": float(target_price),
                    "atr_percent": float(atr_percent)
                }
                
                signal = self.create_signal(
                    side=OrderSide.BUY,
                    price=current_price,
                    confidence=min(confidence, 0.95),
                    metadata=metadata
                )
        
        # Signal de vente: RSI entre en zone de surachat (vendre dans le vert)
        elif current_rsi > self.overbought_threshold:
            # Plus le RSI est haut, plus on est confiant
            confidence = self._calculate_confidence(current_rsi, OrderSide.SELL)
            
            # Ne générer un signal que si:
            # - On vient d'entrer en surachat
            # - OU le RSI continue de monter en zone de surachat
            should_sell = False
            signal_reason = ""
            
            if prev_rsi < self.overbought_threshold:
                # Première entrée en surachat
                should_sell = True
                signal_reason = "entry_overbought"
                confidence *= 0.8  # Confiance modérée
            elif current_rsi > prev_rsi + 2:  # RSI monte de plus de 2 points
                # RSI continue de monter, le marché est vraiment suracheté
                should_sell = True
                signal_reason = "extreme_overbought"
                confidence *= 1.2  # Plus confiant
            
            if should_sell:
                # Calculer les niveaux de stop/target basés sur l'ATR
                atr_percent = self.calculate_atr(df)
                
                # Pour RSI, utiliser des multiples plus conservateurs
                # Stop: 1.5x ATR, Target: 2x ATR (ratio 1.33)
                stop_distance = atr_percent * 1.5
                target_distance = atr_percent * 2.0
                
                # Pour SELL: stop au-dessus, target en dessous
                stop_price = current_price * (1 + stop_distance / 100)
                target_price = current_price * (1 - target_distance / 100)
                
                metadata = {
                    "rsi": float(current_rsi),
                    "rsi_threshold": self.overbought_threshold,
                    "previous_rsi": float(prev_rsi),
                    "reason": signal_reason,
                    "rsi_delta": float(current_rsi - prev_rsi),
                    "stop_price": float(stop_price),
                    "target_price": float(target_price),
                    "atr_percent": float(atr_percent)
                }
                
                signal = self.create_signal(
                    side=OrderSide.SELL,
                    price=current_price,
                    confidence=min(confidence, 0.95),
                    metadata=metadata
                )
        
        # Mettre à jour les valeurs précédentes
        self.prev_rsi = current_rsi
        self.prev_price = current_price
        
        # Mettre à jour le timestamp si un signal est généré
        if signal:
            self.last_signal_time = datetime.now()
        
        return signal
    
    def _calculate_confidence(self, rsi_value: float, side: OrderSide) -> float:
        """
        Calcule le niveau de confiance d'un signal basé sur la valeur RSI.
        Plus on est dans l'extrême (très survendu ou très suracheté), plus la confiance est élevée.
        
        Args:
            rsi_value: Valeur actuelle du RSI
            side: Côté du signal (BUY ou SELL)
            
        Returns:
            Niveau de confiance entre 0.0 et 1.0
        """
        if side == OrderSide.BUY:
            # Plus le RSI est bas, plus la confiance est élevée (acheter dans l'extrême rouge)
            # RSI=5 -> confiance ~0.95, RSI=20 -> confiance ~0.6
            if rsi_value >= self.oversold_threshold:
                # Pas en survente, confiance faible
                return 0.3
            
            # Échelle progressive: RSI 0-20 mappé sur confiance 0.95-0.6
            confidence = 0.95 - (rsi_value / self.oversold_threshold) * 0.35
            
            # Bonus si RSI très bas (< 15)
            if rsi_value < 15:
                confidence *= 1.1
                
            return min(confidence, 0.95)
            
        else:  # SELL
            # Plus le RSI est haut, plus la confiance est élevée (vendre dans l'extrême vert)
            # RSI=95 -> confiance ~0.95, RSI=80 -> confiance ~0.6
            if rsi_value <= self.overbought_threshold:
                # Pas en surachat, confiance faible
                return 0.3
            
            # Échelle progressive: RSI 80-100 mappé sur confiance 0.6-0.95
            remaining_range = 100 - self.overbought_threshold
            position_in_range = (rsi_value - self.overbought_threshold) / remaining_range
            confidence = 0.6 + position_in_range * 0.35
            
            # Bonus si RSI très haut (> 85)
            if rsi_value > 85:
                confidence *= 1.1
                
            return min(confidence, 0.95)