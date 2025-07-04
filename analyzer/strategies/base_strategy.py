"""
Classe de base pour toutes les stratégies de trading.
Définit l'interface commune que toutes les stratégies doivent implémenter.
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List, Deque
from collections import deque

import numpy as np
import pandas as pd

# Importer les modules partagés
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from shared.src.enums import OrderSide, SignalStrength
from shared.src.schemas import StrategySignal, MarketData

# Configuration du logging
logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Classe abstraite de base pour toutes les stratégies de trading.
    Les stratégies concrètes doivent hériter de cette classe et implémenter ses méthodes.
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        """
        Initialise la stratégie de base.
        
        Args:
            symbol: Symbole de trading (ex: 'BTCUSDC')
            params: Paramètres spécifiques à la stratégie
        """
        self.symbol = symbol
        self.params = params or {}
        self.buffer_size = self.params.get('buffer_size', 100)  # Taille par défaut du buffer
        self.data_buffer = deque(maxlen=self.buffer_size)  # Buffer circulaire pour stocker les données
        self.last_signal_time: Optional[datetime] = None
        self.signal_cooldown = self.params.get('signal_cooldown', 45)  # Temps min entre signaux (sec) - réduit pour confluence
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Nom unique de la stratégie.
        Doit être implémenté par les classes dérivées.
        
        Returns:
            Nom de la stratégie
        """
        pass
    
    def add_market_data(self, data: Dict[str, Any]) -> None:
        # Vérifier que les données concernent le bon symbole
        if data.get('symbol') != self.symbol:
            return
    
        # Ajouter au buffer uniquement si le chandelier est fermé
        if data.get('is_closed', False):
            # Vérifier si le buffer est devenu trop grand (sécurité additionnelle)
            if len(self.data_buffer) >= self.buffer_size * 2:
                # Vider la moitié du buffer
                for _ in range(self.buffer_size):
                    if self.data_buffer:
                        self.data_buffer.popleft()
        
            self.data_buffer.append(data)
        
            # Déboguer les données
            logger.info(f"[{self.name}] Données ajoutées pour {self.symbol}: "
                        f"close={data['close']}, time={datetime.fromtimestamp(data['start_time']/1000)}")
    
    def get_data_as_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Convertit les données du buffer en DataFrame pandas.
        
        Returns:
            DataFrame pandas avec les données de marché, ou None si pas assez de données
        """
        if len(self.data_buffer) == 0:
            return None
        
        # Convertir le deque en DataFrame
        df = pd.DataFrame(list(self.data_buffer))
        
        # Convertir les timestamps en datetime
        if 'start_time' in df.columns:
            df['datetime'] = pd.to_datetime(df['start_time'], unit='ms')
            df.set_index('datetime', inplace=True)
        
        return df
    
    def can_generate_signal(self) -> bool:
        """
        Vérifie si la stratégie peut générer un signal basé sur le cooldown.
        
        Returns:
            True si un signal peut être généré, False sinon
        """
        if not self.last_signal_time:
            return True
        
        now = datetime.now()
        elapsed = (now - self.last_signal_time).total_seconds()
        
        return elapsed >= self.signal_cooldown
    
    @abstractmethod
    def generate_signal(self) -> Optional[StrategySignal]:
        """
        Génère un signal de trading basé sur les données de marché.
        Doit être implémenté par les classes dérivées.
        
        Returns:
            Signal de trading ou None si aucun signal n'est généré
        """
        pass
    
    def analyze(self) -> Optional[StrategySignal]:
        """
        Analyse les données de marché et génère un signal si les conditions sont remplies.
        
        Returns:
            Signal de trading ou None
        """
        # Vérifier s'il y a assez de données
        if len(self.data_buffer) < self.get_min_data_points():
            logger.info(f"[{self.name}] Pas assez de données pour générer un signal ({len(self.data_buffer)}/{self.get_min_data_points()})")
            return None
        
        # Vérifier le cooldown des signaux
        if not self.can_generate_signal():
            return None
        
        # Générer un signal
        signal = self.generate_signal()
        
        if signal:
            # Mettre à jour le timestamp du dernier signal
            self.last_signal_time = datetime.now()
            
            # Loguer le signal
            logger.info(f"🔔 [{self.name}] Signal généré pour {self.symbol}: {signal.side} @ {signal.price}")
        
        return signal
    
    def get_min_data_points(self) -> int:
        """
        Retourne le nombre minimum de points de données nécessaires pour générer un signal.
        Peut être surchargé par les classes dérivées.
        
        Returns:
            Nombre minimum de points de données
        """
        # Par défaut, utiliser la taille du buffer ou une valeur minimale
        return min(self.buffer_size, 20)
    
    def create_signal(self, side: OrderSide, price: float, confidence: float = 0.7, 
                    metadata: Dict[str, Any] = None) -> StrategySignal:
        """
        Crée un objet signal standardisé.
        
        Args:
            side: Côté de l'ordre (BUY ou SELL)
            price: Prix actuel
            confidence: Niveau de confiance (0.0 à 1.0)
            metadata: Métadonnées supplémentaires spécifiques à la stratégie
            
        Returns:
            Objet signal standardisé
        """
        # Déterminer la force du signal basée sur la confiance
        strength = SignalStrength.WEAK
        if confidence >= 0.9:
            strength = SignalStrength.VERY_STRONG
        elif confidence >= 0.75:
            strength = SignalStrength.STRONG
        elif confidence >= 0.5:
            strength = SignalStrength.MODERATE
        
        # Créer le signal
        return StrategySignal(
            strategy=self.name,
            symbol=self.symbol,
            side=side,
            timestamp=datetime.now(),
            price=price,
            confidence=confidence,
            strength=strength,
            metadata=metadata or {}
        )
    
    def calculate_atr(self, df: pd.DataFrame = None, period: int = 14) -> float:
        """
        Calcule l'ATR (Average True Range) pour mesurer la volatilité.
        
        Args:
            df: DataFrame avec les données (si None, utilise le buffer)
            period: Période pour le calcul de l'ATR
            
        Returns:
            Valeur ATR en pourcentage du prix actuel
        """
        if df is None:
            df = self.get_data_as_dataframe()
        
        if df is None or len(df) < period + 1:
            return 1.0  # Retourner 1% par défaut si pas assez de données
        
        # Calculer le True Range
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)  
        df['close'] = df['close'].astype(float)
        
        df['previous_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['previous_close'])
        df['tr3'] = abs(df['low'] - df['previous_close'])
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculer l'ATR comme la moyenne mobile du True Range
        atr = df['true_range'].rolling(window=period).mean().iloc[-1]
        
        # Normaliser en pourcentage du prix actuel
        current_price = df['close'].iloc[-1]
        atr_percent = (atr / current_price) * 100
        
        return atr_percent
    
    def calculate_dynamic_stop(self, entry_price: float, side: OrderSide, 
                             atr_percent: float = None) -> Dict[str, float]:
        """
        Calcule seulement le stop dynamique basé sur l'ATR (plus de target avec TrailingStop pur).
        
        Args:
            entry_price: Prix d'entrée
            side: Côté du trade (BUY ou SELL)
            atr_percent: ATR en pourcentage (si None, calculé automatiquement)
            
        Returns:
            Dict avec stop_price seulement
        """
        if atr_percent is None:
            atr_percent = self.calculate_atr()
        
        # Limiter l'ATR pour éviter des stops trop extrêmes
        atr_percent = max(0.3, min(atr_percent, 2.0))
        
        # Pour les paires crypto, ajuster selon la volatilité moyenne
        if 'BTC' in self.symbol:
            atr_multiplier = 0.8
        elif 'ETH' in self.symbol:
            atr_multiplier = 1.0
        else:
            atr_multiplier = 1.2
        
        # Distance de base pour le stop
        # MODIFIÉ: Augmentation du stop loss pour éviter les sorties prématurées
        if 'BTC' in self.symbol:
            base_stop_mult = 2.0  # Augmenté de 1.5 à 2.0 pour BTC
        else:
            base_stop_mult = 3.0  # Augmenté de 2.5 à 3.0 pour les altcoins
        
        stop_distance_percent = atr_percent * atr_multiplier * base_stop_mult
        
        # Calculer le prix de stop
        if side == OrderSide.BUY:
            stop_price = entry_price * (1 - stop_distance_percent / 100)
        else:  # SELL
            stop_price = entry_price * (1 + stop_distance_percent / 100)
        
        return {
            "stop_price": stop_price,
            "atr_percent": atr_percent,
            "stop_distance_percent": stop_distance_percent
        }
    
    def _validate_trend_alignment_for_signal(self) -> Optional[str]:
        """
        Valide la tendance actuelle pour déterminer si un signal est approprié.
        Utilise la même logique que le signal_aggregator pour cohérence.
        Méthode commune disponible pour toutes les stratégies.
        """
        try:
            df = self.get_data_as_dataframe()
            if df is None or len(df) < 50:
                return None
            
            prices = df['close'].values
            
            # Calculer EMA 21 vs EMA 50 (harmonisé avec signal_aggregator)
            def ema(data, period):
                """Calcul EMA simple sans dépendance externe."""
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
    
    def __str__(self) -> str:
        """Représentation sous forme de chaîne de la stratégie."""
        return f"{self.name} Strategy ({self.symbol})"