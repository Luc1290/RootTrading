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
        self.signal_cooldown = self.params.get('signal_cooldown', 3600)  # Temps min entre signaux (sec)
    
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
    
    def __str__(self) -> str:
        """Représentation sous forme de chaîne de la stratégie."""
        return f"{self.name} Strategy ({self.symbol})"