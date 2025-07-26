"""
Classe de base pour toutes les stratégies de trading.
Définit l'interface commune que toutes les stratégies doivent implémenter.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Classe de base abstraite pour toutes les stratégies."""
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        """
        Initialise la stratégie avec les données nécessaires.
        
        Args:
            symbol: Le symbole de trading (ex: BTCUSDC)
            data: Données OHLCV depuis la DB
            indicators: Dictionnaire des indicateurs pré-calculés depuis la DB
        """
        self.symbol = symbol
        self.data = data
        self.indicators = indicators
        self.name = self.__class__.__name__
        
    @abstractmethod
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal de trading basé sur la stratégie.
        
        Returns:
            Dict contenant:
            - side: "BUY", "SELL" ou None
            - confidence: float entre 0 et 1
            - strength: "weak", "moderate", "strong", "very_strong"
            - reason: str description du signal
            - metadata: dict avec infos supplémentaires
        """
        pass
        
    def validate_data(self) -> bool:
        """
        Valide que toutes les données nécessaires sont présentes.
        
        Returns:
            True si les données sont valides, False sinon
        """
        if not self.data or not self.indicators:
            logger.warning(f"{self.name}: Données manquantes")
            return False
        return True
        
    def calculate_confidence(self, base_confidence: float, *factors: float) -> float:
        """
        Calcule la confiance finale en multipliant les facteurs.
        
        Args:
            base_confidence: Confiance de base
            *factors: Facteurs multiplicatifs
            
        Returns:
            Confiance finale entre 0 et 1
        """
        confidence = base_confidence
        for factor in factors:
            confidence *= factor
        return min(max(confidence, 0.0), 1.0)
        
    def get_strength_from_confidence(self, confidence: float) -> str:
        """
        Convertit la confiance en force du signal.
        
        Args:
            confidence: Niveau de confiance (0-1)
            
        Returns:
            Force du signal: weak, moderate, strong, very_strong
        """
        if confidence >= 0.8:
            return "very_strong"
        elif confidence >= 0.6:
            return "strong"
        elif confidence >= 0.4:
            return "moderate"
        else:
            return "weak"
