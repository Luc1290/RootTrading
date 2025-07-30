"""
Classe de base pour tous les validators.
Définit l'interface commune pour la validation des signaux.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseValidator(ABC):
    """Classe de base abstraite pour tous les validators."""
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        """
        Initialise le validator avec les données nécessaires.
        
        Args:
            symbol: Le symbole de trading (ex: BTCUSDC)
            data: Données OHLCV et indicateurs depuis la DB
            context: Contexte du marché (regime, volume, trend, etc.)
        """
        self.symbol = symbol
        self.data = data
        self.context = context
        self.name = self.__class__.__name__
        
    @abstractmethod
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide un signal dans le contexte actuel.
        
        Args:
            signal: Signal à valider contenant side, confidence, strength, etc.
            
        Returns:
            True si le signal est valide, False sinon
        """
        pass
        
    def get_validation_score(self, signal: Dict[str, Any]) -> float:
        """
        Calcule un score de validation entre 0 et 1.
        Par défaut retourne 1.0 si valide, 0.0 sinon.
        
        Args:
            signal: Signal à scorer
            
        Returns:
            Score entre 0 et 1
        """
        return 1.0 if self.validate_signal(signal) else 0.0
        
    def validate_data(self) -> bool:
        """
        Valide que toutes les données nécessaires sont présentes.
        
        Returns:
            True si les données sont valides, False sinon
        """
        if not self.data or not self.context:
            logger.warning(f"{self.name}: Données ou contexte manquants")
            return False
        return True
        
    def get_validation_reason(self, signal: Dict[str, Any], is_valid: bool) -> str:
        """
        Retourne une raison pour la validation/invalidation.
        
        Args:
            signal: Le signal évalué
            is_valid: Résultat de la validation
            
        Returns:
            Raison de la décision
        """
        return f"{self.name}: {'Validé' if is_valid else 'Rejeté'}"
    
    def _convert_strength_to_score(self, strength_str: str) -> float:
        """Convertit une valeur string de strength en score numérique."""
        if strength_str is None:
            return 0.5
        
        # Si c'est déjà un float (déjà converti par FieldConverter)
        if isinstance(strength_str, (int, float)):
            return float(strength_str)
        
        # Si c'est une string, faire la conversion
        if isinstance(strength_str, str):
            strength_map = {
                'WEAK': 0.2,
                'MODERATE': 0.5, 
                'STRONG': 0.8,
                'VERY_STRONG': 0.9,
                'EXTREME': 1.0
            }
            return strength_map.get(strength_str.upper(), 0.5)
        
        return 0.5
    
    def _convert_regime_strength_to_score(self, regime_strength_str: str) -> float:
        """Convertit regime_strength (WEAK/MODERATE/STRONG/EXTREME) en score."""
        return self._convert_strength_to_score(regime_strength_str)
    
    def _convert_trend_strength_to_score(self, trend_strength_str: str) -> float:
        """Convertit trend_strength (WEAK/MODERATE/STRONG/VERY_STRONG) en score."""
        return self._convert_strength_to_score(trend_strength_str)
    
    def _convert_signal_strength_to_score(self, signal_strength_str: str) -> float:
        """Convertit signal_strength (WEAK/MODERATE/STRONG/VERY_STRONG) en score."""
        return self._convert_strength_to_score(signal_strength_str)
    
    def _safe_float(self, value, default: float = 0.0) -> float:
        """Convertit en float de manière sécurisée."""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _safe_format(self, value, format_spec: str = ".2f", default: str = "N/A") -> str:
        """Formate une valeur de manière sécurisée pour éviter les erreurs None.__format__."""
        if value is None:
            return default
        try:
            return f"{value:{format_spec}}"
        except (ValueError, TypeError):
            return str(value) if value is not None else default