"""
Volume Context Detector - Détection intelligente du contexte market pour adaptation des seuils volume.

Ce module analyse les conditions du marché (RSI, CCI, ADX, volatilité) pour déterminer 
le contexte approprié et ajuster automatiquement les seuils de validation volume.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
from .config import VOLUME_CONTEXTS, VOLUME_BUILDUP_CONFIG, ADX_WEAK_TREND_THRESHOLD


@dataclass
class VolumeContext:
    """Représente un contexte volume avec ses paramètres."""
    name: str
    min_ratio: float
    ideal_ratio: float
    description: str
    confidence: float = 1.0  # Confiance dans la détection du contexte


class VolumeContextDetector:
    """Détecteur de contexte market pour adaptation des seuils volume."""
    
    def __init__(self):
        self.contexts = VOLUME_CONTEXTS
        self.buildup_config = VOLUME_BUILDUP_CONFIG
    
    def detect_market_context(
        self,
        rsi: Optional[float] = None,
        cci: Optional[float] = None,
        adx: Optional[float] = None,
        signal_type: str = "BUY",
        volume_history: Optional[List[float]] = None,
        price_trend: Optional[str] = None
    ) -> VolumeContext:
        """
        Détecte le contexte market approprié basé sur les indicateurs techniques.
        
        Args:
            rsi: RSI actuel (0-100)
            cci: CCI actuel
            adx: ADX actuel (0-100)
            signal_type: Type de signal ("BUY" ou "SELL")
            volume_history: Historique des volumes (optionnel)
            price_trend: Tendance des prix ("bullish", "bearish", "neutral")
            
        Returns:
            VolumeContext: Contexte détecté avec seuils adaptés
        """
        detected_contexts = []
        
        # 1. Détection oversold conditions
        if rsi is not None and signal_type == "BUY":
            if rsi < 30 and cci is not None and cci < -200:
                detected_contexts.append(("deep_oversold", 0.9))
            elif rsi < 40 and cci is not None and cci < -150:
                detected_contexts.append(("moderate_oversold", 0.8))
            elif rsi < 40:
                detected_contexts.append(("oversold_bounce", 0.7))
        
        # 2. Détection volatilité basée sur ADX
        if adx is not None:
            if adx < 20:
                detected_contexts.append(("low_volatility", 0.6))
            elif adx > 35:
                detected_contexts.append(("high_volatility", 0.5))
        
        # 3. Détection patterns de volume
        if volume_history is not None and len(volume_history) >= 3:
            if self._detect_volume_buildup(volume_history):
                detected_contexts.append(("consolidation_break", 0.7))
            elif self._detect_volume_spike(volume_history):
                detected_contexts.append(("breakout", 0.8))
        
        # 4. Détection tendance et breakouts
        if price_trend == "breakout":
            # Contexte spécifique breakout avec priorité absolue
            detected_contexts.append(("breakout", 0.9))
        elif price_trend == "bullish" and adx is not None and adx > 25:
            detected_contexts.append(("trend_continuation", 0.6))
        elif price_trend == "pump_start":
            detected_contexts.append(("pump_start", 0.8))
        
        # 5. Sélection du contexte avec la plus haute confiance
        if detected_contexts:
            # Trier par confiance décroissante
            detected_contexts.sort(key=lambda x: x[1], reverse=True)
            best_context_name = detected_contexts[0][0]
            confidence = detected_contexts[0][1]
        else:
            # Contexte par défaut
            best_context_name = "trend_continuation"
            confidence = 0.5
        
        # Récupérer la configuration du contexte
        context_config = self.contexts[best_context_name]
        
        return VolumeContext(
            name=best_context_name,
            min_ratio=context_config["min_ratio"],
            ideal_ratio=context_config["ideal_ratio"],
            description=context_config["description"],
            confidence=confidence
        )
    
    def _detect_volume_buildup(self, volume_history: List[float]) -> bool:
        """Détecte si il y a une accumulation progressive de volume."""
        if len(volume_history) < self.buildup_config["lookback_periods"]:
            return False
        
        recent_volumes = volume_history[-self.buildup_config["lookback_periods"]:]
        increases = 0
        
        for i in range(1, len(recent_volumes)):
            if recent_volumes[i] > recent_volumes[i-1] * self.buildup_config["min_increase_ratio"]:
                increases += 1
        
        increase_ratio = increases / (len(recent_volumes) - 1)
        return increase_ratio >= self.buildup_config["progressive_threshold"]
    
    def _detect_volume_spike(self, volume_history: List[float]) -> bool:
        """Détecte un spike de volume récent."""
        if len(volume_history) < 2:
            return False
        
        current_volume = volume_history[-1]
        avg_volume = sum(volume_history[:-1]) / len(volume_history[:-1])
        
        return current_volume > avg_volume * self.buildup_config["spike_detection_ratio"]
    
    def get_contextual_volume_threshold(
        self,
        base_volume_ratio: float,
        rsi: Optional[float] = None,
        cci: Optional[float] = None,
        adx: Optional[float] = None,
        signal_type: str = "BUY",
        volume_history: Optional[List[float]] = None,
        price_trend: Optional[str] = None
    ) -> Tuple[float, str, float]:
        """
        Calcule le seuil volume contextuel adapté.
        
        Args:
            base_volume_ratio: Ratio de volume de base
            rsi, cci, adx: Indicateurs techniques
            signal_type: Type de signal
            volume_history: Historique des volumes
            price_trend: Tendance des prix
            
        Returns:
            Tuple[float, str, float]: (seuil_ajusté, contexte_détecté, score_contextuel)
        """
        context = self.detect_market_context(
            rsi=rsi,
            cci=cci,
            adx=adx,
            signal_type=signal_type,
            volume_history=volume_history,
            price_trend=price_trend
        )
        
        # Calculer le score contextuel basé sur la distance par rapport au seuil idéal
        if base_volume_ratio >= context.ideal_ratio:
            contextual_score = 1.0  # Score parfait
        elif base_volume_ratio >= context.min_ratio:
            # Score graduel entre min et ideal
            ratio_range = context.ideal_ratio - context.min_ratio
            actual_range = base_volume_ratio - context.min_ratio
            contextual_score = 0.5 + (actual_range / ratio_range) * 0.5
        else:
            # En dessous du minimum, score proportionnel
            contextual_score = (base_volume_ratio / context.min_ratio) * 0.5
        
        # Ajuster le score par la confiance dans la détection
        final_score = contextual_score * context.confidence
        
        return context.min_ratio, context.name, final_score
    
    def get_volume_quality_description(self, volume_ratio: float, context_name: str) -> str:
        """Retourne une description qualitative du volume selon le contexte."""
        context_config = self.contexts[context_name]
        
        if volume_ratio >= context_config["ideal_ratio"]:
            return "Excellent"
        elif volume_ratio >= context_config["min_ratio"] * 1.2:
            return "Très bon"
        elif volume_ratio >= context_config["min_ratio"]:
            return "Acceptable"
        else:
            return "Insuffisant"
    
    def should_apply_volume_tolerance(
        self,
        rsi: Optional[float] = None,
        cci: Optional[float] = None,
        adx: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Détermine si une tolérance volume doit être appliquée.
        
        Returns:
            Tuple[bool, float]: (appliquer_tolérance, facteur_réduction)
        """
        # Conditions pour tolérance volume
        tolerance_conditions = []
        
        # Oversold extrême
        if rsi is not None and rsi < 30:
            tolerance_conditions.append(0.7)  # Réduction de 30%
        
        # CCI oversold sévère
        if cci is not None and cci < -200:
            tolerance_conditions.append(0.6)  # Réduction de 40%
        
        # Marché calme (ADX faible)
        if adx is not None and adx < ADX_WEAK_TREND_THRESHOLD:
            tolerance_conditions.append(0.8)  # Réduction de 20%
        
        if tolerance_conditions:
            # Utiliser la tolérance la plus forte (facteur le plus bas)
            max_tolerance = min(tolerance_conditions)
            return True, max_tolerance
        
        return False, 1.0


# Instance globale pour utilisation dans les modules
volume_context_detector = VolumeContextDetector()