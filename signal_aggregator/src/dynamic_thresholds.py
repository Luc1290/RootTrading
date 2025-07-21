"""
Système de seuils dynamiques basés sur les percentiles historiques
"""
import numpy as np
from typing import Dict, Optional, Deque, Any
from collections import deque
from datetime import datetime
import json
import logging
from .shared.redis_utils import RedisManager

logger = logging.getLogger(__name__)

class DynamicThresholdManager:
    """
    Gère les seuils de confiance dynamiques basés sur l'historique des signaux
    """
    
    def __init__(self, 
                 redis_client=None,
                 history_size: int = 500,
                 min_threshold: float = 0.35,
                 max_threshold: float = 0.85,
                 target_signal_rate: float = 0.02):  # 2% des signaux devraient passer - approche sniper
        """
        Args:
            redis_client: Client Redis pour persistance
            history_size: Nombre de signaux historiques à conserver
            min_threshold: Seuil minimum absolu
            max_threshold: Seuil maximum absolu
            target_signal_rate: Taux cible de signaux qui passent le filtre
        """
        self.redis = redis_client
        self.history_size = history_size
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.target_signal_rate = target_signal_rate
        
        # Historique des confiances de signaux
        self.confidence_history: Deque[float] = deque(maxlen=history_size)
        self.signal_timestamps: Deque[datetime] = deque(maxlen=history_size)
        
        # Seuils actuels - plus sélectifs pour approche sniper
        self.current_confidence_threshold = 0.80
        self.current_vote_threshold = 0.55
        
        # Statistiques
        self.signals_evaluated = 0
        self.signals_passed = 0
        
        # Charger l'historique si disponible
        if self.redis:
            self._load_history()
    
    def _load_history(self):
        """Charge l'historique depuis Redis"""
        try:
            # Utiliser l'utilitaire partagé pour récupération avec désérialisation automatique
            history_data = RedisManager.get_cached_data(self.redis, "dynamic_threshold_history")
            if history_data:
                self.confidence_history = deque(
                    history_data.get('confidence_history', []), 
                    maxlen=self.history_size
                )
                self.current_confidence_threshold = history_data.get(
                    'current_confidence_threshold', 0.55
                )
                self.current_vote_threshold = history_data.get(
                    'current_vote_threshold', 0.35
                )
                logger.info(f"📊 Chargé l'historique des seuils: {len(self.confidence_history)} signaux")
        except Exception as e:
            logger.error(f"Erreur chargement historique seuils: {e}")
    
    def _save_history(self):
        """Sauvegarde l'historique dans Redis"""
        if not self.redis:
            return
            
        try:
            data = {
                'confidence_history': list(self.confidence_history),
                'current_confidence_threshold': self.current_confidence_threshold,
                'current_vote_threshold': self.current_vote_threshold,
                'last_update': datetime.now().isoformat()
            }
            # Utiliser l'utilitaire partagé pour mise en cache avec sérialisation automatique
            RedisManager.set_cached_data(self.redis, "dynamic_threshold_history", data)
        except Exception as e:
            logger.error(f"Erreur sauvegarde historique seuils: {e}")
    
    def add_signal(self, confidence: float, timestamp: Optional[datetime] = None):
        """
        Ajoute un signal à l'historique
        
        Args:
            confidence: Niveau de confiance du signal
            timestamp: Timestamp du signal (par défaut: maintenant)
        """
        self.confidence_history.append(confidence)
        self.signal_timestamps.append(timestamp or datetime.now())
        self.signals_evaluated += 1
        
        # Mettre à jour les seuils tous les 50 signaux
        if self.signals_evaluated % 50 == 0:
            self.update_thresholds()
    
    def update_thresholds(self):
        """
        Met à jour les seuils basés sur les percentiles de l'historique
        """
        if len(self.confidence_history) < 50:
            return  # Pas assez de données
        
        # Calculer les percentiles
        confidence_array = np.array(self.confidence_history)
        
        # Calculer le percentile qui donnerait le taux de signal cible
        target_percentile = 100 * (1 - self.target_signal_rate)
        new_confidence_threshold = np.percentile(confidence_array, target_percentile)
        
        # Appliquer les limites min/max
        new_confidence_threshold = np.clip(
            new_confidence_threshold, 
            self.min_threshold, 
            self.max_threshold
        )
        
        # Calculer le taux de signal actuel
        recent_window = list(self.confidence_history)[-100:]
        current_pass_rate = sum(1 for c in recent_window if c >= self.current_confidence_threshold) / len(recent_window)
        
        # Ajuster plus agressivement si on est loin du taux cible
        if current_pass_rate < self.target_signal_rate * 0.5:
            # Trop peu de signaux passent, baisser le seuil plus rapidement
            adjustment_factor = 0.95
        elif current_pass_rate > self.target_signal_rate * 2:
            # Trop de signaux passent, augmenter le seuil plus rapidement
            adjustment_factor = 1.05
        else:
            adjustment_factor = 1.0
        
        # Appliquer l'ajustement avec lissage
        self.current_confidence_threshold = (
            0.7 * self.current_confidence_threshold + 
            0.3 * new_confidence_threshold * adjustment_factor
        )
        
        # Ajuster aussi le seuil de vote proportionnellement
        self.current_vote_threshold = self.min_threshold * (
            self.current_confidence_threshold / self.max_threshold
        )
        
        logger.info(
            f"📈 Seuils mis à jour - Confiance: {self.current_confidence_threshold:.3f}, "
            f"Vote: {self.current_vote_threshold:.3f}, "
            f"Taux actuel: {current_pass_rate:.1%}, Cible: {self.target_signal_rate:.1%}"
        )
        
        # Sauvegarder
        self._save_history()
    
    def should_accept_signal(self, confidence: float, vote_weight: float) -> bool:
        """
        Détermine si un signal doit être accepté basé sur les seuils dynamiques
        
        Args:
            confidence: Niveau de confiance du signal
            vote_weight: Poids du vote agrégé
            
        Returns:
            True si le signal passe les seuils
        """
        # Ajouter à l'historique
        self.add_signal(confidence)
        
        # Vérifier les seuils
        passes = (
            confidence >= self.current_confidence_threshold and
            vote_weight >= self.current_vote_threshold
        )
        
        if passes:
            self.signals_passed += 1
        
        return passes
    
    def get_current_thresholds(self) -> Dict[str, float]:
        """Retourne les seuils actuels"""
        return {
            'confidence_threshold': self.current_confidence_threshold,
            'vote_threshold': self.current_vote_threshold,
            'min_threshold': self.min_threshold,
            'max_threshold': self.max_threshold,
            'target_rate': self.target_signal_rate
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques des seuils dynamiques"""
        if len(self.confidence_history) == 0:
            return {
                'history_size': 0,
                'current_thresholds': self.get_current_thresholds()
            }
        
        confidence_array = np.array(self.confidence_history)
        recent_100 = list(self.confidence_history)[-100:] if len(self.confidence_history) >= 100 else list(self.confidence_history)
        
        return {
            'history_size': len(self.confidence_history),
            'confidence_stats': {
                'mean': float(np.mean(confidence_array)),
                'std': float(np.std(confidence_array)),
                'min': float(np.min(confidence_array)),
                'max': float(np.max(confidence_array)),
                'p25': float(np.percentile(confidence_array, 25)),
                'p50': float(np.percentile(confidence_array, 50)),
                'p75': float(np.percentile(confidence_array, 75)),
                'p90': float(np.percentile(confidence_array, 90))
            },
            'current_thresholds': self.get_current_thresholds(),
            'pass_rate': {
                'overall': self.signals_passed / self.signals_evaluated if self.signals_evaluated > 0 else 0,
                'recent_100': sum(1 for c in recent_100 if c >= self.current_confidence_threshold) / len(recent_100) if recent_100 else 0
            },
            'signals_evaluated': self.signals_evaluated,
            'signals_passed': self.signals_passed
        }
    
    def adjust_for_market_conditions(self, volatility: float, volume_ratio: float):
        """
        Ajuste les seuils en fonction des conditions de marché
        
        Args:
            volatility: Volatilité actuelle (0-1)
            volume_ratio: Ratio du volume actuel vs moyenne (ex: 1.5 = 150% du volume normal)
        """
        # En haute volatilité, être plus sélectif
        volatility_adjustment = 1 + (volatility - 0.5) * 0.2  # ±10% max
        
        # Avec un volume élevé, être moins sélectif
        volume_adjustment = 1 - (min(2, volume_ratio) - 1) * 0.1  # -10% max si volume 2x
        
        # Appliquer les ajustements
        adjusted_confidence = self.current_confidence_threshold * volatility_adjustment * volume_adjustment
        self.current_confidence_threshold = np.clip(
            adjusted_confidence,
            self.min_threshold,
            self.max_threshold
        )
        
        logger.debug(
            f"🎯 Seuils ajustés pour conditions de marché - "
            f"Volatilité: {volatility:.2f}, Volume ratio: {volume_ratio:.2f}, "
            f"Nouveau seuil: {self.current_confidence_threshold:.3f}"
        )