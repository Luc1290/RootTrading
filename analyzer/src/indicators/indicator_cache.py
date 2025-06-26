"""
Cache LRU pour les indicateurs techniques avec optimisation numpy
"""
import numpy as np
import pandas as pd
from functools import lru_cache
from typing import Union, Tuple, Optional
import hashlib
import time
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class IndicatorCache:
    """Cache optimisé pour les indicateurs techniques"""
    
    def __init__(self, ttl_seconds: int = 120, max_size: int = 128):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.stats = CacheStats()
        self._cache = {}
        self._timestamps = {}
        
    def _get_cache_key(self, indicator: str, data: Union[np.ndarray, pd.Series], **params) -> str:
        """Génère une clé de cache unique basée sur les données et paramètres"""
        # Convertir en numpy si nécessaire
        if isinstance(data, pd.Series):
            data_array = data.values
        else:
            data_array = data
            
        # Hash des données (utilise seulement les 10 dernières valeurs pour la clé)
        data_hash = hashlib.md5(data_array[-10:].tobytes()).hexdigest()[:8]
        
        # Hash des paramètres
        params_str = "_".join(f"{k}{v}" for k, v in sorted(params.items()))
        
        return f"{indicator}_{len(data_array)}_{data_hash}_{params_str}"
    
    def get(self, indicator: str, data: Union[np.ndarray, pd.Series], 
            compute_func, **params) -> np.ndarray:
        """
        Récupère un indicateur du cache ou le calcule si nécessaire
        
        Args:
            indicator: Nom de l'indicateur (ex: 'RSI', 'ATR')
            data: Données d'entrée
            compute_func: Fonction de calcul si cache miss
            **params: Paramètres de l'indicateur
        """
        cache_key = self._get_cache_key(indicator, data, **params)
        
        # Vérifier si en cache et non expiré
        if cache_key in self._cache:
            timestamp = self._timestamps.get(cache_key, 0)
            if time.time() - timestamp < self.ttl_seconds:
                self.stats.hits += 1
                logger.debug(f"Cache HIT pour {indicator} (hit_rate: {self.stats.hit_rate:.2%})")
                return self._cache[cache_key].copy()
            else:
                # Expiré, supprimer
                del self._cache[cache_key]
                del self._timestamps[cache_key]
        
        # Cache miss - calculer
        self.stats.misses += 1
        logger.debug(f"Cache MISS pour {indicator} - calcul en cours...")
        
        result = compute_func(data, **params)
        
        # Stocker en cache
        if len(self._cache) >= self.max_size:
            # LRU eviction - supprimer le plus ancien
            oldest_key = min(self._timestamps, key=self._timestamps.get)
            del self._cache[oldest_key]
            del self._timestamps[oldest_key]
        
        self._cache[cache_key] = result.copy() if isinstance(result, np.ndarray) else result
        self._timestamps[cache_key] = time.time()
        
        return result
    
    def clear(self):
        """Vide le cache"""
        self._cache.clear()
        self._timestamps.clear()
        logger.info(f"Cache vidé. Stats finales: {self.stats.hits} hits, {self.stats.misses} misses")
        
    def get_stats(self) -> dict:
        """Retourne les statistiques du cache"""
        return {
            'hits': self.stats.hits,
            'misses': self.stats.misses,
            'hit_rate': self.stats.hit_rate,
            'size': len(self._cache),
            'max_size': self.max_size
        }

# Instance globale du cache
indicator_cache = IndicatorCache(ttl_seconds=120, max_size=256)