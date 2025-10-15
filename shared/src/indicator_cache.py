"""
Indicator Cache System

High-performance Redis-based caching for technical indicators with:
- Incremental calculation support
- State persistence across service restarts
- Thread-safe operations
- Auto-save functionality
- TTL management
"""

import redis
import pickle
import json
import threading
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Métriques du cache."""

    hit_count: int = 0
    miss_count: int = 0
    save_count: int = 0
    restore_count: int = 0
    error_count: int = 0
    last_save: Optional[float] = None

    @property
    def hit_ratio(self) -> float:
        """Ratio de hits."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

    def to_dict(self) -> Dict:
        """Convertit en dictionnaire."""
        return asdict(self)


class IndicatorCache:
    """
    Cache Redis pour les indicateurs techniques.

    Optimise les performances en:
    - Sauvegardant les états des indicateurs
    - Permettant les calculs incrémentaux
    - Persistant les données entre redémarrages
    - Gérant automatiquement les TTL
    """

    def __init__(
        self,
        redis_host: Optional[str] = None,
        redis_port: Optional[int] = None,
        redis_db: int = 2,  # DB séparée pour les indicateurs
        redis_password: Optional[str] = None,
        ttl_hours: int = 48,
        auto_save_interval: int = 300,
    ):  # 5 minutes
        """
        Args:
            redis_host: Host Redis
            redis_port: Port Redis
            redis_db: Database Redis
            redis_password: Mot de passe Redis
            ttl_hours: TTL en heures (48h par défaut)
            auto_save_interval: Intervalle auto-save en secondes
        """
        # Importer la configuration par défaut si non fournie
        from .config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD

        self.redis_host = redis_host or REDIS_HOST
        self.redis_port = redis_port or REDIS_PORT
        self.redis_db = redis_db
        self.redis_password = redis_password or REDIS_PASSWORD
        self.ttl_seconds = ttl_hours * 3600
        self.auto_save_interval = auto_save_interval

        # Connexion Redis
        self.redis_client: Optional[redis.Redis] = None
        self._connect_redis()

        # Cache en mémoire
        self._memory_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Métriques
        self.metrics = CacheMetrics()

        # Auto-save
        self._auto_save_thread: Optional[threading.Thread] = None
        self._stop_auto_save = threading.Event()
        self._start_auto_save()

        logger.info(
            f"IndicatorCache initialisé - TTL: {ttl_hours}h, Auto-save: {auto_save_interval}s"
        )

    def _connect_redis(self):
        """Établit la connexion Redis."""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                password=self.redis_password,
                decode_responses=False,  # Pour pickle
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )

            # Test de connexion
            self.redis_client.ping()
            logger.info("Connexion Redis établie")

        except Exception as e:
            logger.error(f"Erreur connexion Redis: {e}")
            self.redis_client = None

    def _start_auto_save(self):
        """Démarre l'auto-save en arrière-plan."""
        if self.auto_save_interval > 0:
            self._auto_save_thread = threading.Thread(
                target=self._auto_save_worker,
                daemon=True,
                name="IndicatorCache-AutoSave",
            )
            self._auto_save_thread.start()
            logger.debug("Auto-save thread démarré")

    def _auto_save_worker(self):
        """Worker d'auto-save."""
        while not self._stop_auto_save.wait(self.auto_save_interval):
            try:
                self.save_all_to_redis()
            except Exception as e:
                logger.error(f"Erreur auto-save: {e}")
                self.metrics.error_count += 1

    def get(self, key: str, symbol: Optional[str] = None) -> Optional[Any]:
        """
        Récupère une valeur du cache.

        Args:
            key: Clé de cache
            symbol: Symbole (optionnel)

        Returns:
            Valeur mise en cache ou None
        """
        full_key = self._make_full_key(key, symbol)

        with self._lock:
            # Vérifier cache mémoire d'abord
            if full_key in self._memory_cache:
                self.metrics.hit_count += 1
                return self._memory_cache[full_key]

            # Vérifier Redis si disponible
            if self.redis_client:
                try:
                    redis_value = self.redis_client.get(full_key)
                    if redis_value and isinstance(redis_value, (bytes, str)):
                        # Convertir en bytes si nécessaire pour pickle.loads
                        redis_bytes = (
                            redis_value
                            if isinstance(redis_value, bytes)
                            else redis_value.encode("utf-8")
                        )
                        value = pickle.loads(redis_bytes)
                        # Mettre en cache mémoire
                        self._memory_cache[full_key] = value
                        self._cache_timestamps[full_key] = time.time()
                        self.metrics.hit_count += 1
                        return value

                except Exception as e:
                    logger.warning(f"Erreur lecture Redis {full_key}: {e}")
                    self.metrics.error_count += 1

            self.metrics.miss_count += 1
            return None

    def set(self, key: str, value: Any, symbol: Optional[str] = None):
        """
        Stocke une valeur dans le cache.

        Args:
            key: Clé de cache
            value: Valeur à stocker
            symbol: Symbole (optionnel)
        """
        full_key = self._make_full_key(key, symbol)

        with self._lock:
            # Cache mémoire
            self._memory_cache[full_key] = value
            self._cache_timestamps[full_key] = time.time()

            # Cache Redis (asynchrone pour performance)
            if self.redis_client:
                threading.Thread(
                    target=self._save_to_redis, args=(full_key, value), daemon=True
                ).start()

    def _save_to_redis(self, full_key: str, value: Any):
        """Sauvegarde en Redis (méthode privée pour threading)."""
        try:
            serialized = pickle.dumps(value)
            if self.redis_client is not None:
                self.redis_client.setex(full_key, self.ttl_seconds, serialized)
        except Exception as e:
            logger.warning(f"Erreur écriture Redis {full_key}: {e}")
            self.metrics.error_count += 1

    def delete(self, key: str, symbol: Optional[str] = None):
        """
        Supprime une entrée du cache.

        Args:
            key: Clé de cache
            symbol: Symbole (optionnel)
        """
        full_key = self._make_full_key(key, symbol)

        with self._lock:
            # Supprimer du cache mémoire
            self._memory_cache.pop(full_key, None)
            self._cache_timestamps.pop(full_key, None)

            # Supprimer de Redis
            if self.redis_client:
                try:
                    self.redis_client.delete(full_key)
                except Exception as e:
                    logger.warning(f"Erreur suppression Redis {full_key}: {e}")
                    self.metrics.error_count += 1

    def clear_symbol(self, symbol: str, force: bool = False):
        """
        Efface tous les indicateurs d'un symbole.

        Args:
            symbol: Symbole à effacer
            force: Forcer même si erreurs Redis
        """
        pattern = f"indicators:{symbol}:*"

        with self._lock:
            try:
                # Effacer cache mémoire
                keys_to_remove = [
                    k
                    for k in self._memory_cache.keys()
                    if k.startswith(f"indicators:{symbol}:")
                ]

                for key in keys_to_remove:
                    self._memory_cache.pop(key, None)
                    self._cache_timestamps.pop(key, None)

                # Effacer Redis
                if self.redis_client:
                    redis_keys = self.redis_client.keys(pattern)
                    if redis_keys and isinstance(redis_keys, list):
                        self.redis_client.delete(*redis_keys)

                logger.info(
                    f"Cache effacé pour {symbol}: {len(keys_to_remove)} entrées"
                )

            except Exception as e:
                logger.error(f"Erreur effacement cache {symbol}: {e}")
                self.metrics.error_count += 1
                if not force:
                    raise

    def clear_all(self):
        """Efface tout le cache."""
        with self._lock:
            # Cache mémoire
            self._memory_cache.clear()
            self._cache_timestamps.clear()

            # Redis
            if self.redis_client:
                try:
                    # Effacer seulement les clés indicators
                    keys = self.redis_client.keys("indicators:*")
                    if keys:
                        self.redis_client.delete(*keys)
                    logger.info(f"Cache Redis effacé: {len(keys)} entrées")
                except Exception as e:
                    logger.error(f"Erreur effacement Redis: {e}")
                    self.metrics.error_count += 1

    def save_all_to_redis(self):
        """Sauvegarde tout le cache mémoire vers Redis."""
        if not self.redis_client:
            return

        with self._lock:
            saved_count = 0

            try:
                # Utiliser pipeline pour efficiency
                pipe = self.redis_client.pipeline()

                for full_key, value in self._memory_cache.items():
                    try:
                        serialized = pickle.dumps(value)
                        pipe.setex(full_key, self.ttl_seconds, serialized)
                        saved_count += 1
                    except Exception as e:
                        logger.warning(f"Erreur sérialisation {full_key}: {e}")
                        continue

                # Exécuter le pipeline
                pipe.execute()

                self.metrics.save_count += saved_count
                self.metrics.last_save = time.time()

                if saved_count > 0:
                    logger.debug(f"Auto-save: {saved_count} indicateurs sauvegardés")

            except Exception as e:
                logger.error(f"Erreur pipeline Redis: {e}")
                self.metrics.error_count += 1

    def restore_from_redis(self, symbol: Optional[str] = None) -> int:
        """
        Restaure le cache depuis Redis.

        Args:
            symbol: Symbole spécifique ou None pour tout

        Returns:
            Nombre d'entrées restaurées
        """
        if not self.redis_client:
            return 0

        pattern = f"indicators:{symbol}:*" if symbol else "indicators:*"
        restored_count = 0

        with self._lock:
            try:
                keys = self.redis_client.keys(pattern)

                if keys and isinstance(keys, list):
                    # Utiliser pipeline pour efficiency
                    pipe = self.redis_client.pipeline()
                    for key in keys:
                        pipe.get(key)

                    values = pipe.execute()

                    if isinstance(values, list):
                        for key, value in zip(keys, values):
                            if value:
                                try:
                                    # Décoder key bytes vers string
                                    if isinstance(key, bytes):
                                        key = key.decode("utf-8")

                                    deserialized = pickle.loads(value)
                                    self._memory_cache[key] = deserialized
                                    self._cache_timestamps[key] = time.time()
                                    restored_count += 1

                                except Exception as e:
                                    logger.warning(f"Erreur désérialisation {key}: {e}")
                                    continue

                self.metrics.restore_count += restored_count

                if restored_count > 0:
                    logger.info(f"Restauré {restored_count} indicateurs depuis Redis")

            except Exception as e:
                logger.error(f"Erreur restauration Redis: {e}")
                self.metrics.error_count += 1

        return restored_count

    def get_memory_usage(self) -> Dict[str, int]:
        """Retourne l'usage mémoire du cache."""
        with self._lock:
            total_entries = len(self._memory_cache)

            # Estimation taille (approximative)
            total_size = 0
            for value in self._memory_cache.values():
                try:
                    total_size += len(pickle.dumps(value))
                except:
                    total_size += 1024  # Estimation par défaut

            return {
                "entries": total_entries,
                "estimated_bytes": total_size,
                "estimated_mb": int(total_size / (1024 * 1024)),
            }

    def cleanup_expired(self, max_age_seconds: int = 3600):
        """
        Nettoie les entrées expirées du cache mémoire.

        Args:
            max_age_seconds: Âge maximum en secondes (1h par défaut)
        """
        current_time = time.time()
        expired_keys = []

        with self._lock:
            for key, timestamp in self._cache_timestamps.items():
                if current_time - timestamp > max_age_seconds:
                    expired_keys.append(key)

            for key in expired_keys:
                self._memory_cache.pop(key, None)
                self._cache_timestamps.pop(key, None)

        if expired_keys:
            logger.debug(f"Nettoyage cache: {len(expired_keys)} entrées expirées")

    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        memory_usage = self.get_memory_usage()

        return {
            "metrics": self.metrics.to_dict(),
            "memory_usage": memory_usage,
            "redis_connected": self.redis_client is not None,
            "auto_save_active": not self._stop_auto_save.is_set(),
            "cache_entries": len(self._memory_cache),
            "ttl_hours": self.ttl_seconds / 3600,
        }

    def _make_full_key(self, key: str, symbol: Optional[str] = None) -> str:
        """Génère une clé complète."""
        if symbol:
            return f"indicators:{symbol}:{key}"
        else:
            return f"indicators:{key}"

    def shutdown(self):
        """Arrêt propre du cache."""
        logger.info("Arrêt IndicatorCache...")

        # Arrêter auto-save
        self._stop_auto_save.set()
        if self._auto_save_thread and self._auto_save_thread.is_alive():
            self._auto_save_thread.join(timeout=5)

        # Dernière sauvegarde
        try:
            self.save_all_to_redis()
        except Exception as e:
            logger.error(f"Erreur sauvegarde finale: {e}")

        # Fermer connexion Redis
        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception as e:
                logger.error(f"Erreur fermeture Redis: {e}")

        logger.info("IndicatorCache arrêté")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# Instance globale partagée
_global_cache: Optional[IndicatorCache] = None


def get_indicator_cache() -> IndicatorCache:
    """Retourne l'instance globale du cache."""
    global _global_cache

    if _global_cache is None:
        _global_cache = IndicatorCache()

        # Tentative de restauration au démarrage
        try:
            restored = _global_cache.restore_from_redis()
            if restored > 0:
                logger.info(f"Cache restauré au démarrage: {restored} indicateurs")
        except Exception as e:
            logger.warning(f"Restauration cache échouée: {e}")

    return _global_cache


def shutdown_indicator_cache():
    """Arrêt propre du cache global."""
    global _global_cache

    if _global_cache:
        _global_cache.shutdown()
        _global_cache = None
