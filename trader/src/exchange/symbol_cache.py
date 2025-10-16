# trader/src/exchange/symbol_cache.py
"""
Cache intelligent pour les contraintes de symboles Binance.
Évite les appels API répétés en mettant en cache les informations pendant 10 minutes.
"""
import logging
import time
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


class SymbolConstraintsCache:
    """
    Cache thread-safe pour les contraintes de symboles avec TTL de 10 minutes.
    """

    def __init__(self, ttl_seconds: int = 600):  # 10 minutes par défaut
        """
        Initialise le cache avec une durée de vie configurable.

        Args:
            ttl_seconds: Durée de vie des entrées en cache (défaut: 600s = 10min)
        """
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, dict[str, Any]] = {}
        self._timestamps: dict[str, float] = {}
        self._lock = Lock()

        logger.info(f"✅ Cache contraintes symboles initialisé (TTL: {ttl_seconds}s)")

    def get(self, symbol: str) -> dict[str, Any] | None:
        """
        Récupère les contraintes d'un symbole depuis le cache.

        Args:
            symbol: Symbole (ex: 'BTCUSDC')

        Returns:
            Contraintes si en cache et valides, None sinon
        """
        with self._lock:
            current_time = time.time()

            # Vérifier si le symbole est en cache
            if symbol not in self._cache:
                logger.debug(f"❌ Cache miss pour {symbol} (non présent)")
                return None

            # Vérifier si l'entrée n'a pas expiré
            cached_time = self._timestamps.get(symbol, 0)
            if current_time - cached_time > self.ttl_seconds:
                logger.debug(
                    f"⏰ Cache expiré pour {symbol} (âge: {current_time - cached_time:.1f}s)"
                )
                # Nettoyer l'entrée expirée
                del self._cache[symbol]
                del self._timestamps[symbol]
                return None

            logger.debug(
                f"✅ Cache hit pour {symbol} (âge: {current_time - cached_time:.1f}s)"
            )
            return self._cache[symbol].copy()

    def set(self, symbol: str, constraints: dict[str, Any]) -> None:
        """
        Met en cache les contraintes d'un symbole.

        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            constraints: Contraintes à mettre en cache
        """
        with self._lock:
            self._cache[symbol] = constraints.copy()
            self._timestamps[symbol] = time.time()

            logger.debug(f"💾 Contraintes mises en cache pour {symbol}: {constraints}")

    def invalidate(self, symbol: str) -> bool:
        """
        Invalide l'entrée cache pour un symbole.

        Args:
            symbol: Symbole à invalider

        Returns:
            True si l'entrée existait, False sinon
        """
        with self._lock:
            had_entry = symbol in self._cache

            if had_entry:
                del self._cache[symbol]
                del self._timestamps[symbol]
                logger.debug(f"🗑️ Cache invalidé pour {symbol}")

            return had_entry

    def clear(self) -> None:
        """Vide complètement le cache."""
        with self._lock:
            cleared_count = len(self._cache)
            self._cache.clear()
            self._timestamps.clear()

            logger.info(f"🗑️ Cache vidé ({cleared_count} entrées supprimées)")

    def cleanup_expired(self) -> int:
        """
        Nettoie les entrées expirées du cache.

        Returns:
            Nombre d'entrées supprimées
        """
        with self._lock:
            current_time = time.time()
            expired_symbols = []

            for symbol, cached_time in self._timestamps.items():
                if current_time - cached_time > self.ttl_seconds:
                    expired_symbols.append(symbol)

            for symbol in expired_symbols:
                del self._cache[symbol]
                del self._timestamps[symbol]

            if expired_symbols:
                logger.debug(
                    f"🧹 Nettoyage cache: {len(expired_symbols)} entrées expirées supprimées"
                )

            return len(expired_symbols)

    def get_stats(self) -> dict[str, Any]:
        """
        Retourne les statistiques du cache.

        Returns:
            Dictionnaire avec les statistiques
        """
        with self._lock:
            current_time = time.time()

            stats: dict[str, Any] = {
                "total_entries": len(self._cache),
                "ttl_seconds": self.ttl_seconds,
                "symbols": list(self._cache.keys()),
                "entries_by_age": {},
            }

            # Calculer l'âge de chaque entrée
            for symbol, cached_time in self._timestamps.items():
                age_seconds = current_time - cached_time
                age_minutes = age_seconds / 60
                stats["entries_by_age"][symbol] = {
                    "age_seconds": age_seconds,
                    "age_minutes": age_minutes,
                    "expires_in_seconds": max(0, self.ttl_seconds - age_seconds),
                }

            return stats

    def __len__(self) -> int:
        """Retourne le nombre d'entrées en cache."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, symbol: str) -> bool:
        """Vérifie si un symbole est en cache (sans considérer l'expiration)."""
        with self._lock:
            return symbol in self._cache
