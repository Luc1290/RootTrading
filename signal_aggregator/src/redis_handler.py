"""
Module de gestion des communications Redis pour le Signal Aggregator.
Gère la réception des signaux depuis l'analyzer et la publication vers le coordinator.
"""

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisHandler:
    """Gestionnaire des communications Redis pour le signal aggregator."""

    def __init__(self) -> None:
        self.redis_client: redis.Redis | None = None
        self.subscriber_client: redis.Redis | None = None
        self.is_connected = False

        # Canaux Redis
        self.signals_channel = "analyzer:signals"  # Signaux depuis analyzer
        self.filtered_signals_channel = (
            "roottrading:signals:filtered"  # Vers coordinator
        )

        # Callback pour traitement des signaux
        self.signal_callback: Callable[[str],
                                       None] | Callable[[str], Any] | None = None

        # Statistiques
        self.stats = {
            "signals_received": 0,
            "signals_published": 0,
            "connection_errors": 0,
            "publish_errors": 0,
        }

    async def connect(self, redis_url: str = "redis://redis:6379"):
        """
        Établit la connexion Redis.

        Args:
            redis_url: URL de connexion Redis
        """
        try:
            # Client principal pour publication
            self.redis_client = redis.from_url(
                redis_url, decode_responses=True)
            if self.redis_client is not None:
                await self.redis_client.ping()

            # Client séparé pour abonnement (recommandé par redis-py)
            self.subscriber_client = redis.from_url(
                redis_url, decode_responses=True)
            if self.subscriber_client is not None:
                await self.subscriber_client.ping()

            self.is_connected = True
            logger.info(f"Connexion Redis établie: {redis_url}")

        except Exception:
            logger.exception("Erreur connexion Redis")
            self.stats["connection_errors"] += 1
            raise

    async def disconnect(self):
        """Ferme les connexions Redis."""
        if self.redis_client:
            await self.redis_client.aclose()
            self.redis_client = None

        if self.subscriber_client:
            await self.subscriber_client.aclose()
            self.subscriber_client = None

        self.is_connected = False
        logger.info("Connexions Redis fermées")

    async def subscribe_to_signals(
        self, callback: Callable[[str], None] | Callable[[str], Any]
    ):
        """
        S'abonne aux signaux depuis l'analyzer.

        Args:
            callback: Fonction à appeler pour chaque signal reçu
        """
        if not self.is_connected:
            raise RuntimeError("Redis non connecté")

        self.signal_callback = callback

        # Créer un abonnement
        if self.subscriber_client is None:
            raise RuntimeError("Client subscriber non initialisé")
        pubsub = self.subscriber_client.pubsub()
        await pubsub.subscribe(self.signals_channel)

        logger.info(f"Abonné au canal: {self.signals_channel}")

        # Boucle d'écoute des messages
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        self.stats["signals_received"] += 1
                        signal_data = message["data"]

                        logger.debug(
                            f"Signal reçu depuis Redis: {len(signal_data)} chars"
                        )

                        # Appeler le callback de traitement
                        if self.signal_callback and callable(
                                self.signal_callback):
                            if asyncio.iscoroutinefunction(
                                    self.signal_callback):
                                await self.signal_callback(signal_data)
                            else:
                                self.signal_callback(signal_data)

                    except Exception:
                        logger.exception("Erreur traitement message Redis")

        except Exception:
            logger.exception("Erreur dans l'abonnement Redis")
        finally:
            await pubsub.unsubscribe(self.signals_channel)
            await pubsub.aclose()

    async def publish_validated_signal(self, validated_signal: dict[str, Any]):
        """
        Publie un signal validé vers le coordinator.

        Args:
            validated_signal: Signal validé et scoré
        """
        if not self.is_connected:
            raise RuntimeError("Redis non connecté")

        try:
            # Sérialisation du signal
            signal_json = json.dumps(validated_signal, default=str)

            # Publication vers le coordinator
            if self.redis_client is None:
                raise RuntimeError("Client Redis non initialisé")
            await self.redis_client.publish(self.filtered_signals_channel, signal_json)

            self.stats["signals_published"] += 1

            # Log sécurisé avec vérification des clés
            strategy = validated_signal.get("strategy", "N/A")
            symbol = validated_signal.get("symbol", "N/A")
            timeframe = validated_signal.get("timeframe", "N/A")
            side = validated_signal.get("side", "N/A")

            logger.debug(
                f"Signal publié vers coordinator: {strategy} {symbol} {timeframe} {side}"
            )

        except Exception:
            logger.exception("Erreur publication signal")
            self.stats["publish_errors"] += 1
            raise

    async def publish_multiple_signals(self, validated_signals: list):
        """
        Publie plusieurs signaux validés en batch.

        Args:
            validated_signals: Liste de signaux validés
        """
        if not validated_signals:
            return

        try:
            # Publication en pipeline pour optimiser les performances
            if self.redis_client is None:
                raise RuntimeError("Client Redis non initialisé")
            pipe = self.redis_client.pipeline()

            for signal in validated_signals:
                signal_json = json.dumps(signal, default=str)
                pipe.publish(self.filtered_signals_channel, signal_json)

            await pipe.execute()

            self.stats["signals_published"] += len(validated_signals)

            logger.info(
                f"Batch de {len(validated_signals)} signaux publié vers coordinator"
            )

        except Exception:
            logger.exception("Erreur publication batch")
            self.stats["publish_errors"] += 1
            raise

    async def get_pending_signals_count(self) -> int:
        """
        Récupère le nombre de signaux en attente dans la queue.

        Returns:
            Nombre de signaux en attente
        """
        try:
            # Vérifier la longueur de la liste des signaux en attente
            if self.redis_client is None:
                return 0
            count_result = self.redis_client.llen(
                "roottrading:signals:pending")
            count = (
                await count_result
                if hasattr(count_result, "__await__")
                else count_result
            )
            return int(count) if count is not None else 0
        except Exception:
            logger.exception("Erreur récupération count signaux")
            return 0

    async def health_check(self) -> bool:
        """
        Vérifie l'état de santé des connexions Redis.

        Returns:
            True si les connexions sont saines, False sinon
        """
        try:
            if not self.is_connected:
                return False

            # Test ping sur les deux clients
            if self.redis_client is not None:
                await self.redis_client.ping()
            if self.subscriber_client is not None:
                await self.subscriber_client.ping()
        except Exception as e:
            logger.warning(f"Health check Redis échoué: {e}")
            return False
        else:
            return True

    def get_stats(self) -> dict[str, Any]:
        """
        Récupère les statistiques Redis.

        Returns:
            Dictionnaire des statistiques
        """
        return {
            "is_connected": self.is_connected,
            "signals_received": self.stats["signals_received"],
            "signals_published": self.stats["signals_published"],
            "connection_errors": self.stats["connection_errors"],
            "publish_errors": self.stats["publish_errors"],
            "signals_channel": self.signals_channel,
            "filtered_signals_channel": self.filtered_signals_channel,
        }

    async def reset_stats(self):
        """Remet à zéro les statistiques."""
        self.stats = {
            "signals_received": 0,
            "signals_published": 0,
            "connection_errors": 0,
            "publish_errors": 0,
        }
        logger.info("Statistiques Redis remises à zéro")

    async def flush_channels(self):
        """Vide tous les canaux Redis (utile pour debug/test)."""
        try:
            # Note: Redis pub/sub ne stocke pas les messages, donc pas besoin de flush
            # Cette méthode est plutôt pour compatibilité future
            logger.info(
                "Flush des canaux Redis (pas d'action nécessaire pour pub/sub)")

        except Exception:
            logger.exception("Erreur flush canaux")

    async def set_callback(
        self, callback: Callable[[str], None] | Callable[[str], Any]
    ):
        """
        Définit le callback pour le traitement des signaux.

        Args:
            callback: Fonction de callback
        """
        self.signal_callback = callback
        logger.info("Callback Redis défini")

    async def test_connection(self) -> bool:
        """
        Test de connexion Redis.

        Returns:
            True si la connexion fonctionne, False sinon
        """
        try:
            if not self.redis_client:
                return False

            # Test avec un ping simple
            result = await self.redis_client.ping()
        except Exception:
            logger.exception("Test connexion Redis échoué")
            return False
        else:
            return result is True
