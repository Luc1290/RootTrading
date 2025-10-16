"""
Module de gestion des publications Redis pour l'analyzer.
Publie les signaux générés par les stratégies.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisPublisher:
    """Gestionnaire de publication des signaux vers Redis."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: redis.Redis | None = None

        # Canaux Redis pour les différents types de messages
        self.channels = {
            "signals": "analyzer:signals",
            "health": "analyzer:health",
            "metrics": "analyzer:metrics",
            "errors": "analyzer:errors",
        }

    async def connect(self):
        """Établit la connexion Redis."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Connexion Redis Publisher établie")
        except Exception:
            logger.exception("Erreur connexion Redis Publisher")
            raise

    async def disconnect(self):
        """Ferme la connexion Redis."""
        if self.redis_client is not None:
            await self.redis_client.close()
            logger.info("Connexion Redis Publisher fermée")

    async def publish_signals(
        self, signals: list[dict[str, Any]], mode: str = "individual"
    ):
        """
        Publie une liste de signaux vers Redis.

        Args:
            signals: Liste des signaux à publier
            mode: "individual" (recommandé) ou "batch" (legacy)
        """
        if not signals:
            return

        try:
            if mode == "individual":
                # Mode recommandé : publication individuelle pour consensus
                # adaptatif
                for signal in signals:
                    await self.publish_signal(signal)
                logger.info(
                    f"Publié {len(signals)} signaux individuellement vers Redis"
                )

            else:
                # Mode legacy : publication en batch
                batch_message = {
                    "type": "signal_batch",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "count": len(signals),
                    "signals": signals,
                }

                await self._publish_message(self.channels["signals"], batch_message)
                logger.info(
                    f"Publié batch de {len(signals)} signaux vers Redis")

        except Exception as e:
            logger.exception("Erreur publication signaux")
            await self.publish_error(f"Erreur publication signaux: {e!s}")

    async def publish_signal(self, signal: dict[str, Any]):
        """
        Publie un signal individuel vers Redis.

        Args:
            signal: Signal à publier
        """
        try:
            # Enrichissement du signal avec des métadonnées
            enriched_signal = {
                **signal,
                "analyzer_timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "analyzer",
                "version": "2.0",
            }

            # Publication sur le canal principal
            await self._publish_message(self.channels["signals"], enriched_signal)

            # Publication sur un canal spécifique au symbole (optionnel)
            symbol_channel = f"analyzer:signals:{signal['symbol'].lower()}"
            await self._publish_message(symbol_channel, enriched_signal)

            logger.debug(
                f"Signal publié: {signal['strategy']} {signal['symbol']} {signal['side']}"
            )

        except Exception:
            logger.exception("Erreur publication signal")

    async def publish_health_status(self, status: dict[str, Any]):
        """
        Publie le statut de santé de l'analyzer.

        Args:
            status: Informations de statut
        """
        try:
            health_message = {
                "service": "analyzer",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": status,
            }

            await self._publish_message(self.channels["health"], health_message)

            logger.debug("Statut de santé publié")

        except Exception:
            logger.exception("Erreur publication santé")

    async def publish_metrics(self, metrics: dict[str, Any]):
        """
        Publie les métriques de performance de l'analyzer.

        Args:
            metrics: Métriques à publier
        """
        try:
            metrics_message = {
                "service": "analyzer",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics,
            }

            await self._publish_message(self.channels["metrics"], metrics_message)

            logger.debug("Métriques publiées")

        except Exception:
            logger.exception("Erreur publication métriques")

    async def publish_error(
        self, error_message: str, context: dict[Any, Any] | None = None
    ):
        """
        Publie une erreur vers Redis.

        Args:
            error_message: Message d'erreur
            context: Contexte additionnel (optionnel)
        """
        try:
            error_data = {
                "service": "analyzer",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": error_message,
                "context": context or {},
            }

            await self._publish_message(self.channels["errors"], error_data)

            logger.debug(f"Erreur publiée: {error_message}")

        except Exception:
            logger.exception("Erreur publication erreur")

    async def _publish_message(self, channel: str, message: dict[str, Any]):
        """
        Publie un message sur un canal Redis.

        Args:
            channel: Canal Redis
            message: Message à publier
        """
        if self.redis_client is None:
            logger.warning("Client Redis non connecté")
            return

        # Sérialisation JSON
        message_json = json.dumps(message, default=str)

        # Publication
        await self.redis_client.publish(channel, message_json)

    async def publish_strategy_performance(
        self, strategy_name: str, performance_data: dict[str, Any]
    ):
        """
        Publie les données de performance d'une stratégie.

        Args:
            strategy_name: Nom de la stratégie
            performance_data: Données de performance
        """
        try:
            perf_message = {
                "service": "analyzer",
                "strategy": strategy_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "performance": performance_data,
            }

            performance_channel = f"analyzer:performance:{strategy_name.lower()}"
            await self._publish_message(performance_channel, perf_message)

            logger.debug(f"Performance publiée pour {strategy_name}")

        except Exception:
            logger.exception(f"Erreur publication performance {strategy_name}")


class RedisSubscriber:
    """Gestionnaire d'abonnement aux messages Redis (pour les configurations, etc.)."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: redis.Redis | None = None
        self.pubsub: redis.client.PubSub | None = None
        self.running = False

        # Canaux d'abonnement
        self.subscription_channels = [
            "analyzer:config",
            "analyzer:commands",
            "system:shutdown",
        ]

    async def connect(self):
        """Établit la connexion Redis et l'abonnement."""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            self.pubsub = self.redis_client.pubsub()

            # Abonnement aux canaux
            await self.pubsub.subscribe(*self.subscription_channels)

            logger.info("Redis Subscriber connecté et abonné")

        except Exception:
            logger.exception("Erreur connexion Redis Subscriber")
            raise

    async def disconnect(self):
        """Ferme les connexions Redis."""
        self.running = False

        if self.pubsub is not None:
            await self.pubsub.unsubscribe()
            await self.pubsub.close()

        if self.redis_client is not None:
            await self.redis_client.close()

        logger.info("Redis Subscriber déconnecté")

    async def listen(self, message_handler=None):
        """
        Écoute les messages Redis.

        Args:
            message_handler: Fonction de traitement des messages
        """
        if self.pubsub is None:
            logger.error("PubSub non initialisé")
            return

        self.running = True
        logger.info("Écoute des messages Redis démarrée")

        try:
            async for message in self.pubsub.listen():
                if not self.running:
                    break

                if message["type"] == "message":
                    try:
                        # Décodage du message
                        channel = message["channel"].decode("utf-8")
                        data = json.loads(message["data"].decode("utf-8"))

                        logger.debug(f"Message reçu sur {channel}: {data}")

                        # Traitement du message
                        if message_handler:
                            await message_handler(channel, data)
                        else:
                            await self._default_message_handler(channel, data)

                    except Exception:
                        logger.exception("Erreur traitement message")

        except Exception:
            logger.exception("Erreur écoute Redis")
        finally:
            self.running = False

    async def _default_message_handler(
            self, channel: str, data: dict[str, Any]):
        """Gestionnaire par défaut des messages."""
        logger.info(
            f"Message reçu sur {channel}: {data.get('type', 'unknown')}")

        # Traitement basique selon le canal
        if channel == "analyzer:config":
            logger.info("Configuration mise à jour reçue")
        elif channel == "analyzer:commands":
            command = data.get("command")
            if command == "reload_strategies":
                logger.info("Commande de rechargement des stratégies reçue")
        elif channel == "system:shutdown":
            logger.info("Commande d'arrêt système reçue")
            self.running = False
