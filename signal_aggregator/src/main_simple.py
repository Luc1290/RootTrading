"""
Point d'entrée principal pour le microservice Signal Aggregator - VERSION ULTRA-SIMPLIFIÉE.
Utilise le nouveau système sans validators complexes.
"""

import asyncio
import os
import sys
from datetime import datetime, timezone

import psycopg2
import psycopg2.extensions
from aiohttp import web

# Configuration du logging centralisée
from shared.logging_config import setup_logging

from .context_manager import ContextManager
from .database_manager import DatabaseManager
from .signal_aggregator_simple import SimpleSignalAggregatorService

log_level = "DEBUG" if os.getenv("DEBUG_LOGS", "false").lower() == "true" else "INFO"
logger = setup_logging("signal_aggregator", log_level=log_level)


class SimpleSignalAggregatorApp:
    """Application principale du signal aggregator ultra-simplifié."""

    def __init__(self):
        # Configuration base de données
        self.db_config = {
            "host": os.getenv("DB_HOST", "db"),
            "port": os.getenv("DB_PORT", "5432"),
            "database": os.getenv("DB_NAME", "trading"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "postgres"),
        }

        # Modules simplifiés
        self.db_connection: psycopg2.extensions.connection | None = None
        self.context_manager: ContextManager | None = None
        self.database_manager: DatabaseManager | None = None
        self.aggregator_service: SimpleSignalAggregatorService | None = None

        # Web server pour health checks
        self.web_app = None
        self.web_runner = None

        # Statistiques
        self.start_time = datetime.now(tz=timezone.utc)

    async def initialize(self):
        """Initialise tous les composants simplifiés."""
        logger.info("🚀 Initialisation Signal Aggregator SIMPLIFIÉ...")

        # Connexion base de données
        await self.connect_db()

        # Gestionnaires essentiels seulement
        self.context_manager = ContextManager(self.db_connection)
        self.database_manager = DatabaseManager(self.db_connection)

        # Service d'agrégation simplifié
        self.aggregator_service = SimpleSignalAggregatorService(
            self.context_manager,
            self.database_manager,
            self.db_connection,  # Passer la connexion DB pour les filtres critiques
        )

        # Web server pour health checks
        await self.setup_web_server()

        logger.info("✅ Signal Aggregator simplifié initialisé")

    async def connect_db(self):
        """Établit la connexion à la base de données."""
        try:
            self.db_connection = psycopg2.connect(
                host=str(self.db_config["host"]),
                port=int(self.db_config["port"]),
                database=str(self.db_config["database"]),
                user=str(self.db_config["user"]),
                password=str(self.db_config["password"]),
            )
            self.db_connection.autocommit = (
                True  # Important pour éviter les transactions bloquées
            )
            logger.info("✅ Connexion DB établie")
        except Exception:
            logger.exception("❌ Erreur connexion DB")
            raise

    def ensure_db_connection(self):
        """Vérifie et recrée la connexion DB si nécessaire."""
        try:
            if self.db_connection is None:
                raise psycopg2.OperationalError("Connection is None")

            # Tester si la connexion est toujours valide
            if self.db_connection.closed:
                logger.warning("Connexion DB fermée, reconnexion...")
                self.db_connection = psycopg2.connect(
                    host=str(self.db_config["host"]),
                    port=int(self.db_config["port"]),
                    database=str(self.db_config["database"]),
                    user=str(self.db_config["user"]),
                    password=str(self.db_config["password"]),
                )
                self.db_connection.autocommit = True
                logger.info("✅ Reconnexion DB réussie")
                return self.db_connection

            # Tester avec une requête simple
            with self.db_connection.cursor() as cursor:
                cursor.execute("SELECT 1")
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            logger.warning(f"Connexion DB invalide: {e}, reconnexion...")
            try:
                self.db_connection = psycopg2.connect(
                    host=str(self.db_config["host"]),
                    port=int(self.db_config["port"]),
                    database=str(self.db_config["database"]),
                    user=str(self.db_config["user"]),
                    password=str(self.db_config["password"]),
                )
                self.db_connection.autocommit = True
                logger.info("✅ Reconnexion DB réussie")
            except Exception:
                logger.exception("❌ Échec reconnexion DB: ")
                raise
            else:
                return self.db_connection
        else:
            return self.db_connection

    async def setup_web_server(self):
        """Configure le serveur web pour les health checks."""
        self.web_app = web.Application()

        # Routes simplifiées
        self.web_app.router.add_get("/health", self.health_check)
        self.web_app.router.add_get("/stats", self.get_stats)

        # Démarrage du serveur
        self.web_runner = web.AppRunner(self.web_app)
        await self.web_runner.setup()

        site = web.TCPSite(self.web_runner, "0.0.0.0", 8080)
        await site.start()

        logger.info("✅ Health check server: port 8080")

    async def health_check(self, _request):
        """Endpoint de health check simplifié."""
        try:
            uptime = (datetime.now(tz=timezone.utc) - self.start_time).total_seconds()

            # Test connexion DB
            if self.db_connection is None:
                db_status = "ERROR"
            else:
                with self.db_connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    db_status = "OK"

            # Stats du service simplifié
            stats = (
                self.aggregator_service.get_stats() if self.aggregator_service else {}
            )

            return web.json_response(
                {
                    "status": "healthy",
                    "version": "SIMPLIFIÉ v2.0",
                    "uptime_seconds": uptime,
                    "database_status": db_status,
                    "stats_summary": {
                        "signals_received": stats.get("service_stats", {}).get(
                            "signals_received", 0
                        ),
                        "signals_validated": stats.get("service_stats", {}).get(
                            "signals_validated", 0
                        ),
                        "success_rate": stats.get("processor_stats", {}).get(
                            "success_rate_percent", 0
                        ),
                    },
                    "features": [
                        "Consensus adaptatif seulement",
                        "Filtres critiques minimalistes (4 max)",
                        "Pas de validators complexes",
                        "Performance optimisée",
                    ],
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                }
            )

        except Exception as e:
            logger.exception("❌ Erreur health check")
            return web.json_response(
                {
                    "status": "unhealthy",
                    "version": "SIMPLIFIÉ v2.0",
                    "error": str(e),
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                },
                status=500,
            )

    async def get_stats(self, _request):
        """Endpoint pour les statistiques simplifiées."""
        try:
            if not self.aggregator_service:
                return web.json_response(
                    {"error": "Service non initialisé"}, status=503
                )

            stats = self.aggregator_service.get_stats()

            # Enrichir avec infos système
            enriched_stats = {
                "system_info": {
                    "version": "Signal Aggregator SIMPLIFIÉ v2.0",
                    "uptime_seconds": (
                        datetime.now(tz=timezone.utc) - self.start_time
                    ).total_seconds(),
                    "features_removed": [
                        "23+ validators complexes",
                        "Système hiérarchique",
                        "Pouvoir de veto",
                        "Scoring pondéré complexe",
                    ],
                    "features_active": [
                        "Consensus adaptatif par régime",
                        "Filtres critiques (4 max)",
                        "Buffer intelligent",
                        "Protection contradictions",
                    ],
                },
                **stats,
            }

            return web.json_response(enriched_stats)

        except Exception as e:
            logger.exception("❌ Erreur récupération stats")
            return web.json_response({"error": str(e)}, status=500)

    async def run(self):
        """Lance le service d'agrégation simplifié."""
        try:
            logger.info("🚀 Démarrage service d'agrégation SIMPLIFIÉ...")
            if self.aggregator_service is None:
                raise RuntimeError("Aggregator service not initialized")
            await self.aggregator_service.start()

        except KeyboardInterrupt:
            logger.info("⏹️  Arrêt demandé par l'utilisateur")
        except Exception:
            logger.exception("❌ Erreur service d'agrégation")
            raise
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Arrêt propre de l'application."""
        logger.info("⏹️  Arrêt Signal Aggregator SIMPLIFIÉ...")

        # Arrêt du serveur web
        if self.web_runner:
            await self.web_runner.cleanup()

        # Fermeture DB
        if self.db_connection:
            self.db_connection.close()

        logger.info("✅ Signal Aggregator SIMPLIFIÉ arrêté")


async def main():
    """Point d'entrée principal."""
    app = SimpleSignalAggregatorApp()

    try:
        await app.initialize()
        await app.run()
    except KeyboardInterrupt:
        logger.info("⏹️  Arrêt par Ctrl+C")
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Message de démarrage
    print("🚀 Signal Aggregator SIMPLIFIÉ v2.0")
    print("📋 Features: Consensus adaptatif + Filtres critiques seulement")
    print("⚡ Optimisé pour: Performance + Simplicité")
    print("=" * 60)

    asyncio.run(main())
