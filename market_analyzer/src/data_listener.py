"""
Market Data Listener
Écoute les nouvelles données dans market_data et déclenche les calculs.
Système trigger-based pour traitement temps réel.
"""

import asyncio
import builtins
import contextlib
import json
import logging
import sys
from datetime import datetime, timezone
# Ajouter les chemins pour les imports
from pathlib import Path
from typing import Any

import asyncpg  # type: ignore

from shared.src.config import get_db_config

from .indicator_processor import IndicatorProcessor

sys.path.append(str((Path(__file__).parent / "../../").resolve()))


logger = logging.getLogger(__name__)


class DataListener:
    """
    Écoute les changements dans market_data et déclenche les calculs automatiquement.
    Utilise PostgreSQL LISTEN/NOTIFY pour traitement temps réel.
    """

    def __init__(self):
        self.db_pool = None
        self.listen_conn = None
        self.running = False
        self.processed_count = 0
        self.indicator_processor = IndicatorProcessor()
        # Semaphore pour limiter les connexions DB concurrentes
        self.db_semaphore = asyncio.Semaphore(
            15)  # Max 15 connexions simultanées

        logger.info("📡 DataListener initialisé")

    async def initialize(self):
        """Initialise les connexions et le moteur de calcul."""
        try:
            db_config = get_db_config()

            # Pool principal pour les requêtes - AUGMENTÉ pour traitement
            # historique
            self.db_pool = await asyncpg.create_pool(
                host=db_config["host"],
                port=db_config["port"],
                database=db_config["database"],
                user=db_config["user"],
                password=db_config["password"],
                min_size=5,
                max_size=25,  # Augmenté de 10 à 25
                command_timeout=30,  # Timeout requêtes longues
                server_settings={
                    "application_name": "market_analyzer_pool",
                    "statement_timeout": "30000",  # 30s timeout pour statements
                    # Tuer les "idle in transaction" après 60s
                    "idle_in_transaction_session_timeout": "60000",
                },
            )

            # Connexion dédiée pour LISTEN
            self.listen_conn = await asyncpg.connect(
                host=db_config["host"],
                port=db_config["port"],
                database=db_config["database"],
                user=db_config["user"],
                password=db_config["password"],
                server_settings={
                    "application_name": "market_analyzer_listener",
                    "statement_timeout": "60000",  # 60s timeout pour LISTEN
                    "idle_in_transaction_session_timeout": "120000",  # 2min pour listener
                },
            )

            # Initialiser le processeur d'indicateurs
            await self.indicator_processor.initialize()

            # Créer le trigger si nécessaire
            await self._setup_database_trigger()

            logger.info("✅ DataListener connecté et prêt")

        except Exception:
            logger.exception("❌ Erreur initialisation DataListener")
            raise

    async def _setup_database_trigger(self):
        """
        Configure le trigger PostgreSQL pour notifier les nouveaux inserts.

        IDEMPOTENT: Nettoie d'abord les anciens triggers/fonctions avant de recréer.
        Safe pour rebuild/restart sans intervention manuelle.
        """

        # ÉTAPE 1: Nettoyer proprement les anciens triggers et fonctions
        cleanup_sql = """
            -- Supprimer le trigger s'il existe
            DROP TRIGGER IF EXISTS market_data_change_trigger ON market_data;

            -- Supprimer l'ancienne fonction (CASCADE pour forcer si référencée)
            DROP FUNCTION IF EXISTS notify_market_data_change() CASCADE;
        """

        # ÉTAPE 2: Créer la nouvelle fonction trigger
        trigger_function = """
            CREATE OR REPLACE FUNCTION notify_market_data_change()
            RETURNS TRIGGER AS $$
            BEGIN
                -- Envoyer notification avec les détails de la nouvelle donnée
                PERFORM pg_notify(
                    'market_data_change',
                    json_build_object(
                        'symbol', NEW.symbol,
                        'timeframe', NEW.timeframe,
                        'time', extract(epoch from NEW.time)::bigint,
                        'action', TG_OP
                    )::text
                );
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
        """

        # ÉTAPE 3: Créer le trigger
        trigger_definition = """
            CREATE TRIGGER market_data_change_trigger
                AFTER INSERT OR UPDATE ON market_data
                FOR EACH ROW
                EXECUTE FUNCTION notify_market_data_change();
        """

        try:
            if not self.db_pool:
                raise RuntimeError("db_pool not initialized")
            async with self.db_pool.acquire() as conn:
                # Exécuter le nettoyage d'abord
                logger.info("🧹 Nettoyage des anciens triggers...")
                await conn.execute(cleanup_sql)

                # Créer la fonction
                logger.info("🔧 Création de la fonction trigger...")
                await conn.execute(trigger_function)

                # Créer le trigger
                logger.info("🎯 Création du trigger...")
                await conn.execute(trigger_definition)

            logger.info("✅ Trigger PostgreSQL configuré pour market_data")

        except Exception:
            logger.exception("❌ Erreur configuration trigger")
            # Ne pas raise pour permettre au service de démarrer quand même
            logger.warning("⚠️ Service démarré sans trigger (mode dégradé)")
            # raise  # Commenté pour ne pas bloquer le démarrage

    async def start_listening(self):
        """Démarre l'écoute des notifications."""
        self.running = True
        logger.info("🎧 Démarrage de l'écoute des changements market_data...")

        try:
            # S'abonner aux notifications
            if not self.listen_conn:
                raise RuntimeError("listen_conn not initialized")
            await self.listen_conn.add_listener(
                "market_data_change", self._handle_notification
            )

            logger.info("✅ Écoute active - en attente de nouvelles données...")

            # Boucle principale d'écoute
            while self.running:
                try:
                    # Attendre les notifications (bloquant)
                    # Petite pause pour éviter de surcharger
                    await asyncio.sleep(0.1)

                except asyncio.CancelledError:
                    logger.info("🛑 Écoute interrompue")
                    break
                except Exception:
                    logger.exception("❌ Erreur dans la boucle d'écoute")
                    await asyncio.sleep(1)  # Attendre avant de retry

        except Exception:
            logger.exception("❌ Erreur critique dans l'écoute")
        finally:
            await self._cleanup()

    async def _handle_notification(self, _connection, _pid, _channel, payload):
        """
        Gestionnaire appelé quand une notification est reçue.

        Args:
            payload: JSON avec symbol, timeframe, time, action
        """
        try:
            # Parser la notification
            data = json.loads(payload)
            symbol = data["symbol"]
            timeframe = data["timeframe"]
            timestamp = datetime.fromtimestamp(data["time"], tz=timezone.utc)
            action = data["action"]

            logger.debug(
                f"📬 Notification reçue: {action} {symbol} {timeframe} @ {timestamp}"
            )

            # Ne traiter que les INSERT (nouvelles données)
            if action == "INSERT":
                await self._process_new_data(symbol, timeframe, timestamp)
                self.processed_count += 1

                if self.processed_count % 10 == 0:
                    logger.info(
                        f"📊 {self.processed_count} analyses complétées")

        except Exception:
            logger.exception("❌ Erreur traitement notification")
            logger.exception("Payload: ")

    async def _process_new_data(
            self,
            symbol: str,
            timeframe: str,
            timestamp: datetime):
        """
        Traite une nouvelle donnée en lançant les calculs.

        Args:
            symbol: Symbole (ex: BTCUSDC)
            timeframe: Timeframe (ex: 1m, 5m)
            timestamp: Timestamp de la nouvelle donnée
        """
        try:
            # Vérifier si on a déjà analysé cette donnée
            if await self._is_already_analyzed(symbol, timeframe, timestamp):
                logger.debug(
                    f"⏭️ Déjà analysé: {symbol} {timeframe} @ {timestamp}")
                return

            # Appeler le processeur d'indicateurs
            await self.indicator_processor.process_new_data(
                symbol, timeframe, timestamp
            )

            logger.debug(f"✅ Traitement terminé: {symbol} {timeframe}")

        except Exception:
            logger.exception("❌ Erreur traitement données")

    async def _is_already_analyzed(
        self, symbol: str, timeframe: str, timestamp: datetime
    ) -> bool:
        """Vérifie si cette donnée a déjà été analysée."""

        query = """
            SELECT 1 FROM analyzer_data
            WHERE symbol = $1 AND timeframe = $2 AND time = $3
            LIMIT 1
        """

        try:
            if not self.db_pool:
                raise RuntimeError("db_pool not initialized")
            async with self.db_pool.acquire() as conn:
                result = await asyncio.wait_for(
                    conn.fetchval(query, symbol, timeframe, timestamp), timeout=5.0
                )
                return result is not None

        except Exception:
            logger.exception("❌ Erreur vérification analyse")
            return False

    async def process_historical_optimized(
        self,
        symbol: str | None = None,
        timeframe: str | None = None,
        limit: int = 1000000,
    ):
        """
        Traite les données historiques de manière OPTIMISÉE.
        - Traite par symbole
        - Dans l'ordre chronologique
        - Réutilise les calculs précédents
        """
        logger.info(
            "🚀 Démarrage traitement optimisé des données non analysées...")

        # Si pas de symbole spécifique, traiter chaque symbole séparément
        if symbol is None:
            symbols_to_process = await self._get_symbols_with_gaps()
            logger.info(f"📊 Symboles à traiter: {symbols_to_process}")

            total_processed = 0
            for sym in symbols_to_process:
                processed = await self._process_symbol_historical(sym, timeframe, limit)
                total_processed += processed
                # Petite pause entre chaque symbole
                await asyncio.sleep(1)

            logger.info(
                f"✅ Traitement optimisé terminé: {total_processed} données analysées"
            )
        else:
            processed = await self._process_symbol_historical(symbol, timeframe, limit)
            logger.info(
                f"✅ Traitement optimisé terminé pour {symbol}: {processed} données analysées"
            )

    async def _get_symbols_with_gaps(self) -> list:
        """Récupère la liste des symboles ayant des données non analysées."""
        query = """
            SELECT DISTINCT md.symbol
            FROM market_data md
            LEFT JOIN analyzer_data ad ON (
                md.symbol = ad.symbol AND
                md.timeframe = ad.timeframe AND
                md.time = ad.time
            )
            WHERE ad.time IS NULL
            ORDER BY md.symbol
        """

        if not self.db_pool:
            raise RuntimeError("db_pool not initialized")
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(query)
            return [row["symbol"] for row in rows]

    async def _process_symbol_historical(
        self, symbol: str, timeframe: str | None = None, limit: int = 1000000
    ) -> int:
        """Traite l'historique d'un symbole spécifique dans l'ordre chronologique."""
        logger.info(f"🔄 Traitement historique optimisé pour {symbol}...")

        # Requête optimisée : EXISTS plus rapide que LEFT JOIN
        query = """
            SELECT DISTINCT md.timeframe, md.time
            FROM market_data md
            WHERE md.symbol = $1
            AND NOT EXISTS (
                SELECT 1 FROM analyzer_data ad
                WHERE ad.symbol = md.symbol
                AND ad.timeframe = md.timeframe
                AND ad.time = md.time
            )
        """

        params: list[Any] = [symbol]

        if timeframe:
            query += " AND md.timeframe = $2"
            params.append(timeframe)

        query += " ORDER BY md.timeframe, md.time ASC LIMIT $" + \
            str(len(params) + 1)
        params.append(limit)

        total_processed = 0

        try:
            if not self.db_pool:
                raise RuntimeError("db_pool not initialized")
            async with self.db_pool.acquire() as conn:
                rows = await asyncio.wait_for(conn.fetch(query, *params), timeout=10.0)

            if not rows:
                logger.info(f"✅ {symbol}: Aucune donnée non analysée")
                return 0

            logger.info(f"📊 {symbol}: {len(rows)} données à analyser")

            # Grouper par timeframe pour optimiser
            timeframes: dict[str, list] = {}
            for row in rows:
                tf = row["timeframe"]
                if tf not in timeframes:
                    timeframes[tf] = []
                timeframes[tf].append(row["time"])

            # Traiter chaque timeframe séparément dans l'ordre chronologique
            for tf, timestamps in timeframes.items():
                logger.info(
                    f"⏱️ {symbol} {tf}: {len(timestamps)} données à traiter")

                processed = 0
                errors = 0

                # Traiter dans l'ordre chronologique
                for timestamp in timestamps:
                    try:
                        async with self.db_semaphore:  # Limiter les connexions concurrentes
                            await self.indicator_processor.process_new_data(
                                symbol, tf, timestamp
                            )
                        processed += 1

                        # Log de progression
                        if processed % 100 == 0:
                            percent = (processed / len(timestamps)) * 100
                            logger.info(
                                f"📈 {symbol} {tf}: {processed}/{len(timestamps)} ({percent:.1f}%)"
                            )

                        # Pause raisonnable tous les 10 éléments pour éviter
                        # saturation DB
                        if processed % 10 == 0:
                            await asyncio.sleep(0.01)  # 10ms

                    except Exception:
                        errors += 1
                        if errors <= 5:  # Limiter les logs d'erreur
                            logger.exception(
                                "❌ Erreur {symbol} {tf} @ {timestamp}")
                        continue

                logger.info(
                    f"✅ {symbol} {tf}: {processed} données traitées ({errors} erreurs)"
                )
                total_processed += processed

            logger.info(
                f"✅ {symbol}: Traitement terminé - {total_processed} données analysées"
            )

        except Exception:
            logger.exception("❌ Erreur traitement {symbol}")

        return total_processed

    async def get_stats(self) -> dict[str, Any]:
        """Retourne les statistiques du listener."""

        # Compter les données avec focus sur les gaps récents (24h)
        stats_query = """
            SELECT
                (SELECT COUNT(*) FROM market_data) as total_market_data,
                (SELECT COUNT(*) FROM analyzer_data) as total_analyzer_data,
                (SELECT COUNT(DISTINCT symbol) FROM market_data) as symbols_count,
                (SELECT COUNT(DISTINCT timeframe) FROM market_data) as timeframes_count,
                (SELECT COUNT(*) FROM market_data md
                 LEFT JOIN analyzer_data ad ON (
                     md.symbol = ad.symbol AND
                     md.timeframe = ad.timeframe AND
                     md.time = ad.time
                 )
                 WHERE ad.time IS NULL
                 AND md.time >= NOW() - INTERVAL '24 hours'
                ) as recent_gaps_24h
        """

        try:
            if not self.db_pool:
                raise RuntimeError("db_pool not initialized")
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(stats_query)

                coverage_percent = 0
                if row["total_market_data"] > 0:
                    coverage_percent = (
                        row["total_analyzer_data"] / row["total_market_data"]
                    ) * 100

                return {
                    "running": self.running,
                    "processed_count": self.processed_count,
                    "total_market_data": row["total_market_data"],
                    "total_analyzer_data": row["total_analyzer_data"],
                    "coverage_percent": round(coverage_percent, 2),
                    "symbols_count": row["symbols_count"],
                    "timeframes_count": row["timeframes_count"],
                    "missing_analyses": row[
                        "recent_gaps_24h"
                    ],  # Maintenant seulement les gaps récents
                    "total_historical_gaps": row["total_market_data"]
                    - row["total_analyzer_data"],
                }

        except Exception as e:
            logger.exception("❌ Erreur récupération stats")
            return {
                "running": self.running,
                "processed_count": self.processed_count,
                "error": str(e),
            }

    async def _cleanup(self):
        """
        Nettoie les ressources proprement.

        IMPORTANT: Ferme toutes les connexions pour éviter les orphelins
        lors d'un rebuild du container.
        """
        try:
            # Stopper l'écoute des notifications
            if self.listen_conn:
                try:
                    await self.listen_conn.remove_listener(
                        "market_data_change", self._handle_notification
                    )
                    await self.listen_conn.close()
                    logger.info("✅ Connexion LISTEN fermée")
                except Exception as e:
                    logger.warning(f"⚠️ Erreur fermeture LISTEN: {e}")

            # Fermer le processeur d'indicateurs
            try:
                await self.indicator_processor.close()
                logger.info("✅ Indicator processor fermé")
            except Exception as e:
                logger.warning(f"⚠️ Erreur fermeture processor: {e}")

            # Fermer le pool de connexions
            if self.db_pool:
                try:
                    # Attendre que toutes les connexions se terminent (timeout
                    # 5s)
                    await asyncio.wait_for(self.db_pool.close(), timeout=5.0)
                    logger.info("✅ Pool DB fermé proprement")
                except asyncio.TimeoutError:
                    logger.warning(
                        "⚠️ Timeout fermeture pool - connexions forcées")
                    # Force terminate si timeout
                    await self.db_pool.terminate()
                except Exception as e:
                    logger.warning(f"⚠️ Erreur fermeture pool: {e}")

            logger.info("🧹 DataListener nettoyé complètement")

        except Exception:
            logger.exception("❌ Erreur critique nettoyage")
            # Toujours essayer de terminer le pool
            if self.db_pool:
                with contextlib.suppress(builtins.BaseException):
                    await self.db_pool.terminate()

    async def stop(self):
        """Arrête l'écoute."""
        logger.info("🛑 Arrêt du DataListener...")
        self.running = False
        await self._cleanup()
