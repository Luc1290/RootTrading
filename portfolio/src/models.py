"""
Modèles de données pour le service Portfolio.
Définit les structures de données et les interactions avec la base de données.
"""

import contextlib
import logging
import os

# Importer les modules partagés
import sys
import time
from datetime import datetime
from typing import Any

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, execute_values

from shared.src.config import get_db_url
from shared.src.schemas import AssetBalance, PortfolioSummary

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../")))


# Configuration du logging
logger = logging.getLogger(__name__)


class DBManager:
    """
    Gestionnaire de connexion à la base de données.
    Fournit des méthodes pour interagir avec la base de données PostgreSQL.
    Utilise un pool de connexions pour améliorer les performances.
    """

    # Singleton pattern pour le pool de connexions
    _pool = None

    @classmethod
    def get_pool(cls, db_url=None):
        """
        Obtient ou crée le pool de connexions partagé.

        Args:
            db_url: URL de connexion à la base de données

        Returns:
            Pool de connexions
        """
        if cls._pool is None:
            db_url = db_url or get_db_url()
            try:
                # Créer un pool avec min=5, max=20 connexions
                cls._pool = pool.ThreadedConnectionPool(
                    minconn=5, maxconn=20, dsn=db_url
                )
                logger.info("✅ Pool de connexions DB initialisé")
            except Exception:
                logger.exception("❌ Erreur lors de la création du pool DB")
                raise
        return cls._pool

    def __init__(self, db_url: str | None = None):
        """
        Initialise le gestionnaire de base de données.

        Args:
            db_url: URL de connexion à la base de données
        """
        self.db_url = db_url or get_db_url()
        self.conn = None
        self.pool = None

        # Utiliser le pool de connexions
        try:
            self.pool = self.get_pool(self.db_url)
            self.conn = self.pool.getconn()
            logger.debug(
                "✅ Connexion obtenue depuis le pool DB"
            )  # Changé de INFO à DEBUG
        except Exception:
            logger.exception("❌ Impossible d'obtenir une connexion du pool")
            # Fallback: connexion directe
            self._connect()

    def _connect(self) -> None:
        """
        Établit une connexion directe à la base de données (fallback).
        """
        if self.conn is not None:
            return

        try:
            self.conn = psycopg2.connect(self.db_url)
            logger.info("✅ Connexion directe à la base de données établie")
        except Exception:
            logger.exception(
                "❌ Erreur lors de la connexion à la base de données: "
            )
            self.conn = None

    def _ensure_connection(self) -> bool:
        """
        S'assure que la connexion à la base de données est active.
        Tente de récupérer une nouvelle connexion du pool si nécessaire.

        Returns:
            True si la connexion est active, False sinon
        """
        if self.conn is None:
            if self.pool:
                try:
                    self.conn = self.pool.getconn()
                    return True
                except Exception:
                    logger.exception(
                        "❌ Impossible d'obtenir une connexion du pool: "
                    )
                    # Fallback: connexion directe
                    self._connect()
                    return self.conn is not None
            else:
                self._connect()
                return self.conn is not None

        try:
            # Vérifier si la connexion est active avec un timeout
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                return True
        except Exception as e:
            # Reconnexion si nécessaire
            logger.warning(f"⚠️ Connexion à la base de données perdue: {e!s}")
            try:
                if self.pool and self.conn:
                    # Marquer la connexion comme défectueuse dans le pool
                    self.pool.putconn(self.conn, close=True)
                else:
                    with contextlib.suppress(Exception):
                        self.conn.close()
            except Exception:
                pass

            self.conn = None

            # Tenter d'obtenir une nouvelle connexion
            if self.pool:
                try:
                    self.conn = self.pool.getconn()
                    return True
                except Exception as e:
                    logger.exception(
                        "❌ Impossible d'obtenir une nouvelle connexion: "
                    )
                    # Fallback: connexion directe
                    self._connect()
            else:
                self._connect()

            return self.conn is not None

    def execute_query(
        self,
        query: str,
        params: tuple[Any, ...] | None = None,
        fetch_one: bool = False,
        fetch_all: bool = False,
        commit: bool = False,
        retry: int = 1,
    ) -> list[dict[str, Any]] | dict[str, Any] | bool | None:
        """
        Exécute une requête SQL avec retry.

        Args:
            query: Requête SQL à exécuter
            params: Paramètres de la requête
            fetch_one: Si True, récupère une seule ligne
            fetch_all: Si True, récupère toutes les lignes
            commit: Si True, valide la transaction
            retry: Nombre de tentatives

        Returns:
            Résultat de la requête ou True pour les mises à jour réussies, None en cas d'erreur
        """
        max_retries = max(1, retry)
        current_retry = 0
        last_error = None

        while current_retry < max_retries:
            if not self._ensure_connection():
                logger.error("❌ Pas de connexion à la base de données")
                return None

            try:
                # Log plus léger en production
                if len(query) > 200:
                    logger.debug(f"Exécution de la requête: {query[:200]}...")
                else:
                    logger.debug(f"Exécution de la requête: {query}")

                if params:
                    logger.debug(f"Paramètres: {params}")

                if self.conn:
                    with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                        cursor.execute(query, params)

                        if fetch_one:
                            result = cursor.fetchone()
                        elif fetch_all:
                            result = cursor.fetchall()
                        else:
                            # Pour les requêtes UPDATE/INSERT/DELETE, retourner
                            # True au lieu de None
                            result = True

                        if commit and self.conn:
                            self.conn.commit()

                        return result
                else:
                    return None

            except (psycopg2.InterfaceError, psycopg2.OperationalError) as e:
                # Erreurs de connexion, on peut réessayer
                current_retry += 1
                last_error = e
                logger.warning(
                    f"⚠️ Erreur de connexion DB (tentative {current_retry}/{max_retries}): {e!s}"
                )

                # Marquer la connexion comme défectueuse
                try:
                    if self.pool and self.conn:
                        self.pool.putconn(self.conn, close=True)
                    else:
                        try:
                            if self.conn:
                                self.conn.close()
                        except Exception:
                            pass
                except Exception:
                    pass

                self.conn = None

                # Attendre avant de réessayer
                if current_retry < max_retries:
                    retry_delay = 2**current_retry  # Backoff exponentiel
                    time.sleep(retry_delay)

            except Exception:
                # Autres erreurs, rollback et log
                logger.exception("❌ Erreur lors de l'exécution de la requête")
                if len(query) > 200:
                    logger.exception(f"Query: {query[:200]}...")
                else:
                    logger.exception("Query: ")
                logger.exception("Params: ")
                import traceback

                logger.exception(traceback.format_exc())
                try:
                    if self.conn:
                        self.conn.rollback()
                except Exception:
                    pass
                return None

        if last_error:
            logger.error(
                f"❌ Échec après {max_retries} tentatives: {last_error!s}")
        return None

    def execute_batch(
        self,
        query: str,
        params_list: list[tuple],
        page_size: int = 100,
        commit: bool = True,
    ) -> bool:
        """
        Exécute une requête SQL par lots pour améliorer les performances.

        Args:
            query: Requête SQL avec placeholders
            params_list: Liste des paramètres pour chaque exécution
            page_size: Taille de chaque lot
            commit: Si True, valide la transaction

        Returns:
            True si l'exécution a réussi, False sinon
        """
        if not params_list:
            return True

        if not self._ensure_connection():
            logger.error("❌ Pas de connexion à la base de données")
            return False

        try:
            if self.conn:
                with self.conn.cursor() as cursor:
                    # Utiliser execute_values pour les performances
                    execute_values(
                        cursor, query, params_list, page_size=page_size)

                    if commit and self.conn:
                        self.conn.commit()

                    return True
            else:
                return False

        except Exception:
            logger.exception("❌ Erreur lors de l'exécution par lots")
            try:
                if self.conn:
                    self.conn.rollback()
            except Exception:
                pass
            return False

    def execute_many(
        self, query: str, params_list: list[tuple], commit: bool = True
    ) -> bool:
        """
        Exécute une requête SQL avec plusieurs ensembles de paramètres.

        Args:
            query: Requête SQL à exécuter
            params_list: Liste des paramètres pour chaque exécution
            commit: Si True, valide la transaction

        Returns:
            True si l'exécution a réussi, False sinon
        """
        if not params_list:
            return True

        # Pour les petites listes, utiliser executemany standard
        if len(params_list) < 50:
            if not self._ensure_connection():
                logger.error("❌ Pas de connexion à la base de données")
                return False

            try:
                if self.conn:
                    with self.conn.cursor() as cursor:
                        cursor.executemany(query, params_list)

                        if commit and self.conn:
                            self.conn.commit()

                        return True
                else:
                    return False

            except Exception:
                logger.exception("❌ Erreur lors de l'exécution multiple")
                try:
                    if self.conn:
                        self.conn.rollback()
                except Exception:
                    pass
                return False
        else:
            # Pour les grandes listes, utiliser execute_batch avec pagination
            return self.execute_batch(query, params_list, commit=commit)

    def close(self) -> None:
        """
        Ferme la connexion à la base de données.
        Si un pool est utilisé, retourne la connexion au pool.
        """
        if self.conn:
            try:
                if self.pool:
                    # Retourner la connexion au pool
                    self.pool.putconn(self.conn)
                    logger.debug("✅ Connexion retournée au pool DB")
                else:
                    # Fermer la connexion
                    self.conn.close()
                    logger.debug("✅ Connexion à la base de données fermée")
            except Exception:
                logger.exception(
                    "❌ Erreur lors de la fermeture de la connexion: "
                )
            finally:
                self.conn = None


# Cache à mémoire partagée
class SharedCache:
    """Cache à mémoire partagée pour stocker des résultats fréquemment demandés."""

    _cache: dict[str, tuple[float, Any]] = {}
    _locks: dict[str, Any] = {}

    @classmethod
    def get(cls, key: str, max_age: int = 5):
        """
        Récupère une valeur du cache si elle existe et n'est pas expirée.

        Args:
            key: Clé du cache
            max_age: Âge maximum en secondes

        Returns:
            Valeur mise en cache ou None
        """
        current_time = time.time()

        if key in cls._cache:
            cache_time, cache_data = cls._cache[key]
            if current_time - cache_time < max_age:
                return cache_data

        return None

    @classmethod
    def set(cls, key: str, data: Any):
        """
        Stocke une valeur dans le cache.

        Args:
            key: Clé du cache
            data: Données à mettre en cache
        """
        current_time = time.time()
        cls._cache[key] = (current_time, data)

    @classmethod
    def clear(cls, prefix: str | None = None):
        """
        Efface le cache ou une partie du cache.

        Args:
            prefix: Préfixe des clés à effacer
        """
        if prefix:
            # Effacer les clés commençant par le préfixe
            keys_to_remove = [k for k in cls._cache if k.startswith(prefix)]
            for k in keys_to_remove:
                del cls._cache[k]
        else:
            # Effacer tout le cache
            cls._cache.clear()


class PortfolioModel:
    """
    Modèle pour la gestion du portefeuille.
    Fournit des méthodes pour accéder et manipuler les données du portefeuille.
    """

    def __init__(self, db_manager: DBManager | None = None):
        """
        Initialise le modèle de portefeuille.

        Args:
            db_manager: Gestionnaire de base de données préexistant (optionnel)
        """
        self.db = db_manager or DBManager()

        logger.info("✅ PortfolioModel initialisé")

    def get_latest_balances(self) -> list[AssetBalance]:
        """
        Récupère les derniers soldes du portefeuille.
        Utilise un cache partagé pour améliorer les performances.

        Returns:
            Liste des soldes par actif
        """
        # Vérifier le cache partagé
        cache_key = "latest_balances"
        cached_data = SharedCache.get(cache_key, max_age=5)

        if cached_data:
            return cached_data

        # Si pas en cache, exécuter la requête
        query = """
        WITH latest_balances AS (
            SELECT
                asset,
                MAX(timestamp) as latest_timestamp
            FROM
                portfolio_balances
            GROUP BY
                asset
        )
        SELECT
            pb.asset,
            pb.free,
            pb.locked,
            pb.total,
            pb.value_usdc
        FROM
            portfolio_balances pb
        JOIN
            latest_balances lb ON pb.asset = lb.asset AND pb.timestamp = lb.latest_timestamp
        ORDER BY
            pb.value_usdc DESC NULLS LAST,
            pb.total DESC
        """

        result = self.db.execute_query(query, fetch_all=True, retry=2)

        if not result or not isinstance(result, list):
            return []

        # Convertir en objets AssetBalance
        balances = []
        for row in result:
            balance = AssetBalance(
                asset=row["asset"],
                free=float(row["free"]),
                locked=float(row["locked"]),
                total=float(row["total"]),
                value_usdc=(
                    float(row["value_usdc"]) if row["value_usdc"] is not None else None
                ),
            )
            balances.append(balance)

        # Mettre en cache
        SharedCache.set(cache_key, balances)

        return balances

    def get_portfolio_summary(self) -> PortfolioSummary:
        """
        Récupère un résumé du portefeuille.
        Utilise un cache partagé pour améliorer les performances.

        Returns:
            Résumé du portefeuille
        """
        # Vérifier le cache
        cache_key = "portfolio_summary"
        cached_data = SharedCache.get(cache_key, max_age=5)

        if cached_data:
            return cached_data

        try:
            # Récupérer les soldes récents
            balances = self.get_latest_balances()

            if not balances:
                balances = []

            # Calculer la valeur totale
            total_value = sum(b.value_usdc or 0 for b in balances)

            # Obtenir les performances
            performance_24h = (
                self._calculate_performance(days=1)
                if hasattr(self, "_calculate_performance")
                else None
            )
            performance_7d = (
                self._calculate_performance(days=7)
                if hasattr(self, "_calculate_performance")
                else None
            )

            # Compter les trades actifs
            active_trades = (
                self._count_active_trades()
                if hasattr(self, "_count_active_trades")
                else 0
            )

            # Créer le résumé
            summary = PortfolioSummary(
                balances=balances,
                total_value=total_value,
                performance_24h=performance_24h,
                performance_7d=performance_7d,
                active_trades=active_trades,
                timestamp=datetime.now(tz=timezone.utc),
            )

            # Mettre en cache
            SharedCache.set(cache_key, summary)

            return summary

        except Exception:
            logger.exception(
                "❌ Erreur lors de la récupération du résumé du portefeuille: "
            )
            import traceback

            logger.exception(traceback.format_exc())

            # Retourner un résumé vide en cas d'erreur
            return PortfolioSummary(
                balances=[],
                total_value=0,
                active_trades=0,
                timestamp=datetime.now(tz=timezone.utc))

    def update_balances(self, balances: list[AssetBalance | dict]) -> bool:
        """
        Met à jour les soldes du portefeuille.

        Args:
            balances: Liste des nouveaux soldes (AssetBalance ou dictionnaires)

        Returns:
            True si la mise à jour a réussi, False sinon
        """
        if not balances:
            return False

        now = datetime.now(tz=timezone.utc)
        values = []
        assets_from_binance = set()

        # Préparer les actifs présents sur Binance
        for balance in balances:
            # Accepter les objets AssetBalance ou les dictionnaires
            if isinstance(balance, dict):
                asset = balance.get("asset")
                free = float(balance.get("free", 0))
                locked = float(balance.get("locked", 0))
                total = float(balance.get("total", free + locked))
                value_usdc = balance.get("value_usdc")
                if value_usdc is not None:
                    value_usdc = float(value_usdc)
            else:
                asset = balance.asset
                free = balance.free
                locked = balance.locked
                total = balance.total
                value_usdc = balance.value_usdc

            assets_from_binance.add(asset)
            values.append((asset, free, locked, total, value_usdc, now))

        # NOUVEAU: Récupérer tous les actifs connus (ayant eu une balance > 0
        # récemment)
        known_assets_query = """
        SELECT DISTINCT asset
        FROM portfolio_balances
        WHERE total > 0
        AND timestamp > NOW() - INTERVAL '7 days'
        """
        known_assets_result = self.db.execute_query(
            known_assets_query, fetch_all=True)

        if known_assets_result and isinstance(known_assets_result, list):
            # Pour chaque actif connu mais absent de Binance, ajouter une
            # entrée à 0
            for row in known_assets_result:
                asset = row["asset"]
                if asset not in assets_from_binance:
                    # Ajouter cet actif avec une valeur de 0
                    values.append((asset, 0.0, 0.0, 0.0, 0.0, now))
                    logger.info(f"📉 Actif {asset} absent de Binance, mis à 0")

        # Insérer toutes les valeurs (présentes et à 0)
        success = self.db.execute_batch(
            "INSERT INTO portfolio_balances (asset, free, locked, total, value_usdc, timestamp) VALUES %s",
            values,
        )

        if success:
            logger.info(
                f"✅ Soldes mis à jour: {len(assets_from_binance)} actifs actifs, {len(values) - len(assets_from_binance)} mis à 0"
            )
            # Invalider le cache
            SharedCache.clear("latest_balances")
            SharedCache.clear("portfolio_summary")

        # Nettoyer les anciens enregistrements (garder seulement les 24
        # dernières heures)
        self._cleanup_old_records()

        return success

    def get_trades_history(
        self,
        limit: int = 50,
        offset: int = 0,
        symbol: str | None = None,
        strategy: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Récupère l'historique des trades.

        Args:
            limit: Nombre maximum de résultats
            offset: Décalage pour la pagination
            symbol: Filtrer par symbole
            strategy: Filtrer par stratégie
            start_date: Date de début
            end_date: Date de fin

        Returns:
            Liste des trades
        """
        # Construire la requête avec les filtres
        query = """
        SELECT
            tc.id,
            tc.symbol,
            tc.strategy,
            tc.status,
            tc.entry_price,
            tc.exit_price,
            tc.quantity,
            tc.profit_loss,
            tc.profit_loss_percent,
            tc.created_at,
            tc.completed_at,
            tc.demo
        FROM
            trade_cycles tc
        WHERE 1=1
        """

        params: list[Any] = []

        # Ajouter les filtres si spécifiés
        if symbol:
            query += " AND tc.symbol = %s"
            params.append(symbol)

        if strategy:
            query += " AND tc.strategy = %s"
            params.append(strategy)

        if start_date:
            query += " AND tc.created_at >= %s"
            params.append(start_date)

        if end_date:
            query += " AND tc.created_at <= %s"
            params.append(end_date)

        # Ajouter l'ordre et la pagination
        query += " ORDER BY tc.created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        # Exécuter la requête
        result = self.db.execute_query(query, tuple(params), fetch_all=True)

        return result if isinstance(result, list) else []

    def get_performance_stats(
        self, period: str = "daily", limit: int = 30
    ) -> list[dict[str, Any]]:
        """
        Récupère les statistiques de performance.

        Args:
            period: Période ('daily', 'weekly', 'monthly')
            limit: Nombre de périodes à récupérer

        Returns:
            Liste des statistiques de performance
        """
        # Vérifier si c'est en cache
        cache_key = f"performance_stats_{period}_{limit}"
        cached_data = SharedCache.get(
            cache_key, max_age=30
        )  # Cache plus long pour les stats

        if cached_data:
            return cached_data

        query = """
        SELECT
            symbol,
            strategy,
            period,
            start_date,
            end_date,
            total_trades,
            winning_trades,
            losing_trades,
            break_even_trades,
            profit_loss,
            profit_loss_percent
        FROM
            performance_stats
        WHERE
            period = %s
        ORDER BY
            start_date DESC
        LIMIT %s
        """

        result = self.db.execute_query(query, (period, limit), fetch_all=True)

        if result:
            SharedCache.set(cache_key, result)

        return result if isinstance(result, list) else []

    def get_strategy_performance(self) -> list[dict[str, Any]]:
        """
        Récupère les performances par stratégie.

        Returns:
            Liste des performances par stratégie
        """
        # Vérifier si c'est en cache
        cache_key = "strategy_performance"
        cached_data = SharedCache.get(cache_key, max_age=30)

        if cached_data:
            return cached_data

        query = """
        SELECT * FROM strategy_performance
        """

        result = self.db.execute_query(query, fetch_all=True)

        if result:
            SharedCache.set(cache_key, result)

        return result if isinstance(result, list) else []

    def get_symbol_performance(self) -> list[dict[str, Any]]:
        """
        Récupère les performances par symbole.

        Returns:
            Liste des performances par symbole
        """
        # Vérifier si c'est en cache
        cache_key = "symbol_performance"
        cached_data = SharedCache.get(cache_key, max_age=30)

        if cached_data:
            return cached_data

        query = """
        SELECT * FROM symbol_performance
        """

        result = self.db.execute_query(query, fetch_all=True)

        if result:
            SharedCache.set(cache_key, result)

        return result if isinstance(result, list) else []

    def _cleanup_old_records(self) -> None:
        """
        Nettoie les anciens enregistrements de portfolio_balances.
        Garde seulement les 24 dernières heures de données pour chaque actif,
        mais conserve toujours au moins un enregistrement par actif.
        """
        try:
            # Compter d'abord combien d'enregistrements vont être supprimés
            count_query = """
            SELECT COUNT(*) as count_to_delete
            FROM portfolio_balances
            WHERE timestamp < NOW() - INTERVAL '24 hours'
            AND (asset, timestamp) NOT IN (
                SELECT asset, MAX(timestamp)
                FROM portfolio_balances
                GROUP BY asset
            )
            """
            count_result = self.db.execute_query(count_query, fetch_one=True)
            count_to_delete = (
                count_result.get("count_to_delete", 0)
                if count_result and isinstance(count_result, dict)
                else 0
            )

            if count_to_delete > 0:
                # Maintenant supprimer les enregistrements
                cleanup_query = """
                DELETE FROM portfolio_balances
                WHERE timestamp < NOW() - INTERVAL '24 hours'
                AND (asset, timestamp) NOT IN (
                    SELECT asset, MAX(timestamp)
                    FROM portfolio_balances
                    GROUP BY asset
                )
                """

                result = self.db.execute_query(cleanup_query, commit=True)
                if result:
                    logger.info(
                        f"🧹 Nettoyage: {count_to_delete} anciens enregistrements supprimés"
                    )

        except Exception as e:
            logger.warning(
                f"⚠️ Erreur lors du nettoyage des anciens enregistrements: {e}"
            )

    def get_strategy_configs(
            self, name: str | None = None) -> list[dict[str, Any]]:
        """
        Récupère les configurations de stratégie.

        Args:
            name: Filtrer par nom de stratégie

        Returns:
            Liste des configurations de stratégie
        """
        query = """
        SELECT name, mode, symbols, params, max_simultaneous_trades, enabled
        FROM strategy_configs
        WHERE 1=1
        """
        params: list[Any] = []

        if name:
            query += " AND name = %s"
            params.append(name)

        result = self.db.execute_query(query, tuple(params), fetch_all=True)
        return result if isinstance(result, list) else []

    def close(self) -> None:
        """
        Ferme la connexion à la base de données.
        """
        if self.db:
            self.db.close()
