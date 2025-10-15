"""
Module de gestion de la base de données pour le Signal Aggregator.
Gère le stockage des signaux validés et l'historique de validation.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Gestionnaire de la base de données pour le stockage des signaux."""

    def __init__(self, db_connection):
        """
        Initialise le gestionnaire de base de données.

        Args:
            db_connection: Connexion à la base de données
        """
        self.db_connection = db_connection

        # Statistiques de stockage
        self.stats = {
            "signals_stored": 0,
            "storage_errors": 0,
            "last_storage_time": None,
        }

    def store_validated_signal(self, validated_signal: Dict[str, Any]) -> Optional[int]:
        """
        Stocke un signal validé en base de données.

        Args:
            validated_signal: Signal validé à stocker

        Returns:
            ID du signal stocké ou None en cas d'erreur
        """
        try:
            with self.db_connection.cursor() as cursor:
                # Préparation des données
                signal_data = self._prepare_signal_data(validated_signal)

                # Insertion du signal
                insert_query = """
                    INSERT INTO trading_signals 
                    (strategy, symbol, side, timestamp, confidence, price, metadata, processed)
                    VALUES (%(strategy)s, %(symbol)s, %(side)s, %(timestamp)s, 
                           %(confidence)s, %(price)s, %(metadata)s, %(processed)s)
                    RETURNING id
                """

                cursor.execute(insert_query, signal_data)
                signal_id = cursor.fetchone()[0]

                # Commit de la transaction
                self.db_connection.commit()

                # Mise à jour des statistiques
                self.stats["signals_stored"] += 1
                self.stats["last_storage_time"] = datetime.utcnow()

                logger.debug(f"Signal stocké en DB avec ID: {signal_id}")
                return signal_id

        except Exception as e:
            logger.error(f"Erreur stockage signal en DB: {e}")
            self.stats["storage_errors"] += 1

            # Rollback en cas d'erreur
            try:
                self.db_connection.rollback()
            except Exception as rollback_error:
                logger.error(f"Erreur rollback: {rollback_error}")

            return None

    def _prepare_signal_data(self, validated_signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prépare les données du signal pour l'insertion en DB.

        Args:
            validated_signal: Signal validé

        Returns:
            Dict des données préparées pour l'insertion
        """
        try:
            # Extraction du prix depuis les métadonnées ou contexte
            price = self._extract_price(validated_signal)

            # Préparation des métadonnées (tout sauf les champs de base)
            original_metadata = validated_signal.get("metadata", {})

            metadata = {
                "timeframe": validated_signal.get("timeframe"),
                "strength": validated_signal.get("strength"),
                "reason": validated_signal.get("reason"),
                "validation_score": validated_signal.get("validation_score"),
                "raw_validation_score": validated_signal.get("raw_validation_score"),
                "weighted_validation_score": validated_signal.get(
                    "weighted_validation_score"
                ),
                "validators_passed": validated_signal.get("validators_passed"),
                "total_validators": validated_signal.get("total_validators"),
                "validation_strength": validated_signal.get("validation_strength"),
                "pass_rate": validated_signal.get("pass_rate"),
                "aggregator_confidence": validated_signal.get("aggregator_confidence"),
                "final_score": validated_signal.get("final_score"),
                "category_scores": validated_signal.get("category_scores"),
                "validation_details": validated_signal.get("validation_details"),
                "validation_timestamp": validated_signal.get("validation_timestamp"),
                "original_metadata": original_metadata,
                # Promouvoir les infos de consensus importantes au niveau supérieur
                "strategy_count": original_metadata.get("strategy_count"),
                "has_consensus": original_metadata.get("has_consensus"),
                "is_individual": original_metadata.get("is_individual"),
                "consensus_group": original_metadata.get("consensus_group"),
                "resolution_strategy": original_metadata.get("resolution_strategy"),
                "symbol_analysis": original_metadata.get("symbol_analysis"),
            }

            # Suppression des valeurs None pour optimiser le JSON
            metadata = {k: v for k, v in metadata.items() if v is not None}

            signal_data = {
                "strategy": validated_signal["strategy"],
                "symbol": validated_signal["symbol"],
                "side": validated_signal["side"],
                "timestamp": self._parse_timestamp(
                    validated_signal.get("timestamp") or ""
                ),
                "confidence": float(
                    validated_signal.get(
                        "aggregator_confidence", validated_signal.get("confidence", 0)
                    )
                ),
                "price": price,
                "metadata": json.dumps(metadata),
                "processed": False,  # Signal pas encore traité par le coordinator
            }

            return signal_data

        except Exception as e:
            logger.error(f"Erreur préparation données signal: {e}")
            raise

    def _extract_price(self, validated_signal: Dict[str, Any]) -> float:
        """
        Extrait le prix du signal depuis les métadonnées ou contexte.

        Args:
            validated_signal: Signal validé

        Returns:
            Prix du signal ou 0.0 si non trouvé
        """
        try:
            # D'abord vérifier si le prix est déjà dans le signal
            if "price" in validated_signal and validated_signal["price"] is not None:
                try:
                    price = float(validated_signal["price"])
                    if price > 0:  # Prix valide
                        return price
                except (ValueError, TypeError):
                    pass

            # Essayer de récupérer le prix depuis différentes sources
            metadata = validated_signal.get("metadata", {})

            # Prix depuis les métadonnées originales
            if isinstance(metadata, dict) and "current_price" in metadata:
                return float(metadata["current_price"])

            # Prix depuis le contexte (si disponible)
            if "context" in validated_signal:
                context = validated_signal["context"]
                if isinstance(context, dict):
                    ohlcv_data = context.get("ohlcv_data", [])
                    if ohlcv_data and len(ohlcv_data) > 0:
                        latest_candle = ohlcv_data[-1]
                        return float(latest_candle.get("close", 0))

            # Prix depuis les indicateurs
            indicators = validated_signal.get("indicators", {})
            if isinstance(indicators, dict) and "close" in indicators:
                return float(indicators["close"])

            # Valeur par défaut
            logger.warning(
                f"Prix non trouvé pour signal {validated_signal.get('symbol')}, utilisation de 0.0"
            )
            return 0.0

        except Exception as e:
            logger.error(f"Erreur extraction prix: {e}")
            return 0.0

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """
        Parse un timestamp string en objet datetime.

        Args:
            timestamp_str: Timestamp au format string

        Returns:
            Objet datetime
        """
        try:
            if not timestamp_str:
                return datetime.utcnow()

            # Essayer différents formats
            formats = [
                "%Y-%m-%dT%H:%M:%S.%f",  # ISO avec microsecondes
                "%Y-%m-%dT%H:%M:%S",  # ISO sans microsecondes
                "%Y-%m-%d %H:%M:%S.%f",  # Format standard avec microsecondes
                "%Y-%m-%d %H:%M:%S",  # Format standard sans microsecondes
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue

            # Si aucun format ne marche, utiliser le timestamp actuel
            logger.warning(
                f"Format timestamp non reconnu: {timestamp_str}, utilisation timestamp actuel"
            )
            return datetime.utcnow()

        except Exception as e:
            logger.error(f"Erreur parsing timestamp: {e}")
            return datetime.utcnow()

    def store_multiple_signals(
        self, validated_signals: List[Dict[str, Any]]
    ) -> List[Optional[int]]:
        """
        Stocke plusieurs signaux en batch pour optimiser les performances.

        Args:
            validated_signals: Liste des signaux validés

        Returns:
            Liste des IDs des signaux stockés
        """
        signal_ids: List[Optional[int]] = []

        if not validated_signals:
            return signal_ids

        try:
            with self.db_connection.cursor() as cursor:
                insert_query = """
                    INSERT INTO trading_signals 
                    (strategy, symbol, side, timestamp, confidence, price, metadata, processed)
                    VALUES (%(strategy)s, %(symbol)s, %(side)s, %(timestamp)s, 
                           %(confidence)s, %(price)s, %(metadata)s, %(processed)s)
                    RETURNING id
                """

                for signal in validated_signals:
                    try:
                        signal_data = self._prepare_signal_data(signal)
                        cursor.execute(insert_query, signal_data)
                        signal_id = cursor.fetchone()[0]
                        signal_ids.append(signal_id)

                    except Exception as e:
                        logger.error(f"Erreur stockage signal individuel: {e}")
                        signal_ids.append(None)
                        self.stats["storage_errors"] += 1

                # Commit de toutes les insertions
                self.db_connection.commit()

                # Mise à jour des statistiques
                successful_stores = len([sid for sid in signal_ids if sid is not None])
                self.stats["signals_stored"] += successful_stores
                self.stats["last_storage_time"] = datetime.utcnow()

                logger.info(
                    f"Batch de {successful_stores}/{len(validated_signals)} signaux stocké en DB"
                )

        except Exception as e:
            logger.error(f"Erreur stockage batch signaux: {e}")
            self.stats["storage_errors"] += len(validated_signals)

            # Rollback en cas d'erreur
            try:
                self.db_connection.rollback()
            except Exception as rollback_error:
                logger.error(f"Erreur rollback batch: {rollback_error}")

            # Retourner une liste de None de la même taille
            signal_ids = [None] * len(validated_signals)

        return signal_ids

    def get_recent_signals(
        self,
        limit: int = 50,
        strategy: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Récupère les signaux récents depuis la base de données.

        Args:
            limit: Nombre maximum de signaux à récupérer
            strategy: Filtrer par stratégie (optionnel)
            symbol: Filtrer par symbole (optionnel)

        Returns:
            Liste des signaux récents
        """
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Construction de la requête avec filtres optionnels
                query = "SELECT * FROM trading_signals WHERE 1=1"
                params: List[Any] = []

                if strategy:
                    query += " AND strategy = %s"
                    params.append(strategy)

                if symbol:
                    query += " AND symbol = %s"
                    params.append(symbol)

                query += " ORDER BY timestamp DESC LIMIT %s"
                params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()

                # Conversion en liste de dictionnaires
                signals = []
                for row in rows:
                    signal_dict = dict(row)

                    # Parse des métadonnées JSON
                    if signal_dict.get("metadata"):
                        try:
                            signal_dict["metadata"] = json.loads(
                                signal_dict["metadata"]
                            )
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Erreur parsing métadonnées pour signal {signal_dict['id']}"
                            )

                    signals.append(signal_dict)

                return signals

        except Exception as e:
            logger.error(f"Erreur récupération signaux récents: {e}")
            return []

    def mark_signal_as_processed(self, signal_id: int) -> bool:
        """
        Marque un signal comme traité par le coordinator.

        Args:
            signal_id: ID du signal à marquer

        Returns:
            True si le marquage a réussi, False sinon
        """
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE trading_signals SET processed = true WHERE id = %s",
                    (signal_id,),
                )

                self.db_connection.commit()

                if cursor.rowcount > 0:
                    logger.debug(f"Signal {signal_id} marqué comme traité")
                    return True
                else:
                    logger.warning(f"Signal {signal_id} non trouvé pour marquage")
                    return False

        except Exception as e:
            logger.error(f"Erreur marquage signal {signal_id}: {e}")
            try:
                self.db_connection.rollback()
            except Exception as rollback_error:
                logger.error(f"Erreur rollback marquage: {rollback_error}")
            return False

    def get_validation_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Récupère les statistiques de validation sur une période donnée.

        Args:
            hours: Nombre d'heures à analyser

        Returns:
            Dictionnaire des statistiques
        """
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Statistiques générales
                cursor.execute(
                    """
                    SELECT 
                        COUNT(*) as total_signals,
                        COUNT(CASE WHEN processed = true THEN 1 END) as processed_signals,
                        AVG(confidence) as avg_confidence,
                        COUNT(DISTINCT strategy) as unique_strategies,
                        COUNT(DISTINCT symbol) as unique_symbols
                    FROM trading_signals 
                    WHERE timestamp >= NOW() - INTERVAL '%s hours'
                """,
                    (hours,),
                )

                general_stats = cursor.fetchone()

                # Statistiques par stratégie
                cursor.execute(
                    """
                    SELECT 
                        strategy,
                        COUNT(*) as signal_count,
                        AVG(confidence) as avg_confidence
                    FROM trading_signals 
                    WHERE timestamp >= NOW() - INTERVAL '%s hours'
                    GROUP BY strategy
                    ORDER BY signal_count DESC
                """,
                    (hours,),
                )

                strategy_stats = cursor.fetchall()

                # Statistiques par symbole
                cursor.execute(
                    """
                    SELECT 
                        symbol,
                        COUNT(*) as signal_count,
                        COUNT(CASE WHEN side = 'BUY' THEN 1 END) as buy_signals,
                        COUNT(CASE WHEN side = 'SELL' THEN 1 END) as sell_signals
                    FROM trading_signals 
                    WHERE timestamp >= NOW() - INTERVAL '%s hours'
                    GROUP BY symbol
                    ORDER BY signal_count DESC
                """,
                    (hours,),
                )

                symbol_stats = cursor.fetchall()

                return {
                    "period_hours": hours,
                    "general": dict(general_stats) if general_stats else {},
                    "by_strategy": [dict(row) for row in strategy_stats],
                    "by_symbol": [dict(row) for row in symbol_stats],
                    "storage_stats": self.get_storage_stats(),
                }

        except Exception as e:
            logger.error(f"Erreur récupération statistiques validation: {e}")
            return {"error": str(e)}

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Récupère les statistiques de stockage.

        Returns:
            Dictionnaire des statistiques de stockage
        """
        return {
            "signals_stored": self.stats["signals_stored"],
            "storage_errors": self.stats["storage_errors"],
            "last_storage_time": (
                self.stats["last_storage_time"].isoformat()
                if self.stats["last_storage_time"]
                else None
            ),
            "success_rate": (
                self.stats["signals_stored"]
                / (self.stats["signals_stored"] + self.stats["storage_errors"])
                if (self.stats["signals_stored"] + self.stats["storage_errors"]) > 0
                else 1.0
            ),
        }

    def cleanup_old_signals(self, days: int = 30) -> int:
        """
        Nettoie les anciens signaux de la base de données.

        Args:
            days: Nombre de jours à conserver

        Returns:
            Nombre de signaux supprimés
        """
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM trading_signals 
                    WHERE timestamp < NOW() - INTERVAL '%s days'
                    AND processed = true
                """,
                    (days,),
                )

                deleted_count = cursor.rowcount
                self.db_connection.commit()

                logger.info(f"Nettoyage DB: {deleted_count} anciens signaux supprimés")
                return deleted_count

        except Exception as e:
            logger.error(f"Erreur nettoyage anciens signaux: {e}")
            try:
                self.db_connection.rollback()
            except Exception as rollback_error:
                logger.error(f"Erreur rollback nettoyage: {rollback_error}")
            return 0
