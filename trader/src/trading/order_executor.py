"""
Exécuteur d'ordres simplifié pour le trader.
Logique simple : reçoit ordre du coordinator → exécute sur Binance → stocke historique.
"""

import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from shared.src.db_pool import transaction
from shared.src.enums import OrderSide, OrderStatus
from shared.src.schemas import TradeExecution, TradeOrder
from trader.src.exchange.binance_executor import BinanceExecutor
from trader.src.trading.cycle_manager import CycleManager

logger = logging.getLogger(__name__)


class OrderExecutor:
    """
    Exécuteur d'ordres simplifié.
    Pas de cycles, pas de statuts complexes : juste exécuter et stocker.
    """

    def __init__(self, binance_executor: BinanceExecutor):
        """
        Initialise l'exécuteur d'ordres.

        Args:
            binance_executor: Exécuteur Binance
        """
        self.binance_executor = binance_executor

        # Historique des ordres exécutés
        self.order_history: list[dict[str, Any]] = []

        # Gestionnaire de cycles pour le tracking des P&L
        self.cycle_manager = CycleManager()

        # Cache de déduplication: {hash: (timestamp, order_id)}
        self.dedup_cache: dict[str, tuple[float, str]] = {}
        self.dedup_window_seconds = 10  # Fenêtre de déduplication en secondes

        logger.info("✅ OrderExecutor initialisé (logique simplifiée)")

    def _get_order_hash(self, order_data: dict[str, Any]) -> str:
        """
        Génère un hash unique pour un ordre basé sur ses caractéristiques.

        Args:
            order_data: Données de l'ordre

        Returns:
            Hash de l'ordre
        """
        # Créer une représentation canonique de l'ordre
        key_parts = [
            order_data.get("symbol", ""),
            order_data.get("side", ""),
            str(order_data.get("quantity", 0)),
            str(order_data.get("strategy", "Manual")),
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _clean_dedup_cache(self) -> None:
        """Nettoie le cache de déduplication des entrées expirées."""
        current_time = time.time()
        expired_keys = [
            k
            for k, (timestamp, _) in self.dedup_cache.items()
            if current_time - timestamp > self.dedup_window_seconds
        ]
        for key in expired_keys:
            del self.dedup_cache[key]

    def _is_duplicate_order(self, order_data: dict[str, Any]) -> str | None:
        """
        Vérifie si un ordre est un duplicata récent.

        Args:
            order_data: Données de l'ordre

        Returns:
            order_id du duplicata si trouvé, None sinon
        """
        # Nettoyer le cache
        self._clean_dedup_cache()

        # Générer le hash de l'ordre
        order_hash = self._get_order_hash(order_data)

        # Vérifier si l'ordre existe déjà dans le cache
        if order_hash in self.dedup_cache:
            timestamp, existing_order_id = self.dedup_cache[order_hash]
            time_diff = time.time() - timestamp

            if time_diff <= self.dedup_window_seconds:
                logger.warning(
                    f"⚠️ Ordre dupliqué détecté (créé il y a {time_diff:.1f}s): {existing_order_id}"
                )
                return existing_order_id

        return None

    def execute_order(self, order_data: dict[str, Any]) -> str | None:
        """
        Exécute un ordre simple.

        Args:
            order_data: Données de l'ordre
            Format: {
                "symbol": "BTCUSDC",
                "side": "BUY" ou "SELL",
                "quantity": 0.001,
                "price": 50000 (optionnel, sinon MARKET),
                "strategy": "Manual",
                "timestamp": 1234567890
            }

        Returns:
            ID de l'ordre exécuté ou None si échec
        """
        try:
            # Vérifier les duplicatas
            duplicate_order_id = self._is_duplicate_order(order_data)
            if duplicate_order_id:
                logger.info(
                    f"🔄 Ordre dupliqué ignoré, retour de l'ID existant: {duplicate_order_id}"
                )
                return duplicate_order_id

            # Générer un ID unique pour l'ordre
            order_id = f"order_{uuid.uuid4().hex[:16]}"

            # Valider les données
            required_fields = ["symbol", "side", "quantity"]
            for field in required_fields:
                if field not in order_data:
                    logger.error(f"❌ Champ manquant dans l'ordre: {field}")
                    return None

            symbol = order_data["symbol"]
            side = OrderSide(order_data["side"])
            quantity = float(order_data["quantity"])
            price = (float(order_data.get("price", 0))
                     if order_data.get("price") else None)
            strategy = order_data.get("strategy", "Manual")

            # Ajouter l'ordre au cache de déduplication AVANT l'exécution
            order_hash = self._get_order_hash(order_data)
            self.dedup_cache[order_hash] = (time.time(), order_id)

            side_str = side.value if hasattr(side, "value") else str(side)
            logger.info(
                f"📤 Exécution ordre: {side_str} {quantity} {symbol} @ {price or 'MARKET'}"
            )

            # Créer l'ordre TradeOrder
            trade_order = TradeOrder(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                client_order_id=order_id,
                strategy=strategy,
                leverage=1,  # Mode SPOT, pas de levier
                demo=self.binance_executor.demo_mode,
            )

            # Exécuter sur Binance
            execution = self.binance_executor.execute_order(trade_order)

            if not execution or not execution.order_id:
                logger.error(f"❌ Échec exécution ordre {order_id}")
                return None

            # Vérifier que l'ordre a été exécuté
            if execution.status not in [
                OrderStatus.FILLED,
                OrderStatus.PARTIALLY_FILLED,
            ]:
                logger.error(
                    f"❌ Ordre {order_id} non exécuté: {execution.status}")
                return None

            # Sauvegarder en base de données
            self._save_execution(execution)

            # Ajouter à l'historique
            order_record = {
                "id": order_id,
                "binance_order_id": execution.order_id,
                "symbol": symbol,
                "side": side_str,
                "quantity": quantity,
                "price": execution.price,
                "strategy": strategy,
                "status": (
                    execution.status.value
                    if hasattr(execution.status, "value")
                    else str(execution.status)
                ),
                "timestamp": datetime.now(timezone.utc),
                "demo": self.binance_executor.demo_mode,
            }
            self.order_history.append(order_record)

            status_str = (
                execution.status.value
                if hasattr(execution.status, "value")
                else str(execution.status)
            )
            logger.info(
                f"✅ Ordre exécuté: {order_id} (Binance: {execution.order_id}) - {status_str}"
            )

            # Traiter le trade pour les cycles en arrière-plan (après avoir
            # retourné la réponse)
            cycle_id = None
            try:
                cycle_id = self._process_trade_for_cycles(execution, strategy)
                if cycle_id:
                    # Mettre à jour l'exécution avec le cycle_id
                    self._update_execution_with_cycle_id(
                        execution.order_id, cycle_id)
            except Exception:
                logger.exception("❌ Erreur processing cycle")
                # On ne propage pas l'erreur

            return order_id

        except Exception:
            logger.exception("❌ Erreur exécution ordre")
            return None

    def _save_execution(self, execution: TradeExecution) -> None:
        """
        Sauvegarde une exécution en base de données.

        Args:
            execution: Exécution à sauvegarder
        """
        try:
            with transaction() as cursor:
                cursor.execute(
                    """
                    INSERT INTO trade_executions
                    (order_id, symbol, side, status, price, quantity, quote_quantity,
                     fee, fee_asset, role, timestamp, demo)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (order_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        price = EXCLUDED.price,
                        quantity = EXCLUDED.quantity,
                        quote_quantity = EXCLUDED.quote_quantity,
                        fee = EXCLUDED.fee,
                        timestamp = EXCLUDED.timestamp
                """,
                    (execution.order_id,
                     execution.symbol,
                     (execution.side.value if hasattr(
                         execution.side,
                         "value") else str(
                         execution.side)),
                        (execution.status.value if hasattr(
                            execution.status,
                            "value") else str(
                            execution.status)),
                        execution.price,
                        execution.quantity,
                        execution.quote_quantity,
                        execution.fee,
                        execution.fee_asset,
                        (execution.role.value if execution.role and hasattr(
                            execution.role,
                            "value") else (
                            str(
                                execution.role) if execution.role else None)),
                        execution.timestamp,
                        execution.demo,
                     ),
                )

            logger.debug(f"✅ Exécution {execution.order_id} sauvegardée")

        except Exception:
            logger.exception("❌ Erreur sauvegarde exécution")

    def get_order_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        Récupère l'historique des ordres.

        Args:
            limit: Nombre maximum d'ordres à retourner

        Returns:
            Liste des ordres récents
        """
        return self.order_history[-limit:]

    def get_order_status(self, order_id: str) -> dict[str, Any] | None:
        """
        Récupère le statut d'un ordre.

        Args:
            order_id: ID de l'ordre

        Returns:
            Statut de l'ordre ou None
        """
        for order in self.order_history:
            if order["id"] == order_id:
                return order
        return None

    def _process_trade_for_cycles(
        self, execution: TradeExecution, strategy: str
    ) -> str | None:
        """
        Traite un trade exécuté pour mettre à jour les cycles.
        Cette méthode est appelée pour ne pas bloquer l'exécution.

        Args:
            execution: Exécution du trade
            strategy: Stratégie utilisée

        Returns:
            ID du cycle créé/mis à jour ou None
        """
        try:
            trade_data = {
                "symbol": execution.symbol,
                "side": execution.side,
                "price": execution.price,
                "quantity": execution.quantity,
                "order_id": execution.order_id,
                "timestamp": execution.timestamp,
                "strategy": strategy,
            }

            return self.cycle_manager.process_trade_execution(trade_data)

        except Exception:
            logger.exception("❌ Erreur traitement cycle")
            # On ne propage pas l'erreur pour ne pas impacter le trading
            return None

    def get_stats(self) -> dict[str, Any]:
        """
        Récupère les statistiques de l'exécuteur.

        Returns:
            Statistiques
        """
        total_orders = len(self.order_history)
        successful_orders = sum(
            1
            for order in self.order_history
            if order["status"] in ["FILLED", "PARTIALLY_FILLED"]
        )

        return {
            "total_orders": total_orders,
            "successful_orders": successful_orders,
            "success_rate": (
                (successful_orders / total_orders * 100) if total_orders > 0 else 0
            ),
            "demo_mode": self.binance_executor.demo_mode,
            "last_order_time": (
                str(self.order_history[-1]["timestamp"]) if self.order_history else None
            ),
        }

    def _update_execution_with_cycle_id(
            self, order_id: str, cycle_id: str) -> None:
        """
        Met à jour une exécution existante avec son cycle_id.

        Args:
            order_id: ID de l'ordre à mettre à jour
            cycle_id: ID du cycle à associer
        """
        try:
            with transaction() as cursor:
                cursor.execute(
                    """
                    UPDATE trade_executions
                    SET cycle_id = %s, updated_at = NOW()
                    WHERE order_id = %s
                """,
                    (cycle_id, order_id),
                )

                if cursor.rowcount > 0:
                    logger.debug(
                        f"✅ Exécution {order_id} liée au cycle {cycle_id}")
                else:
                    logger.warning(
                        f"⚠️ Aucune exécution trouvée pour order_id {order_id}"
                    )

        except Exception:
            logger.exception("❌ Erreur mise à jour cycle_id")
