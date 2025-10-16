"""
Gestionnaire de cycles de trading pour ROOT.
Gère la création et le suivi des cycles BUY->SELL sans impacter l'exécution des trades.
"""

import json
import logging
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from shared.src.db_pool import transaction
from shared.src.enums import OrderSide
from shared.src.redis_client import RedisClient

logger = logging.getLogger(__name__)


class CycleManager:
    """
    Gère les cycles de trading (BUY->SELL) pour le calcul des P&L.
    Fonctionne de manière indépendante sans influencer l'exécution des trades.
    """

    def __init__(self):
        """Initialise le gestionnaire de cycles."""
        self.redis_client = RedisClient()
        self.fee_rate = Decimal("0.001")  # 0.1% de frais Binance par trade
        logger.info("✅ CycleManager initialisé")

    def process_trade_execution(self, trade_data: dict[str, Any]) -> str | None:
        """
        Traite une exécution de trade pour mettre à jour les cycles.
        Cette méthode est appelée APRÈS l'exécution réussie d'un trade.

        Args:
            trade_data: Données du trade exécuté
                - symbol: Symbole tradé
                - side: BUY ou SELL
                - price: Prix d'exécution
                - quantity: Quantité exécutée
                - order_id: ID de l'ordre
                - timestamp: Timestamp d'exécution
                - strategy: Stratégie utilisée (optionnel)

        Returns:
            ID du cycle créé/mis à jour ou None si erreur
        """
        try:
            symbol = trade_data.get("symbol")
            side = trade_data.get("side")
            price = Decimal(str(trade_data.get("price", 0)))
            quantity = Decimal(str(trade_data.get("quantity", 0)))
            order_id = trade_data.get("order_id")
            strategy = trade_data.get("strategy", "Unknown")

            if not all([symbol, side, price > 0, quantity > 0]):
                logger.warning(f"⚠️ Données de trade incomplètes: {trade_data}")
                return None

            # Convertir side en OrderSide si nécessaire
            if isinstance(side, str):
                side = OrderSide(side.upper())

            if side == OrderSide.BUY:
                if symbol and order_id:
                    return self._handle_buy_trade(
                        symbol=symbol,
                        price=price,
                        quantity=quantity,
                        order_id=order_id,
                        strategy=strategy,
                    )
                logger.warning(
                    f"⚠️ Symbol ou order_id manquant pour BUY: symbol={symbol}, order_id={order_id}"
                )
                return None
            if side == OrderSide.SELL:
                if symbol and order_id:
                    return self._handle_sell_trade(
                        symbol=symbol,
                        price=price,
                        quantity=quantity,
                        order_id=order_id,
                        strategy=strategy,
                    )
                logger.warning(
                    f"⚠️ Symbol ou order_id manquant pour SELL: symbol={symbol}, order_id={order_id}"
                )
                return None

        except Exception:
            logger.exception("❌ Erreur dans process_trade_execution")
            # On ne propage pas l'erreur pour ne pas impacter le trading
            return None

        # Fallback par défaut si aucun côté reconnu
        logger.warning(f"⚠️ Side non reconnu: {side}")
        return None

    def _handle_buy_trade(
        self,
        symbol: str,
        price: Decimal,
        quantity: Decimal,
        order_id: str,
        strategy: str,
    ) -> str | None:
        """
        Gère un trade BUY en créant ou mettant à jour un cycle.

        Args:
            symbol: Symbole tradé
            price: Prix d'achat
            quantity: Quantité achetée
            order_id: ID de l'ordre
            strategy: Stratégie utilisée

        Returns:
            ID du cycle créé/mis à jour
        """
        try:
            # Chercher un cycle actif pour ce symbole
            active_cycle = self._get_active_cycle(symbol)

            if active_cycle:
                # Mettre à jour le cycle existant (ajout de position)
                self._update_cycle_for_buy(
                    cycle_id=active_cycle["id"],
                    new_price=price,
                    new_quantity=quantity,
                    order_id=order_id,
                )
                logger.info(f"📈 Cycle mis à jour pour {symbol}: +{quantity} @ {price}")
                return active_cycle["id"]
            # Créer un nouveau cycle
            cycle_id = self._create_new_cycle(
                symbol=symbol,
                price=price,
                quantity=quantity,
                order_id=order_id,
                strategy=strategy,
            )
            logger.info(f"🆕 Nouveau cycle créé pour {symbol}: {quantity} @ {price}")
            return cycle_id

        except Exception:
            logger.exception("❌ Erreur dans _handle_buy_trade")
            return None

    def _handle_sell_trade(
        self,
        symbol: str,
        price: Decimal,
        quantity: Decimal,
        order_id: str,
        strategy: str,
    ) -> str | None:
        """
        Gère un trade SELL en fermant le(s) cycle(s) correspondant(s).

        Args:
            symbol: Symbole tradé
            price: Prix de vente
            quantity: Quantité vendue
            order_id: ID de l'ordre
            strategy: Stratégie utilisée

        Returns:
            ID du cycle fermé ou None
        """
        try:
            # Récupérer tous les cycles actifs pour ce symbole
            active_cycles = self._get_active_cycles_for_symbol(symbol)

            if not active_cycles:
                logger.warning(f"⚠️ Aucun cycle actif trouvé pour SELL {symbol}")
                # Créer un cycle orphelin pour tracker ce SELL
                return self._create_orphan_sell_cycle(
                    symbol=symbol,
                    price=price,
                    quantity=quantity,
                    order_id=order_id,
                    strategy=strategy,
                )

            # Fermer tous les cycles du symbole vendu (on vend toujours tout
            # sur un symbole)
            closed_cycle_id = None
            total_cycle_quantity = Decimal("0")

            for cycle in active_cycles:
                cycle_quantity = Decimal(str(cycle["quantity"]))
                total_cycle_quantity += cycle_quantity

                # Fermer complètement ce cycle
                self._close_cycle(
                    cycle_id=cycle["id"],
                    exit_price=price,
                    exit_quantity=cycle_quantity,
                    order_id=order_id,
                )
                closed_cycle_id = cycle["id"]

            # Log si différence de quantité (max 1 USDC de valeur)
            quantity_diff = abs(quantity - total_cycle_quantity)
            if quantity_diff > 0:
                # Calculer la valeur en USDC de la différence
                diff_value_usdc = quantity_diff * price
                if diff_value_usdc < Decimal("1.0"):
                    logger.debug(
                        f"📊 Dust négligeable: {quantity_diff} {symbol} (~{diff_value_usdc:.4f} USDC)"
                    )
                else:
                    logger.warning(
                        f"⚠️ Différence importante: vendu {quantity}, cycles {total_cycle_quantity}, diff {quantity_diff} (~{diff_value_usdc:.2f} USDC)"
                    )

            return closed_cycle_id

        except Exception:
            logger.exception("❌ Erreur dans _handle_sell_trade")
            return None

    def _get_active_cycle(self, symbol: str) -> dict[str, Any] | None:
        """
        Récupère le cycle actif le plus récent pour un symbole.

        Args:
            symbol: Symbole à rechercher

        Returns:
            Cycle actif ou None
        """
        try:
            with transaction() as cursor:
                cursor.execute(
                    """
                    SELECT * FROM trade_cycles
                    WHERE symbol = %s
                    AND status IN ('active_buy', 'waiting_sell')
                    ORDER BY created_at DESC
                    LIMIT 1
                """,
                    (symbol,),
                )

                result = cursor.fetchone()
                if result:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, result))
                return None
        except Exception:
            logger.exception("❌ Erreur _get_active_cycle")
            return None

    def _get_active_cycles_for_symbol(self, symbol: str) -> list[dict[str, Any]]:
        """
        Récupère tous les cycles actifs pour un symbole (ordre FIFO).

        Args:
            symbol: Symbole à rechercher

        Returns:
            Liste des cycles actifs
        """
        try:
            with transaction() as cursor:
                cursor.execute(
                    """
                    SELECT * FROM trade_cycles
                    WHERE symbol = %s
                    AND status IN ('active_buy', 'waiting_sell')
                    ORDER BY created_at ASC
                """,
                    (symbol,),
                )

                results = cursor.fetchall()
                if results:
                    columns = [desc[0] for desc in cursor.description]
                    return [dict(zip(columns, row)) for row in results]
                return []
        except Exception:
            logger.exception("❌ Erreur _get_active_cycles_for_symbol")
            return []

    def _create_new_cycle(
        self,
        symbol: str,
        price: Decimal,
        quantity: Decimal,
        order_id: str,
        strategy: str,
    ) -> str:
        """
        Crée un nouveau cycle de trading.

        Args:
            symbol: Symbole tradé
            price: Prix d'entrée
            quantity: Quantité
            order_id: ID de l'ordre d'entrée
            strategy: Stratégie utilisée

        Returns:
            ID du cycle créé
        """
        try:
            cycle_id = f"cycle_{uuid.uuid4().hex[:16]}"

            with transaction() as cursor:
                cursor.execute(
                    """
                    INSERT INTO trade_cycles (
                        id, symbol, strategy, status, side, entry_order_id,
                        entry_price, quantity, min_price, max_price,
                        created_at, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        NOW(), NOW()
                    )
                """,
                    (
                        cycle_id,
                        symbol,
                        strategy,
                        "active_buy",
                        "BUY",
                        order_id,
                        price,
                        quantity,
                        price,
                        price,
                    ),
                )

            return cycle_id
        except Exception:
            logger.exception("❌ Erreur _create_new_cycle")
            return ""

    def _update_cycle_for_buy(
        self, cycle_id: str, new_price: Decimal, new_quantity: Decimal, order_id: str
    ) -> None:
        """
        Met à jour un cycle existant avec un nouveau BUY (ajout de position).

        Args:
            cycle_id: ID du cycle à mettre à jour
            new_price: Prix du nouveau BUY
            new_quantity: Quantité du nouveau BUY
            order_id: ID de l'ordre
        """
        try:
            with transaction() as cursor:
                # Récupérer le cycle actuel
                cursor.execute(
                    """
                    SELECT entry_price, quantity, min_price, max_price, metadata
                    FROM trade_cycles WHERE id = %s
                """,
                    (cycle_id,),
                )

                result = cursor.fetchone()
                if not result:
                    logger.error(f"Cycle {cycle_id} non trouvé")
                    return

                current_price = Decimal(str(result[0]))
                current_quantity = Decimal(str(result[1]))
                min_price = Decimal(str(result[2])) if result[2] else current_price
                max_price = Decimal(str(result[3])) if result[3] else current_price

                try:
                    metadata = json.loads(result[4]) if result[4] else {}
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

                # Calculer le nouveau prix moyen pondéré
                total_value = (current_price * current_quantity) + (
                    new_price * new_quantity
                )
                new_total_quantity = current_quantity + new_quantity
                new_avg_price = total_value / new_total_quantity

                # Mettre à jour min/max
                min_price = min(min_price, new_price)
                max_price = max(max_price, new_price)

                # Ajouter l'historique des BUY dans metadata
                if "buy_history" not in metadata:
                    metadata["buy_history"] = []

                metadata["buy_history"].append(
                    {
                        "order_id": order_id,
                        "price": str(new_price),
                        "quantity": str(new_quantity),
                        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    }
                )

                # Mettre à jour le cycle
                cursor.execute(
                    """
                    UPDATE trade_cycles SET
                        entry_price = %s,
                        quantity = %s,
                        min_price = %s,
                        max_price = %s,
                        metadata = %s,
                        updated_at = NOW()
                    WHERE id = %s
                """,
                    (
                        new_avg_price,
                        new_total_quantity,
                        min_price,
                        max_price,
                        json.dumps(metadata),
                        cycle_id,
                    ),
                )

        except Exception:
            logger.exception("❌ Erreur _update_cycle_for_buy")

    def _close_cycle(
        self, cycle_id: str, exit_price: Decimal, exit_quantity: Decimal, order_id: str
    ) -> None:
        """
        Ferme complètement un cycle et calcule le P&L.

        Args:
            cycle_id: ID du cycle à fermer
            exit_price: Prix de sortie
            exit_quantity: Quantité vendue
            order_id: ID de l'ordre de sortie
        """
        try:
            with transaction() as cursor:
                # Récupérer les données du cycle incluant le symbole
                cursor.execute(
                    """
                    SELECT entry_price, quantity, metadata, symbol
                    FROM trade_cycles WHERE id = %s
                """,
                    (cycle_id,),
                )

                result = cursor.fetchone()
                if not result:
                    logger.error(f"Cycle {cycle_id} non trouvé")
                    return

                entry_price = Decimal(str(result[0]))
                symbol = result[3]

                try:
                    metadata = json.loads(result[2]) if result[2] else {}
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

                # Récupérer le prix max depuis Redis
                max_price = entry_price  # Valeur par défaut
                try:
                    max_price_key = f"cycle_max_price:{symbol}"
                    max_price_data = self.redis_client.get(max_price_key)

                    if max_price_data:
                        if isinstance(max_price_data, dict):
                            max_price = Decimal(
                                str(max_price_data.get("price", entry_price))
                            )
                        elif isinstance(max_price_data, str | bytes):
                            if isinstance(max_price_data, bytes):
                                max_price_data = max_price_data.decode("utf-8")
                            max_price_dict = json.loads(max_price_data)
                            max_price = Decimal(
                                str(max_price_dict.get("price", entry_price))
                            )

                        # Nettoyer la clé Redis après récupération
                        self.redis_client.delete(max_price_key)
                    else:
                        # Si pas de max en Redis, utiliser le max entre entry
                        # et exit
                        max_price = max(entry_price, exit_price)
                        logger.debug(
                            f"Pas de max_price dans Redis pour {symbol}, utilisation de {max_price}"
                        )

                except Exception as e:
                    logger.warning(f"Erreur récupération max_price depuis Redis: {e}")
                    max_price = max(entry_price, exit_price)

                # Calculer les frais (0.1% à l'achat + 0.1% à la vente = 0.2%
                # total)
                entry_value = entry_price * exit_quantity
                exit_value = exit_price * exit_quantity
                total_fees = (entry_value + exit_value) * self.fee_rate

                # Calculer le P&L net (après frais)
                gross_profit = (exit_price - entry_price) * exit_quantity
                profit_loss = gross_profit - total_fees
                profit_loss_percent = (profit_loss / entry_value) * 100

                # Ajouter les infos de SELL dans metadata
                metadata["sell_info"] = {
                    "order_id": order_id,
                    "price": str(exit_price),
                    "quantity": str(exit_quantity),
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                }

                # Ajouter les détails des frais
                metadata["fees"] = {
                    "entry_fee": str(entry_value * self.fee_rate),
                    "exit_fee": str(exit_value * self.fee_rate),
                    "total_fees": str(total_fees),
                    "gross_profit": str(gross_profit),
                }

                # Ajouter info sur le max atteint
                metadata["max_price_reached"] = str(max_price)
                metadata["max_gain_percent"] = str(
                    ((max_price - entry_price) / entry_price) * 100
                )

                # Mettre à jour le cycle avec le max_price
                cursor.execute(
                    """
                    UPDATE trade_cycles SET
                        status = 'completed',
                        exit_order_id = %s,
                        exit_price = %s,
                        max_price = %s,
                        profit_loss = %s,
                        profit_loss_percent = %s,
                        metadata = %s,
                        completed_at = NOW(),
                        updated_at = NOW()
                    WHERE id = %s
                """,
                    (
                        order_id,
                        exit_price,
                        max_price,
                        profit_loss,
                        profit_loss_percent,
                        json.dumps(metadata),
                        cycle_id,
                    ),
                )

                logger.info(
                    f"✅ Cycle {cycle_id} fermé: P&L={profit_loss:.2f} ({profit_loss_percent:.2f}%), Max atteint={max_price}"
                )

        except Exception:
            logger.exception("❌ Erreur _close_cycle")

    def _partial_close_cycle(
        self, cycle_id: str, exit_price: Decimal, exit_quantity: Decimal, order_id: str
    ) -> None:
        """
        Ferme partiellement un cycle (vente partielle de la position).

        Args:
            cycle_id: ID du cycle
            exit_price: Prix de sortie
            exit_quantity: Quantité vendue
            order_id: ID de l'ordre
        """
        try:
            with transaction() as cursor:
                # Récupérer les données du cycle
                cursor.execute(
                    """
                    SELECT * FROM trade_cycles WHERE id = %s
                """,
                    (cycle_id,),
                )

                result = cursor.fetchone()
                if not result:
                    logger.error(f"Cycle {cycle_id} non trouvé")
                    return

                columns = [desc[0] for desc in cursor.description]
                cycle_data = dict(zip(columns, result))

                current_quantity = Decimal(str(cycle_data["quantity"]))
                remaining_quantity = current_quantity - exit_quantity

                if remaining_quantity <= 0:
                    # Si on vend tout ou plus, fermer complètement
                    self._close_cycle(cycle_id, exit_price, current_quantity, order_id)
                    return

                # Fermer le cycle actuel avec la quantité vendue
                self._close_cycle(cycle_id, exit_price, exit_quantity, order_id)

                # Créer un nouveau cycle avec la quantité restante
                new_cycle_id = f"cycle_{uuid.uuid4().hex[:16]}"

                try:
                    metadata = json.loads(cycle_data.get("metadata", "{}") or "{}")
                except (json.JSONDecodeError, TypeError):
                    metadata = {}

                metadata["partial_from_cycle"] = cycle_id

                cursor.execute(
                    """
                    INSERT INTO trade_cycles (
                        id, symbol, strategy, status, side, entry_order_id,
                        entry_price, quantity, min_price, max_price,
                        metadata, created_at, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, NOW(), NOW()
                    )
                """,
                    (
                        new_cycle_id,
                        cycle_data["symbol"],
                        cycle_data["strategy"],
                        "active_buy",
                        "BUY",
                        cycle_data["entry_order_id"],
                        cycle_data["entry_price"],
                        remaining_quantity,
                        cycle_data["min_price"],
                        cycle_data["max_price"],
                        json.dumps(metadata),
                    ),
                )

                logger.info(
                    f"📊 Fermeture partielle: {exit_quantity} @ {exit_price}, reste {remaining_quantity}"
                )

        except Exception:
            logger.exception("❌ Erreur _partial_close_cycle")

    def _create_orphan_sell_cycle(
        self,
        symbol: str,
        price: Decimal,
        quantity: Decimal,
        order_id: str,
        strategy: str,
    ) -> str | None:
        """
        Crée un cycle "orphelin" pour un SELL sans BUY correspondant.
        Utile pour tracker les ventes de positions préexistantes.

        Args:
            symbol: Symbole vendu
            price: Prix de vente
            quantity: Quantité vendue
            order_id: ID de l'ordre
            strategy: Stratégie utilisée

        Returns:
            ID du cycle orphelin créé
        """
        try:
            cycle_id = f"cycle_orphan_{uuid.uuid4().hex[:16]}"

            metadata = {
                "orphan_sell": True,
                "reason": "No active buy cycle found",
                "order_id": order_id,
            }

            with transaction() as cursor:
                cursor.execute(
                    """
                    INSERT INTO trade_cycles (
                        id, symbol, strategy, status, side,
                        exit_order_id, exit_price, quantity,
                        metadata, created_at, updated_at, completed_at
                    ) VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s,
                        %s, NOW(), NOW(), NOW()
                    )
                """,
                    (
                        cycle_id,
                        symbol,
                        strategy,
                        "completed",
                        "SELL",
                        order_id,
                        price,
                        quantity,
                        json.dumps(metadata),
                    ),
                )

            logger.warning(
                f"⚠️ Cycle orphelin créé pour SELL {symbol}: {quantity} @ {price}"
            )
            return cycle_id

        except Exception:
            logger.exception("❌ Erreur _create_orphan_sell_cycle")
            return None
