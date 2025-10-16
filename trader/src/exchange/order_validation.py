# trader/src/exchange/order_validation.py
"""
Validation des ordres avant envoi à Binance.
S'assure que les ordres respectent les contraintes de l'exchange.
"""
import logging

from shared.src.schemas import TradeOrder
from trader.src.exchange.constraints import BinanceSymbolConstraints

# Configuration du logging
logger = logging.getLogger(__name__)


class OrderValidationError(Exception):
    """Exception levée lorsqu'un ordre ne respecte pas les contraintes."""


class OrderValidator:
    """
    Validateur d'ordres pour Binance.
    Vérifie et ajuste les ordres pour qu'ils respectent les contraintes de l'exchange.
    """

    def __init__(self, constraints: BinanceSymbolConstraints):
        """
        Initialise le validateur d'ordres.

        Args:
            constraints: Contraintes de symbole pour Binance
        """
        self.constraints = constraints
        logger.info("✅ Validateur d'ordres initialisé")

    def validate_and_adjust_order(self, order: TradeOrder) -> TradeOrder:
        """
        Valide et ajuste un ordre pour qu'il respecte les contraintes de Binance.

        Args:
            order: Ordre à valider et ajuster

        Returns:
            Ordre ajusté

        Raises:
            OrderValidationError: Si l'ordre ne peut pas être ajusté pour respecter les contraintes
        """
        try:
            # Clone de l'ordre pour ne pas modifier l'original
            adjusted_order = order.copy()

            # Vérifier que la quantité n'est pas nulle
            if adjusted_order.quantity <= 0:
                logger.error(
                    f"❌ Quantité invalide (zéro ou négative): {adjusted_order.quantity}"
                )
                raise OrderValidationError(
                    f"Quantité invalide (zéro ou négative): {adjusted_order.quantity}"
                )

            # Tronquer la quantité au step size
            original_quantity = adjusted_order.quantity
            adjusted_order.quantity = self.constraints.truncate_quantity(
                adjusted_order.symbol, adjusted_order.quantity
            )

            # Log pour débogage si la quantité a changé significativement
            if (abs(original_quantity - adjusted_order.quantity) /
                    original_quantity > 0.01):  # Changement de plus de 1%
                logger.warning(
                    f"⚠️ Ajustement significatif de quantité: {original_quantity} → {adjusted_order.quantity} pour {adjusted_order.symbol}"
                )

            # Validation de quantité minimale
            min_qty = self.constraints.get_min_qty(adjusted_order.symbol)
            if adjusted_order.quantity < min_qty:
                logger.error(
                    f"❌ Quantité {adjusted_order.quantity} trop faible pour {adjusted_order.symbol} (min: {min_qty})"
                )
                raise OrderValidationError(
                    f"Quantité {adjusted_order.quantity} trop faible pour {adjusted_order.symbol} (min: {min_qty})"
                )

            # Validation de prix (si c'est un ordre limité)
            if adjusted_order.price is not None:
                adjusted_order.price = self.constraints.round_price(
                    adjusted_order.symbol, adjusted_order.price
                )

            # Validation de notional (quantité * prix)
            from trader.src.exchange.binance_utils import BinanceUtils

            binance_utils = BinanceUtils(
                "", ""
            )  # Clés vides car on n'utilise pas d'API ici

            # Récupérer le prix actuel si nécessaire
            price = adjusted_order.price
            if price is None:
                try:
                    price = binance_utils.get_current_price(
                        adjusted_order.symbol)
                except Exception as e:
                    logger.warning(
                        f"⚠️ Impossible de récupérer le prix actuel: {e!s}"
                    )
                    # Utiliser un prix par défaut selon le symbole
                    if adjusted_order.symbol.startswith("BTC"):
                        price = 50000.0
                    elif adjusted_order.symbol.startswith("ETH"):
                        price = 3000.0
                    else:
                        price = 100.0

            # Vérifier que le notional est suffisant
            notional = adjusted_order.quantity * price
            min_notional = self.constraints.get_min_notional(
                adjusted_order.symbol)

            # Ajouter une petite marge de tolérance pour éviter les erreurs d'arrondi
            # Utiliser 1e-8 comme epsilon pour les comparaisons de décimaux
            epsilon = 1e-8

            if notional < min_notional - epsilon:
                logger.error(
                    f"❌ Notional {notional:.8f} trop faible pour {adjusted_order.symbol} (min: {min_notional})"
                )
                raise OrderValidationError(
                    f"Notional trop faible pour {adjusted_order.symbol}: {notional:.8f} (min: {min_notional})"
                )

            logger.info(
                f"✅ Ordre validé et ajusté: {adjusted_order.symbol} {adjusted_order.side} {adjusted_order.quantity}"
            )
            return adjusted_order

        except OrderValidationError:
            # Relancer les erreurs de validation
            raise
        except Exception as e:
            # Capturer les autres erreurs et les convertir en erreurs de
            # validation
            logger.exception(
                f"❌ Erreur inattendue lors de la validation de l'ordre: {e!s}"
            )
            raise OrderValidationError(f"Erreur de validation: {e!s}")
