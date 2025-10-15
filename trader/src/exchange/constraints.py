# trader/src/exchange/constraints.py
"""
Gestion des contraintes de trading de Binance (min. qtés, précisions, etc.)
Fournit des informations sur les limites de trading par symbole.
"""
import logging
import math
from typing import Dict, Any, Optional
from decimal import Decimal, getcontext, ROUND_DOWN
from .symbol_cache import SymbolConstraintsCache

# Configuration de la précision décimale
getcontext().prec = 28

# Configuration du logging
logger = logging.getLogger(__name__)


class BinanceSymbolConstraints:
    """
    Gère les contraintes de trading par symbole pour Binance.
    Fournit des méthodes pour vérifier et ajuster les quantités et prix.
    """

    def __init__(
        self,
        symbol_info: Optional[Dict[str, Dict[str, Any]]] = None,
        cache_ttl: int = 600,
    ):
        """
        Initialise les contraintes de symbole avec des valeurs par défaut.

        Args:
            symbol_info: Informations de symboles obtenues depuis fetch_exchange_info()
            cache_ttl: Durée de vie du cache en secondes (défaut: 10 min)
        """
        self.symbol_info = symbol_info or {}
        self.cache = SymbolConstraintsCache(ttl_seconds=cache_ttl)

        # Quantité minimale par défaut pour les symboles (fallback si pas de données en temps réel)
        self.default_min_quantities = {
            "BTCUSDC": 0.00001,  # minQty selon Binance
            "ETHUSDC": 0.0001,  # minQty selon Binance
            "ETHBTC": 0.0001,  # minQty selon Binance
            "SOLUSDC": 0.001,  # minQty selon Binance
            "XRPUSDC": 0.1,  # minQty selon Binance
        }

        # Pas de quantité par défaut (step size) pour les symboles (fallback)
        self.default_step_sizes = {
            "BTCUSDC": 0.00001,  # stepSize selon Binance (5 décimales)
            "ETHUSDC": 0.0001,  # stepSize selon Binance (4 décimales)
            "ETHBTC": 0.0001,  # stepSize selon Binance (4 décimales)
            "SOLUSDC": 0.001,  # stepSize selon Binance (3 décimales)
            "XRPUSDC": 0.1,  # stepSize selon Binance (1 décimale)
        }

        # Valeur minimale des ordres par défaut (min notional) (fallback)
        self.default_min_notionals = {
            "BTCUSDC": 10.0,
            "ETHUSDC": 10.0,
            "ETHBTC": 0.0001,  # 0.0001 BTC ≈ 10 USDC, pas 10 BTC !
            "SOLUSDC": 10.0,  # 10 USDC minimum
            "XRPUSDC": 10.0,  # 10 USDC minimum
        }

        # Précision des prix par défaut (fallback)
        self.default_price_precisions = {
            "BTCUSDC": 2,  # 2 décimales (ex: 50000.25)
            "ETHUSDC": 2,  # 2 décimales (ex: 3000.50)
            "ETHBTC": 6,  # 6 décimales (ex: 0.024002)
            "SOLUSDC": 2,  # 2 décimales (ex: 147.25)
            "XRPUSDC": 4,  # 4 décimales (ex: 2.2145)
        }

        logger.info(
            f"✅ Contraintes de symbole initialisées avec {len(self.symbol_info)} symboles en temps réel et cache TTL {cache_ttl}s"
        )

    def _get_cached_or_fetch_constraints(self, symbol: str) -> Dict[str, Any]:
        """
        Récupère les contraintes depuis le cache ou les calcule et les met en cache.

        Args:
            symbol: Symbole (ex: 'BTCUSDC')

        Returns:
            Dictionnaire avec les contraintes du symbole
        """
        # Essayer de récupérer depuis le cache
        cached_constraints = self.cache.get(symbol)
        if cached_constraints is not None:
            return cached_constraints

        # Pas en cache, calculer les contraintes
        constraints = {}

        # Récupérer depuis symbol_info si disponible, sinon utiliser les defaults
        if symbol in self.symbol_info:
            symbol_data = self.symbol_info[symbol]
            constraints["min_qty"] = symbol_data.get(
                "min_qty", self.default_min_quantities.get(symbol, 0.001)
            )
            constraints["step_size"] = symbol_data.get(
                "step_size", self.default_step_sizes.get(symbol, 0.0001)
            )
            constraints["min_notional"] = symbol_data.get(
                "min_notional", self.default_min_notionals.get(symbol, 10.0)
            )
            constraints["tick_size"] = symbol_data.get("tick_size")
        else:
            # Utiliser les valeurs par défaut
            constraints["min_qty"] = self.default_min_quantities.get(symbol, 0.001)
            constraints["step_size"] = self.default_step_sizes.get(symbol, 0.0001)
            constraints["min_notional"] = self.default_min_notionals.get(symbol, 10.0)
            constraints["tick_size"] = None

        # Calculer la précision des prix
        if constraints["tick_size"] and constraints["tick_size"] > 0:
            constraints["price_precision"] = max(
                0, -int(math.floor(math.log10(constraints["tick_size"])))
            )
        else:
            constraints["price_precision"] = self.default_price_precisions.get(
                symbol, 2
            )

        # Mettre en cache
        self.cache.set(symbol, constraints)

        logger.debug(
            f"📊 Contraintes calculées et mises en cache pour {symbol}: {constraints}"
        )
        return constraints

    def get_min_qty(self, symbol: str) -> float:
        """
        Récupère la quantité minimale pour un symbole.

        Args:
            symbol: Symbole (ex: 'BTCUSDC')

        Returns:
            Quantité minimale
        """
        constraints = self._get_cached_or_fetch_constraints(symbol)
        return constraints["min_qty"]

    def get_step_size(self, symbol: str) -> float:
        """
        Récupère le pas de quantité pour un symbole.

        Args:
            symbol: Symbole (ex: 'BTCUSDC')

        Returns:
            Pas de quantité
        """
        constraints = self._get_cached_or_fetch_constraints(symbol)
        return constraints["step_size"]

    def get_min_notional(self, symbol: str) -> float:
        """
        Récupère la valeur minimale d'un ordre pour un symbole.

        Args:
            symbol: Symbole (ex: 'BTCUSDC')

        Returns:
            Valeur minimale de l'ordre
        """
        constraints = self._get_cached_or_fetch_constraints(symbol)
        return constraints["min_notional"]

    def get_price_precision(self, symbol: str) -> int:
        """
        Récupère la précision des prix pour un symbole.

        Args:
            symbol: Symbole (ex: 'BTCUSDC')

        Returns:
            Nombre de décimales pour les prix
        """
        constraints = self._get_cached_or_fetch_constraints(symbol)
        return constraints["price_precision"]

    def truncate_quantity(self, symbol: str, quantity: float) -> float:
        """
        Tronque la quantité au pas (stepSize) Binance, sans erreur binaire.

        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            quantity: Quantité à tronquer

        Returns:
            Quantité tronquée
        """
        if quantity <= 0:
            logger.warning(f"⚠️ Quantité négative ou nulle : {quantity}")
            return self.get_min_qty(symbol)

        step_size = self.get_step_size(symbol)  # ex. 0.000001
        step_dec = Decimal(str(step_size))
        qty_dec = Decimal(str(quantity))

        # Arrondi « floor » au multiple de stepSize
        truncated = (qty_dec // step_dec) * step_dec
        truncated = truncated.quantize(step_dec, rounding=ROUND_DOWN)

        # Sécurité : si on tombe à 0, on repasse à la quantité mini
        if truncated <= 0:
            logger.warning(
                f"⚠️ Quantité nulle après troncature : {quantity} → {truncated}"
            )
            truncated = Decimal(str(self.get_min_qty(symbol)))

        # Convertir en float mais s'assurer qu'il n'y a pas de notation scientifique
        result = float(truncated)

        # Si le résultat est très petit, le formater explicitement pour éviter la notation scientifique
        if result < 0.0001:
            # Calculer le nombre de décimales nécessaires selon le step_size
            step_decimals = (
                len(str(step_dec).split(".")[-1]) if "." in str(step_dec) else 0
            )
            # Formater avec suffisamment de décimales et reconvertir en float
            formatted = f"{result:.{step_decimals}f}"
            logger.debug(
                f"🔧 Formatage quantité pour éviter notation scientifique: {result} → {formatted}"
            )
            return float(formatted)

        return result

    def round_price(self, symbol: str, price: float) -> float:
        """
        Arrondit le prix selon le tickSize du symbole.

        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            price: Prix à arrondir

        Returns:
            Prix arrondi
        """
        precision = self.get_price_precision(symbol)
        return round(price, precision)

    def is_quantity_valid(self, symbol: str, quantity: float) -> bool:
        """
        Vérifie si la quantité est valide selon les règles de Binance.

        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            quantity: Quantité à vérifier

        Returns:
            True si la quantité est valide, False sinon
        """
        # Vérifier que la quantité n'est pas négative ou nulle
        if quantity <= 0:
            return False

        # Vérifier si la quantité est supérieure au minimum
        min_qty = self.get_min_qty(symbol)
        return quantity >= min_qty

    def is_notional_valid(self, symbol: str, quantity: float, price: float) -> bool:
        """
        Vérifie si la valeur totale (quantité * prix) est valide selon les règles de Binance.

        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            quantity: Quantité
            price: Prix

        Returns:
            True si la valeur est valide, False sinon
        """
        # Vérifier que la quantité et le prix ne sont pas nuls
        if quantity <= 0 or price <= 0:
            return False

        # Calculer la valeur totale
        notional = quantity * price

        # Vérifier si la valeur est supérieure au minimum
        min_notional = self.get_min_notional(symbol)
        return notional >= min_notional

    def calculate_min_quantity(self, symbol: str, price: float) -> float:
        """
        Calcule la quantité minimale nécessaire pour respecter à la fois
        les contraintes de min_qty et min_notional.

        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            price: Prix actuel

        Returns:
            Quantité minimale arrondie adaptée au step_size
        """
        min_qty = self.get_min_qty(symbol)
        min_notional = self.get_min_notional(symbol)

        # Calculer la quantité minimale pour respecter min_notional
        notional_min_qty = min_notional / price if price > 0 else min_qty

        # Prendre le maximum des deux contraintes
        required_min_qty = max(min_qty, notional_min_qty)

        # Arrondir au step size supérieur
        step_size = self.get_step_size(symbol)
        steps = (required_min_qty / step_size) if step_size > 0 else 0
        rounded_steps = math.ceil(steps)
        rounded_qty = rounded_steps * step_size

        logger.info(
            f"Quantité minimale calculée pour {symbol} @ {price}: {rounded_qty} (min_qty: {min_qty}, notional min: {notional_min_qty})"
        )
        return rounded_qty

    def invalidate_cache(self, symbol: str) -> bool:
        """
        Invalide le cache pour un symbole spécifique.

        Args:
            symbol: Symbole à invalider

        Returns:
            True si une entrée était présente, False sinon
        """
        return self.cache.invalidate(symbol)

    def clear_cache(self) -> None:
        """Vide complètement le cache des contraintes."""
        self.cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques du cache.

        Returns:
            Statistiques détaillées du cache
        """
        return self.cache.get_stats()

    def cleanup_expired_cache(self) -> int:
        """
        Nettoie les entrées expirées du cache.

        Returns:
            Nombre d'entrées supprimées
        """
        return self.cache.cleanup_expired()
