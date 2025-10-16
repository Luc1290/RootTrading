"""
Routes API simplifiées pour le trader.
Plus de cycles complexes : juste exécuter les ordres du coordinator.
"""

import logging
import time
from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify, request

from shared.src.config import TRADING_MODE
from shared.src.db_pool import get_db_metrics

logger = logging.getLogger(__name__)

# Créer un Blueprint pour organiser les routes
routes_bp = Blueprint("api_routes", __name__)

# ============================================================================
# Routes de base
# ============================================================================


@routes_bp.route("/health", methods=["GET"])
def health_check():
    """Point de terminaison pour vérifier l'état du service."""
    order_manager = current_app.config["ORDER_MANAGER"]

    return jsonify(
        {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime": time.time() - order_manager.start_time,
            "mode": TRADING_MODE,
            "symbols": order_manager.symbols,
            "version": "simplified-1.0",
        }
    )


@routes_bp.route("/diagnostic", methods=["GET"])
def diagnostic():
    """Point de terminaison pour le diagnostic du service."""
    order_manager = current_app.config["ORDER_MANAGER"]

    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500

    # Récupérer les statistiques
    stats = order_manager.get_stats()

    diagnostic_info = {
        "status": "operational",
        "timestamp": time.time(),
        "mode": TRADING_MODE,
        "symbols": order_manager.symbols,
        "uptime": stats["uptime"],
        "market_data": {
            "prices": order_manager.get_current_prices(),
            "last_update": stats["last_price_update"],
        },
        "trading_status": {
            "paused_symbols": stats["paused_symbols"],
            "paused_strategies": stats["paused_strategies"],
            "paused_all": stats["paused_all"],
        },
        "executor_stats": stats["executor_stats"],
    }

    # Ajouter les métriques de base de données
    try:
        diagnostic_info["database"] = get_db_metrics()
    except Exception as e:
        logger.warning(
            f"Impossible de récupérer les métriques de base de données: {e!s}"
        )
        diagnostic_info["database"] = {
            "status": "unavailable", "error": str(e)}

    return jsonify(diagnostic_info)


# ============================================================================
# Routes des prix
# ============================================================================


@routes_bp.route("/prices", methods=["GET"])
def get_prices():
    """Récupère les prix actuels pour les symboles demandés."""
    order_manager = current_app.config["ORDER_MANAGER"]

    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500

    symbols_param = request.args.get("symbols", "")
    if not symbols_param:
        return jsonify({"error": "Paramètre 'symbols' requis"}), 400

    symbols = [s.strip() for s in symbols_param.split(",")]
    all_prices = order_manager.get_current_prices()

    prices = {}
    for symbol in symbols:
        if symbol in all_prices:
            prices[symbol] = all_prices[symbol]

    if not prices:
        return jsonify({"error": f"Aucun prix disponible pour {symbols}"}), 404

    return jsonify(prices)


@routes_bp.route("/price/<symbol>", methods=["GET"])
def get_current_price(symbol):
    """Récupère le prix actuel d'un symbole."""
    order_manager = current_app.config["ORDER_MANAGER"]

    try:
        prices = order_manager.get_current_prices()
        price = prices.get(symbol)

        if price:
            return jsonify(
                {
                    "success": True,
                    "symbol": symbol,
                    "price": price,
                    "timestamp": time.time(),
                }
            )
        return (
            jsonify(
                {"success": False, "message": f"Prix non disponible pour {symbol}"}
            ),
            404,
        )

    except Exception as e:
        logger.exception("❌ Erreur lors de la récupération du prix")
        return jsonify({"success": False, "message": f"Erreur: {e!s}"}), 500


# ============================================================================
# Routes des ordres
# ============================================================================


@routes_bp.route("/order", methods=["POST"])
def create_order():
    """
    Crée un ordre (route principale du coordinator).

    Exemple de requête:
    {
        "symbol": "BTCUSDC",
        "side": "BUY",
        "quantity": 0.001,
        "price": 50000,
        "strategy": "Manual"
    }
    """
    order_manager = current_app.config["ORDER_MANAGER"]

    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500

    try:
        data = request.json

        if data is None:
            return jsonify({"error": "Corps de requête JSON manquant"}), 400

        # Valider les paramètres requis
        if not all(k in data for k in ["symbol", "side", "quantity"]):
            return (
                jsonify(
                    {"error": "Paramètres manquants. Requis: symbol, side, quantity"}
                ),
                400,
            )

        # Créer l'ordre
        order_id = order_manager.create_order(data)

        if order_id:
            return (
                jsonify(
                    {
                        "order_id": order_id,
                        "status": "created",
                        "timestamp": time.time(),
                    }
                ),
                201,
            )
        return jsonify(
            {"error": "Ordre non créé - vérifiez les paramètres"}), 422

    except Exception as e:
        logger.exception("❌ Erreur lors de la création de l'ordre")
        return jsonify({"error": str(e)}), 500


@routes_bp.route("/orders", methods=["GET"])
def get_orders():
    """Récupère l'historique des ordres."""
    order_manager = current_app.config["ORDER_MANAGER"]

    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500

    try:
        limit = int(request.args.get("limit", 50))
        orders = order_manager.get_order_history(limit)

        # Convertir les timestamps en ISO format de manière sécurisée
        for order in orders:
            if "timestamp" in order and order["timestamp"] is not None:
                if hasattr(order["timestamp"], "isoformat"):
                    # Si c'est un objet datetime, on le convertit
                    order["timestamp"] = order["timestamp"].isoformat()
                elif isinstance(order["timestamp"], int | float):
                    # Si c'est un timestamp Unix, on le convertit
                    order["timestamp"] = datetime.fromtimestamp(
                        order["timestamp"], tz=timezone.utc
                    ).isoformat()
                else:
                    # Si c'est déjà une string ou autre, on la laisse telle
                    # quelle
                    order["timestamp"] = str(order["timestamp"])

        return jsonify(
            {"success": True, "count": len(orders), "orders": orders})

    except Exception as e:
        logger.exception("❌ Erreur lors de la récupération des ordres")
        return jsonify({"error": str(e)}), 500


@routes_bp.route("/order/<order_id>", methods=["GET"])
def get_order_status(order_id):
    """Récupère le statut d'un ordre."""
    order_manager = current_app.config["ORDER_MANAGER"]

    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500

    try:
        order = order_manager.get_order_status(order_id)

        if order:
            # Convertir le timestamp en ISO format de manière sécurisée
            if "timestamp" in order and order["timestamp"] is not None:
                if hasattr(order["timestamp"], "isoformat"):
                    # Si c'est un objet datetime, on le convertit
                    order["timestamp"] = order["timestamp"].isoformat()
                elif isinstance(order["timestamp"], int | float):
                    # Si c'est un timestamp Unix, on le convertit
                    order["timestamp"] = datetime.fromtimestamp(
                        order["timestamp"], tz=timezone.utc
                    ).isoformat()
                else:
                    # Si c'est déjà une string ou autre, on la laisse telle
                    # quelle
                    order["timestamp"] = str(order["timestamp"])

            return jsonify({"success": True, "order": order})
        return (
            jsonify({"success": False, "message": f"Ordre {order_id} non trouvé"}),
            404,
        )

    except Exception as e:
        logger.exception("❌ Erreur lors de la récupération du statut")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Routes de configuration
# ============================================================================


@routes_bp.route("/config/pause", methods=["POST"])
def pause_trading():
    """
    Met en pause le trading.

    Exemple de requête:
    {
        "symbol": "BTCUSDC",
        "strategy": "Manual"
    }
    """
    order_manager = current_app.config["ORDER_MANAGER"]

    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500

    try:
        data = request.json or {}

        symbol = data.get("symbol")
        strategy = data.get("strategy")

        if symbol:
            order_manager.pause_symbol(symbol)
        elif strategy:
            order_manager.pause_strategy(strategy)
        else:
            order_manager.pause_all()

        return jsonify(
            {
                "status": "paused",
                "symbol": symbol,
                "strategy": strategy,
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        logger.exception("❌ Erreur lors de la mise en pause")
        return jsonify({"error": str(e)}), 500


@routes_bp.route("/config/resume", methods=["POST"])
def resume_trading():
    """
    Reprend le trading.

    Exemple de requête:
    {
        "symbol": "BTCUSDC",
        "strategy": "Manual"
    }
    """
    order_manager = current_app.config["ORDER_MANAGER"]

    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500

    try:
        data = request.json or {}

        symbol = data.get("symbol")
        strategy = data.get("strategy")

        if symbol:
            order_manager.resume_symbol(symbol)
        elif strategy:
            order_manager.resume_strategy(strategy)
        else:
            order_manager.resume_all()

        return jsonify(
            {
                "status": "resumed",
                "symbol": symbol,
                "strategy": strategy,
                "timestamp": time.time(),
            }
        )

    except Exception as e:
        logger.exception("❌ Erreur lors de la reprise")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# Routes utilitaires
# ============================================================================


@routes_bp.route("/stats", methods=["GET"])
def get_stats():
    """Récupère les statistiques du trader."""
    order_manager = current_app.config["ORDER_MANAGER"]

    if not order_manager:
        return jsonify({"error": "OrderManager non initialisé"}), 500

    try:
        stats = order_manager.get_stats()
        return jsonify({"success": True, "stats": stats,
                       "timestamp": time.time()})

    except Exception as e:
        logger.exception("❌ Erreur lors de la récupération des stats")
        return jsonify({"error": str(e)}), 500


@routes_bp.route("/test", methods=["GET"])
def test_route():
    """Route de test simple."""
    return jsonify({"status": "ok",
                    "message": "Trader API fonctionne",
                    "timestamp": time.time()})


# ============================================================================
# Enregistrement des routes
# ============================================================================


def register_routes(app, order_manager):
    """
    Enregistre les routes dans l'application Flask.

    Args:
        app: Application Flask
        order_manager: Gestionnaire d'ordres
    """
    # Stocker le gestionnaire d'ordres dans la configuration
    app.config["ORDER_MANAGER"] = order_manager

    # Enregistrer le blueprint
    app.register_blueprint(routes_bp)

    logger.info("✅ Routes API simplifiées enregistrées")
