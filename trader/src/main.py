"""
Point d'entrée principal pour le microservice Trader.
Démarre le gestionnaire d'ordres et expose une API REST pour les interactions manuelles.
"""

import argparse
import logging
import signal
import sys
import time
import os
import json
import threading
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify, Response
import psutil
import requests
import hmac
import hashlib

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Maintenant que le chemin est configuré, importer les modules nécessaires
from shared.src.config import BINANCE_SECRET_KEY, SYMBOLS, LOG_LEVEL, TRADING_MODE
from shared.src.enums import OrderSide

from trader.src.order_manager import OrderManager

# Configuration du logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("trader.log")],
)
logger = logging.getLogger("trader")


class TraderService:
    """
    Service principal pour le microservice Trader.
    Gère l'OrderManager et expose une API REST pour les interactions.
    """

    def __init__(self, symbols: Optional[Any] = None, port: int = 5002) -> None:
        """
        Initialise le service Trader.

        Args:
            symbols: Liste des symboles à surveiller
            port: Port pour l'API REST
        """
        self.symbols = symbols or SYMBOLS
        self.port = port
        self.process = psutil.Process(os.getpid())
        self.order_manager = None
        self.app = Flask(__name__)
        self.running = False
        self.start_time = time.time()

        # Configurer les routes de l'API
        self.setup_routes()

        logger.info(f"✅ TraderService initialisé pour {len(self.symbols)} symboles")

    def setup_routes(self) -> Any:
        """Configure les routes de l'API Flask."""
        self.app.route("/health", methods=["GET"])(self.health_check)
        self.app.route("/diagnostic", methods=["GET"])(self.diagnostic)
        self.app.route("/orders", methods=["GET"])(self.get_orders)
        self.app.route("/order", methods=["POST"])(self.create_order)
        self.app.route("/order/<order_id>", methods=["DELETE"])(self.cancel_order)
        self.app.route("/close/<cycle_id>", methods=["POST"])(self.close_cycle)
        self.app.route("/config/pause", methods=["POST"])(self.pause_trading)
        self.app.route("/config/resume", methods=["POST"])(self.resume_trading)
        self.app.route("/diag/binance", methods=["GET"])(self.diagnostic_binance)
        self.app.route("/api/diagnostic/signature", methods=["GET"])(
            self.diagnostic_binance_signature
        )

    def health_check(self) -> Any:
        """
        Point de terminaison pour vérifier l'état du service.
        """
        return jsonify(
            {
                "status": "healthy" if self.running else "stopped",
                "timestamp": time.time(),
                "uptime": time.time() - self.start_time,
                "mode": TRADING_MODE,
                "symbols": self.symbols,
            }
        )

    def diagnostic(self) -> Any:
        """
        Point de terminaison pour le diagnostic du service.
        Fournit des informations complètes sur l'état du système.
        """
        if not self.order_manager:
            return jsonify({"error": "OrderManager non initialisé"}), 500

        # Récupérer les cycles actifs
        active_cycles = self.order_manager.get_active_cycles()

        # Collecter les informations sur les derniers prix
        last_prices = (
            self.order_manager.last_prices.copy()
            if hasattr(self.order_manager, "last_prices")
            else {}
        )

        # Vérifier l'état des connexions
        redis_status = (
            "connected"
            if hasattr(self.order_manager.redis_signal_client, "connection")
            else "disconnected"
        )

        # Vérifier l'état d'exécution des threads
        thread_alive = (
            self.order_manager.processing_thread
            and self.order_manager.processing_thread.is_alive()
        )

        # Construire la réponse
        diagnostic_info = {
            "status": "operational" if self.running else "stopped",
            "timestamp": time.time(),
            "mode": TRADING_MODE,
            "symbols": self.symbols,
            "cycles": {
                "active_count": len(active_cycles),
                "cycles": [
                    {
                        "id": cycle.id,
                        "symbol": cycle.symbol,
                        "strategy": cycle.strategy,
                        "status": (
                            cycle.status.value
                            if hasattr(cycle.status, "value")
                            else str(cycle.status)
                        ),
                        "entry_price": cycle.entry_price,
                        "current_price": last_prices.get(cycle.symbol),
                        "quantity": cycle.quantity,
                        "pl_percent": (
                            cycle.profit_loss_percent
                            if hasattr(cycle, "profit_loss_percent")
                            else None
                        ),
                        "created_at": (
                            cycle.created_at.isoformat() if cycle.created_at else None
                        ),
                    }
                    for cycle in active_cycles
                ],
            },
            "market_data": {
                "prices": {symbol: price for symbol, price in last_prices.items()},
                "update_time": (
                    self.order_manager.last_price_update
                    if hasattr(self.order_manager, "last_price_update")
                    else None
                ),
            },
            "connections": {
                "redis": redis_status,
                "processing_thread": "running" if thread_alive else "stopped",
                "binance": self.order_manager.binance_executor.demo_mode
                and "demo"
                or "live",
            },
            "queue_size": (
                self.order_manager.signal_queue.qsize()
                if hasattr(self.order_manager, "signal_queue")
                else None
            ),
            "memory_usage_mb": round(self.process.memory_info().rss / (1024 * 1024), 2),
        }

        return jsonify(diagnostic_info)

    def diagnostic_binance_signature(self) -> Any:
        """
        Endpoint Flask de diagnostic pour tester la génération de signature Binance.
        """
        timestamp = int(time.time() * 1000)
        test_params = {
            "symbol": "BTCUSDC",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": "0.00100",
            "price": "30000.00",
            "timeInForce": "GTC",
            "timestamp": timestamp,
        }
        query_string = "&".join(
            [f"{key}={test_params[key]}" for key in sorted(test_params.keys())]
        )
        signature = hmac.new(
            BINANCE_SECRET_KEY.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return jsonify(
            {
                "query_string": query_string,
                "signature": signature,
                "timestamp": timestamp,
                "message": "Signature générée localement (Flask version)",
            }
        )

    def get_orders(self) -> Any:
        """
        Récupère les ordres actifs.
        """
        if not self.order_manager:
            return jsonify({"error": "OrderManager non initialisé"}), 500

        active_orders = self.order_manager.get_active_orders()
        return jsonify(active_orders)

    def create_order(self) -> Any:
        """
        Crée un ordre manuel.

        Exemple de requête:
        {
            "symbol": "BTCUSDC",
            "side": "BUY",
            "quantity": 0.001,
            "price": 50000  # optionnel
        }
        """
        if not self.order_manager:
            return jsonify({"error": "OrderManager non initialisé"}), 500

        try:
            data = request.json

            # Valider les paramètres requis
            if not all(k in data for k in ["symbol", "side", "quantity"]):
                return (
                    jsonify(
                        {
                            "error": "Paramètres manquants. Requis: symbol, side, quantity"
                        }
                    ),
                    400,
                )

            symbol = data["symbol"]
            side = OrderSide(data["side"])  # Conversion en enum
            quantity = float(data["quantity"])
            price = float(data["price"]) if "price" in data else None

            # Créer l'ordre
            result = self.order_manager.create_manual_order(
                symbol, side, quantity, price
            )

            # Vérifier si c'est un ID ou un message d'erreur
            if isinstance(result, str) and result.startswith("cycle_"):
                return jsonify({"order_id": result, "status": "created"}), 201
            else:
                return jsonify({"error": str(result)}), 400

        except (ValueError, TypeError) as e:
            logger.error(f"❌ Erreur lors de la création de l'ordre: {str(e)}")
            return jsonify({"error": str(e)}), 400
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Problème de connexion: {str(e)}")
            return (
                jsonify(
                    {"error": "Problème de connexion au service. Veuillez réessayer."}
                ),
                503,
            )
        except Exception as e:
            logger.critical(f"Erreur inattendue: {str(e)}")
            return jsonify({"error": "Erreur interne du serveur"}), 500

    def cancel_order(self, order_id: Any) -> Any:
        """
        Annule un ordre existant.
        """
        if not self.order_manager:
            return jsonify({"error": "OrderManager non initialisé"}), 500

        try:
            # Récupérer les paramètres optionnels
            reason = request.args.get("reason", "Annulation manuelle via API")

            # Annuler le cycle
            success = self.order_manager.cycle_manager.cancel_cycle(order_id, reason)

            if success:
                return jsonify({"status": "canceled", "order_id": order_id}), 200
            else:
                return (
                    jsonify({"error": f"Impossible d'annuler l'ordre {order_id}"}),
                    400,
                )

        except (ValueError, TypeError) as e:
            logger.error(f"❌ Erreur lors de la création de l'ordre: {str(e)}")
            return jsonify({"error": str(e)}), 400
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Problème de connexion: {str(e)}")
            return (
                jsonify(
                    {"error": "Problème de connexion au service. Veuillez réessayer."}
                ),
                503,
            )
        except Exception as e:
            logger.critical(f"Erreur inattendue: {str(e)}")
            return jsonify({"error": "Erreur interne du serveur"}), 500

    def close_cycle(self, cycle_id: Any) -> Any:
        """
        Ferme un cycle de trading.

        Exemple de requête:
        {
            "price": 50000  # optionnel
        }
        """
        if not self.order_manager:
            return jsonify({"error": "OrderManager non initialisé"}), 500

        try:
            data = request.json or {}
            price = float(data["price"]) if "price" in data else None

            # Fermer le cycle
            success = self.order_manager.cycle_manager.close_cycle(cycle_id, price)

            if success:
                return jsonify({"status": "closed", "cycle_id": cycle_id}), 200
            else:
                return (
                    jsonify({"error": f"Impossible de fermer le cycle {cycle_id}"}),
                    400,
                )

        except (ValueError, TypeError) as e:
            logger.error(f"❌ Erreur lors de la création de l'ordre: {str(e)}")
            return jsonify({"error": str(e)}), 400
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Problème de connexion: {str(e)}")
            return (
                jsonify(
                    {"error": "Problème de connexion au service. Veuillez réessayer."}
                ),
                503,
            )
        except Exception as e:
            logger.critical(f"Erreur inattendue: {str(e)}")
            return jsonify({"error": "Erreur interne du serveur"}), 500

    def pause_trading(self) -> Any:
        """
        Met en pause les transactions pour un symbole ou une stratégie spécifique.

        Exemple de requête:
        {
            "symbol": "BTCUSDC",  # optionnel
            "strategy": "Bollinger_Strategy",  # optionnel
            "duration": 3600  # durée en secondes, optionnel
        }
        """
        if not self.order_manager:
            return jsonify({"error": "OrderManager non initialisé"}), 500

        try:
            data = request.json or {}

            symbol = data.get("symbol")
            strategy = data.get("strategy")
            duration = data.get("duration", 0)  # 0 = pause indéfinie

            # Appliquer la pause
            if hasattr(self.order_manager, "pause_symbol") and symbol:
                self.order_manager.pause_symbol(symbol, duration)
                logger.info(f"⏸️ Trading en pause pour le symbole {symbol}")

            if hasattr(self.order_manager, "pause_strategy") and strategy:
                self.order_manager.pause_strategy(strategy, duration)
                logger.info(f"⏸️ Trading en pause pour la stratégie {strategy}")

            if not symbol and not strategy and hasattr(self.order_manager, "pause_all"):
                self.order_manager.pause_all(duration)
                logger.info("⏸️ Trading en pause pour tous les symboles et stratégies")

            return jsonify(
                {
                    "status": "paused",
                    "symbol": symbol,
                    "strategy": strategy,
                    "duration": duration,
                    "until": time.time() + duration if duration > 0 else None,
                }
            )

        except (ValueError, TypeError) as e:
            logger.error(f"❌ Erreur lors de la création de l'ordre: {str(e)}")
            return jsonify({"error": str(e)}), 400
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Problème de connexion: {str(e)}")
            return (
                jsonify(
                    {"error": "Problème de connexion au service. Veuillez réessayer."}
                ),
                503,
            )
        except Exception as e:
            logger.critical(f"Erreur inattendue: {str(e)}")
            return jsonify({"error": "Erreur interne du serveur"}), 500

    def resume_trading(self) -> Any:
        """
        Reprend les transactions pour un symbole ou une stratégie spécifique.

        Exemple de requête:
        {
            "symbol": "BTCUSDC",  # optionnel
            "strategy": "Bollinger_Strategy"  # optionnel
        }
        """
        if not self.order_manager:
            return jsonify({"error": "OrderManager non initialisé"}), 500

        try:
            data = request.json or {}

            symbol = data.get("symbol")
            strategy = data.get("strategy")

            # Reprendre le trading
            if hasattr(self.order_manager, "resume_symbol") and symbol:
                self.order_manager.resume_symbol(symbol)
                logger.info(f"▶️ Trading repris pour le symbole {symbol}")

            if hasattr(self.order_manager, "resume_strategy") and strategy:
                self.order_manager.resume_strategy(strategy)
                logger.info(f"▶️ Trading repris pour la stratégie {strategy}")

            if (
                not symbol
                and not strategy
                and hasattr(self.order_manager, "resume_all")
            ):
                self.order_manager.resume_all()
                logger.info("▶️ Trading repris pour tous les symboles et stratégies")

            return jsonify(
                {"status": "resumed", "symbol": symbol, "strategy": strategy}
            )

        except (ValueError, TypeError) as e:
            logger.error(f"❌ Erreur lors de la création de l'ordre: {str(e)}")
            return jsonify({"error": str(e)}), 400
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Problème de connexion: {str(e)}")
            return (
                jsonify(
                    {"error": "Problème de connexion au service. Veuillez réessayer."}
                ),
                503,
            )
        except Exception as e:
            logger.critical(f"Erreur inattendue: {str(e)}")
            return jsonify({"error": "Erreur interne du serveur"}), 500

    def diagnostic_binance(self) -> Any:
        """
        Effectue un diagnostic de la connexion Binance.
        """
        if not self.order_manager:
            return jsonify({"error": "OrderManager non initialisé"}), 500

        try:
            # Extraire les premières/dernières lettres de la clé pour vérification sans exposer toute la clé
            api_key = self.order_manager.binance_executor.api_key
            api_key_preview = (
                f"{api_key[:4]}...{api_key[-4:]}" if api_key else "Non définie"
            )

            # Vérifier si la clé API a un format valide
            api_key_valid = bool(api_key and len(api_key) > 8)

            # Tester une requête public (ne nécessite pas d'authentification)
            ping_success = False
            ping_error = None
            try:
                ping_url = f"{self.order_manager.binance_executor.BASE_URL}/api/v3/ping"
                response = requests.get(ping_url, timeout=5)
                ping_success = response.status_code == 200
            except (ValueError, TypeError) as e:
                ping_error = str(e)

            # Tester une requête privée (nécessite authentification)
            auth_success = False
            auth_error = None
            try:
                timestamp = int(time.time() * 1000)
                params = {"timestamp": timestamp}
                signature = self.order_manager.binance_executor._generate_signature(
                    params
                )
                account_url = (
                    f"{self.order_manager.binance_executor.BASE_URL}/api/v3/account"
                )
                params["signature"] = signature
                headers = {"X-MBX-APIKEY": api_key}
                response = requests.get(
                    account_url, params=params, headers=headers, timeout=5
                )
                auth_success = response.status_code == 200
                if not auth_success:
                    auth_error = response.text
            except Exception as e:
                auth_error = str(e)

            return jsonify(
                {
                    "api_key_preview": api_key_preview,
                    "api_key_valid": api_key_valid,
                    "ping_test": {"success": ping_success, "error": ping_error},
                    "auth_test": {"success": auth_success, "error": auth_error},
                    "demo_mode": self.order_manager.binance_executor.demo_mode,
                }
            )

        except (ValueError, TypeError) as e:
            logger.error(f"❌ Erreur lors de la création de l'ordre: {str(e)}")
            return jsonify({"error": str(e)}), 400
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Problème de connexion: {str(e)}")
            return (
                jsonify(
                    {"error": "Problème de connexion au service. Veuillez réessayer."}
                ),
                503,
            )
        except Exception as e:
            logger.critical(f"Erreur inattendue: {str(e)}")
            return jsonify({"error": "Erreur interne du serveur"}), 500

    def start(self) -> Any:
        """
        Démarre le service Trader.
        """
        if self.running:
            logger.warning("Le service est déjà en cours d'exécution")
            return

        self.running = True
        logger.info("🚀 Démarrage du service Trader RootTrading...")
        logger.info(
            f"Configuration: {len(self.symbols)} symboles, "
            f"mode {'DÉMO' if TRADING_MODE.lower() == 'demo' else 'RÉEL'}"
        )

        try:
            # Initialiser le gestionnaire d'ordres
            self.order_manager = OrderManager(symbols=self.symbols)
            self.order_manager.start()

            logger.info("✅ Service Trader démarré")

        except (ValueError, TypeError) as e:
            logger.error(f"❌ Erreur critique lors du démarrage: {str(e)}")
            self.running = False
            raise
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Problème de connexion: {str(e)}")
            self.running = False
            raise
        except Exception as e:
            logger.critical(f"Erreur inattendue: {str(e)}")
            self.running = False
            raise

    def start_api_server(self) -> Any:
        """
        Démarre le serveur API dans un thread séparé.
        """
        api_thread = threading.Thread(
            target=lambda: self.app.run(
                host="0.0.0.0", port=self.port, debug=False, use_reloader=False
            ),
            daemon=True,
        )
        api_thread.start()
        logger.info(f"✅ API REST démarrée sur le port {self.port}")
        return api_thread

    def stop(self) -> None:
        """
        Arrête proprement le service Trader.
        """
        if not self.running:
            return

        logger.info("Arrêt du service Trader...")
        self.running = False

        # Arrêter le gestionnaire d'ordres proprement
        if self.order_manager:
            self.order_manager.stop()
            self.order_manager = None

        logger.info("Service Trader terminé")


def parse_arguments() -> None:
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(description="Trader RootTrading")
    parser.add_argument("--port", type=int, default=5002, help="Port pour l'API REST")
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Liste de symboles séparés par des virgules (ex: BTCUSDC,ETHUSDC)",
    )
    parser.add_argument("--no-api", action="store_true", help="Désactive l'API REST")
    return parser.parse_args()


def main() -> None:
    """Fonction principale du service Trader."""
    # Parser les arguments
    args = parse_arguments()

    # Configurer les symboles
    symbols = args.symbols.split(",") if args.symbols else SYMBOLS

    # Créer le service
    trader_service = TraderService(symbols=symbols, port=args.port)

    # Configurer les gestionnaires de signaux
    def signal_handler(sig: Any, frame: Any) -> Any:
        logger.info(f"Signal {sig} reçu, arrêt en cours...")
        trader_service.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Démarrer le service
        trader_service.start()

        # Démarrer l'API REST si activée
        if not args.no_api:
            trader_service.start_api_server()

        # Maintenir le programme en vie
        while trader_service.running:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")
    except (ValueError, TypeError) as e:
        logger.error(f"❌ Erreur critique dans le service Trader: {str(e)}")
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Problème de connexion: {str(e)}")
    except Exception as e:
        logger.critical(f"Erreur inattendue: {str(e)}")
        raise
    finally:
        # Arrêter le service
        trader_service.stop()


if __name__ == "__main__":
    main()
