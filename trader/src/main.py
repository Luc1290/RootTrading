"""
Point d'entrée principal pour le microservice Trader.
Démarre le gestionnaire d'ordres et expose une API REST.
"""

import argparse
import logging
import signal
import sys
import time
import os

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Maintenant que le chemin est configuré, importer les modules nécessaires
from shared.src.config import SYMBOLS, LOG_LEVEL
from trader.src.trading.order_manager import OrderManager
from trader.src.api.rest_server import RestApiServer
from trader.src.utils.logging_config import setup_logging

# Configuration du logging
logger = logging.getLogger("trader")


class TraderService:
    """
    Service principal pour le microservice Trader.
    Gère l'OrderManager et expose une API REST pour les interactions.
    """

    def __init__(self, symbols=None, port=5002):
        """
        Initialise le service Trader.

        Args:
            symbols: Liste des symboles à surveiller
            port: Port pour l'API REST
        """
        self.symbols = symbols or SYMBOLS
        self.port = port
        self.order_manager = None
        self.api_server = None
        self.running = False
        self.start_time = time.time()

        logger.info(f"✅ TraderService initialisé pour {len(self.symbols)} symboles")

    def start(self):
        """
        Démarre le service Trader.
        """
        if self.running:
            logger.warning("Le service est déjà en cours d'exécution")
            return

        self.running = True
        logger.info("🚀 Démarrage du service Trader RootTrading...")

        try:
            # Initialiser le gestionnaire d'ordres
            self.order_manager = OrderManager(symbols=self.symbols)
            self.order_manager.start()

            # Initialiser et démarrer le serveur API REST
            self.api_server = RestApiServer(self.order_manager, port=self.port)
            self.api_server.start()

            logger.info("✅ Service Trader démarré")

        except Exception as e:
            logger.error(f"❌ Erreur critique lors du démarrage: {str(e)}")
            self.running = False
            raise

    def stop(self):
        """
        Arrête proprement le service Trader.
        """
        if not self.running:
            return

        logger.info("Arrêt du service Trader...")
        self.running = False

        # Arrêter l'API REST
        if self.api_server:
            self.api_server.stop()

        # Arrêter le gestionnaire d'ordres proprement
        if self.order_manager:
            self.order_manager.stop()
            self.order_manager = None

        logger.info("Service Trader terminé")


def parse_arguments():
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
    parser.add_argument(
        "--log-level",
        type=str,
        default=LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Niveau de journalisation",
    )
    return parser.parse_args()


def setup_signal_handlers(trader_service):
    """
    Configure les gestionnaires de signaux pour arrêter proprement le service.

    Args:
        trader_service: Instance du service Trader
    """

    def signal_handler(sig, frame):
        logger.info(f"Signal {sig} reçu, arrêt en cours...")
        trader_service.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Fonction principale du service Trader."""
    # Parser les arguments
    args = parse_arguments()

    # Configurer le logging
    setup_logging(args.log_level)

    # Configurer les symboles
    symbols = args.symbols.split(",") if args.symbols else SYMBOLS

    # Créer le service
    trader_service = TraderService(symbols=symbols, port=args.port)

    # Configurer les gestionnaires de signaux
    setup_signal_handlers(trader_service)

    try:
        # Démarrer le service
        trader_service.start()

        # Maintenir le programme en vie
        while trader_service.running:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur critique dans le service Trader: {str(e)}")
    finally:
        # Arrêter le service
        trader_service.stop()


if __name__ == "__main__":
    main()
