"""
Serveur API REST pour le trader.
Expose des endpoints pour interagir avec le système de trading.
"""

import logging
import threading
import psutil  # type: ignore[import-untyped]
import os
from flask import Flask
from typing import Optional

from trader.src.trading.order_manager import OrderManager
from trader.src.api.routes import register_routes

# Configuration du logging
logger = logging.getLogger(__name__)


class RestApiServer:
    """
    Serveur API REST pour le trader.
    Expose des endpoints pour interagir avec le système de trading.
    """

    def __init__(self, order_manager: OrderManager, port: int = 5002):
        """
        Initialise le serveur API REST.

        Args:
            order_manager: Gestionnaire d'ordres
            port: Port d'écoute
        """
        self.order_manager = order_manager
        self.port = port
        self.process = psutil.Process(os.getpid())
        self.app = Flask(__name__)
        self.api_thread: Optional[threading.Thread] = None

        # Configurer les routes via le module routes.py
        register_routes(self.app, self.order_manager)

        logger.info(f"✅ RestApiServer initialisé sur le port {port}")

    def start(self):
        """
        Démarre le serveur API Flask dans un thread séparé.
        """
        if self.api_thread is not None and self.api_thread.is_alive():
            logger.warning("Le serveur API est déjà en cours d'exécution")
            return

        def run_flask():
            self.app.run(
                host="0.0.0.0", port=self.port, debug=False, use_reloader=False
            )

        self.api_thread = threading.Thread(target=run_flask, daemon=True)
        self.api_thread.start()

        logger.info(f"✅ API REST démarrée sur le port {self.port}")

    def stop(self):
        """
        Arrête le serveur API Flask.
        """
        # Flask n'a pas de méthode d'arrêt propre
        # L'arrêt se fait généralement en arrêtant le thread parent
        logger.info("Arrêt du serveur API REST")
