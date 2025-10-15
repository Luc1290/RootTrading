"""
Configuration du logging pour le trader.
"""

import logging
import sys
import os
from logging.handlers import RotatingFileHandler


def setup_logging(log_level_str="INFO"):
    """
    Configure le système de logging pour le trader.

    Args:
        log_level_str: Niveau de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Convertir le niveau de log en constante
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    # Configurer le logger principal
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Supprimer les handlers existants
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Créer un formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Configurer la sortie console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Créer le dossier logs s'il n'existe pas
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Configurer la sortie fichier
    file_handler = RotatingFileHandler(
        "logs/trader.log",
        maxBytes=10 * 1024 * 1024,  # 10 Mo
        backupCount=5,  # 5 fichiers de backup
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Configurer le fichier d'erreurs séparé
    error_handler = RotatingFileHandler(
        "logs/error.log",
        maxBytes=10 * 1024 * 1024,  # 10 Mo
        backupCount=5,  # 5 fichiers de backup
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    # Désactiver les logs des bibliothèques tierces trop verbeuses
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    # Logger les informations de configuration
    logger = logging.getLogger("trader")
    logger.info(f"Niveau de log configuré à {log_level_str}")
    logger.info(f"Logs enregistrés dans le dossier {os.path.abspath(logs_dir)}")


def get_logger(name):
    """
    Récupère un logger avec le nom spécifié.

    Args:
        name: Nom du logger

    Returns:
        Logger configuré
    """
    return logging.getLogger(name)
