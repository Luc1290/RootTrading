"""
Configuration centralisée des logs pour ROOT Trading
- Rotation automatique (10 MB max par fichier)
- Conservation de 5 fichiers de backup
- Logs séparés par service
- Logs Docker limités à 10 MB
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(service_name: str, log_level: str = "INFO") -> logging.Logger:
    """
    Configure le logging avec rotation automatique.
    
    Args:
        service_name: Nom du service (trader, analyzer, gateway, etc.)
        log_level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Logger configuré
    """
    # Convertir le niveau
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Logger principal du service
    logger = logging.getLogger(service_name)
    logger.setLevel(level)
    
    # Éviter les doublons
    if logger.handlers:
        return logger
    
    # Format des logs
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # === Console Handler (STDOUT) ===
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # === File Handlers (avec rotation) ===
    # Créer le dossier logs
    logs_dir = Path("/app/logs") if Path("/app").exists() else Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Logs généraux du service
    file_handler = RotatingFileHandler(
        filename=logs_dir / f"{service_name}.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Logs d'erreurs séparés
    error_handler = RotatingFileHandler(
        filename=logs_dir / f"{service_name}_errors.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    # Réduire le bruit des libs tierces
    for noisy_logger in ["urllib3", "requests", "werkzeug", "asyncio"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    
    logger.info(f"✓ Logging configuré: {service_name} [{log_level}]")
    logger.info(f"✓ Logs sauvegardés dans: {logs_dir.absolute()}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Récupère un logger enfant."""
    return logging.getLogger(name)
