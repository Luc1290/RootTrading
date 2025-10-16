"""
Configuration du logging pour le service Portfolio.
"""

import logging
import sys
from datetime import datetime


def setup_logging(log_level="INFO"):
    """
    Configure le système de logging pour le service Portfolio.

    Args:
        log_level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Configuration du format de log
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Configuration du logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"portfolio_{datetime.now(tz=timezone.utc).strftime('%Y%m%d')}.log"),
        ],
    )

    # Réduire la verbosité des loggers externes
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Logger principal
    logger = logging.getLogger("portfolio")
    logger.info(f"📝 Logging configuré au niveau {log_level}")
