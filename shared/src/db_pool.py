"""
Module de gestion du pool de connexions à la base de données.
Fournit un accès centralisé à la base de données avec un pool de connexions.
"""
import logging
import psycopg2
from psycopg2 import pool
from typing import Optional, Any, Dict

# Importer la configuration
from shared.src.config import get_db_url, DB_MIN_CONNECTIONS, DB_MAX_CONNECTIONS

# Configuration du logging
logger = logging.getLogger(__name__)

class DBConnectionPool:
    """Gestionnaire de pool de connexions à la base de données."""
    _instance = None
    
    @classmethod
    def get_instance(cls: List[Any]) -> Any:
        # Validation des paramètres
        if cls is not None and not isinstance(cls, list):
            raise TypeError(f"cls doit être une liste, pas {type(cls).__name__}")
        if cls._instance is None:
            cls._instance = DBConnectionPool()
        return cls._instance
    
    def __init__(self) -> None:
        self.connection_pool = pool.ThreadedConnectionPool(
            DB_MIN_CONNECTIONS,
            DB_MAX_CONNECTIONS,
            get_db_url()
        )
        logger.info(f"✅ Pool de connexions initialisé ({DB_MIN_CONNECTIONS}-{DB_MAX_CONNECTIONS})")
    
    def get_connection(self) -> Any:
        return self.connection_pool.getconn()
    
    def release_connection(self, conn: Any) -> Any:
        self.connection_pool.putconn(conn)
    
    def close(self) -> None:
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Pool de connexions fermé")

class DBContextManager:
    """Gestionnaire de contexte pour utiliser une connexion du pool."""
    
    def __init__(self) -> None:
        self.pool = DBConnectionPool.get_instance()
        self.conn = None
        self.cursor = None
    
    def __enter__(self) -> Any:
        self.conn = self.pool.get_connection()
        self.cursor = self.conn.cursor()
        return self.cursor
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        if self.cursor:
            self.cursor.close()
        
        if self.conn:
            if exc_type is not None:
                self.conn.rollback()
            else:
                self.conn.commit()
            
            self.pool.release_connection(self.conn)