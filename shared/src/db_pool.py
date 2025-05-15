"""
Module de gestion du pool de connexions à la base de données.
Fournit un accès centralisé à la base de données avec un pool de connexions.
"""
import logging
import psycopg2
from psycopg2 import pool, extensions
from typing import Optional, Any, Dict, Union
from contextlib import contextmanager

# Importer la configuration
from shared.src.config import get_db_url, DB_MIN_CONNECTIONS, DB_MAX_CONNECTIONS

# Configuration du logging
logger = logging.getLogger(__name__)

class DBConnectionPool:
    """Gestionnaire de pool de connexions à la base de données."""
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = DBConnectionPool()
        return cls._instance
    
    def __init__(self):
        self.connection_pool = pool.ThreadedConnectionPool(
            DB_MIN_CONNECTIONS,
            DB_MAX_CONNECTIONS,
            get_db_url()
        )
        logger.info(f"✅ Pool de connexions initialisé ({DB_MIN_CONNECTIONS}-{DB_MAX_CONNECTIONS})")
    
    def get_connection(self):
        try:
            conn = self.connection_pool.getconn()
            # S'assurer que autocommit est True par défaut pour éviter les transactions involontaires
            conn.autocommit = True
            return conn
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'obtention d'une connexion: {str(e)}")
            raise
    
    def release_connection(self, conn):
        try:
            # Vérifier si une transaction est encore en cours et la rollback
            if conn and not conn.closed:
                # S'assurer que toute transaction abandonnée est annulée
                if conn.get_transaction_status() != extensions.STATUS_READY:
                    conn.rollback()
                    logger.warning("Transaction abandonnée rollbackée lors de la libération de la connexion")
                
                # Remettre autocommit à True
                conn.autocommit = True
                
                # Rendre la connexion au pool
                self.connection_pool.putconn(conn)
        except Exception as e:
            logger.error(f"❌ Erreur lors de la libération d'une connexion: {str(e)}")
            # En cas d'erreur, on essaie quand même de rendre la connexion
            try:
                if conn and not conn.closed:
                    self.connection_pool.putconn(conn)
            except:
                pass
    
    def close(self):
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Pool de connexions fermé")

class DBContextManager:
    """Gestionnaire de contexte pour utiliser une connexion du pool."""
    
    def __init__(self, auto_transaction=True):
        """
        Initialise le gestionnaire de contexte.
        
        Args:
            auto_transaction: Si True, démarre une transaction automatiquement
        """
        self.pool = DBConnectionPool.get_instance()
        self.conn = None
        self.cursor = None
        self.auto_transaction = auto_transaction
    
    def __enter__(self):
        self.conn = self.pool.get_connection()
        
        # Vérifier et nettoyer toute transaction résiduelle
        if self.conn.get_transaction_status() != extensions.STATUS_READY:
            logger.warning("Transaction résiduelle détectée, rollback automatique")
            self.conn.rollback()
        
        # Configurer la gestion de transaction
        if self.auto_transaction:
            self.conn.autocommit = False
        
        self.cursor = self.conn.cursor()
        return self.cursor
    
    def commit(self):
        """Valide la transaction en cours"""
        if self.conn and not self.conn.closed:
            if self.conn.get_transaction_status() == extensions.STATUS_INTRANS:
                self.conn.commit()
                logger.debug("Transaction validée explicitement")
            else:
                logger.debug("Aucune transaction active à valider")

    def rollback(self):
        """Annule la transaction en cours"""
        if self.conn and not self.conn.closed:
            if self.conn.get_transaction_status() in (extensions.STATUS_INTRANS, extensions.STATUS_INERROR):
                self.conn.rollback()
                logger.debug("Transaction annulée explicitement")
            else:
                logger.debug("Aucune transaction active à annuler")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            # 1) fermer le curseur s'il existe encore
            if self.cursor and not self.cursor.closed:
                self.cursor.close()

            # 2) gérer la transaction si nécessaire
            if self.conn and not self.conn.closed:
                status = self.conn.get_transaction_status()
                
                if self.auto_transaction:
                    if exc_type is None:
                        # Pas d'erreur → commit si transaction active
                        if status == extensions.STATUS_INTRANS:
                            self.conn.commit()
                            logger.debug("Transaction validée automatiquement")
                    else:
                        # Erreur → rollback si nécessaire
                        if status in (extensions.STATUS_INTRANS, extensions.STATUS_INERROR):
                            self.conn.rollback()
                            logger.debug(f"Transaction annulée automatiquement suite à {exc_type.__name__}: {exc_val}")
                else:
                    # En mode sans auto_transaction, vérifier qu'aucune transaction n'est encore active
                    if status in (extensions.STATUS_INTRANS, extensions.STATUS_INERROR):
                        logger.warning("Transaction non terminée détectée, rollback forcé")
                        self.conn.rollback()
        finally:
            # 3) toujours remettre autocommit et libérer la connexion
            if self.conn and not self.conn.closed:
                self.conn.autocommit = True
                self.pool.release_connection(self.conn)

# Helper contextmanager pour les transactions explicites
@contextmanager
def transaction():
    """
    Gestionnaire de contexte pour exécuter du code dans une transaction.
    Assure que la transaction est correctement validée ou annulée.
    
    Exemple:
        with transaction() as cursor:
            cursor.execute("INSERT INTO...")
            cursor.execute("UPDATE...")
    """
    db = DBContextManager(auto_transaction=True)
    try:
        cursor = db.__enter__()
        yield cursor
        db.commit()  # commit explicite si tout s'est bien passé
    except Exception as e:
        logger.error(f"❌ Erreur dans la transaction: {str(e)}")
        db.rollback()  # rollback explicite en cas d'erreur
        raise
    finally:
        db.__exit__(None, None, None)  # nettoyage