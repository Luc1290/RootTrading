"""
Module optimisé pour la gestion du pool de connexions à la base de données.
Améliore les performances et la résilience en cas d'erreurs de connexion.
"""
import logging
import time
import random
import threading
import queue
from typing import Optional, Any, Dict, Union, List, Tuple
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool, extensions, extras
from psycopg2.extras import RealDictCursor, DictCursor

# Importer la configuration
from shared.src.config import get_db_url, DB_MIN_CONNECTIONS, DB_MAX_CONNECTIONS
# Ajouter ce code après les imports dans db_pool.py
# Compatibilité pour différentes versions de psycopg2
if not hasattr(psycopg2.extensions, 'STATUS_READY'):
    psycopg2.extensions.STATUS_READY = 0  # pas de transaction en cours
    
if not hasattr(psycopg2.extensions, 'STATUS_INTRANS'):
    psycopg2.extensions.STATUS_INTRANS = 1  # transaction en cours
    
if not hasattr(psycopg2.extensions, 'STATUS_INERROR'):
    psycopg2.extensions.STATUS_INERROR = 2  # erreur dans la transaction

# Configuration du logging
logger = logging.getLogger(__name__)

# Augmenter les valeurs par défaut pour plus de résilience
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_DELAY = 0.5
DEFAULT_IDLE_TIMEOUT = 600  # 10 minutes avant de fermer les connexions inactives

class DBMetrics:
    """Classe pour collecter des métriques sur l'utilisation de la base de données."""
    
    def __init__(self):
        """Initialise les métriques."""
        self.query_count = 0
        self.transaction_count = 0
        self.error_count = 0
        self.total_duration = 0.0
        self.max_duration = 0.0
        self.last_error = None
        self.last_error_time = None
        self.last_query_time = None
        self.slow_queries = []  # Liste des 10 requêtes les plus lentes
        self.query_types = {}   # Compteur par type de requête
        
        # Protection thread
        self._lock = threading.RLock()
    
    def record_query(self, duration: float, query_type: str = None, query_text: str = None):
        """
        Enregistre une requête exécutée.
        
        Args:
            duration: Durée d'exécution en secondes
            query_type: Type de requête (SELECT, INSERT, etc.)
            query_text: Texte de la requête (pour debug)
        """
        with self._lock:
            self.query_count += 1
            self.total_duration += duration
            self.max_duration = max(self.max_duration, duration)
            self.last_query_time = time.time()
            
            # Enregistrer par type de requête
            if query_type:
                self.query_types[query_type] = self.query_types.get(query_type, 0) + 1
            
            # Enregistrer les requêtes lentes
            if duration > 0.1:  # 100ms
                query_info = {
                    'duration': duration,
                    'time': self.last_query_time,
                    'type': query_type,
                    'query': query_text[:200] if query_text else None
                }
                
                # Insérer la requête lente de manière triée
                if not self.slow_queries or duration > self.slow_queries[-1]['duration']:
                    self.slow_queries.append(query_info)
                    self.slow_queries.sort(key=lambda x: x['duration'], reverse=True)
                    
                    # Garder seulement les 10 plus lentes
                    if len(self.slow_queries) > 10:
                        self.slow_queries.pop()
    
    def record_transaction(self):
        """Enregistre une transaction exécutée."""
        with self._lock:
            self.transaction_count += 1
    
    def record_error(self, error: Exception, query_text: str = None):
        """
        Enregistre une erreur de base de données.
        
        Args:
            error: L'exception levée
            query_text: Texte de la requête qui a échoué
        """
        with self._lock:
            self.error_count += 1
            self.last_error = str(error)
            self.last_error_time = time.time()
            
            # Enregistrer le contexte de l'erreur
            if query_text:
                self.last_error = f"{self.last_error} (Query: {query_text[:200]})"
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Récupère les statistiques d'utilisation.
        
        Returns:
            Dictionnaire des statistiques
        """
        with self._lock:
            stats = {
                "query_count": self.query_count,
                "transaction_count": self.transaction_count,
                "error_count": self.error_count,
                "avg_duration": self.total_duration / max(1, self.query_count),
                "max_duration": self.max_duration,
                "last_error": self.last_error,
                "last_error_time": self.last_error_time,
                "last_query_time": self.last_query_time,
                "query_types": self.query_types,
                "slow_queries": self.slow_queries[:5]  # Top 5 des requêtes lentes
            }
            return stats
    
    def reset(self):
        """Réinitialise les métriques."""
        with self._lock:
            self.query_count = 0
            self.transaction_count = 0
            self.error_count = 0
            self.total_duration = 0.0
            # Ne pas réinitialiser max_duration, last_error, last_query_time pour l'historique
            self.query_types = {}
            # Garder les requêtes lentes pour l'historique

class ConnectionWrapper:
    """
    Wrapper autour d'une connexion pour suivre son utilisation.
    """
    def __init__(self, connection, pool):
        self.connection = connection
        self.pool = pool
        self.last_used = time.time()
        self.created = time.time()
        self.usage_count = 0
        self.transaction_count = 0
        self.in_use = False
        self.idle_timeout = DEFAULT_IDLE_TIMEOUT
    
    def check_health(self) -> bool:
        """
        Vérifie si la connexion est toujours valide.
        
        Returns:
            True si la connexion est saine, False sinon
        """
        try:
            if self.connection.closed:
                return False
                
            # Vérifier si la connexion n'est pas trop vieille ou n'a pas été utilisée depuis trop longtemps
            current_time = time.time()
            if (current_time - self.created > 3600) or (not self.in_use and current_time - self.last_used > self.idle_timeout):
                return False
                
            # Si pas en cours d'utilisation, tester avec un ping
            if not self.in_use:
                cursor = self.connection.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
            
            return True
        except Exception:
            return False
    
    def use(self):
        """Marque la connexion comme utilisée."""
        self.last_used = time.time()
        self.usage_count += 1
        self.in_use = True
    
    def release(self):
        """Marque la connexion comme libérée."""
        self.last_used = time.time()
        self.in_use = False
    
    def begin_transaction(self):
        """Marque le début d'une transaction."""
        self.transaction_count += 1
    
    def close(self):
        """Ferme la connexion."""
        if not self.connection.closed:
            self.connection.close()

class AdvancedConnectionPool:
    """
    Pool de connexions avancé avec gestion des erreurs et des reconnnexions.
    """
    def __init__(self, min_connections: int, max_connections: int, dsn: str):
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.dsn = dsn
        
        # Créer un pool pour les connexions disponibles
        self.available_connections = queue.Queue(maxsize=max_connections)
        
        # Dictionnaire des connexions en cours d'utilisation
        self.in_use_connections = {}
        
        # Verrou pour l'accès au pool
        self.lock = threading.RLock()
        
        # Compteur de connexions
        self.connection_count = 0
        
        # Initialiser le pool avec les connexions minimales
        self._initialize_pool()
        
        # Thread de surveillance
        self._start_monitoring_thread()
    
    def _initialize_pool(self):
        """Initialise le pool avec les connexions minimales."""
        with self.lock:
            for _ in range(self.min_connections):
                try:
                    conn = self._create_connection()
                    self.available_connections.put(conn)
                    self.connection_count += 1
                except Exception as e:
                    logger.error(f"Erreur lors de l'initialisation du pool: {str(e)}")
    
    def _create_connection(self) -> ConnectionWrapper:
        """
        Crée une nouvelle connexion.
        
        Returns:
            ConnectionWrapper contenant la connexion
        """
        try:
            connection = psycopg2.connect(dsn=self.dsn)
            connection.autocommit = True
            return ConnectionWrapper(connection, self)
        except Exception as e:
            logger.error(f"Erreur lors de la création d'une connexion: {str(e)}")
            raise
    
    def _start_monitoring_thread(self):
        """Démarre un thread pour nettoyer les connexions inactives."""
        def monitor_connections():
            while True:
                try:
                    # Vérifier toutes les minutes
                    time.sleep(60)
                    self._cleanup_connections()
                except Exception as e:
                    logger.error(f"Erreur dans le thread de surveillance: {str(e)}")
        
        thread = threading.Thread(target=monitor_connections, daemon=True)
        thread.start()
    
    def _cleanup_connections(self):
        """Nettoie les connexions inactives ou invalides."""
        with self.lock:
            # Récupérer toutes les connexions disponibles
            available_connections = []
            while not self.available_connections.empty():
                try:
                    conn = self.available_connections.get_nowait()
                    available_connections.append(conn)
                except queue.Empty:
                    break
            
            # Vérifier et remettre les connexions valides dans le pool
            for conn in available_connections:
                if conn.check_health():
                    self.available_connections.put(conn)
                else:
                    logger.info(f"Fermeture d'une connexion inactive (utilisée {conn.usage_count} fois)")
                    conn.close()
                    self.connection_count -= 1
            
            # Créer de nouvelles connexions si nécessaire
            while self.connection_count < self.min_connections:
                try:
                    conn = self._create_connection()
                    self.available_connections.put(conn)
                    self.connection_count += 1
                    logger.info("Création d'une nouvelle connexion pour maintenir le minimum")
                except Exception as e:
                    logger.error(f"Impossible de créer une connexion: {str(e)}")
                    break
    
    def getconn(self, timeout: float = 30.0) -> ConnectionWrapper:
        """
        Obtient une connexion du pool.
        
        Args:
            timeout: Timeout en secondes
            
        Returns:
            ConnectionWrapper contenant la connexion
            
        Raises:
            queue.Empty: Si aucune connexion n'est disponible dans le délai imparti
        """
        # Essayer d'obtenir une connexion existante
        try:
            conn = self.available_connections.get(timeout=timeout)
            
            # Vérifier si la connexion est valide
            if not conn.check_health():
                logger.info("Connexion invalide récupérée du pool, création d'une nouvelle")
                conn.close()
                self.connection_count -= 1
                conn = self._create_connection()
                self.connection_count += 1
            
            # Marquer la connexion comme utilisée
            conn.use()
            
            # Enregistrer la connexion comme en cours d'utilisation
            with self.lock:
                self.in_use_connections[id(conn)] = conn
            
            return conn
            
        except queue.Empty:
            # Aucune connexion disponible, en créer une nouvelle si possible
            with self.lock:
                if self.connection_count < self.max_connections:
                    try:
                        conn = self._create_connection()
                        conn.use()
                        self.connection_count += 1
                        self.in_use_connections[id(conn)] = conn
                        return conn
                    except Exception as e:
                        logger.error(f"Impossible de créer une nouvelle connexion: {str(e)}")
                        raise
            
            # Toutes les connexions sont utilisées et le maximum est atteint
            logger.error(f"Pool de connexions épuisé ({self.connection_count}/{self.max_connections})")
            raise queue.Empty("Connection pool exhausted")
    
    def putconn(self, conn: ConnectionWrapper):
        """
        Remet une connexion dans le pool.
        
        Args:
            conn: Connexion à remettre dans le pool
        """
        # Vérifier si la connexion est valide
        if not conn.check_health():
            with self.lock:
                # Fermer la connexion invalide
                conn.close()
                
                # Retirer de la liste des connexions en cours d'utilisation
                self.in_use_connections.pop(id(conn), None)
                
                # Décrémenter le compteur
                self.connection_count -= 1
                
                logger.info("Connexion invalide fermée lors de sa libération")
            return
        
        # S'assurer que autocommit est activé
        if not conn.connection.autocommit:
            # Vérifier s'il y a une transaction active et la rollback
            tx_status = conn.connection.get_transaction_status()
            if tx_status != 0:  # 0 = IDLE/READY (pas de transaction)
                conn.connection.rollback()
                logger.debug("Transaction nettoyée lors de la libération")
            
            # Remettre autocommit à True
            conn.connection.autocommit = True
        
        # Marquer la connexion comme libérée
        conn.release()
        
        # Retirer de la liste des connexions en cours d'utilisation
        with self.lock:
            self.in_use_connections.pop(id(conn), None)
        
        # Remettre la connexion dans le pool
        try:
            self.available_connections.put_nowait(conn)
        except queue.Full:
            # Si le pool est plein, fermer la connexion
            conn.close()
            with self.lock:
                self.connection_count -= 1
            logger.info("Connexion fermée car le pool est plein")
    
    def closeall(self):
        """Ferme toutes les connexions du pool."""
        # Fermer les connexions disponibles
        while not self.available_connections.empty():
            try:
                conn = self.available_connections.get_nowait()
                conn.close()
            except queue.Empty:
                break
        
        # Fermer les connexions en cours d'utilisation
        with self.lock:
            for conn_id, conn in list(self.in_use_connections.items()):
                conn.close()
            
            # Réinitialiser les compteurs
            self.in_use_connections = {}
            self.connection_count = 0
        
        logger.info("Toutes les connexions fermées")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Récupère des statistiques sur le pool.
        
        Returns:
            Dictionnaire des statistiques
        """
        with self.lock:
            available_count = self.available_connections.qsize()
            in_use_count = len(self.in_use_connections)
            
            # Calculer des statistiques d'utilisation
            usage_counts = [conn.usage_count for conn in self.in_use_connections.values()]
            avg_usage = sum(usage_counts) / max(1, len(usage_counts)) if usage_counts else 0
            
            stats = {
                "total_connections": self.connection_count,
                "available_connections": available_count,
                "in_use_connections": in_use_count,
                "min_connections": self.min_connections,
                "max_connections": self.max_connections,
                "usage_percent": (self.connection_count / self.max_connections) * 100 if self.max_connections > 0 else 0,
                "avg_usage_count": avg_usage
            }
            
            return stats

class DBConnectionPool:
    """Gestionnaire avancé de pool de connexions à la base de données."""
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """
        Obtient l'instance unique du pool.
        
        Returns:
            Instance DBConnectionPool
        """
        if cls._instance is None:
            cls._instance = DBConnectionPool()
        return cls._instance
    
    def __init__(self):
        """Initialise le pool de connexions."""
        # Créer le pool avec connexion améliorée
        self.connection_pool = AdvancedConnectionPool(
            DB_MIN_CONNECTIONS,
            DB_MAX_CONNECTIONS,
            get_db_url()
        )
        
        # Métriques
        self.metrics = DBMetrics()
        
        # Démarrer un thread de surveillance
        self._start_monitoring_thread()
        
        logger.info(f"✅ Pool de connexions initialisé ({DB_MIN_CONNECTIONS}-{DB_MAX_CONNECTIONS})")
    
    def _start_monitoring_thread(self):
        """Démarre un thread pour surveiller l'état du pool."""
        def monitor_pool():
            while True:
                try:
                    # Vérifier toutes les 30 minutes
                    time.sleep(1800)
                    
                    # Récupérer les statistiques
                    pool_stats = self.connection_pool.get_stats()
                    db_stats = self.metrics.get_stats()
                    
                    # Logguer les statistiques
                    logger.info(f"📊 DB Pool: {pool_stats['in_use_connections']}/{pool_stats['total_connections']} "
                                f"connexions utilisées ({pool_stats['usage_percent']:.1f}%)")
                    
                    logger.info(f"📊 DB Requêtes: {db_stats['query_count']} requêtes, "
                                f"{db_stats['transaction_count']} transactions, "
                                f"{db_stats['error_count']} erreurs, "
                                f"durée moyenne {db_stats['avg_duration']:.3f}s")
                    
                    # Logguer les requêtes lentes si présentes
                    if db_stats['slow_queries']:
                        logger.warning(f"⚠️ Top requêtes lentes: " + 
                                      ", ".join([f"{q['type']} ({q['duration']:.3f}s)" for q in db_stats['slow_queries'][:3]]))
                    
                    # Réinitialiser certaines métriques
                    self.metrics.reset()
                    
                except Exception as e:
                    logger.error(f"Erreur dans le thread de surveillance du pool: {str(e)}")
        
        thread = threading.Thread(target=monitor_pool, daemon=True)
        thread.start()
    
    def get_connection(self, max_retries=DEFAULT_MAX_RETRIES, retry_delay=DEFAULT_RETRY_DELAY):
        """
        Obtient une connexion du pool avec retry et backoff exponentiel.
        
        Args:
            max_retries: Nombre maximum de tentatives
            retry_delay: Délai initial entre les tentatives
            
        Returns:
            Connexion à la base de données
        """
        attempt = 0
        last_error = None
        
        while attempt <= max_retries:
            try:
                # Obtenir une connexion
                conn_wrapper = self.connection_pool.getconn()
                return conn_wrapper.connection
                
            except Exception as e:
                attempt += 1
                last_error = e
                self.metrics.record_error(e)
                
                if "connection pool exhausted" in str(e):
                    if attempt < max_retries:
                        # Calculer un délai avec jitter pour éviter la tempête de requêtes
                        wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 0.1)
                        logger.warning(f"⚠️ Pool de connexions épuisé (attempt {attempt}/{max_retries}), "
                                      f"attente de {wait_time:.2f}s")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Logguer des informations de diagnostic
                        logger.critical(f"🔥 Pool de connexions épuisé après {max_retries} tentatives")
                        try:
                            pool_stats = self.connection_pool.get_stats()
                            logger.critical(f"Diagnostic: {pool_stats['in_use_connections']}/{pool_stats['total_connections']} "
                                          f"connexions utilisées ({pool_stats['usage_percent']:.1f}%)")
                        except:
                            pass
                
                logger.error(f"❌ Erreur lors de l'obtention d'une connexion: {str(e)}")
                
                if attempt >= max_retries:
                    break
                    
                # Délai exponentiel
                wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 0.1)
                time.sleep(wait_time)
        
        logger.error(f"❌ Échec après {max_retries} tentatives: {str(last_error)}")
        raise last_error
    
    def release_connection(self, conn):
        """
        Libère une connexion et la remet dans le pool.
        
        Args:
            conn: Connexion à libérer
        """
        if conn is None:
            return
            
        try:
            # Trouver le wrapper associé à cette connexion
            conn_wrapper = None
            
            for wrapper in self.connection_pool.in_use_connections.values():
                if wrapper.connection is conn:
                    conn_wrapper = wrapper
                    break
            
            if conn_wrapper:
                # Vérifier si une transaction est encore en cours et la rollback
                if not conn.closed and conn.get_transaction_status() != 0:  # 0 = STATUS_READY
                    conn.rollback()
                    logger.debug("Transaction nettoyée lors de la libération")
                
                # Remettre autocommit à True
                conn.autocommit = True
                
                # Rendre la connexion au pool
                self.connection_pool.putconn(conn_wrapper)
            else:
                logger.warning("Tentative de libération d'une connexion inconnue")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la libération d'une connexion: {str(e)}")
            self.metrics.record_error(e)
            
            # Essayer de fermer la connexion si possible
            try:
                if conn and not conn.closed:
                    conn.close()
            except:
                pass
    
    def close(self):
        """Ferme toutes les connexions du pool."""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("Pool de connexions fermé")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Récupère des informations de diagnostic sur le pool.
        
        Returns:
            Dictionnaire d'informations de diagnostic
        """
        pool_stats = self.connection_pool.get_stats()
        db_stats = self.metrics.get_stats()
        
        return {
            "pool": pool_stats,
            "metrics": db_stats
        }

class DBContextManager:
    """Gestionnaire de contexte pour utiliser une connexion du pool."""
    
    def __init__(self, auto_transaction=True, max_retries=DEFAULT_MAX_RETRIES, cursor_factory=None):
        """
        Initialise le gestionnaire de contexte.
        
        Args:
            auto_transaction: Si True, démarre une transaction automatiquement
            max_retries: Nombre maximum de tentatives pour obtenir une connexion
            cursor_factory: Type de curseur à utiliser (DictCursor, RealDictCursor, etc.)
        """
        self.pool = DBConnectionPool.get_instance()
        self.conn = None
        self.cursor = None
        self.auto_transaction = auto_transaction
        self.max_retries = max_retries
        self.cursor_factory = cursor_factory
        self.start_time = None
        self.query_type = None
        self.query_text = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.conn = self.pool.get_connection(max_retries=self.max_retries)
        
        # Vérifier et nettoyer toute transaction résiduelle
        if self.conn.get_transaction_status() != extensions.STATUS_READY:
            logger.debug("Transaction résiduelle nettoyée")
            self.conn.rollback()
        
        # Configurer la gestion de transaction
        if self.auto_transaction:
            self.conn.autocommit = False
        
        # Créer un curseur avec le factory spécifié
        cursor_args = {}
        if self.cursor_factory:
            cursor_args['cursor_factory'] = self.cursor_factory
        
        self.cursor = self.conn.cursor(**cursor_args)
        return self.cursor
    
    def execute(self, query, params=None):
        """
        Exécute une requête et enregistre son type.
        
        Args:
            query: Requête SQL
            params: Paramètres pour la requête
            
        Returns:
            Résultat de l'exécution
        """
        # Détecter le type de requête (SELECT, INSERT, etc.)
        query_start = query.strip().upper()[:10]
        if "SELECT" in query_start:
            self.query_type = "SELECT"
        elif "INSERT" in query_start:
            self.query_type = "INSERT"
        elif "UPDATE" in query_start:
            self.query_type = "UPDATE"
        elif "DELETE" in query_start:
            self.query_type = "DELETE"
        else:
            self.query_type = "OTHER"
        
        self.query_text = query
        
        return self.cursor.execute(query, params)
    
    def commit(self):
        """Valide la transaction en cours."""
        if self.conn and not self.conn.closed:
            if self.conn.get_transaction_status() == 1:  # 1 = STATUS_INTRANS
                self.conn.commit()
                # Enregistrer la transaction dans les métriques
                self.pool.metrics.record_transaction()
                logger.debug("Transaction validée explicitement")
            else:
                logger.debug("Aucune transaction active à valider")

    def rollback(self):
        """Annule la transaction en cours."""
        if self.conn and not self.conn.closed:
            tx_status = self.conn.get_transaction_status()
            if tx_status in (1, 2):  # 1 = STATUS_INTRANS, 2 = STATUS_INERROR
                self.conn.rollback()
                logger.debug("Transaction annulée explicitement")
            else:
                logger.debug("Aucune transaction active à annuler")
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        
        # Enregistrer l'exécution dans les métriques
        if exc_type:
            self.pool.metrics.record_error(exc_val, self.query_text)
        else:
            self.pool.metrics.record_query(duration, self.query_type, self.query_text)
        
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
                            # Enregistrer la transaction dans les métriques
                            self.pool.metrics.record_transaction()
                            logger.debug("Transaction validée automatiquement")
                    else:
                        # Erreur → rollback si nécessaire
                        if status in (extensions.STATUS_INTRANS, extensions.STATUS_INERROR):
                            self.conn.rollback()
                            logger.debug(f"Transaction annulée automatiquement suite à {exc_type.__name__}: {exc_val}")
                else:
                    # En mode sans auto_transaction, vérifier qu'aucune transaction n'est encore active
                    if status in (extensions.STATUS_INTRANS, extensions.STATUS_INERROR):
                        logger.debug("Transaction non terminée nettoyée")
                        self.conn.rollback()
        finally:
            # 3) toujours s'assurer qu'aucune transaction n'est active avant de changer autocommit
            if self.conn and not self.conn.closed:
                # Vérifier qu'il n'y a plus de transactions actives 
                if self.conn.get_transaction_status() != extensions.STATUS_READY:
                    logger.debug("Nettoyage de transaction avant libération")
                    self.conn.rollback()
                
                # Maintenant on peut changer autocommit en toute sécurité
                self.conn.autocommit = True
                self.pool.release_connection(self.conn)
            
        # Logguer les requêtes lentes (plus de 500ms)
        if duration > 0.5:
            logger.warning(f"⚠️ Requête SQL lente ({self.query_type}): {duration:.3f}s")

# Helper contextmanager pour les transactions explicites
@contextmanager
def transaction(cursor_factory=None):
    """
    Gestionnaire de contexte pour exécuter du code dans une transaction.
    Assure que la transaction est correctement validée ou annulée.
    
    Args:
        cursor_factory: Type de curseur à utiliser (DictCursor, RealDictCursor, etc.)
    
    Exemple:
        with transaction() as cursor:
            cursor.execute("INSERT INTO...")
            cursor.execute("UPDATE...")
    """
    db_ctx = None
    try:
        # Create connection with explicit transaction mode
        db_ctx = DBContextManager(auto_transaction=True, cursor_factory=cursor_factory)
        cursor = db_ctx.__enter__()
        
        # Vérifier que la transaction est bien démarrée
        if cursor.connection.get_transaction_status() != extensions.STATUS_INTRANS:
            # S'assurer qu'aucune transaction n'est active
            if cursor.connection.get_transaction_status() != extensions.STATUS_READY:
                cursor.connection.rollback()
                logger.debug("Transaction nettoyée avant démarrage")
            
            # Démarrer une nouvelle transaction propre
            cursor.connection.autocommit = False
        
        yield cursor
        
        # Commit changes if no exception occurred
        if cursor and cursor.connection and not cursor.connection.closed:
            cursor.connection.commit()
            logger.debug("Transaction committed")
            
    except Exception as e:
        # Rollback on exception
        logger.error(f"❌ Transaction error: {str(e)}")
        if cursor and cursor.connection and not cursor.connection.closed:
            cursor.connection.rollback()
            logger.debug("Transaction rolled back due to error")
        raise
    
    finally:
        # Clean up resources
        if db_ctx:
            db_ctx.__exit__(None, None, None)

# Helper pour les requêtes avec DictCursor
@contextmanager
def dict_cursor(auto_transaction=False):
    """
    Gestionnaire de contexte pour utiliser un DictCursor.
    
    Args:
        auto_transaction: Si True, démarre une transaction automatiquement
    
    Exemple:
        with dict_cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            user = cursor.fetchone()  # Retourne un dictionnaire
    """
    db = DBContextManager(auto_transaction=auto_transaction, cursor_factory=DictCursor)
    try:
        cursor = db.__enter__()
        yield cursor
    finally:
        db.__exit__(None, None, None)

# Helper pour les requêtes avec RealDictCursor
@contextmanager
def real_dict_cursor(auto_transaction=False):
    """
    Gestionnaire de contexte pour utiliser un RealDictCursor.
    
    Args:
        auto_transaction: Si True, démarre une transaction automatiquement
    
    Exemple:
        with real_dict_cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            user = cursor.fetchone()  # Retourne un dictionnaire avec les noms de colonnes préservés
    """
    db = DBContextManager(auto_transaction=auto_transaction, cursor_factory=RealDictCursor)
    try:
        cursor = db.__enter__()
        yield cursor
    finally:
        db.__exit__(None, None, None)

# Fonctions utilitaires pour les requêtes communes

def fetch_one(query, params=None, dict_result=True):
    """
    Exécute une requête et retourne une seule ligne.
    
    Args:
        query: Requête SQL
        params: Paramètres pour la requête
        dict_result: Si True, retourne un dictionnaire
        
    Returns:
        Une ligne de résultat ou None
    """
    cursor_factory = DictCursor if dict_result else None
    
    with DBContextManager(auto_transaction=False, cursor_factory=cursor_factory) as cursor:
        cursor.execute(query, params)
        return cursor.fetchone()

def fetch_all(query, params=None, dict_result=True):
    """
    Exécute une requête et retourne toutes les lignes.
    
    Args:
        query: Requête SQL
        params: Paramètres pour la requête
        dict_result: Si True, retourne une liste de dictionnaires
        
    Returns:
        Liste des résultats
    """
    cursor_factory = DictCursor if dict_result else None
    
    with DBContextManager(auto_transaction=False, cursor_factory=cursor_factory) as cursor:
        cursor.execute(query, params)
        return cursor.fetchall()

def execute(query, params=None, auto_transaction=True):
    """
    Exécute une requête sans retourner de résultat.
    
    Args:
        query: Requête SQL
        params: Paramètres pour la requête
        auto_transaction: Si True, démarre une transaction automatiquement
        
    Returns:
        Nombre de lignes affectées
    """
    with DBContextManager(auto_transaction=auto_transaction) as cursor:
        cursor.execute(query, params)
        return cursor.rowcount

def execute_batch(query, params_list, auto_transaction=True, page_size=100):
    """
    Exécute une requête en batch.
    
    Args:
        query: Requête SQL
        params_list: Liste de paramètres pour la requête
        auto_transaction: Si True, démarre une transaction automatiquement
        page_size: Nombre d'opérations par page
        
    Returns:
        Nombre de lignes affectées
    """
    with DBContextManager(auto_transaction=auto_transaction) as cursor:
        extras.execute_batch(cursor, query, params_list, page_size=page_size)
        return cursor.rowcount

def execute_values(query, values, auto_transaction=True, page_size=100):
    """
    Exécute une requête INSERT avec des valeurs en masse.
    
    Args:
        query: Requête SQL de base (ex: "INSERT INTO table (col1, col2) VALUES %s")
        values: Liste de tuples de valeurs
        auto_transaction: Si True, démarre une transaction automatiquement
        page_size: Nombre d'opérations par page
        
    Returns:
        Nombre de lignes affectées
    """
    with DBContextManager(auto_transaction=auto_transaction) as cursor:
        extras.execute_values(cursor, query, values, page_size=page_size)
        return cursor.rowcount

# Accès simplifié aux métriques de la base de données
def get_db_metrics():
    """
    Récupère les métriques de la base de données.
    
    Returns:
        Dictionnaire des métriques
    """
    pool = DBConnectionPool.get_instance()
    return pool.get_diagnostics()