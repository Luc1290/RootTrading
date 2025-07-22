"""
Utilitaires base de données partagés pour éviter la duplication de code
"""

import asyncpg
import logging
import os
from typing import Dict, Optional, Any, List, Callable
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


def get_db_config() -> Dict[str, Any]:
    """
    Récupère la configuration de base de données depuis les variables d'environnement
    Utilise les variables PostgreSQL standard (PGHOST, PGPORT, etc.)
    
    Returns:
        Configuration de base de données
    """
    return {
        'host': os.getenv('PGHOST', os.getenv('DB_HOST', 'localhost')),
        'port': int(os.getenv('PGPORT', os.getenv('DB_PORT', '5432'))),
        'database': os.getenv('PGDATABASE', os.getenv('DB_NAME', 'trading')),
        'user': os.getenv('PGUSER', os.getenv('DB_USER', 'postgres')),
        'password': os.getenv('PGPASSWORD', os.getenv('DB_PASSWORD', 'postgres'))
    }


class DatabasePoolManager:
    """Gestionnaire centralisé des pools de connexion PostgreSQL"""
    
    _pools: Dict[str, asyncpg.Pool] = {}
    
    @classmethod
    async def create_pool(cls, service_name: str, min_size: int = 1, max_size: int = 3, 
                         command_timeout: int = 10, **kwargs) -> asyncpg.Pool:
        """
        Crée ou récupère un pool de connexions pour un service donné
        
        Args:
            service_name: Nom du service (utilisé pour l'identification)
            min_size: Nombre minimum de connexions
            max_size: Nombre maximum de connexions
            command_timeout: Timeout des commandes en secondes
            **kwargs: Paramètres supplémentaires pour asyncpg.create_pool
            
        Returns:
            Pool de connexions PostgreSQL
        """
        if service_name in cls._pools and not cls._pools[service_name].is_closing():
            return cls._pools[service_name]
        
        try:
            db_config = get_db_config()
            db_config.update(kwargs)  # Permettre l'override
            
            pool = await asyncpg.create_pool(
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['database'],
                user=db_config['user'],
                password=db_config['password'],
                min_size=min_size,
                max_size=max_size,
                command_timeout=command_timeout,
                server_settings={
                    'application_name': f'{service_name}_pool'
                }
            )
            
            cls._pools[service_name] = pool
            logger.info(f"✅ Pool de connexions DB créé pour {service_name} "
                       f"(min: {min_size}, max: {max_size})")
            return pool
            
        except Exception as e:
            logger.error(f"❌ Erreur création pool DB pour {service_name}: {e}")
            raise
    
    @classmethod
    async def get_pool(cls, service_name: str) -> Optional[asyncpg.Pool]:
        """
        Récupère un pool existant
        
        Args:
            service_name: Nom du service
            
        Returns:
            Pool de connexions ou None si inexistant
        """
        pool = cls._pools.get(service_name)
        if pool and not pool.is_closing():
            return pool
        return None
    
    @classmethod
    async def close_pool(cls, service_name: str):
        """
        Ferme un pool de connexions spécifique
        
        Args:
            service_name: Nom du service
        """
        if service_name in cls._pools:
            try:
                await cls._pools[service_name].close()
                del cls._pools[service_name]
                logger.info(f"🔒 Pool DB fermé pour {service_name}")
            except Exception as e:
                logger.error(f"Erreur fermeture pool DB {service_name}: {e}")
    
    @classmethod
    async def close_all_pools(cls):
        """Ferme tous les pools de connexion"""
        for service_name in list(cls._pools.keys()):
            await cls.close_pool(service_name)


class DatabaseUtils:
    """Utilitaires de base de données avec patterns partagés"""
    
    @staticmethod
    @asynccontextmanager
    async def get_connection(pool: asyncpg.Pool):
        """
        Context manager pour récupérer une connexion du pool
        
        Args:
            pool: Pool de connexions
            
        Yields:
            Connexion de base de données
        """
        async with pool.acquire() as connection:
            try:
                yield connection
            except Exception as e:
                # Rollback automatique en cas d'erreur
                try:
                    await connection.rollback()
                except:
                    pass  # Ignorer les erreurs de rollback
                raise e
    
    @staticmethod
    @asynccontextmanager
    async def get_transaction(pool: asyncpg.Pool):
        """
        Context manager pour une transaction complète
        
        Args:
            pool: Pool de connexions
            
        Yields:
            Connexion de base de données en transaction
        """
        async with DatabaseUtils.get_connection(pool) as connection:
            async with connection.transaction():
                yield connection
    
    @staticmethod
    async def execute_query(pool: asyncpg.Pool, query: str, *args) -> List[Dict]:
        """
        Exécute une requête SELECT et retourne les résultats
        
        Args:
            pool: Pool de connexions
            query: Requête SQL
            *args: Paramètres de la requête
            
        Returns:
            Liste des résultats sous forme de dictionnaires
        """
        try:
            async with DatabaseUtils.get_connection(pool) as connection:
                rows = await connection.fetch(query, *args)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Erreur exécution requête: {e}")
            logger.error(f"Requête: {query}")
            logger.error(f"Paramètres: {args}")
            raise
    
    @staticmethod
    async def execute_query_single(pool: asyncpg.Pool, query: str, *args) -> Optional[Dict]:
        """
        Exécute une requête SELECT et retourne un seul résultat
        
        Args:
            pool: Pool de connexions
            query: Requête SQL
            *args: Paramètres de la requête
            
        Returns:
            Premier résultat sous forme de dictionnaire ou None
        """
        results = await DatabaseUtils.execute_query(pool, query, *args)
        return results[0] if results else None
    
    @staticmethod
    async def execute_command(pool: asyncpg.Pool, command: str, *args) -> str:
        """
        Exécute une commande (INSERT, UPDATE, DELETE)
        
        Args:
            pool: Pool de connexions
            command: Commande SQL
            *args: Paramètres de la commande
            
        Returns:
            Status de la commande
        """
        try:
            async with DatabaseUtils.get_connection(pool) as connection:
                return await connection.execute(command, *args)
        except Exception as e:
            logger.error(f"Erreur exécution commande: {e}")
            logger.error(f"Commande: {command}")
            logger.error(f"Paramètres: {args}")
            raise
    
    @staticmethod
    async def batch_insert(pool: asyncpg.Pool, table: str, columns: List[str], 
                          data: List[List[Any]], on_conflict: str = None) -> int:
        """
        Insertion en lot optimisée
        
        Args:
            pool: Pool de connexions
            table: Nom de la table
            columns: Liste des colonnes
            data: Données à insérer (liste de listes)
            on_conflict: Clause ON CONFLICT (optionnel)
            
        Returns:
            Nombre de lignes insérées
        """
        if not data:
            return 0
        
        try:
            placeholders = ', '.join([f'${i+1}' for i in range(len(columns))])
            conflict_clause = f' {on_conflict}' if on_conflict else ''
            
            query = f"""
                INSERT INTO {table} ({', '.join(columns)})
                VALUES ({placeholders}){conflict_clause}
            """
            
            async with DatabaseUtils.get_transaction(pool) as connection:
                await connection.executemany(query, data)
                return len(data)
                
        except Exception as e:
            logger.error(f"Erreur insertion en lot: {e}")
            logger.error(f"Table: {table}, Colonnes: {columns}")
            raise
    
    @staticmethod
    async def health_check(pool: asyncpg.Pool) -> bool:
        """
        Vérifie la santé de la connexion à la base de données
        
        Args:
            pool: Pool de connexions
            
        Returns:
            True si la connexion fonctionne
        """
        try:
            result = await DatabaseUtils.execute_query_single(pool, "SELECT 1 as health")
            return result and result.get('health') == 1
        except Exception as e:
            logger.error(f"❌ Health check DB échoué: {e}")
            return False


class PerformanceTrackingDB:
    """Utilitaires spécialisés pour le tracking des performances"""
    
    @staticmethod
    async def save_strategy_performance(pool: asyncpg.Pool, strategy: str, 
                                      performance_data: Dict[str, Any]):
        """
        Sauvegarde les performances d'une stratégie
        
        Args:
            pool: Pool de connexions
            strategy: Nom de la stratégie
            performance_data: Données de performance
        """
        query = """
            INSERT INTO strategy_performance 
            (strategy_name, win_rate, total_trades, avg_return, last_update, data)
            VALUES ($1, $2, $3, $4, NOW(), $5)
            ON CONFLICT (strategy_name) 
            DO UPDATE SET 
                win_rate = EXCLUDED.win_rate,
                total_trades = EXCLUDED.total_trades,
                avg_return = EXCLUDED.avg_return,
                last_update = NOW(),
                data = EXCLUDED.data
        """
        
        await DatabaseUtils.execute_command(
            pool, query,
            strategy,
            performance_data.get('win_rate', 0.0),
            performance_data.get('total_trades', 0),
            performance_data.get('avg_return', 0.0),
            performance_data
        )
    
    @staticmethod
    async def get_strategy_weights(pool: asyncpg.Pool) -> Dict[str, float]:
        """
        Récupère les poids des stratégies depuis la DB
        
        Args:
            pool: Pool de connexions
            
        Returns:
            Dictionnaire des poids par stratégie
        """
        query = """
            SELECT strategy_name, 
                   CASE 
                       WHEN total_trades > 10 THEN LEAST(win_rate * 1.5, 2.0)
                       ELSE 1.0 
                   END as weight
            FROM strategy_performance 
            WHERE last_update > NOW() - INTERVAL '7 days'
        """
        
        results = await DatabaseUtils.execute_query(pool, query)
        return {row['strategy_name']: row['weight'] for row in results}