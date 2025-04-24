"""
Module d'exportation des logs vers la base de données.
Stocke les logs en base de données et gère la rotation des anciens logs.
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import psycopg2
from psycopg2.extras import execute_values

# Importer les modules partagés
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import get_db_url

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('db_exporter.log')
    ]
)
logger = logging.getLogger(__name__)

class DBExporter:
    """
    Exporte les logs vers la base de données PostgreSQL.
    Gère également la rotation des anciens logs pour éviter une croissance excessive.
    """
    
    def __init__(self, db_url: str = None, retention_days: int = 30):
        """
        Initialise l'exporteur de logs.
        
        Args:
            db_url: URL de connexion à la base de données
            retention_days: Nombre de jours de rétention des logs
        """
        self.db_url = db_url or get_db_url()
        self.retention_days = retention_days
        self.conn = None
        
        # Statistiques
        self.stats = {
            "logs_stored": 0,
            "logs_rotated": 0,
            "errors": 0,
            "last_rotation": None
        }
        
        # Initialiser la connexion et les tables
        self._init_db()
        
        logger.info(f"✅ DBExporter initialisé (rétention: {retention_days} jours)")
    
    def _init_db(self) -> None:
        """
        Initialise la connexion à la base de données et crée les tables si nécessaire.
        """
        try:
            # Établir la connexion
            self.conn = psycopg2.connect(self.db_url)
            
            # Créer la table des logs si elle n'existe pas
            with self.conn.cursor() as cursor:
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS event_logs (
                    id SERIAL PRIMARY KEY,
                    service VARCHAR(50) NOT NULL,
                    level VARCHAR(20) NOT NULL,
                    message TEXT NOT NULL,
                    data JSONB,
                    source VARCHAR(20) NOT NULL,
                    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
                );
                """)
                
                # Créer les index nécessaires
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS event_logs_service_idx 
                ON event_logs(service);
                """)
                
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS event_logs_level_idx 
                ON event_logs(level);
                """)
                
                cursor.execute("""
                CREATE INDEX IF NOT EXISTS event_logs_timestamp_idx 
                ON event_logs(timestamp);
                """)
                
                # Créer une table d'agrégation pour les statistiques
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS log_stats (
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL,
                    service VARCHAR(50) NOT NULL,
                    level VARCHAR(20) NOT NULL,
                    count INT NOT NULL,
                    UNIQUE (date, service, level)
                );
                """)
                
                self.conn.commit()
                logger.info("✅ Tables de logs initialisées")
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation de la base de données: {str(e)}")
            if self.conn:
                self.conn.close()
                self.conn = None
    
    def _ensure_connection(self) -> bool:
        """
        S'assure que la connexion à la base de données est active.
        
        Returns:
            True si la connexion est active, False sinon
        """
        if self.conn is None:
            self._init_db()
            return self.conn is not None
        
        try:
            # Vérifier si la connexion est active
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT 1")
            return True
        
        except psycopg2.OperationalError:
            # Reconnecter si nécessaire
            logger.warning("⚠️ Connexion à la base de données perdue, reconnexion...")
            try:
                self.conn.close()
            except:
                pass
            
            self._init_db()
            return self.conn is not None
    
    def store_logs(self, logs: List[Dict[str, Any]]) -> bool:
        """
        Stocke une liste de logs en base de données.
        
        Args:
            logs: Liste de logs à stocker
            
        Returns:
            True si tous les logs ont été stockés, False sinon
        """
        if not logs:
            return True
        
        if not self._ensure_connection():
            logger.error("❌ Impossible de stocker les logs: pas de connexion à la base de données")
            self.stats["errors"] += 1
            return False
        
        try:
            # Préparer les données pour l'insertion
            data_to_insert = []
            for log in logs:
                # Convertir la chaîne de timestamp en objet datetime si nécessaire
                timestamp = log.get("timestamp")
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    except ValueError:
                        timestamp = datetime.now()
                elif timestamp is None:
                    timestamp = datetime.now()
                
                data_to_insert.append((
                    log.get("service", "unknown"),
                    log.get("level", "info"),
                    log.get("message", ""),
                    log.get("data"),
                    log.get("source", "unknown"),
                    timestamp
                ))
            
            # Insérer les logs
            with self.conn.cursor() as cursor:
                execute_values(
                    cursor,
                    """
                    INSERT INTO event_logs 
                    (service, level, message, data, source, timestamp)
                    VALUES %s
                    """,
                    data_to_insert
                )
                
                # Mettre à jour les statistiques d'agrégation
                cursor.execute("""
                INSERT INTO log_stats (date, service, level, count)
                SELECT 
                    DATE(timestamp), 
                    service, 
                    level, 
                    COUNT(*)
                FROM 
                    event_logs
                WHERE 
                    timestamp >= DATE_TRUNC('day', NOW())
                GROUP BY 
                    DATE(timestamp), service, level
                ON CONFLICT (date, service, level) 
                DO UPDATE SET count = 
                    log_stats.count + EXCLUDED.count
                """)
                
                self.conn.commit()
            
            # Mettre à jour les statistiques
            self.stats["logs_stored"] += len(logs)
            
            # Tenter la rotation des logs si nous n'en avons pas fait récemment
            if (self.stats["last_rotation"] is None or 
                (datetime.now() - self.stats["last_rotation"]).total_seconds() > 3600):  # Toutes les heures
                self._rotate_old_logs()
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du stockage des logs: {str(e)}")
            self.stats["errors"] += 1
            
            try:
                self.conn.rollback()
            except:
                pass
            
            return False
    
    def _rotate_old_logs(self) -> None:
        """
        Supprime les logs plus anciens que la période de rétention.
        """
        if not self._ensure_connection():
            return
        
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        try:
            with self.conn.cursor() as cursor:
                # Nombre de logs à supprimer
                cursor.execute(
                    "SELECT COUNT(*) FROM event_logs WHERE timestamp < %s",
                    (cutoff_date,)
                )
                count = cursor.fetchone()[0]
                
                if count > 0:
                    # Supprimer les logs anciens
                    cursor.execute(
                        "DELETE FROM event_logs WHERE timestamp < %s",
                        (cutoff_date,)
                    )
                    
                    # Valider la transaction
                    self.conn.commit()
                    
                    # Mettre à jour les statistiques
                    self.stats["logs_rotated"] += count
                    self.stats["last_rotation"] = datetime.now()
                    
                    logger.info(f"✅ Rotation: {count} logs plus anciens que {self.retention_days} jours supprimés")
                else:
                    self.stats["last_rotation"] = datetime.now()
                    logger.info("Aucun log à supprimer lors de la rotation")
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la rotation des logs: {str(e)}")
            self.stats["errors"] += 1
            
            try:
                self.conn.rollback()
            except:
                pass
    
    def get_log_summary(self, days: int = 1) -> Dict[str, Any]:
        """
        Récupère un résumé des logs sur la période spécifiée.
        
        Args:
            days: Nombre de jours pour le résumé
            
        Returns:
            Résumé des logs
        """
        if not self._ensure_connection():
            return {"error": "Pas de connexion à la base de données"}
        
        try:
            start_date = datetime.now() - timedelta(days=days)
            
            with self.conn.cursor() as cursor:
                # Récupérer les statistiques globales
                cursor.execute("""
                SELECT 
                    COUNT(*) as total_logs,
                    COUNT(DISTINCT service) as distinct_services,
                    MIN(timestamp) as oldest_log,
                    MAX(timestamp) as newest_log
                FROM 
                    event_logs
                WHERE 
                    timestamp >= %s
                """, (start_date,))
                
                global_stats = cursor.fetchone()
                
                # Récupérer les statistiques par niveau
                cursor.execute("""
                SELECT 
                    level, 
                    COUNT(*) as count
                FROM 
                    event_logs
                WHERE 
                    timestamp >= %s
                GROUP BY 
                    level
                ORDER BY 
                    count DESC
                """, (start_date,))
                
                level_stats = cursor.fetchall()
                
                # Récupérer les statistiques par service
                cursor.execute("""
                SELECT 
                    service, 
                    COUNT(*) as count
                FROM 
                    event_logs
                WHERE 
                    timestamp >= %s
                GROUP BY 
                    service
                ORDER BY 
                    count DESC
                LIMIT 10
                """, (start_date,))
                
                service_stats = cursor.fetchall()
                
                # Récupérer les erreurs récentes
                cursor.execute("""
                SELECT 
                    id,
                    service,
                    timestamp,
                    message
                FROM 
                    event_logs
                WHERE 
                    timestamp >= %s
                    AND level IN ('error', 'critical')
                ORDER BY 
                    timestamp DESC
                LIMIT 10
                """, (start_date,))
                
                recent_errors = cursor.fetchall()
            
            # Construire le résumé
            summary = {
                "period": f"Last {days} days",
                "total_logs": global_stats[0] if global_stats else 0,
                "distinct_services": global_stats[1] if global_stats else 0,
                "oldest_log": global_stats[2].isoformat() if global_stats and global_stats[2] else None,
                "newest_log": global_stats[3].isoformat() if global_stats and global_stats[3] else None,
                "logs_by_level": {level: count for level, count in level_stats} if level_stats else {},
                "top_services": {service: count for service, count in service_stats} if service_stats else {},
                "recent_errors": [
                    {
                        "id": id,
                        "service": service,
                        "timestamp": timestamp.isoformat(),
                        "message": message
                    }
                    for id, service, timestamp, message in recent_errors
                ] if recent_errors else []
            }
            
            return summary
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération du résumé des logs: {str(e)}")
            return {"error": str(e)}
    
    def get_logs(self, service: Optional[str] = None, level: Optional[str] = None, 
                limit: int = 100, offset: int = 0, start_date: Optional[datetime] = None, 
                end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Récupère des logs avec filtrage.
        
        Args:
            service: Filtrer par service (optionnel)
            level: Filtrer par niveau de log (optionnel)
            limit: Nombre maximum de logs à récupérer
            offset: Offset pour la pagination
            start_date: Date de début (optionnel)
            end_date: Date de fin (optionnel)
            
        Returns:
            Liste des logs
        """
        if not self._ensure_connection():
            return []
        
        try:
            # Construire la requête SQL
            query = """
            SELECT 
                id, 
                service, 
                level, 
                message, 
                data, 
                source, 
                timestamp
            FROM 
                event_logs
            WHERE 1=1
            """
            
            params = []
            
            # Ajouter les filtres
            if service:
                query += " AND service = %s"
                params.append(service)
            
            if level:
                query += " AND level = %s"
                params.append(level)
            
            if start_date:
                query += " AND timestamp >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= %s"
                params.append(end_date)
            
            # Ajouter l'ordre et la pagination
            query += " ORDER BY timestamp DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            
            # Exécuter la requête
            with self.conn.cursor() as cursor:
                cursor.execute(query, params)
                
                # Récupérer les résultats
                logs = []
                for row in cursor.fetchall():
                    logs.append({
                        "id": row[0],
                        "service": row[1],
                        "level": row[2],
                        "message": row[3],
                        "data": row[4],
                        "source": row[5],
                        "timestamp": row[6].isoformat()
                    })
                
                return logs
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des logs: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Récupère les statistiques d'exportation.
        
        Returns:
            Statistiques d'exportation
        """
        return {
            "logs_stored": self.stats["logs_stored"],
            "logs_rotated": self.stats["logs_rotated"],
            "errors": self.stats["errors"],
            "last_rotation": self.stats["last_rotation"].isoformat() if self.stats["last_rotation"] else None
        }
    
    def close(self) -> None:
        """
        Ferme la connexion à la base de données.
        """
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("✅ DBExporter fermé")

# Point d'entrée pour les tests
if __name__ == "__main__":
    # Initialiser l'exporteur
    exporter = DBExporter()
    
    # Tester avec quelques logs
    test_logs = [
        {
            "service": "test-service",
            "level": "info",
            "message": "Test log message",
            "timestamp": datetime.now().isoformat(),
            "source": "direct",
            "data": '{"test": "data"}'
        },
        {
            "service": "test-service",
            "level": "error",
            "message": "Test error message",
            "timestamp": datetime.now().isoformat(),
            "source": "direct",
            "data": None
        }
    ]
    
    # Stocker les logs
    success = exporter.store_logs(test_logs)
    print(f"Logs stockés: {success}")
    
    # Récupérer un résumé
    summary = exporter.get_log_summary()
    print(f"Résumé: {summary}")
    
    # Fermer l'exporteur
    exporter.close()