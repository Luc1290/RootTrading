"""
Module de vérification de santé du système.
Effectue des health checks sur différents aspects du système (base de données, messages, etc.)
"""
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
import redis
import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HealthChecker:
    """
    Effectue des vérifications de santé sur différents composants du système.
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialise le vérificateur de santé.
        
        Args:
            config_file: Chemin vers le fichier de configuration (optionnel)
        """
        # Charger la configuration
        self.config = self._load_config(config_file)
        
        # Configuration des connexions
        self.db_url = self.config.get("db_url", "postgresql://postgres:postgres@db:5432/trading")
        self.redis_host = self.config.get("redis_host", "redis")
        self.redis_port = self.config.get("redis_port", 6379)
        
        # Seuils d'alerte
        self.thresholds = self.config.get("thresholds", {})
        
        # Derniers résultats de vérification
        self.last_results = {}
        
        # Connexions
        self.db_conn = None
        self.redis_conn = None
        
        logger.info("✅ HealthChecker initialisé")
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Charge la configuration depuis un fichier ou utilise les valeurs par défaut.
        
        Args:
            config_file: Chemin vers le fichier de configuration
            
        Returns:
            Configuration chargée
        """
        default_config = {
            "db_url": "postgresql://postgres:postgres@db:5432/trading",
            "redis_host": "redis",
            "redis_port": 6379,
            "thresholds": {
                "max_cycle_age_hours": 72,     # Âge maximum d'un cycle en attente (heures)
                "max_stale_data_minutes": 15,  # Âge maximum des données de marché (minutes)
                "min_free_disk_space_mb": 500, # Espace disque minimum (MB)
                "min_cycles_per_day": 3,       # Nombre minimum de cycles par jour
                "max_error_rate": 5.0          # Taux d'erreur maximum (%)
            }
        }
        
        if not config_file:
            return default_config
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            # Fusionner avec les valeurs par défaut
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                elif isinstance(value, dict) and isinstance(config[key], dict):
                    # Pour les dictionnaires imbriqués comme "thresholds"
                    for subkey, subvalue in value.items():
                        if subkey not in config[key]:
                            config[key][subkey] = subvalue
            
            return config
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {str(e)}")
            return default_config
    
    def _get_db_connection(self) -> Optional[psycopg2.extensions.connection]:
        """
        Obtient une connexion à la base de données.
        
        Returns:
            Connexion à la base de données ou None en cas d'erreur
        """
        if self.db_conn and not self.db_conn.closed:
            return self.db_conn
        
        try:
            self.db_conn = psycopg2.connect(self.db_url)
            return self.db_conn
        except Exception as e:
            logger.error(f"Erreur de connexion à la base de données: {str(e)}")
            return None
    
    def _get_redis_connection(self) -> Optional[redis.Redis]:
        """
        Obtient une connexion à Redis.
        
        Returns:
            Connexion à Redis ou None en cas d'erreur
        """
        if self.redis_conn:
            return self.redis_conn
        
        try:
            self.redis_conn = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                socket_timeout=5
            )
            return self.redis_conn
        except Exception as e:
            logger.error(f"Erreur de connexion à Redis: {str(e)}")
            return None
    
    def check_database_connection(self) -> Dict[str, Any]:
        """
        Vérifie la connexion à la base de données.
        
        Returns:
            Résultat de la vérification
        """
        status = {
            "status": "failed",
            "message": "Erreur de connexion à la base de données",
            "details": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            conn = self._get_db_connection()
            if not conn:
                return status
            
            start_time = time.time()
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            response_time = time.time() - start_time
            
            status.update({
                "status": "ok",
                "message": "Connexion à la base de données établie",
                "details": {
                    "response_time": response_time,
                    "connection_string": self.db_url.split("@")[1]  # Ne pas inclure les credentials
                }
            })
            
            if response_time > 1.0:
                status.update({
                    "status": "warning",
                    "message": f"Temps de réponse de la base de données élevé: {response_time:.3f}s"
                })
            
        except Exception as e:
            status.update({
                "status": "failed",
                "message": f"Erreur lors de la vérification de la base de données: {str(e)}"
            })
        
        return status
    
    def check_redis_connection(self) -> Dict[str, Any]:
        """
        Vérifie la connexion à Redis.
        
        Returns:
            Résultat de la vérification
        """
        status = {
            "status": "failed",
            "message": "Erreur de connexion à Redis",
            "details": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            redis_conn = self._get_redis_connection()
            if not redis_conn:
                return status
            
            start_time = time.time()
            ping_response = redis_conn.ping()
            response_time = time.time() - start_time
            
            if ping_response:
                status.update({
                    "status": "ok",
                    "message": "Connexion à Redis établie",
                    "details": {
                        "response_time": response_time,
                        "host": f"{self.redis_host}:{self.redis_port}"
                    }
                })
                
                if response_time > 0.5:
                    status.update({
                        "status": "warning",
                        "message": f"Temps de réponse Redis élevé: {response_time:.3f}s"
                    })
            else:
                status.update({
                    "status": "failed",
                    "message": "Redis ne répond pas au ping"
                })
            
        except Exception as e:
            status.update({
                "status": "failed",
                "message": f"Erreur lors de la vérification de Redis: {str(e)}"
            })
        
        return status
    
    def check_active_cycles(self) -> Dict[str, Any]:
        """
        Vérifie les cycles de trading actifs.
        
        Returns:
            Résultat de la vérification
        """
        status = {
            "status": "unknown",
            "message": "Vérification des cycles non effectuée",
            "details": {},
            "timestamp": datetime.now().isoformat()
        }
        
        conn = self._get_db_connection()
        if not conn:
            status.update({
                "status": "failed",
                "message": "Impossible de vérifier les cycles: connexion DB non disponible"
            })
            return status
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Récupérer les cycles actifs
                cursor.execute("""
                SELECT 
                    status, 
                    COUNT(*) as count,
                    MIN(updated_at) as oldest_update,
                    MAX(created_at) as newest_create
                FROM 
                    trade_cycles
                WHERE 
                    status NOT IN ('completed', 'canceled', 'failed')
                GROUP BY 
                    status
                """)
                
                active_cycles = cursor.fetchall()
                
                if not active_cycles:
                    status.update({
                        "status": "warning",
                        "message": "Aucun cycle actif trouvé",
                        "details": {"active_cycles": 0}
                    })
                    return status
                
                # Calculer le nombre total de cycles actifs
                total_active = sum(cycle['count'] for cycle in active_cycles)
                
                # Vérifier les cycles potentiellement bloqués
                max_cycle_age = self.thresholds.get("max_cycle_age_hours", 72)
                blocked_cycles = []
                
                for cycle in active_cycles:
                    if cycle['oldest_update']:
                        age_hours = (datetime.now() - cycle['oldest_update']).total_seconds() / 3600
                        if age_hours > max_cycle_age:
                            blocked_cycles.append({
                                "status": cycle['status'],
                                "count": cycle['count'],
                                "age_hours": age_hours
                            })
                
                # Préparer les détails de la vérification
                details = {
                    "active_cycles": total_active,
                    "by_status": {cycle['status']: cycle['count'] for cycle in active_cycles},
                    "blocked_cycles": blocked_cycles
                }
                
                # Déterminer le statut global
                if blocked_cycles:
                    status.update({
                        "status": "warning",
                        "message": f"{len(blocked_cycles)} cycles potentiellement bloqués",
                        "details": details
                    })
                else:
                    status.update({
                        "status": "ok",
                        "message": f"{total_active} cycles actifs en cours",
                        "details": details
                    })
            
        except Exception as e:
            status.update({
                "status": "failed",
                "message": f"Erreur lors de la vérification des cycles: {str(e)}"
            })
        
        return status
    
    def check_market_data_freshness(self) -> Dict[str, Any]:
        """
        Vérifie la fraîcheur des données de marché.
        
        Returns:
            Résultat de la vérification
        """
        status = {
            "status": "unknown",
            "message": "Vérification des données de marché non effectuée",
            "details": {},
            "timestamp": datetime.now().isoformat()
        }
        
        conn = self._get_db_connection()
        if not conn:
            status.update({
                "status": "failed",
                "message": "Impossible de vérifier les données de marché: connexion DB non disponible"
            })
            return status
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Récupérer les dernières données de marché par symbole
                cursor.execute("""
                WITH latest_data AS (
                    SELECT 
                        symbol,
                        MAX(time) as last_update
                    FROM 
                        market_data
                    GROUP BY 
                        symbol
                )
                SELECT 
                    symbol,
                    last_update
                FROM 
                    latest_data
                """)
                
                market_data = cursor.fetchall()
                
                if not market_data:
                    status.update({
                        "status": "warning",
                        "message": "Aucune donnée de marché trouvée",
                        "details": {"symbols": 0}
                    })
                    return status
                
                # Vérifier la fraîcheur des données
                max_stale_minutes = self.thresholds.get("max_stale_data_minutes", 15)
                stale_symbols = []
                fresh_symbols = []
                
                for data in market_data:
                    if data['last_update']:
                        age_minutes = (datetime.now() - data['last_update']).total_seconds() / 60
                        if age_minutes > max_stale_minutes:
                            stale_symbols.append({
                                "symbol": data['symbol'],
                                "last_update": data['last_update'].isoformat(),
                                "age_minutes": age_minutes
                            })
                        else:
                            fresh_symbols.append(data['symbol'])
                
                # Préparer les détails de la vérification
                details = {
                    "total_symbols": len(market_data),
                    "fresh_symbols": len(fresh_symbols),
                    "stale_symbols": stale_symbols
                }
                
                # Déterminer le statut global
                if stale_symbols:
                    status.update({
                        "status": "warning",
                        "message": f"Données obsolètes pour {len(stale_symbols)} symboles",
                        "details": details
                    })
                else:
                    status.update({
                        "status": "ok",
                        "message": f"Données à jour pour tous les symboles ({len(fresh_symbols)})",
                        "details": details
                    })
            
        except Exception as e:
            status.update({
                "status": "failed",
                "message": f"Erreur lors de la vérification des données de marché: {str(e)}"
            })
        
        return status
    
    def check_disk_space(self) -> Dict[str, Any]:
        """
        Vérifie l'espace disque disponible.
        
        Returns:
            Résultat de la vérification
        """
        status = {
            "status": "unknown",
            "message": "Vérification de l'espace disque non effectuée",
            "details": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Vérifier l'espace disque pour quelques répertoires clés
            paths_to_check = [
                "/", "/app", "/var/lib/postgresql/data"
            ]
            
            disk_stats = []
            warnings = []
            
            for path in paths_to_check:
                if os.path.exists(path):
                    stat = os.statvfs(path)
                    free_bytes = stat.f_frsize * stat.f_bavail
                    total_bytes = stat.f_frsize * stat.f_blocks
                    free_mb = free_bytes / (1024 * 1024)
                    total_mb = total_bytes / (1024 * 1024)
                    used_percent = 100 - (free_bytes / total_bytes * 100)
                    
                    disk_stats.append({
                        "path": path,
                        "free_mb": free_mb,
                        "total_mb": total_mb,
                        "used_percent": used_percent
                    })
                    
                    # Vérifier si l'espace est suffisant
                    min_free_mb = self.thresholds.get("min_free_disk_space_mb", 500)
                    if free_mb < min_free_mb:
                        warnings.append({
                            "path": path,
                            "free_mb": free_mb,
                            "min_required_mb": min_free_mb
                        })
            
            # Préparer les détails de la vérification
            details = {
                "disk_stats": disk_stats
            }
            
            # Déterminer le statut global
            if warnings:
                status.update({
                    "status": "warning",
                    "message": f"Espace disque insuffisant pour {len(warnings)} chemins",
                    "details": {"disk_stats": disk_stats, "warnings": warnings}
                })
            else:
                status.update({
                    "status": "ok",
                    "message": "Espace disque suffisant",
                    "details": details
                })
            
        except Exception as e:
            status.update({
                "status": "failed",
                "message": f"Erreur lors de la vérification de l'espace disque: {str(e)}"
            })
        
        return status
    
    def check_trading_activity(self) -> Dict[str, Any]:
        """
        Vérifie l'activité de trading récente.
        
        Returns:
            Résultat de la vérification
        """
        status = {
            "status": "unknown",
            "message": "Vérification de l'activité de trading non effectuée",
            "details": {},
            "timestamp": datetime.now().isoformat()
        }
        
        conn = self._get_db_connection()
        if not conn:
            status.update({
                "status": "failed",
                "message": "Impossible de vérifier l'activité de trading: connexion DB non disponible"
            })
            return status
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Récupérer l'activité de trading des dernières 24h
                cursor.execute("""
                SELECT 
                    COUNT(*) as total_cycles,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_cycles,
                    COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as profitable_cycles,
                    SUM(profit_loss) as total_profit
                FROM 
                    trade_cycles
                WHERE 
                    created_at >= NOW() - INTERVAL '24 hours'
                """)
                
                activity = cursor.fetchone()
                
                if not activity or activity['total_cycles'] == 0:
                    status.update({
                        "status": "warning",
                        "message": "Aucune activité de trading dans les dernières 24h",
                        "details": {"total_cycles": 0}
                    })
                    return status
                
                # Vérifier si l'activité est suffisante
                min_cycles = self.thresholds.get("min_cycles_per_day", 3)
                
                details = {
                    "total_cycles": activity['total_cycles'],
                    "completed_cycles": activity['completed_cycles'],
                    "profitable_cycles": activity['profitable_cycles'],
                    "total_profit": float(activity['total_profit'] or 0)
                }
                
                # Calculer le win rate et le profit moyen
                if activity['completed_cycles'] > 0:
                    details["win_rate"] = (activity['profitable_cycles'] / activity['completed_cycles']) * 100
                    details["avg_profit"] = details["total_profit"] / activity['completed_cycles']
                
                # Déterminer le statut global
                if activity['total_cycles'] < min_cycles:
                    status.update({
                        "status": "warning",
                        "message": f"Faible activité de trading: {activity['total_cycles']} cycles (min: {min_cycles})",
                        "details": details
                    })
                else:
                    status.update({
                        "status": "ok",
                        "message": f"Activité de trading normale: {activity['total_cycles']} cycles dans les dernières 24h",
                        "details": details
                    })
            
        except Exception as e:
            status.update({
                "status": "failed",
                "message": f"Erreur lors de la vérification de l'activité de trading: {str(e)}"
            })
        
        return status
    
    def check_error_logs(self) -> Dict[str, Any]:
        """
        Vérifie les logs d'erreurs récents.
        
        Returns:
            Résultat de la vérification
        """
        status = {
            "status": "unknown",
            "message": "Vérification des logs d'erreurs non effectuée",
            "details": {},
            "timestamp": datetime.now().isoformat()
        }
        
        conn = self._get_db_connection()
        if not conn:
            status.update({
                "status": "failed",
                "message": "Impossible de vérifier les logs d'erreurs: connexion DB non disponible"
            })
            return status
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Récupérer les erreurs récentes
                cursor.execute("""
                SELECT 
                    service,
                    level,
                    COUNT(*) as count
                FROM 
                    event_logs
                WHERE 
                    timestamp >= NOW() - INTERVAL '1 hour'
                    AND level IN ('error', 'critical')
                GROUP BY 
                    service, level
                ORDER BY 
                    count DESC
                """)
                
                error_logs = cursor.fetchall()
                
                # Récupérer le nombre total de logs pour calculer le taux d'erreur
                cursor.execute("""
                SELECT 
                    COUNT(*) as total_logs
                FROM 
                    event_logs
                WHERE 
                    timestamp >= NOW() - INTERVAL '1 hour'
                """)
                
                total_result = cursor.fetchone()
                total_logs = total_result['total_logs'] if total_result else 0
                
                if not error_logs or total_logs == 0:
                    status.update({
                        "status": "ok",
                        "message": "Aucune erreur récente détectée",
                        "details": {"error_count": 0, "total_logs": total_logs}
                    })
                    return status
                
                # Calculer les statistiques d'erreurs
                total_errors = sum(log['count'] for log in error_logs)
                error_rate = (total_errors / total_logs) * 100 if total_logs > 0 else 0
                
                details = {
                    "total_errors": total_errors,
                    "total_logs": total_logs,
                    "error_rate": error_rate,
                    "errors_by_service": {f"{log['service']}_{log['level']}": log['count'] for log in error_logs}
                }
                
                # Déterminer le statut global
                max_error_rate = self.thresholds.get("max_error_rate", 5.0)
                
                if error_rate > max_error_rate:
                    status.update({
                        "status": "warning",
                        "message": f"Taux d'erreur élevé: {error_rate:.2f}% (max: {max_error_rate}%)",
                        "details": details
                    })
                else:
                    status.update({
                        "status": "ok",
                        "message": f"Taux d'erreur normal: {error_rate:.2f}%",
                        "details": details
                    })
            
        except Exception as e:
            status.update({
                "status": "failed",
                "message": f"Erreur lors de la vérification des logs d'erreurs: {str(e)}"
            })
        
        return status
    
    def run_all_checks(self) -> Dict[str, Dict[str, Any]]:
        """
        Exécute toutes les vérifications de santé.
        
        Returns:
            Résultats de toutes les vérifications
        """
        results = {}
        
        # Exécuter toutes les vérifications
        results["database"] = self.check_database_connection()
        results["redis"] = self.check_redis_connection()
        results["active_cycles"] = self.check_active_cycles()
        results["market_data"] = self.check_market_data_freshness()
        results["disk_space"] = self.check_disk_space()
        results["trading_activity"] = self.check_trading_activity()
        results["error_logs"] = self.check_error_logs()
        
        # Mettre à jour les derniers résultats
        self.last_results = results
        
        # Déterminer le statut global
        global_status = self._determine_global_status(results)
        results["global"] = {
            "status": global_status,
            "timestamp": datetime.now().isoformat()
        }
        
        return results
    
    def _determine_global_status(self, results: Dict[str, Dict[str, Any]]) -> str:
        """
        Détermine le statut global du système.
        
        Args:
            results: Résultats des vérifications
            
        Returns:
            Statut global: 'ok', 'warning', ou 'critical'
        """
        # Vérifier s'il y a des erreurs critiques
        for key, result in results.items():
            if result.get("status") == "failed":
                return "critical"
        
        # Vérifier s'il y a des avertissements
        for key, result in results.items():
            if result.get("status") == "warning":
                return "warning"
        
        # Sinon, tout va bien
        return "ok"
    
    def close(self) -> None:
        """
        Ferme les connexions et nettoie les ressources.
        """
        if self.db_conn:
            self.db_conn.close()
            self.db_conn = None
        
        if self.redis_conn:
            self.redis_conn = None
        
        logger.info("✅ HealthChecker fermé")

# Point d'entrée pour les tests
if __name__ == "__main__":
    # Initialiser le vérificateur de santé
    checker = HealthChecker()
    
    # Exécuter toutes les vérifications
    results = checker.run_all_checks()
    
    # Afficher les résultats
    for check_name, result in results.items():
        print(f"[{check_name}] {result.get('status', 'unknown')}: {result.get('message', '')}")
    
    # Fermer le vérificateur
    checker.close()