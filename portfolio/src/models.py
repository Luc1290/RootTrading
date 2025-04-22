"""
Modèles de données pour le service Portfolio.
Définit les structures de données et les interactions avec la base de données.
"""
import logging
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from decimal import Decimal

# Importer les modules partagés
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import get_db_url, POCKET_CONFIG
from shared.src.schemas import AssetBalance, PortfolioSummary, PocketSummary

# Configuration du logging
logger = logging.getLogger(__name__)

class DBManager:
    """
    Gestionnaire de connexion à la base de données.
    Fournit des méthodes pour interagir avec la base de données PostgreSQL.
    """
    
    def __init__(self, db_url: str = None):
        """
        Initialise le gestionnaire de base de données.
        
        Args:
            db_url: URL de connexion à la base de données
        """
        self.db_url = db_url or get_db_url()
        self.conn = None
        
        # Initialiser la connexion
        self._connect()
        
        logger.info("✅ DBManager initialisé")
    
    def _connect(self) -> None:
        """
        Établit la connexion à la base de données.
        """
        try:
            self.conn = psycopg2.connect(self.db_url)
            logger.info("✅ Connexion à la base de données établie")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la connexion à la base de données: {str(e)}")
            self.conn = None
    
    def _ensure_connection(self) -> bool:
        """
        S'assure que la connexion à la base de données est active.
        
        Returns:
            True si la connexion est active, False sinon
        """
        if self.conn is None:
            self._connect()
            return self.conn is not None
        
        try:
            # Vérifier si la connexion est active
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                return True
        except Exception:
            # Reconnexion si nécessaire
            logger.warning("⚠️ Connexion à la base de données perdue, tentative de reconnexion...")
            try:
                self.conn.close()
            except:
                pass
            
            self._connect()
            return self.conn is not None
    
    def execute_query(self, query: str, params: tuple = None, fetch_one: bool = False, 
                     fetch_all: bool = False, commit: bool = False) -> Union[List[Dict], Dict, None]:
        """
        Exécute une requête SQL.
        
        Args:
            query: Requête SQL à exécuter
            params: Paramètres de la requête
            fetch_one: Si True, récupère une seule ligne
            fetch_all: Si True, récupère toutes les lignes
            commit: Si True, valide la transaction
            
        Returns:
            Résultat de la requête ou None en cas d'erreur
        """
        if not self._ensure_connection():
            logger.error("❌ Pas de connexion à la base de données")
            return None
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                
                if fetch_one:
                    result = cursor.fetchone()
                elif fetch_all:
                    result = cursor.fetchall()
                else:
                    result = None
                
                if commit:
                    self.conn.commit()
                
                return result
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'exécution de la requête: {str(e)}")
            try:
                self.conn.rollback()
            except:
                pass
            return None
    
    def execute_many(self, query: str, params_list: List[tuple], commit: bool = True) -> bool:
        """
        Exécute une requête SQL avec plusieurs ensembles de paramètres.
        
        Args:
            query: Requête SQL à exécuter
            params_list: Liste des paramètres pour chaque exécution
            commit: Si True, valide la transaction
            
        Returns:
            True si l'exécution a réussi, False sinon
        """
        if not self._ensure_connection():
            logger.error("❌ Pas de connexion à la base de données")
            return False
        
        try:
            with self.conn.cursor() as cursor:
                cursor.executemany(query, params_list)
                
                if commit:
                    self.conn.commit()
                
                return True
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'exécution multiple: {str(e)}")
            try:
                self.conn.rollback()
            except:
                pass
            return False
    
    def close(self) -> None:
        """
        Ferme la connexion à la base de données.
        """
        if self.conn:
            try:
                self.conn.close()
                logger.info("✅ Connexion à la base de données fermée")
            except Exception as e:
                logger.error(f"❌ Erreur lors de la fermeture de la connexion: {str(e)}")
            finally:
                self.conn = None

class PortfolioModel:
    """
    Modèle pour la gestion du portefeuille.
    Fournit des méthodes pour accéder et manipuler les données du portefeuille.
    """
    
    def __init__(self, db_manager: DBManager = None):
        """
        Initialise le modèle de portefeuille.
        
        Args:
            db_manager: Gestionnaire de base de données préexistant (optionnel)
        """
        self.db = db_manager or DBManager()
        
        logger.info("✅ PortfolioModel initialisé")
    
    def get_latest_balances(self) -> List[AssetBalance]:
        """
        Récupère les derniers soldes du portefeuille.
        
        Returns:
            Liste des soldes par actif
        """
        # Vérifier si un cache existe et s'il est toujours valide (moins de 5 secondes)
        cache_key = 'latest_balances'
        current_time = time.time()
    
        if hasattr(self, '_cache') and cache_key in self._cache:
            cache_time, cache_data = self._cache[cache_key]
            if current_time - cache_time < 5:  # 5 secondes de validité
                return cache_data
            
        query = """
        WITH latest_balances AS (
            SELECT 
                asset,
                MAX(timestamp) as latest_timestamp
            FROM 
                portfolio_balances
            GROUP BY 
                asset
        )
        SELECT 
            pb.asset,
            pb.free,
            pb.locked,
            pb.total,
            pb.value_usdc
        FROM 
            portfolio_balances pb
        JOIN 
            latest_balances lb ON pb.asset = lb.asset AND pb.timestamp = lb.latest_timestamp
        ORDER BY 
            pb.value_usdc DESC NULLS LAST,
            pb.total DESC
        """
        
        result = self.db.execute_query(query, fetch_all=True)
        
        if not result:
            return []
        
        # Convertir en objets AssetBalance
        balances = []
        for row in result:
            balance = AssetBalance(
                asset=row['asset'],
                free=float(row['free']),
                locked=float(row['locked']),
                total=float(row['total']),
                value_usdc=float(row['value_usdc']) if row['value_usdc'] is not None else None
            )
            balances.append(balance)       
        
        # Initialiser le cache si nécessaire
        if not hasattr(self, '_cache'):
            self._cache = {}
    
        # Mettre en cache
        self._cache[cache_key] = (current_time, balances)
    
        return balances
    
    def get_portfolio_summary(self):
        """
        Récupère un résumé du portefeuille.
        """
        try:
            # Récupérer les soldes récents
            balances = self.get_latest_balances()
        
            if not balances:
                balances = []
        
            # Calculer la valeur totale
            total_value = sum(b.value_usdc or 0 for b in balances)
        
            # Obtenir les performances
            performance_24h = self._calculate_performance(days=1) if hasattr(self, "_calculate_performance") else None
            performance_7d = self._calculate_performance(days=7) if hasattr(self, "_calculate_performance") else None

            # Compter les trades actifs
            active_trades = self._count_active_trades() if hasattr(self, "_count_active_trades") else 0
        
        
            return PortfolioSummary(
                balances=balances,
                total_value=total_value,
                performance_24h=performance_24h,
                performance_7d=performance_7d,
                active_trades=active_trades,
                timestamp=datetime.utcnow()
            )
    
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération du résumé du portefeuille: {str(e)}")
            return PortfolioSummary(
                balances=[],
                total_value=0,
                active_trades=0,
                timestamp=datetime.utcnow()
            )
    
    def update_balances(self, balances: List[AssetBalance]) -> bool:
        """
        Met à jour les soldes du portefeuille.
        
        Args:
            balances: Liste des nouveaux soldes
            
        Returns:
            True si la mise à jour a réussi, False sinon
        """
        if not balances:
            return False
        
        # Préparer l'insertion
        query = """
        INSERT INTO portfolio_balances
        (asset, free, locked, total, value_usdc, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        now = datetime.now()
        params_list = []
        
        for balance in balances:
            params = (
                balance.asset,
                balance.free,
                balance.locked,
                balance.total,
                balance.value_usdc,
                now
            )
            params_list.append(params)
        
        # Exécuter l'insertion
        success = self.db.execute_many(query, params_list)
        
        if success:
            logger.info(f"✅ Soldes mis à jour pour {len(balances)} actifs")
        
        return success
    
    def get_trades_history(self, limit: int = 50, offset: int = 0, 
                          symbol: Optional[str] = None, strategy: Optional[str] = None,
                          start_date: Optional[datetime] = None, 
                          end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Récupère l'historique des trades.
        
        Args:
            limit: Nombre maximum de résultats
            offset: Décalage pour la pagination
            symbol: Filtrer par symbole
            strategy: Filtrer par stratégie
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            Liste des trades
        """
        # Construire la requête avec les filtres
        query = """
        SELECT 
            tc.id,
            tc.symbol,
            tc.strategy,
            tc.status,
            tc.entry_price,
            tc.exit_price,
            tc.quantity,
            tc.profit_loss,
            tc.profit_loss_percent,
            tc.created_at,
            tc.completed_at,
            tc.pocket,
            tc.demo
        FROM 
            trade_cycles tc
        WHERE 1=1
        """
        
        params = []
        
        # Ajouter les filtres si spécifiés
        if symbol:
            query += " AND tc.symbol = %s"
            params.append(symbol)
        
        if strategy:
            query += " AND tc.strategy = %s"
            params.append(strategy)
        
        if start_date:
            query += " AND tc.created_at >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND tc.created_at <= %s"
            params.append(end_date)
        
        # Ajouter l'ordre et la pagination
        query += " ORDER BY tc.created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        # Exécuter la requête
        result = self.db.execute_query(query, tuple(params), fetch_all=True)
        
        return result or []
    
    def get_performance_stats(self, period: str = 'daily', 
                             limit: int = 30) -> List[Dict[str, Any]]:
        """
        Récupère les statistiques de performance.
        
        Args:
            period: Période ('daily', 'weekly', 'monthly')
            limit: Nombre de périodes à récupérer
            
        Returns:
            Liste des statistiques de performance
        """
        query = """
        SELECT 
            symbol,
            strategy,
            period,
            start_date,
            end_date,
            total_trades,
            winning_trades,
            losing_trades,
            break_even_trades,
            profit_loss,
            profit_loss_percent
        FROM 
            performance_stats
        WHERE 
            period = %s
        ORDER BY 
            start_date DESC
        LIMIT %s
        """
        
        result = self.db.execute_query(query, (period, limit), fetch_all=True)
        
        return result or []
    
    def get_strategy_performance(self) -> List[Dict[str, Any]]:
        """
        Récupère les performances par stratégie.
        
        Returns:
            Liste des performances par stratégie
        """
        query = """
        SELECT * FROM strategy_performance
        """
        
        result = self.db.execute_query(query, fetch_all=True)
        
        return result or []
    
    def get_symbol_performance(self) -> List[Dict[str, Any]]:
        """
        Récupère les performances par symbole.
        
        Returns:
            Liste des performances par symbole
        """
        query = """
        SELECT * FROM symbol_performance
        """
        
        result = self.db.execute_query(query, fetch_all=True)
        
        return result or []
    
    def close(self) -> None:
        """
        Ferme la connexion à la base de données.
        """
        if self.db:
            self.db.close()