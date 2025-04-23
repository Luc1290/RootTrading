"""
Module de suivi et d'enregistrement des profits et pertes (PnL).
Permet de suivre les performances des stratégies de trading et d'enregistrer
les métriques pour analyse ultérieure.
"""
import logging
import json
import pandas as pd
import numpy as np
import os
import time
from decimal import Decimal

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import threading
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from concurrent.futures import ThreadPoolExecutor

# Importer les modules partagés
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import get_db_url, SYMBOLS
from shared.src.enums import CycleStatus

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pnl_logger.log')
    ]
)
logger = logging.getLogger(__name__)

class PnLLogger:
    """
    Logger des profits et pertes (PnL) pour le système de trading.
    Collecte, analyse et enregistre les métriques de performance des stratégies.
    """
    
    def __init__(self, db_url: str = None, export_dir: str = "./exports"):
        """
        Initialise le logger PnL.
        
        Args:
            db_url: URL de connexion à la base de données
            export_dir: Répertoire pour l'exportation des données
        """
        self.db_url = db_url or get_db_url()
        self.export_dir = export_dir
        self.conn = None
        
        # Statistiques en mémoire
        self.stats_cache = {
            "global": {},
            "by_strategy": {},
            "by_symbol": {},
            "daily": {}
        }
        
        # Variables pour le processus en arrière-plan
        self.update_thread = None
        self.stop_event = threading.Event()
        self.update_interval = 3600  # 1 heure par défaut
        
        # Créer le répertoire d'export s'il n'existe pas
        os.makedirs(self.export_dir, exist_ok=True)
        
        # Initialiser la connexion à la base de données
        self._init_db_connection()
        
        logger.info(f"✅ PnLLogger initialisé")
    
    def _init_db_connection(self) -> None:
        """
        Initialise la connexion à la base de données.
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
            self._init_db_connection()
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
            
            self._init_db_connection()
            return self.conn is not None
    
    def calculate_global_stats(self) -> Dict[str, Any]:
        """
        Calcule les statistiques globales de PnL.
        
        Returns:
            Dictionnaire des statistiques globales
        """
        if not self._ensure_connection():
            logger.error("❌ Impossible de calculer les statistiques: pas de connexion à la base de données")
            return {}
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Requête pour les statistiques globales
                query = """
                SELECT 
                    COUNT(*) as total_cycles,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_cycles,
                    SUM(CASE WHEN profit_loss > 0 AND status = 'completed' THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN profit_loss < 0 AND status = 'completed' THEN 1 ELSE 0 END) as losing_trades,
                    SUM(CASE WHEN profit_loss = 0 AND status = 'completed' THEN 1 ELSE 0 END) as break_even_trades,
                    SUM(profit_loss) as total_profit_loss,
                    AVG(CASE WHEN status = 'completed' THEN profit_loss_percent ELSE NULL END) as avg_profit_loss_percent,
                    AVG(CASE WHEN status = 'completed' AND profit_loss > 0 THEN profit_loss_percent ELSE NULL END) as avg_win_percent,
                    AVG(CASE WHEN status = 'completed' AND profit_loss < 0 THEN profit_loss_percent ELSE NULL END) as avg_loss_percent,
                    COUNT(DISTINCT symbol) as symbol_count,
                    COUNT(DISTINCT strategy) as strategy_count
                FROM 
                    trade_cycles
                """
                
                cursor.execute(query)
                result = cursor.fetchone()
                
                if not result:
                    return {}
                
                # Calcul de métriques additionnelles
                stats = {k: float(v) if isinstance(v, (Decimal, float)) else v for k, v in result.items()}
                
                # Calcul du win rate
                if stats.get("completed_cycles", 0) > 0:
                    stats["win_rate"] = (stats.get("winning_trades", 0) / stats["completed_cycles"]) * 100
                else:
                    stats["win_rate"] = 0
                
                # Calcul du profit factor (gains bruts / pertes brutes)
                query_pf = """
                SELECT 
                    SUM(CASE WHEN profit_loss > 0 THEN profit_loss ELSE 0 END) as gross_profit,
                    ABS(SUM(CASE WHEN profit_loss < 0 THEN profit_loss ELSE 0 END)) as gross_loss
                FROM 
                    trade_cycles
                WHERE 
                    status = 'completed'
                """
                
                cursor.execute(query_pf)
                pf_result = cursor.fetchone()
                
                if pf_result and pf_result["gross_loss"] is not None and pf_result["gross_profit"] is not None:
                    gross_loss = float(pf_result["gross_loss"] or 0)
                    gross_profit = float(pf_result["gross_profit"] or 0)
    
                    # Éviter la division par zéro
                    if gross_loss > 0:
                        stats["profit_factor"] = gross_profit / gross_loss
                    elif gross_profit > 0:  # Gross loss est 0 mais il y a du profit
                        stats["profit_factor"] = float('inf')  # profit factor infini
                    else:
                        stats["profit_factor"] = 0  # Pas de profit ni de perte
                else:
                    stats["profit_factor"] = 0
                
                # Ajouter la date de mise à jour
                stats["updated_at"] = datetime.now().isoformat()
                
                return stats
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du calcul des statistiques globales: {str(e)}")
            return {}
    
    def calculate_strategy_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Calcule les statistiques PnL par stratégie.
        
        Returns:
            Dictionnaire des statistiques par stratégie
        """
        if not self._ensure_connection():
            logger.error("❌ Impossible de calculer les statistiques: pas de connexion à la base de données")
            return {}
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Requête pour les statistiques par stratégie
                query = """
                SELECT 
                    strategy,
                    COUNT(*) as total_cycles,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_cycles,
                    SUM(CASE WHEN profit_loss > 0 AND status = 'completed' THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN profit_loss < 0 AND status = 'completed' THEN 1 ELSE 0 END) as losing_trades,
                    SUM(CASE WHEN profit_loss = 0 AND status = 'completed' THEN 1 ELSE 0 END) as break_even_trades,
                    SUM(profit_loss) as total_profit_loss,
                    AVG(CASE WHEN status = 'completed' THEN profit_loss_percent ELSE NULL END) as avg_profit_loss_percent,
                    COUNT(DISTINCT symbol) as symbol_count
                FROM 
                    trade_cycles
                GROUP BY 
                    strategy
                ORDER BY 
                    total_profit_loss DESC
                """
                
                cursor.execute(query)
                results = cursor.fetchall()
                
                # Créer le dictionnaire des statistiques par stratégie
                strategy_stats = {}
                
                for row in results:
                    strategy = row["strategy"]
                    stats = {k: float(v) if isinstance(v, (Decimal, float)) else v for k, v in row.items() if k != "strategy"}
                    
                    # Calcul du win rate
                    if stats.get("completed_cycles", 0) > 0:
                        stats["win_rate"] = (stats.get("winning_trades", 0) / stats["completed_cycles"]) * 100
                    else:
                        stats["win_rate"] = 0
                    
                    # Récupérer les métriques de profitabilité avancées
                    query_pf = """
                    SELECT 
                        SUM(CASE WHEN profit_loss > 0 THEN profit_loss ELSE 0 END) as gross_profit,
                        ABS(SUM(CASE WHEN profit_loss < 0 THEN profit_loss ELSE 0 END)) as gross_loss,
                        MAX(profit_loss_percent) as max_win_percent,
                        MIN(profit_loss_percent) as max_loss_percent,
                        AVG(CASE WHEN profit_loss > 0 THEN profit_loss_percent ELSE NULL END) as avg_win_percent,
                        AVG(CASE WHEN profit_loss < 0 THEN profit_loss_percent ELSE NULL END) as avg_loss_percent
                    FROM 
                        trade_cycles
                    WHERE 
                        status = 'completed' AND
                        strategy = %s
                    """
                    
                    cursor.execute(query_pf, (strategy,))
                    pf_result = cursor.fetchone()
                    
                    if pf_result:
                        for k, v in pf_result.items():
                            if v is not None:
                                stats[k] = float(v) if isinstance(v, (Decimal, float)) else v
                        
                        # Calcul du profit factor
                        if pf_result["gross_loss"] is not None and pf_result["gross_loss"] > 0:
                            stats["profit_factor"] = pf_result["gross_profit"] / pf_result["gross_loss"]
                        else:
                            stats["profit_factor"] = float('inf') if pf_result["gross_profit"] > 0 else 0
                    
                    strategy_stats[strategy] = stats
                
                return strategy_stats
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du calcul des statistiques par stratégie: {str(e)}")
            return {}
    
    def calculate_symbol_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Calcule les statistiques PnL par symbole.
        
        Returns:
            Dictionnaire des statistiques par symbole
        """
        if not self._ensure_connection():
            logger.error("❌ Impossible de calculer les statistiques: pas de connexion à la base de données")
            return {}
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Requête pour les statistiques par symbole
                query = """
                SELECT 
                    symbol,
                    COUNT(*) as total_cycles,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_cycles,
                    SUM(CASE WHEN profit_loss > 0 AND status = 'completed' THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN profit_loss < 0 AND status = 'completed' THEN 1 ELSE 0 END) as losing_trades,
                    SUM(CASE WHEN profit_loss = 0 AND status = 'completed' THEN 1 ELSE 0 END) as break_even_trades,
                    SUM(profit_loss) as total_profit_loss,
                    AVG(CASE WHEN status = 'completed' THEN profit_loss_percent ELSE NULL END) as avg_profit_loss_percent,
                    COUNT(DISTINCT strategy) as strategy_count
                FROM 
                    trade_cycles
                GROUP BY 
                    symbol
                ORDER BY 
                    total_profit_loss DESC
                """
                
                cursor.execute(query)
                results = cursor.fetchall()
                
                # Créer le dictionnaire des statistiques par symbole
                symbol_stats = {}
                
                for row in results:
                    symbol = row["symbol"]
                    stats = {k: float(v) if isinstance(v, (Decimal, float)) else v for k, v in row.items() if k != "symbol"}
                    
                    # Calcul du win rate
                    if stats.get("completed_cycles", 0) > 0:
                        stats["win_rate"] = (stats.get("winning_trades", 0) / stats["completed_cycles"]) * 100
                    else:
                        stats["win_rate"] = 0
                    
                    # Récupérer les métriques de profitabilité avancées
                    query_pf = """
                    SELECT 
                        SUM(CASE WHEN profit_loss > 0 THEN profit_loss ELSE 0 END) as gross_profit,
                        ABS(SUM(CASE WHEN profit_loss < 0 THEN profit_loss ELSE 0 END)) as gross_loss,
                        MAX(profit_loss_percent) as max_win_percent,
                        MIN(profit_loss_percent) as max_loss_percent
                    FROM 
                        trade_cycles
                    WHERE 
                        status = 'completed' AND
                        symbol = %s
                    """
                    
                    cursor.execute(query_pf, (symbol,))
                    pf_result = cursor.fetchone()
                    
                    if pf_result:
                        for k, v in pf_result.items():
                            if v is not None:
                                stats[k] = float(v) if isinstance(v, (Decimal, float)) else v
                        
                        # Calcul du profit factor
                        if pf_result["gross_loss"] is not None and pf_result["gross_loss"] > 0:
                            stats["profit_factor"] = pf_result["gross_profit"] / pf_result["gross_loss"]
                        else:
                            stats["profit_factor"] = float('inf') if pf_result["gross_profit"] > 0 else 0
                    
                    symbol_stats[symbol] = stats
                
                return symbol_stats
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du calcul des statistiques par symbole: {str(e)}")
            return {}
    
    def calculate_daily_stats(self, days: int = 30) -> Dict[str, Dict[str, Any]]:
        """
        Calcule les statistiques PnL quotidiennes sur une période.
        
        Args:
            days: Nombre de jours à inclure
            
        Returns:
            Dictionnaire des statistiques quotidiennes
        """
        if not self._ensure_connection():
            logger.error("❌ Impossible de calculer les statistiques: pas de connexion à la base de données")
            return {}
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Requête pour les statistiques quotidiennes
                query = """
                SELECT 
                    DATE(completed_at) as trade_date,
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losing_trades,
                    SUM(profit_loss) as daily_profit_loss,
                    AVG(profit_loss_percent) as avg_profit_loss_percent
                FROM 
                    trade_cycles
                WHERE 
                    status = 'completed'
                    AND completed_at >= NOW() - INTERVAL '%s days'
                GROUP BY 
                    DATE(completed_at)
                ORDER BY 
                    trade_date DESC
                """
                
                cursor.execute(query, (days,))
                results = cursor.fetchall()
                
                # Créer le dictionnaire des statistiques quotidiennes
                daily_stats = {}
                
                for row in results:
                    date_str = row["trade_date"].isoformat()
                    stats = {k: float(v) if isinstance(v, (Decimal, float)) else v for k, v in row.items() if k != "trade_date"}
                    
                    # Calcul du win rate
                    if stats.get("total_trades", 0) > 0:
                        stats["win_rate"] = (stats.get("winning_trades", 0) / stats["total_trades"]) * 100
                    else:
                        stats["win_rate"] = 0
                    
                    daily_stats[date_str] = stats
                
                return daily_stats
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du calcul des statistiques quotidiennes: {str(e)}")
            return {}
    
    def calculate_drawdown(self, days: int = 90) -> Dict[str, Any]:
        """
        Calcule le drawdown maximal sur une période donnée.
        
        Args:
            days: Nombre de jours à inclure
            
        Returns:
            Informations sur le drawdown
        """
        if not self._ensure_connection():
            logger.error("❌ Impossible de calculer le drawdown: pas de connexion à la base de données")
            return {}
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Requête pour obtenir le PnL quotidien cumulé
                query = """
                WITH daily_pnl AS (
                    SELECT 
                        DATE(completed_at) as trade_date,
                        SUM(profit_loss) as daily_profit_loss
                    FROM 
                        trade_cycles
                    WHERE 
                        status = 'completed'
                        AND completed_at >= NOW() - INTERVAL '%s days'
                    GROUP BY 
                        DATE(completed_at)
                    ORDER BY 
                        trade_date
                )
                SELECT 
                    trade_date,
                    daily_profit_loss,
                    SUM(daily_profit_loss) OVER (ORDER BY trade_date) as cumulative_pnl
                FROM 
                    daily_pnl
                ORDER BY 
                    trade_date
                """
                
                cursor.execute(query, (days,))
                results = cursor.fetchall()
                
                if not results:
                    return {"max_drawdown": 0, "max_drawdown_percent": 0}
                
                # Convertir en DataFrame pour faciliter les calculs
                df = pd.DataFrame(results)
                
                # Calculer le drawdown
                df['cumulative_pnl'] = df['cumulative_pnl'].astype(float)
                df['peak'] = df['cumulative_pnl'].cummax()
                df['drawdown'] = df['peak'] - df['cumulative_pnl']
                
                # Calculer le drawdown maximal
                max_drawdown = df['drawdown'].max()
                
                # Calculer le drawdown maximal en pourcentage
                if len(df) > 0 and df['peak'].max() > 0:
                    peak_before_max_dd = df.loc[df['drawdown'] == max_drawdown, 'peak'].values[0]
                    max_drawdown_percent = (max_drawdown / peak_before_max_dd) * 100 if peak_before_max_dd > 0 else 0
                else:
                    max_drawdown_percent = 0
                
                # Trouver la période du drawdown maximal
                if max_drawdown > 0:
                    max_dd_end_idx = df['drawdown'].idxmax()
                    max_dd_start_idx = df.loc[:max_dd_end_idx]['peak'].idxmax()
                    
                    max_dd_start_date = df.loc[max_dd_start_idx, 'trade_date']
                    max_dd_end_date = df.loc[max_dd_end_idx, 'trade_date']
                    
                    max_dd_duration = (max_dd_end_date - max_dd_start_date).days
                else:
                    max_dd_start_date = None
                    max_dd_end_date = None
                    max_dd_duration = 0
                
                return {
                    "max_drawdown": float(max_drawdown),
                    "max_drawdown_percent": float(max_drawdown_percent),
                    "start_date": max_dd_start_date.isoformat() if max_dd_start_date else None,
                    "end_date": max_dd_end_date.isoformat() if max_dd_end_date else None,
                    "duration_days": max_dd_duration
                }
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du calcul du drawdown: {str(e)}")
            return {"max_drawdown": 0, "max_drawdown_percent": 0}
    
    def calculate_sharpe_ratio(self, days: int = 90, risk_free_rate: float = 0.02) -> float:
        if not self._ensure_connection():
            logger.error("❌ Impossible de calculer le ratio de Sharpe: pas de connexion à la base de données")
            return 0.0
    
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Requête pour obtenir le PnL quotidien
                query = """
                SELECT 
                    DATE(completed_at) as trade_date,
                    SUM(profit_loss_percent) as daily_return
                FROM 
                    trade_cycles
                WHERE 
                    status = 'completed'
                    AND completed_at >= NOW() - INTERVAL '%s days'
                GROUP BY 
                    DATE(completed_at)
                ORDER BY 
                    trade_date
                """
            
                cursor.execute(query, (days,))
                results = cursor.fetchall()
            
                # Vérifier qu'il y a suffisamment de données
                if not results or len(results) < 2:
                    logger.warning("⚠️ Pas assez de données pour calculer le ratio de Sharpe")
                    return 0.0
            
                # Convertir en DataFrame pour faciliter les calculs
                df = pd.DataFrame(results)
                if 'daily_return' not in df.columns or df['daily_return'].isnull().all():
                    logger.warning("⚠️ Données de retour quotidien manquantes ou invalides")
                    return 0.0
                
                df['daily_return'] = df['daily_return'].astype(float)
            
                # Calcul du ratio de Sharpe
                daily_returns = df['daily_return'].values
            
                # Vérifier les valeurs NaN ou None
                if np.isnan(daily_returns).any() or None in daily_returns:
                    logger.warning("⚠️ Valeurs NaN ou None détectées dans les retours quotidiens")
                    daily_returns = np.array([r for r in daily_returns if r is not None and not np.isnan(r)])
                    if len(daily_returns) < 2:
                        return 0.0
            
                # Annualisation des rendements
                avg_daily_return = np.mean(daily_returns)
                annualized_return = avg_daily_return * 252
            
                # Volatilité annualisée
                daily_std = np.std(daily_returns)
            
                # Vérification supplémentaire pour éviter la division par zéro
                if daily_std is None or daily_std <= 0:
                    logger.warning("⚠️ Volatilité nulle, ratio de Sharpe non calculable")
                    return 0.0
                
                annualized_std = daily_std * np.sqrt(252)
            
                # Ratio de Sharpe
                sharpe_ratio = (annualized_return - risk_free_rate) / annualized_std
            
                return float(sharpe_ratio)
    
        except Exception as e:
            logger.error(f"❌ Erreur lors du calcul du ratio de Sharpe: {str(e)}")
            return 0.0
    
    def update_stats(self) -> None:
        """
        Met à jour toutes les statistiques en mémoire.
        """
        logger.info("Mise à jour des statistiques PnL...")
        
        # Utiliser un ThreadPoolExecutor pour paralléliser les calculs
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Soumettre les tâches
            global_future = executor.submit(self.calculate_global_stats)
            strategy_future = executor.submit(self.calculate_strategy_stats)
            symbol_future = executor.submit(self.calculate_symbol_stats)
            daily_future = executor.submit(self.calculate_daily_stats)
            drawdown_future = executor.submit(self.calculate_drawdown)
            sharpe_future = executor.submit(self.calculate_sharpe_ratio)
        
            # Récupérer les résultats
            self.stats_cache["global"] = global_future.result()
            self.stats_cache["by_strategy"] = strategy_future.result()
            self.stats_cache["by_symbol"] = symbol_future.result()
            self.stats_cache["daily"] = daily_future.result()
        
            # Ajouter des métriques avancées
            self.stats_cache["global"]["drawdown"] = drawdown_future.result()
            self.stats_cache["global"]["sharpe_ratio"] = sharpe_future.result()
            self.stats_cache["global"]["updated_at"] = datetime.now().isoformat()
    
        logger.info("✅ Statistiques PnL mises à jour")
    
    def export_stats_to_csv(self, filename: str = None) -> str:
        """
        Exporte les statistiques actuelles vers un fichier Excel.
    
        Args:
            filename: Nom du fichier (optionnel)
        
        Returns:
            Chemin vers le fichier exporté
        """
        if not self.stats_cache["global"]:
            self.update_stats()
    
        # Générer un nom de fichier si non fourni
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pnl_stats_{timestamp}.xlsx"  # Utiliser .xlsx au lieu de .csv
    
        filepath = os.path.join(self.export_dir, filename)
    
        try:
            # Préparer les données pour l'export
            # Pour les statistiques par stratégie
            strategy_df = pd.DataFrame.from_dict(self.stats_cache["by_strategy"], orient='index')
            if not strategy_df.empty:
                strategy_df.index.name = 'strategy'
                strategy_df.reset_index(inplace=True)
            else:
                strategy_df = pd.DataFrame(columns=['strategy'])
        
            # Pour les statistiques par symbole
            symbol_df = pd.DataFrame.from_dict(self.stats_cache["by_symbol"], orient='index')
            if not symbol_df.empty:
                symbol_df.index.name = 'symbol'
                symbol_df.reset_index(inplace=True)
            else:
                symbol_df = pd.DataFrame(columns=['symbol'])
        
            # Pour les statistiques quotidiennes
            daily_df = pd.DataFrame.from_dict(self.stats_cache["daily"], orient='index')
            if not daily_df.empty:
                daily_df.index.name = 'date'
                daily_df.reset_index(inplace=True)
            else:
                daily_df = pd.DataFrame(columns=['date'])
        
            # Statistiques globales
            global_df = pd.DataFrame([self.stats_cache["global"]])
        
            # Exporter vers Excel
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                global_df.to_excel(writer, sheet_name='Global', index=False)
                strategy_df.to_excel(writer, sheet_name='By Strategy', index=False)
                symbol_df.to_excel(writer, sheet_name='By Symbol', index=False)
                daily_df.to_excel(writer, sheet_name='Daily', index=False)
        
            logger.info(f"✅ Statistiques exportées vers {filepath}")
            return filepath
    
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'exportation des statistiques: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return ""
    
    def export_trade_history(self, days: int = 90, filename: str = None) -> str:
        """
        Exporte l'historique des trades vers un fichier CSV.
        
        Args:
            days: Nombre de jours à inclure
            filename: Nom du fichier (optionnel)
            
        Returns:
            Chemin vers le fichier exporté
        """
        if not self._ensure_connection():
            logger.error("❌ Impossible d'exporter l'historique: pas de connexion à la base de données")
            return ""
        
        # Générer un nom de fichier si non fourni
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trade_history_{timestamp}.csv"
        
        filepath = os.path.join(self.export_dir, filename)
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Requête pour l'historique des trades
                query = """
                SELECT 
                    id,
                    symbol,
                    strategy,
                    status,
                    entry_price,
                    exit_price,
                    quantity,
                    target_price,
                    stop_price,
                    profit_loss,
                    profit_loss_percent,
                    created_at,
                    completed_at,
                    EXTRACT(EPOCH FROM (completed_at - created_at))/3600 as duration_hours
                FROM 
                    trade_cycles
                WHERE 
                    created_at >= NOW() - INTERVAL '%s days'
                ORDER BY 
                    created_at DESC
                """
                
                cursor.execute(query, (days,))
                results = cursor.fetchall()
                
                if not results:
                    logger.warning(f"⚠️ Aucun trade trouvé pour les {days} derniers jours")
                    return ""
                
                # Convertir en DataFrame
                df = pd.DataFrame(results)
                
                # Exporter vers CSV
                df.to_csv(filepath, index=False)
                
                logger.info(f"✅ Historique des trades exporté vers {filepath}")
                return filepath
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'exportation de l'historique des trades: {str(e)}")
            return ""
    
    def _update_loop(self) -> None:
        """
        Boucle de mise à jour périodique des statistiques.
        """
        logger.info("Démarrage de la boucle de mise à jour des statistiques...")
        
        while not self.stop_event.is_set():
            try:
                # Mettre à jour les statistiques
                self.update_stats()
                
                # Attendre jusqu'à la prochaine mise à jour
                for _ in range(self.update_interval):
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)
            
            except Exception as e:
                logger.error(f"❌ Erreur dans la boucle de mise à jour: {str(e)}")
                time.sleep(60)  # Pause en cas d'erreur
        
        logger.info("Boucle de mise à jour des statistiques arrêtée")
    
    def start_update_thread(self, interval: int = 3600) -> None:
        """
        Démarre un thread d'arrière-plan pour mettre à jour les statistiques périodiquement.
        
        Args:
            interval: Intervalle de mise à jour en secondes (par défaut: 1 heure)
        """
        if self.update_thread and self.update_thread.is_alive():
            logger.warning("⚠️ Thread de mise à jour déjà en cours")
            return
        
        self.update_interval = interval
        self.stop_event.clear()
        
        # Effectuer une première mise à jour
        self.update_stats()
        
        # Démarrer le thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        logger.info(f"✅ Thread de mise à jour démarré (intervalle: {interval}s)")
    
    def stop_update_thread(self) -> None:
        """
        Arrête le thread de mise à jour des statistiques.
        """
        if not self.update_thread or not self.update_thread.is_alive():
            return
        
        logger.info("Arrêt du thread de mise à jour...")
        self.stop_event.set()
        
        if self.update_thread:
            self.update_thread.join(timeout=10)
            if self.update_thread.is_alive():
                logger.warning("⚠️ Le thread de mise à jour ne s'est pas arrêté proprement")
        
        logger.info("✅ Thread de mise à jour arrêté")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Récupère les statistiques actuelles.
        
        Returns:
            Dictionnaire des statistiques
        """
        if not self.stats_cache["global"] or datetime.fromisoformat(self.stats_cache["global"].get("updated_at", "2000-01-01T00:00:00")) < datetime.now() - timedelta(hours=1):
            self.update_stats()
        
        return self.stats_cache
    
    def close(self) -> None:
        """
        Ferme proprement le logger PnL.
        """
        # Arrêter le thread de mise à jour
        self.stop_update_thread()
        
        # Fermer la connexion à la base de données
        if self.conn:
            self.conn.close()
            self.conn = None
        
        logger.info("✅ PnLLogger fermé")

# Point d'entrée pour les tests
if __name__ == "__main__":
    # Initialiser le logger PnL
    pnl_logger = PnLLogger()
    
    # Mise à jour des statistiques
    pnl_logger.update_stats()
    
    # Récupérer les statistiques
    stats = pnl_logger.get_stats()
    print(f"Statistiques globales: {json.dumps(stats['global'], indent=2)}")
    
    # Exporter les statistiques
    export_path = pnl_logger.export_stats_to_csv()
    print(f"Statistiques exportées vers: {export_path}")
    
    # Fermer le logger
    pnl_logger.close()