"""
Module d'optimisation des stratégies de trading.
Utilise les données historiques pour ajuster les paramètres des stratégies
afin d'améliorer leurs performances.
"""
import logging
import json
import os
import time
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import inspect
import threading
import psycopg2
from psycopg2.extras import RealDictCursor
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from joblib import Parallel, delayed
import importlib

# Importer les modules partagés
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import get_db_url, SYMBOLS
from shared.src.enums import OrderSide

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('strategy_tuner.log')
    ]
)
logger = logging.getLogger(__name__)

class StrategyTuner:
    """
    Optimiseur de stratégies de trading.
    Utilise des techniques d'optimisation pour améliorer les paramètres des stratégies.
    """
    
    def __init__(self, db_url: str = None, 
                strategies_dir: str = None,
                results_dir: str = "./tuning_results"):
        """
        Initialise l'optimiseur de stratégies.
        
        Args:
            db_url: URL de connexion à la base de données
            strategies_dir: Répertoire des stratégies
            results_dir: Répertoire pour les résultats d'optimisation
        """
        self.db_url = db_url or get_db_url()
        
        # Chemin vers le répertoire des stratégies
        if strategies_dir:
            self.strategies_dir = strategies_dir
        else:
            # Par défaut, chercher dans analyzer/strategies/
            self.strategies_dir = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 
                "../../analyzer/strategies"
            ))
        
        self.results_dir = results_dir
        self.conn = None
        
        # Créer le répertoire de résultats s'il n'existe pas
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Dictionnaire pour stocker les instances de stratégies
        self.strategy_instances = {}
        
        # Initialiser la connexion à la base de données
        self._init_db_connection()
        
        logger.info(f"✅ StrategyTuner initialisé")
    
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
    
    def load_strategy_module(self, strategy_name: str) -> Optional[Any]:
        """
        Charge dynamiquement une stratégie.
        
        Args:
            strategy_name: Nom du fichier Python de la stratégie (sans .py)
            
        Returns:
            Module de la stratégie ou None en cas d'échec
        """
        try:
            # S'assurer que le fichier existe
            strategy_file = os.path.join(self.strategies_dir, f"{strategy_name}.py")
            if not os.path.exists(strategy_file):
                logger.error(f"❌ Fichier de stratégie non trouvé: {strategy_file}")
                return None
            
            # Charger le module
            module_name = f"analyzer.strategies.{strategy_name}"
            strategy_module = importlib.import_module(module_name)
            
            logger.info(f"✅ Module de stratégie chargé: {module_name}")
            return strategy_module
            
        except ImportError as e:
            logger.error(f"❌ Erreur lors du chargement de la stratégie {strategy_name}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"❌ Erreur inattendue lors du chargement de la stratégie {strategy_name}: {str(e)}")
            return None
    
    def get_strategy_class(self, strategy_module: Any, strategy_type: str) -> Optional[type]:
        """
        Trouve la classe de stratégie dans un module.
        
        Args:
            strategy_module: Module contenant la stratégie
            strategy_type: Type de stratégie (ex: 'RSIStrategy', 'BollingerStrategy')
            
        Returns:
            Classe de stratégie ou None si non trouvée
        """
        try:
            # Parcourir toutes les classes du module
            for name, obj in inspect.getmembers(strategy_module, inspect.isclass):
                # Rechercher la classe qui correspond au type de stratégie
                if name == strategy_type:
                    # Vérifier que c'est une sous-classe de BaseStrategy
                    if hasattr(obj, '__bases__') and any('BaseStrategy' in str(base) for base in obj.__bases__):
                        return obj
            
            logger.warning(f"⚠️ Classe de stratégie '{strategy_type}' non trouvée dans le module")
            return None
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la recherche de la classe de stratégie: {str(e)}")
            return None
    
    def get_strategy_instance(self, strategy_name: str, symbol: str, params: Dict[str, Any]) -> Optional[Any]:
        """
        Crée une instance de stratégie avec des paramètres spécifiques.
        
        Args:
            strategy_name: Nom de la stratégie (ex: 'rsi' pour le fichier rsi.py)
            symbol: Symbole de trading
            params: Paramètres à utiliser pour l'instance
            
        Returns:
            Instance de stratégie ou None en cas d'échec
        """
        # Déterminer le nom de classe de stratégie à partir du nom de stratégie
        class_name_map = {
            'rsi': 'RSIStrategy',
            'bollinger': 'BollingerStrategy',
            'ema_cross': 'EMACrossStrategy',
            'breakout': 'BreakoutStrategy',
            'reversal_divergence': 'ReversalDivergenceStrategy',
            'ride_or_react': 'RideOrReactStrategy'
        }
        
        if strategy_name not in class_name_map:
            logger.error(f"❌ Nom de stratégie non reconnu: {strategy_name}")
            return None
        
        class_name = class_name_map[strategy_name]
        
        try:
            # Charger le module de stratégie
            strategy_module = self.load_strategy_module(strategy_name)
            if not strategy_module:
                return None
            
            # Obtenir la classe de stratégie
            strategy_class = self.get_strategy_class(strategy_module, class_name)
            if not strategy_class:
                return None
            
            # Créer l'instance
            strategy_instance = strategy_class(symbol=symbol, params=params)
            
            logger.info(f"✅ Instance de stratégie créée: {class_name} pour {symbol}")
            return strategy_instance
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la création de l'instance de stratégie: {str(e)}")
            return None
    
    def get_market_data(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Récupère les données de marché pour un symbole.
        
        Args:
            symbol: Symbole de trading
            days: Nombre de jours de données à récupérer
            
        Returns:
            DataFrame avec les données de marché ou None en cas d'échec
        """
        if not self._ensure_connection():
            logger.error("❌ Impossible de récupérer les données de marché: pas de connexion à la base de données")
            return None
        
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Requête pour les données de marché
                query = """
                SELECT 
                    time,
                    symbol,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM 
                    market_data
                WHERE 
                    symbol = %s
                    AND time >= NOW() - INTERVAL '%s days'
                ORDER BY 
                    time
                """
                
                cursor.execute(query, (symbol, days))
                results = cursor.fetchall()
                
                if not results:
                    logger.warning(f"⚠️ Aucune donnée de marché trouvée pour {symbol}")
                    return None
                
                # Convertir en DataFrame
                df = pd.DataFrame(results)
                
                # Définir le timestamp comme index
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                
                # Convertir les colonnes numériques en float
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    df[col] = df[col].astype(float)
                
                return df
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la récupération des données de marché: {str(e)}")
            return None
    
    def convert_to_market_data_format(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convertit un DataFrame en liste de dictionnaires au format attendu par les stratégies.
        
        Args:
            df: DataFrame avec les données de marché
            
        Returns:
            Liste de dictionnaires au format standard
        """
        market_data = []
        
        for idx, row in df.iterrows():
            data = {
                'symbol': row['symbol'],
                'start_time': int(idx.timestamp() * 1000),  # Timestamp en millisecondes
                'close_time': int((idx + pd.Timedelta(minutes=1)).timestamp() * 1000),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'is_closed': True  # Considérer toutes les données comme fermées pour le backtest
            }
            market_data.append(data)
        
        return market_data
    
    def backtest_strategy(self, strategy_instance: Any, market_data: List[Dict[str, Any]], 
                         use_stop_loss: bool = True, use_take_profit: bool = True) -> Dict[str, Any]:
        """
        Effectue un backtest d'une stratégie sur des données historiques.
        
        Args:
            strategy_instance: Instance de la stratégie à tester
            market_data: Données de marché au format standard
            use_stop_loss: Activer les stop loss dans le backtest
            use_take_profit: Activer les take profit dans le backtest
            
        Returns:
            Résultats du backtest
        """
        # Initialiser les résultats
        results = {
            "trades": [],
            "signals": [],
            "pnl": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_profit": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0
        }
        
        try:
            # Réinitialiser la stratégie
            strategy_instance.data_buffer.clear()
            
            # État du trading
            in_position = False
            entry_price = 0.0
            entry_time = None
            position_type = None
            stop_loss = None
            take_profit = None
            
            # Statistiques
            trades = []
            signals = []
            equity_curve = [1000.0]  # Commencer avec 1000 unités
            
            # Ajouter les données dans l'ordre chronologique
            for data in market_data:
                # Ajouter les données à la stratégie
                strategy_instance.add_market_data(data)
                
                # Vérifier les stop loss et take profit si en position
                if in_position and position_type == "LONG":
                    current_price = data['close']
                    current_time = datetime.fromtimestamp(data['start_time'] / 1000)
                    
                    # Vérifier si le stop loss est atteint
                    if use_stop_loss and stop_loss is not None and current_price <= stop_loss:
                        # Exécuter le stop loss
                        exit_price = stop_loss
                        exit_time = current_time
                        pnl_percent = (exit_price - entry_price) / entry_price * 100
                        
                        # Enregistrer le trade
                        trade = {
                            "entry_time": entry_time,
                            "exit_time": exit_time,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "pnl_percent": pnl_percent,
                            "type": position_type,
                            "exit_reason": "stop_loss"
                        }
                        trades.append(trade)
                        
                        # Mettre à jour l'equity curve
                        last_equity = equity_curve[-1]
                        new_equity = last_equity * (1 + pnl_percent / 100)
                        equity_curve.append(new_equity)
                        
                        # Réinitialiser l'état
                        in_position = False
                        entry_price = 0.0
                        entry_time = None
                        position_type = None
                        stop_loss = None
                        take_profit = None
                        continue
                    
                    # Vérifier si le take profit est atteint
                    if use_take_profit and take_profit is not None and current_price >= take_profit:
                        # Exécuter le take profit
                        exit_price = take_profit
                        exit_time = current_time
                        pnl_percent = (exit_price - entry_price) / entry_price * 100
                        
                        # Enregistrer le trade
                        trade = {
                            "entry_time": entry_time,
                            "exit_time": exit_time,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "pnl_percent": pnl_percent,
                            "type": position_type,
                            "exit_reason": "take_profit"
                        }
                        trades.append(trade)
                        
                        # Mettre à jour l'equity curve
                        last_equity = equity_curve[-1]
                        new_equity = last_equity * (1 + pnl_percent / 100)
                        equity_curve.append(new_equity)
                        
                        # Réinitialiser l'état
                        in_position = False
                        entry_price = 0.0
                        entry_time = None
                        position_type = None
                        stop_loss = None
                        take_profit = None
                        continue
                
                # Générer des signaux
                signal = strategy_instance.analyze()
                
                if signal:
                    signals.append({
                        "time": datetime.fromtimestamp(data['start_time'] / 1000),
                        "price": data['close'],
                        "side": signal.side.value
                    })
                    
                    # Simuler l'exécution du signal
                    if signal.side == OrderSide.BUY and not in_position:
                        # Ouvrir une position longue
                        in_position = True
                        entry_price = data['close']
                        entry_time = datetime.fromtimestamp(data['start_time'] / 1000)
                        position_type = "LONG"
                        
                        # Définir stop loss et take profit si fournis dans le signal
                        if use_stop_loss and 'stop_price' in signal.metadata:
                            stop_loss = float(signal.metadata['stop_price'])
                        else:
                            stop_loss = entry_price * 0.95  # 5% par défaut
                            
                        if use_take_profit and 'target_price' in signal.metadata:
                            take_profit = float(signal.metadata['target_price'])
                        else:
                            take_profit = entry_price * 1.10  # 10% par défaut
                    
                    elif signal.side == OrderSide.SELL and in_position and position_type == "LONG":
                        # Fermer une position longue
                        exit_price = data['close']
                        exit_time = datetime.fromtimestamp(data['start_time'] / 1000)
                        pnl_percent = (exit_price - entry_price) / entry_price * 100
                        
                        # Enregistrer le trade
                        trade = {
                            "entry_time": entry_time,
                            "exit_time": exit_time,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "pnl_percent": pnl_percent,
                            "type": position_type,
                            "exit_reason": "signal"
                        }
                        trades.append(trade)
                        
                        # Mettre à jour l'equity curve
                        last_equity = equity_curve[-1]
                        new_equity = last_equity * (1 + pnl_percent / 100)
                        equity_curve.append(new_equity)
                        
                        # Réinitialiser l'état
                        in_position = False
                        entry_price = 0.0
                        entry_time = None
                        position_type = None
                    
                    elif signal.side == OrderSide.SELL and not in_position:
                        # Ouvrir une position courte (si autorisé)
                        # Note: Pour certains systèmes, les positions courtes ne sont pas autorisées
                        # Dans ce cas, ignorer ce signal
                        pass
            
            # Fermer toute position ouverte à la fin de la période
            if in_position:
                exit_price = market_data[-1]['close']
                exit_time = datetime.fromtimestamp(market_data[-1]['start_time'] / 1000)
                
                if position_type == "LONG":
                    pnl_percent = (exit_price - entry_price) / entry_price * 100
                else:
                    pnl_percent = (entry_price - exit_price) / entry_price * 100
                
                # Enregistrer le trade
                trade = {
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_percent": pnl_percent,
                    "type": position_type,
                    "exit_reason": "end_of_period"
                }
                trades.append(trade)
                
                # Mettre à jour l'equity curve
                last_equity = equity_curve[-1]
                new_equity = last_equity * (1 + pnl_percent / 100)
                equity_curve.append(new_equity)
            
            # Calculer les métriques
            if trades:
                # PnL total
                pnl = sum(trade["pnl_percent"] for trade in trades)
                
                # Win rate
                winning_trades = sum(1 for trade in trades if trade["pnl_percent"] > 0)
                win_rate = winning_trades / len(trades) * 100
                
                # Profit factor
                gross_profit = sum(trade["pnl_percent"] for trade in trades if trade["pnl_percent"] > 0)
                gross_loss = abs(sum(trade["pnl_percent"] for trade in trades if trade["pnl_percent"] < 0))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
                # Profit moyen par trade
                avg_profit = pnl / len(trades)
                
                # Maximum drawdown
                max_dd_percent = 0.0
                peak = equity_curve[0]
                
                for equity in equity_curve:
                    if equity > peak:
                        peak = equity
                    dd = (peak - equity) / peak * 100
                    max_dd_percent = max(max_dd_percent, dd)
                
                # Sharpe ratio
                equity_returns = [(equity_curve[i] / equity_curve[i-1] - 1) for i in range(1, len(equity_curve))]
                if equity_returns and np.std(equity_returns) > 0:
                    sharpe_ratio = np.mean(equity_returns) / np.std(equity_returns) * np.sqrt(252)  # Annualisé
                else:
                    sharpe_ratio = 0.0
                
                # Mettre à jour les résultats
                results.update({
                    "trades": trades,
                    "signals": signals,
                    "pnl": pnl,
                    "win_rate": win_rate,
                    "profit_factor": profit_factor,
                    "avg_profit": avg_profit,
                    "max_drawdown": max_dd_percent,
                    "sharpe_ratio": sharpe_ratio,
                    "total_trades": len(trades),
                    "equity_curve": equity_curve
                })
            
            return results
        
        except Exception as e:
            logger.error(f"❌ Erreur lors du backtest de la stratégie: {str(e)}")
            return results
    
    def objective_function(self, params: List[float], 
                         strategy_name: str, symbol: str, 
                         market_data: List[Dict[str, Any]],
                         param_names: List[str]) -> float:
        """
        Fonction objectif pour l'optimisation des paramètres.
        
        Args:
            params: Valeurs des paramètres à tester
            strategy_name: Nom de la stratégie
            symbol: Symbole de trading
            market_data: Données de marché
            param_names: Noms des paramètres
            
        Returns:
            Score négatif (pour minimisation)
        """
        # Convertir la liste de paramètres en dictionnaire
        params_dict = {param_names[i]: params[i] for i in range(len(params))}
        
        # Créer une instance de stratégie avec ces paramètres
        strategy_instance = self.get_strategy_instance(strategy_name, symbol, params_dict)
        
        if not strategy_instance:
            return float('inf')  # Retourner une valeur très élevée en cas d'échec
        
        # Exécuter le backtest
        results = self.backtest_strategy(strategy_instance, market_data)
        
        # Calculer un score composite (à minimiser, donc négatif)
        # Le score peut être ajusté en fonction des priorités
        score = 0.0
        
        # Le PnL est la métrique principale
        if results["pnl"] != 0:
            score -= results["pnl"]
        
        # Pénaliser un faible nombre de trades
        if results["total_trades"] < 5:
            score += (5 - results["total_trades"]) * 10
        
        # Pénaliser un drawdown élevé
        score += results["max_drawdown"] * 0.5
        
        # Bonus pour un bon ratio de Sharpe
        if results["sharpe_ratio"] > 0:
            score -= results["sharpe_ratio"] * 5
        
        return score
    
    def optimize_strategy_params(self, strategy_name: str, symbol: str, 
                              param_ranges: Dict[str, Tuple[float, float]],
                              days: int = 90,
                              method: str = 'L-BFGS-B') -> Dict[str, Any]:
        """
        Optimise les paramètres d'une stratégie pour un symbole donné.
        
        Args:
            strategy_name: Nom de la stratégie
            symbol: Symbole de trading
            param_ranges: Plages de valeurs pour chaque paramètre
            days: Nombre de jours de données à utiliser
            method: Méthode d'optimisation à utiliser ('L-BFGS-B', 'SLSQP', 'Powell')
            
        Returns:
            Résultats de l'optimisation
        """
        logger.info(f"Optimisation de la stratégie {strategy_name} pour {symbol}...")
        
        # Récupérer les données de marché
        df = self.get_market_data(symbol, days)
        if df is None:
            logger.error(f"❌ Pas de données de marché disponibles pour {symbol}")
            return {"success": False, "error": "Pas de données disponibles"}
        
        # Convertir en format standard
        market_data = self.convert_to_market_data_format(df)
        
        # Préparer les paramètres pour l'optimisation
        param_names = list(param_ranges.keys())
        param_bounds = [param_ranges[param] for param in param_names]
        
        # Paramètres initiaux (milieu de chaque plage)
        initial_params = [(bounds[0] + bounds[1]) / 2 for bounds in param_bounds]
        
        try:
            # Définir la fonction objectif
            obj_func = lambda p: self.objective_function(p, strategy_name, symbol, market_data, param_names)
            
            # Exécuter l'optimisation
            result = minimize(
                obj_func,
                initial_params,
                bounds=param_bounds,
                method=method
            )
            
            # Récupérer les paramètres optimaux
            optimized_params = {param_names[i]: result.x[i] for i in range(len(param_names))}
            
            # Créer une instance avec les paramètres optimisés
            strategy_instance = self.get_strategy_instance(strategy_name, symbol, optimized_params)
            
            # Exécuter un backtest final avec les paramètres optimisés
            backtest_results = self.backtest_strategy(strategy_instance, market_data)
            
            # Préparer les résultats
            optimization_results = {
                "success": True,
                "strategy": strategy_name,
                "symbol": symbol,
                "optimized_params": optimized_params,
                "backtest_results": backtest_results,
                "optimization_info": {
                    "param_ranges": param_ranges,
                    "data_period_days": days,
                    "total_data_points": len(market_data),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Sauvegarder les résultats
            self._save_optimization_results(optimization_results)
            
            logger.info(f"✅ Optimisation terminée pour {strategy_name} ({symbol})")
            logger.info(f"Paramètres optimisés: {optimized_params}")
            logger.info(f"PnL: {backtest_results['pnl']:.2f}%, Win Rate: {backtest_results['win_rate']:.2f}%")
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'optimisation de la stratégie: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _save_optimization_results(self, results: Dict[str, Any]) -> str:
        """
        Sauvegarde les résultats d'optimisation dans un fichier.
        
        Args:
            results: Résultats de l'optimisation
            
        Returns:
            Chemin du fichier sauvegardé
        """
        strategy = results["strategy"]
        symbol = results["symbol"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Créer le nom du fichier
        filename = f"{strategy}_{symbol}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Sauvegarder au format JSON
        try:
            with open(filepath, 'w') as f:
                # Convertir les valeurs numpy en Python natif
                def convert_numpy(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.float64) or isinstance(obj, np.float32):
                        return float(obj)
                    elif isinstance(obj, np.int64) or isinstance(obj, np.int32):
                        return int(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_numpy(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy(item) for item in obj]
                    elif isinstance(obj, datetime):
                        return obj.isoformat()
                    else:
                        return obj
                
                # Convertir les résultats et enregistrer
                clean_results = convert_numpy(results)
                json.dump(clean_results, f, indent=2)
                
            logger.info(f"✅ Résultats d'optimisation sauvegardés dans {filepath}")
            return filepath
        
        except Exception as e:
            logger.error(f"❌ Erreur lors de la sauvegarde des résultats: {str(e)}")
            return ""