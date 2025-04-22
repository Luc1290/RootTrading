"""
Module de chargement et de gestion des stratégies d'analyse.
Charge dynamiquement les stratégies disponibles et les exécute sur les données reçues.
"""
import logging
import importlib
import os
import sys
import inspect
from typing import Dict, List, Any, Type, Optional
import time

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import SYMBOLS
from shared.src.schemas import StrategySignal

from analyzer.strategies.base_strategy import BaseStrategy

# Configuration du logging
logger = logging.getLogger(__name__)

class StrategyLoader:
    """
    Chargeur et gestionnaire des stratégies de trading.
    Découvre, instancie et exécute les stratégies disponibles.
    """
    
    def __init__(self, symbols: List[str] = None, strategy_dir: str = None):
        """
        Initialise le chargeur de stratégies.
        
        Args:
            symbols: Liste des symboles à analyser (par défaut: depuis la config)
            strategy_dir: Répertoire contenant les stratégies
        """
        self.symbols = symbols or SYMBOLS
        self.strategy_dir = strategy_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), "strategies")
        self.strategies: Dict[str, Dict[str, BaseStrategy]] = {}  # {symbol: {strategy_name: strategy_instance}}
        
        # Charger les stratégies
        self._load_strategies()
        
        logger.info(f"✅ StrategyLoader initialisé pour {len(self.symbols)} symboles: {', '.join(self.symbols)}")
    
    def _load_strategies(self) -> None:
        """
        Découvre et charge dynamiquement les stratégies disponibles dans le répertoire des stratégies.
        """
        logger.info(f"Chargement des stratégies depuis: {self.strategy_dir}")
        
        # Initialiser le dictionnaire pour chaque symbole
        for symbol in self.symbols:
            self.strategies[symbol] = {}
        
        # Trouver tous les fichiers Python dans le répertoire des stratégies
        strategy_files = [f for f in os.listdir(self.strategy_dir) 
                         if f.endswith('.py') and f != 'base_strategy.py' and not f.startswith('__')]
        
        # Parcourir chaque fichier de stratégie
        for file_name in strategy_files:
            module_name = file_name[:-3]  # Enlever l'extension .py
            
            try:
                # Charger le module dynamiquement
                module = importlib.import_module(f"analyzer.strategies.{module_name}")
                
                # Parcourir toutes les classes du module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Vérifier si c'est une sous-classe de BaseStrategy (sauf BaseStrategy elle-même)
                    if (issubclass(obj, BaseStrategy) and 
                        obj != BaseStrategy and 
                        obj.__module__ == module.__name__):
                        
                        # Créer une instance pour chaque symbole
                        for symbol in self.symbols:
                            try:
                                strategy_instance = obj(symbol=symbol)
                                strategy_name = strategy_instance.name
                                self.strategies[symbol][strategy_name] = strategy_instance
                                logger.info(f"✅ Stratégie chargée: {strategy_name} pour {symbol}")
                            except Exception as e:
                                logger.error(f"❌ Erreur lors de l'instanciation de {name} pour {symbol}: {str(e)}")
            
            except Exception as e:
                logger.error(f"❌ Erreur lors du chargement du module {module_name}: {str(e)}")
        
        # Compter le nombre total de stratégies chargées
        total_strategies = sum(len(strategies) for strategies in self.strategies.values())
        logger.info(f"📊 Total: {total_strategies} stratégies chargées pour {len(self.symbols)} symboles")
    
    def process_market_data(self, data: Dict[str, Any]) -> List[StrategySignal]:
        """
        Traite les données de marché avec toutes les stratégies pour le symbole concerné.
        
        Args:
            data: Données de marché
            
        Returns:
            Liste des signaux générés
        """
        symbol = data.get('symbol')
        if not symbol or symbol not in self.strategies:
            return []
        
        signals = []
        start_time = time.time()
        
        # Parcourir toutes les stratégies pour ce symbole
        for strategy_name, strategy in self.strategies[symbol].items():
            try:
                # Ajouter les données à la stratégie
                strategy.add_market_data(data)
                
                # Analyser les données et générer un signal si les conditions sont remplies
                signal = strategy.analyze()
                if signal:
                    signals.append(signal)
                    logger.info(f"🔔 Signal généré par {strategy_name} pour {symbol}: {signal.side} @ {signal.price}")
            
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'exécution de {strategy_name} pour {symbol}: {str(e)}")
        
        # Mesurer le temps d'exécution
        execution_time = time.time() - start_time
        if signals:
            logger.info(f"⏱️ Analyse de {symbol} terminée en {execution_time:.3f}s avec {len(signals)} signaux")
        else:
            logger.debug(f"⏱️ Analyse de {symbol} terminée en {execution_time:.3f}s sans signal")
        
        return signals
    
    def get_strategy_list(self) -> Dict[str, List[str]]:
        """
        Retourne la liste des stratégies chargées par symbole.
        
        Returns:
            Dictionnaire de la forme {symbole: [liste_des_stratégies]}
        """
        result = {}
        for symbol, strategies in self.strategies.items():
            result[symbol] = list(strategies.keys())
        return result
    
    def get_strategy_count(self) -> int:
        """
        Retourne le nombre total de stratégies chargées.
        
        Returns:
            Nombre de stratégies
        """
        return sum(len(strategies) for strategies in self.strategies.values())

# Fonction utilitaire pour créer une instance singleton du chargeur de stratégies
_strategy_loader_instance = None

def get_strategy_loader() -> StrategyLoader:
    """
    Retourne l'instance singleton du chargeur de stratégies.
    
    Returns:
        Instance du chargeur de stratégies
    """
    global _strategy_loader_instance
    if _strategy_loader_instance is None:
        _strategy_loader_instance = StrategyLoader()
    return _strategy_loader_instance