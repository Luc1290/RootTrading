"""
Module de chargement et de gestion des stratégies d'analyse.
Charge dynamiquement les stratégies disponibles et les exécute sur les données reçues.
"""

import importlib
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Type, List, Optional, Any

# Ajouter le répertoire parent au path pour les imports
analyzer_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(analyzer_root)

from strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class StrategyLoader:
    """Gestionnaire de chargement dynamique des stratégies."""
    
    def __init__(self) -> None:
        self.strategies: Dict[str, Type[BaseStrategy]] = {}
        self.strategies_path = Path(__file__).parent.parent / "strategies"
        
        # Stratégies à exclure du chargement
        self.excluded_strategies = {
            'base_strategy',
            '__init__',
            '__pycache__'
        }
        
    def load_strategies(self) -> None:
        """Charge toutes les stratégies disponibles depuis le dossier strategies."""
        logger.info("Chargement des stratégies...")
        
        if not self.strategies_path.exists():
            logger.error(f"Dossier strategies non trouvé: {self.strategies_path}")
            return
            
        # Parcourir tous les fichiers Python dans le dossier strategies
        for strategy_file in self.strategies_path.glob("*.py"):
            strategy_name = strategy_file.stem
            
            # Ignorer les fichiers exclus
            if strategy_name in self.excluded_strategies:
                continue
                
            try:
                self._load_strategy_from_file(strategy_name)
            except Exception as e:
                logger.error(f"Erreur lors du chargement de {strategy_name}: {e}")
                
        logger.info(f"Stratégies chargées: {len(self.strategies)}")
        for name in self.strategies.keys():
            logger.info(f"  - {name}")
            
    def _load_strategy_from_file(self, strategy_name: str) -> None:
        """
        Charge une stratégie depuis un fichier spécifique.
        
        Args:
            strategy_name: Nom du fichier de stratégie (sans .py)
        """
        try:
            # Import dynamique du module
            module_path = f"strategies.{strategy_name}"
            module = importlib.import_module(module_path)
            
            # Recherche de la classe de stratégie dans le module
            strategy_class = None
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Vérifier que c'est une sous-classe de BaseStrategy et pas BaseStrategy elle-même
                if (issubclass(obj, BaseStrategy) and 
                    obj != BaseStrategy and 
                    obj.__module__ == module.__name__):
                    strategy_class = obj
                    break
                    
            if strategy_class:
                self.strategies[strategy_name] = strategy_class
                logger.debug(f"Stratégie chargée: {strategy_name} -> {strategy_class.__name__}")
            else:
                logger.warning(f"Aucune classe de stratégie trouvée dans {strategy_name}")
                
        except ImportError as e:
            logger.error(f"Impossible d'importer {strategy_name}: {e}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de {strategy_name}: {e}")
            
    def get_strategy(self, strategy_name: str) -> Optional[Type[BaseStrategy]]:
        """
        Récupère une stratégie par son nom.
        
        Args:
            strategy_name: Nom de la stratégie
            
        Returns:
            Classe de la stratégie ou None si non trouvée
        """
        return self.strategies.get(strategy_name)
        
    def get_all_strategies(self) -> Dict[str, Type[BaseStrategy]]:
        """
        Récupère toutes les stratégies chargées.
        
        Returns:
            Dictionnaire nom -> classe de stratégie
        """
        return self.strategies.copy()
        
    def get_strategy_names(self) -> List[str]:
        """
        Récupère la liste des noms de stratégies disponibles.
        
        Returns:
            Liste des noms de stratégies
        """
        return list(self.strategies.keys())
        
    def reload_strategies(self) -> None:
        """Recharge toutes les stratégies (utile pour le développement)."""
        logger.info("Rechargement des stratégies...")
        
        # Vider le cache des stratégies
        self.strategies.clear()
        
        # Invalider les modules déjà importés pour forcer le rechargement
        modules_to_remove = []
        for module_name in sys.modules:
            if module_name.startswith('strategies.'):
                modules_to_remove.append(module_name)
                
        for module_name in modules_to_remove:
            del sys.modules[module_name]
            
        # Recharger toutes les stratégies
        self.load_strategies()
        
    def validate_strategy(self, strategy_class: Type[BaseStrategy]) -> bool:
        """
        Valide qu'une classe de stratégie respecte l'interface requise.
        
        Args:
            strategy_class: Classe de stratégie à valider
            
        Returns:
            True si la stratégie est valide, False sinon
        """
        try:
            # Vérifier que c'est une sous-classe de BaseStrategy
            if not issubclass(strategy_class, BaseStrategy):
                logger.error(f"{strategy_class.__name__} n'hérite pas de BaseStrategy")
                return False
                
            # Vérifier que la méthode generate_signal est implémentée
            if not hasattr(strategy_class, 'generate_signal'):
                logger.error(f"{strategy_class.__name__} n'implémente pas generate_signal")
                return False
                
            # Vérifier la signature du constructeur
            sig = inspect.signature(strategy_class.__init__)
            required_params = {'symbol', 'data', 'indicators'}
            actual_params = set(sig.parameters.keys()) - {'self'}
            
            if not required_params.issubset(actual_params):
                missing = required_params - actual_params
                logger.error(f"{strategy_class.__name__} manque les paramètres: {missing}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la validation de {strategy_class.__name__}: {e}")
            return False
            
    def get_strategy_info(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """
        Récupère les informations d'une stratégie.
        
        Args:
            strategy_name: Nom de la stratégie
            
        Returns:
            Dictionnaire avec les informations de la stratégie
        """
        strategy_class = self.get_strategy(strategy_name)
        if not strategy_class:
            return None
            
        return {
            'name': strategy_name,
            'class_name': strategy_class.__name__,
            'module': strategy_class.__module__,
            'doc': strategy_class.__doc__ or "Aucune documentation",
            'file': inspect.getfile(strategy_class)
        }
        
    def filter_strategies(self, 
                         enabled_only: bool = True,
                         categories: Optional[List[str]] = None) -> Dict[str, Type[BaseStrategy]]:
        """
        Filtre les stratégies selon des critères.
        
        Args:
            enabled_only: Si True, ne retourne que les stratégies activées
            categories: Liste des catégories à inclure
            
        Returns:
            Dictionnaire filtré des stratégies
        """
        filtered = {}
        
        for name, strategy_class in self.strategies.items():
            # Pour l'instant, on retourne toutes les stratégies
            # Plus tard, on pourra ajouter des attributs de configuration
            filtered[name] = strategy_class
            
        return filtered