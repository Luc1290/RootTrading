"""
Module de chargement et de gestion des validators de signaux.
Charge dynamiquement les validators disponibles et les exécute sur les signaux reçus.
"""

import importlib
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Type, List, Optional, Any

# Ajouter le répertoire parent au path pour les imports
aggregator_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(aggregator_root)

from validators.base_validator import BaseValidator

logger = logging.getLogger(__name__)


class ValidatorLoader:
    """Gestionnaire de chargement dynamique des validators."""
    
    def __init__(self) -> None:
        self.validators: Dict[str, Type[BaseValidator]] = {}
        self.validators_path = Path(__file__).parent.parent / "validators"
        
        # Validators à exclure du chargement
        self.excluded_validators = {
            'base_validator',
            '__init__',
            '__pycache__'
        }
        
    def load_validators(self) -> None:
        """Charge tous les validators disponibles depuis le dossier validators."""
        logger.info("Chargement des validators...")
        
        if not self.validators_path.exists():
            logger.error(f"Dossier validators non trouvé: {self.validators_path}")
            return
            
        # Parcourir tous les fichiers Python dans le dossier validators
        for validator_file in self.validators_path.glob("*.py"):
            validator_name = validator_file.stem
            
            # Ignorer les fichiers exclus
            if validator_name in self.excluded_validators:
                continue
                
            try:
                self._load_validator_from_file(validator_name)
            except Exception as e:
                logger.error(f"Erreur lors du chargement de {validator_name}: {e}")
                
        logger.info(f"Validators chargés: {len(self.validators)}")
        for name in self.validators.keys():
            logger.info(f"  - {name}")
            
    def _load_validator_from_file(self, validator_name: str) -> None:
        """
        Charge un validator depuis un fichier spécifique.
        
        Args:
            validator_name: Nom du fichier de validator (sans .py)
        """
        try:
            # Import dynamique du module
            module_path = f"validators.{validator_name}"
            module = importlib.import_module(module_path)
            
            # Recherche de la classe de validator dans le module
            validator_class = None
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Vérifier que c'est une sous-classe de BaseValidator et pas BaseValidator elle-même
                if (issubclass(obj, BaseValidator) and 
                    obj != BaseValidator and 
                    obj.__module__ == module.__name__):
                    validator_class = obj
                    break
                    
            if validator_class:
                self.validators[validator_name] = validator_class
                logger.debug(f"Validator chargé: {validator_name} -> {validator_class.__name__}")
            else:
                logger.warning(f"Aucune classe de validator trouvée dans {validator_name}")
                
        except ImportError as e:
            logger.error(f"Impossible d'importer {validator_name}: {e}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de {validator_name}: {e}")
            
    def get_validator(self, validator_name: str) -> Optional[Type[BaseValidator]]:
        """
        Récupère un validator par son nom.
        
        Args:
            validator_name: Nom du validator
            
        Returns:
            Classe du validator ou None si non trouvée
        """
        return self.validators.get(validator_name)
        
    def get_all_validators(self) -> Dict[str, Type[BaseValidator]]:
        """
        Récupère tous les validators chargés.
        
        Returns:
            Dictionnaire nom -> classe de validator
        """
        return self.validators.copy()
        
    def get_validator_names(self) -> List[str]:
        """
        Récupère la liste des noms de validators disponibles.
        
        Returns:
            Liste des noms de validators
        """
        return list(self.validators.keys())
        
    def reload_validators(self) -> None:
        """Recharge tous les validators (utile pour le développement)."""
        logger.info("Rechargement des validators...")
        
        # Vider le cache des validators
        self.validators.clear()
        
        # Invalider les modules déjà importés pour forcer le rechargement
        modules_to_remove = []
        for module_name in sys.modules:
            if module_name.startswith('validators.'):
                modules_to_remove.append(module_name)
                
        for module_name in modules_to_remove:
            del sys.modules[module_name]
            
        # Recharger tous les validators
        self.load_validators()
        
    def validate_validator(self, validator_class: Type[BaseValidator]) -> bool:
        """
        Valide qu'une classe de validator respecte l'interface requise.
        
        Args:
            validator_class: Classe de validator à valider
            
        Returns:
            True si le validator est valide, False sinon
        """
        try:
            # Vérifier que c'est une sous-classe de BaseValidator
            if not issubclass(validator_class, BaseValidator):
                logger.error(f"{validator_class.__name__} n'hérite pas de BaseValidator")
                return False
                
            # Vérifier que la méthode validate_signal est implémentée
            if not hasattr(validator_class, 'validate_signal'):
                logger.error(f"{validator_class.__name__} n'implémente pas validate_signal")
                return False
                
            # Vérifier la signature du constructeur
            sig = inspect.signature(validator_class.__init__)
            required_params = {'symbol', 'data', 'context'}
            actual_params = set(sig.parameters.keys()) - {'self'}
            
            if not required_params.issubset(actual_params):
                missing = required_params - actual_params
                logger.error(f"{validator_class.__name__} manque les paramètres: {missing}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la validation de {validator_class.__name__}: {e}")
            return False
            
    def get_validator_info(self, validator_name: str) -> Optional[Dict[str, Any]]:
        """
        Récupère les informations d'un validator.
        
        Args:
            validator_name: Nom du validator
            
        Returns:
            Dictionnaire avec les informations du validator
        """
        validator_class = self.get_validator(validator_name)
        if not validator_class:
            return None
            
        return {
            'name': validator_name,
            'class_name': validator_class.__name__,
            'module': validator_class.__module__,
            'doc': validator_class.__doc__ or "Aucune documentation",
            'file': inspect.getfile(validator_class)
        }
        
    def filter_validators(self, 
                         enabled_only: bool = True,
                         categories: Optional[List[str]] = None) -> Dict[str, Type[BaseValidator]]:
        """
        Filtre les validators selon des critères.
        
        Args:
            enabled_only: Si True, ne retourne que les validators activés
            categories: Liste des catégories à inclure
            
        Returns:
            Dictionnaire filtré des validators
        """
        filtered = {}
        
        for name, validator_class in self.validators.items():
            # Pour l'instant, on retourne tous les validators
            # Plus tard, on pourra ajouter des attributs de configuration
            filtered[name] = validator_class
            
        return filtered
        
    def get_validation_categories(self) -> List[str]:
        """
        Récupère les catégories de validation disponibles.
        
        Returns:
            Liste des catégories de validation
        """
        categories = set()
        
        for validator_name in self.validators.keys():
            # Extraire la catégorie du nom (ex: ADX_TrendStrength_Validator -> TrendStrength)
            if '_' in validator_name:
                parts = validator_name.split('_')
                if len(parts) >= 2 and parts[-1] == 'Validator':
                    category = '_'.join(parts[:-1])
                    categories.add(category)
                    
        return sorted(list(categories))