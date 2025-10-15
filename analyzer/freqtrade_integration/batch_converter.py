"""
BatchConverter - Conversion en masse de stratégies ROOT ↔ Freqtrade.
Permet de convertir toutes les stratégies d'un dossier en une seule opération.
"""

import os
import sys
import importlib
import inspect
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Ajouter le path pour imports ROOT
analyzer_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(analyzer_root)

from strategies.base_strategy import BaseStrategy
from .root_to_freqtrade import RootToFreqtradeAdapter
from .freqtrade_to_root import FreqtradeToRootAdapter

# Import conditionnel Freqtrade
try:
    from freqtrade.strategy import IStrategy
    FREQTRADE_AVAILABLE = True
except ImportError:
    FREQTRADE_AVAILABLE = False
    IStrategy = object

logger = logging.getLogger(__name__)


class BatchConverter:
    """
    Convertisseur en masse de stratégies.
    Supporte ROOT → Freqtrade et Freqtrade → ROOT.
    """

    def __init__(self):
        """Initialise le convertisseur batch."""
        self.strategies_path = Path(analyzer_root) / "strategies"
        self.output_path = Path(analyzer_root) / "freqtrade_integration" / "converted_strategies"

        # Créer dossier de sortie si nécessaire
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Stats de conversion
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }

    def convert_all_root_to_freqtrade(
        self,
        output_dir: Optional[str] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Convertit toutes les stratégies ROOT vers Freqtrade.

        Args:
            output_dir: Dossier de sortie (défaut: converted_strategies/freqtrade)
            exclude_patterns: Patterns de noms de fichiers à exclure

        Returns:
            Dict avec statistiques de conversion
        """
        if not FREQTRADE_AVAILABLE:
            logger.error("Freqtrade non installé")
            return {'error': 'Freqtrade not installed'}

        # Réinitialiser stats
        self.stats = {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0, 'errors': []}

        # Définir dossier de sortie
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = self.output_path / "freqtrade"

        output_path.mkdir(parents=True, exist_ok=True)

        # Patterns à exclure par défaut
        if exclude_patterns is None:
            exclude_patterns = ['base_strategy', '__init__', '__pycache__']

        logger.info(f"Conversion ROOT → Freqtrade: {self.strategies_path}")
        logger.info(f"Sortie: {output_path}")

        # Parcourir tous les fichiers de stratégies
        for strategy_file in self.strategies_path.glob("*.py"):
            strategy_name = strategy_file.stem

            # Vérifier exclusions
            if any(pattern in strategy_name for pattern in exclude_patterns):
                logger.debug(f"Skip: {strategy_name}")
                self.stats['skipped'] += 1
                continue

            self.stats['total'] += 1

            try:
                # Charger la classe de stratégie
                strategy_class = self._load_root_strategy(strategy_name)

                if not strategy_class:
                    self.stats['failed'] += 1
                    self.stats['errors'].append({
                        'strategy': strategy_name,
                        'error': 'Failed to load class'
                    })
                    continue

                # Convertir vers Freqtrade
                adapter = RootToFreqtradeAdapter(strategy_class)
                freqtrade_strategy = adapter.convert()

                # Exporter vers fichier
                output_file = output_path / f"{strategy_name}_freqtrade.py"
                adapter.export_to_file(str(output_file))

                self.stats['success'] += 1
                logger.info(f"✓ Converti: {strategy_name} → {output_file.name}")

            except Exception as e:
                self.stats['failed'] += 1
                self.stats['errors'].append({
                    'strategy': strategy_name,
                    'error': str(e)
                })
                logger.error(f"✗ Erreur {strategy_name}: {e}")

        # Résumé
        logger.info(f"\n=== Résumé conversion ROOT → Freqtrade ===")
        logger.info(f"Total: {self.stats['total']}")
        logger.info(f"Succès: {self.stats['success']}")
        logger.info(f"Échecs: {self.stats['failed']}")
        logger.info(f"Ignorés: {self.stats['skipped']}")

        if self.stats['errors']:
            logger.warning(f"\nErreurs ({len(self.stats['errors'])}):")
            for error in self.stats['errors']:
                logger.warning(f"  - {error['strategy']}: {error['error']}")

        return self.stats.copy()

    def convert_all_freqtrade_to_root(
        self,
        freqtrade_dir: str,
        output_dir: Optional[str] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Convertit toutes les stratégies Freqtrade vers ROOT.

        Args:
            freqtrade_dir: Dossier contenant stratégies Freqtrade
            output_dir: Dossier de sortie (défaut: converted_strategies/root)
            exclude_patterns: Patterns à exclure

        Returns:
            Dict avec statistiques de conversion
        """
        if not FREQTRADE_AVAILABLE:
            logger.error("Freqtrade non installé")
            return {'error': 'Freqtrade not installed'}

        # Réinitialiser stats
        self.stats = {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0, 'errors': []}

        # Définir dossier de sortie
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = self.output_path / "root"

        output_path.mkdir(parents=True, exist_ok=True)

        # Patterns à exclure
        if exclude_patterns is None:
            exclude_patterns = ['__init__', '__pycache__', 'test_']

        freqtrade_path = Path(freqtrade_dir)
        if not freqtrade_path.exists():
            return {'error': f'Directory not found: {freqtrade_dir}'}

        logger.info(f"Conversion Freqtrade → ROOT: {freqtrade_path}")
        logger.info(f"Sortie: {output_path}")

        # Ajouter le dossier Freqtrade au path
        sys.path.insert(0, str(freqtrade_path.parent))

        # Parcourir fichiers Freqtrade
        for strategy_file in freqtrade_path.glob("*.py"):
            strategy_name = strategy_file.stem

            # Vérifier exclusions
            if any(pattern in strategy_name for pattern in exclude_patterns):
                logger.debug(f"Skip: {strategy_name}")
                self.stats['skipped'] += 1
                continue

            self.stats['total'] += 1

            try:
                # Charger la classe de stratégie Freqtrade
                strategy_class = self._load_freqtrade_strategy(strategy_file)

                if not strategy_class:
                    self.stats['failed'] += 1
                    self.stats['errors'].append({
                        'strategy': strategy_name,
                        'error': 'Failed to load Freqtrade class'
                    })
                    continue

                # Convertir vers ROOT
                adapter = FreqtradeToRootAdapter(strategy_class)
                root_strategy = adapter.convert()

                # Exporter vers fichier
                output_file = output_path / f"{strategy_name}_root.py"
                adapter.export_to_file(str(output_file))

                self.stats['success'] += 1
                logger.info(f"✓ Converti: {strategy_name} → {output_file.name}")

            except Exception as e:
                self.stats['failed'] += 1
                self.stats['errors'].append({
                    'strategy': strategy_name,
                    'error': str(e)
                })
                logger.error(f"✗ Erreur {strategy_name}: {e}")

        # Résumé
        logger.info(f"\n=== Résumé conversion Freqtrade → ROOT ===")
        logger.info(f"Total: {self.stats['total']}")
        logger.info(f"Succès: {self.stats['success']}")
        logger.info(f"Échecs: {self.stats['failed']}")
        logger.info(f"Ignorés: {self.stats['skipped']}")

        if self.stats['errors']:
            logger.warning(f"\nErreurs ({len(self.stats['errors'])}):")
            for error in self.stats['errors']:
                logger.warning(f"  - {error['strategy']}: {error['error']}")

        return self.stats.copy()

    def _load_root_strategy(self, strategy_name: str) -> Optional[type]:
        """
        Charge dynamiquement une classe de stratégie ROOT.

        Args:
            strategy_name: Nom du fichier (sans .py)

        Returns:
            Classe de stratégie ou None
        """
        try:
            # Import dynamique
            module_path = f"strategies.{strategy_name}"
            module = importlib.import_module(module_path)

            # Chercher classe héritant de BaseStrategy
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BaseStrategy) and
                    obj != BaseStrategy and
                    obj.__module__ == module.__name__):
                    return obj

            logger.warning(f"Aucune classe BaseStrategy trouvée dans {strategy_name}")
            return None

        except Exception as e:
            logger.error(f"Erreur chargement stratégie ROOT {strategy_name}: {e}")
            return None

    def _load_freqtrade_strategy(self, strategy_file: Path) -> Optional[type]:
        """
        Charge dynamiquement une classe de stratégie Freqtrade.

        Args:
            strategy_file: Path du fichier de stratégie

        Returns:
            Classe de stratégie ou None
        """
        try:
            # Charger le module
            spec = importlib.util.spec_from_file_location(
                strategy_file.stem,
                strategy_file
            )
            if not spec or not spec.loader:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Chercher classe héritant de IStrategy
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, IStrategy) and
                    obj != IStrategy and
                    obj.__module__ == module.__name__):
                    return obj

            logger.warning(f"Aucune classe IStrategy trouvée dans {strategy_file.name}")
            return None

        except Exception as e:
            logger.error(f"Erreur chargement stratégie Freqtrade {strategy_file.name}: {e}")
            return None

    def generate_report(self, output_file: str = "conversion_report.txt") -> None:
        """
        Génère un rapport de conversion détaillé.

        Args:
            output_file: Fichier de sortie du rapport
        """
        try:
            report_path = self.output_path / output_file

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("RAPPORT DE CONVERSION FREQTRADE ↔ ROOT\n")
                f.write("=" * 60 + "\n\n")

                f.write(f"Total stratégies traitées: {self.stats['total']}\n")
                f.write(f"Succès: {self.stats['success']}\n")
                f.write(f"Échecs: {self.stats['failed']}\n")
                f.write(f"Ignorés: {self.stats['skipped']}\n\n")

                if self.stats['errors']:
                    f.write("=" * 60 + "\n")
                    f.write("ERREURS\n")
                    f.write("=" * 60 + "\n")
                    for error in self.stats['errors']:
                        f.write(f"\nStratégie: {error['strategy']}\n")
                        f.write(f"Erreur: {error['error']}\n")

            logger.info(f"Rapport généré: {report_path}")

        except Exception as e:
            logger.error(f"Erreur génération rapport: {e}")


def main():
    """Point d'entrée pour conversion en ligne de commande."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convertisseur batch ROOT ↔ Freqtrade"
    )
    parser.add_argument(
        'direction',
        choices=['root2ft', 'ft2root'],
        help='Direction de conversion (root2ft ou ft2root)'
    )
    parser.add_argument(
        '--input',
        help='Dossier d\'entrée (requis pour ft2root)'
    )
    parser.add_argument(
        '--output',
        help='Dossier de sortie (optionnel)'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Générer rapport de conversion'
    )

    args = parser.parse_args()

    # Configurer logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    converter = BatchConverter()

    if args.direction == 'root2ft':
        stats = converter.convert_all_root_to_freqtrade(output_dir=args.output)
    else:
        if not args.input:
            logger.error("--input requis pour ft2root")
            return

        stats = converter.convert_all_freqtrade_to_root(
            freqtrade_dir=args.input,
            output_dir=args.output
        )

    if args.report:
        converter.generate_report()


if __name__ == '__main__':
    main()
