#!/usr/bin/env python3
"""Script pour exécuter Vulture sur le code en évitant les environnements virtuels."""

import os
import sys
from pathlib import Path
from vulture import Vulture

def should_skip_dir(dir_name):
    """Détermine si un répertoire doit être ignoré."""
    skip_dirs = {'.venv', 'venv', 'node_modules', '__pycache__', '.git',
                 'build', 'dist', '.pytest_cache', '.mypy_cache', 'lib64'}
    return dir_name in skip_dirs

def find_python_files(root_dir):
    """Trouve tous les fichiers Python en évitant les répertoires à ignorer."""
    python_files = []
    root_path = Path(root_dir)

    for py_file in root_path.rglob("*.py"):
        # Vérifie si le fichier est dans un répertoire à ignorer
        skip = False
        for parent in py_file.parents:
            if should_skip_dir(parent.name):
                skip = True
                break

        if not skip:
            python_files.append(str(py_file))

    return python_files

def main():
    """Exécute Vulture sur tous les fichiers Python."""
    root_dir = Path(__file__).parent
    print(f"Analyse du code dans: {root_dir}")

    # Trouve tous les fichiers Python
    python_files = find_python_files(root_dir)
    print(f"Fichiers Python trouvés: {len(python_files)}")

    if not python_files:
        print("Aucun fichier Python trouvé!")
        return 1

    # Exécute Vulture
    v = Vulture(min_confidence=80, sort_by_size=True)
    v.scavenge(python_files)

    # Affiche les résultats
    if v.get_unused_code():
        print(f"\n{'='*80}")
        print(f"CODE MORT DÉTECTÉ - {len(v.get_unused_code())} éléments inutilisés")
        print(f"{'='*80}\n")

        for item in v.get_unused_code():
            print(item)

        return 1
    else:
        print("\n✓ Aucun code mort détecté!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
