#!/usr/bin/env python3
"""
Script pour corriger automatiquement les erreurs Ruff courantes.
"""

import os
import re
from pathlib import Path


def fix_try401(content: str) -> str:
    """
    Corrige TRY401: Supprime {e} des messages logger.exception()

    Exemples:
    - logger.exception("Error") -> logger.exception("Error")
    - logger.exception("Error {x}") -> logger.exception(f"Error {x}")
    """
    # Pattern pour logger.exception avec {e} à la fin
    patterns = [
        # f"message: {e}"
        (r'logger\.exception\(f"([^"]+):\s*\{e\}"\)',
         r'logger.exception("\1")'),
        # f"message {e}"
        (r'logger\.exception\(f"([^"]+)\s+\{e\}"\)',
         r'logger.exception("\1")'),
        # f"message: {e!s}"
        (r'logger\.exception\(f"([^"]+):\s*\{e!s\}"\)',
         r'logger.exception("\1")'),
        # Cas où il y a d'autres variables avant {e}
        (r'logger\.exception\(f"([^"]*\{[^}]+\}[^"]*?):\s*\{e\}"\)',
         r'logger.exception(f"\1")'),
        (r'logger\.exception\(f"([^"]*\{[^}]+\}[^"]*?)\s+\{e\}"\)',
         r'logger.exception(f"\1")'),
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    return content


def fix_dtz003(content: str) -> str:
    """
    Corrige DTZ003: Remplace datetime.now(timezone.utc) par datetime.now(timezone.utc)
    """
    # Vérifier si timezone est déjà importé
    has_timezone_import = 'from datetime import' in content and 'timezone' in content

    if not has_timezone_import and 'datetime.now(timezone.utc)' in content:
        # Ajouter timezone à l'import
        content = re.sub(
            r'from datetime import ([^;\n]+)',
            lambda m: f"from datetime import {m.group(1)}, timezone" if 'timezone' not in m.group(1) else m.group(0),
            content,
            count=1)

    # Remplacer datetime.now(timezone.utc) par datetime.now(timezone.utc)
    content = content.replace(
        'datetime.now(timezone.utc)',
        'datetime.now(timezone.utc)')

    return content


def fix_dtz005(content: str) -> str:
    """
    Corrige DTZ005: Remplace datetime.now() par datetime.now(timezone.utc)

    Note: Seulement pour les cas où timezone.utc devrait être utilisé
    """
    # Pattern pour datetime.now() sans argument
    # On remplace seulement si c'est clairement pour un timestamp UTC
    patterns = [
        # .isoformat() suggère un timestamp UTC
        (r'datetime\.now\(\)\.isoformat\(\)',
         'datetime.now(timezone.utc).isoformat()'),
        # timestamp() suggère un timestamp UTC
        (r'datetime\.now\(\)\.timestamp\(\)',
         'datetime.now(timezone.utc).timestamp()'),
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    return content


def fix_arg002(content: str) -> str:
    """
    Corrige ARG002: Préfixe les paramètres inutilisés avec _

    Note: Cette fonction est basique et peut nécessiter une révision manuelle
    """
    # Pattern pour détecter les paramètres de fonction
    # On cherche les cas simples où le paramètre n'est clairement pas utilisé

    # Liste des paramètres communs qui sont souvent inutilisés
    common_unused = ['request', 'params', 'context', 'kwargs']

    for param in common_unused:
        # Remplacer param: Type par _param: Type si pas déjà préfixé
        pattern = rf'(\(|,\s+)({param}):\s*([^,)]+)'

        def replace_if_unused(match):
            prefix = match.group(1)
            param_name = match.group(2)
            param_type = match.group(3)
            # Ne pas remplacer si déjà préfixé par _
            if param_name.startswith('_'):
                return match.group(0)
            return f"{prefix}_{param_name}: {param_type}"

        # Note: Ce remplacement est conservateur et peut manquer certains cas
        # Une analyse plus poussée du code serait nécessaire pour être précis

    return content


def process_file(file_path: Path) -> tuple[bool, int]:
    """
    Traite un fichier Python pour corriger les erreurs Ruff.

    Returns:
        Tuple (modified, num_fixes) où modified indique si le fichier a été modifié
        et num_fixes est le nombre approximatif de corrections effectuées
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        fixes = 0

        # Appliquer les corrections
        new_content = fix_try401(content)
        if new_content != content:
            fixes += content.count('logger.exception(f"') - \
                new_content.count('logger.exception(f"')
            content = new_content

        new_content = fix_dtz003(content)
        if new_content != content:
            fixes += original_content.count('datetime.now(timezone.utc)') - \
                new_content.count('datetime.now(timezone.utc)')
            content = new_content

        new_content = fix_dtz005(content)
        if new_content != content:
            fixes += 1
            content = new_content

        # Sauvegarder si modifié
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            return True, fixes

        return False, 0

    except Exception as e:
        print(f"Erreur lors du traitement de {file_path}: {e}")
        return False, 0


def main():
    """Fonction principale."""
    # Exclure certains répertoires
    excluded_dirs = {
        '.venv',
        'venv',
        '__pycache__',
        '.git',
        'node_modules',
        '.pytest_cache'}

    # Trouver tous les fichiers Python en utilisant os.walk (plus robuste)
    py_files = []
    for root, dirs, files in os.walk('.'):
        # Filtrer les répertoires exclus
        dirs[:] = [d for d in dirs if d not in excluded_dirs]

        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                try:
                    if file_path.exists():
                        py_files.append(file_path)
                except (OSError, PermissionError):
                    continue

    print(f"Traitement de {len(py_files)} fichiers Python...")

    total_modified = 0
    total_fixes = 0

    for py_file in py_files:
        modified, fixes = process_file(py_file)
        if modified:
            total_modified += 1
            total_fixes += fixes
            print(f"[OK] {py_file} ({fixes} corrections)")

    print(f"\n[TERMINE]")
    print(f"   Fichiers modifies: {total_modified}/{len(py_files)}")
    print(f"   Corrections approximatives: {total_fixes}")
    print(f"\n[INFO] Lancez 'ruff check .' pour verifier les erreurs restantes")


if __name__ == "__main__":
    main()
