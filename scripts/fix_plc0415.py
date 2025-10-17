#!/usr/bin/env python3
"""
Script pour corriger les erreurs PLC0415 (imports outside top-level)
"""
import re
import sys
from pathlib import Path


def extract_imports_from_try_blocks(content: str) -> tuple[set[str], list[tuple[int, str]]]:
    """
    Extrait les imports depuis les try blocks et retourne:
    - Set d'imports uniques à déplacer
    - Liste de (line_num, import_statement) à supprimer
    """
    lines = content.split("\n")
    imports_to_move = set()
    lines_to_remove = []

    in_try_block = False
    try_indent = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Détecter début de try block
        if stripped.startswith("try:"):
            in_try_block = True
            try_indent = len(line) - len(line.lstrip())
            continue

        # Détecter fin de try block
        if in_try_block and line and len(line) - len(line.lstrip()) <= try_indent:
            if stripped.startswith(("except", "finally", "else:")):
                in_try_block = False
            elif not line.strip().startswith("#") and line.strip():
                # Ligne non indentée, sortie du try
                in_try_block = False

        # Si dans un try block, chercher les imports
        if in_try_block and (
            stripped.startswith("import ") or stripped.startswith("from ")
        ):
            imports_to_move.add(stripped)
            lines_to_remove.append((i, stripped))

    return imports_to_move, lines_to_remove


def get_import_insert_position(lines: list[str]) -> int:
    """
    Trouve la position où insérer les imports (après les imports existants, avant le code)
    """
    last_import_pos = 0
    found_import = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip docstrings et commentaires
        if stripped.startswith('"""') or stripped.startswith("'''") or stripped.startswith("#"):
            continue

        # Trouver la dernière ligne d'import au top-level
        if stripped.startswith(("import ", "from ")):
            last_import_pos = i + 1
            found_import = True
        elif found_import and stripped and not stripped.startswith("#"):
            # Premier code après les imports
            break

    return last_import_pos


def fix_plc0415_in_file(file_path: Path) -> int:
    """
    Corrige les erreurs PLC0415 dans un fichier
    Returns: nombre de corrections
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        # Extraire les imports à déplacer
        imports_to_move, lines_to_remove = extract_imports_from_try_blocks(content)

        if not imports_to_move:
            return 0

        # Supprimer les imports des try blocks (en ordre inverse)
        for line_idx, _ in sorted(lines_to_remove, key=lambda x: x[0], reverse=True):
            del lines[line_idx]

        # Trouver où insérer
        insert_pos = get_import_insert_position(lines)

        # Trier les imports (from ... avant import ...)
        sorted_imports = sorted(
            imports_to_move, key=lambda x: (not x.startswith("from"), x)
        )

        # Insérer les imports
        for imp in reversed(sorted_imports):
            lines.insert(insert_pos, imp)

        # Sauvegarder
        file_path.write_text("\n".join(lines), encoding="utf-8")

        return len(imports_to_move)

    except Exception as e:
        print(f"[ERROR] Erreur dans {file_path}: {e}")
        return 0


def main():
    root = Path("E:/RootTrading/RootTrading")

    # Trouver tous les fichiers Python en évitant les répertoires problématiques
    py_files = []
    excluded_dirs = {".venv", "__pycache__", "venv", "node_modules", ".git"}

    # Scanner manuellement les répertoires pour éviter les erreurs d'accès
    for subdir in ["analyzer", "coordinator", "dispatcher", "portfolio", "shared", "trader", "visualization"]:
        subdir_path = root / subdir
        if subdir_path.exists():
            try:
                for item in subdir_path.rglob("*.py"):
                    if any(excluded in item.parts for excluded in excluded_dirs):
                        continue
                    py_files.append(item)
            except OSError as e:
                print(f"[WARN] Erreur acces {subdir}: {e}")
                continue

    print(f"[*] Analyse de {len(py_files)} fichiers Python...")

    total_fixes = 0

    for py_file in py_files:

        fixes = fix_plc0415_in_file(py_file)
        if fixes > 0:
            print(f"  [OK] {py_file.relative_to(root)}: {fixes} import(s) deplaces")
            total_fixes += fixes

    print(f"\n[SUCCESS] Total: {total_fixes} imports deplaces")

    return 0


if __name__ == "__main__":
    sys.exit(main())
