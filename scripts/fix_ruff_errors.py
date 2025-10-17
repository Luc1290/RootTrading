#!/usr/bin/env python3
"""
Script pour corriger automatiquement les erreurs Ruff communes
"""
import re
import sys
from pathlib import Path
from typing import Dict, List, Set


def fix_try300_errors(file_path: Path) -> int:
    """
    Corrige les erreurs TRY300: déplace les return statements dans des else blocks
    """
    content = file_path.read_text(encoding="utf-8")
    lines = content.split("\n")
    modified = False
    fixes = 0

    i = 0
    while i < len(lines):
        line = lines[i]

        # Détecter un try block suivi d'un return avant except
        if "try:" in line:
            indent = len(line) - len(line.lstrip())
            try_indent = indent

            # Chercher le except correspondant
            j = i + 1
            return_line_idx = None
            except_idx = None

            while j < len(lines):
                curr_line = lines[j]
                curr_indent = len(curr_line) - len(curr_line.lstrip())

                # Si on trouve un return au même niveau d'indentation que le try
                if curr_indent == try_indent + 4 and curr_line.strip().startswith("return"):
                    return_line_idx = j

                # Si on trouve except au même niveau que try
                if curr_indent == try_indent and "except" in curr_line:
                    except_idx = j
                    break

                j += 1

            # Si on a un return avant except, le déplacer dans un else
            if return_line_idx and except_idx and return_line_idx < except_idx:
                # Insérer un else avant le return
                return_indent = " " * (try_indent + 4)
                else_line = " " * try_indent + "else:"

                # Vérifier qu'il n'y a pas déjà un else
                if "else:" not in lines[return_line_idx - 1]:
                    lines.insert(return_line_idx, else_line)
                    modified = True
                    fixes += 1

        i += 1

    if modified:
        file_path.write_text("\n".join(lines), encoding="utf-8")

    return fixes


def fix_plc0415_errors(file_path: Path, error_lines: Dict[int, str]) -> int:
    """
    Corrige les erreurs PLC0415: déplace les imports au début du fichier
    """
    content = file_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    # Extraire tous les imports à déplacer
    imports_to_move: Set[str] = set()
    lines_to_remove: List[int] = []

    for line_num, import_statement in error_lines.items():
        if line_num - 1 < len(lines):
            line = lines[line_num - 1].strip()
            if line.startswith(("import ", "from ")):
                imports_to_move.add(line)
                lines_to_remove.append(line_num - 1)

    if not imports_to_move:
        return 0

    # Trouver où insérer les imports (après les imports existants)
    insert_pos = 0
    for i, line in enumerate(lines):
        if line.strip().startswith(("import ", "from ")):
            insert_pos = i + 1
        elif line.strip() and not line.strip().startswith("#"):
            break

    # Supprimer les anciennes lignes d'import (en ordre inverse)
    for idx in sorted(lines_to_remove, reverse=True):
        if idx < len(lines):
            del lines[idx]

    # Insérer les imports au bon endroit
    new_imports = sorted(imports_to_move)
    for imp in reversed(new_imports):
        lines.insert(insert_pos, imp)

    file_path.write_text("\n".join(lines), encoding="utf-8")
    return len(imports_to_move)


def fix_sim102_errors(file_path: Path) -> int:
    """
    Corrige les erreurs SIM102: combine les if imbriqués avec and
    """
    content = file_path.read_text(encoding="utf-8")
    lines = content.split("\n")
    modified = False
    fixes = 0

    i = 0
    while i < len(lines) - 1:
        line = lines[i]
        next_line = lines[i + 1] if i + 1 < len(lines) else ""

        # Détecter if imbriqué simple
        if_match = re.match(r"^(\s*)if (.+):$", line)
        if if_match and next_line.strip().startswith("if "):
            indent = if_match.group(1)
            condition1 = if_match.group(2)

            inner_match = re.match(r"^\s+if (.+):$", next_line)
            if inner_match:
                condition2 = inner_match.group(1)

                # Combiner les conditions
                combined = f"{indent}if {condition1} and {condition2}:"
                lines[i] = combined
                del lines[i + 1]

                modified = True
                fixes += 1
                continue

        i += 1

    if modified:
        file_path.write_text("\n".join(lines), encoding="utf-8")

    return fixes


def parse_ruff_report(report_path: Path) -> Dict[Path, Dict[str, List[Dict]]]:
    """
    Parse le rapport ruff pour extraire les erreurs par fichier

    Returns:
        Dict[file_path, Dict[error_code, List[error_details]]]
    """
    errors_by_file: Dict[Path, Dict[str, List[Dict]]] = {}

    content = report_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    current_error = None

    for line in lines:
        # Détecter une nouvelle erreur: path:line:col: CODE Description
        match = re.match(r"^(.+?):(\d+):(\d+): ([A-Z]+\d+) (.+)$", line)
        if match:
            file_path = Path(match.group(1))
            line_num = int(match.group(2))
            col_num = int(match.group(3))
            error_code = match.group(4)
            description = match.group(5)

            if file_path not in errors_by_file:
                errors_by_file[file_path] = {}

            if error_code not in errors_by_file[file_path]:
                errors_by_file[file_path][error_code] = []

            errors_by_file[file_path][error_code].append({
                "line": line_num,
                "col": col_num,
                "description": description,
            })

    return errors_by_file


def main():
    root = Path("E:/RootTrading/RootTrading")
    report_path = root / "debug" / "ruff-report.txt"

    if not report_path.exists():
        print(f"❌ Rapport non trouvé: {report_path}")
        return 1

    print("📋 Parsing du rapport ruff...")
    errors_by_file = parse_ruff_report(report_path)

    print(f"✅ {len(errors_by_file)} fichiers avec erreurs détectés")

    total_fixes = 0

    # Fix TRY300 errors
    print("\n🔧 Correction des erreurs TRY300...")
    for file_path, errors in errors_by_file.items():
        if "TRY300" in errors:
            full_path = root / file_path
            if full_path.exists():
                fixes = fix_try300_errors(full_path)
                if fixes > 0:
                    print(f"  ✓ {file_path}: {fixes} fix(es)")
                    total_fixes += fixes

    # Fix SIM102 errors
    print("\n🔧 Correction des erreurs SIM102...")
    for file_path, errors in errors_by_file.items():
        if "SIM102" in errors:
            full_path = root / file_path
            if full_path.exists():
                fixes = fix_sim102_errors(full_path)
                if fixes > 0:
                    print(f"  ✓ {file_path}: {fixes} fix(es)")
                    total_fixes += fixes

    print(f"\n✅ Total: {total_fixes} corrections appliquées")
    print("\n💡 Pour les erreurs restantes:")
    print("  - PLR0911 (too many returns): nécessite refactoring manuel")
    print("  - PLC0415 (imports): déplacer manuellement au début du fichier")
    print("  - F821 (undefined names): corriger les typos/références")

    return 0


if __name__ == "__main__":
    sys.exit(main())
