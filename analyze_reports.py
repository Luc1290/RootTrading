#!/usr/bin/env python3
"""
Script qui analyse les résultats des outils d'analyse statique
et génère un rapport de synthèse.
"""
import re
import os

def count_issues(filename):
    """Compte le nombre de problèmes dans un fichier de rapport."""
    if not os.path.exists(filename):
        return 0
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Pour flake8, chaque ligne est une erreur
    if 'flake8' in filename:
        return len(content.strip().split('\n'))
    
    # Pour pylint, chercher le score
    if 'pylint' in filename:
        score_match = re.search(r'Your code has been rated at (-?\d+\.\d+)/10', content)
        if score_match:
            score = float(score_match.group(1))
            return 10 - score  # Convertir le score en nombre d'erreurs approximatif
        else:
            # Si pas de score, compter les lignes contenant des erreurs/warnings
            error_lines = [l for l in content.split('\n') if re.search(r'[CWEF]\d+:', l)]
            return len(error_lines)
    
    # Pour mypy, chaque ligne d'erreur contient "error:"
    if 'mypy' in filename:
        error_count = content.count('error:')
        return error_count
    
    return 0

def analyze_by_module(filename, module_pattern=r'([a-zA-Z_]+)/[^:]+:'):
    """Analyse les problèmes par module."""
    if not os.path.exists(filename):
        return {}
    
    with open(filename, 'r') as f:
        content = f.read()
    
    modules = {}
    for line in content.split('\n'):
        match = re.search(module_pattern, line)
        if match:
            module = match.group(1)
            if module not in modules:
                modules[module] = 0
            modules[module] += 1
    
    return modules

# Analyse principale
flake8_issues = count_issues('flake8_report.txt')
pylint_issues = count_issues('pylint_report.txt')
mypy_issues = count_issues('mypy_report.txt')

# Analyse par module
flake8_modules = analyze_by_module('flake8_report.txt')
pylint_modules = analyze_by_module('pylint_report.txt')
mypy_modules = analyze_by_module('mypy_report.txt')

# Générer un rapport
with open('analysis_summary.txt', 'w') as f:
    f.write("=== RAPPORT D'ANALYSE STATIQUE ===\n\n")
    f.write(f"Total des problèmes flake8: {flake8_issues}\n")
    f.write(f"Approximation des problèmes pylint: {pylint_issues}\n")
    f.write(f"Total des erreurs mypy: {mypy_issues}\n\n")
    
    f.write("=== PROBLÈMES PAR MODULE ===\n\n")
    
    # Fusionner tous les modules
    all_modules = set(flake8_modules.keys()) | set(pylint_modules.keys()) | set(mypy_modules.keys())
    
    f.write(f"{'Module':<15} {'Flake8':<10} {'Pylint':<10} {'MyPy':<10} {'Total':<10}\n")
    f.write("-" * 55 + "\n")
    
    for module in sorted(all_modules):
        flake8_count = flake8_modules.get(module, 0)
        pylint_count = pylint_modules.get(module, 0)
        mypy_count = mypy_modules.get(module, 0)
        total = flake8_count + pylint_count + mypy_count
        
        f.write(f"{module:<15} {flake8_count:<10} {pylint_count:<10} {mypy_count:<10} {total:<10}\n")

print("Rapport d'analyse généré dans 'analysis_summary.txt'")