#!/usr/bin/env python3
"""
Script qui suit la progression de l'amélioration du code.
"""
import subprocess
import datetime
import json
import os

def run_command(cmd):
    """Exécute une commande et retourne sa sortie."""
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    return output.decode('utf-8'), error.decode('utf-8')

def get_pylint_score():
    """Récupère le score pylint."""
    output, _ = run_command("pylint --rcfile=pylintrc shared trader analyzer portfolio coordinator dispatcher gateway logger pnl_tracker risk_manager scheduler")
    
    # Chercher le score final
    import re
    match = re.search(r'Your code has been rated at (-?\d+\.\d+)/10', output)
    if match:
        return float(match.group(1))
    return 0.0

def count_mypy_errors():
    """Compte le nombre d'erreurs mypy."""
    output, _ = run_command("mypy --config-file mypy.ini shared trader analyzer portfolio coordinator dispatcher gateway logger pnl_tracker risk_manager scheduler")
    return output.count("error:")

def count_flake8_errors():
    """Compte le nombre d'erreurs flake8."""
    output, _ = run_command("flake8 shared trader analyzer portfolio coordinator dispatcher gateway logger pnl_tracker risk_manager scheduler")
    return len(output.strip().split('\n'))

def save_progress():
    """Enregistre la progression actuelle."""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    pylint_score = get_pylint_score()
    mypy_errors = count_mypy_errors()
    flake8_errors = count_flake8_errors()
    
    # Charger les données existantes
    data = {}
    if os.path.exists("progress.json"):
        with open("progress.json", "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    
    # Ajouter la nouvelle entrée
    data[today] = {
        "pylint_score": pylint_score,
        "mypy_errors": mypy_errors,
        "flake8_errors": flake8_errors
    }
    
    # Sauvegarder
    with open("progress.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Progression du {today} enregistrée:")
    print(f"- Score pylint: {pylint_score}/10")
    print(f"- Erreurs mypy: {mypy_errors}")
    print(f"- Erreurs flake8: {flake8_errors}")

if __name__ == "__main__":
    save_progress()