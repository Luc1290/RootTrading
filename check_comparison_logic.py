#!/usr/bin/env python3
"""
Script pour détecter les comparaisons illogiques entre indicateurs 0-100 et seuils 0-1.
"""

import os
import re
from typing import Dict, List, Tuple, Set

# Indicateurs qui sont en format 0-100
INDICATORS_0_100 = {
    'rsi', 'rsi_14', 'rsi_21', 
    'stoch_k', 'stoch_d', 'stoch_rsi',
    'williams_r',  # Note: Williams %R est en fait -100 à 0
    'cci', 'cci_20',
    'momentum_score',
    'bb_position',  # Position dans Bollinger (0-1 mais souvent * 100)
    'volume_ratio',  # Peut être > 1
    'atr_percentile',
    'volatility_percentile',
    'pattern_confidence',
    'confluence_score',
    'trend_alignment',
    'break_probability',
    'reversal_probability', 
    'continuation_probability'
}

# Indicateurs qui sont en format 0-1
INDICATORS_0_1 = {
    'trend_strength',
    'support_strength', 
    'resistance_strength',
    'regime_strength',
    'pattern_strength',
    'signal_strength'
}

# Indicateurs spéciaux avec ranges particuliers
SPECIAL_RANGES = {
    'williams_r': (-100, 0),
    'macd_line': 'variable',
    'macd_histogram': 'variable', 
    'adx': (0, 100),
    'di_plus': (0, 100),
    'di_minus': (0, 100),
    'roc': 'percentage',  # Peut être négatif
    'momentum_score': (0, 100)
}

def analyze_file_comparisons(file_path: str) -> List[Dict]:
    """Analyse les comparaisons dans un fichier."""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception:
        return issues
    
    for line_num, line in enumerate(lines, 1):
        # Ignorer commentaires
        if line.strip().startswith('#'):
            continue
            
        # Patterns de comparaison à détecter
        # Exemple: indicator > 0.3, indicator >= 0.5, indicator < 0.8
        comparison_patterns = [
            # Pattern: variable > 0.something
            r'(\w+)\s*([><=!]+)\s*(0\.\d+)',
            # Pattern: 0.something < variable  
            r'(0\.\d+)\s*([><=!]+)\s*(\w+)',
            # Pattern: variable > number (detect small numbers for 0-100 indicators)
            r'(\w+)\s*([><=!]+)\s*([0-9]+(?:\.[0-9]+)?)'
        ]
        
        for pattern in comparison_patterns:
            matches = re.findall(pattern, line)
            for match in matches:
                if len(match) == 3:
                    var1, operator, var2 = match
                    
                    # Cas 1: variable op 0.something
                    if var2.startswith('0.') and '.' in var2:
                        threshold = float(var2)
                        if threshold <= 1.0:
                            # Vérifier si variable est un indicateur 0-100
                            if any(indicator in var1.lower() for indicator in INDICATORS_0_100):
                                issues.append({
                                    'file': os.path.basename(file_path),
                                    'line': line_num,
                                    'issue_type': 'comparison_0_100_vs_0_1',
                                    'code': line.strip(),
                                    'indicator': var1,
                                    'threshold': threshold,
                                    'suggestion': f"Utiliser {threshold * 100} au lieu de {threshold}"
                                })
                    
                    # Cas 2: 0.something op variable
                    elif var1.startswith('0.') and '.' in var1:
                        threshold = float(var1)
                        if threshold <= 1.0:
                            if any(indicator in var2.lower() for indicator in INDICATORS_0_100):
                                issues.append({
                                    'file': os.path.basename(file_path),
                                    'line': line_num,
                                    'issue_type': 'comparison_0_100_vs_0_1',
                                    'code': line.strip(),
                                    'indicator': var2,
                                    'threshold': threshold,
                                    'suggestion': f"Utiliser {threshold * 100} au lieu de {threshold}"
                                })
                    
                    # Cas 3: variable op number (detect suspicious small numbers)
                    else:
                        try:
                            threshold = float(var2)
                            # Indicateur 0-100 comparé avec seuil < 10 (suspect)
                            if (threshold < 10 and threshold > 0.01 and 
                                any(indicator in var1.lower() for indicator in INDICATORS_0_100)):
                                
                                # Exceptions légitimes
                                if not (var1.lower() in ['williams_r'] and threshold < 0):  # Williams %R est négatif
                                    issues.append({
                                        'file': os.path.basename(file_path),
                                        'line': line_num,
                                        'issue_type': 'suspicious_small_threshold',
                                        'code': line.strip(),
                                        'indicator': var1,
                                        'threshold': threshold,
                                        'suggestion': f"Vérifier si {threshold} est correct pour indicateur 0-100"
                                    })
                        except ValueError:
                            continue
    
    return issues

def analyze_specific_indicators(file_path: str) -> List[Dict]:
    """Analyse spécifique pour certains indicateurs problématiques."""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
    except Exception:
        return issues
    
    # Patterns spécifiques
    specific_checks = [
        # RSI comparisons (should be 0-100)
        (r'rsi.*[><=]\s*0\.[0-9]', 'RSI comparé avec 0.x (devrait être 0-100)'),
        # Stochastic comparisons
        (r'stoch.*[><=]\s*0\.[0-9]', 'Stochastic comparé avec 0.x (devrait être 0-100)'),
        # Momentum score
        (r'momentum_score.*[><=]\s*0\.[0-9]', 'momentum_score comparé avec 0.x (devrait être 0-100)'),
        # BB position
        (r'bb_position.*[><=]\s*[2-9][0-9]', 'bb_position comparé avec valeur > 20 (devrait être 0-1 ou 0-100)'),
        # Pattern confidence
        (r'pattern_confidence.*[><=]\s*0\.[0-9]', 'pattern_confidence comparé avec 0.x'),
        # Williams %R incorrect range
        (r'williams_r.*[><=]\s*[1-9][0-9]', 'williams_r comparé avec valeur positive (devrait être -100 à 0)'),
    ]
    
    for line_num, line in enumerate(lines, 1):
        if line.strip().startswith('#'):
            continue
            
        for pattern, description in specific_checks:
            if re.search(pattern, line, re.IGNORECASE):
                issues.append({
                    'file': os.path.basename(file_path),
                    'line': line_num,
                    'issue_type': 'specific_indicator_issue',
                    'code': line.strip(),
                    'description': description
                })
    
    return issues

def analyze_directory(directory: str) -> Dict[str, List[Dict]]:
    """Analyse tous les fichiers Python dans un répertoire."""
    all_issues = {}
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                file_path = os.path.join(root, file)
                
                # Analyse générale des comparaisons
                issues = analyze_file_comparisons(file_path)
                # Analyse spécifique
                issues.extend(analyze_specific_indicators(file_path))
                
                if issues:
                    all_issues[file_path] = issues
    
    return all_issues

def print_results(all_issues: Dict[str, List[Dict]]):
    """Affiche les résultats de l'analyse."""
    print("🔍 ANALYSE DES COMPARAISONS ILLOGIQUES")
    print("=" * 60)
    
    if not all_issues:
        print("🎉 Aucune comparaison illogique détectée !")
        return
    
    total_issues = sum(len(issues) for issues in all_issues.values())
    
    for file_path, issues in all_issues.items():
        file_name = os.path.basename(file_path)
        directory = os.path.basename(os.path.dirname(file_path))
        
        print(f"\n❌ {directory}/{file_name} ({len(issues)} problèmes)")
        
        for issue in issues:
            print(f"   📍 Ligne {issue['line']}: {issue['issue_type']}")
            print(f"      Code: {issue['code']}")
            
            if 'indicator' in issue and 'threshold' in issue:
                print(f"      Indicateur: {issue['indicator']} | Seuil: {issue['threshold']}")
            
            if 'suggestion' in issue:
                print(f"      💡 {issue['suggestion']}")
            elif 'description' in issue:
                print(f"      💡 {issue['description']}")
            print()
    
    print(f"\n{'='*60}")
    print(f"📊 RÉSUMÉ: {total_issues} comparaisons illogiques trouvées dans {len(all_issues)} fichiers")

def main():
    directories = [
        '/mnt/e/RootTrading/RootTrading/analyzer/strategies',
        '/mnt/e/RootTrading/RootTrading/signal_aggregator/validators', 
        '/mnt/e/RootTrading/RootTrading/signal_aggregator/src'
    ]
    
    all_issues = {}
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"🔍 Analyse de {directory}...")
            issues = analyze_directory(directory)
            all_issues.update(issues)
        else:
            print(f"⚠️  Répertoire non trouvé: {directory}")
    
    print_results(all_issues)

if __name__ == "__main__":
    main()