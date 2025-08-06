#!/usr/bin/env python3
"""
Version simplifiée de root_logs pour export (sans emojis ni couleurs)
"""

import subprocess
import re
import sys
import argparse
from collections import Counter, defaultdict

def get_logs(container=None, tail=1000, grep_pattern=None):
    """Récupère les logs avec options"""
    if container:
        cmd = ['docker', 'logs', f'--tail={tail}', container]
    else:
        containers = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], 
                                  capture_output=True, text=True).stdout.strip().split('\n')
        root_containers = [c for c in containers if 'roottrading' in c]
        
        all_logs = []
        for cont in root_containers:
            result = subprocess.run(['docker', 'logs', f'--tail={tail}', cont], 
                                  capture_output=True, text=True, encoding='utf-8', errors='replace')
            logs = result.stderr if result.stderr else result.stdout
            
            if grep_pattern:
                for line in logs.split('\n'):
                    if re.search(grep_pattern, line, re.IGNORECASE):
                        all_logs.append(line)
            else:
                all_logs.extend(logs.split('\n'))
        
        return '\n'.join(all_logs)
    
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    logs = result.stderr if result.stderr else result.stdout
    
    if grep_pattern:
        filtered = []
        for line in logs.split('\n'):
            if re.search(grep_pattern, line, re.IGNORECASE):
                filtered.append(line)
        return '\n'.join(filtered)
    
    return logs

def show_errors(service=None, tail=500):
    """Affiche les erreurs récentes"""
    print("="*60)
    print("ERREURS RECENTES")
    print("="*60)
    print()
    
    if service:
        logs = get_logs(container=f"roottrading-{service}-1", tail=tail, grep_pattern="error|exception|failed|warning")
    else:
        logs = get_logs(tail=tail, grep_pattern="error|exception|failed|warning")
    
    errors = logs.split('\n')
    
    if errors:
        print(f"Total: {len(errors)} erreurs/warnings trouvees\n")
    
    max_display = min(len(errors), tail // 10) if tail > 200 else 20
    
    for error in errors[:max_display]:
        if error.strip():
            # Nettoyer les caractères Unicode problématiques
            clean_error = error.encode('ascii', 'ignore').decode('ascii')
            print(clean_error[:300])

def analyze_crypto(crypto, tail=2000):
    """Analyse complète d'une crypto"""
    print("="*60)
    print(f"ANALYSE COMPLETE: {crypto}USDC")
    print("="*60)
    print()
    
    logs = get_logs(tail=tail)
    lines = logs.split('\n')
    
    crypto_contexts = []
    for i, line in enumerate(lines):
        if f'{crypto}USDC' in line:
            start = max(0, i - 3)
            end = min(len(lines), i + 15)
            crypto_contexts.extend(lines[start:end])
    
    stats = defaultdict(int)
    strategies = Counter()
    confidence_values = []
    
    for line in crypto_contexts:
        if re.search(r'Signal (BUY|SELL) généré', line, re.IGNORECASE):
            if 'BUY' in line.upper():
                stats['buy'] += 1
            else:
                stats['sell'] += 1
            
            strat_match = re.search(r'(\w+_Strategy)', line)
            if strat_match:
                strategies[strat_match.group(1)] += 1
            
            conf_match = re.search(r'confidence[=:]\s*([\d.]+)', line, re.IGNORECASE)
            if conf_match:
                confidence_values.append(float(conf_match.group(1)))
    
    print(f"Signaux BUY: {stats['buy']}")
    print(f"Signaux SELL: {stats['sell']}")
    ratio = stats['buy'] / stats['sell'] if stats['sell'] > 0 else 0
    print(f"Ratio BUY/SELL: {ratio:.2f}")
    
    if confidence_values:
        avg_conf = sum(confidence_values) / len(confidence_values)
        print(f"\nCONFIDENCE:")
        print(f"   Moyenne: {avg_conf:.2%}")
        print(f"   Max: {max(confidence_values):.2%}")
        print(f"   Min: {min(confidence_values):.2%}")
    
    if strategies:
        print(f"\nTOP STRATEGIES:")
        for i, (strategy, count) in enumerate(strategies.most_common(5), 1):
            print(f"   {i}. {strategy}: {count} signaux")

def search_pattern(pattern, tail=1000):
    """Recherche un pattern dans tous les logs"""
    print(f"Recherche: '{pattern}'")
    print("="*60)
    print()
    
    logs = get_logs(tail=tail, grep_pattern=pattern)
    all_matches = logs.split('\n')
    all_matches.reverse()
    matches = all_matches[:200]
    
    if matches:
        total_matches = len(logs.split('\n'))
        print(f"Trouve {total_matches} correspondances (affichage limite a 200)\n")
        for match in matches:
            if match.strip():
                print(match[:300])
    else:
        print("Aucune correspondance trouvee")

def main():
    parser = argparse.ArgumentParser(description='ROOT Trading - Analyseur de logs (version export)')
    
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    crypto_parser = subparsers.add_parser('crypto', help='Analyser une crypto')
    crypto_parser.add_argument('symbol', help='Symbole (ex: BTC, ETH, SOL)')
    crypto_parser.add_argument('-t', '--tail', type=int, default=2000, help='Nombre de lignes')
    
    errors_parser = subparsers.add_parser('errors', help='Afficher les erreurs')
    errors_parser.add_argument('-s', '--service', help='Service specifique')
    errors_parser.add_argument('-t', '--tail', type=int, default=500, help='Nombre de lignes')
    
    search_parser = subparsers.add_parser('search', help='Rechercher un pattern')
    search_parser.add_argument('pattern', help='Pattern a rechercher')
    search_parser.add_argument('-t', '--tail', type=int, default=1000, help='Nombre de lignes')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'crypto':
        analyze_crypto(args.symbol.upper(), args.tail)
    elif args.command == 'errors':
        show_errors(args.service, args.tail)
    elif args.command == 'search':
        search_pattern(args.pattern, args.tail)

if __name__ == '__main__':
    main()