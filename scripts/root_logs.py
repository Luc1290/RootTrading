#!/usr/bin/env python3
"""
ROOT Trading - Analyseur de logs unifié
Usage simple pour toutes les analyses de logs
"""

import subprocess
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
import argparse

class Colors:
    """Couleurs pour l'affichage"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def get_logs(container=None, tail=1000, grep_pattern=None):
    """Récupère les logs avec options"""
    if container:
        cmd = ['docker', 'logs', f'--tail={tail}', container]
    else:
        # Récupérer tous les containers ROOT
        containers = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], 
                                  capture_output=True, text=True).stdout.strip().split('\n')
        root_containers = [c for c in containers if 'roottrading' in c]
        
        all_logs = []
        for cont in root_containers:
            result = subprocess.run(['docker', 'logs', f'--tail={tail}', cont], 
                                  capture_output=True, text=True, encoding='utf-8', errors='replace')
            logs = result.stderr if result.stderr else result.stdout
            
            if grep_pattern:
                # Filtrer chaque ligne
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

def analyze_crypto(crypto, tail=2000):
    """Analyse complète d'une crypto"""
    print(f"\n{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}📊 ANALYSE COMPLÈTE: {crypto}USDC{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*60}{Colors.RESET}\n")
    
    # Récupérer les logs avec contexte
    logs = get_logs(tail=tail)
    lines = logs.split('\n')
    
    # Trouver toutes les lignes pertinentes
    crypto_contexts = []
    for i, line in enumerate(lines):
        if f'{crypto}USDC' in line:
            # Capturer le contexte (lignes avant et après)
            start = max(0, i - 3)
            end = min(len(lines), i + 15)
            crypto_contexts.extend(lines[start:end])
    
    # Analyser
    stats = defaultdict(int)
    strategies = Counter()
    confidence_values = []
    recent_signals = []
    timeframes = Counter()
    
    for line in crypto_contexts:
        # Signaux
        if re.search(r'Signal (BUY|SELL) généré', line, re.IGNORECASE):
            if 'BUY' in line.upper():
                stats['buy'] += 1
                signal_type = 'BUY'
            else:
                stats['sell'] += 1
                signal_type = 'SELL'
            
            # Stratégie
            strat_match = re.search(r'(\w+_Strategy)', line)
            if strat_match:
                strategies[strat_match.group(1)] += 1
            
            # Confidence
            conf_match = re.search(r'confidence[=:]\s*([\d.]+)', line, re.IGNORECASE)
            if conf_match:
                confidence_values.append(float(conf_match.group(1)))
            
            # Garder les signaux récents
            if len(recent_signals) < 10:
                recent_signals.append({
                    'type': signal_type,
                    'line': line.strip(),
                    'confidence': float(conf_match.group(1)) if conf_match else None,
                    'strategy': strat_match.group(1) if strat_match else 'Unknown'
                })
        
        # Timeframes
        tf_match = re.search(f'{crypto}USDC\s+(\d+m)', line)
        if tf_match:
            timeframes[tf_match.group(1)] += 1
        
        # Erreurs
        if 'error' in line.lower() or 'exception' in line.lower():
            stats['errors'] += 1
    
    # Affichage des résultats
    print(f"{Colors.GREEN}✅ Signaux BUY:{Colors.RESET} {stats['buy']}")
    print(f"{Colors.RED}❌ Signaux SELL:{Colors.RESET} {stats['sell']}")
    ratio = stats['buy'] / stats['sell'] if stats['sell'] > 0 else 0
    print(f"{Colors.BLUE}📊 Ratio BUY/SELL:{Colors.RESET} {ratio:.2f}")
    print(f"{Colors.YELLOW}⚠️  Erreurs:{Colors.RESET} {stats['errors']}")
    
    if confidence_values:
        avg_conf = sum(confidence_values) / len(confidence_values)
        print(f"\n{Colors.PURPLE}📈 CONFIDENCE:{Colors.RESET}")
        print(f"   Moyenne: {avg_conf:.2%}")
        print(f"   Max: {max(confidence_values):.2%}")
        print(f"   Min: {min(confidence_values):.2%}")
    
    if timeframes:
        print(f"\n{Colors.CYAN}⏱️  TIMEFRAMES:{Colors.RESET}")
        for tf, count in timeframes.most_common():
            print(f"   {tf}: {count} analyses")
    
    if strategies:
        print(f"\n{Colors.YELLOW}🎯 TOP STRATÉGIES:{Colors.RESET}")
        for i, (strategy, count) in enumerate(strategies.most_common(5), 1):
            print(f"   {i}. {strategy}: {count} signaux")
    
    if recent_signals:
        print(f"\n{Colors.BOLD}📝 DERNIERS SIGNAUX:{Colors.RESET}")
        for signal in recent_signals[-3:]:
            color = Colors.GREEN if signal['type'] == 'BUY' else Colors.RED
            print(f"\n   {color}[{signal['type']}]{Colors.RESET} {signal['strategy']}")
            if signal['confidence']:
                print(f"   Confidence: {signal['confidence']:.2%}")

def show_errors(service=None, tail=500):
    """Affiche les erreurs récentes"""
    print(f"\n{Colors.RED}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.RED}⚠️  ERREURS RÉCENTES{Colors.RESET}")
    print(f"{Colors.RED}{'='*60}{Colors.RESET}\n")
    
    if service:
        logs = get_logs(container=f"roottrading-{service}-1", tail=tail, grep_pattern="error|exception|failed")
    else:
        logs = get_logs(tail=tail, grep_pattern="error|exception|failed")
    
    errors = logs.split('\n')
    
    # Afficher le nombre total d'erreurs trouvées
    if errors:
        print(f"Total: {len(errors)} erreurs/warnings trouvées\n")
    
    # Limiter l'affichage mais montrer plus si demandé
    max_display = min(len(errors), tail // 10) if tail > 200 else 20
    
    for error in errors[:max_display]:
        if error.strip():
            # Colorer selon le type d'erreur
            if 'CRITICAL' in error or 'FATAL' in error:
                print(f"{Colors.RED}{Colors.BOLD}{error[:150]}...{Colors.RESET}")
            elif 'ERROR' in error:
                print(f"{Colors.RED}{error[:150]}...{Colors.RESET}")
            elif 'WARNING' in error:
                print(f"{Colors.YELLOW}{error[:150]}...{Colors.RESET}")
            else:
                print(f"{error[:150]}...")

def search_pattern(pattern, tail=1000):
    """Recherche un pattern dans tous les logs"""
    print(f"\n{Colors.BLUE}🔍 Recherche: '{pattern}'{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}\n")
    
    logs = get_logs(tail=tail, grep_pattern=pattern)
    all_matches = logs.split('\n')
    # Inverser pour avoir les plus récents en premier
    all_matches.reverse()
    matches = all_matches[:200]  # Limiter l'affichage à 200 lignes
    
    if matches:
        total_matches = len(logs.split('\n'))
        print(f"Trouvé {total_matches} correspondances (affichage limité à 200)\n")
        for match in matches:
            if match.strip():
                # Highlighter le pattern
                highlighted = re.sub(f"({pattern})", f"{Colors.YELLOW}\\1{Colors.RESET}", match, flags=re.IGNORECASE)
                # Afficher plus de caractères et sans couper au milieu du pattern
                max_len = 300
                if len(highlighted) > max_len:
                    # Trouver la position du pattern pour centrer l'affichage
                    pattern_pos = match.lower().find(pattern.lower())
                    if pattern_pos > max_len // 2:
                        start = pattern_pos - max_len // 2
                        end = start + max_len
                        print(f"...{highlighted[start:end]}...")
                    else:
                        print(highlighted[:max_len] + "...")
                else:
                    print(highlighted)
    else:
        print("Aucune correspondance trouvée")

def follow_logs(service=None, pattern=None):
    """Suit les logs en temps réel"""
    print(f"\n{Colors.GREEN}📡 LOGS EN TEMPS RÉEL{Colors.RESET}")
    if pattern:
        print(f"Filtre: {Colors.YELLOW}{pattern}{Colors.RESET}")
    print(f"{Colors.GREEN}{'='*60}{Colors.RESET}")
    print(f"{Colors.YELLOW}(Ctrl+C pour arrêter){Colors.RESET}\n")
    
    try:
        if service:
            cmd = ['docker', 'logs', '-f', f'roottrading-{service}-1']
        else:
            # Suivre plusieurs containers
            containers = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], 
                                      capture_output=True, text=True).stdout.strip().split('\n')
            root_containers = [c for c in containers if 'roottrading' in c][:5]  # Limiter à 5
            cmd = ['docker', 'logs', '-f'] + root_containers
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding='utf-8', errors='replace')
        
        for line in process.stdout:
            if pattern:
                if re.search(pattern, line, re.IGNORECASE):
                    # Colorer selon le type
                    if 'ERROR' in line:
                        print(f"{Colors.RED}{line.strip()}{Colors.RESET}")
                    elif 'WARNING' in line:
                        print(f"{Colors.YELLOW}{line.strip()}{Colors.RESET}")
                    elif 'BUY' in line:
                        print(f"{Colors.GREEN}{line.strip()}{Colors.RESET}")
                    elif 'SELL' in line:
                        print(f"{Colors.RED}{line.strip()}{Colors.RESET}")
                    else:
                        print(line.strip())
            else:
                print(line.strip())
    
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}✋ Arrêt du suivi{Colors.RESET}")

def main():
    parser = argparse.ArgumentParser(
        description='ROOT Trading - Analyseur de logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  %(prog)s crypto SOL          # Analyser SOLUSDC
  %(prog)s errors              # Voir toutes les erreurs
  %(prog)s search "pattern"    # Rechercher un pattern
  %(prog)s follow              # Suivre les logs en temps réel
  %(prog)s follow -p BTCUSDC   # Suivre avec filtre
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # Crypto
    crypto_parser = subparsers.add_parser('crypto', help='Analyser une crypto')
    crypto_parser.add_argument('symbol', help='Symbole (ex: BTC, ETH, SOL)')
    crypto_parser.add_argument('-t', '--tail', type=int, default=2000, help='Nombre de lignes')
    
    # Errors
    errors_parser = subparsers.add_parser('errors', help='Afficher les erreurs')
    errors_parser.add_argument('-s', '--service', help='Service spécifique')
    errors_parser.add_argument('-t', '--tail', type=int, default=500, help='Nombre de lignes')
    
    # Search
    search_parser = subparsers.add_parser('search', help='Rechercher un pattern')
    search_parser.add_argument('pattern', help='Pattern à rechercher')
    search_parser.add_argument('-t', '--tail', type=int, default=1000, help='Nombre de lignes')
    
    # Follow
    follow_parser = subparsers.add_parser('follow', help='Suivre en temps réel')
    follow_parser.add_argument('-s', '--service', help='Service spécifique')
    follow_parser.add_argument('-p', '--pattern', help='Pattern à filtrer')
    
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
    elif args.command == 'follow':
        follow_logs(args.service, args.pattern)

if __name__ == '__main__':
    main()