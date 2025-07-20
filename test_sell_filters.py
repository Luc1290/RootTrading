#!/usr/bin/env python3
"""
Script de test pour vérifier les nouveaux filtres SELL
"""
import logging
import asyncio
from datetime import datetime
import sys
import os

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_sell_filters.log')
    ]
)

logger = logging.getLogger(__name__)

def analyze_logs():
    """Analyse les logs pour voir combien de SELL ont été filtrés"""
    try:
        with open('test_sell_filters.log', 'r') as f:
            lines = f.readlines()
        
        blocked_sells = 0
        allowed_sells = 0
        blocked_reasons = {}
        
        for line in lines:
            if "Signal SELL" in line and "BLOQUÉ" in line:
                blocked_sells += 1
                # Extraire la raison
                if "RSI:" in line:
                    reason = "RSI insuffisant"
                elif "MACD bullish" in line:
                    reason = "MACD bullish + ADX fort"
                elif "TREND_UP détecté" in line:
                    reason = "Tendance haussière forte"
                elif "confluence AVOID" in line:
                    reason = "Confluence AVOID sans sommet"
                elif "contexte défavorable" in line:
                    reason = "Contexte défavorable"
                else:
                    reason = "Autre"
                
                blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1
                
            elif "Signal SELL" in line and "AUTORISÉ" in line:
                allowed_sells += 1
        
        print("\n=== RÉSUMÉ DES FILTRES SELL ===")
        print(f"Total SELL bloqués: {blocked_sells}")
        print(f"Total SELL autorisés: {allowed_sells}")
        print(f"Taux de filtrage: {blocked_sells / (blocked_sells + allowed_sells) * 100:.1f}%" if (blocked_sells + allowed_sells) > 0 else "N/A")
        
        print("\n=== RAISONS DE BLOCAGE ===")
        for reason, count in sorted(blocked_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"{reason}: {count} fois")
        
        print("\n=== MODIFICATIONS APPLIQUÉES ===")
        print("✅ RSI Pro: overbought 72→68, extreme 80→75")
        print("✅ RSI Pro: score contexte minimum 35→55, standard 50→70")
        print("✅ EMA Cross: score contexte minimum 30→50")
        print("✅ Signal Aggregator: filtre SELL en TREND_UP ajouté")
        print("✅ Signal Aggregator: seuils SELL uniques augmentés")
        print("✅ Signal Aggregator: validation confluence plus stricte")
        
    except FileNotFoundError:
        print("Fichier de log non trouvé. Lancez d'abord le système de trading.")
    except Exception as e:
        print(f"Erreur lors de l'analyse: {e}")

if __name__ == "__main__":
    print("=== TEST DES NOUVEAUX FILTRES SELL ===")
    print("Ce script analyse les logs pour vérifier l'efficacité des filtres")
    print("\nAssurez-vous que le système de trading tourne depuis au moins 30 minutes")
    print("pour avoir suffisamment de données.\n")
    
    analyze_logs()
    
    print("\n=== RECOMMANDATIONS ===")
    print("1. Si trop de SELL sont encore générés, augmenter encore les seuils RSI")
    print("2. Si aucun SELL n'est généré, assouplir légèrement les conditions")
    print("3. Surveiller particulièrement les SELL en tendance haussière")
    print("4. Vérifier que les vrais sommets génèrent bien des SELL")