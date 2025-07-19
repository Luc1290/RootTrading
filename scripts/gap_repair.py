#!/usr/bin/env python3
"""
Script utilitaire pour détecter et réparer les gaps de données après une coupure
Usage: python gap_repair.py [--symbol SYMBOL] [--timeframe TF] [--hours HOURS] [--dry-run]
"""
import asyncio
import argparse
import sys
import os

# Ajouter le répertoire parent au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from gateway.src.gap_detector import GapDetector
from gateway.src.ultra_data_fetcher import UltraDataFetcher
from shared.src.config import SYMBOLS
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gap_repair")

async def main():
    """Fonction principale de réparation des gaps"""
    parser = argparse.ArgumentParser(description='Détecte et répare les gaps de données')
    parser.add_argument('--symbol', type=str, help='Symbole spécifique à vérifier (ex: BTCUSDC)')
    parser.add_argument('--timeframe', type=str, default='1m', 
                       choices=['1m', '5m', '15m', '1h', '4h'],
                       help='Timeframe à vérifier (défaut: 1m)')
    parser.add_argument('--hours', type=int, default=24, 
                       help='Nombre d\'heures à vérifier en arrière (défaut: 24)')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Mode simulation - affiche les gaps sans les réparer')
    parser.add_argument('--force-reload', action='store_true',
                       help='Force le rechargement même sans gaps détectés')
    
    args = parser.parse_args()
    
    # Déterminer les symboles à vérifier
    symbols_to_check = [args.symbol] if args.symbol else SYMBOLS
    
    logger.info("🔍 Démarrage de la détection de gaps...")
    logger.info(f"📊 Symboles: {', '.join(symbols_to_check)}")
    logger.info(f"⏰ Timeframe: {args.timeframe}")
    logger.info(f"📅 Période: {args.hours} heures en arrière")
    logger.info(f"🔧 Mode: {'Simulation' if args.dry_run else 'Réparation'}")
    
    try:
        # Initialiser le détecteur de gaps
        detector = GapDetector()
        await detector.initialize()
        
        # Détecter les gaps
        logger.info("🔍 Détection des gaps en cours...")
        
        all_gaps_found = False
        gap_summary = {}
        
        for symbol in symbols_to_check:
            gaps = await detector.detect_gaps_for_symbol(symbol, args.timeframe, args.hours)
            
            if gaps:
                all_gaps_found = True
                gap_summary[symbol] = gaps
                
                logger.warning(f"⚠️ {symbol} {args.timeframe}: {len(gaps)} gaps détectés")
                for i, (start, end) in enumerate(gaps[:5], 1):  # Afficher max 5 gaps
                    duration = (end - start).total_seconds() / 60
                    logger.warning(f"  Gap {i}: {start} → {end} ({duration:.1f} minutes)")
                    
                if len(gaps) > 5:
                    logger.warning(f"  ... et {len(gaps) - 5} autres gaps")
            else:
                logger.info(f"✅ {symbol} {args.timeframe}: Aucun gap détecté")
        
        # Générer le rapport final
        if all_gaps_found or args.force_reload:
            if not args.dry_run:
                logger.info("🔧 Réparation des gaps en cours...")
                
                # Utiliser UltraDataFetcher pour réparer
                fetcher = UltraDataFetcher()
                
                # Forcer le mode gap detection
                await fetcher.load_historical_data(
                    days=args.hours // 24 + 1,  # Convertir heures en jours
                    use_gap_detection=True
                )
                
                logger.info("✅ Réparation des gaps terminée")
            else:
                logger.info("📋 Mode simulation - Aucune réparation effectuée")
                
                if gap_summary:
                    total_gaps = sum(len(gaps) for gaps in gap_summary.values())
                    logger.info(f"📊 Résumé: {total_gaps} gaps trouvés sur {len(gap_summary)} symboles")
        else:
            logger.info("✅ Aucun gap détecté - Aucune action nécessaire")
            
        await detector.close()
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la réparation: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)