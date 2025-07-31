#!/usr/bin/env python3
"""
Script pour corriger les erreurs de conversion string->float dans les validators.

Problème identifié:
- volume_pattern: 'DECLINING', 'NORMAL', 'INCREASING' → valeurs catégorielles
- directional_bias: 'BULLISH', 'BEARISH', 'NEUTRAL' → valeurs catégorielles  
- volatility_regime: 'low', 'normal', 'high' → valeurs catégorielles

Solution: Ne pas convertir ces indicateurs en float, les utiliser comme strings.
"""

import os
import re
from typing import List, Dict, Tuple

def create_backup(file_path: str) -> str:
    """Crée une sauvegarde du fichier."""
    backup_path = f"{file_path}.backup_string_fix"
    try:
        with open(file_path, 'r', encoding='utf-8') as source:
            content = source.read()
        with open(backup_path, 'w', encoding='utf-8') as backup:
            backup.write(content)
        print(f"Sauvegarde créée: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"Erreur création sauvegarde {file_path}: {e}")
        return ""

def fix_string_float_conversions(file_path: str, corrections: List[Tuple[str, str]]) -> Tuple[bool, List[str]]:
    """Corrige les conversions float() inappropriées pour les indicateurs catégoriels."""
    if not os.path.exists(file_path):
        return False, [f"Fichier non trouvé: {file_path}"]
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return False, [f"Erreur lecture {file_path}: {e}"]
    
    original_content = content
    applied_corrections = []
    
    for pattern, replacement in corrections:
        new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        if new_content != content:
            applied_corrections.append(f"{pattern[:50]}... -> {replacement[:50]}...")
            content = new_content
    
    if content != original_content:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, applied_corrections
        except Exception as e:
            return False, [f"Erreur écriture {file_path}: {e}"]
    
    return False, []

def main():
    """Fonction principale."""
    print("=== Correction des conversions string->float inappropriées ===\n")
    
    # Corrections par fichier
    file_corrections = {
        "/mnt/e/RootTrading/RootTrading/signal_aggregator/validators/Volume_Buildup_Validator.py": [
            # volume_pattern est catégoriel: 'DECLINING', 'NORMAL', 'INCREASING'
            (r'volume_trend = float\(volume_trend_raw\) if volume_trend_raw is not None else None', 
             'volume_trend = str(volume_trend_raw) if volume_trend_raw is not None else None'),
            
            # Remplacer les comparaisons numériques par des comparaisons de catégories
            (r'if volume_trend is not None and volume_trend >= self\.min_volume_trend:', 
             'if volume_trend is not None and volume_trend in [\"INCREASING\", \"RISING\", \"STRONG\"]:'),
             
            (r'if volume_trend is not None and volume_trend <= -self\.min_volume_trend:', 
             'if volume_trend is not None and volume_trend in [\"DECLINING\", \"FALLING\", \"WEAK\"]:'),
        ],
        
        "/mnt/e/RootTrading/RootTrading/signal_aggregator/validators/Trend_Alignment_Validator.py": [
            # directional_bias est catégoriel: 'BULLISH', 'BEARISH', 'NEUTRAL'
            (r'primary_trend_direction = self\.context\.get\(\'directional_bias\'\)  # \'bullish\', \'bearish\', \'neutral\'', 
             'primary_trend_direction = str(self.context.get(\'directional_bias\', \'neutral\')) if self.context.get(\'directional_bias\') is not None else \'neutral\''),
        ],
        
        "/mnt/e/RootTrading/RootTrading/signal_aggregator/validators/Range_Validator.py": [
            # volatility_regime est catégoriel: 'low', 'normal', 'high'  
            (r'volatility_regime_raw = self\.context\.get\(\'volatility_regime\'\)',
             'volatility_regime_raw = str(self.context.get(\'volatility_regime\', \'normal\')) if self.context.get(\'volatility_regime\') is not None else \'normal\''),
             
            # Éviter float() sur volatility_regime
            (r'volatility_regime = float\(volatility_regime_raw\) if volatility_regime_raw is not None else None',
             'volatility_regime = str(volatility_regime_raw) if volatility_regime_raw is not None else \'normal\''),
        ],
    }
    
    total_files_fixed = 0
    total_corrections = 0
    
    for file_path, corrections in file_corrections.items():
        print(f"🔧 Traitement: {os.path.basename(file_path)}")
        
        # Créer sauvegarde
        backup_path = create_backup(file_path)
        if not backup_path:
            print(f"  ❌ Échec sauvegarde, fichier sauté\n")
            continue
        
        # Appliquer corrections
        success, applied = fix_string_float_conversions(file_path, corrections)
        
        if success and applied:
            total_files_fixed += 1
            total_corrections += len(applied)
            print(f"  ✅ {len(applied)} corrections appliquées:")
            for correction in applied:
                print(f"    🔧 {correction}")
        elif success:
            print(f"  ℹ️  Aucune correction nécessaire")
        else:
            print(f"  ❌ Erreur: {applied}")
        
        print()
    
    print("=== Résumé Corrections String->Float ===")
    print(f"📊 Fichiers traités: {len(file_corrections)}")
    print(f"✅ Fichiers corrigés: {total_files_fixed}")
    print(f"🔧 Total corrections: {total_corrections}")
    
    if total_files_fixed > 0:
        print(f"\n🎉 Corrections string->float terminées!")
        print(f"📝 Indicateurs catégoriels maintenant gérés correctement:")
        print(f"   - volume_pattern: 'DECLINING', 'NORMAL', 'INCREASING'")
        print(f"   - directional_bias: 'BULLISH', 'BEARISH', 'NEUTRAL'")
        print(f"   - volatility_regime: 'low', 'normal', 'high'")
        print(f"💾 Sauvegardes: .backup_string_fix")
        print(f"\n🚀 Redémarrez le signal-aggregator pour appliquer les corrections")
    else:
        print(f"\n⚠️  Aucune correction string->float appliquée")

if __name__ == "__main__":
    main()