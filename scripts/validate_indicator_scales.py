#!/usr/bin/env python3
"""
Script de validation des échelles d'indicateurs dans ROOT Trading Bot
Détecte automatiquement les incohérences d'échelle entre validators et field_converters
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class IndicatorUsage:
    """Usage d'un indicateur dans un validator"""
    validator_name: str
    line_number: int
    usage_pattern: str
    threshold_value: float
    suspected_scale: str  # "0-1", "0-100", "-100-100", "raw", "unknown"

@dataclass
class ScaleInconsistency:
    """Incohérence détectée dans les échelles"""
    indicator_name: str
    usages: List[IndicatorUsage]
    suspected_issue: str
    severity: str  # "critical", "warning", "info"

class IndicatorScaleValidator:
    """Validateur des échelles d'indicateurs"""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.validators_path = self.root_path / "signal_aggregator" / "validators"
        self.field_converters_path = self.root_path / "signal_aggregator" / "src" / "field_converters.py"
        
        # Patterns de détection d'échelle
        self.scale_patterns = {
            "0-1": [
                (r'([a-zA-Z_]+)\s*[<>]=?\s*0\.[0-9]+', "threshold"),
                (r'([a-zA-Z_]+)\s*/\s*100', "conversion_to_0_1"),
                (r'([a-zA-Z_]+)\s*\*\s*100', "conversion_from_0_1"),
            ],
            "0-100": [
                (r'([a-zA-Z_]+)\s*[<>]=?\s*[1-9][0-9]\.0+', "threshold"),
                (r'([a-zA-Z_]+)\s*[<>]=?\s*[1-9][0-9]', "threshold"),
            ],
            "-100-100": [
                (r'([a-zA-Z_]+)\s*[<>]=?\s*-[0-9]+\.0+', "negative_threshold"),
                (r'([a-zA-Z_]+)\s*[<>]=?\s*-[0-9]+', "negative_threshold"),
            ]
        }
        
        # Indicateurs connus avec leur échelle attendue
        self.known_scales = {
            # Échelle 0-1
            'trend_strength': '0-1',
            'regime_strength': '0-1', 
            'pivot_support_strength': '0-1',
            'pivot_resistance_strength': '0-1',
            'bb_position': '0-1',
            'trade_intensity': '0-1',
            'distribution_normality': '0-1',
            'statistical_significance': '0-1',
            'buy_sell_pressure': '0-1',
            'money_flow_index': '0-1',
            
            # Échelle 0-100
            'rsi_14': '0-100',
            'rsi_21': '0-100',
            'momentum_score': '0-100',
            'volume_quality_score': '0-100',
            'regime_confidence': '0-100',
            'regime_stability': '0-100',
            'regime_persistence': '0-100',
            'regime_consensus_score': '0-100',
            'accumulation_distribution_score': '0-100',
            'liquidity_score': '0-100',
            'ema_alignment_score': '0-100',
            'confluence_score': '0-100',
            
            # Z-Scores (-3 à +3)
            'price_zscore': 'zscore',
            'volume_zscore': 'zscore',
            'returns_zscore': 'zscore',
            'zscore_20': 'zscore',
            'zscore_50': 'zscore',
            'zscore_rsi': 'zscore',
            
            # Ratios/multiplicateurs
            'relative_volume': 'ratio',
            'volume_ratio': 'ratio',
            'volume_spike_multiplier': 'ratio',
        }
        
        self.inconsistencies: List[ScaleInconsistency] = []
        self.indicator_usages: Dict[str, List[IndicatorUsage]] = defaultdict(list)
    
    def scan_validators(self) -> None:
        """Scanne tous les validators pour détecter les usages d'indicateurs"""
        if not self.validators_path.exists():
            print(f"❌ Chemin validators introuvable: {self.validators_path}")
            return
            
        for validator_file in self.validators_path.glob("*.py"):
            if validator_file.name.startswith("__"):
                continue
                
            self._scan_validator_file(validator_file)
    
    def _scan_validator_file(self, file_path: Path) -> None:
        """Scanne un fichier validator pour les usages d'indicateurs"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            validator_name = file_path.stem
            
            for line_num, line in enumerate(lines, 1):
                # Ignore les commentaires
                if line.strip().startswith('#') or line.strip().startswith('"""') or line.strip().startswith("'''"):
                    continue
                    
                # Cherche les patterns d'usage d'indicateurs
                self._extract_indicator_usages(validator_name, line_num, line)
                
        except Exception as e:
            print(f"⚠️ Erreur lecture {file_path}: {e}")
    
    def _extract_indicator_usages(self, validator_name: str, line_num: int, line: str) -> None:
        """Extrait les usages d'indicateurs d'une ligne de code"""
        # Pattern pour détecter les comparaisons avec seuils
        comparison_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*([<>]=?)\s*([+-]?\d+(?:\.\d+)?)'
        matches = re.findall(comparison_pattern, line)
        
        for var_name, operator, threshold_str in matches:
            try:
                threshold = float(threshold_str)
                
                # Déterminer l'échelle suspectée
                suspected_scale = self._guess_scale_from_threshold(threshold)
                
                usage = IndicatorUsage(
                    validator_name=validator_name,
                    line_number=line_num,
                    usage_pattern=f"{var_name} {operator} {threshold}",
                    threshold_value=threshold,
                    suspected_scale=suspected_scale
                )
                
                self.indicator_usages[var_name].append(usage)
                
            except ValueError:
                continue
        
        # Pattern pour détecter les .get() avec valeurs par défaut
        get_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=.*\.get\(["\']([^"\']+)["\'],\s*([+-]?\d+(?:\.\d+)?)\)'
        get_matches = re.findall(get_pattern, line)
        
        for var_name, indicator_name, default_str in get_matches:
            try:
                default_val = float(default_str)
                suspected_scale = self._guess_scale_from_threshold(default_val)
                
                usage = IndicatorUsage(
                    validator_name=validator_name,
                    line_number=line_num,
                    usage_pattern=f"default value: {default_val}",
                    threshold_value=default_val,
                    suspected_scale=suspected_scale
                )
                
                self.indicator_usages[indicator_name].append(usage)
                
            except ValueError:
                continue
    
    def _guess_scale_from_threshold(self, threshold: float) -> str:
        """Devine l'échelle basée sur la valeur du seuil"""
        if threshold < 0:
            if threshold >= -3:
                return "zscore"
            elif threshold >= -100:
                return "-100-100"
            else:
                return "raw"
        elif threshold <= 1.0:
            return "0-1"
        elif threshold <= 100:
            return "0-100"
        else:
            return "raw"
    
    def analyze_inconsistencies(self) -> None:
        """Analyse les incohérences dans les usages d'indicateurs"""
        for indicator_name, usages in self.indicator_usages.items():
            if len(usages) < 2:
                continue  # Pas assez d'usages pour détecter une incohérence
                
            # Grouper par échelle suspectée
            scales_used = defaultdict(list)
            for usage in usages:
                scales_used[usage.suspected_scale].append(usage)
            
            # Si plus d'une échelle détectée = incohérence
            if len(scales_used) > 1:
                issue_desc = f"Échelles multiples détectées: {list(scales_used.keys())}"
                severity = "critical" if indicator_name in self.known_scales else "warning"
                
                inconsistency = ScaleInconsistency(
                    indicator_name=indicator_name,
                    usages=usages,
                    suspected_issue=issue_desc,
                    severity=severity
                )
                
                self.inconsistencies.append(inconsistency)
            
            # Vérifier contre les échelles connues
            elif indicator_name in self.known_scales:
                expected_scale = self.known_scales[indicator_name]
                actual_scale = list(scales_used.keys())[0]
                
                if not self._scales_compatible(expected_scale, actual_scale):
                    issue_desc = f"Échelle attendue: {expected_scale}, détectée: {actual_scale}"
                    severity = "critical"
                    
                    inconsistency = ScaleInconsistency(
                        indicator_name=indicator_name,
                        usages=usages,
                        suspected_issue=issue_desc,
                        severity=severity
                    )
                    
                    self.inconsistencies.append(inconsistency)
    
    def _scales_compatible(self, expected: str, actual: str) -> bool:
        """Vérifie si deux échelles sont compatibles"""
        if expected == actual:
            return True
        
        # Certaines conversions sont acceptables
        compatible_pairs = [
            ("0-1", "0-100"),  # Conversion possible
            ("zscore", "0-1"),  # Z-score peut être normalisé
        ]
        
        return (expected, actual) in compatible_pairs or (actual, expected) in compatible_pairs
    
    def generate_report(self) -> str:
        """Génère un rapport des incohérences détectées"""
        report = ["# 🔍 RAPPORT DE VALIDATION DES ÉCHELLES D'INDICATEURS", ""]
        report.append(f"📊 **Statistiques:**")
        report.append(f"- Indicateurs analysés: {len(self.indicator_usages)}")
        report.append(f"- Incohérences détectées: {len(self.inconsistencies)}")
        report.append(f"- Critiques: {len([i for i in self.inconsistencies if i.severity == 'critical'])}")
        report.append(f"- Avertissements: {len([i for i in self.inconsistencies if i.severity == 'warning'])}")
        report.append("")
        
        if not self.inconsistencies:
            report.append("✅ **Aucune incohérence détectée !**")
            return "\\n".join(report)
        
        # Trier par sévérité
        critical = [i for i in self.inconsistencies if i.severity == "critical"]
        warnings = [i for i in self.inconsistencies if i.severity == "warning"]
        
        if critical:
            report.append("## 🚨 INCOHÉRENCES CRITIQUES")
            report.append("")
            for inc in critical:
                report.extend(self._format_inconsistency(inc))
                report.append("")
        
        if warnings:
            report.append("## ⚠️ AVERTISSEMENTS")
            report.append("")
            for inc in warnings:
                report.extend(self._format_inconsistency(inc))
                report.append("")
        
        # Recommandations
        report.append("## 🔧 RECOMMANDATIONS")
        report.append("")
        report.append("1. **Standardiser les échelles** : Choisir une échelle cohérente pour chaque indicateur")
        report.append("2. **Documenter les conversions** : Ajouter des commentaires expliquant les conversions")
        report.append("3. **Tester après corrections** : Vérifier que les validators fonctionnent correctement")
        report.append("4. **Automatiser la validation** : Intégrer ce script dans le CI/CD")
        
        return "\\n".join(report)
    
    def _format_inconsistency(self, inc: ScaleInconsistency) -> List[str]:
        """Formate une incohérence pour le rapport"""
        lines = []
        severity_icon = "🚨" if inc.severity == "critical" else "⚠️"
        
        lines.append(f"### {severity_icon} `{inc.indicator_name}`")
        lines.append(f"**Problème:** {inc.suspected_issue}")
        lines.append("")
        lines.append("**Usages détectés:**")
        
        for usage in inc.usages:
            lines.append(f"- `{usage.validator_name}:{usage.line_number}` - {usage.usage_pattern} (échelle: {usage.suspected_scale})")
        
        # Suggestion de correction si disponible
        if inc.indicator_name in self.known_scales:
            expected = self.known_scales[inc.indicator_name]
            lines.append("")
            lines.append(f"**✅ Échelle recommandée:** `{expected}`")
        
        return lines
    
    def run_validation(self) -> None:
        """Lance la validation complète"""
        print("🔍 Validation des échelles d'indicateurs - ROOT Trading Bot")
        print("=" * 60)
        
        print("📂 Scan des validators...", end=" ")
        self.scan_validators()
        print(f"✅ {len(self.indicator_usages)} indicateurs trouvés")
        
        print("🔎 Analyse des incohérences...", end=" ")
        self.analyze_inconsistencies()
        print(f"✅ {len(self.inconsistencies)} incohérences détectées")
        
        print("📝 Génération du rapport...")
        report = self.generate_report()
        
        # Sauvegarde du rapport
        report_path = self.root_path / "scripts" / "indicator_scale_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Rapport sauvegardé: {report_path}")
        
        # Affichage résumé
        critical_count = len([i for i in self.inconsistencies if i.severity == "critical"])
        if critical_count > 0:
            print(f"🚨 {critical_count} incohérences CRITIQUES détectées !")
            sys.exit(1)
        else:
            print("✅ Aucune incohérence critique détectée")

def main():
    """Point d'entrée principal"""
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
    else:
        # Détecter automatiquement le chemin ROOT
        current_path = Path(__file__).parent.parent
        if (current_path / "signal_aggregator").exists():
            root_path = str(current_path)
        else:
            print("❌ Impossible de détecter le chemin ROOT automatiquement")
            print("Usage: python validate_indicator_scales.py [chemin_root]")
            sys.exit(1)
    
    validator = IndicatorScaleValidator(root_path)
    validator.run_validation()

if __name__ == "__main__":
    main()