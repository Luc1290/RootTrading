"""
Système de validation hiérarchique pour les signaux de trading.
Remplace l'ancien système de scoring simple par une approche multi-niveaux.
"""

from typing import Dict, List, Any, Tuple, Optional
import logging
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.validator_hierarchy import (
    VALIDATOR_HIERARCHY, 
    VALIDATION_THRESHOLDS,
    COMBINATION_RULES,
    get_validator_level,
    has_veto_power
)

logger = logging.getLogger(__name__)


class HierarchicalValidator:
    """
    Gestionnaire de validation hiérarchique des signaux.
    """
    
    def __init__(self):
        self.hierarchy = VALIDATOR_HIERARCHY
        self.thresholds = VALIDATION_THRESHOLDS
        self.combination_rules = COMBINATION_RULES
        
    def validate_with_hierarchy(self, validation_results: List[Dict[str, Any]], 
                               signal: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Valide les résultats selon la hiérarchie définie.
        
        Args:
            validation_results: Résultats de tous les validators
            signal: Signal original pour contexte
            
        Returns:
            Tuple (is_valid, final_score, detailed_analysis)
        """
        # Grouper les résultats par niveau
        results_by_level = self._group_results_by_level(validation_results)
        
        # Analyser chaque niveau
        level_analysis = {}
        veto_triggered = False
        veto_reason = None
        
        for level in ['critical', 'important', 'standard']:
            if level not in results_by_level:
                continue
                
            analysis = self._analyze_level(level, results_by_level[level])
            level_analysis[level] = analysis
            
            # Vérifier le veto pour les validators critiques
            if level == 'critical' and self.hierarchy[level]['veto_power']:
                for result in results_by_level[level]:
                    if not result['is_valid']:
                        veto_triggered = True
                        veto_reason = f"{result['validator_name']}: {result['reason']}"
                        logger.warning(f"VETO déclenché par {result['validator_name']} "
                                     f"pour {signal['symbol']} {signal['side']}: {result['reason']}")
                        break
                        
            # Vérifier le taux de passage minimum
            if not analysis['meets_min_pass_rate']:
                if level == 'critical':
                    veto_triggered = True
                    veto_reason = f"Taux de passage critique insuffisant: {analysis['pass_rate']:.1%}"
        
        # Si veto, rejeter immédiatement
        if veto_triggered:
            return False, 0.0, {
                'veto': True,
                'veto_reason': veto_reason,
                'level_analysis': level_analysis
            }
            
        # Calculer le score final hiérarchique
        final_score = self._calculate_hierarchical_score(results_by_level, level_analysis)
        
        # Appliquer les bonus de combinaison
        combination_bonuses = self._apply_combination_bonuses(validation_results, final_score)
        final_score = combination_bonuses['final_score']
        
        # Décision finale
        is_valid = self._make_final_decision(level_analysis, final_score)
        
        detailed_analysis = {
            'veto': False,
            'level_analysis': level_analysis,
            'final_score': final_score,
            'combination_bonuses': combination_bonuses['applied_bonuses'],
            'is_valid': is_valid
        }
        
        return is_valid, final_score, detailed_analysis
        
    def _group_results_by_level(self, validation_results: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Groupe les résultats par niveau hiérarchique."""
        grouped = {'critical': [], 'important': [], 'standard': [], 'unknown': []}
        
        for result in validation_results:
            level = get_validator_level(result['validator_name'])
            grouped[level].append(result)
            
        return grouped
        
    def _analyze_level(self, level: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyse les résultats d'un niveau spécifique."""
        if not results:
            return {
                'total': 0,
                'passed': 0,
                'pass_rate': 0.0,
                'avg_score': 0.0,
                'meets_min_pass_rate': True,  # Pas de validators = pas de contrainte
                'validators': []
            }
            
        total = len(results)
        passed = sum(1 for r in results if r['is_valid'])
        pass_rate = passed / total if total > 0 else 0.0
        
        scores = [r['score'] for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        # Vérifier le taux de passage minimum
        min_pass_rate = self.hierarchy[level]['min_pass_rate']
        meets_min_pass_rate = pass_rate >= min_pass_rate
        
        # Détail des validators
        validators = [{
            'name': r['validator_name'],
            'passed': r['is_valid'],
            'score': r['score'],
            'reason': r['reason']
        } for r in results]
        
        return {
            'total': total,
            'passed': passed,
            'pass_rate': pass_rate,
            'avg_score': avg_score,
            'meets_min_pass_rate': meets_min_pass_rate,
            'min_pass_rate_required': min_pass_rate,
            'validators': validators
        }
        
    def _calculate_hierarchical_score(self, results_by_level: Dict[str, List[Dict]], 
                                     level_analysis: Dict[str, Dict]) -> float:
        """Calcule le score final en appliquant les multiplicateurs hiérarchiques."""
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for level, results in results_by_level.items():
            if level == 'unknown' or not results:
                continue
                
            level_config = self.hierarchy.get(level, {})
            multiplier = level_config.get('weight_multiplier', 1.0)
            
            for result in results:
                weighted_score = result['score'] * multiplier
                total_weighted_score += weighted_score
                total_weight += multiplier
                
        # Score moyen pondéré
        base_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Appliquer des pénalités si certains niveaux ne passent pas
        for level, analysis in level_analysis.items():
            if not analysis['meets_min_pass_rate'] and level in ['important']:
                penalty = 0.2  # 20% de pénalité si les validators importants échouent
                base_score *= (1 - penalty)
                logger.debug(f"Pénalité appliquée pour échec niveau {level}: -{penalty:.0%}")
                
        return min(1.0, max(0.0, base_score))
        
    def _apply_combination_bonuses(self, validation_results: List[Dict[str, Any]], 
                                   current_score: float) -> Dict[str, Any]:
        """Applique les bonus selon les combinaisons de validators."""
        applied_bonuses = []
        final_score = current_score
        
        # Créer un dictionnaire pour accès rapide
        results_dict = {r['validator_name']: r for r in validation_results}
        
        # Vérifier chaque règle de combinaison
        for rule_name, rule in self.combination_rules.items():
            if rule_name == 'perfect_critical':
                # Tous les critiques passent avec score élevé
                critical_results = [r for r in validation_results 
                                  if get_validator_level(r['validator_name']) == 'critical']
                if critical_results:
                    all_pass = all(r['is_valid'] for r in critical_results)
                    avg_score = sum(r['score'] for r in critical_results) / len(critical_results)
                    
                    if all_pass and avg_score >= rule['min_avg_score']:
                        bonus = rule['bonus']
                        final_score = min(1.0, final_score + bonus)
                        applied_bonuses.append({
                            'rule': rule_name,
                            'bonus': bonus,
                            'reason': rule['description']
                        })
                        
            elif rule_name == 'strong_trend_volume':
                # Tendance et volume alignés
                validators_needed = rule['validators']
                if all(v in results_dict for v in validators_needed):
                    all_valid = all(results_dict[v]['is_valid'] for v in validators_needed)
                    all_high_score = all(results_dict[v]['score'] >= rule['min_scores'] 
                                       for v in validators_needed)
                    
                    if all_valid and all_high_score:
                        bonus = rule['bonus']
                        final_score = min(1.0, final_score + bonus)
                        applied_bonuses.append({
                            'rule': rule_name,
                            'bonus': bonus,
                            'reason': rule['description']
                        })
                        
            elif rule_name == 'multi_tf_consensus':
                # Consensus multi-timeframe fort
                validator = rule['validator']
                if validator in results_dict:
                    result = results_dict[validator]
                    if result['is_valid'] and result['score'] >= rule['min_score']:
                        bonus = rule['bonus']
                        final_score = min(1.0, final_score + bonus)
                        applied_bonuses.append({
                            'rule': rule_name,
                            'bonus': bonus,
                            'reason': rule['description']
                        })
                        
        return {
            'final_score': final_score,
            'applied_bonuses': applied_bonuses,
            'total_bonus': final_score - current_score
        }
        
    def _make_final_decision(self, level_analysis: Dict[str, Dict], final_score: float) -> bool:
        """Prend la décision finale basée sur l'analyse hiérarchique."""
        # Vérifier que tous les niveaux requis passent
        for level in ['critical', 'important']:
            if level in level_analysis and not level_analysis[level]['meets_min_pass_rate']:
                return False
                
        # Vérifier le score final minimum (plus strict)
        min_final_score = 0.7  # Score minimum rehaussé pour la qualité
        if final_score < min_final_score:
            return False
            
        return True
        
    def get_validator_importance(self, validator_name: str) -> str:
        """Retourne l'importance d'un validator pour l'affichage."""
        level = get_validator_level(validator_name)
        importance_map = {
            'critical': '🔴 CRITIQUE',
            'important': '🟡 IMPORTANT',
            'standard': '🟢 STANDARD',
            'unknown': '⚪ INCONNU'
        }
        return importance_map.get(level, '⚪ INCONNU')