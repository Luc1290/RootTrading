"""
Visualization Source Module - Opportunity Calculator Pro Suite

Ce package contient les modules de calcul et d'analyse d'opportunités de trading :
- opportunity_scoring: Système de scoring multi-niveaux (0-100) - v4.1
- opportunity_scoring_v5: Système de scoring optimisé exploitant 25+ indicateurs DB - v5.0
- opportunity_validator: Validation stricte en 4 niveaux (gates)
- opportunity_early_detector: Détection précoce avec leading indicators
- opportunity_calculator_pro: Orchestrateur principal (utilise v5.0)

Version: 5.0 - Maximum DB Indicator Utilization
"""

__version__ = "5.0.0"
__all__ = [
    "EarlySignal",
    "OpportunityCalculatorPro",
    "OpportunityEarlyDetector",
    "OpportunityScore",
    "OpportunityScoring",
    "OpportunityScoringV5",
    "OpportunityValidator",
    "TradingOpportunity",
    "ValidationSummary",
]
