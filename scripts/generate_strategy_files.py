#!/usr/bin/env python3
"""
Script pour générer automatiquement les fichiers de stratégies et validators.
"""

import os
from pathlib import Path

# Liste des stratégies
STRATEGIES = [
    "RSI_Cross_Strategy",
    "StochRSI_Rebound_Strategy",
    "CCI_Reversal_Strategy",
    "ROC_Threshold_Strategy",
    "TRIX_Crossover_Strategy",
    "PPO_Crossover_Strategy",
    "MACD_Crossover_Strategy",
    "ADX_Direction_Strategy",
    "EMA_Cross_Strategy",
    "TEMA_Slope_Strategy",
    "HullMA_Slope_Strategy",
    "Supertrend_Reversal_Strategy",
    "ParabolicSAR_Bounce_Strategy",
    "Bollinger_Touch_Strategy",
    "ATR_Breakout_Strategy",
    "Stochastic_Oversold_Buy_Strategy",
    "WilliamsR_Rebound_Strategy",
    "OBV_Crossover_Strategy",
    "VWAP_Support_Resistance_Strategy",
    "Donchian_Breakout_Strategy",
    "ZScore_Extreme_Reversal_Strategy",
    "Support_Breakout_Strategy",
    "Resistance_Rejection_Strategy",
    "Liquidity_Sweep_Buy_Strategy",
    "Pump_Dump_Pattern_Strategy",
    "Spike_Reaction_Buy_Strategy",
    "Range_Breakout_Confirmation_Strategy",
    "MultiTF_ConfluentEntry_Strategy"
]

# Liste des validators
VALIDATORS = [
    "Trend_Alignment_Validator",
    "ADX_TrendStrength_Validator",
    "MACD_Regime_Validator",
    "RSI_Regime_Validator",
    "Regime_Strength_Validator",
    "Market_Structure_Validator",
    "Trend_Smoothness_Validator",
    "ATR_Volatility_Validator",
    "Bollinger_Width_Validator",
    "Volatility_Regime_Validator",
    "VWAP_Context_Validator",
    "Volume_Ratio_Validator",
    "Volume_Buildup_Validator",
    "Volume_Spike_Validator",
    "Liquidity_Sweep_Validator",
    "Volume_Quality_Score_Validator",
    "S_R_Level_Proximity_Validator",
    "Pivot_Strength_Validator",
    "Psychological_Level_Validator",
    "MultiTF_Consensus_Validator",
    "Range_Validator",
    "ZScore_Context_Validator"
]

# Template pour les stratégies
STRATEGY_TEMPLATE = '''"""
{class_name} - Stratégie basée sur {indicator}.
"""

from typing import Dict, Any
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class {class_name}(BaseStrategy):
    """
    Stratégie {strategy_type} utilisant {indicator}.
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # TODO: Paramètres spécifiques à la stratégie
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur {indicator}.
        """
        if not self.validate_data():
            return {{
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Données insuffisantes",
                "metadata": {{}}
            }}
            
        # TODO: Implémenter la logique de la stratégie
        
        return {{
            "side": None,
            "confidence": 0.0,
            "strength": "weak",
            "reason": "À implémenter",
            "metadata": {{
                "strategy": self.name,
                "symbol": self.symbol
            }}
        }}
'''

# Template pour les validators
VALIDATOR_TEMPLATE = '''"""
{class_name} - Validator pour {validation_type}.
"""

from typing import Dict, Any
from .base_validator import BaseValidator
import logging

logger = logging.getLogger(__name__)


class {class_name}(BaseValidator):
    """
    Valide les signaux en fonction de {validation_type}.
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], context: Dict[str, Any]):
        super().__init__(symbol, data, context)
        # TODO: Paramètres spécifiques au validator
        
    def validate_signal(self, signal: Dict[str, Any]) -> bool:
        """
        Valide le signal en fonction de {validation_type}.
        """
        if not self.validate_data():
            return False
            
        # TODO: Implémenter la logique de validation
        
        return True
'''


def generate_files():
    """Génère tous les fichiers de stratégies et validators."""
    
    # Chemin de base
    base_path = Path("/mnt/e/RootTrading/RootTrading")
    
    # Génération des stratégies
    strategies_path = base_path / "analyzer" / "strategies"
    strategies_path.mkdir(parents=True, exist_ok=True)
    
    print("Génération des stratégies...")
    for strategy in STRATEGIES:
        file_path = strategies_path / f"{strategy}.py"
        if not file_path.exists():
            # Extraire les infos du nom
            parts = strategy.replace("_Strategy", "").split("_")
            indicator = " ".join(parts[:-1]) if len(parts) > 1 else parts[0]
            strategy_type = parts[-1] if len(parts) > 1 else "trading"
            
            content = STRATEGY_TEMPLATE.format(
                class_name=strategy,
                indicator=indicator,
                strategy_type=strategy_type
            )
            
            with open(file_path, "w") as f:
                f.write(content)
            print(f"  ✓ {strategy}.py")
        else:
            print(f"  - {strategy}.py (existe déjà)")
    
    # Génération des validators
    validators_path = base_path / "signal_aggregator" / "validators"
    validators_path.mkdir(parents=True, exist_ok=True)
    
    print("\nGénération des validators...")
    for validator in VALIDATORS:
        file_path = validators_path / f"{validator}.py"
        if not file_path.exists():
            # Extraire les infos du nom
            validation_type = validator.replace("_Validator", "").replace("_", " ")
            
            content = VALIDATOR_TEMPLATE.format(
                class_name=validator,
                validation_type=validation_type
            )
            
            with open(file_path, "w") as f:
                f.write(content)
            print(f"  ✓ {validator}.py")
        else:
            print(f"  - {validator}.py (existe déjà)")
    
    # Mise à jour des __init__.py
    print("\nMise à jour des __init__.py...")
    
    # __init__.py pour les stratégies
    strategies_init = strategies_path / "__init__.py"
    strategy_imports = "\n".join([f"from .{s} import {s}" for s in STRATEGIES])
    strategy_all = "['" + "', '".join(STRATEGIES) + "']"
    
    init_content = f'''"""
Module strategies - Toutes les stratégies de trading.
"""

from .base_strategy import BaseStrategy
{strategy_imports}

__all__ = ['BaseStrategy'] + {strategy_all}
'''
    
    with open(strategies_init, "w") as f:
        f.write(init_content)
    print("  ✓ analyzer/strategies/__init__.py")
    
    # __init__.py pour les validators
    validators_init = validators_path / "__init__.py"
    validator_imports = "\n".join([f"from .{v} import {v}" for v in VALIDATORS])
    validator_all = "['" + "', '".join(VALIDATORS) + "']"
    
    init_content = f'''"""
Module validators - Tous les validators de signaux.
"""

from .base_validator import BaseValidator
{validator_imports}

__all__ = ['BaseValidator'] + {validator_all}
'''
    
    with open(validators_init, "w") as f:
        f.write(init_content)
    print("  ✓ signal_aggregator/validators/__init__.py")
    
    print("\nGénération terminée!")


if __name__ == "__main__":
    generate_files()