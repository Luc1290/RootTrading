"""
Module strategies - Toutes les stratégies de trading.
"""

from .base_strategy import BaseStrategy
from .RSI_Cross_Strategy import RSI_Cross_Strategy
from .StochRSI_Rebound_Strategy import StochRSI_Rebound_Strategy
from .CCI_Reversal_Strategy import CCI_Reversal_Strategy
from .ROC_Threshold_Strategy import ROC_Threshold_Strategy
from .TRIX_Crossover_Strategy import TRIX_Crossover_Strategy
from .PPO_Crossover_Strategy import PPO_Crossover_Strategy
from .MACD_Crossover_Strategy import MACD_Crossover_Strategy
from .ADX_Direction_Strategy import ADX_Direction_Strategy
from .EMA_Cross_Strategy import EMA_Cross_Strategy
from .TEMA_Slope_Strategy import TEMA_Slope_Strategy
from .HullMA_Slope_Strategy import HullMA_Slope_Strategy
from .Supertrend_Reversal_Strategy import Supertrend_Reversal_Strategy
from .ParabolicSAR_Bounce_Strategy import ParabolicSAR_Bounce_Strategy
from .Bollinger_Touch_Strategy import Bollinger_Touch_Strategy
from .ATR_Breakout_Strategy import ATR_Breakout_Strategy
from .Stochastic_Oversold_Buy_Strategy import Stochastic_Oversold_Buy_Strategy
from .WilliamsR_Rebound_Strategy import WilliamsR_Rebound_Strategy
from .OBV_Crossover_Strategy import OBV_Crossover_Strategy
from .VWAP_Support_Resistance_Strategy import VWAP_Support_Resistance_Strategy
from .Donchian_Breakout_Strategy import Donchian_Breakout_Strategy
from .ZScore_Extreme_Reversal_Strategy import ZScore_Extreme_Reversal_Strategy
from .Support_Breakout_Strategy import Support_Breakout_Strategy
from .Resistance_Rejection_Strategy import Resistance_Rejection_Strategy
from .Liquidity_Sweep_Buy_Strategy import Liquidity_Sweep_Buy_Strategy
from .Pump_Dump_Pattern_Strategy import Pump_Dump_Pattern_Strategy
from .Spike_Reaction_Buy_Strategy import Spike_Reaction_Buy_Strategy
from .Range_Breakout_Confirmation_Strategy import Range_Breakout_Confirmation_Strategy
from .MultiTF_ConfluentEntry_Strategy import MultiTF_ConfluentEntry_Strategy

__all__ = ['BaseStrategy'] + ['RSI_Cross_Strategy', 'StochRSI_Rebound_Strategy', 'CCI_Reversal_Strategy', 'ROC_Threshold_Strategy', 'TRIX_Crossover_Strategy', 'PPO_Crossover_Strategy', 'MACD_Crossover_Strategy', 'ADX_Direction_Strategy', 'EMA_Cross_Strategy', 'TEMA_Slope_Strategy', 'HullMA_Slope_Strategy', 'Supertrend_Reversal_Strategy', 'ParabolicSAR_Bounce_Strategy', 'Bollinger_Touch_Strategy', 'ATR_Breakout_Strategy', 'Stochastic_Oversold_Buy_Strategy', 'WilliamsR_Rebound_Strategy', 'OBV_Crossover_Strategy', 'VWAP_Support_Resistance_Strategy', 'Donchian_Breakout_Strategy', 'ZScore_Extreme_Reversal_Strategy', 'Support_Breakout_Strategy', 'Resistance_Rejection_Strategy', 'Liquidity_Sweep_Buy_Strategy', 'Pump_Dump_Pattern_Strategy', 'Spike_Reaction_Buy_Strategy', 'Range_Breakout_Confirmation_Strategy', 'MultiTF_ConfluentEntry_Strategy']
