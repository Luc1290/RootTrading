"""
Module de gestion de l'allocation dynamique du capital.
Calcule les montants optimaux selon la force des signaux et les balances disponibles.
"""
import logging
import os
from typing import Dict, Any, Tuple, Optional
from shared.src.schemas import StrategySignal
from shared.src.enums import SignalStrength, OrderSide

logger = logging.getLogger(__name__)


class AllocationManager:
    """
    Gère l'allocation dynamique du capital selon la force des signaux.
    Applique des marges de sécurité pour éviter les échecs d'ordres.
    """
    
    def __init__(self):
        """
        Initialise le gestionnaire d'allocation.
        """
        # Pourcentages d'allocation par force de signal (configurables via env)
        self.allocation_percentages = {
            SignalStrength.WEAK: float(os.getenv('ALLOCATION_WEAK_PCT', 2.0)),
            SignalStrength.MODERATE: float(os.getenv('ALLOCATION_MODERATE_PCT', 5.0)),
            SignalStrength.STRONG: float(os.getenv('ALLOCATION_STRONG_PCT', 8.0)),
            SignalStrength.VERY_STRONG: float(os.getenv('ALLOCATION_VERY_STRONG_PCT', 12.0))
        }
        
        # Montants minimum par devise
        self.min_amounts = {
            'USDC': float(os.getenv('MIN_TRADE_USDC', 10.0)),
            'BTC': float(os.getenv('MIN_TRADE_BTC', 0.0001)),
            'ETH': float(os.getenv('MIN_TRADE_ETH', 0.003)),
            'BNB': float(os.getenv('MIN_TRADE_BNB', 0.02))
        }
        
        # Montants maximum par devise
        self.max_amounts = {
            'USDC': float(os.getenv('MAX_TRADE_USDC', 250.0)),
            'BTC': float(os.getenv('MAX_TRADE_BTC', 0.005)),
            'ETH': float(os.getenv('MAX_TRADE_ETH', 0.13)),
            'BNB': float(os.getenv('MAX_TRADE_BNB', 2.0))
        }
        
        # Marges de sécurité pour éviter les échecs d'ordres
        self.safety_margins = {
            'BUY': 0.95,   # Utiliser 95% de la balance quote pour BUY
            'SELL': 0.90   # Utiliser 90% de la balance base pour SELL
        }
        
    def calculate_trade_amount(self, signal: StrategySignal, 
                              available_balance: float,
                              quote_asset: str) -> Tuple[float, str]:
        """
        Calcule le montant optimal pour un trade.
        
        Args:
            signal: Signal de trading
            available_balance: Balance disponible
            quote_asset: Actif de quote (USDC, BTC, etc.)
            
        Returns:
            Tuple (montant, actif)
        """
        # Récupérer le pourcentage d'allocation selon la force
        base_percentage = self.allocation_percentages.get(signal.strength, 5.0)
        
        # Calculer le montant basé sur le pourcentage
        calculated_amount = available_balance * (base_percentage / 100.0)
        
        # Appliquer les limites min/max
        min_amount = self.min_amounts.get(quote_asset, 10.0)
        max_amount = self.max_amounts.get(quote_asset, 100.0)
        
        final_amount = max(min_amount, min(calculated_amount, max_amount))
        
        logger.info(f"Allocation dynamique {signal.symbol}: {base_percentage}% de "
                   f"{available_balance:.8f} {quote_asset} = {final_amount:.6f} {quote_asset}")
        
        return final_amount, quote_asset
        
    def calculate_constraining_balance(self, signal: StrategySignal,
                                     balances: Dict[str, Dict[str, float]],
                                     base_asset: str, quote_asset: str) -> Dict[str, Any]:
        """
        Calcule la balance contraignante avec marge de sécurité.
        
        Args:
            signal: Signal de trading
            balances: Balances disponibles par actif
            base_asset: Actif de base (BTC, ETH, etc.)
            quote_asset: Actif de quote (USDC, etc.)
            
        Returns:
            Dict avec constraining_balance et details
        """
        try:
            if signal.side == OrderSide.BUY:
                # BUY: besoin de quote_asset
                available_quote = balances[quote_asset]['binance_free']
                constraining_balance = available_quote * self.safety_margins['BUY']
                
                logger.info(f"💡 BUY {signal.symbol}: balance contraignante basée sur {quote_asset}: "
                           f"{available_quote:.6f} * {self.safety_margins['BUY']} = "
                           f"{constraining_balance:.6f} {quote_asset}")
                           
            else:  # OrderSide.SELL
                # SELL: besoin de base_asset, convertir en équivalent quote
                available_base = balances[base_asset]['binance_free']
                constraining_balance = available_base * signal.price * self.safety_margins['SELL']
                
                logger.info(f"💡 SELL {signal.symbol}: balance contraignante basée sur {base_asset}: "
                           f"{available_base:.6f} * {signal.price:.6f} * "
                           f"{self.safety_margins['SELL']} = {constraining_balance:.6f} {quote_asset}")
                           
            return {
                'sufficient': True,
                'constraining_balance': constraining_balance,
                'reason': 'Balances suffisantes',
                'details': {
                    'base_available': balances[base_asset]['binance_free'],
                    'quote_available': balances[quote_asset]['binance_free'],
                    'safety_margin': self.safety_margins[signal.side.value]
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur calcul balance contraignante: {str(e)}")
            return {
                'sufficient': False,
                'constraining_balance': 0.0,
                'reason': f'Erreur calcul: {str(e)}',
                'details': {}
            }
            
    def check_trade_feasibility(self, signal: StrategySignal,
                               balances: Dict[str, Dict[str, float]],
                               base_asset: str, quote_asset: str) -> Dict[str, Any]:
        """
        Vérifie si un trade est faisable avec les balances actuelles.
        
        Args:
            signal: Signal de trading
            balances: Balances par actif
            base_asset: Actif de base
            quote_asset: Actif de quote
            
        Returns:
            Dict avec résultat de la vérification
        """
        try:
            # Vérifier que les actifs existent dans les balances
            if base_asset not in balances or quote_asset not in balances:
                missing = [asset for asset in [base_asset, quote_asset] if asset not in balances]
                return {
                    'sufficient': False,
                    'constraining_balance': 0.0,
                    'reason': f"Actifs manquants dans les balances: {missing}",
                    'details': {}
                }
                
            # Calculer la balance contraignante
            constraint_result = self.calculate_constraining_balance(
                signal, balances, base_asset, quote_asset
            )
            
            if not constraint_result['sufficient']:
                return constraint_result
                
            # Calculer le montant de trade proposé
            constraining_balance = constraint_result['constraining_balance']
            trade_amount, _ = self.calculate_trade_amount(
                signal, constraining_balance, quote_asset
            )
            
            # Vérifier si le montant est faisable
            if trade_amount > constraining_balance:
                return {
                    'sufficient': False,
                    'constraining_balance': constraining_balance,
                    'reason': f"Montant calculé ({trade_amount:.6f}) > balance contraignante ({constraining_balance:.6f})",
                    'details': constraint_result['details']
                }
                
            # Tout est OK
            logger.info(f"✅ Balances suffisantes pour {signal.side} {signal.symbol}: "
                       f"{base_asset}={balances[base_asset]['binance_free']:.6f}, "
                       f"{quote_asset}={balances[quote_asset]['binance_free']:.6f}")
                       
            return {
                'sufficient': True,
                'constraining_balance': constraining_balance,
                'trade_amount': trade_amount,
                'reason': 'Balances suffisantes',
                'details': constraint_result['details']
            }
            
        except Exception as e:
            logger.error(f"Erreur vérification faisabilité: {str(e)}")
            return {
                'sufficient': False,
                'constraining_balance': 0.0,
                'reason': f'Erreur vérification: {str(e)}',
                'details': {}
            }
            
    def get_asset_limits(self, asset: str) -> Dict[str, float]:
        """
        Retourne les limites min/max pour un actif.
        
        Args:
            asset: Nom de l'actif
            
        Returns:
            Dict avec min_amount et max_amount
        """
        return {
            'min_amount': self.min_amounts.get(asset, 10.0),
            'max_amount': self.max_amounts.get(asset, 100.0)
        }