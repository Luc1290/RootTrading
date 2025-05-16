# trader/src/exchange/constraints.py
"""
Gestion des contraintes de trading de Binance (min. qtés, précisions, etc.)
Fournit des informations sur les limites de trading par symbole.
"""
import logging
import math
from typing import Dict, Any, Optional
from decimal import Decimal, getcontext, ROUND_DOWN

# Configuration de la précision décimale
getcontext().prec = 28

# Configuration du logging
logger = logging.getLogger(__name__)

class BinanceSymbolConstraints:
    """
    Gère les contraintes de trading par symbole pour Binance.
    Fournit des méthodes pour vérifier et ajuster les quantités et prix.
    """
    
    def __init__(self):
        """
        Initialise les contraintes de symbole avec des valeurs par défaut.
        """
        # Limites minimales pour les symboles courants
        self.min_quantities = {
            "BTCUSDC": 0.001,
            "ETHUSDC": 0.01,
        }
        
        # Pas de quantité (step size) pour les symboles
        self.step_sizes = {
            "BTCUSDC": 0.00001,
            "ETHUSDC": 0.001,
        }
        
        # Valeur minimale des ordres (min notional)
        self.min_notionals = {
            "BTCUSDC": 10.0,
            "ETHUSDC": 10.0,
        }
        
        # Précision des prix
        self.price_precisions = {
            "BTCUSDC": 2,  # 2 décimales (ex: 50000.25)
            "ETHUSDC": 2,  # 2 décimales (ex: 3000.50)
        }
        
        logger.info("✅ Contraintes de symbole initialisées")
    
    def get_min_qty(self, symbol: str) -> float:
        """
        Récupère la quantité minimale pour un symbole.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            
        Returns:
            Quantité minimale
        """
        return self.min_quantities.get(symbol, 0.001)
    
    def get_step_size(self, symbol: str) -> float:
        """
        Récupère le pas de quantité pour un symbole.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            
        Returns:
            Pas de quantité
        """
        return self.step_sizes.get(symbol, 0.0001)
    
    def get_min_notional(self, symbol: str) -> float:
        """
        Récupère la valeur minimale d'un ordre pour un symbole.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            
        Returns:
            Valeur minimale de l'ordre
        """
        return self.min_notionals.get(symbol, 10.0)
    
    def get_price_precision(self, symbol: str) -> int:
        """
        Récupère la précision des prix pour un symbole.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            
        Returns:
            Nombre de décimales pour les prix
        """
        return self.price_precisions.get(symbol, 2)
    
    def truncate_quantity(self, symbol: str, quantity: float) -> float:
        """
        Tronque la quantité au pas (stepSize) Binance, sans erreur binaire.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            quantity: Quantité à tronquer
            
        Returns:
            Quantité tronquée
        """
        if quantity <= 0:
            logger.warning(f"⚠️ Quantité négative ou nulle : {quantity}")
            return self.get_min_qty(symbol)
        
        step_size = self.get_step_size(symbol)          # ex. 0.000001
        step_dec = Decimal(str(step_size))
        qty_dec = Decimal(str(quantity))
        
        # Arrondi « floor » au multiple de stepSize
        truncated = (qty_dec // step_dec) * step_dec
        truncated = truncated.quantize(step_dec, rounding=ROUND_DOWN)
        
        # Sécurité : si on tombe à 0, on repasse à la quantité mini
        if truncated <= 0:
            logger.warning(f"⚠️ Quantité nulle après troncature : {quantity} → {truncated}")
            truncated = Decimal(str(self.get_min_qty(symbol)))
        
        return float(truncated)
    
    def round_price(self, symbol: str, price: float) -> float:
        """
        Arrondit le prix selon le tickSize du symbole.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            price: Prix à arrondir
            
        Returns:
            Prix arrondi
        """
        precision = self.get_price_precision(symbol)
        return round(price, precision)
    
    def is_quantity_valid(self, symbol: str, quantity: float) -> bool:
        """
        Vérifie si la quantité est valide selon les règles de Binance.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            quantity: Quantité à vérifier
            
        Returns:
            True si la quantité est valide, False sinon
        """
        # Vérifier que la quantité n'est pas négative ou nulle
        if quantity <= 0:
            return False
        
        # Vérifier si la quantité est supérieure au minimum
        min_qty = self.get_min_qty(symbol)
        return quantity >= min_qty
    
    def is_notional_valid(self, symbol: str, quantity: float, price: float) -> bool:
        """
        Vérifie si la valeur totale (quantité * prix) est valide selon les règles de Binance.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            quantity: Quantité
            price: Prix
            
        Returns:
            True si la valeur est valide, False sinon
        """
        # Vérifier que la quantité et le prix ne sont pas nuls
        if quantity <= 0 or price <= 0:
            return False
        
        # Calculer la valeur totale
        notional = quantity * price
        
        # Vérifier si la valeur est supérieure au minimum
        min_notional = self.get_min_notional(symbol)
        return notional >= min_notional