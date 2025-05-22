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
    
    def __init__(self, symbol_info: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialise les contraintes de symbole avec des valeurs par défaut.
        
        Args:
            symbol_info: Informations de symboles obtenues depuis fetch_exchange_info()
        """
        self.symbol_info = symbol_info or {}
        
        # Quantité minimale par défaut pour les symboles (fallback si pas de données en temps réel)
        self.default_min_quantities = {
            "BTCUSDC": 0.00001,   # minQty selon Binance
            "ETHUSDC": 0.0001,    # minQty selon Binance
            "ETHBTC": 0.0001,     # minQty selon Binance
        }
        
        # Pas de quantité par défaut (step size) pour les symboles (fallback)
        self.default_step_sizes = {
            "BTCUSDC": 0.00001,   # stepSize selon Binance (5 décimales)
            "ETHUSDC": 0.0001,    # stepSize selon Binance (4 décimales)
            "ETHBTC": 0.0001,     # stepSize selon Binance (4 décimales)
        }
        
        # Valeur minimale des ordres par défaut (min notional) (fallback)
        self.default_min_notionals = {
            "BTCUSDC": 10.0,
            "ETHUSDC": 10.0,
            "ETHBTC": 0.0001,     # 0.0001 BTC ≈ 10 USDC, pas 10 BTC !
        }
        
        # Précision des prix par défaut (fallback)
        self.default_price_precisions = {
            "BTCUSDC": 2,  # 2 décimales (ex: 50000.25)
            "ETHUSDC": 2,  # 2 décimales (ex: 3000.50)
            "ETHBTC": 5,   # 5 décimales (ex: 0.02402)
        }
        
        logger.info(f"✅ Contraintes de symbole initialisées avec {len(self.symbol_info)} symboles en temps réel")
    
    def get_min_qty(self, symbol: str) -> float:
        """
        Récupère la quantité minimale pour un symbole.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            
        Returns:
            Quantité minimale
        """
        # Utiliser les données en temps réel si disponibles
        if symbol in self.symbol_info and 'min_qty' in self.symbol_info[symbol]:
            return self.symbol_info[symbol]['min_qty']
        # Sinon, utiliser les valeurs par défaut
        return self.default_min_quantities.get(symbol, 0.001)
    
    def get_step_size(self, symbol: str) -> float:
        """
        Récupère le pas de quantité pour un symbole.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            
        Returns:
            Pas de quantité
        """
        # Utiliser les données en temps réel si disponibles
        if symbol in self.symbol_info and 'step_size' in self.symbol_info[symbol]:
            return self.symbol_info[symbol]['step_size']
        # Sinon, utiliser les valeurs par défaut
        return self.default_step_sizes.get(symbol, 0.0001)
    
    def get_min_notional(self, symbol: str) -> float:
        """
        Récupère la valeur minimale d'un ordre pour un symbole.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            
        Returns:
            Valeur minimale de l'ordre
        """
        # Utiliser les données en temps réel si disponibles
        if symbol in self.symbol_info and 'min_notional' in self.symbol_info[symbol]:
            return self.symbol_info[symbol]['min_notional']
        # Sinon, utiliser les valeurs par défaut
        return self.default_min_notionals.get(symbol, 10.0)
    
    def get_price_precision(self, symbol: str) -> int:
        """
        Récupère la précision des prix pour un symbole.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            
        Returns:
            Nombre de décimales pour les prix
        """
        # Pour la précision des prix, calculer depuis tick_size si disponible
        if symbol in self.symbol_info and 'tick_size' in self.symbol_info[symbol]:
            tick_size = self.symbol_info[symbol]['tick_size']
            # Calculer le nombre de décimales depuis tick_size
            # Ex: 0.01 -> 2 décimales, 0.001 -> 3 décimales
            import math
            if tick_size > 0:
                return max(0, -int(math.floor(math.log10(tick_size))))
        # Sinon, utiliser les valeurs par défaut
        return self.default_price_precisions.get(symbol, 2)
    
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
        
    def calculate_min_quantity(self, symbol: str, price: float) -> float:
        """
        Calcule la quantité minimale nécessaire pour respecter à la fois
        les contraintes de min_qty et min_notional.
        
        Args:
            symbol: Symbole (ex: 'BTCUSDC')
            price: Prix actuel
            
        Returns:
            Quantité minimale arrondie adaptée au step_size
        """
        min_qty = self.get_min_qty(symbol)
        min_notional = self.get_min_notional(symbol)
        
        # Calculer la quantité minimale pour respecter min_notional
        notional_min_qty = min_notional / price if price > 0 else min_qty
        
        # Prendre le maximum des deux contraintes
        required_min_qty = max(min_qty, notional_min_qty)
        
        # Arrondir au step size supérieur
        step_size = self.get_step_size(symbol)
        steps = (required_min_qty / step_size) if step_size > 0 else 0
        rounded_steps = math.ceil(steps)
        rounded_qty = rounded_steps * step_size
        
        logger.info(f"Quantité minimale calculée pour {symbol} @ {price}: {rounded_qty} (min_qty: {min_qty}, notional min: {notional_min_qty})")
        return rounded_qty