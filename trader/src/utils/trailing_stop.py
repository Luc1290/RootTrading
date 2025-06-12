"""
Trailing-stop "pur" : un seul filet de sécurité.
- LONG  : suit le plus-haut (max) et se place en dessous.
- SHORT : suit le plus-bas  (min)  et se place au-dessus.
"""
import logging
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger(__name__)

class Side(Enum):
    LONG = auto()
    SHORT = auto()

class TrailingStop:
    """
    Trailing-stop "pur" : un seul filet de sécurité.
    - LONG  : suit le plus-haut (max) et se place en dessous.
    - SHORT : suit le plus-bas  (min)  et se place au-dessus.
    """

    def __init__(self, side: Side, entry_price: float, stop_pct: float = 3.0):
        """
        Initialise le trailing stop.
        
        Args:
            side: LONG ou SHORT
            entry_price: Prix d'entrée
            stop_pct: Pourcentage de retracement toléré (défaut: 3%)
        """
        self.side = side
        self.entry_price = entry_price
        self.stop_pct = stop_pct
        
        # Extrêmes favorables
        self.max_price = entry_price  # pour LONG
        self.min_price = entry_price  # pour SHORT
        
        # Stop initial
        self.stop_price = self._calc_stop(entry_price)
        
        logger.info(f"🎯 TrailingStop créé: {side.name} @ {entry_price:.6f}, "
                   f"stop initial @ {self.stop_price:.6f} (-{stop_pct}%)")

    def _calc_stop(self, ref_price: float) -> float:
        """Calcule le stop par rapport à l'extrême favorable."""
        if self.side == Side.LONG:
            return ref_price * (1 - self.stop_pct / 100)
        else:  # SHORT
            return ref_price * (1 + self.stop_pct / 100)

    def update(self, price: float) -> bool:
        """
        Met à jour l'extrême favorable et le stop.
        
        Args:
            price: Prix actuel
            
        Returns:
            True si le stop est touché ➜ il faut sortir.
        """
        old_stop = self.stop_price
        stop_hit = False
        
        if self.side == Side.LONG:
            # Nouveau record à la hausse ?
            if price > self.max_price:
                self.max_price = price
                new_stop = self._calc_stop(self.max_price)
                
                # Le stop ne peut que monter (jamais redescendre)
                if new_stop > self.stop_price:
                    self.stop_price = new_stop
                    logger.debug(f"📈 Stop LONG mis à jour: {old_stop:.6f} → {self.stop_price:.6f} "
                               f"(nouveau max: {self.max_price:.6f})")
            
            # Stop déclenché ?
            stop_hit = price <= self.stop_price
            
        else:  # SHORT
            # Nouveau record à la baisse ?
            if price < self.min_price:
                self.min_price = price
                new_stop = self._calc_stop(self.min_price)
                
                # Le stop ne peut que descendre (pour protéger plus de profit) pour SHORT
                if new_stop < self.stop_price:
                    self.stop_price = new_stop
                    logger.debug(f"📈 Stop SHORT mis à jour: {old_stop:.6f} → {self.stop_price:.6f} "
                               f"(nouveau min: {self.min_price:.6f})")
            
            # Stop déclenché ?
            stop_hit = price >= self.stop_price
        
        if stop_hit:
            profit_pct = self._calculate_profit_pct(price)
            logger.info(f"🔴 STOP DÉCLENCHÉ ! Prix: {price:.6f}, Stop: {self.stop_price:.6f}, "
                       f"Profit: {profit_pct:+.2f}%")
        
        return stop_hit

    def _calculate_profit_pct(self, exit_price: float) -> float:
        """Calcule le pourcentage de profit/perte."""
        if self.side == Side.LONG:
            return ((exit_price - self.entry_price) / self.entry_price) * 100
        else:  # SHORT
            return ((self.entry_price - exit_price) / self.entry_price) * 100

    def get_profit_if_exit_now(self, current_price: float) -> float:
        """Calcule le profit potentiel si on sortait maintenant."""
        return self._calculate_profit_pct(current_price)

    def get_status(self) -> dict:
        """Retourne l'état actuel du trailing stop."""
        return {
            'side': self.side.name,
            'entry_price': self.entry_price,
            'stop_price': self.stop_price,
            'stop_pct': self.stop_pct,
            'max_price': self.max_price,
            'min_price': self.min_price,
            'extreme_price': self.max_price if self.side == Side.LONG else self.min_price
        }

    def __str__(self) -> str:
        """Représentation string du trailing stop."""
        extreme = self.max_price if self.side == Side.LONG else self.min_price
        return (f"TrailingStop({self.side.name}, entry={self.entry_price:.6f}, "
               f"stop={self.stop_price:.6f}, extreme={extreme:.6f})")


# Exemple d'utilisation pour tests
if __name__ == "__main__":
    # Test LONG
    ts = TrailingStop(Side.LONG, entry_price=100.0, stop_pct=3.0)
    prices = [101, 104, 107, 105, 103, 102]  # flux de prix fictif
    
    print("=== Test LONG ===")
    for p in prices:
        hit = ts.update(p)
        status = ts.get_status()
        profit = ts.get_profit_if_exit_now(p)
        print(f"Prix {p:6.2f} | Stop {status['stop_price']:6.2f} | "
              f"Profit {profit:+5.1f}% | {'EXIT!' if hit else 'Hold'}")
        if hit:
            break
    
    print("\n=== Test SHORT ===")
    # Test SHORT
    ts_short = TrailingStop(Side.SHORT, entry_price=100.0, stop_pct=3.0)
    prices_short = [99, 96, 93, 95, 97, 98]
    
    for p in prices_short:
        hit = ts_short.update(p)
        status = ts_short.get_status()
        profit = ts_short.get_profit_if_exit_now(p)
        print(f"Prix {p:6.2f} | Stop {status['stop_price']:6.2f} | "
              f"Profit {profit:+5.1f}% | {'EXIT!' if hit else 'Hold'}")
        if hit:
            break