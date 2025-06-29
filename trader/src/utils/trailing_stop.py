"""
Trailing-stop "pur" : un seul filet de sécurité.
- BUY  : suit le plus-haut (max) et se place en dessous.
- SELL : suit le plus-bas  (min)  et se place au-dessus.
"""
import logging
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger(__name__)

class Side(Enum):
    BUY = auto()
    SELL = auto()

class TrailingStop:
    """
    Trailing-stop "pur" : un seul filet de sécurité.
    - BUY  : suit le plus-haut (max) et se place en dessous.
    - SELL : suit le plus-bas  (min)  et se place au-dessus.
    """

    def __init__(self, side: Side, entry_price: float, stop_pct: float = 8.0, atr_multiplier: float = 3.0, min_stop_pct: float = 6.0):
        """
        Initialise le trailing stop.
        
        Args:
            side: BUY ou SELL
            entry_price: Prix d'entrée
            stop_pct: Pourcentage de retracement toléré (défaut: 8.0% - utilisé si pas d'ATR)
            atr_multiplier: Multiplicateur ATR pour calcul adaptatif (défaut: 3.0)
            min_stop_pct: Pourcentage minimum de stop (défaut: 6.0%)
        """
        self.side = side
        self.entry_price = entry_price
        self.stop_pct = stop_pct
        self.atr_multiplier = atr_multiplier
        self.min_stop_pct = min_stop_pct
        self.current_atr = None  # Sera mis à jour via update_atr()
        
        # Extrêmes favorables
        self.max_price = entry_price  # pour BUY
        self.min_price = entry_price  # pour SELL
        
        # Stop initial
        self.stop_price = self._calc_stop(entry_price)
        
        logger.info(f"🎯 TrailingStop créé: {side.name} @ {entry_price:.6f}, "
                   f"stop initial @ {self.stop_price:.6f} (stop_pct: {stop_pct}%, ATR: {atr_multiplier}x)")

    def _calc_stop(self, ref_price: float) -> float:
        """Calcule le stop par rapport à l'extrême favorable en utilisant l'ATR si disponible."""
        # Calculer le pourcentage de stop adaptatif basé sur ATR
        effective_stop_pct = self._get_effective_stop_percentage(ref_price)
        
        if self.side == Side.BUY:
            return ref_price * (1 - effective_stop_pct / 100)
        else:  # SELL
            return ref_price * (1 + effective_stop_pct / 100)
    
    def _get_effective_stop_percentage(self, ref_price: float) -> float:
        """Calcule le pourcentage de stop effectif basé sur ATR ou valeur fixe."""
        if self.current_atr is not None and self.current_atr > 0:
            # Calcul ATR-based: max(ATR * multiplier / prix * 100, min_stop_pct)
            atr_stop_pct = (self.current_atr * self.atr_multiplier / ref_price) * 100
            effective_pct = max(atr_stop_pct, self.min_stop_pct)
            logger.debug(f"🧮 Stop ATR-based: {atr_stop_pct:.2f}% → effectif: {effective_pct:.2f}% "
                        f"(ATR: {self.current_atr:.6f}, prix: {ref_price:.6f})")
            return effective_pct
        else:
            # Fallback sur pourcentage fixe
            logger.debug(f"🧮 Stop fixe: {self.stop_pct:.2f}% (pas d'ATR disponible)")
            return self.stop_pct

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
        
        if self.side == Side.BUY:
            # Nouveau record à la hausse ?
            if price > self.max_price:
                self.max_price = price
                new_stop = self._calc_stop(self.max_price)
                
                # Le stop ne peut que monter (jamais redescendre)
                if new_stop > self.stop_price:
                    self.stop_price = new_stop
                    logger.debug(f"📈 Stop BUY mis à jour: {old_stop:.6f} → {self.stop_price:.6f} "
                               f"(nouveau max: {self.max_price:.6f})")
            
            # Stop déclenché ?
            stop_hit = price <= self.stop_price
            
        else:  # SELL
            # Nouveau record à la baisse ?
            if price < self.min_price:
                self.min_price = price
                new_stop = self._calc_stop(self.min_price)
                
                # Le stop ne peut que descendre (pour protéger plus de profit) pour SELL
                if new_stop < self.stop_price:
                    self.stop_price = new_stop
                    logger.debug(f"📈 Stop SELL mis à jour: {old_stop:.6f} → {self.stop_price:.6f} "
                               f"(nouveau min: {self.min_price:.6f})")
            
            # Stop déclenché ?
            stop_hit = price >= self.stop_price
        
        if stop_hit:
            profit_pct = self._calculate_profit_pct(price)
            logger.info(f"🔴 STOP DÉCLENCHÉ ! Prix: {price:.6f}, Stop: {self.stop_price:.6f}, "
                       f"Profit: {profit_pct:+.2f}%")
        
        return stop_hit
    
    def update_atr(self, atr_value: float):
        """
        Met à jour la valeur ATR pour les calculs adaptatifs.
        
        Args:
            atr_value: Nouvelle valeur ATR (Average True Range)
        """
        old_atr = self.current_atr
        self.current_atr = atr_value
        
        if old_atr != atr_value:
            logger.debug(f"📊 ATR mis à jour: {old_atr} → {atr_value:.6f}")

    def _calculate_profit_pct(self, exit_price: float) -> float:
        """Calcule le pourcentage de profit/perte."""
        if self.side == Side.BUY:
            return ((exit_price - self.entry_price) / self.entry_price) * 100
        else:  # SELL
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
            'extreme_price': self.max_price if self.side == Side.BUY else self.min_price,
            'current_atr': self.current_atr,
            'atr_multiplier': self.atr_multiplier,
            'min_stop_pct': self.min_stop_pct,
            'is_atr_based': self.current_atr is not None
        }

    def __str__(self) -> str:
        """Représentation string du trailing stop."""
        extreme = self.max_price if self.side == Side.BUY else self.min_price
        return (f"TrailingStop({self.side.name}, entry={self.entry_price:.6f}, "
               f"stop={self.stop_price:.6f}, extreme={extreme:.6f})")


# Exemple d'utilisation pour tests
if __name__ == "__main__":
    # Test BUY
    ts = TrailingStop(Side.BUY, entry_price=100.0, stop_pct=3.0)
    prices = [101, 104, 107, 105, 103, 102]  # flux de prix fictif
    
    print("=== Test BUY ===")
    for p in prices:
        hit = ts.update(p)
        status = ts.get_status()
        profit = ts.get_profit_if_exit_now(p)
        print(f"Prix {p:6.6f} | Stop {status['stop_price']:6.6f} | "
              f"Profit {profit:+5.6f}% | {'EXIT!' if hit else 'Hold'}")
        if hit:
            break
    
    print("\n=== Test SELL ===")
    # Test SELL
    ts_SELL = TrailingStop(Side.SELL, entry_price=100.0, stop_pct=3.0)
    prices_SELL = [99, 96, 93, 95, 97, 98]
    
    for p in prices_SELL:
        hit = ts_SELL.update(p)
        status = ts_SELL.get_status()
        profit = ts_SELL.get_profit_if_exit_now(p)
        print(f"Prix {p:6.6f} | Stop {status['stop_price']:6.6f} | "
              f"Profit {profit:+5.1f}% | {'EXIT!' if hit else 'Hold'}")
        if hit:
            break