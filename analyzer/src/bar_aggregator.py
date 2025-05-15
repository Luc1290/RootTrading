"""
Agrégateur de ticks → bougies « 1 minute ».
Si le tick reçu est déjà une bougie fermée (`is_closed` True),
on le relaie tel quel ; sinon on cumule jusqu’à la fin de la minute.
"""
import math
from typing import Dict, Optional

class BarAggregator:
    def __init__(self, interval_ms: int = 60_000):
        self.interval_ms = interval_ms
        self.current: Optional[Dict] = None          # bougie en cours

    def _bucket_start(self, start_time_ms: int) -> int:
        """Arrondit un timestamp (ms) au début de son intervalle."""
        return (start_time_ms // self.interval_ms) * self.interval_ms

    def add(self, tick: Dict) -> Optional[Dict]:
        """
        Ajoute un tick ou une bougie partielle.
        Retourne une bougie fermée dès qu’elle l’est, sinon None.
        """
        # Si le tick est déjà marqué fermé, on le renvoie immédiatement
        if tick.get("is_closed", False):
            return tick

        ts = tick["start_time"]
        bucket = self._bucket_start(ts)

        if self.current is None or bucket != self.current["start_time"]:
            # Nouveau bucket : terminer l’ancien si présent
            closed = self._close_current(bucket)
            # Initialiser une nouvelle bougie
            self.current = {
                "symbol": tick["symbol"],
                "start_time": bucket,
                "open": tick["open"],
                "high": tick["high"],
                "low": tick["low"],
                "close": tick["close"],
                "volume": tick.get("volume", 0.0),
                "interval": "1m",
                "is_closed": False,
            }
            return closed    # peut être None

        # Mise à jour de la bougie courante
        self.current["high"] = max(self.current["high"], tick["high"])
        self.current["low"]  = min(self.current["low"],  tick["low"])
        self.current["close"] = tick["close"]
        self.current["volume"] += tick.get("volume", 0.0)

        # Si le tick termine la minute
        if tick.get("is_closed", False):
            return self._close_current(bucket)

        return None

    def _close_current(self, bucket: int) -> Optional[Dict]:
        """Ferme la bougie en cours (si elle existe) et la renvoie."""
        if self.current and self.current["start_time"] == bucket and not self.current["is_closed"]:
            self.current["is_closed"] = True
            closed_bar = self.current
            self.current = None
            return closed_bar
        return None
