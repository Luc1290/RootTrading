import requests
from typing import Dict, Any

class BinanceSymbolConstraints:
    BASE_URL = "https://api.binance.com"
    
    def __init__(self):
        self.constraints: Dict[str, Dict[str, Any]] = {}
        self._load_constraints()

    def _load_constraints(self):
        url = f"{self.BASE_URL}/api/v3/exchangeInfo"
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        for symbol_data in data["symbols"]:
            symbol = symbol_data["symbol"]
            constraints = {}
            for f in symbol_data["filters"]:
                if f["filterType"] == "LOT_SIZE":
                    constraints["stepSize"] = float(f["stepSize"])
                    constraints["minQty"] = float(f["minQty"])
                if f["filterType"] == "MIN_NOTIONAL":
                    constraints["minNotional"] = float(f["minNotional"])
            self.constraints[symbol] = constraints

    def get_step_size(self, symbol: str) -> float:
        if symbol not in self.constraints:
            # Log warning and return a default value
            print(f"Warning: Symbol {symbol} not found in constraints, using default step size")
            return 0.00001
        return self.constraints[symbol].get("stepSize", 0.00001)

    def get_min_qty(self, symbol: str) -> float:
        if symbol not in self.constraints:
            print(f"Warning: Symbol {symbol} not found in constraints, using default min quantity")
            return 0.0001
        return self.constraints[symbol].get("minQty", 0.0001)

    def get_min_notional(self, symbol: str) -> float:
        if symbol not in self.constraints:
            # Log warning and return a default value
            print(f"Warning: Symbol {symbol} not found in constraints, using default min notional")
            return 0.0
        return self.constraints[symbol].get("minNotional", 0.0)

    def truncate_quantity(self, symbol: str, quantity: float) -> float:
        step = self.get_step_size(symbol)
        truncated = (quantity // step) * step
        return float(f"{truncated:.8f}")

    def is_quantity_valid(self, symbol: str, quantity: float) -> bool:
        return quantity >= self.get_min_qty(symbol)

    def is_notional_valid(self, symbol: str, quantity: float, price: float) -> bool:
        notional = quantity * price
        return notional >= self.get_min_notional(symbol)
