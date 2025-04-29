"""
Schémas partagés pour la validation des données entre microservices.
Utilise Pydantic pour définir et valider la structure des messages.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from pydantic import BaseModel, Field, validator

from .enums import OrderSide, OrderStatus, TradeRole, CycleStatus, SignalStrength, StrategyMode

class MarketData(BaseModel):
    """Données de marché provenant de Binance."""
    symbol: str
    start_time: int
    close_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: Optional[datetime] = None

    @validator('open', 'high', 'low', 'close', 'volume')
    def check_positive_values(cls: List[Any], v: Any) -> Any:
        # Validation des paramètres
        if cls is not None and not isinstance(cls, list):
            raise TypeError(f"cls doit être une liste, pas {type(cls).__name__}")
        """Vérifier que les valeurs sont positives."""
        if v < 0:
            raise ValueError("Les valeurs de prix et volume doivent être positives")
        return v
    
    @validator('timestamp', pre=True, always=True)
    def set_timestamp(cls: List[Any], v: Any, values: List[Any]) -> Any:
        # Validation des paramètres
        if cls is not None and not isinstance(cls, list):
            raise TypeError(f"cls doit être une liste, pas {type(cls).__name__}")
        if values is not None and not isinstance(values, list):
            raise TypeError(f"values doit être une liste, pas {type(values).__name__}")
        """Si timestamp n'est pas fourni, le calculer à partir de start_time."""
        if v is None and 'start_time' in values:
            return datetime.fromtimestamp(values['start_time'] / 1000)
        return v

class StrategySignal(BaseModel):
    """Signal généré par une stratégie de trading."""
    strategy: str
    symbol: str
    side: OrderSide
    timestamp: datetime
    price: float
    confidence: Optional[float] = None
    strength: Optional[SignalStrength] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True

class TradeOrder(BaseModel):
    """Ordre de trading à exécuter sur Binance."""
    symbol: str
    side: OrderSide
    quantity: float
    price: Optional[float] = None  # None pour un ordre au marché
    client_order_id: Optional[str] = None
    strategy: Optional[str] = None
    stop_price: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_delta: Optional[float] = None
    leverage: Optional[int] = Field(1, ge=1, le=10)  # Levier (1-10x)
    demo: bool = False
    
    class Config:
        use_enum_values = True

class TradeExecution(BaseModel):
    """Exécution d'un ordre sur Binance."""
    order_id: str
    symbol: str
    side: OrderSide
    status: OrderStatus
    price: float
    quantity: float
    quote_quantity: float
    fee: Optional[float] = None
    fee_asset: Optional[str] = None
    role: Optional[TradeRole] = None
    timestamp: datetime
    demo: bool = False
    
    class Config:
        use_enum_values = True

class TradeCycle(BaseModel):
    """Cycle complet de trading (entrée + sortie)."""
    id: str
    symbol: str
    strategy: str
    status: CycleStatus
    entry_order_id: Optional[str] = None
    exit_order_id: Optional[str] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    quantity: Optional[float] = None
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    trailing_delta: Optional[float] = None
    profit_loss: Optional[float] = None
    profit_loss_percent: Optional[float] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    pocket: Optional[str] = None
    demo: bool = False
    
    class Config:
        use_enum_values = True

class AssetBalance(BaseModel):
    """Solde d'un actif dans un portefeuille."""
    asset: str
    free: float
    locked: float
    total: float
    value_usdc: Optional[float] = None

class PortfolioSummary(BaseModel):
    """Résumé du portefeuille complet."""
    balances: List[AssetBalance]
    total_value: float
    performance_24h: Optional[float] = None
    performance_7d: Optional[float] = None
    active_trades: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PocketSummary(BaseModel):
    """État d'une poche de trading."""
    pocket_type: str
    allocation_percent: float
    current_value: float
    used_value: float
    available_value: float
    active_cycles: int = 0

class ErrorMessage(BaseModel):
    """Message d'erreur standardisé."""
    service: str
    error_type: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StrategyConfig(BaseModel):
    """Configuration d'une stratégie de trading."""
    name: str
    mode: StrategyMode
    params: Dict[str, Any]
    symbols: List[str]
    max_simultaneous_trades: int = 3
    enabled: bool = True
    
    class Config:
        use_enum_values = True

class LogMessage(BaseModel):
    """Message de log standardisé."""
    service: str
    level: str
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)