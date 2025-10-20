"""
Schémas partagés pour la validation des données entre microservices.
Utilise Pydantic pour définir et valider la structure des messages.
"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .enums import CycleStatus, OrderSide, OrderStatus, SignalStrength, TradeRole


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
    timestamp: datetime | None = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def set_timestamp(cls, v, info):
        """Si timestamp n'est pas fourni, le calculer à partir de start_time."""
        if v is None and info.data.get("start_time"):
            return datetime.fromtimestamp(info.data["start_time"] / 1000, tz=timezone.utc)
        return v


class StrategySignal(BaseModel):
    """Signal généré par une stratégie de trading."""

    strategy: str
    symbol: str
    side: OrderSide
    timestamp: datetime
    price: float
    confidence: float | None = None
    strength: SignalStrength | None = None
    metadata: dict[str, Any] | None = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class TradeOrder(BaseModel):
    """Ordre de trading à exécuter sur Binance."""

    symbol: str
    side: OrderSide
    quantity: float
    price: float | None = None  # None pour un ordre au marché
    client_order_id: str | None = None
    strategy: str | None = None
    stop_price: float | None = None
    take_profit: float | None = None
    trailing_delta: float | None = None
    leverage: int | None = Field(1, ge=1, le=10)  # Levier (1-10x)
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
    fee: float | None = None
    fee_asset: str | None = None
    role: TradeRole | None = None
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
    side: (
        # Direction du cycle: BUY (position longue) ou SELL (position courte)
        OrderSide
    )
    entry_order_id: str | None = None
    exit_order_id: str | None = None
    entry_price: float | None = None
    exit_price: float | None = None
    quantity: float | None = None
    stop_price: float | None = None
    trailing_delta: float | None = None
    min_price: float | None = None
    max_price: float | None = None
    profit_loss: float | None = None
    profit_loss_percent: float | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    confirmed: bool = False
    demo: bool = False
    metadata: dict[str, Any] | None = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class AssetBalance(BaseModel):
    """Solde d'un actif dans un portefeuille."""

    asset: str
    free: float
    locked: float
    total: float
    value_usdc: float | None = None


class PortfolioSummary(BaseModel):
    """Résumé du portefeuille complet."""

    balances: list[AssetBalance]
    total_value: float
    performance_24h: float | None = None
    performance_7d: float | None = None
    active_trades: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorMessage(BaseModel):
    """Message d'erreur standardisé."""

    service: str
    error_type: str
    message: str
    details: dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StrategyConfig(BaseModel):
    """Configuration d'une stratégie de trading."""

    name: str
    mode: str
    params: dict[str, Any]
    symbols: list[str]
    max_simultaneous_trades: int = 3
    enabled: bool = True


class LogMessage(BaseModel):
    """Message de log standardisé."""

    service: str
    level: str
    message: str
    data: dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
