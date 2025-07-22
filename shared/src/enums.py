"""
Définition des énumérations partagées entre les microservices.
"""
from enum import Enum

class OrderSide(str, Enum):
    """Type d'ordre: achat ou vente."""
    BUY = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    """Type d'ordre sur l'exchange."""
    LIMIT = "LIMIT"             # Ordre limite
    MARKET = "MARKET"           # Ordre au marché
    STOP_LOSS = "STOP_LOSS"     # Stop loss
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"  # Stop loss limite
    TAKE_PROFIT = "TAKE_PROFIT"  # Take profit
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"  # Take profit limite
    LIMIT_MAKER = "LIMIT_MAKER"  # Ordre limite maker only

class OrderStatus(str, Enum):
    """Statut d'un ordre."""
    NEW = "NEW"                 # Ordre créé et envoyé à Binance
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # Partiellement exécuté
    FILLED = "FILLED"           # Complètement exécuté
    CANCELED = "CANCELED"       # Annulé par l'utilisateur ou le système
    REJECTED = "REJECTED"       # Rejeté par Binance
    EXPIRED = "EXPIRED"         # Expiré selon les paramètres de l'ordre
    PENDING_CANCEL = "PENDING_CANCEL"  # En attente d'annulation

class TradeRole(str, Enum):
    """Rôle dans une transaction: maker ou taker."""
    MAKER = "maker"  # Fournisseur de liquidité
    TAKER = "taker"  # Preneur de liquidité

class CycleStatus(str, Enum):
    """Statut d'un cycle de trading."""
    INITIATING = "initiating"            # Cycle en cours d'initialisation
    WAITING_BUY = "waiting_buy"          # En attente d'achat
    ACTIVE_BUY = "active_buy"            # Achat en cours
    WAITING_SELL = "waiting_sell"        # En attente de vente
    ACTIVE_SELL = "active_sell"          # Vente en cours
    COMPLETED = "completed"              # Cycle terminé avec succès
    CANCELED = "canceled"                # Cycle annulé
    FAILED = "failed"                    # Cycle échoué

class SignalStrength(str, Enum):
    """Force du signal généré par une stratégie."""
    VERY_WEAK = "very_weak"  # Signal très faible
    WEAK = "weak"        # Signal faible
    MODERATE = "moderate"  # Signal modéré
    STRONG = "strong"    # Signal fort
    VERY_STRONG = "very_strong"  # Signal très fort

class TimeFrame(str, Enum):
    """Intervalles de temps pour les données de marché."""
    ONE_MINUTE = "1m"
    THREE_MINUTES = "3m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    TWO_HOURS = "2h"
    FOUR_HOURS = "4h"
    SIX_HOURS = "6h"
    EIGHT_HOURS = "8h"
    TWELVE_HOURS = "12h"
    ONE_DAY = "1d"
    THREE_DAYS = "3d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"

class MarketCondition(str, Enum):
    """Condition générale du marché."""

    CONSOLIDATING = "consolidating"      # Marché en consolidation
    RANGING = "ranging"                  # Marché en range
    STRONG_TREND_DOWN = "downtrend"      # Forte tendance baissière
    STRONG_TREND_UP = "up_trend"         # Forte tendance haussière
    BEARISH = "bearish"                  # Marché baissier
    BULLISH = "bullish"                  # Marché haussier
    SIDEWAYS = "sideways"                # Marché latéral
    HIGH_VOLATILITY = "high_volatility"  # Forte volatilité
    LOW_VOLATILITY = "low_volatility"    # Faible volatilité