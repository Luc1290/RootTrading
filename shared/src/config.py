"""
Configuration centralisée pour tous les services RootTrading.
Charge les variables d'environnement depuis .env et les rend disponibles dans l'application.
"""
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# Binance API
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY", "")

# Redis
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Kafka
KAFKA_BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "roottrading")

# Base de données PostgreSQL
PGUSER = os.getenv("PGUSER", "postgres")
PGPASSWORD = os.getenv("PGPASSWORD", "postgres")
PGDATABASE = os.getenv("PGDATABASE", "trading")
PGHOST = os.getenv("PGHOST", "db")
PGPORT = int(os.getenv("PGPORT", 5432))

# Ajouter des paramètres pour le pool de connexions
DB_MIN_CONNECTIONS = int(os.getenv("DB_MIN_CONNECTIONS", "1"))
DB_MAX_CONNECTIONS = int(os.getenv("DB_MAX_CONNECTIONS", "10"))

# Paramètres de trading
DEFAULT_SYMBOL = os.getenv("SYMBOL", "BTCUSDC")
SYMBOLS = os.getenv("SYMBOLS", "BTCUSDC,ETHUSDC").split(",")
INTERVAL = os.getenv("INTERVAL", "1m")
TRADING_MODE = os.getenv("TRADING_MODE", "demo")  # 'demo' ou 'live'
TRADE_QUANTITY_BTC = float(os.getenv("TRADE_QUANTITY_BTC", 0.001))
TRADE_QUANTITY_ETH = float(os.getenv("TRADE_QUANTITY_ETH", 0.03))
TRADE_QUANTITIES = {
    "BTCUSDC": TRADE_QUANTITY_BTC,
    "ETHUSDC": TRADE_QUANTITY_ETH
}
TRADE_QUANTITY = TRADE_QUANTITY_BTC  # pour compatibilité avec le code existant

# Paramètres des stratégies
STRATEGY_PARAMS = {
    "rsi": {
        "window": int(os.getenv("RSI_WINDOW", 14)),
        "overbought": int(os.getenv("RSI_OVERBOUGHT", 70)),
        "oversold": int(os.getenv("RSI_OVERSOLD", 30)),
    },
    "ema_cross": {
        "short_window": int(os.getenv("SHORT_WINDOW", 5)),
        "long_window": int(os.getenv("LONG_WINDOW", 20)),
    },
    "bollinger": {
        "window": int(os.getenv("BB_WINDOW", 20)),
        "num_std": float(os.getenv("BB_STD", 2.0)),
    },
    "ride_or_react": {
        "thresholds": {
            "1h": float(os.getenv("ROD_1H_THRESHOLD", 0.8)),
            "3h": float(os.getenv("ROD_3H_THRESHOLD", 2.5)),
            "6h": float(os.getenv("ROD_6H_THRESHOLD", 3.6)),
            "12h": float(os.getenv("ROD_12H_THRESHOLD", 5.1)),
            "24h": float(os.getenv("ROD_24H_THRESHOLD", 7.8)),
        }
    }
}

# Configuration des canaux Redis
CHANNEL_PREFIX = os.getenv("CHANNEL_PREFIX", "roottrading")

# Chemins de Kafka Topics
KAFKA_TOPIC_MARKET_DATA = "market.data"
KAFKA_TOPIC_SIGNALS = "signals"
KAFKA_TOPIC_ORDERS = "orders"
KAFKA_TOPIC_EXECUTIONS = "executions"
KAFKA_TOPIC_ERRORS = "errors"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Configuration des poches de trading
POCKET_CONFIG = {
    "active": float(os.getenv("POCKET_ACTIVE_PERCENT", 60)) / 100,  # Poche pour trades actifs
    "buffer": float(os.getenv("POCKET_BUFFER_PERCENT", 30)) / 100,  # Poche tampon
    "safety": float(os.getenv("POCKET_SAFETY_PERCENT", 10)) / 100,  # Poche de sécurité
}

def get_db_url() -> str:
    """Retourne l'URL de connexion à la base de données PostgreSQL."""
    return f"postgresql://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"

def get_strategy_param(strategy_name: str, param_name: str, default: Any = None) -> Any:
        # Validation des paramètres
        if strategy_name is not None and not isinstance(strategy_name, str):
            raise TypeError(f"strategy_name doit être une chaîne, pas {type(strategy_name).__name__}")
        if param_name is not None and not isinstance(param_name, str):
            raise TypeError(f"param_name doit être une chaîne, pas {type(param_name).__name__}")
        """Obtient un paramètre de stratégie spécifique."""
        strategy_config = STRATEGY_PARAMS.get(strategy_name, {})
        return strategy_config.get(param_name, default)

def is_live_mode() -> bool:
    """Vérifie si le bot est en mode réel (live) ou démo."""
    return TRADING_MODE.lower() == 'live'

def get_redis_channel(channel_type: str, symbol: Optional[str] = None) -> str:
        # Validation des paramètres
        if channel_type is not None and not isinstance(channel_type, str):
            raise TypeError(f"channel_type doit être une chaîne, pas {type(channel_type).__name__}")
        if symbol is not None and not isinstance(symbol, str):
            raise TypeError(f"symbol doit être une chaîne, pas {type(symbol).__name__}")
        """Génère un nom de canal Redis basé sur le type et le symbole."""
        base_channel = f"{CHANNEL_PREFIX}:{channel_type}"
        if symbol:
            return f"{base_channel}:{symbol}"
        return base_channel

# Vérifier et ajuster les allocations
pocket_sum = sum(POCKET_CONFIG.values())
if abs(pocket_sum - 1.0) > 0.0001:  # Tolérance pour les erreurs d'arrondi
    print(f"Attention: La somme des allocations de poches est {pocket_sum*100}%, ajustement...")
    # Ajuster la dernière poche pour obtenir exactement 100%
    POCKET_CONFIG["safety"] = 1.0 - (POCKET_CONFIG["active"] + POCKET_CONFIG["buffer"])