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
DB_MAX_CONNECTIONS = int(os.getenv("DB_MAX_CONNECTIONS", "50"))

def get_db_config() -> Dict[str, Any]:
    """Retourne la configuration de la base de données."""
    return {
        'host': PGHOST,
        'port': PGPORT,
        'database': PGDATABASE,
        'user': PGUSER,
        'password': PGPASSWORD
    }

# Paramètres de trading
DEFAULT_SYMBOL = os.getenv("SYMBOL", "SOLUSDC")
SYMBOLS = os.getenv("SYMBOLS", "SOLUSDC,XRPUSDC").split(",")
INTERVAL = os.getenv("INTERVAL", "1m")
VALIDATION_INTERVAL = os.getenv("VALIDATION_INTERVAL", "5m")  # Changé pour scalping
SCALPING_INTERVALS = os.getenv("SCALPING_INTERVALS", "1m,3m,5m").split(",")  # Multi-timeframes
TRADING_MODE = os.getenv("TRADING_MODE", "demo")  # 'demo' ou 'live'

# Quantités individuelles par symbole (doivent correspondre aux SYMBOLS)
TRADE_QUANTITY_SOLUSDC = float(os.getenv("TRADE_QUANTITY_SOLUSDC", 0.17))  # ~25$ par trade
TRADE_QUANTITY_XRPUSDC = float(os.getenv("TRADE_QUANTITY_XRPUSDC", 11.0))   # ~24$ par trade
TRADE_QUANTITY_SOL = float(os.getenv("TRADE_QUANTITY_SOL", 0.17))   # Pour compatibilité
TRADE_QUANTITY_XRP = float(os.getenv("TRADE_QUANTITY_XRP", 11.0))   # Pour compatibilité

# Dictionnaire centralisé
TRADE_QUANTITIES = {
    "SOLUSDC": TRADE_QUANTITY_SOLUSDC,
    "XRPUSDC": TRADE_QUANTITY_XRPUSDC,
    "SOL": TRADE_QUANTITY_SOL,
    "XRP": TRADE_QUANTITY_XRP,    
}

# Valeur par défaut utilisée par compatibilité
TRADE_QUANTITY = TRADE_QUANTITIES.get(DEFAULT_SYMBOL, TRADE_QUANTITY_SOLUSDC)


# Paramètres des stratégies
STRATEGY_PARAMS = {
    "rsi": {
        "window": int(os.getenv("RSI_WINDOW", 14)),
        "overbought": int(os.getenv("RSI_OVERBOUGHT", 70)),
        "oversold": int(os.getenv("RSI_OVERSOLD", 30)),
    },
    "ema_cross": {
        "SELL_window": int(os.getenv("SELL_WINDOW", 5)),
        "BUY_window": int(os.getenv("BUY_WINDOW", 20)),
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
    },
    "macd": {
        "fast_period": int(os.getenv("MACD_FAST_PERIOD", 12)),
        "slow_period": int(os.getenv("MACD_SLOW_PERIOD", 26)),
        "signal_period": int(os.getenv("MACD_SIGNAL_PERIOD", 9)),
        "histogram_threshold": float(os.getenv("MACD_HISTOGRAM_THRESHOLD", 0.001)),
    }
}

# ADX Hybrid Configuration
ADX_SMOOTHING_PERIOD = int(os.getenv("ADX_SMOOTHING_PERIOD", 3))
ADX_HYBRID_MODE = os.getenv("ADX_HYBRID_MODE", "true").lower() == "true"

# Regime Detection Thresholds (Optimized for crypto volatility)
ADX_NO_TREND_THRESHOLD = float(os.getenv("ADX_NO_TREND_THRESHOLD", 18))
ADX_WEAK_TREND_THRESHOLD = float(os.getenv("ADX_WEAK_TREND_THRESHOLD", 23))
ADX_TREND_THRESHOLD = float(os.getenv("ADX_TREND_THRESHOLD", 32))
ADX_STRONG_TREND_THRESHOLD = float(os.getenv("ADX_STRONG_TREND_THRESHOLD", 42))

# Signal Aggregator Hybrid Settings
SIGNAL_COOLDOWN_MINUTES = int(os.getenv("SIGNAL_COOLDOWN_MINUTES", 3))
VOTE_THRESHOLD = float(os.getenv("VOTE_THRESHOLD", 0.35))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.60))

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

def get_db_url() -> str:
    """Retourne l'URL de connexion à la base de données PostgreSQL."""
    return f"postgresql://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"

def get_strategy_param(strategy_name: str, param_name: str, default: Any = None) -> Any:
    """Obtient un paramètre de stratégie spécifique."""
    strategy_config = STRATEGY_PARAMS.get(strategy_name, {})
    return strategy_config.get(param_name, default)

def is_live_mode() -> bool:
    """Vérifie si le bot est en mode réel (live) ou démo."""
    return TRADING_MODE.lower() == 'live'

def get_redis_channel(channel_type: str, symbol: Optional[str] = None) -> str:
    """Génère un nom de canal Redis basé sur le type et le symbole."""
    base_channel = f"{CHANNEL_PREFIX}:{channel_type}"
    if symbol:
        return f"{base_channel}:{symbol}"
    return base_channel