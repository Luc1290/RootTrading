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
DEFAULT_SYMBOL = "BTCUSDC"
SYMBOLS = ["BTCUSDC", "ETHUSDC", "SOLUSDC", "XRPUSDC", "BNBUSDC", "ADAUSDC", "AVAXUSDC", "DOGEUSDC", "LINKUSDC", "AAVEUSDC", "SUIUSDC", "PEPEUSDC", "BONKUSDC", "HYPERUSDC", "LDOUSDC"]
INTERVAL = "1m"
VALIDATION_INTERVAL = "15m"
SCALPING_INTERVALS = ["1m", "5m", "15m", "1h", "4h"]
TRADING_MODE = os.getenv("TRADING_MODE", "demo")  # 'demo' ou 'live' - reste dans .env car peut changer

# Allocation dynamique par pourcentage de capital
ALLOCATION_WEAK_PCT = 15.0
ALLOCATION_MODERATE_PCT = 25.0
ALLOCATION_STRONG_PCT = 35.0
ALLOCATION_VERY_STRONG_PCT = 50.0

# Limites d'allocation par devise
MIN_TRADE_USDC = 15.0
MAX_TRADE_USDC = 200.0
MIN_TRADE_SOL = 0.15
MAX_TRADE_SOL = 1.5
MIN_TRADE_XRP = 10.0
MAX_TRADE_XRP = 100.0

# Quantités individuelles par symbole (legacy)
TRADE_QUANTITY_SOLUSDC = 0.17
TRADE_QUANTITY_XRPUSDC = 11.0
TRADE_QUANTITY_SOL = 0.17
TRADE_QUANTITY_XRP = 11.0

# Dictionnaire centralisé
TRADE_QUANTITIES = {
    "SOLUSDC": TRADE_QUANTITY_SOLUSDC,
    "XRPUSDC": TRADE_QUANTITY_XRPUSDC,
    "SOL": TRADE_QUANTITY_SOL,
    "XRP": TRADE_QUANTITY_XRP,    
}

# Valeur par défaut
TRADE_QUANTITY = TRADE_QUANTITIES.get(DEFAULT_SYMBOL, TRADE_QUANTITY_SOLUSDC)


# Paramètres pour TechnicalIndicators (nécessaires pour compatibilité)
# Les stratégies Pro gèrent leurs propres paramètres avancés
STRATEGY_PARAMS = {
    # Paramètres de base pour TechnicalIndicators
    "rsi": {
        "window": 14,
        "overbought": 70,
        "oversold": 30,
    },
    "ema_cross": {
        "fast_window": 12,
        "slow_window": 26,
    },
    "bollinger": {
        "window": 20,
        "num_std": 2.0,
    },
    "macd": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
        "histogram_threshold": 0.0005,
    },
    # Paramètres globaux pour toutes les stratégies Pro
    "global": {
        "confluence_threshold": 55.0,
        "min_adx_trend": 25.0,
        "min_volume_ratio": 1.2,
        "context_score_threshold": 50.0
    }
}

# ADX Hybrid Configuration
ADX_SMOOTHING_PERIOD = 3
ADX_HYBRID_MODE = True

# Regime Detection Thresholds (Optimized for crypto volatility)
ADX_NO_TREND_THRESHOLD = 18.0
ADX_WEAK_TREND_THRESHOLD = 23.0
ADX_TREND_THRESHOLD = 32.0
ADX_STRONG_TREND_THRESHOLD = 42.0

# Signal Aggregator Settings - Optimisés pour stratégies Pro
SIGNAL_COOLDOWN_MINUTES = 3
VOTE_THRESHOLD = 0.40  # Augmenté pour stratégies Pro plus sélectives
CONFIDENCE_THRESHOLD = 0.70  # Augmenté pour qualité supérieure

# Configuration des canaux Redis
CHANNEL_PREFIX = "roottrading"

# Chemins de Kafka Topics
KAFKA_TOPIC_MARKET_DATA = "market.data"
KAFKA_TOPIC_SIGNALS = "signals"
KAFKA_TOPIC_ORDERS = "orders"
KAFKA_TOPIC_EXECUTIONS = "executions"
KAFKA_TOPIC_ERRORS = "errors"

# Ports des services
GATEWAY_PORT = 5010
ANALYZER_PORT = 5012
SIGNAL_AGGREGATOR_PORT = 5013
TRADER_PORT = 5002
PORTFOLIO_PORT = 8000
VISUALIZATION_PORT = 5009
COORDINATOR_PORT = 5003
DISPATCHER_PORT = 5004

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