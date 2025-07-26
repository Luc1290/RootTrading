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
SYMBOLS = ["BTCUSDC", "ETHUSDC", "SOLUSDC", "XRPUSDC", "ADAUSDC", "AVAXUSDC", "LINKUSDC", "AAVEUSDC", "SUIUSDC", "LDOUSDC"]
INTERVAL = "1m"
VALIDATION_INTERVAL = "3m"
SCALPING_INTERVALS = ["1m", "3m", "5m", "15m"]
TRADING_MODE = os.getenv("TRADING_MODE", "demo")  # 'demo' ou 'live' - reste dans .env car peut changer

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

def is_live_mode() -> bool:
    """Vérifie si le bot est en mode réel (live) ou démo."""
    return TRADING_MODE.lower() == 'live'

def get_redis_channel(channel_type: str, symbol: Optional[str] = None) -> str:
    """Génère un nom de canal Redis basé sur le type et le symbole."""
    base_channel = f"{CHANNEL_PREFIX}:{channel_type}"
    if symbol:
        return f"{base_channel}:{symbol}"
    return base_channel