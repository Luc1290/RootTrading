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
SYMBOLS = ["BTCUSDC", "ETHUSDC", "SOLUSDC", "XRPUSDC", "ADAUSDC", "AVAXUSDC", "DOGEUSDC", "LINKUSDC", "AAVEUSDC", "SUIUSDC", "PEPEUSDC", "BONKUSDC", "LDOUSDC"]
INTERVAL = "1m"
VALIDATION_INTERVAL = "5m"
SCALPING_INTERVALS = ["1m", "3m", "5m", "15m"]
TRADING_MODE = os.getenv("TRADING_MODE", "demo")  # 'demo' ou 'live' - reste dans .env car peut changer

# Allocation dynamique maintenant gérée par le Coordinator
# (Anciens hardcoded values supprimés - allocation dynamique selon capital)

# Quantités legacy supprimées - utilisation allocation dynamique


# Paramètres pour TechnicalIndicators (nécessaires pour compatibilité)
# Les stratégies Pro gèrent leurs propres paramètres avancés
STRATEGY_PARAMS: Dict[str, Dict[str, Any]] = {
    # Paramètres de base pour TechnicalIndicators
    "rsi": {
        "window": 14,
        "overbought": 65,  # STANDARDISÉ: Cohérent avec toutes les stratégies Pro
        "oversold": 40,   # STANDARDISÉ: Cohérent avec toutes les stratégies Pro
    },
    "ema_cross": {
        "fast_window": 7,   # MIGRATION BINANCE: 12 → 7 (plus réactif)
        "slow_window": 26,  # Inchangé (déjà optimal)
        "long_window": 99,  # NOUVEAU: Ajout EMA long terme (remplacement de 50)
    },
    "bollinger": {
        "window": 20,
        "num_std": 2.0,
    },
    "macd": {
        "fast_period": 7,  # Aligné avec EMA Binance (7/26/99)
        "slow_period": 26,
        "signal_period": 9,
        "histogram_threshold": 0.00005,  # STANDARDISÉ: Momentum faible minimum (sera MACD_HISTOGRAM_WEAK)
    },
    # Paramètres globaux pour toutes les stratégies Pro - STANDARDISÉS
    "global": {
        "confluence_threshold": 55.0,
        "min_adx_trend": 25.0,         # STANDARDISÉ: ADX_TREND_THRESHOLD
        "min_volume_ratio": 1.0,       # STANDARDISÉ: Volume acceptable minimum
        "context_score_threshold": 50.0
    }
}

# ADX Hybrid Configuration
ADX_SMOOTHING_PERIOD = 3
ADX_HYBRID_MODE = True

# Regime Detection Thresholds - Standardisation ADX (Analyse technique standard)
ADX_NO_TREND_THRESHOLD = 15.0        # < 15 : Pas de tendance (range/consolidation)
ADX_WEAK_TREND_THRESHOLD = 20.0      # >= 15 à <20 : Tendance faible  
ADX_MODERATE_TREND_THRESHOLD = 25.0  # >= 20 à <25 : Tendance modérée
ADX_TREND_THRESHOLD = 25.0           # >= 25 : Forte tendance (alias pour compatibilité)
ADX_STRONG_TREND_THRESHOLD = 35.0    # >= 35 : Très forte tendance

# MACD Histogram Thresholds - STANDARDISÉS pour tous les modules
MACD_HISTOGRAM_VERY_STRONG = 0.001   # >= 0.001 : Momentum très fort (positif/négatif)
MACD_HISTOGRAM_STRONG = 0.0005       # >= 0.0005 : Momentum fort
MACD_HISTOGRAM_MODERATE = 0.0001     # >= 0.0001 : Momentum modéré
MACD_HISTOGRAM_WEAK = 0.00005        # >= 0.00005 : Momentum faible
MACD_HISTOGRAM_NEUTRAL = 0.00005     # < 0.00005 : Momentum neutre/insignifiant

# ATR Multipliers - Standardisation pour la volatilité et risk management
ATR_MULTIPLIER_EXTREME = 3.0       # Volatilité extrême (crash protection)
ATR_MULTIPLIER_VERY_HIGH = 2.5     # Volatilité très élevée (range trading)
ATR_MULTIPLIER_HIGH = 2.0          # Volatilité élevée (forte tendance)
ATR_MULTIPLIER_MODERATE = 1.5      # Volatilité modérée (standard)
ATR_MULTIPLIER_LOW = 1.0           # Volatilité faible (ETH, majors)
ATR_MULTIPLIER_VERY_LOW = 0.8      # Volatilité très faible (BTC)
ATR_MULTIPLIER_ALTCOINS = 1.2      # Volatilité altcoins (légèrement plus élevée)

# ATR Threshold Values - Seuils standardisés pour les stratégies (en % du prix)
ATR_THRESHOLD_EXTREME = 0.008       # 0.8% - Volatilité extrême
ATR_THRESHOLD_VERY_HIGH = 0.006     # 0.6% - Volatilité très élevée
ATR_THRESHOLD_HIGH = 0.005          # 0.5% - Volatilité élevée
ATR_THRESHOLD_MODERATE = 0.003      # 0.3% - Volatilité modérée
ATR_THRESHOLD_LOW = 0.002           # 0.2% - Volatilité faible
ATR_THRESHOLD_VERY_LOW = 0.001      # 0.1% - Volatilité très faible

# ATR Minimum Values - Valeurs minimales par type d'actif (en % du prix)
ATR_MIN_BTC = 0.0012                # 0.12% - BTC volatilité minimale
ATR_MIN_ETH = 0.0015                # 0.15% - ETH volatilité minimale
ATR_MIN_ALTCOINS = 0.0018           # 0.18% - Altcoins volatilité minimale

# Volume Context Configuration - Seuils adaptatifs selon le contexte market
VOLUME_CONTEXTS = {
    "deep_oversold": {
        "min_ratio": 0.6,
        "ideal_ratio": 1.2,
        "rsi_threshold": 30,
        "cci_threshold": -200,
        "description": "RSI < 30 et CCI < -200 - Oversold extrême, volume plus tolérant"
    },
    "moderate_oversold": {
        "min_ratio": 0.8,
        "ideal_ratio": 1.0,
        "rsi_threshold": 40,
        "cci_threshold": -150,
        "description": "RSI < 40 et CCI < -150 - Oversold modéré, légèrement tolérant"
    },
    "oversold_bounce": {
        "min_ratio": 0.8,
        "ideal_ratio": 1.2,
        "rsi_threshold": 40,
        "cci_threshold": -100,
        "description": "Rebond technique depuis oversold, volume réduit acceptable"
    },
    "trend_continuation": {
        "min_ratio": 1.0,
        "ideal_ratio": 1.5,
        "description": "Continuation de tendance établie, volume standard"
    },
    "consolidation_break": {
        "min_ratio": 1.3,
        "ideal_ratio": 1.8,
        "description": "Cassure de consolidation, volume confirmation requis"
    },
    "breakout": {
        "min_ratio": 1.5,
        "ideal_ratio": 2.0,
        "description": "Breakout de résistance/support, volume élevé obligatoire"
    },
    "pump_start": {
        "min_ratio": 1.2,
        "ideal_ratio": 2.5,
        "description": "Début de pump détecté, volume confirmation importante"
    },
    "low_volatility": {
        "min_ratio": 0.9,
        "ideal_ratio": 1.3,
        "adx_threshold": 20,
        "description": "Marché calme (ADX < 20), volume réduit acceptable"
    },
    "high_volatility": {
        "min_ratio": 1.1,
        "ideal_ratio": 1.6,
        "adx_threshold": 35,
        "description": "Marché volatile (ADX > 35), volume légèrement plus élevé"
    }
}

# Volume Progressive Pattern Detection
VOLUME_BUILDUP_CONFIG = {
    "lookback_periods": 5,           # Nombre de bougies à analyser
    "min_increase_ratio": 1.1,       # Augmentation minimale entre bougies (10%)
    "progressive_threshold": 0.7,    # 70% des bougies doivent suivre la progression
    "spike_detection_ratio": 2.0,    # Ratio pour détecter un spike volume
    "accumulation_threshold": 1.5    # Seuil d'accumulation progressive
}

# Signal Aggregator Settings - Optimisés pour stratégies Pro
SIGNAL_COOLDOWN_MINUTES = 15  # Cooldown plus long pour éviter sur-trading (15min entre signaux)
VOTE_THRESHOLD = 0.50  # Seuil de vote équilibré - au moins 50% des stratégies doivent s'accorder
CONFIDENCE_THRESHOLD = 0.70  # Seuil de confiance modéré - permet plus de signaux BUY tout en restant sélectif

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