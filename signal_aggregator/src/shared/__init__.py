"""
Shared utilities package for signal_aggregator
"""

# Import conditionnels pour éviter les erreurs de dépendances
try:
    from .redis_utils import RedisManager, SignalCacheManager
except ImportError as e:
    print(f"Erreur import redis_utils: {e}")
    RedisManager = None
    SignalCacheManager = None

try:
    from .technical_utils import TechnicalCalculators, VolumeAnalyzer, SignalValidators
except ImportError as e:
    print(f"Erreur import technical_utils: {e}")
    TechnicalCalculators = None
    VolumeAnalyzer = None
    SignalValidators = None

try:
    from .db_utils import DatabasePoolManager, DatabaseUtils
except ImportError as e:
    print(f"Erreur import db_utils: {e}")
    DatabasePoolManager = None
    DatabaseUtils = None

__all__ = [
    'RedisManager',
    'SignalCacheManager', 
    'TechnicalCalculators',
    'VolumeAnalyzer',
    'SignalValidators',
    'DatabasePoolManager',
    'DatabaseUtils'
]