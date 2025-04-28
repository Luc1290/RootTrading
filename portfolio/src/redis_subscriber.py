import threading
import logging
from shared.src.redis_client import RedisClient
from shared.src.config import SYMBOLS

logger = logging.getLogger(__name__)

def start_redis_subscriptions():
    threading.Thread(target=_subscribe_market_data, daemon=True).start()

def _subscribe_market_data():
    redis = RedisClient()
    channels = [f"roottrading:market:data:{symbol.lower()}" for symbol in SYMBOLS]
    redis.subscribe(channels, _handle_market_data)
    logger.info(f"✅ Abonné à {len(channels)} channels Redis.")

def _handle_market_data(channel, data):
    # Tu peux enrichir ça plus tard
    logger.debug(f"📩 Market data reçu: {channel} -> {data}")
