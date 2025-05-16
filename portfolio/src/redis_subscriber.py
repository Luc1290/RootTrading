import threading
import logging
from shared.src.redis_client import RedisClient
from shared.src.config import SYMBOLS

logger = logging.getLogger(__name__)

def start_redis_subscriptions():
    threading.Thread(target=_subscribe_market_data, daemon=True).start()
    threading.Thread(target=_subscribe_cycle_created, daemon=True).start()

def _subscribe_market_data():
    redis = RedisClient()
    channels = [f"roottrading:market:data:{symbol.lower()}" for symbol in SYMBOLS]
    redis.subscribe(channels, _handle_market_data)
    logger.info(f"✅ Abonné à {len(channels)} channels Redis.")

def _handle_market_data(channel, data):
    # Tu peux enrichir ça plus tard
    logger.debug(f"📩 Market data reçu: {channel} -> {data}")

def _subscribe_cycle_created():
    redis = RedisClient()
    redis.subscribe("roottrading:cycle:created", _handle_cycle_created)
    logger.info("✅ Abonné au canal Redis des cycles créés.")

def _handle_cycle_created(channel, data):
    try:
        cycle_id = data.get("cycle_id")
        pocket = data.get("pocket", "active")
        quantity = float(data.get("quantity", 0.0))

        if not cycle_id or quantity <= 0:
            logger.warning(f"⛔ Données cycle invalides: {data}")
            return

        # Mettre à jour la poche
        pocket_key = f"roottrading:pocket:{pocket}:amount"
        cycles_key = f"roottrading:pocket:{pocket}:cycles"

        redis = RedisClient()
        
        # Utiliser INCRBYFLOAT au lieu de DECRBYFLOAT (qui n'existe pas)
        redis.incrbyfloat(pocket_key, -quantity)  # Utilise incrbyfloat avec valeur négative
        redis.sadd(cycles_key, cycle_id)

        logger.info(f"📦 Cycle {cycle_id} ajouté à la poche '{pocket}' ({quantity} décrémentés)")

    except Exception as e:
        logger.error(f"❌ Erreur dans _handle_cycle_created: {str(e)}")