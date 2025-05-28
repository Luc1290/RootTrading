import threading
import logging
from shared.src.redis_client import RedisClient
from shared.src.config import SYMBOLS

logger = logging.getLogger(__name__)

def start_redis_subscriptions():
    threading.Thread(target=_subscribe_market_data, daemon=True).start()
    threading.Thread(target=_subscribe_cycle_created, daemon=True).start()
    threading.Thread(target=_subscribe_cycle_closed, daemon=True).start()
    threading.Thread(target=_subscribe_cycle_canceled, daemon=True).start()

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
        
        # Utiliser decrbyfloat qui gère correctement l'appel à Redis
        redis.decrbyfloat(pocket_key, quantity)
        redis.sadd(cycles_key, cycle_id)

        logger.info(f"📦 Cycle {cycle_id} ajouté à la poche '{pocket}' ({quantity} décrémentés)")

    except Exception as e:
        logger.error(f"❌ Erreur dans _handle_cycle_created: {str(e)}")

def _subscribe_cycle_closed():
    redis = RedisClient()
    redis.subscribe("roottrading:cycle:closed", _handle_cycle_closed)
    logger.info("✅ Abonné au canal Redis des cycles fermés.")

def _handle_cycle_closed(channel, data):
    try:
        cycle_id = data.get("cycle_id")
        pocket = data.get("pocket", "active")
        entry_price = float(data.get("entry_price", 0.0))
        quantity = float(data.get("quantity", 0.0))
        exit_price = float(data.get("exit_price", 0.0))

        if not cycle_id or quantity <= 0:
            logger.warning(f"⛔ Données cycle fermé invalides: {data}")
            return

        # Calculer le montant à rembourser (valeur de sortie)
        exit_value = exit_price * quantity

        # Mettre à jour la poche - rembourser la valeur de sortie
        pocket_key = f"roottrading:pocket:{pocket}:amount"
        cycles_key = f"roottrading:pocket:{pocket}:cycles"

        redis = RedisClient()
        
        # Incrémenter le montant disponible avec la valeur de sortie
        redis.incrbyfloat(pocket_key, exit_value)
        # Retirer le cycle de la liste des cycles actifs
        redis.srem(cycles_key, cycle_id)

        logger.info(f"💰 Cycle {cycle_id} retiré de la poche '{pocket}' (+{exit_value:.2f} remboursés)")

    except Exception as e:
        logger.error(f"❌ Erreur dans _handle_cycle_closed: {str(e)}")

def _subscribe_cycle_canceled():
    redis = RedisClient()
    redis.subscribe("roottrading:cycle:canceled", _handle_cycle_canceled)
    logger.info("✅ Abonné au canal Redis des cycles annulés.")

def _handle_cycle_canceled(channel, data):
    try:
        cycle_id = data.get("cycle_id")
        pocket = data.get("pocket", "active")
        entry_price = float(data.get("entry_price", 0.0))
        quantity = float(data.get("quantity", 0.0))

        if not cycle_id or quantity <= 0:
            logger.warning(f"⛔ Données cycle annulé invalides: {data}")
            return

        # Calculer le montant à rembourser (valeur d'entrée car pas de sortie)
        entry_value = entry_price * quantity

        # Mettre à jour la poche - rembourser la valeur d'entrée
        pocket_key = f"roottrading:pocket:{pocket}:amount"
        cycles_key = f"roottrading:pocket:{pocket}:cycles"

        redis = RedisClient()
        
        # Incrémenter le montant disponible avec la valeur d'entrée
        redis.incrbyfloat(pocket_key, entry_value)
        # Retirer le cycle de la liste des cycles actifs
        redis.srem(cycles_key, cycle_id)

        logger.info(f"🚫 Cycle {cycle_id} annulé et retiré de la poche '{pocket}' (+{entry_value:.2f} remboursés)")

    except Exception as e:
        logger.error(f"❌ Erreur dans _handle_cycle_canceled: {str(e)}")