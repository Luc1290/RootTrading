import threading
import logging
from shared.src.redis_client import RedisClient
from shared.src.config import SYMBOLS

logger = logging.getLogger(__name__)

def start_redis_subscriptions():
    """
    Démarre les souscriptions Redis pour le monitoring uniquement.
    Le portfolio ne modifie JAMAIS les montants directement - c'est le coordinator
    qui utilise l'API du portfolio pour gérer les allocations.
    """
    threading.Thread(target=_subscribe_market_data, daemon=True).start()
    threading.Thread(target=_subscribe_cycle_created, daemon=True).start()
    threading.Thread(target=_subscribe_cycle_completed, daemon=True).start()
    threading.Thread(target=_subscribe_cycle_canceled, daemon=True).start()
    threading.Thread(target=_subscribe_cycle_failed, daemon=True).start()

def _subscribe_market_data():
    redis = RedisClient()
    channels = [f"roottrading:market:data:{symbol.lower()}" for symbol in SYMBOLS]
    redis.subscribe(channels, _handle_market_data)
    logger.info(f"✅ Abonné à {len(channels)} channels Redis pour le monitoring.")

def _handle_market_data(channel, data):
    # Monitoring uniquement - pas de modification
    logger.debug(f"📩 Market data reçu: {channel} -> {data}")

def _subscribe_cycle_created():
    redis = RedisClient()
    redis.subscribe("roottrading:cycle:created", _handle_cycle_created)
    logger.info("✅ Abonné au canal Redis des cycles créés (monitoring uniquement).")

def _handle_cycle_created(channel, data):
    """
    Log uniquement - le coordinator gère l'allocation via l'API du portfolio
    """
    try:
        cycle_id = data.get("cycle_id")
        pocket = data.get("pocket", "active")
        symbol = data.get("symbol", "N/A")
        quantity = float(data.get("quantity", 0.0))
        entry_price = float(data.get("entry_price", 0.0))
        
        logger.info(f"📊 [MONITORING] Cycle créé: {cycle_id} | Poche: {pocket} | "
                   f"Symbol: {symbol} | Qty: {quantity} | Prix: {entry_price}")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du monitoring de cycle_created: {str(e)}")

def _subscribe_cycle_completed():
    redis = RedisClient()
    redis.subscribe("roottrading:cycle:completed", _handle_cycle_completed)
    logger.info("✅ Abonné au canal Redis des cycles complétés (monitoring uniquement).")

def _handle_cycle_completed(channel, data):
    """
    Log uniquement - le coordinator gère le remboursement via l'API du portfolio
    """
    try:
        cycle_id = data.get("cycle_id")
        pocket = data.get("pocket", "active")
        symbol = data.get("symbol", "N/A")
        entry_price = float(data.get("entry_price", 0.0))
        exit_price = float(data.get("exit_price", 0.0))
        quantity = float(data.get("quantity", 0.0))
        pnl = float(data.get("pnl", 0.0))
        
        logger.info(f"💰 [MONITORING] Cycle complété: {cycle_id} | Poche: {pocket} | "
                   f"Symbol: {symbol} | PnL: {pnl:.2f} | "
                   f"Entry: {entry_price} -> Exit: {exit_price}")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du monitoring de cycle_completed: {str(e)}")

def _subscribe_cycle_canceled():
    redis = RedisClient()
    redis.subscribe("roottrading:cycle:canceled", _handle_cycle_canceled)
    logger.info("✅ Abonné au canal Redis des cycles annulés (monitoring uniquement).")

def _handle_cycle_canceled(channel, data):
    """
    Log uniquement - le coordinator gère le remboursement via l'API du portfolio
    """
    try:
        cycle_id = data.get("cycle_id")
        pocket = data.get("pocket", "active")
        symbol = data.get("symbol", "N/A")
        reason = data.get("reason", "Non spécifié")
        
        logger.info(f"🚫 [MONITORING] Cycle annulé: {cycle_id} | Poche: {pocket} | "
                   f"Symbol: {symbol} | Raison: {reason}")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du monitoring de cycle_canceled: {str(e)}")

def _subscribe_cycle_failed():
    redis = RedisClient()
    redis.subscribe("roottrading:cycle:failed", _handle_cycle_failed)
    logger.info("✅ Abonné au canal Redis des cycles échoués (monitoring uniquement).")

def _handle_cycle_failed(channel, data):
    """
    Log uniquement - le coordinator gère le remboursement via l'API du portfolio
    """
    try:
        cycle_id = data.get("cycle_id")
        pocket = data.get("pocket", "active")
        symbol = data.get("symbol", "N/A")
        error = data.get("error", "Non spécifié")
        
        logger.info(f"❌ [MONITORING] Cycle échoué: {cycle_id} | Poche: {pocket} | "
                   f"Symbol: {symbol} | Erreur: {error}")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du monitoring de cycle_failed: {str(e)}")