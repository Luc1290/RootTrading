"""
Fonctions de sécurité et de protection pour le trader.
Fournit des mécanismes pour éviter les erreurs et problèmes.
"""
import logging
import time
import functools
from typing import Dict, Any, Callable, Optional, TypeVar, List
import traceback
import threading

from shared.src.redis_client import RedisClient

# Configuration du logging
logger = logging.getLogger(__name__)

# Type pour les fonctions décorées
T = TypeVar('T')

def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0,
          exceptions: List[Exception] = [Exception]) -> Callable:
    """
    Décorateur pour retenter une fonction en cas d'échec.
    
    Args:
        max_attempts: Nombre maximum de tentatives
        delay: Délai initial entre les tentatives (en secondes)
        backoff: Facteur multiplicatif pour augmenter le délai
        exceptions: Liste des exceptions à intercepter
        
    Returns:
        Fonction décorée
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except tuple(exceptions) as e:
                    attempt += 1
                    if attempt == max_attempts:
                        logger.error(f"Échec de la fonction {func.__name__} après {attempt} tentatives: {str(e)}")
                        raise
                    
                    logger.warning(f"Échec de la fonction {func.__name__} (tentative {attempt}/{max_attempts}): {str(e)}")
                    logger.warning(f"Nouvelle tentative dans {current_delay:.2f}s")
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
        
        return wrapper
    
    return decorator

def circuit_breaker(failure_threshold: int = 3, reset_timeout: float = 60.0) -> Callable:
    """
    Décorateur implémentant un circuit breaker.
    Arrête les tentatives après un certain nombre d'échecs.
    
    Args:
        failure_threshold: Nombre d'échecs avant d'ouvrir le circuit
        reset_timeout: Délai avant de réinitialiser le circuit (en secondes)
        
    Returns:
        Fonction décorée
    """
    # État du circuit breaker
    state = {
        "failures": 0,
        "open": False,
        "last_failure": 0,
        "lock": threading.RLock()
    }
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with state["lock"]:
                # Vérifier si le circuit est ouvert
                if state["open"]:
                    # Vérifier si le délai de réinitialisation est passé
                    if time.time() - state["last_failure"] > reset_timeout:
                        logger.info(f"Circuit breaker réinitialisé pour {func.__name__}")
                        state["open"] = False
                        state["failures"] = 0
                    else:
                        # Circuit toujours ouvert
                        remaining = reset_timeout - (time.time() - state["last_failure"])
                        logger.warning(f"Circuit breaker ouvert pour {func.__name__}, réinitialisation dans {remaining:.2f}s")
                        raise Exception(f"Circuit breaker ouvert pour {func.__name__}")
            
            try:
                # Exécuter la fonction
                result = func(*args, **kwargs)
                
                # Réinitialiser les échecs en cas de succès
                with state["lock"]:
                    if state["failures"] > 0:
                        logger.info(f"Circuit breaker réinitialisé après succès pour {func.__name__}")
                        state["failures"] = 0
                
                return result
                
            except Exception as e:
                # Incrémenter le compteur d'échecs
                with state["lock"]:
                    state["failures"] += 1
                    state["last_failure"] = time.time()
                    
                    # Ouvrir le circuit si le seuil est atteint
                    if state["failures"] >= failure_threshold:
                        state["open"] = True
                        logger.error(f"Circuit breaker ouvert pour {func.__name__} après {state['failures']} échecs")
                
                # Relancer l'exception
                raise
        
        return wrapper
    
    return decorator

def redis_lock(lock_name: str, timeout: int = 30) -> Callable:
    """
    Décorateur pour implémenter un verrou distribué via Redis.
    Empêche l'exécution simultanée de la fonction par plusieurs instances.
    
    Args:
        lock_name: Nom du verrou Redis
        timeout: Délai d'expiration du verrou (en secondes)
        
    Returns:
        Fonction décorée
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Générer un identifiant unique pour cette instance du verrou
            lock_id = f"{func.__name__}_{time.time()}"
            lock_key = f"roottrading:lock:{lock_name}"
            
            # Client Redis
            redis = RedisClient()
            
            # Essayer d'acquérir le verrou
            acquired = redis.set(lock_key, lock_id, nx=True, expiration=timeout)
            
            if not acquired:
                logger.warning(f"Impossible d'acquérir le verrou Redis pour {func.__name__}")
                raise Exception(f"Verrou Redis déjà acquis pour {func.__name__}")
            
            try:
                # Exécuter la fonction
                return func(*args, **kwargs)
            finally:
                # Libérer le verrou seulement si on est propriétaire
                try:
                    current_value = redis.get(lock_key)
                    if current_value == lock_id:
                        redis.delete(lock_key)
                        logger.debug(f"Verrou Redis libéré pour {func.__name__}")
                except Exception as e:
                    logger.error(f"Erreur lors de la libération du verrou Redis: {str(e)}")
        
        return wrapper
    
    return decorator

def safe_execute(func: Callable, *args, **kwargs) -> Optional[Any]:
    """
    Exécute une fonction en toute sécurité, en capturant les exceptions.
    
    Args:
        func: Fonction à exécuter
        *args: Arguments positionnels
        **kwargs: Arguments nommés
        
    Returns:
        Résultat de la fonction ou None en cas d'erreur
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'exécution de {func.__name__}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def notify_error(message: str, additional_info: Dict[str, Any] = None) -> None:
    """
    Notifie une erreur via Redis.
    
    Args:
        message: Message d'erreur
        additional_info: Informations supplémentaires (optionnel)
    """
    try:
        redis = RedisClient()
        
        data = {
            "type": "error",
            "service": "trader",
            "message": message,
            "timestamp": time.time()
        }
        
        if additional_info:
            data["info"] = additional_info
        
        redis.publish("roottrading:notifications", data)
        logger.debug(f"Notification d'erreur envoyée: {message}")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'envoi de la notification: {str(e)}")

def validate_price(symbol: str, price: float, max_deviation: float = 0.1) -> bool:
    """
    Valide un prix par rapport au prix du marché.
    
    Args:
        symbol: Symbole concerné
        price: Prix à valider
        max_deviation: Déviation maximale autorisée (en pourcentage)
        
    Returns:
        True si le prix est valide, False sinon
    """
    if price <= 0:
        logger.warning(f"❌ Prix invalide: {price} <= 0")
        return False
    
    try:
        # Récupérer le prix actuel via Redis
        redis = RedisClient()
        market_data = redis.get(f"roottrading:market:last_price:{symbol.lower()}")
        
        if not market_data:
            logger.warning(f"⚠️ Pas de prix récent pour {symbol}")
            return True  # Par défaut, accepter le prix
        
        market_price = float(market_data)
        
        # Calculer la déviation
        deviation = abs(price - market_price) / market_price
        
        if deviation > max_deviation:
            logger.warning(f"❌ Déviation de prix trop importante pour {symbol}: {deviation:.2%} > {max_deviation:.2%}")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Erreur lors de la validation du prix: {str(e)}")
        return True  # En cas d'erreur, accepter le prix