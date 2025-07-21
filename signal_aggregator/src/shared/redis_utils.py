"""
Utilitaires Redis partagés pour éviter la duplication de code
"""

import json
import logging
from typing import Any, Optional, Dict, Union, Callable
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RedisManager:
    """Gestionnaire Redis centralisé avec patterns unifiés"""
    
    @staticmethod
    def get_cached_data(redis_client, key: str, deserializer: Optional[Callable] = None) -> Optional[Any]:
        """
        Récupère des données depuis Redis avec désérialisation automatique
        
        Args:
            redis_client: Client Redis
            key: Clé Redis
            deserializer: Fonction de désérialisation personnalisée (optionnel)
            
        Returns:
            Données désérialisées ou None si non trouvé
        """
        try:
            cached = redis_client.get(key)
            
            if cached is None:
                return None
            
            # Désérialisation personnalisée
            if deserializer:
                return deserializer(cached)
            
            # Désérialisation automatique JSON
            if isinstance(cached, str):
                try:
                    return json.loads(cached)
                except json.JSONDecodeError:
                    return cached  # Retourner tel quel si pas JSON
            else:
                return cached
                
        except Exception as e:
            logger.error(f"Erreur récupération cache Redis {key}: {e}")
            return None
    
    @staticmethod
    def set_cached_data(redis_client, key: str, data: Any, expiration: Optional[int] = None, 
                       serializer: Optional[Callable] = None) -> bool:
        """
        Met en cache des données dans Redis avec sérialisation automatique
        
        Args:
            redis_client: Client Redis
            key: Clé Redis
            data: Données à mettre en cache
            expiration: Durée d'expiration en secondes (optionnel)
            serializer: Fonction de sérialisation personnalisée (optionnel)
            
        Returns:
            True si réussi
        """
        try:
            # Sérialisation personnalisée
            if serializer:
                serialized_data = serializer(data)
            else:
                # Sérialisation automatique JSON pour dict/list
                if isinstance(data, (dict, list)):
                    serialized_data = json.dumps(data)
                else:
                    serialized_data = data
            
            if expiration:
                redis_client.set(key, serialized_data, expiration=expiration)
            else:
                redis_client.set(key, serialized_data)
                
            return True
            
        except Exception as e:
            logger.error(f"Erreur mise en cache Redis {key}: {e}")
            return False
    
    @staticmethod
    def get_with_fallback(redis_client, keys: list, deserializer: Optional[Callable] = None) -> Optional[Any]:
        """
        Essaie plusieurs clés Redis en cascade jusqu'à trouver des données
        
        Args:
            redis_client: Client Redis
            keys: Liste des clés à essayer dans l'ordre
            deserializer: Fonction de désérialisation personnalisée (optionnel)
            
        Returns:
            Première valeur trouvée ou None
        """
        for key in keys:
            data = RedisManager.get_cached_data(redis_client, key, deserializer)
            if data is not None:
                return data
        return None
    
    @staticmethod
    def get_market_data_multi_timeframe(redis_client, symbol: str, 
                                      timeframes: list = None) -> Optional[Dict]:
        """
        Récupère les données de marché avec fallback multi-timeframes
        
        Args:
            redis_client: Client Redis
            symbol: Symbole concerné
            timeframes: Liste des timeframes à essayer (défaut: ['1m', '5m', '15m'])
            
        Returns:
            Données de marché ou None si non trouvé
        """
        if timeframes is None:
            timeframes = ['1m', '5m', '15m']
        
        keys = [f"market_data:{symbol}:{tf}" for tf in timeframes]
        return RedisManager.get_with_fallback(redis_client, keys)
    
    @staticmethod
    def cache_with_timestamp(redis_client, key: str, data: Any, expiration: int = 60) -> bool:
        """
        Met en cache des données avec timestamp automatique
        
        Args:
            redis_client: Client Redis
            key: Clé Redis
            data: Données à mettre en cache
            expiration: Durée d'expiration en secondes (défaut: 60s)
            
        Returns:
            True si réussi
        """
        cache_data = {
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(seconds=expiration)).isoformat()
        }
        return RedisManager.set_cached_data(redis_client, key, cache_data, expiration)
    
    @staticmethod
    def get_timestamped_data(redis_client, key: str, max_age_seconds: Optional[int] = None) -> Optional[Any]:
        """
        Récupère des données avec vérification d'âge optionnel
        
        Args:
            redis_client: Client Redis
            key: Clé Redis
            max_age_seconds: Âge maximum acceptable en secondes (optionnel)
            
        Returns:
            Données si valides ou None
        """
        cached = RedisManager.get_cached_data(redis_client, key)
        
        if not cached or not isinstance(cached, dict) or 'data' not in cached:
            return None
        
        # Vérifier l'âge si requis
        if max_age_seconds and 'timestamp' in cached:
            try:
                timestamp = datetime.fromisoformat(cached['timestamp'])
                age = (datetime.now() - timestamp).total_seconds()
                if age > max_age_seconds:
                    return None
            except (ValueError, TypeError):
                return None
        
        return cached['data']
    
    @staticmethod
    def increment_counter(redis_client, key: str, expiration: Optional[int] = None) -> int:
        """
        Incrémente un compteur Redis avec expiration optionnelle
        
        Args:
            redis_client: Client Redis
            key: Clé du compteur
            expiration: Durée d'expiration en secondes (optionnel)
            
        Returns:
            Nouvelle valeur du compteur
        """
        try:
            value = redis_client.incr(key)
            if expiration and value == 1:  # Première création
                redis_client.expire(key, expiration)
            return value
        except Exception as e:
            logger.error(f"Erreur incrémentation compteur {key}: {e}")
            return 0
    
    @staticmethod
    def get_counter(redis_client, key: str) -> int:
        """
        Récupère la valeur d'un compteur Redis
        
        Args:
            redis_client: Client Redis
            key: Clé du compteur
            
        Returns:
            Valeur du compteur (0 si inexistant)
        """
        try:
            value = redis_client.get(key)
            return int(value) if value else 0
        except Exception as e:
            logger.error(f"Erreur récupération compteur {key}: {e}")
            return 0

    @staticmethod
    def publish_json(redis_client, channel: str, data: Any) -> bool:
        """
        Publie des données JSON sur un canal Redis
        
        Args:
            redis_client: Client Redis
            channel: Canal Redis
            data: Données à publier (sérialisées en JSON automatiquement)
            
        Returns:
            True si succès, False sinon
        """
        try:
            # Sérialiser en JSON
            if isinstance(data, (dict, list)):
                json_data = json.dumps(data)
            else:
                json_data = str(data)
            
            redis_client.publish(channel, json_data)
            return True
        except Exception as e:
            logger.error(f"Erreur publish_json Redis sur {channel}: {e}")
            return False


class SignalCacheManager:
    """Gestionnaire de cache spécialisé pour les signaux"""
    
    @staticmethod
    def cache_signal_decision(redis_client, strategy: str, symbol: str, regime: str, 
                            accepted: bool, confidence: float, expiration: int = 300) -> bool:
        """
        Met en cache une décision de signal pour éviter les recalculs
        
        Args:
            redis_client: Client Redis
            strategy: Nom de la stratégie
            symbol: Symbole concerné
            regime: Régime de marché
            accepted: Signal accepté ou rejeté
            confidence: Niveau de confiance
            expiration: Durée de cache en secondes (défaut: 5min)
            
        Returns:
            True si réussi
        """
        key = f"signal_decision:{strategy}:{symbol}:{regime}"
        data = {
            'accepted': accepted,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        return RedisManager.cache_with_timestamp(redis_client, key, data, expiration)
    
    @staticmethod
    def get_cached_signal_decision(redis_client, strategy: str, symbol: str, 
                                 regime: str, max_age_seconds: int = 300) -> Optional[Dict]:
        """
        Récupère une décision de signal en cache
        
        Args:
            redis_client: Client Redis
            strategy: Nom de la stratégie
            symbol: Symbole concerné
            regime: Régime de marché
            max_age_seconds: Âge maximum acceptable (défaut: 5min)
            
        Returns:
            Décision cachée ou None
        """
        key = f"signal_decision:{strategy}:{symbol}:{regime}"
        return RedisManager.get_timestamped_data(redis_client, key, max_age_seconds)
    
    @staticmethod
    def cache_regime_analysis(redis_client, symbol: str, regime: str, metrics: Dict, 
                            expiration: int = 60) -> bool:
        """
        Met en cache une analyse de régime de marché
        
        Args:
            redis_client: Client Redis
            symbol: Symbole concerné
            regime: Régime détecté
            metrics: Métriques d'analyse
            expiration: Durée de cache en secondes (défaut: 1min)
            
        Returns:
            True si réussi
        """
        key = f"regime_analysis:{symbol}"
        data = {
            'regime': regime,
            'metrics': metrics
        }
        return RedisManager.cache_with_timestamp(redis_client, key, data, expiration)
    
    @staticmethod
    def get_cached_regime_analysis(redis_client, symbol: str, 
                                 max_age_seconds: int = 60) -> Optional[Dict]:
        """
        Récupère une analyse de régime en cache
        
        Args:
            redis_client: Client Redis
            symbol: Symbole concerné
            max_age_seconds: Âge maximum acceptable (défaut: 1min)
            
        Returns:
            Analyse cachée ou None
        """
        key = f"regime_analysis:{symbol}"
        return RedisManager.get_timestamped_data(redis_client, key, max_age_seconds)