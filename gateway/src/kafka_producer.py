"""
Module Kafka Producer pour le Gateway.
Convertit les données WebSocket Binance en messages Kafka.
"""
import json
import logging
import time
from typing import Dict, Any, Optional

# Importer les clients partagés
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import KAFKA_TOPIC_MARKET_DATA, KAFKA_BROKER
from shared.src.kafka_client import KafkaClient

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("kafka_producer")

class KafkaProducer:
    """
    Producteur Kafka pour le Gateway.
    Publie les données de marché sur les topics Kafka.
    """
    
    def __init__(self, broker: str = KAFKA_BROKER):
        """
        Initialise le producteur Kafka.
        
        Args:
            broker: Adresse du broker Kafka (host:port)
        """
        self.client = KafkaClient(broker=broker)
        logger.info(f"✅ Producteur Kafka initialisé pour {broker}")
    
    def publish_market_data(self, data: Dict[str, Any], key: Optional[str] = None) -> None:
        """
        Publie les données de marché sur le topic Kafka approprié.
        
        Args:
            data: Données de marché à publier
            key: Clé à utiliser pour le partitionnement (généralement le symbole)
        """
        if not data or 'symbol' not in data:
            logger.error("❌ Données de marché invalides, impossible de publier")
            return
        
        symbol = data['symbol'].lower()
        topic = f"{KAFKA_TOPIC_MARKET_DATA}.{symbol}"
        
        try:
            # Utiliser le symbole comme clé si non fournie
            message_key = key or symbol
            
            # Publier le message
            self.client.produce(topic=topic, message=data, key=message_key)
            
            # Log pour le débogage (uniquement pour les chandeliers fermés)
            if data.get('is_closed', False):
                logger.info(f"📊 Publié sur {topic}: {data['close']} [O:{data['open']} H:{data['high']} L:{data['low']}]")
            else:
                # Ajouter un nouveau log pour les mises à jour en cours
                logger.info(f"🔄 Mis à jour sur {topic}: prix actuel {data['close']}")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la publication sur Kafka: {str(e)}")
    
    def publish_account_data(self, data: Dict[str, Any], data_type: str) -> None:
        """
        Publie les données de compte sur le topic Kafka approprié.
        
        Args:
            data: Données de compte à publier
            data_type: Type de données (balances, orders, etc.)
        """
        topic = f"account.{data_type}"
        
        try:
            self.client.produce(topic=topic, message=data)
            logger.info(f"💰 Publié données de compte sur {topic}")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la publication des données de compte: {str(e)}")
    
    def flush(self) -> None:
        """
        Force l'envoi de tous les messages en attente.
        """
        self.client.flush()
        logger.info("🔄 Producteur Kafka vidé")
    
    def close(self) -> None:
        """
        Ferme le producteur Kafka proprement.
        """
        self.flush()
        self.client.close()
        logger.info("👋 Producteur Kafka fermé")

# Fonction utilitaire pour créer une instance singleton du producteur
_producer_instance = None

def get_producer() -> KafkaProducer:
    """
    Retourne l'instance singleton du producteur Kafka.
    
    Returns:
        Instance du producteur Kafka
    """
    global _producer_instance
    if _producer_instance is None:
        _producer_instance = KafkaProducer()
    return _producer_instance

# Point d'entrée pour les tests
if __name__ == "__main__":
    producer = get_producer()
    
    # Exemple de publication de données de marché
    test_data = {
        "symbol": "BTCUSDC",
        "start_time": int(time.time() * 1000),
        "close_time": int(time.time() * 1000) + 60000,
        "open": 50000.0,
        "high": 50100.0,
        "low": 49900.0,
        "close": 50050.0,
        "volume": 2.5,
        "is_closed": True
    }
    
    producer.publish_market_data(test_data)
    producer.flush()
    
    logger.info("✅ Test de publication Kafka réussi")
    producer.close()