"""
Module Kafka Producer pour le Gateway.
Convertit les données WebSocket Binance en messages Kafka.
"""

import logging
import os

# Importer les clients partagés
import sys
import time
from typing import Any

from shared.src.config import KAFKA_BROKER, KAFKA_TOPIC_MARKET_DATA
from shared.src.kafka_client import KafkaClient
from shared.src.redis_client import RedisClient

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../")))


# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
        self.redis_client = RedisClient()
        logger.info(f"✅ Producteur Kafka+Redis initialisé pour {broker}")

    def publish_market_data(
        self, data: dict[str, Any], key: str | None = None
    ) -> None:
        """
        Publie les données de marché sur le topic Kafka approprié.

        Args:
            data: Données de marché à publier
            key: Clé à utiliser pour le partitionnement (généralement le symbole)
        """
        if not data or "symbol" not in data:
            logger.error(
                "❌ Données de marché invalides, impossible de publier")
            return

        symbol = data["symbol"].lower()
        timeframe = data.get("timeframe", "1m")
        topic = f"{KAFKA_TOPIC_MARKET_DATA}.{symbol}.{timeframe}"

        try:
            # Utiliser le symbole comme clé si non fournie
            message_key = key or symbol

            # Publier le message sur Kafka
            self.client.produce(topic=topic, message=data, key=message_key)

            # Publier sur Redis avec le même format que Kafka
            redis_channel = (
                f"roottrading:market:data:{symbol}:{data.get('timeframe', '1m')}"
            )
            self.redis_client.publish(redis_channel, data)

            # Log pour le débogage (uniquement pour les chandeliers fermés)
            if data.get("is_closed", False):
                logger.info(
                    f"📊 OHLCV brutes publiées {symbol.upper()}: {data['close']} [O:{data['open']} H:{data['high']} L:{data['low']} V:{data.get('volume', 'N/A')}]"
                )
            else:
                # Log plus discret pour les mises à jour en cours
                logger.debug(
                    f"🔄 Données en cours {symbol.upper()}: prix actuel {data['close']}"
                )
        except Exception as e:
            error_msg = str(e).replace("{", "{{").replace("}", "}}")
            logger.exception(
                "❌ Erreur lors de la publication sur Kafka: ")

    def publish_to_topic(
        self, topic: str, data: dict[str, Any], key: str | None = None
    ) -> None:
        """
        Publie des données sur un topic Kafka spécifique.

        Args:
            topic: Nom du topic Kafka
            data: Données à publier
            key: Clé pour le partitionnement
        """
        try:
            self.client.produce(topic=topic, message=data, key=key)

            # Log simple pour les données publiées
            if data.get("is_closed", False):
                symbol = data.get("symbol", "N/A")
                timeframe = data.get("timeframe", "N/A")
                price = data.get("close", "N/A")
                logger.info(
                    f"📊 Données publiées {symbol} {timeframe}: {price}")

        except Exception:
            logger.exception("❌ Erreur publication topic {topic}")

    def publish_account_data(
            self, data: dict[str, Any], data_type: str) -> None:
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
            error_msg = str(e).replace("{", "{{").replace("}", "}}")
            logger.exception(
                "❌ Erreur lors de la publication des données de compte: "
            )

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
        "is_closed": True,
    }

    producer.publish_market_data(test_data)
    producer.flush()

    logger.info("✅ Test de publication Kafka réussi")
    producer.close()
