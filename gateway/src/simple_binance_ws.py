"""
Module de connexion WebSocket SIMPLE à Binance.
ARCHITECTURE PROPRE : Reçoit uniquement les données OHLCV brutes en temps réel.
AUCUN calcul d'indicateur - transmission pure des données de marché.
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional
import asyncio
import websockets
from websockets.client import ClientProtocol as WebSocketClientProtocol
from websockets.exceptions import ConnectionClosed, InvalidStatus

# Importer les clients partagés
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import SYMBOLS, INTERVAL
from shared.src.kafka_client import KafkaClient
from gateway.src.kafka_producer import get_producer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("simple_binance_ws")


class SimpleBinanceWebSocket:
    """
    Gestionnaire SIMPLE de connexion WebSocket à Binance.
    Récupère uniquement les données OHLCV brutes en temps réel.
    AUCUN calcul d'indicateur technique.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        intervals: Optional[List[str]] = None,
        kafka_client: Optional[KafkaClient] = None,
    ):
        """
        Initialise la connexion WebSocket Binance.

        Args:
            symbols: Liste des symboles à surveiller (ex: ['BTCUSDC', 'ETHUSDC'])
            intervals: Liste des intervalles à surveiller (ex: ['1m', '3m', '5m', '15m', '1h', '1d'])
            kafka_client: Client Kafka pour la publication des données
        """
        self.symbols = symbols or SYMBOLS
        self.intervals = intervals or ["1m", "3m", "5m", "15m", "1h", "1d"]
        self.kafka_client = kafka_client or get_producer()
        self.ws: Optional[WebSocketClientProtocol] = None
        self.running = False
        self.reconnect_delay = 1  # Secondes, pour backoff exponentiel
        self.last_message_time = 0.0
        self.heartbeat_interval = 60  # Secondes

        # Générateur de streams pour le WebSocket
        self.stream_paths = self._generate_stream_paths()

        logger.info(f"📡 SimpleBinanceWebSocket initialisé - OHLCV brutes uniquement")
        logger.info(f"🎯 Symboles: {', '.join(self.symbols)}")
        logger.info(f"⏱️ Intervalles: {', '.join(self.intervals)}")

    def _generate_stream_paths(self) -> List[str]:
        """
        Génère les chemins de streams WebSocket pour tous les symboles/intervalles.
        """
        streams = []
        for symbol in self.symbols:
            for interval in self.intervals:
                # Stream kline pour chaque symbole/intervalle
                stream_name = f"{symbol.lower()}@kline_{interval}"
                streams.append(stream_name)

        logger.info(
            f"🔗 {len(streams)} streams générés: {len(self.symbols)} symboles × {len(self.intervals)} intervalles"
        )
        return streams

    async def start(self):
        """
        Démarre la connexion WebSocket avec gestion de reconnexion automatique.
        """
        self.running = True
        logger.info("🚀 Démarrage de la connexion WebSocket Binance...")

        while self.running:
            try:
                await self._connect_and_listen()
            except Exception as e:
                logger.error(f"❌ Erreur WebSocket: {e}")
                if self.running:
                    logger.info(
                        f"🔄 Reconnexion dans {self.reconnect_delay} secondes..."
                    )
                    await asyncio.sleep(self.reconnect_delay)
                    self.reconnect_delay = min(
                        self.reconnect_delay * 2, 60
                    )  # Backoff exponentiel

    async def _connect_and_listen(self):
        """
        Se connecte au WebSocket et écoute les messages.
        """
        # URL du WebSocket combiné
        base_url = "wss://stream.binance.com:9443/ws/"
        stream_params = "/".join(self.stream_paths)
        websocket_url = f"{base_url}{stream_params}"

        logger.info(f"🔗 Connexion à: {websocket_url}")

        async with websockets.connect(websocket_url) as ws:
            self.ws = ws
            self.reconnect_delay = 1  # Reset du délai après connexion réussie
            logger.info("✅ WebSocket connecté")

            # Boucle d'écoute des messages
            async for message in ws:
                if not self.running:
                    break

                await self._handle_message(message)
                self.last_message_time = time.time()

    async def _handle_message(self, message: str):
        """
        Traite un message reçu du WebSocket.

        Args:
            message: Message JSON brut du WebSocket
        """
        try:
            data = json.loads(message)

            # Traiter uniquement les messages kline
            if "k" in data:
                await self._process_kline_data(data)
            else:
                logger.debug(f"⚠️ Message non-kline ignoré: {data.get('e', 'unknown')}")

        except json.JSONDecodeError as e:
            logger.error(f"❌ Erreur décodage JSON: {e}")
        except Exception as e:
            logger.error(f"❌ Erreur traitement message: {e}")

    async def _process_kline_data(self, data: Dict[str, Any]):
        """
        Traite les données de chandelier (kline) SANS calcul d'indicateur.

        Args:
            data: Données kline du WebSocket Binance
        """
        try:
            kline = data["k"]

            # Extraire uniquement les données OHLCV brutes
            raw_candle = {
                "symbol": kline["s"],
                "time": kline["t"],  # Open time
                "close_time": kline["T"],  # Close time
                "interval": kline["i"],
                "open": float(kline["o"]),
                "high": float(kline["h"]),
                "low": float(kline["l"]),
                "close": float(kline["c"]),
                "volume": float(kline["v"]),
                "is_closed": kline["x"],  # Bougie fermée ou en cours
                "quote_asset_volume": float(kline["q"]),
                "number_of_trades": kline["n"],
                "taker_buy_base_asset_volume": float(kline["V"]),
                "taker_buy_quote_asset_volume": float(kline["Q"]),
                "source": "binance_websocket",
                "timestamp": time.time(),
            }

            # Publier uniquement les bougies fermées
            if raw_candle["is_closed"]:
                await self._publish_raw_data(raw_candle)
                logger.debug(
                    f"📤 Bougie fermée publiée: {raw_candle['symbol']} @ {raw_candle['close']}"
                )
            else:
                logger.debug(f"⏳ Bougie en cours ignorée: {raw_candle['symbol']}")

        except KeyError as e:
            logger.error(f"❌ Champ manquant dans les données kline: {e}")
        except Exception as e:
            logger.error(f"❌ Erreur traitement kline: {e}")

    async def _publish_raw_data(self, candle_data: Dict[str, Any]):
        """
        Publie les données brutes vers Kafka.

        Args:
            candle_data: Données OHLCV brutes de la bougie
        """
        try:
            # Adapter les données pour le KafkaProducer
            market_data = {
                "symbol": candle_data["symbol"],
                "timeframe": candle_data["interval"],
                "time": candle_data["time"],
                "close_time": candle_data["close_time"],
                "open": candle_data["open"],
                "high": candle_data["high"],
                "low": candle_data["low"],
                "close": candle_data["close"],
                "volume": candle_data["volume"],
                "is_closed": candle_data["is_closed"],
                "quote_asset_volume": candle_data["quote_asset_volume"],
                "number_of_trades": candle_data["number_of_trades"],
                "taker_buy_base_asset_volume": candle_data[
                    "taker_buy_base_asset_volume"
                ],
                "taker_buy_quote_asset_volume": candle_data[
                    "taker_buy_quote_asset_volume"
                ],
                "source": candle_data["source"],
                "timestamp": candle_data["timestamp"],
            }

            # Utiliser la méthode correcte du KafkaProducer
            if hasattr(self.kafka_client, "publish_market_data"):
                self.kafka_client.publish_market_data(
                    market_data, key=candle_data["symbol"]
                )
            else:
                # Fallback - essayer d'autres méthodes disponibles
                if hasattr(self.kafka_client, "send"):
                    self.kafka_client.send(
                        "market_data", market_data, key=candle_data["symbol"]
                    )
                else:
                    logger.warning(
                        f"Méthode de publication Kafka non trouvée pour {type(self.kafka_client)}"
                    )

            logger.debug(
                f"📡 Données brutes publiées: {candle_data['symbol']} @ {candle_data['interval']}"
            )

        except Exception as e:
            logger.error(f"❌ Erreur publication Kafka: {e}")

    async def stop(self):
        """
        Arrête la connexion WebSocket proprement.
        """
        logger.info("🛑 Arrêt de la connexion WebSocket...")
        self.running = False

        if self.ws:
            await self.ws.close()
            self.ws = None

        logger.info("✅ WebSocket arrêté")

    def get_status(self) -> Dict[str, Any]:
        """
        Retourne le statut de la connexion WebSocket.

        Returns:
            Dictionnaire avec les informations de statut
        """
        return {
            "connected": self.ws is not None,
            "running": self.running,
            "last_message_time": self.last_message_time,
            "symbols": self.symbols,
            "intervals": self.intervals,
            "stream_count": len(self.stream_paths),
            "architecture": "multi_timeframe_raw_data",
        }
