"""
Module de connexion WebSocket à Binance.
Reçoit les données de marché en temps réel et les transmet au Kafka producer.
"""
import json
import logging
import time
from typing import Dict, Any, Callable, List
import asyncio
import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatusCode

# Importer les clients partagés
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import BINANCE_API_KEY, BINANCE_SECRET_KEY, SYMBOLS, INTERVAL, KAFKA_TOPIC_MARKET_DATA
from shared.src.kafka_client import KafkaClient
from gateway.src.kafka_producer import get_producer   # NEW


# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("binance_ws")

class BinanceWebSocket:
    """
    Gestionnaire de connexion WebSocket à Binance pour les données de marché.
    Récupère les données en temps réel via WebSocket et les transmet via Kafka.
    """
    
    def __init__(self, symbols: List[str] = None, interval: str = INTERVAL, kafka_client: KafkaClient = None):
        """
        Initialise la connexion WebSocket Binance.
        
        Args:
            symbols: Liste des symboles à surveiller (ex: ['BTCUSDC', 'ETHUSDC'])
            interval: Intervalle des chandeliers (ex: '1m', '5m', '1h')
            kafka_client: Client Kafka pour la publication des données
        """
        self.symbols = symbols or SYMBOLS
        self.interval = interval
        self.kafka_client = kafka_client or get_producer()   # utilise le producteur singleton
        self.ws = None
        self.running = False
        self.reconnect_delay = 1  # Secondes, pour backoff exponentiel
        self.last_message_time = 0
        self.heartbeat_interval = 60  # Secondes
        
        # URL de l'API WebSocket Binance
        self.base_url = "wss://stream.binance.com:9443/ws"
        
        # Crée les chemins des streams pour chaque paire
        self.stream_paths = [f"{symbol.lower()}@kline_{interval}" for symbol in self.symbols]
    
    async def _connect(self) -> None:
        """
        Établit la connexion WebSocket avec Binance et souscrit aux streams.
        """
        uri = self.base_url
        
        try:
            logger.info(f"Connexion à Binance WebSocket: {uri}")
            self.ws = await websockets.connect(uri)
            
            # S'abonner aux streams
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": self.stream_paths,
                "id": int(time.time() * 1000)
            }
            
            await self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Abonnement envoyé pour: {', '.join(self.stream_paths)}")
            
            # Attendre la confirmation d'abonnement
            response = await self.ws.recv()
            response_data = json.loads(response)
            
            if 'result' in response_data and response_data['result'] is None:
                logger.info("✅ Abonnement aux streams Binance confirmé")
            else:
                logger.warning(f"Réponse d'abonnement inattendue: {response_data}")
            
            # Réinitialiser le délai de reconnexion
            self.reconnect_delay = 1
            
        except (ConnectionClosed, InvalidStatusCode) as e:
            logger.error(f"❌ Erreur lors de la connexion WebSocket: {str(e)}")
            await self._handle_reconnect()
        except Exception as e:
            logger.error(f"❌ Erreur inattendue lors de la connexion: {str(e)}")
            await self._handle_reconnect()
    
    async def _handle_reconnect(self) -> None:
        """
        Gère la reconnexion en cas de perte de connexion.
        Utilise un backoff exponentiel pour éviter de surcharger le serveur.
        """
        # Vérifier si le service est encore en cours d'exécution
        if not self.running:
            return
    
        # Limiter le nombre de tentatives
        max_retries = 20
        retry_count = 0
    
        while self.running and retry_count < max_retries:
            logger.warning(f"Tentative de reconnexion {retry_count+1}/{max_retries} dans {self.reconnect_delay} secondes...")
            await asyncio.sleep(self.reconnect_delay)
        
            # Backoff exponentiel avec plafond plus élevé
            self.reconnect_delay = min(300, self.reconnect_delay * 2)  # 5 minutes max
        
            try:
                # Fermer proprement la connexion existante si elle existe
                if self.ws:
                    try:
                        await self.ws.close()
                    except Exception as e:
                        logger.warning(f"Erreur lors de la fermeture de la connexion existante: {str(e)}")
                
                # Établir une nouvelle connexion
                await self._connect()
                return  # Reconnexion réussie
            except Exception as e:
                retry_count += 1
                logger.error(f"❌ Échec de reconnexion: {str(e)}")
    
        logger.critical("Impossible de se reconnecter après plusieurs tentatives. Arrêt du service.")
        # Signaler l'arrêt au service principal
        self.running = False
    
    async def _heartbeat_check(self) -> None:
        """
        Vérifie régulièrement si nous recevons toujours des messages.
        Reconnecte si aucun message n'a été reçu depuis trop longtemps.
        """
        while self.running:
            await asyncio.sleep(self.heartbeat_interval)
            
            # Si aucun message reçu depuis 2x l'intervalle de heartbeat
            if time.time() - self.last_message_time > self.heartbeat_interval * 2:
                logger.warning(f"❗ Aucun message reçu depuis {self.heartbeat_interval * 2} secondes. Reconnexion...")
                
                # Fermer la connexion existante
                if self.ws:
                    try:
                        await self.ws.close()
                    except Exception as e:
                        logger.warning(f"Erreur lors de la fermeture de la connexion: {str(e)}")
                
                # Reconnecter
                await self._connect()
    
    def _process_kline_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traite un message de chandelier Binance et le convertit en format standardisé.
        
        Args:
            message: Message brut de Binance
            
        Returns:
            Message formaté pour le système RootTrading
        """
        # Extraire les données pertinentes
        kline = message['k']
        
        # Créer un message standardisé
        processed_data = {
            "symbol": message['s'],
            "start_time": kline['t'],
            "close_time": kline['T'],
            "open": float(kline['o']),
            "high": float(kline['h']),
            "low": float(kline['l']),
            "close": float(kline['c']),
            "volume": float(kline['v']),
            "is_closed": kline['x']  # Si la période est fermée
        }
        
        return processed_data
    
    async def _listen(self) -> None:
        """
        Écoute les messages WebSocket et les traite.
        """
        while self.running:
            try:
                if not self.ws:
                    logger.warning("Connexion WebSocket perdue, reconnexion...")
                    await self._connect()
                    continue
                
                # Recevoir un message
                message = await self.ws.recv()
                self.last_message_time = time.time()
                
                # Traiter le message
                try:
                    data = json.loads(message)
                    
                    # Vérifier le type de message
                    if 'e' in data:
                        event_type = data['e']
                        
                        # Traiter les données de chandelier
                        if event_type == 'kline':
                            processed_data = self._process_kline_message(data)

                            # 👉 on ignore les bougies encore ouvertes
                            if not processed_data['is_closed']:
                                continue

                            symbol = processed_data['symbol']
                            key = f"{symbol}:{processed_data['start_time']}"        # clé idempotente

                            self.kafka_client.publish_market_data(                  # on passe par le wrapper
                                data=processed_data,
                                key=key
                            )
        
                            # Log pour le débogage (seulement si le chandelier est fermé)
                            if processed_data['is_closed']:
                                logger.info(f"📊 {symbol} @ {self.interval}: {processed_data['close']} "
                                            f"[O:{processed_data['open']} H:{processed_data['high']} L:{processed_data['low']}]")
                except json.JSONDecodeError:
                    logger.error(f"Message non-JSON reçu: {message[:100]}...")
                except Exception as e:
                    logger.error(f"Erreur lors du traitement du message: {str(e)}")
                
            except ConnectionClosed as e:
                logger.warning(f"Connexion WebSocket fermée: {str(e)}")
                await self._handle_reconnect()
            except Exception as e:
                logger.error(f"Erreur durant l'écoute: {str(e)}")
                await asyncio.sleep(1)  # Pause pour éviter une boucle d'erreur infinie
    
    async def start(self) -> None:
        """
        Démarre la connexion WebSocket et l'écoute des messages.
        """
        if self.running:
            logger.warning("Le WebSocket est déjà en cours d'exécution.")
            return
            
        self.running = True
        self.last_message_time = time.time()
        
        try:
            # Démarrer la connexion
            await self._connect()
            
            # Démarrer le vérificateur de heartbeat dans une tâche séparée
            heartbeat_task = asyncio.create_task(self._heartbeat_check())
            
            # Écouter les messages
            await self._listen()
            
            # Annuler la tâche de heartbeat si on sort de la boucle
            heartbeat_task.cancel()
            
        except Exception as e:
            logger.error(f"Erreur critique dans la boucle WebSocket: {str(e)}")
            self.running = False
    
    async def stop(self) -> None:
        """
        Arrête la connexion WebSocket proprement.
        """
        logger.info("Arrêt de la connexion WebSocket Binance...")
        self.running = False
        
        if self.ws:
            try:
                # Désabonnement des streams
                unsubscribe_msg = {
                    "method": "UNSUBSCRIBE",
                    "params": self.stream_paths,
                    "id": int(time.time() * 1000)
                }
                
                await self.ws.send(json.dumps(unsubscribe_msg))
                logger.info("Message de désabonnement envoyé")
                
                # Fermer la connexion
                await self.ws.close()
                logger.info("Connexion WebSocket fermée")
                
            except Exception as e:
                logger.error(f"Erreur lors de la fermeture du WebSocket: {str(e)}")
        
        # Fermer le client Kafka
        if self.kafka_client:
            self.kafka_client.flush()
            self.kafka_client.close()
            logger.info("Client Kafka fermé")

# Fonction principale pour exécuter le WebSocket de manière asynchrone
async def run_binance_websocket():
    """
    Fonction principale pour exécuter le WebSocket Binance.
    """
    # Créer le client Kafka
    kafka_client = KafkaClient()
    
    # Créer et démarrer le WebSocket
    ws_client = BinanceWebSocket(kafka_client=kafka_client)
    
    try:
        logger.info("🚀 Démarrage du WebSocket Binance...")
        await ws_client.start()
    except KeyboardInterrupt:
        logger.info("Interruption clavier détectée")
    except Exception as e:
        logger.error(f"Erreur dans la boucle principale: {str(e)}")
    finally:
        logger.info("Arrêt du WebSocket Binance...")
        await ws_client.stop()

# Point d'entrée pour exécution directe
if __name__ == "__main__":
    try:
        asyncio.run(run_binance_websocket())
    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")