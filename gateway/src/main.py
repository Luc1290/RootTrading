"""
Point d'entrée principal pour le microservice Gateway.
Gère les connexions WebSocket à Binance et la publication des données vers Kafka.
"""
import asyncio
import logging
import signal
import sys
import os

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from binance_ws import BinanceWebSocket
from kafka_producer import get_producer
from shared.src.config import SYMBOLS, INTERVAL

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gateway.log')
    ]
)
logger = logging.getLogger("gateway")

# Variables pour le contrôle du service
ws_client = None
running = True

async def shutdown(signal_type, loop):
    """
    Gère l'arrêt propre du service en cas de signal (SIGINT, SIGTERM).
    
    Args:
        signal_type: Type de signal reçu
        loop: Boucle asyncio en cours
    """
    global running, ws_client
    
    logger.info(f"Signal {signal_type.name} reçu, arrêt en cours...")
    running = False
    
    # Arrêter le WebSocket
    if ws_client:
        await ws_client.stop()
    
    # Fermer le producteur Kafka
    producer = get_producer()
    producer.close()
    
    # Arrêter la boucle asyncio
    loop.stop()
    
    logger.info("Service Gateway arrêté proprement")

async def main():
    """
    Fonction principale qui démarre le Gateway.
    """
    global ws_client
    
    logger.info("🚀 Démarrage du service Gateway RootTrading...")
    logger.info(f"Configuration: {', '.join(SYMBOLS)} @ {INTERVAL}")
    
    # Créer le client WebSocket Binance
    ws_client = BinanceWebSocket(symbols=SYMBOLS, interval=INTERVAL)
    
    try:
        # Démarrer le client WebSocket
        await ws_client.start()
    except Exception as e:
        logger.error(f"❌ Erreur critique dans le Gateway: {str(e)}")
    finally:
        if ws_client:
            await ws_client.stop()
        
        logger.info("Service Gateway terminé")

if __name__ == "__main__":
    # Obtenir la boucle asyncio
    loop = asyncio.get_event_loop()
    
    # Configurer les gestionnaires de signaux pour l'arrêt propre
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig, 
            lambda s=sig: asyncio.create_task(shutdown(s, loop))
        )
    
    try:
        # Exécuter la fonction principale
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Interruption clavier détectée")
    finally:
        # Fermer la boucle asyncio
        loop.close()
        logger.info("Boucle asyncio fermée")