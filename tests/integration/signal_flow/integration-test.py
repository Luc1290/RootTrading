#!/usr/bin/env python3
"""
Script de test d'intégration pour RootTrading.
Vérifie le flux complet : génération de données de marché → signaux → ordres.
"""

import sys
import os
import time
import json
import logging
from datetime import datetime, timedelta
import random
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("integration-test")

# Ajouter le répertoire racine au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Charger les variables d'environnement
load_dotenv()

def generate_market_data(symbols, num_candles=5):
    """Génère des données de marché fictives pour les symboles spécifiés."""
    try:
        from shared.src.kafka_client import KafkaClient
        from shared.src.redis_client import RedisClient
        from shared.src.schemas import MarketData
        from shared.src.config import KAFKA_TOPIC_MARKET_DATA, get_redis_channel
        
        logger.info("Initialisation des clients Kafka et Redis...")
        kafka_client = KafkaClient()
        redis_client = RedisClient()
        
        symbols_list = symbols if isinstance(symbols, list) else symbols.split(',')
        
        logger.info(f"Génération de données de marché pour les symboles: {', '.join(symbols_list)}")
        
        # Tendance simulée pour les symboles
        trends = {}
        for symbol in symbols_list:
            # Simuler différentes tendances de marché (haussière, baissière, neutre)
            trends[symbol] = random.choice([-1, 0, 1]) * 0.001  # +/- 0.1% par candle ou neutre
        
        for i in range(num_candles):
            for symbol in symbols_list:
                # Prix de référence
                base_price = 20000.0 if symbol.startswith("BTC") else 1500.0
                trend = trends[symbol]
                
                # Créer une candle avec tendance
                now = datetime.now()
                start_time = int((now - timedelta(minutes=i+1)).timestamp() * 1000)
                close_time = int((now - timedelta(minutes=i)).timestamp() * 1000)
                
                # Simuler le mouvement de prix
                change = trend + random.uniform(-0.003, 0.003)  # +/- 0.3% volatilité autour de la tendance
                open_price = base_price * (1 + (i * trend) + random.uniform(-0.001, 0.001))
                close_price = open_price * (1 + change)
                high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.002))
                low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.002))
                volume = random.uniform(0.5, 5.0)
                
                # Créer l'objet MarketData
                market_data = MarketData(
                    symbol=symbol,
                    start_time=start_time,
                    close_time=close_time,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    timestamp=datetime.fromtimestamp(start_time / 1000)
                )
                
                # Convertir en dictionnaire et publier
                try:
                    data_dict = market_data.model_dump()
                except AttributeError:
                    # Fallback pour la compatibilité avec les anciennes versions de Pydantic
                    data_dict = market_data.dict()
                    
                # Convertir le timestamp en chaîne ISO
                data_dict['timestamp'] = market_data.timestamp.isoformat()
                
                # Publier sur Kafka
                kafka_topic = f"{KAFKA_TOPIC_MARKET_DATA}.{symbol.lower()}"
                logger.info(f"Publication sur le topic Kafka: {kafka_topic}")
                kafka_client.produce(kafka_topic, data_dict)
                
                # Publier aussi sur Redis
                redis_channel = get_redis_channel('market', symbol)
                logger.info(f"Publication sur le canal Redis: {redis_channel}")
                redis_client.publish(redis_channel, data_dict)
                
            # Pause entre les lots de données
            time.sleep(1)
        
        # Forcer l'envoi des messages Kafka
        kafka_client.flush()
        logger.info("✅ Données de marché simulées publiées avec succès!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la génération des données de marché: {str(e)}")
        return False

def listen_for_signals(timeout=30):
    """Écoute les signaux de trading pendant une période spécifiée."""
    try:
        from shared.src.redis_client import RedisClient
        from shared.src.config import get_redis_channel
        
        logger.info(f"Écoute des signaux de trading pendant {timeout} secondes...")
        redis_client = RedisClient()
        
        # Canal des signaux
        signal_channel = get_redis_channel('analyze', 'signal')
        
        signals_received = []
        
        # Fonction de callback pour les messages Redis
        def signal_callback(channel, message):
            if channel == signal_channel:
                logger.info(f"Signal reçu: {message}")
                signals_received.append(message)
        
        # S'abonner au canal des signaux
        redis_client.subscribe(signal_channel, signal_callback)
        
        # Attendre les signaux pendant le délai spécifié
        start_time = time.time()
        while time.time() - start_time < timeout:
            time.sleep(1)
            if len(signals_received) > 0:
                logger.info(f"Reçu {len(signals_received)} signaux.")
                break
        
        # Se désabonner
        redis_client.unsubscribe()
        
        if len(signals_received) > 0:
            logger.info("✅ Signaux de trading reçus avec succès!")
            return signals_received
        else:
            logger.warning("⚠️ Aucun signal reçu pendant la période d'écoute.")
            return []
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'écoute des signaux: {str(e)}")
        return []

def listen_for_orders(timeout=30):
    """Écoute les ordres générés pendant une période spécifiée."""
    try:
        from shared.src.redis_client import RedisClient
        from shared.src.config import get_redis_channel
        
        logger.info(f"Écoute des ordres pendant {timeout} secondes...")
        redis_client = RedisClient()
        
        # Canal des ordres
        order_channel = get_redis_channel('trade', 'order')
        
        orders_received = []
        
        # Fonction de callback pour les messages Redis
        def order_callback(channel, message):
            if channel == order_channel:
                logger.info(f"Ordre reçu: {message}")
                orders_received.append(message)
        
        # S'abonner au canal des ordres
        redis_client.subscribe(order_channel, order_callback)
        
        # Attendre les ordres pendant le délai spécifié
        start_time = time.time()
        while time.time() - start_time < timeout:
            time.sleep(1)
            if len(orders_received) > 0:
                logger.info(f"Reçu {len(orders_received)} ordres.")
                break
        
        # Se désabonner
        redis_client.unsubscribe()
        
        if len(orders_received) > 0:
            logger.info("✅ Ordres reçus avec succès!")
            return orders_received
        else:
            logger.warning("⚠️ Aucun ordre reçu pendant la période d'écoute.")
            return []
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'écoute des ordres: {str(e)}")
        return []

def check_active_cycles():
    """Vérifie les cycles de trading actifs dans la base de données."""
    try:
        import psycopg2
        from shared.src.config import get_db_url
        
        logger.info("Vérification des cycles de trading actifs...")
        conn = psycopg2.connect(get_db_url())
        cursor = conn.cursor()
        
        # Requête pour récupérer les cycles actifs
        cursor.execute("""
            SELECT id, symbol, strategy, status, created_at
            FROM trade_cycles
            WHERE status NOT IN ('completed', 'canceled', 'failed')
            ORDER BY created_at DESC
            LIMIT 10;
        """)
        
        cycles = cursor.fetchall()
        
        if cycles:
            logger.info(f"✅ {len(cycles)} cycles actifs trouvés:")
            for cycle in cycles:
                logger.info(f"  - Cycle {cycle[0]}: {cycle[1]}, {cycle[2]}, {cycle[3]}, créé à {cycle[4]}")
            return cycles
        else:
            logger.warning("⚠️ Aucun cycle actif trouvé.")
            return []
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la vérification des cycles: {str(e)}")
        return []

def main():
    """Fonction principale du test d'intégration."""
    logger.info("=== TEST D'INTÉGRATION ROOTTRADING ===")
    
    # Étape 1: Générer des données de marché
    logger.info("\n=== ÉTAPE 1: GÉNÉRATION DE DONNÉES DE MARCHÉ ===")
    if not generate_market_data(["BTCUSDC", "ETHUSDC"], num_candles=10):
        logger.error("❌ Échec de la génération des données de marché, arrêt du test.")
        return False
    
    # Étape 2: Écouter les signaux
    logger.info("\n=== ÉTAPE 2: ÉCOUTE DES SIGNAUX ===")
    signals = listen_for_signals(timeout=20)
    
    # Étape 3: Écouter les ordres (seulement si des signaux ont été reçus)
    if signals:
        logger.info("\n=== ÉTAPE 3: ÉCOUTE DES ORDRES ===")
        orders = listen_for_orders(timeout=20)
    else:
        logger.warning("⚠️ Aucun signal reçu, impossible de vérifier les ordres.")
        orders = []
    
    # Étape 4: Vérifier les cycles actifs
    logger.info("\n=== ÉTAPE 4: VÉRIFICATION DES CYCLES ACTIFS ===")
    cycles = check_active_cycles()
    
    # Résumé des résultats
    logger.info("\n=== RÉSUMÉ DU TEST D'INTÉGRATION ===")
    logger.info(f"Données de marché générées: ✅")
    logger.info(f"Signaux reçus: {'✅ ' + str(len(signals)) if signals else '❌ 0'}")
    logger.info(f"Ordres reçus: {'✅ ' + str(len(orders)) if orders else '❌ 0'}")
    logger.info(f"Cycles actifs: {'✅ ' + str(len(cycles)) if cycles else '❌ 0'}")
    
    # Validation du test
    success = len(signals) > 0
    if success:
        logger.info("✅ Test d'intégration réussi! Le flux de données fonctionne.")
    else:
        logger.warning("⚠️ Test d'intégration partiellement réussi. Aucun signal détecté.")
    
    return success

if __name__ == "__main__":
    main()