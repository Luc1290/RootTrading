import sys
import os
import time
import json
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Ajouter le répertoire racine au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Charger les variables d'environnement
load_dotenv()

def simulate_gateway():
    """Simule le comportement du Gateway en produisant des données de marché fictives."""
    try:
        from shared.src.kafka_client import KafkaClient
        from shared.src.redis_client import RedisClient
        from shared.src.schemas import MarketData
        from shared.src.config import SYMBOLS, KAFKA_TOPIC_MARKET_DATA, get_redis_channel
        
        print("Initialisation des clients Kafka et Redis...")
        kafka_client = KafkaClient()
        redis_client = RedisClient()
        
        # Nombre de candles à générer
        num_candles = 5
        symbols = SYMBOLS if isinstance(SYMBOLS, list) else SYMBOLS.split(',')
        
        print(f"Simulation de données de marché pour les symboles: {', '.join(symbols)}")
        
        for symbol in symbols:
            base_price = 20000.0 if symbol.startswith("BTC") else 1500.0
            
            # Générer quelques candles fictives
            for i in range(num_candles):
                # Créer une candle fictive avec des prix légèrement différents
                now = datetime.now()
                start_time = int((now - timedelta(minutes=i+1)).timestamp() * 1000)
                close_time = int((now - timedelta(minutes=i)).timestamp() * 1000)
                
                # Simulation de mouvement de prix aléatoire
                price_change = random.uniform(-0.5, 0.5) / 100  # ±0.5%
                open_price = base_price * (1 + random.uniform(-0.1, 0.1) / 100)
                close_price = open_price * (1 + price_change)
                high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.1) / 100)
                low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.1) / 100)
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
                
                # Convertir en dictionnaire et publier sur Kafka
                data_dict = market_data.model_dump()
                # Convertir le timestamp en chaîne ISO
                data_dict['timestamp'] = market_data.timestamp.isoformat()
                
                # Publier sur Kafka
                kafka_topic = f"{KAFKA_TOPIC_MARKET_DATA}.{symbol.lower()}"
                print(f"Publication sur le topic Kafka: {kafka_topic}")
                kafka_client.produce(kafka_topic, data_dict)
                
                # Publier aussi sur Redis pour les consommateurs qui utilisent Redis
                redis_channel = get_redis_channel('market', symbol)
                print(f"Publication sur le canal Redis: {redis_channel}")
                redis_client.publish(redis_channel, data_dict)
                
                time.sleep(0.5)  # Pause entre les données
        
        # Forcer l'envoi de tous les messages Kafka en attente
        kafka_client.flush()
        print("✅ Données de marché simulées publiées avec succès!")
        
        # Vérification simple: essayer de récupérer les dernières données depuis Redis
        for symbol in symbols:
            key = f"market_data:{symbol}:latest"
            redis_client.set(key, {"timestamp": datetime.now().isoformat(), "symbol": symbol, "close": base_price})
            
            # Vérifier qu'on peut récupérer cette donnée
            data = redis_client.get(key)
            if data:
                print(f"✅ Données Redis récupérées pour {symbol}: {data}")
            else:
                print(f"❌ Échec de récupération des données Redis pour {symbol}")
        
        return True
            
    except ImportError as e:
        print(f"❌ Erreur d'importation des modules: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Erreur lors de la simulation du Gateway: {str(e)}")
        return False

if __name__ == "__main__":
    simulate_gateway()