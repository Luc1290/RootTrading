import sys
import os
import time
from dotenv import load_dotenv

# Ajouter le répertoire racine au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Charger les variables d'environnement
load_dotenv()

def test_redis_connection():
    """Teste la connexion au serveur Redis."""
    try:
        from shared.src.redis_client import RedisClient
        
        print("Initialisation du client Redis...")
        redis_client = RedisClient()
        
        # Test de ping
        print("Test de ping Redis...")
        ping_result = redis_client.redis.ping()
        if ping_result:
            print("✅ Connexion Redis établie avec succès!")
        else:
            print("❌ Échec du ping Redis!")
            
        # Test de set/get
        test_key = "test:connection:" + str(time.time())
        test_value = "Connexion réussie!"
        
        print(f"Test de set/get avec la clé '{test_key}'...")
        redis_client.set(test_key, test_value, expiration=60)
        retrieved_value = redis_client.get(test_key)
        
        if retrieved_value == test_value:
            print(f"✅ Test set/get réussi: {retrieved_value}")
        else:
            print(f"❌ Échec du test set/get: attendu '{test_value}', obtenu '{retrieved_value}'")
        
        # Nettoyer la clé de test
        redis_client.delete(test_key)
        
        return True
    except Exception as e:
        print(f"❌ Erreur lors du test Redis: {str(e)}")
        return False

# Correction pour le test Kafka dans middleware-test.py

def test_kafka_connection():
    """Teste la connexion au cluster Kafka."""
    try:
        from shared.src.kafka_client import KafkaClient
        from confluent_kafka.admin import AdminClient
        
        print("Initialisation du client Kafka...")
        kafka_client = KafkaClient()
        
        # Tester la création d'un producteur
        print("Test de création du producteur Kafka...")
        producer = kafka_client._create_producer()
        if producer:
            print("✅ Producteur Kafka créé avec succès!")
        else:
            print("❌ Échec de création du producteur Kafka!")
        
        # Essayer de lister les topics en créant directement un AdminClient
        try:
            print("Tentative de listage des topics Kafka...")
            # Créer directement l'AdminClient sans passer par _ensure_topics_exist
            admin_conf = {'bootstrap.servers': kafka_client.broker}
            admin_client = AdminClient(admin_conf)
            
            if admin_client:
                topics = admin_client.list_topics(timeout=10).topics
                
                print("Liste des topics Kafka disponibles:")
                if topics:
                    for topic in topics:
                        print(f" - {topic}")
                else:
                    print("Aucun topic trouvé!")
                    
                print("✅ Connexion au cluster Kafka réussie!")
            else:
                print("❌ Impossible de créer l'admin_client Kafka")
                return False
                
        except Exception as e:
            print(f"❌ Erreur lors du listage des topics: {str(e)}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Erreur lors du test Kafka: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== Test de connexion Redis ===")
    redis_success = test_redis_connection()
    
    print("\n=== Test de connexion Kafka ===")
    kafka_success = test_kafka_connection()
    
    if redis_success and kafka_success:
        print("\n✅ Tous les tests de connexion ont réussi!")
    else:
        failed = []
        if not redis_success:
            failed.append("Redis")
        if not kafka_success:
            failed.append("Kafka")
        print(f"\n❌ Échec des tests de connexion pour: {', '.join(failed)}")