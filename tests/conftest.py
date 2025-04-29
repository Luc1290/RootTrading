import pytest
import os
import redis
import time
import asyncio
from dotenv import load_dotenv

# Charger les variables d'environnement pour les tests
try:
    load_dotenv("tests/.env.test")
except:
    print("Fichier .env.test non trouvé, utilisation des valeurs par défaut")

@pytest.fixture(scope="session")
def event_loop():
    """Créer une nouvelle boucle d'événements pour chaque session de test."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def mock_environment():
    """Configurer l'environnement de test."""
    # Sauvegarder les variables d'environnement originales
    original_env = os.environ.copy()
    
    # Définir les variables d'environnement pour les tests
    os.environ["REDIS_HOST"] = "localhost"
    os.environ["KAFKA_BROKER"] = "localhost:9092"
    os.environ["TRADING_MODE"] = "demo"
    os.environ["PGHOST"] = "localhost"
    os.environ["PGUSER"] = "postgres"
    os.environ["PGPASSWORD"] = "postgres"
    os.environ["PGDATABASE"] = "postgres_test"
    
    yield
    
    # Restaurer les variables d'environnement
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture(scope="session")
def redis_connection():
    """Établir une connexion Redis pour les tests."""
    client = None
    try:
        client = redis.Redis(host=os.environ.get("REDIS_HOST", "localhost"))
        # Vérifier que Redis est disponible
        client.ping()
    except (redis.ConnectionError, redis.TimeoutError):
        pytest.skip("Redis n'est pas disponible pour les tests")
    
    yield client
    
    # Nettoyage
    if client:
        client.flushdb()
        client.close()