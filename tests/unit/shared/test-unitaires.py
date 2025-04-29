import unittest
import sys
import os

# Ajouter le répertoire racine au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.src.schemas import MarketData, StrategySignal
from shared.src.enums import OrderSide, SignalStrength

class TestSchemas(unittest.TestCase):
    """Tests pour les schémas Pydantic."""
    
    def test_market_data(self):
        """Tester la validation des données de marché."""
        # Création d'un objet MarketData valide
        data = MarketData(
            symbol="BTCUSDC",
            start_time=1609459200000,
            close_time=1609459260000,
            open=20000.0,
            high=20100.0,
            low=19900.0,
            close=20050.0,
            volume=1.5
        )
        
        # Vérification de la conversion automatique du timestamp
        self.assertIsNotNone(data.timestamp)
        
        # Test avec des données invalides
        with self.assertRaises(Exception):
            # Prix négatif devrait échouer
            MarketData(
                symbol="BTCUSDC",
                start_time=1609459200000,
                close_time=1609459260000,
                open=-20000.0,  # Prix négatif
                high=20100.0,
                low=19900.0,
                close=20050.0,
                volume=1.5
            )
    
    def test_strategy_signal(self):
        """Tester la création de signaux de stratégie."""
        signal = StrategySignal(
            strategy="rsi",
            symbol="BTCUSDC",
            side=OrderSide.BUY,
            timestamp="2023-01-01T12:00:00",
            price=20000.0,
            strength=SignalStrength.STRONG
        )
        
        self.assertEqual(signal.strategy, "rsi")
        self.assertEqual(signal.side, "BUY")  # Devrait être converti en chaîne
        self.assertEqual(signal.strength, "strong")  # Devrait être converti en chaîne

class TestRedisClient(unittest.TestCase):
    """Tests pour le client Redis."""
    
    def test_import(self):
        """Vérifier que le module peut être importé."""
        try:
            from shared.src.redis_client import RedisClient
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Échec d'importation du module RedisClient: {e}")

class TestKafkaClient(unittest.TestCase):
    """Tests pour le client Kafka."""
    
    def test_import(self):
        """Vérifier que le module peut être importé."""
        try:
            from shared.src.kafka_client import KafkaClient
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Échec d'importation du module KafkaClient: {e}")

if __name__ == "__main__":
    unittest.main()