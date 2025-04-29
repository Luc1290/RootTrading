import math
import sys
import os
from dotenv import load_dotenv

# Ajouter le répertoire racine au chemin Python
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Charger les variables d'environnement
load_dotenv()

def validate_config():
    """Valide la configuration du système."""
    try:
        from shared.src.config import (
            BINANCE_API_KEY, BINANCE_SECRET_KEY, 
            KAFKA_BROKER, KAFKA_GROUP_ID,
            REDIS_HOST, REDIS_PORT,
            PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE,
            SYMBOLS, TRADING_MODE, TRADE_QUANTITIES,
            POCKET_CONFIG
        )
        
        # Vérification des configurations critiques
        config_checks = {
            "Binance API Key": bool(BINANCE_API_KEY),
            "Binance Secret Key": bool(BINANCE_SECRET_KEY),
            "Kafka Broker": bool(KAFKA_BROKER),
            "Redis Host": bool(REDIS_HOST),
            "Database Host": bool(PGHOST),
            "Trading Symbols": bool(SYMBOLS),
            "Trading Mode": TRADING_MODE in ['demo', 'live'],
            "Pocket Config": math.isclose(sum(POCKET_CONFIG.values()), 1.0, abs_tol=0.001)# Vérifier que les allocations totalisent 100%
        }
        
        print("Résultats de validation de la configuration:")
        all_valid = True
        
        for name, is_valid in config_checks.items():
            status = "✅" if is_valid else "❌"
            print(f"{status} {name}")
            all_valid = all_valid and is_valid
        
        if all_valid:
            print("\nTous les paramètres de configuration sont valides!")
        else:
            print("\nCertains paramètres de configuration ne sont pas valides!")
            
        # Afficher les symboles configurés
        print(f"\nSymboles configurés: {', '.join(SYMBOLS)}")
        print(f"Mode de trading: {TRADING_MODE}")
        
        # Afficher la configuration des quantités
        print("\nQuantités de trading configurées:")
        for symbol, quantity in TRADE_QUANTITIES.items():
            print(f" - {symbol}: {quantity}")
        
        # Afficher la configuration des poches
        print("\nAllocation des poches de capital:")
        for pocket_type, allocation in POCKET_CONFIG.items():
            print(f" - {pocket_type}: {allocation*100}%")
        
        return all_valid
        
    except ImportError as e:
        print(f"❌ Erreur d'importation du module de configuration: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Erreur lors de la validation de la configuration: {str(e)}")
        return False

def validate_schemas():
    """Valide les schémas de données."""
    try:
        from shared.src.schemas import (
            MarketData, StrategySignal, TradeOrder, 
            TradeExecution, TradeCycle, AssetBalance, 
            PortfolioSummary, PocketSummary, ErrorMessage
        )
        from shared.src.enums import (
            OrderSide, OrderStatus, TradeRole, 
            CycleStatus, SignalStrength, StrategyMode
        )
        
        from datetime import datetime
        
        # Valider MarketData
        print("Test de création de MarketData...")
        market_data = MarketData(
            symbol="BTCUSDC",
            start_time=int(datetime.now().timestamp() * 1000),
            close_time=int(datetime.now().timestamp() * 1000) + 60000,
            open=20000.0,
            high=20100.0,
            low=19900.0,
            close=20050.0,
            volume=1.5
        )
        print(f"✅ Validation réussie: {market_data}")
        
        # Valider StrategySignal
        print("\nTest de création de StrategySignal...")
        signal = StrategySignal(
            strategy="rsi",
            symbol="BTCUSDC",
            side=OrderSide.BUY,
            timestamp=datetime.now(),
            price=20000.0,
            strength=SignalStrength.STRONG,
            metadata={"rsi_value": 29.5}
        )
        print(f"✅ Validation réussie: {signal}")
        
        # Valider TradeOrder
        print("\nTest de création de TradeOrder...")
        order = TradeOrder(
            symbol="BTCUSDC",
            side=OrderSide.BUY,
            quantity=0.001,
            price=20000.0,
            client_order_id="test_order_123",
            strategy="rsi",
            demo=True
        )
        print(f"✅ Validation réussie: {order}")
        
        # Valider TradeCycle
        print("\nTest de création de TradeCycle...")
        cycle = TradeCycle(
            id="cycle_123",
            symbol="BTCUSDC",
            strategy="rsi",
            status=CycleStatus.WAITING_BUY,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            demo=True
        )
        print(f"✅ Validation réussie: {cycle}")
        
        print("\nTous les schémas sont validés avec succès!")
        return True
        
    except ImportError as e:
        print(f"❌ Erreur d'importation des modules de schéma: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Erreur lors de la validation des schémas: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== Validation de la configuration ===")
    config_valid = validate_config()
    
    print("\n=== Validation des schémas ===")
    schemas_valid = validate_schemas()
    
    if config_valid and schemas_valid:
        print("\n✅ Validation globale réussie! La configuration et les schémas sont valides.")
    else:
        issues = []
        if not config_valid:
            issues.append("configuration")
        if not schemas_valid:
            issues.append("schémas")
            
        print(f"\n❌ Validation échouée pour: {', '.join(issues)}")