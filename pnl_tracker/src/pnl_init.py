"""
Script d'initialisation pour le PnL Tracker.
Crée des données minimales dans la base de données pour éviter les erreurs
lorsqu'il n'y a pas encore de trades.
"""
import sys
import os
import logging
import psycopg2
from datetime import datetime, timedelta
import uuid

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import get_db_url, SYMBOLS

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pnl_init")

def initialize_db():
    """
    Initialise la base de données avec des données minimales pour le PnL Tracker.
    """
    conn = None
    try:
        # Connexion à la base de données
        db_url = get_db_url()
        logger.info(f"Connexion à la base de données: {db_url}")
        conn = psycopg2.connect(db_url)
        
        # Vérifier si des cycles de trading existent déjà
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM trade_cycles")
            count = cursor.fetchone()[0]
            
            if count > 0:
                logger.info(f"La base de données contient déjà {count} cycles de trading. Pas besoin d'initialisation.")
                return
        
        # Créer des exemples de cycles de trading
        logger.info("Création de données d'exemple pour les cycles de trading...")
        
        # Données d'exemple
        strategies = ["rsi", "bollinger", "ema_cross"]
        statuses = ["completed", "canceled", "failed"]
        
        # Date actuelle et il y a 30 jours
        now = datetime.now()
        thirty_days_ago = now - timedelta(days=30)
        
        # Créer des cycles pour chaque symbole et stratégie
        cycles = []
        for symbol in SYMBOLS:
            for strategy in strategies:
                # Un cycle complété avec profit
                cycle_id = str(uuid.uuid4())
                entry_price = 50000.0 if symbol == "BTCUSDC" else 2000.0
                exit_price = entry_price * 1.05  # 5% de profit
                quantity = 0.01 if symbol == "BTCUSDC" else 0.1
                profit = (exit_price - entry_price) * quantity
                completed_at = now - timedelta(days=2)
                
                cycles.append({
                    "id": cycle_id,
                    "symbol": symbol,
                    "strategy": strategy,
                    "status": "completed",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "quantity": quantity,
                    "profit_loss": profit,
                    "profit_loss_percent": 5.0,
                    "created_at": thirty_days_ago,
                    "completed_at": completed_at
                })
                
                # Un cycle complété avec perte
                cycle_id = str(uuid.uuid4())
                entry_price = 52000.0 if symbol == "BTCUSDC" else 2100.0
                exit_price = entry_price * 0.97  # 3% de perte
                quantity = 0.01 if symbol == "BTCUSDC" else 0.1
                profit = (exit_price - entry_price) * quantity
                completed_at = now - timedelta(days=5)
                
                cycles.append({
                    "id": cycle_id,
                    "symbol": symbol,
                    "strategy": strategy,
                    "status": "completed",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "quantity": quantity,
                    "profit_loss": profit,
                    "profit_loss_percent": -3.0,
                    "created_at": thirty_days_ago + timedelta(days=10),
                    "completed_at": completed_at
                })
        
        # Insérer les cycles dans la base de données
        with conn.cursor() as cursor:
            for cycle in cycles:
                cursor.execute("""
                INSERT INTO trade_cycles
                (id, symbol, strategy, status, entry_price, exit_price, quantity, profit_loss, profit_loss_percent, created_at, updated_at, completed_at)
                VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    cycle["id"],
                    cycle["symbol"],
                    cycle["strategy"],
                    cycle["status"],
                    cycle["entry_price"],
                    cycle["exit_price"],
                    cycle["quantity"],
                    cycle["profit_loss"],
                    cycle["profit_loss_percent"],
                    cycle["created_at"],
                    cycle["created_at"],  # updated_at = created_at pour la simplicité
                    cycle["completed_at"]
                ))
            
            conn.commit()
            logger.info(f"✅ {len(cycles)} cycles de trading d'exemple créés avec succès.")

        logger.info("Initialisation terminée avec succès.")

    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation de la base de données: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    initialize_db()