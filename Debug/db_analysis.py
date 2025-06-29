
import os
import psycopg2
from collections import Counter

def analyze_db():
    try:
        conn = psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            database=os.getenv("PGDATABASE", "trading"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "postgres"))

        cursor = conn.cursor()

        print("--- Analyse des Cycles de Trading (Détail) ---")
        cursor.execute("SELECT id, symbol, strategy, status, created_at, updated_at, completed_at, profit_loss FROM trade_cycles ORDER BY created_at DESC LIMIT 20;")
        cycles = cursor.fetchall()

        if not cycles:
            print("Aucun cycle de trading trouvé dans la base de données.")
            return

        print("\n" + "-"*100)
        print(f"{'ID':<10} | {'Symbole':<10} | {'Stratégie':<15} | {'Statut':<15} | {'Créé le':<20} | {'Mis à jour le':<20} | {'Terminé le':<20} | {'PnL':>8}")
        print("-"*100)

        for cycle in cycles:
            cycle_id, symbol, strategy, status, created_at, updated_at, completed_at, profit_loss = cycle
            print(f"{cycle_id[:8]:<10} | {symbol:<10} | {strategy:<15} | {status:<15} | {created_at.strftime('%Y-%m-%d %H:%M:%S'):<20} | {updated_at.strftime('%Y-%m-%d %H:%M:%S'):<20} | {completed_at.strftime('%Y-%m-%d %H:%M:%S') if completed_at else 'N/A':<20} | {profit_loss if profit_loss is not None else 'N/A':>8.2f}")
        print("-"*100)

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Une erreur est survenue: {e}")

if __name__ == "__main__":
    analyze_db()
