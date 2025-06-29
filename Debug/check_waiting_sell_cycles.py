

import os
import psycopg2

def check_waiting_sell_cycles():
    try:
        conn = psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            database=os.getenv("PGDATABASE", "trading"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "postgres"))

        cursor = conn.cursor()

        print("--- Vérification des Cycles BUY en attente de Vente ---")
        cursor.execute("SELECT id, symbol, strategy, entry_price, quantity, created_at FROM trade_cycles WHERE status = 'waiting_sell' AND side = 'BUY';")
        waiting_sell_cycles = cursor.fetchall()

        if not waiting_sell_cycles:
            print("Aucun cycle BUY en statut 'waiting_sell' trouvé.")
        else:
            print(f"Found {len(waiting_sell_cycles)} BUY cycles in 'waiting_sell' status:")
            print("\n" + "-"*80)
            print(f"{'ID':<38} | {'Symbole':<10} | {'Stratégie':<15} | {'Prix Entrée':>12} | {'Quantité':>10} | {'Créé le':<20}")
            print("-"*80)
            for cycle in waiting_sell_cycles:
                cycle_id, symbol, strategy, entry_price, quantity, created_at = cycle
                print(f"{cycle_id:<38} | {symbol:<10} | {strategy:<15} | {entry_price:>12.8f} | {quantity:>10.8f} | {str(created_at):<20}")
            print("-"*80)

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Une erreur est survenue: {e}")

if __name__ == "__main__":
    check_waiting_sell_cycles()

