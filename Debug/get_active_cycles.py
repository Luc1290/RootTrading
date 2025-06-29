

import os
import psycopg2

def get_active_cycles():
    try:
        conn = psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            database=os.getenv("PGDATABASE", "trading"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "postgres"))

        cursor = conn.cursor()

        print("--- Recherche de Cycles Actifs ---")
        cursor.execute("SELECT id, symbol, strategy, status, entry_price, quantity FROM trade_cycles WHERE status NOT IN ('completed', 'canceled', 'failed');")
        active_cycles = cursor.fetchall()

        if not active_cycles:
            print("Aucun cycle actif trouvé dans la base de données.")
            return None

        print("\nCycles Actifs Trouvés:")
        for cycle in active_cycles:
            print(f"- ID: {cycle[0]}, Symbole: {cycle[1]}, Stratégie: {cycle[2]}, Statut: {cycle[3]}, Prix d'entrée: {cycle[4]}, Quantité: {cycle[5]}")
        
        # Retourne le premier symbole trouvé pour l'observation en temps réel
        return active_cycles[0][1] if active_cycles else None

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Une erreur est survenue lors de la recherche des cycles actifs: {e}")
        return None

if __name__ == "__main__":
    symbol_to_monitor = get_active_cycles()
    if symbol_to_monitor:
        print(f"\nSymbole à surveiller en temps réel: {symbol_to_monitor}")
    else:
        print("Aucun symbole à surveiller.")

