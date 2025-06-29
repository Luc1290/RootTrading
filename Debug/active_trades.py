
import os
import psycopg2

def analyze_active_trades():
    try:
        conn = psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            database=os.getenv("PGDATABASE", "trading"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "postgres"))

        cursor = conn.cursor()

        print("--- Analyse des Cycles de Trading Actifs ---")
        # Sélectionner les colonnes les plus pertinentes pour l'analyse
        query = """
            SELECT id, symbol, side, status, entry_price, stop_price, min_price, max_price, created_at, updated_at
            FROM trade_cycles 
            WHERE status NOT IN ('completed', 'canceled', 'failed')
            ORDER BY created_at DESC;
        """
        cursor.execute(query)
        active_cycles = cursor.fetchall()

        if not active_cycles:
            print("Aucun cycle actif trouvé.")
            return

        # Afficher les en-têtes
        headers = [desc[0] for desc in cursor.description]
        print(f"{headers[0]:<28} | {headers[1]:<10} | {headers[2]:<5} | {headers[3]:<12} | {headers[4]:>12} | {headers[5]:>12} | {headers[8]:>20}")
        print("-"*110)

        # Afficher chaque cycle
        for row in active_cycles:
            # Formater les valeurs pour un affichage propre
            row = list(row)
            row[4] = f"{row[4]:.4f}" if row[4] is not None else 'N/A'
            row[5] = f"{row[5]:.4f}" if row[5] is not None else 'N/A'
            row[8] = row[8].strftime('%Y-%m-%d %H:%M:%S') if row[8] is not None else 'N/A'
            
            print(f"{row[0]:<28} | {row[1]:<10} | {row[2]:<5} | {row[3]:<12} | {row[4]:>12} | {row[5]:>12} | {row[8]:>20}")

        print("-"*110)

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Une erreur est survenue: {e}")

if __name__ == "__main__":
    analyze_active_trades()
