import os
import psycopg2
from decimal import Decimal

def analyze_cycle_details():
    try:
        conn = psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            database=os.getenv("PGDATABASE", "trading"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "postgres"))

        cursor = conn.cursor()

        print("--- Analyse Détaillée des Cycles de Trading (20 derniers trades) ---")
        # Récupérer les cycles complétés avec toutes les infos de prix
        query = """
            SELECT symbol, side, entry_price, min_price, max_price, status, created_at, profit_loss_percent
            FROM trade_cycles 
            WHERE status = 'completed' AND entry_price IS NOT NULL AND min_price IS NOT NULL AND max_price IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 20;
        """
        cursor.execute(query)
        cycles = cursor.fetchall()

        if not cycles:
            print("Aucun cycle complété avec les informations de prix nécessaires n'a été trouvé.")
            return

        results = []
        for cycle in cycles:
            symbol, side, entry_price, min_price, max_price, status, created_at, final_pnl_pct = cycle
            
            if not isinstance(entry_price, Decimal) or entry_price == 0:
                continue

            # Calcul du gain/perte max potentiel
            if side.upper() == 'BUY':
                max_gain_pct = ((max_price - entry_price) / entry_price) * 100
                min_gain_pct = ((min_price - entry_price) / entry_price) * 100
            else: # SELL
                max_gain_pct = ((entry_price - min_price) / entry_price) * 100
                min_gain_pct = ((entry_price - max_price) / entry_price) * 100

            results.append({
                "symbol": symbol,
                "side": side,
                "entry_price": entry_price,
                "min_price": min_price,
                "max_price": max_price,
                "max_gain_pct": max_gain_pct,
                "min_gain_pct": min_gain_pct,
                "final_pnl_pct": final_pnl_pct,
                "status": status,
                "created": created_at.strftime('%Y-%m-%d %H:%M')
            })

        # Affichage des résultats
        header = f"{'Symbol':<10} | {'Side':<4} | {'Entry':>10} | {'Min Price':>10} | {'Max Price':>10} | {'Max Gain %':>12} | {'Max Loss %':>12} | {'Final PnL %':>12} | {'Created':<16}"
        print("\n" + header)
        print("-" * len(header))

        for res in results:
            print(f"{res['symbol']:<10} | {res['side']:<4} | {res['entry_price']:>10.4f} | {res['min_price']:>10.4f} | {res['max_price']:>10.4f} | {res['max_gain_pct']:>11.2f}% | {res['min_gain_pct']:>11.2f}% | {res['final_pnl_pct']:>11.2f}% | {res['created']:<16}")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Une erreur est survenue: {e}")

if __name__ == "__main__":
    analyze_cycle_details()
