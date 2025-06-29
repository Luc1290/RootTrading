

import os
import psycopg2
from datetime import datetime

def analyze_detailed_cycles():
    try:
        conn = psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            database=os.getenv("PGDATABASE", "trading"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "postgres"))

        cursor = conn.cursor()

        print("--- Analyse Détaillée des Cycles de Trading Terminés ---")
        
        query = """
        SELECT 
            symbol,
            side,
            entry_price,
            exit_price,
            max_price,
            min_price,
            CASE
                WHEN side = 'BUY' AND entry_price IS NOT NULL AND max_price IS NOT NULL AND entry_price != 0
                    THEN (max_price - entry_price) / entry_price * 100
                WHEN side = 'SELL' AND entry_price IS NOT NULL AND min_price IS NOT NULL AND entry_price != 0
                    THEN (entry_price - min_price) / entry_price * 100
                ELSE NULL
            END as max_gain_pct,
            profit_loss_percent,
            DATE(completed_at) as day,
            status
        FROM 
            trade_cycles
        WHERE 
            status = 'completed'
        ORDER BY 
            completed_at DESC
        LIMIT 50; -- Limiter aux 50 derniers cycles pour la lisibilité
        """

        cursor.execute(query)
        cycles = cursor.fetchall()

        if not cycles:
            print("Aucun cycle terminé trouvé avec les critères spécifiés.")
            return

        # Affichage des résultats
        print("\n" + "-"*120)
        print(f"{'Symbol':<10} | {'Side':<5} | {'Entry Price':>12} | {'Exit Price':>12} | {'Max Price':>12} | {'Min Price':>12} | {'Max Gain %':>12} | {'PnL %':>10} | {'Day':<10} | {'Status':<10}")
        print("-"*120)

        for cycle in cycles:
            symbol, side, entry_price, exit_price, max_price, min_price, max_gain_pct, profit_loss_percent, day, status = cycle
            
            print(f"{str(symbol):<10} | {str(side):<5} | {float(entry_price):>12.4f} | {float(exit_price):>12.4f} | {float(max_price):>12.4f} | {float(min_price):>12.4f} | {float(max_gain_pct) if max_gain_pct is not None else 'N/A':>12.2f} | {float(profit_loss_percent):>9.2f}% | {str(day):<10} | {str(status):<10}")
        
        print("-"*120)

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Une erreur est survenue: {e}")

if __name__ == "__main__":
    analyze_detailed_cycles()

