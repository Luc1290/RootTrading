

import os
import psycopg2

def analyze_active_cycles():
    try:
        conn = psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            database=os.getenv("PGDATABASE", "trading"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "postgres"))

        cursor = conn.cursor()

        print("--- Analyse des Cycles de Trading Actifs ---")
        # Sélectionner les colonnes pertinentes de la vue active_cycles
        cursor.execute("""
            SELECT 
                id, symbol, strategy, status, side, entry_price, 
                min_price, max_price, current_price, unrealized_pl_percent
            FROM 
                active_cycles
            ORDER BY 
                created_at DESC;
        """)
        active_cycles = cursor.fetchall()

        if not active_cycles:
            print("Aucun cycle de trading actif trouvé dans la base de données.")
            return

        print("\n" + "-"*100)
        print(f"{'ID Cycle':<10} | {'Symbole':<10} | {'Stratégie':<15} | {'Statut':<15} | {'Side':<5} | {'Entrée':>10} | {'Min Prix':>10} | {'Max Prix':>10} | {'Actuel':>10} | {'Gain Actuel%':>12} | {'Gain Potentiel Max%':>18}")
        print("-"*100)

        for cycle in active_cycles:
            (cycle_id, symbol, strategy, status, side, entry_price,
             min_price, max_price, current_price, unrealized_pl_percent) = cycle

            max_potential_gain_percent = 0.0
            if entry_price is not None:
                if side == 'BUY' and max_price is not None:
                    max_potential_gain_percent = ((max_price - entry_price) / entry_price) * 100
                elif side == 'SELL' and min_price is not None:
                    max_potential_gain_percent = ((entry_price - min_price) / entry_price) * 100
            
            # Formater les valeurs pour l'affichage
            entry_price_f = f'{entry_price:.2f}' if entry_price is not None else 'N/A'
            min_price_f = f'{min_price:.2f}' if min_price is not None else 'N/A'
            max_price_f = f'{max_price:.2f}' if max_price is not None else 'N/A'
            current_price_f = f'{current_price:.2f}' if current_price is not None else 'N/A'
            unrealized_pl_percent_f = f'{unrealized_pl_percent:.2f}%' if unrealized_pl_percent is not None else 'N/A'
            max_potential_gain_percent_f = f'{max_potential_gain_percent:.2f}%'

            print(f"{cycle_id:<10.10} | {symbol:<10} | {strategy:<15.15} | {status:<15} | {side:<5} | {entry_price_f:>10} | {min_price_f:>10} | {max_price_f:>10} | {current_price_f:>10} | {unrealized_pl_percent_f:>12} | {max_potential_gain_percent_f:>18}")
        
        print("-"*100)

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Une erreur est survenue: {e}")

if __name__ == "__main__":
    analyze_active_cycles()
