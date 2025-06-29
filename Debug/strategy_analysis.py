

import os
import psycopg2
from collections import defaultdict

def analyze_strategies():
    try:
        conn = psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            database=os.getenv("PGDATABASE", "trading"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "postgres"))

        cursor = conn.cursor()

        print("--- Analyse de Performance par Stratégie ---")
        cursor.execute("SELECT strategy, profit_loss FROM trade_cycles WHERE status = 'completed' AND profit_loss IS NOT NULL;")
        completed_cycles = cursor.fetchall()

        if not completed_cycles:
            print("Aucun cycle complété avec PnL trouvé.")
            return

        # Grouper les résultats par stratégie
        strategy_performance = defaultdict(list)
        for strategy, pnl in completed_cycles:
            strategy_performance[strategy].append(pnl)

        print("\n" + "-"*80)
        print(f"{'Stratégie':<25} | {'Trades':>7} | {'Win Rate':>10} | {'PnL Total':>12} | {'PnL Moyen':>12}")
        print("-"*80)

        # Calculer et afficher les métriques pour chaque stratégie
        for strategy, pnl_list in sorted(strategy_performance.items()):
            trade_count = len(pnl_list)
            win_count = sum(1 for pnl in pnl_list if pnl > 0)
            win_rate = (win_count / trade_count) * 100 if trade_count > 0 else 0
            total_pnl = sum(pnl_list)
            avg_pnl = total_pnl / trade_count if trade_count > 0 else 0

            print(f"{strategy:<25} | {trade_count:>7} | {win_rate:>9.2f}% | {total_pnl:>11.2f}$ | {avg_pnl:>11.2f}$")
        
        print("-"*80)

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Une erreur est survenue: {e}")

if __name__ == "__main__":
    analyze_strategies()

