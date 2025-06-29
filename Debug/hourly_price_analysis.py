

import os
import psycopg2
from datetime import datetime, timedelta

def analyze_hourly_prices():
    try:
        conn = psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            database=os.getenv("PGDATABASE", "trading"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "postgres"))

        cursor = conn.cursor()

        print("--- Analyse des Mouvements de Prix Horaires (Dernières 24h) ---")
        
        query = """
        SELECT
            symbol,
            date_trunc('hour', time) as hour,
            MIN(close) as min_price,
            MAX(close) as max_price,
            (MAX(close) - MIN(close)) / MIN(close) * 100 as hourly_range_pct
        FROM
            market_data
        WHERE
            time >= NOW() - INTERVAL '24 hours'
        GROUP BY
            symbol,
            date_trunc('hour', time)
        ORDER BY
            symbol,
            hour;
        """
        cursor.execute(query)
        hourly_data = cursor.fetchall()

        if not hourly_data:
            print("Aucune donnée de marché horaire trouvée pour les dernières 24 heures.")
            return

        print("\n" + "-"*80)
        print(f"{'Symbole':<10} | {'Heure':<20} | {'Min Prix':>12} | {'Max Prix':>12} | {'Range %':>10}")
        print("-"*80)

        for row in hourly_data:
            symbol, hour, min_price, max_price, hourly_range_pct = row
            print(f"{symbol:<10} | {hour.strftime('%Y-%m-%d %H:%M'):<20} | {min_price:>12.4f} | {max_price:>12.4f} | {hourly_range_pct:>9.2f}%")
        
        print("-"*80)

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"Une erreur est survenue: {e}")

if __name__ == "__main__":
    analyze_hourly_prices()

