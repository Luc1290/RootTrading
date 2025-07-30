#!/usr/bin/env python3
"""
Script de debug pour le signal_aggregator
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime

# Connexion à la base de données
conn = psycopg2.connect(
    host="localhost",
    port=5433,
    database="trading",
    user="postgres",
    password="trading123"
)

cursor = conn.cursor(cursor_factory=RealDictCursor)

# 1. Vérifier les données dans analyzer_data
print("=== VERIFICATION DES DONNEES ANALYZER ===")
cursor.execute("""
    SELECT symbol, timeframe, 
           trend_strength, regime_strength, signal_strength,
           market_regime, volatility_regime,
           support_strength, resistance_strength,
           volume_trend, volume_pattern,
           pattern_detected
    FROM analyzer_data 
    WHERE symbol IN ('BTCUSDC', 'ETHUSDC', 'SUIUSDC')
    ORDER BY time DESC 
    LIMIT 5
""")

for row in cursor.fetchall():
    print(f"\n{row['symbol']} {row['timeframe']}:")
    for key, value in row.items():
        if key not in ['symbol', 'timeframe']:
            print(f"  {key}: {value} (type: {type(value).__name__})")

# 2. Vérifier une ligne complète
print("\n\n=== LIGNE COMPLETE POUR BTCUSDC ===")
cursor.execute("""
    SELECT * FROM analyzer_data 
    WHERE symbol = 'BTCUSDC' AND timeframe = '1m'
    ORDER BY time DESC 
    LIMIT 1
""")

row = cursor.fetchone()
if row:
    for key, value in row.items():
        if value is not None and key not in ['time', 'symbol', 'timeframe', 'analysis_timestamp']:
            print(f"{key}: {value} (type: {type(value).__name__})")

# 3. Vérifier les valeurs None
print("\n\n=== VERIFICATION DES VALEURS NULL ===")
cursor.execute("""
    SELECT 
        COUNT(*) as total,
        COUNT(trend_strength) as trend_strength_ok,
        COUNT(regime_strength) as regime_strength_ok,
        COUNT(signal_strength) as signal_strength_ok,
        COUNT(support_strength) as support_strength_ok,
        COUNT(resistance_strength) as resistance_strength_ok
    FROM analyzer_data
    WHERE symbol = 'BTCUSDC' AND timeframe = '1m'
""")

row = cursor.fetchone()
print(f"Total lignes: {row['total']}")
print(f"trend_strength non-null: {row['trend_strength_ok']}")
print(f"regime_strength non-null: {row['regime_strength_ok']}")
print(f"signal_strength non-null: {row['signal_strength_ok']}")
print(f"support_strength non-null: {row['support_strength_ok']}")
print(f"resistance_strength non-null: {row['resistance_strength_ok']}")

cursor.close()
conn.close()

print("\n\n=== FIN DU DEBUG ===")