# Database Commands - RootTrading

## 🔌 Connexion à la base de données

### Accès rapide via Docker
```bash
# Se connecter à la base trading
docker exec roottrading-db-1 psql -U postgres -d trading

# Exécuter une commande SQL directement
docker exec roottrading-db-1 psql -U postgres -d trading -c "SELECT COUNT(*) FROM market_data;"

# Lister toutes les tables
docker exec roottrading-db-1 psql -U postgres -d trading -c "\dt"

# Voir la structure d'une table
docker exec roottrading-db-1 psql -U postgres -d trading -c "\d market_data"
```

## 📊 Market Data

### État et monitoring
```bash
# Compter les lignes total
docker exec roottrading-db-1 psql -U postgres -d trading -c "SELECT COUNT(*) as total_rows FROM market_data;"

# Voir les données par symbole et timeframe
docker exec roottrading-db-1 psql -U postgres -d trading -c "
SELECT symbol, timeframe, COUNT(*) as points, 
       MIN(time) as oldest, MAX(time) as newest
FROM market_data 
WHERE time > NOW() - INTERVAL '1 hour'
GROUP BY symbol, timeframe
ORDER BY symbol, timeframe;"

# Dernières entrées avec indicateurs
docker exec roottrading-db-1 psql -U postgres -d trading -c "
SELECT time, symbol, close, volume, rsi_14, ema_7, ema_26, macd_line 
FROM market_data 
WHERE symbol = 'SOLUSDC' 
ORDER BY time DESC LIMIT 10;"
```

### Maintenance et nettoyage
```bash
# Vider complètement market_data (ATTENTION!)
docker exec roottrading-db-1 psql -U postgres -d trading -c "TRUNCATE TABLE market_data;"

# Vider pour un symbole spécifique
docker exec roottrading-db-1 psql -U postgres -d trading -c "DELETE FROM market_data WHERE symbol = 'BTCUSDC';"

# Supprimer données anciennes
docker exec roottrading-db-1 psql -U postgres -d trading -c "DELETE FROM market_data WHERE time < NOW() - INTERVAL '30 days';"

# Reclaim disk space après suppression
docker exec roottrading-db-1 psql -U postgres -d trading -c "VACUUM ANALYZE market_data;"
```

## 📈 Cycles de Trading

### Monitoring actif
```bash
# Vue d'ensemble des statuts
docker exec roottrading-db-1 psql -U postgres -d trading -c "SELECT status, COUNT(*) FROM trade_cycles GROUP BY status;"

# Cycles actifs avec détails
docker exec roottrading-db-1 psql -U postgres -d trading -c "
SELECT id, symbol, side, entry_price, min_price, max_price, 
ROUND(((max_price - entry_price) / entry_price * 100)::numeric, 2) as max_gain_pct,
status, DATE_TRUNC('minute', created_at) as created 
FROM trade_cycles 
WHERE status IN ('waiting_buy', 'waiting_sell', 'active_buy', 'active_sell')
ORDER BY created_at DESC;"
```

### Analyse de performance
```bash
# Top 10 meilleurs trades
docker exec roottrading-db-1 psql -U postgres -d trading -c "
SELECT symbol, side, entry_price, exit_price,
ROUND(profit_loss_percent::numeric, 2) as profit_pct,
DATE_TRUNC('day', completed_at) as date
FROM trade_cycles 
WHERE status = 'completed' AND profit_loss > 0
ORDER BY profit_loss_percent DESC LIMIT 10;"

# Performance par symbole
docker exec roottrading-db-1 psql -U postgres -d trading -c "
SELECT symbol, 
       COUNT(*) as total_trades,
       SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
       ROUND(AVG(profit_loss_percent)::numeric, 2) as avg_profit_pct,
       ROUND(MAX(profit_loss_percent)::numeric, 2) as best_trade_pct
FROM trade_cycles 
WHERE status = 'completed'
GROUP BY symbol
ORDER BY avg_profit_pct DESC;"
```

## 🔍 Debug et Logs

### Logs des services
```bash
# Gateway (données market)
docker logs roottrading-gateway-1 --tail 50 | grep -E "SOLUSDC.*1m:|Enrichissement|indicateurs"

# Coordinator (signaux)
docker logs roottrading-coordinator-1 --tail 100 | grep -E "Signal.*BUY|Signal.*SELL|Balance"

# Trader (ordres et stops)
docker logs roottrading-trader-1 --tail 100 | grep -E "prix|Stop.*mis à jour|P&L|Order"

# Analyzer (stratégies)
docker logs roottrading-analyzer-1 --tail 100 | grep -E "Bollinger|RSI|MACD|Signal"
```

### État des conteneurs
```bash
# Vérifier que tous les services tournent
docker ps --format "table {{.Names}}\t{{.Status}}" | grep roottrading

# Santé détaillée
docker ps -a | grep roottrading
```

## ⚙️ TimescaleDB

### Gestion des hypertables
```bash
# Voir les chunks (partitions)
docker exec roottrading-db-1 psql -U postgres -d trading -c "
SELECT hypertable_name, chunk_name, 
       pg_size_pretty(total_bytes) as size
FROM timescaledb_information.chunks 
WHERE hypertable_name = 'market_data'
ORDER BY range_start DESC LIMIT 10;"

# Statistiques de compression
docker exec roottrading-db-1 psql -U postgres -d trading -c "
SELECT * FROM timescaledb_information.compression_stats 
WHERE hypertable_name = 'market_data';"

# Forcer la compression manuelle
docker exec roottrading-db-1 psql -U postgres -d trading -c "
SELECT compress_chunk(i) FROM show_chunks('market_data', older_than => INTERVAL '7 days') i;"
```

## 📏 Tailles et statistiques

### Taille des tables
```bash
# Taille de toutes les tables
docker exec roottrading-db-1 psql -U postgres -d trading -c "
SELECT table_name,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;"

# Taille totale de la base
docker exec roottrading-db-1 psql -U postgres -d trading -c "
SELECT pg_size_pretty(pg_database_size('trading')) as database_size;"
```

### Row counts toutes tables
```bash
docker exec roottrading-db-1 psql -U postgres -d trading -c "
SELECT 'trade_cycles' as table_name, COUNT(*) as row_count FROM trade_cycles
UNION ALL SELECT 'market_data', COUNT(*) FROM market_data
UNION ALL SELECT 'event_logs', COUNT(*) FROM event_logs
UNION ALL SELECT 'trading_signals', COUNT(*) FROM trading_signals
UNION ALL SELECT 'trade_executions', COUNT(*) FROM trade_executions
UNION ALL SELECT 'performance_stats', COUNT(*) FROM performance_stats
UNION ALL SELECT 'portfolio_balances', COUNT(*) FROM portfolio_balances
ORDER BY row_count DESC;"
```

## 🧹 Cleanup Scripts

### Preview cleanup (sans supprimer)
```bash
# Utiliser le script de preview
./debug/preview_cleanup.sh

# Ou directement
docker exec roottrading-db-1 psql -U postgres -d trading -c "
SELECT 'Would delete from trade_cycles:' as info, COUNT(*) as rows 
FROM trade_cycles WHERE created_at < NOW() - INTERVAL '30 days'
UNION ALL
SELECT 'Would delete from market_data:', COUNT(*) 
FROM market_data WHERE time < NOW() - INTERVAL '30 days';"
```

### Cleanup avec confirmation
```bash
# Script interactif
./debug/cleanup_database.sh

# Cleanup manuel par date
docker exec roottrading-db-1 psql -U postgres -d trading -c "
DELETE FROM market_data WHERE time < '2025-07-10 10:00:00';
DELETE FROM event_logs WHERE timestamp < '2025-07-10 10:00:00';
DELETE FROM trading_signals WHERE created_at < '2025-07-10 10:00:00';
VACUUM ANALYZE;"
```

## 📝 Informations de connexion
- **Host**: localhost (ou roottrading-db-1 depuis Docker)
- **Port**: 5432
- **Database**: trading
- **User**: postgres
- **Password**: postgres

## 🏷️ Notes importantes
- La base utilise **TimescaleDB** pour optimiser les séries temporelles
- `market_data` est une **hypertable** avec compression automatique après 7 jours
- Les données sont partitionnées par **chunks de 1 jour**
- Toujours faire un `VACUUM ANALYZE` après de grosses suppressions
- Les indicateurs techniques sont stockés directement dans `market_data`