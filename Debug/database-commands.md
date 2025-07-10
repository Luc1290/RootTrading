# Database Commands - RootTrading

## Cycle Management

### Check cycle statuses
```bash
docker exec roottrading-db-1 psql -U postgres -d trading -c "SELECT status, COUNT(*) FROM trade_cycles GROUP BY status;"
```

### View active cycles
```bash
docker exec roottrading-db-1 psql -U postgres -d trading -c "
SELECT id, symbol, side, entry_price, min_price, max_price, 
ROUND(((max_price - entry_price) / entry_price * 100)::numeric, 2) as max_gain_pct,
status, DATE_TRUNC('minute', created_at) as created 
FROM trade_cycles WHERE status IN ('waiting_buy', 'waiting_sell');"
```

### Clean up terminated cycles
```bash
# Remove completed, failed, and canceled cycles
docker exec roottrading-db-1 psql -U postgres -d trading -c "DELETE FROM trade_cycles WHERE status IN ('completed', 'failed', 'canceled');"
```

### Performance analysis
```bash
# View best performing completed cycles
docker exec roottrading-db-1 psql -U postgres -d trading -c "
SELECT symbol, side, entry_price, max_price, min_price,
ROUND(((max_price - entry_price) / entry_price * 100)::numeric, 2) as max_gain_pct
FROM trade_cycles WHERE status = 'completed' 
ORDER BY max_gain_pct DESC LIMIT 10;"
```

## Container Management

### List all containers
```bash
docker ps
```

### Access database directly
```bash
docker exec roottrading-db-1 psql -U postgres -d trading
```

### List all tables
```bash
docker exec roottrading-db-1 psql -U postgres -d trading -c "\dt"
```

## Logs Monitoring

### Trader logs (price updates)
```bash
docker logs roottrading-trader-1 --tail 100 | grep "Prix.*USDC"
```

### Coordinator logs (signals)
```bash
docker logs roottrading-coordinator-1 --tail 100 | grep -E "Signal.*BUY|Signal.*SELL"
```

### Gateway logs (market data)
```bash
docker logs roottrading-gateway-1 --tail 50 | grep "SOLUSDC.*1m:"
```
# Guide de Debug RootTrading

## Commandes Base de Données Essentielles

### Accès à la DB PostgreSQL
```bash
# Accéder à la DB via Docker
docker exec roottrading-db-1 psql -U postgres -d trading

# Lister les tables
docker exec roottrading-db-1 psql -U postgres -d trading -c "\dt"

# Voir la structure d'une table
docker exec roottrading-db-1 psql -U postgres -d trading -c "\d trade_cycles"
```

### Requêtes d'Analyse des Cycles

#### Cycles en cours
```sql
SELECT id, symbol, side, entry_price, max_price, min_price, status, 
       DATE_TRUNC('minute', created_at) as created 
FROM trade_cycles 
WHERE status = 'waiting_sell' 
ORDER BY created_at DESC;
```

#### Performance des cycles complétés
```sql
SELECT symbol, side, COUNT(*) as nb_cycles, 
       SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as profitable,
       ROUND(AVG(profit_loss_percent)::numeric, 2) as avg_pct,
       ROUND(MAX(((max_price - entry_price) / entry_price * 100))::numeric, 2) as max_gain_pct
FROM trade_cycles 
WHERE status = 'completed' AND symbol IN ('SOLUSDC', 'XRPUSDC')
GROUP BY symbol, side 
ORDER BY symbol, side;
```

#### Plus gros gains historiques
```sql
SELECT symbol, side, entry_price, max_price, min_price,
       ROUND(((max_price - entry_price) / entry_price * 100)::numeric, 2) as max_gain_pct,
       profit_loss_percent, DATE_TRUNC('day', created_at) as day, status
FROM trade_cycles 
WHERE symbol IN ('SOLUSDC', 'XRPUSDC') AND created_at > NOW() - INTERVAL '7 days'
ORDER BY max_gain_pct DESC LIMIT 20;
```

#### Distribution des trades par heure
```sql
SELECT DATE_TRUNC('hour', created_at) as hour, side, COUNT(*) as trades
FROM trade_cycles 
WHERE symbol = 'SOLUSDC' AND created_at > NOW() - INTERVAL '12 hours' 
      AND status = 'completed'
GROUP BY hour, side 
ORDER BY hour DESC;
```

## Commandes Docker de Debug

### Logs des Services
```bash
# Logs du coordinator (signaux)
docker logs roottrading-coordinator-1 --tail 100 | grep -E "Signal.*BUY|Signal.*SELL|Balance insuffisante"

# Logs du trader (prix et stops)
docker logs roottrading-trader-1 --tail 100 | grep -E "prix.*150\.|Stop.*mis à jour|P&L"

# Logs de l'analyzer (stratégies)
docker logs roottrading-analyzer-1 --tail 100 | grep -E "Bollinger.*Signal|RSI.*Signal"

# Logs du gateway (données market)
docker logs roottrading-gateway-1 --tail 50 | grep -E "SOLUSDC.*1m:"
```

### État des conteneurs
```bash
# Vérifier que tous les services tournent
docker ps | grep roottrading

# Santé des services
docker ps --format "table {{.Names}}\t{{.Status}}"
```

## Analyses de Performance

### Identifier les problèmes de performance
```bash
# Vérifier si les données market arrivent
docker exec roottrading-db-1 psql -U postgres -d trading -c "
SELECT symbol, COUNT(*) as points, 
       MIN(time) as oldest, MAX(time) as newest
FROM market_data 
WHERE time > NOW() - INTERVAL '1 hour'
GROUP BY symbol;"

# Voir l'évolution récente des prix
docker exec roottrading-db-1 psql -U postgres -d trading -c "
SELECT time, symbol, close 
FROM market_data 
WHERE symbol = 'SOLUSDC' AND time > NOW() - INTERVAL '2 hours' 
ORDER BY time DESC LIMIT 20;"
```

### Debugging des Trailing Stops
```bash
# Voir les trailing stops actifs
docker logs roottrading-trader-1 --tail 200 | grep -E "Stop.*mis à jour.*150\.|nouveau max.*150\."

# Vérifier les prix reçus par le trader
docker logs roottrading-trader-1 --tail 100 | grep "Prix SOLUSDC:"
docker logs roottrading-trader-1 --tail 100 | grep "Prix XRPUSDC:"
```


## Data Cleanup Commands

### Preview what data would be deleted (READ-ONLY)
```bash
# Preview cleanup without deleting data
./debug/preview_cleanup.sh

# Or run SQL directly
docker exec -i roottrading-db-1 psql -U postgres -d trading -f - < debug/preview_cleanup.sql
```

### Delete data older than July 10, 2025 10:00 AM
```bash
# Interactive cleanup with confirmation
./debug/cleanup_database.sh

# Or run SQL directly (BE CAREFUL!)
docker exec -i roottrading-db-1 psql -U postgres -d trading -f - < debug/delete_old_data.sql
```

### Manual cleanup commands
```bash
# Delete specific data ranges
docker exec roottrading-db-1 psql -U postgres -d trading -c "
DELETE FROM trade_cycles WHERE created_at < '2025-07-10 10:00:00';
DELETE FROM market_data WHERE time < '2025-07-10 10:00:00';
DELETE FROM event_logs WHERE timestamp < '2025-07-10 10:00:00';
DELETE FROM trading_signals WHERE created_at < '2025-07-10 10:00:00';
DELETE FROM trade_executions WHERE created_at < '2025-07-10 10:00:00';
DELETE FROM performance_stats WHERE start_date < '2025-07-10';
"

# Reclaim disk space after cleanup
docker exec roottrading-db-1 psql -U postgres -d trading -c "VACUUM ANALYZE;"
```

### Check data size and counts
```bash
# Show row counts for all tables
docker exec roottrading-db-1 psql -U postgres -d trading -c "
SELECT 'trade_cycles' as table_name, COUNT(*) as row_count FROM trade_cycles
UNION ALL
SELECT 'market_data', COUNT(*) FROM market_data
UNION ALL
SELECT 'event_logs', COUNT(*) FROM event_logs
UNION ALL
SELECT 'trading_signals', COUNT(*) FROM trading_signals
UNION ALL
SELECT 'trade_executions', COUNT(*) FROM trade_executions
UNION ALL
SELECT 'performance_stats', COUNT(*) FROM performance_stats
ORDER BY table_name;
"

# Show database size
docker exec roottrading-db-1 psql -U postgres -d trading -c "
SELECT pg_size_pretty(pg_database_size('trading')) as database_size;
"
```

---
*Database maintenance commands for RootTrading project*