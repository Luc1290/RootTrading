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
