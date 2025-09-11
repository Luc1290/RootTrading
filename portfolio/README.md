# Portfolio Service

## Vue d'ensemble
Le service **Portfolio** est le gestionnaire de portefeuille dans l'écosystème RootTrading. Il synchronise les balances, track les positions, calcule les PnL et fournit une vue unifiée de la performance du portefeuille.

## Responsabilités
- **Synchronisation des balances** : Mise à jour temps réel depuis Binance
- **Tracking des positions** : Suivi des positions ouvertes et fermées
- **Calcul PnL** : Profit & Loss réalisé et non réalisé
- **Métriques de performance** : ROI, Sharpe ratio, drawdown
- **API REST** : Interface web pour consultation et gestion

## Architecture

### Composants principaux
- `main.py` : Service principal et serveur FastAPI
- `binance_account_manager.py` : Interface avec l'API Binance
- `models.py` : Modèles de données et ORM
- `sync.py` : Tâches de synchronisation périodiques
- `redis_subscriber.py` : Écoute des événements de trading
- `api.py` : Routes et endpoints REST

### Flux de données
```
Binance API → Portfolio → Base de données → API REST → Dashboard
     ↑                                         ↓
Redis events ← Trading executions → Sync tasks
```

## Configuration

### Variables d'environnement requises
- `BINANCE_API_KEY` : Clé API Binance (lecture compte)
- `BINANCE_SECRET_KEY` : Clé secrète Binance
- `DATABASE_URL` : Connexion PostgreSQL
- `REDIS_URL` : URL Redis pour les événements
- `LOG_LEVEL` : Niveau de logging

### Ports
- **8000** : Port API REST principal

## Endpoints API

### Portfolio Overview
```http
GET /portfolio/overview
```
Vue d'ensemble : balances, PnL total, positions actives.

### Balances détaillées
```http
GET /portfolio/balances
```
Détail de toutes les balances par asset.

### Positions historiques
```http
GET /portfolio/positions?status=open&limit=50
```
Liste des positions avec filtres (open, closed, all).

### Performance metrics
```http
GET /portfolio/metrics?period=7d
```
Métriques de performance sur une période donnée.

### PnL par symbole
```http
GET /portfolio/pnl-by-symbol
```
Répartition des profits/pertes par symbole.

### Historique des trades
```http
GET /portfolio/trade-history?symbol=BTCUSDT&limit=100
```
Historique détaillé des transactions.

## Fonctionnalités avancées

### Synchronisation temps réel
- **Binance WebSocket** : Balance updates en temps réel
- **Redis Events** : Écoute des exécutions de trades
- **Batch updates** : Optimisation des écritures DB
- **Conflict resolution** : Gestion des données contradictoires

### Calculs de performance
```python
# Métriques calculées automatiquement
{
    "total_pnl": 1250.45,
    "realized_pnl": 850.30,
    "unrealized_pnl": 400.15,
    "roi_percent": 12.5,
    "sharpe_ratio": 1.85,
    "max_drawdown": -5.2,
    "win_rate": 0.68
}
```

### Gestion des positions
- **Position tracking** : Ouverture/fermeture automatique
- **Cost basis** : Calcul du prix moyen pondéré
- **Fees tracking** : Suivi des frais de trading
- **Multi-timeframe** : Positions sur différentes échelles

## Modèles de données

### Balance Model
```python
{
    "asset": "BTC",
    "free": 0.5,
    "locked": 0.0,
    "total": 0.5,
    "usd_value": 22500.0,
    "timestamp": "2024-01-01T12:00:00Z"
}
```

### Position Model
```python
{
    "symbol": "BTCUSDT",
    "side": "LONG",
    "quantity": 0.001,
    "entry_price": 45000.0,
    "current_price": 46000.0,
    "pnl": 1.0,
    "pnl_percent": 2.22,
    "status": "open",
    "open_time": "2024-01-01T12:00:00Z"
}
```

## Synchronisation

### Tâches périodiques
- **Balance sync** : Toutes les 30 secondes
- **Position update** : Toutes les minutes
- **PnL calculation** : Toutes les 5 minutes
- **Performance metrics** : Toutes les heures

### Stratégies de sync
- **Full sync** : Synchronisation complète au démarrage
- **Incremental sync** : Mise à jour des changements uniquement
- **Conflict resolution** : Priorité aux données les plus récentes
- **Retry mechanism** : Retry automatique avec backoff

## Démarrage

### Avec Docker
```bash
docker-compose up portfolio
```

### En développement
```bash
cd portfolio/src
python main.py --host 0.0.0.0 --port 8000
```

## Monitoring et alertes

### Health checks
- Connexion à Binance API
- Synchronisation des données
- Performance de la base de données
- Latence des réponses API

### Métriques de monitoring
```python
{
    "sync_status": "healthy",
    "last_sync": "2024-01-01T12:00:00Z",
    "sync_lag_seconds": 15,
    "api_response_time_ms": 250,
    "database_connection": "ok",
    "redis_connection": "ok"
}
```

### Alertes configurables
- Perte importante détectée (> seuil)
- Échec de synchronisation répété
- Anomalie dans les balances
- Position trop importante (> limite)

## Sécurité et compliance

### Protection des données
- Chiffrement des clés API
- Logs sans informations sensibles
- Access control sur l'API
- Audit trail complet

### Validation des données
- Vérification cohérence des balances
- Détection d'anomalies de PnL
- Validation croisée multi-sources
- Protection contre la corruption de données

## Performance

### Optimisations
- **Connection pooling** : Pool de connexions DB optimisé
- **Batch operations** : Requêtes groupées
- **Caching** : Cache Redis pour les données fréquentes
- **Indexing** : Index DB pour les requêtes rapides

### Scalabilité
- Architecture asynchrone (FastAPI + asyncio)
- Séparation lecture/écriture possible
- Sharding par symbole si nécessaire
- Load balancing sur multiple instances

## Dépendances
- **Binance API** : Source des données de compte
- **PostgreSQL** : Persistance des données portfolio
- **Redis** : Cache et événements temps réel
- **FastAPI** : Framework API REST moderne

## Architecture technique
```
PortfolioService
├── BinanceAccountManager (sync balances)
├── DBManager (persistance)
├── RedisSubscriber (événements)
└── FastAPI App (API REST)
    ├── /portfolio/* (routes portfolio)
    ├── /metrics/* (routes performance)
    └── /admin/* (routes administration)
```