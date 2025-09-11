# Visualization Service

## Vue d'ensemble
Le service **Visualization** est l'interface de monitoring et d'analyse visuelle dans l'écosystème RootTrading. Il fournit des dashboards temps réel, des graphiques interactifs et des statistiques de performance via une interface web moderne.

## Responsabilités
- **Dashboard temps réel** : Interface web pour le monitoring du système
- **Graphiques financiers** : Charts interactifs avec indicateurs techniques
- **Statistiques de performance** : Métriques et KPIs en temps réel
- **WebSocket Hub** : Communication bidirectionnelle pour les updates temps réel
- **API REST** : Endpoints pour l'accès aux données et graphiques

## Architecture

### Composants principaux
- `main.py` : Service principal FastAPI et routage
- `data_manager.py` : Gestionnaire de données multi-sources
- `chart_service.py` : Génération des graphiques financiers
- `websocket_hub.py` : Hub WebSocket pour temps réel
- `statistics_service.py` : Calculs et agrégations statistiques

### Flux de données
```
PostgreSQL/Redis → DataManager → ChartService/StatisticsService
                      ↓
              WebSocketHub ← Frontend (React/Legacy)
                      ↓
                FastAPI REST API
```

## Configuration

### Variables d'environnement requises
- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB` : Configuration PostgreSQL
- `POSTGRES_USER`, `POSTGRES_PASSWORD` : Credentials base de données
- `REDIS_HOST`, `REDIS_PORT` : Configuration Redis
- `VISUALIZATION_PORT` : Port du service (défaut: 5009)

### Ports
- **5009** : Port principal FastAPI

## Interface utilisateur

### Frontend options
- **React App** : Interface moderne (si `frontend/dist` existe)
- **Legacy Templates** : Templates Jinja2 de fallback
- **API-only mode** : Utilisation pure API REST

### Dashboard principal
- Vue d'ensemble du portefeuille
- Graphiques temps réel des positions
- Métriques de performance
- Logs des transactions récentes

## Endpoints API

### Health Check
```http
GET /api/health
```
Statut du service de visualisation.

### Données de marché
```http
GET /api/market-data/{symbol}?timeframe=1h&limit=1000
```
Données OHLCV avec indicateurs techniques.

### Graphiques interactifs
```http
GET /api/chart/{symbol}?timeframe=1d&indicators=sma,rsi
```
Données formatées pour les graphiques (Plotly/Chart.js).

### Statistiques portfolio
```http
GET /api/portfolio/stats?period=7d
```
Métriques de performance sur une période.

### Signaux récents
```http
GET /api/signals/recent?limit=50
```
Derniers signaux de trading générés.

### Positions actives
```http
GET /api/positions/active
```
Vue en temps réel des positions ouvertes.

## WebSocket Endpoints

### Connection temps réel
```javascript
// Connection WebSocket
const ws = new WebSocket('ws://localhost:5009/ws');

// Messages reçus
{
  "type": "market_update",
  "symbol": "BTCUSDT",
  "price": 45000.0,
  "change_24h": 2.5
}

{
  "type": "signal_alert",
  "signal_type": "BUY",
  "symbol": "ETHUSD",
  "strength": 0.85
}

{
  "type": "portfolio_update",
  "total_value": 50000.0,
  "pnl_today": 250.0
}
```

## Services de données

### DataManager
- **Multi-source** : PostgreSQL, Redis, services REST
- **Caching intelligent** : Cache des données fréquentes
- **Real-time updates** : Synchronisation temps réel
- **Error handling** : Gestion robuste des pannes

### ChartService
```python
# Types de graphiques supportés
chart_types = [
    "candlestick",      # Chandeliers japonais
    "line",             # Courbes de prix
    "volume",           # Histogrammes de volume
    "indicators",       # Indicateurs techniques
    "heatmap",          # Cartes de chaleur
    "scatter"           # Graphiques de dispersion
]
```

### StatisticsService
```python
# Métriques calculées
stats = {
    "performance": {
        "total_return": 12.5,
        "sharpe_ratio": 1.85,
        "max_drawdown": -5.2,
        "win_rate": 0.68
    },
    "portfolio": {
        "total_value": 50000.0,
        "available_balance": 45000.0,
        "positions_count": 5,
        "daily_pnl": 250.0
    },
    "trading": {
        "signals_today": 25,
        "trades_executed": 12,
        "success_rate": 0.75,
        "avg_trade_duration": "2h 15m"
    }
}
```

## Graphiques et visualisations

### Types de graphiques
- **Candlestick charts** : Graphiques en chandeliers avec volume
- **Technical indicators** : RSI, MACD, Bollinger Bands, etc.
- **Portfolio composition** : Répartition des actifs
- **Performance timeline** : Évolution des performances
- **Correlation matrix** : Matrice de corrélation des actifs
- **Risk metrics** : Visualisation des métriques de risque

### Indicateurs supportés
```python
indicators = {
    "trend": ["SMA", "EMA", "Hull MA", "VWAP"],
    "momentum": ["RSI", "MACD", "Stochastic", "CCI"],
    "volatility": ["Bollinger Bands", "ATR", "Donchian"],
    "volume": ["OBV", "Volume Profile", "VWAP"]
}
```

## Démarrage

### Avec Docker
```bash
docker-compose up visualization
```

### En développement
```bash
cd visualization/src
python main.py
```

### Interface web
- Accès via http://localhost:5009
- Dashboard automatique si React build disponible
- Fallback templates sinon

## Configuration frontend

### React App (recommandé)
```bash
# Build du frontend React
cd frontend
npm install
npm run build
# → génère frontend/dist/
```

### Templates legacy
- Utilisation de Jinja2 templates
- Interface basique mais fonctionnelle
- Pas de build process requis

## Performance et optimisation

### Optimisations temps réel
- **WebSocket pooling** : Gestion efficace des connexions
- **Data compression** : Compression des données WebSocket
- **Selective updates** : Mise à jour sélective des composants
- **Caching stratégique** : Cache multi-niveaux

### Scalabilité
- **Connection limits** : Limitation des connexions WebSocket
- **Data pagination** : Pagination des gros datasets
- **Lazy loading** : Chargement paresseux des graphiques
- **Resource cleanup** : Nettoyage automatique des ressources

## Monitoring et debugging

### Métriques internes
```json
{
    "websocket_connections": 5,
    "active_subscriptions": 12,
    "charts_rendered": 150,
    "data_cache_hit_rate": 0.85,
    "avg_response_time_ms": 45
}
```

### Debugging
- Logs structurés par composant
- Traces des requêtes WebSocket
- Monitoring des performances de rendu
- Alertes sur les erreurs critiques

## Sécurité

### Protection des données
- CORS configuré pour les origines autorisées
- Validation des inputs WebSocket
- Rate limiting sur les API endpoints
- Sanitization des données affichées

## Dépendances
- **FastAPI** : Framework web moderne et performant
- **WebSockets** : Communication temps réel
- **PostgreSQL** : Source de données principales
- **Redis** : Cache et pub/sub
- **Plotly/Chart.js** : Librairies de graphiques (frontend)

## Architecture technique
```
VisualizationService (FastAPI)
├── DataManager (sources multiples)
├── ChartService (graphiques)
├── StatisticsService (métriques)
├── WebSocketHub (temps réel)
└── Static/Templates (interface)
    ├── React App (moderne)
    └── Jinja2 Templates (fallback)
```

## Extensions possibles
- **Alertes personnalisées** : Notifications configurables
- **Backtesting interface** : Interface de test de stratégies
- **Mobile responsiveness** : Adaptation mobile
- **Export capabilities** : Export PDF/Excel des rapports
- **Multi-timezone support** : Support des fuseaux horaires