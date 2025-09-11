# Trader Service

## Vue d'ensemble
Le service **Trader** est le moteur d'exécution des ordres dans l'écosystème RootTrading. Il reçoit les signaux consolidés et exécute les ordres de trading sur l'exchange Binance avec gestion des risques intégrée.

## Responsabilités
- **Exécution d'ordres** : Passage des ordres BUY/SELL sur Binance
- **Gestion des risques** : Stop-loss, take-profit, et limitations de position
- **Suivi des positions** : Monitoring des positions ouvertes et PnL
- **API REST** : Interface de contrôle et monitoring
- **Validation pré-trade** : Vérifications avant exécution

## Architecture

### Composants principaux
- `main.py` : Service principal et orchestration
- `order_manager.py` : Gestionnaire principal des ordres
- `binance_executor.py` : Interface avec l'API Binance
- `rest_server.py` : API REST pour le contrôle
- `risk_manager.py` : Gestion des risques et validations

### Flux de données
```
Signaux consolidés → Trader → Validation risques → Binance API → Suivi positions
```

## Configuration

### Variables d'environnement requises
- `BINANCE_API_KEY` : Clé API Binance (trading)
- `BINANCE_SECRET_KEY` : Clé secrète Binance
- `SYMBOLS` : Liste des symboles autorisés au trading
- `MAX_POSITION_SIZE` : Taille max de position par symbole
- `RISK_PERCENTAGE` : Pourcentage de risque max par trade

### Ports
- **5002** : Port API REST principal

## Endpoints API

### Health Check
```http
GET /health
```
Statut du service et connexion à l'exchange.

### Positions ouvertes
```http
GET /positions
```
Liste des positions actuellement ouvertes.

### Historique des ordres
```http
GET /orders?symbol=BTCUSDT&limit=100
```
Historique des ordres avec filtres.

### Balance du compte
```http
GET /balance
```
Soldes disponibles sur l'exchange.

### Exécution manuelle
```http
POST /execute-order
Content-Type: application/json

{
  "signal_type": "BUY",
  "symbol": "BTCUSDT",
  "quantity": 0.001,
  "order_type": "MARKET"
}
```

### Fermeture de position
```http
POST /close-position
Content-Type: application/json

{
  "symbol": "BTCUSDT",
  "reason": "manual_exit"
}
```

## Gestion des ordres

### Types d'ordres supportés
- **MARKET** : Exécution immédiate au prix marché
- **LIMIT** : Ordre à prix limite
- **STOP_LOSS** : Stop-loss automatique
- **TAKE_PROFIT** : Prise de bénéfice automatique

### Stratégies d'exécution
- **TWAP** : Time Weighted Average Price
- **Volume-based** : Adaptation au volume disponible  
- **Smart routing** : Optimisation des frais

## Gestion des risques

### Contrôles pré-trade
- Vérification des fonds disponibles
- Limite de taille de position
- Validation des prix et spreads
- Contrôle de fréquence des ordres

### Protection en temps réel
- Stop-loss dynamique
- Monitoring de la drawdown
- Alerte sur positions importantes
- Circuit breaker en cas d'anomalie

### Règles de risk management
```python
# Exemples de limites
MAX_POSITION_SIZE = 1000  # USDT
STOP_LOSS_PERCENT = 2.0   # 2%
MAX_DAILY_LOSS = 5000     # USDT
MAX_CONCURRENT_POSITIONS = 5
```

## Démarrage

### Avec Docker
```bash
docker-compose up trader
```

### En développement
```bash
cd trader/src
python main.py --port 5002
```

## Monitoring et logging

### Métriques de trading
- Nombre d'ordres exécutés
- Volume total tradé
- PnL réalisé/non réalisé
- Taux de succès des ordres
- Frais de trading total

### Alertes
- Échec d'exécution d'ordre
- Dépassement de limits de risque
- Anomalies de prix
- Problèmes de connectivité

## Sécurité

### Protection des clés API
- Stockage chiffré des credentials
- Utilisation de variables d'environnement
- Logs sans informations sensibles
- Validation des permissions API

### Contrôles d'accès
- Authentication sur API REST
- Whitelisting des IP autorisées
- Rate limiting sur les endpoints
- Audit trail complet

## Interface avec Binance

### API utilisées
- **Spot Trading API** : Ordres spot uniquement
- **Market Data** : Prix temps réel et orderbook
- **Account Information** : Balances et positions
- **Trade History** : Historique des transactions

### Gestion des erreurs
- Retry automatique avec backoff
- Gestion des limites de rate
- Fallback sur différents endpoints
- Notification des erreurs critiques

## Structure des données

### Format des signaux entrants
```json
{
  "signal_type": "BUY|SELL",
  "symbol": "BTCUSDT",
  "strength": 0.8,
  "price": 45000.0,
  "quantity": 0.001,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Format des ordres exécutés
```json
{
  "order_id": "123456789",
  "symbol": "BTCUSDT",
  "side": "BUY",
  "quantity": 0.001,
  "executed_price": 45050.0,
  "status": "FILLED",
  "fees": 0.045,
  "timestamp": "2024-01-01T12:00:01Z"
}
```

## Dépendances
- **Binance API** : Exécution des ordres
- **PostgreSQL** : Persistance des données de trading
- **Redis** : Communication temps réel
- **Signal Aggregator** : Source des signaux consolidés

## Architecture technique
```
TraderService
├── OrderManager (gestion ordres)
├── BinanceExecutor (interface exchange)
├── RiskManager (contrôles risques)
└── RestApiServer (API contrôle)
```