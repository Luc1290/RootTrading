# Coordinator Service

## Vue d'ensemble
Le service **Coordinator** est l'orchestrateur central de l'écosystème RootTrading. Il coordonne les interactions entre tous les services, valide les signaux de trading et gère l'exécution intelligente des stratégies.

## Responsabilités
- **Orchestration** : Coordination entre Signal Aggregator, Trader et Portfolio
- **Validation finale** : Vérifications ultimes avant exécution
- **Gestion d'univers** : Sélection dynamique des actifs à trader
- **Trailing Stop Manager** : Gestion des stops intelligents
- **Monitoring global** : Supervision de l'ensemble du système

## Architecture

### Composants principaux
- `main.py` : Service principal et API Flask
- `coordinator.py` : Moteur de coordination principal
- `universe_manager.py` : Gestion de l'univers de trading
- `trailing_sell_manager.py` : Gestionnaire des trailing stops
- `service_client.py` : Interface avec les autres services

### Flux de données
```
Signal Aggregator → Coordinator → Validation → Trader
                       ↓
                 Universe Manager
                       ↓
              Trailing Stop Manager
                       ↓
                Portfolio Sync
```

## Configuration

### Variables d'environnement requises
- `TRADER_API_URL` : URL du service Trader (défaut: http://trader:5002)
- `PORTFOLIO_API_URL` : URL du service Portfolio (défaut: http://portfolio:8000)
- `REDIS_URL` : URL Redis pour la communication inter-services
- `MAX_CONCURRENT_TRADES` : Nombre max de trades simultanés
- `UNIVERSE_SIZE` : Taille de l'univers de trading

### Ports
- **5003** : Port API HTTP principal

## Endpoints API

### Health Check
```http
GET /health
```
Statut général du coordinateur et de ses composants.

### Diagnostic complet
```http
GET /diagnostic
```
Informations détaillées sur tous les services connectés.

### Statut des services
```http
GET /status
```
État de connexion avec Trader, Portfolio et autres services.

### Statistiques globales
```http
GET /stats
```
Métriques de coordination et performance du système.

### Univers de trading
```http
GET /universe
```
Liste actuelle des actifs sélectionnés pour le trading.

## Fonctionnalités principales

### Orchestration des signaux
```python
# Flux de validation
signal → universe_check → risk_validation → trader_execution
```

### Universe Manager
- **Sélection dynamique** : Choix des meilleurs actifs selon critères
- **Rotation automatique** : Mise à jour périodique de l'univers
- **Filtres de qualité** : Volume, liquidité, volatilité
- **Blacklist management** : Exclusion d'actifs problématiques

### Trailing Stop Manager
- **Stops dynamiques** : Ajustement automatique selon la performance
- **Multi-stratégies** : Différents types de trailing selon l'actif
- **Protection drawdown** : Limitation des pertes maximales
- **Profit locking** : Sécurisation des gains

## Validation des signaux

### Contrôles pre-execution
- **Universe membership** : L'actif est-il dans l'univers ?
- **Risk limits** : Respect des limites de risque
- **Portfolio constraints** : Diversification et concentration
- **Market conditions** : Conditions de marché favorables

### Règles de validation
```python
def validate_signal(signal):
    checks = [
        universe_manager.is_tradeable(signal.symbol),
        risk_manager.within_limits(signal),
        portfolio_manager.allows_position(signal),
        market_filter.is_favorable(signal)
    ]
    return all(checks)
```

## Coordination inter-services

### Service Health Monitoring
- **Ping périodique** : Vérification de l'état des services
- **Circuit breaker** : Arrêt automatique si service critique en panne
- **Graceful degradation** : Fonctionnement dégradé si services secondaires indisponibles
- **Auto-recovery** : Reconnexion automatique après récupération

### Communication patterns
- **Request-Response** : API REST synchrone pour les commandes
- **Publish-Subscribe** : Redis pour les événements asynchrones
- **Health checks** : Monitoring continu de l'écosystème
- **Error propagation** : Remontée des erreurs critiques

## Gestion d'univers

### Critères de sélection
```python
universe_criteria = {
    "min_volume_24h": 1000000,    # Volume minimum
    "max_spread_percent": 0.1,     # Spread maximum
    "min_market_cap": 50000000,    # Capitalisation minimum
    "volatility_range": (0.02, 0.15),  # Volatilité acceptable
    "correlation_limit": 0.8       # Limite de corrélation
}
```

### Rotation et mise à jour
- **Analyse quotidienne** : Réévaluation de l'univers
- **Scoring dynamique** : Score de qualité par actif
- **Transition graduelle** : Changements progressifs pour éviter les chocs
- **Performance tracking** : Suivi des performances par actif

## Démarrage

### Avec Docker
```bash
docker-compose up coordinator
```

### En développement
```bash
cd coordinator/src
python main.py --port 5003
```

## Monitoring et métriques

### Métriques de coordination
```json
{
    "signals_processed": 1250,
    "signals_validated": 850,
    "signals_rejected": 400,
    "validation_success_rate": 0.68,
    "active_positions": 5,
    "universe_size": 20,
    "trailing_stops_active": 3
}
```

### Dashboard de santé
- **Service status** : État de tous les services connectés
- **Signal flow** : Flux des signaux en temps réel  
- **Universe health** : Qualité de l'univers de trading
- **Performance overview** : Vue d'ensemble des performances

## Gestion d'erreurs et resilience

### Stratégies de récupération
- **Retry automatique** : Tentatives multiples avec backoff
- **Fallback modes** : Modes de fonctionnement dégradé
- **Circuit breakers** : Protection contre les pannes en cascade
- **Data validation** : Vérification de la cohérence des données

### Logging et alertes
- **Structured logging** : Logs structurés pour l'analyse
- **Error aggregation** : Regroupement des erreurs similaires
- **Alert thresholds** : Seuils d'alerte configurables
- **Performance monitoring** : Suivi des temps de réponse

## Sécurité

### Protection des communications
- **API authentication** : Authentication sur les appels inter-services
- **Rate limiting** : Limitation du débit des requêtes
- **Input validation** : Validation stricte des entrées
- **Audit logging** : Traçabilité complète des actions

## Dépendances
- **Trader Service** : Exécution des ordres
- **Portfolio Service** : Gestion du portefeuille
- **Signal Aggregator** : Source des signaux consolidés
- **Redis** : Communication inter-services
- **PostgreSQL** : Persistance des données de coordination

## Architecture technique
```
CoordinatorService
├── Coordinator (moteur principal)
├── UniverseManager (sélection actifs)
├── TrailingStopManager (stops intelligents)
├── ServiceClient (communication)
└── Flask API (interface REST)
    ├── /health (santé système)
    ├── /stats (métriques)
    ├── /universe (actifs tradés)
    └── /diagnostic (debug)
```

## Configuration avancée

### Paramètres de coordination
```python
coordination_config = {
    "signal_timeout": 30,          # Timeout traitement signal (s)
    "max_pending_signals": 100,    # Signals en attente max
    "validation_threads": 4,       # Threads de validation
    "health_check_interval": 60,   # Fréquence health checks (s)
    "universe_update_interval": 3600  # MAJ univers (s)
}
```