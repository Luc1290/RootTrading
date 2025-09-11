# Market Analyzer Service

## Vue d'ensemble
Le service **Market Analyzer** est le moteur de calcul des indicateurs techniques dans l'écosystème RootTrading. Il traite les données OHLCV brutes pour générer tous les indicateurs techniques nécessaires aux stratégies de trading.

## Responsabilités
- **Calcul d'indicateurs** : Traitement des données de marché pour générer tous les indicateurs techniques
- **Écoute temps réel** : Traitement en continu des nouvelles données de marché
- **Traitement historique** : Analyse des données passées pour combler les lacunes
- **Optimisation** : Traitement par batch pour des performances maximales
- **Monitoring** : API de santé et statistiques de couverture

## Architecture

### Composants principaux
- `main.py` : Service principal et API REST
- `data_listener.py` : Écoute des données et orchestration
- `indicator_processor.py` : Moteur de calcul des indicateurs

### Flux de données
```
market_data (DB) → Market Analyzer → analyzer_data (DB)
                ↓
        API REST (stats, coverage, process)
```

## Configuration

### Variables d'environnement requises
- `SYMBOLS` : Liste des symboles cryptos à analyser
- `DATABASE_URL` : Connexion à la base TimescaleDB
- `REDIS_*` : Configuration Redis pour la communication

### Ports
- **5020** : Port HTTP pour l'API REST

## Endpoints API

### Health Check
```http
GET /health
```
Retourne le statut du service et la configuration.

### Statistiques
```http
GET /stats
```
Métriques de performance et couverture des données.

### Analyse de couverture
```http
GET /coverage
```
Détails de couverture par symbole et timeframe.

### Traitement historique
```http
POST /process-historical
Content-Type: application/json

{
  "symbol": "BTCUSDT",
  "timeframe": "1m",
  "limit": 1000
}
```
Lance le traitement des données historiques.

## Base de données

### Tables utilisées
- **market_data** : Données OHLCV sources
- **analyzer_data** : Indicateurs calculés

### Indicateurs calculés
- Moyennes mobiles (SMA, EMA, HullMA)
- RSI, MACD, Bollinger Bands
- ADX, CCI, OBV, ParabolicSAR
- Donchian Channels, ATR
- Et plus...

## Démarrage

### Avec Docker
```bash
docker-compose up market_analyzer
```

### En développement
```bash
cd market_analyzer
python src/main.py
```

## Performance

### Optimisations
- **Traitement par batch** : Calcul optimisé par lots
- **Connexions DB async** : Pool de connexions PostgreSQL
- **Limitation automatique** : Max 100k données au démarrage
- **Cache intelligent** : Évite le recalcul des données existantes

### Monitoring
- Couverture en temps réel des analyses
- Statistiques de performance par timeframe
- Détection des données manquantes
- Métriques de débit de traitement

## Gestion d'erreurs
- Reconnexion automatique à la base de données
- Reprise après interruption
- Logging détaillé des performances
- API de diagnostic complète

## Dépendances
- **TimescaleDB** : Stockage des données financières
- **Redis** : Communication inter-services
- **asyncpg** : Accès asynchrone à PostgreSQL
- **numpy/pandas** : Calculs des indicateurs techniques

## Architecture technique
```
DataListener
├── Écoute Redis (données temps réel)
├── Traitement historique (rattrapage)
└── IndicatorProcessor (calculs optimisés)
```