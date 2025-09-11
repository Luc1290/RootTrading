# Gateway Service

## Vue d'ensemble
Le service **Gateway** est le point d'entrée de données dans l'écosystème RootTrading. Il récupère les données de marché depuis Binance et les transmet vers les autres services via Kafka.

## Responsabilités
- **Récupération de données OHLCV** : Collecte des données de marché depuis Binance API
- **WebSocket temps réel** : Écoute des flux de données en temps réel
- **Détection de gaps** : Identification et comblement des données manquantes
- **Multi-timeframes** : Support de multiples échelles de temps (1m, 3m, 5m, 15m, 1h, 1d)
- **Optimisation intelligente** : Récupération sélective des données manquantes uniquement

## Architecture

### Composants principaux
- `main_simple.py` : Service principal et orchestration
- `simple_data_fetcher.py` : Récupération des données historiques
- `simple_binance_ws.py` : WebSocket Binance temps réel
- `gap_detector.py` : Détection et comblement des données manquantes
- `kafka_producer.py` : Publication des données vers Kafka

### Flux de données
```
Binance API/WebSocket → Gateway → Kafka Topics → Dispatcher
```

## Configuration

### Variables d'environnement requises
- `BINANCE_API_KEY` : Clé API Binance
- `BINANCE_SECRET_KEY` : Clé secrète Binance
- `SYMBOLS` : Liste des symboles cryptos à surveiller
- `INTERVAL` : Intervalles de temps par défaut
- `KAFKA_BROKER` : Adresse du broker Kafka

### Ports
- **5010** : Port HTTP pour l'API de santé et contrôle

## Topics Kafka produits
- `market.data.{symbol}.{timeframe}` : Données OHLCV par timeframe
- `market.data.{symbol}` : Données de marché (format legacy)

## Fonctionnalités avancées

### Détection intelligente de gaps
- **Analyse automatique** : Vérification des données manquantes sur 24h
- **Remplissage optimisé** : Récupération sélective des gaps uniquement  
- **Estimation temporelle** : Calcul du temps de synchronisation
- **Fallback automatique** : Récupération classique en cas d'erreur

### SmartDataFetcher
```python
# Mode intelligent avec détection de gaps
if gaps_detected:
    fill_missing_data()
else:
    websocket_only_mode()
```

### WebSocket multi-timeframes
- Connexions simultanées pour tous les symboles
- Agrégation automatique des données 1m vers autres timeframes
- Reconnexion automatique en cas de déconnexion
- Gestion des erreurs de flux

## Endpoints API

### Health Check
```http
GET /health
```
Retourne le statut du service et statistiques de collecte.

### Contrôle manuel
```http
POST /fetch-historical
Content-Type: application/json

{
  "symbol": "BTCUSDT",
  "timeframe": "1m",
  "lookback_hours": 24
}
```

### Statistiques
```http
GET /stats
```
Métriques de performance et couverture des données.

## Démarrage

### Avec Docker
```bash
docker-compose up gateway
```

### En développement
```bash
cd gateway
python src/main_simple.py
```

## Optimisations

### Performance
- **Requêtes par batch** : Récupération optimisée par lots
- **Limitation de débit** : Respect des limites Binance API
- **Cache intelligent** : Évite les requêtes redondantes
- **Parallélisation** : Traitement concurrent des symboles

### Robustesse
- Gestion des timeouts et reconnexions
- Retry automatique avec backoff exponentiel  
- Validation des données reçues
- Logging détaillé pour le debugging

## Gestion d'erreurs
- Reconnexion WebSocket automatique
- Fallback API REST si WebSocket échoue
- Gestion des erreurs de rate limiting Binance
- Sauvegarde des données critiques

## Dépendances
- **Binance API** : Source de données
- **Kafka** : Distribution des messages
- **PostgreSQL** : Vérification des gaps
- **WebSocket** : Données temps réel

## Architecture technique
```
Gateway Service
├── SmartDataFetcher (gaps intelligents)
├── SimpleBinanceWebSocket (temps réel)  
├── GapDetector (analyse manquants)
└── KafkaProducer (publication)
```