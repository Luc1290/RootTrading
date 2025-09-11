# Dispatcher Service

## Vue d'ensemble
Le service **Dispatcher** est le point central de routage des messages dans l'écosystème RootTrading. Il reçoit tous les messages Kafka et les route intelligemment vers les destinations Redis appropriées.

## Responsabilités
- **Routage de messages** : Réception et distribution des messages Kafka vers Redis
- **Multi-timeframes** : Support des données de marché sur différentes échelles de temps (1m, 3m, 5m, 15m, 1h, 1d)
- **Monitoring** : Endpoints de santé et diagnostics
- **Persistance** : Sauvegarde des messages via le module database_persister

## Architecture

### Composants principaux
- `main.py` : Point d'entrée et service principal
- `message_router.py` : Logique de routage des messages
- `database_persister.py` : Persistance des données

### Flux de données
```
Kafka Topics → Dispatcher → Redis Channels → Services consommateurs
```

## Configuration

### Variables d'environnement requises
- `KAFKA_BROKER` : Adresse du broker Kafka
- `KAFKA_GROUP_ID` : ID du groupe de consommateurs
- `SYMBOLS` : Liste des symboles cryptos à traiter
- `LOG_LEVEL` : Niveau de logging

### Ports
- **5004** : Port HTTP pour les endpoints de santé

## Topics Kafka consommés
- `market.data.{symbol}.{timeframe}` : Données de marché par timeframe
- `market.data.{symbol}` : Données de marché (format legacy)
- `signals` : Signaux de trading
- `executions` : Exécutions d'ordres
- `orders` : Ordres de trading
- `analyzer.signals` : Signaux d'analyse

## Endpoints API

### Health Check
```http
GET /health
```
Retourne le statut du service et les statistiques de routage.

### Diagnostic
```http
GET /diagnostic
```
Informations détaillées sur les connexions et performances.

## Démarrage

### Avec Docker
```bash
docker-compose up dispatcher
```

### En développement
```bash
cd dispatcher
python src/main.py
```

## Monitoring

### Métriques disponibles
- Uptime du service
- Statut des connexions Kafka/Redis
- Statistiques de routage
- Performance des messages

### Logs
Les logs sont disponibles dans `dispatcher.log` avec différents niveaux de verbosité.

## Dépendances
- **Kafka** : Réception des messages
- **Redis** : Distribution des messages
- **Shared** : Modules communs (config, kafka_client, redis_client)

## Gestion d'erreurs
- Reconnexion automatique aux services
- Gestion gracieuse des arrêts (SIGINT/SIGTERM)
- Logging détaillé des erreurs
- Endpoints de diagnostic pour le troubleshooting