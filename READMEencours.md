# RootTrading - Plateforme de Trading Algorithmique

Système modulaire de trading automatisé pour les cryptomonnaies, fonctionnant sur l'API Binance.

## Architecture

RootTrading utilise une architecture microservices avec les composants suivants:

- **Gateway**: Connecteur WebSocket à Binance, récupère les données de marché en temps réel
- **Analyzer**: Exécute les stratégies de trading pour générer des signaux
- **Trader**: Exécute les ordres sur Binance et gère les cycles de trading
- **Portfolio**: Gère le portefeuille et les poches de capital
- **Frontend**: Interface utilisateur pour le monitoring et la gestion

## Services d'infrastructure

- **Redis**: Bus de communication entre les services
- **Kafka**: Messaging pour les données à haute fréquence
- **PostgreSQL + TimescaleDB**: Stockage persistant des données

## Prérequis

- Docker et Docker Compose
- Compte Binance avec clés API (pour le mode réel)
- Python 3.10+ (pour le développement local)

## Installation

1. Cloner le dépôt:
   ```bash
   git clone https://github.com/yourusername/roottrading.git
   cd roottrading
   ```

2. Configurer les variables d'environnement:
   ```bash
   cp .env.example .env
   # Modifier le fichier .env avec vos informations
   ```

3. Démarrer les services:
   ```bash
   make up
   # ou docker-compose up -d
   ```

## Configuration

Les principales configurations se trouvent dans le fichier `.env`:

- `TRADING_MODE`: `demo` ou `live` pour le mode de trading
- `SYMBOLS`: Liste des paires de trading séparées par des virgules
- `BINANCE_API_KEY` et `BINANCE_SECRET_KEY`: Clés API Binance
- Paramètres des stratégies (RSI, EMA, etc.)

## Utilisation

### Interface Web

L'interface utilisateur est accessible à l'adresse:
```
http://localhost:3000
```

### API REST

Les API des différents services sont accessibles:

- Portfolio API: `http://localhost:8000`
- Trader API: `http://localhost:5002`

### Commandes utiles

```bash
# Voir les logs de tous les services
make logs

# Voir les logs d'un service spécifique
make logs-analyzer

# Redémarrer un service
docker-compose restart analyzer

# Arrêter tous les services
make down
```

## Structure du projet

```
roottrading/
├── .env                  # Variables d'environnement
├── docker-compose.yml    # Configuration Docker Compose
├── Makefile              # Commandes utiles
├── requirements-shared.txt # Dépendances partagées
│
├── gateway/              # Service de connexion à Binance
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── main.py
│       ├── binance_ws.py
│       └── kafka_producer.py
│
├── analyzer/             # Service d'analyse et de stratégies
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── src/
│   │   ├── main.py
│   │   ├── multiproc_manager.py
│   │   ├── redis_subscriber.py
│   │   └── strategy_loader.py
│   └── strategies/
│       ├── base_strategy.py
│       ├── rsi.py
│       ├── bollinger.py
│       └── ema_cross.py
│
├── trader/               # Service d'exécution d'ordres
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── main.py
│       ├── binance_executor.py
│       ├── cycle_manager.py
│       └── order_manager.py
│
├── portfolio/            # Service de gestion du portefeuille
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── api.py
│       ├── main.py
│       ├── models.py
│       └── pockets.py
│
├── shared/               # Modules partagés
│   └── src/
│       ├── config.py
│       ├── enums.py
│       ├── kafka_client.py
│       ├── redis_client.py
│       └── schemas.py
│
└── database/             # Schéma de base de données
    └── schema.sql
```

## Ajouter une nouvelle stratégie

1. Créer un fichier dans `analyzer/strategies/`, en héritant de `BaseStrategy`
2. Implémenter les méthodes requises (`name` et `generate_signal`)
3. Ajouter les paramètres de configuration dans `.env`
4. Redémarrer le service analyzer: `docker-compose restart analyzer`

## Mode démo vs Mode réel

- Mode démo (`TRADING_MODE=demo`): Simule les ordres sans les exécuter réellement
- Mode réel (`TRADING_MODE=live`): Exécute les ordres sur Binance avec votre capital

## Sécurité

- Stockez vos clés API en sécurité, ne les exposez jamais publiquement
- Utilisez des clés API avec les permissions minimales nécessaires
- Pour le trading réel, testez d'abord en démo

## Contribuer

Les contributions sont les bienvenues! Veuillez:

1. Forker le dépôt
2. Créer une branche (`git checkout -b feature/amazing-feature`)
3. Commiter vos changements (`git commit -m 'Add some amazing feature'`)
4. Pousser vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrir une Pull Request

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.


Checklist de ce que nous avons fait
1. Architecture des services

✅ Création de l'architecture microservices
✅ Définition des responsabilités de chaque service
✅ Établissement des flux de communication

2. Configuration partagée

✅ Création du module shared avec configurations communes
✅ Implémentation de la gestion des variables d'environnement
✅ Définition des schémas et énumérations partagés

3. Communication entre services

✅ Client Redis pour communication légère et rapide
✅ Client Kafka pour communication haute performance
✅ Schémas de messages standardisés

4. Gateway (Connecteur Binance)

✅ WebSocket pour connexion temps réel à Binance
✅ Producteur Kafka pour diffuser les données
✅ Gestion des reconnexions et erreurs

5. Analyzer (Moteur d'analyse)

✅ Classe de base pour les stratégies
✅ Système de chargement dynamique des stratégies
✅ Implémentation de la stratégie RSI
✅ Traitement multiprocessus pour analyse parallèle

6. Trader (Exécution)

✅ Exécuteur d'ordres Binance
✅ Gestionnaire de cycles de trading
✅ Gestionnaire d'ordres et de signaux

7. Portfolio (Gestion du capital)

✅ Modèles de données pour le portefeuille
✅ Gestionnaire de poches de capital
✅ API REST pour accès aux données

8. Base de données

✅ Schéma SQL complet avec toutes les tables
✅ Vues et fonctions pour requêtes communes
✅ Procédures stockées pour statistiques

9. Infrastructure Docker

✅ Dockerfiles pour chaque service
✅ Configuration docker-compose
✅ Fichiers requirements.txt pour les dépendances

10. Documentation

✅ README.md avec instructions d'installation
✅ Makefile pour les commandes courantes

11. Corrections

✅ Correction des bugs dans le gestionnaire de cycles
✅ Résolution des problèmes d'imports manquants

Description pour la suite : Emplacement des fichiers et composants
Structure principale du projet
roottrading/
├── .env                    # Variables d'environnement
├── docker-compose.yml      # Configuration des services Docker
├── Makefile                # Commandes utilitaires
├── requirements-shared.txt # Dépendances communes
│
├── shared/src/             # Modules partagés entre services
│   ├── config.py           # Configuration centralisée
│   ├── enums.py            # Énumérations communes
│   ├── redis_client.py     # Client Redis
│   ├── kafka_client.py     # Client Kafka
│   └── schemas.py          # Modèles de données Pydantic
│
├── gateway/                # Service de connexion à Binance
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── main.py         # Point d'entrée
│       ├── binance_ws.py   # WebSocket Binance
│       └── kafka_producer.py # Producteur Kafka
│
├── analyzer/               # Service d'analyse
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── src/
│   │   ├── main.py         # Point d'entrée
│   │   ├── multiproc_manager.py # Gestion multiprocessus
│   │   ├── redis_subscriber.py  # Abonnement Redis
│   │   └── strategy_loader.py   # Chargeur de stratégies
│   └── strategies/
│       ├── base_strategy.py     # Classe de base
│       └── rsi.py               # Stratégie RSI
│
├── trader/                 # Service d'exécution d'ordres
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── main.py         # Point d'entrée et API
│       ├── binance_executor.py  # Exécution sur Binance
│       ├── cycle_manager.py     # Gestion des cycles
│       └── order_manager.py     # Gestion des ordres
│
├── portfolio/              # Service de gestion du portefeuille
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
│       ├── main.py         # Point d'entrée
│       ├── api.py          # API REST FastAPI
│       ├── models.py       # Modèles de données
│       └── pockets.py      # Gestion des poches
│
└── database/              # Base de données
    └── schema.sql         # Schéma SQL
Flux de données dans le système

Données de marché:

Reçues par gateway/src/binance_ws.py
Publiées sur Kafka par gateway/src/kafka_producer.py
Consommées par analyzer/src/redis_subscriber.py


Signaux de trading:

Générés par les stratégies dans analyzer/strategies/
Publiés sur Redis par analyzer/src/redis_subscriber.py
Consommés par trader/src/order_manager.py


Ordres et cycles:

Créés par trader/src/order_manager.py
Exécutés par trader/src/binance_executor.py
Gérés par trader/src/cycle_manager.py
Stockés dans PostgreSQL


Portefeuille et poches:

Gérés par portfolio/src/pockets.py
Exposés via API dans portfolio/src/api.py
Stockés dans PostgreSQL



Pour continuer le développement
Prochaines étapes recommandées:

Ajout de stratégies supplémentaires:

Créer de nouveaux fichiers dans analyzer/strategies/
S'inspirer de rsi.py en héritant de base_strategy.py


Développement du frontend:

Créer un répertoire frontend/ avec une application React
Connecter aux APIs du portfolio et du trader


Mise en place du Risk Manager:

Implémenter les règles de gestion des risques
Connecter au trader pour limiter l'exposition


Tests et validation:

Créer des tests unitaires pour chaque service
Tester le système en mode démo avant activation réelle


Monitoring et logging:

Ajouter des dashboards Grafana
Mettre en place une centralisation des logs



Démarrage du système:

Configuration:
bashcp .env.example .env
# Éditer .env avec vos clés API et paramètres

Lancement:
bashdocker-compose up -d
# ou
make up

Vérification:
bashdocker-compose logs -f gateway
# ou
make logs-gateway

Interface:

Portfolio API: http://localhost:8000
Trader API: http://localhost:5002
Frontend (à développer): http://localhost:3000



Le système est conçu de manière modulaire, ce qui permet de développer ou remplacer n'importe quel composant sans impacter les autres, tant que les interfaces restent compatibles.