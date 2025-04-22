# 📈 RootTrading - Système de Trading Automatisé

RootTrading est une plateforme complète de trading automatisé conçue pour analyser les marchés de crypto-monnaies, générer des signaux de trading, exécuter des trades et gérer un portefeuille de manière autonome. Le système est construit comme une architecture microservices hautement modulaire, permettant une scalabilité, une maintenance et une évolution efficaces.

## 📋 Table des matières

- [Architecture globale](#architecture-globale)
- [Services principaux](#services-principaux)
- [Services secondaires](#services-secondaires)
- [Infrastructure](#infrastructure)
- [Flux de données](#flux-de-données)
- [Configuration et déploiement](#configuration-et-déploiement)
- [API REST](#api-rest)
- [Stratégies de trading](#stratégies-de-trading)
- [Gestion du portefeuille](#gestion-du-portefeuille)
- [Gestion des risques](#gestion-des-risques)
- [Interface utilisateur](#interface-utilisateur)
- [Modes de fonctionnement](#modes-de-fonctionnement)
- [Journalisation et monitoring](#journalisation-et-monitoring)
- [Commandes utiles](#commandes-utiles)
- [Dépannage](#dépannage)

## 🏗️ Architecture globale

RootTrading est construit comme un ensemble de microservices communiquant entre eux via Kafka, Redis et des API REST. Cette architecture permet d'isoler les responsabilités, de scaler indépendamment chaque composant et de maintenir une haute disponibilité.

![Architecture RootTrading](architecture_diagram.png)

## 🔍 Services principaux

### Gateway (Port 5000)

Le Gateway est le point d'entrée des données de marché. Il:
- Se connecte aux WebSockets de Binance pour récupérer les données en temps réel
- Convertit et nettoie les données de marché
- Publie les données sur Kafka pour être consommées par les autres services
- Assure la persistance des connexions et la gestion des reconnexions

**Technologies**: Python, WebSockets, Kafka
**Dépendances**: Kafka, Redis

### Analyzer (Port 5001)

L'Analyzer est le cerveau analytique du système. Il:
- Consomme les données de marché depuis Kafka
- Exécute diverses stratégies de trading sur ces données
- Génère des signaux d'achat/vente lorsque les conditions sont remplies
- Publie les signaux sur Redis pour être traités par le Trader
- Utilise un système multiprocessus pour exécuter les stratégies en parallèle

**Technologies**: Python, NumPy, Pandas, TA-Lib, multiprocessing
**Dépendances**: Redis, Kafka

### Trader (Port 5002)

Le Trader gère l'exécution des ordres. Il:
- Écoute les signaux générés par l'Analyzer
- Valide les signaux selon les règles commerciales et les vérifications de risques
- Crée et gère des cycles de trading (de l'entrée à la sortie)
- Exécute les ordres sur Binance (ou simule en mode démo)
- Gère les stop-loss, take-profit et trailing stops
- Expose une API REST pour le contrôle manuel

**Technologies**: Python, Flask, PostgreSQL
**Dépendances**: Redis, PostgreSQL

### Portfolio (Port 8000)

Le Portfolio gère le suivi des actifs et l'allocation du capital. Il:
- Maintient un registre des soldes d'actifs
- Divise le capital en poches (active, buffer, safety)
- Calcule les métriques de performance
- Optimise l'allocation des fonds
- Expose une API REST pour la visualisation et la gestion

**Technologies**: Python, FastAPI, PostgreSQL, TimescaleDB
**Dépendances**: Redis, PostgreSQL

### Frontend (Port 3000)

Le Frontend fournit une interface utilisateur pour visualiser et contrôler le système. Il:
- Affiche le tableau de bord avec les métriques clés
- Visualise les trades actifs et l'historique
- Permet de créer et gérer des trades manuellement
- Affiche les signaux générés et les performances par stratégie

**Technologies**: React, Recharts, TailwindCSS
**Dépendances**: APIs des autres services

## 🧩 Services secondaires

### Coordinator (Port 5003)

Le Coordinator fait le lien entre les signaux et les exécutions. Il:
- Reçoit les signaux de l'Analyzer via Redis
- Coordonne avec Portfolio pour vérifier la disponibilité des fonds
- Applique des filtres basés sur les conditions de marché
- Transmet les ordres validés au Trader

**Technologies**: Python
**Dépendances**: Redis

### Dispatcher (Port 5004)

Le Dispatcher route les messages entre Kafka et Redis. Il:
- Convertit les messages Kafka en messages Redis et vice-versa
- Assure la compatibilité entre les différents systèmes de messagerie
- Standardise le format des messages

**Technologies**: Python, Kafka, Redis
**Dépendances**: Kafka, Redis

### Logger (Port 5005)

Le Logger centralise la journalisation de tous les services. Il:
- Collecte les logs depuis Kafka et Redis
- Normalise et stocke les logs dans PostgreSQL
- Permet une recherche et une analyse des logs
- Gère la rotation et l'archivage des logs

**Technologies**: Python, PostgreSQL
**Dépendances**: Kafka, Redis, PostgreSQL

### PnL Tracker (Port 5006)

Le PnL Tracker analyse les performances et optimise les stratégies. Il:
- Calcule les métriques de profit et perte
- Génère des rapports de performance
- Optimise les paramètres des stratégies via backtesting
- Exporte les statistiques pour analyse externe

**Technologies**: Python, Pandas, NumPy, PostgreSQL
**Dépendances**: PostgreSQL

### Risk Manager (Port 5007)

Le Risk Manager applique les règles de gestion des risques. Il:
- Surveille l'exposition par actif et par stratégie
- Applique des règles de risque configurables
- Peut limiter ou bloquer les trades en cas de risque élevé
- S'adapte aux conditions de marché

**Technologies**: Python, YAML
**Dépendances**: Redis, PostgreSQL

### Scheduler (Port 5008)

Le Scheduler gère les tâches périodiques et surveille la santé du système. Il:
- Effectue des vérifications de santé régulières
- Génère des rapports sur l'état du système
- Exécute des tâches planifiées (nettoyage, synchronisation)
- Peut redémarrer des services en cas de problème

**Technologies**: Python
**Dépendances**: HTTP vers les autres services

## 🏢 Infrastructure

### Redis (Port 6379)

Redis est utilisé comme broker de messages et cache:
- Canal pour les signaux de trading
- Canal pour les données de marché en temps réel
- État partagé entre les services
- Cache pour les données fréquemment accédées
- Communication publish/subscribe entre services

### Kafka (Port 9092)

Kafka est utilisé pour la distribution des données à haut débit:
- Transport des données de marché brutes
- Journalisation distribuée
- Communication asynchrone entre services
- Tampon pour les pics de charge

### PostgreSQL/TimescaleDB (Port 5432)

La base de données est le stockage persistant du système:
- Historique des trades et des cycles
- Données de marché historiques
- État du portefeuille et des poches
- Métriques de performance
- Utilise TimescaleDB pour optimiser les séries temporelles

## 🔄 Flux de données

1. Le **Gateway** se connecte aux WebSockets de Binance et reçoit les données de marché en temps réel
2. Les données sont publiées sur les topics Kafka spécifiques à chaque symbole
3. Le **Dispatcher** relaie ces données vers Redis pour une consommation plus facile
4. L'**Analyzer** traite ces données via ses différentes stratégies
5. Lorsqu'une condition de trading est remplie, l'**Analyzer** génère un signal
6. Le **Coordinator** reçoit le signal, vérifie sa validité et la disponibilité des fonds via le **Portfolio**
7. Si le signal est validé, un ordre est transmis au **Trader**
8. Le **Trader** crée un cycle de trading et exécute l'ordre sur Binance
9. Le **Portfolio** met à jour les soldes et l'allocation des poches
10. Le **PnL Tracker** calcule et enregistre les performances
11. Le **Risk Manager** surveille continuellement les risques et peut intervenir à tout moment
12. Le **Frontend** visualise toutes ces données et permet le contrôle manuel

## ⚙️ Configuration et déploiement

### Fichier .env

Le fichier `.env` contient toutes les variables de configuration du système:

```
# Mode de trading
TRADING_MODE=demo        # demo ou live

# API Binance
BINANCE_API_KEY=votre_api_key
BINANCE_SECRET_KEY=votre_secret_key

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
CHANNEL_PREFIX=roottrading

# Kafka
KAFKA_BROKER=kafka:9092
KAFKA_GROUP_ID=roottrading

# Base de données PostgreSQL
PGUSER=postgres
PGPASSWORD=postgres
PGDATABASE=trading
PGHOST=db
PGPORT=5432

# Paramètres de trading
SYMBOLS=BTCUSDC,ETHUSDC
INTERVAL=1m
TRADE_QUANTITY=0.00017

# Configuration des poches
POCKET_ACTIVE_PERCENT=60
POCKET_BUFFER_PERCENT=30
POCKET_SAFETY_PERCENT=10

# Logging
LOG_LEVEL=INFO

# Ports des services
GATEWAY_PORT=5000
ANALYZER_PORT=5001
TRADER_PORT=5002
PORTFOLIO_PORT=8000
FRONTEND_PORT=3000
COORDINATOR_PORT=5003
DISPATCHER_PORT=5004
LOGGER_PORT=5005
PNL_TRACKER_PORT=5006
RISK_MANAGER_PORT=5007
SCHEDULER_PORT=5008
```

### Déploiement avec Docker Compose

1. Créez votre fichier `.env` à partir du modèle `.env.exemple`
2. Lancez d'abord l'infrastructure:
   ```bash
   make up-infra
   ```
3. Puis lancez les services principaux:
   ```bash
   make up-gateway
   make up-analyzer
   make up-trader
   make up-portfolio
   make up-frontend
   ```
4. Enfin, lancez les services secondaires:
   ```bash
   docker-compose up -d coordinator dispatcher logger pnl_tracker risk_manager scheduler
   ```

## 📡 API REST

### Portfolio API (Port 8000)

- `GET /summary` - Récupère un résumé du portefeuille
- `GET /balances` - Récupère les soldes actuels
- `GET /pockets` - Récupère l'état des poches de capital
- `PUT /pockets/sync` - Synchronise les poches avec les trades actifs
- `PUT /pockets/allocation` - Met à jour l'allocation des poches
- `POST /pockets/{pocket_type}/reserve` - Réserve des fonds
- `POST /pockets/{pocket_type}/release` - Libère des fonds réservés
- `GET /trades` - Récupère l'historique des trades avec pagination et filtrage
- `GET /performance/{period}` - Récupère les statistiques de performance
- `GET /performance/strategy` - Récupère les performances par stratégie
- `GET /performance/symbol` - Récupère les performances par symbole
- `POST /balances/update` - Met à jour les soldes manuellement

### Trader API (Port 5002)

- `GET /health` - Vérifie l'état du service
- `GET /orders` - Récupère les ordres actifs
- `POST /order` - Crée un ordre manuel
- `DELETE /order/{order_id}` - Annule un ordre existant
- `POST /close/{cycle_id}` - Ferme un cycle de trading

## 📊 Stratégies de trading

RootTrading implémente plusieurs stratégies de trading qui peuvent être exécutées en parallèle:

### RSI (Relative Strength Index)

La stratégie RSI utilise l'indicateur de surachat/survente pour détecter les retournements potentiels:
- Achat lorsque le RSI passe sous le niveau de survente puis remonte
- Vente lorsque le RSI passe au-dessus du niveau de surachat puis redescend
- Paramètres configurables: période RSI, niveaux de surachat/survente

### Bollinger Bands

La stratégie Bollinger utilise les bandes de volatilité:
- Achat lorsque le prix touche la bande inférieure et commence à remonter
- Vente lorsque le prix touche la bande supérieure et commence à redescendre
- Paramètres configurables: période, nombre d'écarts-types

### EMA Cross

La stratégie de croisement de moyennes mobiles exponentielles:
- Achat lorsque l'EMA courte croise l'EMA longue vers le haut
- Vente lorsque l'EMA courte croise l'EMA longue vers le bas
- Paramètres configurables: périodes courte et longue

### Breakout

La stratégie de cassure de niveau:
- Achat lorsque le prix casse à la hausse un niveau de résistance
- Vente lorsque le prix casse à la baisse un niveau de support
- Paramètres configurables: période de recherche, confirmation

### Reversal Divergence

La stratégie de divergence avec les oscillateurs:
- Détecte les divergences entre le prix et les oscillateurs (RSI, MACD)
- Signale les retournements potentiels du marché
- Paramètres configurables: type d'oscillateur, période, seuil

### Ride or React

La stratégie adaptative qui s'ajuste aux conditions de marché:
- Mode "Ride" en tendance forte: laisse courir les positions, filtre les signaux opposés
- Mode "React" en consolidation: plus réactif, prend les profits plus rapidement
- Paramètres configurables: seuils de détection de tendance, périodes d'analyse

## 💼 Gestion du portefeuille

Le système divise le capital en trois types de poches:

### Poche Active (60% par défaut)

- Capital dédié aux trades actifs
- Réserve automatiquement des fonds lors de l'ouverture de trades
- Libère les fonds à la fermeture des trades

### Poche Buffer (30% par défaut)

- Sert de tampon pour augmenter la capacité de trading
- Utilisée lorsque la poche active est épuisée
- Permet d'exploiter les opportunités additionnelles

### Poche Safety (10% par défaut)

- Capital de sécurité non utilisé pour le trading
- Sert de réserve en cas de besoin
- Peut être utilisé pour des situations d'urgence ou des opportunités exceptionnelles

## ⚠️ Gestion des risques

Le Risk Manager applique un ensemble de règles configurables dans `risk_manager/src/rules.yaml`, notamment:

- Limite du nombre maximum de trades actifs simultanés
- Arrêt du trading si la perte quotidienne dépasse un seuil
- Limitation de l'exposition maximale par symbole
- Adaptation aux périodes de volatilité
- Limitation du nombre de trades par jour
- Protection contre les crashs soudains

## 🖥️ Interface utilisateur

Le Frontend fournit:

- Un tableau de bord avec la valeur du portefeuille, performances et allocations
- Visualisation des cycles de trading actifs et historiques
- Graphiques de distribution des signaux
- Interface pour créer et gérer des trades manuellement
- Visualisation des performances par stratégie et par symbole

## 🔄 Modes de fonctionnement

RootTrading peut fonctionner en deux modes:

### Mode Démo

- Simule les exécutions d'ordres sans interaction réelle avec Binance
- Parfait pour tester des stratégies sans risque financier
- Utilise un ensemble de données de marché réelles mais des ordres simulés

### Mode Live

- Exécute réellement les ordres sur Binance
- Nécessite des clés API valides avec les permissions appropriées
- Utilise des fonds réels, donc implique des risques financiers

## 📝 Journalisation et monitoring

Le système utilise plusieurs approches pour la journalisation et le monitoring:

- Logs centralisés via le service Logger
- Métriques de performance stockées dans la base de données
- Vérifications de santé périodiques par le Scheduler
- Alertes en cas de problèmes détectés
- Rapports de performance générés par le PnL Tracker

## 🛠️ Commandes utiles

Le Makefile fournit plusieurs commandes utiles:

```bash
# Services
make build                # Construit toutes les images Docker
make up                   # Démarre tous les services
make down                 # Arrête tous les services
make logs                 # Affiche les logs de tous les services
make ps                   # Liste les services en cours d'exécution
make restart              # Redémarre tous les services
make clean                # Nettoie tout (y compris les volumes)

# Infrastructure
make up-infra             # Démarre uniquement l'infrastructure (Redis, Kafka, PostgreSQL)

# Services spécifiques
make up-gateway           # Démarre le service Gateway
make up-analyzer          # Démarre le service Analyzer
make up-trader            # Démarre le service Trader
make up-portfolio         # Démarre le service Portfolio
make up-frontend          # Démarre le service Frontend

# Logs
make logs-gateway         # Affiche les logs du service Gateway
make logs-analyzer        # Affiche les logs du service Analyzer
make logs-trader          # Affiche les logs du service Trader
make logs-portfolio       # Affiche les logs du service Portfolio

# Base de données
make db-init              # Initialise la base de données
make db-backup            # Sauvegarde la base de données
make db-reset             # Réinitialise la base de données
```

## 🔧 Dépannage

### Connexion à Binance impossible

- Vérifiez vos clés API dans le fichier `.env`
- Assurez-vous que les clés ont les permissions nécessaires
- Vérifiez votre connexion internet

### Services qui ne démarrent pas

- Vérifiez les logs avec `make logs-<service>`
- Assurez-vous que les services dépendants sont en cours d'exécution
- Vérifiez les variables d'environnement dans `.env`

### Problèmes de base de données

- Vérifiez la connexion à PostgreSQL
- Réinitialisez la base de données avec `make db-reset`
- Vérifiez l'espace disque disponible

### Données de marché manquantes

- Vérifiez les logs du Gateway avec `make logs-gateway`
- Assurez-vous que les symboles sont correctement configurés dans `.env`
- Vérifiez la connexion entre Gateway et Kafka

### Signaux non générés

- Vérifiez les logs de l'Analyzer avec `make logs-analyzer`
- Assurez-vous que les stratégies sont correctement configurées
- Vérifiez si les données de marché sont reçues correctement

### Ordres non exécutés

- Vérifiez les logs du Trader avec `make logs-trader`
- Assurez-vous que le mode de trading est correctement configuré
- Vérifiez les soldes disponibles dans les poches

---

## 🔐 Sécurité

N'oubliez pas:
- Ne partagez jamais vos clés API Binance
- Utilisez des clés API avec les permissions minimales nécessaires
- Déployez le système sur un serveur sécurisé
- Sauvegardez régulièrement la base de données
- Commencez avec de petits montants en mode Live

---

## 📜 Licence

Ce projet est sous licence MIT.

---

Ce README fournit une vue d'ensemble technique du système RootTrading. Pour des informations plus détaillées sur chaque composant, consultez la documentation spécifique dans chaque répertoire de service.