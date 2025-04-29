ootTrading – Plateforme de Trading Crypto Automatisé
RootTrading est un système modulaire de trading automatique de crypto-monnaies. Son objectif est d’analyser en continu les marchés, de générer des signaux d’achat/vente basés sur diverses stratégies et d’exécuter les ordres de manière autonome. Le projet vise à créer une architecture microservices évolutive et robuste, permettant de déployer facilement de nouveaux composants ou stratégies. Le système se veut très modulaire et scalable​
github.com
​
github.com
. Chaque service (gestion des données de marché, analyse, exécution d’ordres, gestion de portefeuille, etc.) est isolé, communique via des files de messages (Kafka, Redis) et expose, le cas échéant, une API REST (FastAPI/Flask) pour l’intégration ou le contrôle externe.
Table des matières
Introduction
Architecture générale
Installation
Utilisation
Détail du code
Optimisations & améliorations
FAQ
Introduction
RootTrading se présente comme une plateforme complète de trading automatisé. Conçue pour le trading de cryptomonnaies, elle analyse en temps réel les flux de données des marchés, génère des signaux de trading (achat/vente) et exécute les ordres de façon autonome​
github.com
. L’architecture est hautement modulaire : chaque fonctionnalité est implémentée dans un microservice dédié (Gateway, Analyzer, Trader, Portfolio, etc.), ce qui facilite l’extension du système, la maintenance et la montée en charge​
github.com
​
github.com
. Par exemple, de nouveaux indicateurs ou stratégies peuvent être ajoutés dans le service d’analyse sans toucher au reste du système. La vision est de fournir une base solide et évolutive pour élaborer des stratégies de trading quantitatif, tout en surveillant et gérant le portefeuille de manière intelligente.
Architecture générale
RootTrading repose sur une architecture microservices où chaque composant traite un aspect du workflow de trading​
github.com
. Les microservices communiquent principalement via Kafka (pour les données de marché brutes et les logs) et Redis (pour les signaux rapides), et exposent des API REST pour la configuration et le monitoring. Cette architecture permet d’isoler les responsabilités et de déployer ou scaler chaque service indépendamment​
github.com
. Les composants principaux sont :
Gateway (Port 5000) : point d’entrée des données de marché. Il se connecte aux WebSockets de Binance pour récupérer les cotations en temps réel, normalise ces flux et les publie dans Kafka​
github.com
. Il gère aussi la persistance de la connexion au flux de données (reconnexion automatique). (Techno : Python, WebSockets)
Analyzer (Port 5001) : cœur analytique qui consomme les données de marché depuis Kafka, applique diverses stratégies de trading en parallèle et génère des signaux d’achat/vente lorsque les conditions sont remplies​
github.com
. Les signaux produits sont publiés dans Redis pour être consommés par le Trader. (Techno : Python, NumPy/Pandas, TA-Lib, multiprocessing)
Trader (Port 5002) : gestionnaire des ordres. Il écoute les signaux depuis Redis, valide chaque signal selon des règles métier et de risque, construit le cycle complet d’un trade (ouverture/fermeture), puis envoie les ordres à l’API Binance (ou les simule en mode démo)​
github.com
. Il gère aussi les stop-loss, take-profit, trailing stop, etc., et expose une API REST pour le contrôle manuel des trades. (Techno : Python, Flask, PostgreSQL)
Portfolio (Port 8000) : assure le suivi financier global. Il maintient les soldes de chaque actif (divisés en poches par exemple « active », « safety », « reserve »), calcule les indicateurs de performance, gère l’allocation du capital et fournit des API REST (via FastAPI) pour visualiser les données du portefeuille​
github.com
. Les données sont stockées dans une base Postgres/Timescale pour l’historique et l’analyse. (Techno : Python, FastAPI, PostgreSQL/TimescaleDB)
Frontend (Port 3000) : interface utilisateur en React. Elle affiche un tableau de bord graphique avec les métriques clés, l’historique des trades, les signaux générés par stratégie, et permet d’interagir manuellement avec le système (créer/annuler des ordres, modifier des paramètres, etc.)​
github.com
. (Techno : React, TailwindCSS)
D’autres services dits « secondaires » complètent l’architecture : le Coordinator (coordination des signaux et réservations de fonds), le Dispatcher (conversion et acheminement des messages Kafka ↔ Redis), le Logger (agrégateur de logs centralisés), le PnL Tracker (suivi et optimisation des performances), le Risk Manager (application des règles de risque), et le Scheduler (tâches périodiques et monitoring)​
github.com
​
github.com
. Chaque service est conteneurisé (Docker) et s’appuie sur des middleware : Redis pour le caching de signaux et communication rapide, Kafka pour le flux de données brutes et la journalisation, PostgreSQL/Timescale pour le stockage persistant​
github.com
​
github.com
​
github.com
.
Installation
Prérequis : installer Docker et docker-compose. (Optionnel : Python >=3.10 et Node.js >=18 pour exécuter localement chaque service sans conteneur).
Cloner le dépôt :
bash
Copier
Modifier
git clone https://github.com/Luc1290/RootTrading.git
cd RootTrading
Configuration : copier le fichier modèle d’environnement si nécessaire (.env.example → .env) et définir les variables essentielles : clés d’API Binance, ports (GATEWAY_PORT, PORTFOLIO_PORT, etc.), identifiants PostgreSQL (PGUSER, PGPASSWORD, PGDATABASE), mode démo/production, etc. Par défaut, l’application écoute sur les ports 5000 (Gateway), 5001 (Analyzer), 5002 (Trader), 8000 (Portfolio), 3000 (Frontend), etc., mais vous pouvez personnaliser via .env.
Services tiers : le docker-compose.yml inclut toutes les dépendances :
Redis (image redis:7.0-alpine) servant de broker/cache​
github.com
.
Kafka (image bitnami/kafka:3.4) pour le bus de données​
github.com
. Le script d’initialisation crée automatiquement les topics nécessaires (market.data, logs, etc.).
PostgreSQL/TimescaleDB (image timescale/timescaledb:latest-pg14) pour la base de données centrale​
github.com
. Un schéma SQL (database/schema.sql) initialise les tables (signals de trading, portefeuilles, journaux, etc.).
Lancer le système : exécuter docker-compose up -d. Cela construit et démarre tous les conteneurs RootTrading ainsi que Redis/Kafka/Postgres. Un conteneur kafka-init attend la disponibilité de Kafka avant de créer les topics.
Vérification : utiliser docker ps pour s’assurer que tous les services sont « healthy ». Les logs initiaux s’affichent dans docker-compose logs.
Alternativement, vous pouvez installer manuellement les dépendances Python pour chaque service (pip install -r requirements.txt depuis chaque dossier), et lancer les scripts avec Python. Le dépôt fournit les Dockerfiles et requirements-*.txt correspondants.
Utilisation
Démarrage/arrêt : utilisez docker-compose up -d pour lancer tous les services, et docker-compose down pour tout stopper proprement. Pour exécuter un service isolé, par exemple l’Analyzer, vous pouvez faire docker-compose up -d analyzer.
Accès aux APIs :
Le Gateway (port configurable) ne propose pas d’interface utilisateur, mais on vérifie qu’il se connecte aux flux Binance et publie dans Kafka (voir logs).
L’Analyzer et le Trader démarrent automatiquement en consommant les topics. Aucun point d’entrée manuel n’est prévu – ils tournent en boucle, relançant périodiquement les analyses/ordres.
Le Portfolio expose une API REST (FastAPI) sur son port (par défaut 8000) ; par exemple GET /api/balance pour voir les soldes actuels.
Le Frontend est accessible par défaut sur http://localhost:3000. Il affiche le dashboard et communique avec les APIs du Portfolio, du Trader, etc.
Supervision : chaque service émet des logs sur stdout (capturés par Docker). Vous pouvez visualiser les logs d’un service via docker-compose logs -f <service>. Le service Logger consomme également tous les logs pour les stocker en base (Postgres) et permettre des requêtes centralisées. Le Scheduler effectue des checks de santé périodiques (vérification de la disponibilité des topics Kafka, de Redis, etc.) et publie un rapport global dans les logs.
Commandes utiles :
docker-compose logs gateway pour le flux de données.
docker-compose logs analyzer pour les signaux générés.
docker-compose logs trader pour le suivi des trades.
En base de données PostgreSQL : se connecter (par ex. via pgAdmin) et inspecter les tables trading_signals, event_logs, wallets, etc. (schéma dans [database/schema.sql]).
Exemple de requête API :
bash
Copier
Modifier
curl http://localhost:8000/api/signals?limit=10
renverra les 10 derniers signaux calculés (si l’API a été développée pour cela). De même, le frontend permet de déclencher des actions (ex. création d’un trade manuel) via l’API REST du Trader.
Détail du code
Chaque service est organisé en différents fichiers. Voici un tour d’horizon de l’implémentation principale (pour chaque dossier de service) :
Gateway (gateway/):
gateway/src/main.py : point d’entrée qui établit la connexion au WebSocket de Binance (ex. Binance Spot) et récupère les tickers en direct. Les données brutes sont formatées puis publiées sur les topics Kafka adéquats (ex. market.data.btcusdc)​
github.com
. En cas de déconnexion, il gère la reconnexion.
Analyzer (analyzer/):
analyzer/src/main.py : lance le flux d’analyse en créant un manager multi-processus. Il récupère les topics Kafka des marchés et alimente un service de gestion multi-process (défini en dessous) avec ces données.
analyzer/src/multiproc_manager.py : classe qui coordonne plusieurs processus Python pour exécuter en parallèle différentes stratégies sur les mêmes données (simultanéité augmentée)​
github.com
. Elle segmente le travail et remonte les signaux.
analyzer/src/redis_subscriber.py : (si présent) gère la publication des signaux générés vers Redis, pour consommation immédiate par le Trader.
analyzer/src/strategy_loader.py : charge dynamiquement les stratégies définies dans analyzer/strategies/. Chaque stratégie implémente une fonction d’analyse sur les données de marché.
Trader (trader/):
trader/src/main.py : service principal du trader. Il écoute Redis pour les nouveaux signaux d’achat/vente. À chaque signal, il exécute la logique de trading (vérification des risques, calcul du montant d’achat/vente via PocketManager, envoi d’ordre à Binance)​
github.com
. Il gère le cycle complet d’un trade (ouverture, suivi de stop-loss/take-profit, clôture) et les traite en mode réel ou « dry-run » selon la configuration.
Pas de structure exacte fournie, mais on peut s’attendre à une classe TraderService ou similaire qui encapsule cette logique, utilisant possiblement SQLAlchemy pour stocker les cycles de trade dans PostgreSQL.
Portfolio (portfolio/):
portfolio/src/api.py : application FastAPI qui expose les endpoints de gestion du portefeuille. Par exemple, routes pour consulter le solde de chaque actif, la répartition des poches, l’historique des transactions, etc. (Les détails exacts des routes ne sont pas listés ici, mais l’API est décrite dans le README source pour la partie Portfolio).
portfolio/src/binance_account_manager.py : gère l’interaction avec l’API client Binance (spot/futures). Permet de récupérer les soldes, exécuter des ordres sur le compte, vérifier les états des ordres. C’est un wrapper autour des endpoints API Binance.
portfolio/src/pockets.py : implémente le concept de poches (résultat du système de gestion du capital). Contient notamment la classe PocketManager qui calcule combien de capital allouer à chaque trade/poche et stocke les mouvements. Elle gère les exceptions liées aux poches (par exemple PocketNotFoundError), organise les répartitions (active, sécurité, buffer) et tient à jour le portefeuille total et le capital disponible​
github.com
.
portfolio/src/models.py : définit les modèles de données (SQLAlchemy/Pydantic) pour le portfolio : tables Wallet, Trade, Order, etc. Utilise le schéma initial fourni dans database/schema.sql.
Coordinator (coordinator/):
coordinator/src/main.py : service qui fait le lien entre signaux et gestion financière. Il puise les signaux via Redis et sollicite le PocketManager pour réserver le capital nécessaire avant d’autoriser un trade.
coordinator/src/signal_handler.py : classe SignalHandler (longue, ~2200 lignes) qui traite chaque signal : validation des règles métier, communication avec le portefeuille (envoi aux poches appropriées) et finalement envoi au Trader. On y trouve aussi un circuit breaker pour gérer la fiabilité. Des commentaires indiquent des TODO (par exemple, affiner la logique d’ajustement du trade en fonction du portefeuille total​
github.com
).
coordinator/src/pocket_checker.py : classe PocketChecker qui implémente la logique de réservation/libération de fonds au sein des poches. Elle interroge Binance pour vérifier les balances et informe le Portfolio pour bloquer les fonds pour un trade donné.
Dispatcher (dispatcher/):
dispatcher/src/main.py : démarre le service de dispatching. Il consomme des messages (transactions, signaux, logs…) et les achemine d’un broker à l’autre.
dispatcher/src/message_router.py : classe MessageRouter qui traduit les messages Kafka en appels Redis (et inversement). Par exemple, les signaux de trading produits dans Kafka peuvent être envoyés sur Redis pour le Trader, ou vice-versa pour maintenir la synchro entre files.
Logger (logger/):
logger/src/consumer.py : abonné Kafka/Redis qui collecte tous les logs et métriques (différents topics) et publie éventuellement des métriques.
logger/src/db_exporter.py : classe DBExporter qui lit les logs standardisés et les insère dans la base PostgreSQL (tables event_logs, etc.), pour audit et recherche ultérieure.
logger/src/main.py (s’il existe) : lance les deux processus ci-dessus.
PnL Tracker (pnl_tracker/):
pnl_tracker/src/main.py : service qui calcule les profits & pertes globaux et par stratégie. Il interagit avec la base de données pour récupérer l’historique des trades exécutés et génère des rapports de performance. Il ajuste aussi (automatiquement ou semi-automatiquement) les paramètres de stratégie via backtests (inclusion/exclusion).
pnl_tracker/src/strategy_tuner.py et pnl_tracker/src/pnl_logger.py : outils internes pour tuner les hyperparamètres de stratégie et logger les résultats de backtests.
Risk Manager (risk_manager/):
risk_manager/src/main.py : service d’application des règles de risque. Il lit un fichier rules.yaml configurable, surveille les positions du portefeuille et bloque ou ajuste les ordres en fonction de seuils (exposition maximum, drawdown, etc.).
risk_manager/src/checker.py : implémente la logique concrète d’évaluation de chaque règle (par exemple, limiter le levier par paire, maximum de perte journalière, etc.). C’est un ensemble de fonctions d’évaluation sur les données du portefeuille.
Scheduler (scheduler/):
scheduler/src/health_check.py : vérifie périodiquement la santé de tous les services (connexion Kafka, intégrité de la DB, quotas Binance, etc.) et alerte en cas de dérive.
scheduler/src/main.py : programme principal qui lance les jobs planifiés et le monitoring.
scheduler/src/monitor.py : (s’il existe) gère la remontée d’alertes et le redémarrage automatique de services défaillants.
Frontend (frontend/):
Fichiers principaux : App.jsx, main.jsx – composants React pour l’interface utilisateur.
src/components/ : composants UI (tableaux de bord, graphiques, formulaires de configuration de trades).
src/api/ : clients front-end pour appeler les API REST du portfolio et du trader (hooks React Query comme useCycle.js, usePortfolio.js, useSignals.js).
Chaque fichier et classe a été conçu pour un rôle unique et interagit par messages avec les autres composants, comme résumé ci-dessus.
Optimisations & améliorations
Bonnes pratiques et points à renforcer :
Gestion d’erreurs : plusieurs blocs try/except Exception as e capturent toutes les exceptions, ce qui peut masquer des problèmes spécifiques. Il faudrait affiner ces blocs (ne capturer que les exceptions attendues) et enrichir la journalisation avec des niveaux adéquats. Par exemple, dans PocketManager et SignalHandler, de nombreux except Exception généraux existent​
github.com
.
Configuration centralisée : de nombreux paramètres (Ports, clés API, seuils de risque) sont gérés via .env ou fichiers YAML, mais certains sont codés en dur. Il serait utile de centraliser toute la configuration (p.ex. avec Pydantic/BaseSettings) pour éviter la duplication.
Tests automatisés : le dépôt ne contient pas de tests unitaires ou d’intégration. Ajouter des tests (pytest) pour valider les stratégies de trading, la logique de répartition de capital (PocketManager), et la connectivité Kafka/Redis améliorerait la fiabilité.
Structures de données : l’utilisation de Redis pour les signaux pourrait être optimisée (p.ex. utiliser des streams Redis plutôt que des listes simples). Le code multiprocessing du Analyzer peut présenter des goulots en partageant des queues Python — on pourrait envisager multiprocessing.Queue ou des workers plus légers (asyncio) pour de gros volumes.
Sécurité : s’assurer que les clés d’API Binance ne sont pas exposées. Idéalement, utiliser un système de secrets manager ou Docker secrets plutôt que .env en clair.
Évolutivité Kafka : le docker-compose utilise Kafka en mode KRaft sur un seul nœud. Pour du scale, passer à un cluster Kafka (multi-broker + Zookeeper).
Erreurs détectées : dans la classe SignalHandler, un TODO indique que la logique d’ajustement du montant des trades devrait être enrichie (« baser sur le portefeuille total et les limites de risque par trade »)​
github.com
. De même, vérifier la synchronisation entre les fonds réservés et réels (gestion des cas où Binance rejette un ordre) reste à fiabiliser.
Interface et supervision : le Frontend actuel semble basique. On pourrait ajouter plus d’indicateurs temps réel (WebSocket au lieu de polling REST), et des alertes visuelles (notifications). Un système de monitoring comme Grafana/Prometheus pourrait être intégré pour remplacer une simple vérification par logs.
FAQ
Q : Comment configurer mes clés Binance et le mode de trading (réel vs démo) ?
R : Définissez vos clés d’API Binance (API_KEY, API_SECRET) dans le fichier .env (ou via Docker secrets). Pour un mode démo, activez le flag correspondant dans .env (par exemple DEMO_MODE=true) ; le Trader utilisera alors des ordres simulés plutôt qu’un vrai compte. Assurez-vous que votre compte Binance autorise les connexions API.
Q : Le système ne récupère pas les données de marché / les services ne démarrent pas. Que vérifier ?
R : Vérifiez d’abord que Redis et Kafka sont bien opérationnels (via docker-compose ps ou docker-compose logs). Laissez Kafka terminer l’init (le conteneur kafka-init prend quelques secondes pour créer les topics). Vérifiez aussi votre configuration de réseau : par défaut, les services communiquent sur le réseau Docker interne. Les logs du service Gateway indiqueront s’il y a un problème de connexion à l’API Binance.
Q : Comment lancer une stratégie de trading personnalisée ?
R : Déposez votre script de stratégie dans analyzer/strategies/. Il doit fournir une fonction d’analyse qui prend les ticks du marché et retourne un signal. Le Analyzer charge automatiquement toutes les stratégies présentes dans ce dossier (via strategy_loader.py) au démarrage. Redémarrez le service Analyzer pour prendre en compte la nouvelle stratégie.
Q : Les trades ne se déclenchent pas malgré des signaux.
R : Plusieurs causes possibles : d’abord, le Coordinator peut bloquer le signal s’il n’y a pas assez de fonds disponibles. Vérifiez que le Portfolio indique suffisamment de capital (table wallets en DB). Ensuite, le Risk Manager peut rejeter le trade selon les règles (consultez rules.yaml). Enfin, consultez les logs (docker-compose logs trader) pour voir s’il y a eu des erreurs lors de l’envoi d’ordre à Binance (limites de compte, erreur réseau, etc.).
Q : Comment superviser le système en temps réel ?
R : Les logs unifiés sont stockés en base via le service Logger. Vous pouvez lancer une requête SQL dans la table event_logs pour rechercher des erreurs globales. Le Frontend affiche aussi l’état des derniers trades et signaux. Pour plus de monitoring avancé, il est recommandé d’ajouter des métriques exposées (p.ex. avec Prometheus) ou de configurer des alertes (Webhook, e-mail) dans le Scheduler ou un outil externe.
Q : Peut-on ajouter de nouvelles sources de données ou brokers ?
R : Oui. Le Gateway est conçu pour Binance (WebSocket). Pour d’autres exchanges, il faudrait implémenter une nouvelle source dans gateway/src qui publie vers Kafka. Côté exécution, il faudrait adapter BinanceAccountManager ou créer un équivalent pour un autre broker. L’architecture microservices permet d’intégrer ces changements sans tout reconfigurer.
Ce README synthétise le fonctionnement détaillé de RootTrading. Il décrit l’architecture globale, l’installation pas-à-pas, l’utilisation des différents services, et donne un aperçu ligne-à-ligne du code. L’esprit est de rester transparent sur les limites actuelles (erreurs connues, TODO) tout en soulignant la possibilité d’évolution du système vers de nouvelles fonctionnalités et optimisations.


## Table des matières

1. [Architecture](#architecture)
2. [Microservices](#microservices)
3. [Base de données](#base-de-données)
4. [Middleware](#middleware)
5. [Flux de données](#flux-de-données)
6. [Gestion du capital](#gestion-du-capital)
7. [Stratégies](#stratégies)
8. [Installation et déploiement](#installation-et-déploiement)
9. [Configuration](#configuration)
10. [API REST](#api-rest)
11. [Optimisations possibles](#optimisations-possibles)
12. [Glossaire](#glossaire)
13. [Dépannage](#dépannage)

## Architecture

RootTrading est construit sur une architecture microservices où chaque composant a une responsabilité spécifique et communique avec d'autres services à travers Kafka et Redis. Cette conception permet une évolutivité horizontale et une résilience améliorée.

### Vue d'ensemble

Le système est composé des microservices suivants qui travaillent ensemble pour créer un pipeline complet de trading:

- **Gateway**: Collecte des données en temps réel de Binance et envoi vers Kafka
- **Dispatcher**: Routage des messages Kafka vers Redis
- **Analyzer**: Analyse des données de marché et génération de signaux
- **Trader**: Exécution des ordres sur Binance
- **Portfolio**: Gestion des actifs et du capital
- **Coordinator**: Coordination entre signaux et allocation des ressources
- **Risk Manager**: Surveillance et contrôle des risques
- **Logger**: Centralisation des logs du système
- **PNL Tracker**: Suivi des profits et pertes
- **Scheduler**: Planification des tâches périodiques

### Technologies principales

- **Backend**: Python 3.10+
- **Message queue**: Apache Kafka
- **Pub/Sub**: Redis
- **Base de données**: PostgreSQL avec TimescaleDB pour les séries temporelles
- **Containerisation**: Docker et Docker Compose
- **API**: REST avec Flask/FastAPI
- **Web frontend**: React

## Microservices

### Gateway

Le Gateway est le point d'entrée principal des données de marché dans le système. Il se connecte aux WebSockets de Binance pour obtenir des données en temps réel et les publie dans Kafka.

**Fichiers principaux**:
- `gateway/src/main.py` - Point d'entrée
- `gateway/src/binance_ws.py` - Client WebSocket Binance
- `gateway/src/kafka_producer.py` - Producteur Kafka
- `gateway/src/historical_data_fetcher.py` - Récupération des données historiques

**Fonctionnalités**:
- Connexion websocket à Binance avec gestion des reconnexions
- Publication des données sur Kafka
- Récupération de données historiques au démarrage
- Endpoints HTTP pour le monitoring et la santé du service

**Classes clés**:
- `BinanceWebSocket`: Gère la connexion WebSocket à Binance et la réception des données
- `KafkaProducer`: Publie les données sur Kafka
- `HistoricalDataFetcher`: Récupère les données historiques de l'API REST Binance

**Points d'attention**:
- Le heartbeat du WebSocket est vérifié régulièrement pour assurer la connectivité
- Les données historiques sont récupérées pour initialiser les stratégies au démarrage
- Les reconnexions sont gérées avec un backoff exponentiel

### Dispatcher

Le Dispatcher fait le pont entre Kafka et Redis. Il permet de prendre les messages de Kafka et de les republier sur les canaux appropriés de Redis pour consommation par les autres services.

**Fichiers principaux**:
- `dispatcher/src/main.py` - Point d'entrée
- `dispatcher/src/message_router.py` - Routage des messages

**Fonctionnalités**:
- Consommation des messages Kafka
- Transformation et enrichissement des messages
- Publication sur les canaux Redis appropriés
- File d'attente interne pour la résilience en cas de problème avec Redis

**Classes clés**:
- `MessageRouter`: Gère la logique de routage des messages entre Kafka et Redis
- `DispatcherService`: Service principal qui coordonne l'activité du dispatcher

**Points d'attention**:
- Utilise une file d'attente en mémoire pour gérer les cas où Redis est indisponible
- Transforme les messages pour assurer la compatibilité entre les systèmes
- Fournit des statistiques de routage pour le monitoring

### Analyzer

L'Analyzer est le cerveau du système responsable de l'analyse des données de marché et de la génération des signaux de trading basés sur différentes stratégies.

**Fichiers principaux**:
- `analyzer/src/main.py` - Point d'entrée
- `analyzer/src/multiproc_manager.py` - Gestion des processus d'analyse parallèles
- `analyzer/src/redis_subscriber.py` - Abonnement aux données Redis
- `analyzer/src/strategy_loader.py` - Chargement dynamique des stratégies
- `analyzer/strategies/base_strategy.py` - Classe de base pour toutes les stratégies

**Fonctionnalités**:
- Analyse des données de marché avec diverses stratégies
- Support multiprocessing pour maximiser les performances
- Génération de signaux de trading
- Publication des signaux sur Redis

**Classes clés**:
- `AnalyzerManager`: Gère les processus/threads d'analyse
- `StrategyLoader`: Découvre et charge dynamiquement les stratégies
- `BaseStrategy`: Classe abstraite dont toutes les stratégies héritent
- `RedisSubscriber`: Gère les abonnements Redis

**Points d'attention**:
- L'analyse peut être parallélisée sur plusieurs cœurs
- Les stratégies sont chargées dynamiquement à partir du répertoire strategies/
- Chaque stratégie analyse indépendamment les données pour son symbole

### Trader

Le Trader est responsable de l'exécution des ordres sur Binance, de la gestion des cycles de trading et du suivi des positions.

**Fichiers principaux**:
- `trader/src/main.py` - Point d'entrée
- `trader/src/order_manager.py` - Gestion des ordres
- `trader/src/cycle_manager.py` - Gestion des cycles de trading
- `trader/src/binance_executor.py` - Exécution des ordres sur Binance

**Fonctionnalités**:
- Traitement des signaux de trading
- Exécution des ordres sur Binance
- Gestion des cycles de trading (entrée, sortie)
- Stop-loss et Take-profit automatiques
- Mode démo disponible pour le test sans risque

**Classes clés**:
- `OrderManager`: Gère les signaux entrants et crée les cycles
- `CycleManager`: Gère l'état et le cycle de vie des positions
- `BinanceExecutor`: Exécute les ordres sur Binance
- `TraderService`: Coordonne le fonctionnement du service

**Points d'attention**:
- Prend en charge les ordres au marché et à cours limité
- Gère les stops et les targets automatiquement
- Fournit un mode démo pour tester sans passer d'ordres réels
- Vérifie les limites minimales de Binance (taille d'ordre, prix, etc.)

### Portfolio

Le Portfolio gère le suivi des actifs, des balances et des allocations de capital entre différentes poches.

**Fichiers principaux**:
- `portfolio/src/main.py` - Point d'entrée
- `portfolio/src/api.py` - API REST
- `portfolio/src/models.py` - Modèles de données
- `portfolio/src/pockets.py` - Gestion des poches
- `portfolio/src/binance_account_manager.py` - Synchronisation avec le compte Binance

**Fonctionnalités**:
- Suivi des soldes du portefeuille
- Gestion des poches de capital (active, buffer, safety)
- Évaluation des actifs en USDC
- Synchronisation avec le compte Binance
- Statistiques de performance

**Classes clés**:
- `PortfolioModel`: Gère les données du portefeuille
- `PocketManager`: Gère les poches de capital
- `DBManager`: Gère les interactions avec la base de données
- `BinanceAccountManager`: Synchronise avec le compte Binance
- `SharedCache`: Cache pour améliorer les performances

**Points d'attention**:
- Utilise un système de poches pour allouer le capital (active, buffer, safety)
- Maintient un cache pour réduire les requêtes à la base de données
- Synchronise régulièrement les données avec Binance
- Publie des notifications lors des mises à jour

### Coordinator

Le Coordinator orchestre le flux entre signaux et exécution des ordres, en vérifiant notamment les ressources disponibles.

**Fichiers principaux**:
- `coordinator/src/main.py` - Point d'entrée
- `coordinator/src/signal_handler.py` - Gestion des signaux
- `coordinator/src/pocket_checker.py` - Vérification des poches

**Fonctionnalités**:
- Gestion des signaux de trading
- Vérification de la disponibilité des fonds
- Réservation des fonds dans les poches
- Filtrage des signaux selon les conditions de marché

**Classes clés**:
- `SignalHandler`: Traite les signaux et gère les ressources
- `PocketChecker`: Vérifie et réserve les ressources dans les poches
- `CoordinatorService`: Service principal qui coordonne l'activité

**Points d'attention**:
- S'assure que suffisamment de capital est disponible avant d'exécuter un ordre
- Implémente une logique de circuit breaker pour éviter les appels en cascade aux services en échec
- Filtre certains signaux basés sur les conditions de marché (mode "Ride or React")

### Risk Manager

Le Risk Manager surveille l'ensemble du système pour appliquer les règles de gestion des risques et intervenir si nécessaire.

**Fichiers principaux**:
- `risk_manager/src/main.py` - Point d'entrée
- `risk_manager/src/checker.py` - Vérification des règles
- `risk_manager/src/rules.yaml` - Configuration des règles

**Fonctionnalités**:
- Vérification des règles de gestion de risque
- Déclenchement d'actions en cas de risque excessif
- Monitoring des métriques du système

**Classes clés**:
- `RuleChecker`: Évalue les règles et déclenche les actions
- `CircuitBreaker`: Évite les appels répétés aux services en échec

**Points d'attention**:
- Les règles sont définies dans un fichier YAML
- Différents types de règles: exposition, drawdown, volatilité, etc.
- Différentes actions possibles: pause des trades, désactivation, etc.

## Base de données

RootTrading utilise PostgreSQL avec l'extension TimescaleDB pour le stockage des données, en particulier pour les séries temporelles comme les données de marché.

### Schéma de la base de données

Le schéma principal est défini dans `database/schema.sql` et inclut les tables suivantes:

1. **trade_executions** - Enregistre les exécutions d'ordres
   - `order_id`: Identifiant unique de l'ordre
   - `symbol`: Symbole de trading (ex: BTCUSDC)
   - `side`: Côté de l'ordre (BUY, SELL)
   - `status`: Statut de l'ordre
   - `price`: Prix d'exécution
   - `quantity`: Quantité
   - `quote_quantity`: Montant total en devise de cotation
   - Autres informations sur l'exécution

2. **trade_cycles** - Suivi des cycles complets de trading
   - `id`: Identifiant unique du cycle
   - `symbol`: Symbole de trading
   - `strategy`: Stratégie utilisée
   - `status`: État du cycle
   - `entry_price`, `exit_price`: Prix d'entrée et de sortie
   - `profit_loss`, `profit_loss_percent`: Performance du cycle
   - Autres informations de gestion du cycle

3. **market_data** - Données de marché (chandeliers)
   - Table hypertable TimescaleDB pour les séries temporelles
   - `time`: Horodatage
   - `symbol`: Symbole de trading
   - `open`, `high`, `low`, `close`: Données OHLC
   - `volume`: Volume d'échanges

4. **portfolio_balances** - Soldes du portefeuille
   - `asset`: Actif (ex: BTC, ETH)
   - `free`: Montant disponible
   - `locked`: Montant verrouillé dans des ordres
   - `total`: Montant total
   - `value_usdc`: Valeur en USDC

5. **capital_pockets** - Poches de capital
   - `pocket_type`: Type de poche (active, buffer, safety)
   - `allocation_percent`: Pourcentage d'allocation
   - `current_value`: Valeur totale actuelle
   - `used_value`: Valeur utilisée
   - `available_value`: Valeur disponible

6. **performance_stats** - Statistiques de performance
   - Statistiques agrégées par période, stratégie, symbole, etc.

### Optimisations de base de données

Le schéma inclut plusieurs optimisations:

- Utilisation de TimescaleDB pour les données de séries temporelles
- Indices appropriés pour les requêtes fréquentes
- Compression des données historiques
- Vues matérialisées pour les requêtes d'analyse fréquentes
- Procédures stockées pour les opérations complexes

### Connexion à la base de données

La connexion à la base de données est gérée par `DBManager` dans `portfolio/src/models.py` qui fournit:

- Pool de connexions pour une utilisation efficace des ressources
- Retry automatique pour les opérations critiques
- Requêtes paramétrées pour prévenir les injections SQL
- Transactions pour garantir l'intégrité des données

## Middleware

### Kafka

Kafka est utilisé comme message broker pour transmettre les données de marché du Gateway aux autres services.

**Configuration**:
- Les topics Kafka sont définis dans `shared/src/config.py`
- Les producteurs et consommateurs sont implémentés dans `shared/src/kafka_client.py`

**Topics principaux**:
- `market.data.<symbol>` - Données de marché par symbole
- `signals` - Signaux de trading
- `executions` - Exécutions d'ordres
- `orders` - Ordres envoyés
- `logs.info`, `logs.error`, `logs.debug` - Logs du système

### Redis

Redis est utilisé pour la communication pub/sub en temps réel entre les services et comme cache.

**Configuration**:
- La configuration Redis est définie dans `shared/src/config.py`
- Le client Redis est implémenté dans `shared/src/redis_client.py`

**Canaux principaux**:
- `roottrading:market:data:<symbol>` - Données de marché en temps réel
- `roottrading:analyze:signal` - Signaux générés par l'analyzer
- `roottrading:trade:execution` - Exécutions d'ordres
- `roottrading:trade:order` - Ordres à exécuter
- `roottrading:order:failed` - Notifications d'échec d'ordre
- `roottrading:notification:balance_updated` - Notifications de mise à jour de solde

## Flux de données

1. **Acquisition des données**:
   - Le Gateway se connecte aux WebSockets de Binance
   - Les données de marché en temps réel sont reçues
   - Ces données sont publiées sur Kafka

2. **Acheminement des données**:
   - Le Dispatcher consomme les messages Kafka
   - Les messages sont transformés et publiés sur Redis

3. **Analyse et signaux**:
   - L'Analyzer s'abonne aux canaux Redis pour recevoir les données
   - Les stratégies chargées analysent les données
   - Des signaux de trading sont générés et publiés sur Redis

4. **Coordination et vérification**:
   - Le Coordinator reçoit les signaux
   - Le PocketChecker vérifie la disponibilité des fonds
   - Le Risk Manager vérifie les règles de risque

5. **Exécution des ordres**:
   - Le Trader reçoit les ordres validés
   - Les ordres sont envoyés à Binance via l'API
   - Les résultats d'exécution sont enregistrés

6. **Suivi et mise à jour**:
   - Le Portfolio est mis à jour avec les nouvelles positions
   - Les poches de capital sont ajustées
   - Le PNL Tracker calcule les performances

## Gestion du capital

RootTrading utilise un système de "poches" pour allouer et gérer le capital:

### Types de poches

1. **Active** (60% par défaut):
   - Capital dédié aux trades actifs quotidiens
   - Première poche utilisée pour les nouveaux trades

2. **Buffer** (30% par défaut):
   - Poche tampon utilisée quand la poche active est insuffisante
   - Permet de saisir des opportunités supplémentaires

3. **Safety** (10% par défaut):
   - Réserve de sécurité, utilisée uniquement dans des cas spéciaux
   - Protection contre les drawdowns importants

### Mécanisme de réservation

Lorsqu'un trade est initié:
1. Le capital nécessaire est calculé
2. Le PocketChecker vérifie la disponibilité dans la poche appropriée
3. Les fonds sont réservés pour le cycle de trading
4. À la fin du cycle, les fonds (avec profit/perte) sont libérés

### Synchronisation et réconciliation

- Synchronisation régulière avec les trades actifs
- Réconciliation périodique pour corriger les désynchronisations
- Réallocation basée sur la valeur totale du portefeuille

## Stratégies

Les stratégies de trading sont implémentées comme des classes Python qui héritent de `BaseStrategy` définie dans `analyzer/strategies/base_strategy.py`.

### Structure d'une stratégie

Chaque stratégie doit implémenter:
- Une propriété `name` qui renvoie un nom unique
- Une méthode `generate_signal()` qui analyse les données et génère des signaux

### Chargement dynamique

Les stratégies sont chargées dynamiquement au démarrage de l'Analyzer:
- Tous les fichiers Python dans le répertoire `analyzer/strategies/` sont examinés
- Les classes qui héritent de `BaseStrategy` sont instanciées
- Chaque stratégie est initialisée pour chaque symbole configuré

### Filtrage et gestion des signaux

Les signaux générés par les stratégies peuvent être filtrés:
- Par force du signal (WEAK, MODERATE, STRONG, VERY_STRONG)
- Par règles du Risk Manager
- Par disponibilité des fonds
- Par mode de marché (Ride or React)

## Installation et déploiement

### Prérequis

- Docker et Docker Compose
- Python 3.10+
- Clés API Binance (pour le trading réel)

### Installation

1. Cloner le dépôt:
```bash
git clone https://github.com/Luc1290/RootTrading.git
cd RootTrading
```

2. Configurer les variables d'environnement:
```bash
cp .env.example .env
# Éditer .env avec vos clés API et configurations
```

3. Démarrer les services avec Docker Compose:
```bash
docker-compose up -d
```

### Déploiement manuel (sans Docker)

Pour chaque service, dans un terminal séparé:

```bash
# Exemple pour le service Portfolio
cd RootTrading
python -m portfolio.src.main
```

## Configuration

La configuration du système est centralisée dans le module `shared/src/config.py`, avec des options spécifiques définies dans les fichiers `.env` ou les variables d'environnement.

### Variables principales

- `BINANCE_API_KEY`, `BINANCE_SECRET_KEY`: Clés API Binance
- `TRADING_MODE`: 'live' ou 'demo'
- `SYMBOLS`: Liste des symboles à trader (ex: BTCUSDC,ETHUSDC)
- `INTERVAL`: Intervalle des chandeliers (ex: 1m, 5m, 15m)
- `KAFKA_BROKER`, `KAFKA_GROUP_ID`: Configuration Kafka
- `REDIS_HOST`, `REDIS_PORT`: Configuration Redis
- `DB_*`: Configuration de la base de données
- `POCKET_CONFIG`: Allocation des poches de capital

### Ports des services

- Gateway: 5000
- Analyzer: 5001
- Trader: 5002
- Coordinator: 5003
- Dispatcher: 5004
- Logger: 5005
- PNL Tracker: 5006
- Risk Manager: 5007
- Scheduler: 5008
- Portfolio: 8000
- Frontend: 3000

## API REST

Plusieurs services exposent des API REST pour l'interaction et le monitoring:

### Gateway API

- `GET /health`: État de santé du service
- `GET /diagnostic`: Informations détaillées sur l'état du service

### Portfolio API

- `GET /summary`: Résumé du portefeuille
- `GET /balances`: Soldes actuels
- `GET /pockets`: État des poches de capital
- `PUT /pockets/sync`: Force la synchronisation des poches
- `PUT /pockets/allocation`: Met à jour l'allocation des poches
- `GET /trades`: Historique des trades avec filtrage et pagination
- `GET /performance/{period}`: Statistiques de performance

### Trader API

- `GET /health`: État de santé du service
- `GET /orders`: Liste des ordres actifs
- `POST /order`: Crée un ordre manuel
- `DELETE /order/{order_id}`: Annule un ordre
- `POST /close/{cycle_id}`: Ferme un cycle de trading
- `POST /config/pause`: Met en pause le trading
- `POST /config/resume`: Reprend le trading

### Risk Manager API

- `GET /health`: État de santé du service
- `GET /rules`: Liste des règles et leur état

## Optimisations possibles

### Performances

1. **Traitement parallèle**: 
   - Analyzer utilise déjà le multiprocessing, mais pourrait être optimisé davantage
   - Le traitement des données pourrait être encore plus parallélisé

2. **Bases de données**:
   - Optimisation des requêtes et des indices
   - Partitionnement des données historiques
   - Implémentation de vues matérialisées pour les requêtes fréquentes

3. **Mise en cache**:
   - Extension du cache Redis pour plus de types de données
   - Mise en place d'une hiérarchie de cache plus sophistiquée

### Robustesse

1. **Gestion d'erreurs**:
   - Meilleure isolation des erreurs de service
   - Retry patterns plus avancés

2. **Surveillance**:
   - Ajout de métriques plus détaillées
   - Alertes avancées

3. **Haute disponibilité**:
   - Déploiement multi-instances de chaque service
   - Configuration de réplication pour Redis et Kafka

### Fonctionnalités

1. **Stratégies**:
   - Support pour des stratégies plus complexes (ML, apprentissage par renforcement)
   - Backtesting plus avancé

2. **Gestion des risques**:
   - Règles plus sophistiquées
   - Analyse de scénarios et tests de stress

3. **Interface utilisateur**:
   - Dashboard plus complet avec visualisations avancées
   - Interface mobile

## Glossaire

- **Cycle de trading**: Processus complet d'entrée et de sortie d'une position
- **Chandelier**: Représentation des mouvements de prix dans un intervalle de temps (OHLC)
- **OHLC**: Open, High, Low, Close - données standard d'un chandelier
- **Signal**: Recommandation d'achat ou de vente générée par une stratégie
- **Poche de capital**: Allocation spécifique du capital pour différents objectifs
- **Stop-loss**: Prix auquel une position est automatiquement fermée pour limiter les pertes
- **Take-profit**: Prix cible auquel une position est fermée pour sécuriser les gains
- **Drawdown**: Baisse de la valeur du portefeuille par rapport à un sommet précédent
- **PnL**: Profit and Loss - bénéfice ou perte d'une position ou du portefeuille
- **Ride or React**: Mode adaptatif où les stratégies ajustent leur comportement selon les conditions de marché

## Dépannage

### Problèmes communs

1. **Connection refusée à Kafka ou Redis**:
   - Vérifier que les services sont démarrés
   - Vérifier les paramètres d'hôte et de port

2. **Erreurs d'authentification Binance**:
   - Vérifier les clés API
   - Vérifier les permissions des clés API

3. **Synchronisation des poches incohérente**:
   - Exécuter une réconciliation manuelle via l'API Portfolio
   - Vérifier les logs pour des erreurs de transaction

4. **Aucun signal généré**:
   - Vérifier que les données de marché arrivent correctement
   - Vérifier les configurations des stratégies

### Journalisation et débogage

- Les logs sont disponibles dans chaque conteneur Docker
- Le service Logger centralise les logs importants
- Le niveau de détail peut être ajusté via `LOG_LEVEL` dans la configuration

### Support

Pour les questions techniques ou les problèmes, veuillez créer une issue sur GitHub.