# RootTrading - Plateforme de Trading Automatisé

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Architecture](#architecture)
   - [Diagramme de flux](#diagramme-de-flux)
   - [Services](#services)
3. [Configuration](#configuration)
   - [Variables d'environnement](#variables-denvironnement)
   - [Sécurité](#sécurité)
4. [Installation et démarrage](#installation-et-démarrage)
5. [Guide de développement](#guide-de-développement)
   - [Ajouter une nouvelle stratégie](#ajouter-une-nouvelle-stratégie)
   - [Ajouter une nouvelle paire de trading](#ajouter-une-nouvelle-paire-de-trading)
   - [Personnaliser les signaux de trading](#personnaliser-les-signaux-de-trading)
6. [Structure des données](#structure-des-données)
   - [Market Data](#market-data)
   - [Signaux de trading](#signaux-de-trading)
   - [Transactions](#transactions)
7. [Communication entre services](#communication-entre-services)
   - [Canaux Redis](#canaux-redis)
   - [Format des messages](#format-des-messages)
8. [Base de données](#base-de-données)
   - [Schéma](#schéma)
   - [Requêtes communes](#requêtes-communes)
9. [APIs REST](#apis-rest)
   - [Portfolio API](#portfolio-api)
   - [Trader API](#trader-api)
10. [Optimisation](#optimisation)
    - [Gestion des appels API Binance](#gestion-des-appels-api-binance)
    - [Performance Redis](#performance-redis)
11. [Mode démo vs Mode réel](#mode-démo-vs-mode-réel)
12. [Logging et diagnostic](#logging-et-diagnostic)
13. [Extensions futures](#extensions-futures)

## Vue d'ensemble

RootTrading est une plateforme modulaire de trading automatisé pour le marché des cryptomonnaies, construite sur une architecture microservices. Elle permet l'exécution automatique de stratégies de trading, la gestion des ordres, le suivi du portefeuille et l'analyse des performances.

**Caractéristiques principales:**
- Architecture microservices avec Docker
- Connexion à l'API Binance
- Stratégies de trading personnalisables
- Communication asynchrone via Redis
- Stockage persistant avec PostgreSQL
- Interface utilisateur React/Tailwind
- Mode démo pour tester sans risque

## Architecture

### Diagramme de flux

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Gateway   │────▶│  Analyzer   │────▶│   Trader    │────▶│  Portfolio  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                                        │                  │
       │                                        │                  │
       ▼                                        ▼                  ▼
┌─────────────┐                         ┌─────────────┐     ┌─────────────┐
│   Binance   │                         │  PostgreSQL │     │  Frontend   │
└─────────────┘                         └─────────────┘     └─────────────┘
       ▲                                        ▲                  │
       │                                        │                  │
       └────────────────────────────────────────┴──────────────────┘
```

### Services

1. **Gateway**:
   - **Technologie**: Node.js
   - **Rôle**: Se connecte à l'API Binance, récupère les données de marché et les publie sur Redis
   - **Fichiers clés**: 
     - `gateway/src/index.js` - Point d'entrée principal
     - `gateway/src/binance-connector.js` - Gestion des connexions Binance
     - `gateway/src/redis-publisher.js` - Publication des données vers Redis

2. **Analyzer**:
   - **Technologie**: Python
   - **Rôle**: Reçoit les données de marché, exécute les stratégies et génère des signaux de trading
   - **Fichiers clés**:
     - `analyzer/main.py` - Point d'entrée
     - `analyzer/strategy_manager.py` - Gestion des stratégies de trading
     - `analyzer/strategies/` - Dossier contenant toutes les stratégies

3. **Trader**:
   - **Technologie**: Python
   - **Rôle**: Reçoit les signaux, exécute les ordres sur Binance et les enregistre en base de données
   - **Fichiers clés**:
     - `trader/main.py` - Point d'entrée et API
     - `trader/binance_executor.py` - Exécution des ordres sur Binance
     - `trader/order_manager.py` - Gestion et enregistrement des ordres

4. **Portfolio**:
   - **Technologie**: Python (FastAPI)
   - **Rôle**: Fournit une API pour accéder aux données du portefeuille et à l'historique des trades
   - **Fichiers clés**:
     - `portfolio/src/main.py` - Point d'entrée et configuration FastAPI
     - `portfolio/src/routes.py` - Endpoints API
     - `portfolio/src/manager.py` - Gestion des données du portefeuille

5. **Frontend**:
   - **Technologie**: React, Tailwind CSS
   - **Rôle**: Interface utilisateur pour visualiser le portefeuille et passer des ordres manuels
   - **Fichiers clés**:
     - `frontend/src/App.jsx` - Point d'entrée React
     - `frontend/src/components/` - Composants UI
     - `frontend/src/api/` - Clients API pour backend

6. **Services partagés**:
   - **Redis**: Communication entre services
   - **PostgreSQL**: Stockage des données de transactions

## Configuration

### Variables d'environnement

Toutes les configurations sont centralisées dans un fichier `.env` à la racine:

```bash
# Clés API Binance
BINANCE_API_KEY=votre_clé_api
BINANCE_SECRET_KEY=votre_clé_secrète

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# PostgreSQL
PGUSER=postgres
PGPASSWORD=postgres
PGDATABASE=trading
PGHOST=db
PGPORT=5432

# Paramètres de trading
SYMBOL=BTCUSDC           # Paire de trading principale
INTERVAL=1m              # Intervalle des bougies (1m, 5m, 15m, etc.)
TRADING_MODE=demo        # demo ou live
TRADE_QUANTITY=0.00017   # Quantité par défaut pour les ordres

# Paramètres des stratégies
RSI_WINDOW=14            # Période RSI
RSI_OVERBOUGHT=70        # Seuil de surachat
RSI_OVERSOLD=30          # Seuil de survente
SHORT_WINDOW=5           # Période courte pour MA Cross
LONG_WINDOW=20           # Période longue pour MA Cross
```

Ces variables sont accessibles dans tous les services via les fichiers de configuration (`utils/config.py` pour Python, et via `dotenv` pour Node.js).

### Sécurité

Les clés API Binance doivent avoir les permissions minimales nécessaires:
- Lecture du solde du compte
- Lecture des données de marché
- Passage d'ordres (uniquement si TRADING_MODE=live)

## Installation et démarrage

### Prérequis

- Docker et Docker Compose

### Étapes de démarrage

1. Configurer le fichier .env:
   ```
   # Modifier .env avec vos clés API et configurations
   ```

2. Lancer l'application avec Docker Compose:
   ```
   docker-compose up -d
   ou docker-compose up -- build
   ```

3. Accéder à l'interface:
   ```
   http://localhost:3000
   ```

4. Vérifier les logs:
   ```bash
   docker-compose logs -f
   # ou pour un service spécifique:
   docker-compose logs -f analyzer
   ```

## Guide de développement

### Ajouter une nouvelle stratégie

1. Créer un nouveau fichier Python dans `analyzer/strategies/` (par exemple `bollinger_bands.py`):

```python
# analyzer/strategies/bollinger_bands.py
import os
import numpy as np
import pandas as pd
from collections import deque
from typing import Optional, Dict, Any
from .base_strategy import BaseStrategy

class BollingerBandsStrategy(BaseStrategy):
    """
    Stratégie basée sur les bandes de Bollinger
    """
    def __init__(self):
        self.window = int(os.getenv('BB_WINDOW', 20))
        self.num_std = float(os.getenv('BB_STD', 2.0))
        self.prices = deque(maxlen=self.window * 3)
        self.prev_price = None

    @property
    def name(self) -> str:
        return 'Bollinger_Bands'

    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        price = market_data.get('close')
        if price is None:
            return None

        # Ajouter le nouveau prix
        self.prices.append(price)

        # Attendre d'avoir assez de données
        if len(self.prices) < self.window:
            self.prev_price = price
            return None

        # Calcul des bandes de Bollinger
        prices_array = np.array(self.prices)
        mean = np.mean(prices_array[-self.window:])
        std = np.std(prices_array[-self.window:])
        upper_band = mean + (std * self.num_std)
        lower_band = mean - (std * self.num_std)

        signal = None
        if self.prev_price:
            # Signal d'achat: prix traverse la bande inférieure vers le haut
            if self.prev_price <= lower_band and price > lower_band:
                signal = 'BUY'
            # Signal de vente: prix traverse la bande supérieure vers le bas
            elif self.prev_price >= upper_band and price < upper_band:
                signal = 'SELL'

        self.prev_price = price

        if signal:
            return {
                'strategy': self.name,
                'symbol': market_data.get('symbol'),
                'side': signal,
                'timestamp': market_data.get('startTime'),
                'price': price,
                'bb_upper': upper_band,
                'bb_lower': lower_band,
                'bb_mean': mean
            }
        return None
```

2. Ajouter les variables de configuration dans le fichier `.env`:
   ```
   BB_WINDOW=20
   BB_STD=2.0
   ```

3. Redémarrer uniquement le service Analyzer:
   ```bash
   docker-compose restart analyzer
   ```

4. Vérifier les logs pour confirmer le chargement de la stratégie:
   ```bash
   docker-compose logs -f analyzer
   # chercher: "Stratégies chargées: RSI_Strategy, SimpleMA_Cross, Bollinger_Bands"
   ```

### Ajouter une nouvelle paire de trading

Pour surveiller et trader plusieurs paires de cryptomonnaies, vous pouvez:

1. Modifier le fichier `.env` pour changer la paire principale:
   ```
   SYMBOL=ETHUSDC
   ```

2. Pour supporter plusieurs paires simultanément, modifiez le Gateway:
   
   Dans `gateway/src/index.js`, ajoutez des abonnements supplémentaires:

   ```javascript
   // S'abonner à plusieurs paires
   const symbols = ['BTCUSDC', 'ETHUSDC', 'SOLUSDC'];
   const interval = process.env.INTERVAL || '1m';
   
   symbols.forEach(symbol => {
     subscribeToCandles(binanceClient, symbol, interval, (candleData) => {
       const channel = `market:data:${candleData.symbol}`;
       publishToRedis(redisPublisher, channel, candleData);
       logger.debug(`Publié sur ${channel} → ${JSON.stringify(candleData)}`);
     });
   });
   ```

3. Modifier le subscriber dans `analyzer/redis_subscriber.py` pour écouter plusieurs canaux:

   ```python
   def subscribe_market_data(callback, stop_event=None):
       """
       Souscrit aux channels de plusieurs symboles.
       """
       symbols = os.getenv('SYMBOLS', 'BTCUSDC,ETHUSDC,SOLUSDC').split(',')
       
       # Connexion Redis
       redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT)
       pubsub = redis_client.pubsub(ignore_subscribe_messages=True)
       
       # S'abonner à chaque symbole
       channels = []
       for symbol in symbols:
           channel = f"{CHANNEL_PREFIX}:{symbol}"
           pubsub.subscribe(channel)
           channels.append(channel)
           logger.info(f"✅ Abonné au channel Redis {channel}")
       
       # Boucle d'écoute
       for message in pubsub.listen():
           # [...le reste du code reste identique...]
   ```

4. Redémarrer les services Gateway et Analyzer:
   ```bash
   docker-compose restart gateway analyzer
   ```

### Personnaliser les signaux de trading

Vous pouvez personnaliser les conditions qui génèrent des signaux dans n'importe quelle stratégie:

1. Dans les fichiers de stratégie (`analyzer/strategies/`), modifiez la logique de la méthode `generate_signal()`:

   Par exemple, ajoutons une condition de volume pour la stratégie RSI:

   ```python
   def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
       price = market_data.get('close')
       volume = market_data.get('volume', 0)
       
       # Ignorer si volume trop faible
       min_volume = float(os.getenv('MIN_VOLUME', 10))
       if volume < min_volume:
           return None
           
       # Reste du code RSI...
   ```

2. Ajouter le nouveau paramètre dans `.env`:
   ```
   MIN_VOLUME=10
   ```

3. Redémarrer l'analyzer:
   ```bash
   docker-compose restart analyzer
   ```

## Structure des données

### Market Data

Format des données de marché publiées par le Gateway:

```json
{
  "symbol": "BTCUSDC",
  "startTime": 1616554800000,
  "closeTime": 1616554859999,
  "open": 55432.10,
  "high": 55450.00,
  "low": 55421.20,
  "close": 55445.80,
  "volume": 2.34567
}
```

### Signaux de trading

Format des signaux générés par l'Analyzer et envoyés au Trader:

```json
{
  "strategy": "RSI_Strategy",
  "symbol": "BTCUSDC",
  "side": "BUY",
  "timestamp": 1616554800000,
  "price": 55445.80,
  "rsi": 28.5
}
```

### Transactions

Structure d'une transaction stockée en base de données:

```json
{
  "id": 12345678,
  "symbol": "BTCUSDC",
  "side": "BUY",
  "price": 55445.80,
  "quantity": 0.00017,
  "quote_quantity": 9.42,
  "fee": 0.0094,
  "role": "taker",
  "timestamp": "2023-01-15T14:30:00Z",
  "status": "FILLED",
  "demo": false
}
```

## Communication entre services

### Canaux Redis

Les services communiquent via ces canaux Redis:

| Canal | Description | Publication | Abonnement |
|-------|-------------|------------|------------|
| `market:data:{SYMBOL}` | Données de marché en temps réel | Gateway | Analyzer |
| `analyze:signal` | Signaux de trading générés | Analyzer | Trader |
| `account:balances` | Mise à jour des soldes du compte | Gateway | Portfolio |
| `account:total_balance` | Valeur totale du portefeuille | Gateway | Portfolio |

### Format des messages

Tous les messages sont en format JSON. Exemple:

```json
// Canal: market:data:BTCUSDC
{
  "symbol": "BTCUSDC",
  "startTime": 1616554800000,
  "closeTime": 1616554859999,
  "open": 55432.10,
  "high": 55450.00,
  "low": 55421.20,
  "close": 55445.80,
  "volume": 2.34567
}

// Canal: analyze:signal
{
  "strategy": "RSI_Strategy",
  "symbol": "BTCUSDC",
  "side": "BUY",
  "timestamp": 1616554800000,
  "price": 55445.80,
  "metadata": {
    "rsi": 28.5
  }
}

// Canal: account:balances
[
  {
    "asset": "BTC",
    "disponible": 0.00445841,
    "en_ordre": 0,
    "total": 0.00445841,
    "eur_value": 330.96
  },
  {
    "asset": "USDC",
    "disponible": 66.8915,
    "en_ordre": 0,
    "total": 66.8915,
    "eur_value": 59.54
  }
]
```

## Base de données

### Schéma

La base de données PostgreSQL contient les tables suivantes:

**Table: trades**
```sql
CREATE TABLE trades (
    id BIGINT PRIMARY KEY,
    order_id BIGINT,
    symbol VARCHAR NOT NULL,
    side VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    price NUMERIC(16, 8) NOT NULL,
    quantity NUMERIC(16, 8) NOT NULL,
    fee NUMERIC(16, 8),
    role VARCHAR,
    quote_quantity NUMERIC(16, 8),
    timestamp TIMESTAMP NOT NULL,
    demo BOOLEAN NOT NULL DEFAULT FALSE
);
```

### Requêtes communes

Exemples de requêtes SQL pour analyser les performances:

```sql
-- Résumé des trades par symbole
SELECT symbol, 
       COUNT(*) as trade_count,
       SUM(CASE WHEN side = 'BUY' THEN quote_quantity ELSE 0 END) as buy_volume,
       SUM(CASE WHEN side = 'SELL' THEN quote_quantity ELSE 0 END) as sell_volume,
       SUM(CASE WHEN side = 'SELL' THEN quote_quantity ELSE -quote_quantity END) as net_pnl
FROM trades
GROUP BY symbol
ORDER BY net_pnl DESC;

-- Performance par stratégie (si la stratégie est enregistrée)
SELECT t.strategy,
       COUNT(*) as trade_count,
       SUM(CASE WHEN side = 'SELL' THEN quote_quantity ELSE -quote_quantity END) as net_pnl
FROM trades t
GROUP BY t.strategy
ORDER BY net_pnl DESC;
```

## APIs REST

### Portfolio API

| Endpoint | Méthode | Description |
|---------|---------|-------------|
| `/portfolio/summary` | GET | Résumé du portefeuille (valeur totale, trades) |
| `/portfolio/balance` | GET | Soldes actuels par actif |
| `/portfolio/trades` | GET | Historique des trades avec pagination |

Paramètres pour `/portfolio/trades`:
- `symbol` (optionnel): Filtrer par paire
- `limit` (défaut: 50): Nombre de résultats par page
- `page` (défaut: 1): Numéro de page

### Trader API

| Endpoint | Méthode | Description |
|---------|---------|-------------|
| `/trade` | POST | Passer un ordre manuellement |

Format de la requête pour `/trade`:
```json
{
  "symbol": "BTCUSDC",
  "side": "BUY",
  "quantity": 0.001
}
```

## Optimisation

### Gestion des appels API Binance

Pour éviter d'atteindre les limites de l'API Binance:

1. Minimiser les appels en mettant en cache les données:
   - Pour les prix, utiliser les WebSockets au lieu des requêtes REST
   - Mettre en cache les informations qui changent rarement (info sur les actifs)

2. Utiliser les bons intervalles de mise à jour:
   - Soldes du compte: toutes les minutes
   - Prix pour conversion: mises à jour périodiques

3. Regrouper les requêtes:
   - Au lieu de demander le prix de chaque actif séparément, utiliser `getPrices()`

### Performance Redis

Redis est utilisé comme bus de communication:

1. Optimiser la taille des messages:
   - N'inclure que les champs nécessaires
   - Pour les grandes quantités de données, utiliser la compression

2. Gestion des connexions:
   - Réutiliser les connexions Redis
   - Implémenter une logique de reconnexion en cas d'échec

## Mode démo vs Mode réel

Le système fonctionne dans deux modes:

1. **Mode démo** (`TRADING_MODE=demo`):
   - Les ordres ne sont pas réellement exécutés sur Binance
   - Simulation locale dans `binance_executor.py`
   - Les trades simulés sont marqués avec `demo=True` dans la base de données
   - Parfait pour tester les stratégies sans risque

2. **Mode réel** (`TRADING_MODE=live`):
   - Les ordres sont exécutés sur Binance avec les vraies clés API
   - Nécessite des permissions d'API pour le trading
   - Les transactions réelles sont persistées en base de données

Pour basculer entre les modes:
1. Modifier la variable `TRADING_MODE` dans `.env`
2. Redémarrer les services:
   ```bash
   docker-compose restart trader
   ```

## Logging et diagnostic

Chaque service a son propre système de logging:

1. Niveau de log configurable via `.env`:
   ```
   LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
   ```

2. Voir les logs des services:
   ```bash
   # Tous les services
   docker-compose logs -f
   
   # Service spécifique
   docker-compose logs -f analyzer
   ```

3. Debug des stratégies:
   - Mettre `LOG_LEVEL=DEBUG` pour voir les calculs d'indicateurs
   - Les signaux générés sont toujours loggés en niveau INFO

## Extensions futures


🧱 Phase 1 – Scalabilité de base
🔁 1. Ajouter d’autres cryptos
✅ Déjà prêt côté Gateway/Analyzer.

🔧 À faire :

Étendre la liste SYMBOLS dans .env (BTCUSDC, ETHUSDC, SOLUSDC, etc.)

Adapter gateway/src/index.js pour tous les abonnements.

Adapter analyzer/redis_subscriber.py pour écouter toutes les paires.

🧠 2. Ajouter de nouvelles stratégies
🧩 Crée des fichiers dans analyzer/strategies/

🎛️ Ajoute les variables .env associées (ex : BB_STD, EMA_SHORT, etc.)

🧪 Redémarre le service analyzer et vérifie les logs

🚀 Phase 2 – Interface & Monitoring
🧾 3. Frontend amélioré
📊 Ajouter :

Tableau PnL par paire et stratégie

Graphiques Candle + signaux (avec Chart.js ou Recharts)

Historique des positions ouvertes/fermées

Stats temps réel (balance, nombre de trades, % win)

⚙️ Organiser les composants :

Dashboard.jsx, TradeHistory.jsx, SignalChart.jsx, etc.

Créer un useStats() hook qui appelle le backend pour live refresh

📡 4. Logs & Stats live
Backend :

Ajouter une route /stats/live dans portfolio

Calculer le nombre de signaux, trades, performances instantanées

Frontend :

useEffect avec interval pour fetch régulier (ou via WebSocket plus tard)

Bonus : intégrer socket.io pour stats en temps réel (optionnel mais sexy)

🛡️ Phase 3 – Gestion du risque
🔒 5. Stop-Loss / Take-Profit
Implémenter dans trader/order_manager.py :

Ajout de stop_price / target_price selon stratégie

Vérification continue (peut être via un watcher de marché)

Ajouter au message signal :

json
Copier
Modifier
{
  "stop_loss": 55000,
  "take_profit": 57000
}
🔁 6. Trailing Stop
Stocker le trailing_start_price à l’achat

À chaque nouveau candle, ajuster le stop_loss à max_price - trailing_gap

Fermer la position si prix chute sous le trailing stop dynamique

🚨 7. Gestion des limites d’exposition
Exemple : max 20% de ton capital sur une seule crypto

Implémenter un risk_manager.py dans trader/

Vérifie avant d’envoyer un ordre si la limite est déjà atteinte

📅 strategie à mettre en place plus tard, a voir si autorisé sur binance France car pas de futures ni de short long:

1. Trend‑Following avec Cross EMA
Horizon : 2–10 jours

Indicateurs : EMA 21 vs EMA 50 sur l’unité journalier (D1).

Entrée : quand EMA 21 croise au‑dessus de l’EMA 50 = début de tendance haussière.

Sortie :

Take‑profit automatique à + 5–10 % (ajuste selon la volatilité).

Stop‑loss sous le dernier plus bas swing (- 3 % par ex.).

Bot‑flow : scan D1 toutes les 4 h, ouvre position spot (BTC, ETH, LINK…) dès le cross, ferme au TP/SL.

Pourquoi : tu suis la tendance, tu restes dans le train quand ça part.

2. Reversal RSI Divergence
Horizon : 1–5 jours

Indicateur : RSI 14 en daily ou 4 h.

Entrée :

Prix fait un plus bas plus bas (bearish structure)

RSI fait un plus bas plus haut (divergence haussière)

Sortie : clôture quand RSI revient > 50 ou atteint target + 4–6 %.

Bot‑flow : vérifie daily + H4, repère divergences, place ordre limit sur le support identifié.

Pourquoi : tu captes les retournements précoces, rendement sympa sans courir après la hype.

3. Breakout de Range
Horizon : 3–7 jours

Setup : repère consolidation (range horizontal) au moins 5 barres D1.

Entrée :

Buy stop quelques pips au‑dessus de la résistance pour buy breakout.

Sell stop sous le support pour take advantage d’un effondrement.

Sortie :

TP = largeur du range (hauteur) projetée à la sortie.

SL = juste de l’autre côté du range.

Bot‑flow : détecte range en scan (plus haut/plus bas) 5–10 jours, place OCO (TP+SL), exécute dès la cassure.

Pourquoi : tu profites du momentum généré par la sortie de zone de congestion.

4. Rotation de Pairs Corrélées
Horizon : 5–14 jours

Principe : passer de la crypto la plus faible à la plus forte sur un panier.

Étapes :

Chaque jour, calcule la perf 7 jours de BTC, ETH, LINK, DOT…

Sell la plus faible (en USDC) + Buy la plus forte.

Rééquilibre hebdo ou quand la perf se renverse.

Bot‑flow : récupère perf 7 jours via l’API, génère ordre market USDC→top crypto et inverse pour la plus faible.

Pourquoi : tu capturés les leaders de marché, tu suis la dynamique sectorielle.

💡 Bonus Risk‑Mgmt
Taille de position : 2–5 % par trade.

Stop‑loss “volatilité” : place ton SL à 1,5× ATR(14) pour laisser respirer.

Take‑profit partiel : ferme 50 % à + 5 %, laisse le reste courir avec un trailing SL.



1. Triangular Spot (USDC ⇄ ALT ⇄ ALT ⇄ USDC)
Pourquoi : tu joues sur les micro‑écarts entre paires spot sans jamais sortir de Binance.

Loop type : USDC → LINK → ETH → USDC (ou remplace LINK/ETH par ATOM, BNB…)

Bot‑flow :

Récupère en temps réel les order‑books USDC/ALT1, ALT1/ALT2, ALT2/USDC.

Simule le swap en chaîne ; si profit net > frais+0,15 %, envoie les 3 ordres “post‑only”.

Répète dès que tu trouves un spread exploitable.

2. Stablecoin‑Swap & Savings
Pourquoi : profite des différences de taux et rewards entre USDC, BUSD et DAI.

Options :

Convertisseur instantané (USDC↔BUSD↔DAI) pour capter 0,02–0,1 % de micro‑arbitrage.

Binance Earn (Flex Savings vs Locked) : déplace tes stablecoins vers la meilleure offre (ex : DAI Locked 7 % APY vs USDC Flex 3 %).

Bot‑flow : scrute /sapi/v1/bswap/poolList et /sapi/v1/lending/union/position/list, déplace les fonds quand l’écart de yield dépasse ton seuil (ex. + 1 %).

3. BNB Fee‑Rebate Hack
Principe : utilise BNB pour payer tous tes frais spot et gagne 25 % de discount, que tu peux convertir en BUSD/USDC.

Bot‑flow :

Vérifie chaque ordre : si tu peux payer en BNB, active l’option “Discount”.

Stocke automatiquement le BNB économisé dans un wallet “rebate” et convertis-le en stable dès que tu atteins ton palier (ex 50 USDC).

Résultat : ton portefeuille croît “gratuitement” à chaque trade.

4. Cross‑Pair Swing Spot
Principe : exploiter les divergences de momentum entre deux paires corrélées (ex BTC/USDC ↔ ETH/USDC).

Setup :

Monitor RSI ou MACD court terme sur BTC/USDC et ETH/USDC.

Si ETH tape un support (RSI<30) alors que BTC reste stable, passe un ordre limit ETH/USDC, et reverse si ETH → surachat.

Bot‑flow : check indicateurs toutes les 5 min, place OCO (Take‑Profit + Stop‑Loss) dès qu’un signal se confirme.

1. Funding‑Rate Hedge (Spot ⇄ Perpétuels)
Identique à avant, mais avec USDC comme référence si tu veux te couvrir en stable :

Quand funding ⬆︎ sur ETH/BTC/LINK perp > 0 → Short perp + Long spot (en USDC)

Quand funding ⬇︎ (< 0) → Long perp + Short spot

API Binance toutes les 15 min, seuil ≈ 0,01 % de funding, close à 0 ou target heures de funding encaissé.

2. Calendar‑Spread (Perpétuel vs Futures Trimestriels)
Toujours top pour du roll‐yield sans directionnel :

Compare SYMBOL_PERP vs SYMBOL_YYYYMMDD (BTC, ETH, LINK…)

If prix(trimestriel) > prix(perp) + frais → Short trimestriel + Long perp

Else inverse

Trigger > 0,2 % de spread, rebalance avant expiry ou take‑profit.

3. Triangular Spot (USDC ⇄ ALT ⇄ ALT ⇄ USDC)
Exploite les écarts entre paires spot USDC/ALT :

Boucle type : USDC → LINK → ETH → USDC

Récupère order books pour USDC/LINK, LINK/ETH, ETH/USDC

Simule profit net > frais + buffer (0,2 %), puis exécute les 3 ordres en “post‐only”.

Rapide (ms) et 100 % on‑exchange.

4. Stablecoin Arbitrage & Liquid Swap
Sans USDT, tu joues sur USDC, BUSD et DAI :

Liquid Swap Pools : surveille les APY et les taux de swap USDC–BUSD et USDC–DAI.

Quand écart > 0,02 % ou APY promo > 10 % (récompenses BNB/SXP…), dépose/enlève du pool.

Utilise le convertisseur “instantané” pour passer USDC ↔ BUSD ↔ DAI en un clic et capter le micro‑arbitrage.