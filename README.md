# 🚀 ROOT Trading Bot

**Système de trading crypto automatisé SPOT avec cycles BUY/SELL optimisés**

*C'est mon avenir et ma vie en jeu* - Architecture professionnelle pour performance maximale.

---

## 📋 Vue d'ensemble

ROOT est un bot de trading crypto entièrement automatisé focalisé sur le trading SPOT uniquement. Le système utilise une architecture de microservices pour analyser les marchés, générer des signaux et exécuter des trades avec un capital entièrement réinvesti jusqu'en juin 2026.

### 🎯 Objectifs
- **Trading SPOT uniquement** (aucun effet de levier)
- **Cycles BUY/SELL optimisés** avec trailing stop intelligent
- **Allocation dynamique** par actif selon la performance
- **Logique de régime** pour adaptation aux conditions de marché
- **Performance trackée** avec métriques avancées
- **Capital entièrement réinvesti** jusqu'en juin 2026

---

## 🏗️ Architecture

### Flux de données principal
```
Binance API → Gateway → Dispatcher → Database → Market Analyzer → Database
```

### 🔧 Services

| Service | Port | Rôle |
|---------|------|------|
| **Gateway** | 5010 | Récupération données OHLCV brutes depuis Binance |
| **Dispatcher** | 5004 | Routage et persistence des données market |
| **Market Analyzer** | 5020 | Calcul de TOUS les indicateurs techniques |
| **Analyzer** | 5012 | Génération de signaux de trading |
| **Signal Aggregator** | 5013 | Agrégation et validation des signaux |
| **Coordinator** | 5003 | Agrégation et validation des trades |
| **Trader** | 5002 | Exécution des ordres |
| **Portfolio** | 8000 | Gestion du portefeuille et allocation |
| **Visualization** | 5009 | Interface de monitoring |

### 🗄️ Base de données
- **PostgreSQL + TimescaleDB** pour les données temporelles
- **Redis** pour le cache haute performance (1-5ms)
- Tables principales :
  - `market_data` : Données OHLCV brutes
  - `analyzer_data` : 80+ indicateurs techniques calculés
  - `signals` : Signaux de trading générés
  - `trades` : Historique des trades

---

## 📊 Indicateurs techniques

Le Market Analyzer calcule automatiquement **80+ indicateurs** :

### Tendance
- **Moyennes mobiles** : EMA (7,12,26,50,99), SMA (20,50), WMA, DEMA, TEMA, Hull MA, KAMA
- **MACD** complet : Line, Signal, Histogram, PPO
- **ADX** : Tendance et force directionnelle

### Momentum  
- **RSI** (14,21), Stochastic RSI
- **Stochastic** (normal + fast)
- **Williams %R**, **CCI**
- **Momentum**, **ROC** (Rate of Change)

### Volatilité
- **ATR** avec stops dynamiques
- **Bollinger Bands** + squeeze/expansion
- **Keltner Channels**
- **NATR** (Normalized ATR)

### Volume
- **OBV** (On Balance Volume)
- **VWAP** (Volume Weighted Average Price)
- **Volume context** et patterns

### Détections avancées
- **Régime de marché** (trending, ranging, volatile)
- **Support/Résistance** automatique
- **Volume spikes** et accumulation
- **Patterns** de chandelier

---

## 🛠️ Stack technique

- **Python 3.11+**
- **Docker Compose** pour l'orchestration
- **PostgreSQL + TimescaleDB** pour les données temporelles
- **Redis** pour le cache haute performance
- **Kafka** pour la messagerie temps réel
- **FastAPI/aiohttp** pour les APIs
- **React + Vite** pour la visualisation
- **Binance API** (SPOT uniquement)

---

## 🚦 État actuel

### ✅ Implémenté
- [x] Gateway : Récupération données OHLCV
- [x] Dispatcher : Persistence market_data  
- [x] Market Analyzer : Calcul automatique de tous les indicateurs
- [x] Database : Schéma complet market_data + analyzer_data
- [x] Trigger temps réel : PostgreSQL LISTEN/NOTIFY
- [x] Architecture microservices complète

### 🔄 En cours
- [ ] Analyzer : Génération de signaux
- [ ] Signal Aggregator : Logique d'agrégation


### 📋 À venir
- [ ] Trailing stop intelligent
- [ ] Régimes de marché avancés
- [ ] Alertes et notifications

---

## 🔧 Configuration

### Variables d'environnement
```bash
# Base de données
PGHOST=localhost
PGPORT=5432
PGDATABASE=trading
PGUSER=postgres
PGPASSWORD=postgres

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Binance
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
```

### Symboles surveillés
Configurés dans `shared/src/config.py` :
```python
SYMBOLS = ['BTCUSDC', 'ETHUSDC', 'ADAUSDC', ...]
```

---

## 🚀 Démarrage rapide

```bash
# Cloner le projet
git clone <repository-url>
cd root-trading-bot

# Démarrer l'infrastructure
docker-compose up -d redis kafka db

# Démarrer les services
docker-compose up -d gateway dispatcher market_analyzer

# Vérifier les logs
docker-compose logs -f market_analyzer
```

---

## 📈 Commandes utiles

### Base de données
```bash
# Se connecter à la DB
docker exec roottrading-db-1 psql -U postgres -d trading

# Voir les données récentes
docker exec roottrading-db-1 psql -U postgres -d trading -c "SELECT COUNT(*) FROM market_data;"

# Vérifier les indicateurs
docker exec roottrading-db-1 psql -U postgres -d trading -c "SELECT COUNT(*) FROM analyzer_data;"
```

### Monitoring
```bash
# API Health checks
curl http://localhost:5010/health  # Gateway
curl http://localhost:5004/health  # Dispatcher  
curl http://localhost:5020/health  # Market Analyzer

# Statistiques Market Analyzer
curl http://localhost:5020/stats
```

---

## 📝 Notes importantes

- **SPOT UNIQUEMENT** : Aucun trading avec effet de levier
- **Capital à risque** : Système conçu pour la performance long terme
- **Monitoring constant** : Logs et métriques détaillés
- **Architecture évolutive** : Ajout facile de nouvelles stratégies
- **Sécurité** : Aucune clé privée en dur, variables d'environnement

---

## 📞 Support

Pour toute question ou amélioration :
- Vérifier les logs : `docker-compose logs -f <service>`
- APIs de diagnostic : `/health` et `/stats` sur chaque service
- Base de données : Commandes SQL dans `database/database-commands.md`

---

*ROOT Trading Bot v1.0 - Architecture microservices pour trading crypto automatisé*