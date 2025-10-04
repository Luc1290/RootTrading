# RootTrading - Système de Trading Crypto Automatisé

## Vue d'ensemble
RootTrading est un écosystème de trading cryptographique automatisé, conçu avec une architecture microservices pour le trading SPOT sur Binance. Le système combine analyse technique multi-stratégies, agrégation de signaux intelligente et exécution automatisée avec gestion des risques.

## 🎯 Objectifs
- **Trading automatisé** : Exécution de stratégies de trading crypto sans intervention manuelle
- **Multi-stratégies** : Support de 28 stratégies techniques simultanées
- **Gestion des risques** : Contrôles intégrés et trailing stops intelligents
- **Monitoring temps réel** : Dashboard web et métriques de performance
- **Architecture scalable** : Microservices containerisés avec Docker

## 🏗️ Architecture Système

### Vue d'ensemble
```
┌─────────────────────────────────────────────────────────────┐
│                     RootTrading Ecosystem                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Binance API ──► Gateway ──► Dispatcher ──► Market Analyzer │
│                     │             │              │          │
│                     ▼             ▼              ▼          │
│              ┌─────────────────────────────────────────┐    │
│              │            Kafka + Redis               │    │
│              └─────────────────────────────────────────┘    │
│                     │             │              │          │
│                     ▼             ▼              ▼          │
│              Analyzer ──► Signal Aggregator ──► Coordinator │
│                                   │              │          │
│                                   ▼              ▼          │
│              Portfolio ◄────── Trader ◄──────────┘          │
│                     │                                       │
│                     ▼                                       │
│              Visualization Dashboard                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Infrastructure
- **Base de données** : PostgreSQL + TimescaleDB pour les données financières
- **Message Broker** : Kafka pour la communication asynchrone
- **Cache** : Redis pour les données temps réel
- **Conteneurisation** : Docker Compose pour l'orchestration

## 🔧 Services Microservices

### 📊 Data Pipeline
| Service | Port | Description |
|---------|------|-------------|
| **Gateway** | 5010 | Collecte des données Binance (WebSocket + REST API) |
| **Dispatcher** | 5004 | Routage des messages Kafka vers Redis |
| **Market Analyzer** | 5020 | Calcul des indicateurs techniques |

### 🧠 Signal Processing
| Service | Port | Description |
|---------|------|-------------|
| **Analyzer** | 5012 | Exécution des stratégies de trading (28 stratégies) |
| **Signal Aggregator** | 5013 | Consensus et validation des signaux |

### 💰 Trading & Portfolio
| Service | Port | Description |
|---------|------|-------------|
| **Coordinator** | 5003 | Orchestration et validation finale |
| **Trader** | 5002 | Exécution des ordres sur Binance |
| **Portfolio** | 8000 | Gestion du portefeuille et PnL |

### 📈 Monitoring
| Service | Port | Description |
|---------|------|-------------|
| **Visualization** | 5009 | Dashboard web et graphiques temps réel |

## 🚀 Démarrage Rapide

### Prérequis
- Docker et Docker Compose installés
- Clés API Binance (trading activé)
- Minimum 8GB RAM pour l'ensemble du stack

### Installation
```bash
# 1. Cloner le repository
git clone <repository-url>
cd RootTrading

# 2. Configuration
cp .env.example .env
# Éditer .env avec vos clés API Binance

# 3. Démarrage complet
docker-compose up -d

# 4. Vérification des services
docker-compose ps
```

### Accès aux interfaces
- **Dashboard Principal** : http://localhost:5009
- **Portfolio API** : http://localhost:8000/docs
- **Health Checks** : Disponibles sur tous les services

## ⚙️ Configuration

### Variables d'environnement principales
```env
# Trading
TRADING_MODE=live                    # live ou paper
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key

# Symboles tradés (22 cryptos par défaut)
TRADING_SYMBOLS=BTCUSDC,ETHUSDC,SOLUSDC,XRPUSDC...

# Infrastructure
KAFKA_BROKER=kafka:9092
REDIS_HOST=redis
PGDATABASE=trading
```

### Symboles supportés
Le système trade 22 cryptomonnaies par défaut :
- **Majeurs** : BTC, ETH, SOL, XRP, ADA
- **DeFi** : UNI, AAVE, COMP, LINK
- **Gaming/NFT** : SAND, AXS, ENJ
- **AI/Compute** : FET, RENDER
- **Memes** : PEPE, DOGE, SHIB
- **Autres** : AVAX, ATOM, NEAR, ALGO, POL

## 📋 Stratégies de Trading

### 28 Stratégies Actives

#### Stratégies de Tendance (7)
1. **ADX Direction** : Tendance basée sur l'ADX
2. **EMA Cross** : Croisements moyennes mobiles exponentielles
3. **Hull MA Slope** : Pente Hull Moving Average
4. **MACD Crossover** : Signaux MACD classiques
5. **Supertrend Reversal** : Retournements Supertrend
6. **TEMA Slope** : Triple EMA direction
7. **TRIX Crossover** : Triple EMA oscillator

#### Stratégies de Momentum (8)
8. **CCI Reversal** : Retournements Commodity Channel Index
9. **PPO Crossover** : Price Percentage Oscillator
10. **ROC Threshold** : Rate of Change avec seuils
11. **RSI Cross** : Croisements RSI avec niveaux
12. **Stochastic Oversold Buy** : Stochastique survendu
13. **StochRSI Rebound** : Rebond StochRSI
14. **Williams R Rebound** : Williams %R reversal
15. **ZScore Extreme Reversal** : Z-Score extrêmes

#### Stratégies de Volatilité (4)
16. **ATR Breakout** : Cassures avec Average True Range
17. **Bollinger Touch** : Touches des bandes de Bollinger
18. **Donchian Breakout** : Cassures des canaux de Donchian
19. **Range Breakout Confirmation** : Validation cassures de range

#### Stratégies de Support/Résistance (4)
20. **Parabolic SAR Bounce** : Rebonds Parabolic SAR
21. **Resistance Rejection** : Rejets de résistances
22. **Support Breakout** : Cassures de supports
23. **VWAP Support/Resistance** : Niveaux VWAP dynamiques

#### Stratégies de Volume (2)
24. **OBV Crossover** : On-Balance Volume divergences
25. **Spike Reaction Buy** : Réactions aux pics de volume

#### Stratégies Avancées (4)
26. **Liquidity Sweep Buy** : Détection sweeps de liquidité
27. **Multi-TF Confluent Entry** : Confluence multi-timeframes
28. **Pump & Dump Pattern** : Détection patterns anormaux

## 🛡️ Gestion des Risques

### Contrôles automatiques
- **Stop-loss adaptatifs** : 1.2-1.8% dynamiques (ATR + régime de marché)
- **Trailing sell intelligent** : Activation à +1.5%, marges adaptatives 0.8-1.5%
- **Breakeven protection** : +1.2% → entry+fees, +2.0% → entry+0.2%
- **Position sizing** : Limitation par actif (1000 USDC max)
- **Daily loss limit** : Limitation des pertes journalières
- **Concentration limits** : Max 5 positions simultanées

### Trailing Sell Intelligent

Le système utilise un trailing sell adaptatif multi-niveaux optimisé pour le scalp :

**Activation progressive** :
- Breakeven niveau 1 : +1.2% → stop @ entry + 2×fees (protection perte)
- Breakeven niveau 2 : +2.0% → stop @ entry + 0.2% (profit garanti)
- Trailing activé : +1.5% → marges adaptatives selon palier

**Marges adaptatives dynamiques** :
- Palier ≥8% : marge 0.4% (protection stricte des gros gains)
- Palier 5-8% : marge 0.6%
- Palier 3-5% : marge 0.8%
- Palier 2-3% : marge 1.0%
- Palier 1.5-2% : marge 1.2%
- Palier <1.5% : marge 1.5% (laisser respirer)

**Protection avancée** :
- Clés Redis par `position_id` (évite collisions multi-positions)
- TTL 7 jours avec refresh automatique
- Pump rider mode : désactive TP progressif sur gains >5% rapides (<10min)
- TP progressif inversé : strict sur gros gains, permissif sur petits

### Validation multi-niveaux
1. **Analyzer** : Validation technique des signaux
2. **Signal Aggregator** : Consensus et filtres critiques
3. **Coordinator** : Validation finale et orchestration
4. **Trader** : Contrôles pré-exécution

## 📊 Monitoring & Performance

### Métriques clés
- **ROI** : Retour sur investissement
- **Sharpe Ratio** : Ratio risque/rendement
- **Win Rate** : Taux de succès des trades
- **Max Drawdown** : Perte maximale
- **Volume quotidien** : Volume de trading

### Dashboard temps réel
- Positions ouvertes et PnL
- Graphiques techniques avec indicateurs
- Performance par stratégie
- Alertes et notifications

## 🔍 Health Checks & Diagnostics

### Endpoints de santé
Chaque service expose des endpoints de monitoring :
```bash
# Vérification globale
curl http://localhost:5009/api/health

# Services individuels
curl http://localhost:5010/health  # Gateway
curl http://localhost:5012/health  # Analyzer
curl http://localhost:5003/health  # Coordinator
```

### Logs centralisés
```bash
# Logs en temps réel
docker-compose logs -f

# Logs par service
docker-compose logs -f gateway
docker-compose logs -f trader
```

## 🚨 Sécurité

### Protection des clés API
- Variables d'environnement sécurisées
- Pas de logs des credentials
- Permissions API minimales requises
- Séparation trading/lecture

### Contrôles de trading
- Validation des ordres pré-exécution
- Limites de fréquence strictes
- Circuit breakers en cas d'anomalie
- Audit trail complet

## 📁 Structure du Projet

```
RootTrading/
├── 📊 Data Pipeline
│   ├── gateway/          # Collecte données Binance
│   ├── dispatcher/       # Routage Kafka→Redis  
│   └── market_analyzer/  # Calcul indicateurs
├── 🧠 Signal Processing
│   ├── analyzer/         # Stratégies trading
│   └── signal_aggregator/ # Consensus signaux
├── 💰 Trading & Portfolio
│   ├── coordinator/      # Orchestration
│   ├── trader/          # Exécution ordres
│   └── portfolio/       # Gestion portefeuille
├── 📈 Monitoring
│   └── visualization/   # Dashboard web
├── 🔧 Infrastructure
│   ├── shared/          # Modules communs
│   ├── database/        # Schémas DB
│   └── docker-compose.yml
└── 📋 Configuration
    ├── .env            # Variables environnement
    └── README.md       # Documentation
```

## 🛠️ Développement

### Architecture microservices
- Chaque service est indépendant
- Communication via Kafka/Redis
- APIs REST pour les interactions
- Health checks standardisés

### Standards de qualité
- Logging structuré
- Gestion d'erreurs robuste
- Tests automatisés
- Documentation complète

## 📈 Performance & Scalabilité

### Optimisations
- **Traitement parallèle** : Multi-processing pour les stratégies
- **Cache intelligent** : Redis pour les données fréquentes
- **Batch processing** : Traitement par lots optimisé
- **Connection pooling** : Pool de connexions DB

### Limites actuelles
- **22 symboles** simultanés maximum
- **28 stratégies** par symbole
- **5 positions** ouvertes simultanément
- **Mode SPOT uniquement** (pas de futures/margin)

## 🐛 Troubleshooting

### Problèmes courants

**Services ne démarrent pas**
```bash
# Vérifier les logs
docker-compose logs

# Redémarrer les services
docker-compose down && docker-compose up -d
```

**Problèmes de connectivité Binance**
```bash
# Vérifier les clés API
curl https://api.binance.com/api/v3/account

# Logs du gateway
docker-compose logs gateway
```

## 📞 Support & Contribution

### Documentation
- README détaillé dans chaque service
- API documentation avec FastAPI/Swagger
- Architecture diagrams dans `/docs`

### Monitoring
- Métriques Prometheus (optionnel)
- Logs centralisés avec ELK Stack
- Alerting configurable

## ⚖️ Disclaimers

⚠️ **AVERTISSEMENT** : Le trading de cryptomonnaies présente des risques de perte importants. Ce système est fourni à des fins éducatives et de recherche. 

- **Testez toujours** en mode paper trading d'abord
- **Commencez petit** avec des montants que vous pouvez vous permettre de perdre  
- **Surveillez régulièrement** les performances et ajustez si nécessaire
- **Aucune garantie** de profits n'est fournie

## 📄 License

MIT License - Voir LICENSE file pour les détails.

---

**RootTrading v1.0.9.908** - Système de trading crypto automatisé
📅 Dernière mise à jour : 2025-10-02
🔧 Trailing Sell v2.0 (optimisé scalp intraday)
🚀 28 stratégies actives | Architecture microservices | Gestion risques avancée
