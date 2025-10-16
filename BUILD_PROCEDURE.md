# 🏗️ ROOT Trading - Procédure de Build

## ⚙️ Build Initial (une seule fois)

```bash
# Build l'image de base avec TA-Lib (5-10 min)
docker build -t roottrading-base:latest -f Dockerfile.base .
```

## 🚀 Démarrage des Services (dans l'ordre)

```bash
# Infrastructure
docker-compose up -d --build redis db

# Messaging
docker-compose up -d --build zookeeper
docker-compose up -d --build kafka

# Services Core
docker-compose up -d --build dispatcher
docker-compose up -d --build gateway
docker-compose up -d --build market_analyzer
docker-compose up -d --build analyzer
docker-compose up -d --build signal_aggregator

# Services Trading
docker-compose up -d --build trader
docker-compose up -d --build coordinator
docker-compose up -d --build portfolio

# Interface
docker-compose up -d --build visualization
```

## 🔄 Rebuild Après Changements

```bash
# Rebuild un service spécifique
docker-compose up -d --build analyzer

# Rebuild tous les services
docker-compose build && docker-compose up -d
```

## ⚠️ Notes

- **Image de base** : À rebuild uniquement si TA-Lib ou Python change
- **Services** : Rebuild automatique avec `--build` si code modifié
- **Ordre** : Respecter l'ordre (infrastructure → messaging → services)
