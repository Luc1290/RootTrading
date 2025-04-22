.PHONY: build up down logs ps restart clean

# Variables
COMPOSE = docker-compose
ENV_FILE = .env

# Commandes principales
build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

ps:
	$(COMPOSE) ps

restart:
	$(COMPOSE) restart

clean:
	$(COMPOSE) down -v --remove-orphans

# Commandes spécifiques par service
up-infra:
	$(COMPOSE) up -d redis kafka db

up-gateway:
	$(COMPOSE) up -d gateway

up-analyzer:
	$(COMPOSE) up -d analyzer

up-trader:
	$(COMPOSE) up -d trader

up-portfolio:
	$(COMPOSE) up -d portfolio

up-frontend:
	$(COMPOSE) up -d frontend

# Commandes de logs par service
logs-gateway:
	$(COMPOSE) logs -f gateway

logs-analyzer:
	$(COMPOSE) logs -f analyzer

logs-trader:
	$(COMPOSE) logs -f trader

logs-portfolio:
	$(COMPOSE) logs -f portfolio

# Commandes d'installation
install-requirements:
	pip install -r requirements-shared.txt
	pip install -r gateway/requirements.txt
	pip install -r analyzer/requirements.txt
	pip install -r trader/requirements.txt
	pip install -r portfolio/requirements.txt

# Commandes de base de données
db-init:
	psql -h localhost -U postgres -d trading -f database/schema.sql

db-backup:
	mkdir -p backups
	pg_dump -h localhost -U postgres -d trading > backups/trading_backup_$$(date +"%Y%m%d_%H%M%S").sql

db-reset:
	$(COMPOSE) down -v --remove-orphans db
	$(COMPOSE) up -d db
	sleep 5
	psql -h localhost -U postgres -d trading -f database/schema.sql

# Commandes de test et développement
test:
	echo "Exécution des tests (à implémenter)"

lint:
	echo "Vérification du code (à implémenter)"

# Commandes de déploiement
deploy-dev:
	echo "Déploiement en environnement de développement"
	$(COMPOSE) -f docker-compose.yml up -d

deploy-prod:
	echo "Déploiement en environnement de production"
	$(COMPOSE) -f docker-compose.yml -f docker-compose.prod.yml up -d

# Aide
help:
	@echo "Commandes disponibles:"
	@echo "  make build          - Construit les images Docker"
	@echo "  make up             - Démarre tous les services"
	@echo "  make down           - Arrête tous les services"
	@echo "  make logs           - Affiche les logs de tous les services"
	@echo "  make ps             - Liste les services en cours d'exécution"
	@echo "  make restart        - Redémarre tous les services"
	@echo "  make clean          - Nettoie tout (y compris les volumes)"
	@echo "  make up-infra       - Démarre uniquement l'infrastructure (Redis, Kafka, PostgreSQL)"
	@echo "  make up-[service]   - Démarre un service spécifique"
	@echo "  make logs-[service] - Affiche les logs d'un service spécifique"
	@echo "  make db-init        - Initialise la base de données"
	@echo "  make db-backup      - Sauvegarde la base de données"
	@echo "  make db-reset       - Réinitialise la base de données"