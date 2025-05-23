FROM python:3.10-slim

WORKDIR /app

# Mise à jour du système et installation des dépendances
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copie des requirements et installation
COPY requirements-shared.txt /app/

# Ensuite copier les requirements spécifiques au service
COPY pnl_tracker/requirements.txt /app/
    
# Installer les dépendances
# Remplacer la référence relative par le chemin absolu
RUN sed -i 's|-r ../requirements-shared.txt|-r /app/requirements-shared.txt|g' /app/requirements.txt && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copie des modules partagés
COPY shared /app/shared

# Copie du code du service
COPY pnl_tracker /app/pnl_tracker

# Copie des stratégies pour le strategy tuner (optionnel)
COPY analyzer/strategies /app/analyzer/strategies || echo "Stratégies non trouvées, optimisation désactivée"

# Créer les répertoires analyzer si nécessaire
RUN mkdir -p /app/analyzer && touch /app/analyzer/__init__.py
RUN test -d /app/analyzer/strategies && touch /app/analyzer/strategies/__init__.py || echo "Pas de stratégies à initialiser"

# Exposition du port
EXPOSE 5006

# Créer le répertoire pour les exports
RUN mkdir -p /app/exports && chmod 777 /app/exports
RUN mkdir -p /app/tuning_results && chmod 777 /app/tuning_results

# Commande d'initialisation et de démarrage
CMD ["sh", "-c", "python pnl_tracker/src/pnl_init.py && python pnl_tracker/src/main.py"]