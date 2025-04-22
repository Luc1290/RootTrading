FROM python:3.10-slim

WORKDIR /app

# Mise à jour du système et installation des dépendances
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    librdkafka-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copie des requirements et installation
COPY requirements-shared.txt /app/

# Ensuite copier les requirements spécifiques au service
COPY logger/requirements.txt /app/

# Installer les dépendances
# Remplacer la référence relative par le chemin absolu
RUN sed -i 's|-r ../requirements-shared.txt|-r /app/requirements-shared.txt|g' /app/requirements.txt && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copie des modules partagés
COPY shared /app/shared

# Copie du code du service
COPY logger /app/logger

# Exposition du port
EXPOSE 5005

# Commande de démarrage
CMD ["python", "logger/src/consumer.py"]