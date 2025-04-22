FROM python:3.10-slim

WORKDIR /app

# Mise à jour du système et installation des dépendances
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie des requirements et installation
COPY scheduler/requirements.txt .
COPY requirements-shared.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie des modules partagés
COPY shared /app/shared

# Copie du code du service
COPY scheduler /app/scheduler

# Exposition du port
EXPOSE 5008

# Commande de démarrage
CMD ["python", "scheduler/src/monitor.py"]