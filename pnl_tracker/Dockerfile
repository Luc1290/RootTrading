FROM python:3.10-slim

WORKDIR /app

# Mise à jour du système et installation des dépendances
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copie des requirements et installation
COPY pnl_tracker/requirements.txt .
COPY requirements-shared.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie des modules partagés
COPY shared /app/shared

# Copie du code du service
COPY pnl_tracker /app/pnl_tracker

# Exposition du port
EXPOSE 5006

# Commande de démarrage
CMD ["python", "pnl_tracker/src/pnl_logger.py"]