FROM python:3.12-slim

WORKDIR /app

# Installer les dépendances système pour Snappy
RUN apt-get update && apt-get install -y \
    libsnappy-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements
COPY signal_aggregator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier les modules partagés
COPY shared/ ./shared/

# Copier le code source du signal_aggregator
COPY signal_aggregator/src/ ./src/

# Attendre que les services soient prêts puis démarrer
CMD ["sh", "-c", "sleep 10 && python -u src/main.py"]