# ============================================================================
# ROOT TRADING - Dispatcher Service (OPTIMIZED)
# ============================================================================
# Utilise l'image de base avec Python 3.11 et dépendances communes
# ============================================================================

FROM roottrading-base:latest

WORKDIR /app

# Copie des requirements et installation
COPY requirements-shared.txt /app/

# Ensuite copier les requirements spécifiques au service
COPY dispatcher/requirements.txt /app/

# Installer les dépendances
RUN sed -i 's|-r ../requirements-shared.txt|-r /app/requirements-shared.txt|g' /app/requirements.txt && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copie des modules partagés
COPY shared /app/shared

# Copie du code du service
COPY dispatcher /app/dispatcher

# Configuration de l'environnement Python
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Exposition du port
EXPOSE 5004

# Commande de démarrage
CMD ["python", "dispatcher/src/main.py"]