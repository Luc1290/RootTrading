# ============================================================================
# ROOT TRADING - Portfolio Service (OPTIMIZED)
# ============================================================================
# Utilise l'image de base avec Python 3.11 et dépendances communes
# ============================================================================

FROM roottrading-base:latest

WORKDIR /app

# Copie des requirements et installation
COPY requirements-shared.txt /app/

# Ensuite copier les requirements spécifiques au service
COPY portfolio/requirements.txt /app/

# Installer les dépendances
# Remplacer la référence relative par le chemin absolu
RUN sed -i 's|-r ../requirements-shared.txt|-r /app/requirements-shared.txt|g' /app/requirements.txt && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copie des modules partagés
COPY shared /app/shared

# Copie du code du service
COPY portfolio /app/portfolio

# Configuration de l'environnement Python
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Exposition du port
EXPOSE 8000

# Commande de démarrage
WORKDIR /app/portfolio/src
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]