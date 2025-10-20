# ============================================================================
# ROOT TRADING - Gateway Service (OPTIMIZED)
# ============================================================================
# Utilise l'image de base avec Python 3.11 et dépendances communes
# ============================================================================

FROM roottrading-base:latest

WORKDIR /app

# Copy requirements files (layer cache optimization)
COPY requirements-shared.txt /app/
COPY gateway/requirements.txt /app/gateway-requirements.txt

# Fix relative path in requirements
RUN sed -i 's|-r ../requirements-shared.txt|-r /app/requirements-shared.txt|g' /app/gateway-requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements-shared.txt && \
    pip install --no-cache-dir -r /app/gateway-requirements.txt

# Copy shared modules
COPY shared /app/shared

# Copy gateway code
COPY gateway /app/gateway

# Configuration de l'environnement Python
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5010

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5010/health').read()" || exit 1

# Start command (using -m to run as module for proper imports)
CMD ["python", "-m", "gateway.src.main_simple"]
