# ============================================================================
# ROOT TRADING - Signal Aggregator Service (OPTIMIZED)
# ============================================================================
# Utilise l'image de base avec TA-Lib précompilé et Python 3.11
# ============================================================================

FROM roottrading-base:latest

WORKDIR /app

# Copy shared requirements first
COPY requirements-shared.txt /app/

# Copy signal_aggregator requirements and modify
COPY signal_aggregator/requirements.txt /app/
RUN sed -i 's|-r ../requirements-shared.txt|-r /app/requirements-shared.txt|g' /app/requirements.txt

# Install NumPy first at a compatible version
RUN pip install --no-cache-dir numpy==1.23.5

# Install other dependencies but exclude TA-Lib
RUN pip install --no-cache-dir -r /app/requirements-shared.txt && \
    grep -v "ta-lib" /app/requirements.txt | pip install --no-cache-dir -r /dev/stdin

# Install TA-Lib Python wrapper separately with NumPy already installed
# Version 0.4.32 fixes const pointer compatibility issues
RUN pip install --no-cache-dir ta-lib==0.4.32

# Note: pandas-ta removed - incompatible with Python 3.10
# Using TA-Lib for technical indicators instead

# Copy shared modules
COPY shared /app/shared

# Copy signal_aggregator code
COPY signal_aggregator /app/signal_aggregator

# Configuration de l'environnement Python
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Attendre que les services soient prêts puis démarrer
# NOUVEAU: Version simplifiée sans validators complexes - IMPORTS CORRIGÉS
CMD ["python", "signal_aggregator/src/main_simple.py"]