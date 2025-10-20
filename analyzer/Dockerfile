# ============================================================================
# ROOT TRADING - Analyzer Service (OPTIMIZED)
# ============================================================================
# Utilise l'image de base avec TA-Lib précompilé
# ============================================================================

FROM roottrading-base:latest

WORKDIR /app

# Copy requirements files (layer cache optimization)
COPY requirements-shared.txt /app/
COPY analyzer/requirements.txt /app/analyzer-requirements.txt

# Fix relative path in requirements
RUN sed -i 's|-r ../requirements-shared.txt|-r /app/requirements-shared.txt|g' /app/analyzer-requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements-shared.txt && \
    pip install --no-cache-dir -r /app/analyzer-requirements.txt

# Copy shared modules
COPY shared /app/shared

# Copy analyzer code
COPY analyzer /app/analyzer

# Configuration de l'environnement Python
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5012

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5012/health').read()" || exit 1

# Start command
CMD ["python", "-m", "analyzer.src.main"]
