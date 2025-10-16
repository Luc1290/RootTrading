# ============================================================================
# ROOT TRADING - Market Analyzer Service (OPTIMIZED)
# ============================================================================
# Utilise l'image de base avec TA-Lib précompilé
# ============================================================================

FROM roottrading-base:latest

WORKDIR /app

# Copy requirements files (layer cache optimization)
COPY requirements-shared.txt /app/
COPY market_analyzer/requirements.txt /app/market-analyzer-requirements.txt

# Fix relative path in requirements
RUN sed -i 's|-r ../requirements-shared.txt|-r /app/requirements-shared.txt|g' /app/market-analyzer-requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements-shared.txt && \
    pip install --no-cache-dir -r /app/market-analyzer-requirements.txt

# Copy shared modules
COPY shared /app/shared

# Copy market_analyzer code
COPY market_analyzer /app/market_analyzer

# Expose API port
EXPOSE 5020

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5020/health').read()" || exit 1

# Command to start the Market Analyzer service
CMD ["python", "market_analyzer/src/main.py"]
