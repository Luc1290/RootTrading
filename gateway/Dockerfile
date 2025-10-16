# ============================================================================
# ROOT TRADING - Gateway Service (OPTIMIZED)
# ============================================================================
# Service simple sans TA-Lib - websockets + kafka uniquement
# ============================================================================

FROM python:3.10-slim

WORKDIR /app

# Install minimal build dependencies (pour confluent-kafka)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    librdkafka-dev \
    && rm -rf /var/lib/apt/lists/*

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

# Expose port
EXPOSE 5010

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5010/health').read()" || exit 1

# Start command
CMD ["python", "gateway/src/main_simple.py"]
