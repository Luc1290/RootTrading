FROM python:3.10-slim

WORKDIR /app

# Install build dependencies including TA-Lib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    librdkafka-dev \
    wget \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib from source
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Copy shared requirements first
COPY requirements-shared.txt /app/

# Copy market_analyzer requirements and modify
COPY market_analyzer/requirements.txt /app/
RUN sed -i 's|-r ../requirements-shared.txt|-r /app/requirements-shared.txt|g' /app/requirements.txt

# Install NumPy first at a compatible version
RUN pip install --no-cache-dir numpy==1.23.5

# Install other dependencies
RUN pip install --no-cache-dir -r /app/requirements-shared.txt && \
    pip install --no-cache-dir -r /app/requirements.txt

# Install TA-Lib Python wrapper separately with NumPy already installed
RUN pip install --no-cache-dir ta-lib==0.4.24

# Copy shared modules
COPY shared /app/shared

# Copy market_analyzer code
COPY market_analyzer /app/market_analyzer

# Expose API port
EXPOSE 5020

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5020/health || exit 1

# Command to start the Market Analyzer service
CMD ["python", "market_analyzer/src/main.py"]