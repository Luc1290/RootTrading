FROM python:3.10-slim

WORKDIR /app

# Install build dependencies including TA-Lib (COPIE EXACTE DE ANALYZER)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    librdkafka-dev \
    wget \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib from source (COPIE EXACTE DE ANALYZER)
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

# Copy gateway requirements and modify
COPY gateway/requirements.txt /app/
RUN sed -i 's|-r ../requirements-shared.txt|-r /app/requirements-shared.txt|g' /app/requirements.txt

# Install NumPy first at a compatible version (COPIE EXACTE DE ANALYZER)
RUN pip install --no-cache-dir numpy==1.23.5

# Install other dependencies
RUN pip install --no-cache-dir -r /app/requirements-shared.txt && \
    pip install --no-cache-dir -r /app/requirements.txt

# Install TA-Lib Python wrapper separately with NumPy already installed (COPIE EXACTE DE ANALYZER)
RUN pip install --no-cache-dir ta-lib==0.4.24

# Copie des modules partagés
COPY shared /app/shared

# Copie du code du service
COPY gateway /app/gateway

# Exposition du port
EXPOSE 5010

# Commande de démarrage avec mode ultra-enrichi
CMD ["python", "gateway/src/main.py", "--debug"]