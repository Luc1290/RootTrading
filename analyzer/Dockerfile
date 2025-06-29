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

# Copy analyzer requirements and modify
COPY analyzer/requirements.txt /app/
RUN sed -i 's|-r ../requirements-shared.txt|-r /app/requirements-shared.txt|g' /app/requirements.txt

# Install NumPy first at a compatible version
RUN pip install --no-cache-dir numpy==1.23.5

# Install other dependencies but exclude TA-Lib
RUN pip install --no-cache-dir -r /app/requirements-shared.txt && \
    grep -v "ta-lib" /app/requirements.txt | pip install --no-cache-dir -r /dev/stdin

# Install TA-Lib Python wrapper separately with NumPy already installed
RUN pip install --no-cache-dir ta-lib==0.4.24

# Install pandas-ta for advanced indicators
RUN pip install --no-cache-dir pandas-ta==0.3.14b0

# Copy shared modules
COPY shared /app/shared

# Copy analyzer code
COPY analyzer /app/analyzer

# Expose port
EXPOSE 5012

# Start command
CMD ["python", "analyzer/src/main.py"]