FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy frontend package files
COPY visualization/frontend/package*.json ./
RUN npm install

# Copy frontend source code
COPY visualization/frontend ./
RUN npm run build

# Python stage
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy shared requirements first
COPY requirements-shared.txt /app/

# Copy visualization requirements and modify
COPY visualization/requirements.txt /app/
RUN sed -i 's|-r ../requirements-shared.txt|-r /app/requirements-shared.txt|g' /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements-shared.txt && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy shared modules
COPY shared /app/shared

# Copy notifications module
COPY notifications /app/notifications

# Copy application code
COPY visualization/src/ ./src/

# Copy built frontend from frontend-builder stage
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port
EXPOSE 5009

# Run the application
CMD ["python", "src/main.py"]