# RootTrading Deployment Guide

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.10+
- PostgreSQL client tools
- 8GB RAM minimum (16GB recommended)
- 50GB disk space
- Binance API credentials

## Environment Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd RootTrading
```

### 2. Environment Configuration

Create `.env` file in the project root:

```bash
# Binance API Configuration
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
BINANCE_TESTNET=false

# Database Configuration
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=roottrading
POSTGRES_USER=rootuser
POSTGRES_PASSWORD=secure_password_here

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=redis_password_here

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
KAFKA_GROUP_ID=roottrading

# Service Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
MAX_CONCURRENT_TRADES=5
POSITION_SIZE_PERCENT=2.0

# Trading Pairs
TRADING_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT
TRADING_TIMEFRAMES=1m,5m,15m,1h
```

### 3. Create Docker Networks

```bash
docker network create roottrading_network
```

## Deployment Steps

### 1. Build Docker Images

```bash
# Build all services
docker-compose build

# Or build specific service
docker-compose build gateway
```

### 2. Initialize Database

```bash
# Start only database service
docker-compose up -d postgres

# Wait for PostgreSQL to be ready
sleep 10

# Apply migrations
python database/apply_migrations.py

# Create indexes for performance
psql -h localhost -U rootuser -d roottrading < database/indexes.sql
```

### 3. Start Infrastructure Services

```bash
# Start message brokers and cache
docker-compose up -d redis kafka zookeeper
```

### 4. Deploy Application Services

```bash
# Start all services
docker-compose up -d

# Or start services individually in order
docker-compose up -d gateway
docker-compose up -d analyzer
docker-compose up -d signal_aggregator
docker-compose up -d coordinator
docker-compose up -d trader
docker-compose up -d portfolio
docker-compose up -d dispatcher
```

### 5. Verify Deployment

```bash
# Check service status
docker-compose ps

# Check logs
docker-compose logs -f --tail=100

# Test health endpoints
curl http://localhost:8001/health  # Gateway
curl http://localhost:8002/health  # Analyzer
curl http://localhost:8003/health  # Signal Aggregator
```

## Production Configuration

### 1. Resource Limits

Update `docker-compose.yml` with production limits:

```yaml
services:
  gateway:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G

  analyzer:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 4G
        reservations:
          cpus: '2'
          memory: 2G
```

### 2. Persistence Volumes

Configure persistent storage:

```yaml
volumes:
  postgres_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/postgres

  redis_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /data/redis
```

### 3. Logging Configuration

Set up centralized logging:

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "100m"
    max-file: "10"
    labels: "service"
```

## Monitoring Setup

### 1. Health Checks

Configure Docker health checks:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### 2. Metrics Collection

Enable Prometheus metrics:

```bash
# Add to .env
ENABLE_METRICS=true
METRICS_PORT=9090

# Deploy Prometheus
docker-compose -f docker-compose.monitoring.yml up -d
```

### 3. Alerting Rules

Create `alerts.yml`:

```yaml
groups:
  - name: trading_alerts
    rules:
      - alert: ServiceDown
        expr: up{job="roottrading"} == 0
        for: 5m
        annotations:
          summary: "Service {{ $labels.instance }} is down"

      - alert: HighErrorRate
        expr: rate(errors_total[5m]) > 0.1
        for: 5m
        annotations:
          summary: "High error rate detected"

      - alert: DatabaseConnectionFailure
        expr: postgres_connections_active == 0
        for: 1m
        annotations:
          summary: "Database connection lost"
```

## Security Hardening

### 1. Network Security

```bash
# Create internal network
docker network create --internal roottrading_internal

# Update docker-compose.yml
networks:
  internal:
    external:
      name: roottrading_internal
  external:
    external:
      name: roottrading_network
```

### 2. Secrets Management

Use Docker secrets:

```bash
# Create secrets
echo "your_api_key" | docker secret create binance_api_key -
echo "your_api_secret" | docker secret create binance_api_secret -

# Reference in docker-compose.yml
secrets:
  binance_api_key:
    external: true
  binance_api_secret:
    external: true
```

### 3. SSL/TLS Configuration

For production, use nginx reverse proxy:

```nginx
server {
    listen 443 ssl;
    server_name api.roottrading.com;

    ssl_certificate /etc/ssl/certs/roottrading.crt;
    ssl_certificate_key /etc/ssl/private/roottrading.key;

    location / {
        proxy_pass http://gateway:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Backup and Recovery

### 1. Database Backup

Create backup script `backup.sh`:

```bash
#!/bin/bash
BACKUP_DIR=/backups/postgres
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup
docker exec postgres pg_dump -U rootuser roottrading | gzip > $BACKUP_DIR/backup_$TIMESTAMP.sql.gz

# Keep only last 7 days
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +7 -delete
```

### 2. Automated Backups

Add to crontab:

```bash
# Daily backup at 2 AM
0 2 * * * /opt/roottrading/backup.sh
```

### 3. Recovery Procedure

```bash
# Stop services
docker-compose stop

# Restore database
gunzip < backup_20240115_020000.sql.gz | docker exec -i postgres psql -U rootuser roottrading

# Restart services
docker-compose start
```

## Scaling Guidelines

### Horizontal Scaling

1. **Analyzer Service**: Can run multiple instances
   ```bash
   docker-compose up -d --scale analyzer=3
   ```

2. **Signal Aggregator**: Partition by symbol
   ```yaml
   environment:
     - PARTITION_KEY=symbol
     - INSTANCE_ID=${INSTANCE_ID}
   ```

### Vertical Scaling

1. **Database Optimization**:
   - Increase shared_buffers
   - Tune work_mem
   - Add read replicas

2. **Redis Optimization**:
   - Enable persistence
   - Configure maxmemory policy
   - Use Redis Cluster for large datasets

## Troubleshooting

### Common Issues

1. **Service Won't Start**
   ```bash
   # Check logs
   docker-compose logs service_name
   
   # Restart service
   docker-compose restart service_name
   ```

2. **Database Connection Errors**
   ```bash
   # Check PostgreSQL status
   docker exec postgres pg_isready
   
   # Check connections
   docker exec postgres psql -U rootuser -c "SELECT count(*) FROM pg_stat_activity;"
   ```

3. **Memory Issues**
   ```bash
   # Check memory usage
   docker stats
   
   # Increase memory limits in docker-compose.yml
   ```

### Debug Mode

Enable debug logging:

```bash
# Set in .env
LOG_LEVEL=DEBUG

# Or per service
docker-compose run -e LOG_LEVEL=DEBUG analyzer
```

## Maintenance

### Regular Tasks

1. **Weekly**:
   - Check disk space
   - Review error logs
   - Update dependencies

2. **Monthly**:
   - Performance analysis
   - Database vacuum
   - Security updates

3. **Quarterly**:
   - Full system backup
   - Disaster recovery test
   - Strategy performance review

### Update Procedure

```bash
# Pull latest changes
git pull origin main

# Rebuild images
docker-compose build

# Rolling update
docker-compose up -d --no-deps --build gateway
# Wait and verify
docker-compose up -d --no-deps --build analyzer
# Continue for each service...
```