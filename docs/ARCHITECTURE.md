# RootTrading System Architecture

## Overview

RootTrading is a distributed cryptocurrency trading system built using microservices architecture. The system is designed for high availability, scalability, and real-time performance.

## Architecture Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Binance API    │────▶│    Gateway      │────▶│     Redis       │
│   WebSocket     │     │    Service      │     │   Pub/Sub       │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                          │
                              ┌───────────────────────────┼───────────────────────────┐
                              │                           │                           │
                              ▼                           ▼                           ▼
                    ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
                    │                 │         │                 │         │                 │
                    │    Analyzer     │         │  Signal Aggr.   │         │   Dispatcher    │
                    │    Service      │         │    Service      │         │    Service      │
                    │                 │         │                 │         │                 │
                    └────────┬────────┘         └────────┬────────┘         └────────┬────────┘
                             │                           │                           │
                             └───────────────┬───────────┘                           │
                                             │                                       │
                                             ▼                                       ▼
                                   ┌─────────────────┐                     ┌─────────────────┐
                                   │                 │                     │                 │
                                   │  Coordinator    │◀────────────────────│     Kafka       │
                                   │    Service      │                     │   Message Bus   │
                                   │                 │                     │                 │
                                   └────────┬────────┘                     └─────────────────┘
                                            │
                              ┌─────────────┼─────────────┐
                              │             │             │
                              ▼             ▼             ▼
                    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
                    │                 │ │                 │ │                 │
                    │     Trader      │ │   Portfolio     │ │   PostgreSQL    │
                    │    Service      │ │    Service      │ │    Database     │
                    │                 │ │                 │ │                 │
                    └─────────────────┘ └─────────────────┘ └─────────────────┘
                              │                   │
                              ▼                   ▼
                    ┌─────────────────┐ ┌─────────────────┐
                    │                 │ │                 │
                    │  Binance API    │ │  Binance API    │
                    │   REST/Orders   │ │  REST/Account   │
                    │                 │ │                 │
                    └─────────────────┘ └─────────────────┘
```

## Service Communication Patterns

### 1. Market Data Flow
- **Protocol**: WebSocket → Redis Pub/Sub
- **Pattern**: Publisher-Subscriber
- **Data Format**: JSON messages with OHLCV data
- **Frequency**: Real-time (tick-by-tick)

### 2. Signal Generation Flow
- **Protocol**: Redis Pub/Sub + PostgreSQL
- **Pattern**: Event-driven processing
- **Data Flow**: 
  1. Gateway publishes market data to Redis
  2. Analyzer subscribes to market data
  3. Analyzer publishes signals to Redis
  4. Signal Aggregator processes and weights signals
  5. Coordinator receives aggregated signals

### 3. Trade Execution Flow
- **Protocol**: Kafka + REST API
- **Pattern**: Command-Event Sourcing
- **Flow**:
  1. Coordinator creates trade cycle
  2. Publishes trade command to Kafka
  3. Trader service executes order
  4. Portfolio service updates balances
  5. Events published back to Kafka

## Data Storage Architecture

### PostgreSQL Schema

```sql
-- Core Tables
market_data
├── id (BIGSERIAL)
├── symbol (VARCHAR)
├── open_time (TIMESTAMP)
├── close_time (TIMESTAMP)
├── open (DECIMAL)
├── high (DECIMAL)
├── low (DECIMAL)
├── close (DECIMAL)
├── volume (DECIMAL)
└── timeframe (VARCHAR)

signals
├── id (BIGSERIAL)
├── symbol (VARCHAR)
├── strategy (VARCHAR)
├── signal_type (VARCHAR)
├── strength (DECIMAL)
├── price (DECIMAL)
├── timestamp (TIMESTAMP)
└── metadata (JSONB)

trade_cycles
├── id (UUID)
├── symbol (VARCHAR)
├── status (VARCHAR)
├── entry_signal_id (BIGINT)
├── entry_price (DECIMAL)
├── exit_price (DECIMAL)
├── quantity (DECIMAL)
├── pnl (DECIMAL)
├── created_at (TIMESTAMP)
└── updated_at (TIMESTAMP)

orders
├── id (UUID)
├── cycle_id (UUID)
├── binance_order_id (BIGINT)
├── symbol (VARCHAR)
├── side (VARCHAR)
├── type (VARCHAR)
├── quantity (DECIMAL)
├── price (DECIMAL)
├── status (VARCHAR)
├── created_at (TIMESTAMP)
└── updated_at (TIMESTAMP)
```

### Redis Data Structures

```
# Market Data Channels
market:BTCUSDT:1m    # Real-time candles
market:BTCUSDT:5m
market:BTCUSDT:15m

# Signal Channels  
signals:raw          # Raw signals from analyzer
signals:aggregated   # Weighted signals from aggregator

# State Management
positions:active     # Active positions (Hash)
cycles:active        # Active trade cycles (Set)
balances:current     # Current account balances (Hash)

# Configuration
config:symbols       # Trading pairs (Set)
config:strategies    # Active strategies (Hash)
```

## Scalability Considerations

### Horizontal Scaling
- **Stateless Services**: All services except coordinator are stateless
- **Load Balancing**: Can run multiple instances of analyzer/trader services
- **Partitioning**: Kafka topics partitioned by symbol

### Vertical Scaling
- **Resource Allocation**:
  - Gateway: CPU-bound (WebSocket processing)
  - Analyzer: CPU-intensive (indicator calculations)
  - Database: Memory and I/O intensive
  - Redis: Memory-intensive

### Performance Optimizations
1. **Caching**: Redis for hot data (recent candles, active positions)
2. **Batch Processing**: Bulk inserts for market data
3. **Connection Pooling**: PostgreSQL and Redis connection pools
4. **Async Processing**: Non-blocking I/O in Python services

## Security Architecture

### API Security
- **Authentication**: API key/secret stored in environment variables
- **Encryption**: TLS for all external communications
- **Rate Limiting**: Respects Binance API limits

### Internal Security
- **Network Isolation**: Docker network isolation
- **Secrets Management**: Environment variables for sensitive data
- **Access Control**: Service-to-service authentication via shared secrets

## Monitoring and Observability

### Metrics Collection
- **Service Health**: Health check endpoints
- **Performance Metrics**: Response times, throughput
- **Business Metrics**: Trade success rate, P&L

### Logging Strategy
- **Structured Logging**: JSON format logs
- **Log Levels**: DEBUG, INFO, WARNING, ERROR
- **Centralized Collection**: Docker logs aggregation

### Alerting
- **Critical Alerts**:
  - Service down
  - Database connection failure
  - API rate limit exceeded
  - Abnormal trading activity

## Disaster Recovery

### Backup Strategy
- **Database**: Daily PostgreSQL backups
- **Configuration**: Version controlled in Git
- **State Recovery**: Redis persistence for critical data

### Failure Scenarios
1. **Service Failure**: Docker restart policies
2. **Database Failure**: Read replicas for failover
3. **Network Partition**: Eventual consistency model
4. **Exchange API Failure**: Circuit breaker pattern

## Development Workflow

### Local Development
```bash
# Start infrastructure
docker-compose up -d postgres redis kafka

# Run services locally
python Services/gateway/src/main.py
python Services/analyzer/src/main.py
```

### Testing Strategy
- **Unit Tests**: Per-service testing
- **Integration Tests**: Multi-service scenarios
- **Performance Tests**: Load testing with historical data
- **Backtesting**: Strategy validation against historical data