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

### 1. Market Data Flow (Enhanced)
- **Protocol**: WebSocket → Kafka → PostgreSQL
- **Pattern**: Publisher-Subscriber with persistent storage
- **Data Format**: JSON messages with OHLCV + 18 technical indicators
- **Frequency**: Real-time (5m candles with enriched data)
- **Enhancement**: UltraDataFetcher calculates and stores ADX, RSI, MACD, Bollinger, Stochastic, etc.

### 2. Signal Generation Flow (Hybrid Approach)
- **Protocol**: Kafka + Enhanced Regime Detection
- **Pattern**: Event-driven with intelligent filtering
- **Data Flow**: 
  1. Gateway calculates enriched market data with ADX smoothing
  2. DatabasePersister saves all 18 indicators to PostgreSQL
  3. Analyzer generates signals with technical indicators
  4. **Enhanced Signal Aggregator** with adaptive filtering:
     - ADX-based regime detection (18/23/32/42 thresholds)
     - Adaptive debounce (0.5x-1.8x based on trend strength)
     - Balanced BUY/SELL signal generation
  5. **Enhanced Coordinator** trusts aggregated signals (no re-validation)

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
-- Core Tables (Enhanced)
market_data
├── time (TIMESTAMP)
├── symbol (VARCHAR)
├── open (DECIMAL)
├── high (DECIMAL)
├── low (DECIMAL)
├── close (DECIMAL)
├── volume (DECIMAL)
├── enhanced (BOOLEAN)
├── ultra_enriched (BOOLEAN)
├── -- Technical Indicators (18 total)
├── rsi_14 (DECIMAL)
├── ema_12, ema_26, ema_50 (DECIMAL)
├── sma_20, sma_50 (DECIMAL)
├── macd_line, macd_signal, macd_histogram (DECIMAL)
├── bb_upper, bb_middle, bb_lower, bb_position, bb_width (DECIMAL)
├── atr_14 (DECIMAL)
├── adx_14, plus_di, minus_di (DECIMAL) -- New: ADX with smoothing
├── stoch_k, stoch_d, stoch_rsi (DECIMAL)
├── williams_r, cci_20, mfi_14 (DECIMAL)
├── vwap_10, roc_10, roc_20 (DECIMAL)
├── obv, trend_angle, pivot_count (DECIMAL)
├── momentum_10, volume_ratio, avg_volume_20 (DECIMAL)
└── supertrend, supertrend_direction (DECIMAL) -- New: Complete Supertrend

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

## Hybrid Trading Approach (2025 Enhancement)

### Enhanced Technical Analysis
- **ADX Smoothing**: EMA(3) applied to ADX for stable regime detection
- **Adaptive Thresholds**: Optimized for crypto volatility (18/23/32/42 vs 20/25/35/45)
- **18 Technical Indicators**: Comprehensive market analysis stored in database
- **Complete Supertrend**: Full historical implementation with band continuity rules

### Intelligent Signal Processing
- **Adaptive Debounce**: 
  - Strong trends (ADX ≥ 42): 0.5x debounce (faster signals)
  - Moderate trends (ADX 23-42): 1.0x debounce (normal)
  - Range markets (ADX < 23): 1.8x debounce (slower, less noise)
- **Balanced Signal Generation**: Eliminates BUY/SELL imbalance
- **Enhanced OBV Validation**: Historical trend analysis with linear regression
- **Trust-Based Architecture**: Coordinator trusts signal aggregator decisions

### Performance Targets
- **Frequency**: 8-12 trades/day (vs previous imbalance)
- **Win Rate**: Improved through intelligent filtering
- **Regime Stability**: 15-30 min stable periods vs erratic changes
- **Signal Quality**: Higher precision through multi-layer validation

### Configuration Management
```bash
# Hybrid Mode Settings (.env)
ADX_HYBRID_MODE=true
ADX_SMOOTHING_PERIOD=3
ADX_NO_TREND_THRESHOLD=18
ADX_WEAK_TREND_THRESHOLD=23
ADX_TREND_THRESHOLD=32
ADX_STRONG_TREND_THRESHOLD=42
SIGNAL_COOLDOWN_MINUTES=3
VOTE_THRESHOLD=0.35
CONFIDENCE_THRESHOLD=0.60
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