# RootTrading - Cryptocurrency Trading System

A microservices-based cryptocurrency trading system for Binance exchange, featuring real-time market analysis, signal generation, and automated trading execution.

## 🏗️ Architecture Overview

RootTrading is built using a distributed microservices architecture with the following core components:

### Core Services

- **Gateway Service** - Market data ingestion from Binance WebSocket streams
- **Analyzer Service** - Technical analysis and indicator calculation
- **Signal Aggregator** - Signal aggregation and Bayesian weight optimization
- **Coordinator Service** - Trade orchestration and lifecycle management
- **Trader Service** - Order execution and position management
- **Portfolio Service** - Portfolio tracking and performance monitoring
- **Dispatcher Service** - Message routing and service communication

### Infrastructure Components

- **PostgreSQL** - Primary data storage for trades, signals, and market data
- **Redis** - Real-time data caching and pub/sub messaging
- **Kafka** - Event streaming for service communication
- **Docker Compose** - Container orchestration

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.10+
- Binance API credentials

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd RootTrading
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your Binance API credentials and configuration

# Hybrid Mode Configuration (recommended)
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

3. Start the services:
```bash
docker-compose up -d
```

4. Apply database migrations:
```bash
python database/apply_migrations.py
```

## 📊 Services Documentation

### Gateway Service
Connects to Binance WebSocket streams to receive real-time market data for configured trading pairs.

**Key Features:**
- Real-time OHLCV data ingestion
- Multi-timeframe candle aggregation
- Market data normalization

### Analyzer Service
Performs technical analysis on incoming market data using 18+ comprehensive indicators.

**Supported Indicators:**
- Moving Averages (SMA, EMA) - Multiple periods
- RSI (Relative Strength Index) - 14/21 periods
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands with position and width
- Stochastic Oscillator (K/D)
- ADX with DI+ and DI- (Enhanced with EMA smoothing)
- ATR (Average True Range)
- OBV (On-Balance Volume) with trend validation
- Supertrend with band continuity
- Williams %R, CCI, MFI, ROC
- Custom indicators via TA-Lib

### Signal Aggregator (Enhanced Hybrid Mode)
Aggregates signals from multiple analysis strategies with intelligent market regime detection.

**Features:**
- Multi-strategy signal aggregation with market regime filtering
- Bayesian weight optimization
- Signal strength normalization
- Historical performance tracking
- **Adaptive Debounce**: 0.5x-1.8x based on ADX strength
- **Balanced Signal Generation**: Eliminates BUY/SELL imbalance
- **ADX Smoothing**: EMA(3) for stable regime detection
- **Enhanced OBV Validation**: Linear regression trend analysis

### Coordinator Service (Trust-Based Architecture)
Manages the complete trade lifecycle from signal to execution with enhanced trust model.

**Responsibilities:**
- Trade cycle creation and management
- Risk management rule enforcement
- Position sizing calculation
- Trade state coordination
- **Enhanced Trust Model**: Trusts signal aggregator decisions (no re-validation)
- **Regime-Aware Processing**: Adapts to market conditions

### Trader Service
Executes trades on Binance exchange with advanced order management.

**Features:**
- Market and limit order execution
- Trailing stop-loss management
- Position tracking
- Order status monitoring

### Portfolio Service
Tracks portfolio performance and manages account balances.

**Features:**
- Real-time P&L calculation
- Multi-asset portfolio tracking
- Performance metrics calculation
- Balance synchronization with Binance

## 🔧 Configuration

### Trading Pairs
Configure trading pairs in the gateway service configuration:
```python
# Services/gateway/src/config.py
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
```

### Risk Management
Adjust risk parameters in the coordinator service:
```python
# Services/coordinator/src/config.py
MAX_POSITION_SIZE = 0.02  # 2% of portfolio per position
MAX_CONCURRENT_TRADES = 5
STOP_LOSS_PERCENTAGE = 0.02  # 2% stop loss
```

### Technical Indicators
Configure analysis strategies in the analyzer service:
```python
# Services/analyzer/strategies/
# Add custom strategy files here
```

## 📈 Monitoring

### Debug Tools
The project includes various debug utilities in the `Debug/` directory:

- `active_cycles_analysis.py` - Monitor active trading cycles
- `strategy_analysis.py` - Analyze strategy performance
- `db_analysis.py` - Database health checks
- `hourly_price_analysis.py` - Price movement analysis

### Logging
Each service logs to stdout/stderr. View logs using:
```bash
docker-compose logs -f [service-name]
```

## 🧪 Development

### Running Tests
```bash
# Type checking
mypy .

# Linting
ruff check .

# Format code
ruff format .
```

### Adding New Strategies
1. Create a new strategy file in `Services/analyzer/strategies/`
2. Implement the `BaseStrategy` interface
3. Register the strategy in `strategy_loader.py`

## 🚨 Important Notes

- **API Keys Security**: Never commit API keys to version control
- **Risk Management**: Always test strategies with small amounts first
- **Market Conditions**: System performance varies with market volatility
- **Maintenance**: Regular database maintenance recommended for optimal performance

## 📝 License

[Your License Here]

## 🤝 Contributing

[Contributing Guidelines]