# RootTrading API Documentation

## Service APIs

Each service in the RootTrading system exposes internal APIs for inter-service communication. This document details the API contracts and message formats.

## 1. Gateway Service API

### WebSocket Subscriptions
The Gateway service subscribes to Binance WebSocket streams and publishes normalized data.

#### Published Events (Redis Pub/Sub)

**Channel**: `market:{symbol}:{timeframe}`

**Message Format**:
```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1m",
  "open_time": 1699920000000,
  "close_time": 1699920059999,
  "open": "43250.50",
  "high": "43280.00",
  "low": "43240.00",
  "close": "43270.25",
  "volume": "125.543",
  "trades": 1523,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### Configuration API

**GET** `/api/config/symbols`
```json
{
  "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
  "timeframes": ["1m", "5m", "15m", "1h"]
}
```

## 2. Analyzer Service API

### Signal Publishing

**Channel**: `signals:raw`

**Message Format**:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "symbol": "BTCUSDT",
  "strategy": "macd_crossover",
  "signal_type": "BUY",
  "strength": 0.85,
  "price": "43270.25",
  "timeframe": "15m",
  "indicators": {
    "macd": 125.5,
    "signal": 120.3,
    "histogram": 5.2,
    "rsi": 65.4
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Strategy Management

**GET** `/api/strategies`
```json
{
  "strategies": [
    {
      "name": "macd_crossover",
      "enabled": true,
      "parameters": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
      }
    },
    {
      "name": "rsi_oversold",
      "enabled": true,
      "parameters": {
        "period": 14,
        "oversold": 30,
        "overbought": 70
      }
    }
  ]
}
```

## 3. Signal Aggregator API

### Aggregated Signals

**Channel**: `signals:aggregated`

**Message Format**:
```json
{
  "id": "660e8400-e29b-41d4-a716-446655440001",
  "symbol": "BTCUSDT",
  "signal_type": "BUY",
  "aggregated_strength": 0.78,
  "price": "43270.25",
  "strategies": [
    {
      "name": "macd_crossover",
      "strength": 0.85,
      "weight": 0.6
    },
    {
      "name": "rsi_oversold",
      "strength": 0.65,
      "weight": 0.4
    }
  ],
  "bayesian_confidence": 0.82,
  "timestamp": "2024-01-15T10:30:15Z"
}
```

### Weight Management

**GET** `/api/weights/{symbol}`
```json
{
  "symbol": "BTCUSDT",
  "weights": {
    "macd_crossover": 0.6,
    "rsi_oversold": 0.4,
    "bollinger_squeeze": 0.55
  },
  "last_updated": "2024-01-15T10:00:00Z"
}
```

## 4. Coordinator Service API

### Trade Cycle Management

**POST** `/api/cycles/create`
```json
{
  "signal_id": "660e8400-e29b-41d4-a716-446655440001",
  "symbol": "BTCUSDT",
  "side": "BUY",
  "quantity": 0.1,
  "entry_price": "43270.25",
  "stop_loss": "42400.00",
  "take_profit": "44100.00"
}
```

**Response**:
```json
{
  "cycle_id": "770e8400-e29b-41d4-a716-446655440002",
  "status": "CREATED",
  "created_at": "2024-01-15T10:30:30Z"
}
```

### Kafka Events

**Topic**: `trade.events`

**Event Types**:

1. **CycleCreated**
```json
{
  "event_type": "CycleCreated",
  "cycle_id": "770e8400-e29b-41d4-a716-446655440002",
  "symbol": "BTCUSDT",
  "side": "BUY",
  "quantity": 0.1,
  "entry_price": "43270.25",
  "timestamp": "2024-01-15T10:30:30Z"
}
```

2. **CycleUpdated**
```json
{
  "event_type": "CycleUpdated",
  "cycle_id": "770e8400-e29b-41d4-a716-446655440002",
  "status": "FILLED",
  "fill_price": "43268.50",
  "timestamp": "2024-01-15T10:30:45Z"
}
```

## 5. Trader Service API

### Order Execution

**Flask REST API** (Port: 5001)

**POST** `/execute_order`
```json
{
  "symbol": "BTCUSDT",
  "side": "BUY",
  "quantity": 0.1,
  "order_type": "MARKET",
  "cycle_id": "770e8400-e29b-41d4-a716-446655440002"
}
```

**Response**:
```json
{
  "order_id": "123456789",
  "binance_order_id": "987654321",
  "status": "FILLED",
  "executed_quantity": 0.1,
  "executed_price": "43268.50",
  "commission": 0.0001,
  "commission_asset": "BTC"
}
```

### Position Management

**GET** `/api/positions/active`
```json
{
  "positions": [
    {
      "symbol": "BTCUSDT",
      "side": "LONG",
      "quantity": 0.1,
      "entry_price": "43268.50",
      "current_price": "43350.00",
      "unrealized_pnl": 8.15,
      "unrealized_pnl_percent": 0.19
    }
  ]
}
```

## 6. Portfolio Service API

### Account Information

**GET** `/api/account/balance`
```json
{
  "balances": {
    "BTC": {
      "free": "0.5432",
      "locked": "0.1000",
      "total": "0.6432"
    },
    "USDT": {
      "free": "10543.21",
      "locked": "4325.00",
      "total": "14868.21"
    }
  },
  "total_value_usdt": "42765.43",
  "last_updated": "2024-01-15T10:30:00Z"
}
```

### Performance Metrics

**GET** `/api/performance/summary`
```json
{
  "total_trades": 156,
  "winning_trades": 98,
  "losing_trades": 58,
  "win_rate": 0.628,
  "total_pnl": 2543.67,
  "total_pnl_percent": 12.45,
  "sharpe_ratio": 1.85,
  "max_drawdown": -5.23,
  "average_trade_duration": "4h 23m",
  "best_trade": {
    "symbol": "BTCUSDT",
    "pnl": 543.21,
    "pnl_percent": 5.43
  },
  "worst_trade": {
    "symbol": "ETHUSDT",
    "pnl": -123.45,
    "pnl_percent": -2.10
  }
}
```

## 7. Dispatcher Service API

### Message Routing

The Dispatcher service routes messages between services based on configuration.

**Kafka Topics**:
- `market.data` - Market data events
- `signals.raw` - Raw trading signals
- `signals.aggregated` - Aggregated signals
- `trade.events` - Trade lifecycle events
- `portfolio.updates` - Portfolio changes

### Event Subscriptions

**GET** `/api/subscriptions`
```json
{
  "subscriptions": [
    {
      "service": "coordinator",
      "topics": ["signals.aggregated", "portfolio.updates"]
    },
    {
      "service": "trader",
      "topics": ["trade.events"]
    }
  ]
}
```

## Common Response Formats

### Success Response
```json
{
  "status": "success",
  "data": { ... },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Error Response
```json
{
  "status": "error",
  "error": {
    "code": "INSUFFICIENT_BALANCE",
    "message": "Insufficient balance for order",
    "details": {
      "required": "100 USDT",
      "available": "95 USDT"
    }
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Rate Limiting

Internal APIs implement rate limiting to prevent service overload:

- **Default Rate**: 100 requests per minute
- **Burst Rate**: 20 requests per second
- **Headers**:
  - `X-RateLimit-Limit`: Maximum requests allowed
  - `X-RateLimit-Remaining`: Requests remaining
  - `X-RateLimit-Reset`: Reset timestamp

## Authentication

Internal services authenticate using shared secrets:

```http
Authorization: Bearer {SERVICE_TOKEN}
X-Service-Name: {SERVICE_NAME}
```

## WebSocket Connections

Some services expose WebSocket endpoints for real-time updates:

### Trader Service WebSocket
```javascript
ws://trader:5001/ws

// Subscribe to position updates
{
  "action": "subscribe",
  "channel": "positions"
}

// Position update event
{
  "event": "position_update",
  "data": {
    "symbol": "BTCUSDT",
    "quantity": 0.1,
    "unrealized_pnl": 15.43
  }
}
```