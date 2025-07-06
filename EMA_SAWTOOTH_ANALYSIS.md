# EMA Sawtooth Pattern Analysis - RootTrading System

## Executive Summary

After analyzing the entire RootTrading codebase, I've identified **multiple potential causes** of EMA sawtooth patterns across different timeframes. The system has a **hybrid approach** with some services using incremental EMA calculations while others still use traditional recalculation methods, creating inconsistencies.

## Key Findings

### ✅ **Incremental EMA Implementation EXISTS**
The system has a sophisticated incremental EMA calculation system in:
- `shared/src/technical_indicators.py`
- Gateway services (`binance_ws.py`, `ultra_data_fetcher.py`)
- Signal Aggregator incremental cache

### ❌ **BUT: Inconsistent Application Across Timeframes**

## Root Causes of Sawtooth Patterns

### 1. **Database Storage Limitation - CRITICAL ISSUE**

**Location**: `database/schema.sql` and migrations
**Problem**: The database `market_data` table only stores **1-minute data** but the system trades on **multiple timeframes** (1m, 5m, 15m, 1h, 4h).

```sql
-- Only stores 1m data, but system needs 5m, 15m, 1h, 4h
CREATE TABLE IF NOT EXISTS market_data (
    time TIMESTAMP NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    -- EMA columns store only 1m timeframe data
    ema_12 NUMERIC(16,8),
    ema_26 NUMERIC(16,8),
    ema_50 NUMERIC(16,8),
    -- No timeframe differentiation!
    PRIMARY KEY (time, symbol)
);
```

**Impact**: EMA values for different timeframes are either:
- All mixed together (causing confusion)
- Only calculated for 1m (causing recalculation for other timeframes)
- Lost between database reads

### 2. **Gateway Multi-Timeframe Processing**

**Location**: `gateway/src/ultra_data_fetcher.py` and `gateway/src/binance_ws.py`

**Problem**: The gateway fetches and processes data for multiple timeframes:

```python
# ultra_data_fetcher.py:32
self.timeframes = ['1m', '5m', '15m', '1h', '4h']  # Multi-timeframes

# binance_ws.py:55-67
for symbol in self.symbols:
    for tf in self.timeframes:
        self.stream_paths.append(f"{symbol_lower}@kline_{tf}")
```

But the incremental cache is timeframe-specific:
```python
# binance_ws.py:93-104
self.incremental_cache[symbol][tf] = {}
```

However, **database persistence ignores timeframe differentiation**.

### 3. **Analyzer Database Retrieval Issue**

**Location**: `analyzer/src/indicators/db_indicators.py`

**Problem**: The analyzer retrieves "enriched" data from the database, but there's no timeframe specification:

```python
# db_indicators.py:66-68
WHERE symbol = %s AND enhanced = true
ORDER BY time DESC 
LIMIT %s
```

This means the analyzer gets **mixed timeframe data** or **only 1m data**, then strategies that expect other timeframes will have incorrect EMA values.

### 4. **Strategy Timeframe Assumptions**

**Location**: `analyzer/strategies/ema_cross.py`

**Problem**: Strategies assume they're getting the correct timeframe data, but there's no timeframe validation:

```python
# ema_cross.py:53-54
current_ema12 = self._get_current_indicator(indicators, 'ema_12')
current_ema26 = self._get_current_indicator(indicators, 'ema_26')
```

The EMAs might be from **different timeframes** than expected, causing inappropriate signals.

### 5. **Signal Aggregator Multi-Timeframe Conflation**

**Location**: `signal_aggregator/src/signal_aggregator.py`

**Problem**: The signal aggregator processes signals from multiple timeframes but has a shared incremental cache:

```python
# signal_aggregator.py:91-92
self.ema_incremental_cache = defaultdict(lambda: defaultdict(dict))
```

This could mix EMA calculations across timeframes.

### 6. **Database Persister Timeframe Loss**

**Location**: `dispatcher/src/database_persister.py`

**Problem**: When data is saved to the database, timeframe information is extracted but **not stored**:

```python
# database_persister.py:76-77
symbol = parts[2].upper()
timeframe = parts[3]  # Extracted but not stored in DB!
```

The database schema doesn't have a `timeframe` column, so all EMA values get mixed together.

## Specific Sawtooth Scenarios

### Scenario 1: Database Retrieval Mixing
1. Gateway calculates EMAs for 1m, 5m, 15m timeframes
2. Database stores only the latest values without timeframe distinction
3. Analyzer retrieves "mixed" EMA values
4. Strategy gets EMA12 from 1m but EMA26 from 5m data
5. **Result**: Sawtooth pattern due to timeframe mismatch

### Scenario 2: Fallback to Traditional Calculation
1. Incremental cache is empty or corrupted
2. System falls back to traditional EMA calculation:
```python
# technical_indicators.py:204-209
if self.talib_available:
    try:
        ema_values = talib.EMA(prices_array, timeperiod=period)
        return [float(val) if not np.isnan(val) else None for val in ema_values]
```
3. TA-Lib recalculates entire EMA series from scratch
4. **Result**: Discontinuity between incremental and traditional calculations

### Scenario 3: Multi-Service Cache Inconsistency
1. Gateway has incremental EMAs for timeframe A
2. Signal Aggregator has different incremental EMAs for same timeframe
3. Database has yet another set of EMAs
4. **Result**: Different services using different EMA values = sawtooth

## Solutions Required

### 1. **Database Schema Fix - HIGH PRIORITY**

Add timeframe column to market_data table:

```sql
-- Add timeframe column
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS interval VARCHAR(10) NOT NULL DEFAULT '1m';

-- Update primary key to include timeframe
ALTER TABLE market_data DROP CONSTRAINT market_data_pkey;
ALTER TABLE market_data ADD PRIMARY KEY (time, symbol, interval);

-- Create separate EMA columns per timeframe or use JSONB
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS emas_by_timeframe JSONB;
```

### 2. **Unified Incremental Cache**

Create a shared incremental cache service that all components use:

```python
class SharedIncrementalCache:
    def get_ema(self, symbol: str, timeframe: str, period: int) -> Optional[float]:
        # Single source of truth for EMA values
    
    def update_ema(self, symbol: str, timeframe: str, period: int, value: float):
        # Update with timeframe awareness
```

### 3. **Timeframe-Aware Data Retrieval**

Update analyzer to specify timeframe:

```python
def get_enriched_market_data(self, symbol: str, timeframe: str = '1m', limit: int = 200):
    # Retrieve data for specific timeframe only
```

### 4. **Consistent EMA Calculation Path**

Ensure all services use the same EMA calculation method:
- **Always use incremental** when possible
- **Clear fallback strategy** when incremental cache is unavailable
- **Validation** that incremental and traditional methods produce consistent results

### 5. **Timeframe Validation in Strategies**

Add timeframe validation to ensure strategies get the expected timeframe data:

```python
def analyze(self, symbol: str, df: pd.DataFrame, indicators: Dict, expected_timeframe: str = '1m'):
    # Validate that indicators match expected timeframe
```

## Testing Recommendations

1. **EMA Continuity Test**: Verify EMAs don't have unexpected jumps between candles
2. **Timeframe Isolation Test**: Ensure EMAs for different timeframes are completely separate
3. **Cross-Service Consistency Test**: Verify all services return same EMA for same symbol/timeframe/time
4. **Cache Recovery Test**: Ensure smooth transition when incremental cache is lost/rebuilt

## Priority Actions

1. **IMMEDIATE**: Add timeframe column to database schema
2. **HIGH**: Implement shared incremental cache service
3. **HIGH**: Update database persister to store timeframe-specific EMAs
4. **MEDIUM**: Update analyzer to request specific timeframe data
5. **LOW**: Add comprehensive EMA consistency monitoring

The sawtooth patterns are primarily caused by **timeframe data mixing** and **inconsistent calculation paths** across the distributed system. The solution requires both database schema changes and unified cache management.