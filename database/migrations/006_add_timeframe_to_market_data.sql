-- Migration 006: Add timeframe column to market_data table
-- Fixes EMA sawtooth patterns by properly separating timeframe-specific data

BEGIN;

-- Step 1: Add timeframe column with default value
ALTER TABLE market_data 
ADD COLUMN timeframe VARCHAR(10) DEFAULT '1m' NOT NULL;

-- Step 2: Create index for timeframe-based queries (performance optimization)
CREATE INDEX CONCURRENTLY IF NOT EXISTS market_data_timeframe_symbol_time_idx 
ON market_data (timeframe, symbol, time DESC);

-- Step 3: Drop old primary key constraint
ALTER TABLE market_data DROP CONSTRAINT market_data_pkey;

-- Step 4: Create new composite primary key including timeframe
ALTER TABLE market_data 
ADD CONSTRAINT market_data_pkey PRIMARY KEY (time, symbol, timeframe);

-- Step 5: Update existing data to have '1m' timeframe (preserve current data)
UPDATE market_data SET timeframe = '1m' WHERE timeframe IS NULL;

-- Step 6: Create specialized indexes for common queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS market_data_enhanced_timeframe_idx 
ON market_data (enhanced, ultra_enriched, symbol, timeframe, time DESC)
WHERE enhanced = true;

CREATE INDEX CONCURRENTLY IF NOT EXISTS market_data_signal_aggregator_timeframe_idx 
ON market_data (symbol, timeframe, time DESC, enhanced, adx_14, rsi_14) 
WHERE enhanced = true;

-- Step 7: Add constraint to ensure valid timeframes
ALTER TABLE market_data 
ADD CONSTRAINT valid_timeframe_check 
CHECK (timeframe IN ('1m', '5m', '15m', '1h', '4h', '1d'));

-- Step 8: Drop old indexes that don't include timeframe (they're now less useful)
DROP INDEX IF EXISTS market_data_enhanced_idx;
DROP INDEX IF EXISTS market_data_signal_aggregator_idx;
DROP INDEX IF EXISTS market_data_timeframe_idx;

COMMIT;

-- Notes:
-- 1. This migration preserves all existing data by defaulting to '1m' timeframe
-- 2. New data will need to specify explicit timeframe values
-- 3. Queries will need to be updated to include timeframe filters
-- 4. EMA calculations will now be properly separated by timeframe
-- 5. Performance should improve due to timeframe-specific indexes