-- Migration 007: Fix numeric field overflow for high volume cryptocurrencies
-- Addresses issue with memecoins (PEPE, SHIB) having massive volumes that exceed NUMERIC(16,8) limits

BEGIN;

-- Step 1: Increase volume precision from NUMERIC(16,8) to NUMERIC(20,8)
-- This allows volumes up to 999,999,999,999.99999999 (999 billion with 8 decimals)
ALTER TABLE market_data ALTER COLUMN volume TYPE NUMERIC(20,8);

-- Step 2: Increase OBV precision from NUMERIC(20,8) to NUMERIC(24,8) 
-- OBV accumulates volumes over time and needs even more precision
-- This allows values up to 9,999,999,999,999,999.99999999 (9 quadrillion)
ALTER TABLE market_data ALTER COLUMN obv TYPE NUMERIC(24,8);

-- Step 3: Increase avg_volume_20 precision from NUMERIC(16,8) to NUMERIC(20,8)
-- 20-period average of large volumes also needs increased precision
ALTER TABLE market_data ALTER COLUMN avg_volume_20 TYPE NUMERIC(20,8);

-- Step 4: Update other volume-related indicators that might overflow
-- These are derived from volume and could also hit limits with memecoins
ALTER TABLE market_data ALTER COLUMN volume_sma_20 TYPE NUMERIC(20,8);

-- Step 5: Fix trading tables for memecoin compatibility
-- trade_cycles table - quantities can be massive for low-price tokens
ALTER TABLE trade_cycles ALTER COLUMN quantity TYPE NUMERIC(20,8);
ALTER TABLE trade_cycles ALTER COLUMN entry_price TYPE NUMERIC(20,12);  -- More precision for micro prices
ALTER TABLE trade_cycles ALTER COLUMN exit_price TYPE NUMERIC(20,12);
ALTER TABLE trade_cycles ALTER COLUMN stop_price TYPE NUMERIC(20,12);
ALTER TABLE trade_cycles ALTER COLUMN min_price TYPE NUMERIC(20,12);
ALTER TABLE trade_cycles ALTER COLUMN max_price TYPE NUMERIC(20,12);

-- trade_executions table - same issues with quantities and micro prices
ALTER TABLE trade_executions ALTER COLUMN quantity TYPE NUMERIC(20,8);
ALTER TABLE trade_executions ALTER COLUMN quote_quantity TYPE NUMERIC(20,8);
ALTER TABLE trade_executions ALTER COLUMN price TYPE NUMERIC(20,12);

-- Step 6: Add validation constraints to prevent extreme values while allowing legitimate high volumes
-- This protects against data corruption while accommodating real market conditions
ALTER TABLE market_data ADD CONSTRAINT volume_sanity_check 
    CHECK (volume >= 0 AND volume < 1e15);  -- Max 1 quadrillion

ALTER TABLE market_data ADD CONSTRAINT obv_sanity_check 
    CHECK (obv > -1e20 AND obv < 1e20);  -- OBV can be negative, allow ±100 quintillion

-- Trading constraints for memecoins
ALTER TABLE trade_cycles ADD CONSTRAINT quantity_sanity_check 
    CHECK (quantity >= 0 AND quantity < 1e15);  -- Max 1 quadrillion tokens

ALTER TABLE trade_executions ADD CONSTRAINT quantity_sanity_check 
    CHECK (quantity >= 0 AND quantity < 1e15);

COMMIT;

-- Notes:
-- 1. PEPE/SHIB volumes can exceed 400 billion daily, requiring NUMERIC(20,8)
-- 2. OBV accumulates these volumes indefinitely, requiring NUMERIC(24,8)  
-- 3. Sanity checks prevent corruption while allowing legitimate market data
-- 4. Performance impact minimal as these are storage-only changes
-- 5. Resolves "numeric field overflow" errors in dispatcher insertion