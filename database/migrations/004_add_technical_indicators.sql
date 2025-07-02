-- Migration 004: Ajout des indicateurs techniques à la table market_data
-- Date: 2025-07-01
-- Description: Ajoute les colonnes pour les indicateurs techniques calculés par le Gateway

-- Ajouter les colonnes d'indicateurs techniques
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS rsi_14 NUMERIC(8,4);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS ema_12 NUMERIC(16,8);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS ema_26 NUMERIC(16,8);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS ema_50 NUMERIC(16,8);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS sma_20 NUMERIC(16,8);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS sma_50 NUMERIC(16,8);

-- MACD
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS macd_line NUMERIC(16,8);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS macd_signal NUMERIC(16,8);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS macd_histogram NUMERIC(16,8);

-- Bollinger Bands
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS bb_upper NUMERIC(16,8);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS bb_middle NUMERIC(16,8);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS bb_lower NUMERIC(16,8);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS bb_position NUMERIC(8,4);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS bb_width NUMERIC(8,4);

-- ATR et autres indicateurs
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS atr_14 NUMERIC(16,8);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS momentum_10 NUMERIC(8,4);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS volume_ratio NUMERIC(8,4);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS avg_volume_20 NUMERIC(16,8);

-- Indicateurs additionnels (optionnels)
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS stoch_rsi NUMERIC(8,4);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS adx_14 NUMERIC(8,4);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS williams_r NUMERIC(8,4);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS cci_20 NUMERIC(8,4);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS vwap_10 NUMERIC(16,8);

-- Métadonnées d'enrichissement
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS enhanced BOOLEAN DEFAULT FALSE;
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS ultra_enriched BOOLEAN DEFAULT FALSE;

-- Créer un index pour les requêtes sur les données enrichies
CREATE INDEX IF NOT EXISTS market_data_enhanced_idx ON market_data (enhanced, ultra_enriched, symbol, time DESC);

-- Commentaires pour documentation
COMMENT ON COLUMN market_data.rsi_14 IS 'RSI 14 périodes';
COMMENT ON COLUMN market_data.ema_12 IS 'EMA 12 périodes';
COMMENT ON COLUMN market_data.ema_26 IS 'EMA 26 périodes';
COMMENT ON COLUMN market_data.macd_line IS 'MACD Line (EMA12 - EMA26)';
COMMENT ON COLUMN market_data.macd_signal IS 'MACD Signal Line (EMA9 du MACD)';
COMMENT ON COLUMN market_data.bb_upper IS 'Bollinger Band supérieure';
COMMENT ON COLUMN market_data.bb_lower IS 'Bollinger Band inférieure';
COMMENT ON COLUMN market_data.enhanced IS 'Données enrichies avec indicateurs techniques';
COMMENT ON COLUMN market_data.ultra_enriched IS 'Données ultra-enrichies avec tous les indicateurs';

-- Log de la migration
DO $$
BEGIN
    RAISE NOTICE 'Migration 004: Colonnes indicateurs techniques ajoutées à market_data';
END $$;