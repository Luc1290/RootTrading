-- Migration 005: Ajout des indicateurs manquants pour Signal Aggregator
-- Date: 2025-07-01
-- Description: Ajoute ADX, Stochastic, ROC et autres indicateurs utilisés par Signal Aggregator

-- ADX et Directional Indicators
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS adx_14 NUMERIC(8,4);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS plus_di NUMERIC(8,4);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS minus_di NUMERIC(8,4);

-- Stochastic Oscillator
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS stoch_k NUMERIC(8,4);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS stoch_d NUMERIC(8,4);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS stoch_rsi NUMERIC(8,4);

-- Rate of Change et autres indicateurs momentum
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS roc_10 NUMERIC(8,4);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS roc_20 NUMERIC(8,4);

-- On Balance Volume
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS obv NUMERIC(20,8);

-- Williams %R
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS williams_r NUMERIC(8,4);

-- Commodity Channel Index
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS cci_20 NUMERIC(8,4);

-- Money Flow Index
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS mfi_14 NUMERIC(8,4);

-- VWAP (Volume Weighted Average Price)
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS vwap_10 NUMERIC(16,8);

-- Indicateurs de régime pour Signal Aggregator
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS trend_angle NUMERIC(8,4);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS pivot_count INTEGER;

-- Index pour optimiser les requêtes Signal Aggregator
CREATE INDEX IF NOT EXISTS market_data_signal_aggregator_idx ON market_data (
    symbol, 
    time DESC,
    enhanced,
    adx_14,
    rsi_14
) WHERE enhanced = true;

-- Index pour recherche par timeframe
CREATE INDEX IF NOT EXISTS market_data_timeframe_idx ON market_data (
    symbol,
    time DESC
) WHERE enhanced = true;

-- Commentaires pour documentation
COMMENT ON COLUMN market_data.adx_14 IS 'Average Directional Index 14 périodes - Force de tendance';
COMMENT ON COLUMN market_data.plus_di IS 'Directional Indicator + (DI+) - Direction haussière';
COMMENT ON COLUMN market_data.minus_di IS 'Directional Indicator - (DI-) - Direction baissière';
COMMENT ON COLUMN market_data.stoch_k IS 'Stochastic %K - Momentum oscillator';
COMMENT ON COLUMN market_data.stoch_d IS 'Stochastic %D - Signal line de %K';
COMMENT ON COLUMN market_data.roc_10 IS 'Rate of Change 10 périodes - Momentum directionnel';
COMMENT ON COLUMN market_data.obv IS 'On Balance Volume - Volume directionnel cumulé';
COMMENT ON COLUMN market_data.williams_r IS 'Williams %R - Oscillateur momentum';
COMMENT ON COLUMN market_data.cci_20 IS 'Commodity Channel Index 20 périodes';
COMMENT ON COLUMN market_data.mfi_14 IS 'Money Flow Index 14 périodes';
COMMENT ON COLUMN market_data.vwap_10 IS 'Volume Weighted Average Price 10 périodes';
COMMENT ON COLUMN market_data.trend_angle IS 'Angle de tendance (degrés) - Régression linéaire';
COMMENT ON COLUMN market_data.pivot_count IS 'Nombre de pivots détectés - Support/Résistance';

-- Vue pour faciliter les requêtes Signal Aggregator
CREATE OR REPLACE VIEW signal_aggregator_data AS
SELECT 
    time,
    symbol,
    open,
    high,
    low,
    close,
    volume,
    -- Indicateurs essentiels pour régime
    rsi_14,
    adx_14,
    plus_di,
    minus_di,
    bb_upper,
    bb_middle,
    bb_lower,
    bb_position,
    bb_width,
    roc_10,
    atr_14,
    volume_ratio,
    trend_angle,
    pivot_count,
    enhanced,
    ultra_enriched
FROM market_data
WHERE enhanced = true
ORDER BY symbol, time DESC;

-- Grant permissions pour la vue
GRANT SELECT ON signal_aggregator_data TO postgres;

-- Log de la migration
DO $$
BEGIN
    RAISE NOTICE 'Migration 005: Indicateurs Signal Aggregator ajoutés à market_data';
    RAISE NOTICE 'Nouvelles colonnes: ADX, DI+/-, Stochastic, ROC, OBV, Williams %R, CCI, MFI, VWAP';
    RAISE NOTICE 'Index créés pour optimiser les requêtes Signal Aggregator';
    RAISE NOTICE 'Vue signal_aggregator_data créée';
END $$;