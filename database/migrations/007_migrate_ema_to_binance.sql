-- Migration 007: Migration des EMAs vers configuration Binance (7/26/99)
-- Date: 2025-01-18
-- Description: Migre de EMA 12/26/50 vers EMA 7/26/99 pour meilleure réactivité

-- 1. Ajouter les nouvelles colonnes EMA
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS ema_7 NUMERIC(16,8);
ALTER TABLE market_data ADD COLUMN IF NOT EXISTS ema_99 NUMERIC(16,8);

-- 2. Créer des index pour les nouvelles colonnes
CREATE INDEX IF NOT EXISTS idx_market_data_ema_7 ON market_data (symbol, time DESC) WHERE ema_7 IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_market_data_ema_99 ON market_data (symbol, time DESC) WHERE ema_99 IS NOT NULL;

-- 3. Commentaires pour documentation
COMMENT ON COLUMN market_data.ema_7 IS 'EMA 7 périodes - Ultra rapide (config Binance)';
COMMENT ON COLUMN market_data.ema_99 IS 'EMA 99 périodes - Long terme (config Binance)';

-- 4. Migration des données (optionnel - garder les anciennes pour historique)
-- Les colonnes ema_12 et ema_50 sont conservées pour compatibilité
UPDATE market_data 
SET enhanced = FALSE 
WHERE enhanced = TRUE 
  AND ema_7 IS NULL;

-- 5. Log de la migration
DO $$
BEGIN
    RAISE NOTICE 'Migration 007: Migration EMA vers configuration Binance (7/26/99) effectuée';
    RAISE NOTICE 'Anciennes colonnes (ema_12, ema_50) conservées pour historique';
    RAISE NOTICE 'Nouvelles colonnes: ema_7 (ultra rapide), ema_99 (long terme)';
END $$;