-- Migration: Ajout des colonnes min_price et max_price pour le trailing stop
-- Date: 2025-06-05

-- Ajouter les colonnes si elles n'existent pas
ALTER TABLE trade_cycles 
ADD COLUMN IF NOT EXISTS min_price NUMERIC(16, 8), 
ADD COLUMN IF NOT EXISTS max_price NUMERIC(16, 8);

-- Initialiser les valeurs NULL avec entry_price pour les cycles actifs
UPDATE trade_cycles 
SET min_price = entry_price, 
    max_price = entry_price 
WHERE (min_price IS NULL OR max_price IS NULL) 
  AND status NOT IN ('completed', 'canceled', 'failed');

-- Créer des index pour améliorer les performances des requêtes sur ces colonnes
CREATE INDEX IF NOT EXISTS trade_cycles_min_price_idx ON trade_cycles(min_price) 
WHERE status NOT IN ('completed', 'canceled', 'failed');

CREATE INDEX IF NOT EXISTS trade_cycles_max_price_idx ON trade_cycles(max_price) 
WHERE status NOT IN ('completed', 'canceled', 'failed');