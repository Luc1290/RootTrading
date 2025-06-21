-- Script de migration pour changer BUY/SELL en LONG/sell
-- Date: 2025-06-13

-- 1. D'abord, supprimer les anciennes contraintes CHECK
ALTER TABLE trade_executions DROP CONSTRAINT IF EXISTS trade_executions_side_check;
ALTER TABLE trade_cycles DROP CONSTRAINT IF EXISTS trade_cycles_status_check;

-- 2. Mettre à jour les données existantes (si nécessaire)
-- Note: Si vous avez des données avec BUY/SELL, décommentez ces lignes :
-- UPDATE trade_executions SET side = 'LONG' WHERE side = 'BUY';
-- UPDATE trade_executions SET side = 'sell' WHERE side = 'SELL';

-- 3. Ajouter les nouvelles contraintes CHECK
ALTER TABLE trade_executions 
ADD CONSTRAINT trade_executions_side_check 
CHECK (side IN ('LONG', 'sell'));

-- 4. Vérifier aussi la table trade_cycles si elle a des contraintes à modifier
-- (Le schema.sql montre que trade_cycles utilise des statuts, pas des sides)

-- 5. Afficher un message de confirmation
DO $$ 
BEGIN
    RAISE NOTICE 'Migration terminée : Les contraintes ont été mises à jour pour utiliser LONG/sell';
END $$;