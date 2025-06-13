-- Script de migration complet pour changer BUY/SELL en LONG/SHORT
-- Date: 2025-06-13

-- 1. D'abord, supprimer les anciennes contraintes CHECK
ALTER TABLE trade_executions DROP CONSTRAINT IF EXISTS trade_executions_side_check;

-- 2. Mettre à jour les données existantes
UPDATE trade_executions SET side = 'LONG' WHERE side = 'BUY';
UPDATE trade_executions SET side = 'SHORT' WHERE side = 'SELL';

-- 3. Ajouter les nouvelles contraintes CHECK
ALTER TABLE trade_executions 
ADD CONSTRAINT trade_executions_side_check 
CHECK (side IN ('LONG', 'SHORT'));

-- 4. Vérifier le résultat
SELECT 'Valeurs uniques dans trade_executions.side:' as info;
SELECT DISTINCT side FROM trade_executions;

-- 5. Afficher un message de confirmation
DO $$ 
BEGIN
    RAISE NOTICE 'Migration terminée : Les données ont été migrées de BUY/SELL vers LONG/SHORT';
END $$;