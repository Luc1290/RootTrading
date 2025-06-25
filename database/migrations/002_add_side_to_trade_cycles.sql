-- Migration 002: Ajouter la colonne 'side' à la table trade_cycles
-- Date: 2025-06-24
-- Auteur: Claude Code
-- Description: Ajoute le champ 'side' (BUY/SELL) directement dans les cycles 
--              pour éviter les jointures avec trade_executions

BEGIN;

-- 1. Ajouter la colonne side
ALTER TABLE trade_cycles 
ADD COLUMN side VARCHAR(4) CHECK (side IN ('BUY', 'SELL'));

-- 2. Créer un index pour les requêtes par side
CREATE INDEX idx_trade_cycles_side ON trade_cycles(side);

-- 3. Créer un index composite pour les requêtes fréquentes
CREATE INDEX idx_trade_cycles_symbol_side_status ON trade_cycles(symbol, side, status);

-- 4. Mettre à jour les cycles existants en déduisant le side depuis les exécutions
UPDATE trade_cycles 
SET side = (
    SELECT te.side 
    FROM trade_executions te 
    WHERE te.order_id = trade_cycles.entry_order_id 
    LIMIT 1
)
WHERE side IS NULL 
AND entry_order_id IS NOT NULL;

-- 5. Pour les cycles sans exécution, déduire depuis le statut
UPDATE trade_cycles 
SET side = CASE 
    WHEN status IN ('waiting_sell', 'active_sell') THEN 'BUY'  -- Position longue
    WHEN status IN ('waiting_buy', 'active_buy') THEN 'SELL'   -- Position courte
    ELSE 'BUY'  -- Défaut pour les cas ambigus
END
WHERE side IS NULL;

-- 6. Rendre la colonne NOT NULL maintenant qu'elle est populée
ALTER TABLE trade_cycles 
ALTER COLUMN side SET NOT NULL;

-- 7. Ajouter un commentaire pour documenter
COMMENT ON COLUMN trade_cycles.side IS 'Direction du cycle: BUY (position longue) ou SELL (position courte)';

COMMIT;

-- Vérification post-migration
SELECT 
    side,
    status,
    COUNT(*) as count
FROM trade_cycles 
GROUP BY side, status
ORDER BY side, status;