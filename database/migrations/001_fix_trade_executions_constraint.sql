-- Migration pour corriger la contrainte trade_executions_side_check
-- Cette migration corrige la contrainte qui n'autorisait que 'LONG' et 'sell'
-- pour accepter 'BUY' et 'SELL' comme utilisé par le trader

-- Supprimer l'ancienne contrainte si elle existe
ALTER TABLE trade_executions DROP CONSTRAINT IF EXISTS trade_executions_side_check;

-- Recréer la contrainte avec les bonnes valeurs
ALTER TABLE trade_executions 
ADD CONSTRAINT trade_executions_side_check 
CHECK (side IN ('BUY', 'SELL', 'LONG', 'sell'));

-- Ajouter un commentaire pour documenter le changement
COMMENT ON CONSTRAINT trade_executions_side_check ON trade_executions 
IS 'Contrainte mise à jour pour accepter BUY/SELL (nouveau) et LONG/sell (legacy)';