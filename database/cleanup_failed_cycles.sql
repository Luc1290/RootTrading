-- Script de nettoyage des cycles failed
-- Ce script réinitialise les metadata des cycles failed pour pouvoir repartir sur une base saine

-- 1. Afficher un résumé des cycles failed
SELECT 'Résumé des cycles failed:' AS info;
SELECT status, COUNT(*) as count 
FROM trade_cycles 
GROUP BY status;

-- 2. Afficher les cycles failed récents
SELECT 'Cycles failed récents:' AS info;
SELECT id, symbol, created_at, entry_order_id, exit_order_id, profit_loss
FROM trade_cycles 
WHERE status = 'failed' 
ORDER BY created_at DESC 
LIMIT 10;

-- 3. Réinitialiser les metadata des cycles failed
-- Ceci permet de nettoyer les cycles fantômes et de recommencer proprement
UPDATE trade_cycles 
SET metadata = jsonb_build_object(
    'cleaned_at', NOW()::text,
    'cleanup_reason', 'Manual cleanup of phantom cycles'
)
WHERE status = 'failed' 
  AND metadata IS NULL OR metadata = '{}';

-- 4. Optionnel: Si vous voulez supprimer complètement les cycles failed sans profit/loss
-- ATTENTION: Décommentez seulement si vous êtes sûr de vouloir supprimer ces données
-- DELETE FROM trade_cycles 
-- WHERE status = 'failed' 
--   AND (profit_loss IS NULL OR profit_loss = 0)
--   AND created_at < NOW() - INTERVAL '1 day';

-- 5. Afficher le résultat final
SELECT 'Résultat après nettoyage:' AS info;
SELECT status, COUNT(*) as count 
FROM trade_cycles 
GROUP BY status;

-- 6. Vérifier s'il reste des cycles actifs qui pourraient être fantômes
SELECT 'Cycles potentiellement actifs:' AS info;
SELECT id, symbol, status, entry_order_id, exit_order_id, created_at, updated_at
FROM trade_cycles 
WHERE status NOT IN ('completed', 'canceled', 'failed')
ORDER BY created_at DESC;