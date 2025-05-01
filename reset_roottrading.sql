
-- 🔥 PURGE DES CYCLES NON CONFIRMÉS
DELETE FROM trade_cycles
WHERE confirmed = false OR confirmed IS NULL;

-- 🔥 PURGE DES EXÉCUTIONS ORPHELINES
DELETE FROM trade_executions
WHERE cycle_id IS NULL
   OR cycle_id NOT IN (SELECT id FROM trade_cycles);

-- 🔄 RESET DES POCHES
UPDATE capital_pockets
SET used_value = 0,
    available_value = current_value,
    active_cycles = 0,
    updated_at = NOW();

-- 🧹 PURGE DES TRANSACTIONS DE POCHES FICTIVES (optionnel)
DELETE FROM pocket_transactions
WHERE created_at < '2025-05-01T15:51:03.700807';

-- 🟢 RÉINITIALISATION DU RISK MANAGER
-- Cela dépend du fonctionnement. En général on redémarre le process,
-- mais si une table de config (comme `risk_flags`) existe :
-- UPDATE risk_flags SET status = 'enabled', blocked_until = NULL;

-- 🔁 COORDINATOR : via API ou simplement relancer le service

-- ✅ RÉINITIALISATION DU PORTFOLIO : forcée via endpoint /pockets/reconcile
-- ou simplement redémarrer le container
