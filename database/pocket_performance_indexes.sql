-- Index de performance pour les requêtes de pocket_transactions
-- Améliore les performances des requêtes SUM() sur les réservations/libérations

-- Index composite principal pour les requêtes de réservation/libération par cycle
CREATE INDEX IF NOT EXISTS idx_pocket_transactions_cycle_performance 
ON pocket_transactions (cycle_id, pocket_type, asset, transaction_type);

-- Index pour améliorer les requêtes par type de transaction
CREATE INDEX IF NOT EXISTS idx_pocket_transactions_type_date 
ON pocket_transactions (transaction_type, created_at);

-- Index pour améliorer les requêtes de réservation (ex: reservation_id)
CREATE INDEX IF NOT EXISTS idx_pocket_transactions_reservation_optimized 
ON pocket_transactions (reservation_id) 
WHERE reservation_id IS NOT NULL;

-- Index pour optimiser les requêtes de reconciliation
CREATE INDEX IF NOT EXISTS idx_pocket_transactions_reconciliation 
ON pocket_transactions (pocket_type, asset, created_at);

-- Statistiques pour l'optimiseur de requêtes
ANALYZE pocket_transactions;

-- Vérification des index créés
SELECT 
    indexname,
    indexdef
FROM pg_indexes 
WHERE tablename = 'pocket_transactions' 
AND schemaname = 'public'
ORDER BY indexname;