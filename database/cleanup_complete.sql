-- Script de nettoyage complet pour repartir sur une base propre
-- ATTENTION: Ce script supprime TOUTES les données de trading !

-- 1. Afficher un résumé avant nettoyage
SELECT 'État actuel de la base de données:' AS info;
SELECT 
    'Cycles' as table_name,
    COUNT(*) as total,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed,
    COUNT(CASE WHEN status = 'canceled' THEN 1 END) as canceled,
    COUNT(CASE WHEN status NOT IN ('completed', 'failed', 'canceled') THEN 1 END) as active
FROM trade_cycles;

SELECT 
    'Exécutions' as table_name,
    COUNT(*) as total
FROM trade_executions;

-- 2. Supprimer toutes les exécutions
SELECT 'Suppression des exécutions...' AS info;
TRUNCATE TABLE trade_executions CASCADE;

-- 3. Supprimer tous les cycles
SELECT 'Suppression des cycles...' AS info;
TRUNCATE TABLE trade_cycles CASCADE;

-- 4. Réinitialiser les réservations de fonds dans portfolio_fund_reservations
SELECT 'Nettoyage des réservations de fonds...' AS info;
DELETE FROM portfolio_fund_reservations WHERE reservation_id LIKE 'cycle_%';
DELETE FROM portfolio_fund_reservations WHERE reservation_id LIKE 'temp_%';

-- 5. Vérifier que tout est propre
SELECT 'Vérification après nettoyage:' AS info;
SELECT 'Cycles restants: ' || COUNT(*) FROM trade_cycles;
SELECT 'Exécutions restantes: ' || COUNT(*) FROM trade_executions;
SELECT 'Réservations cycle restantes: ' || COUNT(*) FROM portfolio_fund_reservations WHERE reservation_id LIKE 'cycle_%';

-- 6. Nettoyer aussi les clés Redis (à faire manuellement)
SELECT 'IMPORTANT: Exécuter aussi ces commandes Redis:' AS info;
SELECT '- redis-cli KEYS "roottrading:cycle:*" | xargs redis-cli DEL' AS commande;

-- 7. Message de confirmation
SELECT 'Base de données nettoyée avec succès !' AS info;
SELECT 'Les services peuvent maintenant être redémarrés.' AS info;