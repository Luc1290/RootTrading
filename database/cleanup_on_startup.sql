-- Script SQL exécuté au démarrage pour nettoyer les données incohérentes
-- Ce script garantit un état propre au démarrage du système

-- 1. Normaliser tous les statuts existants en minuscules
UPDATE trade_cycles 
SET status = LOWER(status), updated_at = NOW()
WHERE status != LOWER(status);

-- 2. Marquer comme FAILED les cycles trop anciens (> 24h) qui sont encore actifs
UPDATE trade_cycles 
SET status = 'failed', 
    updated_at = NOW(),
    metadata = jsonb_set(
        COALESCE(metadata, '{}'::jsonb),
        '{fail_reason}',
        '"Cycle marqué failed au démarrage (> 24h)"'::jsonb
    )
WHERE status IN ('waiting_buy', 'active_buy', 'waiting_sell', 'active_sell')
AND created_at < NOW() - INTERVAL '24 hours';

-- 3. Annuler les cycles non confirmés depuis plus d'1 heure
UPDATE trade_cycles 
SET status = 'canceled', 
    updated_at = NOW(),
    metadata = jsonb_set(
        COALESCE(metadata, '{}'::jsonb),
        '{cancel_reason}',
        '"Cycle non confirmé annulé au démarrage"'::jsonb
    )
WHERE confirmed = false 
AND created_at < NOW() - INTERVAL '1 hour'
AND status NOT IN ('completed', 'canceled', 'failed');

-- 4. Corriger les cycles 'completed' sans prix de sortie
UPDATE trade_cycles 
SET status = 'failed', 
    updated_at = NOW(),
    metadata = jsonb_set(
        COALESCE(metadata, '{}'::jsonb),
        '{fail_reason}',
        '"Cycle marqué completed sans prix de sortie"'::jsonb
    )
WHERE status = 'completed' AND exit_price IS NULL;

-- 5. Afficher un résumé des nettoyages
DO $$
DECLARE
    v_failed_count INTEGER;
    v_canceled_count INTEGER;
    v_normalized_count INTEGER;
BEGIN
    -- Compter les modifications (approximatif)
    SELECT COUNT(*) INTO v_failed_count 
    FROM trade_cycles 
    WHERE status = 'failed' 
    AND metadata->>'fail_reason' LIKE '%démarrage%';
    
    SELECT COUNT(*) INTO v_canceled_count 
    FROM trade_cycles 
    WHERE status = 'canceled' 
    AND metadata->>'cancel_reason' LIKE '%démarrage%';
    
    RAISE NOTICE '✅ Nettoyage au démarrage terminé:';
    RAISE NOTICE '   - Cycles failed: %', v_failed_count;
    RAISE NOTICE '   - Cycles canceled: %', v_canceled_count;
    RAISE NOTICE '   - Statuts normalisés';
END $$;