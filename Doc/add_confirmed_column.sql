
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'trade_cycles' AND column_name = 'confirmed'
    ) THEN
        ALTER TABLE trade_cycles ADD COLUMN confirmed BOOLEAN DEFAULT FALSE;
        RAISE NOTICE '✅ Colonne confirmed ajoutée.';
    ELSE
        RAISE NOTICE 'ℹ️ Colonne confirmed déjà présente.';
    END IF;
END$$;
