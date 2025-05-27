-- Migration pour ajouter le support des actifs dans les poches
-- Étape 1: Ajouter la colonne asset
ALTER TABLE capital_pockets ADD COLUMN IF NOT EXISTS asset VARCHAR(10) NOT NULL DEFAULT 'USDC';

-- Étape 2: Supprimer l'ancien index unique et en créer un nouveau avec asset
DROP INDEX IF EXISTS capital_pockets_type_unique_idx;
CREATE UNIQUE INDEX IF NOT EXISTS capital_pockets_type_asset_unique_idx ON capital_pockets(pocket_type, asset);

-- Étape 3: Mettre à jour les contraintes
ALTER TABLE capital_pockets DROP CONSTRAINT IF EXISTS capital_pockets_pocket_type_check;
ALTER TABLE capital_pockets ADD CONSTRAINT capital_pockets_pocket_type_check 
    CHECK (pocket_type IN ('active', 'buffer', 'safety'));

-- Étape 4: Créer les poches pour chaque actif si elles n'existent pas
INSERT INTO capital_pockets (pocket_type, asset, allocation_percent, current_value, used_value, available_value, active_cycles, reserved_amount)
VALUES 
    -- Poches USDC (déjà existantes, mais on s'assure qu'elles ont le bon asset)
    ('active', 'USDC', 80.0, 0.0, 0.0, 0.0, 0, 0.0),
    ('buffer', 'USDC', 10.0, 0.0, 0.0, 0.0, 0, 0.0),
    ('safety', 'USDC', 10.0, 0.0, 0.0, 0.0, 0, 0.0),
    
    -- Poches BTC
    ('active', 'BTC', 80.0, 0.0, 0.0, 0.0, 0, 0.0),
    ('buffer', 'BTC', 10.0, 0.0, 0.0, 0.0, 0, 0.0),
    ('safety', 'BTC', 10.0, 0.0, 0.0, 0.0, 0, 0.0),
    
    -- Poches ETH
    ('active', 'ETH', 80.0, 0.0, 0.0, 0.0, 0, 0.0),
    ('buffer', 'ETH', 10.0, 0.0, 0.0, 0.0, 0, 0.0),
    ('safety', 'ETH', 10.0, 0.0, 0.0, 0.0, 0, 0.0),
    
    -- Poches BNB
    ('active', 'BNB', 80.0, 0.0, 0.0, 0.0, 0, 0.0),
    ('buffer', 'BNB', 10.0, 0.0, 0.0, 0.0, 0, 0.0),
    ('safety', 'BNB', 10.0, 0.0, 0.0, 0.0, 0, 0.0)
ON CONFLICT (pocket_type, asset) DO NOTHING;

-- Étape 5: Mettre à jour la table des transactions pour supporter les actifs
ALTER TABLE pocket_transactions ADD COLUMN IF NOT EXISTS asset VARCHAR(10) NOT NULL DEFAULT 'USDC';

-- Étape 6: Créer des index pour les performances
CREATE INDEX IF NOT EXISTS capital_pockets_asset_idx ON capital_pockets(asset);
CREATE INDEX IF NOT EXISTS pocket_transactions_asset_idx ON pocket_transactions(asset);

-- Étape 7: Mettre à jour les contraintes pour les transactions
ALTER TABLE pocket_transactions DROP CONSTRAINT IF EXISTS pocket_transactions_pocket_type_check;
ALTER TABLE pocket_transactions ADD CONSTRAINT pocket_transactions_pocket_type_check 
    CHECK (pocket_type IN ('active', 'buffer', 'safety'));

ALTER TABLE pocket_transactions DROP CONSTRAINT IF EXISTS pocket_transactions_transaction_type_check;
ALTER TABLE pocket_transactions ADD CONSTRAINT pocket_transactions_transaction_type_check 
    CHECK (transaction_type IN ('reserve', 'release', 'allocate', 'reallocate'));