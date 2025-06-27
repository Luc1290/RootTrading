-- Migration 003: Ajout des colonnes id et metadata à trade_executions
-- Date: 27 Juin 2025
-- Objectif: Supporter le système de renforcement (DCA) avec traçabilité

-- Ajouter la colonne id optionnelle pour compatibilité avec certains codes
ALTER TABLE trade_executions ADD COLUMN IF NOT EXISTS id VARCHAR(50);

-- Ajouter la colonne metadata pour traçabilité des renforcements
ALTER TABLE trade_executions ADD COLUMN IF NOT EXISTS metadata JSONB;

-- Créer les index pour les nouvelles colonnes
CREATE INDEX IF NOT EXISTS trade_executions_metadata_idx ON trade_executions USING GIN (metadata);
CREATE INDEX IF NOT EXISTS trade_executions_id_idx ON trade_executions(id) WHERE id IS NOT NULL;

-- Commentaires pour documentation
COMMENT ON COLUMN trade_executions.id IS 'ID optionnel pour compatibilité avec certains codes existants';
COMMENT ON COLUMN trade_executions.metadata IS 'Métadonnées JSON pour traçabilité (renforcements DCA, etc.)';