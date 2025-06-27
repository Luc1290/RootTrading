-- Schéma de base de données complet pour RootTrading
-- Version 1.0.9.0.3 - Inclut support renforcement DCA et prévention hedging
-- Date: 27 Juin 2025

-- Activer les extensions nécessaires
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb" CASCADE;  -- Pour les données temporelles

-- Configuration PostgreSQL optimisée pour trading
SET timezone = 'UTC';
SET default_text_search_config = 'pg_catalog.english';

-- Table des exécutions d'ordres
CREATE TABLE IF NOT EXISTS trade_executions (
    order_id VARCHAR(50) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    status VARCHAR(20) NOT NULL CHECK (status IN ('NEW', 'PARTIALLY_FILLED', 'FILLED', 'CANCELED', 'REJECTED', 'EXPIRED', 'PENDING_CANCEL')),
    price NUMERIC(16, 8) NOT NULL,
    quantity NUMERIC(16, 8) NOT NULL,
    quote_quantity NUMERIC(16, 8) NOT NULL,
    fee NUMERIC(16, 8),
    fee_asset VARCHAR(10),
    role VARCHAR(10) CHECK (role IN ('maker', 'taker')),
    timestamp TIMESTAMP NOT NULL,
    cycle_id VARCHAR(50),
    demo BOOLEAN NOT NULL DEFAULT FALSE,
    id VARCHAR(50),  -- Colonne id optionnelle pour compatibilité
    metadata JSONB,  -- Métadonnées pour traçabilité (renforcements, etc.)
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Index pour les exécutions
CREATE INDEX IF NOT EXISTS trade_executions_cycle_id_idx ON trade_executions(cycle_id);
CREATE INDEX IF NOT EXISTS trade_executions_timestamp_idx ON trade_executions(timestamp);
CREATE INDEX IF NOT EXISTS trade_executions_symbol_idx ON trade_executions(symbol);
CREATE INDEX IF NOT EXISTS trade_executions_status_idx ON trade_executions(status);
-- Index GIN pour les recherches dans les métadonnées JSON des exécutions
CREATE INDEX IF NOT EXISTS trade_executions_metadata_idx ON trade_executions USING GIN (metadata);
-- Index optionnel pour la colonne id si utilisée
CREATE INDEX IF NOT EXISTS trade_executions_id_idx ON trade_executions(id) WHERE id IS NOT NULL;

-- Table des cycles de trading (avec tous les champs requis)
CREATE TABLE IF NOT EXISTS trade_cycles (
    id VARCHAR(50) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    -- Statuts en minuscules pour cohérence avec les énumérations Python
    status VARCHAR(20) NOT NULL CHECK (status IN ('initiating', 'waiting_buy', 'active_buy', 'waiting_sell', 'active_sell', 'completed', 'canceled', 'failed')),
    -- Direction du cycle: BUY (position longue) ou SELL (position courte)
    side VARCHAR(4) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    entry_order_id VARCHAR(50),
    exit_order_id VARCHAR(50),
    entry_price NUMERIC(16, 8),
    exit_price NUMERIC(16, 8),
    quantity NUMERIC(16, 8),
    stop_price NUMERIC(16, 8),
    trailing_delta NUMERIC(16, 8),
    min_price NUMERIC(16, 8),
    max_price NUMERIC(16, 8),
    profit_loss NUMERIC(16, 8),
    profit_loss_percent NUMERIC(16, 8),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,
    confirmed BOOLEAN NOT NULL DEFAULT FALSE,
    -- Champ metadata pour stocker informations supplémentaires (réconciliation, raisons d'annulation, etc.)
    metadata JSONB,
    demo BOOLEAN NOT NULL DEFAULT FALSE    
);

-- Index optimisés pour les cycles
CREATE INDEX IF NOT EXISTS trade_cycles_symbol_idx ON trade_cycles(symbol);
CREATE INDEX IF NOT EXISTS trade_cycles_strategy_idx ON trade_cycles(strategy);
CREATE INDEX IF NOT EXISTS trade_cycles_side_idx ON trade_cycles(side);
CREATE INDEX IF NOT EXISTS trade_cycles_symbol_side_status_idx ON trade_cycles(symbol, side, status);
CREATE INDEX IF NOT EXISTS trade_cycles_status_idx ON trade_cycles(status);
CREATE INDEX IF NOT EXISTS trade_cycles_created_at_idx ON trade_cycles(created_at);
CREATE INDEX IF NOT EXISTS trade_cycles_completed_at_idx ON trade_cycles(completed_at);
-- Index composés pour requêtes fréquentes
CREATE INDEX IF NOT EXISTS trade_cycles_symbol_created_idx ON trade_cycles(symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS trade_cycles_strategy_created_idx ON trade_cycles(strategy, created_at DESC);
CREATE INDEX IF NOT EXISTS trade_cycles_status_updated_idx ON trade_cycles(status, updated_at DESC);
-- Index GIN pour les recherches dans les métadonnées JSON
CREATE INDEX IF NOT EXISTS trade_cycles_metadata_idx ON trade_cycles USING GIN (metadata);
-- Index pour les calculs de performance
CREATE INDEX IF NOT EXISTS trade_cycles_profit_status_idx ON trade_cycles(profit_loss_percent, status) 
WHERE status = 'completed';
-- Index pour les cycles actifs (très fréquent)
CREATE INDEX IF NOT EXISTS trade_cycles_active_idx ON trade_cycles(status, symbol, updated_at) 
WHERE status NOT IN ('completed', 'canceled', 'failed');

-- Table des signaux de trading
CREATE TABLE IF NOT EXISTS trading_signals (
    id SERIAL PRIMARY KEY,
    strategy VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    timestamp TIMESTAMP NOT NULL,
    price NUMERIC(16, 8) NOT NULL,
    confidence NUMERIC(5, 4),
    strength VARCHAR(20) CHECK (strength IN ('weak', 'moderate', 'strong', 'very_strong')),
    metadata JSONB,
    cycle_id VARCHAR(50),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Index pour les signaux
CREATE INDEX IF NOT EXISTS trading_signals_symbol_idx ON trading_signals(symbol);
CREATE INDEX IF NOT EXISTS trading_signals_strategy_idx ON trading_signals(strategy);
CREATE INDEX IF NOT EXISTS trading_signals_timestamp_idx ON trading_signals(timestamp);
CREATE INDEX IF NOT EXISTS trading_signals_strength_idx ON trading_signals(strength);
-- Index GIN pour les recherches dans les métadonnées JSON
CREATE INDEX IF NOT EXISTS trading_signals_metadata_idx ON trading_signals USING GIN (metadata);

-- Table des données de marché (séries temporelles optimisées)
CREATE TABLE IF NOT EXISTS market_data (
    time TIMESTAMP NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    open NUMERIC(16, 8) NOT NULL,
    high NUMERIC(16, 8) NOT NULL,
    low NUMERIC(16, 8) NOT NULL,
    close NUMERIC(16, 8) NOT NULL,
    volume NUMERIC(16, 8) NOT NULL,
    PRIMARY KEY (time, symbol)
);

-- Convertir market_data en table hypertable (TimescaleDB)
SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);

-- Optimisation de TimescaleDB pour market_data
SELECT set_chunk_time_interval('market_data', INTERVAL '1 day');
ALTER TABLE market_data SET (
    timescaledb.compress = true,
    timescaledb.compress_segmentby = 'symbol'
);
SELECT add_compression_policy('market_data', INTERVAL '7 days');

-- Index pour les données de marché
CREATE INDEX IF NOT EXISTS market_data_symbol_idx ON market_data(symbol, time DESC);
-- Index BRIN plus efficace pour les grandes tables temporelles
CREATE INDEX IF NOT EXISTS market_data_time_brin_idx ON market_data USING BRIN (time) WITH (pages_per_range = 128);

-- Table des soldes de portefeuille
CREATE TABLE IF NOT EXISTS portfolio_balances (
    id SERIAL PRIMARY KEY,
    asset VARCHAR(10) NOT NULL,
    free NUMERIC(24, 8) NOT NULL,
    locked NUMERIC(24, 8) NOT NULL,
    total NUMERIC(24, 8) NOT NULL,
    value_usdc NUMERIC(24, 8),
    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Index pour les soldes
CREATE INDEX IF NOT EXISTS portfolio_balances_asset_idx ON portfolio_balances(asset);
CREATE INDEX IF NOT EXISTS portfolio_balances_timestamp_idx ON portfolio_balances(timestamp);
CREATE INDEX IF NOT EXISTS portfolio_balances_asset_timestamp_idx ON portfolio_balances(asset, timestamp DESC);

-- Table des paramètres des stratégies
CREATE TABLE IF NOT EXISTS strategy_configs (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    mode VARCHAR(20) NOT NULL CHECK (mode IN ('active', 'monitoring', 'paused', 'disabled')),
    symbols JSONB NOT NULL,  -- Array de symboles
    params JSONB NOT NULL,   -- Paramètres de la stratégie
    max_simultaneous_trades INTEGER NOT NULL DEFAULT 3,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Index pour les stratégies
CREATE INDEX IF NOT EXISTS strategy_configs_name_idx ON strategy_configs(name);
CREATE INDEX IF NOT EXISTS strategy_configs_mode_idx ON strategy_configs(mode);
CREATE INDEX IF NOT EXISTS strategy_configs_enabled_idx ON strategy_configs(enabled);
-- Index GIN pour les recherches dans les tableaux de symboles
CREATE INDEX IF NOT EXISTS strategy_configs_symbols_idx ON strategy_configs USING GIN (symbols);
CREATE INDEX IF NOT EXISTS strategy_configs_params_idx ON strategy_configs USING GIN (params);

-- Table des journaux d'événements (optimisée)
CREATE TABLE IF NOT EXISTS event_logs (
    id SERIAL PRIMARY KEY,
    service VARCHAR(50) NOT NULL,
    level VARCHAR(20) NOT NULL CHECK (level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    message TEXT NOT NULL,
    data JSONB,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Index pour les journaux (optimisés pour les requêtes fréquentes)
CREATE INDEX IF NOT EXISTS event_logs_service_level_idx ON event_logs(service, level);
CREATE INDEX IF NOT EXISTS event_logs_timestamp_idx ON event_logs(timestamp);
CREATE INDEX IF NOT EXISTS event_logs_level_timestamp_idx ON event_logs(level, timestamp DESC);
-- Index partiel pour les erreurs critiques
CREATE INDEX IF NOT EXISTS event_logs_critical_idx ON event_logs(timestamp DESC, service) 
WHERE level IN ('ERROR', 'CRITICAL');

-- Table des statistiques de performance (améliorée)
CREATE TABLE IF NOT EXISTS performance_stats (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    period VARCHAR(20) NOT NULL CHECK (period IN ('daily', 'weekly', 'monthly', 'yearly')),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    total_trades INTEGER NOT NULL DEFAULT 0,
    winning_trades INTEGER NOT NULL DEFAULT 0,
    losing_trades INTEGER NOT NULL DEFAULT 0,
    break_even_trades INTEGER NOT NULL DEFAULT 0,
    profit_loss NUMERIC(24, 8) NOT NULL DEFAULT 0,
    profit_loss_percent NUMERIC(8, 4) NOT NULL DEFAULT 0,
    max_drawdown NUMERIC(8, 4),
    average_holding_time INTERVAL,
    sharpe_ratio NUMERIC(8, 4),
    win_rate NUMERIC(5, 2),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Index pour les stats
CREATE UNIQUE INDEX IF NOT EXISTS performance_stats_unique_idx ON performance_stats(symbol, strategy, period, start_date);
CREATE INDEX IF NOT EXISTS performance_stats_period_date_idx ON performance_stats(period, start_date DESC);
CREATE INDEX IF NOT EXISTS performance_stats_strategy_period_idx ON performance_stats(strategy, period);

-- Table des règles de gestion des risques
CREATE TABLE IF NOT EXISTS risk_rules (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    rule_type VARCHAR(20) NOT NULL CHECK (rule_type IN ('exposure', 'volatility', 'drawdown', 'position_size', 'daily_loss')),
    symbol VARCHAR(20),  -- NULL pour règle globale
    strategy VARCHAR(50),  -- NULL pour règle globale
    threshold NUMERIC(10, 4) NOT NULL,
    action VARCHAR(20) NOT NULL CHECK (action IN ('warn', 'pause', 'disable', 'force_close')),
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    triggered_count INTEGER NOT NULL DEFAULT 0,
    last_triggered TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Index pour les règles
CREATE INDEX IF NOT EXISTS risk_rules_type_enabled_idx ON risk_rules(rule_type, enabled);
CREATE INDEX IF NOT EXISTS risk_rules_symbol_strategy_idx ON risk_rules(symbol, strategy);
CREATE INDEX IF NOT EXISTS risk_rules_enabled_idx ON risk_rules(enabled);

-- Table des alertes (améliorée)
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    alert_type VARCHAR(20) NOT NULL,
    level VARCHAR(20) NOT NULL CHECK (level IN ('info', 'warning', 'critical')),
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    symbol VARCHAR(20),
    strategy VARCHAR(50),
    cycle_id VARCHAR(50),
    data JSONB,
    is_read BOOLEAN NOT NULL DEFAULT FALSE,
    is_acknowledged BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    acknowledged_at TIMESTAMP
);

-- Index pour les alertes
CREATE INDEX IF NOT EXISTS alerts_type_level_idx ON alerts(alert_type, level);
CREATE INDEX IF NOT EXISTS alerts_created_at_idx ON alerts(created_at DESC);
CREATE INDEX IF NOT EXISTS alerts_is_read_idx ON alerts(is_read, created_at DESC);
CREATE INDEX IF NOT EXISTS alerts_level_read_idx ON alerts(level, is_read, created_at DESC);
-- Index partiel pour les alertes non lues
CREATE INDEX IF NOT EXISTS alerts_unread_idx ON alerts(created_at DESC) WHERE is_read = FALSE;

-- Table des contraintes de trading Binance (cache pour éviter les appels API répétés)
CREATE TABLE IF NOT EXISTS binance_constraints (
    symbol VARCHAR(20) PRIMARY KEY,
    min_qty NUMERIC(16, 8) NOT NULL,
    max_qty NUMERIC(16, 8) NOT NULL,
    step_size NUMERIC(16, 8) NOT NULL,
    min_notional NUMERIC(16, 8) NOT NULL,
    price_precision INTEGER NOT NULL,
    qty_precision INTEGER NOT NULL,
    last_updated TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Index pour les contraintes
CREATE INDEX IF NOT EXISTS binance_constraints_updated_idx ON binance_constraints(last_updated);

-- Vues pour faciliter les requêtes communes

-- Vue des performances par stratégie (optimisée)
CREATE OR REPLACE VIEW strategy_performance AS
SELECT 
    strategy,
    COUNT(*) as total_cycles,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_cycles,
    SUM(CASE WHEN profit_loss > 0 AND status = 'completed' THEN 1 ELSE 0 END) as winning_trades,
    SUM(CASE WHEN profit_loss < 0 AND status = 'completed' THEN 1 ELSE 0 END) as losing_trades,
    SUM(CASE WHEN profit_loss = 0 AND status = 'completed' THEN 1 ELSE 0 END) as break_even_trades,
    SUM(profit_loss) as total_profit_loss,
    AVG(CASE WHEN status = 'completed' THEN profit_loss_percent ELSE NULL END) as avg_profit_loss_percent,
    CASE 
        WHEN SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) > 0 THEN
            SUM(CASE WHEN profit_loss > 0 AND status = 'completed' THEN 1 ELSE 0 END)::NUMERIC / 
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) * 100
        ELSE 0
    END as win_rate,
    COUNT(DISTINCT symbol) as symbol_count,
    COUNT(CASE WHEN status NOT IN ('completed', 'canceled', 'failed') THEN 1 END) as active_cycles
FROM 
    trade_cycles
GROUP BY 
    strategy
ORDER BY 
    total_profit_loss DESC;

-- Vue des performances par symbole (optimisée)
CREATE OR REPLACE VIEW symbol_performance AS
SELECT 
    symbol,
    COUNT(*) as total_cycles,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_cycles,
    SUM(CASE WHEN profit_loss > 0 AND status = 'completed' THEN 1 ELSE 0 END) as winning_trades,
    SUM(CASE WHEN profit_loss < 0 AND status = 'completed' THEN 1 ELSE 0 END) as losing_trades,
    SUM(CASE WHEN profit_loss = 0 AND status = 'completed' THEN 1 ELSE 0 END) as break_even_trades,
    SUM(profit_loss) as total_profit_loss,
    AVG(CASE WHEN status = 'completed' THEN profit_loss_percent ELSE NULL END) as avg_profit_loss_percent,
    CASE 
        WHEN SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) > 0 THEN
            SUM(CASE WHEN profit_loss > 0 AND status = 'completed' THEN 1 ELSE 0 END)::NUMERIC / 
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) * 100
        ELSE 0
    END as win_rate,
    COUNT(DISTINCT strategy) as strategy_count,
    COUNT(CASE WHEN status NOT IN ('completed', 'canceled', 'failed') THEN 1 END) as active_cycles
FROM 
    trade_cycles
GROUP BY 
    symbol
ORDER BY 
    total_profit_loss DESC;

-- Vue des cycles actifs optimisée avec prix actuels
CREATE OR REPLACE VIEW active_cycles AS
WITH latest_prices AS (
    SELECT DISTINCT ON (symbol)
        symbol, 
        close as price,
        time
    FROM 
        market_data
    ORDER BY 
        symbol, time DESC
)
SELECT 
    tc.id, 
    tc.symbol, 
    tc.strategy, 
    tc.status, 
    tc.entry_price, 
    tc.quantity, 
    tc.stop_price,
    lp.price as current_price,
    lp.time as price_timestamp,
    CASE 
        WHEN tc.entry_price IS NOT NULL AND lp.price IS NOT NULL THEN
            CASE 
                WHEN tc.status LIKE '%BUY%' THEN 
                    (lp.price - tc.entry_price) / tc.entry_price * 100
                WHEN tc.status LIKE '%SELL%' THEN 
                    (tc.entry_price - lp.price) / tc.entry_price * 100
                ELSE NULL
            END
        ELSE NULL
    END as unrealized_pl_percent,
    tc.created_at,
    tc.updated_at,
    EXTRACT(EPOCH FROM (NOW() - tc.created_at))/3600 as hours_active,
    tc.metadata
FROM 
    trade_cycles tc
LEFT JOIN 
    latest_prices lp ON tc.symbol = lp.symbol
WHERE 
    tc.status NOT IN ('completed', 'canceled', 'failed')
ORDER BY 
    tc.created_at DESC;

-- Vue des performances quotidiennes
CREATE OR REPLACE VIEW daily_performance AS
SELECT 
    DATE(completed_at) as trade_date,
    COUNT(*) as total_trades,
    SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losing_trades,
    SUM(profit_loss) as daily_profit_loss,
    AVG(profit_loss_percent) as avg_profit_loss_percent,
    CASE 
        WHEN COUNT(*) > 0 THEN
            SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END)::NUMERIC / COUNT(*) * 100
        ELSE 0
    END as daily_win_rate
FROM 
    trade_cycles
WHERE 
    status = 'completed'
    AND completed_at IS NOT NULL
GROUP BY 
    DATE(completed_at)
ORDER BY 
    trade_date DESC;

-- Vue des alertes actives
CREATE OR REPLACE VIEW active_alerts AS
SELECT 
    id,
    alert_type,
    level,
    title,
    message,
    symbol,
    strategy,
    cycle_id,
    created_at,
    EXTRACT(EPOCH FROM (NOW() - created_at))/3600 as hours_old
FROM 
    alerts
WHERE 
    is_read = FALSE
ORDER BY 
    CASE level 
        WHEN 'critical' THEN 1 
        WHEN 'warning' THEN 2 
        WHEN 'info' THEN 3 
    END,
    created_at DESC;

-- Fonctions utiles

-- Fonction pour calculer la performance sur une période
CREATE OR REPLACE FUNCTION calculate_performance(
    p_start_date TIMESTAMP,
    p_end_date TIMESTAMP,
    p_symbol VARCHAR DEFAULT NULL,
    p_strategy VARCHAR DEFAULT NULL
)
RETURNS TABLE (
    total_trades BIGINT,
    winning_trades BIGINT,
    losing_trades BIGINT,
    win_rate NUMERIC,
    total_profit_loss NUMERIC,
    avg_profit_loss_percent NUMERIC,
    max_profit_percent NUMERIC,
    max_loss_percent NUMERIC,
    avg_holding_hours NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_trades,
        SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
        SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losing_trades,
        CASE 
            WHEN COUNT(*) > 0 THEN 
                SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END)::NUMERIC / COUNT(*) * 100
            ELSE 0
        END as win_rate,
        SUM(profit_loss) as total_profit_loss,
        AVG(profit_loss_percent) as avg_profit_loss_percent,
        MAX(profit_loss_percent) as max_profit_percent,
        MIN(profit_loss_percent) as max_loss_percent,
        AVG(EXTRACT(EPOCH FROM (completed_at - created_at))/3600) as avg_holding_hours
    FROM 
        trade_cycles
    WHERE 
        status = 'completed'
        AND completed_at BETWEEN p_start_date AND p_end_date
        AND (p_symbol IS NULL OR symbol = p_symbol)
        AND (p_strategy IS NULL OR strategy = p_strategy);
END;
$$ LANGUAGE plpgsql;

-- Fonction pour nettoyer les cycles fantômes (utilisée par la réconciliation)
CREATE OR REPLACE FUNCTION cleanup_phantom_cycles()
RETURNS TABLE (
    cleaned_count INTEGER,
    details JSONB
) AS $$
DECLARE
    phantom_count INTEGER := 0;
    details_json JSONB := '{}';
BEGIN
    -- Marquer les cycles active_sell sans exit_order_id comme canceled
    UPDATE trade_cycles 
    SET 
        status = 'canceled', 
        updated_at = NOW(),
        metadata = COALESCE(metadata, '{}') || '{"cancel_reason": "phantom_cycle_cleanup", "cleanup_timestamp": "' || NOW()::text || '"}'
    WHERE 
        status = 'active_sell' 
        AND exit_order_id IS NULL;
    
    GET DIAGNOSTICS phantom_count = ROW_COUNT;
    
    details_json := jsonb_build_object(
        'phantom_cycles_cleaned', phantom_count,
        'cleanup_timestamp', NOW()
    );
    
    RETURN QUERY SELECT phantom_count, details_json;
END;
$$ LANGUAGE plpgsql;

-- Procédure optimisée pour calculer et stocker les statistiques quotidiennes
CREATE OR REPLACE PROCEDURE update_daily_stats(p_date DATE DEFAULT CURRENT_DATE)
LANGUAGE plpgsql
AS $$
DECLARE
    v_start_timestamp TIMESTAMP;
    v_end_timestamp TIMESTAMP;
BEGIN
    v_start_timestamp := p_date::TIMESTAMP;
    v_end_timestamp := (p_date + INTERVAL '1 day')::TIMESTAMP;
    
    -- Supprimer les anciennes stats pour cette date
    DELETE FROM performance_stats 
    WHERE period = 'daily' AND start_date = p_date;
    
    -- Calculer et insérer les nouvelles statistiques
    WITH completed_cycles AS (
        SELECT 
            symbol,
            strategy,
            profit_loss,
            profit_loss_percent,
            completed_at,
            created_at
        FROM 
            trade_cycles
        WHERE 
            status = 'completed'
            AND completed_at >= v_start_timestamp
            AND completed_at < v_end_timestamp
    ),
    daily_stats AS (
        SELECT 
            symbol,
            strategy,
            COUNT(*) as total_trades,
            SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
            SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losing_trades,
            SUM(CASE WHEN profit_loss = 0 THEN 1 ELSE 0 END) as break_even_trades,
            SUM(profit_loss) as profit_loss,
            AVG(profit_loss_percent) as profit_loss_percent,
            AVG(completed_at - created_at) as average_holding_time,
            CASE 
                WHEN COUNT(*) > 0 THEN
                    SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END)::NUMERIC / COUNT(*) * 100
                ELSE 0
            END as win_rate
        FROM 
            completed_cycles
        GROUP BY 
            symbol, strategy
    )
    INSERT INTO performance_stats (
        symbol, strategy, period, start_date, end_date,
        total_trades, winning_trades, losing_trades, break_even_trades,
        profit_loss, profit_loss_percent, average_holding_time, win_rate
    )
    SELECT 
        symbol,
        strategy,
        'daily',
        p_date,
        p_date,
        total_trades,
        winning_trades,
        losing_trades,
        break_even_trades,
        profit_loss,
        profit_loss_percent,
        average_holding_time,
        win_rate
    FROM 
        daily_stats;
        
    RAISE NOTICE 'Statistiques quotidiennes calculées pour %', p_date;
END;
$$;

-- Trigger pour mettre à jour les timestamps automatiquement
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Créer les triggers pour les tables principales
DROP TRIGGER IF EXISTS update_trade_executions_timestamp ON trade_executions;
CREATE TRIGGER update_trade_executions_timestamp
BEFORE UPDATE ON trade_executions
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();

DROP TRIGGER IF EXISTS update_trade_cycles_timestamp ON trade_cycles;
CREATE TRIGGER update_trade_cycles_timestamp
BEFORE UPDATE ON trade_cycles
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();

DROP TRIGGER IF EXISTS update_strategy_configs_timestamp ON strategy_configs;
CREATE TRIGGER update_strategy_configs_timestamp
BEFORE UPDATE ON strategy_configs
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();

DROP TRIGGER IF EXISTS update_risk_rules_timestamp ON risk_rules;
CREATE TRIGGER update_risk_rules_timestamp
BEFORE UPDATE ON risk_rules
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();

-- Fonction de nettoyage périodique (améliorée)
CREATE OR REPLACE PROCEDURE maintenance_cleanup(older_than_days INT DEFAULT 90)
LANGUAGE plpgsql
AS $$
DECLARE
    signals_deleted INT;
    logs_deleted INT;
    alerts_deleted INT;
BEGIN
    -- Nettoyer les anciennes données de trading_signals
    DELETE FROM trading_signals
    WHERE created_at < NOW() - (older_than_days || ' days')::INTERVAL;
    GET DIAGNOSTICS signals_deleted = ROW_COUNT;
    
    -- Nettoyer les vieux logs d'événements (garder uniquement les erreurs pour les périodes plus anciennes)
    DELETE FROM event_logs
    WHERE timestamp < NOW() - (older_than_days || ' days')::INTERVAL
    AND level NOT IN ('ERROR', 'CRITICAL');
    GET DIAGNOSTICS logs_deleted = ROW_COUNT;
    
    -- Nettoyer les alertes lues anciennes
    DELETE FROM alerts
    WHERE is_read = TRUE 
    AND created_at < NOW() - (older_than_days || ' days')::INTERVAL;
    GET DIAGNOSTICS alerts_deleted = ROW_COUNT;
    
    -- VACUUM ANALYZE pour optimiser les performances après suppression
    ANALYZE trading_signals;
    ANALYZE event_logs;
    ANALYZE alerts;
    
    RAISE NOTICE 'Maintenance terminée. Supprimé: % signaux, % logs, % alertes', 
                 signals_deleted, logs_deleted, alerts_deleted;
END;
$$;

-- Fonction pour analyser la santé de la base de données
CREATE OR REPLACE FUNCTION db_health_check() 
RETURNS TABLE (
    table_name TEXT,
    row_count BIGINT,
    table_size TEXT,
    index_size TEXT,
    bloat_pct NUMERIC,
    last_vacuum TIMESTAMP,
    last_analyze TIMESTAMP
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH table_stats AS (
        SELECT
            schemaname || '.' || relname AS full_table_name,
            n_live_tup AS row_estimate,
            pg_size_pretty(pg_total_relation_size('"' || schemaname || '"."' || relname || '"')) AS total_size,
            pg_size_pretty(pg_indexes_size('"' || schemaname || '"."' || relname || '"')) AS index_size,
            CASE WHEN n_dead_tup > 0 THEN
                round(n_dead_tup * 100.0 / (n_live_tup + n_dead_tup), 1)
            ELSE 0 END AS bloat_pct,
            last_vacuum,
            last_autovacuum,
            last_analyze,
            last_autoanalyze
        FROM
            pg_stat_user_tables
        WHERE
            schemaname = 'public'
    )
    SELECT
        split_part(ts.full_table_name, '.', 2) AS table_name,
        ts.row_estimate,
        ts.total_size,
        ts.index_size,
        ts.bloat_pct,
        COALESCE(ts.last_vacuum, ts.last_autovacuum) AS last_vacuum,
        COALESCE(ts.last_analyze, ts.last_autoanalyze) AS last_analyze
    FROM
        table_stats ts
    ORDER BY
        row_estimate DESC;
END;
$$;

-- Insérer les contraintes Binance par défaut (basées sur les vraies valeurs de l'API)
INSERT INTO binance_constraints (symbol, min_qty, max_qty, step_size, min_notional, price_precision, qty_precision) VALUES
('BTCUSDC', 0.00001, 9000.0, 0.00001, 5.0, 2, 5),
('ETHUSDC', 0.0001, 90000.0, 0.0001, 5.0, 2, 4),
('ETHBTC', 0.001, 100000.0, 0.001, 0.0001, 6, 3)
ON CONFLICT (symbol) DO UPDATE SET
    min_qty = EXCLUDED.min_qty,
    max_qty = EXCLUDED.max_qty,
    step_size = EXCLUDED.step_size,
    min_notional = EXCLUDED.min_notional,
    price_precision = EXCLUDED.price_precision,
    qty_precision = EXCLUDED.qty_precision,
    last_updated = NOW();

-- Insérer les configurations de stratégies par défaut
INSERT INTO strategy_configs (name, mode, symbols, params, max_simultaneous_trades, enabled) VALUES
('RSI_Strategy', 'active', '["BTCUSDC", "ETHUSDC"]', '{"window": 14, "overbought": 70, "oversold": 30}', 3, true),
('EMA_Cross_Strategy', 'active', '["BTCUSDC", "ETHUSDC"]', '{"SELL_window": 5, "BUY_window": 20}', 2, true),
('Bollinger_Bands_Strategy', 'monitoring', '["BTCUSDC", "ETHUSDC"]', '{"window": 20, "std_dev": 2.0}', 2, false),
('Ride_or_React_Strategy', 'active', '["BTCUSDC", "ETHUSDC", "ETHBTC"]', '{"thresholds": {"1h": 0.8, "3h": 2.5, "6h": 3.6, "12h": 5.1, "24h": 7.8}}', 1, true)
ON CONFLICT (name) DO UPDATE SET
    mode = EXCLUDED.mode,
    symbols = EXCLUDED.symbols,
    params = EXCLUDED.params,
    max_simultaneous_trades = EXCLUDED.max_simultaneous_trades,
    enabled = EXCLUDED.enabled,
    updated_at = NOW();

-- Insérer les règles de risque par défaut
INSERT INTO risk_rules (name, rule_type, threshold, action, enabled) VALUES
('Max Daily Loss', 'daily_loss', 5.0, 'pause', true),
('Max Drawdown', 'drawdown', 10.0, 'warn', true),
('Max Position Size', 'position_size', 20.0, 'warn', true),
('High Volatility Warning', 'volatility', 15.0, 'warn', true)
ON CONFLICT (name) DO UPDATE SET
    rule_type = EXCLUDED.rule_type,
    threshold = EXCLUDED.threshold,
    action = EXCLUDED.action,
    enabled = EXCLUDED.enabled,
    updated_at = NOW();

-- Ajouter les colonnes si elles n'existent pas
ALTER TABLE trade_cycles 
ADD COLUMN IF NOT EXISTS min_price NUMERIC(16, 8), 
ADD COLUMN IF NOT EXISTS max_price NUMERIC(16, 8);

-- Initialiser les valeurs NULL avec entry_price pour les cycles actifs
UPDATE trade_cycles 
SET min_price = entry_price, 
    max_price = entry_price 
WHERE (min_price IS NULL OR max_price IS NULL) 
  AND status NOT IN ('completed', 'canceled', 'failed');

-- Créer des index pour améliorer les performances des requêtes sur ces colonnes
CREATE INDEX IF NOT EXISTS trade_cycles_min_price_idx ON trade_cycles(min_price) 
WHERE status NOT IN ('completed', 'canceled', 'failed');

CREATE INDEX IF NOT EXISTS trade_cycles_max_price_idx ON trade_cycles(max_price) 
WHERE status NOT IN ('completed', 'canceled', 'failed');

-- Insérer des commentaires sur les tables pour la documentation
COMMENT ON TABLE trade_cycles IS 'Cycles de trading complets avec métadonnées pour la réconciliation';
COMMENT ON COLUMN trade_cycles.metadata IS 'Métadonnées JSON pour stocker des infos supplémentaires (réconciliation, raisons d''annulation, etc.)';
COMMENT ON COLUMN trade_cycles.status IS 'Statut du cycle en minuscules pour cohérence avec les énumérations Python';

COMMENT ON TABLE binance_constraints IS 'Cache des contraintes de trading Binance pour éviter les appels API répétés';

-- Politique de rétention pour TimescaleDB (données de marché)
SELECT add_retention_policy('market_data', INTERVAL '1 year');

-- Exécuter ANALYZE pour mettre à jour les statistiques
ANALYZE;