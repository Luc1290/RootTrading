-- Schéma de base de données pour RootTrading
-- Contient toutes les tables nécessaires au fonctionnement du système

-- Activer les extensions nécessaires
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb" CASCADE;  -- Pour les données temporelles

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
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Index pour les exécutions
CREATE INDEX IF NOT EXISTS trade_executions_cycle_id_idx ON trade_executions(cycle_id);
CREATE INDEX IF NOT EXISTS trade_executions_timestamp_idx ON trade_executions(timestamp);
CREATE INDEX IF NOT EXISTS trade_executions_symbol_idx ON trade_executions(symbol);

-- Table des cycles de trading
CREATE TABLE IF NOT EXISTS trade_cycles (
    id VARCHAR(50) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('initiating', 'waiting_buy', 'active_buy', 'waiting_sell', 'active_sell', 'completed', 'canceled', 'failed')),
    entry_order_id VARCHAR(50),
    exit_order_id VARCHAR(50),
    entry_price NUMERIC(16, 8),
    exit_price NUMERIC(16, 8),
    quantity NUMERIC(16, 8),
    target_price NUMERIC(16, 8),
    stop_price NUMERIC(16, 8),
    trailing_delta NUMERIC(16, 8),
    profit_loss NUMERIC(16, 8),
    profit_loss_percent NUMERIC(16, 8),
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    pocket VARCHAR(20),
    metadata JSONB,
    demo BOOLEAN NOT NULL DEFAULT FALSE
);

-- Index pour les cycles
CREATE INDEX IF NOT EXISTS trade_cycles_symbol_idx ON trade_cycles(symbol);
CREATE INDEX IF NOT EXISTS trade_cycles_strategy_idx ON trade_cycles(strategy);
CREATE INDEX IF NOT EXISTS trade_cycles_status_idx ON trade_cycles(status);
CREATE INDEX IF NOT EXISTS trade_cycles_created_at_idx ON trade_cycles(created_at);
CREATE INDEX IF NOT EXISTS trade_cycles_completed_at_idx ON trade_cycles(completed_at);
-- Index pour les requêtes fréquentes
CREATE INDEX IF NOT EXISTS trade_cycles_symbol_created_idx ON trade_cycles(symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS trade_cycles_strategy_created_idx ON trade_cycles(strategy, created_at DESC);
-- Index GIN pour les recherches dans les métadonnées JSON
CREATE INDEX IF NOT EXISTS trade_cycles_metadata_idx ON trade_cycles USING GIN (metadata);
-- Index pour les calculs de performance
CREATE INDEX IF NOT EXISTS trade_cycles_profit_status_idx ON trade_cycles(profit_loss_percent, status) 
WHERE status = 'completed';

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
-- Index GIN pour les recherches dans les métadonnées JSON
CREATE INDEX IF NOT EXISTS trading_signals_metadata_idx ON trading_signals USING GIN (metadata);

-- Table des données de marché (séries temporelles)
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

-- Table des poches de capital
CREATE TABLE IF NOT EXISTS capital_pockets (
    id SERIAL PRIMARY KEY,
    pocket_type VARCHAR(20) NOT NULL CHECK (pocket_type IN ('active', 'buffer', 'safety')),
    allocation_percent NUMERIC(5, 2) NOT NULL,
    current_value NUMERIC(24, 8) NOT NULL,
    used_value NUMERIC(24, 8) NOT NULL,
    available_value NUMERIC(24, 8) NOT NULL,
    active_cycles INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Table des paramètres des stratégies
CREATE TABLE IF NOT EXISTS strategy_configs (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
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
-- Index GIN pour les recherches dans les tableaux de symboles
CREATE INDEX IF NOT EXISTS strategy_configs_symbols_idx ON strategy_configs USING GIN (symbols);
CREATE INDEX IF NOT EXISTS strategy_configs_params_idx ON strategy_configs USING GIN (params);

-- Table des journaux d'événements
CREATE TABLE IF NOT EXISTS event_logs (
    id SERIAL PRIMARY KEY,
    service VARCHAR(50) NOT NULL,
    level VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    data JSONB,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Index pour les journaux
CREATE INDEX IF NOT EXISTS event_logs_service_idx ON event_logs(service);
CREATE INDEX IF NOT EXISTS event_logs_level_idx ON event_logs(level);
CREATE INDEX IF NOT EXISTS event_logs_timestamp_idx ON event_logs(timestamp);

-- Table des statistiques de performance
CREATE TABLE IF NOT EXISTS performance_stats (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    period VARCHAR(20) NOT NULL,  -- 'daily', 'weekly', 'monthly'
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
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Index pour les stats
CREATE INDEX IF NOT EXISTS performance_stats_symbol_strategy_idx ON performance_stats(symbol, strategy);
CREATE INDEX IF NOT EXISTS performance_stats_period_idx ON performance_stats(period, start_date);

-- Table des règles de gestion des risques
CREATE TABLE IF NOT EXISTS risk_rules (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    rule_type VARCHAR(20) NOT NULL,  -- 'exposure', 'volatility', 'drawdown', etc.
    symbol VARCHAR(20),  -- NULL pour règle globale
    strategy VARCHAR(50),  -- NULL pour règle globale
    threshold NUMERIC(10, 4) NOT NULL,
    action VARCHAR(20) NOT NULL,  -- 'warn', 'pause', 'disable'
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Index pour les règles
CREATE INDEX IF NOT EXISTS risk_rules_type_idx ON risk_rules(rule_type);
CREATE INDEX IF NOT EXISTS risk_rules_symbol_strategy_idx ON risk_rules(symbol, strategy);

-- Table des alertes
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    alert_type VARCHAR(20) NOT NULL,
    level VARCHAR(20) NOT NULL CHECK (level IN ('info', 'warning', 'critical')),
    message TEXT NOT NULL,
    symbol VARCHAR(20),
    strategy VARCHAR(50),
    data JSONB,
    is_read BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Index pour les alertes
CREATE INDEX IF NOT EXISTS alerts_type_level_idx ON alerts(alert_type, level);
CREATE INDEX IF NOT EXISTS alerts_created_at_idx ON alerts(created_at);
CREATE INDEX IF NOT EXISTS alerts_is_read_idx ON alerts(is_read);

-- Vues pour faciliter les requêtes communes

-- Vue des performances par stratégie
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
    COUNT(DISTINCT symbol) as symbol_count
FROM 
    trade_cycles
GROUP BY 
    strategy
ORDER BY 
    total_profit_loss DESC;

-- Vue des performances par symbole
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
    COUNT(DISTINCT strategy) as strategy_count
FROM 
    trade_cycles
GROUP BY 
    symbol
ORDER BY 
    total_profit_loss DESC;

-- Vue des cycles actifs optimisée
CREATE OR REPLACE VIEW active_cycles AS
WITH latest_prices AS (
    SELECT 
        symbol, 
        close AS price,
        MAX(time) as latest_time
    FROM 
        market_data
    GROUP BY 
        symbol, close
)
SELECT 
    tc.id, 
    tc.symbol, 
    tc.strategy, 
    tc.status, 
    tc.entry_price, 
    tc.quantity, 
    tc.target_price, 
    tc.stop_price,
    lp.price as current_price,
    CASE 
        WHEN tc.status = 'active_buy' AND lp.price IS NOT NULL THEN 
            (lp.price - tc.entry_price) / tc.entry_price * 100
        WHEN tc.status = 'active_sell' AND lp.price IS NOT NULL THEN 
            (tc.entry_price - lp.price) / tc.entry_price * 100
        ELSE NULL
    END as unrealized_pl_percent,
    tc.created_at,
    tc.updated_at,
    tc.pocket
FROM 
    trade_cycles tc
LEFT JOIN 
    latest_prices lp ON tc.symbol = lp.symbol
WHERE 
    tc.status NOT IN ('completed', 'canceled', 'failed');

-- Vue des performances quotidiennes
CREATE OR REPLACE VIEW daily_performance AS
SELECT 
    DATE(completed_at) as trade_date,
    COUNT(*) as total_trades,
    SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losing_trades,
    SUM(profit_loss) as daily_profit_loss,
    AVG(profit_loss_percent) as avg_profit_loss_percent
FROM 
    trade_cycles
WHERE 
    status = 'completed'
    AND completed_at IS NOT NULL
GROUP BY 
    DATE(completed_at)
ORDER BY 
    trade_date DESC;

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
    avg_profit_loss_percent NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_trades,
        SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
        SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losing_trades,
        CASE 
            WHEN COUNT(*) > 0 THEN 
                SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END)::NUMERIC / COUNT(*)
            ELSE 0
        END as win_rate,
        SUM(profit_loss) as total_profit_loss,
        AVG(profit_loss_percent) as avg_profit_loss_percent
    FROM 
        trade_cycles
    WHERE 
        status = 'completed'
        AND completed_at BETWEEN p_start_date AND p_end_date
        AND (p_symbol IS NULL OR symbol = p_symbol)
        AND (p_strategy IS NULL OR strategy = p_strategy);
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
    
    -- Utilisation de CTE pour optimisation
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
    stats_by_strategy AS (
        SELECT 
            symbol,
            strategy,
            COUNT(*) as total_trades,
            SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
            SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losing_trades,
            SUM(CASE WHEN profit_loss = 0 THEN 1 ELSE 0 END) as break_even_trades,
            SUM(profit_loss) as profit_loss,
            AVG(profit_loss_percent) as profit_loss_percent,
            AVG(completed_at - created_at) as average_holding_time
        FROM 
            completed_cycles
        GROUP BY 
            symbol, strategy
    ),
    stats_global AS (
        SELECT 
            'ALL' as symbol,
            'ALL' as strategy,
            COUNT(*) as total_trades,
            SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
            SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losing_trades,
            SUM(CASE WHEN profit_loss = 0 THEN 1 ELSE 0 END) as break_even_trades,
            SUM(profit_loss) as profit_loss,
            AVG(profit_loss_percent) as profit_loss_percent,
            AVG(completed_at - created_at) as average_holding_time
        FROM 
            completed_cycles
    ),
    all_stats AS (
        SELECT * FROM stats_by_strategy
        UNION ALL
        SELECT * FROM stats_global
    )
    -- Supprimer et insérer en une seule transaction
    DELETE FROM performance_stats 
    WHERE period = 'daily' AND start_date = p_date;
    
    INSERT INTO performance_stats (
        symbol, strategy, period, start_date, end_date,
        total_trades, winning_trades, losing_trades, break_even_trades,
        profit_loss, profit_loss_percent, max_drawdown, average_holding_time
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
        NULL, -- max_drawdown calculé séparément
        average_holding_time
    FROM 
        all_stats;
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
CREATE TRIGGER update_trade_executions_timestamp
BEFORE UPDATE ON trade_executions
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER update_trade_cycles_timestamp
BEFORE UPDATE ON trade_cycles
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER update_strategy_configs_timestamp
BEFORE UPDATE ON strategy_configs
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER update_risk_rules_timestamp
BEFORE UPDATE ON risk_rules
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER update_capital_pockets_timestamp
BEFORE UPDATE ON capital_pockets
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();

-- Fonctions pour la gestion des cycles

-- Fonction pour obtenir le nombre de cycles actifs par poche
CREATE OR REPLACE FUNCTION get_active_cycles_by_pocket()
RETURNS TABLE (
    pocket VARCHAR,
    active_count BIGINT,
    total_value NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        tc.pocket,
        COUNT(*) as active_count,
        SUM(tc.quantity * tc.entry_price) as total_value
    FROM 
        trade_cycles tc
    WHERE 
        tc.status NOT IN ('completed', 'canceled', 'failed')
    GROUP BY 
        tc.pocket;
END;
$$ LANGUAGE plpgsql;

-- Fonction pour obtenir les statistiques de performance par stratégie
CREATE OR REPLACE FUNCTION get_strategy_stats(
    p_days INTEGER DEFAULT 30
)
RETURNS TABLE (
    strategy VARCHAR,
    win_rate NUMERIC,
    avg_profit NUMERIC,
    max_drawdown NUMERIC,
    sharpe_ratio NUMERIC,
    total_trades BIGINT,
    active_trades BIGINT
) AS $$
BEGIN
    RETURN QUERY
    WITH completed_trades AS (
        SELECT 
            strategy,
            profit_loss_percent,
            completed_at
        FROM 
            trade_cycles
        WHERE 
            status = 'completed'
            AND completed_at >= NOW() - INTERVAL '1 day' * p_days
    ),
    daily_returns AS (
        SELECT 
            strategy,
            DATE(completed_at) as trade_date,
            SUM(profit_loss_percent) as daily_return
        FROM 
            completed_trades
        GROUP BY 
            strategy, DATE(completed_at)
    ),
    stats AS (
        SELECT 
            strategy,
            COUNT(*) as total_trades,
            SUM(CASE WHEN profit_loss_percent > 0 THEN 1 ELSE 0 END)::NUMERIC / 
                NULLIF(COUNT(*), 0) as win_rate,
            AVG(profit_loss_percent) as avg_profit,
            -- Calcul approximatif du max drawdown à partir des rendements quotidiens
            COALESCE((
                SELECT 
                    MIN(cumulative_return - max_cumulative_return) as max_drawdown
                FROM (
                    SELECT 
                        dr.strategy,
                        dr.trade_date,
                        dr.daily_return,
                        SUM(dr.daily_return) OVER (PARTITION BY dr.strategy ORDER BY dr.trade_date) as cumulative_return,
                        MAX(SUM(dr.daily_return)) OVER (PARTITION BY dr.strategy ORDER BY dr.trade_date) as max_cumulative_return
                    FROM 
                        daily_returns dr
                    WHERE 
                        dr.strategy = c.strategy
                ) as sub
            ), 0) as max_drawdown,
            -- Calcul approximatif du ratio de Sharpe
            CASE 
                WHEN STDDEV(daily_return) OVER (PARTITION BY strategy) > 0 THEN
                    AVG(daily_return) OVER (PARTITION BY strategy) / NULLIF(STDDEV(daily_return) OVER (PARTITION BY strategy), 0) * SQRT(252)
                ELSE NULL
            END as sharpe_ratio
        FROM 
            completed_trades c
        LEFT JOIN 
            daily_returns dr ON c.strategy = dr.strategy
        GROUP BY 
            c.strategy
    )
    SELECT 
        s.strategy,
        s.win_rate,
        s.avg_profit,
        s.max_drawdown,
        s.sharpe_ratio,
        s.total_trades,
        COUNT(tc.id) as active_trades
    FROM 
        stats s
    LEFT JOIN 
        trade_cycles tc ON s.strategy = tc.strategy AND tc.status NOT IN ('completed', 'canceled', 'failed')
    GROUP BY 
        s.strategy, s.win_rate, s.avg_profit, s.max_drawdown, s.sharpe_ratio, s.total_trades
    ORDER BY 
        s.avg_profit DESC;
END;
$$ LANGUAGE plpgsql;

-- Fonction de nettoyage périodique
CREATE OR REPLACE PROCEDURE maintenance_cleanup(older_than_days INT DEFAULT 90)
LANGUAGE plpgsql
AS $$
BEGIN
    -- Nettoyer les anciennes données de trading_signals
    DELETE FROM trading_signals
    WHERE created_at < NOW() - (older_than_days || ' days')::INTERVAL;
    
    -- Nettoyer les vieux logs d'événements (garder uniquement les erreurs pour les périodes plus anciennes)
    DELETE FROM event_logs
    WHERE timestamp < NOW() - (older_than_days || ' days')::INTERVAL
    AND level NOT IN ('ERROR', 'CRITICAL');
    
    -- Nettoyer les alertes lues anciennes
    DELETE FROM alerts
    WHERE is_read = TRUE 
    AND created_at < NOW() - (older_than_days || ' days')::INTERVAL;
    
    -- VACUUM ANALYZE pour optimiser les performances après suppression
    ANALYZE trading_signals;
    ANALYZE event_logs;
    ANALYZE alerts;
    
    RAISE NOTICE 'Maintenance terminée. Données nettoyées plus anciennes que % jours', older_than_days;
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

-- Exécuter ANALYZE pour mettre à jour les statistiques
ANALYZE;