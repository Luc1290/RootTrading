-- Schéma de base de données COMPLET et PROPRE pour RootTrading
-- Version 2.0 - Architecture Nettoyée : Gateway OHLCV bruts + Market Analyzer calculs séparés
-- Date: 22 Juillet 2025

-- =====================================================
-- CONFIGURATION ET EXTENSIONS
-- =====================================================

-- Activer les extensions nécessaires
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "timescaledb" CASCADE;  -- Pour les données temporelles

-- Configuration PostgreSQL optimisée pour trading
SET timezone = 'UTC';
SET default_text_search_config = 'pg_catalog.english';

-- =====================================================
-- TABLE MARKET_DATA - DONNÉES OHLCV BRUTES
-- =====================================================

-- Table des données de marché OHLCV brutes (Gateway uniquement)
CREATE TABLE IF NOT EXISTS market_data (
    -- Identifiants temporels
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(5) NOT NULL,
    
    -- Données OHLCV de base (Gateway)
    open DECIMAL(20,8) NOT NULL,
    high DECIMAL(20,8) NOT NULL,
    low DECIMAL(20,8) NOT NULL,
    close DECIMAL(20,8) NOT NULL,
    volume DECIMAL(28,8) NOT NULL,
    
    -- Métadonnées Binance additionnelles
    quote_asset_volume DECIMAL(28,8),
    number_of_trades INTEGER,
    taker_buy_base_asset_volume DECIMAL(28,8),
    taker_buy_quote_asset_volume DECIMAL(28,8),
    
    -- Métadonnées système
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Contraintes
    PRIMARY KEY (time, symbol, timeframe),
    CONSTRAINT market_data_symbol_check CHECK (symbol ~ '^[A-Z0-9]+$'),
    CONSTRAINT market_data_timeframe_check CHECK (timeframe IN ('1m', '3m', '5m', '15m', '1h', '1d')),
    CONSTRAINT market_data_prices_check CHECK (high >= low AND high >= open AND high >= close AND low <= open AND low <= close)
);

-- Convertir en hypertable pour TimescaleDB
SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);

-- Index optimisés pour les données de marché
CREATE INDEX IF NOT EXISTS market_data_symbol_time_idx ON market_data (symbol, time DESC);
CREATE INDEX IF NOT EXISTS market_data_timeframe_idx ON market_data (timeframe, time DESC);
CREATE INDEX IF NOT EXISTS market_data_symbol_timeframe_idx ON market_data (symbol, timeframe, time DESC);

-- Politique de rétention (garder 2 ans de données OHLCV)
SELECT add_retention_policy('market_data', INTERVAL '2 years', if_not_exists => TRUE);

-- =====================================================
-- TABLE ANALYZER_DATA - ANALYSES AVANCÉES
-- =====================================================

-- Table des analyses avancées calculées par le Market Analyzer
CREATE TABLE IF NOT EXISTS analyzer_data (
    -- Identifiants
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(5) NOT NULL,
    
    -- Métadonnées
    analysis_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    analyzer_version VARCHAR(10) DEFAULT '1.0',
    
    -- === MOYENNES MOBILES AVANCÉES ===
    wma_20 DECIMAL(20,8),          -- Weighted Moving Average
    dema_12 DECIMAL(20,8),         -- Double Exponential MA
    tema_12 DECIMAL(20,8),         -- Triple Exponential MA
    hull_20 DECIMAL(20,8),         -- Hull Moving Average
    kama_14 DECIMAL(20,8),         -- Kaufman Adaptive MA
    
    -- === INDICATEURS DE BASE (pour compatibilité) ===
    rsi_14 DECIMAL(20,8),          -- RSI standard
    rsi_21 DECIMAL(20,8),          -- RSI 21 périodes
    ema_7 DECIMAL(20,8),           -- EMA rapide
    ema_12 DECIMAL(20,8),          -- EMA standard
    ema_26 DECIMAL(20,8),          -- EMA lente
    ema_50 DECIMAL(20,8),          -- EMA moyen terme
    ema_99 DECIMAL(20,8),          -- EMA long terme
    sma_20 DECIMAL(20,8),          -- SMA standard
    sma_50 DECIMAL(20,8),          -- SMA moyen terme
    
    -- === MACD COMPLET ===
    macd_line DECIMAL(20,8),       -- MACD Line
    macd_signal DECIMAL(20,8),     -- Signal Line
    macd_histogram DECIMAL(20,8),  -- Histogram
    ppo DECIMAL(20,8),             -- Percentage Price Oscillator
    macd_zero_cross BOOLEAN,       -- Crossover de la ligne zéro
    macd_signal_cross BOOLEAN,     -- Crossover signal
    macd_trend VARCHAR(10),        -- BULLISH/BEARISH/NEUTRAL
    
    -- === BOLLINGER BANDS COMPLET ===
    bb_upper DECIMAL(20,8),        -- Bande supérieure
    bb_middle DECIMAL(20,8),       -- Bande médiane (SMA)
    bb_lower DECIMAL(20,8),        -- Bande inférieure
    bb_position DECIMAL(20,8),     -- Position dans les bandes (0-1)
    bb_width DECIMAL(20,8),        -- Largeur des bandes
    bb_squeeze BOOLEAN,            -- Squeeze détecté
    bb_expansion BOOLEAN,          -- Expansion détectée
    bb_breakout_direction VARCHAR(10), -- UP/DOWN/NONE
    keltner_upper DECIMAL(20,8),   -- Keltner Channel Upper
    keltner_lower DECIMAL(20,8),   -- Keltner Channel Lower
    
    -- === STOCHASTIC COMPLET ===
    stoch_k DECIMAL(20,8),         -- %K standard
    stoch_d DECIMAL(20,8),         -- %D standard
    stoch_rsi DECIMAL(20,8),       -- Stochastic RSI
    stoch_fast_k DECIMAL(20,8),    -- Fast %K
    stoch_fast_d DECIMAL(20,8),    -- Fast %D
    stoch_divergence BOOLEAN,      -- Divergence détectée
    stoch_signal VARCHAR(10),      -- OVERBOUGHT/OVERSOLD/NEUTRAL
    
    -- === ATR & VOLATILITÉ ===
    atr_14 DECIMAL(20,8),          -- ATR standard
    atr_percentile DECIMAL(5,2),   -- ATR percentile (0-100)
    natr DECIMAL(20,8),            -- Normalized ATR
    volatility_regime VARCHAR(15), -- low/normal/high/extreme
    atr_stop_long DECIMAL(20,8),   -- ATR-based stop loss (long)
    atr_stop_short DECIMAL(20,8),  -- ATR-based stop loss (short)
    
    -- === ADX & DIRECTIONAL MOVEMENT ===
    adx_14 DECIMAL(20,8),          -- ADX standard
    plus_di DECIMAL(20,8),         -- +DI
    minus_di DECIMAL(20,8),        -- -DI
    dx DECIMAL(20,8),              -- Directional Movement Index
    adxr DECIMAL(20,8),            -- ADX Rating
    trend_strength VARCHAR(15),    -- WEAK/MODERATE/STRONG/VERY_STRONG
    directional_bias VARCHAR(10),  -- BULLISH/BEARISH/NEUTRAL
    trend_angle DECIMAL(20,8),     -- Angle de tendance
    
    -- === OSCILLATEURS ===
    williams_r DECIMAL(20,8),      -- Williams %R
    mfi_14 DECIMAL(20,8),          -- Money Flow Index
    cci_20 DECIMAL(20,8),          -- Commodity Channel Index
    momentum_10 DECIMAL(20,8),     -- Momentum
    roc_10 DECIMAL(20,8),          -- Rate of Change 10
    roc_20 DECIMAL(20,8),          -- Rate of Change 20
    
    -- === VOLUME AVANCÉ ===
    vwap_10 DECIMAL(20,8),         -- VWAP court terme (volume de base)
    vwap_quote_10 DECIMAL(20,8),   -- VWAP court terme (quote asset - plus précis)
    anchored_vwap DECIMAL(20,8),   -- VWAP ancré
    vwap_upper_band DECIMAL(20,8), -- VWAP + 1 std
    vwap_lower_band DECIMAL(20,8), -- VWAP - 1 std
    volume_ratio DECIMAL(10,4),    -- Ratio volume vs moyenne
    avg_volume_20 DECIMAL(28,8),   -- Volume moyen 20 périodes
    quote_volume_ratio DECIMAL(10,4), -- Ratio du volume en quote asset (USDC)
    avg_trade_size DECIMAL(28,8),  -- Taille moyenne des trades (volume/nb trades)
    trade_intensity DECIMAL(10,4), -- Intensité du trading (nb trades vs moyenne)
    obv DECIMAL(28,8),             -- On Balance Volume
    obv_ma_10 DECIMAL(28,8),       -- OBV Moving Average 10
    obv_oscillator DECIMAL(28,8),  -- OBV Oscillator
    ad_line DECIMAL(28,8),         -- Accumulation/Distribution Line
    
    -- === VOLUME PROFILE ===
    volume_profile_poc DECIMAL(20,8), -- Point of Control
    volume_profile_vah DECIMAL(20,8), -- Value Area High
    volume_profile_val DECIMAL(20,8), -- Value Area Low
    
    -- === DÉTECTION DE RÉGIME ===
    market_regime VARCHAR(20),     -- TRENDING_BULL/BEAR, RANGING, VOLATILE, BREAKOUT_BULL/BEAR, TRANSITION
    regime_strength VARCHAR(15),   -- WEAK/MODERATE/STRONG/EXTREME
    regime_confidence DECIMAL(5,2), -- Confiance (0-100%)
    regime_duration INTEGER,       -- Durée du régime en périodes
    trend_alignment DECIMAL(5,2),  -- Alignement des EMA (0-100%)
    momentum_score DECIMAL(5,2),   -- Score momentum (0-100%)
    
    -- === SUPPORT & RÉSISTANCE ===
    support_levels JSONB,          -- Array des niveaux support [{price, strength, touches}]
    resistance_levels JSONB,       -- Array des niveaux résistance
    nearest_support DECIMAL(20,8), -- Support le plus proche
    nearest_resistance DECIMAL(20,8), -- Résistance la plus proche
    support_strength VARCHAR(15),  -- WEAK/MODERATE/STRONG/MAJOR
    resistance_strength VARCHAR(15),
    break_probability DECIMAL(5,2), -- Probabilité de cassure (0-1)
    pivot_count INTEGER,           -- Nombre de pivots détectés
    
    -- === CONTEXTE VOLUME ===
    volume_context VARCHAR(20),    -- DEEP_OVERSOLD, BREAKOUT, PUMP_START, etc.
    volume_pattern VARCHAR(15),    -- BUILDUP/SPIKE/SUSTAINED_HIGH/DECLINING
    volume_quality_score DECIMAL(5,2), -- Qualité du volume (0-100%)
    relative_volume DECIMAL(10,4), -- Volume relatif vs moyenne
    volume_buildup_periods INTEGER, -- Nombre de périodes d'accumulation
    volume_spike_multiplier DECIMAL(10,4), -- Multiplicateur du spike
    
    -- === PATTERNS & SIGNAUX ===
    pattern_detected VARCHAR(30),  -- Pattern détecté (HAMMER, DOJI, etc.)
    pattern_confidence DECIMAL(5,2), -- Confiance dans le pattern
    signal_strength VARCHAR(15),   -- WEAK/MODERATE/STRONG/VERY_STRONG
    confluence_score DECIMAL(5,2), -- Score de confluence (0-100%)
    
    -- === MÉTADONNÉES PERFORMANCE ===
    calculation_time_ms INTEGER,   -- Temps de calcul en ms
    cache_hit_ratio DECIMAL(5,2),  -- Ratio de cache hit (0-100%)
    data_quality VARCHAR(15),      -- EXCELLENT/GOOD/FAIR/POOR
    anomaly_detected BOOLEAN,      -- Anomalie dans les données
    
    -- Contraintes
    PRIMARY KEY (time, symbol, timeframe),
    CONSTRAINT analyzer_data_symbol_check CHECK (symbol ~ '^[A-Z0-9]+$'),
    CONSTRAINT analyzer_data_timeframe_check CHECK (timeframe IN ('1m', '3m', '5m', '15m', '1h', '1d')),
    CONSTRAINT analyzer_data_regime_check CHECK (market_regime IN (
        'TRENDING_BULL', 'TRENDING_BEAR', 'RANGING', 'VOLATILE', 
        'BREAKOUT_BULL', 'BREAKOUT_BEAR', 'TRANSITION', 'UNKNOWN'
    )),
    CONSTRAINT analyzer_data_strength_check CHECK (regime_strength IN ('WEAK', 'MODERATE', 'STRONG', 'EXTREME')),
    CONSTRAINT analyzer_data_confidence_check CHECK (regime_confidence >= 0 AND regime_confidence <= 100),
    CONSTRAINT analyzer_data_quality_check CHECK (data_quality IN ('EXCELLENT', 'GOOD', 'FAIR', 'POOR'))
);

-- Convertir en hypertable pour TimescaleDB
SELECT create_hypertable('analyzer_data', 'time', if_not_exists => TRUE);

-- Index optimisés pour les requêtes fréquentes
CREATE INDEX IF NOT EXISTS analyzer_data_symbol_time_idx ON analyzer_data (symbol, time DESC);
CREATE INDEX IF NOT EXISTS analyzer_data_regime_idx ON analyzer_data (market_regime, regime_strength);
CREATE INDEX IF NOT EXISTS analyzer_data_timeframe_idx ON analyzer_data (timeframe, time DESC);
CREATE INDEX IF NOT EXISTS analyzer_data_pattern_idx ON analyzer_data (pattern_detected, pattern_confidence);
CREATE INDEX IF NOT EXISTS analyzer_data_signal_idx ON analyzer_data (signal_strength, confluence_score);
CREATE INDEX IF NOT EXISTS analyzer_data_volume_context_idx ON analyzer_data (volume_context, volume_quality_score);

-- Index pour support/résistance (JSONB)
CREATE INDEX IF NOT EXISTS analyzer_data_support_gin_idx ON analyzer_data USING GIN (support_levels);
CREATE INDEX IF NOT EXISTS analyzer_data_resistance_gin_idx ON analyzer_data USING GIN (resistance_levels);

-- Index composite pour performance
CREATE INDEX IF NOT EXISTS analyzer_data_performance_idx ON analyzer_data (
    symbol, timeframe, market_regime, time DESC
) WHERE regime_confidence > 70;

-- Politique de rétention (garder 1 an de données d'analyse)
SELECT add_retention_policy('analyzer_data', INTERVAL '1 year', if_not_exists => TRUE);

-- =====================================================
-- TABLES DE TRADING
-- =====================================================

-- Table des exécutions d'ordres
CREATE TABLE IF NOT EXISTS trade_executions (
    order_id VARCHAR(50) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    status VARCHAR(20) NOT NULL CHECK (status IN ('NEW', 'PARTIALLY_FILLED', 'FILLED', 'CANCELED', 'REJECTED', 'EXPIRED', 'PENDING_CANCEL')),
    price NUMERIC(20, 12) NOT NULL,
    quantity NUMERIC(20, 8) NOT NULL,
    quote_quantity NUMERIC(20, 8) NOT NULL,
    fee NUMERIC(16, 8),
    fee_asset VARCHAR(10),
    role VARCHAR(10) CHECK (role IN ('maker', 'taker')),
    timestamp TIMESTAMP NOT NULL,
    cycle_id VARCHAR(50),
    demo BOOLEAN NOT NULL DEFAULT FALSE,
    id VARCHAR(50),
    metadata JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Table des cycles de trading
CREATE TABLE IF NOT EXISTS trade_cycles (
    id VARCHAR(50) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('initiating', 'waiting_buy', 'active_buy', 'waiting_sell', 'active_sell', 'completed', 'canceled', 'failed')),
    side VARCHAR(4) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    entry_order_id VARCHAR(50),
    exit_order_id VARCHAR(50),
    entry_price NUMERIC(20, 12),
    exit_price NUMERIC(20, 12),
    quantity NUMERIC(20, 8),
    stop_price NUMERIC(20, 12),
    trailing_delta NUMERIC(16, 8),
    min_price NUMERIC(20, 12),
    max_price NUMERIC(20, 12),
    profit_loss NUMERIC(16, 8),
    profit_loss_percent NUMERIC(16, 8),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,
    confirmed BOOLEAN NOT NULL DEFAULT FALSE,
    metadata JSONB,
    demo BOOLEAN NOT NULL DEFAULT FALSE    
);

-- Table des signaux de trading
CREATE TABLE IF NOT EXISTS trading_signals (
    id SERIAL PRIMARY KEY,
    strategy VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    timestamp TIMESTAMP NOT NULL,
    confidence DECIMAL(5,2) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    metadata JSONB,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),

    -- Contrainte unique pour prévenir les signaux dupliqués
    CONSTRAINT unique_signal_constraint
    UNIQUE (strategy, symbol, side, timestamp, confidence)
);

-- Table des signaux Telegram
CREATE TABLE IF NOT EXISTS telegram_signals (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('BUY', 'SELL')),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Informations du signal Telegram
    score INTEGER NOT NULL CHECK (score >= 0 AND score <= 100),
    price DECIMAL(20,8) NOT NULL,
    action VARCHAR(30) NOT NULL,

    -- Targets et Stop Loss
    tp1 DECIMAL(20,8),
    tp2 DECIMAL(20,8),
    tp3 DECIMAL(20,8),
    stop_loss DECIMAL(20,8),

    -- Métadonnées d'analyse
    reason TEXT,
    momentum DECIMAL(10,2),
    volume_ratio DECIMAL(10,4),
    regime VARCHAR(30),
    estimated_hold_time VARCHAR(50),
    grade VARCHAR(5),
    rr_ratio DECIMAL(10,4),
    risk_level VARCHAR(20),

    -- Métadonnées système
    telegram_sent BOOLEAN DEFAULT TRUE,
    telegram_message_id VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Métadonnées additionnelles (JSON pour flexibilité)
    metadata JSONB,

    CONSTRAINT telegram_signals_symbol_check CHECK (symbol ~ '^[A-Z0-9]+$'),
    CONSTRAINT telegram_signals_action_check CHECK (action IN (
        'BUY_NOW', 'BUY_DCA', 'EARLY_ENTRY', 'WAIT_PULLBACK', 'WAIT_BREAKOUT',
        'WAIT_OVERSOLD', 'WAIT', 'SELL_OVERBOUGHT', 'AVOID'
    ))
);

-- Table de configuration des stratégies
CREATE TABLE IF NOT EXISTS strategy_configs (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(50) NOT NULL UNIQUE,
    symbol VARCHAR(20) NOT NULL,
    config JSONB NOT NULL,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Table des positions (pour tracking)
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('LONG', 'SHORT')),
    size NUMERIC(20, 8) NOT NULL,
    entry_price NUMERIC(20, 12) NOT NULL,
    current_price NUMERIC(20, 12),
    unrealized_pnl NUMERIC(16, 8),
    realized_pnl NUMERIC(16, 8) DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    closed_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'CLOSED'))
);

-- Table des balances du portefeuille
CREATE TABLE IF NOT EXISTS portfolio_balances (
    id SERIAL PRIMARY KEY,
    asset VARCHAR(20) NOT NULL,
    free DECIMAL(20,8) NOT NULL DEFAULT 0,
    locked DECIMAL(20,8) NOT NULL DEFAULT 0,
    total DECIMAL(20,8) NOT NULL DEFAULT 0,
    value_usdc DECIMAL(20,8),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =====================================================
-- INDEX POUR LES TABLES DE TRADING
-- =====================================================

-- Index pour trade_executions
CREATE INDEX IF NOT EXISTS trade_executions_cycle_id_idx ON trade_executions(cycle_id);
CREATE INDEX IF NOT EXISTS trade_executions_timestamp_idx ON trade_executions(timestamp);
CREATE INDEX IF NOT EXISTS trade_executions_symbol_idx ON trade_executions(symbol);
CREATE INDEX IF NOT EXISTS trade_executions_status_idx ON trade_executions(status);
CREATE INDEX IF NOT EXISTS trade_executions_metadata_idx ON trade_executions USING GIN (metadata);
CREATE INDEX IF NOT EXISTS trade_executions_id_idx ON trade_executions(id) WHERE id IS NOT NULL;

-- Index pour trade_cycles
CREATE INDEX IF NOT EXISTS trade_cycles_symbol_idx ON trade_cycles(symbol);
CREATE INDEX IF NOT EXISTS trade_cycles_strategy_idx ON trade_cycles(strategy);
CREATE INDEX IF NOT EXISTS trade_cycles_side_idx ON trade_cycles(side);
CREATE INDEX IF NOT EXISTS trade_cycles_symbol_side_status_idx ON trade_cycles(symbol, side, status);
CREATE INDEX IF NOT EXISTS trade_cycles_status_idx ON trade_cycles(status);
CREATE INDEX IF NOT EXISTS trade_cycles_created_at_idx ON trade_cycles(created_at);
CREATE INDEX IF NOT EXISTS trade_cycles_completed_at_idx ON trade_cycles(completed_at);
CREATE INDEX IF NOT EXISTS trade_cycles_symbol_created_idx ON trade_cycles(symbol, created_at DESC);
CREATE INDEX IF NOT EXISTS trade_cycles_strategy_created_idx ON trade_cycles(strategy, created_at DESC);
CREATE INDEX IF NOT EXISTS trade_cycles_status_updated_idx ON trade_cycles(status, updated_at DESC);
CREATE INDEX IF NOT EXISTS trade_cycles_metadata_idx ON trade_cycles USING GIN (metadata);
CREATE INDEX IF NOT EXISTS trade_cycles_profit_status_idx ON trade_cycles(profit_loss_percent, status) WHERE status = 'completed';
CREATE INDEX IF NOT EXISTS trade_cycles_active_idx ON trade_cycles(status, symbol, updated_at) WHERE status NOT IN ('completed', 'canceled', 'failed');

-- Index pour trading_signals
CREATE INDEX IF NOT EXISTS trading_signals_strategy_idx ON trading_signals(strategy);
CREATE INDEX IF NOT EXISTS trading_signals_symbol_idx ON trading_signals(symbol);
CREATE INDEX IF NOT EXISTS trading_signals_timestamp_idx ON trading_signals(timestamp);
CREATE INDEX IF NOT EXISTS trading_signals_processed_idx ON trading_signals(processed);
CREATE INDEX IF NOT EXISTS trading_signals_metadata_idx ON trading_signals USING GIN (metadata);

-- Index pour telegram_signals
CREATE INDEX IF NOT EXISTS telegram_signals_symbol_idx ON telegram_signals(symbol);
CREATE INDEX IF NOT EXISTS telegram_signals_timestamp_idx ON telegram_signals(timestamp DESC);
CREATE INDEX IF NOT EXISTS telegram_signals_symbol_timestamp_idx ON telegram_signals(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS telegram_signals_side_idx ON telegram_signals(side);
CREATE INDEX IF NOT EXISTS telegram_signals_action_idx ON telegram_signals(action);
CREATE INDEX IF NOT EXISTS telegram_signals_metadata_idx ON telegram_signals USING GIN (metadata);
CREATE INDEX IF NOT EXISTS telegram_signals_performance_idx ON telegram_signals(symbol, timestamp DESC, side);

-- Index pour strategy_configs
CREATE INDEX IF NOT EXISTS strategy_configs_symbol_idx ON strategy_configs(symbol);
CREATE INDEX IF NOT EXISTS strategy_configs_active_idx ON strategy_configs(active);

-- Index pour positions
CREATE INDEX IF NOT EXISTS positions_symbol_idx ON positions(symbol);
CREATE INDEX IF NOT EXISTS positions_status_idx ON positions(status);
CREATE INDEX IF NOT EXISTS positions_symbol_status_idx ON positions(symbol, status);

-- Index pour portfolio_balances
CREATE INDEX IF NOT EXISTS portfolio_balances_asset_idx ON portfolio_balances(asset);
CREATE INDEX IF NOT EXISTS portfolio_balances_timestamp_idx ON portfolio_balances(timestamp DESC);
CREATE INDEX IF NOT EXISTS portfolio_balances_asset_timestamp_idx ON portfolio_balances(asset, timestamp DESC);
CREATE INDEX IF NOT EXISTS portfolio_balances_total_idx ON portfolio_balances(total) WHERE total > 0;

-- =====================================================
-- COMMENTAIRES POUR DOCUMENTATION
-- =====================================================

COMMENT ON TABLE market_data IS 'Données OHLCV brutes reçues du Gateway - aucun indicateur calculé';
COMMENT ON TABLE analyzer_data IS 'Données d''analyse avancée calculées par le Market Analyzer avec tous les indicateurs et détections';
COMMENT ON COLUMN analyzer_data.market_regime IS 'Régime de marché détecté par l''analyzer';
COMMENT ON COLUMN analyzer_data.support_levels IS 'Niveaux de support sous format JSON: [{"price": 50000, "strength": "STRONG", "touches": 3}]';
COMMENT ON COLUMN analyzer_data.resistance_levels IS 'Niveaux de résistance sous format JSON';
COMMENT ON COLUMN analyzer_data.confluence_score IS 'Score de confluence entre différents indicateurs (0-100%)';

COMMENT ON TABLE trade_executions IS 'Exécutions d''ordres avec métadonnées complètes';
COMMENT ON TABLE trade_cycles IS 'Cycles de trading complets avec gestion des statuts';
COMMENT ON TABLE trading_signals IS 'Signaux générés par les stratégies d''analyse';
COMMENT ON TABLE telegram_signals IS 'Signaux de trading envoyés via Telegram avec métadonnées complètes pour affichage sur les graphiques';
COMMENT ON TABLE strategy_configs IS 'Configuration des stratégies de trading';
COMMENT ON TABLE positions IS 'Suivi des positions ouvertes et fermées';
COMMENT ON TABLE portfolio_balances IS 'Historique des balances du portefeuille par asset';

-- =====================================================
-- TRIGGERS POUR UPDATED_AT
-- =====================================================

-- Fonction pour mettre à jour updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers pour les tables avec updated_at
CREATE TRIGGER update_market_data_updated_at BEFORE UPDATE ON market_data FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_trade_executions_updated_at BEFORE UPDATE ON trade_executions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_trade_cycles_updated_at BEFORE UPDATE ON trade_cycles FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_strategy_configs_updated_at BEFORE UPDATE ON strategy_configs FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- ANALYSE ET OPTIMISATION
-- =====================================================

-- Exécuter ANALYZE pour mettre à jour les statistiques
ANALYZE;

-- Message de confirmation
DO $$
BEGIN
    RAISE NOTICE 'Schéma RootTrading v2.0 créé avec succès !';
    RAISE NOTICE 'Architecture propre : Gateway (OHLCV bruts) + Market Analyzer (calculs séparés)';
    RAISE NOTICE 'Tables principales : market_data, analyzer_data, trade_cycles, trade_executions';
END $$;