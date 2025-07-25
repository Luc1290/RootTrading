export interface MarketData {
  timestamps: string[];
  open: number[];
  high: number[];
  low: number[];
  close: number[];
  volume: number[];
}

export interface TradingSignal {
  timestamp: string;
  price: number;
  strategy: string;
  strength?: number | string | null;
  type?: 'buy' | 'sell';
}

export interface Indicators {
  // RSI et momentum
  rsi_14?: number[];
  rsi_21?: number[];
  // EMAs
  ema_7?: number[];
  ema_12?: number[];
  ema_26?: number[];
  ema_50?: number[];
  ema_99?: number[];
  // SMAs
  sma_20?: number[];
  sma_50?: number[];
  // MACD
  macd_line?: number[];
  macd_signal?: number[];
  macd_histogram?: number[];
  // Bollinger Bands
  bb_upper?: number[];
  bb_middle?: number[];
  bb_lower?: number[];
  bb_position?: number[];
  bb_width?: number[];
  // Oscillateurs
  stoch_k?: number[];
  stoch_d?: number[];
  williams_r?: number[];
  cci_20?: number[];
  // Volatilité et tendance
  atr_14?: number[];
  adx_14?: number[];
  // Momentum
  momentum_10?: number[];
  roc_10?: number[];
  roc_20?: number[];
  // Volume
  obv?: number[];
  vwap_10?: number[];
  vwap_quote_10?: number[];
  volume_ratio?: number[];
  avg_volume_20?: number[];
  quote_volume_ratio?: number[];
  avg_trade_size?: number[];
  trade_intensity?: number[];
  // Régime et contexte
  market_regime?: string[];
  regime_strength?: string[];
  regime_confidence?: number[];
  volume_context?: string[];
  volume_pattern?: string[];
  pattern_detected?: string[];
  data_quality?: string[];
}

export interface ChartData {
  market_data: MarketData;
  indicators?: Indicators;
}

export interface PerformanceData {
  timestamps: string[];
  values: number[];
}

export interface WebSocketMessage {
  type: 'update' | 'error' | 'connected' | 'disconnected';
  data?: any;
  message?: string;
}

export interface ChartConfig {
  symbol: string;
  interval: string;
  limit: number;
  signalFilter: string;
  period: string;
  emaToggles: {
    ema7: boolean;
    ema12: boolean;
    ema26: boolean;
    ema50: boolean;
    ema99: boolean;
  };
  smaToggles: {
    sma20: boolean;
    sma50: boolean;
  };
  indicatorToggles: {
    rsi: boolean;
    macd: boolean;
    bollinger: boolean;
    stochastic: boolean;
    adx: boolean;
    volume_advanced: boolean;
    regime_info: boolean;
  };
}

export interface ZoomState {
  xRange: [number, number] | null;
  yRange: [number, number] | null;
}

export type ChartType = 'market' | 'volume' | 'rsi' | 'macd' | 'performance';

export type WebSocketStatus = 'connected' | 'disconnected' | 'connecting' | 'error';

export type TimeInterval = '1m' | '3m' | '5m' | '15m' | '30m' | '1d' | string;

export type TradingSymbol = 'BTCUSDC' | 'ETHUSDC' | 'SOLUSDC' | 'XRPUSDC' | string;

export type SignalFilter = 'all' | 'Aggregated_2' | 'Aggregated_3' | 'Aggregated_4' | 'Aggregated_2,Aggregated_3,Aggregated_4' | string;

export type PerformancePeriod = '24h' | '7d' | '30d' | string;