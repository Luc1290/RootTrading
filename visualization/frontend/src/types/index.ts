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
  strength: number;
  type: 'buy' | 'sell';
}

export interface Indicators {
  rsi?: number[];
  macd?: number[];
  macd_signal?: number[];
  macd_histogram?: number[];
  ema_12?: number[];
  ema_26?: number[];
  ema_50?: number[];
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
    ema12: boolean;
    ema26: boolean;
    ema50: boolean;
  };
}

export interface ZoomState {
  xRange: [number, number] | null;
  yRange: [number, number] | null;
}

export type ChartType = 'market' | 'volume' | 'rsi' | 'macd' | 'performance';

export type WebSocketStatus = 'connected' | 'disconnected' | 'connecting' | 'error';

export type TimeInterval = '1m' | '5m' | '15m' | '30m' | '1h' | '4h' | '1d' | string;

export type TradingSymbol = 'BTCUSDC' | 'ETHUSDC' | 'SOLUSDC' | 'XRPUSDC' | string;

export type SignalFilter = 'all' | 'Aggregated_2' | 'Aggregated_3' | 'Aggregated_4' | 'Aggregated_2,Aggregated_3,Aggregated_4' | string;

export type PerformancePeriod = '1h' | '24h' | '7d' | '30d' | string;