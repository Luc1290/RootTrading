import { create, StoreApi, UseBoundStore } from 'zustand';
import type {
  ChartConfig,
  ZoomState,
  MarketData,
  TradingSignal,
  Indicators,
  TimeInterval,
  TradingSymbol,
  SignalFilter,
} from '@/types';

interface ChartStore {
  // Configuration
  config: ChartConfig;
  setConfig: (config: Partial<ChartConfig>) => void;

  // Données
  marketData: MarketData | null;
  signals: { buy: TradingSignal[]; sell: TradingSignal[] } | null;
  indicators: Indicators | null;

  // État de zoom
  zoomState: ZoomState;
  setZoomState: (zoom: Partial<ZoomState>) => void;
  resetZoom: () => void;

  // État de chargement
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;

  // Interaction utilisateur
  isUserInteracting: boolean;
  setIsUserInteracting: (interacting: boolean) => void;

  // Actions
  setMarketData: (data: MarketData) => void;
  setSignals: (signals: { buy: TradingSignal[]; sell: TradingSignal[] }) => void;
  setIndicators: (indicators: Indicators) => void;

  // Helpers
  updateSymbol: (symbol: TradingSymbol) => void;
  updateInterval: (interval: TimeInterval) => void;
  updateSignalFilter: (filter: SignalFilter) => void;
  toggleEMA: (type: 'ema7' | 'ema12' | 'ema26' | 'ema50' | 'ema99') => void;
}

// Cache des stores par symbole
const storeCache = new Map<string, UseBoundStore<StoreApi<ChartStore>>>();

export function createChartStore(symbol: string, interval: string = '1m') {
  const storeKey = `${symbol}-${interval}`;

  // Retourner le store existant si déjà créé
  if (storeCache.has(storeKey)) {
    return storeCache.get(storeKey)!;
  }

  // Créer un nouveau store
  const store = create<ChartStore>()((set) => ({
    // Configuration initiale
    config: {
      symbol,
      interval,
      limit: 1000,
      signalFilter: 'all',
      period: '24h',
      emaToggles: {
        ema7: false,
        ema12: false,
        ema26: false,
        ema50: false,
        ema99: false,
      },
      smaToggles: {
        sma20: false,
        sma50: false,
      },
      indicatorToggles: {
        rsi: false,
        macd: false,
        bollinger: false,
        stochastic: false,
        adx: false,
        volume_advanced: false,
        regime_info: false,
      },
    },

    // Données
    marketData: null,
    signals: null,
    indicators: null,

    // État
    zoomState: {
      xRange: null,
      yRange: null,
    },
    isLoading: false,
    isUserInteracting: false,

    // Setters
    setConfig: (config) => set((state) => ({
      config: { ...state.config, ...config },
    })),

    setZoomState: (zoom) => set((state) => ({
      zoomState: { ...state.zoomState, ...zoom },
    })),

    resetZoom: () => set({
      zoomState: { xRange: null, yRange: null },
    }),

    setIsLoading: (loading) => set({ isLoading: loading }),

    setIsUserInteracting: (interacting) => set({ isUserInteracting: interacting }),

    setMarketData: (data) => set({ marketData: data }),

    setSignals: (signals) => set({ signals }),

    setIndicators: (indicators) => set({ indicators }),

    // Helpers
    updateSymbol: (symbol) => set((state) => ({
      config: { ...state.config, symbol },
      zoomState: { xRange: null, yRange: null },
    })),

    updateInterval: (interval) => set((state) => ({
      config: { ...state.config, interval },
      zoomState: { xRange: null, yRange: null },
    })),

    updateSignalFilter: (signalFilter) => set((state) => ({
      config: { ...state.config, signalFilter },
    })),

    toggleEMA: (type) => set((state) => ({
      config: {
        ...state.config,
        emaToggles: {
          ...state.config.emaToggles,
          [type]: !state.config.emaToggles[type],
        },
      },
    })),
  }));

  storeCache.set(storeKey, store);
  return store;
}

// Fonction pour nettoyer le cache si nécessaire
export function clearStoreCache() {
  storeCache.clear();
}
