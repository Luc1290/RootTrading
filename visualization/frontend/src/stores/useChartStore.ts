import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type {
  ChartConfig,
  ZoomState,
  MarketData,
  TradingSignal,
  Indicators,
  PerformanceData,
  TimeInterval,
  TradingSymbol,
  SignalFilter,
  PerformancePeriod,
} from '@/types';

interface ChartStore {
  // Configuration
  config: ChartConfig;
  setConfig: (config: Partial<ChartConfig>) => void;
  
  // Données
  marketData: MarketData | null;
  signals: { buy: TradingSignal[]; sell: TradingSignal[] } | null;
  indicators: Indicators | null;
  performanceData: PerformanceData | null;
  
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
  
  // Dernière mise à jour
  lastUpdate: Date | null;
  setLastUpdate: (date: Date) => void;
  
  // Protection contre les refreshs multiples
  lastSymbolChange: string | null;
  lastIntervalChange: string | null;
  
  // Actions
  setMarketData: (data: MarketData) => void;
  setSignals: (signals: { buy: TradingSignal[]; sell: TradingSignal[] }) => void;
  setIndicators: (indicators: Indicators) => void;
  setPerformanceData: (data: PerformanceData) => void;
  
  // Helpers
  updateSymbol: (symbol: TradingSymbol) => void;
  updateInterval: (interval: TimeInterval) => void;
  updateSignalFilter: (filter: SignalFilter) => void;
  updatePeriod: (period: PerformancePeriod) => void;
  toggleEMA: (type: 'ema7' | 'ema26' | 'ema99') => void;
}

export const useChartStore = create<ChartStore>()(
  subscribeWithSelector((set, get) => ({
    // Configuration initiale
    config: {
      symbol: 'SOLUSDC',
      interval: '1m',
      limit: 2880,
      signalFilter: 'all',
      period: '24h',
      emaToggles: {
        ema7: true,
        ema26: true,
        ema99: true,
      },
    },
    
    // Données
    marketData: null,
    signals: null,
    indicators: null,
    performanceData: null,
    
    // État
    zoomState: {
      xRange: null,
      yRange: null,
    },
    isLoading: false,
    isUserInteracting: false,
    lastUpdate: null,
    lastSymbolChange: null,
    lastIntervalChange: null,
    
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
    
    setLastUpdate: (date) => set({ lastUpdate: date }),
    
    setMarketData: (data) => set({ marketData: data }),
    
    setSignals: (signals) => set({ signals }),
    
    setIndicators: (indicators) => set({ indicators }),
    
    setPerformanceData: (data) => set({ performanceData: data }),
    
    // Helpers
    updateSymbol: (symbol) => set((state) => {
      const currentTime = Date.now().toString();
      return {
        config: { ...state.config, symbol },
        zoomState: { xRange: null, yRange: null }, // Reset zoom sur changement symbole
        lastSymbolChange: currentTime,
      };
    }),
    
    updateInterval: (interval) => set((state) => {
      const currentTime = Date.now().toString();
      return {
        config: { ...state.config, interval },
        zoomState: { xRange: null, yRange: null }, // Reset zoom sur changement intervalle
        lastIntervalChange: currentTime,
      };
    }),
    
    updateSignalFilter: (signalFilter) => set((state) => ({
      config: { ...state.config, signalFilter },
    })),
    
    updatePeriod: (period) => set((state) => ({
      config: { ...state.config, period },
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
  })));