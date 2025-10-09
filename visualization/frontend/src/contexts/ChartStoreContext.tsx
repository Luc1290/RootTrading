import React, { createContext, useContext, ReactNode } from 'react';
import { StoreApi, UseBoundStore } from 'zustand';
import { createChartStore } from '@/stores/useChartStoreFactory';

interface ChartStore {
  config: any;
  setConfig: (config: any) => void;
  marketData: any;
  signals: any;
  indicators: any;
  zoomState: any;
  setZoomState: (zoom: any) => void;
  resetZoom: () => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
  isUserInteracting: boolean;
  setIsUserInteracting: (interacting: boolean) => void;
  setMarketData: (data: any) => void;
  setSignals: (signals: any) => void;
  setIndicators: (indicators: any) => void;
  updateSymbol: (symbol: any) => void;
  updateInterval: (interval: any) => void;
  updateSignalFilter: (filter: any) => void;
  toggleEMA: (type: any) => void;
}

const ChartStoreContext = createContext<UseBoundStore<StoreApi<ChartStore>> | null>(null);

interface ChartStoreProviderProps {
  symbol: string;
  interval: string;
  children: ReactNode;
}

export function ChartStoreProvider({ symbol, interval, children }: ChartStoreProviderProps) {
  const store = createChartStore(symbol, interval);

  return (
    <ChartStoreContext.Provider value={store}>
      {children}
    </ChartStoreContext.Provider>
  );
}

export function useChartStoreContext() {
  const store = useContext(ChartStoreContext);
  if (!store) {
    throw new Error('useChartStoreContext must be used within ChartStoreProvider');
  }
  return store;
}
