import { useState, useEffect, useCallback } from 'react';
import { apiService } from '@/services/api';
import { useChartStore } from '@/stores/useChartStore';
import type { TradingSymbol, TimeInterval, PerformancePeriod } from '@/types';

interface UseApiState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

interface UseApiOptions {
  immediate?: boolean;
  onSuccess?: (data: any) => void;
  onError?: (error: Error) => void;
}

function useApi<T>(
  apiCall: () => Promise<T>,
  deps: any[] = [],
  options: UseApiOptions = {}
): UseApiState<T> & { refetch: () => Promise<void> } {
  const [state, setState] = useState<UseApiState<T>>({
    data: null,
    loading: false,
    error: null,
  });
  
  const { immediate = true, onSuccess, onError } = options;
  
  const fetchData = useCallback(async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));
    
    try {
      const result = await apiCall();
      setState({ data: result, loading: false, error: null });
      onSuccess?.(result);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setState({ data: null, loading: false, error: errorMessage });
      onError?.(error as Error);
    }
  }, deps);
  
  useEffect(() => {
    if (immediate) {
      fetchData();
    }
  }, [fetchData, immediate]);
  
  return {
    ...state,
    refetch: fetchData,
  };
}

export function useMarketData(
  symbol: TradingSymbol,
  interval: TimeInterval,
  limit: number = 2880
) {
  const { setMarketData } = useChartStore();
  
  return useApi(
    () => apiService.getMarketData(symbol, interval, limit),
    [symbol, interval, limit],
    {
      onSuccess: (data) => setMarketData(data.data),
    }
  );
}

export function useTradingSignals(symbol: TradingSymbol) {
  const { setSignals } = useChartStore();
  
  return useApi(
    () => apiService.getTradingSignals(symbol),
    [symbol],
    {
      onSuccess: (data) => setSignals(data.signals),
    }
  );
}

export function useIndicators(
  symbol: TradingSymbol,
  indicators: string,
  interval: TimeInterval,
  limit: number = 2880
) {
  const { setIndicators } = useChartStore();
  
  return useApi(
    () => apiService.getIndicators(symbol, indicators, interval, limit),
    [symbol, indicators, interval, limit],
    {
      onSuccess: (data) => setIndicators(data.indicators || {}),
    }
  );
}

export function usePerformanceData(
  period: PerformancePeriod,
  metric: string = 'pnl'
) {
  const { setPerformanceData } = useChartStore();
  
  return useApi(
    () => apiService.getPerformanceData(period, metric),
    [period, metric],
    {
      onSuccess: (data) => setPerformanceData(data.data),
    }
  );
}

export function useAvailableSymbols() {
  return useApi(
    () => apiService.getAvailableSymbols(),
    [],
    { immediate: true }
  );
}

export function useAllChartData(
  symbol: TradingSymbol,
  interval: TimeInterval,
  limit: number = 2880
) {
  const { setMarketData, setIndicators, setSignals } = useChartStore();
  
  return useApi(
    () => apiService.getAllChartData(symbol, interval, limit),
    [symbol, interval, limit],
    {
      onSuccess: (data) => {
        setMarketData(data.marketData);
        setIndicators(data.indicators);
        setSignals(data.signals);
      },
    }
  );
}

export default useApi;