import { useEffect, useCallback } from 'react';
import { useChartStore } from '@/stores/useChartStore';
import { useAllChartData } from './useApi';
import { useWebSocket } from './useWebSocket';
import { debounce } from '@/utils';
import type { TradingSymbol, TimeInterval } from '@/types';

interface UseChartOptions {
  autoUpdate?: boolean;
  updateInterval?: number;
  enableWebSocket?: boolean;
}

export function useChart(options: UseChartOptions = {}) {
  const { autoUpdate = true, updateInterval = 10000, enableWebSocket = true } = options;
  
  const {
    config,
    zoomState,
    isLoading,
    isUserInteracting,
    lastUpdate,
    marketData,
    signals,
    indicators,
    performanceData,
    setConfig,
    setZoomState,
    setIsLoading,
    setIsUserInteracting,
    setLastUpdate,
    resetZoom,
    updateSymbol,
    updateInterval: updateIntervalConfig,
    updateSignalFilter,
    updatePeriod,
    toggleEMA,
  } = useChartStore();
  
  const { refetch } = useAllChartData(
    config.symbol,
    config.interval,
    config.limit
  );
  
  const { subscribe, unsubscribe, isConnected } = useWebSocket({
    autoConnect: enableWebSocket,
    onMessage: (message) => {
      if (message.type === 'update') {
        // Traiter les mises à jour en temps réel
        console.log('Real-time update:', message.data);
        // Ici on pourrait mettre à jour les données sans refetch complet
      }
    },
  });
  
  // Fonction de mise à jour avec debounce
  const debouncedUpdate = useCallback(
    debounce(async () => {
      if (isUserInteracting) return;
      
      setIsLoading(true);
      try {
        await refetch();
        setLastUpdate(new Date());
      } catch (error) {
        console.error('Error updating chart data:', error);
      } finally {
        setIsLoading(false);
      }
    }, 1000),
    [refetch, isUserInteracting, setIsLoading, setLastUpdate]
  );
  
  // Force update function - defined before handlers
  const forceUpdate = useCallback(async () => {
    setIsLoading(true);
    try {
      await refetch();
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Error forcing chart update:', error);
    } finally {
      setIsLoading(false);
    }
  }, [refetch, setIsLoading, setLastUpdate]);
  
  // Mise à jour automatique
  useEffect(() => {
    if (!autoUpdate) return;
    
    const interval = setInterval(() => {
      debouncedUpdate();
    }, updateInterval);
    
    return () => clearInterval(interval);
  }, [autoUpdate, updateInterval, debouncedUpdate]);
  
  // Abonnement WebSocket
  useEffect(() => {
    if (enableWebSocket && isConnected) {
      const channel = `market:${config.symbol}:${config.interval}`;
      subscribe(channel);
      
      return () => {
        unsubscribe(channel);
      };
    }
  }, [config.symbol, config.interval, enableWebSocket, isConnected, subscribe, unsubscribe]);
  
  // Handlers pour les changements de configuration
  const handleSymbolChange = useCallback((symbol: TradingSymbol) => {
    // Clear data immediately to avoid showing stale data
    useChartStore.setState({ 
      marketData: null, 
      signals: null, 
      indicators: null,
      isLoading: true 
    });
    updateSymbol(symbol);
    resetZoom();
    // Force immediate update instead of debounced
    forceUpdate();
  }, [updateSymbol, resetZoom, forceUpdate]);
  
  const handleIntervalChange = useCallback((interval: TimeInterval) => {
    // Clear data immediately to avoid showing stale data
    useChartStore.setState({ 
      marketData: null, 
      signals: null, 
      indicators: null,
      isLoading: true 
    });
    updateIntervalConfig(interval);
    resetZoom();
    // Force immediate update instead of debounced
    forceUpdate();
  }, [updateIntervalConfig, resetZoom, forceUpdate]);
  
  const handleSignalFilterChange = useCallback((filter: string) => {
    updateSignalFilter(filter);
    debouncedUpdate();
  }, [updateSignalFilter, debouncedUpdate]);
  
  const handlePeriodChange = useCallback((period: string) => {
    updatePeriod(period);
    debouncedUpdate();
  }, [updatePeriod, debouncedUpdate]);
  
  const handleEMAToggle = useCallback((type: 'ema12' | 'ema26' | 'ema50') => {
    toggleEMA(type);
    debouncedUpdate();
  }, [toggleEMA, debouncedUpdate]);
  
  const handleZoomChange = useCallback((xRange: [number, number] | null, yRange: [number, number] | null) => {
    setZoomState({ xRange, yRange });
  }, [setZoomState]);
  
  const handleUserInteraction = useCallback((interacting: boolean) => {
    setIsUserInteracting(interacting);
  }, [setIsUserInteracting]);
  
  return {
    // État
    config,
    zoomState,
    isLoading,
    isUserInteracting,
    lastUpdate,
    marketData,
    signals,
    indicators,
    performanceData,
    
    // Actions
    handleSymbolChange,
    handleIntervalChange,
    handleSignalFilterChange,
    handlePeriodChange,
    handleEMAToggle,
    handleZoomChange,
    handleUserInteraction,
    forceUpdate,
    resetZoom,
    
    // WebSocket
    isConnected,
    
    // Utilitaires
    debouncedUpdate,
  };
}

export default useChart;