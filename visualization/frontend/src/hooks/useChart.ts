import { useEffect, useCallback } from 'react';
import { useChartStore } from '@/stores/useChartStore';
import { useAllChartData } from './useApi';
import { useWebSocket } from './useWebSocket';
import { debounce } from '@/utils';
import type { TradingSymbol, TimeInterval } from '@/types';

// Fonction pour convertir l'intervalle en millisecondes
const getIntervalMs = (interval: TimeInterval): number => {
  switch (interval) {
    case '1m': return 60000;      // 1 minute
    case '5m': return 300000;     // 5 minutes
    case '15m': return 900000;    // 15 minutes
    case '30m': return 1800000;   // 30 minutes
    case '1h': return 3600000;    // 1 heure
    case '4h': return 14400000;   // 4 heures
    case '1d': return 86400000;   // 1 jour
    default: return 60000;        // Par défaut 1 minute
  }
};

interface UseChartOptions {
  autoUpdate?: boolean;
  updateInterval?: number;
  enableWebSocket?: boolean;
}

export function useChart(options: UseChartOptions = {}) {
  const { autoUpdate = true, updateInterval = 60000, enableWebSocket = true } = options;
  
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
    toggleSMA,
    toggleIndicator,
  } = useChartStore();
  
  const { refetch, data: chartData, loading: apiLoading, error: apiError } = useAllChartData(
    config.symbol,
    config.interval,
    config.limit
  );
  
  // Log API errors
  useEffect(() => {
    if (apiError) {
      console.error('API Error in useChart:', apiError);
    }
  }, [apiError]);
  
  // Update store when data is loaded
  useEffect(() => {
    if (chartData && !apiLoading) {
      console.log('Chart data loaded:', {
        marketData: chartData.marketData ? 'OK' : 'Missing',
        indicators: chartData.indicators ? 'OK' : 'Missing',
        signals: chartData.signals ? 'OK' : 'Missing'
      });
    }
  }, [chartData, apiLoading]);
  
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
  
  // Fonction de mise à jour avec debounce et limite basée sur l'intervalle
  const debouncedUpdate = useCallback(
    debounce(async () => {
      if (isUserInteracting || apiLoading) return;
      
      // Ne pas actualiser plus souvent que l'intervalle des données
      const now = Date.now();
      const intervalMs = getIntervalMs(config.interval);
      if (lastUpdate && (now - lastUpdate.getTime()) < intervalMs) {
        console.log('Skipping update - too frequent for interval:', config.interval);
        return;
      }
      
      console.log('Updating chart data for:', config.symbol, config.interval);
      setIsLoading(true);
      try {
        await refetch();
        setLastUpdate(new Date());
        console.log('Chart data updated successfully');
      } catch (error) {
        console.error('Error updating chart data:', error);
      } finally {
        setIsLoading(false);
      }
    }, 2000),
    [refetch, isUserInteracting, setIsLoading, setLastUpdate, config.interval, lastUpdate, apiLoading]
  );
  
  // Force update function with cancellation protection
  const forceUpdate = useCallback(async () => {
    if (isLoading || apiLoading) {
      console.log('Update already in progress, skipping...');
      return;
    }
    
    console.log('Force updating chart data for:', config.symbol, config.interval);
    setIsLoading(true);
    try {
      await refetch();
      setLastUpdate(new Date());
      console.log('Force update completed successfully');
    } catch (error) {
      console.error('Error forcing chart update:', error);
    } finally {
      setIsLoading(false);
    }
  }, [refetch, setIsLoading, setLastUpdate, isLoading, apiLoading, config.symbol, config.interval]);
  
  // Mise à jour automatique adaptée à l'intervalle des données
  useEffect(() => {
    if (!autoUpdate) return;
    
    // Adapter l'intervalle de mise à jour à l'intervalle des données
    const dataIntervalMs = getIntervalMs(config.interval);
    const actualUpdateInterval = Math.max(updateInterval, dataIntervalMs);
    
    // Log seulement si l'intervalle change réellement
    const intervalId = setInterval(() => {
      debouncedUpdate();
    }, actualUpdateInterval);
    
    return () => clearInterval(intervalId);
  }, [autoUpdate, updateInterval, debouncedUpdate, config.interval]);
  
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
  
  // Handlers pour les changements de configuration avec debounce
  const handleSymbolChange = useCallback((symbol: TradingSymbol) => {
    // Prevent multiple calls for the same symbol
    if (config.symbol === symbol) {
      return;
    }
    
    // Clear data immediately to avoid showing stale data
    useChartStore.setState({ 
      marketData: null, 
      signals: null, 
      indicators: null,
      isLoading: true 
    });
    updateSymbol(symbol);
    resetZoom();
    
    // Force immediate update instead of debounced to prevent stale data display
    forceUpdate();
  }, [updateSymbol, resetZoom, forceUpdate, config.symbol]);
  
  const handleIntervalChange = useCallback((interval: TimeInterval) => {
    // Prevent multiple calls for the same interval
    if (config.interval === interval) {
      return;
    }
    
    // Clear data immediately to avoid showing stale data
    useChartStore.setState({ 
      marketData: null, 
      signals: null, 
      indicators: null,
      isLoading: true 
    });
    updateIntervalConfig(interval);
    resetZoom();
    
    // Force immediate update instead of debounced to prevent stale data display
    forceUpdate();
  }, [updateIntervalConfig, resetZoom, forceUpdate, config.interval]);
  
  const handleSignalFilterChange = useCallback((filter: string) => {
    updateSignalFilter(filter);
    debouncedUpdate();
  }, [updateSignalFilter, debouncedUpdate]);
  
  const handlePeriodChange = useCallback((period: string) => {
    updatePeriod(period);
    debouncedUpdate();
  }, [updatePeriod, debouncedUpdate]);
  
  const handleEMAToggle = useCallback((type: 'ema7' | 'ema26' | 'ema99') => {
    toggleEMA(type);
    debouncedUpdate();
  }, [toggleEMA, debouncedUpdate]);
  
  const handleSMAToggle = useCallback((type: 'sma20' | 'sma50') => {
    toggleSMA(type);
    debouncedUpdate();
  }, [toggleSMA, debouncedUpdate]);
  
  const handleIndicatorToggle = useCallback((type: 'rsi' | 'macd' | 'bollinger' | 'stochastic' | 'adx' | 'volume_advanced' | 'regime_info') => {
    toggleIndicator(type);
    debouncedUpdate();
  }, [toggleIndicator, debouncedUpdate]);
  
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
    isLoading: isLoading || apiLoading,
    isUserInteracting,
    lastUpdate,
    marketData,
    signals,
    indicators,
    performanceData,
    apiError,
    
    // Actions
    handleSymbolChange,
    handleIntervalChange,
    handleSignalFilterChange,
    handlePeriodChange,
    handleEMAToggle,
    handleSMAToggle,
    handleIndicatorToggle,
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