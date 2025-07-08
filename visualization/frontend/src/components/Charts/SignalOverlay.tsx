import React, { useEffect, useRef } from 'react';
import { IChartApi, ISeriesApi, Time } from 'lightweight-charts';
import { useChartStore } from '@/stores/useChartStore';
import type { TradingSignal } from '@/types';

interface SignalOverlayProps {
  chart: IChartApi | null;
  candlestickSeries: ISeriesApi<'Candlestick'> | null;
}

function SignalOverlay({ chart, candlestickSeries }: SignalOverlayProps) {
  const buyMarkersRef = useRef<any[]>([]);
  const sellMarkersRef = useRef<any[]>([]);
  
  const { signals, config } = useChartStore();
  
  // Mise à jour des signaux
  useEffect(() => {
    if (!signals || !chart || !candlestickSeries) return;
    
    // Filtrer les signaux selon la configuration
    let filteredBuySignals = signals.buy;
    let filteredSellSignals = signals.sell;
    
    if (config.signalFilter !== 'all') {
      const allowedStrategies = config.signalFilter.split(',');
      filteredBuySignals = signals.buy.filter((s: any) => 
        allowedStrategies.some((strategy: string) => s.strategy.includes(strategy))
      );
      filteredSellSignals = signals.sell.filter((s: any) => 
        allowedStrategies.some((strategy: string) => s.strategy.includes(strategy))
      );
    }
    
    // Créer les marqueurs pour les signaux d'achat
    const buyMarkers = filteredBuySignals.map((signal: any) => ({
      time: Math.floor(new Date(signal.timestamp).getTime() / 1000) as Time,
      position: 'belowBar' as const,
      color: '#00ff88',
      shape: 'arrowUp' as const,
      text: `BUY ${signal.strategy}`,
      size: 2,
    }));
    
    // Créer les marqueurs pour les signaux de vente
    const sellMarkers = filteredSellSignals.map((signal: any) => ({
      time: Math.floor(new Date(signal.timestamp).getTime() / 1000) as Time,
      position: 'aboveBar' as const,
      color: '#ff4444',
      shape: 'arrowDown' as const,
      text: `SELL ${signal.strategy}`,
      size: 2,
    }));
    
    // Appliquer tous les marqueurs à la série candlestick
    const allMarkers = [...buyMarkers, ...sellMarkers].sort((a, b) => (a.time as number) - (b.time as number));
    candlestickSeries.setMarkers(allMarkers);
    
    buyMarkersRef.current = buyMarkers;
    sellMarkersRef.current = sellMarkers;
  }, [signals, config.signalFilter, chart, candlestickSeries]);
  
  return null; // Ce composant n'a pas de rendu visuel, il modifie juste le graphique
}

export default SignalOverlay;