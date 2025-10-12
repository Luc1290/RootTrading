import React, { useEffect, useRef, useState } from 'react';
import { IChartApi, ISeriesApi, Time } from 'lightweight-charts';
import { useChartStore } from '@/stores/useChartStore';
import { apiService } from '@/services/api';
import type { TradingSignal } from '@/types';

interface SignalOverlayProps {
  chart: IChartApi | null;
  candlestickSeries: ISeriesApi<'Candlestick'> | null;
}

function SignalOverlay({ chart, candlestickSeries }: SignalOverlayProps) {
  const buyMarkersRef = useRef<any[]>([]);
  const sellMarkersRef = useRef<any[]>([]);
  const [telegramSignals, setTelegramSignals] = useState<any[]>([]);

  const { signals, config } = useChartStore();

  // Récupérer les signaux Telegram
  useEffect(() => {
    const fetchTelegramSignals = async () => {
      try {
        console.log('[SignalOverlay] Fetching Telegram signals for', config.symbol);
        const response = await apiService.getTelegramSignals(config.symbol, 24);
        console.log('[SignalOverlay] Telegram signals received:', response.signals);
        setTelegramSignals(response.signals || []);
      } catch (error) {
        console.error('[SignalOverlay] Error fetching Telegram signals:', error);
      }
    };

    if (config.symbol) {
      fetchTelegramSignals();
      // Rafraîchir toutes les 30 secondes
      const interval = setInterval(fetchTelegramSignals, 30000);
      return () => clearInterval(interval);
    }
  }, [config.symbol]);
  
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

    // Créer les marqueurs pour les signaux Telegram (étoiles dorées)
    const telegramMarkers = telegramSignals.map((signal: any) => {
      const marker = {
        time: Math.floor(new Date(signal.timestamp).getTime() / 1000) as Time,
        position: signal.side === 'BUY' ? 'belowBar' as const : 'aboveBar' as const,
        color: '#FFD700', // Or
        shape: 'circle' as const, // Cercle pour les distinguer
        text: `TG ${signal.action} (${signal.score})`,
        size: 3, // Plus grand pour être visible
      };
      console.log('[SignalOverlay] Created Telegram marker:', marker);
      return marker;
    });

    console.log('[SignalOverlay] Total markers - Buy:', buyMarkers.length, 'Sell:', sellMarkers.length, 'Telegram:', telegramMarkers.length);

    // Appliquer tous les marqueurs à la série candlestick
    const allMarkers = [...buyMarkers, ...sellMarkers, ...telegramMarkers].sort((a, b) => (a.time as number) - (b.time as number));
    console.log('[SignalOverlay] Setting', allMarkers.length, 'markers on chart');
    candlestickSeries.setMarkers(allMarkers);

    buyMarkersRef.current = buyMarkers;
    sellMarkersRef.current = sellMarkers;
  }, [signals, config.signalFilter, chart, candlestickSeries, telegramSignals]);
  
  return null; // Ce composant n'a pas de rendu visuel, il modifie juste le graphique
}

export default SignalOverlay;