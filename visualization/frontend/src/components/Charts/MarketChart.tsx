import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi, CandlestickData, LineData, Time, LineStyle } from 'lightweight-charts';
import { useChartStore } from '@/stores/useChartStore';
import { useChart } from '@/hooks/useChart';
import { formatNumber } from '@/utils';
import { apiService } from '@/services/api';
import type { TradeCycle } from '@/components/Cycles/CyclesPage';

interface MarketChartProps {
  height?: number;
}

function MarketChart({ height = 750 }: MarketChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const emaSeriesRef = useRef<{
    ema7?: ISeriesApi<'Line'>;
    ema26?: ISeriesApi<'Line'>;
    ema99?: ISeriesApi<'Line'>;
  }>({});
  
  // État pour l'infobulle des signaux
  const [tooltip, setTooltip] = useState<{
    visible: boolean;
    x: number;
    y: number;
    signal: any;
  }>({
    visible: false,
    x: 0,
    y: 0,
    signal: null,
  });
  const signalSeriesRef = useRef<{
    buy?: ISeriesApi<'Line'>;
    sell?: ISeriesApi<'Line'>;
  }>({});
  
  const { marketData, signals, indicators, config, zoomState, setZoomState, setIsUserInteracting } = useChartStore();
  
  // État pour les cycles de trading
  const [tradeCycles, setTradeCycles] = useState<TradeCycle[]>([]);
  const cycleSeriesRef = useRef<(ISeriesApi<'Line'> | ISeriesApi<'Area'>)[]>([]);
  
  // Configuration de la précision basée sur le symbole
  const getPriceFormat = () => {
    const symbol = config.symbol;
    if (symbol.includes('PEPE') || symbol.includes('BONK')) {
      return { precision: 10, minMove: 0.0000000001 };
    } else if (symbol.includes('BTC')) {
      return { precision: 2, minMove: 0.01 };
    } else {
      return { precision: 6, minMove: 0.000001 };
    }
  };
  
  // Initialisation du graphique
  useEffect(() => {
    if (!chartContainerRef.current) return;
    
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: height,
      layout: {
        background: { type: ColorType.Solid, color: '#1a1a1a' },
        textColor: '#ffffff',
      },
      grid: {
        vertLines: { color: '#333333' },
        horzLines: { color: '#333333' },
      },
      crosshair: {
        mode: 1,
        vertLine: {
          color: '#888888',
          width: 1,
          style: 2,
        },
        horzLine: {
          color: '#888888',
          width: 1,
          style: 2,
        },
      },
      rightPriceScale: {
        borderColor: '#444444',
        textColor: '#ffffff',
      },
      leftPriceScale: {
        borderColor: '#444444',
        textColor: '#ffffff',
        visible: true,
      },
      timeScale: {
        borderColor: '#444444',
        timeVisible: true,
        secondsVisible: false,
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
      },
    });
    
    // Création de la série candlestick
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
      priceFormat: {
        type: 'price',
        ...getPriceFormat(),
      },
    });
    
    chartRef.current = chart;
    candlestickSeriesRef.current = candlestickSeries;
    
    // Gestion des événements de zoom
    chart.timeScale().subscribeVisibleTimeRangeChange((timeRange) => {
      if (timeRange) {
        setZoomState({ xRange: [timeRange.from as number, timeRange.to as number] });
      }
    });
    
    // Gestion des interactions utilisateur
    chartContainerRef.current.addEventListener('mouseenter', () => {
      setIsUserInteracting(true);
    });
    
    chartContainerRef.current.addEventListener('mouseleave', () => {
      setIsUserInteracting(false);
    });
    
    // Redimensionnement automatique
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    
    window.addEventListener('resize', handleResize);
    
    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [height]);
  
  // Récupérer les cycles de trading pour le symbole actuel
  useEffect(() => {
    const loadCycles = async () => {
      try {
        const response = await apiService.getTradeCycles(config.symbol);
        setTradeCycles(response.cycles);
      } catch (error) {
        console.error('Error loading trade cycles:', error);
      }
    };
    
    if (config.symbol) {
      loadCycles();
    }
  }, [config.symbol]);
  
  // Nettoyage complet lors du changement de symbole ou interval
  useEffect(() => {
    if (!chartRef.current || !candlestickSeriesRef.current) return;
    
    try {
      // Nettoyer toutes les données des séries existantes
      candlestickSeriesRef.current.setData([]);
      
      // Nettoyer les séries EMA
      Object.values(emaSeriesRef.current).forEach(series => {
        if (series && chartRef.current) {
          try {
            chartRef.current.removeSeries(series);
          } catch (e) {
            console.warn('Error removing EMA series:', e);
          }
        }
      });
      emaSeriesRef.current = {};
      
      // Nettoyer les séries de signaux
      Object.values(signalSeriesRef.current).forEach(series => {
        if (series && chartRef.current) {
          try {
            chartRef.current.removeSeries(series);
          } catch (e) {
            console.warn('Error removing signal series:', e);
          }
        }
      });
      signalSeriesRef.current = {};
      
      // Nettoyer les séries de cycles
      cycleSeriesRef.current.forEach(series => {
        if (chartRef.current) {
          try {
            chartRef.current.removeSeries(series);
          } catch (e) {
            console.warn('Error removing cycle series:', e);
          }
        }
      });
      cycleSeriesRef.current = [];
      
    } catch (error) {
      console.error('Error during chart cleanup:', error);
    }
    
  }, [config.symbol, config.interval]); // Se déclenche quand symbole OU interval change
  
  // Mise à jour des données candlestick
  useEffect(() => {
    if (!marketData?.timestamps || !candlestickSeriesRef.current) return;
    
    const candlestickData: CandlestickData[] = marketData.timestamps.map((timestamp: string, index: number) => ({
      time: Math.floor(new Date(timestamp).getTime() / 1000) as Time,
      open: marketData.open[index],
      high: marketData.high[index],
      low: marketData.low[index],
      close: marketData.close[index],
    })).filter((item) => 
      item.open != null && 
      item.high != null && 
      item.low != null && 
      item.close != null &&
      !isNaN(item.open) &&
      !isNaN(item.high) &&
      !isNaN(item.low) &&
      !isNaN(item.close)
    ) as CandlestickData[];
    
    candlestickSeriesRef.current.setData(candlestickData);
  }, [marketData]);
  
  // Mise à jour des indicateurs EMA
  useEffect(() => {
    if (!indicators || !chartRef.current) return;
    
    // Nettoyer les anciennes séries EMA
    Object.values(emaSeriesRef.current).forEach(series => {
      if (series) {
        chartRef.current?.removeSeries(series);
      }
    });
    emaSeriesRef.current = {};
    
    // EMA 7
    if (config.emaToggles.ema7 && indicators.ema_7 && marketData?.timestamps) {
      const ema7Series = chartRef.current.addLineSeries({
        color: '#ff6b35',
        lineWidth: 2,
        title: '',
        priceScaleId: 'left',
        lineStyle: 0, // Solid line
        priceLineVisible: false, // Hide price line
      });
      
      const ema7Data: LineData[] = marketData.timestamps.map((timestamp: string, index: number) => ({
        time: Math.floor(new Date(timestamp).getTime() / 1000) as Time,
        value: indicators.ema_7![index],
      })).filter((item) => item.value !== null && item.value !== undefined) as LineData[];
      
      ema7Series.setData(ema7Data);
      emaSeriesRef.current.ema7 = ema7Series;
    }
    
    // EMA 26
    if (config.emaToggles.ema26 && indicators.ema_26 && marketData?.timestamps) {
      const ema26Series = chartRef.current.addLineSeries({
        color: '#f7931e',
        lineWidth: 2,
        title: '',
        priceScaleId: 'left',
        lineStyle: 0, // Solid line
        priceLineVisible: false, // Hide price line
      });
      
      const ema26Data: LineData[] = marketData.timestamps.map((timestamp: string, index: number) => ({
        time: Math.floor(new Date(timestamp).getTime() / 1000) as Time,
        value: indicators.ema_26![index],
      })).filter((item) => item.value !== null && item.value !== undefined) as LineData[];
      
      ema26Series.setData(ema26Data);
      emaSeriesRef.current.ema26 = ema26Series;
    }
    
    // EMA 99
    if (config.emaToggles.ema99 && indicators.ema_99 && marketData?.timestamps) {
      const ema99Series = chartRef.current.addLineSeries({
        color: '#ffd700',
        lineWidth: 2,
        title: '',
        priceScaleId: 'left',
        lineStyle: 0, // Solid line
        priceLineVisible: false, // Hide price line
      });
      
      const ema99Data: LineData[] = marketData.timestamps.map((timestamp: string, index: number) => ({
        time: Math.floor(new Date(timestamp).getTime() / 1000) as Time,
        value: indicators.ema_99![index],
      })).filter((item) => item.value !== null && item.value !== undefined) as LineData[];
      
      ema99Series.setData(ema99Data);
      emaSeriesRef.current.ema99 = ema99Series;
    }
  }, [indicators, config.emaToggles, marketData]);
  
  // Mise à jour des signaux
  useEffect(() => {
    if (!signals || !chartRef.current || !candlestickSeriesRef.current) return;
    
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
      time: new Date(signal.timestamp).getTime() / 1000,
      position: 'belowBar' as const,
      color: '#00ff88',
      shape: 'arrowUp' as const,
      text: `BUY`,
      size: 2,
    }));
    
    // Créer les marqueurs pour les signaux de vente
    const sellMarkers = filteredSellSignals.map((signal: any) => ({
      time: new Date(signal.timestamp).getTime() / 1000,
      position: 'aboveBar' as const,
      color: '#ff4444',
      shape: 'arrowDown' as const,
      text: `SELL`,
      size: 2,
    }));
    
    // Créer un mapping des signaux par timestamp pour la recherche rapide
    const signalMap = new Map();
    filteredBuySignals.forEach(signal => {
      const time = new Date(signal.timestamp).getTime() / 1000;
      signalMap.set(time, { ...signal, type: 'buy' });
    });
    filteredSellSignals.forEach(signal => {
      const time = new Date(signal.timestamp).getTime() / 1000;
      signalMap.set(time, { ...signal, type: 'sell' });
    });

    // Appliquer tous les marqueurs à la série candlestick
    const allMarkers = [...buyMarkers, ...sellMarkers].sort((a, b) => (a.time as number) - (b.time as number));
    candlestickSeriesRef.current.setMarkers(allMarkers as any);
    
    // Utiliser l'API native pour détecter les survols
    chartRef.current.subscribeCrosshairMove((param) => {
      if (!param.time || !chartContainerRef.current) {
        setTooltip(prev => ({ ...prev, visible: false }));
        return;
      }
      
      const timeValue = param.time as number;
      const signal = signalMap.get(timeValue);
      
      if (signal && param.point) {
        const rect = chartContainerRef.current.getBoundingClientRect();
        setTooltip({
          visible: true,
          x: param.point.x,
          y: param.point.y,
          signal: signal,
        });
      } else {
        setTooltip(prev => ({ ...prev, visible: false }));
      }
    });
  }, [signals, config.signalFilter]);
  
  // Afficher les cycles de trading sur le graphique
  useEffect(() => {
    if (!chartRef.current || !marketData || !marketData.timestamps || tradeCycles.length === 0) return;
    
    // Nettoyer les anciennes séries de cycles
    cycleSeriesRef.current.forEach(series => {
      if (chartRef.current) {
        try {
          chartRef.current.removeSeries(series);
        } catch (e) {
          console.warn('Error removing cycle series:', e);
        }
      }
    });
    cycleSeriesRef.current = [];
    
    // Créer des zones colorées pour chaque cycle
    tradeCycles.forEach(cycle => {
      if (!cycle.entry_price || !chartRef.current) return;
      
      // Déterminer les timestamps de début et fin
      const startTime = Math.floor(new Date(cycle.created_at).getTime() / 1000) as Time;
      let endTime: Time;
      
      if (cycle.completed_at) {
        endTime = Math.floor(new Date(cycle.completed_at).getTime() / 1000) as Time;
      } else {
        // Si le cycle est actif, étendre jusqu'au dernier timestamp disponible
        endTime = Math.floor(new Date(marketData.timestamps[marketData.timestamps.length - 1]).getTime() / 1000) as Time;
      }
      
      // Créer une série de lignes pour marquer la période
      const topLineSeries = chartRef.current.addLineSeries({
        color: cycle.status === 'completed' 
          ? (cycle.profit_loss && cycle.profit_loss > 0 ? 'rgba(0, 255, 136, 0.6)' : 'rgba(255, 68, 68, 0.6)')
          : 'rgba(33, 150, 243, 0.6)',
        lineWidth: 2,
        lineStyle: LineStyle.Solid,
        priceScaleId: 'right',
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      });
      
      const bottomLineSeries = chartRef.current.addLineSeries({
        color: cycle.status === 'completed' 
          ? (cycle.profit_loss && cycle.profit_loss > 0 ? 'rgba(0, 255, 136, 0.6)' : 'rgba(255, 68, 68, 0.6)')
          : 'rgba(33, 150, 243, 0.6)',
        lineWidth: 2,
        lineStyle: LineStyle.Solid,
        priceScaleId: 'right',
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      });
      
      // Trouver les prix min et max dans cette période
      const pricesInPeriod = marketData.timestamps
        .map((ts, idx) => {
          const time = Math.floor(new Date(ts).getTime() / 1000);
          const startTimeNum = startTime as number;
          const endTimeNum = endTime as number;
          if (time >= startTimeNum && time <= endTimeNum) {
            return {
              high: marketData.high[idx],
              low: marketData.low[idx]
            };
          }
          return null;
        })
        .filter(p => p !== null);
      
      if (pricesInPeriod.length > 0) {
        const maxPrice = Math.max(...pricesInPeriod.map(p => p!.high));
        const minPrice = Math.min(...pricesInPeriod.map(p => p!.low));
        const padding = (maxPrice - minPrice) * 0.1;
        
        // Créer les lignes supérieures et inférieures
        topLineSeries.setData([
          { time: startTime, value: maxPrice + padding },
          { time: endTime, value: maxPrice + padding }
        ]);
        
        bottomLineSeries.setData([
          { time: startTime, value: minPrice - padding },
          { time: endTime, value: minPrice - padding }
        ]);
        
        // Créer une zone remplie entre les deux lignes
        if (chartRef.current) {
          const fillSeries = chartRef.current.addAreaSeries({
            topColor: cycle.status === 'completed' 
              ? (cycle.profit_loss && cycle.profit_loss > 0 ? 'rgba(0, 255, 136, 0.25)' : 'rgba(255, 68, 68, 0.25)')
              : 'rgba(33, 150, 243, 0.25)',
            bottomColor: cycle.status === 'completed' 
              ? (cycle.profit_loss && cycle.profit_loss > 0 ? 'rgba(0, 255, 136, 0.05)' : 'rgba(255, 68, 68, 0.05)')
              : 'rgba(33, 150, 243, 0.05)',
            lineColor: cycle.status === 'completed' 
              ? (cycle.profit_loss && cycle.profit_loss > 0 ? 'rgba(0, 255, 136, 0.7)' : 'rgba(255, 68, 68, 0.7)')
              : 'rgba(33, 150, 243, 0.7)',
            lineWidth: 2,
            lineStyle: LineStyle.Solid,
            priceScaleId: 'right',
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
          });
          
          // Données pour remplir la zone
          const areaData = [];
          const startTimeNum = startTime as number;
          const endTimeNum = endTime as number;
          for (let i = 0; i < marketData.timestamps.length; i++) {
            const time = Math.floor(new Date(marketData.timestamps[i]).getTime() / 1000);
            if (time >= startTimeNum && time <= endTimeNum) {
              areaData.push({
                time: time as Time,
                value: marketData.high[i] * 1.01 // Légèrement au-dessus du prix
              });
            }
          }
          
          if (areaData.length > 0) {
            fillSeries.setData(areaData);
            cycleSeriesRef.current.push(fillSeries);
          }
        }
        
        cycleSeriesRef.current.push(topLineSeries, bottomLineSeries);
      }
    });
  }, [tradeCycles, marketData]);
  
  // Application du zoom
  useEffect(() => {
    if (!chartRef.current || !marketData || !marketData.timestamps || marketData.timestamps.length === 0) return;
    
    // Si xRange est null, c'est un reset intentionnel - on utilise fitContent
    if (!zoomState.xRange) {
      try {
        chartRef.current.timeScale().fitContent();
      } catch (error) {
        console.warn('Error fitting content:', error);
      }
      return;
    }
    
    // Vérifier que les valeurs sont des nombres valides et que les données existent
    if (
      zoomState.xRange[0] != null && 
      zoomState.xRange[1] != null &&
      typeof zoomState.xRange[0] === 'number' &&
      typeof zoomState.xRange[1] === 'number' &&
      zoomState.xRange[0] < zoomState.xRange[1] &&
      !isNaN(zoomState.xRange[0]) &&
      !isNaN(zoomState.xRange[1])
    ) {
      try {
        chartRef.current.timeScale().setVisibleRange({
          from: zoomState.xRange[0] as Time,
          to: zoomState.xRange[1] as Time,
        });
      } catch (error) {
        console.warn('Error setting visible range:', error);
      }
    }
  }, [zoomState.xRange, marketData]);
  
  const currentPrice = marketData?.close?.[marketData.close.length - 1];
  
  return (
    <div className="relative market-chart-container">
      {/* Informations sur le prix actuel */}
      {currentPrice && (
        <div className="absolute top-2 left-2 z-10 bg-black/70 backdrop-blur-sm rounded-md px-3 py-1 text-sm">
          <span className="text-gray-300">Prix actuel: </span>
          <span className="text-white font-mono font-medium">
            {formatNumber(currentPrice, 4)} USDC
          </span>
        </div>
      )}
      
      {/* Conteneur du graphique */}
      <div
        ref={chartContainerRef}
        className="w-full"
        style={{ height: `${height}px` }}
      />
      
      {/* Infobulle des signaux */}
      {tooltip.visible && tooltip.signal && (
        <div
          className="absolute z-20 bg-dark-200 border border-gray-600 rounded-lg p-3 shadow-lg pointer-events-none"
          style={{
            left: tooltip.x,
            top: tooltip.y - 10,
            transform: 'translate(-50%, -100%)',
          }}
        >
          <div className="text-sm space-y-1">
            <div className={`font-semibold ${tooltip.signal.type === 'buy' ? 'text-green-400' : 'text-red-400'}`}>
              {tooltip.signal.type === 'buy' ? '📈 SIGNAL D\'ACHAT' : '📉 SIGNAL DE VENTE'}
            </div>
            <div className="text-gray-300">
              <span className="font-medium">Stratégie:</span> {tooltip.signal.strategy}
            </div>
            <div className="text-gray-300">
              <span className="font-medium">Prix:</span> {tooltip.signal.price}$
            </div>
            <div className="text-gray-300">
              <span className="font-medium">Force:</span> {tooltip.signal.strength}
            </div>
            <div className="text-gray-300">
              <span className="font-medium">Heure:</span> {new Date(tooltip.signal.timestamp).toLocaleTimeString()}
            </div>
          </div>
        </div>
      )}
      
      {/* Contrôles du graphique */}
      <div className="absolute bottom-2 left-2 z-10 flex items-center space-x-2">
        <button
          onClick={() => chartRef.current?.timeScale().fitContent()}
          className="bg-black/70 backdrop-blur-sm text-white px-3 py-1.5 rounded text-xs hover:bg-black/80 transition-colors"
          title="Ajuster à la taille"
        >
          Ajuster
        </button>
        <button
          onClick={() => chartRef.current?.timeScale().resetTimeScale()}
          className="bg-black/70 backdrop-blur-sm text-white px-3 py-1.5 rounded text-xs hover:bg-black/80 transition-colors"
          title="Reset zoom"
        >
          Reset
        </button>
      </div>
      
      {/* Légende des cycles */}
      <div className="absolute bottom-2 right-2 z-10 bg-black/70 backdrop-blur-sm rounded-md px-3 py-2 text-xs space-y-1">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-blue-500/30 rounded"></div>
          <span className="text-gray-300">Cycle actif</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-green-500/30 rounded"></div>
          <span className="text-gray-300">Cycle gagnant</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-red-500/30 rounded"></div>
          <span className="text-gray-300">Cycle perdant</span>
        </div>
      </div>
    </div>
  );
}

export default MarketChart;