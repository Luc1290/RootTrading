import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi, CandlestickData, LineData, Time } from 'lightweight-charts';
import { useChartStore } from '@/stores/useChartStore';
import { useChart } from '@/hooks/useChart';
import { formatNumber } from '@/utils';

interface MarketChartProps {
  height?: number;
}

function MarketChart({ height = 750 }: MarketChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const emaSeriesRef = useRef<{
    ema12?: ISeriesApi<'Line'>;
    ema26?: ISeriesApi<'Line'>;
    ema50?: ISeriesApi<'Line'>;
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
      
    } catch (error) {
      console.error('Error during chart cleanup:', error);
    }
    
  }, [config.symbol, config.interval]); // Se déclenche quand symbole OU interval change
  
  // Mise à jour des données candlestick
  useEffect(() => {
    if (!marketData || !candlestickSeriesRef.current) return;
    
    const candlestickData: CandlestickData[] = marketData.timestamps.map((timestamp: string, index: number) => ({
      time: Math.floor(new Date(timestamp).getTime() / 1000) as Time,
      open: marketData.open[index],
      high: marketData.high[index],
      low: marketData.low[index],
      close: marketData.close[index],
    }));
    
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
    
    // EMA 12
    if (config.emaToggles.ema12 && indicators.ema_12 && marketData) {
      const ema12Series = chartRef.current.addLineSeries({
        color: '#ff6b35',
        lineWidth: 2,
        title: '',
        priceScaleId: 'left',
        lineStyle: 0, // Solid line
        priceLineVisible: false, // Hide price line
      });
      
      const ema12Data: LineData[] = marketData.timestamps.map((timestamp: string, index: number) => ({
        time: Math.floor(new Date(timestamp).getTime() / 1000) as Time,
        value: indicators.ema_12![index],
      })).filter((item) => item.value !== null && item.value !== undefined) as LineData[];
      
      ema12Series.setData(ema12Data);
      emaSeriesRef.current.ema12 = ema12Series;
    }
    
    // EMA 26
    if (config.emaToggles.ema26 && indicators.ema_26 && marketData) {
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
    
    // EMA 50
    if (config.emaToggles.ema50 && indicators.ema_50 && marketData) {
      const ema50Series = chartRef.current.addLineSeries({
        color: '#ffd700',
        lineWidth: 2,
        title: '',
        priceScaleId: 'left',
        lineStyle: 0, // Solid line
        priceLineVisible: false, // Hide price line
      });
      
      const ema50Data: LineData[] = marketData.timestamps.map((timestamp: string, index: number) => ({
        time: Math.floor(new Date(timestamp).getTime() / 1000) as Time,
        value: indicators.ema_50![index],
      })).filter((item) => item.value !== null && item.value !== undefined) as LineData[];
      
      ema50Series.setData(ema50Data);
      emaSeriesRef.current.ema50 = ema50Series;
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
  
  // Application du zoom
  useEffect(() => {
    if (!chartRef.current || !zoomState.xRange) return;
    
    chartRef.current.timeScale().setVisibleRange({
      from: zoomState.xRange[0] as Time,
      to: zoomState.xRange[1] as Time,
    });
  }, [zoomState.xRange]);
  
  const currentPrice = marketData?.close[marketData.close.length - 1];
  
  return (
    <div className="relative">
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
    </div>
  );
}

export default MarketChart;