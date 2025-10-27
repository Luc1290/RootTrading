import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi, CandlestickData, LineData, Time } from 'lightweight-charts';
import { useChartStore } from '@/stores/useChartStore';
import { formatNumber } from '@/utils';

interface MarketChartProps {
  height?: number;
  useStore?: any; // Store custom optionnel
}

function MarketChart({ height = 750, useStore }: MarketChartProps) {
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
    isHoveringTooltip: boolean;
    isPinned: boolean;
  }>({
    visible: false,
    x: 0,
    y: 0,
    signal: null,
    isHoveringTooltip: false,
    isPinned: false,
  });
  const signalSeriesRef = useRef<{
    buy?: ISeriesApi<'Line'>;
    sell?: ISeriesApi<'Line'>;
  }>({});
  const tooltipTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Utiliser le store custom si fourni, sinon le store global
  const defaultStore = useChartStore();
  const store = useStore ? useStore() : defaultStore;
  const { marketData, signals, indicators, config, zoomState, setZoomState, setIsUserInteracting } = store;
  
  // Cycles désactivés pour performance
  // const [tradeCycles, setTradeCycles] = useState<TradeCycle[]>([]);
  // const cycleSeriesRef = useRef<(ISeriesApi<'Line'> | ISeriesApi<'Area'>)[]>([]);


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
        visible: false,
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
      if (tooltipTimeoutRef.current) {
        clearTimeout(tooltipTimeoutRef.current);
      }
      chart.remove();
    };
  }, [height]);
  
  // Cycles désactivés pour performance
  // useEffect(() => {
  //   const loadCycles = async () => {
  //     try {
  //       const response = await apiService.getTradeCycles(config.symbol);
  //       setTradeCycles(response.cycles);
  //     } catch (error) {
  //       console.error('Error loading trade cycles:', error);
  //     }
  //   };
  //
  //   if (config.symbol) {
  //     loadCycles();
  //   }
  // }, [config.symbol]);
  
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
      
      // Cycles désactivés
      // cycleSeriesRef.current.forEach(series => {
      //   if (chartRef.current) {
      //     try {
      //       chartRef.current.removeSeries(series);
      //     } catch (e) {
      //       console.warn('Error removing cycle series:', e);
      //     }
      //   }
      // });
      // cycleSeriesRef.current = [];
      
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
    })).filter((item: CandlestickData) =>
      item.open != null &&
      item.high != null &&
      item.low != null &&
      item.close != null &&
      !isNaN(item.open as number) &&
      !isNaN(item.high as number) &&
      !isNaN(item.low as number) &&
      !isNaN(item.close as number)
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
        priceScaleId: 'right',
        lineStyle: 0, // Solid line
        priceLineVisible: false, // Hide price line
      });
      
      const ema7Data: LineData[] = marketData.timestamps.map((timestamp: string, index: number) => ({
        time: Math.floor(new Date(timestamp).getTime() / 1000) as Time,
        value: indicators.ema_7![index],
      })).filter((item: LineData) => item.value !== null && item.value !== undefined) as LineData[];

      ema7Series.setData(ema7Data);
      emaSeriesRef.current.ema7 = ema7Series;
    }

    // EMA 26
    if (config.emaToggles.ema26 && indicators.ema_26 && marketData?.timestamps) {
      const ema26Series = chartRef.current.addLineSeries({
        color: '#f7931e',
        lineWidth: 2,
        title: '',
        priceScaleId: 'right',
        lineStyle: 0, // Solid line
        priceLineVisible: false, // Hide price line
      });

      const ema26Data: LineData[] = marketData.timestamps.map((timestamp: string, index: number) => ({
        time: Math.floor(new Date(timestamp).getTime() / 1000) as Time,
        value: indicators.ema_26![index],
      })).filter((item: LineData) => item.value !== null && item.value !== undefined) as LineData[];

      ema26Series.setData(ema26Data);
      emaSeriesRef.current.ema26 = ema26Series;
    }

    // EMA 99
    if (config.emaToggles.ema99 && indicators.ema_99 && marketData?.timestamps) {
      const ema99Series = chartRef.current.addLineSeries({
        color: '#ffd700',
        lineWidth: 2,
        title: '',
        priceScaleId: 'right',
        lineStyle: 0, // Solid line
        priceLineVisible: false, // Hide price line
      });

      const ema99Data: LineData[] = marketData.timestamps.map((timestamp: string, index: number) => ({
        time: Math.floor(new Date(timestamp).getTime() / 1000) as Time,
        value: indicators.ema_99![index],
      })).filter((item: LineData) => item.value !== null && item.value !== undefined) as LineData[];
      
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
      size: 1,
    }));
    
    // Créer les marqueurs pour les signaux de vente
    const sellMarkers = filteredSellSignals.map((signal: any) => ({
      time: new Date(signal.timestamp).getTime() / 1000,
      position: 'aboveBar' as const,
      color: '#ff4444',
      shape: 'arrowDown' as const,
      text: `SELL`,
      size: 1,
    }));

    console.log('[MarketChart] Total markers - Buy:', buyMarkers.length, 'Sell:', sellMarkers.length);

    // Créer un mapping des signaux par timestamp pour la recherche rapide
    // Utiliser un tableau pour chaque timestamp pour gérer les signaux multiples
    const signalMap = new Map();
    filteredBuySignals.forEach((signal: any) => {
      const time = Math.floor(new Date(signal.timestamp).getTime() / 1000);
      if (!signalMap.has(time)) {
        signalMap.set(time, []);
      }
      signalMap.get(time).push({ ...signal, type: 'buy' });
    });
    filteredSellSignals.forEach((signal: any) => {
      const time = Math.floor(new Date(signal.timestamp).getTime() / 1000);
      if (!signalMap.has(time)) {
        signalMap.set(time, []);
      }
      signalMap.get(time).push({ ...signal, type: 'sell' });
    });

    // Log removed for performance

    // Appliquer tous les marqueurs à la série candlestick
    const allMarkers = [...buyMarkers, ...sellMarkers].sort((a, b) => (a.time as number) - (b.time as number));
    console.log('[MarketChart] Setting', allMarkers.length, 'total markers on chart');
    candlestickSeriesRef.current.setMarkers(allMarkers as any);

    // Hover désactivé pour performance - seulement clic
    // chartRef.current.subscribeCrosshairMove((param) => { ... });

    // Ajouter la gestion du clic pour épingler le tooltip
    chartRef.current.subscribeClick((param) => {
      if (!param.time || !chartContainerRef.current) {
        // Clic ailleurs = désépingler
        setTooltip(prev => ({ ...prev, isPinned: false, visible: false }));
        return;
      }
      
      const timeValue = param.time as number;
      
      // Chercher s'il y a un signal à cet endroit
      let nearestSignals = null;
      let minDistance = Infinity;
      const tolerance = 30;
      
      for (const [signalTime, signalArray] of signalMap.entries()) {
        const distance = Math.abs(signalTime - timeValue);
        if (distance <= tolerance && distance < minDistance) {
          minDistance = distance;
          nearestSignals = signalArray;
        }
      }
      
      if (nearestSignals && param.point) {
        // Clic sur un signal = épingler le tooltip
        console.log('Pinning tooltip for signals:', nearestSignals);
        setTooltip({
          visible: true,
          x: param.point!.x,
          y: param.point!.y,
          signal: nearestSignals,
          isHoveringTooltip: false,
          isPinned: true, // Épingler !
        });
      } else {
        // Clic ailleurs = désépingler
        setTooltip(prev => ({ ...prev, isPinned: false, visible: false }));
      }
    });
  }, [signals, config.signalFilter]);
  
  // Cycles désactivés pour performance
  // useEffect(() => {
  //   if (!chartRef.current || !marketData || !marketData.timestamps || tradeCycles.length === 0) return;
  //   ...code désactivé...
  // }, [tradeCycles, marketData]);
  
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
      
      {/* Tooltip avec zone de connexion */}
      {tooltip.visible && tooltip.signal && (
        <>
          
          {/* Infobulle des signaux */}
          <div
            className="absolute bg-gray-900 border border-gray-600 rounded-lg p-3 shadow-xl text-white"
            style={{
              left: tooltip.x,
              top: tooltip.y - 10,
              transform: 'translate(-50%, -100%)',
              minWidth: '250px',
              maxWidth: '400px',
              maxHeight: '300px',
              overflowY: 'auto',
              pointerEvents: 'auto',
              zIndex: 1000,
              borderColor: tooltip.isPinned ? '#3b82f6' : '#4b5563', // Bordure bleue si épinglé
            }}
          >
          <div className="text-sm space-y-2">
            {/* En-tête avec bouton de fermeture */}
            <div className="flex justify-between items-center border-b border-gray-700 pb-1">
              {Array.isArray(tooltip.signal) ? (
                <div className="font-semibold text-blue-400">
                  📊 {tooltip.signal.length} SIGNAL{tooltip.signal.length > 1 ? 'S' : ''} SIMULTANÉ{tooltip.signal.length > 1 ? 'S' : ''}
                </div>
              ) : (
                <div className="font-semibold text-blue-400">
                  📊 SIGNAL
                </div>
              )}
              <div className="flex items-center space-x-2">
                {tooltip.isPinned && (
                  <span className="text-xs bg-blue-600 px-1.5 py-0.5 rounded text-white">📌 Épinglé</span>
                )}
                <button
                  onClick={() => setTooltip(prev => ({ ...prev, visible: false, isPinned: false }))}
                  className="text-gray-400 hover:text-white text-lg leading-none"
                  title="Fermer"
                >
                  ×
                </button>
              </div>
            </div>
            
            {Array.isArray(tooltip.signal) ? (
              <>
                {tooltip.signal.map((signal, index) => (
                  <div key={index} className="border-l-2 border-gray-600 pl-2 space-y-1">
                    <div className={`font-medium ${signal.type === 'buy' ? 'text-green-400' : 'text-red-400'}`}>
                      {signal.type === 'buy' ? '📈 ACHAT' : '📉 VENTE'} #{index + 1}
                    </div>
                    <div className="text-gray-300">
                      <span className="font-medium">Stratégie:</span> {signal.strategy || 'N/A'}
                    </div>
                    <div className="text-gray-300">
                      <span className="font-medium">Prix:</span> {signal.price ? `${parseFloat(signal.price).toFixed(4)}$` : 'N/A'}
                    </div>
                    {signal.strength && (
                      <div className="text-gray-300">
                        <span className="font-medium">Force:</span> {signal.strength}
                      </div>
                    )}
                    {index === 0 && (
                      <div className="text-gray-300">
                        <span className="font-medium">Heure:</span> {new Date(signal.timestamp).toLocaleString()}
                      </div>
                    )}
                    {signal.metadata && (
                      <div className="text-gray-300 text-xs">
                        <span className="font-medium">Métadonnées:</span>
                        <pre className="text-xs mt-1 text-gray-400 max-h-20 overflow-y-auto">
                          {JSON.stringify(signal.metadata, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                ))}
              </>
            ) : (
              // Fallback pour un seul signal (rétrocompatibilité)
              <>
                <div className={`font-semibold ${tooltip.signal.type === 'buy' ? 'text-green-400' : 'text-red-400'}`}>
                  {tooltip.signal.type === 'buy' ? '📈 SIGNAL D\'ACHAT' : '📉 SIGNAL DE VENTE'}
                </div>
                <div className="text-gray-300">
                  <span className="font-medium">Stratégie:</span> {tooltip.signal.strategy || 'N/A'}
                </div>
                <div className="text-gray-300">
                  <span className="font-medium">Prix:</span> {tooltip.signal.price ? `${parseFloat(tooltip.signal.price).toFixed(4)}$` : 'N/A'}
                </div>
                {tooltip.signal.strength && (
                  <div className="text-gray-300">
                    <span className="font-medium">Force:</span> {tooltip.signal.strength}
                  </div>
                )}
                <div className="text-gray-300">
                  <span className="font-medium">Heure:</span> {new Date(tooltip.signal.timestamp).toLocaleString()}
                </div>
                {tooltip.signal.metadata && (
                  <div className="text-gray-300 text-xs pt-1 border-t border-gray-700">
                    <span className="font-medium">Métadonnées:</span>
                    <pre className="text-xs mt-1 text-gray-400">
                      {JSON.stringify(tooltip.signal.metadata, null, 2)}
                    </pre>
                  </div>
                )}
              </>
            )}
          </div>
          </div>
        </>
      )}
      
      {/* Contrôles du graphique - repositionnés */}
      <div className="absolute top-2 right-2 z-10 flex items-center space-x-2">
        <button
          onClick={() => chartRef.current?.timeScale().fitContent()}
          className="bg-black/70 backdrop-blur-sm text-white px-2 py-1 rounded text-xs hover:bg-black/80 transition-colors"
          title="Ajuster à la taille"
        >
          📏
        </button>
        <button
          onClick={() => chartRef.current?.timeScale().resetTimeScale()}
          className="bg-black/70 backdrop-blur-sm text-white px-2 py-1 rounded text-xs hover:bg-black/80 transition-colors"
          title="Reset zoom"
        >
          🔄
        </button>
        <div className="bg-black/70 backdrop-blur-sm text-gray-300 px-2 py-1 rounded text-xs">
          💡 Cliquez flèche = épingler
        </div>
      </div>
      
    </div>
  );
}

export default MarketChart;