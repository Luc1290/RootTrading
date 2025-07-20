import React, { useEffect, useRef } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi, LineData, Time } from 'lightweight-charts';
import { useChartStore } from '@/stores/useChartStore';
import { formatNumber } from '@/utils';

interface RSIChartProps {
  height?: number;
}

function RSIChart({ height = 200 }: RSIChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const rsiSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const overboughtLineRef = useRef<ISeriesApi<'Line'> | null>(null);
  const oversoldLineRef = useRef<ISeriesApi<'Line'> | null>(null);
  const neutralLineRef = useRef<ISeriesApi<'Line'> | null>(null);
  
  const { marketData, indicators, zoomState, config } = useChartStore();
  
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
      rightPriceScale: {
        borderColor: '#444444',
        textColor: '#ffffff',
        scaleMargins: {
          top: 0.1,
          bottom: 0.1,
        },
      },
      timeScale: {
        borderColor: '#444444',
        visible: false,
      },
      crosshair: {
        horzLine: {
          color: '#888888',
          width: 1,
          style: 2,
        },
        vertLine: {
          color: '#888888',
          width: 1,
          style: 2,
        },
      },
    });
    
    // Création de la série RSI
    const rsiSeries = chart.addLineSeries({
      color: '#ff9800',
      lineWidth: 2,
      title: 'RSI',
      priceFormat: {
        type: 'price',
        precision: 2,
        minMove: 0.01,
      },
    });
    
    // Lignes de référence
    const overboughtLine = chart.addLineSeries({
      color: '#ff0000',
      lineWidth: 1,
      lineStyle: 2, // Dashed
      title: 'Surachat (70)',
      lastValueVisible: false,
      priceLineVisible: false,
    });
    
    const oversoldLine = chart.addLineSeries({
      color: '#00ff00',
      lineWidth: 1,
      lineStyle: 2, // Dashed
      title: 'Survente (30)',
      lastValueVisible: false,
      priceLineVisible: false,
    });
    
    const neutralLine = chart.addLineSeries({
      color: '#888888',
      lineWidth: 1,
      lineStyle: 3, // Dotted
      title: 'Neutre (50)',
      lastValueVisible: false,
      priceLineVisible: false,
    });
    
    chartRef.current = chart;
    rsiSeriesRef.current = rsiSeries;
    overboughtLineRef.current = overboughtLine;
    oversoldLineRef.current = oversoldLine;
    neutralLineRef.current = neutralLine;
    
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
  
  // Nettoyage lors du changement de symbole ou interval
  useEffect(() => {
    if (!rsiSeriesRef.current) return;
    
    try {
      // Nettoyer toutes les données
      rsiSeriesRef.current.setData([]);
      overboughtLineRef.current?.setData([]);
      oversoldLineRef.current?.setData([]);
      neutralLineRef.current?.setData([]);
    } catch (error) {
      console.error('Error during RSI cleanup:', error);
    }
    
  }, [config.symbol, config.interval]);
  
  // Mise à jour des données RSI
  useEffect(() => {
    if (!indicators?.rsi || !marketData || !rsiSeriesRef.current) return;
    
    const rsiData = marketData.timestamps.map((timestamp: string, index: number) => ({
      time: Math.floor(new Date(timestamp).getTime() / 1000) as Time,
      value: indicators.rsi![index],
    })).filter((item) => item.value !== null && item.value !== undefined) as LineData[];
    
    rsiSeriesRef.current.setData(rsiData);
    
    // Mise à jour des lignes de référence
    if (overboughtLineRef.current && oversoldLineRef.current && neutralLineRef.current) {
      const referenceData = marketData.timestamps.map((timestamp: string) => ({
        time: Math.floor(new Date(timestamp).getTime() / 1000) as Time,
      }));
      
      overboughtLineRef.current.setData(referenceData.map((item: { time: Time }) => ({ ...item, value: 70 })));
      oversoldLineRef.current.setData(referenceData.map((item: { time: Time }) => ({ ...item, value: 30 })));
      neutralLineRef.current.setData(referenceData.map((item: { time: Time }) => ({ ...item, value: 50 })));
    }
  }, [indicators, marketData]);
  
  // Synchronisation du zoom avec le graphique principal
  useEffect(() => {
    if (!chartRef.current || !marketData || marketData.timestamps.length === 0) return;
    
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
  
  const currentRSI = indicators?.rsi?.[indicators.rsi.length - 1];
  
  const getRSIStatus = (rsi: number) => {
    if (rsi > 70) return { text: 'Surachat', color: 'text-red-400' };
    if (rsi < 30) return { text: 'Survente', color: 'text-green-400' };
    return { text: 'Neutre', color: 'text-gray-400' };
  };
  
  const rsiStatus = currentRSI ? getRSIStatus(currentRSI) : null;
  
  return (
    <div className="relative">
      {/* Informations sur le RSI actuel */}
      {currentRSI && rsiStatus && (
        <div className="absolute top-2 left-2 z-10 bg-black/70 backdrop-blur-sm rounded-md px-3 py-1 text-sm">
          <span className="text-gray-300">RSI: </span>
          <span className="text-white font-mono font-medium">
            {formatNumber(currentRSI, 2)}
          </span>
          <span className={`ml-2 ${rsiStatus.color} text-xs`}>
            ({rsiStatus.text})
          </span>
        </div>
      )}
      
      {/* Conteneur du graphique */}
      <div
        ref={chartContainerRef}
        className="w-full"
        style={{ height: `${height}px` }}
      />
    </div>
  );
}

export default RSIChart;