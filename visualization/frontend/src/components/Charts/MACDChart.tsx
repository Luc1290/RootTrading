import React, { useEffect, useRef } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi, LineData, HistogramData, Time } from 'lightweight-charts';
import { useChartStore } from '@/stores/useChartStore';
import { formatNumber } from '@/utils';

interface MACDChartProps {
  height?: number;
}

function MACDChart({ height = 200 }: MACDChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const macdSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const signalSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const histogramSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const zeroLineRef = useRef<ISeriesApi<'Line'> | null>(null);
  
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
        visible: true,
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
    
    // Création de la série MACD
    const macdSeries = chart.addLineSeries({
      color: '#2196f3',
      lineWidth: 2,
      title: 'MACD',
      priceFormat: {
        type: 'price',
        precision: 6,
        minMove: 0.000001,
      },
    });
    
    // Création de la série Signal
    const signalSeries = chart.addLineSeries({
      color: '#ff5722',
      lineWidth: 2,
      title: 'Signal',
      priceFormat: {
        type: 'price',
        precision: 6,
        minMove: 0.000001,
      },
    });
    
    // Création de l'histogramme
    const histogramSeries = chart.addHistogramSeries({
      color: '#00ff00',
      priceFormat: {
        type: 'price',
        precision: 6,
        minMove: 0.000001,
      },
      priceScaleId: 'right',
    });
    
    // Ligne zéro
    const zeroLine = chart.addLineSeries({
      color: '#888888',
      lineWidth: 1,
      lineStyle: 2, // Dashed
      title: 'Zero',
      lastValueVisible: false,
      priceLineVisible: false,
    });
    
    chartRef.current = chart;
    macdSeriesRef.current = macdSeries;
    signalSeriesRef.current = signalSeries;
    histogramSeriesRef.current = histogramSeries;
    zeroLineRef.current = zeroLine;
    
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
    if (!macdSeriesRef.current || !signalSeriesRef.current || !histogramSeriesRef.current) return;
    
    try {
      // Nettoyer toutes les données
      macdSeriesRef.current.setData([]);
      signalSeriesRef.current.setData([]);
      histogramSeriesRef.current.setData([]);
      zeroLineRef.current?.setData([]);
    } catch (error) {
      console.error('Error during MACD cleanup:', error);
    }
    
  }, [config.symbol, config.interval]);
  
  // Mise à jour des données MACD
  useEffect(() => {
    if (!indicators || !marketData) return;
    
    const timeData = marketData.timestamps.map((timestamp: string) => ({
      time: Math.floor(new Date(timestamp).getTime() / 1000) as Time,
    }));
    
    // Ligne MACD
    if (indicators.macd && macdSeriesRef.current) {
      const macdData: LineData[] = marketData.timestamps.map((timestamp: string, index: number) => ({
        time: Math.floor(new Date(timestamp).getTime() / 1000) as Time,
        value: indicators.macd![index],
      })).filter((item) => item.value !== null && item.value !== undefined) as LineData[];
      
      macdSeriesRef.current.setData(macdData);
    }
    
    // Ligne Signal
    if (indicators.macd_signal && signalSeriesRef.current) {
      const signalData: LineData[] = marketData.timestamps.map((timestamp: string, index: number) => ({
        time: Math.floor(new Date(timestamp).getTime() / 1000) as Time,
        value: indicators.macd_signal![index],
      })).filter((item) => item.value !== null && item.value !== undefined) as LineData[];
      
      signalSeriesRef.current.setData(signalData);
    }
    
    // Histogramme
    if (indicators.macd_histogram && histogramSeriesRef.current) {
      const histogramData: HistogramData[] = marketData.timestamps.map((timestamp: string, index: number) => {
        const value = indicators.macd_histogram![index];
        return {
          time: Math.floor(new Date(timestamp).getTime() / 1000) as Time,
          value: value,
          color: value >= 0 ? '#00ff00' : '#ff0000',
        };
      }).filter((item) => item.value !== null && item.value !== undefined) as HistogramData[];
      
      histogramSeriesRef.current.setData(histogramData);
    }
    
    // Ligne zéro
    if (zeroLineRef.current) {
      const zeroData = timeData.map((item: { time: Time }) => ({ ...item, value: 0 }));
      zeroLineRef.current.setData(zeroData);
    }
  }, [indicators, marketData]);
  
  // Synchronisation du zoom avec le graphique principal
  useEffect(() => {
    if (!chartRef.current || !zoomState.xRange) return;
    
    chartRef.current.timeScale().setVisibleRange({
      from: zoomState.xRange[0] as Time,
      to: zoomState.xRange[1] as Time,
    });
  }, [zoomState.xRange]);
  
  const currentMACD = indicators?.macd?.[indicators.macd.length - 1];
  const currentSignal = indicators?.macd_signal?.[indicators.macd_signal.length - 1];
  const currentHistogram = indicators?.macd_histogram?.[indicators.macd_histogram.length - 1];
  
  const getMACDStatus = (macd: number, signal: number) => {
    if (macd > signal) return { text: 'Haussier', color: 'text-green-400' };
    return { text: 'Baissier', color: 'text-red-400' };
  };
  
  const macdStatus = currentMACD && currentSignal ? getMACDStatus(currentMACD, currentSignal) : null;
  
  return (
    <div className="relative">
      {/* Informations sur le MACD actuel */}
      {currentMACD && currentSignal && macdStatus && (
        <div className="absolute top-2 left-2 z-10 bg-black/70 backdrop-blur-sm rounded-md px-3 py-1 text-sm">
          <div className="flex items-center space-x-3">
            <span className="text-gray-300">MACD: </span>
            <span className="text-white font-mono font-medium">
              {formatNumber(currentMACD, 6)}
            </span>
            <span className="text-gray-300">Signal: </span>
            <span className="text-white font-mono font-medium">
              {formatNumber(currentSignal, 6)}
            </span>
            <span className={`${macdStatus.color} text-xs`}>
              ({macdStatus.text})
            </span>
          </div>
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

export default MACDChart;