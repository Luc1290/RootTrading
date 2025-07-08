import React, { useEffect, useRef } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi, AreaData, Time } from 'lightweight-charts';
import { useChartStore } from '@/stores/useChartStore';
import { formatCurrency, formatPercent } from '@/utils';

interface PerformanceChartProps {
  height?: number;
}

function PerformanceChart({ height = 300 }: PerformanceChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const performanceSeriesRef = useRef<ISeriesApi<'Area'> | null>(null);
  const zeroLineRef = useRef<ISeriesApi<'Line'> | null>(null);
  
  const { performanceData, config } = useChartStore();
  
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
        timeVisible: true,
        secondsVisible: false,
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
    
    // Création de la série de performance (aire)
    const performanceSeries = chart.addAreaSeries({
      lineColor: '#ffd700',
      lineWidth: 3,
      topColor: 'rgba(255, 215, 0, 0.4)',
      bottomColor: 'rgba(255, 215, 0, 0.0)',
      title: 'P&L',
      priceFormat: {
        type: 'price',
        precision: 2,
        minMove: 0.01,
      },
    });
    
    // Ligne zéro
    const zeroLine = chart.addLineSeries({
      color: '#888888',
      lineWidth: 1,
      lineStyle: 2, // Dashed
      title: 'Break-even',
      lastValueVisible: false,
      priceLineVisible: false,
    });
    
    chartRef.current = chart;
    performanceSeriesRef.current = performanceSeries;
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
  
  // Mise à jour des données de performance
  useEffect(() => {
    if (!performanceData || !performanceSeriesRef.current) return;
    
    const performanceAreaData = performanceData.timestamps.map((timestamp: string, index: number) => ({
      time: Math.floor(new Date(timestamp).getTime() / 1000) as Time,
      value: performanceData.values[index],
    })) as AreaData[];
    
    performanceSeriesRef.current.setData(performanceAreaData);
    
    // Ligne zéro
    if (zeroLineRef.current) {
      const zeroData = performanceData.timestamps.map((timestamp: string) => ({
        time: Math.floor(new Date(timestamp).getTime() / 1000) as Time,
        value: 0,
      }));
      zeroLineRef.current.setData(zeroData);
    }
  }, [performanceData]);
  
  // Calcul des statistiques de performance
  const getPerformanceStats = () => {
    if (!performanceData || performanceData.values.length === 0) return null;
    
    const values = performanceData.values;
    const currentValue = values[values.length - 1];
    const initialValue = values[0];
    const maxValue = Math.max(...values);
    const minValue = Math.min(...values);
    
    const totalReturn = currentValue - initialValue;
    const totalReturnPercent = initialValue !== 0 ? (totalReturn / Math.abs(initialValue)) * 100 : 0;
    
    return {
      current: currentValue,
      totalReturn,
      totalReturnPercent,
      maxValue,
      minValue,
      isPositive: totalReturn >= 0,
    };
  };
  
  const stats = getPerformanceStats();
  
  return (
    <div className="relative">
      {/* Statistiques de performance */}
      {stats && (
        <div className="absolute top-2 left-2 z-10 bg-black/70 backdrop-blur-sm rounded-md px-3 py-2 text-sm">
          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
            <div>
              <span className="text-gray-300">P&L Actuel: </span>
              <span className={`font-mono font-medium ${
                stats.isPositive ? 'text-green-400' : 'text-red-400'
              }`}>
                {formatCurrency(stats.current)}
              </span>
            </div>
            <div>
              <span className="text-gray-300">Total: </span>
              <span className={`font-mono font-medium ${
                stats.isPositive ? 'text-green-400' : 'text-red-400'
              }`}>
                {formatPercent(stats.totalReturnPercent)}
              </span>
            </div>
            <div>
              <span className="text-gray-300">Max: </span>
              <span className="text-white font-mono font-medium">
                {formatCurrency(stats.maxValue)}
              </span>
            </div>
            <div>
              <span className="text-gray-300">Min: </span>
              <span className="text-white font-mono font-medium">
                {formatCurrency(stats.minValue)}
              </span>
            </div>
          </div>
        </div>
      )}
      
      {/* Indicateur de période */}
      <div className="absolute top-2 right-2 z-10 bg-black/70 backdrop-blur-sm rounded-md px-3 py-1 text-sm">
        <span className="text-gray-300">Période: </span>
        <span className="text-white font-medium">
          {config.period === '1h' ? '1 Heure' :
           config.period === '24h' ? '24 Heures' :
           config.period === '7d' ? '7 Jours' :
           config.period === '30d' ? '30 Jours' : config.period}
        </span>
      </div>
      
      {/* Conteneur du graphique */}
      <div
        ref={chartContainerRef}
        className="w-full"
        style={{ height: `${height}px` }}
      />
    </div>
  );
}

export default PerformanceChart;