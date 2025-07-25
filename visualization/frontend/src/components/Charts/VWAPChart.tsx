import React, { useEffect, useRef } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi, LineData, Time } from 'lightweight-charts';
import { useChartStore } from '@/stores/useChartStore';

interface VWAPChartProps {
  height?: number;
}

function VWAPChart({ height = 300 }: VWAPChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const vwapSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const vwapQuoteSeriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  
  const { marketData, indicators, config } = useChartStore();
  
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
      },
      rightPriceScale: {
        borderColor: '#444444',
        textColor: '#ffffff',
      },
      timeScale: {
        borderColor: '#444444',
        timeVisible: true,
        secondsVisible: false,
      },
    });
    
    // Série VWAP standard
    const vwapSeries = chart.addLineSeries({
      color: '#2196F3',
      lineWidth: 2,
      title: 'VWAP',
      priceLineVisible: false,
    });
    
    // Série VWAP Quote (plus précis)
    const vwapQuoteSeries = chart.addLineSeries({
      color: '#4CAF50',
      lineWidth: 2,
      title: 'VWAP Quote',
      priceLineVisible: false,
    });
    
    chartRef.current = chart;
    vwapSeriesRef.current = vwapSeries;
    vwapQuoteSeriesRef.current = vwapQuoteSeries;
    
    return () => {
      chart.remove();
    };
  }, [height]);
  
  // Mise à jour des données VWAP
  useEffect(() => {
    if (!indicators || !marketData || !vwapSeriesRef.current || !vwapQuoteSeriesRef.current) return;
    
    // VWAP standard
    if (indicators.vwap_10) {
      const vwapData: LineData[] = marketData.timestamps.map((timestamp: string, index: number) => ({
        time: Math.floor(new Date(timestamp).getTime() / 1000) as Time,
        value: indicators.vwap_10![index],
      })).filter((item) => item.value !== null && item.value !== undefined && !isNaN(item.value)) as LineData[];
      
      vwapSeriesRef.current.setData(vwapData);
    }
    
    // VWAP Quote (plus précis)
    if (indicators.vwap_quote_10) {
      const vwapQuoteData: LineData[] = marketData.timestamps.map((timestamp: string, index: number) => ({
        time: Math.floor(new Date(timestamp).getTime() / 1000) as Time,
        value: indicators.vwap_quote_10![index],
      })).filter((item) => item.value !== null && item.value !== undefined && !isNaN(item.value)) as LineData[];
      
      vwapQuoteSeriesRef.current.setData(vwapQuoteData);
    }
  }, [indicators, marketData]);
  
  // Redimensionnement
  useEffect(() => {
    if (!chartRef.current || !chartContainerRef.current) return;
    
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };
    
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-lg font-semibold text-white">VWAP Analysis</h3>
        <div className="flex items-center space-x-4 text-sm">
          <div className="flex items-center">
            <div className="w-3 h-0.5 bg-blue-500 mr-2"></div>
            <span className="text-gray-300">VWAP Standard</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-0.5 bg-green-500 mr-2"></div>
            <span className="text-gray-300">VWAP Quote (précis)</span>
          </div>
        </div>
      </div>
      <div ref={chartContainerRef} className="w-full" />
      {indicators?.vwap_quote_10 && (
        <div className="mt-2 text-sm text-gray-400">
          <p>VWAP Quote utilise le volume en USDC pour une précision améliorée</p>
        </div>
      )}
    </div>
  );
}

export default VWAPChart;