import React, { useEffect, useRef } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi, HistogramData, Time } from 'lightweight-charts';
import { useChartStore } from '@/stores/useChartStore';
import { formatVolume } from '@/utils';

interface VolumeChartProps {
  height?: number;
}

function VolumeChart({ height = 150 }: VolumeChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  
  const { marketData, zoomState, config } = useChartStore();
  
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
          visible: false,
        },
        vertLine: {
          color: '#888888',
          width: 1,
          style: 2,
        },
      },
    });
    
    // Création de la série histogramme pour le volume
    const volumeSeries = chart.addHistogramSeries({
      color: '#26a69a',
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: 'right',
    });
    
    chartRef.current = chart;
    volumeSeriesRef.current = volumeSeries;
    
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
    if (!volumeSeriesRef.current) return;
    
    try {
      // Nettoyer toutes les données de volume
      volumeSeriesRef.current.setData([]);
    } catch (error) {
      console.error('Error during Volume cleanup:', error);
    }
    
  }, [config.symbol, config.interval]);
  
  // Mise à jour des données de volume
  useEffect(() => {
    if (!marketData || !volumeSeriesRef.current) return;
    
    const volumeData: HistogramData[] = marketData.timestamps.map((timestamp: string, index: number) => {
      const isUp = marketData.close[index] >= marketData.open[index];
      return {
        time: Math.floor(new Date(timestamp).getTime() / 1000) as Time,
        value: marketData.volume[index],
        color: isUp ? '#26a69a' : '#ef5350',
      };
    });
    
    volumeSeriesRef.current.setData(volumeData);
  }, [marketData]);
  
  // Synchronisation du zoom avec le graphique principal
  useEffect(() => {
    if (!chartRef.current || !zoomState.xRange) return;
    
    // Vérifier que les valeurs ne sont pas null
    if (zoomState.xRange[0] && zoomState.xRange[1]) {
      try {
        chartRef.current.timeScale().setVisibleRange({
          from: zoomState.xRange[0] as Time,
          to: zoomState.xRange[1] as Time,
        });
      } catch (error) {
        console.warn('Error setting visible range:', error);
      }
    }
  }, [zoomState.xRange]);
  
  const currentVolume = marketData?.volume[marketData.volume.length - 1];
  
  return (
    <div className="relative">
      {/* Informations sur le volume actuel */}
      {currentVolume && (
        <div className="absolute top-2 left-2 z-10 bg-black/70 backdrop-blur-sm rounded-md px-3 py-1 text-sm">
          <span className="text-gray-300">Volume: </span>
          <span className="text-white font-mono font-medium">
            {formatVolume(currentVolume)}
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

export default VolumeChart;