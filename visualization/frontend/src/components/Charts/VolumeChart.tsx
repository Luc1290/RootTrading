import React, { useEffect, useRef } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi, HistogramData, Time } from 'lightweight-charts';
import { useChartStore } from '@/stores/useChartStore';
import { formatVolume } from '@/utils';

interface VolumeChartProps {
  height?: number;
  useStore?: any;
}

function VolumeChart({ height = 150, useStore }: VolumeChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);

  const defaultStore = useChartStore();
  const store = useStore ? useStore() : defaultStore;
  const { marketData, zoomState, config } = store;
  
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
    if (!marketData?.timestamps || !marketData?.volume || !marketData?.close || !marketData?.open || !volumeSeriesRef.current) return;
    
    const volumeData: HistogramData[] = marketData.timestamps.map((timestamp: string, index: number) => {
      const isUp = marketData.close[index] >= marketData.open[index];
      return {
        time: Math.floor(new Date(timestamp).getTime() / 1000) as Time,
        value: marketData.volume[index],
        color: isUp ? '#26a69a' : '#ef5350',
      };
    }).filter((item: HistogramData) =>
      item.value != null &&
      !isNaN(item.value as number) &&
      (item.value as number) >= 0
    ) as HistogramData[];
    
    volumeSeriesRef.current.setData(volumeData);
  }, [marketData]);
  
  // Synchronisation du zoom avec le graphique principal
  useEffect(() => {
    if (!chartRef.current || !marketData?.timestamps || marketData.timestamps.length === 0) return;
    
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
        // Ignore zoom errors silently
      }
    }
  }, [zoomState.xRange, marketData]);
  
  const currentVolume = marketData?.volume && marketData.volume.length > 0 ? marketData.volume[marketData.volume.length - 1] : null;
  
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