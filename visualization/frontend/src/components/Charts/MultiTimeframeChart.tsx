import React, { useState, useEffect } from 'react';
import { apiService } from '@/services/api';
import { MarketData, TimeInterval, TradingSymbol } from '@/types';
import { useChartStore } from '@/stores/useChartStore';

interface TimeframeData {
  interval: TimeInterval;
  label: string;
  data: MarketData | null;
  loading: boolean;
  error: string | null;
}

function MultiTimeframeChart() {
  const { config } = useChartStore();
  const [timeframes, setTimeframes] = useState<TimeframeData[]>([
    { interval: '3m', label: '3M', data: null, loading: true, error: null },
    { interval: '5m', label: '5M', data: null, loading: true, error: null },
    { interval: '15m', label: '15M', data: null, loading: true, error: null },
  ]);

  useEffect(() => {
    fetchAllTimeframes();
    const interval = setInterval(fetchAllTimeframes, 60000);
    return () => clearInterval(interval);
  }, [config.symbol]);

  const fetchAllTimeframes = async () => {
    const promises = timeframes.map(async (tf) => {
      try {
        const response = await apiService.getMarketData(
          config.symbol as TradingSymbol,
          tf.interval,
          100
        );
        return {
          ...tf,
          data: response.data,
          loading: false,
          error: null,
        };
      } catch (error) {
        return {
          ...tf,
          data: null,
          loading: false,
          error: 'Erreur de chargement',
        };
      }
    });

    const results = await Promise.all(promises);
    setTimeframes(results);
  };

  const renderMiniChart = (tf: TimeframeData) => {
    if (tf.loading) {
      return (
        <div className="h-24 bg-dark-200 rounded flex items-center justify-center">
          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-500"></div>
        </div>
      );
    }

    if (tf.error || !tf.data) {
      return (
        <div className="h-24 bg-dark-200 rounded flex items-center justify-center">
          <span className="text-red-400 text-xs">Erreur</span>
        </div>
      );
    }

    // Calculer les données pour le mini chart
    const prices = tf.data.close || [];
    if (prices.length === 0) {
      return (
        <div className="h-24 bg-dark-200 rounded flex items-center justify-center">
          <span className="text-gray-400 text-xs">Pas de données</span>
        </div>
      );
    }

    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;
    
    const lastPrice = prices[prices.length - 1];
    const previousPrice = prices[prices.length - 2] || lastPrice;
    const change = lastPrice - previousPrice;
    const changePercent = previousPrice > 0 ? ((change / previousPrice) * 100) : 0;

    // Créer les points du graphique
    const points = prices.map((price, index) => {
      const x = (index / Math.max(prices.length - 1, 1)) * 100;
      const y = priceRange > 0 ? ((maxPrice - price) / priceRange) * 100 : 50;
      return `${x},${y}`;
    }).join(' ');

    return (
      <div className="h-24 bg-dark-200 rounded p-2 relative">
        {/* Prix et variation */}
        <div className="flex justify-between items-start text-xs mb-1">
          <span className="text-white font-medium">
            ${lastPrice.toFixed(2)}
          </span>
          <span className={`${change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {change >= 0 ? '+' : ''}{changePercent.toFixed(2)}%
          </span>
        </div>

        {/* Mini graphique */}
        <div className="relative h-12">
          <svg viewBox="0 0 100 100" className="w-full h-full">
            <polyline
              points={points}
              fill="none"
              stroke={change >= 0 ? '#10b981' : '#ef4444'}
              strokeWidth="2"
              opacity="0.8"
            />
            <defs>
              <linearGradient id={`gradient-${tf.interval}`} x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor={change >= 0 ? '#10b981' : '#ef4444'} stopOpacity="0.3"/>
                <stop offset="100%" stopColor={change >= 0 ? '#10b981' : '#ef4444'} stopOpacity="0"/>
              </linearGradient>
            </defs>
            <polyline
              points={`${points} 100,100 0,100`}
              fill={`url(#gradient-${tf.interval})`}
              stroke="none"
            />
          </svg>
        </div>
      </div>
    );
  };

  return (
    <div className="h-64 bg-dark-300 rounded-lg p-4">
      <div className="grid grid-cols-3 gap-4 h-full">
        {timeframes.map((tf) => (
          <div key={tf.interval} className="flex flex-col">
            <div className="text-sm font-medium text-gray-400 mb-2 text-center">
              {tf.label}
            </div>
            {renderMiniChart(tf)}
          </div>
        ))}
      </div>
    </div>
  );
}

export default MultiTimeframeChart;