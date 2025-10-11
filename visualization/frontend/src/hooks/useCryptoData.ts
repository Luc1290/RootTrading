import { useState, useEffect } from 'react';
import { apiService } from '@/services/api';
import type { TradingSymbol } from '@/types';

export function useCryptoData(symbol: TradingSymbol, interval: string = '1m', limit: number = 1000) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;

    const loadData = async () => {
      try {
        setLoading(true);

        // Stratégie 2 temps : d'abord 200 bougies (rapide), puis le reste
        const quickLimit = Math.min(200, limit);

        // Premier chargement rapide
        const quickResult = await apiService.getAllChartData(symbol, interval, quickLimit);
        if (isMounted) {
          setData(quickResult);
          setLoading(false);
        }

        // Si on veut plus de données, charger en arrière-plan
        if (limit > quickLimit && isMounted) {
          const fullResult = await apiService.getAllChartData(symbol, interval, limit);
          if (isMounted) {
            setData(fullResult);
          }
        }
      } catch (error) {
        console.error(`Error loading ${symbol}:`, error);
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    loadData();

    return () => {
      isMounted = false;
    };
  }, [symbol, interval, limit]);

  return { data, loading };
}
