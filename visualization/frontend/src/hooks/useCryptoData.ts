import { useState, useEffect } from 'react';
import { apiService } from '@/services/api';
import type { TradingSymbol } from '@/types';

export function useCryptoData(symbol: TradingSymbol, interval: string = '1m', limit: number = 1000) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        const result = await apiService.getAllChartData(symbol, interval, limit);
        setData(result);
      } catch (error) {
        console.error(`Error loading ${symbol}:`, error);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [symbol, interval, limit]);

  return { data, loading };
}
