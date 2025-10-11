import React, { memo, useEffect, useMemo } from 'react';
import { useCryptoData } from '@/hooks/useCryptoData';
import { formatNumber } from '@/utils';
import { createChartStore } from '@/stores/useChartStoreFactory';
import MarketChart from '@/components/Charts/MarketChart';
import VolumeChart from '@/components/Charts/VolumeChart';
import RSIChart from '@/components/Charts/RSIChart';
import MACDChart from '@/components/Charts/MACDChart';

interface CryptoSectionProps {
  symbol: string;
  netSignal: number;
  buyCount: number;
  sellCount: number;
}

function CryptoSection({ symbol, netSignal, buyCount, sellCount }: CryptoSectionProps) {
  const [interval, setInterval] = React.useState<string>('1m');
  const [limit, setLimit] = React.useState<number>(2000);
  const { data, loading } = useCryptoData(symbol as any, interval, limit);

  // Créer un store dédié pour cette crypto
  const useStore = useMemo(() => createChartStore(symbol, interval), [symbol, interval]);
  const { setMarketData, setSignals, setIndicators, setConfig } = useStore();

  // Mettre à jour le store quand les données changent
  useEffect(() => {
    if (data) {
      setConfig({ symbol, interval });
      setMarketData(data.marketData);
      setSignals(data.signals);
      setIndicators(data.indicators);
    }
  }, [data, symbol, interval, setConfig, setMarketData, setSignals, setIndicators]);

  return (
    <div className="space-y-4 pb-8 border-b-4 border-gray-700">
      {/* En-tête crypto */}
      <div className="bg-dark-200 border border-gray-700 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h2 className="text-2xl font-bold text-white">{symbol}</h2>
            <div className={`text-xl font-bold ${netSignal > 0 ? 'text-green-400' : 'text-red-400'}`}>
              {netSignal > 0 ? `+${netSignal}` : netSignal}
            </div>
            {loading ? (
              <div className="text-sm text-gray-400 animate-pulse">Chargement...</div>
            ) : data && (
              <div className="text-lg text-gray-300">
                {formatNumber(data.marketData.close[data.marketData.close.length - 1], 4)} USDC
              </div>
            )}
          </div>
          <div className="flex items-center space-x-4">
            {/* Timeframe selector */}
            <div className="flex items-center space-x-2">
              <span className="text-gray-400 text-sm">Timeframe:</span>
              <select
                value={interval}
                onChange={(e) => setInterval(e.target.value)}
                aria-label="Sélectionner le timeframe"
                className="bg-dark-300 text-white border border-gray-600 rounded px-2 py-1 text-sm"
              >
                <option value="1m">1m</option>
                <option value="5m">5m</option>
                <option value="15m">15m</option>
                <option value="30m">30m</option>
                <option value="1h">1h</option>
                <option value="4h">4h</option>
                <option value="1d">1d</option>
              </select>
            </div>
            {/* Candles limit */}
            <div className="flex items-center space-x-2">
              <span className="text-gray-400 text-sm">Bougies:</span>
              <select
                value={limit}
                onChange={(e) => setLimit(Number(e.target.value))}
                aria-label="Sélectionner le nombre de bougies"
                className="bg-dark-300 text-white border border-gray-600 rounded px-2 py-1 text-sm"
              >
                <option value="500">500</option>
                <option value="1000">1000</option>
                <option value="2000">2000</option>
                <option value="5000">5000</option>
              </select>
            </div>
            <div className="flex items-center space-x-2 text-lg">
              <span className="text-green-400 font-bold">{buyCount} 🟢</span>
              <span className="text-red-400 font-bold">{sellCount} 🔴</span>
            </div>
          </div>
        </div>
      </div>

      {/* Market + Signaux */}
      <div className="chart-container">
        <div className="chart-title">📈 Market + Signaux</div>
        {loading ? (
          <div className="h-[700px] flex items-center justify-center bg-dark-300 rounded animate-pulse">
            <div className="text-gray-500">Chargement du graphique...</div>
          </div>
        ) : (
          <MarketChart height={700} useStore={useStore} />
        )}
      </div>

      {/* Volume */}
      <div className="chart-container">
        <div className="chart-title">📊 Volume</div>
        {loading ? (
          <div className="h-[200px] flex items-center justify-center bg-dark-300 rounded animate-pulse">
            <div className="text-gray-500 text-sm">Chargement volume...</div>
          </div>
        ) : (
          <VolumeChart height={200} useStore={useStore} />
        )}
      </div>

      {/* RSI + MACD */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="chart-container">
          <div className="chart-title">📉 RSI</div>
          {loading ? (
            <div className="h-[200px] flex items-center justify-center bg-dark-300 rounded animate-pulse">
              <div className="text-gray-500 text-sm">Chargement RSI...</div>
            </div>
          ) : (
            <RSIChart height={200} useStore={useStore} />
          )}
        </div>
        <div className="chart-container">
          <div className="chart-title">📉 MACD</div>
          {loading ? (
            <div className="h-[200px] flex items-center justify-center bg-dark-300 rounded animate-pulse">
              <div className="text-gray-500 text-sm">Chargement MACD...</div>
            </div>
          ) : (
            <MACDChart height={200} useStore={useStore} />
          )}
        </div>
      </div>
    </div>
  );
}

export default memo(CryptoSection);
