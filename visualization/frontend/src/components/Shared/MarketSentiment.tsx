import React, { useState, useEffect } from 'react';

interface SentimentData {
  buySignals: number;
  sellSignals: number;
  netSignal: number;
  totalCryptos: number;
  opportunitiesCount: number;
  topSymbol: string;
  topNetSignal: number;
  sentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
}

interface TimeframeSentiment {
  '15m': SentimentData | null;
  '1h': SentimentData | null;
  '4h': SentimentData | null;
}

interface EmaSentimentData {
  sentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL' | 'MIXED';
  bullish_count: number;
  bearish_count: number;
  neutral_count: number;
  mixed_count: number;
  total_symbols: number;
  bullish_percent: number;
  bearish_percent: number;
  top_bullish: Array<{
    symbol: string;
    score: number;
    sentiment: string;
  }>;
  top_bearish: Array<{
    symbol: string;
    score: number;
    sentiment: string;
  }>;
}

export const MarketSentiment: React.FC = () => {
  const [sentiments, setSentiments] = useState<TimeframeSentiment>({
    '15m': null,
    '1h': null,
    '4h': null
  });
  const [emaSentiment, setEmaSentiment] = useState<EmaSentimentData | null>(null);
  const [loading, setLoading] = useState(true);

  const loadSentiment = async () => {
    try {
      setLoading(true);

      // Charger les 3 timeframes + EMA sentiment en parallèle
      const [data15m, data1h, data4h, emaData] = await Promise.all([
        fetch('/api/top-signals?timeframe_minutes=15&limit=100').then(r => r.json()),
        fetch('/api/top-signals?timeframe_minutes=60&limit=100').then(r => r.json()),
        fetch('/api/top-signals?timeframe_minutes=240&limit=100').then(r => r.json()),
        fetch('/api/ema-sentiment?timeframe=1m&limit=100').then(r => r.json()).catch(() => null)
      ]);

      const processSentiment = (data: any): SentimentData => {
        const signals = data.signals || [];

        // Compter les symboles avec consensus BUY ou SELL (pas la somme des counts)
        const buySignals = signals.filter((s: any) => s.net_signal > 0).length;
        const sellSignals = signals.filter((s: any) => s.net_signal < 0).length;
        const netSignal = buySignals - sellSignals;
        const opportunitiesCount = buySignals; // Les opportunités = consensus BUY

        // Trouver le top symbole
        const topSignal = signals.reduce((max: any, s: any) => {
          return (s.net_signal || 0) > (max?.net_signal || 0) ? s : max;
        }, null);

        // Déterminer sentiment
        let sentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL' = 'NEUTRAL';
        const buyRatio = buySignals / (buySignals + sellSignals);
        if (buyRatio >= 0.60) sentiment = 'BULLISH';
        else if (buyRatio <= 0.40) sentiment = 'BEARISH';

        return {
          buySignals,
          sellSignals,
          netSignal,
          totalCryptos: signals.length,
          opportunitiesCount,
          topSymbol: topSignal?.symbol?.replace('USDC', '') || 'N/A',
          topNetSignal: topSignal?.net_signal || 0,
          sentiment
        };
      };

      setSentiments({
        '15m': processSentiment(data15m),
        '1h': processSentiment(data1h),
        '4h': processSentiment(data4h)
      });

      // Stocker le sentiment EMA
      if (emaData) {
        setEmaSentiment(emaData);
      }
    } catch (error) {
      console.error('Erreur chargement sentiment:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSentiment();
    const interval = setInterval(loadSentiment, 60000); // Refresh 1min
    return () => clearInterval(interval);
  }, []);

  if (loading || !sentiments['15m']) {
    return (
      <div className="bg-dark-200 border border-gray-700 rounded-lg p-4">
        <div className="text-gray-400">Chargement sentiment marché...</div>
      </div>
    );
  }

  const getSentimentColor = (sentiment: string) => {
    if (sentiment === 'BULLISH') return 'text-green-400';
    if (sentiment === 'BEARISH') return 'text-red-400';
    return 'text-gray-400';
  };

  const getSentimentBg = (sentiment: string) => {
    if (sentiment === 'BULLISH') return 'bg-green-600';
    if (sentiment === 'BEARISH') return 'bg-red-600';
    return 'bg-gray-600';
  };

  const getSentimentEmoji = (sentiment: string) => {
    if (sentiment === 'BULLISH') return '📈';
    if (sentiment === 'BEARISH') return '📉';
    return '➖';
  };

  // Utiliser le sentiment 15m comme principal
  const mainSentiment = sentiments['15m']!;
  const buyPercent = (mainSentiment.buySignals / (mainSentiment.buySignals + mainSentiment.sellSignals)) * 100;
  const sellPercent = 100 - buyPercent;

  return (
    <div className="bg-dark-200 border border-gray-700 rounded-lg p-4">
      <div className="flex items-start justify-between gap-4">

        {/* Colonne 1 - Sentiment principal */}
        <div className="flex-shrink-0 w-72">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-bold text-white">📊 Sentiment Marché</h3>
            <div className={`px-3 py-1 rounded-lg ${getSentimentBg(mainSentiment.sentiment)}`}>
              <span className="text-white font-bold text-sm">
                {getSentimentEmoji(mainSentiment.sentiment)} {mainSentiment.sentiment}
              </span>
            </div>
          </div>

          {/* BUY vs SELL */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                <span className="text-gray-400 text-sm">BUY</span>
              </div>
              <span className="text-green-400 font-bold text-xl">{mainSentiment.buySignals}</span>
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                <span className="text-gray-400 text-sm">SELL</span>
              </div>
              <span className="text-red-400 font-bold text-xl">{mainSentiment.sellSignals}</span>
            </div>

            <div className="pt-2 border-t border-gray-700 flex items-center justify-between">
              <span className="text-gray-400 text-sm font-semibold">Net Signal</span>
              <span className={`font-bold text-xl ${mainSentiment.netSignal >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {mainSentiment.netSignal >= 0 ? '+' : ''}{mainSentiment.netSignal}
              </span>
            </div>
          </div>
        </div>

        {/* Colonne 2 - Statistiques */}
        <div className="flex-1 grid grid-cols-3 gap-3">
          <div className="bg-dark-300 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-blue-400 mb-1">{mainSentiment.opportunitiesCount}</div>
            <div className="text-xs text-gray-500">Opportunités</div>
            <div className="text-xs text-gray-600 mt-1">(Net Signal +)</div>
          </div>

          <div className="bg-dark-300 rounded-lg p-4 text-center">
            <div className="text-2xl font-bold text-white mb-1">{mainSentiment.totalCryptos}</div>
            <div className="text-xs text-gray-500">Cryptos actives</div>
            <div className="text-xs text-gray-600 mt-1">sur 15min</div>
          </div>

          <div className="bg-dark-300 rounded-lg p-4 text-center">
            <div className="text-xl font-bold text-green-400 mb-1">{mainSentiment.topSymbol}</div>
            <div className="text-xs text-gray-500">Top Crypto</div>
            <div className="text-xs text-green-600 mt-1">+{mainSentiment.topNetSignal} signal</div>
          </div>

          {/* Barres de proportion */}
          <div className="col-span-3 space-y-2 mt-2">
            <div className="flex items-center gap-3">
              <span className="text-xs text-gray-500 w-12">BUY</span>
              <div className="flex-1 h-6 bg-gray-700 rounded overflow-hidden">
                <div
                  className="h-full bg-green-500 flex items-center justify-center transition-all duration-500"
                  style={{ width: `${buyPercent}%` }}
                >
                  <span className="text-xs font-bold text-white">{buyPercent.toFixed(0)}%</span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <span className="text-xs text-gray-500 w-12">SELL</span>
              <div className="flex-1 h-6 bg-gray-700 rounded overflow-hidden">
                <div
                  className="h-full bg-red-500 flex items-center justify-center transition-all duration-500"
                  style={{ width: `${sellPercent}%` }}
                >
                  <span className="text-xs font-bold text-white">{sellPercent.toFixed(0)}%</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Colonne 3 - Tendances timeframes */}
        <div className="flex-shrink-0 w-56">
          <div className="text-sm text-gray-400 mb-3 font-semibold">📉 Multi-timeframes</div>
          <div className="space-y-2">
            {(['15m', '1h', '4h'] as const).map((tf) => {
              const data = sentiments[tf];
              if (!data) return null;

              return (
                <div key={tf} className="bg-dark-300 rounded-lg p-3 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="text-xs font-bold text-gray-400 w-8">{tf}</div>
                    <div className={`text-xl ${getSentimentColor(data.sentiment)}`}>
                      {getSentimentEmoji(data.sentiment)}
                    </div>
                  </div>
                  <div className={`text-base font-bold ${data.netSignal >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {data.netSignal >= 0 ? '+' : ''}{data.netSignal}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Colonne 4 - Sentiment EMA */}
        {emaSentiment && (
          <div className="flex-shrink-0 w-56">
            <div className="text-sm text-gray-400 mb-3 font-semibold">🔀 Alignement EMA</div>
            <div className="bg-dark-300 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <div className={`text-2xl ${getSentimentColor(emaSentiment.sentiment)}`}>
                    {getSentimentEmoji(emaSentiment.sentiment)}
                  </div>
                  <div className="text-sm font-bold text-white">{emaSentiment.sentiment}</div>
                </div>
                <div className="text-xs text-gray-500">
                  {emaSentiment.total_symbols}
                </div>
              </div>

              {/* Distribution */}
              <div className="space-y-2 text-xs">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-gray-400">Bull</span>
                  </div>
                  <span className="text-green-400 font-bold">
                    {emaSentiment.bullish_count} ({emaSentiment.bullish_percent.toFixed(0)}%)
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                    <span className="text-gray-400">Bear</span>
                  </div>
                  <span className="text-red-400 font-bold">
                    {emaSentiment.bearish_count} ({emaSentiment.bearish_percent.toFixed(0)}%)
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-gray-500 rounded-full"></div>
                    <span className="text-gray-400">Mix</span>
                  </div>
                  <span className="text-gray-400 font-bold">
                    {emaSentiment.neutral_count + emaSentiment.mixed_count}
                  </span>
                </div>
              </div>

              {/* Top cryptos */}
              {emaSentiment.top_bullish.length > 0 && (
                <div className="mt-3 pt-3 border-t border-gray-700">
                  <div className="text-xs text-gray-500 mb-1">Top:</div>
                  <div className="flex flex-wrap gap-1">
                    {emaSentiment.top_bullish.slice(0, 3).map((s) => (
                      <span key={s.symbol} className="text-xs bg-green-900/30 text-green-400 px-2 py-1 rounded">
                        {s.symbol.replace('USDC', '')}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

      </div>
    </div>
  );
};
