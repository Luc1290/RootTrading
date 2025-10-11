import React, { useState, useEffect } from 'react';
import CryptoSection from './CryptoSection';
import { apiService } from '@/services/api';
import { MarketSentiment } from '@/components/Shared/MarketSentiment';

interface TopSignal {
  symbol: string;
  net_signal: number;
  buy_count: number;
  sell_count: number;
  dominant_side: string;
  last_signal_time: string;
}

interface CryptoWithSignal {
  symbol: string;
  netSignal: number;
  buyCount: number;
  sellCount: number;
}

function Dashboard() {
  const [allCryptos, setAllCryptos] = useState<CryptoWithSignal[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // Charger TOUTES les cryptos configurées + leurs signaux
  const loadAllCryptos = async () => {
    try {
      setLoading(true);

      // 1. Récupérer toutes les cryptos configurées
      const configuredResponse = await apiService.getConfiguredSymbols();
      const allSymbols = configuredResponse.symbols.map(s => s.endsWith('USDC') ? s : s + 'USDC');

      // 2. Récupérer les signaux récents
      const signalsResponse = await fetch('/api/top-signals?timeframe_minutes=15&limit=100');
      const signalsData = await signalsResponse.json();
      const signalsMap = new Map<string, TopSignal>(
        (signalsData.signals || []).map((s: TopSignal) => [s.symbol, s])
      );

      // 3. Combiner : toutes les cryptos avec leurs signaux (0 si pas de signaux)
      const cryptosWithSignals = allSymbols.map(symbol => {
        const signal = signalsMap.get(symbol);
        return {
          symbol,
          netSignal: signal?.net_signal || 0,
          buyCount: signal?.buy_count || 0,
          sellCount: signal?.sell_count || 0,
        };
      });

      // 4. Trier par net_signal DESC (les plus de signaux BUY en premier)
      cryptosWithSignals.sort((a, b) => b.netSignal - a.netSignal);

      setAllCryptos(cryptosWithSignals);

      // Sélectionner automatiquement la meilleure opportunité
      if (cryptosWithSignals.length > 0 && !selectedSymbol) {
        setSelectedSymbol(cryptosWithSignals[0].symbol);
      }
    } catch (error) {
      console.error('Erreur chargement cryptos:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAllCryptos();
    // Auto-refresh désactivé pour performance
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="flex items-center space-x-3">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
          <span className="text-white text-lg">Chargement des cryptos...</span>
        </div>
      </div>
    );
  }

  // Séparer les cryptos avec signaux positifs et les autres
  const cryptosWithSignals = allCryptos.filter(c => c.netSignal > 0);
  const cryptosWithoutSignals = allCryptos.filter(c => c.netSignal <= 0);
  const selectedCrypto = allCryptos.find(c => c.symbol === selectedSymbol);

  return (
    <div className="space-y-6 p-6">
      {/* Market Sentiment */}
      <MarketSentiment />

      {/* En-tête + Bandeau opportunités */}
      <div className="bg-dark-200 border border-gray-700 rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">
              📊 Dashboard Trading
            </h1>
            <p className="text-gray-400">
              {cryptosWithSignals.length} opportunités actives • {allCryptos.length} cryptos configurées
            </p>
          </div>
          <button
            type="button"
            onClick={loadAllCryptos}
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white px-4 py-2 rounded-lg font-medium transition-colors"
          >
            {loading ? '⏳ Chargement...' : '🔄 Rafraîchir'}
          </button>
        </div>

        {/* Bandeau toutes les cryptos (opportunités d'abord, puis les autres) */}
        {allCryptos.length > 0 && (
          <div className="border-t border-gray-700 pt-4 space-y-3">
            {/* Cryptos avec signaux positifs */}
            {cryptosWithSignals.length > 0 && (
              <div>
                <div className="text-sm text-gray-400 mb-2 font-semibold">🔥 Opportunités actives</div>
                <div className="flex flex-wrap gap-2">
                  {cryptosWithSignals.map((crypto) => {
                const baseAsset = crypto.symbol.replace('USDC', '');
                const isSelected = crypto.symbol === selectedSymbol;
                const bgColor = isSelected
                  ? 'bg-blue-600 border-blue-500'
                  : crypto.netSignal >= 5
                  ? 'bg-green-900/20 border-green-500/50 hover:bg-green-800/30'
                  : crypto.netSignal >= 3
                  ? 'bg-blue-900/20 border-blue-500/50 hover:bg-blue-800/30'
                  : 'bg-gray-800/20 border-gray-600 hover:bg-gray-700/30';

                return (
                  <button
                    key={crypto.symbol}
                    onClick={() => setSelectedSymbol(crypto.symbol)}
                    className={`px-4 py-2 rounded-lg border-2 transition-all cursor-pointer ${bgColor}`}
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-base font-bold text-white">{baseAsset}</span>
                      <span className={`text-sm font-bold ${
                        crypto.netSignal >= 5 ? 'text-green-400' :
                        crypto.netSignal >= 3 ? 'text-blue-400' : 'text-gray-400'
                      }`}>
                        +{crypto.netSignal}
                      </span>
                      <div className="flex items-center gap-1 text-xs">
                        <span className="text-green-400">{crypto.buyCount}🟢</span>
                        <span className="text-red-400">{crypto.sellCount}🔴</span>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
              </div>
            )}

            {/* Cryptos sans signaux */}
            {cryptosWithoutSignals.length > 0 && (
              <div>
                <div className="text-sm text-gray-500 mb-2 font-semibold">📋 Autres cryptos</div>
                <div className="flex flex-wrap gap-2">
                  {cryptosWithoutSignals.map((crypto) => {
                    const baseAsset = crypto.symbol.replace('USDC', '');
                    const isSelected = crypto.symbol === selectedSymbol;
                    const bgColor = isSelected
                      ? 'bg-blue-600 border-blue-500'
                      : 'bg-gray-800/20 border-gray-600 hover:bg-gray-700/30';

                    return (
                      <button
                        key={crypto.symbol}
                        onClick={() => setSelectedSymbol(crypto.symbol)}
                        className={`px-3 py-1.5 rounded-lg border transition-all cursor-pointer ${bgColor}`}
                      >
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium text-gray-300">{baseAsset}</span>
                          {crypto.netSignal !== 0 && (
                            <span className="text-xs text-gray-500">{crypto.netSignal}</span>
                          )}
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Affichage de la crypto sélectionnée */}
      {selectedCrypto ? (
        <CryptoSection
          key={selectedCrypto.symbol}
          symbol={selectedCrypto.symbol}
          netSignal={selectedCrypto.netSignal}
          buyCount={selectedCrypto.buyCount}
          sellCount={selectedCrypto.sellCount}
        />
      ) : (
        <div className="text-center py-12 bg-dark-200 border border-gray-700 rounded-lg">
          <div className="text-gray-400 text-lg">
            {allCryptos.length === 0
              ? 'Aucune crypto configurée'
              : 'Aucune opportunité détectée'}
          </div>
        </div>
      )}
    </div>
  );
}

export default Dashboard;