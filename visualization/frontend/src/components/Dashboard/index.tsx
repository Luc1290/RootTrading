import React, { useState, useEffect } from 'react';
import CryptoSection from './CryptoSection';
import { apiService } from '@/services/api';

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

  return (
    <div className="space-y-8 p-6">
      {/* En-tête */}
      <div className="bg-dark-200 border border-gray-700 rounded-lg p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">
              📊 Dashboard Multi-Crypto
            </h1>
            <p className="text-gray-400">
              {allCryptos.length} cryptos configurées • {allCryptos.filter(c => c.netSignal > 0).length} avec signaux BUY récents (15min)
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
      </div>

      {/* Sections par crypto */}
      {allCryptos.map((crypto) => (
        <CryptoSection
          key={crypto.symbol}
          symbol={crypto.symbol}
          netSignal={crypto.netSignal}
          buyCount={crypto.buyCount}
          sellCount={crypto.sellCount}
        />
      ))}

      {allCryptos.length === 0 && (
        <div className="text-center py-12">
          <div className="text-gray-400 text-lg">
            Aucune crypto configurée
          </div>
        </div>
      )}
    </div>
  );
}

export default Dashboard;