import React, { useState, useEffect } from 'react';
import { format } from 'date-fns';
import { fr } from 'date-fns/locale';
import SignalsFilters from './SignalsFilters';
import SignalsTable from './SignalsTable';
import { apiService } from '@/services/api';
import type { TradingSignal, TradingSymbol } from '@/types';

interface ExtendedTradingSignal extends TradingSignal {
  symbol: TradingSymbol;
  type: 'buy' | 'sell';
}

function SignalsPage() {
  const [signals, setSignals] = useState<ExtendedTradingSignal[]>([]);
  const [filteredSignals, setFilteredSignals] = useState<ExtendedTradingSignal[]>([]);
  const [availableSymbols, setAvailableSymbols] = useState<TradingSymbol[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filtres
  const [filters, setFilters] = useState({
    symbol: 'all' as TradingSymbol | 'all',
    type: 'all' as 'all' | 'buy' | 'sell',
    strategy: 'all',
    period: '24h' as '1h' | '24h' | '7d' | '30d',
  });

  // Charger les symboles disponibles
  useEffect(() => {
    const loadSymbols = async () => {
      try {
        const response = await apiService.getAvailableSymbols();
        setAvailableSymbols(response.symbols);
      } catch (err) {
        console.error('Erreur lors du chargement des symboles:', err);
      }
    };

    loadSymbols();
  }, []);

  // Charger tous les signaux de toutes les paires
  const loadAllSignals = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const symbolsToLoad = availableSymbols.length > 0 ? availableSymbols : ['BTCUSDC', 'ETHUSDC', 'SOLUSDC', 'XRPUSDC'];
      
      const signalsPromises = symbolsToLoad.map(async (symbol) => {
        try {
          const response = await apiService.getTradingSignals(symbol);
          const allSignals = [
            ...response.signals.buy.map(s => ({ ...s, symbol, type: 'buy' as const })),
            ...response.signals.sell.map(s => ({ ...s, symbol, type: 'sell' as const }))
          ];
          return allSignals;
        } catch (err) {
          console.error(`Erreur pour ${symbol}:`, err);
          return [];
        }
      });

      const allSignalsArrays = await Promise.all(signalsPromises);
      const allSignals = allSignalsArrays.flat();
      
      // Trier par timestamp décroissant (plus récent en premier)
      allSignals.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
      
      setSignals(allSignals);
    } catch (err) {
      setError('Erreur lors du chargement des signaux');
      console.error('Erreur signaux:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Charger les signaux au démarrage et quand les symboles changent
  useEffect(() => {
    if (availableSymbols.length > 0) {
      loadAllSignals();
    }
  }, [availableSymbols]);

  // Filtrer les signaux selon les critères
  useEffect(() => {
    let filtered = [...signals];
    
    // Filtre par période
    const now = Date.now();
    const periodHours = {
      '1h': 1,
      '24h': 24,
      '7d': 24 * 7,
      '30d': 24 * 30,
    };
    
    const cutoffTime = now - periodHours[filters.period] * 60 * 60 * 1000;
    filtered = filtered.filter(signal => 
      new Date(signal.timestamp).getTime() >= cutoffTime
    );
    
    // Filtre par symbole
    if (filters.symbol !== 'all') {
      filtered = filtered.filter(signal => signal.symbol === filters.symbol);
    }
    
    // Filtre par type
    if (filters.type !== 'all') {
      filtered = filtered.filter(signal => signal.type === filters.type);
    }
    
    // Filtre par stratégie
    if (filters.strategy !== 'all') {
      filtered = filtered.filter(signal => signal.strategy === filters.strategy);
    }
    
    setFilteredSignals(filtered);
  }, [signals, filters]);

  const handleFilterChange = (newFilters: Partial<typeof filters>) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
  };

  return (
    <div className="space-y-6">
      {/* En-tête */}
      <div className="bg-dark-200 border border-gray-700 rounded-lg p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white mb-2">
              📊 Signaux de Trading
            </h1>
            <p className="text-gray-400">
              Vue d'ensemble des signaux BUY/SELL générés par les stratégies
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            <button
              onClick={loadAllSignals}
              disabled={isLoading}
              className="bg-primary-500 hover:bg-primary-600 disabled:opacity-50 text-white px-4 py-2 rounded-lg transition-colors"
            >
              {isLoading ? '🔄 Actualisation...' : '🔄 Actualiser'}
            </button>
            
            <div className="text-sm text-gray-400">
              {filteredSignals.length} signaux affichés
            </div>
          </div>
        </div>
      </div>

      {/* Filtres */}
      <SignalsFilters
        filters={filters}
        availableSymbols={availableSymbols}
        availableStrategies={[...new Set(signals.map(s => s.strategy))]}
        onFilterChange={handleFilterChange}
      />

      {/* Messages d'état */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4">
          <p className="text-red-400">{error}</p>
        </div>
      )}

      {/* Tableau des signaux */}
      <SignalsTable
        signals={filteredSignals}
        isLoading={isLoading}
      />
    </div>
  );
}

export default SignalsPage;