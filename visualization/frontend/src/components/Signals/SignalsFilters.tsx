import React from 'react';
import type { TradingSymbol } from '@/types';

interface SignalsFiltersProps {
  filters: {
    symbol: TradingSymbol | 'all';
    type: 'all' | 'buy' | 'sell';
    strategy: string;
    period: '1h' | '24h' | '7d' | '30d';
  };
  availableSymbols: TradingSymbol[];
  availableStrategies: string[];
  onFilterChange: (filters: Partial<SignalsFiltersProps['filters']>) => void;
}

function SignalsFilters({
  filters,
  availableSymbols,
  availableStrategies,
  onFilterChange,
}: SignalsFiltersProps) {
  return (
    <div className="bg-dark-200 border border-gray-700 rounded-lg p-4">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Filtre par paire */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            💰 Paire
          </label>
          <select
            value={filters.symbol}
            onChange={(e) => onFilterChange({ symbol: e.target.value as TradingSymbol | 'all' })}
            className="w-full bg-dark-100 border border-gray-600 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="all">Toutes les paires</option>
            {availableSymbols.map(symbol => (
              <option key={symbol} value={symbol}>
                {symbol}
              </option>
            ))}
          </select>
        </div>

        {/* Filtre par type */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            🎯 Type
          </label>
          <select
            value={filters.type}
            onChange={(e) => onFilterChange({ type: e.target.value as 'all' | 'buy' | 'sell' })}
            className="w-full bg-dark-100 border border-gray-600 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="all">Tous les types</option>
            <option value="buy">🟢 BUY seulement</option>
            <option value="sell">🔴 SELL seulement</option>
          </select>
        </div>

        {/* Filtre par stratégie */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            🧠 Stratégie
          </label>
          <select
            value={filters.strategy}
            onChange={(e) => onFilterChange({ strategy: e.target.value })}
            className="w-full bg-dark-100 border border-gray-600 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="all">Toutes les stratégies</option>
            {availableStrategies.map(strategy => (
              <option key={strategy} value={strategy}>
                {strategy}
              </option>
            ))}
          </select>
        </div>

        {/* Filtre par période */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            📅 Période
          </label>
          <select
            value={filters.period}
            onChange={(e) => onFilterChange({ period: e.target.value as '1h' | '24h' | '7d' | '30d' })}
            className="w-full bg-dark-100 border border-gray-600 rounded-lg px-3 py-2 text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="1h">Dernière heure</option>
            <option value="24h">Dernières 24h</option>
            <option value="7d">7 derniers jours</option>
            <option value="30d">30 derniers jours</option>
          </select>
        </div>
      </div>

      {/* Bouton reset */}
      <div className="mt-4 flex justify-end">
        <button
          onClick={() => onFilterChange({
            symbol: 'all',
            type: 'all',
            strategy: 'all',
            period: '24h'
          })}
          className="text-sm text-gray-400 hover:text-white transition-colors px-3 py-1 rounded-lg hover:bg-dark-100"
        >
          🔄 Réinitialiser les filtres
        </button>
      </div>
    </div>
  );
}

export default SignalsFilters;