import React from 'react';
import type { TradingSymbol } from '@/types';

interface CyclesFiltersProps {
  filters: {
    symbol: TradingSymbol | 'all';
    status: 'all' | 'active_buy' | 'active_sell' | 'completed' | 'cancelled';
    strategy: string;
    period: '24h' | '7d' | '30d' | 'all';
  };
  availableSymbols: TradingSymbol[];
  availableStrategies: string[];
  onFilterChange: (filters: Partial<CyclesFiltersProps['filters']>) => void;
}

function CyclesFilters({ filters, availableSymbols, availableStrategies, onFilterChange }: CyclesFiltersProps) {
  return (
    <div className="bg-dark-200 border border-gray-700 rounded-lg p-4">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Filtre par symbole */}
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">
            Symbole
          </label>
          <select
            value={filters.symbol}
            onChange={(e) => onFilterChange({ symbol: e.target.value as TradingSymbol | 'all' })}
            className="w-full bg-dark-300 border border-gray-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary-500"
          >
            <option value="all">Tous les symboles</option>
            {availableSymbols.map(symbol => (
              <option key={symbol} value={symbol}>
                {symbol}
              </option>
            ))}
          </select>
        </div>

        {/* Filtre par statut */}
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">
            Statut
          </label>
          <select
            value={filters.status}
            onChange={(e) => onFilterChange({ status: e.target.value as any })}
            className="w-full bg-dark-300 border border-gray-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary-500"
          >
            <option value="all">Tous les statuts</option>
            <option value="active_buy">🔵 BUY Actif</option>
            <option value="active_sell">🟠 SELL Actif</option>
            <option value="completed">✅ Complété</option>
            <option value="cancelled">❌ Annulé</option>
          </select>
        </div>

        {/* Filtre par stratégie */}
        <div>
          <label className="block text-sm font-medium text-gray-400 mb-2">
            Stratégie
          </label>
          <select
            value={filters.strategy}
            onChange={(e) => onFilterChange({ strategy: e.target.value })}
            className="w-full bg-dark-300 border border-gray-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary-500"
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
          <label className="block text-sm font-medium text-gray-400 mb-2">
            Période
          </label>
          <select
            value={filters.period}
            onChange={(e) => onFilterChange({ period: e.target.value as any })}
            className="w-full bg-dark-300 border border-gray-600 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-primary-500"
          >
            <option value="24h">Dernières 24h</option>
            <option value="7d">7 derniers jours</option>
            <option value="30d">30 derniers jours</option>
            <option value="all">Tout l'historique</option>
          </select>
        </div>
      </div>
    </div>
  );
}

export default CyclesFilters;