import React, { useState, useEffect } from 'react';
import { format } from 'date-fns';
import { fr } from 'date-fns/locale';
import CyclesFilters from './CyclesFilters';
import CyclesTable from './CyclesTable';
import CyclesStats from './CyclesStats';
import { apiService } from '@/services/api';
import type { TradingSymbol } from '@/types';

export interface TradeCycle {
  id: string;
  symbol: TradingSymbol;
  strategy: string;
  status: 'active_buy' | 'active_sell' | 'completed' | 'cancelled';
  side: 'BUY' | 'SELL';
  entry_order_id?: string;
  exit_order_id?: string;
  entry_price?: number;
  exit_price?: number;
  quantity?: number;
  profit_loss?: number;
  profit_loss_percent?: number;
  created_at: string;
  updated_at: string;
  completed_at?: string;
}

function CyclesPage() {
  const [cycles, setCycles] = useState<TradeCycle[]>([]);
  const [filteredCycles, setFilteredCycles] = useState<TradeCycle[]>([]);
  const [availableSymbols, setAvailableSymbols] = useState<TradingSymbol[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filtres
  const [filters, setFilters] = useState({
    symbol: 'all' as TradingSymbol | 'all',
    status: 'all' as 'all' | 'active_buy' | 'active_sell' | 'completed' | 'cancelled',
    strategy: 'all',
    period: '24h' as '24h' | '7d' | '30d' | 'all',
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

  // Charger tous les cycles
  const loadAllCycles = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await apiService.getTradeCycles();
      
      // Trier par date de création décroissante (plus récent en premier)
      const sortedCycles = response.cycles.sort((a, b) => 
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      );
      
      setCycles(sortedCycles);
    } catch (err) {
      setError('Erreur lors du chargement des cycles');
      console.error('Erreur cycles:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Charger les cycles au démarrage
  useEffect(() => {
    loadAllCycles();
  }, []);

  // Filtrer les cycles selon les critères
  useEffect(() => {
    let filtered = [...cycles];
    
    // Filtre par période
    if (filters.period !== 'all') {
      const now = Date.now();
      const periodHours = {
        '24h': 24,
        '7d': 24 * 7,
        '30d': 24 * 30,
      };
      
      const cutoffTime = now - periodHours[filters.period] * 60 * 60 * 1000;
      filtered = filtered.filter(cycle => 
        new Date(cycle.created_at).getTime() >= cutoffTime
      );
    }
    
    // Filtre par symbole
    if (filters.symbol !== 'all') {
      filtered = filtered.filter(cycle => cycle.symbol === filters.symbol);
    }
    
    // Filtre par statut
    if (filters.status !== 'all') {
      filtered = filtered.filter(cycle => cycle.status === filters.status);
    }
    
    // Filtre par stratégie
    if (filters.strategy !== 'all') {
      filtered = filtered.filter(cycle => cycle.strategy === filters.strategy);
    }
    
    setFilteredCycles(filtered);
  }, [cycles, filters]);

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
              🔄 Cycles de Trading
            </h1>
            <p className="text-gray-400">
              Vue d'ensemble des cycles BUY/SELL et leur performance
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            <button
              onClick={loadAllCycles}
              disabled={isLoading}
              className="bg-primary-500 hover:bg-primary-600 disabled:opacity-50 text-white px-4 py-2 rounded-lg transition-colors"
            >
              {isLoading ? '🔄 Actualisation...' : '🔄 Actualiser'}
            </button>
            
            <div className="text-sm text-gray-400">
              {filteredCycles.length} cycles affichés
            </div>
          </div>
        </div>
      </div>

      {/* Statistiques */}
      <CyclesStats cycles={filteredCycles} />

      {/* Filtres */}
      <CyclesFilters
        filters={filters}
        availableSymbols={availableSymbols}
        availableStrategies={[...new Set(cycles.map(c => c.strategy))]}
        onFilterChange={handleFilterChange}
      />

      {/* Messages d'état */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4">
          <p className="text-red-400">{error}</p>
        </div>
      )}

      {/* Tableau des cycles */}
      <CyclesTable
        cycles={filteredCycles}
        isLoading={isLoading}
      />
    </div>
  );
}

export default CyclesPage;