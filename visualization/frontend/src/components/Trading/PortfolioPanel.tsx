import React, { useState, useEffect } from 'react';
import { apiService } from '@/services/api';

interface PortfolioMetrics {
  total_value: number;
  available_balance: number;
  invested_balance: number;
  total_pnl: number;
  total_pnl_percentage: number;
  daily_pnl: number;
  daily_pnl_percentage: number;
  total_trades: number;
  win_rate: number;
  sharpe_ratio: number;
  max_drawdown: number;
}

interface PortfolioPosition {
  symbol: string;
  quantity: number;
  avg_price: number;
  current_price: number;
  pnl: number;
  pnl_percentage: number;
  allocation_percentage: number;
  last_updated: string;
}

function PortfolioPanel() {
  const [metrics, setMetrics] = useState<PortfolioMetrics | null>(null);
  const [positions, setPositions] = useState<PortfolioPosition[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchPortfolioData();
    const interval = setInterval(fetchPortfolioData, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchPortfolioData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Récupérer les vraies données du portfolio
      const [summaryResponse, balancesResponse, performanceResponse] = await Promise.all([
        apiService.getPortfolioSummary(),
        apiService.getPortfolioBalances(),
        apiService.getPortfolioPerformance('daily')
      ]);

      // Transformer les données pour les métriques
      const summary = summaryResponse;
      const balances = balancesResponse;
      const performance = performanceResponse?.data || [];

      const realMetrics: PortfolioMetrics = {
        total_value: summary.total_value || 0,
        available_balance: summary.available_balance || 0,
        invested_balance: summary.invested_balance || 0,
        total_pnl: summary.total_pnl || 0,
        total_pnl_percentage: summary.total_pnl_percentage || 0,
        daily_pnl: summary.daily_pnl || 0,
        daily_pnl_percentage: summary.daily_pnl_percentage || 0,
        total_trades: summary.total_trades || 0,
        win_rate: summary.win_rate || 0,
        sharpe_ratio: summary.sharpe_ratio || 0,
        max_drawdown: summary.max_drawdown || 0
      };

      // Transformer les balances en positions
      const realPositions: PortfolioPosition[] = balances
        .filter((balance: any) => balance.total > 0 && balance.asset !== 'USDC')
        .map((balance: any) => ({
          symbol: `${balance.asset}USDC`,
          quantity: balance.total,
          avg_price: balance.value_usdc / balance.total,
          current_price: balance.value_usdc / balance.total,
          pnl: 0, // Sera calculé avec les données de performance
          pnl_percentage: 0,
          allocation_percentage: summary.total_value > 0 ? (balance.value_usdc / summary.total_value) * 100 : 0,
          last_updated: new Date().toISOString()
        }));

      setMetrics(realMetrics);
      setPositions(realPositions);
    } catch (err) {
      setError('Erreur lors du chargement des données portfolio');
      console.error('Portfolio fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (value: number): string => {
    return new Intl.NumberFormat('fr-FR', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  const formatPercentage = (value: number): string => {
    return `${value > 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  if (loading) {
    return (
      <div className="h-96 bg-dark-300 rounded-lg p-4 flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-96 bg-dark-300 rounded-lg p-4 flex items-center justify-center">
        <span className="text-red-400">{error}</span>
      </div>
    );
  }

  return (
    <div className="h-96 bg-dark-300 rounded-lg p-4 overflow-y-auto">
      {/* Métriques principales */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-dark-200 rounded-lg p-3">
          <div className="text-gray-400 text-sm">Valeur Totale</div>
          <div className="text-xl font-bold text-white">
            {formatCurrency(metrics?.total_value || 0)}
          </div>
        </div>
        
        <div className="bg-dark-200 rounded-lg p-3">
          <div className="text-gray-400 text-sm">P&L Total</div>
          <div className={`text-xl font-bold ${(metrics?.total_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {formatCurrency(metrics?.total_pnl || 0)}
          </div>
          <div className={`text-sm ${(metrics?.total_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {formatPercentage(metrics?.total_pnl_percentage || 0)}
          </div>
        </div>
        
        <div className="bg-dark-200 rounded-lg p-3">
          <div className="text-gray-400 text-sm">P&L Journalier</div>
          <div className={`text-lg font-bold ${(metrics?.daily_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {formatCurrency(metrics?.daily_pnl || 0)}
          </div>
          <div className={`text-sm ${(metrics?.daily_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {formatPercentage(metrics?.daily_pnl_percentage || 0)}
          </div>
        </div>
        
        <div className="bg-dark-200 rounded-lg p-3">
          <div className="text-gray-400 text-sm">Win Rate</div>
          <div className="text-lg font-bold text-white">
            {metrics?.win_rate?.toFixed(1) || 0}%
          </div>
          <div className="text-sm text-gray-400">
            {metrics?.total_trades || 0} trades
          </div>
        </div>
      </div>

      {/* Métriques de performance */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-dark-200 rounded-lg p-3">
          <div className="text-gray-400 text-sm">Sharpe Ratio</div>
          <div className="text-lg font-bold text-white">
            {metrics?.sharpe_ratio?.toFixed(2) || 0}
          </div>
        </div>
        
        <div className="bg-dark-200 rounded-lg p-3">
          <div className="text-gray-400 text-sm">Max Drawdown</div>
          <div className="text-lg font-bold text-red-400">
            {formatPercentage(metrics?.max_drawdown || 0)}
          </div>
        </div>
      </div>

      {/* Répartition du capital */}
      <div className="mb-4">
        <h3 className="text-sm font-medium text-gray-400 mb-2">Répartition du Capital</h3>
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Disponible</span>
            <span className="text-white">{formatCurrency(metrics?.available_balance || 0)}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Investi</span>
            <span className="text-white">{formatCurrency(metrics?.invested_balance || 0)}</span>
          </div>
        </div>
      </div>

      {/* Positions top */}
      <div>
        <h3 className="text-sm font-medium text-gray-400 mb-2">Top Positions</h3>
        <div className="space-y-2">
          {positions.slice(0, 3).map((position) => (
            <div key={position.symbol} className="flex justify-between items-center text-sm">
              <div className="flex items-center space-x-2">
                <span className="text-white font-medium">{position.symbol}</span>
                <span className="text-gray-400">{position.allocation_percentage.toFixed(1)}%</span>
              </div>
              <div className="text-right">
                <div className={`${position.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {formatCurrency(position.pnl)}
                </div>
                <div className={`text-xs ${position.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {formatPercentage(position.pnl_percentage)}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default PortfolioPanel;