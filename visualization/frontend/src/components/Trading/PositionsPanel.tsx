import React, { useState, useEffect } from 'react';
import { apiService } from '@/services/api';

interface Position {
  symbol: string;
  side: 'LONG' | 'SHORT';
  quantity: number;
  entry_price: number;
  current_price: number;
  pnl: number;
  pnl_percentage: number;
  value: number;
  margin_used: number;
  timestamp: string;
  status: 'ACTIVE' | 'CLOSED' | 'PENDING';
}

function PositionsPanel() {
  const [positions, setPositions] = useState<Position[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchPositions();
    const interval = setInterval(fetchPositions, 15000);
    return () => clearInterval(interval);
  }, []);

  const fetchPositions = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Récupérer les vraies données des balances
      const balancesResponse = await apiService.getPortfolioBalances();
      
      // Transformer les balances en positions actives
      const realPositions: Position[] = balancesResponse
        .filter((balance: any) => balance.total > 0 && balance.asset !== 'USDC')
        .map((balance: any) => {
          const currentPrice = balance.value_usdc / balance.total;
          const entryPrice = currentPrice; // Approximation, il faudrait les vraies données d'entrée
          
          return {
            symbol: `${balance.asset}USDC`,
            side: 'LONG' as const,
            quantity: balance.total,
            entry_price: entryPrice,
            current_price: currentPrice,
            pnl: 0, // Sera calculé avec les données de performance
            pnl_percentage: 0,
            value: balance.value_usdc,
            margin_used: balance.value_usdc,
            timestamp: new Date().toISOString(),
            status: 'ACTIVE' as const
          };
        });

      setPositions(realPositions);
    } catch (err) {
      setError('Erreur lors du chargement des positions');
      console.error('Positions fetch error:', err);
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

  const formatTimeAgo = (timestamp: string): string => {
    const now = new Date();
    const time = new Date(timestamp);
    const diff = now.getTime() - time.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    }
    return `${minutes}m`;
  };

  if (loading) {
    return (
      <div className="h-64 bg-dark-300 rounded-lg p-4 flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-64 bg-dark-300 rounded-lg p-4 flex items-center justify-center">
        <span className="text-red-400">{error}</span>
      </div>
    );
  }

  return (
    <div className="h-64 bg-dark-300 rounded-lg p-4 overflow-y-auto">
      <div className="space-y-3">
        {positions.length === 0 ? (
          <div className="text-center text-gray-400 py-8">
            Aucune position active
          </div>
        ) : (
          positions.map((position, index) => (
            <div key={`${position.symbol}-${index}`} className="bg-dark-200 rounded-lg p-3">
              {/* Header */}
              <div className="flex justify-between items-start mb-2">
                <div className="flex items-center space-x-2">
                  <span className="text-white font-medium">{position.symbol}</span>
                  <span className={`text-xs px-2 py-1 rounded ${
                    position.side === 'LONG' ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
                  }`}>
                    {position.side}
                  </span>
                  <span className={`text-xs px-2 py-1 rounded ${
                    position.status === 'ACTIVE' ? 'bg-blue-900 text-blue-300' : 'bg-gray-900 text-gray-300'
                  }`}>
                    {position.status}
                  </span>
                </div>
                <div className="text-right">
                  <div className={`text-sm font-medium ${
                    position.pnl >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {formatCurrency(position.pnl)}
                  </div>
                  <div className={`text-xs ${
                    position.pnl >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {formatPercentage(position.pnl_percentage)}
                  </div>
                </div>
              </div>

              {/* Details */}
              <div className="grid grid-cols-2 gap-4 text-xs">
                <div>
                  <div className="text-gray-400">Quantité</div>
                  <div className="text-white">{position.quantity}</div>
                </div>
                <div>
                  <div className="text-gray-400">Prix d'entrée</div>
                  <div className="text-white">{formatCurrency(position.entry_price)}</div>
                </div>
                <div>
                  <div className="text-gray-400">Prix actuel</div>
                  <div className="text-white">{formatCurrency(position.current_price)}</div>
                </div>
                <div>
                  <div className="text-gray-400">Valeur</div>
                  <div className="text-white">{formatCurrency(position.value)}</div>
                </div>
              </div>

              {/* Footer */}
              <div className="flex justify-between items-center mt-2 pt-2 border-t border-gray-700">
                <div className="text-xs text-gray-400">
                  Ouvert il y a {formatTimeAgo(position.timestamp)}
                </div>
                <div className="text-xs text-gray-400">
                  Margin: {formatCurrency(position.margin_used)}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default PositionsPanel;