import React, { useState, useEffect } from 'react';
import { apiService } from '@/services/api';

interface Trade {
  id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  total_value: number;
  pnl: number;
  pnl_percentage: number;
  strategy: string;
  timestamp: string;
  status: 'COMPLETED' | 'PENDING' | 'FAILED';
  fees: number;
}

function TradeHistoryPanel() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchTradeHistory();
    const interval = setInterval(fetchTradeHistory, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchTradeHistory = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Récupérer les vraies données des trades depuis le portfolio
      const tradesResponse = await apiService.getTradeHistory(1, 10);
      
      // Transformer les données pour le format attendu
      const realTrades: Trade[] = tradesResponse.trades.map((trade: any) => {
        const isCompleted = trade.status === 'completed';
        const side = trade.exit_price ? 'SELL' : 'BUY';
        const price = parseFloat(trade.exit_price || trade.entry_price || '0');
        const quantity = parseFloat(trade.quantity || '0');
        const pnl = parseFloat(trade.profit_loss || '0');
        const pnlPercent = parseFloat(trade.profit_loss_percent || '0');
        
        return {
          id: trade.id,
          symbol: trade.symbol,
          side: side as 'BUY' | 'SELL',
          quantity: quantity,
          price: price,
          total_value: quantity * price,
          pnl: pnl,
          pnl_percentage: pnlPercent,
          strategy: trade.strategy,
          timestamp: trade.created_at,
          status: isCompleted ? 'COMPLETED' : 'PENDING',
          fees: (quantity * price) * 0.001 // Approximation des frais 0.1%
        };
      });

      setTrades(realTrades);
    } catch (err) {
      setError('Erreur lors du chargement de l\'historique');
      console.error('Trade history fetch error:', err);
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
    if (value === 0) return '0.00%';
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
      <div className="space-y-2">
        {trades.length === 0 ? (
          <div className="text-center text-gray-400 py-8">
            Aucun trade récent
          </div>
        ) : (
          trades.map((trade) => (
            <div key={trade.id} className="bg-dark-200 rounded-lg p-3">
              {/* Header */}
              <div className="flex justify-between items-start mb-2">
                <div className="flex items-center space-x-2">
                  <span className="text-white font-medium">{trade.symbol}</span>
                  <span className={`text-xs px-2 py-1 rounded ${
                    trade.side === 'BUY' ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
                  }`}>
                    {trade.side}
                  </span>
                  <span className={`text-xs px-2 py-1 rounded ${
                    trade.status === 'COMPLETED' ? 'bg-blue-900 text-blue-300' : 
                    trade.status === 'PENDING' ? 'bg-yellow-900 text-yellow-300' : 
                    'bg-red-900 text-red-300'
                  }`}>
                    {trade.status}
                  </span>
                </div>
                <div className="text-right">
                  <div className="text-xs text-gray-400">
                    {formatTimeAgo(trade.timestamp)}
                  </div>
                </div>
              </div>

              {/* Details */}
              <div className="grid grid-cols-3 gap-2 text-xs mb-2">
                <div>
                  <div className="text-gray-400">Quantité</div>
                  <div className="text-white">{trade.quantity}</div>
                </div>
                <div>
                  <div className="text-gray-400">Prix</div>
                  <div className="text-white">{formatCurrency(trade.price)}</div>
                </div>
                <div>
                  <div className="text-gray-400">Valeur</div>
                  <div className="text-white">{formatCurrency(trade.total_value)}</div>
                </div>
              </div>

              {/* P&L and Strategy */}
              <div className="flex justify-between items-center">
                <div className="text-xs text-gray-400">
                  {trade.strategy}
                </div>
                <div className="text-right">
                  {trade.pnl !== 0 && (
                    <>
                      <div className={`text-xs font-medium ${
                        trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {formatCurrency(trade.pnl)}
                      </div>
                      <div className={`text-xs ${
                        trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {formatPercentage(trade.pnl_percentage)}
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* Fees */}
              <div className="text-xs text-gray-400 mt-1">
                Frais: {formatCurrency(trade.fees)}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default TradeHistoryPanel;