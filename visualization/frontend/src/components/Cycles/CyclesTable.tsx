import React from 'react';
import { format } from 'date-fns';
import { fr } from 'date-fns/locale';
import type { TradeCycle } from './CyclesPage';

interface CyclesTableProps {
  cycles: TradeCycle[];
  isLoading: boolean;
}

function CyclesTable({ cycles, isLoading }: CyclesTableProps) {
  if (isLoading) {
    return (
      <div className="bg-dark-200 border border-gray-700 rounded-lg p-8">
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500"></div>
        </div>
      </div>
    );
  }

  if (cycles.length === 0) {
    return (
      <div className="bg-dark-200 border border-gray-700 rounded-lg p-8">
        <p className="text-gray-400 text-center">Aucun cycle trouvé pour les critères sélectionnés</p>
      </div>
    );
  }

  const getStatusBadge = (status: TradeCycle['status']) => {
    const statusConfig = {
      'active_buy': { color: 'bg-blue-500/20 text-blue-400 border-blue-500/30', label: '🔵 BUY Actif' },
      'active_sell': { color: 'bg-orange-500/20 text-orange-400 border-orange-500/30', label: '🟠 SELL Actif' },
      'completed': { color: 'bg-green-500/20 text-green-400 border-green-500/30', label: '✅ Complété' },
      'cancelled': { color: 'bg-gray-500/20 text-gray-400 border-gray-500/30', label: '❌ Annulé' },
    };

    const config = statusConfig[status] || statusConfig.cancelled;
    
    return (
      <span className={`px-2 py-1 text-xs font-medium rounded-full border ${config.color}`}>
        {config.label}
      </span>
    );
  };

  const formatPrice = (price?: number) => {
    if (!price) return '-';
    return price.toFixed(price < 10 ? 6 : 2);
  };

  const formatProfitLoss = (profitLoss?: number, profitLossPercent?: number) => {
    if (!profitLoss) return '-';
    
    const isPositive = profitLoss > 0;
    const color = isPositive ? 'text-green-400' : 'text-red-400';
    const sign = isPositive ? '+' : '';
    
    return (
      <div className={color}>
        <div>{sign}{profitLoss.toFixed(6)} USDC</div>
        {profitLossPercent && (
          <div className="text-xs">{sign}{profitLossPercent.toFixed(2)}%</div>
        )}
      </div>
    );
  };

  return (
    <div className="bg-dark-200 border border-gray-700 rounded-lg overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-700">
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                Date
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                Symbole
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                Stratégie
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                Statut
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                Prix Entrée
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                Prix Sortie
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                Quantité
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                P&L
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                Durée
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-700">
            {cycles.map((cycle) => {
              const duration = cycle.completed_at 
                ? new Date(cycle.completed_at).getTime() - new Date(cycle.created_at).getTime()
                : Date.now() - new Date(cycle.created_at).getTime();
              
              const hours = Math.floor(duration / (1000 * 60 * 60));
              const minutes = Math.floor((duration % (1000 * 60 * 60)) / (1000 * 60));
              
              return (
                <tr key={cycle.id} className="hover:bg-dark-300 transition-colors">
                  <td className="px-4 py-3 text-sm text-gray-300">
                    <div>
                      {format(new Date(cycle.created_at), 'dd MMM HH:mm', { locale: fr })}
                    </div>
                    {cycle.completed_at && (
                      <div className="text-xs text-gray-500">
                        → {format(new Date(cycle.completed_at), 'HH:mm', { locale: fr })}
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-3 text-sm font-medium text-white">
                    {cycle.symbol}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-300">
                    {cycle.strategy}
                  </td>
                  <td className="px-4 py-3">
                    {getStatusBadge(cycle.status)}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-300">
                    {formatPrice(cycle.entry_price)}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-300">
                    {formatPrice(cycle.exit_price)}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-300">
                    {cycle.quantity ? cycle.quantity.toFixed(8) : '-'}
                  </td>
                  <td className="px-4 py-3 text-sm">
                    {formatProfitLoss(cycle.profit_loss, cycle.profit_loss_percent)}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-300">
                    {hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default CyclesTable;