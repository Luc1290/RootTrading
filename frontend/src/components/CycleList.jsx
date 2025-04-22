import React from 'react';
import { ArrowUp, ArrowDown, Clock } from 'lucide-react';
import { Link } from 'react-router-dom';

const CycleList = ({ cycles = [], limit, showAll = false }) => {
  // Format currency values
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', { 
      style: 'currency', 
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  // Format percent values
  const formatPercent = (value) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  // Status badge color
  const getStatusColor = (status) => {
    switch (status) {
      case 'initiating':
      case 'waiting_buy':
      case 'waiting_sell':
        return 'bg-yellow-100 text-yellow-800';
      case 'active_buy':
      case 'active_sell':
        return 'bg-blue-100 text-blue-800';
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'canceled':
      case 'failed':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  // Handle empty state
  if (!cycles || cycles.length === 0) {
    return (
      <div className="text-center p-6 text-gray-500">
        No trading cycles found
      </div>
    );
  }

  // Limit the number of cycles to display if specified
  const displayCycles = limit ? cycles.slice(0, limit) : cycles;

  return (
    <div className="overflow-hidden overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Symbol</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Strategy</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Price</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Quantity</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">P&L</th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Time</th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {displayCycles.map((cycle) => {
            // Calculate P&L for active cycles
            let pnl = cycle.profit_loss;
            let pnlPercent = cycle.profit_loss_percent;
            
            if (['active_buy', 'waiting_sell'].includes(cycle.status) && cycle.current_price) {
              // For active buy positions: current_price - entry_price
              pnl = (cycle.current_price - cycle.entry_price) * cycle.quantity;
              pnlPercent = ((cycle.current_price - cycle.entry_price) / cycle.entry_price) * 100;
            } else if (['active_sell', 'waiting_buy'].includes(cycle.status) && cycle.current_price) {
              // For active sell positions: entry_price - current_price
              pnl = (cycle.entry_price - cycle.current_price) * cycle.quantity;
              pnlPercent = ((cycle.entry_price - cycle.current_price) / cycle.entry_price) * 100;
            }
            
            return (
              <tr key={cycle.id}>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                  {cycle.symbol}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {cycle.strategy}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${getStatusColor(cycle.status)}`}>
                    {cycle.status}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {cycle.entry_price ? formatCurrency(cycle.entry_price) : '-'}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  {cycle.quantity}
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  {pnl !== null && pnlPercent !== null ? (
                    <div className={`flex items-center ${pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {pnl >= 0 ? <ArrowUp size={16} className="mr-1" /> : <ArrowDown size={16} className="mr-1" />}
                      {formatCurrency(pnl)}
                      <span className="ml-1 text-xs">
                        ({formatPercent(pnlPercent)})
                      </span>
                    </div>
                  ) : (
                    <span className="text-gray-400">-</span>
                  )}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                  <div className="flex items-center">
                    <Clock size={14} className="mr-1" />
                    {new Date(cycle.created_at).toLocaleTimeString()}
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      
      {/* Show link to all cycles if limited */}
      {limit && cycles.length > limit && !showAll && (
        <div className="px-6 py-3 bg-gray-50 text-right">
          <Link to="/trades" className="text-blue-600 hover:text-blue-800 text-sm font-medium">
            View all {cycles.length} cycles →
          </Link>
        </div>
      )}
    </div>
  );
};

export default CycleList;