import React from 'react';
import type { TradeCycle } from './CyclesPage';

interface CyclesStatsProps {
  cycles: TradeCycle[];
}

function CyclesStats({ cycles }: CyclesStatsProps) {
  // Calculer les statistiques
  const completedCycles = cycles.filter(c => c.status === 'completed');
  const activeCycles = cycles.filter(c => c.status.startsWith('active'));
  
  const totalProfitLoss = completedCycles.reduce((sum, c) => sum + (c.profit_loss || 0), 0);
  const winningCycles = completedCycles.filter(c => (c.profit_loss || 0) > 0);
  const losingCycles = completedCycles.filter(c => (c.profit_loss || 0) < 0);
  
  const winRate = completedCycles.length > 0 
    ? (winningCycles.length / completedCycles.length) * 100 
    : 0;
  
  const avgProfitPercent = completedCycles.length > 0
    ? completedCycles.reduce((sum, c) => sum + (c.profit_loss_percent || 0), 0) / completedCycles.length
    : 0;

  // Calculer les stats par symbole
  const statsBySymbol = cycles.reduce((acc, cycle) => {
    if (!acc[cycle.symbol]) {
      acc[cycle.symbol] = {
        total: 0,
        completed: 0,
        active: 0,
        profitLoss: 0,
        wins: 0,
        losses: 0
      };
    }
    
    acc[cycle.symbol].total++;
    
    if (cycle.status === 'completed') {
      acc[cycle.symbol].completed++;
      acc[cycle.symbol].profitLoss += cycle.profit_loss || 0;
      
      if ((cycle.profit_loss || 0) > 0) {
        acc[cycle.symbol].wins++;
      } else if ((cycle.profit_loss || 0) < 0) {
        acc[cycle.symbol].losses++;
      }
    } else if (cycle.status.startsWith('active')) {
      acc[cycle.symbol].active++;
    }
    
    return acc;
  }, {} as Record<string, any>);

  // Trier par profit décroissant
  const topSymbols = Object.entries(statsBySymbol)
    .map(([symbol, stats]) => ({ symbol, ...stats }))
    .sort((a, b) => b.profitLoss - a.profitLoss)
    .slice(0, 5);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Statistiques globales */}
      <div className="bg-dark-200 border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">📊 Statistiques Globales</h3>
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-gray-400">Total Cycles</p>
            <p className="text-2xl font-bold text-white">{cycles.length}</p>
          </div>
          
          <div>
            <p className="text-sm text-gray-400">Cycles Actifs</p>
            <p className="text-2xl font-bold text-blue-400">{activeCycles.length}</p>
          </div>
          
          <div>
            <p className="text-sm text-gray-400">P&L Total</p>
            <p className={`text-2xl font-bold ${totalProfitLoss >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {totalProfitLoss >= 0 ? '+' : ''}{totalProfitLoss.toFixed(2)} USDC
            </p>
          </div>
          
          <div>
            <p className="text-sm text-gray-400">Win Rate</p>
            <p className="text-2xl font-bold text-white">{winRate.toFixed(1)}%</p>
          </div>
          
          <div>
            <p className="text-sm text-gray-400">Trades Gagnants</p>
            <p className="text-xl font-semibold text-green-400">{winningCycles.length}</p>
          </div>
          
          <div>
            <p className="text-sm text-gray-400">Trades Perdants</p>
            <p className="text-xl font-semibold text-red-400">{losingCycles.length}</p>
          </div>
          
          <div className="col-span-2">
            <p className="text-sm text-gray-400">Profit Moyen</p>
            <p className={`text-xl font-semibold ${avgProfitPercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {avgProfitPercent >= 0 ? '+' : ''}{avgProfitPercent.toFixed(2)}%
            </p>
          </div>
        </div>
      </div>

      {/* Top 5 Symboles */}
      <div className="bg-dark-200 border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">🏆 Top 5 Symboles</h3>
        
        <div className="space-y-3">
          {topSymbols.map((item, index) => (
            <div key={item.symbol} className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <span className="text-lg font-bold text-gray-500">#{index + 1}</span>
                <div>
                  <p className="font-medium text-white">{item.symbol}</p>
                  <p className="text-sm text-gray-400">
                    {item.completed} trades ({item.wins}W/{item.losses}L)
                  </p>
                </div>
              </div>
              
              <div className="text-right">
                <p className={`font-semibold ${item.profitLoss >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {item.profitLoss >= 0 ? '+' : ''}{item.profitLoss.toFixed(4)} USDC
                </p>
                {item.active > 0 && (
                  <p className="text-xs text-blue-400">{item.active} actif(s)</p>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default CyclesStats;