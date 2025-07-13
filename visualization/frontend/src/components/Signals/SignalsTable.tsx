import React from 'react';
import { format } from 'date-fns';
import { fr } from 'date-fns/locale';
import type { TradingSignal, TradingSymbol } from '@/types';

interface ExtendedTradingSignal extends TradingSignal {
  symbol: TradingSymbol;
  type: 'buy' | 'sell';
}

interface SignalsTableProps {
  signals: ExtendedTradingSignal[];
  isLoading: boolean;
}

function SignalsTable({ signals, isLoading }: SignalsTableProps) {
  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('fr-FR', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 6,
    }).format(price);
  };

  const formatStrength = (strength: number | string | undefined | null) => {
    if (strength === undefined || strength === null) {
      return 'N/A';
    }
    
    // Si c'est une string (very_strong, strong, etc.)
    if (typeof strength === 'string') {
      const strengthMap = {
        'very_strong': 'Très Fort',
        'strong': 'Fort', 
        'moderate': 'Modéré',
        'weak': 'Faible'
      };
      return strengthMap[strength as keyof typeof strengthMap] || strength;
    }
    
    // Si c'est un nombre
    if (typeof strength === 'number' && !isNaN(strength)) {
      return `${(strength * 100).toFixed(1)}%`;
    }
    
    return 'N/A';
  };

  const getTypeColor = (type: 'buy' | 'sell') => {
    return type === 'buy' ? 'text-green-400' : 'text-red-400';
  };

  const getTypeIcon = (type: 'buy' | 'sell') => {
    return type === 'buy' ? '🟢' : '🔴';
  };

  const getStrengthColor = (strength: number | string | undefined | null) => {
    if (strength === undefined || strength === null) {
      return 'text-gray-400';
    }
    
    // Si c'est une string
    if (typeof strength === 'string') {
      switch (strength) {
        case 'very_strong': return 'text-green-400';
        case 'strong': return 'text-green-300';
        case 'moderate': return 'text-yellow-400';
        case 'weak': return 'text-red-400';
        default: return 'text-gray-400';
      }
    }
    
    // Si c'est un nombre
    if (typeof strength === 'number' && !isNaN(strength)) {
      if (strength >= 0.8) return 'text-green-400';
      if (strength >= 0.6) return 'text-yellow-400';
      return 'text-red-400';
    }
    
    return 'text-gray-400';
  };

  if (isLoading) {
    return (
      <div className="bg-dark-200 border border-gray-700 rounded-lg p-8">
        <div className="flex items-center justify-center space-x-3">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-500"></div>
          <span className="text-gray-400">Chargement des signaux...</span>
        </div>
      </div>
    );
  }

  if (signals.length === 0) {
    return (
      <div className="bg-dark-200 border border-gray-700 rounded-lg p-8">
        <div className="text-center">
          <div className="text-4xl mb-4">📊</div>
          <h3 className="text-lg font-medium text-white mb-2">
            Aucun signal trouvé
          </h3>
          <p className="text-gray-400">
            Aucun signal ne correspond aux filtres sélectionnés pour cette période.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-dark-200 border border-gray-700 rounded-lg overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-dark-100 border-b border-gray-700">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                Heure
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                Paire
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                Type
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                Prix
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                Stratégie
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                Force
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-700">
            {signals.map((signal, index) => (
              <tr
                key={`${signal.symbol}-${signal.timestamp}-${index}`}
                className="hover:bg-dark-100/50 transition-colors"
              >
                <td className="px-6 py-4 whitespace-nowrap text-sm text-white">
                  <div>
                    <div className="font-medium">
                      {format(new Date(signal.timestamp), 'HH:mm:ss', { locale: fr })}
                    </div>
                    <div className="text-xs text-gray-400">
                      {format(new Date(signal.timestamp), 'dd/MM/yyyy', { locale: fr })}
                    </div>
                  </div>
                </td>
                
                <td className="px-6 py-4 whitespace-nowrap text-sm text-white">
                  <span className="font-mono font-medium bg-dark-100 px-2 py-1 rounded">
                    {signal.symbol}
                  </span>
                </td>
                
                <td className="px-6 py-4 whitespace-nowrap text-sm">
                  <div className={`flex items-center space-x-2 ${getTypeColor(signal.type)}`}>
                    <span>{getTypeIcon(signal.type)}</span>
                    <span className="font-medium uppercase">
                      {signal.type}
                    </span>
                  </div>
                </td>
                
                <td className="px-6 py-4 whitespace-nowrap text-sm text-white">
                  <span className="font-mono">
                    ${formatPrice(signal.price)}
                  </span>
                </td>
                
                <td className="px-6 py-4 whitespace-nowrap text-sm text-white">
                  <span className="bg-primary-500/20 text-primary-300 px-2 py-1 rounded text-xs font-medium">
                    {signal.strategy}
                  </span>
                </td>
                
                <td className="px-6 py-4 whitespace-nowrap text-sm">
                  <div className="flex items-center space-x-2">
                    <div className={`font-medium ${getStrengthColor(signal.strength)}`}>
                      {formatStrength(signal.strength)}
                    </div>
                    {signal.strength !== undefined && signal.strength !== null ? (
                      <div className="w-16 bg-gray-700 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full transition-all ${
                            typeof signal.strength === 'string' ? (
                              signal.strength === 'very_strong' ? 'bg-green-400' :
                              signal.strength === 'strong' ? 'bg-green-300' :
                              signal.strength === 'moderate' ? 'bg-yellow-400' : 'bg-red-400'
                            ) : (
                              signal.strength >= 0.8 ? 'bg-green-400' :
                              signal.strength >= 0.6 ? 'bg-yellow-400' : 'bg-red-400'
                            )
                          }`}
                          style={{ 
                            width: typeof signal.strength === 'string' ? (
                              signal.strength === 'very_strong' ? '100%' :
                              signal.strength === 'strong' ? '80%' :
                              signal.strength === 'moderate' ? '60%' : '40%'
                            ) : `${signal.strength * 100}%`
                          }}
                        />
                      </div>
                    ) : (
                      <div className="w-16 bg-gray-700 rounded-full h-2">
                        <div className="w-full h-2 bg-gray-500 rounded-full opacity-50" />
                      </div>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {/* Footer avec statistiques */}
      <div className="bg-dark-100 px-6 py-3 border-t border-gray-700">
        <div className="flex items-center justify-between text-sm text-gray-400">
          <div>
            Total: {signals.length} signaux
          </div>
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-1">
              <span className="text-green-400">🟢</span>
              <span>BUY: {signals.filter(s => s.type === 'buy').length}</span>
            </div>
            <div className="flex items-center space-x-1">
              <span className="text-red-400">🔴</span>
              <span>SELL: {signals.filter(s => s.type === 'sell').length}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SignalsTable;