import React from 'react';
import { formatTime } from '@/utils';
import { useChartStore } from '@/stores/useChartStore';

function Header() {
  const { lastUpdate, isLoading } = useChartStore();
  
  return (
    <header className="gradient-header shadow-lg">
      <div className="container mx-auto px-4 py-6 max-w-7xl">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <span className="text-3xl">🚀</span>
              <div>
                <h1 className="text-2xl font-bold text-white text-shadow-lg">
                  RootTrading
                </h1>
                <p className="text-primary-100 text-sm opacity-90">
                  Visualisation en temps réel des données de trading
                </p>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${
                isLoading ? 'bg-yellow-400 animate-pulse' : 'bg-green-400'
              }`} />
              <span className="text-sm text-primary-100">
                {isLoading ? '🟡 UPDATING...' : '🟢 LIVE'}
              </span>
            </div>
            
            {lastUpdate && (
              <div className="text-xs text-primary-200 opacity-75">
                Dernière mise à jour: {formatTime(lastUpdate.toISOString())}
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}

export default Header;