import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { formatTime } from '@/utils';
import { useChartStore } from '@/stores/useChartStore';

function Header() {
  const { lastUpdate, isLoading } = useChartStore();
  const location = useLocation();
  
  return (
    <header className="gradient-header shadow-lg">
      <div className="container mx-auto px-4 py-6 max-w-none">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-3">
              <img 
                src="/assets/logo.svg" 
                alt="Root Logo" 
                className="w-64 h-36"
              />
              <div>
                <p className="text-primary-100 text-sm opacity-90">
                  Visualisation en temps réel des données de trading
                </p>
              </div>
            </div>
            
            <nav className="flex items-center space-x-4">
              <Link
                to="/"
                className={`px-4 py-2 rounded-lg transition-colors text-sm font-medium ${
                  location.pathname === '/'
                    ? 'bg-primary-500 text-white'
                    : 'text-primary-100 hover:bg-primary-500/20 hover:text-white'
                }`}
              >
                📈 Dashboard
              </Link>
              <Link
                to="/manual-trading"
                className={`px-4 py-2 rounded-lg transition-colors text-sm font-medium ${
                  location.pathname === '/manual-trading'
                    ? 'bg-primary-500 text-white'
                    : 'text-primary-100 hover:bg-primary-500/20 hover:text-white'
                }`}
              >
                🎯 Trading Manuel
              </Link>
              <Link
                to="/position-tracker"
                className={`px-4 py-2 rounded-lg transition-colors text-sm font-medium ${
                  location.pathname === '/position-tracker'
                    ? 'bg-primary-500 text-white'
                    : 'text-primary-100 hover:bg-primary-500/20 hover:text-white'
                }`}
              >
                📊 Positions
              </Link>
              <Link
                to="/signals"
                className={`px-4 py-2 rounded-lg transition-colors text-sm font-medium ${
                  location.pathname === '/signals'
                    ? 'bg-primary-500 text-white'
                    : 'text-primary-100 hover:bg-primary-500/20 hover:text-white'
                }`}
              >
                📊 Signaux
              </Link>
              <Link
                to="/cycles"
                className={`px-4 py-2 rounded-lg transition-colors text-sm font-medium ${
                  location.pathname === '/cycles'
                    ? 'bg-primary-500 text-white'
                    : 'text-primary-100 hover:bg-primary-500/20 hover:text-white'
                }`}
              >
                🔄 Cycles
              </Link>
              <Link
                to="/statistics"
                className={`px-4 py-2 rounded-lg transition-colors text-sm font-medium ${
                  location.pathname === '/statistics'
                    ? 'bg-primary-500 text-white'
                    : 'text-primary-100 hover:bg-primary-500/20 hover:text-white'
                }`}
              >
                📊 Statistiques
              </Link>
            </nav>
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