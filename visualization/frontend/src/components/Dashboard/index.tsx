import React from 'react';
import Controls from '@/components/Controls';
import MarketChart from '@/components/Charts/MarketChart';
import VolumeChart from '@/components/Charts/VolumeChart';
import RSIChart from '@/components/Charts/RSIChart';
import MACDChart from '@/components/Charts/MACDChart';
import PerformanceChart from '@/components/Charts/PerformanceChart';
import { GlobalStatusMessage } from '@/components/Common/StatusMessage';
import PortfolioPanel from '@/components/Trading/PortfolioPanel';
import PositionsPanel from '@/components/Trading/PositionsPanel';
import TradeHistoryPanel from '@/components/Trading/TradeHistoryPanel';
import AlertsPanel from '@/components/Trading/AlertsPanel';
import RecentOrdersPanel from '@/components/Trading/RecentOrdersPanel';
import MultiTimeframeChart from '@/components/Charts/MultiTimeframeChart';
import OrderBookPanel from '@/components/Trading/OrderBookPanel';
import { useChart } from '@/hooks/useChart';

function Dashboard() {
  const { isLoading } = useChart({
    autoUpdate: true,
    updateInterval: 10000,
    enableWebSocket: true,
  });
  
  return (
    <div className="space-y-6">
      <Controls />
      
      <GlobalStatusMessage />
      
      {/* Layout ultrawide avec grille optimisée */}
      <div className="grid grid-cols-12 gap-6">
        {/* Colonne gauche - Graphiques principaux */}
        <div className="col-span-12 lg:col-span-8 xl:col-span-9 2xl:col-span-8 3xl:col-span-9 space-y-6">
          {/* Graphique principal */}
          <div className="chart-container">
            <div className="chart-title">
              📈 Graphique de Marché
            </div>
            <MarketChart />
          </div>
          
          {/* Graphique de volume */}
          <div className="chart-container">
            <div className="chart-title">
              📉 Volume
            </div>
            <VolumeChart />
          </div>
          
          {/* Grille des indicateurs */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="chart-container">
              <div className="chart-title">
                📉 RSI (Relative Strength Index)
              </div>
              <RSIChart />
            </div>
            
            <div className="chart-container">
              <div className="chart-title">
                📉 MACD
              </div>
              <MACDChart />
            </div>
          </div>

          {/* Timeframes multiples */}
          <div className="chart-container">
            <div className="chart-title">
              🕐 Timeframes Multiples
            </div>
            <MultiTimeframeChart />
          </div>
          
          {/* Graphique de performance */}
          <div className="chart-container">
            <div className="chart-title">
              💰 Performance du Portfolio
            </div>
            <PerformanceChart />
          </div>
        </div>
        
        {/* Colonne droite - Métriques et informations */}
        <div className="col-span-12 lg:col-span-4 xl:col-span-3 2xl:col-span-4 3xl:col-span-3 space-y-6">
          {/* Panel Trading & Portfolio */}
          <div className="chart-container">
            <div className="chart-title">
              💼 Portfolio & Trading
            </div>
            <PortfolioPanel />
          </div>
          
          {/* Panel Positions */}
          <div className="chart-container">
            <div className="chart-title">
              📊 Positions Actuelles
            </div>
            <PositionsPanel />
          </div>
          
          {/* Panel Historique */}
          <div className="chart-container">
            <div className="chart-title">
              📈 Historique Trades
            </div>
            <TradeHistoryPanel />
          </div>
          
          {/* Panel Alertes */}
          <div className="chart-container">
            <div className="chart-title">
              🚨 Alertes Système
            </div>
            <AlertsPanel />
          </div>

          {/* Panel OrderBook */}
          <div className="chart-container">
            <div className="chart-title">
              📋 Carnet d'Ordres
            </div>
            <OrderBookPanel />
          </div>

          {/* Panel Ordres Récents */}
          <div className="chart-container">
            <div className="chart-title">
              ⚡ Ordres Récents
            </div>
            <RecentOrdersPanel />
          </div>
        </div>
      </div>
      
      {/* Overlay de loading */}
      {isLoading && (
        <div className="fixed inset-0 bg-black/20 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-dark-200 border border-gray-700 rounded-lg p-6 flex items-center space-x-3">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-500"></div>
            <span className="text-white font-medium">Mise à jour des données...</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default Dashboard;