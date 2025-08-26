import React, { memo, useState, useMemo, useEffect } from 'react';
import Controls from '@/components/Controls';
import MarketChart from '@/components/Charts/MarketChart';
import VolumeChart from '@/components/Charts/VolumeChart';
import RSIChart from '@/components/Charts/RSIChart';
import MACDChart from '@/components/Charts/MACDChart';
import VWAPChart from '@/components/Charts/VWAPChart';
import RegimeInfo from '@/components/Charts/RegimeInfo';
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
import toast from 'react-hot-toast';

// Composants memoizés pour optimiser les performances
const MemoizedMarketChart = memo(MarketChart);
const MemoizedVolumeChart = memo(VolumeChart);
const MemoizedRSIChart = memo(RSIChart);
const MemoizedMACDChart = memo(MACDChart);
const MemoizedVWAPChart = memo(VWAPChart);
const MemoizedPerformanceChart = memo(PerformanceChart);
const MemoizedMultiTimeframeChart = memo(MultiTimeframeChart);

// Composants de trading memoizés
const MemoizedRegimeInfo = memo(RegimeInfo);
const MemoizedPortfolioPanel = memo(PortfolioPanel);
const MemoizedPositionsPanel = memo(PositionsPanel);
const MemoizedTradeHistoryPanel = memo(TradeHistoryPanel);
const MemoizedAlertsPanel = memo(AlertsPanel);
const MemoizedOrderBookPanel = memo(OrderBookPanel);
const MemoizedRecentOrdersPanel = memo(RecentOrdersPanel);

function Dashboard() {
  const [activeTab, setActiveTab] = useState('main');
  
  const { isLoading, apiError, marketData } = useChart({
    autoUpdate: true,
    updateInterval: 30000, // 30s au lieu de 10s pour réduire la charge
    enableWebSocket: true,
  });
  
  // Afficher les erreurs API
  useEffect(() => {
    if (apiError) {
      toast.error(`Erreur API: ${apiError}`, {
        duration: 5000,
        position: 'top-center'
      });
    }
  }, [apiError]);
  
  // Log de débogage pour les données
  useEffect(() => {
    console.log('Dashboard - Market data status:', {
      hasMarketData: !!marketData,
      timestamps: marketData?.timestamps?.length || 0,
      activeTab
    });
  }, [marketData, activeTab]);
  
  // Contenu des onglets memoizé pour optimiser les performances
  const tabContent = useMemo(() => {
    switch (activeTab) {
      case 'main':
        return (
          <div className="grid grid-cols-12 gap-6">
            {/* Colonne gauche - Graphiques essentiels */}
            <div className="col-span-12 lg:col-span-8 xl:col-span-9 space-y-6">
              {/* Graphique principal */}
              <div className="chart-container">
                <div className="chart-title">📈 Graphique de Marché</div>
                <MemoizedMarketChart />
              </div>
              
              {/* Graphique de volume */}
              <div className="chart-container">
                <div className="chart-title">📉 Volume</div>
                <MemoizedVolumeChart />
              </div>
              
              {/* Indicateurs essentiels */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="chart-container">
                  <div className="chart-title">📉 RSI</div>
                  <MemoizedRSIChart />
                </div>
                <div className="chart-container">
                  <div className="chart-title">📉 MACD</div>
                  <MemoizedMACDChart />
                </div>
              </div>
            </div>
            
            {/* Colonne droite - Infos essentielles */}
            <div className="col-span-12 lg:col-span-4 xl:col-span-3 space-y-6">
              <div className="chart-container">
                <div className="chart-title">🎯 Market Regime</div>
                <MemoizedRegimeInfo />
              </div>
              <div className="chart-container">
                <div className="chart-title">💼 Portfolio</div>
                <MemoizedPortfolioPanel />
              </div>
              <div className="chart-container">
                <div className="chart-title">📊 Positions</div>
                <MemoizedPositionsPanel />
              </div>
            </div>
          </div>
        );
        
      case 'advanced':
        return (
          <div className="grid grid-cols-12 gap-6">
            <div className="col-span-12 lg:col-span-8 space-y-6">
              <div className="chart-container">
                <div className="chart-title">📈 VWAP Analysis</div>
                <MemoizedVWAPChart />
              </div>
              <div className="chart-container">
                <div className="chart-title">🕐 Timeframes Multiples</div>
                <MemoizedMultiTimeframeChart />
              </div>
              <div className="chart-container">
                <div className="chart-title">💰 Performance</div>
                <MemoizedPerformanceChart />
              </div>
            </div>
            <div className="col-span-12 lg:col-span-4 space-y-6">
              <div className="chart-container">
                <div className="chart-title">📈 Historique Trades</div>
                <MemoizedTradeHistoryPanel />
              </div>
              <div className="chart-container">
                <div className="chart-title">🚨 Alertes</div>
                <MemoizedAlertsPanel />
              </div>
            </div>
          </div>
        );
        
      case 'trading':
        return (
          <div className="grid grid-cols-12 gap-6">
            <div className="col-span-12 lg:col-span-6 space-y-6">
              <div className="chart-container">
                <div className="chart-title">📋 Carnet d'Ordres</div>
                <MemoizedOrderBookPanel />
              </div>
            </div>
            <div className="col-span-12 lg:col-span-6 space-y-6">
              <div className="chart-container">
                <div className="chart-title">⚡ Ordres Récents</div>
                <MemoizedRecentOrdersPanel />
              </div>
            </div>
          </div>
        );
        
      default:
        return null;
    }
  }, [activeTab]);

  return (
    <div className="space-y-6">
      <Controls />
      <GlobalStatusMessage />
      
      {/* Navigation par onglets pour réduire la charge */}
      <div className="border-b border-gray-700">
        <nav className="flex space-x-8">
          {[
            { id: 'main', label: '📊 Principal', desc: 'Graphiques essentiels' },
            { id: 'advanced', label: '📈 Avancé', desc: 'Analyses approfondies' },
            { id: 'trading', label: '💰 Trading', desc: 'Ordres et carnet' }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                activeTab === tab.id
                  ? 'border-primary-500 text-primary-500'
                  : 'border-transparent text-gray-400 hover:text-gray-300'
              }`}
            >
              <div>{tab.label}</div>
              <div className="text-xs opacity-75">{tab.desc}</div>
            </button>
          ))}
        </nav>
      </div>
      
      {/* Contenu de l'onglet actif */}
      {tabContent}
      
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