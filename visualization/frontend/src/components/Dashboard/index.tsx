import React from 'react';
import Controls from '@/components/Controls';
import MarketChart from '@/components/Charts/MarketChart';
import VolumeChart from '@/components/Charts/VolumeChart';
import RSIChart from '@/components/Charts/RSIChart';
import MACDChart from '@/components/Charts/MACDChart';
import PerformanceChart from '@/components/Charts/PerformanceChart';
import { GlobalStatusMessage } from '@/components/Common/StatusMessage';
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
      
      {/* Graphique de performance */}
      <div className="chart-container">
        <div className="chart-title">
          💰 Performance du Portfolio
        </div>
        <PerformanceChart />
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