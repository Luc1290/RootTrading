import React, { useState, useEffect } from 'react';
import { ArrowUp, ArrowDown, TrendingUp, AlertTriangle, Activity } from 'lucide-react';
import SignalChart from './SignalChart';
import CycleList from './CycleList';
import usePortfolio from '../api/usePortfolio';
import useCycle from '../api/useCycle';
import useSignals from '../api/useSignals';
import useLiveStats from '../hooks/useLiveStats';

const Dashboard = () => {
  const { summary, loading: portfolioLoading, error: portfolioError } = usePortfolio();
  const { cycles, loading: cyclesLoading, error: cyclesError } = useCycle();
  const { signals, loading: signalsLoading, error: signalsError } = useSignals();
  const { stats, loading: statsLoading } = useLiveStats();
  
  // Format currency values
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', { 
      style: 'currency', 
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  // Format percentage values
  const formatPercent = (value) => {
    return new Intl.NumberFormat('en-US', { 
      style: 'percent',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value / 100);
  };

  if (portfolioLoading || cyclesLoading || signalsLoading) {
    return (
      <div className="flex justify-center items-center h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (portfolioError || cyclesError || signalsError) {
    return (
      <div className="text-center text-red-500 p-4">
        <AlertTriangle className="mx-auto h-12 w-12 mb-4" />
        <h2 className="text-2xl font-bold mb-2">Error Loading Data</h2>
        <p>{portfolioError || cyclesError || signalsError}</p>
      </div>
    );
  }

  const activeTrades = cycles?.filter(cycle => 
    ['initiating', 'waiting_buy', 'active_buy', 'waiting_sell', 'active_sell'].includes(cycle.status)
  ) || [];
  
  // Get the most recent signals (up to 5)
  const recentSignals = [...(signals || [])].sort((a, b) => 
    new Date(b.timestamp) - new Date(a.timestamp)
  ).slice(0, 5);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Trading Dashboard</h1>
      
      {/* Portfolio Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4">Portfolio Value</h2>
          <div className="flex items-baseline">
            <span className="text-3xl font-bold">{formatCurrency(summary?.total_value || 0)}</span>
            {summary?.performance_24h && (
              <span className={`ml-2 flex items-center ${summary.performance_24h >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                {summary.performance_24h >= 0 ? <ArrowUp size={16} /> : <ArrowDown size={16} />}
                {formatPercent(summary.performance_24h)}
              </span>
            )}
          </div>
          <p className="text-gray-500 mt-1">24h change</p>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4">Active Trades</h2>
          <div className="flex items-baseline">
            <span className="text-3xl font-bold">{activeTrades.length}</span>
            <span className="ml-2 text-gray-500">of {cycles?.length || 0} total</span>
          </div>
          <p className="text-gray-500 mt-1">Trading cycles</p>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4">Win Rate</h2>
          <div className="flex items-baseline">
            <span className="text-3xl font-bold">
              {stats?.win_rate ? formatPercent(stats.win_rate) : 'N/A'}
            </span>
          </div>
          <p className="text-gray-500 mt-1">Last 30 days</p>
        </div>
      </div>

      {/* Trading signals and cycle list */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent signals */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4">Recent Signals</h2>
          {recentSignals.length > 0 ? (
            <div className="space-y-4">
              {recentSignals.map((signal, index) => (
                <div key={index} className="flex items-center p-3 border-b last:border-0">
                  <div className={`p-2 rounded-full mr-4 ${
                    signal.side === 'BUY' ? 'bg-green-100 text-green-600' : 'bg-red-100 text-red-600'
                  }`}>
                    {signal.side === 'BUY' ? <TrendingUp size={20} /> : <TrendingUp size={20} className="transform rotate-180" />}
                  </div>
                  <div className="flex-1">
                    <div className="flex justify-between">
                      <span className="font-medium">{signal.symbol}</span>
                      <span className={signal.side === 'BUY' ? 'text-green-600' : 'text-red-600'}>
                        {signal.side}
                      </span>
                    </div>
                    <div className="flex justify-between text-sm text-gray-500">
                      <span>{signal.strategy}</span>
                      <span>{new Date(signal.timestamp).toLocaleTimeString()}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-center py-4">No recent signals</p>
          )}
        </div>

        {/* Signal Chart */}
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold mb-4">Signal Distribution</h2>
          <SignalChart signals={signals} />
        </div>
      </div>

      {/* Active trading cycles */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-semibold mb-4">Active Trading Cycles</h2>
        <CycleList cycles={activeTrades} limit={5} />
      </div>
    </div>
  );
};

export default Dashboard;