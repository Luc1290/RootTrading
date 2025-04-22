import React, { useState } from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { ArrowUp, ArrowDown, Percent, DollarSign, Hash, AlertTriangle, Calendar, TrendingUp } from 'lucide-react';
import usePortfolio from '../api/usePortfolio';
import useLiveStats from '../hooks/useLiveStats';

const StatsPanel = () => {
  const { summary, performance, loading: portfolioLoading, error: portfolioError } = usePortfolio();
  const { stats, loading: statsLoading, error: statsError } = useLiveStats();
  
  // Stats view period
  const [period, setPeriod] = useState('daily');
  
  // Stats view category
  const [category, setCategory] = useState('overall');
  
  // Format currency
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', { 
      style: 'currency', 
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };
  
  // Format percentage
  const formatPercent = (value) => {
    return new Intl.NumberFormat('en-US', { 
      style: 'percent',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value / 100);
  };
  
  // Loading state
  if (portfolioLoading || statsLoading) {
    return (
      <div className="flex justify-center items-center h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  // Error state
  if (portfolioError || statsError) {
    return (
      <div className="text-center text-red-500 p-4">
        <AlertTriangle className="mx-auto h-12 w-12 mb-4" />
        <h2 className="text-2xl font-bold mb-2">Error Loading Stats</h2>
        <p>{portfolioError || statsError}</p>
      </div>
    );
  }
  
  // Get performance data based on selected period
  const periodData = performance?.[period] || [];
  
  // Format data for charts
  const performanceChartData = periodData.map(item => ({
    date: new Date(item.start_date).toLocaleDateString(),
    profit: item.profit_loss || 0,
    profitPercent: item.profit_loss_percent || 0,
    totalTrades: item.total_trades || 0,
    winRate: item.winning_trades / Math.max(item.total_trades, 1) * 100 || 0
  }));
  
  // Get strategy and symbol performance
  const strategyPerformance = stats?.strategy_performance || [];
  const symbolPerformance = stats?.symbol_performance || [];
  
  // Determine which performance data to show based on category
  let categoryData;
  if (category === 'strategy') {
    categoryData = strategyPerformance;
  } else if (category === 'symbol') {
    categoryData = symbolPerformance;
  }
  
  // Create data for category bar chart
  const categoryChartData = categoryData ? categoryData.map(item => ({
    name: item.strategy || item.symbol,
    profit: item.total_profit_loss || 0,
    profitPercent: item.avg_profit_loss_percent || 0,
    winRate: item.winning_trades / Math.max(item.total_cycles, 1) * 100,
    trades: item.total_cycles
  })) : [];

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Performance Statistics</h1>
      
      {/* Stats Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex items-center">
            <div className="bg-blue-100 p-3 rounded-full mr-4">
              <DollarSign className="text-blue-600" size={24} />
            </div>
            <div>
              <p className="text-sm text-gray-500">Total Profit/Loss</p>
              <h3 className="text-xl font-bold">{formatCurrency(stats?.total_profit || 0)}</h3>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex items-center">
            <div className="bg-green-100 p-3 rounded-full mr-4">
              <Percent className="text-green-600" size={24} />
            </div>
            <div>
              <p className="text-sm text-gray-500">Win Rate</p>
              <h3 className="text-xl font-bold">{formatPercent(stats?.win_rate || 0)}</h3>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex items-center">
            <div className="bg-purple-100 p-3 rounded-full mr-4">
              <Hash className="text-purple-600" size={24} />
            </div>
            <div>
              <p className="text-sm text-gray-500">Total Trades</p>
              <h3 className="text-xl font-bold">{stats?.total_trades || 0}</h3>
            </div>
          </div>
        </div>
        
        <div className="bg-white rounded-lg shadow p-4">
          <div className="flex items-center">
            <div className="bg-orange-100 p-3 rounded-full mr-4">
              <TrendingUp className="text-orange-600" size={24} />
            </div>
            <div>
              <p className="text-sm text-gray-500">Avg. Profit per Trade</p>
              <h3 className="text-xl font-bold">{formatCurrency(stats?.avg_profit || 0)}</h3>
            </div>
          </div>
        </div>
      </div>
      
      {/* Performance Over Time Chart */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-lg font-semibold">Performance Over Time</h2>
          
          <div className="flex space-x-2">
            <button
              onClick={() => setPeriod('daily')}
              className={`px-3 py-1 text-sm rounded-md ${
                period === 'daily' 
                  ? 'bg-blue-100 text-blue-600' 
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              Daily
            </button>
            <button
              onClick={() => setPeriod('weekly')}
              className={`px-3 py-1 text-sm rounded-md ${
                period === 'weekly' 
                  ? 'bg-blue-100 text-blue-600' 
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              Weekly
            </button>
            <button
              onClick={() => setPeriod('monthly')}
              className={`px-3 py-1 text-sm rounded-md ${
                period === 'monthly' 
                  ? 'bg-blue-100 text-blue-600' 
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              Monthly
            </button>
          </div>
        </div>
        
        {performanceChartData.length > 0 ? (
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={performanceChartData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip />
                <Legend />
                <Line 
                  yAxisId="left"
                  type="monotone" 
                  dataKey="profit" 
                  name="Profit/Loss" 
                  stroke="#3b82f6" 
                  activeDot={{ r: 8 }} 
                />
                <Line 
                  yAxisId="right"
                  type="monotone" 
                  dataKey="winRate" 
                  name="Win Rate %" 
                  stroke="#10b981" 
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="flex justify-center items-center h-80 text-gray-500">
            No performance data available for this period
          </div>
        )}
      </div>
      
      {/* Category Performance Chart */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-lg font-semibold">Performance By Category</h2>
          
          <div className="flex space-x-2">
            <button
              onClick={() => setCategory('strategy')}
              className={`px-3 py-1 text-sm rounded-md ${
                category === 'strategy' 
                  ? 'bg-blue-100 text-blue-600' 
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              By Strategy
            </button>
            <button
              onClick={() => setCategory('symbol')}
              className={`px-3 py-1 text-sm rounded-md ${
                category === 'symbol' 
                  ? 'bg-blue-100 text-blue-600' 
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              By Symbol
            </button>
          </div>
        </div>
        
        {categoryChartData.length > 0 ? (
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={categoryChartData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="profit" name="Total Profit/Loss" fill="#3b82f6" />
                <Bar dataKey="trades" name="Number of Trades" fill="#8b5cf6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="flex justify-center items-center h-80 text-gray-500">
            No {category} performance data available
          </div>
        )}
      </div>
      
      {/* Detailed Statistics */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="p-6 border-b">
          <h2 className="text-lg font-semibold">Detailed Performance Metrics</h2>
        </div>
        
        <div className="p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Wins vs. Losses */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-gray-500">Trading Results</h3>
              <div className="space-y-1">
                <div className="flex justify-between">
                  <span className="text-gray-600">Winning Trades:</span>
                  <span className="font-medium text-green-600">{stats?.winning_trades || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Losing Trades:</span>
                  <span className="font-medium text-red-600">{stats?.losing_trades || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Break Even:</span>
                  <span className="font-medium">{stats?.break_even_trades || 0}</span>
                </div>
                <div className="flex justify-between border-t pt-1 mt-1">
                  <span className="text-gray-600 font-medium">Total Trades:</span>
                  <span className="font-medium">{stats?.total_trades || 0}</span>
                </div>
              </div>
            </div>
            
            {/* Profitability */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-gray-500">Profitability</h3>
              <div className="space-y-1">
                <div className="flex justify-between">
                  <span className="text-gray-600">Average Win:</span>
                  <span className="font-medium text-green-600">{formatCurrency(stats?.avg_win || 0)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Average Loss:</span>
                  <span className="font-medium text-red-600">{formatCurrency(stats?.avg_loss || 0)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Win Rate:</span>
                  <span className="font-medium">{formatPercent(stats?.win_rate || 0)}</span>
                </div>
                <div className="flex justify-between border-t pt-1 mt-1">
                  <span className="text-gray-600 font-medium">Profit Factor:</span>
                  <span className="font-medium">{stats?.profit_factor?.toFixed(2) || 'N/A'}</span>
                </div>
              </div>
            </div>
            
            {/* Risk Metrics */}
            <div className="space-y-2">
              <h3 className="text-sm font-medium text-gray-500">Risk Metrics</h3>
              <div className="space-y-1">
                <div className="flex justify-between">
                  <span className="text-gray-600">Max Drawdown:</span>
                  <span className="font-medium text-red-600">{formatPercent((stats?.max_drawdown || 0) / 100)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Avg. Holding Time:</span>
                  <span className="font-medium">{stats?.avg_holding_time || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Sharpe Ratio:</span>
                  <span className="font-medium">{stats?.sharpe_ratio?.toFixed(2) || 'N/A'}</span>
                </div>
                <div className="flex justify-between border-t pt-1 mt-1">
                  <span className="text-gray-600 font-medium">Risk/Reward:</span>
                  <span className="font-medium">{stats?.risk_reward_ratio?.toFixed(2) || 'N/A'}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StatsPanel;