import React, { useMemo } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';

const SignalChart = ({ signals = [] }) => {
  // Process signals data for chart visualization
  const chartData = useMemo(() => {
    if (!signals || signals.length === 0) {
      return [];
    }

    // Group signals by strategy and side
    const strategyData = {};
    
    signals.forEach(signal => {
      const key = signal.strategy;
      if (!strategyData[key]) {
        strategyData[key] = {
          name: key,
          value: 0,
          buys: 0,
          sells: 0
        };
      }
      
      strategyData[key].value += 1;
      
      if (signal.side === 'BUY') {
        strategyData[key].buys += 1;
      } else if (signal.side === 'SELL') {
        strategyData[key].sells += 1;
      }
    });

    // Convert to array format for PieChart
    return Object.values(strategyData);
  }, [signals]);

  // For side distribution (buy vs sell)
  const sideData = useMemo(() => {
    if (!signals || signals.length === 0) {
      return [];
    }

    const buyCount = signals.filter(s => s.side === 'BUY').length;
    const sellCount = signals.filter(s => s.side === 'SELL').length;

    return [
      { name: 'Buy Signals', value: buyCount },
      { name: 'Sell Signals', value: sellCount }
    ];
  }, [signals]);

  // Colors for strategies and sides
  const STRATEGY_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#6366f1', '#ec4899', '#8b5cf6'];
  const SIDE_COLORS = ['#10b981', '#ef4444'];

  // Custom tooltip
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-3 border rounded shadow-md">
          <p className="font-medium">{data.name}</p>
          <p>Total Signals: {data.value}</p>
          {data.buys !== undefined && (
            <>
              <p className="text-green-500">Buy Signals: {data.buys}</p>
              <p className="text-red-500">Sell Signals: {data.sells}</p>
            </>
          )}
        </div>
      );
    }
    return null;
  };

  if (chartData.length === 0) {
    return (
      <div className="flex justify-center items-center h-64">
        <p className="text-gray-500">No signal data available</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Strategy Distribution */}
      <div className="h-64">
        <h3 className="text-sm font-medium text-gray-500 mb-2">By Strategy</h3>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              labelLine={false}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
              nameKey="name"
            >
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={STRATEGY_COLORS[index % STRATEGY_COLORS.length]} />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Buy vs Sell Distribution */}
      <div className="h-64">
        <h3 className="text-sm font-medium text-gray-500 mb-2">By Signal Direction</h3>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={sideData}
              cx="50%"
              cy="50%"
              labelLine={false}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
              nameKey="name"
            >
              {sideData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={SIDE_COLORS[index % SIDE_COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default SignalChart;