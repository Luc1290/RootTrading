import React from 'react';
import { useChart } from '@/hooks/useChart';

function IndicatorToggles() {
  const { config, handleIndicatorToggle } = useChart();
  
  const indicatorOptions = [
    { 
      key: 'rsi' as const, 
      label: 'RSI', 
      color: 'text-purple-400',
      description: 'RSI 14 & 21'
    },
    { 
      key: 'macd' as const, 
      label: 'MACD', 
      color: 'text-blue-400',
      description: 'MACD Line, Signal & Histogram'
    },
    { 
      key: 'bollinger' as const, 
      label: 'Bollinger', 
      color: 'text-green-400',
      description: 'Bollinger Bands'
    },
    { 
      key: 'stochastic' as const, 
      label: 'Stochastic', 
      color: 'text-orange-400',
      description: 'Stochastic K & D'
    },
    { 
      key: 'adx' as const, 
      label: 'ADX', 
      color: 'text-red-400',
      description: 'Trend Strength'
    },
    { 
      key: 'volume_advanced' as const, 
      label: 'Volume+', 
      color: 'text-cyan-400',
      description: 'Advanced Volume Metrics'
    },
    { 
      key: 'regime_info' as const, 
      label: 'Regime', 
      color: 'text-yellow-400',
      description: 'Market Regime Analysis'
    },
  ];
  
  return (
    <div className="flex flex-col space-y-1">
      <label className="text-xs text-gray-300 font-medium">
        Indicateurs Techniques
      </label>
      <div className="grid grid-cols-2 gap-2">
        {indicatorOptions.map((option) => (
          <label
            key={option.key}
            className="flex items-center space-x-2 cursor-pointer hover:bg-gray-700 px-2 py-1 rounded transition-colors"
            title={option.description}
          >
            <input
              type="checkbox"
              checked={config.indicatorToggles[option.key]}
              onChange={() => handleIndicatorToggle(option.key)}
              className="rounded border-gray-600 bg-dark-300 text-primary-600 focus:ring-primary-500 focus:ring-2"
            />
            <span className={`text-xs font-medium ${option.color}`}>
              {option.label}
            </span>
          </label>
        ))}
      </div>
    </div>
  );
}

export default IndicatorToggles;