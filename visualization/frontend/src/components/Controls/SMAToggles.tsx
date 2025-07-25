import React from 'react';
import { useChart } from '@/hooks/useChart';

function SMAToggles() {
  const { config, handleSMAToggle } = useChart();
  
  const smaOptions = [
    { 
      key: 'sma20' as const, 
      label: 'SMA 20', 
      color: 'text-yellow-400',
      period: 20
    },
    { 
      key: 'sma50' as const, 
      label: 'SMA 50', 
      color: 'text-orange-400',
      period: 50
    },
  ];
  
  return (
    <div className="flex flex-col space-y-1">
      <label className="text-xs text-gray-300 font-medium">
        Simple Moving Average
      </label>
      <div className="flex space-x-2">
        {smaOptions.map((option) => (
          <label
            key={option.key}
            className="flex items-center space-x-1 cursor-pointer hover:bg-gray-700 px-2 py-1 rounded transition-colors"
          >
            <input
              type="checkbox"
              checked={config.smaToggles[option.key]}
              onChange={() => handleSMAToggle(option.key)}
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

export default SMAToggles;