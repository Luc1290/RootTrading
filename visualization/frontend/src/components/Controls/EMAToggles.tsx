import React from 'react';
import { useChart } from '@/hooks/useChart';

function EMAToggles() {
  const { config, handleEMAToggle } = useChart();
  
  const emaOptions = [
    { key: 'ema12' as const, label: 'EMA 12', color: 'text-orange-400' },
    { key: 'ema26' as const, label: 'EMA 26', color: 'text-yellow-400' },
    { key: 'ema50' as const, label: 'EMA 50', color: 'text-yellow-300' },
  ];
  
  return (
    <div className="flex flex-col space-y-1">
      <label className="text-xs text-gray-300 font-medium">
        Affichage EMA
      </label>
      <div className="flex items-center space-x-3">
        {emaOptions.map((option) => (
          <label
            key={option.key}
            className="flex items-center space-x-1 cursor-pointer hover:bg-gray-700 px-2 py-1 rounded transition-colors"
          >
            <input
              type="checkbox"
              checked={config.emaToggles[option.key]}
              onChange={() => handleEMAToggle(option.key)}
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

export default EMAToggles;