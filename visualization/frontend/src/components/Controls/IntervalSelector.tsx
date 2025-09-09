import React from 'react';
import { useChart } from '@/hooks/useChart';
import type { TimeInterval } from '@/types';

function IntervalSelector() {
  const { config, handleIntervalChange } = useChart();
  
  const intervals: { value: TimeInterval; label: string }[] = [
    { value: '1m', label: '1 minute' },
    { value: '3m', label: '3 minutes' },
    { value: '5m', label: '5 minutes' },
    { value: '15m', label: '15 minutes' },
    { value: '1h', label: '1 heure' },
    { value: '1d', label: '1 jour' },
  ];
  
  return (
    <div className="flex flex-col space-y-1">
      <label className="text-xs text-gray-300 font-medium">
        Intervalle
      </label>
      <select
        value={config.interval}
        onChange={(e) => handleIntervalChange(e.target.value as TimeInterval)}
        className="select min-w-[120px]"
      >
        {intervals.map((interval) => (
          <option key={interval.value} value={interval.value}>
            {interval.label}
          </option>
        ))}
      </select>
    </div>
  );
}

export default IntervalSelector;