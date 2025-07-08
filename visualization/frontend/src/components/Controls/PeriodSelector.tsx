import React from 'react';
import { useChart } from '@/hooks/useChart';
import type { PerformancePeriod } from '@/types';

function PeriodSelector() {
  const { config, handlePeriodChange } = useChart();
  
  const periods: { value: PerformancePeriod; label: string }[] = [
    { value: '1h', label: '1 Heure' },
    { value: '24h', label: '24 Heures' },
    { value: '7d', label: '7 Jours' },
    { value: '30d', label: '30 Jours' },
  ];
  
  return (
    <div className="flex flex-col space-y-1">
      <label className="text-xs text-gray-300 font-medium">
        Période Performance
      </label>
      <select
        value={config.period}
        onChange={(e) => handlePeriodChange(e.target.value as PerformancePeriod)}
        className="select min-w-[120px]"
        title="Période Performance"
      >
        {periods.map((period) => (
          <option key={period.value} value={period.value}>
            {period.label}
          </option>
        ))}
      </select>
    </div>
  );
}

export default PeriodSelector;