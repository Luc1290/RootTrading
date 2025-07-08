import React from 'react';
import { useChart } from '@/hooks/useChart';
import type { SignalFilter as SignalFilterType } from '@/types';

function SignalFilter() {
  const { config, handleSignalFilterChange } = useChart();
  
  const filters: { value: SignalFilterType; label: string }[] = [
    { value: 'all', label: 'Tous les signaux' },
    { value: 'Aggregated_2', label: 'Aggregated_2 (2 stratégies)' },
    { value: 'Aggregated_3', label: 'Aggregated_3 (3 stratégies)' },
    { value: 'Aggregated_4', label: 'Aggregated_4 (4 stratégies)' },
    { value: 'Aggregated_2,Aggregated_3,Aggregated_4', label: 'Multi-stratégies (2+)' },
  ];
  
  return (
    <div className="flex flex-col space-y-1">
      <label className="text-xs text-gray-300 font-medium">
        Filtrer Signaux
      </label>
      <select
        value={config.signalFilter}
        onChange={(e) => handleSignalFilterChange(e.target.value as SignalFilterType)}
        className="select min-w-[180px]"
      >
        {filters.map((filter) => (
          <option key={filter.value} value={filter.value}>
            {filter.label}
          </option>
        ))}
      </select>
    </div>
  );
}

export default SignalFilter;