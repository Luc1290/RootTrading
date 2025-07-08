import React from 'react';
import { ArrowPathIcon } from '@heroicons/react/24/outline';
import { useChart } from '@/hooks/useChart';

function RefreshButton() {
  const { forceUpdate, isLoading } = useChart();
  
  const handleRefresh = () => {
    forceUpdate();
  };
  
  return (
    <button
      onClick={handleRefresh}
      disabled={isLoading}
      className="btn btn-primary px-4 py-2 flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
      title="Actualiser les données"
    >
      <ArrowPathIcon className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
      <span className="text-sm font-medium">
        {isLoading ? 'Actualisation...' : 'Actualiser'}
      </span>
    </button>
  );
}

export default RefreshButton;