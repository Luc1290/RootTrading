import React from 'react';
import SymbolSelector from './SymbolSelector';
import IntervalSelector from './IntervalSelector';
import SignalFilter from './SignalFilter';
import PeriodSelector from './PeriodSelector';
import EMAToggles from './EMAToggles';
import RefreshButton from './RefreshButton';
import WebSocketToggle from './WebSocketToggle';

function Controls() {
  return (
    <div className="bg-dark-200 border border-gray-700 rounded-lg p-4 shadow-lg">
      <div className="flex flex-wrap gap-4 items-center">
        <SymbolSelector />
        <IntervalSelector />
        <SignalFilter />
        <PeriodSelector />
        <EMAToggles />
        
        <div className="flex items-center space-x-2 ml-auto">
          <RefreshButton />
          <WebSocketToggle />
        </div>
      </div>
    </div>
  );
}

export default Controls;