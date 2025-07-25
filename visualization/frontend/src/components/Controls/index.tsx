import React from 'react';
import SymbolSelector from './SymbolSelector';
import IntervalSelector from './IntervalSelector';
import SignalFilter from './SignalFilter';
import PeriodSelector from './PeriodSelector';
import EMAToggles from './EMAToggles';
import SMAToggles from './SMAToggles';
import IndicatorToggles from './IndicatorToggles';
import RefreshButton from './RefreshButton';
import WebSocketToggle from './WebSocketToggle';
import SnapshotButton from './SnapshotButton';

function Controls() {
  return (
    <div className="bg-dark-200 border border-gray-700 rounded-lg p-4 shadow-lg">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Première ligne - Sélecteurs principaux */}
        <div className="flex flex-wrap gap-4 items-center">
          <SymbolSelector />
          <IntervalSelector />
          <SignalFilter />
          <PeriodSelector />
        </div>
        
        {/* Actions */}
        <div className="flex items-center justify-end space-x-2">
          <SnapshotButton />
          <RefreshButton />
          <WebSocketToggle />
        </div>
        
        {/* Deuxième ligne - Indicateurs */}
        <div className="col-span-full grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t border-gray-600">
          <EMAToggles />
          <SMAToggles />
          <IndicatorToggles />
        </div>
      </div>
    </div>
  );
}

export default Controls;