import React from 'react';
import { useChart } from '@/hooks/useChart';
import { useAvailableSymbols } from '@/hooks/useApi';
import type { TradingSymbol } from '@/types';

function SymbolSelector() {
  const { config, handleSymbolChange } = useChart();
  const { data: symbolsData } = useAvailableSymbols();
  
  const symbols = symbolsData?.symbols || ['BTCUSDC', 'ETHUSDC', 'SOLUSDC', 'XRPUSDC'];
  
  return (
    <div className="flex flex-col space-y-1">
      <label className="text-xs text-gray-300 font-medium">
        Symbole
      </label>
      <select
        value={config.symbol}
        onChange={(e) => handleSymbolChange(e.target.value as TradingSymbol)}
        className="select min-w-[120px]"
      >
        {symbols.map((symbol) => (
          <option key={symbol} value={symbol}>
            {symbol.replace('USDC', '/USDC')}
          </option>
        ))}
      </select>
    </div>
  );
}

export default SymbolSelector;