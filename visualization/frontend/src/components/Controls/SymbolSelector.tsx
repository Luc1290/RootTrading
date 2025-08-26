import React, { useState, useEffect } from 'react';
import { useChart } from '@/hooks/useChart';
import { apiService } from '@/services/api';
import type { TradingSymbol } from '@/types';

interface OwnedSymbol {
  symbol: string;
  asset: string;
  price: number;
  price_change_24h: number;
  current_balance?: number;
}

function SymbolSelector() {
  const { config, handleSymbolChange } = useChart();
  const [ownedSymbols, setOwnedSymbols] = useState<OwnedSymbol[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchOwnedSymbols();
    const interval = setInterval(fetchOwnedSymbols, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, []);

  const fetchOwnedSymbols = async () => {
    try {
      const [symbols, configuredResponse] = await Promise.all([
        apiService.getAllTradedSymbolsWithVariations(),
        apiService.getConfiguredSymbols()
      ]);
      
      // Récupérer les symboles configurés depuis shared/config
      const allowedSymbols = configuredResponse.symbols;
      
      // Créer un Map pour accès rapide aux données du portfolio
      const portfolioData = new Map();
      (symbols || []).forEach((symbol: OwnedSymbol) => {
        portfolioData.set(symbol.symbol, symbol);
      });
      
      // Utiliser directement les symboles tradés, plus les symboles configurés manquants
      let validSymbols = [...(symbols || [])];
      
      // Ajouter les symboles configurés manquants avec prix 0
      const tradedSymbolNames = new Set((symbols || []).map((s: OwnedSymbol) => s.symbol));
      const missingConfigured = allowedSymbols.filter(symbol => !tradedSymbolNames.has(symbol));
      
      const missingSymbols = missingConfigured.map(symbol => ({
        symbol: symbol,
        asset: symbol.replace('USDC', ''),
        price: 0,
        price_change_24h: 0,
        current_balance: 0
      }));
      
      validSymbols = [...validSymbols, ...missingSymbols];
      
      setOwnedSymbols(validSymbols);
    } catch (error) {
      console.error('Error fetching owned symbols:', error);
      // Fallback : récupérer au moins les symboles configurés
      try {
        const configuredResponse = await apiService.getConfiguredSymbols();
        const fallbackSymbols = configuredResponse.symbols.map(symbol => ({
          symbol: symbol,
          asset: symbol.replace('USDC', ''),
          price: 0,
          price_change_24h: 0,
          current_balance: 0
        }));
        setOwnedSymbols(fallbackSymbols);
      } catch (configError) {
        console.error('Error fetching configured symbols:', configError);
        // Dernier fallback avec quelques symboles statiques
        setOwnedSymbols([
          { symbol: 'BTCUSDC', asset: 'BTC', price: 0, price_change_24h: 0, current_balance: 0 },
          { symbol: 'ETHUSDC', asset: 'ETH', price: 0, price_change_24h: 0, current_balance: 0 },
          { symbol: 'SOLUSDC', asset: 'SOL', price: 0, price_change_24h: 0, current_balance: 0 }
        ]);
      }
    } finally {
      setLoading(false);
    }
  };

  const formatPercentage = (value: number): string => {
    const sign = value > 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
  };

  const formatPrice = (price: number): string => {
    if (price >= 1000) {
      return `$${(price / 1000).toFixed(1)}k`;
    }
    return `$${price.toFixed(2)}`;
  };

  const getVariationColor = (change: number): string => {
    if (change > 0) return 'text-green-400';
    if (change < 0) return 'text-red-400';
    return 'text-gray-400';
  };

  const getVariationBg = (change: number): string => {
    if (change > 0) return 'bg-green-900/30 border-green-700/50';
    if (change < 0) return 'bg-red-900/30 border-red-700/50';
    return 'bg-gray-900/30 border-gray-700/50';
  };

  if (loading) {
    return (
      <div className="flex flex-col space-y-2">
        <label className="text-xs text-gray-300 font-medium">Symboles Possédés</label>
        <div className="flex items-center justify-center p-4 bg-dark-300 rounded-lg">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-500"></div>
          <span className="ml-2 text-gray-400">Chargement des symboles...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col space-y-2">
      <label className="text-xs text-gray-300 font-medium">
        Tous les Symboles Tradés ({ownedSymbols.length}) - Variations 24h
      </label>
      
      <div className="flex flex-wrap gap-2 p-3 bg-dark-300 rounded-lg max-h-32 overflow-y-auto">
        {ownedSymbols.map((symbolData) => (
          <button
            key={symbolData.symbol}
            onClick={() => handleSymbolChange(symbolData.symbol as TradingSymbol)}
            className={`flex flex-col items-center p-2 rounded-lg border transition-all duration-200 hover:scale-105 min-w-[80px] ${
              config.symbol === symbolData.symbol 
                ? 'bg-primary-900/50 border-primary-500 ring-1 ring-primary-500' 
                : `${getVariationBg(symbolData.price_change_24h)} hover:bg-dark-200`
            }`}
          >
            {/* Symbole */}
            <div className="text-white font-bold text-sm flex items-center">
              {symbolData.asset}
              {(symbolData.current_balance || 0) > 0 && (
                <div className="ml-1 w-2 h-2 bg-green-400 rounded-full" title="Actuellement possédé" />
              )}
            </div>
            
            {/* Prix */}
            {symbolData.price > 0 && (
              <div className="text-gray-300 text-xs">
                {formatPrice(symbolData.price)}
              </div>
            )}
            
            {/* Variation */}
            <div className={`text-xs font-medium ${getVariationColor(symbolData.price_change_24h)}`}>
              {formatPercentage(symbolData.price_change_24h)}
            </div>
          </button>
        ))}
        
        {ownedSymbols.length === 0 && (
          <div className="text-center text-gray-400 py-4 w-full">
            Aucun symbole possédé trouvé
          </div>
        )}
      </div>
    </div>
  );
}

export default SymbolSelector;