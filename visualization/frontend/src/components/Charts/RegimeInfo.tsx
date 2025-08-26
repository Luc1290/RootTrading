import React from 'react';
import { useChartStore } from '@/stores/useChartStore';

interface RegimeInfoProps {
  className?: string;
}

function RegimeInfo({ className = '' }: RegimeInfoProps) {
  const { indicators, marketData } = useChartStore();
  
  // Debug des indicateurs disponibles
  React.useEffect(() => {
    if (indicators) {
      console.log('Indicateurs disponibles pour RegimeInfo:', Object.keys(indicators));
    }
  }, [indicators]);
  
  if (!indicators || !marketData?.timestamps) {
    return (
      <div className={`bg-gray-800 rounded-lg p-4 ${className}`}>
        <h3 className="text-lg font-semibold text-white mb-4">Market Analysis</h3>
        <div className="text-center text-gray-400 py-8">
          <div className="animate-pulse">
            <div className="h-4 bg-gray-700 rounded w-3/4 mx-auto mb-2"></div>
            <div className="h-4 bg-gray-700 rounded w-1/2 mx-auto"></div>
          </div>
          <p className="mt-4 text-sm">Chargement des données de régime...</p>
        </div>
      </div>
    );
  }
  
  // Prendre les dernières valeurs
  const latestIndex = marketData.timestamps.length - 1;
  const marketRegime = indicators.market_regime?.[latestIndex];
  const regimeStrength = indicators.regime_strength?.[latestIndex];
  const regimeConfidence = indicators.regime_confidence?.[latestIndex];
  const volumeContext = indicators.volume_context?.[latestIndex];
  const volumePattern = indicators.volume_pattern?.[latestIndex];
  const patternDetected = indicators.pattern_detected?.[latestIndex];
  const dataQuality = indicators.data_quality?.[latestIndex];
  
  // Fallback: utiliser des indicateurs basiques pour déduire le régime
  let fallbackRegime = null;
  let fallbackStrength = null;
  
  if (!marketRegime && indicators.rsi_14 && indicators.ema_7 && indicators.ema_26) {
    const rsi = indicators.rsi_14[latestIndex];
    const ema7 = indicators.ema_7[latestIndex];
    const ema26 = indicators.ema_26[latestIndex];
    const currentPrice = marketData.close[latestIndex];
    
    if (ema7 > ema26 && currentPrice > ema7 && rsi > 50) {
      fallbackRegime = 'TRENDING_BULL';
      fallbackStrength = rsi > 70 ? 'STRONG' : 'MODERATE';
    } else if (ema7 < ema26 && currentPrice < ema7 && rsi < 50) {
      fallbackRegime = 'TRENDING_BEAR';
      fallbackStrength = rsi < 30 ? 'STRONG' : 'MODERATE';
    } else {
      fallbackRegime = 'RANGING';
      fallbackStrength = 'WEAK';
    }
  }
  
  const getRegimeColor = (regime: string) => {
    switch (regime) {
      case 'TRENDING_BULL':
        return 'text-green-400 bg-green-900';
      case 'TRENDING_BEAR':
        return 'text-red-400 bg-red-900';
      case 'BREAKOUT_BULL':
        return 'text-green-300 bg-green-800';
      case 'BREAKOUT_BEAR':
        return 'text-red-300 bg-red-800';
      case 'RANGING':
        return 'text-yellow-400 bg-yellow-900';
      case 'VOLATILE':
        return 'text-purple-400 bg-purple-900';
      default:
        return 'text-gray-400 bg-gray-700';
    }
  };
  
  const getStrengthColor = (strength: string) => {
    switch (strength) {
      case 'STRONG':
      case 'EXTREME':
        return 'text-green-400';
      case 'MODERATE':
        return 'text-yellow-400';
      case 'WEAK':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };
  
  const getVolumeContextColor = (context: string) => {
    switch (context) {
      case 'BREAKOUT':
      case 'PUMP_START':
        return 'text-green-400 bg-green-900';
      case 'DEEP_OVERSOLD':
      case 'MODERATE_OVERSOLD':
        return 'text-blue-400 bg-blue-900';
      case 'HIGH_VOLATILITY':
        return 'text-purple-400 bg-purple-900';
      default:
        return 'text-gray-400 bg-gray-700';
    }
  };
  
  const getQualityColor = (quality: string) => {
    switch (quality) {
      case 'EXCELLENT':
        return 'text-green-400';
      case 'GOOD':
        return 'text-blue-400';
      case 'FAIR':
        return 'text-yellow-400';
      case 'POOR':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };
  
  return (
    <div className={`bg-gray-800 rounded-lg p-4 ${className}`}>
      <h3 className="text-lg font-semibold text-white mb-4">Market Analysis</h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Régime de marché */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-gray-300">Market Regime</h4>
          {(marketRegime || fallbackRegime) && (
            <div className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium ${getRegimeColor(marketRegime || fallbackRegime || '')}`}>
              {(marketRegime || fallbackRegime || '').replace('_', ' ')}
              {fallbackRegime && !marketRegime && (
                <span className="ml-1 text-xs opacity-75">(estimé)</span>
              )}
            </div>
          )}
          {(regimeStrength || fallbackStrength) && (
            <div className="flex items-center space-x-2">
              <span className="text-xs text-gray-400">Strength:</span>
              <span className={`text-xs font-medium ${getStrengthColor(regimeStrength || fallbackStrength || '')}`}>
                {regimeStrength || fallbackStrength}
                {fallbackStrength && !regimeStrength && (
                  <span className="ml-1 opacity-75">(estimé)</span>
                )}
              </span>
            </div>
          )}
          {regimeConfidence !== undefined && (
            <div className="flex items-center space-x-2">
              <span className="text-xs text-gray-400">Confidence:</span>
              <span className="text-xs text-white font-medium">
                {regimeConfidence.toFixed(1)}%
              </span>
              <div className="flex-1 bg-gray-700 rounded-full h-2">
                <div 
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${regimeConfidence}%` }}
                />
              </div>
            </div>
          )}
        </div>
        
        {/* Contexte volume */}
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-gray-300">Volume Context</h4>
          {volumeContext && (
            <div className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium ${getVolumeContextColor(volumeContext)}`}>
              {volumeContext.replace('_', ' ')}
            </div>
          )}
          {volumePattern && (
            <div className="flex items-center space-x-2">
              <span className="text-xs text-gray-400">Pattern:</span>
              <span className="text-xs text-white">
                {volumePattern.replace('_', ' ')}
              </span>
            </div>
          )}
          {patternDetected && (
            <div className="flex items-center space-x-2">
              <span className="text-xs text-gray-400">Detected:</span>
              <span className="text-xs text-green-400">
                {patternDetected}
              </span>
            </div>
          )}
        </div>
      </div>
      
      {/* Qualité des données */}
      {dataQuality && (
        <div className="mt-4 pt-4 border-t border-gray-700">
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-400">Data Quality:</span>
            <span className={`text-xs font-medium ${getQualityColor(dataQuality)}`}>
              {dataQuality}
            </span>
          </div>
        </div>
      )}
      
      {/* Métriques de volume avancées */}
      <div className="mt-4 grid grid-cols-3 gap-4 text-center">
        {indicators.quote_volume_ratio?.[latestIndex] && (
          <div>
            <div className="text-xs text-gray-400">Quote Volume Ratio</div>
            <div className="text-sm font-medium text-white">
              {indicators.quote_volume_ratio[latestIndex].toFixed(2)}x
            </div>
          </div>
        )}
        {indicators.trade_intensity?.[latestIndex] && (
          <div>
            <div className="text-xs text-gray-400">Trade Intensity</div>
            <div className="text-sm font-medium text-white">
              {indicators.trade_intensity[latestIndex].toFixed(2)}x
            </div>
          </div>
        )}
        {indicators.avg_trade_size?.[latestIndex] && (
          <div>
            <div className="text-xs text-gray-400">Avg Trade Size</div>
            <div className="text-sm font-medium text-white">
              {indicators.avg_trade_size[latestIndex].toFixed(6)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default RegimeInfo;