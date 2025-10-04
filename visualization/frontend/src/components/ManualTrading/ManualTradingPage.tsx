import React, { useState, useEffect } from 'react';
import toast from 'react-hot-toast';
import { apiService } from '@/services/api';
import { formatCurrency, formatPercent } from '@/utils';

interface TradingOpportunity {
  symbol: string;
  score: number;
  score_details?: {
    trend: number;
    momentum: number;
    volume: number;
    price_action: number;
    consensus: number;
  };
  score_explanation?: {
    trend: string;
    momentum: string;
    volume: string;
    price_action: string;
    consensus: string;
  };
  currentPrice: number;
  signals24h: number;
  signalsConfidence: number;
  momentum: number;
  volumeRatio: number;
  regime: string;
  adx?: number;
  rsi?: number;
  mfi?: number;
  volume_context?: string;
  volume_quality_score?: number;
  nearest_support?: number;
  nearest_resistance?: number;
  break_probability?: number;
  entryZone: { min: number; max: number };
  targets: { tp1: number; tp2: number; tp3: number };
  stopLoss: number;
  recommendedSize: { min: number; max: number };
  action: 'BUY_NOW' | 'WAIT' | 'WAIT_PULLBACK' | 'WAIT_BREAKOUT' | 'WAIT_OVERSOLD' | 'SELL_OVERBOUGHT' | 'AVOID';
  reason: string;
  estimatedHoldTime?: string;
}

function ManualTradingPage() {
  const [opportunities, setOpportunities] = useState<TradingOpportunity[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [capitalSize, setCapitalSize] = useState<number>(5000); // Taille position par défaut
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Fonction pour calculer les opportunités
  const loadOpportunities = async () => {
    try {
      setLoading(true);

      // Récupérer les données nécessaires
      const [symbolsResponse] = await Promise.all([
        apiService.getConfiguredSymbols()
      ]);

      const symbols = symbolsResponse.symbols.map(s => s.replace('USDC', '') + 'USDC');

      // Pour chaque symbole, récupérer signaux + données techniques
      const opportunitiesData = await Promise.all(
        symbols.slice(0, 15).map(async (symbol) => {
          try {
            // Récupérer signaux et market data en parallèle
            const [signalsData, marketData] = await Promise.all([
              fetch(`/api/trading-opportunities/${symbol}`).then(r => r.json()).catch(() => null),
              apiService.getMarketData(symbol as any, '1m', 100).catch(() => null)
            ]);

            if (!signalsData || !marketData) return null;

            const currentPrice = marketData.data.close[marketData.data.close.length - 1] || 0;

            return {
              symbol,
              score: signalsData.score || 0,
              score_details: signalsData.score_details,
              score_explanation: signalsData.score_explanation,
              currentPrice,
              signals24h: signalsData.signals_count || 0,
              signalsConfidence: signalsData.avg_confidence || 0,
              momentum: signalsData.momentum_score || 0,
              volumeRatio: signalsData.volume_ratio || 0,
              regime: signalsData.market_regime || 'UNKNOWN',
              adx: signalsData.adx,
              rsi: signalsData.rsi,
              mfi: signalsData.mfi,
              volume_context: signalsData.volume_context,
              volume_quality_score: signalsData.volume_quality_score,
              nearest_support: signalsData.nearest_support,
              nearest_resistance: signalsData.nearest_resistance,
              break_probability: signalsData.break_probability,
              entryZone: signalsData.entry_zone || { min: currentPrice * 0.998, max: currentPrice * 1.002 },
              targets: signalsData.targets || {
                tp1: currentPrice * 1.01,
                tp2: currentPrice * 1.015,
                tp3: currentPrice * 1.02
              },
              stopLoss: signalsData.stop_loss || currentPrice * 0.988,
              recommendedSize: signalsData.recommended_size || { min: 3000, max: 7000 },
              action: signalsData.action || 'WAIT_PULLBACK',
              reason: signalsData.reason || 'Analyse en cours',
              estimatedHoldTime: signalsData.estimated_hold_time
            } as TradingOpportunity;
          } catch (err) {
            console.error(`Error loading ${symbol}:`, err);
            return null;
          }
        })
      );

      // Filtrer les nulls et trier par score
      const validOpportunities = opportunitiesData
        .filter((opp): opp is TradingOpportunity => opp !== null && opp.score > 0)
        .sort((a, b) => b.score - a.score);

      setOpportunities(validOpportunities);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Erreur lors du chargement des opportunités:', error);
      toast.error('Erreur lors du chargement des opportunités');
    } finally {
      setLoading(false);
    }
  };

  // Chargement initial
  useEffect(() => {
    loadOpportunities();
  }, []);

  // Auto-refresh toutes les 30 secondes
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      loadOpportunities();
    }, 30000);

    return () => clearInterval(interval);
  }, [autoRefresh]);

  const handleRefresh = () => {
    toast.promise(
      loadOpportunities(),
      {
        loading: 'Actualisation des opportunités...',
        success: 'Opportunités mises à jour',
        error: 'Erreur lors de l\'actualisation'
      }
    );
  };

  const getActionColor = (action: string) => {
    switch (action) {
      case 'BUY_NOW': return 'bg-green-500/20 text-green-400 border-green-500/50';
      case 'SELL_OVERBOUGHT': return 'bg-red-500/20 text-red-400 border-red-500/50';
      case 'WAIT': return 'bg-gray-500/20 text-gray-300 border-gray-500/50';
      case 'WAIT_PULLBACK': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50';
      case 'WAIT_BREAKOUT': return 'bg-blue-500/20 text-blue-400 border-blue-500/50';
      case 'WAIT_OVERSOLD': return 'bg-cyan-500/20 text-cyan-400 border-cyan-500/50';
      case 'AVOID': return 'bg-gray-700/20 text-gray-500 border-gray-700/50';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/50';
    }
  };

  const getActionIcon = (action: string) => {
    switch (action) {
      case 'BUY_NOW': return '🟢';
      case 'SELL_OVERBOUGHT': return '🔴';
      case 'WAIT': return '⚪';
      case 'WAIT_PULLBACK': return '🟡';
      case 'WAIT_BREAKOUT': return '🔵';
      case 'WAIT_OVERSOLD': return '🔵';
      case 'AVOID': return '⚫';
      default: return '⚪';
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 90) return 'text-emerald-400';
    if (score >= 80) return 'text-green-400';
    if (score >= 70) return 'text-lime-400';
    if (score >= 60) return 'text-yellow-400';
    if (score >= 50) return 'text-orange-400';
    return 'text-red-400';
  };

  const calculatePotentialGain = (entry: number, target: number, size: number) => {
    const gainPercent = ((target - entry) / entry) * 100;
    const gainAmount = (target - entry) * (size / entry);
    return { percent: gainPercent, amount: gainAmount };
  };

  if (loading && opportunities.length === 0) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="flex items-center space-x-3">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
          <span className="text-white text-lg">Chargement des opportunités...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* En-tête */}
      <div className="bg-dark-200 border border-gray-700 rounded-lg p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">
              🎯 Trading Manuel - Opportunités
            </h1>
            <p className="text-gray-400">
              Analyse en temps réel des meilleures opportunités pour scalping/day trading
            </p>
            <p className="text-xs text-gray-500 mt-1">
              Dernière mise à jour: {lastUpdate.toLocaleTimeString()}
            </p>
          </div>

          <div className="flex items-center space-x-4">
            {/* Configuration taille position */}
            <div className="flex items-center space-x-2">
              <label className="text-sm text-gray-400">Taille position:</label>
              <input
                type="number"
                value={capitalSize}
                onChange={(e) => setCapitalSize(Number(e.target.value))}
                className="bg-dark-300 border border-gray-600 rounded px-3 py-2 text-white w-32"
                min="1000"
                max="50000"
                step="1000"
                placeholder="Entrez la taille de position"
                title="Taille de position en USDC"
              />
              <span className="text-sm text-gray-400">USDC</span>
            </div>

            {/* Toggle auto-refresh */}
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`px-4 py-2 rounded-lg transition-colors ${
                autoRefresh
                  ? 'bg-green-500/20 text-green-400 border border-green-500/50'
                  : 'bg-gray-500/20 text-gray-400 border border-gray-500/50'
              }`}
            >
              {autoRefresh ? '🔄 Auto' : '⏸️ Manuel'}
            </button>

            <button
              onClick={handleRefresh}
              disabled={loading}
              className="bg-primary-500 hover:bg-primary-600 disabled:opacity-50 text-white px-6 py-2 rounded-lg transition-colors"
            >
              {loading ? '🔄 Actualisation...' : '🔄 Actualiser'}
            </button>
          </div>
        </div>
      </div>

      {/* Résumé rapide */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
        <div className="chart-container">
          <div className="text-center">
            <div className="text-3xl font-bold text-white">
              {opportunities.length}
            </div>
            <div className="text-sm text-gray-400">Opportunités</div>
          </div>
        </div>
        <div className="chart-container">
          <div className="text-center">
            <div className="text-3xl font-bold text-green-400">
              {opportunities.filter(o => o.action === 'BUY_NOW').length}
            </div>
            <div className="text-sm text-gray-400">🟢 ACHETER</div>
          </div>
        </div>
        <div className="chart-container">
          <div className="text-center">
            <div className="text-3xl font-bold text-red-400">
              {opportunities.filter(o => o.action === 'SELL_OVERBOUGHT').length}
            </div>
            <div className="text-sm text-gray-400">🔴 VENDRE (Overbought)</div>
          </div>
        </div>
        <div className="chart-container">
          <div className="text-center">
            <div className="text-3xl font-bold text-gray-400">
              {opportunities.filter(o => o.action.startsWith('WAIT')).length}
            </div>
            <div className="text-sm text-gray-400">⚪ ATTENDRE</div>
          </div>
        </div>
        <div className="chart-container">
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-400">
              {opportunities.filter(o => o.score >= 60).length}
            </div>
            <div className="text-sm text-gray-400">⭐ Score ≥ 60</div>
          </div>
        </div>
      </div>

      {/* Liste des opportunités */}
      <div className="space-y-4">
        {opportunities.slice(0, 10).map((opp, index) => {
          const tp1Gain = calculatePotentialGain(opp.currentPrice, opp.targets.tp1, capitalSize);
          const tp2Gain = calculatePotentialGain(opp.currentPrice, opp.targets.tp2, capitalSize);
          const slLoss = calculatePotentialGain(opp.currentPrice, opp.stopLoss, capitalSize);

          return (
            <div
              key={opp.symbol}
              className="bg-dark-200 border border-gray-700 rounded-lg p-6 hover:border-primary-500/50 transition-colors"
            >
              <div className="grid grid-cols-12 gap-6">
                {/* Colonne 1: Rang & Symbole */}
                <div className="col-span-2">
                  <div className="flex items-center space-x-3">
                    <div className="text-4xl font-bold text-gray-600">
                      {index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : `#${index + 1}`}
                    </div>
                    <div>
                      <div className="text-xl font-bold text-white">{opp.symbol}</div>
                      <div className={`text-2xl font-mono font-bold ${getScoreColor(opp.score)}`}>
                        {opp.score}/100
                      </div>
                    </div>
                  </div>
                </div>

                {/* Colonne 2: Action & Prix */}
                <div className="col-span-3">
                  <div className={`inline-block px-4 py-2 rounded-lg border font-bold mb-3 ${getActionColor(opp.action)}`}>
                    {getActionIcon(opp.action)} {opp.action.replace('_', ' ')}
                  </div>
                  <div className="space-y-1">
                    <div className="text-sm text-gray-400">Prix actuel</div>
                    <div className="text-2xl font-mono font-bold text-white">
                      {formatCurrency(opp.currentPrice)}
                    </div>
                    <div className="text-xs text-gray-500">
                      Entry: {formatCurrency(opp.entryZone.min)} - {formatCurrency(opp.entryZone.max)}
                    </div>
                  </div>
                </div>

                {/* Colonne 3: Signaux & Régime & Durée Hold */}
                <div className="col-span-2">
                  <div className="space-y-2">
                    <div>
                      <div className="text-xs text-gray-400">Signaux 24h</div>
                      <div className="text-lg font-bold text-white">{opp.signals24h}</div>
                      <div className="text-xs text-gray-500">Conf: {formatPercent(opp.signalsConfidence)}</div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-400">Régime</div>
                      <div className={`text-sm font-bold ${
                        opp.regime.includes('BULL') ? 'text-green-400' :
                        opp.regime.includes('BEAR') ? 'text-red-400' : 'text-gray-400'
                      }`}>
                        {opp.regime}
                      </div>
                    </div>
                    {opp.estimatedHoldTime && (
                      <div>
                        <div className="text-xs text-gray-400">⏱️ Durée Hold</div>
                        <div className="text-sm font-bold text-yellow-400">
                          {opp.estimatedHoldTime}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Colonne 4: Targets & Gains */}
                <div className="col-span-3">
                  <div className="space-y-2">
                    <div className="text-xs text-gray-400 mb-2">Targets & Gains potentiels (sur {formatCurrency(capitalSize)})</div>
                    <div className="space-y-1">
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-green-400">TP1 (+{tp1Gain.percent.toFixed(1)}%)</span>
                        <span className="font-mono text-sm text-green-400">{formatCurrency(opp.targets.tp1)}</span>
                        <span className="font-bold text-green-400">+{formatCurrency(tp1Gain.amount)}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-xs text-green-400">TP2 (+{tp2Gain.percent.toFixed(1)}%)</span>
                        <span className="font-mono text-sm text-green-400">{formatCurrency(opp.targets.tp2)}</span>
                        <span className="font-bold text-green-400">+{formatCurrency(tp2Gain.amount)}</span>
                      </div>
                      <div className="flex justify-between items-center border-t border-gray-700 pt-1">
                        <span className="text-xs text-red-400">SL ({slLoss.percent.toFixed(1)}%)</span>
                        <span className="font-mono text-sm text-red-400">{formatCurrency(opp.stopLoss)}</span>
                        <span className="font-bold text-red-400">{formatCurrency(slLoss.amount)}</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Colonne 5: Momentum & Volume */}
                <div className="col-span-2">
                  <div className="space-y-2">
                    <div>
                      <div className="text-xs text-gray-400">Momentum</div>
                      <div className={`text-lg font-bold ${
                        opp.momentum >= 70 ? 'text-green-400' :
                        opp.momentum >= 50 ? 'text-yellow-400' : 'text-orange-400'
                      }`}>
                        {opp.momentum.toFixed(0)}/100
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-400">Volume Ratio</div>
                      <div className={`text-lg font-bold ${
                        opp.volumeRatio >= 2.0 ? 'text-green-400' :
                        opp.volumeRatio >= 1.5 ? 'text-yellow-400' : 'text-orange-400'
                      }`}>
                        {opp.volumeRatio.toFixed(1)}x
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Raison / Conseil */}
              <div className="mt-4 pt-4 border-t border-gray-700">
                <div className="text-sm text-gray-400">
                  💡 <span className="font-medium">{opp.reason}</span>
                </div>
              </div>

              {/* Explications détaillées des scores (collapsible) */}
              {opp.score_explanation && (
                <details className="mt-3 pt-3 border-t border-gray-700">
                  <summary className="cursor-pointer text-xs text-blue-400 hover:text-blue-300 font-medium">
                    📊 Voir le détail des calculs de score
                  </summary>
                  <div className="mt-3 space-y-2 text-xs bg-gray-800 p-3 rounded">
                    <div>
                      <div className="text-yellow-400 font-bold">🔹 Trend ({opp.score_details?.trend || 0}/25):</div>
                      <div className="text-gray-300 ml-3 mt-1">{opp.score_explanation.trend}</div>
                    </div>
                    <div>
                      <div className="text-green-400 font-bold">🔹 Momentum ({opp.score_details?.momentum || 0}/25):</div>
                      <div className="text-gray-300 ml-3 mt-1">{opp.score_explanation.momentum}</div>
                    </div>
                    <div>
                      <div className="text-blue-400 font-bold">🔹 Volume ({opp.score_details?.volume || 0}/20):</div>
                      <div className="text-gray-300 ml-3 mt-1">{opp.score_explanation.volume}</div>
                    </div>
                    <div>
                      <div className="text-purple-400 font-bold">🔹 Price Action ({opp.score_details?.price_action || 0}/20):</div>
                      <div className="text-gray-300 ml-3 mt-1">{opp.score_explanation.price_action}</div>
                    </div>
                    <div>
                      <div className="text-pink-400 font-bold">🔹 Consensus ({opp.score_details?.consensus || 0}/10):</div>
                      <div className="text-gray-300 ml-3 mt-1">{opp.score_explanation.consensus}</div>
                    </div>
                  </div>
                </details>
              )}
            </div>
          );
        })}
      </div>

      {opportunities.length === 0 && !loading && (
        <div className="text-center py-12">
          <div className="text-gray-400 text-lg">
            Aucune opportunité détectée pour le moment
          </div>
          <button
            onClick={handleRefresh}
            className="mt-4 bg-primary-500 hover:bg-primary-600 text-white px-6 py-2 rounded-lg transition-colors"
          >
            🔄 Recharger
          </button>
        </div>
      )}
    </div>
  );
}

export default ManualTradingPage;
