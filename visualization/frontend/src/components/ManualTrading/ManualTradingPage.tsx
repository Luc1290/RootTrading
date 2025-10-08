import React, { useState, useEffect } from 'react';
import toast from 'react-hot-toast';
import { apiService } from '@/services/api';
import { formatCurrency, formatPercent } from '@/utils';

interface AutoSignal {
  has_signal: boolean;
  validated: boolean;
  side?: string;
  confidence?: number;
  consensus_strength?: number;
  strategies_count?: number;
  strategies?: string[];
  signals_count?: number;
  buy_signals?: number;
  sell_signals?: number;
  rejection_reason?: string;
  last_signal_time?: string;
}

interface TradingOpportunity {
  symbol: string;
  currentPrice: number;
  // Nouvelles données conditions-based (pas de score)
  conditions?: {
    volume_breakout: boolean;
    momentum_alignment: boolean;
    trend_quality: boolean;
    no_resistance: boolean;
  };
  // Données techniques
  signals24h: number;
  signalsConfidence: number;
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
  // Targets
  entryZone: { min: number; max: number };
  targets: { tp1: number; tp2: number; tp3?: number };
  stopLoss: number;
  recommendedSize: { min: number; max: number };
  // Action
  action: 'BUY_NOW' | 'WAIT' | 'WAIT_HIGHER_TF' | 'WAIT_QUALITY_GATE';
  reason: string;
  estimatedHoldTime?: string;
  autoSignal?: AutoSignal; // Signaux automatiques
}

function ManualTradingPage() {
  const [opportunities, setOpportunities] = useState<TradingOpportunity[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [capitalSize, setCapitalSize] = useState<number>(5000); // Taille position par défaut
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Calculateur rapide
  const [calcEntry, setCalcEntry] = useState<string>('');
  const [calcExit, setCalcExit] = useState<string>('');
  const [calcSize, setCalcSize] = useState<string>('');
  const [calcVariation, setCalcVariation] = useState<string>('--');
  const [calcPnL, setCalcPnL] = useState<string>('--');
  const [calcColor, setCalcColor] = useState<string>('text-white');

  // Top signaux
  const [topSignals, setTopSignals] = useState<any[]>([]);

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
        symbols.map(async (symbol) => {
          try {
            // Récupérer signaux manuel, market data ET signaux auto en parallèle
            const [signalsData, marketData, autoSignals] = await Promise.all([
              fetch(`/api/trading-opportunities/${symbol}`).then(r => r.json()).catch(() => null),
              apiService.getMarketData(symbol as any, '1m', 100).catch(() => null),
              fetch(`/api/automatic-signals/${symbol}`).then(r => r.json()).catch(() => null)
            ]);

            if (!signalsData || !marketData) return null;

            const currentPrice = marketData.data.close[marketData.data.close.length - 1] || 0;

            return {
              symbol,
              currentPrice,
              conditions: signalsData.conditions,
              signals24h: signalsData.signals_count || 0,
              signalsConfidence: signalsData.avg_confidence || 0,
              volumeRatio: signalsData.raw_data?.rel_volume || 0,
              regime: signalsData.market_regime || 'UNKNOWN',
              adx: signalsData.raw_data?.adx,
              rsi: signalsData.raw_data?.rsi,
              mfi: signalsData.raw_data?.mfi,
              volume_context: signalsData.volume_context,
              volume_quality_score: signalsData.raw_data?.volume_quality_score,
              nearest_support: signalsData.nearest_support,
              nearest_resistance: signalsData.raw_data?.nearest_resistance,
              break_probability: signalsData.break_probability,
              entryZone: signalsData.entry_zone || { min: currentPrice * 0.998, max: currentPrice * 1.002 },
              targets: signalsData.targets || {
                tp1: currentPrice * 1.01,
                tp2: currentPrice * 1.015,
              },
              stopLoss: signalsData.stop_loss || currentPrice * 0.988,
              recommendedSize: signalsData.recommended_size || { min: 3000, max: 7000 },
              action: signalsData.action || 'WAIT',
              reason: signalsData.reason || 'Analyse en cours',
              estimatedHoldTime: signalsData.estimated_hold_time,
              autoSignal: autoSignals || undefined
            } as TradingOpportunity;
          } catch (err) {
            console.error(`Error loading ${symbol}:`, err);
            return null;
          }
        })
      );

      // Filtrer les nulls seulement (tri fait dans useEffect après chargement des signaux)
      const validOpportunities = opportunitiesData
        .filter((opp): opp is TradingOpportunity => opp !== null);

      setOpportunities(validOpportunities);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Erreur lors du chargement des opportunités:', error);
      toast.error('Erreur lors du chargement des opportunités');
    } finally {
      setLoading(false);
    }
  };

  // Charger les top signaux
  const loadTopSignals = async () => {
    try {
      const response = await fetch(`/api/top-signals?timeframe_minutes=15&limit=50`);
      const data = await response.json();
      setTopSignals(data.signals || []);
    } catch (error) {
      console.error('Erreur chargement top signaux:', error);
    }
  };

  // Chargement initial et auto-refresh
  useEffect(() => {
    const loadData = async () => {
      await loadTopSignals(); // Charger les signaux d'abord
      await loadOpportunities(); // Puis les opportunités (pour tri par net_signal)
    };
    loadData();

    const interval = setInterval(loadData, 60000); // Refresh toutes les 60s (1min = plus court timeframe)
    return () => clearInterval(interval);
  }, []);

  // Retrier les opportunités quand topSignals change
  useEffect(() => {
    if (opportunities.length > 0 && topSignals.length > 0) {
      const sorted = [...opportunities].sort((a, b) => {
        // BUY_NOW en premier
        if (a.action === 'BUY_NOW' && b.action !== 'BUY_NOW') return -1;
        if (a.action !== 'BUY_NOW' && b.action === 'BUY_NOW') return 1;

        // Si aucun BUY_NOW, utiliser le net_signal des top signaux
        const aSignal = topSignals.find(s => s.symbol === a.symbol);
        const bSignal = topSignals.find(s => s.symbol === b.symbol);

        if (aSignal && bSignal) {
          return bSignal.net_signal - aSignal.net_signal;
        }
        if (aSignal) return -1;
        if (bSignal) return 1;

        // Sinon par nombre de conditions remplies
        const aConditions = a.conditions ? Object.values(a.conditions).filter(Boolean).length : 0;
        const bConditions = b.conditions ? Object.values(b.conditions).filter(Boolean).length : 0;
        return bConditions - aConditions;
      });

      // Vérifier si l'ordre a changé avant de mettre à jour (éviter boucle infinie)
      const orderChanged = sorted.some((opp, idx) => opp.symbol !== opportunities[idx].symbol);
      if (orderChanged) {
        setOpportunities(sorted);
      }
    }
  }, [topSignals, opportunities]);

  // Calculateur variation en temps réel
  useEffect(() => {
    const entry = parseFloat(calcEntry || '0');
    const exit = parseFloat(calcExit || '0');
    const size = parseFloat(calcSize || '0');

    if (entry > 0 && exit > 0) {
      const variation = ((exit - entry) / entry) * 100;
      const color = variation >= 0 ? 'text-green-400' : 'text-red-400';
      setCalcVariation(`${variation >= 0 ? '+' : ''}${variation.toFixed(2)}%`);
      setCalcColor(color);

      if (size > 0) {
        const pnl = (exit - entry) * (size / entry);
        setCalcPnL(`${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)} USDC`);
      } else {
        setCalcPnL('--');
      }
    } else {
      setCalcVariation('--');
      setCalcPnL('--');
      setCalcColor('text-white');
    }
  }, [calcEntry, calcExit, calcSize]);

  // Auto-refresh toutes les 60 secondes
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      loadOpportunities();
      loadTopSignals();
    }, 60000);

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
      case 'WAIT_QUALITY_GATE': return 'bg-orange-500/20 text-orange-400 border-orange-500/50';
      case 'AVOID': return 'bg-gray-700/20 text-gray-500 border-gray-700/50';
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/50';
    }
  };

  const getActionText = (action: string) => {
    switch (action) {
      case 'BUY_NOW': return 'ACHETER MAINTENANT ✅';
      case 'WAIT': return 'ATTENDRE ⏳';
      case 'WAIT_HIGHER_TF': return 'ATTENDRE - 5m pas aligné 📊';
      case 'WAIT_QUALITY_GATE': return 'BLOQUÉ - SETUP POURRI 🚫';
      default: return action;
    }
  };

  const getActionIcon = (action: string) => {
    switch (action) {
      case 'BUY_NOW': return '🟢';
      case 'WAIT': return '⚪';
      case 'WAIT_HIGHER_TF': return '🟡';
      case 'WAIT_QUALITY_GATE': return '🚫';
      default: return '⚪';
    }
  };

  const getConditionsColor = (count: number) => {
    // Couleur basée sur nombre de conditions remplies (sur 4)
    if (count === 4) return 'text-emerald-400';  // 4/4 = BUY_NOW
    if (count === 3) return 'text-green-400';    // 3/4 = proche
    if (count === 2) return 'text-yellow-400';   // 2/4 = moyen
    if (count === 1) return 'text-orange-400';   // 1/4 = faible
    return 'text-red-400';                        // 0/4 = très mauvais
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
    <div className="flex gap-6">
      {/* Contenu principal */}
      <div className="flex-1 space-y-6">
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
            {/* Calculateur variation rapide */}
            <div className="bg-dark-300 border border-gray-600 rounded-lg px-4 py-2">
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-2">
                  <label className="text-xs text-gray-400">Entrée:</label>
                  <input
                    type="number"
                    step="0.00000001"
                    value={calcEntry}
                    onChange={(e) => setCalcEntry(e.target.value)}
                    className="bg-dark-400 border border-gray-600 rounded px-2 py-1 text-white w-28 text-sm"
                    placeholder="Prix"
                  />
                </div>
                <div className="flex items-center space-x-2">
                  <label className="text-xs text-gray-400">Sortie:</label>
                  <input
                    type="number"
                    step="0.00000001"
                    value={calcExit}
                    onChange={(e) => setCalcExit(e.target.value)}
                    className="bg-dark-400 border border-gray-600 rounded px-2 py-1 text-white w-28 text-sm"
                    placeholder="Prix"
                  />
                </div>
                <div className="flex items-center space-x-2">
                  <label className="text-xs text-gray-400">Position:</label>
                  <input
                    type="number"
                    step="100"
                    value={calcSize}
                    onChange={(e) => setCalcSize(e.target.value)}
                    className="bg-dark-400 border border-gray-600 rounded px-2 py-1 text-white w-24 text-sm"
                    placeholder="USDC"
                  />
                </div>
                <div className="border-l border-gray-600 pl-3">
                  <div className="text-sm font-mono">
                    <span className="text-gray-400">Var: </span>
                    <span className={calcColor}>{calcVariation}</span>
                  </div>
                  <div className="text-xs font-mono">
                    <span className="text-gray-400">P&L: </span>
                    <span className={calcColor}>{calcPnL}</span>
                  </div>
                </div>
              </div>
            </div>

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

            {/* Toggle auto-refresh (switch style) */}
            <div className="flex items-center space-x-3">
              <span className="text-sm text-gray-400">Auto-refresh (60s)</span>
              <button
                onClick={() => setAutoRefresh(!autoRefresh)}
                aria-label={autoRefresh ? "Désactiver auto-refresh" : "Activer auto-refresh"}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  autoRefresh ? 'bg-green-500' : 'bg-gray-600'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    autoRefresh ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>

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
            <div className="text-3xl font-bold text-gray-400">
              {opportunities.filter(o => o.action.startsWith('WAIT')).length}
            </div>
            <div className="text-sm text-gray-400">⚪ ATTENDRE</div>
          </div>
        </div>
        <div className="chart-container">
          <div className="text-center">
            <div className="text-3xl font-bold text-orange-400">
              {opportunities.filter(o => o.action === 'WAIT_QUALITY_GATE').length}
            </div>
            <div className="text-sm text-gray-400">🚫 SETUP POURRI</div>
          </div>
        </div>
        <div className="chart-container">
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-400">
              {opportunities.filter(o => {
                const count = o.conditions ? Object.values(o.conditions).filter(Boolean).length : 0;
                return count >= 3;
              }).length}
            </div>
            <div className="text-sm text-gray-400">⭐ 3-4/4 conditions</div>
          </div>
        </div>
      </div>

      {/* Liste des opportunités */}
      <div className="space-y-4">
        {opportunities.map((opp, index) => {
          const tp1Gain = calculatePotentialGain(opp.currentPrice, opp.targets.tp1, capitalSize);
          const tp2Gain = calculatePotentialGain(opp.currentPrice, opp.targets.tp2, capitalSize);
          const slLoss = calculatePotentialGain(opp.currentPrice, opp.stopLoss, capitalSize);

          return (
            <div
              key={opp.symbol}
              className="bg-dark-200 border border-gray-700 rounded-lg p-6 hover:border-primary-500/50 transition-colors"
            >
              <div className="grid grid-cols-12 gap-6">
                {/* Colonne 1: Rang & Symbole & Conditions */}
                <div className="col-span-2">
                  <div className="flex items-center space-x-3">
                    <div className="text-4xl font-bold text-gray-600">
                      {index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : `#${index + 1}`}
                    </div>
                    <div>
                      <div className="text-xl font-bold text-white">{opp.symbol}</div>
                      {opp.conditions && (
                        <div className={`text-2xl font-mono font-bold ${getConditionsColor(Object.values(opp.conditions).filter(Boolean).length)}`}>
                          {Object.values(opp.conditions).filter(Boolean).length}/4 ✓
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Colonne 2: Action & Prix */}
                <div className="col-span-3">
                  <div className={`inline-block px-4 py-2 rounded-lg border font-bold mb-3 ${getActionColor(opp.action)}`}>
                    {getActionIcon(opp.action)} {getActionText(opp.action)}
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

                {/* Colonne 5: Conditions détaillées */}
                <div className="col-span-2">
                  {opp.conditions ? (
                    <div className="space-y-2">
                      <div className="text-xs text-gray-400 font-semibold mb-2">4 Conditions Critiques</div>
                      <div className="space-y-1">
                        <div className={`text-xs ${opp.conditions.volume_breakout ? 'text-green-400' : 'text-red-400'}`}>
                          {opp.conditions.volume_breakout ? '✅' : '❌'} Volume Breakout
                        </div>
                        <div className={`text-xs ${opp.conditions.momentum_alignment ? 'text-green-400' : 'text-red-400'}`}>
                          {opp.conditions.momentum_alignment ? '✅' : '❌'} Momentum Aligned
                        </div>
                        <div className={`text-xs ${opp.conditions.trend_quality ? 'text-green-400' : 'text-red-400'}`}>
                          {opp.conditions.trend_quality ? '✅' : '❌'} Trend Quality
                        </div>
                        <div className={`text-xs ${opp.conditions.no_resistance ? 'text-green-400' : 'text-red-400'}`}>
                          {opp.conditions.no_resistance ? '✅' : '❌'} Smart Resistance
                        </div>
                      </div>
                      <div className="text-xs text-gray-500 mt-2">
                        Vol: {opp.volumeRatio.toFixed(1)}x | ADX: {opp.adx?.toFixed(0) || 'N/A'} | RSI: {opp.rsi?.toFixed(0) || 'N/A'}
                      </div>
                    </div>
                  ) : (
                    <div className="text-xs text-gray-500">Conditions non disponibles</div>
                  )}
                </div>
              </div>

              {/* Raison / Conseil */}
              <div className="mt-4 pt-4 border-t border-gray-700">
                <div className="text-sm text-gray-400">
                  💡 <span className="font-medium">{opp.reason}</span>
                </div>
              </div>

              {/* NEW: Comparaison Auto vs Manuel */}
              {opp.autoSignal && (
                <div className="mt-4 pt-4 border-t border-gray-700">
                  <div className="text-xs text-gray-400 mb-3 font-semibold">🔀 Comparaison Sources de Signaux</div>
                  <div className="grid grid-cols-2 gap-4">
                    {/* Colonne Manuel */}
                    <div className="bg-dark-300 p-3 rounded border border-purple-500/30">
                      <div className="text-xs text-purple-400 font-semibold mb-2">📊 MANUEL (opportunity_calculator)</div>
                      <div className="space-y-1 text-xs">
                        <div className="flex justify-between">
                          <span className="text-gray-400">Conditions:</span>
                          <span className={`font-bold ${getConditionsColor(opp.conditions ? Object.values(opp.conditions).filter(Boolean).length : 0)}`}>
                            {opp.conditions ? Object.values(opp.conditions).filter(Boolean).length : 0}/4 ✓
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Action:</span>
                          <span className={`font-semibold ${getActionColor(opp.action).split(' ')[1]}`}>
                            {opp.action}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Timeframe:</span>
                          <span className="text-white font-semibold">1m/5m</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-gray-400">Logique:</span>
                          <span className="text-gray-300">Momentum Adapt.</span>
                        </div>
                      </div>
                    </div>

                    {/* Colonne Auto */}
                    <div className={`bg-dark-300 p-3 rounded border ${
                      opp.autoSignal.validated ? 'border-green-500/30' : 'border-red-500/30'
                    }`}>
                      <div className="text-xs text-blue-400 font-semibold mb-2">🤖 AUTO (signal_aggregator)</div>
                      {opp.autoSignal.has_signal ? (
                        <div className="space-y-1 text-xs">
                          <div className="flex justify-between">
                            <span className="text-gray-400">Consensus:</span>
                            <span className="font-bold text-blue-400">
                              {opp.autoSignal.consensus_strength?.toFixed(2) || 'N/A'}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Stratégies:</span>
                            <span className="text-white font-semibold">{opp.autoSignal.strategies_count || 0}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Side:</span>
                            <span className={`font-semibold ${
                              opp.autoSignal.side === 'BUY' ? 'text-green-400' :
                              opp.autoSignal.side === 'SELL' ? 'text-red-400' : 'text-gray-400'
                            }`}>
                              {opp.autoSignal.side}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-400">Status:</span>
                            <span className={`font-semibold ${
                              opp.autoSignal.validated ? 'text-green-400' : 'text-red-400'
                            }`}>
                              {opp.autoSignal.validated ? '✅ Validé' : '❌ Rejeté'}
                            </span>
                          </div>
                          {opp.autoSignal.rejection_reason && (
                            <div className="text-xs text-red-400 mt-2 italic">
                              {opp.autoSignal.rejection_reason}
                            </div>
                          )}
                          {opp.autoSignal.validated && opp.action === 'BUY_NOW' && opp.autoSignal.side === 'BUY' && (
                            <div className="mt-2 p-2 bg-green-500/10 border border-green-500/30 rounded text-center">
                              <span className="text-green-400 font-bold text-xs">🔥 DOUBLE CONFIRMATION BUY</span>
                              <div className="text-xs text-gray-400 mt-1">Manuel + Auto alignés</div>
                            </div>
                          )}
                          {opp.autoSignal.validated && opp.action !== 'BUY_NOW' && opp.autoSignal.side === 'BUY' && (
                            <div className="mt-2 p-2 bg-yellow-500/10 border border-yellow-500/30 rounded text-center">
                              <span className="text-yellow-400 font-bold text-xs">⚠️ DIVERGENCE</span>
                              <div className="text-xs text-gray-400 mt-1">Auto dit BUY, Manuel dit {opp.action}</div>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="text-xs text-gray-500 italic">
                          Aucun signal automatique récent (15min)
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Détails techniques additionnels (collapsible) */}
              <details className="mt-3 pt-3 border-t border-gray-700">
                <summary className="cursor-pointer text-xs text-blue-400 hover:text-blue-300 font-medium">
                  📊 Voir les détails techniques complets
                </summary>
                <div className="mt-3 space-y-2 text-xs bg-gray-800 p-3 rounded">
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <div className="text-yellow-400 font-bold mb-1">📈 Trend</div>
                      <div className="text-gray-300">ADX: {opp.adx?.toFixed(1) || 'N/A'}</div>
                      <div className="text-gray-300">Régime: {opp.regime}</div>
                    </div>
                    <div>
                      <div className="text-green-400 font-bold mb-1">⚡ Momentum</div>
                      <div className="text-gray-300">RSI: {opp.rsi?.toFixed(0) || 'N/A'}</div>
                      <div className="text-gray-300">MFI: {opp.mfi?.toFixed(0) || 'N/A'}</div>
                    </div>
                    <div>
                      <div className="text-blue-400 font-bold mb-1">🔊 Volume</div>
                      <div className="text-gray-300">Ratio: {opp.volumeRatio.toFixed(2)}x</div>
                      <div className="text-gray-300">Context: {opp.volume_context || 'N/A'}</div>
                      <div className="text-gray-300">Quality: {opp.volume_quality_score?.toFixed(0) || 'N/A'}/100</div>
                    </div>
                  </div>
                  {(opp.nearest_support || opp.nearest_resistance) && (
                    <div className="mt-3 pt-3 border-t border-gray-700">
                      <div className="text-purple-400 font-bold mb-1">🎯 Support / Résistance</div>
                      <div className="grid grid-cols-2 gap-2">
                        {opp.nearest_support && (
                          <div className="text-gray-300">Support: {opp.nearest_support.toFixed(6)}</div>
                        )}
                        {opp.nearest_resistance && (
                          <div className="text-gray-300">Résistance: {opp.nearest_resistance.toFixed(6)}</div>
                        )}
                      </div>
                      {opp.break_probability !== undefined && (
                        <div className="text-gray-300 mt-1">Break prob: {opp.break_probability.toFixed(0)}%</div>
                      )}
                    </div>
                  )}
                </div>
              </details>
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

    {/* Panneau latéral - Top Signaux */}
    <div className="w-80 space-y-4">
      <div className="bg-dark-200 border border-gray-700 rounded-lg p-4 sticky top-4 max-h-screen overflow-y-auto">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-bold text-white">
            🔥❄️ Top Signaux
          </h2>
          <div className="text-right">
            <div className="text-xs text-gray-400">15min</div>
            <div className="text-xs text-blue-400">
              {topSignals.length} signaux
            </div>
          </div>
        </div>

        {/* Toggle BUY/SELL */}
        <div className="space-y-3 pr-2">
          {topSignals.length === 0 ? (
            <div className="text-center text-gray-400 py-4">
              Aucun signal récent
            </div>
          ) : (
            topSignals.map((signal, index) => {
              const isBuy = signal.dominant_side === 'BUY';
              const isSell = signal.dominant_side === 'SELL';
              const hoverColor = isBuy ? 'hover:border-green-500' : (isSell ? 'hover:border-red-500' : 'hover:border-gray-500');
              const icon = isBuy ? '🔥' : (isSell ? '❄️' : '⚪');

              const baseAsset = signal.symbol.replace('USDC', '');
              const binanceUrl = `https://www.binance.com/en/trade/${baseAsset}_USDC?type=spot`;

              return (
              <div
                key={signal.symbol}
                onClick={() => window.open(binanceUrl, '_blank')}
                className={`bg-dark-300 border border-gray-600 rounded-lg p-3 ${hoverColor} transition-all cursor-pointer hover:scale-102 hover:shadow-lg`}
                title={`Cliquez pour ouvrir ${baseAsset}/USDC sur Binance`}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <span className="text-2xl">{icon}</span>
                    <span className="text-base font-bold text-white">{baseAsset}</span>
                    <span className={`text-base font-bold ${isBuy ? 'text-green-400' : (isSell ? 'text-red-400' : 'text-gray-400')}`}>
                      {signal.net_signal > 0 ? `+${signal.net_signal}` : signal.net_signal}
                    </span>
                  </div>
                  <div className="text-right">
                    <div className="text-sm space-x-1">
                      <span className="text-green-400 font-semibold">{signal.buy_count}🟢</span>
                      <span className="text-red-400 font-semibold">{signal.sell_count}🔴</span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between text-sm text-gray-400">
                  <span>
                    {new Date(signal.last_signal_time).toLocaleTimeString('fr-FR', {
                      hour: '2-digit',
                      minute: '2-digit'
                    })}
                  </span>
                  <span className="font-semibold">
                    {Math.max(signal.buy_confidence, signal.sell_confidence) * 100 | 0}%
                  </span>
                </div>
              </div>
              );
            })
          )}
        </div>
      </div>
    </div>
    </div>
  );
}

export default ManualTradingPage;
