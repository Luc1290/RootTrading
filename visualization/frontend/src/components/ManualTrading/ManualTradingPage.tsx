import React, { useState, useEffect } from 'react';
import toast from 'react-hot-toast';
import { apiService } from '@/services/api';
import { formatCurrency, formatPercent } from '@/utils';
import { OpportunityCard } from './OpportunityCard';
import { MarketSentiment } from '@/components/Shared/MarketSentiment';

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

  // Système PRO - Score & Grade
  score?: {
    total: number;
    grade: string;
    confidence: number;
  };

  // Catégories PRO (score détaillé par catégorie)
  categoryScores?: {
    trend?: { score: number; confidence: number };
    momentum?: { score: number; confidence: number };
    volume?: { score: number; confidence: number };
    volatility?: { score: number; confidence: number };
    support_resistance?: { score: number; confidence: number };
    pattern?: { score: number; confidence: number };
    confluence?: { score: number; confidence: number };
  };

  // Validation
  validation?: {
    all_passed: boolean;
    overall_score: number;
  };

  // Détails validation (pour afficher les problèmes)
  validationDetails?: {
    blocking_issues?: string[];
    warnings?: string[];
  };

  // Ancien système (rétrocompatibilité)
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

  // Pricing PRO
  pricing?: {
    current_price: number;
    entry_optimal: number;
    entry_aggressive: number;
  };

  // Risk PRO
  risk?: {
    rr_ratio: number;
    risk_level: string;
    max_position_size_pct: number;
  };

  // Targets
  entryZone: { min: number; max: number };
  targets: { tp1: number | { price: number; percent: number }; tp2: number | { price: number; percent: number }; tp3?: number | { price: number; percent: number } };
  stopLoss: number | { price: number; percent: number };
  recommendedSize: { min: number; max: number };

  // Action
  action: 'BUY_NOW' | 'BUY_DCA' | 'EARLY_ENTRY' | 'WAIT' | 'AVOID' | 'WAIT_HIGHER_TF' | 'WAIT_QUALITY_GATE';
  reason: string;
  reasons?: string[];
  warnings?: string[];
  recommendations?: string[];
  estimatedHoldTime?: string;
  autoSignal?: AutoSignal;
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

            // Extraire targets (nouveau format PRO ou ancien format)
            const tp1 = signalsData.targets?.tp1?.price || signalsData.targets?.tp1 || currentPrice * 1.01;
            const tp2 = signalsData.targets?.tp2?.price || signalsData.targets?.tp2 || currentPrice * 1.015;
            const tp3 = signalsData.targets?.tp3?.price || signalsData.targets?.tp3;
            const stopLoss = signalsData.stop_loss?.price || signalsData.stop_loss || currentPrice * 0.988;

            // Construire entry zone (PRO ou ancien)
            const entryOptimal = signalsData.pricing?.entry_optimal || currentPrice * 0.998;
            const entryAggressive = signalsData.pricing?.entry_aggressive || currentPrice;

            // Extraire les raw_data du debug (analyzer_data complet)
            const rawData = signalsData.debug?.raw_data || {};

            // Extraire les catégories de score
            const categoryScores = signalsData.debug?.score_details?.category_scores ? {
              trend: {
                score: signalsData.debug.score_details.category_scores.trend?.score || 0,
                confidence: signalsData.debug.score_details.category_scores.trend?.confidence || 0
              },
              momentum: {
                score: signalsData.debug.score_details.category_scores.momentum?.score || 0,
                confidence: signalsData.debug.score_details.category_scores.momentum?.confidence || 0
              },
              volume: {
                score: signalsData.debug.score_details.category_scores.volume?.score || 0,
                confidence: signalsData.debug.score_details.category_scores.volume?.confidence || 0
              },
              volatility: {
                score: signalsData.debug.score_details.category_scores.volatility?.score || 0,
                confidence: signalsData.debug.score_details.category_scores.volatility?.confidence || 0
              },
              support_resistance: {
                score: signalsData.debug.score_details.category_scores.support_resistance?.score || 0,
                confidence: signalsData.debug.score_details.category_scores.support_resistance?.confidence || 0
              },
              pattern: {
                score: signalsData.debug.score_details.category_scores.pattern?.score || 0,
                confidence: signalsData.debug.score_details.category_scores.pattern?.confidence || 0
              },
              confluence: {
                score: signalsData.debug.score_details.category_scores.confluence?.score || 0,
                confidence: signalsData.debug.score_details.category_scores.confluence?.confidence || 0
              }
            } : undefined;

            // Extraire les détails de validation
            const validationDetails = signalsData.debug?.validation_details ? {
              blocking_issues: signalsData.debug.validation_details.blocking_issues || [],
              warnings: signalsData.debug.validation_details.warnings || []
            } : undefined;

            return {
              symbol,
              currentPrice,

              // Système PRO
              score: signalsData.score,
              categoryScores,
              validation: signalsData.validation,
              validationDetails,
              pricing: signalsData.pricing,
              risk: signalsData.risk,

              // Ancien système (rétrocompatibilité)
              conditions: signalsData.conditions,

              signals24h: signalsData.signals_count || 0,
              signalsConfidence: signalsData.avg_confidence || signalsData.confidence || 0,
              volumeRatio: rawData.relative_volume || rawData.rel_volume || 1.0,
              regime: signalsData.context?.market_regime || 'UNKNOWN',
              adx: rawData.adx_14,
              rsi: rawData.rsi_14,
              mfi: rawData.mfi_14,
              volume_context: signalsData.context?.volume_context,
              volume_quality_score: rawData.volume_quality_score,
              nearest_support: rawData.nearest_support,
              nearest_resistance: rawData.nearest_resistance,
              break_probability: rawData.break_probability,

              entryZone: { min: entryOptimal, max: entryAggressive },
              targets: { tp1, tp2, tp3 },
              stopLoss,
              recommendedSize: signalsData.risk ?
                { min: capitalSize * (signalsData.risk.max_position_size_pct / 100) * 0.8, max: capitalSize * (signalsData.risk.max_position_size_pct / 100) } :
                { min: 3000, max: 7000 },

              action: signalsData.action || 'WAIT',
              reason: signalsData.reasons?.[0] || signalsData.reason || 'Analyse en cours',
              reasons: signalsData.reasons,
              warnings: signalsData.warnings,
              recommendations: signalsData.recommendations,
              estimatedHoldTime: signalsData.timing?.estimated_hold_time,
              autoSignal: autoSignals || undefined
            } as TradingOpportunity;
          } catch (err) {
            console.error(`Error loading ${symbol}:`, err);
            return null;
          }
        })
      );

      // Filtrer les nulls ET TRIER UNE SEULE FOIS (évite boucle de re-render)
      const validOpportunities = opportunitiesData
        .filter((opp): opp is TradingOpportunity => opp !== null)
        .sort((a, b) => {
          // 1. BUY_NOW en premier absolu
          if (a.action === 'BUY_NOW' && b.action !== 'BUY_NOW') return -1;
          if (a.action !== 'BUY_NOW' && b.action === 'BUY_NOW') return 1;

          // 2. EARLY_ENTRY en deuxième (signaux précoces prioritaires)
          if (a.action === 'EARLY_ENTRY' && !['BUY_NOW'].includes(b.action)) return -1;
          if (b.action === 'EARLY_ENTRY' && !['BUY_NOW'].includes(a.action)) return 1;

          // 3. BUY_DCA en troisième
          if (a.action === 'BUY_DCA' && !['BUY_NOW', 'EARLY_ENTRY'].includes(b.action)) return -1;
          if (b.action === 'BUY_DCA' && !['BUY_NOW', 'EARLY_ENTRY'].includes(a.action)) return 1;

          // 4. Ensuite par score total (système PRO)
          const aScore = a.score?.total || 0;
          const bScore = b.score?.total || 0;

          if (aScore !== bScore) {
            return bScore - aScore; // Score décroissant
          }

          // 5. Si scores égaux, par validation score
          const aValidation = a.validation?.overall_score || 0;
          const bValidation = b.validation?.overall_score || 0;

          return bValidation - aValidation;
        });

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
      await loadOpportunities(); // Puis les opportunités (tri fait dans loadOpportunities)
    };
    loadData();

    const interval = setInterval(loadData, 60000); // Refresh toutes les 60s (1min = plus court timeframe)
    return () => clearInterval(interval);
  }, []);

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
      case 'EARLY_ENTRY': return 'bg-purple-500/20 text-purple-400 border-purple-500/50';
      case 'BUY_DCA': return 'bg-blue-500/20 text-blue-400 border-blue-500/50';
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
      case 'EARLY_ENTRY': return '⚡ ENTRY PRÉCOCE';
      case 'BUY_DCA': return 'ACHETER 📊';
      case 'WAIT': return 'ATTENDRE ⏳';
      case 'WAIT_HIGHER_TF': return 'ATTENDRE - 5m pas aligné 📊';
      case 'WAIT_QUALITY_GATE': return 'BLOQUÉ - SETUP POURRI 🚫';
      case 'AVOID': return 'ÉVITER 🚫';
      default: return action;
    }
  };

  const getActionIcon = (action: string) => {
    switch (action) {
      case 'BUY_NOW': return '🟢';
      case 'EARLY_ENTRY': return '🟣';
      case 'BUY_DCA': return '🔵';
      case 'WAIT': return '⚪';
      case 'WAIT_HIGHER_TF': return '🟡';
      case 'WAIT_QUALITY_GATE': return '🚫';
      case 'AVOID': return '🔴';
      default: return '⚪';
    }
  };

  const getGradeColor = (grade: string) => {
    switch (grade) {
      case 'S': return 'text-pink-400';
      case 'A': return 'text-green-400';
      case 'B': return 'text-blue-400';
      case 'C': return 'text-yellow-400';
      case 'D': return 'text-orange-400';
      case 'F': return 'text-red-400';
      default: return 'text-gray-400';
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
      {/* Sentiment Marché */}
      <MarketSentiment />

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
        {opportunities.map((opp, index) => (
          <OpportunityCard
            key={opp.symbol}
            opportunity={opp}
            rank={index + 1}
            autoSignal={opp.autoSignal}
          />
        ))}
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
                key={`${signal.symbol}-${index}`}
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
