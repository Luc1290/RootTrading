import React, { useState } from 'react';
import { ChevronDown, ChevronUp, TrendingUp, TrendingDown, AlertCircle, CheckCircle, XCircle } from 'lucide-react';
import { formatCurrency, formatPercent } from '@/utils';

interface OpportunityCardProps {
  opportunity: any;
  rank: number;
  autoSignal?: any;
}

export const OpportunityCard: React.FC<OpportunityCardProps> = ({ opportunity: opp, rank, autoSignal }) => {
  const [expanded, setExpanded] = useState(false);

  // Couleurs selon grade
  const gradeColors: Record<string, string> = {
    'S': 'from-purple-600 to-pink-600',
    'A': 'from-green-600 to-emerald-600',
    'B': 'from-blue-600 to-cyan-600',
    'C': 'from-yellow-600 to-orange-600',
    'D': 'from-orange-600 to-red-600',
    'F': 'from-red-600 to-red-800'
  };

  // Couleurs selon action
  const actionConfig: Record<string, { bg: string; text: string; icon: string; emoji: string }> = {
    'BUY_NOW': { bg: 'bg-green-600', text: 'text-white', icon: '🚀', emoji: '✅' },
    'BUY_DCA': { bg: 'bg-blue-600', text: 'text-white', icon: '📊', emoji: '🔵' },
    'EARLY_ENTRY': { bg: 'bg-purple-600', text: 'text-white', icon: '⚡', emoji: '🟣' },
    'WAIT': { bg: 'bg-yellow-600', text: 'text-black', icon: '⏸️', emoji: '⚪' },
    'AVOID': { bg: 'bg-red-600', text: 'text-white', icon: '🛑', emoji: '🔴' }
  };

  const config = actionConfig[opp.action] || actionConfig.AVOID;
  const grade = opp.score?.grade || 'F';
  const gradeColor = gradeColors[grade] || gradeColors['F'];

  // Prix et targets
  const currentPrice = opp.pricing?.current_price || opp.currentPrice || 0;

  // Calculer les pourcentages si non fournis
  const calculatePercent = (targetPrice: number, currentPrice: number) => {
    if (!targetPrice || !currentPrice) return 0;
    return ((targetPrice - currentPrice) / currentPrice) * 100;
  };

  const tp1Raw = typeof opp.targets?.tp1 === 'object' ? opp.targets.tp1 : { price: opp.targets?.tp1 || 0, percent: opp.targets?.tp1_percent || 0 };
  const tp2Raw = typeof opp.targets?.tp2 === 'object' ? opp.targets.tp2 : { price: opp.targets?.tp2 || 0, percent: opp.targets?.tp2_percent || 0 };
  const slRaw = typeof opp.stop_loss === 'object' ? opp.stop_loss : (typeof opp.stopLoss === 'object' ? opp.stopLoss : { price: opp.stop_loss?.price || opp.stopLoss || 0, percent: opp.stop_loss?.percent || 0 });

  // Fallback: calculer percent si manquant
  const tp1 = {
    price: tp1Raw.price || 0,
    percent: tp1Raw.percent || calculatePercent(tp1Raw.price, currentPrice)
  };
  const tp2 = {
    price: tp2Raw.price || 0,
    percent: tp2Raw.percent || calculatePercent(tp2Raw.price, currentPrice)
  };
  const sl = {
    price: slRaw.price || 0,
    percent: slRaw.percent || Math.abs(calculatePercent(slRaw.price, currentPrice))
  };

  // Calculer gains potentiels (sur 5000 USDC)
  const capital = 5000;
  const tp1Gain = tp1.percent ? (capital * tp1.percent / 100) : 0;
  const tp2Gain = tp2.percent ? (capital * tp2.percent / 100) : 0;
  const slLoss = sl.percent ? (capital * sl.percent / 100) : 0;

  // Traduction actions
  const actionText: Record<string, string> = {
    'BUY_NOW': 'Acheter maintenant',
    'BUY_DCA': 'Acheter',
    'EARLY_ENTRY': '⚡ Entry Précoce',
    'WAIT': 'Attendre',
    'WAIT_HIGHER_TF': 'Attendre (5m)',
    'WAIT_QUALITY_GATE': 'Setup faible',
    'AVOID': 'Éviter'
  };

  // Trier les scores par valeur décroissante
  const categoryScores = opp.score?.category_scores || opp.categoryScores || {};
  const sortedScores = Object.entries(categoryScores).sort(([, a]: [string, any], [, b]: [string, any]) => {
    return (b.score || 0) - (a.score || 0);
  });

  // Lien Binance
  const baseAsset = opp.symbol.replace('USDC', '');
  const binanceUrl = `https://www.binance.com/en/trade/${baseAsset}_USDC?type=spot`;

  return (
    <div
      className="bg-dark-200 rounded border border-gray-700 hover:border-gray-600 transition-all p-4 cursor-pointer hover:shadow-lg"
      onClick={() => window.open(binanceUrl, '_blank')}
      title={`Cliquer pour ouvrir ${baseAsset}/USDC sur Binance`}
    >
      {/* === HEADER COMPACT === */}
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-gray-700">
        <div className="flex items-center gap-4">
          {/* Rank + Symbol */}
          <div className="flex items-center gap-2">
            <span className="text-xl">{getRankEmoji(rank)}</span>
            <div>
              <h3 className="text-xl font-bold text-white">{opp.symbol}</h3>
              <div className="text-xs text-gray-500">{formatCurrency(currentPrice)}</div>
            </div>
          </div>

          {/* Score + Grade */}
          <div className="text-center px-3 py-1 bg-dark-300 rounded">
            <div className={`text-2xl font-bold ${
              grade === 'S' || grade === 'A' ? 'text-green-400' :
              grade === 'B' ? 'text-blue-400' :
              grade === 'C' ? 'text-yellow-400' :
              grade === 'D' ? 'text-orange-400' : 'text-gray-400'
            }`}>
              {opp.score?.total_score?.toFixed(0) || opp.score?.total?.toFixed(0) || 0}
              <span className="text-sm text-gray-500">/100</span>
            </div>
            <div className="text-xs text-gray-400">Grade {grade}</div>
          </div>
        </div>

        {/* Action button */}
        <div className={`px-6 py-2 rounded ${getActionBg(opp.action)}`}>
          <div className={`text-base font-bold ${getActionTextColor(opp.action)}`}>
            {actionText[opp.action] || opp.action}
          </div>
        </div>
      </div>

      {/* === EARLY SIGNAL WARNING (NOUVEAU) === */}
      {opp.is_early_entry && opp.early_signal && (
        <div className="mb-4 p-3 bg-purple-900/20 border border-purple-500/40 rounded-lg">
          <div className="flex items-center justify-between mb-2">
            <div className="font-semibold text-purple-300">
              🚀 EARLY ENTRY SIGNAL
            </div>
            <div className="text-sm text-purple-400 font-mono">
              Score: {opp.early_signal.score?.toFixed(0)}/100
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <span className="text-purple-400">Niveau:</span>
              <span className="ml-2 text-white font-semibold uppercase">
                {opp.early_signal.level}
              </span>
            </div>
            <div>
              <span className="text-purple-400">Entry window:</span>
              <span className="ml-2 text-white font-semibold">
                ~{opp.early_signal.estimated_entry_window_seconds}s
              </span>
            </div>
            <div>
              <span className="text-purple-400">Vélocité:</span>
              <span className="ml-2 text-white">
                {opp.early_signal.velocity_score?.toFixed(0)}/35
              </span>
            </div>
            <div>
              <span className="text-purple-400">Volume Buildup:</span>
              <span className="ml-2 text-white">
                {opp.early_signal.volume_buildup_score?.toFixed(0)}/25
              </span>
            </div>
          </div>
          {opp.early_signal.reasons && opp.early_signal.reasons.length > 0 && (
            <div className="mt-2 text-xs text-purple-300 space-y-0.5">
              {opp.early_signal.reasons.slice(0, 3).map((reason: string, i: number) => (
                <div key={i}>• {reason}</div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* === VALIDATION WARNINGS === */}
      {!opp.validation?.all_passed && opp.validationDetails?.blocking_issues && opp.validationDetails.blocking_issues.length > 0 && (
        <div className="mb-4 p-3 bg-orange-900/20 border border-orange-500/40 rounded-lg">
          <div className="font-semibold text-orange-300 mb-1">⚠️ Problèmes bloquants</div>
          <div className="text-sm text-orange-400 space-y-1">
            {opp.validationDetails.blocking_issues.map((issue: string, i: number) => (
              <div key={i}>• {issue}</div>
            ))}
          </div>
        </div>
      )}

      {/* === LAYOUT 2 COLONNES === */}
      <div className="grid grid-cols-2 gap-6">
        {/* ========== COLONNE GAUCHE: SCORES + CONTEXTE ========== */}
        <div className="space-y-4">
          {/* Bloc Scores (triés) */}
          <div className="bg-dark-300 rounded-lg p-4">
            <h4 className="text-xs text-gray-500 uppercase font-semibold mb-3 tracking-wide">📊 Scores Catégories</h4>
            <div className="space-y-2">
              {sortedScores.map(([key, value]: [string, any]) => {
                const score = value.score || 0;
                const categoryName = getCategoryName(key);
                return (
                  <div key={key} className="flex items-center justify-between">
                    <span className="text-sm text-gray-300">{categoryName}</span>
                    <div className="flex items-center gap-2">
                      <div className="w-20 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${
                            score >= 80 ? 'bg-green-400' :
                            score >= 60 ? 'bg-blue-400' :
                            score >= 40 ? 'bg-yellow-400' : 'bg-gray-500'
                          }`}
                          style={{ width: `${score}%` }}
                        />
                      </div>
                      <span className={`text-base font-bold w-8 text-right ${
                        score >= 80 ? 'text-green-400' :
                        score >= 60 ? 'text-blue-400' :
                        score >= 40 ? 'text-yellow-400' : 'text-gray-500'
                      }`}>
                        {score.toFixed(0)}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Bloc Contexte */}
          <div className="bg-dark-300 rounded-lg p-4">
            <h4 className="text-xs text-gray-500 uppercase font-semibold mb-3 tracking-wide">🌍 Contexte</h4>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Régime</span>
                <span className={`text-sm font-bold ${getRegimeColor(opp.context?.market_regime || opp.regime)}`}>
                  {((opp.context?.market_regime || opp.regime) || 'N/A').replace('TRENDING_', '').replace('BREAKOUT_', '')}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Validation</span>
                <span className="text-sm font-bold text-white">
                  {opp.validation?.overall_score?.toFixed(0) || 0}/100
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Volatilité</span>
                <span className="text-sm font-bold text-white">
                  {opp.context?.volatility_regime?.toUpperCase() || categoryScores?.volatility?.score?.toFixed(0) || 'N/A'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* ========== COLONNE DROITE: TRADING + INDICATEURS ========== */}
        <div className="space-y-4">
          {/* Bloc Trading */}
          <div className="bg-dark-300 rounded-lg p-4">
            <h4 className="text-xs text-gray-500 uppercase font-semibold mb-3 tracking-wide">💰 Niveaux Trading</h4>
            <div className="space-y-3">
              {/* TP1 */}
              <div className="flex items-center justify-between pb-2 border-b border-gray-700">
                <span className="text-sm text-gray-400 font-medium">TP1</span>
                <div className="text-right">
                  <div className="text-base font-bold text-white">{formatCurrency(tp1.price)}</div>
                  <div className="text-xs text-green-400 font-semibold">+{tp1.percent?.toFixed(2)}%</div>
                </div>
              </div>

              {/* TP2 */}
              <div className="flex items-center justify-between pb-2 border-b border-gray-700">
                <span className="text-sm text-gray-400 font-medium">TP2</span>
                <div className="text-right">
                  <div className="text-base font-bold text-white">{formatCurrency(tp2.price)}</div>
                  <div className="text-xs text-green-400 font-semibold">+{tp2.percent?.toFixed(2)}%</div>
                </div>
              </div>

              {/* SL */}
              <div className="flex items-center justify-between pb-2 border-b border-gray-700">
                <span className="text-sm text-gray-400 font-medium">Stop Loss</span>
                <div className="text-right">
                  <div className="text-base font-bold text-white">{formatCurrency(sl.price)}</div>
                  <div className="text-xs text-red-400 font-semibold">-{sl.percent?.toFixed(2)}%</div>
                </div>
              </div>

              {/* R/R + Risk */}
              <div className="grid grid-cols-2 gap-4 pt-1">
                <div>
                  <div className="text-xs text-gray-500">Risk/Reward</div>
                  <div className={`text-lg font-bold ${
                    (opp.risk?.rr_ratio || 0) >= 2 ? 'text-green-400' :
                    (opp.risk?.rr_ratio || 0) >= 1.5 ? 'text-yellow-400' : 'text-orange-400'
                  }`}>
                    {opp.risk?.rr_ratio?.toFixed(2) || 'N/A'}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-500">Risk Level</div>
                  <div className="text-lg font-bold text-white">{opp.risk?.risk_level || 'N/A'}</div>
                </div>
              </div>
            </div>
          </div>

          {/* Bloc Indicateurs Techniques */}
          <div className="bg-dark-300 rounded-lg p-4">
            <h4 className="text-xs text-gray-500 uppercase font-semibold mb-3 tracking-wide">📈 Indicateurs</h4>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <div className="text-xs text-gray-500">RSI</div>
                <div className={`text-lg font-bold ${
                  (opp.indicators?.rsi || 0) > 70 ? 'text-orange-400' :
                  (opp.indicators?.rsi || 0) < 30 ? 'text-blue-400' : 'text-white'
                }`}>
                  {opp.indicators?.rsi?.toFixed(0) || 'N/A'}
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500">MFI</div>
                <div className={`text-lg font-bold ${
                  (opp.indicators?.mfi || 0) > 80 ? 'text-orange-400' :
                  (opp.indicators?.mfi || 0) < 20 ? 'text-blue-400' : 'text-white'
                }`}>
                  {opp.indicators?.mfi?.toFixed(0) || 'N/A'}
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500">ADX</div>
                <div className={`text-lg font-bold ${
                  (opp.indicators?.adx || 0) > 25 ? 'text-green-400' : 'text-gray-400'
                }`}>
                  {opp.indicators?.adx?.toFixed(1) || 'N/A'}
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500">Volume</div>
                <div className={`text-lg font-bold ${
                  (opp.indicators?.volume_spike || 0) > 2 ? 'text-green-400' : 'text-white'
                }`}>
                  {opp.indicators?.volume_spike?.toFixed(2)}x
                </div>
              </div>
            </div>

            {/* v5.0 - Nouveaux indicateurs */}
            {(opp.raw_data?.analyzer_data?.pattern_detected || opp.raw_data?.analyzer_data?.confluence_score) && (
              <div className="mt-3 pt-3 border-t border-gray-700 grid grid-cols-2 gap-3">
                {opp.raw_data?.analyzer_data?.pattern_detected && opp.raw_data.analyzer_data.pattern_detected !== 'NORMAL' && (
                  <div className="col-span-2">
                    <div className="text-xs text-gray-500">Pattern v5.0</div>
                    <div className={`text-sm font-bold ${
                      opp.raw_data.analyzer_data.pattern_detected.includes('SPIKE') ? 'text-green-400' :
                      opp.raw_data.analyzer_data.pattern_detected === 'DUMP' ? 'text-red-400' : 'text-yellow-400'
                    }`}>
                      {opp.raw_data.analyzer_data.pattern_detected}
                      {opp.raw_data.analyzer_data.pattern_confidence > 0 && (
                        <span className="text-xs ml-1">({(opp.raw_data.analyzer_data.pattern_confidence * 100).toFixed(0)}%)</span>
                      )}
                    </div>
                  </div>
                )}
                {opp.raw_data?.analyzer_data?.confluence_score && (
                  <div>
                    <div className="text-xs text-gray-500">Confluence</div>
                    <div className={`text-lg font-bold ${
                      opp.raw_data.analyzer_data.confluence_score > 60 ? 'text-green-400' :
                      opp.raw_data.analyzer_data.confluence_score > 40 ? 'text-yellow-400' : 'text-gray-400'
                    }`}>
                      {opp.raw_data.analyzer_data.confluence_score.toFixed(0)}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* === BLOC AVERTISSEMENTS (en dessous des 2 colonnes) === */}
      {opp.validationDetails?.warnings && opp.validationDetails.warnings.length > 0 && (
        <div className="mt-4 bg-yellow-900/10 border border-yellow-500/30 rounded-lg p-4">
          <h4 className="text-xs text-yellow-400 uppercase font-semibold mb-2 tracking-wide">⚠️ Avertissements</h4>
          <div className="text-xs text-yellow-300/90 space-y-1">
            {opp.validationDetails.warnings.map((w: string, i: number) => (
              <div key={i} className="flex items-start gap-1">
                <span className="text-yellow-500">•</span>
                <span>{w}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// Helper functions
const getRankEmoji = (rank: number): string => {
  if (rank === 1) return '🥇';
  if (rank === 2) return '🥈';
  if (rank === 3) return '🥉';
  return `#${rank}`;
};

const getRegimeColor = (regime: string): string => {
  if (!regime || regime === 'UNKNOWN') return 'text-gray-500';
  if (regime.includes('BULL')) return 'text-green-400';
  if (regime.includes('BEAR')) return 'text-red-400';
  if (regime === 'RANGING') return 'text-blue-400';
  if (regime === 'TRANSITION') return 'text-yellow-400';
  return 'text-gray-400';
};

const getActionTextColor = (action: string): string => {
  if (action === 'BUY_NOW') return 'text-white';
  if (action === 'BUY_DCA') return 'text-white';
  if (action === 'EARLY_ENTRY') return 'text-white';
  if (action === 'WAIT') return 'text-black';
  return 'text-white';
};

const getActionBg = (action: string): string => {
  if (action === 'BUY_NOW') return 'bg-green-600';
  if (action === 'BUY_DCA') return 'bg-blue-600';
  if (action === 'EARLY_ENTRY') return 'bg-purple-600';
  if (action === 'WAIT') return 'bg-yellow-600';
  return 'bg-red-600';
};

const getCategoryName = (key: string): string => {
  const names: Record<string, string> = {
    // v5.0 - 9 categories
    'vwap_position': 'VWAP',
    'pattern_detection': 'Pattern',
    'ema_trend': 'Trend EMA',
    'volume_flow': 'Volume Flow',
    'confluence': 'Confluence',
    'momentum': 'Momentum',
    'bollinger': 'Bollinger',
    'volume_profile': 'Vol Profile',
    'macd': 'MACD',
    // v4.1 legacy
    'trend': 'Tendance',
    'volume': 'Volume',
    'volatility': 'Volatilité',
    'support_resistance': 'S/R',
    'rsi_momentum': 'RSI',
    'pattern': 'Pattern'
  };
  return names[key] || key;
};
