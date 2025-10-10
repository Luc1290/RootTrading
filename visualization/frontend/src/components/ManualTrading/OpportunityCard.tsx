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
    'WAIT': { bg: 'bg-yellow-600', text: 'text-black', icon: '⏸️', emoji: '⚪' },
    'AVOID': { bg: 'bg-red-600', text: 'text-white', icon: '🛑', emoji: '🔴' }
  };

  const config = actionConfig[opp.action] || actionConfig.AVOID;
  const grade = opp.score?.grade || 'F';
  const gradeColor = gradeColors[grade] || gradeColors['F'];

  // Prix et targets
  const currentPrice = opp.pricing?.current_price || opp.currentPrice || 0;
  const tp1 = typeof opp.targets?.tp1 === 'object' ? opp.targets.tp1 : { price: opp.targets?.tp1 || 0, percent: 0 };
  const tp2 = typeof opp.targets?.tp2 === 'object' ? opp.targets.tp2 : { price: opp.targets?.tp2 || 0, percent: 0 };
  const sl = typeof opp.stopLoss === 'object' ? opp.stopLoss : { price: opp.stopLoss || 0, percent: 0 };

  // Calculer gains potentiels (sur 5000 USDC)
  const capital = 5000;
  const tp1Gain = tp1.percent ? (capital * tp1.percent / 100) : 0;
  const tp2Gain = tp2.percent ? (capital * tp2.percent / 100) : 0;
  const slLoss = sl.percent ? (capital * sl.percent / 100) : 0;

  return (
    <div className="bg-dark-200 rounded-lg border border-gray-700 hover:border-gray-600 transition-all">
      {/* === HEADER COMPACT === */}
      <div className="p-4">
        <div className="flex items-center justify-between mb-3">
          {/* Symbol + Rank */}
          <div className="flex items-center gap-3">
            <div className="text-2xl">{getRankEmoji(rank)}</div>
            <div>
              <h3 className="text-lg font-bold text-white">{opp.symbol}</h3>
              <div className="text-sm text-gray-400">{formatCurrency(currentPrice)}</div>
            </div>
          </div>

          {/* Score + Grade + Action */}
          <div className="flex items-center gap-3">
            {/* Grade Badge */}
            <div className={`px-3 py-1 rounded-lg bg-gradient-to-r ${gradeColor} text-white font-bold text-sm`}>
              {opp.score?.total?.toFixed(0) || 0}/100
              <div className="text-xs opacity-90">{grade}</div>
            </div>

            {/* Action Badge */}
            <div className={`${config.bg} ${config.text} px-4 py-2 rounded-lg font-bold text-sm flex items-center gap-2`}>
              <span>{config.emoji}</span>
              <span>{opp.action.replace('_', ' ')}</span>
            </div>
          </div>
        </div>

        {/* === VALIDATION STATUS === */}
        {!opp.validation?.all_passed && opp.validationDetails?.blocking_issues && (
          <div className="mb-3 p-3 bg-red-900/20 border border-red-500/30 rounded-lg">
            <div className="flex items-start gap-2">
              <XCircle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
              <div className="flex-1">
                <div className="text-sm font-semibold text-red-400 mb-1">
                  🚫 Validation échouée - {opp.validationDetails.blocking_issues.length} problème(s)
                </div>
                <ul className="text-xs text-red-300 space-y-0.5">
                  {opp.validationDetails.blocking_issues.map((issue: string, i: number) => (
                    <li key={i}>• {issue}</li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}

        {/* === GRID PRINCIPAL === */}
        <div className="grid grid-cols-3 gap-3">
          {/* Colonne 1: Targets */}
          <div className="bg-dark-300 p-3 rounded-lg border border-gray-700">
            <div className="text-xs text-gray-400 mb-2 font-semibold">🎯 Targets</div>
            <div className="space-y-1.5">
              <div className="flex justify-between items-center text-xs">
                <span className="text-green-400">TP1 (+{tp1.percent?.toFixed(1)}%)</span>
                <div className="text-right">
                  <div className="text-white font-semibold">{formatCurrency(tp1.price)}</div>
                  <div className="text-green-400 text-[10px]">+{formatCurrency(tp1Gain)}</div>
                </div>
              </div>
              <div className="flex justify-between items-center text-xs">
                <span className="text-green-400">TP2 (+{tp2.percent?.toFixed(1)}%)</span>
                <div className="text-right">
                  <div className="text-white font-semibold">{formatCurrency(tp2.price)}</div>
                  <div className="text-green-400 text-[10px]">+{formatCurrency(tp2Gain)}</div>
                </div>
              </div>
              <div className="flex justify-between items-center text-xs border-t border-gray-600 pt-1.5">
                <span className="text-red-400">SL (-{sl.percent?.toFixed(1)}%)</span>
                <div className="text-right">
                  <div className="text-white font-semibold">{formatCurrency(sl.price)}</div>
                  <div className="text-red-400 text-[10px]">-{formatCurrency(slLoss)}</div>
                </div>
              </div>
            </div>
          </div>

          {/* Colonne 2: Risk */}
          <div className="bg-dark-300 p-3 rounded-lg border border-gray-700">
            <div className="text-xs text-gray-400 mb-2 font-semibold">⚖️ Risk</div>
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">R/R Ratio:</span>
                <span className={`font-bold ${
                  (opp.risk?.rr_ratio || 0) >= 2 ? 'text-green-400' :
                  (opp.risk?.rr_ratio || 0) >= 1.5 ? 'text-yellow-400' : 'text-red-400'
                }`}>
                  {opp.risk?.rr_ratio?.toFixed(2) || 'N/A'}
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Risk Level:</span>
                <span className={`font-semibold ${
                  opp.risk?.risk_level === 'LOW' ? 'text-green-400' :
                  opp.risk?.risk_level === 'MEDIUM' ? 'text-yellow-400' :
                  opp.risk?.risk_level === 'HIGH' ? 'text-orange-400' : 'text-red-400'
                }`}>
                  {opp.risk?.risk_level || 'N/A'}
                </span>
              </div>
              <div className="flex justify-between text-xs border-t border-gray-600 pt-1.5">
                <span className="text-gray-400">Max Position:</span>
                <span className="text-white font-bold">
                  {opp.risk?.max_position_size_pct?.toFixed(1) || 'N/A'}%
                </span>
              </div>
            </div>
          </div>

          {/* Colonne 3: Régime */}
          <div className="bg-dark-300 p-3 rounded-lg border border-gray-700">
            <div className="text-xs text-gray-400 mb-2 font-semibold">📊 Régime</div>
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">Marché:</span>
                <span className={`font-semibold text-xs ${getRegimeColor(opp.regime)}`}>
                  {opp.regime || 'N/A'}
                </span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">ADX:</span>
                <span className="text-white font-semibold">{opp.adx?.toFixed(1) || 'N/A'}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-gray-400">RSI:</span>
                <span className={`font-semibold ${
                  (opp.rsi || 0) > 70 ? 'text-red-400' :
                  (opp.rsi || 0) < 30 ? 'text-green-400' : 'text-gray-300'
                }`}>
                  {opp.rsi?.toFixed(0) || 'N/A'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* === EXPAND BUTTON === */}
        <button
          onClick={() => setExpanded(!expanded)}
          className="w-full mt-3 py-2 text-xs text-gray-400 hover:text-white flex items-center justify-center gap-2 border border-gray-700 rounded hover:border-gray-600 transition-colors"
        >
          <span>{expanded ? 'Masquer détails' : 'Afficher détails'}</span>
          {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>
      </div>

      {/* === EXPANDED DETAILS === */}
      {expanded && (
        <div className="border-t border-gray-700 p-4 space-y-4 bg-dark-300/50">
          {/* Catégories Score */}
          {opp.categoryScores && (
            <div>
              <div className="text-sm font-semibold text-gray-300 mb-3">📈 Catégories PRO</div>
              <div className="grid grid-cols-4 gap-2">
                {Object.entries(opp.categoryScores).map(([key, value]: [string, any]) => (
                  <div key={key} className="bg-dark-200 p-2 rounded border border-gray-700">
                    <div className="text-xs text-gray-400 capitalize">{key.replace('_', ' ')}</div>
                    <div className="text-lg font-bold text-white">{value.score?.toFixed(0) || 0}</div>
                    <div className="text-[10px] text-gray-500">Conf: {value.confidence?.toFixed(0) || 0}%</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Divergence Auto/Manuel */}
          {autoSignal && (
            <div>
              <div className="text-sm font-semibold text-gray-300 mb-3">🔀 Comparaison Sources</div>
              <div className="grid grid-cols-2 gap-3">
                {/* Manuel */}
                <div className="bg-dark-200 p-3 rounded border border-purple-500/30">
                  <div className="text-xs text-purple-400 font-semibold mb-2">📊 MANUEL</div>
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Score:</span>
                      <span className="font-bold text-white">{opp.score?.total?.toFixed(0) || 0}/100</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Grade:</span>
                      <span className="font-bold text-white">{grade}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Action:</span>
                      <span className={`font-semibold ${getActionTextColor(opp.action)}`}>
                        {opp.action}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Auto */}
                <div className={`bg-dark-200 p-3 rounded border ${
                  autoSignal.validated ? 'border-green-500/30' : 'border-red-500/30'
                }`}>
                  <div className="text-xs text-blue-400 font-semibold mb-2">🤖 AUTO</div>
                  {autoSignal.has_signal ? (
                    <div className="space-y-1 text-xs">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Consensus:</span>
                        <span className="font-bold text-blue-400">
                          {autoSignal.consensus_strength?.toFixed(2) || 'N/A'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Stratégies:</span>
                        <span className="text-white font-semibold">{autoSignal.strategies_count || 0}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Side:</span>
                        <span className={`font-semibold ${
                          autoSignal.side === 'BUY' ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {autoSignal.side}
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div className="text-xs text-gray-400">Aucun signal récent</div>
                  )}
                </div>
              </div>

              {/* Divergence warning */}
              {autoSignal.has_signal && autoSignal.side !== opp.action.split('_')[0] && (
                <div className="mt-2 p-2 bg-yellow-900/20 border border-yellow-500/30 rounded text-xs text-yellow-400">
                  ⚠️ DIVERGENCE: Auto dit {autoSignal.side}, Manuel dit {opp.action}
                </div>
              )}
            </div>
          )}

          {/* Warnings */}
          {opp.validationDetails?.warnings && opp.validationDetails.warnings.length > 0 && (
            <div>
              <div className="text-sm font-semibold text-gray-300 mb-2">⚠️ Avertissements</div>
              <ul className="space-y-1">
                {opp.validationDetails.warnings.map((warning: string, i: number) => (
                  <li key={i} className="text-xs text-yellow-400 flex items-start gap-2">
                    <AlertCircle className="w-3 h-3 mt-0.5 flex-shrink-0" />
                    <span>{warning}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
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
  if (action === 'BUY_NOW') return 'text-green-400';
  if (action === 'BUY_DCA') return 'text-blue-400';
  if (action === 'WAIT') return 'text-yellow-400';
  return 'text-red-400';
};
