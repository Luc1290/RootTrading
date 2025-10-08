import React, { useState, useEffect } from 'react';
import toast from 'react-hot-toast';
import { apiService } from '@/services/api';
import { formatCurrency, formatPercent } from '@/utils';

interface ManualPosition {
  id: string;
  symbol: string;
  entryPrice: number;
  quantity: number;
  entryDate: string;
  notes?: string;
  maxPrice?: number;  // Prix max atteint pendant la position
  maxPriceDate?: string;  // Date du prix max
}

interface PositionWithPrice extends ManualPosition {
  currentPrice: number;
  pnlPercent: number;
  pnlUsdc: number;
  maxPnlPercent?: number;  // P&L max atteint (%)
  maxPnlUsdc?: number;  // P&L max atteint (USDC)
  signal?: {
    action: string;
    reason: string;
    estimatedHoldTime?: string;
    stopLoss?: number;
  };
  timeElapsed?: string;
  recommendedStopLoss?: number;
  stopLossPercent?: number;
}

function PositionTrackerPage() {
  const [positions, setPositions] = useState<ManualPosition[]>([]);
  const [positionsWithPrices, setPositionsWithPrices] = useState<PositionWithPrice[]>([]);
  const [loading, setLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [editingPrice, setEditingPrice] = useState<{id: string, value: string} | null>(null);

  // Form pour ajouter position
  const [showAddForm, setShowAddForm] = useState(false);
  const [formData, setFormData] = useState({
    symbol: '',
    entryPrice: '',
    amountUsdc: '',
    notes: ''
  });

  // Charger positions depuis localStorage
  useEffect(() => {
    const savedPositions = localStorage.getItem('manual_positions');
    if (savedPositions) {
      setPositions(JSON.parse(savedPositions));
    }
  }, []);

  // Sauvegarder positions dans localStorage
  const savePositions = (newPositions: ManualPosition[]) => {
    localStorage.setItem('manual_positions', JSON.stringify(newPositions));
    setPositions(newPositions);
  };

  // Charger prix actuels, signaux et calculer PnL
  const updatePrices = async () => {
    if (positions.length === 0) return;

    setLoading(true);
    try {
      const positionsData = await Promise.all(
        positions.map(async (pos) => {
          try {
            // Récupérer prix actuel + signal depuis API
            const [marketData, signalData] = await Promise.all([
              apiService.getMarketData(pos.symbol as any, '1m', 1),
              fetch(`/api/trading-opportunities/${pos.symbol}`).then(r => r.json()).catch(() => null)
            ]);

            const currentPrice = marketData.data.close[marketData.data.close.length - 1] || 0;
            const pnlPercent = ((currentPrice - pos.entryPrice) / pos.entryPrice) * 100;
            const pnlUsdc = (currentPrice - pos.entryPrice) * pos.quantity;

            // Mettre à jour le prix max si dépassé
            let updatedMaxPrice = pos.maxPrice || pos.entryPrice;
            let updatedMaxPriceDate = pos.maxPriceDate;

            if (currentPrice > updatedMaxPrice) {
              updatedMaxPrice = currentPrice;
              updatedMaxPriceDate = new Date().toISOString();

              // Sauvegarder le nouveau max dans localStorage
              const updatedPositions = positions.map(p =>
                p.id === pos.id
                  ? { ...p, maxPrice: updatedMaxPrice, maxPriceDate: updatedMaxPriceDate }
                  : p
              );
              savePositions(updatedPositions);
            }

            // Calculer P&L max
            const maxPnlPercent = ((updatedMaxPrice - pos.entryPrice) / pos.entryPrice) * 100;
            const maxPnlUsdc = (updatedMaxPrice - pos.entryPrice) * pos.quantity;

            // Calculer temps écoulé depuis ouverture
            const entryTime = new Date(pos.entryDate).getTime();
            const now = Date.now();
            const elapsedMinutes = Math.floor((now - entryTime) / 60000);
            const timeElapsed = elapsedMinutes < 60
              ? `${elapsedMinutes} min`
              : `${Math.floor(elapsedMinutes / 60)}h ${elapsedMinutes % 60}min`;

            // Calculer stop-loss recommandé depuis le prix d'entrée
            const recommendedStopLoss = signalData?.stop_loss || (pos.entryPrice * 0.992); // -0.8% par défaut
            const stopLossPercent = ((recommendedStopLoss - pos.entryPrice) / pos.entryPrice) * 100;

            return {
              ...pos,
              currentPrice,
              pnlPercent,
              pnlUsdc,
              maxPrice: updatedMaxPrice,
              maxPriceDate: updatedMaxPriceDate,
              maxPnlPercent,
              maxPnlUsdc,
              signal: signalData ? {
                action: signalData.action,
                reason: signalData.reason,
                estimatedHoldTime: signalData.estimated_hold_time,
                stopLoss: signalData.stop_loss
              } : undefined,
              timeElapsed,
              recommendedStopLoss,
              stopLossPercent
            };
          } catch (err) {
            console.error(`Error loading price for ${pos.symbol}:`, err);
            return {
              ...pos,
              currentPrice: pos.entryPrice,
              pnlPercent: 0,
              pnlUsdc: 0
            };
          }
        })
      );

      setPositionsWithPrices(positionsData);
    } catch (err) {
      console.error('Error updating prices:', err);
      toast.error('Erreur lors de la mise à jour des prix');
    } finally {
      setLoading(false);
    }
  };

  // Auto-refresh toutes les 60 secondes
  useEffect(() => {
    if (!autoRefresh) return;

    updatePrices();
    const interval = setInterval(updatePrices, 60000);

    return () => clearInterval(interval);
  }, [positions, autoRefresh]);

  // Ajouter position
  const handleAddPosition = () => {
    if (!formData.symbol || !formData.entryPrice || !formData.amountUsdc) {
      toast.error('Remplis tous les champs obligatoires');
      return;
    }

    const entryPrice = parseFloat(formData.entryPrice);
    const amountUsdc = parseFloat(formData.amountUsdc);
    const quantity = amountUsdc / entryPrice;  // Calcul automatique de la quantité

    const newPosition: ManualPosition = {
      id: Date.now().toString(),
      symbol: formData.symbol.toUpperCase().replace('USDC', '') + 'USDC',
      entryPrice,
      quantity,
      entryDate: new Date().toISOString(),
      notes: formData.notes
    };

    savePositions([...positions, newPosition]);
    setFormData({ symbol: '', entryPrice: '', amountUsdc: '', notes: '' });
    setShowAddForm(false);
    toast.success(`Position ${newPosition.symbol} ajoutée: ${quantity.toFixed(8)} tokens pour ${amountUsdc} USDC`);
  };

  // Supprimer position
  const handleDeletePosition = (id: string) => {
    if (confirm('Supprimer cette position ?')) {
      savePositions(positions.filter(p => p.id !== id));
      toast.success('Position supprimée');
    }
  };

  // Rafraîchir une seule position
  const handleRefreshSingle = async (symbol: string) => {
    try {
      const marketData = await apiService.getMarketData(symbol as any, '1m', 1);
      const currentPrice = marketData.data.close[marketData.data.close.length - 1] || 0;

      // Mettre à jour uniquement cette position
      setPositionsWithPrices(prev => prev.map(pos => {
        if (pos.symbol !== symbol) return pos;

        const pnlPercent = ((currentPrice - pos.entryPrice) / pos.entryPrice) * 100;
        const pnlUsdc = (currentPrice - pos.entryPrice) * pos.quantity;

        return {
          ...pos,
          currentPrice,
          pnlPercent,
          pnlUsdc
        };
      }));

      toast.success(`${symbol.replace('USDC', '')} mis à jour: ${currentPrice.toFixed(8)} USDC`);
    } catch (err) {
      console.error(`Error refreshing ${symbol}:`, err);
      toast.error(`Erreur rafraîchissement ${symbol}`);
    }
  };

  // Mettre à jour manuellement le prix d'une position
  const handleManualPriceUpdate = (posId: string, newPrice: number) => {
    setPositionsWithPrices(prev => prev.map(pos => {
      if (pos.id !== posId) return pos;

      const pnlPercent = ((newPrice - pos.entryPrice) / pos.entryPrice) * 100;
      const pnlUsdc = (newPrice - pos.entryPrice) * pos.quantity;

      return {
        ...pos,
        currentPrice: newPrice,
        pnlPercent,
        pnlUsdc
      };
    }));
    setEditingPrice(null);
    toast.success(`Prix mis à jour: ${newPrice.toFixed(8)} USDC`);
  };

  // Calculer total PnL
  const totalPnlUsdc = positionsWithPrices.reduce((sum, pos) => sum + pos.pnlUsdc, 0);
  const totalInvestedUsdc = positionsWithPrices.reduce((sum, pos) => sum + (pos.entryPrice * pos.quantity), 0);
  const totalPnlPercent = totalInvestedUsdc > 0 ? (totalPnlUsdc / totalInvestedUsdc) * 100 : 0;

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white">📊 Suivi Positions Manuelles</h1>
          <p className="text-gray-400 mt-1">Suivi en temps réel de tes positions ouvertes</p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              autoRefresh
                ? 'bg-green-600 hover:bg-green-700 text-white'
                : 'bg-gray-600 hover:bg-gray-700 text-white'
            }`}
          >
            {autoRefresh ? '🔄 Auto (10s)' : '⏸️ Pause'}
          </button>
          <button
            onClick={updatePrices}
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white px-6 py-2 rounded-lg font-medium transition-colors"
          >
            {loading ? '⏳ Chargement...' : '🔄 Tout Rafraîchir'}
          </button>
          <button
            onClick={() => setShowAddForm(!showAddForm)}
            className="bg-primary-500 hover:bg-primary-600 text-white px-6 py-2 rounded-lg font-medium transition-colors"
          >
            {showAddForm ? '❌ Annuler' : '➕ Ajouter Position'}
          </button>
        </div>
      </div>

      {/* Formulaire ajout position */}
      {showAddForm && (
        <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 className="text-xl font-bold text-white mb-4">➕ Nouvelle Position</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">
                Symbole (ex: BTC, ETH, SOL)
              </label>
              <input
                type="text"
                value={formData.symbol}
                onChange={(e) => setFormData({ ...formData, symbol: e.target.value })}
                placeholder="BTC"
                className="w-full bg-gray-700 text-white px-4 py-2 rounded-lg border border-gray-600 focus:border-primary-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">
                Prix d'entrée (USDC)
              </label>
              <input
                type="number"
                step="0.00000001"
                value={formData.entryPrice}
                onChange={(e) => setFormData({ ...formData, entryPrice: e.target.value })}
                placeholder="58423.50"
                className="w-full bg-gray-700 text-white px-4 py-2 rounded-lg border border-gray-600 focus:border-primary-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">
                Montant investi (USDC)
              </label>
              <input
                type="number"
                step="0.01"
                value={formData.amountUsdc}
                onChange={(e) => setFormData({ ...formData, amountUsdc: e.target.value })}
                placeholder="5000"
                className="w-full bg-gray-700 text-white px-4 py-2 rounded-lg border border-gray-600 focus:border-primary-500 focus:outline-none"
              />
              {formData.entryPrice && formData.amountUsdc && (
                <p className="text-xs text-gray-400 mt-1">
                  → {(parseFloat(formData.amountUsdc) / parseFloat(formData.entryPrice)).toFixed(8)} tokens
                </p>
              )}
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">
                Notes (optionnel)
              </label>
              <input
                type="text"
                value={formData.notes}
                onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
                placeholder="Breakout BB squeeze"
                className="w-full bg-gray-700 text-white px-4 py-2 rounded-lg border border-gray-600 focus:border-primary-500 focus:outline-none"
              />
            </div>
          </div>
          <button
            onClick={handleAddPosition}
            className="mt-4 bg-green-600 hover:bg-green-700 text-white px-8 py-3 rounded-lg font-bold transition-colors"
          >
            ✅ Ajouter
          </button>
        </div>
      )}

      {/* Total PnL */}
      {positionsWithPrices.length > 0 && (
        <div className="bg-gradient-to-r from-gray-800 to-gray-900 rounded-lg p-6 border border-gray-700">
          <h3 className="text-lg font-bold text-gray-300 mb-4">📈 Total Portfolio</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <div className="text-sm text-gray-400">Capital Investi</div>
              <div className="text-2xl font-bold text-white mt-1">
                {formatCurrency(totalInvestedUsdc)}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-400">PnL Total</div>
              <div className={`text-2xl font-bold mt-1 ${
                totalPnlUsdc >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {totalPnlUsdc >= 0 ? '+' : ''}{formatCurrency(totalPnlUsdc)}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-400">PnL %</div>
              <div className={`text-2xl font-bold mt-1 ${
                totalPnlPercent >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {totalPnlPercent >= 0 ? '+' : ''}{totalPnlPercent.toFixed(2)}%
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Liste positions */}
      <div className="space-y-4">
        {positionsWithPrices.map((pos) => {
          const isProfit = pos.pnlUsdc >= 0;
          const positionValue = pos.currentPrice * pos.quantity;

          return (
            <div
              key={pos.id}
              className={`bg-gray-800 rounded-lg p-6 border-2 transition-all ${
                isProfit
                  ? 'border-green-500/30 hover:border-green-500/50'
                  : 'border-red-500/30 hover:border-red-500/50'
              }`}
            >
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="text-2xl font-bold text-white">{pos.symbol.replace('USDC', '')}</h3>
                  <p className="text-xs text-gray-400 mt-1">
                    Ouvert le {new Date(pos.entryDate).toLocaleString('fr-FR')}
                  </p>
                  {pos.notes && (
                    <p className="text-sm text-gray-400 mt-1">💬 {pos.notes}</p>
                  )}
                </div>
                <div className="flex gap-3">
                  <button
                    onClick={() => handleRefreshSingle(pos.symbol)}
                    className="bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium px-3 py-1 rounded transition-colors"
                  >
                    🔄 Refresh
                  </button>
                  <button
                    onClick={() => handleDeletePosition(pos.id)}
                    className="text-red-400 hover:text-red-300 text-sm font-medium"
                  >
                    🗑️ Supprimer
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
                {/* Prix entrée */}
                <div>
                  <div className="text-xs text-gray-400">Prix Entrée</div>
                  <div className="text-lg font-bold text-gray-300 mt-1">
                    {pos.entryPrice.toFixed(8)} USDC
                  </div>
                  {pos.recommendedStopLoss && (
                    <div className="mt-2 pt-2 border-t border-gray-700">
                      <div className="text-xs text-red-400">🛑 Stop-Loss</div>
                      <div className="text-sm font-bold text-red-400">
                        {pos.recommendedStopLoss.toFixed(8)}
                      </div>
                      <div className="text-xs text-gray-500">
                        ({pos.stopLossPercent?.toFixed(2)}%)
                      </div>
                    </div>
                  )}
                </div>

                {/* Prix actuel - Éditable */}
                <div>
                  <div className="text-xs text-gray-400">Prix Actuel</div>
                  {editingPrice?.id === pos.id ? (
                    <div className="flex gap-2 mt-1">
                      <input
                        type="number"
                        step="0.00000001"
                        value={editingPrice.value}
                        onChange={(e) => setEditingPrice({ id: pos.id, value: e.target.value })}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            handleManualPriceUpdate(pos.id, parseFloat(editingPrice.value));
                          } else if (e.key === 'Escape') {
                            setEditingPrice(null);
                          }
                        }}
                        placeholder="Prix actuel"
                        aria-label="Prix actuel"
                        className="w-32 bg-gray-700 text-white px-2 py-1 rounded text-sm border border-primary-500 focus:outline-none"
                        autoFocus
                      />
                      <button
                        type="button"
                        onClick={() => handleManualPriceUpdate(pos.id, parseFloat(editingPrice.value))}
                        className="bg-green-600 hover:bg-green-700 text-white px-2 py-1 rounded text-xs"
                      >
                        ✓
                      </button>
                      <button
                        type="button"
                        onClick={() => setEditingPrice(null)}
                        className="bg-gray-600 hover:bg-gray-700 text-white px-2 py-1 rounded text-xs"
                      >
                        ✗
                      </button>
                    </div>
                  ) : (
                    <div
                      onClick={() => setEditingPrice({ id: pos.id, value: pos.currentPrice.toFixed(8) })}
                      className="text-lg font-bold text-white mt-1 cursor-pointer hover:bg-gray-700 px-2 py-1 rounded transition-colors"
                      title="Clique pour modifier manuellement"
                    >
                      {pos.currentPrice.toFixed(8)} USDC ✏️
                    </div>
                  )}
                </div>

                {/* PnL Actuel */}
                <div>
                  <div className="text-xs text-gray-400">PnL Actuel</div>
                  <div className={`text-xl font-bold mt-1 ${
                    isProfit ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {isProfit ? '+' : ''}{pos.pnlPercent.toFixed(2)}%
                  </div>
                  <div className={`text-sm mt-1 ${
                    isProfit ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {isProfit ? '+' : ''}{pos.pnlUsdc.toFixed(2)} USDC
                  </div>
                </div>

                {/* PnL Max (Prix Peak) */}
                {pos.maxPnlPercent !== undefined && pos.maxPnlPercent > pos.pnlPercent && (
                  <div className="border-l-2 border-yellow-500 pl-3">
                    <div className="text-xs text-yellow-400 flex items-center gap-1">
                      🏆 Max Atteint
                    </div>
                    <div className="text-lg font-bold text-yellow-400 mt-1">
                      +{pos.maxPnlPercent.toFixed(2)}%
                    </div>
                    <div className="text-sm text-yellow-300 mt-1">
                      +{pos.maxPnlUsdc?.toFixed(2)} USDC
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      @ {pos.maxPrice?.toFixed(8)}
                    </div>
                  </div>
                )}

                {/* Valeur position */}
                <div>
                  <div className="text-xs text-gray-400">Valeur Position</div>
                  <div className="text-lg font-bold text-white mt-1">
                    {positionValue.toFixed(2)} USDC
                  </div>
                  <div className="text-xs text-gray-400 mt-1">
                    {pos.quantity.toFixed(8)} tokens
                  </div>
                </div>
              </div>

              {/* Suggestions TP (Take Profit) */}
              <div className="mt-4 pt-4 border-t border-gray-700">
                <div className="text-xs text-gray-400 mb-3">🎯 Suggestions Take Profit</div>
                <div className="grid grid-cols-3 gap-3">
                  {[0.5, 1.0, 2.0].map(tpPercent => {
                    const tpPrice = pos.entryPrice * (1 + tpPercent / 100);
                    const tpUsdc = (tpPrice - pos.entryPrice) * pos.quantity;
                    const reached = pos.currentPrice >= tpPrice;
                    const maxReached = (pos.maxPrice || 0) >= tpPrice;

                    return (
                      <div
                        key={tpPercent}
                        className={`rounded-lg p-3 border-2 ${
                          reached
                            ? 'bg-green-900/20 border-green-500'
                            : maxReached
                            ? 'bg-yellow-900/20 border-yellow-500'
                            : 'bg-dark-300 border-gray-600'
                        }`}
                      >
                        <div className="text-xs font-semibold text-gray-300">
                          TP +{tpPercent}%
                        </div>
                        <div className="text-sm font-bold text-white mt-1">
                          {tpPrice.toFixed(8)}
                        </div>
                        <div className="text-sm text-green-400 font-semibold mt-1">
                          +{tpUsdc.toFixed(2)} USDC
                        </div>
                        {reached && (
                          <div className="text-xs text-green-400 mt-1">✅ Atteint</div>
                        )}
                        {!reached && maxReached && (
                          <div className="text-xs text-yellow-400 mt-1">⚠️ Était atteint</div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Recommandation & Signal en temps réel */}
              {pos.signal && (
                <div className="mt-4 pt-4 border-t border-gray-700">
                  <div className="flex items-start gap-4">
                    {/* Temps écoulé */}
                    <div className="flex-shrink-0">
                      <div className="text-xs text-gray-400">⏱️ Position ouverte depuis</div>
                      <div className="text-lg font-bold text-yellow-400">{pos.timeElapsed}</div>
                      {pos.signal.estimatedHoldTime && (
                        <div className="text-xs text-gray-500">Recommandé: {pos.signal.estimatedHoldTime}</div>
                      )}
                    </div>

                    {/* Signal actuel */}
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <div className="text-xs text-gray-400">📡 Signal Actuel</div>
                        <div className={`px-3 py-1 rounded font-bold text-sm ${
                          pos.signal.action === 'SELL_OVERBOUGHT' ? 'bg-red-600 text-white' :
                          pos.signal.action === 'BUY_NOW' ? 'bg-green-600 text-white' :
                          pos.signal.action === 'WAIT' ? 'bg-yellow-600 text-white' :
                          'bg-gray-600 text-white'
                        }`}>
                          {pos.signal.action === 'SELL_OVERBOUGHT' ? '🔴 VENDRE' :
                           pos.signal.action === 'BUY_NOW' ? '🟢 TENIR (momentum fort)' :
                           pos.signal.action === 'WAIT' ? '🟡 SURVEILLER' :
                           pos.signal.action}
                        </div>
                      </div>
                      <div className="text-xs text-gray-300">
                        {pos.signal.reason}
                      </div>
                    </div>

                    {/* Recommandation finale */}
                    <div className="flex-shrink-0 bg-gray-900 rounded-lg p-3 border-2 border-dashed min-w-[200px]">
                      <div className="text-xs text-gray-400 mb-1">💡 Recommandation</div>
                      {(() => {
                        const elapsedMin = parseInt(pos.timeElapsed || '0');
                        const holdRange = pos.signal.estimatedHoldTime?.split('-') || ['0', '999'];
                        const holdMin = parseInt(holdRange[0] || '0');
                        const holdMax = parseInt(holdRange[1]?.replace(' min', '') || '999');

                        // Décision basée sur signal + temps + PnL
                        if (pos.signal.action === 'SELL_OVERBOUGHT') {
                          return (
                            <div className="text-red-400 font-bold text-sm">
                              🔴 SORTIR MAINTENANT<br/>
                              <span className="text-xs font-normal">Marché suracheté, correction probable</span>
                            </div>
                          );
                        } else if (pos.pnlPercent >= 1.5) {
                          return (
                            <div className="text-green-400 font-bold text-sm">
                              ✅ PRENDRE PROFIT<br/>
                              <span className="text-xs font-normal">Gain {pos.pnlPercent.toFixed(2)}% atteint !</span>
                            </div>
                          );
                        } else if (pos.recommendedStopLoss && pos.currentPrice <= pos.recommendedStopLoss) {
                          return (
                            <div className="text-red-400 font-bold text-sm">
                              ❌ STOP-LOSS ATTEINT<br/>
                              <span className="text-xs font-normal">Prix {pos.currentPrice.toFixed(8)} ≤ SL {pos.recommendedStopLoss.toFixed(8)}</span>
                            </div>
                          );
                        } else if (pos.pnlPercent <= -0.8) {
                          return (
                            <div className="text-red-400 font-bold text-sm">
                              ❌ STOP-LOSS<br/>
                              <span className="text-xs font-normal">Limite perte atteinte (-{Math.abs(pos.pnlPercent).toFixed(2)}%)</span>
                            </div>
                          );
                        } else if (elapsedMin >= holdMax) {
                          return (
                            <div className="text-orange-400 font-bold text-sm">
                              ⏰ SORTIR (temps max)<br/>
                              <span className="text-xs font-normal">Déjà {elapsedMin}min, risque retournement</span>
                            </div>
                          );
                        } else if (elapsedMin >= holdMin && pos.pnlPercent >= 0.5) {
                          return (
                            <div className="text-yellow-400 font-bold text-sm">
                              💰 SORTIR SI +0.8%<br/>
                              <span className="text-xs font-normal">Temps min atteint, sécuriser profit</span>
                            </div>
                          );
                        } else if (pos.signal.action === 'BUY_NOW') {
                          return (
                            <div className="text-green-400 font-bold text-sm">
                              🟢 TENIR<br/>
                              <span className="text-xs font-normal">Momentum encore fort, viser +1%</span>
                            </div>
                          );
                        } else {
                          return (
                            <div className="text-gray-400 font-bold text-sm">
                              👀 SURVEILLER<br/>
                              <span className="text-xs font-normal">Sortir si signal SELL ou {holdMax}min</span>
                            </div>
                          );
                        }
                      })()}
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Empty state */}
      {positions.length === 0 && (
        <div className="text-center py-12">
          <div className="text-6xl mb-4">📊</div>
          <div className="text-gray-400 text-lg mb-2">
            Aucune position enregistrée
          </div>
          <div className="text-gray-500 text-sm mb-6">
            Clique sur "➕ Ajouter Position" pour commencer le suivi
          </div>
        </div>
      )}

      {/* Loading state */}
      {loading && positions.length > 0 && (
        <div className="text-center py-4">
          <div className="text-gray-400">🔄 Mise à jour des prix...</div>
        </div>
      )}
    </div>
  );
}

export default PositionTrackerPage;
