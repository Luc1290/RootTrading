import React, { useState, useEffect, useRef } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi, LineData, Time } from 'lightweight-charts';
import toast from 'react-hot-toast';
import { apiService } from '@/services/api';
import { formatCurrency, formatPercent } from '@/utils';
import type { TradingSymbol } from '@/types';

interface GlobalStatistics {
  totalTrades: number;
  totalVolume: number;
  totalPnl: number;
  winRate: number;
  profitFactor: number;
  avgTradeSize: number;
  totalFees: number;
  activePositions: number;
  availableBalance: number;
  totalBalance: number;
  unrealizedPnl: number;
  realizedPnl: number;
}

interface SymbolStatistics {
  symbol: string;
  trades: number;
  volume: number;
  pnl: number;
  winRate: number;
  avgTradeSize: number;
  fees: number;
  lastPrice: number;
  priceChange24h: number;
}

interface PerformanceHistory {
  timestamps: string[];
  pnl: number[];
  balance: number[];
  winRate: number[];
  volume: number[];
}

interface StrategyStatistics {
  strategy: string;
  type?: 'CONSENSUS' | 'INDIVIDUAL';
  trades: number;
  winRate: number;
  avgPnl: number;
  totalPnl: number;
  avgDuration: number;
  maxDrawdown: number;
  sharpeRatio: number;
  // Nouveaux champs optimisés
  total_signals_emitted?: number;
  buy_signals_emitted?: number;
  sell_signals_emitted?: number;
  trades_executed?: number;
  buy_trades_executed?: number;
  sell_trades_executed?: number;
  signal_to_trade_rate?: number;
  buy_conversion_rate?: number;
  sell_conversion_rate?: number;
  max_gain?: number;
  max_loss?: number;
  max_gain_percent?: number;
  max_loss_percent?: number;
}

function StatisticsPage() {
  // État pour les données statistiques
  const [globalStats, setGlobalStats] = useState<GlobalStatistics | null>(null);
  const [symbolStats, setSymbolStats] = useState<SymbolStatistics[]>([]);
  const [performanceHistory, setPerformanceHistory] = useState<PerformanceHistory | null>(null);
  const [strategyStats, setStrategyStats] = useState<StrategyStatistics[]>([]);
  const [activeStats, setActiveStats] = useState<StrategyStatistics[]>([]);
  const [profitableStats, setProfitableStats] = useState<StrategyStatistics[]>([]);
  
  // État UI
  const [selectedSymbol, setSelectedSymbol] = useState<TradingSymbol>('BTCUSDC');
  const [selectedTimeframe, setSelectedTimeframe] = useState<'24h' | '7d' | '30d' | '90d'>('7d');
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  
  // Symboles disponibles
  const [availableSymbols, setAvailableSymbols] = useState<TradingSymbol[]>([]);
  
  // Refs pour les graphiques
  const pnlChartRef = useRef<HTMLDivElement>(null);
  const balanceChartRef = useRef<HTMLDivElement>(null);
  const volumeChartRef = useRef<HTMLDivElement>(null);
  const winRateChartRef = useRef<HTMLDivElement>(null);
  
  const pnlChartInstance = useRef<IChartApi | null>(null);
  const balanceChartInstance = useRef<IChartApi | null>(null);
  const volumeChartInstance = useRef<IChartApi | null>(null);
  const winRateChartInstance = useRef<IChartApi | null>(null);

  // Fonction pour charger toutes les données
  const loadAllData = async () => {
    try {
      setLoading(true);
      
      // Charger les données avec gestion d'erreur individuelle
      const results = await Promise.allSettled([
        apiService.getGlobalStatistics(),
        apiService.getSymbolStatistics(selectedSymbol),
        apiService.getPerformanceHistory(selectedTimeframe),
        apiService.getStrategiesStatistics(),
        apiService.getConfiguredSymbols()
      ]);
      
      const globalResponse = results[0].status === 'fulfilled' ? results[0].value as GlobalStatistics : null;
      const symbolResponse = results[1].status === 'fulfilled' ? results[1].value as { symbols: SymbolStatistics[] } : null;
      const performanceResponse = results[2].status === 'fulfilled' ? results[2].value as PerformanceHistory : null;
      const strategiesResponse = results[3].status === 'fulfilled' ? results[3].value as {
        strategies?: StrategyStatistics[];
        individual_strategies?: StrategyStatistics[];
        active_strategies?: StrategyStatistics[];
        profitable_strategies?: StrategyStatistics[];
      } : null;
      const symbolsResponse = results[4].status === 'fulfilled' ? results[4].value as { symbols: string[] } : null;
      
      // Définir des valeurs par défaut pour éviter les erreurs undefined
      const defaultGlobalStats: GlobalStatistics = {
        totalTrades: 0,
        totalVolume: 0,
        totalPnl: 0,
        winRate: 0,
        profitFactor: 0,
        avgTradeSize: 0,
        totalFees: 0,
        activePositions: 0,
        availableBalance: 0,
        totalBalance: 0,
        unrealizedPnl: 0,
        realizedPnl: 0
      };
      
      // Appliquer les résultats avec fallbacks
      if (globalResponse) {
        setGlobalStats({
          ...defaultGlobalStats,
          ...globalResponse
        });
      }
      
      if (symbolResponse && symbolResponse.symbols) {
        setSymbolStats(symbolResponse.symbols);
      }
      
      if (performanceResponse) {
        setPerformanceHistory(performanceResponse);
      }
      
      if (strategiesResponse) {
        setStrategyStats(strategiesResponse.individual_strategies || strategiesResponse.strategies || []);
        setActiveStats(strategiesResponse.active_strategies || []);
        setProfitableStats(strategiesResponse.profitable_strategies || []);
      }
      
      if (symbolsResponse && symbolsResponse.symbols) {
        setAvailableSymbols(symbolsResponse.symbols);
      }
      
      setLastUpdate(new Date());
      
      // Log des erreurs individuelles
      results.forEach((result, index) => {
        if (result.status === 'rejected') {
          console.error(`API Error ${index}:`, result.reason);
        }
      });
      
    } catch (error) {
      console.error('Erreur lors du chargement des statistiques:', error);
      toast.error('Erreur lors du chargement des statistiques');
    } finally {
      setLoading(false);
    }
  };

  // Initialisation des graphiques
  useEffect(() => {
    const initChart = (container: HTMLDivElement, title: string, color: string) => {
      return createChart(container, {
        width: container.clientWidth,
        height: 200,
        layout: {
          background: { type: ColorType.Solid, color: '#1a1a1a' },
          textColor: '#ffffff',
        },
        grid: {
          vertLines: { color: '#333333' },
          horzLines: { color: '#333333' },
        },
        rightPriceScale: {
          borderColor: '#444444',
          textColor: '#ffffff',
        },
        timeScale: {
          borderColor: '#444444',
          timeVisible: true,
          secondsVisible: false,
        },
        crosshair: {
          horzLine: { color: '#888888', width: 1, style: 2 },
          vertLine: { color: '#888888', width: 1, style: 2 },
        },
      });
    };

    if (pnlChartRef.current && !pnlChartInstance.current) {
      pnlChartInstance.current = initChart(pnlChartRef.current, 'P&L', '#ffd700');
    }
    
    if (balanceChartRef.current && !balanceChartInstance.current) {
      balanceChartInstance.current = initChart(balanceChartRef.current, 'Balance', '#26a69a');
    }
    
    if (volumeChartRef.current && !volumeChartInstance.current) {
      volumeChartInstance.current = initChart(volumeChartRef.current, 'Volume', '#42a5f5');
    }
    
    if (winRateChartRef.current && !winRateChartInstance.current) {
      winRateChartInstance.current = initChart(winRateChartRef.current, 'Win Rate', '#ab47bc');
    }

    return () => {
      pnlChartInstance.current?.remove();
      balanceChartInstance.current?.remove();
      volumeChartInstance.current?.remove();
      winRateChartInstance.current?.remove();
      pnlChartInstance.current = null;
      balanceChartInstance.current = null;
      volumeChartInstance.current = null;
      winRateChartInstance.current = null;
    };
  }, []);

  // Mise à jour des données des graphiques
  useEffect(() => {
    if (!performanceHistory) return;

    const updateChart = (
      chart: IChartApi | null,
      data: number[],
      color: string
    ) => {
      if (!chart) return;
      
      const series = chart.addLineSeries({
        color,
        lineWidth: 2,
        priceFormat: { type: 'price', precision: 2 },
      });
      
      const chartData = (performanceHistory?.timestamps || []).map((timestamp, index) => ({
        time: Math.floor(new Date(timestamp).getTime() / 1000) as Time,
        value: data[index] || 0,
      })) as LineData[];
      
      series.setData(chartData);
      chart.timeScale().fitContent();
    };

    // Vider les graphiques existants
    [pnlChartInstance, balanceChartInstance, volumeChartInstance, winRateChartInstance]
      .forEach(chart => {
        if (chart.current) {
          chart.current.remove();
          chart.current = null;
        }
      });

    // Recréer et mettre à jour les graphiques
    setTimeout(() => {
      if (pnlChartRef.current) {
        pnlChartInstance.current = createChart(pnlChartRef.current, {
          width: pnlChartRef.current.clientWidth,
          height: 200,
          layout: { background: { type: ColorType.Solid, color: '#1a1a1a' }, textColor: '#ffffff' },
          grid: { vertLines: { color: '#333333' }, horzLines: { color: '#333333' } },
          rightPriceScale: { borderColor: '#444444', textColor: '#ffffff' },
          timeScale: { borderColor: '#444444', timeVisible: true, secondsVisible: false },
        });
        updateChart(pnlChartInstance.current, performanceHistory?.pnl || [], '#ffd700');
      }
      
      if (balanceChartRef.current) {
        balanceChartInstance.current = createChart(balanceChartRef.current, {
          width: balanceChartRef.current.clientWidth,
          height: 200,
          layout: { background: { type: ColorType.Solid, color: '#1a1a1a' }, textColor: '#ffffff' },
          grid: { vertLines: { color: '#333333' }, horzLines: { color: '#333333' } },
          rightPriceScale: { borderColor: '#444444', textColor: '#ffffff' },
          timeScale: { borderColor: '#444444', timeVisible: true, secondsVisible: false },
        });
        updateChart(balanceChartInstance.current, performanceHistory?.balance || [], '#26a69a');
      }
      
      if (volumeChartRef.current) {
        volumeChartInstance.current = createChart(volumeChartRef.current, {
          width: volumeChartRef.current.clientWidth,
          height: 200,
          layout: { background: { type: ColorType.Solid, color: '#1a1a1a' }, textColor: '#ffffff' },
          grid: { vertLines: { color: '#333333' }, horzLines: { color: '#333333' } },
          rightPriceScale: { borderColor: '#444444', textColor: '#ffffff' },
          timeScale: { borderColor: '#444444', timeVisible: true, secondsVisible: false },
        });
        updateChart(volumeChartInstance.current, performanceHistory?.volume || [], '#42a5f5');
      }
      
      if (winRateChartRef.current) {
        winRateChartInstance.current = createChart(winRateChartRef.current, {
          width: winRateChartRef.current.clientWidth,
          height: 200,
          layout: { background: { type: ColorType.Solid, color: '#1a1a1a' }, textColor: '#ffffff' },
          grid: { vertLines: { color: '#333333' }, horzLines: { color: '#333333' } },
          rightPriceScale: { borderColor: '#444444', textColor: '#ffffff' },
          timeScale: { borderColor: '#444444', timeVisible: true, secondsVisible: false },
        });
        updateChart(winRateChartInstance.current, performanceHistory?.winRate || [], '#ab47bc');
      }
    }, 100);
  }, [performanceHistory]);

  // Chargement initial et mise à jour
  useEffect(() => {
    loadAllData();
    
    const interval = setInterval(loadAllData, 30000); // Mise à jour toutes les 30 secondes
    
    return () => clearInterval(interval);
  }, [selectedSymbol, selectedTimeframe]);

  const handleRefresh = () => {
    toast.promise(
      loadAllData(),
      {
        loading: 'Actualisation des statistiques...',
        success: 'Statistiques mises à jour',
        error: 'Erreur lors de l\'actualisation'
      }
    );
  };

  if (loading && !globalStats) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="flex items-center space-x-3">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
          <span className="text-white text-lg">Chargement des statistiques...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* En-tête */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white">📊 Statistiques de Trading</h1>
          <p className="text-gray-400 mt-1">
            Dernière mise à jour: {lastUpdate.toLocaleTimeString()}
          </p>
        </div>
        
        <div className="flex items-center space-x-4">
          {/* Sélecteur de timeframe */}
          <select
            value={selectedTimeframe}
            onChange={(e) => setSelectedTimeframe(e.target.value as any)}
            className="select"
          >
            <option value="24h">24 Heures</option>
            <option value="7d">7 Jours</option>
            <option value="30d">30 Jours</option>
            <option value="90d">90 Jours</option>
          </select>
          
          {/* Sélecteur de symbole */}
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value as TradingSymbol)}
            className="select"
          >
            {availableSymbols.map(symbol => (
              <option key={symbol} value={symbol}>{symbol}</option>
            ))}
          </select>
          
          <button
            onClick={handleRefresh}
            disabled={loading}
            className="btn-primary px-4 py-2"
          >
            {loading ? '⏳' : '🔄'} Actualiser
          </button>
        </div>
      </div>

      {/* Cartes de métriques globales */}
      {globalStats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
          <div className="chart-container">
            <div className="text-center">
              <div className="text-2xl font-bold text-white">
                {formatCurrency(globalStats.totalPnl)}
              </div>
              <div className="text-sm text-gray-400">P&L Total (Net)</div>
              <div className={`text-sm mt-1 ${globalStats.totalPnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {globalStats.totalPnl >= 0 ? '📈' : '📉'} Frais inclus
              </div>
            </div>
          </div>

          <div className="chart-container">
            <div className="text-center">
              <div className="text-2xl font-bold text-white">
                {formatPercent(globalStats.winRate)}
              </div>
              <div className="text-sm text-gray-400">Taux de Réussite</div>
              <div className={`text-sm mt-1 ${globalStats.winRate >= 50 ? 'text-green-400' : 'text-red-400'}`}>
                🎯 {globalStats.totalTrades} trades
              </div>
            </div>
          </div>

          <div className="chart-container">
            <div className="text-center">
              <div className="text-2xl font-bold text-white">
                {formatCurrency(globalStats.totalVolume)}
              </div>
              <div className="text-sm text-gray-400">Volume Total</div>
              <div className="text-sm text-blue-400 mt-1">
                💰 {formatCurrency(globalStats.avgTradeSize)} / trade
              </div>
            </div>
          </div>

          <div className="chart-container">
            <div className="text-center">
              <div className="text-2xl font-bold text-white">
                {formatCurrency(globalStats.totalBalance)}
              </div>
              <div className="text-sm text-gray-400">Balance Totale</div>
              <div className="text-sm text-green-400 mt-1">
                💳 {formatCurrency(globalStats.availableBalance)} disponible
              </div>
            </div>
          </div>

          <div className="chart-container">
            <div className="text-center">
              <div className={`text-2xl font-bold ${
                globalStats.profitFactor >= 2.0 ? 'text-emerald-400' :
                globalStats.profitFactor >= 1.5 ? 'text-green-400' :
                globalStats.profitFactor >= 1.3 ? 'text-lime-400' :
                globalStats.profitFactor >= 1.1 ? 'text-yellow-400' :
                globalStats.profitFactor >= 1.0 ? 'text-orange-400' : 'text-red-400'
              }`}>
                {globalStats.profitFactor.toFixed(2)}
              </div>
              <div className="text-sm text-gray-400">Profit Factor</div>
              <div className={`text-sm mt-1 ${
                globalStats.profitFactor >= 2.0 ? 'text-emerald-400' :
                globalStats.profitFactor >= 1.5 ? 'text-green-400' :
                globalStats.profitFactor >= 1.3 ? 'text-lime-400' :
                globalStats.profitFactor >= 1.1 ? 'text-yellow-400' :
                globalStats.profitFactor >= 1.0 ? 'text-orange-400' : 'text-red-400'
              }`}>
                {globalStats.profitFactor >= 2.0 ? '💎 Exceptionnel' :
                 globalStats.profitFactor >= 1.5 ? '🟢 Excellent' :
                 globalStats.profitFactor >= 1.3 ? '🌟 Très Bon' :
                 globalStats.profitFactor >= 1.1 ? '🟡 Bon' :
                 globalStats.profitFactor >= 1.0 ? '🟠 Profitable' : '🔴 Déficitaire'}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Graphiques de performance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="chart-container">
          <div className="chart-title">📈 Évolution P&L</div>
          <div ref={pnlChartRef} className="w-full h-50" />
        </div>

        <div className="chart-container">
          <div className="chart-title">💰 Évolution Balance</div>
          <div ref={balanceChartRef} className="w-full h-50" />
        </div>

        <div className="chart-container">
          <div className="chart-title">📊 Volume de Trading</div>
          <div ref={volumeChartRef} className="w-full h-50" />
        </div>

        <div className="chart-container">
          <div className="chart-title">🎯 Taux de Réussite</div>
          <div ref={winRateChartRef} className="w-full h-50" />
        </div>
      </div>

      {/* Section résumé optimisée */}
      {strategyStats.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
          <div className="chart-container">
            <div className="text-center">
              <div className="text-2xl font-bold text-white">
                {strategyStats.length}
              </div>
              <div className="text-sm text-gray-400">Stratégies Total</div>
              <div className="text-sm text-blue-400 mt-1">
                📊 Système complet
              </div>
            </div>
          </div>
          <div className="chart-container">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-400">
                {activeStats.length}
              </div>
              <div className="text-sm text-gray-400">Stratégies Actives</div>
              <div className="text-sm text-green-400 mt-1">
                🚀 Émettent des signaux
              </div>
            </div>
          </div>
          <div className="chart-container">
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-400">
                {profitableStats.length}
              </div>
              <div className="text-sm text-gray-400">Stratégies Profitables</div>
              <div className="text-sm text-yellow-400 mt-1">
                💰 P&L positif
              </div>
            </div>
          </div>
          <div className="chart-container">
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-400">
                {strategyStats.filter(s => (s.total_signals_emitted || 0) > 0 && (s.trades_executed || 0) === 0).length}
              </div>
              <div className="text-sm text-gray-400">Stratégies Filtrées</div>
              <div className="text-sm text-purple-400 mt-1">
                🚫 Signaux non retenus
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tableau des stratégies optimisé */}
      {strategyStats.length > 0 && (
        <div className="chart-container">
          <div className="chart-title">🎯 Performance des Stratégies (Signaux → Trades Réels)</div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-3 px-4 text-gray-300">Stratégie</th>
                  <th className="text-right py-3 px-4 text-gray-300">Signaux Émis</th>
                  <th className="text-right py-3 px-4 text-gray-300">Trades Participés</th>
                  <th className="text-right py-3 px-4 text-gray-300">Taux Participation</th>
                  <th className="text-right py-3 px-4 text-gray-300">Win Rate</th>
                  <th className="text-right py-3 px-4 text-gray-300">P&L Moy</th>
                  <th className="text-right py-3 px-4 text-gray-300">P&L Total</th>
                  <th className="text-right py-3 px-4 text-gray-300">Max Gain</th>
                  <th className="text-right py-3 px-4 text-gray-300">Max Perte</th>
                </tr>
              </thead>
              <tbody>
                {strategyStats.map((strategy, index) => {
                  const hasSignals = (strategy.total_signals_emitted || 0) > 0;
                  const hasTrades = (strategy.trades_executed || 0) > 0;
                  const conversionRate = strategy.signal_to_trade_rate || 0;
                  
                  return (
                  <tr key={index} className={`border-b border-gray-800 hover:bg-gray-800/30 ${
                    !hasSignals ? 'opacity-50' : hasTrades ? '' : 'bg-red-900/10'
                  }`}>
                    <td className="py-3 px-4 text-white font-medium">
                      <span className={`${
                        !hasSignals ? 'text-gray-500' : 
                        hasTrades ? (strategy.totalPnl >= 0 ? 'text-green-400' : 'text-red-400') :
                        'text-orange-400'
                      }`}>
                        {!hasSignals ? '😴' : hasTrades ? '✅' : '⚠️'}
                      </span> {strategy.strategy}
                    </td>
                    <td className="py-3 px-4 text-right text-gray-300">
                      {strategy.total_signals_emitted || 0}
                      {(strategy.buy_signals_emitted || 0) > 0 && (strategy.sell_signals_emitted || 0) > 0 && (
                        <div className="text-xs text-gray-500">
                          {strategy.buy_signals_emitted}B/{strategy.sell_signals_emitted}S
                        </div>
                      )}
                    </td>
                    <td className="py-3 px-4 text-right text-gray-300">
                      {strategy.trades_executed || 0}
                      {(strategy.buy_trades_executed || 0) > 0 && (strategy.sell_trades_executed || 0) > 0 && (
                        <div className="text-xs text-gray-500">
                          {strategy.buy_trades_executed}B/{strategy.sell_trades_executed}S
                        </div>
                      )}
                    </td>
                    <td className={`py-3 px-4 text-right font-mono ${
                      conversionRate >= 25 ? 'text-green-400' : 
                      conversionRate >= 10 ? 'text-yellow-400' :
                      conversionRate > 0 ? 'text-orange-400' : 'text-gray-500'
                    }`}>
                      {conversionRate > 0 ? `${conversionRate.toFixed(1)}%` : '-'}
                      {(strategy.buy_conversion_rate || 0) !== (strategy.sell_conversion_rate || 0) && (
                        <div className="text-xs text-gray-500">
                          B:{(strategy.buy_conversion_rate || 0).toFixed(1)}% S:{(strategy.sell_conversion_rate || 0).toFixed(1)}%
                        </div>
                      )}
                    </td>
                    <td className={`py-3 px-4 text-right font-mono ${
                      strategy.winRate >= 50 ? 'text-green-400' : 
                      strategy.winRate > 0 ? 'text-orange-400' : 'text-gray-500'
                    }`}>
                      {strategy.winRate > 0 ? formatPercent(strategy.winRate) : '-'}
                    </td>
                    <td className={`py-3 px-4 text-right font-mono ${
                      strategy.avgPnl >= 0 ? 'text-green-400' : 
                      strategy.avgPnl < 0 ? 'text-red-400' : 'text-gray-500'
                    }`}>
                      {strategy.avgPnl !== 0 ? formatCurrency(strategy.avgPnl) : '-'}
                    </td>
                    <td className={`py-3 px-4 text-right font-mono font-bold ${
                      strategy.totalPnl >= 0 ? 'text-green-400' : 
                      strategy.totalPnl < 0 ? 'text-red-400' : 'text-gray-500'
                    }`}>
                      {strategy.totalPnl !== 0 ? formatCurrency(strategy.totalPnl) : '-'}
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-green-400">
                      {strategy.max_gain ? formatCurrency(strategy.max_gain) : '-'}
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-red-400">
                      {strategy.max_loss ? formatCurrency(Math.abs(strategy.max_loss || 0)) : '-'}
                    </td>
                  </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tableau des statistiques par symbole */}
      {symbolStats.length > 0 && (
        <div className="chart-container">
          <div className="chart-title">💱 Performance par Symbole</div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-3 px-4 text-gray-300">Symbole</th>
                  <th className="text-right py-3 px-4 text-gray-300">Trades</th>
                  <th className="text-right py-3 px-4 text-gray-300">Volume</th>
                  <th className="text-right py-3 px-4 text-gray-300">P&L</th>
                  <th className="text-right py-3 px-4 text-gray-300">Win Rate</th>
                  <th className="text-right py-3 px-4 text-gray-300">Prix Actuel</th>
                  <th className="text-right py-3 px-4 text-gray-300">24h %</th>
                </tr>
              </thead>
              <tbody>
                {symbolStats.map((symbol, index) => (
                  <tr key={index} className="border-b border-gray-800 hover:bg-gray-800/30">
                    <td className="py-3 px-4 text-white font-bold">{symbol.symbol}</td>
                    <td className="py-3 px-4 text-right text-gray-300">{symbol.trades}</td>
                    <td className="py-3 px-4 text-right text-gray-300 font-mono">
                      {formatCurrency(symbol.volume)}
                    </td>
                    <td className={`py-3 px-4 text-right font-mono ${
                      symbol.pnl >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {formatCurrency(symbol.pnl)}
                    </td>
                    <td className={`py-3 px-4 text-right font-mono ${
                      symbol.winRate >= 50 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {formatPercent(symbol.winRate)}
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-white">
                      {formatCurrency(symbol.lastPrice)}
                    </td>
                    <td className={`py-3 px-4 text-right font-mono ${
                      symbol.priceChange24h >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {formatPercent(symbol.priceChange24h)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Indicateurs additionnels */}
      {globalStats && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="chart-container">
            <div className="chart-title">💸 Frais de Trading</div>
            <div className="text-center py-4">
              <div className="text-xl font-bold text-red-400">
                {formatCurrency(globalStats.totalFees)}
              </div>
              <div className="text-sm text-gray-400 mt-2">
                Total des frais payés
              </div>
            </div>
          </div>

          <div className="chart-container">
            <div className="chart-title">📋 Positions Actives</div>
            <div className="text-center py-4">
              <div className="text-xl font-bold text-blue-400">
                {globalStats.activePositions}
              </div>
              <div className="text-sm text-gray-400 mt-2">
                Positions ouvertes
              </div>
            </div>
          </div>

          <div className="chart-container">
            <div className="chart-title">📏 Taille Moy. Trade</div>
            <div className="text-center py-4">
              <div className="text-xl font-bold text-yellow-400">
                {formatCurrency(globalStats.avgTradeSize)}
              </div>
              <div className="text-sm text-gray-400 mt-2">
                Par transaction
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default StatisticsPage;