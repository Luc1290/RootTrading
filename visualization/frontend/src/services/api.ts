import type {
  MarketData,
  TradingSignal,
  Indicators,
  PerformanceData,
  ChartData,
  TimeInterval,
  TradingSymbol,
  PerformancePeriod,
  GlobalStatistics,
  SymbolStatistics,
  PerformanceHistory,
  StrategyStatistics,
} from '@/types';

class ApiService {
  private baseUrl: string;
  
  constructor(baseUrl: string = '') {
    this.baseUrl = baseUrl;
  }
  
  private async request<T>(endpoint: string): Promise<T> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`API Error [${endpoint}]:`, error);
      throw error;
    }
  }
  
  // Données de marché
  async getMarketData(
    symbol: TradingSymbol,
    interval: TimeInterval,
    limit: number = 2880
  ): Promise<{ data: MarketData }> {
    return this.request(`/api/charts/market/${symbol}?interval=${interval}&limit=${limit}`);
  }
  
  // Indicateurs techniques
  async getIndicators(
    symbol: TradingSymbol,
    indicators: string,
    interval: TimeInterval,
    limit: number = 2880
  ): Promise<ChartData> {
    const response: ChartData = await this.request(
      `/api/charts/indicators/${symbol}?indicators=${indicators}&interval=${interval}&limit=${limit}`
    );
    
    // Ajouter des alias pour compatibilité frontend
    if (response.indicators) {
      // Alias RSI
      if (response.indicators.rsi_14 && !response.indicators.rsi) {
        response.indicators.rsi = response.indicators.rsi_14;
      }
      
      // Alias MACD
      if (response.indicators.macd_line && !response.indicators.macd) {
        response.indicators.macd = response.indicators.macd_line;
      }
    }
    
    return response;
  }
  
  // Signaux de trading
  async getTradingSignals(
    symbol: TradingSymbol
  ): Promise<{ signals: { buy: TradingSignal[]; sell: TradingSignal[] } }> {
    return this.request(`/api/charts/signals/${symbol}`);
  }
  
  // Données de performance
  async getPerformanceData(
    period: PerformancePeriod,
    metric: string = 'pnl'
  ): Promise<{ data: PerformanceData }> {
    return this.request(`/api/charts/performance?period=${period}&metric=${metric}`);
  }

  // Portfolio API (via proxy)
  async getPortfolioSummary(): Promise<any> {
    return this.request(`/api/portfolio/summary`);
  }

  async getPortfolioBalances(): Promise<any> {
    return this.request(`/api/portfolio/balances`);
  }

  async getTradeHistory(page: number = 1, pageSize: number = 20): Promise<any> {
    return this.request(`/api/portfolio/trades?page=${page}&page_size=${pageSize}`);
  }

  async getPortfolioPerformance(period: string = 'daily'): Promise<any> {
    return this.request(`/api/portfolio/performance/${period}`);
  }

  async getActivePositions(): Promise<any> {
    return this.request(`/api/portfolio/positions/active`);
  }

  async getRecentPositions(hours: number = 24): Promise<any> {
    return this.request(`/api/portfolio/positions/recent?hours=${hours}`);
  }

  async getOwnedSymbolsWithVariations(): Promise<any> {
    return this.request(`/api/portfolio/symbols/owned`);
  }

  // Trader API (via proxy)
  async getTraderStats(): Promise<any> {
    return this.request(`/api/trader/stats`);
  }

  async getOrderHistory(limit: number = 50): Promise<any> {
    return this.request(`/api/trader/orders?limit=${limit}`);
  }

  async getTraderHealth(): Promise<any> {
    return this.request(`/api/trader/health`);
  }

  // Alertes système (diagnostic multi-services)
  async getSystemAlerts(): Promise<any> {
    return this.request('/api/system/alerts');
  }
  
  // Symboles disponibles
  async getAvailableSymbols(): Promise<{ symbols: TradingSymbol[] }> {
    return this.request('/api/available-symbols');
  }

  // Symboles configurés depuis shared/config
  async getConfiguredSymbols(): Promise<{ symbols: string[] }> {
    return this.request('/api/configured-symbols');
  }
  
  // Cycles de trading
  async getTradeCycles(
    symbol?: TradingSymbol,
    status?: string,
    limit: number = 100
  ): Promise<{ cycles: any[] }> {
    let query = `/api/trade-cycles?limit=${limit}`;
    if (symbol) query += `&symbol=${symbol}`;
    if (status) query += `&status=${status}`;
    return this.request(query);
  }

  // Statistics API endpoints
  async getGlobalStatistics(): Promise<GlobalStatistics> {
    return this.request('/api/statistics/global');
  }

  async getSymbolStatistics(symbol: TradingSymbol): Promise<{ symbols: SymbolStatistics[] }> {
    return this.request(`/api/statistics/symbol/${symbol}`);
  }

  async getPerformanceHistory(
    timeframe: '24h' | '7d' | '30d' | '90d' = '7d'
  ): Promise<PerformanceHistory> {
    return this.request(`/api/statistics/performance-history?timeframe=${timeframe}`);
  }

  async getStrategiesStatistics(): Promise<{ 
    strategies: StrategyStatistics[],
    consensus_strategies?: StrategyStatistics[],
    individual_strategies?: StrategyStatistics[]
  }> {
    return this.request('/api/statistics/strategies');
  }
  
  // Méthodes spécifiques pour les indicateurs
  async getRSI(
    symbol: TradingSymbol,
    interval: TimeInterval,
    limit: number = 2880
  ): Promise<ChartData> {
    return this.getIndicators(symbol, 'rsi', interval, limit);
  }
  
  async getMACD(
    symbol: TradingSymbol,
    interval: TimeInterval,
    limit: number = 2880
  ): Promise<ChartData> {
    return this.getIndicators(symbol, 'macd', interval, limit);
  }
  
  async getEMA(
    symbol: TradingSymbol,
    interval: TimeInterval,
    limit: number = 2880
  ): Promise<ChartData> {
    return this.getIndicators(symbol, 'ema', interval, limit);
  }
  
  // Méthodes combinées
  async getAllChartData(
    symbol: TradingSymbol,
    interval: TimeInterval,
    limit: number = 2880
  ): Promise<{
    marketData: MarketData;
    indicators: Indicators;
    signals: { buy: TradingSignal[]; sell: TradingSignal[] };
  }> {
    try {
      const [marketResponse, indicatorsResponse, signalsResponse] = await Promise.all([
        this.getMarketData(symbol, interval, limit),
        this.getIndicators(symbol, 'rsi,macd,ema', interval, limit),
        this.getTradingSignals(symbol),
      ]);
      
      return {
        marketData: marketResponse.data,
        indicators: indicatorsResponse.indicators || {},
        signals: signalsResponse.signals,
      };
    } catch (error) {
      console.error('Error fetching all chart data:', error);
      throw error;
    }
  }
}

export const apiService = new ApiService();
export default apiService;