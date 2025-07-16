import type {
  MarketData,
  TradingSignal,
  Indicators,
  PerformanceData,
  ChartData,
  TimeInterval,
  TradingSymbol,
  PerformancePeriod,
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
    return this.request(
      `/api/charts/indicators/${symbol}?indicators=${indicators}&interval=${interval}&limit=${limit}`
    );
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

  // Portfolio API (port 8000)
  async getPortfolioSummary(): Promise<any> {
    return this.request(`http://localhost:8000/summary`);
  }

  async getPortfolioBalances(): Promise<any> {
    return this.request(`http://localhost:8000/balances`);
  }

  async getTradeHistory(page: number = 1, pageSize: number = 20): Promise<any> {
    return this.request(`http://localhost:8000/trades?page=${page}&page_size=${pageSize}`);
  }

  async getPortfolioPerformance(period: string = 'daily'): Promise<any> {
    return this.request(`http://localhost:8000/performance/${period}`);
  }

  async getActivePositions(): Promise<any> {
    return this.request(`http://localhost:8000/positions/active`);
  }

  async getRecentPositions(hours: number = 24): Promise<any> {
    return this.request(`http://localhost:8000/positions/recent?hours=${hours}`);
  }

  // Trader API (port 5002)
  async getTraderStats(): Promise<any> {
    return this.request(`http://localhost:5002/stats`);
  }

  async getOrderHistory(limit: number = 50): Promise<any> {
    return this.request(`http://localhost:5002/orders?limit=${limit}`);
  }

  async getTraderHealth(): Promise<any> {
    return this.request(`http://localhost:5002/health`);
  }

  // Alertes système (diagnostic multi-services)
  async getSystemAlerts(): Promise<any> {
    try {
      const [portfolioHealth, traderHealth] = await Promise.all([
        this.request(`http://localhost:8000/health`).catch(() => ({ status: 'offline', service: 'portfolio' })),
        this.request(`http://localhost:5002/health`).catch(() => ({ status: 'offline', service: 'trader' }))
      ]);

      return {
        portfolio: portfolioHealth,
        trader: traderHealth
      };
    } catch (error) {
      console.error('System alerts fetch error:', error);
      return { portfolio: { status: 'error' }, trader: { status: 'error' } };
    }
  }
  
  // Symboles disponibles
  async getAvailableSymbols(): Promise<{ symbols: TradingSymbol[] }> {
    return this.request('/api/available-symbols');
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