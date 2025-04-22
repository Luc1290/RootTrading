import { useState, useEffect } from 'react';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';

/**
 * Hook for accessing trading signals
 * Provides functions to fetch and filter signals
 */
const useSignals = () => {
  const [signals, setSignals] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch signals with optional filters
  const fetchSignals = async (filters = {}) => {
    setLoading(true);
    setError(null);
    
    try {
      // Build query string from filters
      const queryParams = new URLSearchParams();
      
      if (filters.symbol) queryParams.append('symbol', filters.symbol);
      if (filters.strategy) queryParams.append('strategy', filters.strategy);
      if (filters.side) queryParams.append('side', filters.side);
      if (filters.startDate) queryParams.append('start_date', filters.startDate);
      if (filters.endDate) queryParams.append('end_date', filters.endDate);
      if (filters.limit) queryParams.append('limit', filters.limit);
      
      const queryString = queryParams.toString();
      const url = `${API_BASE_URL}/signals${queryString ? `?${queryString}` : ''}`;
      
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch signals: ${response.statusText}`);
      }
      
      const data = await response.json();
      setSignals(data || []);
      
      return data;
    } catch (err) {
      setError(err.message);
      console.error('Error fetching signals:', err);
      return [];
    } finally {
      setLoading(false);
    }
  };
  
  // Get signals by strategy
  const getSignalsByStrategy = () => {
    if (!signals.length) return [];
    
    // Group signals by strategy
    const strategyMap = {};
    
    signals.forEach(signal => {
      if (!strategyMap[signal.strategy]) {
        strategyMap[signal.strategy] = [];
      }
      strategyMap[signal.strategy].push(signal);
    });
    
    // Convert to array format
    return Object.entries(strategyMap).map(([strategy, signals]) => ({
      strategy,
      signals,
      count: signals.length,
      buyCount: signals.filter(s => s.side === 'BUY').length,
      sellCount: signals.filter(s => s.side === 'SELL').length,
    }));
  };
  
  // Get signals by symbol
  const getSignalsBySymbol = () => {
    if (!signals.length) return [];
    
    // Group signals by symbol
    const symbolMap = {};
    
    signals.forEach(signal => {
      if (!symbolMap[signal.symbol]) {
        symbolMap[signal.symbol] = [];
      }
      symbolMap[signal.symbol].push(signal);
    });
    
    // Convert to array format
    return Object.entries(symbolMap).map(([symbol, signals]) => ({
      symbol,
      signals,
      count: signals.length,
      buyCount: signals.filter(s => s.side === 'BUY').length,
      sellCount: signals.filter(s => s.side === 'SELL').length,
    }));
  };
  
  // Get recent signals
  const getRecentSignals = (limit = 5) => {
    if (!signals.length) return [];
    
    // Sort by timestamp in descending order and take the first 'limit' signals
    return [...signals]
      .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
      .slice(0, limit);
  };
  
  // Effect to fetch signals on component mount
  useEffect(() => {
    fetchSignals();
    
    // Set up polling interval to refresh signals periodically
    const intervalId = setInterval(() => {
      fetchSignals();
    }, 60000); // Refresh every minute
    
    // Clean up interval on component unmount
    return () => clearInterval(intervalId);
  }, []);
  
  return {
    signals,
    loading,
    error,
    fetchSignals,
    getSignalsByStrategy,
    getSignalsBySymbol,
    getRecentSignals
  };
};

export default useSignals;