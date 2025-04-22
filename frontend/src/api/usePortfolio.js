import { useState, useEffect } from 'react';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/**
 * Hook for accessing portfolio data
 * Provides access to portfolio summary, balances, and performance metrics
 */
const usePortfolio = () => {
  const [summary, setSummary] = useState(null);
  const [balances, setBalances] = useState([]);
  const [performance, setPerformance] = useState({
    daily: [],
    weekly: [],
    monthly: []
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch summary data
  const fetchSummary = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/summary`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch portfolio summary: ${response.statusText}`);
      }
      
      const data = await response.json();
      setSummary(data);
      
      return data;
    } catch (err) {
      setError(err.message);
      console.error('Error fetching portfolio summary:', err);
      return null;
    }
  };

  // Fetch balances
  const fetchBalances = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/balances`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch balances: ${response.statusText}`);
      }
      
      const data = await response.json();
      setBalances(data);
      
      return data;
    } catch (err) {
      setError(err.message);
      console.error('Error fetching balances:', err);
      return [];
    }
  };

  // Fetch performance data for a specific period
  const fetchPerformance = async (period) => {
    try {
      const response = await fetch(`${API_BASE_URL}/performance/${period}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch ${period} performance: ${response.statusText}`);
      }
      
      const result = await response.json();
      return result.data || [];
    } catch (err) {
      console.error(`Error fetching ${period} performance:`, err);
      return [];
    }
  };

  // Fetch all portfolio data
  const fetchAllData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Fetch summary first
      const summaryData = await fetchSummary();
      
      // Fetch balances
      await fetchBalances();
      
      // Fetch performance metrics for all periods
      const [dailyPerf, weeklyPerf, monthlyPerf] = await Promise.all([
        fetchPerformance('daily'),
        fetchPerformance('weekly'),
        fetchPerformance('monthly')
      ]);
      
      setPerformance({
        daily: dailyPerf,
        weekly: weeklyPerf,
        monthly: monthlyPerf
      });
      
      // Fetch strategy and symbol performance
      await Promise.all([
        fetchPerformanceByStrategy(),
        fetchPerformanceBySymbol()
      ]);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching portfolio data:', err);
    } finally {
      setLoading(false);
    }
  };
  
  // Fetch performance by strategy
  const fetchPerformanceByStrategy = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/performance/strategy`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch strategy performance: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      // Store in performance object for now
      setPerformance(prev => ({
        ...prev,
        strategy: result.data || []
      }));
      
      return result.data || [];
    } catch (err) {
      console.error('Error fetching strategy performance:', err);
      return [];
    }
  };
  
  // Fetch performance by symbol
  const fetchPerformanceBySymbol = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/performance/symbol`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch symbol performance: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      // Store in performance object
      setPerformance(prev => ({
        ...prev,
        symbol: result.data || []
      }));
      
      return result.data || [];
    } catch (err) {
      console.error('Error fetching symbol performance:', err);
      return [];
    }
  };

  // Effect to fetch data on component mount
  useEffect(() => {
    fetchAllData();
    
    // Set up polling interval to refresh data periodically
    const intervalId = setInterval(() => {
      fetchAllData();
    }, 60000); // Refresh every minute
    
    // Clean up interval on component unmount
    return () => clearInterval(intervalId);
  }, []);
  
  // Method to manually refresh data
  const refreshData = () => {
    return fetchAllData();
  };
  
  // Update balances, e.g. after a manual order
  const updateBalances = async (newBalances) => {
    try {
      const response = await fetch(`${API_BASE_URL}/balances/update`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newBalances),
      });
      
      if (!response.ok) {
        throw new Error(`Failed to update balances: ${response.statusText}`);
      }
      
      await fetchBalances();
      return true;
    } catch (err) {
      setError(err.message);
      console.error('Error updating balances:', err);
      return false;
    }
  };

  return {
    summary,
    balances,
    performance,
    loading,
    error,
    refreshData,
    updateBalances
  };
};

export default usePortfolio;