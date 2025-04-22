import { useState, useEffect } from 'react';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5002';

/**
 * Hook for managing trading cycles
 * Provides functions to fetch, create, and close trading cycles
 */
const useCycle = () => {
  const [cycles, setCycles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch all cycles
  const fetchCycles = async (filters = {}) => {
    setLoading(true);
    setError(null);
    
    try {
      // Build query string from filters
      const queryParams = new URLSearchParams();
      
      if (filters.symbol) queryParams.append('symbol', filters.symbol);
      if (filters.strategy) queryParams.append('strategy', filters.strategy);
      if (filters.status) queryParams.append('status', filters.status);
      if (filters.startDate) queryParams.append('start_date', filters.startDate);
      if (filters.endDate) queryParams.append('end_date', filters.endDate);
      
      // Default pagination
      queryParams.append('page', filters.page || 1);
      queryParams.append('page_size', filters.pageSize || 100);
      
      const queryString = queryParams.toString();
      const url = `${API_BASE_URL}/trades${queryString ? `?${queryString}` : ''}`;
      
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch cycles: ${response.statusText}`);
      }
      
      const data = await response.json();
      setCycles(data.trades || []);
      
      return data;
    } catch (err) {
      setError(err.message);
      console.error('Error fetching cycles:', err);
      return { trades: [], total_count: 0 };
    } finally {
      setLoading(false);
    }
  };
  
  // Create manual order
  const createOrder = async (orderData) => {
    try {
      const response = await fetch(`${API_BASE_URL}/order`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(orderData),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Failed to create order: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      // Refresh cycles list
      fetchCycles();
      
      return result;
    } catch (err) {
      setError(err.message);
      console.error('Error creating order:', err);
      throw err;
    }
  };
  
  // Close a cycle
  const closeCycle = async (cycleId, price = null) => {
    try {
      const url = `${API_BASE_URL}/close/${cycleId}`;
      const payload = price ? { price } : {};
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Failed to close cycle: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      // Refresh cycles list
      fetchCycles();
      
      return result;
    } catch (err) {
      setError(err.message);
      console.error(`Error closing cycle ${cycleId}:`, err);
      throw err;
    }
  };
  
  // Cancel a cycle
  const cancelCycle = async (cycleId, reason = 'Manual cancellation') => {
    try {
      const response = await fetch(`${API_BASE_URL}/order/${cycleId}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ reason }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Failed to cancel cycle: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      // Refresh cycles list
      fetchCycles();
      
      return result;
    } catch (err) {
      setError(err.message);
      console.error(`Error canceling cycle ${cycleId}:`, err);
      throw err;
    }
  };
  
  // Effect to fetch cycles on component mount
  useEffect(() => {
    fetchCycles();
    
    // Set up polling interval to refresh cycles periodically
    const intervalId = setInterval(() => {
      fetchCycles();
    }, 30000); // Refresh every 30 seconds
    
    // Clean up interval on component unmount
    return () => clearInterval(intervalId);
  }, []);
  
  return {
    cycles,
    loading,
    error,
    fetchCycles,
    createOrder,
    closeCycle,
    cancelCycle
  };
};

export default useCycle;