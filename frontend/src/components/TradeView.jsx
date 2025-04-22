import React, { useState, useEffect } from 'react';
import { ArrowUp, ArrowDown, Clock, Check, X, AlertTriangle, RefreshCw } from 'lucide-react';
import useCycle from '../api/useCycle';
import usePortfolio from '../api/usePortfolio';

const TradeView = () => {
  const { cycles, loading: cyclesLoading, error: cyclesError, createOrder, closeCycle } = useCycle();
  const { summary, loading: portfolioLoading } = usePortfolio();
  
  // State for manual order form
  const [orderForm, setOrderForm] = useState({
    symbol: 'BTCUSDC',
    side: 'BUY',
    quantity: '',
    price: ''
  });
  
  // State for active tab
  const [activeTab, setActiveTab] = useState('active');
  
  // State for form submission and feedback
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [feedback, setFeedback] = useState(null);
  
  // Set default symbols from portfolio if available
  useEffect(() => {
    if (summary?.balances && summary.balances.length > 0) {
      // Find symbols from balances that end with USDC
      const symbols = summary.balances
        .filter(balance => balance.asset !== 'USDC')
        .map(balance => `${balance.asset}USDC`);
      
      if (symbols.length > 0) {
        setOrderForm(prev => ({ ...prev, symbol: symbols[0] }));
      }
    }
  }, [summary]);
  
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setOrderForm(prev => ({ ...prev, [name]: value }));
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setFeedback(null);
    
    try {
      // Convert string values to numbers
      const orderData = {
        ...orderForm,
        quantity: parseFloat(orderForm.quantity),
        price: orderForm.price ? parseFloat(orderForm.price) : null
      };
      
      const result = await createOrder(orderData);
      
      setFeedback({
        type: 'success',
        message: `Order successfully created with ID: ${result.order_id}`
      });
      
      // Reset form
      setOrderForm(prev => ({
        ...prev,
        quantity: '',
        price: ''
      }));
    } catch (error) {
      setFeedback({
        type: 'error',
        message: error.message || 'Failed to create order'
      });
    } finally {
      setIsSubmitting(false);
    }
  };
  
  const handleCloseCycle = async (cycleId) => {
    try {
      await closeCycle(cycleId);
      setFeedback({
        type: 'success',
        message: `Cycle ${cycleId} closed successfully`
      });
    } catch (error) {
      setFeedback({
        type: 'error',
        message: error.message || 'Failed to close cycle'
      });
    }
  };
  
  // Format currency
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', { 
      style: 'currency', 
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };
  
  // Filter cycles based on active tab
  const filteredCycles = cycles?.filter(cycle => {
    if (activeTab === 'active') {
      return ['initiating', 'waiting_buy', 'active_buy', 'waiting_sell', 'active_sell'].includes(cycle.status);
    } else if (activeTab === 'completed') {
      return cycle.status === 'completed';
    } else if (activeTab === 'canceled') {
      return ['canceled', 'failed'].includes(cycle.status);
    }
    return true; // All tab
  }) || [];
  
  // Status badge color
  const getStatusColor = (status) => {
    switch (status) {
      case 'initiating':
      case 'waiting_buy':
      case 'waiting_sell':
        return 'bg-yellow-100 text-yellow-800';
      case 'active_buy':
      case 'active_sell':
        return 'bg-blue-100 text-blue-800';
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'canceled':
      case 'failed':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  // Loading state
  if (cyclesLoading || portfolioLoading) {
    return (
      <div className="flex justify-center items-center h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  // Error state
  if (cyclesError) {
    return (
      <div className="text-center text-red-500 p-4">
        <AlertTriangle className="mx-auto h-12 w-12 mb-4" />
        <h2 className="text-2xl font-bold mb-2">Error Loading Cycles</h2>
        <p>{cyclesError}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Trade Management</h1>
      
      {/* Manual Order Form */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-semibold mb-4">Create Manual Order</h2>
        
        {feedback && (
          <div className={`p-3 mb-4 rounded ${feedback.type === 'success' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
            {feedback.type === 'success' ? <Check className="inline mr-2" size={16} /> : <AlertTriangle className="inline mr-2" size={16} />}
            {feedback.message}
          </div>
        )}
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Symbol</label>
              <select
                name="symbol"
                value={orderForm.symbol}
                onChange={handleInputChange}
                className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              >
                {['BTCUSDC', 'ETHUSDC', 'SOLUSDC'].map(symbol => (
                  <option key={symbol} value={symbol}>{symbol}</option>
                ))}
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Side</label>
              <select
                name="side"
                value={orderForm.side}
                onChange={handleInputChange}
                className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              >
                <option value="BUY">BUY</option>
                <option value="SELL">SELL</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Quantity</label>
              <input
                type="number"
                name="quantity"
                value={orderForm.quantity}
                onChange={handleInputChange}
                placeholder="0.001"
                step="0.000001"
                min="0"
                required
                className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Price (optional)</label>
              <input
                type="number"
                name="price"
                value={orderForm.price}
                onChange={handleInputChange}
                placeholder="Market price if empty"
                step="0.01"
                min="0"
                className="w-full p-2 border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
              />
              <p className="text-sm text-gray-500 mt-1">Leave empty for market price</p>
            </div>
          </div>
          
          <div className="flex justify-end">
            <button
              type="submit"
              disabled={isSubmitting}
              className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50"
            >
              {isSubmitting ? (
                <>
                  <RefreshCw className="inline mr-2 animate-spin" size={16} />
                  Submitting...
                </>
              ) : (
                'Create Order'
              )}
            </button>
          </div>
        </form>
      </div>
      
      {/* Cycles List */}
      <div className="bg-white rounded-lg shadow overflow-hidden">
        <div className="border-b">
          <nav className="flex">
            {['active', 'completed', 'canceled', 'all'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`py-4 px-6 focus:outline-none ${
                  activeTab === tab 
                    ? 'border-b-2 border-blue-500 font-medium text-blue-600' 
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
                {tab === 'active' && filteredCycles.length > 0 && (
                  <span className="ml-2 bg-blue-100 text-blue-600 py-0.5 px-2 rounded-full text-xs">
                    {filteredCycles.length}
                  </span>
                )}
              </button>
            ))}
          </nav>
        </div>
        
        <div className="overflow-x-auto">
          {filteredCycles.length > 0 ? (
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ID/Symbol</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Strategy</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Entry</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Exit/Current</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">P&L</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Created</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {filteredCycles.map((cycle) => {
                  // Calculate P&L for active cycles
                  let pnl = cycle.profit_loss;
                  let pnlPercent = cycle.profit_loss_percent;
                  
                  if (['active_buy', 'waiting_sell'].includes(cycle.status) && cycle.current_price) {
                    // For active buy positions: current_price - entry_price
                    pnl = (cycle.current_price - cycle.entry_price) * cycle.quantity;
                    pnlPercent = ((cycle.current_price - cycle.entry_price) / cycle.entry_price) * 100;
                  } else if (['active_sell', 'waiting_buy'].includes(cycle.status) && cycle.current_price) {
                    // For active sell positions: entry_price - current_price
                    pnl = (cycle.entry_price - cycle.current_price) * cycle.quantity;
                    pnlPercent = ((cycle.entry_price - cycle.current_price) / cycle.entry_price) * 100;
                  }
                  
                  return (
                    <tr key={cycle.id}>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="font-medium text-gray-900">{cycle.id.substring(0, 8)}...</div>
                        <div className="text-sm text-gray-500">{cycle.symbol}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {cycle.strategy}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full ${getStatusColor(cycle.status)}`}>
                          {cycle.status}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="font-medium">{formatCurrency(cycle.entry_price)}</div>
                        <div className="text-sm text-gray-500">{cycle.quantity} units</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        {cycle.exit_price ? (
                          <div className="font-medium">{formatCurrency(cycle.exit_price)}</div>
                        ) : cycle.current_price ? (
                          <div className="font-medium">{formatCurrency(cycle.current_price)}</div>
                        ) : (
                          <div className="text-gray-400">-</div>
                        )}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        {pnl !== null && pnl !== undefined ? (
                          <div className={`flex items-center ${pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {pnl >= 0 ? <ArrowUp size={16} className="mr-1" /> : <ArrowDown size={16} className="mr-1" />}
                            {formatCurrency(pnl)}
                            <span className="ml-1 text-xs">
                              ({pnlPercent >= 0 ? '+' : ''}{pnlPercent.toFixed(2)}%)
                            </span>
                          </div>
                        ) : (
                          <div className="text-gray-400">-</div>
                        )}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        <div className="flex items-center">
                          <Clock size={14} className="mr-1" />
                          {new Date(cycle.created_at).toLocaleDateString()}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {['active_buy', 'active_sell', 'waiting_buy', 'waiting_sell'].includes(cycle.status) && (
                          <button
                            onClick={() => handleCloseCycle(cycle.id)}
                            className="text-blue-600 hover:text-blue-800"
                          >
                            Close
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          ) : (
            <div className="text-center p-6 text-gray-500">
              No trading cycles found
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TradeView;