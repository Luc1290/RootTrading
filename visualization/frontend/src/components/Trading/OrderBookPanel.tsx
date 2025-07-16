import React, { useState, useEffect } from 'react';
import { apiService } from '@/services/api';
import { useChartStore } from '@/stores/useChartStore';

interface OrderBookEntry {
  price: number;
  quantity: number;
  total: number;
}

interface OrderBookData {
  bids: OrderBookEntry[];
  asks: OrderBookEntry[];
  lastUpdateId: number;
}

function OrderBookPanel() {
  const { config } = useChartStore();
  const [orderBook, setOrderBook] = useState<OrderBookData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchOrderBook();
    const interval = setInterval(fetchOrderBook, 5000);
    return () => clearInterval(interval);
  }, [config.symbol]);

  const fetchOrderBook = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Simuler des données d'orderbook en attendant l'API
      const mockOrderBook: OrderBookData = {
        bids: [
          { price: 43195.50, quantity: 0.125, total: 5399.44 },
          { price: 43194.25, quantity: 0.340, total: 14686.05 },
          { price: 43193.00, quantity: 0.055, total: 2375.62 },
          { price: 43192.75, quantity: 0.180, total: 7774.70 },
          { price: 43191.50, quantity: 0.225, total: 9718.09 },
          { price: 43190.25, quantity: 0.095, total: 4103.07 },
          { price: 43189.00, quantity: 0.310, total: 13388.59 },
          { price: 43188.75, quantity: 0.160, total: 6910.20 },
          { price: 43187.50, quantity: 0.205, total: 8853.44 },
          { price: 43186.25, quantity: 0.145, total: 6262.01 },
        ],
        asks: [
          { price: 43196.75, quantity: 0.085, total: 3671.72 },
          { price: 43197.00, quantity: 0.275, total: 11879.18 },
          { price: 43198.25, quantity: 0.140, total: 6047.76 },
          { price: 43199.50, quantity: 0.190, total: 8207.91 },
          { price: 43200.75, quantity: 0.110, total: 4752.08 },
          { price: 43201.00, quantity: 0.255, total: 11016.26 },
          { price: 43202.25, quantity: 0.175, total: 7560.39 },
          { price: 43203.50, quantity: 0.095, total: 4104.33 },
          { price: 43204.75, quantity: 0.210, total: 9073.00 },
          { price: 43205.00, quantity: 0.125, total: 5400.63 },
        ],
        lastUpdateId: Date.now(),
      };

      setOrderBook(mockOrderBook);
    } catch (err) {
      setError('Erreur lors du chargement de l\'orderbook');
      console.error('OrderBook fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatPrice = (price: number): string => {
    return price.toFixed(2);
  };

  const formatQuantity = (quantity: number): string => {
    return quantity.toFixed(3);
  };

  const formatTotal = (total: number): string => {
    return total.toFixed(0);
  };

  const getBarWidth = (quantity: number, maxQuantity: number): string => {
    return `${(quantity / maxQuantity) * 100}%`;
  };

  if (loading) {
    return (
      <div className="h-64 bg-dark-300 rounded-lg p-4 flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-64 bg-dark-300 rounded-lg p-4 flex items-center justify-center">
        <span className="text-red-400">{error}</span>
      </div>
    );
  }

  if (!orderBook) {
    return (
      <div className="h-64 bg-dark-300 rounded-lg p-4 flex items-center justify-center">
        <span className="text-gray-400">Pas de données</span>
      </div>
    );
  }

  const maxQuantity = Math.max(
    ...orderBook.bids.map(b => b.quantity),
    ...orderBook.asks.map(a => a.quantity)
  );

  return (
    <div className="h-64 bg-dark-300 rounded-lg p-4 overflow-hidden">
      {/* Header */}
      <div className="grid grid-cols-3 gap-4 text-xs font-medium text-gray-400 mb-2">
        <div>Prix</div>
        <div className="text-center">Quantité</div>
        <div className="text-right">Total</div>
      </div>

      {/* Asks (Ventes) */}
      <div className="h-20 overflow-y-auto mb-2">
        <div className="space-y-1">
          {orderBook.asks.slice().reverse().map((ask, index) => (
            <div
              key={`ask-${index}`}
              className="grid grid-cols-3 gap-4 text-xs py-1 relative"
            >
              {/* Background bar */}
              <div
                className="absolute inset-0 bg-red-900/30 rounded"
                style={{ width: getBarWidth(ask.quantity, maxQuantity) }}
              />
              
              <div className="text-red-400 relative z-10">{formatPrice(ask.price)}</div>
              <div className="text-white text-center relative z-10">{formatQuantity(ask.quantity)}</div>
              <div className="text-gray-400 text-right relative z-10">{formatTotal(ask.total)}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Spread */}
      <div className="border-t border-b border-gray-700 py-2 mb-2">
        <div className="text-center text-xs">
          <span className="text-gray-400">Spread: </span>
          <span className="text-white">
            {(orderBook.asks[0].price - orderBook.bids[0].price).toFixed(2)}
          </span>
        </div>
      </div>

      {/* Bids (Achats) */}
      <div className="h-20 overflow-y-auto">
        <div className="space-y-1">
          {orderBook.bids.map((bid, index) => (
            <div
              key={`bid-${index}`}
              className="grid grid-cols-3 gap-4 text-xs py-1 relative"
            >
              {/* Background bar */}
              <div
                className="absolute inset-0 bg-green-900/30 rounded"
                style={{ width: getBarWidth(bid.quantity, maxQuantity) }}
              />
              
              <div className="text-green-400 relative z-10">{formatPrice(bid.price)}</div>
              <div className="text-white text-center relative z-10">{formatQuantity(bid.quantity)}</div>
              <div className="text-gray-400 text-right relative z-10">{formatTotal(bid.total)}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default OrderBookPanel;