import React, { useState, useEffect } from 'react';
import { apiService } from '@/services/api';

interface RecentOrder {
  id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  status: string;
  strategy: string;
  timestamp: string;
  binance_order_id: string;
}

function RecentOrdersPanel() {
  const [orders, setOrders] = useState<RecentOrder[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchRecentOrders();
    const interval = setInterval(fetchRecentOrders, 15000);
    return () => clearInterval(interval);
  }, []);

  const fetchRecentOrders = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Récupérer les ordres récents du trader
      const ordersResponse = await apiService.getOrderHistory(10);
      
      // Transformer les données pour le format attendu
      const realOrders: RecentOrder[] = ordersResponse.orders.map((order: any) => ({
        id: order.id,
        symbol: order.symbol,
        side: order.side as 'BUY' | 'SELL',
        quantity: parseFloat(order.quantity || '0'),
        price: parseFloat(order.price || '0'),
        status: order.status,
        strategy: order.strategy,
        timestamp: order.timestamp,
        binance_order_id: order.binance_order_id
      }));

      setOrders(realOrders);
    } catch (err) {
      setError('Erreur lors du chargement des ordres');
      console.error('Recent orders fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (value: number): string => {
    return new Intl.NumberFormat('fr-FR', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 8
    }).format(value);
  };

  const formatTimeAgo = (timestamp: string): string => {
    const now = new Date();
    const time = new Date(timestamp);
    const diff = now.getTime() - time.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    }
    return `${minutes}m`;
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

  return (
    <div className="h-64 bg-dark-300 rounded-lg p-4 overflow-y-auto">
      <div className="space-y-2">
        {orders.length === 0 ? (
          <div className="text-center text-gray-400 py-8">
            Aucun ordre récent
          </div>
        ) : (
          orders.map((order) => (
            <div key={order.id} className="bg-dark-200 rounded-lg p-3">
              {/* Header */}
              <div className="flex justify-between items-start mb-2">
                <div className="flex items-center space-x-2">
                  <span className="text-white font-medium">{order.symbol}</span>
                  <span className={`text-xs px-2 py-1 rounded ${
                    order.side === 'BUY' ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
                  }`}>
                    {order.side}
                  </span>
                  <span className={`text-xs px-2 py-1 rounded ${
                    order.status === 'FILLED' ? 'bg-blue-900 text-blue-300' : 'bg-yellow-900 text-yellow-300'
                  }`}>
                    {order.status}
                  </span>
                </div>
                <div className="text-xs text-gray-400">
                  {formatTimeAgo(order.timestamp)}
                </div>
              </div>

              {/* Details */}
              <div className="grid grid-cols-3 gap-2 text-xs mb-2">
                <div>
                  <div className="text-gray-400">Quantité</div>
                  <div className="text-white">{order.quantity}</div>
                </div>
                <div>
                  <div className="text-gray-400">Prix</div>
                  <div className="text-white">{formatCurrency(order.price)}</div>
                </div>
                <div>
                  <div className="text-gray-400">Valeur</div>
                  <div className="text-white">{formatCurrency(order.quantity * order.price)}</div>
                </div>
              </div>

              {/* Footer */}
              <div className="flex justify-between items-center text-xs">
                <div className="text-gray-400">
                  {order.strategy}
                </div>
                <div className="text-gray-400">
                  #{order.binance_order_id}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default RecentOrdersPanel;