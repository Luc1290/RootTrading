import React, { useState, useEffect } from 'react';
import { apiService } from '@/services/api';

interface SystemAlert {
  id: string;
  type: 'error' | 'warning' | 'info' | 'success';
  service: string;
  message: string;
  timestamp: string;
  resolved: boolean;
  details?: string;
}

function AlertsPanel() {
  const [alerts, setAlerts] = useState<SystemAlert[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchAlerts();
    const interval = setInterval(fetchAlerts, 10000);
    return () => clearInterval(interval);
  }, []);

  const fetchAlerts = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Récupérer les vraies données de santé des services
      const healthResponse = await apiService.getSystemAlerts();
      
      const realAlerts: SystemAlert[] = [];
      
      // Traiter les alertes du portfolio
      if (healthResponse.portfolio) {
        const isPortfolioHealthy = healthResponse.portfolio.status === 'ok' || healthResponse.portfolio.status === 'healthy';
        const portfolioAlert: SystemAlert = {
          id: `portfolio_${Date.now()}`,
          type: isPortfolioHealthy ? 'success' : 'error',
          service: 'portfolio',
          message: isPortfolioHealthy ? 'Service Portfolio opérationnel' : `Service Portfolio: ${healthResponse.portfolio.status}`,
          timestamp: new Date().toISOString(),
          resolved: isPortfolioHealthy,
          details: isPortfolioHealthy 
            ? `DB: ${healthResponse.portfolio.database || 'OK'}, Uptime: ${healthResponse.portfolio.uptime || 'N/A'}` 
            : `Erreur: ${healthResponse.portfolio.error || healthResponse.portfolio.status}`
        };
        realAlerts.push(portfolioAlert);
      }
      
      // Traiter les alertes du trader
      if (healthResponse.trader) {
        const isTraderHealthy = healthResponse.trader.status === 'healthy' || healthResponse.trader.status === 'ok';
        const traderAlert: SystemAlert = {
          id: `trader_${Date.now()}`,
          type: isTraderHealthy ? 'success' : 'error',
          service: 'trader',
          message: isTraderHealthy ? 'Service Trader opérationnel' : `Service Trader: ${healthResponse.trader.status}`,
          timestamp: new Date().toISOString(),
          resolved: isTraderHealthy,
          details: isTraderHealthy 
            ? `Mode: ${healthResponse.trader.mode || 'N/A'}, Symboles: ${healthResponse.trader.symbols?.length || 0}, Uptime: ${Math.floor((healthResponse.trader.uptime || 0) / 3600)}h`
            : `Erreur: ${healthResponse.trader.error || healthResponse.trader.status}`
        };
        realAlerts.push(traderAlert);
      }
      
      // Ajouter une alerte générale si pas de données
      if (realAlerts.length === 0) {
        realAlerts.push({
          id: 'general_alert',
          type: 'info',
          service: 'système',
          message: 'Surveillance des services en cours',
          timestamp: new Date().toISOString(),
          resolved: true,
          details: 'Monitoring des services en cours d\'initialisation'
        });
      }

      setAlerts(realAlerts);
    } catch (err) {
      setError('Erreur lors du chargement des alertes');
      console.error('Alerts fetch error:', err);
    } finally {
      setLoading(false);
    }
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

  const getAlertIcon = (type: SystemAlert['type']): string => {
    switch (type) {
      case 'error': return '🚨';
      case 'warning': return '⚠️';
      case 'info': return 'ℹ️';
      case 'success': return '✅';
      default: return '📢';
    }
  };

  const getAlertColor = (type: SystemAlert['type']): string => {
    switch (type) {
      case 'error': return 'border-red-500 bg-red-900/20';
      case 'warning': return 'border-yellow-500 bg-yellow-900/20';
      case 'info': return 'border-blue-500 bg-blue-900/20';
      case 'success': return 'border-green-500 bg-green-900/20';
      default: return 'border-gray-500 bg-gray-900/20';
    }
  };

  const getTextColor = (type: SystemAlert['type']): string => {
    switch (type) {
      case 'error': return 'text-red-400';
      case 'warning': return 'text-yellow-400';
      case 'info': return 'text-blue-400';
      case 'success': return 'text-green-400';
      default: return 'text-gray-400';
    }
  };

  if (loading) {
    return (
      <div className="h-48 bg-dark-300 rounded-lg p-4 flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-48 bg-dark-300 rounded-lg p-4 flex items-center justify-center">
        <span className="text-red-400">{error}</span>
      </div>
    );
  }

  return (
    <div className="h-48 bg-dark-300 rounded-lg p-4 overflow-y-auto">
      <div className="space-y-2">
        {alerts.length === 0 ? (
          <div className="text-center text-gray-400 py-8">
            Aucune alerte récente
          </div>
        ) : (
          alerts.slice(0, 10).map((alert) => (
            <div
              key={alert.id}
              className={`border-l-4 rounded-r-lg p-3 ${getAlertColor(alert.type)} ${
                alert.resolved ? 'opacity-60' : ''
              }`}
            >
              {/* Header */}
              <div className="flex justify-between items-start mb-1">
                <div className="flex items-center space-x-2">
                  <span className="text-sm">{getAlertIcon(alert.type)}</span>
                  <span className="text-xs font-medium text-gray-300">
                    {alert.service.toUpperCase()}
                  </span>
                  {alert.resolved && (
                    <span className="text-xs px-2 py-1 rounded bg-gray-700 text-gray-300">
                      RÉSOLU
                    </span>
                  )}
                </div>
                <div className="text-xs text-gray-400">
                  {formatTimeAgo(alert.timestamp)}
                </div>
              </div>

              {/* Message */}
              <div className={`text-sm font-medium ${getTextColor(alert.type)} mb-1`}>
                {alert.message}
              </div>

              {/* Details */}
              {alert.details && (
                <div className="text-xs text-gray-400">
                  {alert.details}
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default AlertsPanel;