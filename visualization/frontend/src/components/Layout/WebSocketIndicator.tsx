import React from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { cn } from '@/utils';

function WebSocketIndicator() {
  const { status, connect, disconnect, isConnected } = useWebSocket();
  
  const getStatusConfig = () => {
    switch (status) {
      case 'connected':
        return {
          icon: '🟢',
          text: 'Connecté',
          className: 'status-success',
        };
      case 'connecting':
        return {
          icon: '🟡',
          text: 'Connexion...',
          className: 'status-warning animate-pulse',
        };
      case 'error':
        return {
          icon: '❌',
          text: 'Erreur',
          className: 'status-error',
        };
      default:
        return {
          icon: '🔴',
          text: 'Déconnecté',
          className: 'status-error',
        };
    }
  };
  
  const statusConfig = getStatusConfig();
  
  const handleToggle = () => {
    if (isConnected) {
      disconnect();
    } else {
      connect();
    }
  };
  
  return (
    <div className="fixed top-4 right-4 z-50 flex items-center space-x-2">
      <div className={cn('flex items-center space-x-1', statusConfig.className)}>
        <span className="text-sm">{statusConfig.icon}</span>
        <span className="text-sm font-medium">{statusConfig.text}</span>
      </div>
      
      <button
        onClick={handleToggle}
        className="btn btn-secondary text-xs px-2 py-1 rounded-md hover:bg-gray-600 transition-colors"
        title={isConnected ? 'Déconnecter WebSocket' : 'Connecter WebSocket'}
      >
        📡 {isConnected ? 'Déconnecter' : 'Connecter'}
      </button>
    </div>
  );
}

export default WebSocketIndicator;