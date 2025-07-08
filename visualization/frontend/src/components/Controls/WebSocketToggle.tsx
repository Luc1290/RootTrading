import React from 'react';
import { SignalIcon, SignalSlashIcon } from '@heroicons/react/24/outline';
import { useWebSocket } from '@/hooks/useWebSocket';
import { cn } from '@/utils';

function WebSocketToggle() {
  const { status, connect, disconnect, isConnected } = useWebSocket();
  
  const handleToggle = () => {
    if (isConnected) {
      disconnect();
    } else {
      connect();
    }
  };
  
  const getButtonText = () => {
    switch (status) {
      case 'connected':
        return 'Déconnecter';
      case 'connecting':
        return 'Connexion...';
      case 'error':
        return 'Erreur';
      default:
        return 'Connecter';
    }
  };
  
  const getButtonClass = () => {
    switch (status) {
      case 'connected':
        return 'btn-success';
      case 'connecting':
        return 'btn-secondary animate-pulse';
      case 'error':
        return 'btn-danger';
      default:
        return 'btn-secondary';
    }
  };
  
  return (
    <button
      onClick={handleToggle}
      disabled={status === 'connecting'}
      className={cn('btn px-4 py-2 flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed', getButtonClass())}
      title={`${isConnected ? 'Déconnecter' : 'Connecter'} WebSocket`}
    >
      {isConnected ? (
        <SignalIcon className="w-4 h-4" />
      ) : (
        <SignalSlashIcon className="w-4 h-4" />
      )}
      <span className="text-sm font-medium">{getButtonText()}</span>
    </button>
  );
}

export default WebSocketToggle;