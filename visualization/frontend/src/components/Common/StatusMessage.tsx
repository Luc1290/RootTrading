import React, { useState, useEffect } from 'react';
import { XMarkIcon } from '@heroicons/react/24/outline';
import { cn } from '@/utils';

interface StatusMessageProps {
  message?: string;
  type?: 'success' | 'error' | 'warning' | 'info';
  duration?: number;
  onClose?: () => void;
}

function StatusMessage({ message, type = 'info', duration = 5000, onClose }: StatusMessageProps) {
  const [isVisible, setIsVisible] = useState(!!message);
  const [currentMessage, setCurrentMessage] = useState(message);
  
  useEffect(() => {
    if (message) {
      setCurrentMessage(message);
      setIsVisible(true);
      
      const timer = setTimeout(() => {
        setIsVisible(false);
        onClose?.();
      }, duration);
      
      return () => clearTimeout(timer);
    }
  }, [message, duration, onClose]);
  
  const handleClose = () => {
    setIsVisible(false);
    onClose?.();
  };
  
  if (!isVisible || !currentMessage) {
    return null;
  }
  
  const getStatusClass = () => {
    switch (type) {
      case 'success':
        return 'status-success';
      case 'error':
        return 'status-error';
      case 'warning':
        return 'status-warning';
      default:
        return 'status-info';
    }
  };
  
  return (
    <div className={cn('flex items-center justify-between p-3 rounded-lg', getStatusClass())}>
      <span className="text-sm font-medium">{currentMessage}</span>
      <button
        onClick={handleClose}
        className="ml-4 p-1 hover:bg-black/10 rounded transition-colors"
        title="Fermer"
      >
        <XMarkIcon className="w-4 h-4" />
      </button>
    </div>
  );
}

// Hook pour gérer les messages de statut globalement
let statusMessageSetter: ((message: string, type?: 'success' | 'error' | 'warning' | 'info') => void) | null = null;

export function useStatusMessage() {
  const [message, setMessage] = useState<string | null>(null);
  const [type, setType] = useState<'success' | 'error' | 'warning' | 'info'>('info');
  
  const showMessage = (msg: string, msgType: 'success' | 'error' | 'warning' | 'info' = 'info') => {
    setMessage(msg);
    setType(msgType);
  };
  
  const clearMessage = () => {
    setMessage(null);
  };
  
  // Exposer la fonction globalement
  statusMessageSetter = showMessage;
  
  return {
    message,
    type,
    showMessage,
    clearMessage,
  };
}

// Fonction utilitaire pour afficher des messages depuis n'importe où
export function showStatusMessage(message: string, type: 'success' | 'error' | 'warning' | 'info' = 'info') {
  statusMessageSetter?.(message, type);
}

// Composant wrapper qui utilise le hook
export function GlobalStatusMessage() {
  const { message, type, clearMessage } = useStatusMessage();
  
  return (
    <StatusMessage
      message={message || undefined}
      type={type}
      onClose={clearMessage}
    />
  );
}

export default GlobalStatusMessage;