import { useEffect, useCallback, useRef } from 'react';
import { webSocketService } from '@/services/websocket';
import { useWebSocketStore } from '@/stores/useWebSocketStore';
import type { WebSocketMessage, WebSocketStatus } from '@/types';

interface UseWebSocketOptions {
  autoConnect?: boolean;
  onMessage?: (message: WebSocketMessage) => void;
  onStatusChange?: (status: WebSocketStatus) => void;
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const { autoConnect = false, onMessage, onStatusChange } = options;
  const { status, connect, disconnect, sendMessage } = useWebSocketStore();
  const cleanupRef = useRef<(() => void)[]>([]);
  const isMountedRef = useRef(true);
  
  // Configurer les handlers
  useEffect(() => {
    const cleanups: (() => void)[] = [];
    
    if (onMessage) {
      const cleanup = webSocketService.onMessage(onMessage);
      cleanups.push(cleanup);
    }
    
    if (onStatusChange) {
      const cleanup = webSocketService.onStatusChange(onStatusChange);
      cleanups.push(cleanup);
    }
    
    cleanupRef.current = cleanups;
    
    return () => {
      cleanups.forEach(cleanup => cleanup());
    };
  }, [onMessage, onStatusChange]);
  
  // Connexion automatique
  useEffect(() => {
    if (autoConnect && status === 'disconnected') {
      connect();
    }
    
    return () => {
      // Éviter la déconnexion sur démontage pour éviter React #185
      // La déconnexion se fera automatiquement à la fermeture de la page
    };
  }, [autoConnect, connect, disconnect, status]);
  
  // Cleanup lors du démontage
  useEffect(() => {
    return () => {
      isMountedRef.current = false;
    };
  }, []);
  
  const subscribe = useCallback((channel: string) => {
    webSocketService.subscribe(channel);
  }, []);
  
  const unsubscribe = useCallback((channel: string) => {
    webSocketService.unsubscribe(channel);
  }, []);
  
  const send = useCallback((message: any) => {
    sendMessage(message);
  }, [sendMessage]);
  
  return {
    status,
    connect,
    disconnect,
    subscribe,
    unsubscribe,
    send,
    isConnected: status === 'connected',
    isConnecting: status === 'connecting',
    isDisconnected: status === 'disconnected',
    hasError: status === 'error',
  };
}

export function useWebSocketChannel(
  channel: string,
  onMessage?: (message: WebSocketMessage) => void,
  autoSubscribe: boolean = true
) {
  const { subscribe, unsubscribe, isConnected } = useWebSocket({ onMessage });
  
  useEffect(() => {
    if (autoSubscribe && isConnected) {
      subscribe(channel);
      
      return () => {
        unsubscribe(channel);
      };
    }
  }, [channel, subscribe, unsubscribe, isConnected, autoSubscribe]);
  
  return {
    subscribe: () => subscribe(channel),
    unsubscribe: () => unsubscribe(channel),
  };
}

export function useMarketDataWebSocket(symbol: string, interval: string = '1m') {
  const channel = `market:${symbol}:${interval}`;
  const { subscribe, unsubscribe, isConnected } = useWebSocket();
  
  useEffect(() => {
    if (isConnected) {
      subscribe(channel);
      
      return () => {
        unsubscribe(channel);
      };
    }
  }, [channel, subscribe, unsubscribe, isConnected]);
  
  return {
    subscribe: () => subscribe(channel),
    unsubscribe: () => unsubscribe(channel),
  };
}

export default useWebSocket;