import { create } from 'zustand';
import type { WebSocketStatus, WebSocketMessage } from '@/types';

interface WebSocketStore {
  // État de connexion
  status: WebSocketStatus;
  ws: WebSocket | null;
  clientId: string;
  
  // Actions
  connect: () => void;
  disconnect: () => void;
  sendMessage: (message: any) => void;
  
  // Callbacks
  onMessage: (callback: (message: WebSocketMessage) => void) => void;
  onStatusChange: (callback: (status: WebSocketStatus) => void) => void;
  
  // Internes
  setStatus: (status: WebSocketStatus) => void;
  setWs: (ws: WebSocket | null) => void;
  
  // Gestion des canaux
  subscribe: (channel: string) => void;
  unsubscribe: (channel: string) => void;
  subscribedChannels: Set<string>;
}

const generateClientId = () => {
  return 'web_' + Math.random().toString(36).substr(2, 9);
};

export const useWebSocketStore = create<WebSocketStore>()((set, get) => {
  let messageCallbacks: ((message: WebSocketMessage) => void)[] = [];
  let statusCallbacks: ((status: WebSocketStatus) => void)[] = [];
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let reconnectAttempts = 0;
  const maxReconnectAttempts = 5;
  const reconnectDelay = 3000;
  
  const notifyStatusChange = (status: WebSocketStatus) => {
    statusCallbacks.forEach(callback => callback(status));
  };
  
  const notifyMessage = (message: WebSocketMessage) => {
    messageCallbacks.forEach(callback => callback(message));
  };
  
  const attemptReconnect = () => {
    if (reconnectAttempts >= maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }
    
    reconnectAttempts++;
    console.log(`Attempting to reconnect (${reconnectAttempts}/${maxReconnectAttempts})...`);
    
    reconnectTimer = setTimeout(() => {
      const { connect } = get();
      connect();
    }, reconnectDelay);
  };
  
  const createWebSocket = (): WebSocket => {
    const { clientId } = get();
    const ws = new WebSocket(`ws://localhost:5009/ws/charts/${clientId}`);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      set({ status: 'connected' });
      notifyStatusChange('connected');
      reconnectAttempts = 0; // Reset sur connexion réussie
      
      // Ré-s'abonner aux canaux
      const { subscribedChannels } = get();
      subscribedChannels.forEach(channel => {
        ws.send(JSON.stringify({
          action: 'subscribe',
          channel
        }));
      });
    };
    
    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        notifyMessage(message);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      set({ status: 'disconnected', ws: null });
      notifyStatusChange('disconnected');
      
      // Tentative de reconnexion automatique
      if (reconnectAttempts < maxReconnectAttempts) {
        attemptReconnect();
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      set({ status: 'error' });
      notifyStatusChange('error');
    };
    
    return ws;
  };
  
  return {
    status: 'disconnected',
    ws: null,
    clientId: generateClientId(),
    subscribedChannels: new Set(),
    
    connect: () => {
      const { ws, status } = get();
      
      // Eviter les multiples connexions
      if ((ws && status === 'connected') || status === 'connecting') {
        return;
      }
      
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      
      // Fermer la connexion existante si elle existe
      if (ws) {
        try {
          ws.close();
        } catch (e) {
          console.warn('Error closing existing WebSocket:', e);
        }
      }
      
      set({ status: 'connecting' });
      notifyStatusChange('connecting');
      
      try {
        const newWs = createWebSocket();
        set({ ws: newWs });
      } catch (error) {
        console.error('Error creating WebSocket:', error);
        set({ status: 'error' });
        notifyStatusChange('error');
      }
    },
    
    disconnect: () => {
      const { ws } = get();
      
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
      
      reconnectAttempts = maxReconnectAttempts; // Empêcher la reconnexion auto
      
      if (ws) {
        ws.close();
      }
      
      set({ status: 'disconnected', ws: null });
      notifyStatusChange('disconnected');
    },
    
    sendMessage: (message) => {
      const { ws, status } = get();
      
      if (ws && status === 'connected') {
        ws.send(JSON.stringify(message));
      } else {
        console.warn('WebSocket not connected, cannot send message');
      }
    },
    
    subscribe: (channel) => {
      const { subscribedChannels, sendMessage } = get();
      
      subscribedChannels.add(channel);
      sendMessage({ action: 'subscribe', channel });
    },
    
    unsubscribe: (channel) => {
      const { subscribedChannels, sendMessage } = get();
      
      subscribedChannels.delete(channel);
      sendMessage({ action: 'unsubscribe', channel });
    },
    
    onMessage: (callback) => {
      messageCallbacks.push(callback);
      
      // Retourner une fonction de nettoyage
      return () => {
        messageCallbacks = messageCallbacks.filter(cb => cb !== callback);
      };
    },
    
    onStatusChange: (callback) => {
      statusCallbacks.push(callback);
      
      // Retourner une fonction de nettoyage
      return () => {
        statusCallbacks = statusCallbacks.filter(cb => cb !== callback);
      };
    },
    
    setStatus: (status) => set({ status }),
    setWs: (ws) => set({ ws }),
  };
});