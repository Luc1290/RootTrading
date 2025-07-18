import type { WebSocketMessage, WebSocketStatus } from '@/types';

type MessageHandler = (message: WebSocketMessage) => void;
type StatusHandler = (status: WebSocketStatus) => void;

class WebSocketService {
  private ws: WebSocket | null = null;
  private clientId: string;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 3000;
  private messageHandlers: MessageHandler[] = [];
  private statusHandlers: StatusHandler[] = [];
  private subscribedChannels = new Set<string>();
  private status: WebSocketStatus = 'disconnected';
  
  constructor() {
    this.clientId = this.generateClientId();
  }
  
  private generateClientId(): string {
    return 'web_' + Math.random().toString(36).substr(2, 9);
  }
  
  private notifyStatusChange(status: WebSocketStatus): void {
    this.status = status;
    this.statusHandlers.forEach(handler => handler(status));
  }
  
  private notifyMessage(message: WebSocketMessage): void {
    this.messageHandlers.forEach(handler => handler(message));
  }
  
  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      return;
    }
    
    this.reconnectAttempts++;
    console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
    
    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, this.reconnectDelay);
  }
  
  private setupWebSocket(): void {
    if (!this.ws) return;
    
    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.notifyStatusChange('connected');
      this.reconnectAttempts = 0;
      
      // Ré-s'abonner aux canaux
      this.subscribedChannels.forEach(channel => {
        this.sendMessage({ action: 'subscribe', channel });
      });
    };
    
    this.ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        this.notifyMessage(message);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };
    
    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
      this.ws = null;
      this.notifyStatusChange('disconnected');
      
      // Tentative de reconnexion automatique
      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        this.attemptReconnect();
      }
    };
    
    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.notifyStatusChange('error');
    };
  }
  
  connect(): void {
    if (this.ws && this.status === 'connected') {
      console.log('WebSocket already connected');
      return;
    }
    
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    
    this.notifyStatusChange('connecting');
    
    try {
      this.ws = new WebSocket(`ws://localhost:5009/ws/charts/${this.clientId}`);
      this.setupWebSocket();
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      this.notifyStatusChange('error');
    }
  }
  
  disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    
    this.reconnectAttempts = this.maxReconnectAttempts; // Empêcher la reconnexion auto
    
    if (this.ws) {
      this.ws.close();
    }
    
    this.notifyStatusChange('disconnected');
  }
  
  sendMessage(message: any): void {
    if (this.ws && this.status === 'connected') {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected, cannot send message');
    }
  }
  
  subscribe(channel: string): void {
    this.subscribedChannels.add(channel);
    
    // Envoyer le message seulement si connecté, sinon il sera envoyé lors de la connexion
    if (this.status === 'connected') {
      this.sendMessage({ action: 'subscribe', channel });
    }
  }
  
  unsubscribe(channel: string): void {
    this.subscribedChannels.delete(channel);
    
    // Envoyer le message seulement si connecté
    if (this.status === 'connected') {
      this.sendMessage({ action: 'unsubscribe', channel });
    }
  }
  
  onMessage(handler: MessageHandler): () => void {
    this.messageHandlers.push(handler);
    
    // Retourner une fonction de nettoyage
    return () => {
      this.messageHandlers = this.messageHandlers.filter(h => h !== handler);
    };
  }
  
  onStatusChange(handler: StatusHandler): () => void {
    this.statusHandlers.push(handler);
    
    // Retourner une fonction de nettoyage
    return () => {
      this.statusHandlers = this.statusHandlers.filter(h => h !== handler);
    };
  }
  
  getStatus(): WebSocketStatus {
    return this.status;
  }
  
  isConnected(): boolean {
    return this.status === 'connected';
  }
  
  getClientId(): string {
    return this.clientId;
  }
  
  getSubscribedChannels(): Set<string> {
    return new Set(this.subscribedChannels);
  }
}

export const webSocketService = new WebSocketService();
export default webSocketService;