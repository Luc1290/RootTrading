import asyncio
import json
import logging
from typing import Dict, Set, Optional, List, Any
from fastapi import WebSocket
from datetime import datetime, timedelta
from data_manager import DataManager

logger = logging.getLogger(__name__)

class WebSocketHub:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # client_id -> set of channels
        self.channel_clients: Dict[str, Set[str]] = {}  # channel -> set of client_ids
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
    async def start(self):
        """Start the WebSocket hub"""
        self._running = True
        logger.info("WebSocket hub started")
        
        # Subscribe to common Redis channels for broadcasting
        await self._subscribe_to_redis_channels()
        
    async def stop(self):
        """Stop the WebSocket hub"""
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
            
        # Close all connections
        for client_id in list(self.connections.keys()):
            await self.disconnect(client_id)
            
        logger.info("WebSocket hub stopped")
        
    async def connect(self, websocket: WebSocket, client_id: str):
        """Connect a new WebSocket client"""
        await websocket.accept()
        self.connections[client_id] = websocket
        self.subscriptions[client_id] = set()
        
        logger.info(f"Client {client_id} connected")
        
        # Send initial connection message
        await self._send_to_client(client_id, {
            "type": "connection",
            "status": "connected",
            "client_id": client_id,
            "timestamp": datetime.utcnow().isoformat() + 'Z'
        })
        
    async def disconnect(self, client_id: str):
        """Disconnect a WebSocket client"""
        if client_id in self.connections:
            # Remove from all channel subscriptions
            for channel in self.subscriptions.get(client_id, set()):
                if channel in self.channel_clients:
                    self.channel_clients[channel].discard(client_id)
                    
            # Close connection
            try:
                await self.connections[client_id].close()
            except Exception as e:
                logger.error(f"Error closing connection for {client_id}: {e}")
                
            # Clean up
            del self.connections[client_id]
            if client_id in self.subscriptions:
                del self.subscriptions[client_id]
                
            logger.info(f"Client {client_id} disconnected")
            
    async def subscribe_client(self, client_id: str, channel: str, params: Optional[Dict[Any, Any]] = None):
        """Subscribe a client to a channel"""
        if client_id not in self.connections:
            return
            
        # Add to subscriptions
        self.subscriptions[client_id].add(channel)
        
        if channel not in self.channel_clients:
            self.channel_clients[channel] = set()
        self.channel_clients[channel].add(client_id)
        
        logger.info(f"Client {client_id} subscribed to {channel}")
        
        # Send confirmation
        await self._send_to_client(client_id, {
            "type": "subscription",
            "action": "subscribed",
            "channel": channel,
            "timestamp": datetime.utcnow().isoformat() + 'Z'
        })
        
        # Start channel-specific updates if needed
        if channel.startswith("market:"):
            await self._start_market_updates(channel, params)
        elif channel.startswith("signals:"):
            await self._start_signal_updates(channel, params)
        elif channel.startswith("performance:"):
            await self._start_performance_updates(channel, params)
            
    async def unsubscribe_client(self, client_id: str, channel: str):
        """Unsubscribe a client from a channel"""
        if client_id not in self.connections:
            return
            
        # Remove from subscriptions
        self.subscriptions[client_id].discard(channel)
        
        if channel in self.channel_clients:
            self.channel_clients[channel].discard(client_id)
            
        logger.info(f"Client {client_id} unsubscribed from {channel}")
        
        # Send confirmation
        await self._send_to_client(client_id, {
            "type": "subscription",
            "action": "unsubscribed",
            "channel": channel,
            "timestamp": datetime.utcnow().isoformat() + 'Z'
        })
        
    async def broadcast_to_channel(self, channel: str, data: Dict):
        """Broadcast data to all clients subscribed to a channel"""
        if channel not in self.channel_clients:
            return
            
        clients = self.channel_clients[channel].copy()
        
        for client_id in clients:
            await self._send_to_client(client_id, {
                "type": "update",
                "channel": channel,
                "data": data,
                "timestamp": datetime.utcnow().isoformat() + 'Z'
            })
            
    async def _send_to_client(self, client_id: str, data: Dict):
        """Send data to a specific client"""
        if client_id not in self.connections:
            return
            
        try:
            await self.connections[client_id].send_json(data)
        except Exception as e:
            logger.error(f"Error sending to client {client_id}: {e}")
            await self.disconnect(client_id)
            
    async def _subscribe_to_redis_channels(self):
        """Subscribe to Redis channels for real-time updates"""
        # Subscribe to ticker updates
        async def ticker_callback(data):
            symbol = data.get("symbol")
            if symbol:
                await self.broadcast_to_channel(f"ticker:{symbol}", data)
                
        await self.data_manager.subscribe_to_channel("ticker:*", ticker_callback)
        
        # Subscribe to signal updates
        async def signal_callback(data):
            await self.broadcast_to_channel("signals:all", data)
            
        await self.data_manager.subscribe_to_channel("signals:aggregated", signal_callback)
        
    async def _start_market_updates(self, channel: str, params: Optional[Dict[Any, Any]] = None):
        """Start periodic market data updates for a channel"""
        parts = channel.split(":")
        if len(parts) < 3:
            return
            
        symbol = parts[1]
        interval = parts[2] if len(parts) > 2 else "1m"
        
        async def update_loop():
            while self._running and channel in self.channel_clients and self.channel_clients[channel]:
                try:
                    # Get latest candle
                    candles = await self.data_manager.get_market_data(
                        symbol, interval, limit=1
                    )
                    
                    if candles:
                        await self.broadcast_to_channel(channel, {
                            "symbol": symbol,
                            "interval": interval,
                            "candle": candles[0]
                        })
                        
                except Exception as e:
                    logger.error(f"Error in market update loop: {e}")
                    
                await asyncio.sleep(1)  # Update every second
                
        task = asyncio.create_task(update_loop())
        self._tasks.append(task)
        
    async def _start_signal_updates(self, channel: str, params: Optional[Dict[Any, Any]] = None):
        """Start periodic signal updates for a channel"""
        symbol = params.get("symbol") if params else None
        
        async def update_loop():
            while self._running and channel in self.channel_clients and self.channel_clients[channel]:
                try:
                    # Get latest signals
                    signals = await self.data_manager.get_trading_signals(
                        symbol or "BTCUSDT",
                        start_time=datetime.utcnow() - timedelta(minutes=5)
                    )
                    
                    if signals:
                        await self.broadcast_to_channel(channel, {
                            "signals": signals[-10:]  # Last 10 signals
                        })
                        
                except Exception as e:
                    logger.error(f"Error in signal update loop: {e}")
                    
                await asyncio.sleep(5)  # Update every 5 seconds
                
        task = asyncio.create_task(update_loop())
        self._tasks.append(task)
        
    async def _start_performance_updates(self, channel: str, params: Optional[Dict[Any, Any]] = None):
        """Start periodic performance updates"""
        async def update_loop():
            while self._running and channel in self.channel_clients and self.channel_clients[channel]:
                try:
                    # Get performance data
                    perf_data = await self.data_manager.get_portfolio_performance("1h")
                    
                    if perf_data and perf_data.get("balances"):
                        latest_idx = -1
                        await self.broadcast_to_channel(channel, {
                            "balance": perf_data["balances"][latest_idx],
                            "pnl": perf_data["pnl"][latest_idx],
                            "pnl_percentage": perf_data["pnl_percentage"][latest_idx],
                            "win_rate": perf_data["win_rate"][latest_idx],
                            "timestamp": perf_data["timestamps"][latest_idx]
                        })
                        
                except Exception as e:
                    logger.error(f"Error in performance update loop: {e}")
                    
                await asyncio.sleep(10)  # Update every 10 seconds
                
        task = asyncio.create_task(update_loop())
        self._tasks.append(task)