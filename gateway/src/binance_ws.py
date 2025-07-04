"""
Module de connexion WebSocket à Binance.
Reçoit les données de marché en temps réel et les transmet au Kafka producer.
"""
import json
import logging
import time
from typing import Dict, Any, Callable, List, Optional, Tuple
import asyncio
import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatusCode

# Importer les clients partagés
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import BINANCE_API_KEY, BINANCE_SECRET_KEY, SYMBOLS, INTERVAL, KAFKA_TOPIC_MARKET_DATA
from shared.src.kafka_client import KafkaClient
from gateway.src.kafka_producer import get_producer   # NEW


# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("binance_ws")

class BinanceWebSocket:
    """
    Gestionnaire de connexion WebSocket à Binance pour les données de marché.
    Récupère les données en temps réel via WebSocket et les transmet via Kafka.
    """
    
    def __init__(self, symbols: List[str] = None, interval: str = INTERVAL, kafka_client: KafkaClient = None):
        """
        Initialise la connexion WebSocket Binance.
        
        Args:
            symbols: Liste des symboles à surveiller (ex: ['BTCUSDC', 'ETHUSDC'])
            interval: Intervalle des chandeliers (ex: '1m', '5m', '1h')
            kafka_client: Client Kafka pour la publication des données
        """
        self.symbols = symbols or SYMBOLS
        self.interval = interval
        self.kafka_client = kafka_client or get_producer()   # utilise le producteur singleton
        self.ws = None
        self.running = False
        self.reconnect_delay = 1  # Secondes, pour backoff exponentiel
        self.last_message_time = 0
        self.heartbeat_interval = 60  # Secondes
        
        # URL de l'API WebSocket Binance
        self.base_url = "wss://stream.binance.com:9443/ws"
        
        # Multi-timeframes et données enrichies pour trading précis
        self.timeframes = ['1m', '5m', '15m', '1h', '4h']  # Timeframes multiples
        self.stream_paths = []
        
        for symbol in self.symbols:
            symbol_lower = symbol.lower()
            # Klines multi-timeframes
            for tf in self.timeframes:
                self.stream_paths.append(f"{symbol_lower}@kline_{tf}")
            # Ticker 24h pour contexte
            self.stream_paths.append(f"{symbol_lower}@ticker")
            # Orderbook pour spread
            self.stream_paths.append(f"{symbol_lower}@depth5")
            
        # Cache pour données enrichies
        self.ticker_cache = {}
        self.orderbook_cache = {}
        
        # ULTRA-AVANCÉ : Buffers pour calculs techniques en temps réel
        self.price_buffers = {}
        self.volume_buffers = {}
        self.high_buffers = {}
        self.low_buffers = {}
        self.rsi_buffers = {}
        self.macd_buffers = {}
        
        # Initialiser les buffers pour chaque symbole/timeframe
        for symbol in self.symbols:
            self.price_buffers[symbol] = {}
            self.volume_buffers[symbol] = {}
            self.high_buffers[symbol] = {}
            self.low_buffers[symbol] = {}
            self.rsi_buffers[symbol] = {}
            self.macd_buffers[symbol] = {}
            for tf in self.timeframes:
                self.price_buffers[symbol][tf] = []
                self.volume_buffers[symbol][tf] = []
                self.high_buffers[symbol][tf] = []
                self.low_buffers[symbol][tf] = []
                self.rsi_buffers[symbol][tf] = []
                self.macd_buffers[symbol][tf] = {'ema12': None, 'ema26': None, 'signal9': None}
                
        logger.info(f"🔥 Gateway ULTRA-AVANCÉ : {len(self.stream_paths)} streams + indicateurs temps réel")
    
    async def _connect(self) -> None:
        """
        Établit la connexion WebSocket avec Binance et souscrit aux streams.
        """
        uri = self.base_url
        
        try:
            logger.info(f"Connexion à Binance WebSocket: {uri}")
            self.ws = await websockets.connect(uri)
            
            # S'abonner aux streams
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": self.stream_paths,
                "id": int(time.time() * 1000)
            }
            
            await self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Abonnement envoyé pour: {', '.join(self.stream_paths)}")
            
            # Attendre la confirmation d'abonnement
            response = await self.ws.recv()
            response_data = json.loads(response)
            
            if 'result' in response_data and response_data['result'] is None:
                logger.info("✅ Abonnement aux streams Binance confirmé")
            else:
                logger.warning(f"Réponse d'abonnement inattendue: {response_data}")
            
            # Réinitialiser le délai de reconnexion
            self.reconnect_delay = 1
            
        except (ConnectionClosed, InvalidStatusCode) as e:
            logger.error(f"❌ Erreur lors de la connexion WebSocket: {str(e)}")
            await self._handle_reconnect()
        except Exception as e:
            logger.error(f"❌ Erreur inattendue lors de la connexion: {str(e)}")
            await self._handle_reconnect()
    
    async def _handle_reconnect(self) -> None:
        """
        Gère la reconnexion en cas de perte de connexion.
        Utilise un backoff exponentiel pour éviter de surcharger le serveur.
        """
        # Vérifier si le service est encore en cours d'exécution
        if not self.running:
            return
    
        # Limiter le nombre de tentatives
        max_retries = 20
        retry_count = 0
    
        while self.running and retry_count < max_retries:
            logger.warning(f"Tentative de reconnexion {retry_count+1}/{max_retries} dans {self.reconnect_delay} secondes...")
            await asyncio.sleep(self.reconnect_delay)
        
            # Backoff exponentiel avec plafond plus élevé
            self.reconnect_delay = min(300, self.reconnect_delay * 2)  # 5 minutes max
        
            try:
                # Fermer proprement la connexion existante si elle existe
                if self.ws:
                    try:
                        await self.ws.close()
                    except Exception as e:
                        logger.warning(f"Erreur lors de la fermeture de la connexion existante: {str(e)}")
                
                # Établir une nouvelle connexion
                await self._connect()
                return  # Reconnexion réussie
            except Exception as e:
                retry_count += 1
                logger.error(f"❌ Échec de reconnexion: {str(e)}")
    
        logger.critical("Impossible de se reconnecter après plusieurs tentatives. Arrêt du service.")
        # Signaler l'arrêt au service principal
        self.running = False
    
    async def _heartbeat_check(self) -> None:
        """
        Vérifie régulièrement si nous recevons toujours des messages.
        Reconnecte si aucun message n'a été reçu depuis trop longtemps.
        """
        while self.running:
            await asyncio.sleep(self.heartbeat_interval)
            
            # Si aucun message reçu depuis 2x l'intervalle de heartbeat
            if time.time() - self.last_message_time > self.heartbeat_interval * 2:
                logger.warning(f"❗ Aucun message reçu depuis {self.heartbeat_interval * 2} secondes. Reconnexion...")
                
                # Fermer la connexion existante
                if self.ws:
                    try:
                        await self.ws.close()
                    except Exception as e:
                        logger.warning(f"Erreur lors de la fermeture de la connexion: {str(e)}")
                
                # Reconnecter
                await self._connect()
    
    def _process_kline_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Traite un message de chandelier Binance avec données enrichies.
        
        Args:
            message: Message brut de Binance
            
        Returns:
            Message formaté enrichi pour le système RootTrading
        """
        # Extraire les données pertinentes
        kline = message['k']
        symbol = message['s']
        
        # Créer un message enrichi
        processed_data = {
            # Données de base
            "symbol": symbol,
            "timeframe": kline['i'],
            "start_time": kline['t'],
            "close_time": kline['T'],
            "open": float(kline['o']),
            "high": float(kline['h']),
            "low": float(kline['l']),
            "close": float(kline['c']),
            "volume": float(kline['v']),
            "quote_volume": float(kline['q']),
            "trade_count": int(kline['n']),
            "taker_buy_volume": float(kline['V']),
            "taker_buy_quote_volume": float(kline['Q']),
            "is_closed": kline['x'],
            
            # Données enrichies si disponibles
            "enhanced": True
        }
        
        # Ajouter les données de ticker si disponibles
        if symbol in self.ticker_cache:
            ticker = self.ticker_cache[symbol]
            processed_data.update({
                "price_change_24h": ticker.get('priceChange', 0),
                "price_change_pct_24h": ticker.get('priceChangePercent', 0),
                "volume_24h": ticker.get('volume', 0),
                "high_24h": ticker.get('highPrice', 0),
                "low_24h": ticker.get('lowPrice', 0)
            })
            
        # Ajouter les données d'orderbook enrichies si disponibles
        if symbol in self.orderbook_cache:
            book = self.orderbook_cache[symbol]
            bids = book.get('bids', [])
            asks = book.get('asks', [])
            sentiment_analysis = book.get('sentiment_analysis', {})
            
            if bids and asks:
                bid_price = float(bids[0][0])
                ask_price = float(asks[0][0])
                mid_price = (bid_price + ask_price) / 2
                spread_pct = ((ask_price - bid_price) / mid_price) * 100
                
                processed_data.update({
                    "bid_price": bid_price,
                    "ask_price": ask_price,
                    "bid_qty": float(bids[0][1]),
                    "ask_qty": float(asks[0][1]),
                    "spread_pct": spread_pct,
                    
                    # ULTRA-AVANCÉ : Sentiment analysis complete
                    "orderbook_sentiment": sentiment_analysis
                })
        
        # ULTRA-AVANCÉ : Calculer tous les indicateurs techniques en temps réel
        if processed_data['is_closed']:
            self._update_technical_indicators(symbol, processed_data)
        
        return processed_data
        
    def _update_technical_indicators(self, symbol: str, candle_data: Dict) -> None:
        """ULTRA-AVANCÉ : Met à jour tous les indicateurs techniques en temps réel"""
        timeframe = candle_data['timeframe']
        close_price = candle_data['close']
        high_price = candle_data['high']
        low_price = candle_data['low']
        volume = candle_data['volume']
        
        # Ajouter aux buffers (garder 200 dernières valeurs max)
        if len(self.price_buffers[symbol][timeframe]) >= 200:
            self.price_buffers[symbol][timeframe].pop(0)
        self.price_buffers[symbol][timeframe].append(close_price)
        
        if len(self.high_buffers[symbol][timeframe]) >= 200:
            self.high_buffers[symbol][timeframe].pop(0)
        self.high_buffers[symbol][timeframe].append(high_price)
        
        if len(self.low_buffers[symbol][timeframe]) >= 200:
            self.low_buffers[symbol][timeframe].pop(0)
        self.low_buffers[symbol][timeframe].append(low_price)
        
        if len(self.volume_buffers[symbol][timeframe]) >= 200:
            self.volume_buffers[symbol][timeframe].pop(0)
        self.volume_buffers[symbol][timeframe].append(volume)
        
        prices = self.price_buffers[symbol][timeframe]
        highs = self.high_buffers[symbol][timeframe]
        lows = self.low_buffers[symbol][timeframe]
        volumes = self.volume_buffers[symbol][timeframe]
        
        # Utiliser calculate_all_indicators pour calculer TOUS les indicateurs d'un coup
        if len(prices) >= 20 and len(highs) >= 20 and len(lows) >= 20 and len(volumes) >= 20:
            logger.debug(f"📊 Calcul de tous les indicateurs pour {symbol} {timeframe}")
            
            # Utiliser la méthode qui calcule tout
            from shared.src.technical_indicators import indicators
            all_indicators = indicators.calculate_all_indicators(highs, lows, prices, volumes)
            
            # Ajouter tous les indicateurs calculés
            for indicator_name, value in all_indicators.items():
                if value is not None:
                    candle_data[indicator_name] = value
            
            logger.debug(f"✅ {len(all_indicators)} indicateurs calculés pour {symbol}")
            
            # Calculer aussi les indicateurs custom non inclus dans calculate_all_indicators
            # Stochastic RSI (pas dans calculate_all_indicators)
            stoch_rsi = self._calculate_stoch_rsi(prices, 14)
            if stoch_rsi:
                candle_data['stoch_rsi'] = stoch_rsi
                
            # VWAP 10 (custom, pas dans calculate_all_indicators)
            if len(volumes) >= 10:
                vwap = self._calculate_vwap(prices[-10:], volumes[-10:])
                if vwap:
                    candle_data['vwap_10'] = vwap
                    
            # Williams %R (custom implementation)
            williams_r = self._calculate_williams_r(symbol, timeframe, 14)
            if williams_r:
                candle_data['williams_r'] = williams_r
                
            # CCI 20 (custom implementation)
            cci = self._calculate_cci(symbol, timeframe, 20)
            if cci:
                candle_data['cci_20'] = cci
                
            # Marquer les données comme enrichies
            candle_data['enhanced'] = True
            candle_data['ultra_enriched'] = True
            
        elif len(prices) >= 14:
            # Calcul partiel pour avoir au moins RSI et quelques indicateurs de base
            logger.debug(f"📊 Calcul partiel des indicateurs pour {symbol} {timeframe}")
            
            # RSI seul si pas assez de données pour tous les indicateurs
            rsi = self._calculate_rsi(prices, 14)
            if rsi:
                candle_data['rsi_14'] = rsi
                
            # Marquer comme partiellement enrichi
            candle_data['enhanced'] = True
            candle_data['ultra_enriched'] = False
                
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Calcule le RSI via le module centralisé"""
        from shared.src.technical_indicators import calculate_rsi
        return calculate_rsi(prices, period)
        
    def _calculate_stoch_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Calcule le Stochastic RSI"""
        if len(prices) < period * 2:
            return None
            
        # Calculer RSI sur les dernières périodes
        rsi_values = []
        for i in range(period, len(prices) + 1):
            rsi = self._calculate_rsi(prices[:i], period)
            if rsi:
                rsi_values.append(rsi)
                
        if len(rsi_values) < period:
            return None
            
        recent_rsi = rsi_values[-period:]
        min_rsi = min(recent_rsi)
        max_rsi = max(recent_rsi)
        current_rsi = rsi_values[-1]
        
        if max_rsi == min_rsi:
            return 50
            
        return ((current_rsi - min_rsi) / (max_rsi - min_rsi)) * 100
        
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calcule l'EMA via le module centralisé"""
        from shared.src.technical_indicators import calculate_ema
        result = calculate_ema(prices, period)
        return result if result is not None else (prices[-1] if prices else 0)
        
    def _calculate_macd(self, prices: List[float]) -> Optional[Dict]:
        """Calcule MACD complet via le module centralisé"""
        from shared.src.technical_indicators import calculate_macd
        
        result = calculate_macd(prices)
        if result is None or any(v is None for v in result.values()):
            return None
            
        return {
            'macd_line': result['macd_line'],
            'macd_signal': result['macd_signal'],
            'macd_histogram': result['macd_histogram']
        }
        
    def _calculate_bollinger_bands(self, prices: List[float], period: int, std_dev: float) -> Optional[Dict]:
        """Calcule les Bollinger Bands via le module centralisé"""
        from shared.src.technical_indicators import calculate_bollinger_bands
        
        result = calculate_bollinger_bands(prices, period, std_dev)
        if result is None or any(v is None for v in result.values()):
            return None
            
        return {
            'bb_upper': result['bb_upper'],
            'bb_middle': result['bb_middle'], 
            'bb_lower': result['bb_lower'],
            'bb_position': result['bb_position'],
            'bb_width': result['bb_width']
        }
        
    def _calculate_vwap(self, prices: List[float], volumes: List[float]) -> Optional[float]:
        """Calcule le VWAP (Volume Weighted Average Price)"""
        if len(prices) != len(volumes) or not prices:
            return None
            
        total_pv = sum(p * v for p, v in zip(prices, volumes))
        total_volume = sum(volumes)
        
        return total_pv / total_volume if total_volume > 0 else None
        
    def _analyze_volume_profile(self, volumes: List[float]) -> Dict:
        """Analyse avancée du profil de volume"""
        if len(volumes) < 5:
            return {}
            
        recent_vol = volumes[-1]
        avg_vol = sum(volumes) / len(volumes)
        
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1
        
        # Détection de pic de volume
        vol_spike = vol_ratio > 2.0
        
        # Tendance du volume
        if len(volumes) >= 3:
            vol_trend = "increasing" if volumes[-1] > volumes[-2] > volumes[-3] else "decreasing"
        else:
            vol_trend = "stable"
            
        return {
            'volume_ratio': vol_ratio,
            'volume_spike': vol_spike,
            'volume_trend': vol_trend,
            'avg_volume': avg_vol
        }
        
    def _calculate_momentum(self, prices: List[float], period: int) -> Optional[float]:
        """Calcule le momentum"""
        if len(prices) < period + 1:
            return None
            
        return ((prices[-1] - prices[-period-1]) / prices[-period-1]) * 100
        
    def _calculate_atr(self, symbol: str, timeframe: str, period: int) -> Optional[float]:
        """Calcule l'ATR (Average True Range)"""
        # Pour l'ATR, on a besoin des high/low précédents, simplifié ici
        prices = self.price_buffers[symbol][timeframe]
        if len(prices) < period + 1:
            return None
            
        # Approximation : utiliser les variations de prix comme proxy
        ranges = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        if len(ranges) < period:
            return None
            
        return sum(ranges[-period:]) / period
        
    def _calculate_adx(self, symbol: str, timeframe: str, period: int) -> Optional[float]:
        """Calcule l'ADX en utilisant le module partagé"""
        if symbol not in self.high_buffers or symbol not in self.low_buffers:
            return None
            
        if timeframe not in self.high_buffers[symbol] or timeframe not in self.low_buffers[symbol]:
            return None
            
        highs = self.high_buffers[symbol][timeframe]
        lows = self.low_buffers[symbol][timeframe]
        closes = self.price_buffers[symbol][timeframe]
        
        if len(highs) < period + 1 or len(lows) < period + 1 or len(closes) < period + 1:
            return None
        
        try:
            # Utiliser le module partagé pour le calcul ADX
            from shared.src.technical_indicators import TechnicalIndicators
            indicators = TechnicalIndicators()
            
            adx, plus_di, minus_di = indicators.calculate_adx(highs, lows, closes, period)
            return adx  # Retourner seulement la valeur ADX
            
        except Exception as e:
            logger.debug(f"Erreur calcul ADX pour {symbol}: {e}")
            return None
        
    def _calculate_williams_r(self, symbol: str, timeframe: str, period: int) -> Optional[float]:
        """Calcule Williams %R"""
        prices = self.price_buffers[symbol][timeframe]
        if len(prices) < period:
            return None
            
        recent_prices = prices[-period:]
        highest = max(recent_prices)
        lowest = min(recent_prices)
        current = prices[-1]
        
        if highest == lowest:
            return -50
            
        return ((highest - current) / (highest - lowest)) * -100
        
    def _calculate_cci(self, symbol: str, timeframe: str, period: int) -> Optional[float]:
        """Calcule CCI (Commodity Channel Index)"""
        prices = self.price_buffers[symbol][timeframe]
        if len(prices) < period:
            return None
            
        recent_prices = prices[-period:]
        typical_price = sum(recent_prices) / len(recent_prices)
        sma = typical_price
        
        # Approximation de l'écart moyen
        mean_deviation = sum(abs(p - sma) for p in recent_prices) / len(recent_prices)
        
        if mean_deviation == 0:
            return 0
            
        return (prices[-1] - sma) / (0.015 * mean_deviation)
        
    def _process_ticker_message(self, data: Dict[str, Any]) -> None:
        """Traite les données ticker 24h"""
        symbol = data['s']
        self.ticker_cache[symbol] = {
            'priceChange': float(data.get('P', 0)),
            'priceChangePercent': float(data.get('p', 0)),
            'volume': float(data.get('v', 0)),
            'highPrice': float(data.get('h', 0)),
            'lowPrice': float(data.get('l', 0))
        }
        
    def _process_depth_message(self, data: Dict[str, Any]) -> None:
        """Traite les données orderbook avec analyse de sentiment avancée"""
        symbol = data['s']
        bids = data.get('b', [])
        asks = data.get('a', [])
        
        # Analyse avancée du sentiment via orderbook
        sentiment_analysis = self._analyze_orderbook_sentiment(bids, asks)
        
        self.orderbook_cache[symbol] = {
            'bids': bids,
            'asks': asks,
            'sentiment_analysis': sentiment_analysis,
            'timestamp': time.time()
        }
        
    def _analyze_orderbook_sentiment(self, bids: List, asks: List) -> Dict:
        """ULTRA-AVANCÉ : Analyse complète du sentiment via l'orderbook"""
        try:
            if not bids or not asks:
                return {}
                
            # Convertir en float pour calculs
            bid_data = [(float(price), float(qty)) for price, qty in bids[:20]]  # Top 20
            ask_data = [(float(price), float(qty)) for price, qty in asks[:20]]
            
            if not bid_data or not ask_data:
                return {}
                
            # 1. Ratio bid/ask volume
            total_bid_volume = sum(qty for _, qty in bid_data)
            total_ask_volume = sum(qty for _, qty in ask_data)
            
            bid_ask_ratio = total_bid_volume / total_ask_volume if total_ask_volume > 0 else 1
            
            # 2. Spread analysis
            best_bid = bid_data[0][0]
            best_ask = ask_data[0][0]
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
            spread_pct = (spread / mid_price) * 100
            
            # 3. Order book imbalance
            imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
            
            # 4. Depth analysis (support/resistance strength)
            bid_depth_strength = self._calculate_depth_strength(bid_data, best_bid, 'support')
            ask_depth_strength = self._calculate_depth_strength(ask_data, best_ask, 'resistance')
            
            # 5. Wall detection (gros ordres)
            bid_walls = self._detect_walls(bid_data, 'bid')
            ask_walls = self._detect_walls(ask_data, 'ask')
            
            # 6. Price level concentration
            bid_concentration = self._calculate_price_concentration(bid_data)
            ask_concentration = self._calculate_price_concentration(ask_data)
            
            # 7. Sentiment global
            sentiment_score = self._calculate_sentiment_score(
                bid_ask_ratio, imbalance, bid_depth_strength, ask_depth_strength,
                len(bid_walls), len(ask_walls)
            )
            
            # 8. Liquidité totale
            total_liquidity = total_bid_volume + total_ask_volume
            
            # 9. Ratio des 5 premiers niveaux vs le reste
            top5_bid_vol = sum(qty for _, qty in bid_data[:5])
            top5_ask_vol = sum(qty for _, qty in ask_data[:5])
            top5_ratio = (top5_bid_vol + top5_ask_vol) / total_liquidity if total_liquidity > 0 else 0
            
            return {
                'bid_ask_ratio': round(bid_ask_ratio, 3),
                'spread_pct': round(spread_pct, 4),
                'imbalance': round(imbalance, 3),
                'bid_depth_strength': round(bid_depth_strength, 3),
                'ask_depth_strength': round(ask_depth_strength, 3),
                'bid_walls': bid_walls,
                'ask_walls': ask_walls,
                'bid_concentration': round(bid_concentration, 3),
                'ask_concentration': round(ask_concentration, 3),
                'sentiment_score': round(sentiment_score, 3),
                'total_liquidity': round(total_liquidity, 2),
                'top5_concentration': round(top5_ratio, 3),
                'sentiment_signal': self._interpret_sentiment(sentiment_score, imbalance, bid_ask_ratio)
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse sentiment orderbook: {e}")
            return {}
            
    def _calculate_depth_strength(self, orders: List[Tuple], best_price: float, side: str) -> float:
        """Calcule la force de profondeur des ordres"""
        if not orders:
            return 0
            
        # Calculer la force basée sur le volume et la distance du meilleur prix
        total_strength = 0
        
        for price, qty in orders:
            # Distance relative du meilleur prix
            if side == 'support':
                distance = (best_price - price) / best_price
            else:  # resistance
                distance = (price - best_price) / best_price
                
            # Force = volume / (1 + distance) - plus près = plus fort
            strength = qty / (1 + distance * 100)  # *100 pour normaliser
            total_strength += strength
            
        return total_strength
        
    def _detect_walls(self, orders: List[Tuple], side: str) -> List[Dict]:
        """Détecte les murs d'ordres (gros volumes)"""
        if not orders:
            return []
            
        # Calculer le volume moyen
        volumes = [qty for _, qty in orders]
        avg_volume = sum(volumes) / len(volumes)
        
        # Détecter les volumes > 3x la moyenne
        walls = []
        for price, qty in orders:
            if qty > avg_volume * 3:
                walls.append({
                    'price': price,
                    'volume': qty,
                    'ratio_to_avg': qty / avg_volume,
                    'side': side
                })
                
        return sorted(walls, key=lambda x: x['volume'], reverse=True)[:5]  # Top 5
        
    def _calculate_price_concentration(self, orders: List[Tuple]) -> float:
        """Calcule la concentration des ordres par niveau de prix"""
        if len(orders) < 3:
            return 0
            
        # Calculer l'écart-type des volumes pour mesurer la concentration
        volumes = [qty for _, qty in orders]
        avg_vol = sum(volumes) / len(volumes)
        
        variance = sum((vol - avg_vol) ** 2 for vol in volumes) / len(volumes)
        std_dev = variance ** 0.5
        
        # Normaliser : plus l'écart-type est élevé, plus la concentration est inégale
        return std_dev / avg_vol if avg_vol > 0 else 0
        
    def _calculate_sentiment_score(self, bid_ask_ratio: float, imbalance: float, 
                                 bid_strength: float, ask_strength: float,
                                 bid_walls: int, ask_walls: int) -> float:
        """Calcule un score de sentiment global (-1 = très bearish, +1 = très bullish)"""
        
        # Composants du sentiment
        ratio_score = 0
        if bid_ask_ratio > 1.2:
            ratio_score = 0.3  # Plus de volume bid = bullish
        elif bid_ask_ratio < 0.8:
            ratio_score = -0.3  # Plus de volume ask = bearish
            
        # Imbalance score (directement utilisable entre -1 et 1)
        imbalance_score = imbalance * 0.4
        
        # Depth strength comparison
        if bid_strength > ask_strength * 1.2:
            depth_score = 0.2  # Support plus fort
        elif ask_strength > bid_strength * 1.2:
            depth_score = -0.2  # Résistance plus forte
        else:
            depth_score = 0
            
        # Wall advantage
        wall_score = 0
        if bid_walls > ask_walls:
            wall_score = 0.1  # Plus de murs bid
        elif ask_walls > bid_walls:
            wall_score = -0.1  # Plus de murs ask
            
        # Score final
        total_score = ratio_score + imbalance_score + depth_score + wall_score
        
        # Limiter entre -1 et 1
        return max(-1, min(1, total_score))
        
    def _interpret_sentiment(self, sentiment_score: float, imbalance: float, bid_ask_ratio: float) -> str:
        """Interprète le score de sentiment en signal actionnable"""
        
        if sentiment_score > 0.5:
            return "VERY_BULLISH"
        elif sentiment_score > 0.2:
            return "BULLISH"
        elif sentiment_score < -0.5:
            return "VERY_BEARISH"
        elif sentiment_score < -0.2:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    async def _listen(self) -> None:
        """
        Écoute les messages WebSocket et les traite.
        """
        while self.running:
            try:
                if not self.ws:
                    logger.warning("Connexion WebSocket perdue, reconnexion...")
                    await self._connect()
                    continue
                
                # Recevoir un message
                message = await self.ws.recv()
                self.last_message_time = time.time()
                
                # Traiter le message
                try:
                    data = json.loads(message)
                    
                    # Vérifier le type de message
                    if 'e' in data:
                        event_type = data['e']
                        
                        # Traiter les données de chandelier
                        if event_type == 'kline':
                            processed_data = self._process_kline_message(data)

                            # 👉 on ignore les bougies encore ouvertes
                            if not processed_data['is_closed']:
                                continue

                            symbol = processed_data['symbol']
                            timeframe = processed_data['timeframe']
                            
                            # Topic spécifique au timeframe
                            topic = f"market.data.{symbol.lower()}.{timeframe}"
                            key = f"{symbol}:{timeframe}:{processed_data['start_time']}"

                            # Nettoyer les données pour éviter les erreurs de format
                            clean_data = {}
                            for k, v in processed_data.items():
                                if isinstance(v, str):
                                    # Échapper les caractères de formatage problématiques
                                    clean_data[k] = v.replace('{', '{{').replace('}', '}}') if '{' in v or '}' in v else v
                                else:
                                    clean_data[k] = v
                            
                            # Log des clés pour débogage
                            logger.debug(f"Données envoyées à Kafka - Clés: {list(clean_data.keys())}")

                            # Publier avec la nouvelle méthode
                            self.kafka_client.publish_to_topic(
                                topic=topic,
                                data=clean_data,
                                key=key
                            )
        
                            # Log enrichi
                            spread_pct = processed_data.get('spread_pct', 0)
                            spread_info = f" Spread:{spread_pct:.3f}%" if isinstance(spread_pct, (int, float)) and 'spread_pct' in processed_data else ""
                            logger.info(f"📊 {symbol} {timeframe}: {processed_data['close']} "
                                      f"[O:{processed_data['open']} H:{processed_data['high']} L:{processed_data['low']}]"
                                      f"{spread_info}")
                                      
                        # Traiter les données ticker
                        elif event_type == '24hrTicker':
                            self._process_ticker_message(data)
                            
                        # Traiter les données orderbook
                        elif event_type == 'depthUpdate':
                            self._process_depth_message(data)
                except json.JSONDecodeError:
                    logger.error(f"Message non-JSON reçu: {message[:100]}...")
                except Exception as e:
                    error_msg = str(e).replace('{', '{{').replace('}', '}}')
                    logger.error(f"Erreur lors du traitement du message: {error_msg}")
                
            except ConnectionClosed as e:
                logger.warning(f"Connexion WebSocket fermée: {str(e)}")
                await self._handle_reconnect()
            except Exception as e:
                logger.error(f"Erreur durant l'écoute: {str(e)}")
                await asyncio.sleep(1)  # Pause pour éviter une boucle d'erreur infinie
    
    async def start(self) -> None:
        """
        Démarre la connexion WebSocket et l'écoute des messages.
        """
        if self.running:
            logger.warning("Le WebSocket est déjà en cours d'exécution.")
            return
            
        self.running = True
        self.last_message_time = time.time()
        
        try:
            # Démarrer la connexion
            await self._connect()
            
            # Démarrer le vérificateur de heartbeat dans une tâche séparée
            heartbeat_task = asyncio.create_task(self._heartbeat_check())
            
            # Écouter les messages
            await self._listen()
            
            # Annuler la tâche de heartbeat si on sort de la boucle
            heartbeat_task.cancel()
            
        except Exception as e:
            logger.error(f"Erreur critique dans la boucle WebSocket: {str(e)}")
            self.running = False
    
    async def stop(self) -> None:
        """
        Arrête la connexion WebSocket proprement.
        """
        logger.info("Arrêt de la connexion WebSocket Binance...")
        self.running = False
        
        if self.ws:
            try:
                # Désabonnement des streams
                unsubscribe_msg = {
                    "method": "UNSUBSCRIBE",
                    "params": self.stream_paths,
                    "id": int(time.time() * 1000)
                }
                
                await self.ws.send(json.dumps(unsubscribe_msg))
                logger.info("Message de désabonnement envoyé")
                
                # Fermer la connexion
                await self.ws.close()
                logger.info("Connexion WebSocket fermée")
                
            except Exception as e:
                logger.error(f"Erreur lors de la fermeture du WebSocket: {str(e)}")
        
        # Fermer le client Kafka
        if self.kafka_client:
            self.kafka_client.flush()
            self.kafka_client.close()
            logger.info("Client Kafka fermé")

# Fonction principale pour exécuter le WebSocket de manière asynchrone
async def run_binance_websocket():
    """
    Fonction principale pour exécuter le WebSocket Binance.
    """
    # Créer le client Kafka
    kafka_client = KafkaClient()
    
    # Créer et démarrer le WebSocket
    ws_client = BinanceWebSocket(kafka_client=kafka_client)
    
    try:
        logger.info("🚀 Démarrage du WebSocket Binance...")
        await ws_client.start()
    except KeyboardInterrupt:
        logger.info("Interruption clavier détectée")
    except Exception as e:
        logger.error(f"Erreur dans la boucle principale: {str(e)}")
    finally:
        logger.info("Arrêt du WebSocket Binance...")
        await ws_client.stop()

# Point d'entrée pour exécution directe
if __name__ == "__main__":
    try:
        asyncio.run(run_binance_websocket())
    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")