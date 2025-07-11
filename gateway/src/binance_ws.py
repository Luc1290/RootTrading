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
        
        # 🚀 NOUVEAU : Cache pour indicateurs incrémentaux (évite dents de scie)
        self.incremental_cache = {}
        
        # Initialiser les buffers pour chaque symbole/timeframe
        for symbol in self.symbols:
            self.price_buffers[symbol] = {}
            self.volume_buffers[symbol] = {}
            self.high_buffers[symbol] = {}
            self.low_buffers[symbol] = {}
            self.rsi_buffers[symbol] = {}
            self.macd_buffers[symbol] = {}
            
            # 🚀 NOUVEAU : Initialiser cache incrémental pour EMA/MACD lisses
            self.incremental_cache[symbol] = {}
            
            for tf in self.timeframes:
                self.price_buffers[symbol][tf] = []
                self.volume_buffers[symbol][tf] = []
                self.high_buffers[symbol][tf] = []
                self.low_buffers[symbol][tf] = []
                self.rsi_buffers[symbol][tf] = []
                self.macd_buffers[symbol][tf] = {'ema12': None, 'ema26': None, 'signal9': None}
                
                # Cache pour chaque timeframe
                self.incremental_cache[symbol][tf] = {}
                
        # Configuration pour sauvegarde périodique des buffers
        self.redis_client = None
        self._init_redis_for_buffers()
        self.buffer_save_interval = 300  # 5 minutes
        self.last_buffer_save_time = 0
        
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
        
        # 🚀 HYBRIDE : Calcul incrémental pour EMA/MACD + traditionnel pour le reste
        if len(prices) >= 20 and len(highs) >= 20 and len(lows) >= 20 and len(volumes) >= 20:
            logger.info(f"📊 HYBRIDE WebSocket {symbol} {timeframe}: buffers=[P:{len(prices)},H:{len(highs)},L:{len(lows)},V:{len(volumes)}] → calcul des 33 indicateurs")
            
            # 📊 TRADITIONNEL : Calculer TOUS les indicateurs d'abord
            from shared.src.technical_indicators import indicators
            all_indicators = indicators.calculate_all_indicators(highs, lows, prices, volumes)
            
            # Debug: vérifier si ADX est calculé
            if 'adx_14' in all_indicators and all_indicators['adx_14'] is not None:
                logger.info(f"✅ ADX calculé dans WebSocket {symbol} {timeframe}: {all_indicators['adx_14']}")
            else:
                logger.warning(f"❌ ADX manquant dans WebSocket {symbol} {timeframe}, buffer sizes: H={len(highs)}, L={len(lows)}, C={len(prices)}")
                if 'adx_14' in all_indicators:
                    logger.warning(f"ADX value is None: {all_indicators['adx_14']}")
            
            # 🚀 NOUVEAU : Calcul incrémental pour EMA/MACD (évite dents de scie)
            incremental_indicators = self._calculate_smooth_indicators(symbol, timeframe, candle_data, all_indicators)
            
            # FUSION INTELLIGENTE : Garder tous les traditionnels + override seulement EMA/MACD avec versions lisses
            final_indicators = all_indicators.copy()
            # Override seulement les indicateurs EMA/MACD avec les versions incrémentales lisses
            for indicator_name, value in incremental_indicators.items():
                if value is not None and indicator_name in ['ema_12', 'ema_26', 'ema_50', 'macd_line', 'macd_signal', 'macd_histogram']:
                    final_indicators[indicator_name] = value
            
            # Ajouter tous les indicateurs calculés
            for indicator_name, value in final_indicators.items():
                if value is not None:
                    candle_data[indicator_name] = value
            
            logger.debug(f"✅ {len(final_indicators)} indicateurs calculés pour {symbol} (🚀 {len(incremental_indicators)} EMA/MACD incrémentaux, {len(all_indicators)} traditionnels)")
            
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
            # Log pour debug: pourquoi on n'atteint pas les 20 points
            logger.info(f"🔍 WebSocket {symbol} {timeframe}: buffers=[P:{len(prices)},H:{len(highs)},L:{len(lows)},V:{len(volumes)}] → calcul PARTIEL seulement")
            # Calcul partiel pour avoir au moins RSI et quelques indicateurs de base
            logger.debug(f"📊 Calcul partiel des indicateurs pour {symbol} {timeframe}")
            
            # 🚀 NOUVEAU : Même pour calcul partiel, utiliser les EMA incrémentales si possible
            incremental_indicators = self._calculate_smooth_indicators(symbol, timeframe, candle_data)
            for indicator_name, value in incremental_indicators.items():
                if value is not None:
                    candle_data[indicator_name] = value
            
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
            # 🔄 NOUVEAU : Restaurer les buffers WebSocket depuis Redis
            logger.info("🔄 Restoration des buffers WebSocket depuis Redis...")
            restored_buffers = await self._restore_buffers_from_redis()
            
            if restored_buffers > 0:
                logger.info(f"✅ {restored_buffers} buffers WebSocket restaurés - CONTINUITÉ PRÉSERVÉE")
            else:
                logger.info("🆕 Aucun buffer restauré - Premier démarrage ou cache vide")
            
            # 🚀 NOUVEAU : S'assurer que les buffers ont assez de données pour tous les indicateurs
            await self._ensure_sufficient_buffer_data()
            
            # 🚀 Initialiser le cache incrémental (compléter si pas restauré)
            logger.info("💾 Initialisation du cache incrémental pour EMA/MACD lisses...")
            for symbol in self.symbols:
                for timeframe in self.timeframes:
                    self._initialize_incremental_cache_simple(symbol, timeframe)
            logger.info("✅ Cache incrémental initialisé pour tous les symboles/timeframes")
            
            # Démarrer la connexion
            await self._connect()
            
            # Démarrer les tâches parallèles
            heartbeat_task = asyncio.create_task(self._heartbeat_check())
            buffer_save_task = asyncio.create_task(self._periodic_buffer_save())
            
            logger.info("🚀 WebSocket + Sauvegarde périodique démarrés")
            
            # Écouter les messages
            await self._listen()
            
            # Annuler les tâches parallèles si on sort de la boucle
            heartbeat_task.cancel()
            buffer_save_task.cancel()
            
        except Exception as e:
            logger.error(f"Erreur critique dans la boucle WebSocket: {str(e)}")
            self.running = False
    
    async def stop(self) -> None:
        """
        Arrête la connexion WebSocket proprement.
        """
        logger.info("Arrêt de la connexion WebSocket Binance...")
        self.running = False
        
        # 💾 Sauvegarder les buffers avant l'arrêt pour préserver la continuité
        logger.info("💾 Sauvegarde finale des buffers WebSocket...")
        try:
            await self._save_buffers_to_redis()
            logger.info("✅ Buffers sauvegardés pour la prochaine restoration")
        except Exception as e:
            logger.warning(f"Erreur sauvegarde finale: {e}")
        
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

    def _calculate_smooth_indicators(self, symbol: str, timeframe: str, candle_data: Dict, all_indicators: Dict = None) -> Dict:
        """
        🚀 NOUVEAU : Calcule EMA/MACD de manière incrémentale pour éviter les dents de scie.
        
        Args:
            symbol: Symbole tradé
            timeframe: Intervalle de temps  
            candle_data: Données de la bougie actuelle
            
        Returns:
            Dict avec indicateurs EMA/MACD lisses
        """
        result = {}
        current_price = candle_data['close']
        
        try:
            # Cache pour ce symbole/timeframe
            cache = self.incremental_cache[symbol][timeframe]
            
            # 📈 EMA 12, 26, 50 (incrémentaux avec initialisation intelligente)
            from shared.src.technical_indicators import calculate_ema_incremental
            
            for period in [12, 26, 50]:
                cache_ema_key = f'ema_{period}'
                prev_ema = cache.get(cache_ema_key)
                
                if prev_ema is None:
                    # Première fois : utiliser la valeur déjà calculée dans all_indicators
                    if all_indicators and cache_ema_key in all_indicators and all_indicators[cache_ema_key] is not None:
                        traditional_ema = all_indicators[cache_ema_key]
                        cache[cache_ema_key] = traditional_ema
                        result[cache_ema_key] = traditional_ema
                        logger.debug(f"🎯 EMA {period} initialisée depuis all_indicators: {traditional_ema:.4f}")
                        continue
                    else:
                        logger.debug(f"⚠️ EMA {period} non disponible dans all_indicators, skip incrémental")
                
                # Calcul incrémental : EMA_new = α × price + (1-α) × EMA_prev
                new_ema = calculate_ema_incremental(current_price, prev_ema, period)
                result[cache_ema_key] = new_ema
                
                # Mettre à jour le cache
                cache[cache_ema_key] = new_ema
            
            # 📊 MACD incrémental (basé sur EMA 12/26 du cache)
            from shared.src.technical_indicators import calculate_macd_incremental
            
            prev_ema_fast = cache.get('macd_ema_fast')  # EMA 12 pour MACD
            prev_ema_slow = cache.get('macd_ema_slow')  # EMA 26 pour MACD  
            prev_macd_signal = cache.get('macd_signal')
            
            # Utiliser les EMA du cache si disponibles
            if cache.get('ema_12') is not None and cache.get('ema_26') is not None:
                # Synchroniser les EMA MACD avec les EMA principales
                if prev_ema_fast is None:
                    cache['macd_ema_fast'] = cache['ema_12']
                    prev_ema_fast = cache['ema_12']
                if prev_ema_slow is None:
                    cache['macd_ema_slow'] = cache['ema_26']
                    prev_ema_slow = cache['ema_26']
            
            macd_result = calculate_macd_incremental(
                current_price, prev_ema_fast, prev_ema_slow, prev_macd_signal
            )
            
            result.update({
                'macd_line': macd_result['macd_line'],
                'macd_signal': macd_result['macd_signal'],
                'macd_histogram': macd_result['macd_histogram']
            })
            
            # Mettre à jour le cache MACD
            cache['macd_ema_fast'] = macd_result['ema_fast']
            cache['macd_ema_slow'] = macd_result['ema_slow'] 
            cache['macd_signal'] = macd_result['macd_signal']
            
            if result:
                logger.debug(f"🚀 Indicateurs lisses calculés pour {symbol} {timeframe}: "
                            f"EMA12={result.get('ema_12', 0):.4f}, "
                            f"MACD={result.get('macd_line', 0):.4f}")
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul indicateurs incrémentaux {symbol} {timeframe}: {e}")
            # En cas d'erreur, retourner un dict vide (fallback vers calcul traditionnel)
            result = {}
        
        return result

    def _initialize_incremental_cache_simple(self, symbol: str, timeframe: str):
        """
        🔄 Initialise le cache incrémental de manière simple.
        Les premières EMA seront calculées normalement, puis continuées de manière incrémentale.
        """
        try:
            # Pour l'instant, initialisation simple : le cache commence vide
            # Les premières valeurs EMA/MACD seront calculées par la méthode traditionnelle
            # puis les suivantes seront incrémentales et lisses
            
            cache = self.incremental_cache[symbol][timeframe]
            
            # Cache initialement vide - sera rempli au premier calcul
            cache['ema_12'] = None
            cache['ema_26'] = None
            cache['ema_50'] = None
            cache['macd_ema_fast'] = None
            cache['macd_ema_slow'] = None
            cache['macd_signal'] = None
            
            logger.debug(f"💾 Cache incrémental initialisé (vide) pour {symbol} {timeframe}")
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur initialisation cache pour {symbol} {timeframe}: {e}")
    
    def _init_redis_for_buffers(self):
        """Initialise Redis pour la sauvegarde des buffers WebSocket"""
        try:
            from shared.src.redis_client import RedisClient
            self.redis_client = RedisClient()
            logger.info("✅ WebSocket connecté à Redis pour persistence des buffers")
        except Exception as e:
            logger.warning(f"⚠️ WebSocket sans Redis: {e}")
            self.redis_client = None
    
    async def _save_buffers_to_redis(self):
        """Sauvegarde périodique des buffers WebSocket vers Redis"""
        if not self.redis_client:
            return
        
        try:
            saved_count = 0
            ttl = 24 * 3600  # 24h TTL pour buffers WebSocket
            
            for symbol in self.symbols:
                for timeframe in self.timeframes:
                    # Sauvegarder les price_buffers
                    if symbol in self.price_buffers and timeframe in self.price_buffers[symbol]:
                        if self.price_buffers[symbol][timeframe]:
                            key = f"ws_buffers:prices:{symbol}:{timeframe}"
                            buffer_data = {
                                'prices': self.price_buffers[symbol][timeframe][-100:],  # Garder 100 derniers
                                'timestamp': time.time()
                            }
                            self.redis_client.set(key, json.dumps(buffer_data), expiration=ttl)
                            saved_count += 1
                    
                    # Sauvegarder incremental_cache
                    if symbol in self.incremental_cache and timeframe in self.incremental_cache[symbol]:
                        if self.incremental_cache[symbol][timeframe]:
                            key = f"ws_buffers:incremental:{symbol}:{timeframe}"
                            cache_data = {
                                'cache': self.incremental_cache[symbol][timeframe],
                                'timestamp': time.time()
                            }
                            self.redis_client.set(key, json.dumps(cache_data), expiration=ttl)
                            saved_count += 1
            
            if saved_count > 0:
                logger.debug(f"💾 {saved_count} buffers WebSocket sauvegardés vers Redis")
                
        except Exception as e:
            logger.warning(f"Erreur sauvegarde buffers: {e}")
    
    async def _restore_buffers_from_redis(self):
        """Restaure les buffers WebSocket depuis Redis au démarrage"""
        if not self.redis_client:
            return 0
        
        try:
            restored_count = 0
            
            for symbol in self.symbols:
                for timeframe in self.timeframes:
                    # Restaurer price_buffers
                    try:
                        key = f"ws_buffers:prices:{symbol}:{timeframe}"
                        buffer_data = self.redis_client.get(key)
                        if buffer_data:
                            # RedisClientPool fait déjà le parsing JSON automatiquement
                            if isinstance(buffer_data, str):
                                buffer_data = json.loads(buffer_data)
                            self.price_buffers[symbol][timeframe] = buffer_data.get('prices', [])
                            restored_count += 1
                            logger.debug(f"🔄 Prices buffer restauré: {symbol} {timeframe} ({len(self.price_buffers[symbol][timeframe])} prix)")
                    except Exception as e:
                        logger.warning(f"Erreur restoration prices {symbol} {timeframe}: {e}")
                    
                    # Restaurer incremental_cache
                    try:
                        key = f"ws_buffers:incremental:{symbol}:{timeframe}"
                        cache_data = self.redis_client.get(key)
                        if cache_data:
                            # RedisClientPool fait déjà le parsing JSON automatiquement
                            if isinstance(cache_data, str):
                                cache_data = json.loads(cache_data)
                            self.incremental_cache[symbol][timeframe].update(cache_data.get('cache', {}))
                            restored_count += 1
                            logger.debug(f"🔄 Cache incrémental restauré: {symbol} {timeframe}")
                    except Exception as e:
                        logger.warning(f"Erreur restoration cache {symbol} {timeframe}: {e}")
            
            if restored_count > 0:
                logger.info(f"🔄 {restored_count} buffers WebSocket restaurés depuis Redis")
            
            return restored_count
            
        except Exception as e:
            logger.error(f"Erreur restoration buffers: {e}")
            return 0
    
    async def _periodic_buffer_save(self):
        """Tâche périodique de sauvegarde des buffers"""
        while self.running:
            try:
                current_time = time.time()
                if current_time - self.last_buffer_save_time > self.buffer_save_interval:
                    await self._save_buffers_to_redis()
                    self.last_buffer_save_time = current_time
                
                await asyncio.sleep(60)  # Vérifier toutes les minutes
                
            except Exception as e:
                logger.warning(f"Erreur sauvegarde périodique: {e}")
                await asyncio.sleep(60)

    async def _ensure_sufficient_buffer_data(self):
        """
        S'assure que les buffers WebSocket ont suffisamment de données (≥20 points)
        pour calculer immédiatement tous les 33 indicateurs techniques.
        """
        try:
            import aiohttp
            
            total_fetched = 0
            
            for symbol in self.symbols:
                for timeframe in self.timeframes:
                    current_buffer_size = len(self.price_buffers[symbol][timeframe])
                    
                    # Vérifier TOUS les buffers pour ce symbole/timeframe  
                    highs_size = len(self.high_buffers[symbol][timeframe])
                    lows_size = len(self.low_buffers[symbol][timeframe])
                    volumes_size = len(self.volume_buffers[symbol][timeframe])
                    min_buffer_size = min(current_buffer_size, highs_size, lows_size, volumes_size)
                    
                    if min_buffer_size < 20:
                        # Fetch recent klines from Binance API to fill buffers
                        needed_points = 25 - min_buffer_size  # Fetch a few extra
                        
                        try:
                            url = "https://api.binance.com/api/v3/klines"
                            params = {
                                'symbol': symbol,
                                'interval': timeframe,
                                'limit': needed_points
                            }
                            
                            async with aiohttp.ClientSession() as session:
                                async with session.get(url, params=params, timeout=10) as response:
                                    if response.status == 200:
                                        klines = await response.json()
                                        
                                        # Add to buffers (but don't exceed buffer limit)
                                        for kline in klines:
                                            if len(self.price_buffers[symbol][timeframe]) >= 100:
                                                break  # Respect buffer limit
                                                
                                            self.price_buffers[symbol][timeframe].append(float(kline[4]))  # close
                                            self.high_buffers[symbol][timeframe].append(float(kline[2]))   # high
                                            self.low_buffers[symbol][timeframe].append(float(kline[3]))    # low
                                            self.volume_buffers[symbol][timeframe].append(float(kline[5])) # volume
                                            total_fetched += 1
                                        
                                        logger.debug(f"📊 Pré-rempli {symbol} {timeframe}: +{len(klines)} points → buffers=[P:{len(self.price_buffers[symbol][timeframe])},H:{len(self.high_buffers[symbol][timeframe])},L:{len(self.low_buffers[symbol][timeframe])},V:{len(self.volume_buffers[symbol][timeframe])}]")
                                    
                                    # Respect rate limits
                                    await asyncio.sleep(0.1)
                                    
                        except Exception as e:
                            logger.warning(f"Erreur pré-remplissage {symbol} {timeframe}: {e}")
            
            if total_fetched > 0:
                logger.info(f"🚀 Buffers WebSocket pré-remplis: {total_fetched} points ajoutés pour calcul immédiat des 33 indicateurs")
            else:
                logger.info("💾 Buffers WebSocket déjà suffisamment remplis")
                
        except Exception as e:
            logger.error(f"❌ Erreur lors du pré-remplissage des buffers: {e}")

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