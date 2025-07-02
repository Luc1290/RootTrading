"""
Ultra Data Fetcher - Service de récupération de données enrichies multi-timeframes
Compatible avec le système UltraConfluence pour signaux de qualité institutionnelle
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Tuple
import aiohttp
import sys
import os

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from shared.src.config import SYMBOLS
from shared.src.redis_client import RedisClient
from shared.src.technical_indicators import indicators

logger = logging.getLogger(__name__)

class UltraDataFetcher:
    """
    Récupérateur ultra-avancé de données multi-timeframes avec 20+ indicateurs techniques.
    Fournit les données nécessaires pour l'UltraConfluence et le scoring institutionnel.
    """
    
    def __init__(self):
        self.symbols = SYMBOLS
        self.timeframes = ['1m', '5m', '15m', '1h', '4h']  # Multi-timeframes pour confluence
        self.redis_client = RedisClient()
        self.running = False
        
        # URLs Binance API
        self.base_url = "https://api.binance.com"
        self.klines_endpoint = "/api/v3/klines"
        self.depth_endpoint = "/api/v3/depth"
        self.ticker_endpoint = "/api/v3/ticker/24hr"
        
        # Buffers pour calculs techniques en temps réel (comme dans binance_ws.py)
        self.price_buffers = {}
        self.volume_buffers = {}
        
        # Initialiser les buffers pour chaque symbole/timeframe
        for symbol in self.symbols:
            self.price_buffers[symbol] = {}
            self.volume_buffers[symbol] = {}
            for tf in self.timeframes:
                self.price_buffers[symbol][tf] = []
                self.volume_buffers[symbol][tf] = []
        
        # Limites API
        self.rate_limit_delay = 0.1  # 100ms entre requêtes
        self.last_request_time = 0
        
        logger.info(f"🔥 UltraDataFetcher initialisé pour {len(self.symbols)} symboles x {len(self.timeframes)} timeframes")
    
    async def start(self):
        """Démarre la récupération ultra-enrichie périodique"""
        self.running = True
        logger.info("🚀 Démarrage de l'UltraDataFetcher")
        
        while self.running:
            try:
                # Récupération multi-timeframes pour tous les symboles
                for symbol in self.symbols:
                    # Données klines multi-timeframes
                    for timeframe in self.timeframes:
                        await self._fetch_ultra_enriched_data(symbol, timeframe)
                        await self._respect_rate_limit()
                    
                    # Données orderbook pour sentiment
                    await self._fetch_orderbook_sentiment(symbol)
                    await self._respect_rate_limit()
                    
                    # Données ticker 24h pour contexte
                    await self._fetch_ticker_data(symbol)
                    await self._respect_rate_limit()
                
                # Cycle complet toutes les 2 minutes (données très riches)
                logger.info("✅ Cycle ultra-enrichi terminé, pause 120s")
                await asyncio.sleep(120)
                
            except Exception as e:
                logger.error(f"❌ Erreur dans la boucle ultra-enrichie: {e}")
                await asyncio.sleep(30)
    
    async def stop(self):
        """Arrête la récupération"""
        self.running = False
        logger.info("⏹️ Arrêt de l'UltraDataFetcher")
    
    async def _fetch_initialization_data(self):
        """
        Récupère les données initiales pour remplir les caches Redis.
        Utilisé au démarrage pour éviter d'attendre le premier cycle complet.
        """
        logger.info("🔄 Initialisation des données ultra-enrichies...")
        
        try:
            for symbol in self.symbols:
                logger.info(f"📊 Initialisation {symbol}...")
                
                # Données multi-timeframes
                for timeframe in self.timeframes:
                    await self._fetch_ultra_enriched_data(symbol, timeframe)
                    await self._respect_rate_limit()
                
                # Données de sentiment et ticker
                await self._fetch_orderbook_sentiment(symbol)
                await self._respect_rate_limit()
                
                await self._fetch_ticker_data(symbol)
                await self._respect_rate_limit()
            
            logger.info("✅ Initialisation ultra-enrichie terminée")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation: {e}")
            raise
    
    async def _respect_rate_limit(self):
        """Respecte les limites de taux API"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        
        self.last_request_time = time.time()
    
    async def _fetch_ultra_enriched_data(self, symbol: str, timeframe: str):
        """
        Récupère et traite les données ultra-enrichies pour un symbole/timeframe.
        Calcule TOUS les indicateurs nécessaires pour UltraConfluence.
        """
        try:
            # Récupérer suffisamment de klines pour tous les calculs (100 bougies)
            klines = await self._fetch_klines(symbol, timeframe, limit=100)
            
            if not klines:
                logger.warning(f"Aucune kline pour {symbol} {timeframe}")
                return
            
            # Traitement ultra-enrichi avec TOUS les indicateurs
            enriched_data = await self._process_ultra_enriched_klines(klines, symbol, timeframe)
            
            # Mettre à jour les buffers
            self._update_price_buffers(symbol, timeframe, enriched_data)
            
            # Stocker dans Redis avec format standardisé
            redis_key = f"market_data:{symbol}:{timeframe}"
            await self._store_in_redis(redis_key, enriched_data)
            
            logger.debug(f"🔥 {symbol} {timeframe}: {len(enriched_data)} indicateurs calculés")
            
        except Exception as e:
            error_msg = str(e).replace('{', '{{').replace('}', '}}') if e else "Exception vide"
            logger.error(f"❌ Erreur données ultra-enrichies {symbol} {timeframe}: {error_msg}")
    
    async def _fetch_klines(self, symbol: str, timeframe: str, limit: int = 100) -> List:
        """Récupère les klines depuis Binance"""
        try:
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'limit': limit
            }
            
            url = f"{self.base_url}{self.klines_endpoint}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"❌ API Binance error {response.status} pour {symbol} {timeframe}")
                        return []
                        
        except Exception as e:
            error_msg = str(e).replace('{', '{{').replace('}', '}}') if e else "Exception vide"
            logger.error(f"❌ Erreur fetch klines {symbol} {timeframe}: {error_msg}")
            return []
    
    async def _process_ultra_enriched_klines(self, klines: List, symbol: str, timeframe: str) -> Dict:
        """
        Traite les klines avec TOUS les indicateurs ultra-confluents.
        Utilise le module partagé technical_indicators pour les calculs optimisés.
        """
        try:
            # Extraire les prix et volumes
            prices = []
            volumes = []
            highs = []
            lows = []
            opens = []
            
            for kline in klines:
                opens.append(float(kline[1]))
                highs.append(float(kline[2]))
                lows.append(float(kline[3]))
                prices.append(float(kline[4]))  # close
                volumes.append(float(kline[5]))
            
            if not prices:
                return {}
            
            # Données de base
            enriched_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'enhanced': True,
                'ultra_enriched': True,
                'close': prices[-1],
                'volume': volumes[-1],
                'timestamp': time.time(),
                'last_update': time.time()
            }
            
            # **NOUVEAU**: Utiliser le module partagé pour tous les indicateurs
            logger.debug(f"🔧 Calcul indicateurs avec module partagé pour {symbol} {timeframe}")
            
            # Calcul de tous les indicateurs en une fois avec le module partagé
            calculated_indicators = indicators.calculate_all_indicators(highs, lows, prices, volumes)
            
            # Ajouter tous les indicateurs calculés
            enriched_data.update(calculated_indicators)
            
            # Indicateurs additionnels non couverts par le module partagé
            if len(prices) >= 20:
                # Stochastic RSI (pas dans le module partagé)
                enriched_data['stoch_rsi'] = self._calculate_stoch_rsi(prices, 14)
                
                # ADX (sera ajouté au module partagé plus tard)
                enriched_data['adx_14'] = self._calculate_adx(highs, lows, prices, 14)
                
                # Williams %R
                enriched_data['williams_r'] = self._calculate_williams_r(highs, lows, prices, 14)
                
                # CCI
                enriched_data['cci_20'] = self._calculate_cci(highs, lows, prices, 20)
            
            # VWAP
            if len(prices) >= 10:
                enriched_data['vwap_10'] = self._calculate_vwap(prices[-10:], volumes[-10:])
            
            logger.debug(f"✅ {symbol} {timeframe}: {len(enriched_data)} indicateurs calculés (module partagé + additionnels)")
            return enriched_data
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement ultra-enrichi {symbol} {timeframe}: {e}")
            return {}
    
    async def _fetch_orderbook_sentiment(self, symbol: str):
        """Récupère et analyse le sentiment orderbook"""
        try:
            params = {
                'symbol': symbol,
                'limit': 20  # Top 20 niveaux
            }
            
            url = f"{self.base_url}{self.depth_endpoint}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        orderbook = await response.json()
                        sentiment_data = self._analyze_orderbook_sentiment(orderbook)
                        
                        # Stocker dans Redis
                        redis_key = f"orderbook_sentiment:{symbol}"
                        await self._store_in_redis(redis_key, sentiment_data)
                        
                        logger.debug(f"📊 Sentiment orderbook {symbol} mis à jour")
                        
        except Exception as e:
            logger.error(f"❌ Erreur sentiment orderbook {symbol}: {e}")
    
    async def _fetch_ticker_data(self, symbol: str):
        """Récupère les données ticker 24h"""
        try:
            params = {'symbol': symbol}
            url = f"{self.base_url}{self.ticker_endpoint}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        ticker = await response.json()
                        
                        # Stocker dans Redis
                        redis_key = f"ticker_24h:{symbol}"
                        await self._store_in_redis(redis_key, ticker)
                        
                        logger.debug(f"📈 Ticker 24h {symbol} mis à jour")
                        
        except Exception as e:
            logger.error(f"❌ Erreur ticker 24h {symbol}: {e}")
    
    def _update_price_buffers(self, symbol: str, timeframe: str, data: Dict):
        """Met à jour les buffers de prix pour calculs en temps réel"""
        try:
            if 'close' in data:
                self.price_buffers[symbol][timeframe].append(data['close'])
                # Garder seulement les 100 derniers prix
                if len(self.price_buffers[symbol][timeframe]) > 100:
                    self.price_buffers[symbol][timeframe] = self.price_buffers[symbol][timeframe][-100:]
            
            if 'volume' in data:
                self.volume_buffers[symbol][timeframe].append(data['volume'])
                if len(self.volume_buffers[symbol][timeframe]) > 100:
                    self.volume_buffers[symbol][timeframe] = self.volume_buffers[symbol][timeframe][-100:]
                    
        except Exception as e:
            logger.error(f"❌ Erreur mise à jour buffers {symbol} {timeframe}: {e}")
    
    async def _store_in_redis(self, key: str, data: Dict):
        """Stocke les données dans Redis"""
        try:
            self.redis_client.set(key, json.dumps(data), expiration=3600)  # Expire après 1h
        except Exception as e:
            logger.error(f"❌ Erreur stockage Redis {key}: {e}")
    
    # =================== MÉTHODES DE CALCUL D'INDICATEURS ===================
    # (Copies des méthodes de binance_ws.py pour cohérence)
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Calcule le RSI"""
        if len(prices) < period + 1:
            return None
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return None
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return round(rsi, 2)
    
    def _calculate_stoch_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Calcule le Stochastic RSI"""
        if len(prices) < period * 2:
            return None
        
        # Calculer RSI pour chaque point
        rsi_values = []
        for i in range(period, len(prices)):
            subset = prices[i-period:i+1]
            rsi = self._calculate_rsi(subset, period)
            if rsi is not None:
                rsi_values.append(rsi)
        
        if len(rsi_values) < period:
            return None
        
        # Calculer Stochastic RSI
        recent_rsi = rsi_values[-period:]
        min_rsi = min(recent_rsi)
        max_rsi = max(recent_rsi)
        
        if max_rsi == min_rsi:
            return 50
        
        stoch_rsi = ((rsi_values[-1] - min_rsi) / (max_rsi - min_rsi)) * 100
        return round(stoch_rsi, 2)
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calcule EMA"""
        if not prices or period <= 0:
            return prices[-1] if prices else 0
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return round(ema, 6)
    
    def _calculate_macd(self, prices: List[float]) -> Dict:
        """Calcule MACD complet"""
        if len(prices) < 35:
            return {}
        
        ema12 = self._calculate_ema(prices, 12)
        ema26 = self._calculate_ema(prices, 26)
        macd_line = ema12 - ema26
        
        # Signal line (EMA 9 du MACD)
        macd_values = []
        for i in range(26, len(prices)):
            subset = prices[:i+1]
            if len(subset) >= 26:
                e12 = self._calculate_ema(subset, 12)
                e26 = self._calculate_ema(subset, 26)
                macd_values.append(e12 - e26)
        
        if len(macd_values) >= 9:
            signal_line = self._calculate_ema(macd_values, 9)
            histogram = macd_line - signal_line
        else:
            signal_line = macd_line
            histogram = 0
        
        return {
            'macd_line': round(macd_line, 6),
            'macd_signal': round(signal_line, 6),
            'macd_histogram': round(histogram, 6)
        }
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int, std_dev: float) -> Dict:
        """Calcule les Bollinger Bands"""
        if len(prices) < period:
            return {}
        
        sma = sum(prices[-period:]) / period
        variance = sum((x - sma) ** 2 for x in prices[-period:]) / period
        std = variance ** 0.5
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        # Position dans les bandes (0 = bande basse, 1 = bande haute)
        current_price = prices[-1]
        if upper_band != lower_band:
            bb_position = (current_price - lower_band) / (upper_band - lower_band)
        else:
            bb_position = 0.5
        
        return {
            'bb_upper': round(upper_band, 6),
            'bb_middle': round(sma, 6),
            'bb_lower': round(lower_band, 6),
            'bb_position': round(bb_position, 3),
            'bb_width': round(((upper_band - lower_band) / sma) * 100, 2)
        }
    
    def _calculate_adx(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
        """Calcule ADX (simplifié)"""
        if len(closes) < period + 1:
            return None
        
        # Calcul simplifié du momentum directionnel
        ups = []
        downs = []
        
        for i in range(1, len(closes)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            
            if up_move > down_move and up_move > 0:
                ups.append(up_move)
                downs.append(0)
            elif down_move > up_move and down_move > 0:
                ups.append(0)
                downs.append(down_move)
            else:
                ups.append(0)
                downs.append(0)
        
        if len(ups) < period:
            return None
        
        avg_up = sum(ups[-period:]) / period
        avg_down = sum(downs[-period:]) / period
        
        if avg_up + avg_down == 0:
            return 0
        
        # ADX simplifié
        dx = abs(avg_up - avg_down) / (avg_up + avg_down) * 100
        
        return round(dx, 2)
    
    def _calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
        """Calcule ATR"""
        if len(closes) < period + 1:
            return None
        
        true_ranges = []
        
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            
            true_range = max(high_low, high_close, low_close)
            true_ranges.append(true_range)
        
        if len(true_ranges) < period:
            return None
        
        atr = sum(true_ranges[-period:]) / period
        return round(atr, 6)
    
    def _calculate_williams_r(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
        """Calcule Williams %R"""
        if len(closes) < period:
            return None
        
        highest_high = max(highs[-period:])
        lowest_low = min(lows[-period:])
        current_close = closes[-1]
        
        if highest_high == lowest_low:
            return -50
        
        williams_r = ((highest_high - current_close) / (highest_high - lowest_low)) * -100
        return round(williams_r, 2)
    
    def _calculate_cci(self, highs: List[float], lows: List[float], closes: List[float], period: int = 20) -> Optional[float]:
        """Calcule CCI"""
        if len(closes) < period:
            return None
        
        # Typical Price
        typical_prices = []
        for i in range(len(closes)):
            tp = (highs[i] + lows[i] + closes[i]) / 3
            typical_prices.append(tp)
        
        if len(typical_prices) < period:
            return None
        
        # SMA des typical prices
        sma_tp = sum(typical_prices[-period:]) / period
        
        # Déviation moyenne
        mean_deviation = sum(abs(tp - sma_tp) for tp in typical_prices[-period:]) / period
        
        if mean_deviation == 0:
            return 0
        
        current_tp = typical_prices[-1]
        cci = (current_tp - sma_tp) / (0.015 * mean_deviation)
        
        return round(cci, 2)
    
    def _calculate_vwap(self, prices: List[float], volumes: List[float]) -> Optional[float]:
        """Calcule VWAP"""
        if not prices or not volumes or len(prices) != len(volumes):
            return None
        
        total_volume = sum(volumes)
        if total_volume == 0:
            return None
        
        vwap = sum(p * v for p, v in zip(prices, volumes)) / total_volume
        return round(vwap, 6)
    
    def _calculate_momentum(self, prices: List[float], period: int) -> Optional[float]:
        """Calcule le momentum"""
        if len(prices) < period + 1:
            return None
        
        momentum = ((prices[-1] / prices[-period-1]) - 1) * 100
        return round(momentum, 3)
    
    def _analyze_volume(self, volumes: List[float]) -> Dict:
        """Analyse du volume"""
        if len(volumes) < 20:
            return {}
        
        recent_vol = volumes[-10:]
        avg_vol = sum(volumes[-20:]) / 20
        
        current_vol = volumes[-1]
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
        
        # Détection du spike
        volume_spike = vol_ratio > 2.0
        
        # Tendance du volume
        if len(recent_vol) >= 5:
            early_avg = sum(recent_vol[:5]) / 5
            late_avg = sum(recent_vol[5:]) / 5
            
            if late_avg > early_avg * 1.1:
                volume_trend = 'increasing'
            elif late_avg < early_avg * 0.9:
                volume_trend = 'decreasing'
            else:
                volume_trend = 'stable'
        else:
            volume_trend = 'stable'
        
        return {
            'volume_ratio': round(vol_ratio, 2),
            'volume_spike': volume_spike,
            'volume_trend': volume_trend,
            'avg_volume_20': round(avg_vol, 2)
        }
    
    def _analyze_orderbook_sentiment(self, orderbook: Dict) -> Dict:
        """Analyse le sentiment de l'orderbook"""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                return {}
            
            # Calculs de base
            bid_price = float(bids[0][0])
            ask_price = float(asks[0][0])
            spread = ask_price - bid_price
            spread_pct = (spread / bid_price) * 100
            
            # Volumes totaux
            total_bid_volume = sum(float(bid[1]) for bid in bids[:10])
            total_ask_volume = sum(float(ask[1]) for ask in asks[:10])
            
            # Ratio et imbalance
            total_volume = total_bid_volume + total_ask_volume
            if total_volume > 0:
                bid_ask_ratio = total_bid_volume / total_ask_volume
                imbalance = (total_bid_volume - total_ask_volume) / total_volume
            else:
                bid_ask_ratio = 1.0
                imbalance = 0.0
            
            # Signal de sentiment
            if bid_ask_ratio > 1.5:
                sentiment_signal = 'VERY_BULLISH'
            elif bid_ask_ratio > 1.2:
                sentiment_signal = 'BULLISH'
            elif bid_ask_ratio < 0.67:
                sentiment_signal = 'VERY_BEARISH'
            elif bid_ask_ratio < 0.83:
                sentiment_signal = 'BEARISH'
            else:
                sentiment_signal = 'NEUTRAL'
            
            return {
                'bid_price': bid_price,
                'ask_price': ask_price,
                'spread_pct': round(spread_pct, 4),
                'bid_ask_ratio': round(bid_ask_ratio, 3),
                'imbalance': round(imbalance, 3),
                'sentiment_signal': sentiment_signal,
                'sentiment_score': round(imbalance, 3),
                'total_bid_volume': round(total_bid_volume, 2),
                'total_ask_volume': round(total_ask_volume, 2),
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse sentiment: {e}")
            return {}

    async def load_historical_data(self, days: int = 5) -> None:
        """
        Charge les données historiques pour tous les symboles et timeframes.
        Publie aussi les données sur Kafka pour persistance en DB.
        
        Args:
            days: Nombre de jours d'historique à charger (défaut: 5)
        """
        logger.info(f"🔄 Chargement de {days} jours de données historiques...")
        
        # Importer le producteur Kafka pour publier les données historiques
        try:
            from kafka_producer import get_producer
            kafka_producer = get_producer()
        except ImportError:
            logger.warning("⚠️ Producteur Kafka non disponible, données historiques non persistées en DB")
            kafka_producer = None
        
        # Calculer les limites optimisées pour chaque timeframe (selon besoins des indicateurs)
        timeframe_limits = {
            '1m': 180,    # 3 heures (suffisant pour tous les indicateurs)
            '5m': 288,    # 24 heures (1 jour complet)  
            '15m': 192,   # 2 jours (48h / 15m = 192 points)
            '1h': 168,    # 7 jours (168 heures)
            '4h': 84      # 14 jours (84 * 4h = 14 jours)
        }
        
        total_expected = 0
        total_loaded = 0
        total_published = 0
        
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                expected_count = timeframe_limits[timeframe]
                total_expected += expected_count
                
                try:
                    # Charger les données historiques avec pagination si nécessaire
                    historical_data = await self._fetch_historical_klines(
                        symbol, timeframe, expected_count
                    )
                    
                    if historical_data:
                        loaded_count = len(historical_data)
                        total_loaded += loaded_count
                        
                        # **NOUVEAU**: Enrichir et publier chaque kline historique sur Kafka
                        if kafka_producer and historical_data:
                            try:
                                # Enrichir toutes les données historiques avec indicateurs techniques
                                enriched_historical = await self._enrich_historical_batch(
                                    historical_data, symbol, timeframe
                                )
                                
                                # Publier chaque point enrichi sur Kafka
                                for enriched_point in enriched_historical:
                                    try:
                                        kafka_topic = f"market.data.{symbol.lower()}.{timeframe}"
                                        kafka_producer.publish_market_data(enriched_point, kafka_topic)
                                        total_published += 1
                                        
                                    except Exception as e:
                                        logger.warning(f"Erreur publication Kafka point enrichi: {e}")
                                        
                                logger.info(f"📊 {symbol} {timeframe}: {len(enriched_historical)} points enrichis publiés")
                                
                            except Exception as e:
                                logger.error(f"❌ Erreur enrichissement historique {symbol} {timeframe}: {e}")
                        
                        # Traiter et enrichir la dernière donnée pour Redis (indicateurs actuels)
                        enriched_data = await self._process_historical_klines(
                            historical_data, symbol, timeframe
                        )
                        
                        # Stocker dans Redis
                        redis_key = f"market_data:{symbol}:{timeframe}_history"
                        await self._store_in_redis(redis_key, enriched_data)
                        
                        logger.info(f"📊 {symbol} {timeframe}: {loaded_count}/{expected_count} points chargés")
                    else:
                        logger.warning(f"⚠️ Aucune donnée historique pour {symbol} {timeframe}")
                        
                except Exception as e:
                    logger.error(f"❌ Erreur chargement historique {symbol} {timeframe}: {e}")
        
        # Attendre que tous les messages Kafka soient envoyés
        if kafka_producer:
            kafka_producer.flush()
            logger.info(f"📤 {total_published} points historiques publiés sur Kafka pour persistance DB")
        
        success_rate = (total_loaded / total_expected * 100) if total_expected > 0 else 0
        logger.info(f"✅ Chargement historique terminé: {total_loaded}/{total_expected} points ({success_rate:.1f}%)")
    
    async def _fetch_historical_klines(self, symbol: str, timeframe: str, limit: int) -> List:
        """
        Récupère les klines historiques avec pagination pour les gros volumes.
        Binance limite à 1000 klines par requête.
        """
        all_klines = []
        max_per_request = 1000
        
        try:
            while len(all_klines) < limit:
                # Calculer combien récupérer dans cette requête
                remaining = limit - len(all_klines)
                current_limit = min(remaining, max_per_request)
                
                # Calculer l'endTime pour récupérer les données plus anciennes
                if all_klines:
                    # Utiliser le timestamp de la première kline comme endTime
                    end_time = all_klines[0][0] - 1  # -1ms pour éviter les doublons
                else:
                    # Première requête, pas d'endTime
                    end_time = None
                
                # Faire la requête
                batch_klines = await self._fetch_klines_batch(
                    symbol, timeframe, current_limit, end_time
                )
                
                if not batch_klines:
                    logger.warning(f"Pas de données reçues pour {symbol} {timeframe}")
                    break
                
                # Ajouter au début (ordre chronologique inverse)
                all_klines = batch_klines + all_klines
                
                # Log de progression
                logger.debug(f"📊 {symbol} {timeframe}: {len(all_klines)}/{limit} points récupérés")
                
                # Pause pour éviter la limite de taux
                await asyncio.sleep(0.1)
                
                # Si on a reçu moins que demandé, on a atteint la fin
                if len(batch_klines) < current_limit:
                    break
            
            # Trier par timestamp pour être sûr de l'ordre chronologique
            all_klines.sort(key=lambda x: x[0])
            
            logger.info(f"📚 {symbol} {timeframe}: {len(all_klines)} klines historiques récupérées")
            return all_klines
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération historique {symbol} {timeframe}: {e}")
            return []
    
    async def _fetch_klines_batch(self, symbol: str, timeframe: str, limit: int, end_time: Optional[int] = None) -> List:
        """Récupère un batch de klines avec gestion de l'endTime"""
        try:
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'limit': limit
            }
            
            if end_time:
                params['endTime'] = end_time
            
            url = f"{self.base_url}{self.klines_endpoint}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"❌ API Binance error {response.status} pour {symbol} {timeframe}")
                        return []
                        
        except Exception as e:
            logger.error(f"❌ Erreur fetch batch {symbol} {timeframe}: {e}")
            return []
    
    async def _enrich_historical_batch(self, klines: List, symbol: str, timeframe: str) -> List[Dict]:
        """
        Enrichit les données historiques avec approche vectorisée optimisée.
        
        Args:
            klines: Liste des klines historiques brutes
            symbol: Symbole de trading
            timeframe: Timeframe des données
            
        Returns:
            Liste des points enrichis avec indicateurs
        """
        enriched_points = []
        
        try:
            logger.info(f"🚀 Enrichissement vectorisé de {len(klines)} points pour {symbol} {timeframe}")
            
            # **OPTIMISATION**: Calculer les indicateurs une seule fois pour tout le dataset
            latest_indicators = {}
            if len(klines) >= 20:
                try:
                    latest_indicators = await self._process_ultra_enriched_klines(klines, symbol, timeframe)
                    logger.debug(f"Indicateurs calculés pour {symbol} {timeframe}")
                except Exception as e:
                    logger.warning(f"Erreur calcul indicateurs pour {symbol} {timeframe}: {e}")
            
            # Créer un point enrichi pour chaque kline avec ses vraies valeurs OHLCV
            for i, kline in enumerate(klines):
                enriched_point = {
                    'symbol': symbol,
                    'interval': timeframe,
                    'start_time': kline[0],  # Utiliser start_time pour compatibilité avec dispatcher
                    'open_time': kline[0],
                    'close_time': kline[6],
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),  # **FIX**: Utiliser la vraie valeur de fermeture de cette kline
                    'volume': float(kline[5]),
                    'is_closed': True,
                    'is_historical': True,
                    'enhanced': True,
                    'ultra_enriched': True
                }
                
                # Ajouter les indicateurs seulement aux derniers points (les plus récents ont les meilleurs indicateurs)
                if i >= len(klines) - 50 and latest_indicators:
                    for key, value in latest_indicators.items():
                        if key not in ['symbol', 'timeframe', 'enhanced', 'ultra_enriched', 'close', 'volume', 'timestamp', 'last_update']:
                            if isinstance(value, (int, float)):
                                enriched_point[key] = value
                
                enriched_points.append(enriched_point)
            
            logger.info(f"✨ {symbol} {timeframe}: {len(enriched_points)} points enrichis")
            return enriched_points
            
        except Exception as e:
            logger.error(f"❌ Erreur enrichissement vectorisé {symbol} {timeframe}: {e}")
            # En cas d'erreur, retourner au moins les données de base
            basic_points = []
            for kline in klines:
                basic_point = {
                    'symbol': symbol,
                    'interval': timeframe,
                    'start_time': kline[0],  # start_time pour dispatcher
                    'open_time': kline[0],
                    'close_time': kline[6],
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),  # **FIX**: Vraie valeur de fermeture
                    'volume': float(kline[5]),
                    'is_closed': True,
                    'is_historical': True
                }
                basic_points.append(basic_point)
            return basic_points

    async def _process_historical_klines(self, klines: List, symbol: str, timeframe: str) -> Dict:
        """Traite et enrichit un batch de klines historiques"""
        try:
            if not klines:
                return {}
            
            # Prendre la dernière kline comme données actuelles (plus récente)
            latest_kline = klines[-1]
            
            # Traiter comme une kline normale pour obtenir les indicateurs
            enriched_data = await self._process_ultra_enriched_klines(klines, symbol, timeframe)
            
            # Ajouter les métadonnées historiques
            enriched_data.update({
                'symbol': symbol,
                'timeframe': timeframe,
                'historical_count': len(klines),
                'oldest_timestamp': klines[0][0],
                'newest_timestamp': klines[-1][0],
                'is_historical': True
            })
            
            return enriched_data
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement historique {symbol} {timeframe}: {e}")
            return {}

# Point d'entrée pour test standalone
async def main():
    """Test standalone de l'UltraDataFetcher"""
    fetcher = UltraDataFetcher()
    await fetcher.start()

if __name__ == "__main__":
    asyncio.run(main())