"""
Ultra Data Fetcher - Service de récupération de données enrichies multi-timeframes
Compatible avec le système UltraConfluence pour signaux de qualité institutionnelle
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
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
                
                # Cycle complet toutes les 60 secondes (aligné avec bougies 1min)
                logger.info("✅ Cycle ultra-enrichi terminé, pause 60s")
                await asyncio.sleep(60)
                
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
            # Récupérer BEAUCOUP plus de klines pour indicateurs ultra-précis
            timeframe_limits = {
                '1m': 1000,   # 16.7 heures → EMA/SMA 200 ultra-précis
                '5m': 1000,   # 3.5 jours → Tendances moyennes parfaites
                '15m': 800,   # 8.3 jours → Indicateurs long terme solides
                '1h': 500,    # 20.8 jours → Tendance mensuelle
                '4h': 300     # 50 jours → Tendance long terme
            }
            limit = timeframe_limits.get(timeframe, 100)
            klines = await self._fetch_klines(symbol, timeframe, limit=limit)
            
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
            
            # 🚀 HYBRIDE : Calcul incrémental pour EMA/MACD + traditionnel pour le reste
            logger.debug(f"🔧 Calcul HYBRIDE indicateurs pour {symbol} {timeframe}")
            
            # 🚀 NOUVEAU : Calcul incrémental pour EMA/MACD (évite dents de scie)
            incremental_indicators = self._calculate_smooth_indicators_ultra(symbol, timeframe, prices, highs, lows, volumes)
            
            # 📊 TRADITIONNEL : Calcul normal pour les autres indicateurs
            calculated_indicators = indicators.calculate_all_indicators(highs, lows, prices, volumes)
            
            # FUSION : Priorité aux incrémentaux (EMA/MACD lisses)
            final_indicators = calculated_indicators.copy()
            final_indicators.update(incremental_indicators)  # Override EMA/MACD with smooth versions
            
            # Ajouter tous les indicateurs calculés
            enriched_data.update(final_indicators)
            
            # **CRITIQUE**: Sauvegarder les indicateurs dans le cache persistant IMMÉDIATEMENT
            from shared.src.technical_indicators import indicator_cache
            saved_indicators = 0
            for indicator_name, indicator_value in final_indicators.items():
                # Sauvegarder seulement les indicateurs numériques valides
                if isinstance(indicator_value, (int, float)) and not (isinstance(indicator_value, float) and indicator_value != indicator_value):
                    indicator_cache.set(symbol, timeframe, indicator_name, indicator_value)
                    saved_indicators += 1
            
            if saved_indicators > 0:
                logger.debug(f"💾 {saved_indicators} indicateurs sauvegardés dans cache persistant {symbol} {timeframe}")
            
            if incremental_indicators:
                logger.debug(f"✅ {len(final_indicators)} indicateurs calculés et sauvegardés pour {symbol} (🚀 EMA/MACD incrémentaux)")
            
            # Indicateurs additionnels non couverts par le module partagé
            additional_indicators = {}
            if len(prices) >= 20:
                # Stochastic RSI (pas dans le module partagé)
                additional_indicators['stoch_rsi'] = self._calculate_stoch_rsi(prices, 14)
                
                # ADX déjà calculé par calculate_all_indicators avec plus_di/minus_di
                # Pas besoin de recalculer individuellement
                
                # Williams %R
                additional_indicators['williams_r'] = self._calculate_williams_r(highs, lows, prices, 14)
                
                # CCI
                additional_indicators['cci_20'] = self._calculate_cci(highs, lows, prices, 20)
            
            # VWAP
            if len(prices) >= 10:
                additional_indicators['vwap_10'] = self._calculate_vwap(prices[-10:], volumes[-10:])
            
            # Ajouter et sauvegarder les indicateurs additionnels
            enriched_data.update(additional_indicators)
            for indicator_name, indicator_value in additional_indicators.items():
                if isinstance(indicator_value, (int, float)) and not (isinstance(indicator_value, float) and indicator_value != indicator_value):
                    indicator_cache.set(symbol, timeframe, indicator_name, indicator_value)
            
            logger.debug(f"✅ {symbol} {timeframe}: {len(enriched_data)} indicateurs calculés et sauvegardés (module partagé + additionnels)")
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
        """Stocke les données dans Redis avec TTL étendu pour continuité"""
        try:
            # TTL étendu : 48h pour données historiques, 24h pour sentiment/ticker
            if "market_data:" in key:
                ttl = 48 * 3600  # 48 heures pour données de marché
            elif "orderbook_sentiment:" in key or "ticker:" in key:
                ttl = 24 * 3600  # 24 heures pour sentiment et ticker
            else:
                ttl = 12 * 3600  # 12 heures par défaut
            
            self.redis_client.set(key, json.dumps(data), expiration=ttl)
            logger.debug(f"💾 Données stockées Redis {key} (TTL: {ttl/3600}h)")
        except Exception as e:
            logger.error(f"❌ Erreur stockage Redis {key}: {e}")
    
    # =================== MÉTHODES DE CALCUL D'INDICATEURS ===================
    # (Copies des méthodes de binance_ws.py pour cohérence)
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Calcule le RSI via le module centralisé"""
        from shared.src.technical_indicators import calculate_rsi
        return calculate_rsi(prices, period)
    
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
        """Calcule EMA via le module centralisé"""
        from shared.src.technical_indicators import calculate_ema
        result = calculate_ema(prices, period)
        return result if result is not None else (prices[-1] if prices else 0)
    
    def _calculate_macd(self, prices: List[float]) -> Dict:
        """Calcule MACD complet via le module centralisé"""
        from shared.src.technical_indicators import calculate_macd
        
        result = calculate_macd(prices)
        if result is None or any(v is None for v in result.values()):
            return {}
            
        return {
            'macd_line': round(result['macd_line'], 6),
            'macd_signal': round(result['macd_signal'], 6),
            'macd_histogram': round(result['macd_histogram'], 6)
        }
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int, std_dev: float) -> Dict:
        """Calcule les Bollinger Bands via le module centralisé"""
        from shared.src.technical_indicators import calculate_bollinger_bands
        
        result = calculate_bollinger_bands(prices, period, std_dev)
        if result is None or any(v is None for v in result.values()):
            return {}
            
        return {
            'bb_upper': round(result['bb_upper'], 6),
            'bb_middle': round(result['bb_middle'], 6),
            'bb_lower': round(result['bb_lower'], 6),
            'bb_position': round(result['bb_position'], 3),
            'bb_width': round(result['bb_width'], 2)
        }
    
    def _calculate_adx(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
        """Calcule ADX en utilisant le module partagé"""
        from shared.src.technical_indicators import calculate_adx
        
        try:
            adx_result = calculate_adx(highs, lows, closes, period)
            # calculate_adx retourne (ADX, DI+, DI-)
            adx_value = adx_result[0] if adx_result[0] is not None else None
            return round(adx_value, 2) if adx_value is not None else None
        except Exception as e:
            logger.warning(f"Erreur calcul ADX: {e}")
            return None
    
    def _calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
        """Calcule ATR via le module centralisé"""
        from shared.src.technical_indicators import calculate_atr
        result = calculate_atr(highs, lows, closes, period)
        return round(result, 6) if result is not None else None
    
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

    async def load_historical_data(self, days: int = 5, use_gap_detection: bool = True) -> None:
        """
        Charge les données historiques pour tous les symboles et timeframes.
        Publie aussi les données sur Kafka pour persistance en DB.
        
        Args:
            days: Nombre de jours d'historique à charger (défaut: 5)
            use_gap_detection: Si True, détecte et charge uniquement les données manquantes
        """
        logger.info(f"🔄 Chargement de {days} jours de données historiques...")
        
        # Importer le producteur Kafka pour publier les données historiques
        try:
            from kafka_producer import get_producer
            kafka_producer = get_producer()
        except ImportError:
            logger.warning("⚠️ Producteur Kafka non disponible, données historiques non persistées en DB")
            kafka_producer = None
            
        # Utiliser la détection de gaps si activée
        gap_filling_plan = None
        if use_gap_detection:
            try:
                from gap_detector import GapDetector
                detector = GapDetector()
                await detector.initialize()
                
                # Détecter les gaps sur la période demandée
                lookback_hours = days * 24
                all_gaps = await detector.detect_all_gaps(self.symbols, lookback_hours)
                
                # Générer le plan de remplissage optimisé
                gap_filling_plan = detector.generate_gap_filling_plan(all_gaps)
                
                if gap_filling_plan:
                    # Estimer le temps de remplissage
                    estimated_time = detector.estimate_fill_time(gap_filling_plan)
                    logger.info(f"🎯 Mode intelligent: Remplissage ciblé des gaps uniquement")
                    logger.info(f"⏱️ Temps estimé: {estimated_time:.1f}s ({estimated_time/60:.1f} minutes)")
                else:
                    logger.info("✅ Aucun gap détecté - données déjà complètes")
                    await detector.close()
                    return
                    
                await detector.close()
                
            except Exception as e:
                logger.warning(f"⚠️ Erreur détection gaps: {e} - Fallback sur chargement complet")
                gap_filling_plan = None
        
        # Calculer les limites optimisées pour indicateurs ULTRA-PRÉCIS
        timeframe_limits = {
            '1m': 2000,   # 33.3 heures → Intraday ultra-précis
            '5m': 1500,   # 5.2 jours → Pattern detection parfait
            '15m': 1000,  # 10.4 jours → Swing trading optimal  
            '1h': 720,    # 30 jours → Tendance mensuelle complète
            '4h': 500     # 83.3 jours → Cycle long terme
        }
        
        total_expected = 0
        total_loaded = 0
        total_published = 0
        
        # Si on a un plan de remplissage de gaps, l'utiliser
        if gap_filling_plan:
            for symbol, timeframe_periods in gap_filling_plan.items():
                for timeframe, periods in timeframe_periods.items():
                    for start_time, end_time in periods:
                        try:
                            # Charger uniquement les données pour cette période de gap
                            historical_data = await self._fetch_historical_klines_for_period(
                                symbol, timeframe, start_time, end_time
                            )
                            
                            if historical_data:
                                loaded_count = len(historical_data)
                                total_loaded += loaded_count
                                logger.info(f"📊 {symbol} {timeframe}: Gap {start_time} → {end_time} rempli ({loaded_count} points)")
                                
                                # Enrichir et publier sur Kafka
                                if kafka_producer and historical_data:
                                    try:
                                        enriched_historical = await self._enrich_historical_batch(
                                            historical_data, symbol, timeframe
                                        )
                                        
                                        for enriched_point in enriched_historical:
                                            enriched_point['symbol'] = symbol
                                            enriched_point['timeframe'] = timeframe
                                            kafka_producer.publish_market_data(enriched_point, symbol)
                                            total_published += 1
                                            
                                    except Exception as e:
                                        logger.error(f"❌ Erreur enrichissement gap {symbol} {timeframe}: {e}")
                                        
                        except Exception as e:
                            logger.error(f"❌ Erreur remplissage gap {symbol} {timeframe} {start_time}: {e}")
        else:
            # Mode classique: charger toutes les données
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
                                    for i, enriched_point in enumerate(enriched_historical):
                                        try:
                                            # Assurer que les champs symbol et timeframe sont présents pour la génération automatique du topic
                                            enriched_point['symbol'] = symbol  
                                            enriched_point['timeframe'] = timeframe
                                            
                                            # Debug: Log d'un échantillon pour voir ce qui est publié
                                            if i == len(enriched_historical) - 1:  # Dernier point
                                                indicators_in_point = [k for k in enriched_point.keys() if k not in ['symbol', 'interval', 'start_time', 'open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume', 'is_closed', 'is_historical', 'enhanced', 'ultra_enriched', 'timeframe']]
                                                logger.error(f"🔍 KAFKA PUBLISH {symbol} {timeframe}: {len(indicators_in_point)} indicateurs dans le point")
                                                logger.error(f"🔍 Indicateurs dans le point: {indicators_in_point}")
                                            
                                            kafka_producer.publish_market_data(enriched_point, symbol)
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
    
    async def _fetch_historical_klines_for_period(self, 
                                                  symbol: str, 
                                                  timeframe: str, 
                                                  start_time: datetime, 
                                                  end_time: datetime) -> List:
        """
        Récupère les klines historiques pour une période spécifique (gap filling)
        """
        try:
            # Convertir les datetime en timestamps milliseconds
            start_timestamp = int(start_time.timestamp() * 1000)
            end_timestamp = int(end_time.timestamp() * 1000)
            
            # Calculer combien de klines maximum pour cette période
            interval_seconds = {
                '1m': 60, '5m': 300, '15m': 900, '1h': 3600, '4h': 14400
            }.get(timeframe, 60)
            
            duration_seconds = (end_time - start_time).total_seconds()
            max_klines = min(1000, int(duration_seconds / interval_seconds) + 10)  # +10 de marge
            
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'startTime': start_timestamp,
                'endTime': end_timestamp,
                'limit': max_klines
            }
            
            url = f"{self.base_url}{self.klines_endpoint}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status == 200:
                        klines = await response.json()
                        logger.debug(f"📚 Gap fill {symbol} {timeframe}: {len(klines)} klines récupérées "
                                   f"pour {start_time} → {end_time}")
                        return klines
                    else:
                        logger.error(f"❌ API Binance error {response.status} pour gap {symbol} {timeframe}")
                        return []
                        
        except Exception as e:
            logger.error(f"❌ Erreur fetch gap {symbol} {timeframe} {start_time}: {e}")
            return []

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
        Enrichit les données historiques avec calcul correct des indicateurs.
        APPROCHE CORRIGÉE: Calcule les indicateurs progressivement pour chaque point.
        
        Args:
            klines: Liste des klines historiques brutes (triées chronologiquement)
            symbol: Symbole de trading
            timeframe: Timeframe des données
            
        Returns:
            Liste des points enrichis avec indicateurs calculés correctement
        """
        from shared.src.technical_indicators import indicator_cache, indicators, TechnicalIndicators
        import numpy as np
        
        enriched_points = []
        
        try:
            logger.info(f"🔄 Enrichissement de {len(klines)} points historiques pour {symbol} {timeframe}")
            
            # Extraire TOUTES les données OHLCV
            prices = []
            highs = []
            lows = []
            volumes = []
            timestamps = []
            
            for kline in klines:
                prices.append(float(kline[4]))    # close
                highs.append(float(kline[2]))     # high
                lows.append(float(kline[3]))      # low
                volumes.append(float(kline[5]))   # volume
                timestamps.append(kline[0])       # timestamp
            
            logger.info(f"📊 Calcul arrays d'indicateurs sur {len(prices)} points pour {symbol} {timeframe}")
            
            # Calculer TOUS les indicateurs en une fois avec les arrays complets
            tech_indicators = TechnicalIndicators()
            indicators_arrays = tech_indicators.calculate_all_indicators_array(highs, lows, prices, volumes)
            
            logger.info(f"✅ {len(indicators_arrays)} indicateurs calculés en arrays")
            
            # Traiter chaque kline avec les indicateurs correspondants
            for i, kline in enumerate(klines):
                try:
                    # Extraire les données OHLCV
                    timestamp = kline[0]
                    open_price = float(kline[1])
                    high_price = float(kline[2])
                    low_price = float(kline[3])
                    close_price = float(kline[4])
                    volume = float(kline[5])
                    close_time = kline[6]
                    
                    # Créer le point enrichi de base
                    enriched_point = {
                        'symbol': symbol,
                        'interval': timeframe,
                        'start_time': timestamp,
                        'open_time': timestamp,
                        'close_time': close_time,
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'volume': volume,
                        'is_closed': True,
                        'is_historical': True,
                        'enhanced': True,
                        'ultra_enriched': True
                    }
                    
                    # Ajouter les indicateurs correspondant à l'index i
                    indicators_count = 0
                    if indicators_arrays:
                        for indicator_name, indicator_array in indicators_arrays.items():
                            if i < len(indicator_array):
                                value = indicator_array[i]
                                # Vérifier que la valeur est valide (pas NaN ou inf)
                                if not (np.isnan(value) or np.isinf(value)):
                                    enriched_point[indicator_name] = float(value)
                                    indicators_count += 1
                                    # Sauvegarder le dernier indicateur dans le cache pour continuité
                                    if i == len(klines) - 1:  # Dernière kline
                                        indicator_cache.set(symbol, timeframe, indicator_name, float(value))
                    
                    enriched_points.append(enriched_point)
                    
                    # Log de progression
                    if (i + 1) % max(1, len(klines) // 10) == 0 or i == len(klines) - 1:
                        logger.debug(f"  ⚡ Point {i+1}/{len(klines)}: {indicators_count} indicateurs ajoutés")
                        
                except Exception as e:
                    logger.warning(f"Erreur traitement point {i} pour {symbol} {timeframe}: {e}")
                    # En cas d'erreur sur un point, ajouter au moins les données de base
                    basic_point = {
                        'symbol': symbol,
                        'interval': timeframe,
                        'start_time': kline[0],
                        'open_time': kline[0],
                        'close_time': kline[6],
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5]),
                        'is_closed': True,
                        'is_historical': True
                    }
                    enriched_points.append(basic_point)
            
            # Log final avec statistiques détaillées
            if enriched_points:
                sample_indicators = len([k for k in enriched_points[-1].keys() 
                                       if k not in ['symbol', 'interval', 'start_time', 'open_time', 'close_time', 
                                                   'open', 'high', 'low', 'close', 'volume', 'is_closed', 
                                                   'is_historical', 'enhanced', 'ultra_enriched']])
                logger.info(f"✅ {symbol} {timeframe}: {len(enriched_points)} points avec {sample_indicators} indicateurs chacun")
            else:
                logger.warning(f"⚠️ Aucun point enrichi généré pour {symbol} {timeframe}")
            
            return enriched_points
            
        except Exception as e:
            logger.error(f"❌ Erreur enrichissement séquentiel {symbol} {timeframe}: {e}")
            # En cas d'erreur globale, retourner au moins les données de base
            basic_points = []
            for kline in klines:
                basic_point = {
                    'symbol': symbol,
                    'interval': timeframe,
                    'start_time': kline[0],
                    'open_time': kline[0],
                    'close_time': kline[6],
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'is_closed': True,
                    'is_historical': True
                }
                basic_points.append(basic_point)
            return basic_points

    async def _calculate_point_indicators(self, close_price: float, high_price: float, low_price: float, volume: float, symbol: str, timeframe: str) -> Dict:
        """
        Calcule les indicateurs pour un point unique de manière séquentielle.
        Maintient l'état des indicateurs incrémentaux dans le cache global.
        
        Args:
            close_price: Prix de fermeture
            high_price: Prix le plus haut
            low_price: Prix le plus bas
            volume: Volume
            symbol: Symbole de trading
            timeframe: Timeframe des données
            
        Returns:
            Dictionnaire avec tous les indicateurs calculés
        """
        from shared.src.technical_indicators import indicators
        
        try:
            # Créer des listes avec ce point unique pour compatibilité avec les fonctions existantes
            prices = [close_price]
            highs = [high_price]
            lows = [low_price]
            volumes = [volume]
            
            # Données de base
            point_indicators = {
                'symbol': symbol,
                'timeframe': timeframe,
                'enhanced': True,
                'ultra_enriched': True,
                'close': close_price,
                'volume': volume,
                'timestamp': time.time(),
                'last_update': time.time()
            }
            
            # **CRITIQUE**: Calcul incrémental pour EMA/MACD (maintient la continuité)
            incremental_indicators = self._calculate_smooth_indicators_ultra(symbol, timeframe, prices, highs, lows, volumes)
            
            # Calcul des autres indicateurs avec les données actuelles
            # Note: Pour un seul point, certains indicateurs ne peuvent pas être calculés
            # mais les incrémentaux (EMA/MACD) peuvent l'être grâce au cache
            if incremental_indicators:
                point_indicators.update(incremental_indicators)
                
                # Calculs simples possibles avec un point
                point_indicators['rsi_14'] = incremental_indicators.get('rsi_14', 50.0)  # Valeur par défaut neutre
                
                # **NOUVEAU**: Sauvegarder explicitement les indicateurs dans le cache persistant
                from shared.src.technical_indicators import indicator_cache
                for indicator_name, indicator_value in incremental_indicators.items():
                    # Sauvegarder seulement les indicateurs numériques valides
                    if isinstance(indicator_value, (int, float)) and not (isinstance(indicator_value, float) and indicator_value != indicator_value):
                        indicator_cache.set(symbol, timeframe, indicator_name, indicator_value)
                
                logger.debug(f"Point {symbol} {timeframe}: {len(point_indicators)} indicateurs calculés et sauvegardés")
            
            return point_indicators
            
        except Exception as e:
            logger.warning(f"Erreur calcul indicateurs point {symbol} {timeframe}: {e}")
            # Retourner au moins les données de base
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'close': close_price,
                'volume': volume
            }

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

    def _calculate_smooth_indicators_ultra(self, symbol: str, timeframe: str, prices: List[float], highs: List[float], lows: List[float], volumes: List[float]) -> Dict:
        """
        🚀 NOUVEAU : Calcule EMA/MACD avec séries optimisées pour éviter les dents de scie.
        Utilise le module technical_indicators pour un calcul lisse.
        
        Args:
            symbol: Symbole tradé
            timeframe: Intervalle de temps  
            prices: Liste des prix de clôture
            highs: Liste des prix hauts
            lows: Liste des prix bas
            volumes: Liste des volumes
            
        Returns:
            Dict avec indicateurs EMA/MACD lisses
        """
        result = {}
        
        try:
            if len(prices) < 12:  # Pas assez de données
                return {}
            
            # 📈 EMA 12, 26, 50 avec séries optimisées (évite dents de scie)
            for period in [12, 26, 50]:
                if len(prices) >= period:
                    ema_key = f'ema_{period}'
                    
                    # Utiliser la méthode série du module indicators
                    ema_series = indicators.calculate_ema_series(prices, period)
                    
                    # Prendre la dernière valeur calculée (la plus récente)
                    if ema_series and len(ema_series) > 0 and ema_series[-1] is not None:
                        result[ema_key] = float(ema_series[-1])
            
            # 📊 MACD avec séries optimisées  
            if len(prices) >= 26:  # Assez de données pour MACD
                macd_data = indicators.calculate_macd_series(prices)
                
                if macd_data and isinstance(macd_data, dict):
                    # Prendre les dernières valeurs calculées
                    if 'macd_line' in macd_data and len(macd_data['macd_line']) > 0 and macd_data['macd_line'][-1] is not None:
                        result['macd_line'] = float(macd_data['macd_line'][-1])
                    if 'macd_signal' in macd_data and len(macd_data['macd_signal']) > 0 and macd_data['macd_signal'][-1] is not None:
                        result['macd_signal'] = float(macd_data['macd_signal'][-1])
                    if 'macd_histogram' in macd_data and len(macd_data['macd_histogram']) > 0 and macd_data['macd_histogram'][-1] is not None:
                        result['macd_histogram'] = float(macd_data['macd_histogram'][-1])
            
            if result:
                logger.debug(f"🚀 [ULTRA] EMA/MACD lisses calculés pour {symbol} {timeframe}: "
                            f"EMA12={result.get('ema_12', 0):.4f}, "
                            f"MACD={result.get('macd_line', 0):.4f}")
            
        except Exception as e:
            logger.error(f"❌ [ULTRA] Erreur calcul indicateurs lisses {symbol} {timeframe}: {e}")
            # En cas d'erreur, retourner un dict vide (fallback vers calcul traditionnel)
            result = {}
        
        return result

# Point d'entrée pour test standalone
async def main():
    """Test standalone de l'UltraDataFetcher"""
    fetcher = UltraDataFetcher()
    await fetcher.start()

if __name__ == "__main__":
    asyncio.run(main())