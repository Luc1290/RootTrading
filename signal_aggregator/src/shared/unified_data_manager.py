#!/usr/bin/env python3
"""
Gestionnaire unifié de données pour assurer la cohérence entre tous les modules.
Centralise l'accès aux données de marché pour éviter les incohérences.
"""
import logging
from typing import Dict, Optional, List, Any, Tuple
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass
class UnifiedMarketData:
    """Structure unifiée pour les données de marché"""
    symbol: str
    timeframe: str
    timestamp: datetime
    
    # OHLCV
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    # Indicateurs techniques
    rsi_14: Optional[float] = None
    adx_14: Optional[float] = None
    plus_di: Optional[float] = None
    minus_di: Optional[float] = None
    
    # EMAs
    ema_7: Optional[float] = None
    ema_26: Optional[float] = None
    ema_99: Optional[float] = None
    
    # SMAs
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    
    # MACD
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    
    # Bollinger Bands
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_width: Optional[float] = None
    bb_position: Optional[float] = None
    
    # Volume
    volume_ratio: Optional[float] = None
    avg_volume_20: Optional[float] = None
    
    # Autres
    atr_14: Optional[float] = None
    momentum_10: Optional[float] = None
    
    # Source et fraîcheur
    data_source: str = "unified"
    is_fresh: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour compatibilité"""
        data = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'data_source': self.data_source,
            'is_fresh': self.is_fresh
        }
        
        # Ajouter tous les indicateurs non-None
        indicators = [
            'rsi_14', 'adx_14', 'plus_di', 'minus_di',
            'ema_7', 'ema_26', 'ema_99', 'sma_20', 'sma_50',
            'macd_line', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
            'volume_ratio', 'avg_volume_20', 'atr_14', 'momentum_10'
        ]
        
        for indicator in indicators:
            value = getattr(self, indicator)
            if value is not None:
                data[indicator] = float(value) if isinstance(value, (Decimal, int)) else value
                
        return data


class UnifiedDataManager:
    """Gestionnaire centralisé pour toutes les données de marché"""
    
    def __init__(self, db_manager=None, redis_client=None):
        self.db_manager = db_manager
        self.redis_client = redis_client
        self._cache = {}  # Cache en mémoire pour éviter les requêtes répétées
        self._cache_ttl = 5  # TTL du cache en secondes
        
    async def get_latest_market_data(
        self, 
        symbol: str, 
        timeframe: str = "1m",
        prefer_source: str = "db"  # "db", "redis", ou "auto"
    ) -> Optional[UnifiedMarketData]:
        """
        Récupère les dernières données de marché de manière cohérente.
        
        Args:
            symbol: Symbole crypto (ex: BTCUSDC)
            timeframe: Timeframe des données (1m, 5m, etc)
            prefer_source: Source préférée (db pour données enrichies, redis pour temps réel)
            
        Returns:
            UnifiedMarketData ou None si pas de données
        """
        try:
            # Vérifier le cache d'abord
            cache_key = f"{symbol}:{timeframe}"
            cached_data = self._get_from_cache(cache_key)
            if cached_data:
                logger.debug(f"✅ Données du cache pour {symbol}:{timeframe}")
                return cached_data
            
            # Stratégie de récupération selon la préférence
            data = None
            
            if prefer_source == "db" or prefer_source == "auto":
                # Essayer la DB en premier (données enrichies)
                data = await self._get_from_db(symbol, timeframe)
                
            if not data and (prefer_source == "redis" or prefer_source == "auto"):
                # Fallback vers Redis
                data = await self._get_from_redis(symbol, timeframe)
                
            if data:
                # Mettre en cache
                self._cache[cache_key] = {
                    'data': data,
                    'timestamp': datetime.now()
                }
                logger.info(f"📊 Données unifiées récupérées pour {symbol}:{timeframe} depuis {data.data_source}")
                
            return data
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération données unifiées pour {symbol}: {e}")
            return None
    
    async def _get_from_db(self, symbol: str, timeframe: str) -> Optional[UnifiedMarketData]:
        """Récupère les données depuis la base de données"""
        try:
            if not self.db_manager:
                return None
                
            # Récupérer les données enrichies
            data = await self.db_manager.get_enriched_market_data(
                symbol=symbol,
                interval=timeframe,
                limit=1,
                include_indicators=True
            )
            
            if not data or len(data) == 0:
                return None
                
            latest = data[-1]
            
            # Convertir en UnifiedMarketData
            return UnifiedMarketData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=latest.get('time', datetime.now()),
                open=float(latest.get('open', 0)),
                high=float(latest.get('high', 0)),
                low=float(latest.get('low', 0)),
                close=float(latest.get('close', 0)),
                volume=float(latest.get('volume', 0)),
                
                # Indicateurs
                rsi_14=self._safe_float(latest.get('rsi_14')),
                adx_14=self._safe_float(latest.get('adx_14')),
                plus_di=self._safe_float(latest.get('plus_di')),
                minus_di=self._safe_float(latest.get('minus_di')),
                
                # EMAs
                ema_7=self._safe_float(latest.get('ema_7')),
                ema_26=self._safe_float(latest.get('ema_26')),
                ema_99=self._safe_float(latest.get('ema_99')),
                
                # SMAs
                sma_20=self._safe_float(latest.get('sma_20')),
                sma_50=self._safe_float(latest.get('sma_50')),
                
                # MACD
                macd_line=self._safe_float(latest.get('macd_line')),
                macd_signal=self._safe_float(latest.get('macd_signal')),
                macd_histogram=self._safe_float(latest.get('macd_histogram')),
                
                # Bollinger
                bb_upper=self._safe_float(latest.get('bb_upper')),
                bb_middle=self._safe_float(latest.get('bb_middle')),
                bb_lower=self._safe_float(latest.get('bb_lower')),
                bb_width=self._safe_float(latest.get('bb_width')),
                bb_position=self._safe_float(latest.get('bb_position')),
                
                # Volume
                volume_ratio=self._safe_float(latest.get('volume_ratio')),
                avg_volume_20=self._safe_float(latest.get('avg_volume_20')),
                
                # Autres
                atr_14=self._safe_float(latest.get('atr_14')),
                momentum_10=self._safe_float(latest.get('momentum_10')),
                
                data_source="db",
                is_fresh=True
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération DB pour {symbol}: {e}")
            return None
    
    async def _get_from_redis(self, symbol: str, timeframe: str) -> Optional[UnifiedMarketData]:
        """Récupère les données depuis Redis"""
        try:
            if not self.redis_client:
                return None
                
            import json
            
            # Clé Redis standard
            key = f"market_data:{symbol}:{timeframe}"
            data = self.redis_client.get(key)
            
            if not data:
                return None
                
            # Parser les données
            if isinstance(data, str):
                parsed = json.loads(data)
            else:
                parsed = data
                
            # Convertir en UnifiedMarketData
            return UnifiedMarketData(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.fromtimestamp(parsed.get('timestamp', 0)),
                open=float(parsed.get('open', parsed.get('close', 0))),
                high=float(parsed.get('high', parsed.get('close', 0))),
                low=float(parsed.get('low', parsed.get('close', 0))),
                close=float(parsed.get('close', 0)),
                volume=float(parsed.get('volume', 0)),
                
                # Indicateurs (peuvent ne pas être présents dans Redis)
                rsi_14=self._safe_float(parsed.get('rsi_14') or parsed.get('rsi')),
                adx_14=self._safe_float(parsed.get('adx_14') or parsed.get('adx')),
                
                # Volume
                volume_ratio=self._safe_float(parsed.get('volume_ratio')),
                
                data_source="redis",
                is_fresh=True
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération Redis pour {symbol}: {e}")
            return None
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Conversion sûre en float"""
        if value is None:
            return None
        try:
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, Decimal):
                return float(value)
            elif isinstance(value, str):
                return float(value)
            else:
                return None
        except:
            return None
    
    def _get_from_cache(self, cache_key: str) -> Optional[UnifiedMarketData]:
        """Récupère depuis le cache si pas expiré"""
        if cache_key not in self._cache:
            return None
            
        cached = self._cache[cache_key]
        age = (datetime.now() - cached['timestamp']).total_seconds()
        
        if age > self._cache_ttl:
            # Cache expiré
            del self._cache[cache_key]
            return None
            
        return cached['data']
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Vide le cache"""
        if symbol:
            # Vider seulement pour un symbole
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{symbol}:")]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            # Vider tout le cache
            self._cache.clear()
    
    async def get_synchronized_data(
        self, 
        symbol: str,
        timeframes: List[str] = ["1m", "5m", "15m"]
    ) -> Dict[str, UnifiedMarketData]:
        """
        Récupère des données synchronisées pour plusieurs timeframes.
        Assure que toutes les données viennent de la même source.
        """
        result = {}
        
        # Déterminer la meilleure source
        # Essayer DB d'abord pour le 1m
        test_data = await self.get_latest_market_data(symbol, "1m", prefer_source="db")
        preferred_source = "db" if test_data and test_data.data_source == "db" else "redis"
        
        # Récupérer toutes les données de la même source
        for tf in timeframes:
            data = await self.get_latest_market_data(symbol, tf, prefer_source=preferred_source)
            if data:
                result[tf] = data
                
        logger.info(f"📊 Données synchronisées récupérées pour {symbol}: {list(result.keys())} depuis {preferred_source}")
        return result


# Instance globale (sera initialisée par signal_aggregator)
unified_data_manager: Optional[UnifiedDataManager] = None


def get_unified_data_manager() -> Optional[UnifiedDataManager]:
    """Récupère l'instance globale du gestionnaire"""
    return unified_data_manager


def set_unified_data_manager(manager: UnifiedDataManager):
    """Définit l'instance globale du gestionnaire"""
    global unified_data_manager
    unified_data_manager = manager