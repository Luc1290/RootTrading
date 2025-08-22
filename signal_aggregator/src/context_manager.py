"""
Module de gestion du contexte de marché pour la validation des signaux.
Récupère et structure les données nécessaires depuis la base de données.
"""

import logging
from typing import Dict, Any, Optional, List
import psycopg2
from psycopg2.extras import RealDictCursor
from field_converters import FieldConverter

logger = logging.getLogger(__name__)


class ContextManager:
    """Gestionnaire du contexte de marché pour la validation."""
    
    def __init__(self, db_connection):
        """
        Initialise le gestionnaire de contexte.
        
        Args:
            db_connection: Connexion à la base de données
        """
        self.db_connection = db_connection
        
        # Cache pour optimiser les requêtes répétées
        self.context_cache = {}
        self.cache_ttl = 60  # Durée de vie du cache en secondes
        
    def get_unified_market_regime(self, symbol: str) -> Dict[str, Any]:
        """
        Récupère le régime de marché unifié basé sur le timeframe de référence (15m).
        
        Args:
            symbol: Symbole à analyser
            
        Returns:
            Dict contenant le régime unifié et ses métadonnées
        """
        # MODIFIÉ: Utiliser 1m pour réactivité maximale (détection pump immédiate)
        # Le bruit est déjà filtré car on n'émet pas de signaux 1m
        reference_timeframe = "1m"  # Était 15m, trop lent pour crypto
        cache_key = f"regime_unified_{symbol}"
        
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Récupérer le régime du timeframe de référence (plus récent)
                cursor.execute("""
                    SELECT market_regime, regime_strength, regime_confidence, 
                           directional_bias, volatility_regime, time
                    FROM analyzer_data 
                    WHERE symbol = %s AND timeframe = %s
                    ORDER BY time DESC 
                    LIMIT 1
                """, (symbol, reference_timeframe))
                
                regime_data = cursor.fetchone()
                
                if regime_data:
                    return {
                        'market_regime': regime_data['market_regime'],
                        'regime_strength': regime_data['regime_strength'], 
                        'regime_confidence': regime_data['regime_confidence'],
                        'directional_bias': regime_data['directional_bias'],
                        'volatility_regime': regime_data['volatility_regime'],
                        'regime_source': f"{reference_timeframe}_unified",
                        'regime_timestamp': regime_data['time']
                    }
                else:
                    # Fallback si pas de données 1m
                    logger.warning(f"Pas de régime 1m pour {symbol}, fallback sur 3m")
                    cursor.execute("""
                        SELECT market_regime, regime_strength, regime_confidence,
                               directional_bias, volatility_regime, time
                        FROM analyzer_data 
                        WHERE symbol = %s AND timeframe = '1m'
                        ORDER BY time DESC 
                        LIMIT 1
                    """, (symbol,))
                    
                    fallback_data = cursor.fetchone()
                    if fallback_data:
                        return {
                            'market_regime': fallback_data['market_regime'],
                            'regime_strength': fallback_data['regime_strength'], 
                            'regime_confidence': fallback_data['regime_confidence'],
                            'directional_bias': fallback_data['directional_bias'],
                            'volatility_regime': fallback_data['volatility_regime'],
                            'regime_source': "1m_fallback",
                            'regime_timestamp': fallback_data['time']
                        }
                        
                return {
                    'market_regime': 'UNKNOWN',
                    'regime_strength': 0.5,
                    'regime_confidence': 50.0,
                    'directional_bias': 'NEUTRAL',
                    'volatility_regime': 'normal',
                    'regime_source': 'default',
                    'regime_timestamp': None
                }
                
        except Exception as e:
            logger.error(f"Erreur récupération régime unifié {symbol}: {e}")
            return {
                'market_regime': 'UNKNOWN',
                'regime_strength': 0.5,
                'regime_confidence': 50.0,
                'directional_bias': 'NEUTRAL',
                'volatility_regime': 'normal',
                'regime_source': 'error',
                'regime_timestamp': None
            }

    def get_market_context(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Récupère le contexte de marché complet pour un symbole et timeframe.
        
        Args:
            symbol: Symbole à analyser (ex: BTCUSDC)
            timeframe: Timeframe à analyser (ex: 1m, 5m, 15m, 1h)
            
        Returns:
            Dict contenant tout le contexte nécessaire à la validation
        """
        cache_key = f"{symbol}_{timeframe}"
        
        # Vérifier le cache (pour optimiser les performances)
        if cache_key in self.context_cache:
            cached_context = self.context_cache[cache_key]
            # TODO: Implémenter vérification TTL si nécessaire
            
        try:
            # Récupérer les composants du contexte
            ohlcv_data = self._get_ohlcv_data(symbol, timeframe)
            indicators = self._get_indicators(symbol, timeframe)
            market_structure = self._get_market_structure(symbol, timeframe)
            volume_profile = self._get_volume_profile(symbol, timeframe)
            multi_timeframe = self._get_multi_timeframe_context(symbol)
            correlation_data = self._get_correlation_context(symbol)
            
            # Récupérer le régime unifié (15m de référence) 
            unified_regime = self.get_unified_market_regime(symbol)
            
            # Construire le contexte avec champs racine pour compatibility
            context = {
                'symbol': symbol,
                'timeframe': timeframe,
                'ohlcv_data': ohlcv_data,
                'indicators': indicators,
                'market_structure': market_structure,
                'volume_profile': volume_profile,
                'multi_timeframe': multi_timeframe,
                'correlation_data': correlation_data,
                'unified_regime': unified_regime  # Régime unifié pour tous les signaux
            }
            
            # Exposer les champs critiques au niveau racine pour compatibilité validators
            if indicators:
                # Sauvegarder le régime du timeframe original avant update
                original_regime = indicators.get('market_regime')
                context.update(indicators)  # Tous les indicateurs au niveau racine
                # Ajouter le régime original sous un autre nom pour référence
                if original_regime:
                    context['timeframe_regime'] = original_regime
            
            # OVERRIDE: FORCER le régime unifié après l'update des indicators
            # IMPORTANT: Doit être APRÈS context.update(indicators) pour ne pas être écrasé
            if unified_regime:
                context.update({
                    'market_regime': unified_regime['market_regime'],          # Régime unifié (15m) - FORCE L'OVERRIDE
                    'regime_strength': unified_regime['regime_strength'],      
                    'regime_confidence': unified_regime['regime_confidence'],
                    'directional_bias': unified_regime['directional_bias'],
                    'volatility_regime': unified_regime['volatility_regime'],
                    'regime_source': unified_regime['regime_source']           # Pour debugging
                })
            
            if market_structure and 'current_price' in market_structure:
                context['current_price'] = market_structure['current_price']  # Prix actuel au niveau racine
            
            # Fallback pour current_price si absent
            if 'current_price' not in context and ohlcv_data:
                context['current_price'] = ohlcv_data[-1]['close'] if ohlcv_data else 0
            
            # Mise en cache du contexte
            self.context_cache[cache_key] = context
            
            return context
            
        except Exception as e:
            logger.error(f"Erreur récupération contexte {symbol} {timeframe}: {e}")
            return {}
            
    def _get_ohlcv_data(self, symbol: str, timeframe: str, limit: int = 100) -> List[Dict[str, float]]:
        """
        Récupère les données OHLCV historiques.
        
        Args:
            symbol: Symbole
            timeframe: Timeframe
            limit: Nombre de bougies à récupérer
            
        Returns:
            Liste des données OHLCV
        """
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT time, open, high, low, close, volume, quote_asset_volume
                    FROM market_data 
                    WHERE symbol = %s AND timeframe = %s 
                    ORDER BY time DESC 
                    LIMIT %s
                """, (symbol, timeframe, limit))
                
                rows = cursor.fetchall()
                
                return [
                    {
                        'timestamp': row['time'],
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']),
                        'quote_volume': float(row['quote_asset_volume'])
                    }
                    for row in reversed(rows)  # Ordre chronologique
                ]
                
        except Exception as e:
            logger.error(f"Erreur récupération OHLCV {symbol} {timeframe}: {e}")
            return []
            
    def _get_indicators(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Récupère tous les indicateurs techniques pré-calculés.
        
        Args:
            symbol: Symbole
            timeframe: Timeframe
            
        Returns:
            Dict des indicateurs avec conversion de type robuste
        """
        try:
            with self.db_connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM analyzer_data 
                    WHERE symbol = %s AND timeframe = %s 
                    ORDER BY time DESC 
                    LIMIT 1
                """, (symbol, timeframe))
                
                row = cursor.fetchone()
                if not row:
                    return {}
                    
                # Conversion des indicateurs via FieldConverter
                raw_indicators = {}
                for key, value in row.items():
                    if key not in ['time', 'symbol', 'timeframe', 'analysis_timestamp', 'analyzer_version']:
                        raw_indicators[key] = value
                
                # Utiliser le convertisseur pour harmoniser les types
                indicators = FieldConverter.convert_indicators(raw_indicators)
                
                # Log temporaire pour debug
                logger.debug(f"Indicateurs récupérés pour {symbol} {timeframe}: {len(indicators)} champs")
                if 'atr_14' in indicators:
                    logger.debug(f"ATR_14 trouvé: {indicators['atr_14']}")
                if 'atr_percentile' in indicators:
                    logger.debug(f"ATR_percentile trouvé: {indicators['atr_percentile']}")
                
                return indicators
                
        except Exception as e:
            logger.error(f"Erreur récupération indicateurs {symbol} {timeframe}: {e}")
            return {}
            
    def _get_market_structure(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Analyse la structure de marché (support/résistance, pivots, etc.).
        
        Args:
            symbol: Symbole
            timeframe: Timeframe
            
        Returns:
            Dict de la structure de marché
        """
        try:
            # Récupérer les données de prix récentes pour analyser la structure
            ohlcv_data = self._get_ohlcv_data(symbol, timeframe, 50)
            
            if not ohlcv_data:
                return {}
                
            # Extraction des prix pour analyse
            highs = [candle['high'] for candle in ohlcv_data]
            lows = [candle['low'] for candle in ohlcv_data]
            closes = [candle['close'] for candle in ohlcv_data]
            
            current_price = closes[-1] if closes else 0
            
            # Calculs de structure basiques
            recent_high = max(highs[-20:]) if len(highs) >= 20 else max(highs)
            recent_low = min(lows[-20:]) if len(lows) >= 20 else min(lows)
            
            structure: Dict[str, Any] = {
                'current_price': current_price,
                'recent_high': recent_high,
                'recent_low': recent_low,
                'price_range': recent_high - recent_low,
                'distance_to_high': recent_high - current_price,
                'distance_to_low': current_price - recent_low,
                'range_position': ((current_price - recent_low) / (recent_high - recent_low)) if recent_high != recent_low else 0.5
            }
            
            # Détection de niveaux psychologiques simples
            psychological_levels = self._find_psychological_levels(current_price)
            structure['psychological_levels'] = psychological_levels
            
            return structure
            
        except Exception as e:
            logger.error(f"Erreur analyse structure marché {symbol} {timeframe}: {e}")
            return {}
            
    def _get_volume_profile(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Analyse le profil de volume.
        
        Args:
            symbol: Symbole
            timeframe: Timeframe
            
        Returns:
            Dict du profil de volume
        """
        try:
            ohlcv_data = self._get_ohlcv_data(symbol, timeframe, 50)
            
            if not ohlcv_data:
                return {}
                
            volumes = [candle['volume'] for candle in ohlcv_data]
            quote_volumes = [candle['quote_volume'] for candle in ohlcv_data]
            
            # Calculs de volume
            avg_volume = sum(volumes) / len(volumes) if volumes else 0
            current_volume = volumes[-1] if volumes else 0
            
            volume_profile = {
                'current_volume': current_volume,
                'average_volume': avg_volume,
                'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1,
                'total_quote_volume': sum(quote_volumes),
                'volume_trend': self._calculate_volume_trend(volumes)
            }
            
            return volume_profile
            
        except Exception as e:
            logger.error(f"Erreur profil volume {symbol} {timeframe}: {e}")
            return {}
            
    def _get_multi_timeframe_context(self, symbol: str) -> Dict[str, Any]:
        """
        Récupère le contexte multi-timeframe.
        
        Args:
            symbol: Symbole
            
        Returns:
            Dict du contexte multi-timeframe
        """
        try:
            timeframes = ['1m', '5m', '15m', '1h']
            mtf_context = {}
            
            for tf in timeframes:
                indicators = self._get_indicators(symbol, tf)
                if indicators:
                    mtf_context[tf] = {
                        'trend_direction': indicators.get('directional_bias'),
                        'trend_strength': indicators.get('trend_strength'),
                        'momentum_score': indicators.get('momentum_score'),
                        'market_regime': indicators.get('market_regime')
                    }
                    
            return mtf_context
            
        except Exception as e:
            logger.error(f"Erreur contexte multi-timeframe {symbol}: {e}")
            return {}
            
    def _get_correlation_context(self, symbol: str) -> Dict[str, Any]:
        """
        Récupère le contexte de corrélation avec d'autres actifs.
        
        Args:
            symbol: Symbole
            
        Returns:
            Dict du contexte de corrélation
        """
        try:
            # Pour l'instant, retourner un contexte basique
            # Plus tard, on pourra implémenter des corrélations réelles
            correlation_context = {
                'major_pairs_sentiment': 'neutral',
                'sector_correlation': 'neutral',
                'market_wide_sentiment': 'neutral'
            }
            
            return correlation_context
            
        except Exception as e:
            logger.error(f"Erreur contexte corrélation {symbol}: {e}")
            return {}
            
    def _find_psychological_levels(self, price: float) -> List[float]:
        """
        Trouve les niveaux psychologiques proches du prix actuel.
        
        Args:
            price: Prix actuel
            
        Returns:
            Liste des niveaux psychologiques
        """
        try:
            levels: List[float] = []
            
            # Niveaux ronds basés sur la magnitude du prix
            if price >= 1000:
                # Pour les prix élevés, utiliser des niveaux de 100
                base = float(int(price / 100) * 100)
                levels.extend([base, base + 100.0, base - 100.0])
            elif price >= 100:
                # Pour les prix moyens, utiliser des niveaux de 10
                base = float(int(price / 10) * 10)
                levels.extend([base, base + 10.0, base - 10.0])
            elif price >= 10:
                # Pour les prix bas, utiliser des niveaux de 1
                base = float(int(price))
                levels.extend([base, base + 1.0, base - 1.0])
            else:
                # Pour les très petits prix, utiliser des décimales
                base = round(price, 1)
                levels.extend([base, base + 0.1, base - 0.1])
                
            # Filtrer les niveaux négatifs et trier
            levels = [level for level in levels if level > 0]
            levels.sort()
            
            return levels
            
        except Exception as e:
            logger.error(f"Erreur niveaux psychologiques: {e}")
            return []
            
    def _calculate_volume_trend(self, volumes: List[float]) -> str:
        """
        Calcule la tendance du volume.
        
        Args:
            volumes: Liste des volumes
            
        Returns:
            Tendance du volume ('increasing', 'decreasing', 'stable')
        """
        try:
            if len(volumes) < 5:
                return 'stable'
                
            # Comparer les 5 derniers volumes avec les 5 précédents
            recent_avg = sum(volumes[-5:]) / 5
            previous_avg = sum(volumes[-10:-5]) / 5
            
            if recent_avg > previous_avg * 1.2:
                return 'increasing'
            elif recent_avg < previous_avg * 0.8:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f"Erreur calcul tendance volume: {e}")
            return 'stable'
            
    def clear_cache(self):
        """Vide le cache du contexte."""
        self.context_cache.clear()
        logger.info("Cache contexte vidé")
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Récupère les statistiques du cache.
        
        Returns:
            Dict des statistiques du cache
        """
        return {
            'cache_size': len(self.context_cache),
            'cached_symbols': list(set(key.split('_')[0] for key in self.context_cache.keys())),
            'cache_ttl': self.cache_ttl
        }