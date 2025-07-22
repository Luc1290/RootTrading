#!/usr/bin/env python3
"""
Module pour l'analyse technique et les calculs d'indicateurs.
Contient toutes les méthodes d'analyse technique extraites du signal_aggregator principal.
"""

import logging
import math
from typing import Dict, List, Any, Optional, Union
import sys
import os

# Add path to shared modules BEFORE imports
sys.path.append(os.path.dirname(__file__))

try:
    from .shared.technical_utils import TechnicalCalculators
except ImportError:
    # Fallback pour l'exécution dans le conteneur
    sys.path.append(os.path.join(os.path.dirname(__file__), 'shared'))
    from technical_utils import TechnicalCalculators
try:
    from .shared.redis_utils import RedisManager
except ImportError:
    # Fallback pour l'exécution dans le conteneur
    from redis_utils import RedisManager

logger = logging.getLogger(__name__)


class TechnicalAnalysis:
    """Classe pour l'analyse technique et le calcul d'indicateurs"""
    
    def __init__(self, redis_client, ema_incremental_cache: Optional[Dict] = None):
        self.redis = redis_client
        self.ema_incremental_cache = ema_incremental_cache or {}
    
    async def get_technical_context(self, symbol: str) -> Dict[str, Any]:
        """
        Récupère le contexte technique enrichi pour un symbole
        
        Returns:
            Dictionnaire avec les indicateurs techniques actuels
        """
        try:
            # CORRECTION: Import direct avec chemin complet au lieu de manipulation sys.path
            from shared.src.technical_indicators import TechnicalIndicators
            indicators = TechnicalIndicators()
            
            # Récupérer les données 3m depuis Redis avec utilitaire partagé
            market_data_key = f"market_data:{symbol}:3m"
            data_3m = RedisManager.get_cached_data(self.redis, market_data_key)
            
            context: Dict[str, Any] = {
                'macd': None,
                'obv': None, 
                'roc': None,
                'available': False,
                'symbol': symbol  # Ajouter le symbol pour validate_obv_trend
            }
            
            if not data_3m or not isinstance(data_3m, dict):
                return context
                
            # Extraire les prix historiques
            prices = data_3m.get('prices', [])
            volumes = data_3m.get('volumes', [])
            highs = data_3m.get('highs', [])
            lows = data_3m.get('lows', [])
            
            if len(prices) < 30:  # Minimum pour les calculs
                return context
            
            # Utiliser highs/lows pour calculer des niveaux de support/résistance
            if len(highs) >= 20 and len(lows) >= 20:
                # Support/résistance sur les 20 dernières bougies
                context['resistance_level'] = max(highs[-20:])
                context['support_level'] = min(lows[-20:])
                
                # True Range actuel
                if len(prices) >= 2:
                    current_tr = max(
                        highs[-1] - lows[-1],
                        abs(highs[-1] - prices[-2]),
                        abs(lows[-1] - prices[-2])
                    )
                    context['current_true_range'] = current_tr
            
            # Calculer MACD
            macd_data = indicators.calculate_macd(prices)
            if macd_data['macd_line'] is not None:
                context['macd'] = macd_data
            
            # Calculer OBV approximatif (si volumes disponibles)
            if len(volumes) >= len(prices):
                try:
                    obv_value = indicators.calculate_obv(prices, volumes)
                    if obv_value is not None:
                        context['obv'] = obv_value
                except (ValueError, TypeError, IndexError, ZeroDivisionError) as e:
                    logger.debug(f"Erreur calcul OBV pour {symbol}: {e}")  # OBV optionnel
            
            # Calculer ROC
            try:
                roc_value = indicators.calculate_roc(prices, period=10)
                if roc_value is not None:
                    context['roc'] = roc_value
            except (ValueError, TypeError, IndexError, ZeroDivisionError) as e:
                logger.debug(f"Erreur calcul ROC pour {symbol}: {e}")  # ROC optionnel
            
            # Ajouter le volume ratio si disponible
            volume_ratio = data_3m.get('volume_ratio', 1.0)
            context['volume_ratio'] = volume_ratio
            
            context['available'] = True
            return context
            
        except Exception as e:
            logger.error(f"Erreur récupération contexte technique pour {symbol}: {e}")
            return {'macd': None, 'obv': None, 'roc': None, 'available': False, 'symbol': symbol}
    
    def validate_macd_trend(self, technical_context: Dict[str, Any], expected_trend: str) -> Optional[bool]:
        """
        Valide si le MACD confirme la tendance attendue
        
        Args:
            technical_context: Contexte technique
            expected_trend: 'bullish' ou 'bearish'
            
        Returns:
            True/False si MACD confirme, None si pas de données
        """
        try:
            macd_data = technical_context.get('macd')
            if not macd_data or macd_data.get('macd_line') is None:
                return None
                
            macd_line = macd_data['macd_line']
            macd_signal = macd_data.get('macd_signal')
            macd_histogram = macd_data.get('macd_histogram')
            
            if macd_signal is None:
                return None
            
            if expected_trend == 'bullish':
                # Tendance haussière: MACD au-dessus signal ET histogram positif
                return macd_line > macd_signal and (macd_histogram is None or macd_histogram > 0)
            elif expected_trend == 'bearish':
                # Tendance baissière: MACD en-dessous signal ET histogram négatif
                return macd_line < macd_signal and (macd_histogram is None or macd_histogram < 0)
                
            return None
            
        except Exception as e:
            logger.error(f"Erreur validation MACD: {e}")
            return None
    
    async def get_atr(self, symbol: str) -> Dict[str, float | None]:
        """
        Récupère l'ATR (Average True Range) pour un symbole depuis Redis.
        Utilise la méthode centralisée de TechnicalIndicators.
        
        Args:
            symbol: Symbole de trading
            
        Returns:
            Dict avec atr_percent et atr_value
        """
        try:
            # Récupérer les données 3m depuis Redis avec utilitaire partagé
            market_data_key = f"market_data:{symbol}:3m"
            data_3m = RedisManager.get_cached_data(self.redis, market_data_key)
            
            if not data_3m or not isinstance(data_3m, dict):
                logger.debug(f"Pas de données 3m pour {symbol}, utilisation valeur par défaut ATR")
                return {"atr_percent": 1.0, "atr_value": None}
            
            # Vérifier si ATR est déjà calculé dans les données
            atr_value = data_3m.get('atr_14')
            close_price = data_3m.get('close')
            
            if atr_value and close_price:
                atr_percent = (float(atr_value) / float(close_price)) * 100
                return {
                    "atr_percent": atr_percent,
                    "atr_value": float(atr_value)
                }
            
            # Fallback: utiliser TechnicalIndicators.calculate_atr() si données OHLC disponibles
            prices = data_3m.get('prices', [])
            highs = data_3m.get('highs', [])
            lows = data_3m.get('lows', [])
            
            if len(prices) >= 15 and len(highs) >= 15 and len(lows) >= 15:
                from shared.src.technical_indicators import TechnicalIndicators
                indicators = TechnicalIndicators()
                
                # Utiliser la méthode centralisée calculate_atr
                atr_value = indicators.calculate_atr(highs[-15:], lows[-15:], prices[-15:], period=14)
                if atr_value and prices and not math.isnan(atr_value) and not math.isinf(atr_value):
                    atr_percent = (atr_value / prices[-1]) * 100
                    return {
                        "atr_percent": atr_percent,
                        "atr_value": atr_value
                    }
            
            # Valeur par défaut si calcul impossible
            return {"atr_percent": 1.0, "atr_value": None}
            
        except Exception as e:
            logger.error(f"Erreur récupération ATR pour {symbol}: {e}")
            return {"atr_percent": 1.0, "atr_value": None}
    
    def validate_obv_trend(self, technical_context: Dict[str, Any], side: str) -> Optional[bool]:
        """
        Valide si l'OBV confirme le côté du signal avec analyse de tendance historique
        
        Args:
            technical_context: Contexte technique
            side: 'BUY' ou 'SELL'
            
        Returns:
            True si OBV confirme, False sinon, None si pas de données
        """
        try:
            obv_current = technical_context.get('obv')
            if obv_current is None:
                return None
            
            # Essayer de récupérer l'historique OBV récent
            symbol = technical_context.get('symbol', 'UNKNOWN')
            
            try:
                # Récupérer les dernières données depuis Redis pour calculer la tendance OBV
                import json
                recent_key = f"market_data:{symbol}:recent"
                recent_data = self.redis.lrange(recent_key, 0, 9)  # 10 dernières valeurs
                
                if len(recent_data) >= 3:  # Besoin d'au moins 3 points pour une tendance
                    obv_history = []
                    
                    for data_str in recent_data:
                        try:
                            data = json.loads(data_str)
                            if 'obv' in data and data['obv'] is not None:
                                obv_history.append(float(data['obv']))
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue
                    
                    # Ajouter l'OBV actuel
                    obv_history.append(float(obv_current))
                    
                    if len(obv_history) >= 3:
                        # Calculer la tendance OBV sur les derniers points
                        recent_slope = self._calculate_obv_slope(obv_history[-3:])
                        overall_slope = self._calculate_obv_slope(obv_history)
                        
                        # Validation basée sur la tendance OBV
                        if side == 'BUY':
                            # Pour BUY: OBV doit être en tendance haussière (volume d'achat croissant)
                            obv_confirms = recent_slope > 0 or (overall_slope > 0 and recent_slope >= -0.1)
                            logger.debug(f"📈 OBV validation BUY {symbol}: recent_slope={recent_slope:.2f}, overall_slope={overall_slope:.2f} → {obv_confirms}")
                            return obv_confirms
                        else:  # SELL
                            # Pour SELL: OBV doit être en tendance baissière (volume de vente croissant) 
                            obv_confirms = recent_slope < 0 or (overall_slope < 0 and recent_slope <= 0.1)
                            logger.debug(f"📉 OBV validation SELL {symbol}: recent_slope={recent_slope:.2f}, overall_slope={overall_slope:.2f} → {obv_confirms}")
                            return obv_confirms
                            
            except Exception as redis_error:
                logger.warning(f"Impossible de récupérer l'historique OBV pour {symbol}: {redis_error}")
            
            # Fallback: validation simplifiée si pas d'historique disponible
            # Comparer l'OBV actuel avec une valeur de référence basique
            volume_ratio = technical_context.get('volume_ratio', 1.0)
            
            if side == 'BUY':
                # BUY: favorable si volume ratio élevé (indique plus d'activité d'achat)
                return volume_ratio >= 1.0  # STANDARDISÉ: Acceptable
            else:  # SELL  
                # SELL: essoufflement (très bon pour SELL) ou volume correct
                return volume_ratio >= 0.7  # STANDARDISÉ: < 1.0 essoufflement (très bon pour SELL)
            
        except Exception as e:
            logger.error(f"Erreur validation OBV: {e}")
            return None
    
    def _calculate_obv_slope(self, obv_values: List[float]) -> float:
        """
        Calcule la pente de la tendance OBV
        
        Args:
            obv_values: Liste des valeurs OBV chronologiques
            
        Returns:
            Pente (positive = tendance haussière, négative = tendance baissière)
        """
        try:
            if len(obv_values) < 2:
                return 0.0
                
            # Calcul simple de la pente moyenne
            n = len(obv_values)
            x_values = list(range(n))
            
            # Régression linéaire simple: slope = Σ((x-x̄)(y-ȳ)) / Σ((x-x̄)²)
            x_mean = sum(x_values) / n
            y_mean = sum(obv_values) / n
            
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, obv_values))
            denominator = sum((x - x_mean) ** 2 for x in x_values)
            
            if denominator == 0:
                return 0.0
                
            slope = numerator / denominator
            return slope
            
        except Exception as e:
            logger.error(f"Erreur calcul pente OBV: {e}")
            return 0.0
    
    def check_roc_acceleration(self, technical_context: Dict[str, Any], side: str) -> bool:
        """
        Vérifie si le ROC indique une accélération dans la direction du signal
        
        Args:
            technical_context: Contexte technique
            side: 'BUY' ou 'SELL'
            
        Returns:
            True si accélération détectée, False sinon
        """
        try:
            roc_value = technical_context.get('roc')
            if roc_value is None:
                return False
            
            # ROC positif = accélération haussière, ROC négatif = accélération baissière
            if side == 'BUY':
                # Pour BUY: chercher accélération haussière (ROC > 2%)
                return roc_value > 2.0
            elif side == 'SELL':
                # Pour SELL: chercher accélération baissière (ROC < -2%)
                return roc_value < -2.0
                
            return False
            
        except Exception as e:
            logger.error(f"Erreur vérification ROC: {e}")
            return False
    
    async def calculate_atr_based_stop_loss(self, symbol: str, entry_price: float, side: str) -> Optional[float]:
        """
        Calcule un stop-loss adaptatif basé sur l'ATR pour optimiser selon la volatilité
        
        Args:
            symbol: Symbole du trading
            entry_price: Prix d'entrée
            side: 'BUY' ou 'SELL'
            
        Returns:
            Prix de stop-loss adaptatif ou None si impossible
        """
        try:
            # CORRECTION: Import direct avec chemin complet au lieu de manipulation sys.path
            from shared.src.technical_indicators import TechnicalIndicators
            indicators = TechnicalIndicators()
            
            # Récupérer les données OHLC depuis Redis
            market_data_key = f"market_data:{symbol}:3m"
            data_3m = self.redis.get(market_data_key)

            if not data_3m or not isinstance(data_3m, dict):
                logger.debug(f"Données 3m non disponibles pour ATR {symbol}")
                return None
                
            # Extraire les données OHLC
            highs = data_3m.get('highs', [])
            lows = data_3m.get('lows', [])
            closes = data_3m.get('closes', data_3m.get('prices', []))

            if len(highs) < 14 or len(lows) < 14 or len(closes) < 14:
                logger.debug(f"Pas assez de données OHLC pour ATR {symbol}")
                return None
            
            # Calculer ATR(14)
            atr_value = indicators.calculate_atr(highs, lows, closes, period=14)
            if atr_value is None or math.isnan(atr_value) or math.isinf(atr_value):
                logger.debug(f"Calcul ATR échoué ou invalide pour {symbol}: {atr_value}")
                return None
            
            # Récupérer ADX pour adapter le multiplicateur
            adx_value = await self._get_current_adx(symbol)
            
            # Multiplicateur ATR adaptatif selon l'ADX (force de tendance) - STANDARDISÉ
            from shared.src.config import (ADX_STRONG_TREND_THRESHOLD, ADX_TREND_THRESHOLD,
                                         ATR_MULTIPLIER_HIGH, ATR_MULTIPLIER_VERY_HIGH, ATR_MULTIPLIER_EXTREME)
            if adx_value is not None:
                if adx_value > ADX_STRONG_TREND_THRESHOLD:  # Tendance très forte
                    atr_multiplier = ATR_MULTIPLIER_HIGH  # 2.0 - Stop plus serré en tendance forte
                    logger.debug(f"ADX forte ({adx_value:.1f}): multiplicateur ATR {atr_multiplier}x")
                elif adx_value > ADX_TREND_THRESHOLD:  # Tendance modérée
                    atr_multiplier = ATR_MULTIPLIER_VERY_HIGH  # 2.5 - Standard
                    logger.debug(f"ADX modérée ({adx_value:.1f}): multiplicateur ATR {atr_multiplier}x")
                else:  # Tendance faible ou range
                    atr_multiplier = ATR_MULTIPLIER_EXTREME  # 3.0 - Stop plus large en range
                    logger.debug(f"ADX faible ({adx_value:.1f}): multiplicateur ATR {atr_multiplier}x")
            else:
                atr_multiplier = ATR_MULTIPLIER_VERY_HIGH  # 2.5 - Par défaut
                logger.debug(f"ADX non disponible: multiplicateur ATR par défaut {atr_multiplier}x")
            
            # Calculer le stop-loss selon le côté
            atr_distance = atr_value * atr_multiplier
            
            # SANITY CHECK: Vérifier si ATR est en % ou en prix absolu
            atr_as_percent = atr_value / entry_price
            if atr_as_percent > 0.5:  # ATR > 50% du prix = probablement en prix absolu aberrant
                logger.error(f"🚨 ATR ABERRANT {symbol}: {atr_value:.2f} = {atr_as_percent*100:.1f}% du prix {entry_price:.2f}")
                # Forcer un ATR raisonnable basé sur la volatilité crypto (2-5%)
                # Utiliser ATR_MULTIPLIER_MODERATE (1.5) pour un ATR de 3% de base
                from shared.src.config import ATR_MULTIPLIER_MODERATE
                atr_value = entry_price * 0.02  # 2% base ATR pour crypto
                atr_multiplier = ATR_MULTIPLIER_MODERATE  # 1.5 - Volatilité modérée (standard)
                atr_distance = atr_value * atr_multiplier
                logger.warning(f"🔧 ATR corrigé à 3% (2% base × 1.5): nouvelle distance = {atr_distance:.2f}")
            
            # DEBUG: Log des valeurs pour diagnostiquer le problème
            logger.info(f"🔍 ATR Debug {symbol}: atr_value={atr_value:.6f}, multiplier={atr_multiplier}x, distance={atr_distance:.6f}, entry_price={entry_price:.6f}")
            
            if side == 'BUY':
                # BUY: stop en dessous du prix d'entrée
                stop_loss = entry_price - atr_distance
            else:  # SELL
                # SELL: stop au-dessus du prix d'entrée
                stop_loss = entry_price + atr_distance
            
            # Validation: s'assurer que le stop n'est pas trop proche (minimum 0.5%)
            min_distance_percent = 0.005  # 0.5%
            min_distance = entry_price * min_distance_percent
            
            if side == 'BUY':
                if entry_price - stop_loss < min_distance:
                    stop_loss = entry_price - min_distance
                    logger.debug(f"Stop BUY ajusté au minimum 0.5%: {stop_loss:.4f}")
            else:  # SELL
                if stop_loss - entry_price < min_distance:
                    stop_loss = entry_price + min_distance
                    logger.debug(f"Stop SELL ajusté au minimum 0.5%: {stop_loss:.4f}")
            
            # Validation: s'assurer que le stop n'est pas trop loin (maximum 10%)
            max_distance_percent = 0.10  # 10%
            max_distance = entry_price * max_distance_percent
            
            if side == 'BUY':
                if entry_price - stop_loss > max_distance:
                    stop_loss = entry_price - max_distance
                    logger.debug(f"Stop BUY plafonné à 10%: {stop_loss:.4f}")
            else:  # SELL
                if stop_loss - entry_price > max_distance:
                    stop_loss = entry_price + max_distance
                    logger.debug(f"Stop SELL plafonné à 10%: {stop_loss:.4f}")
            
            # PROTECTION ABSOLUE: Forcer un hard-cap à 15% maximum (emergency fix)
            max_emergency_percent = 0.15  # 15% maximum absolu
            max_emergency_distance = entry_price * max_emergency_percent
            
            if side == 'BUY':
                if entry_price - stop_loss > max_emergency_distance:
                    stop_loss = entry_price - max_emergency_distance
                    logger.warning(f"🚨 EMERGENCY CAP: Stop BUY forcé à 15% pour {symbol}: {stop_loss:.4f}")
            else:  # SELL
                if stop_loss - entry_price > max_emergency_distance:
                    stop_loss = entry_price + max_emergency_distance
                    logger.warning(f"🚨 EMERGENCY CAP: Stop SELL forcé à 15% pour {symbol}: {stop_loss:.4f}")
            
            distance_percent = abs(stop_loss - entry_price) / entry_price * 100
            logger.info(f"🎯 Stop ATR calculé pour {symbol} {side}: {stop_loss:.4f} "
                       f"(distance: {distance_percent:.2f}%, ATR: {atr_value:.4f}, mult: {atr_multiplier}x)")
            
            return round(stop_loss, 6)
            
        except Exception as e:
            logger.error(f"Erreur calcul stop ATR pour {symbol}: {e}")
            return None
    
    async def _get_current_adx(self, symbol: str) -> Optional[float]:
        """
        Récupère la valeur ADX actuelle depuis Redis - utilise l'implémentation partagée
        
        Args:
            symbol: Symbole concerné
            
        Returns:
            Valeur ADX ou None si non disponible
        """
        return await TechnicalCalculators.get_current_adx(self.redis, symbol)
    
    def get_or_calculate_indicator_incremental(self, symbol: str, current_candle: Dict, indicator_type: str, **params) -> Union[Optional[float], Dict[str, Union[float, None]]]:
        """
        Méthode générique pour calculer n'importe quel indicateur de manière incrémentale.
        Évite les dents de scie pour MACD, RSI, ATR, Stochastic, etc.
        
        Args:
            symbol: Symbole tradé
            current_candle: Bougie actuelle avec OHLCV
            indicator_type: Type d'indicateur ('macd', 'rsi', 'atr', 'stoch', etc.)
            **params: Paramètres spécifiques (period, etc.)
        """
        timeframe = "1m"
        
        # Initialiser le cache si nécessaire
        if symbol not in self.ema_incremental_cache:
            self.ema_incremental_cache[symbol] = {}
        if timeframe not in self.ema_incremental_cache[symbol]:
            self.ema_incremental_cache[symbol][timeframe] = {}
            
        cache = self.ema_incremental_cache[symbol][timeframe]
        
        try:
            if indicator_type == 'macd':
                # MACD incrémental (line, signal, histogram)
                prev_ema_fast = cache.get('macd_ema_fast')
                prev_ema_slow = cache.get('macd_ema_slow') 
                prev_macd_signal = cache.get('macd_signal')
                
                from shared.src.technical_indicators import calculate_macd_incremental
                result = calculate_macd_incremental(
                    current_candle['close'], prev_ema_fast, prev_ema_slow, prev_macd_signal
                )
                
                # Mettre à jour le cache
                cache['macd_ema_fast'] = result['ema_fast']
                cache['macd_ema_slow'] = result['ema_slow']
                cache['macd_signal'] = result['macd_signal']
                cache['macd_line'] = result['macd_line']
                cache['macd_histogram'] = result['macd_histogram']
                
                return result
                
            elif indicator_type == 'rsi':
                # RSI incrémental (nécessite historique de gains/pertes)
                period = params.get('period', 14)
                prev_rsi = cache.get(f'rsi_{period}')
                prev_avg_gain = cache.get(f'rsi_{period}_avg_gain')
                prev_avg_loss = cache.get(f'rsi_{period}_avg_loss')
                prev_price = cache.get('prev_close')
                
                if prev_price is None:
                    # Première fois : utiliser valeur neutre
                    cache['prev_close'] = current_candle['close']
                    # Initialiser avec une valeur de base ou utiliser prev_rsi si disponible
                    initial_rsi = prev_rsi if prev_rsi is not None else 50.0
                    cache[f'rsi_{period}'] = initial_rsi
                    return initial_rsi
                
                # Calculer gain/perte pour cette période
                price_change = current_candle['close'] - prev_price
                gain = max(price_change, 0)
                loss = max(-price_change, 0)
                
                if prev_avg_gain is None or prev_avg_loss is None:
                    # Initialisation
                    avg_gain = gain
                    avg_loss = loss
                else:
                    # Calcul incrémental des moyennes
                    alpha = 1.0 / period
                    avg_gain = alpha * gain + (1 - alpha) * prev_avg_gain
                    avg_loss = alpha * loss + (1 - alpha) * prev_avg_loss
                
                # Calculer RSI
                if avg_loss == 0:
                    rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100.0 - (100.0 / (1.0 + rs))
                
                # Mettre à jour le cache
                cache[f'rsi_{period}'] = rsi
                cache[f'rsi_{period}_avg_gain'] = avg_gain
                cache[f'rsi_{period}_avg_loss'] = avg_loss
                cache['prev_close'] = current_candle['close']
                
                return rsi
                
            elif indicator_type == 'sma':
                # SMA incrémental
                period = params.get('period', 20)
                prev_sma = cache.get(f'sma_{period}')
                
                from shared.src.technical_indicators import TechnicalIndicators
                indicators = TechnicalIndicators()
                new_sma = indicators.calculate_sma_incremental(current_candle['close'], prev_sma, period)
                
                cache[f'sma_{period}'] = new_sma
                return new_sma
                
            elif indicator_type == 'atr':
                # ATR incrémental pour stop-loss plus précis
                period = params.get('period', 14)
                prev_atr = cache.get(f'atr_{period}')
                prev_close = cache.get('atr_prev_close')
                
                if prev_close is None:
                    cache['atr_prev_close'] = current_candle['close']
                    cache[f'atr_{period}'] = current_candle['high'] - current_candle['low']
                    return cache[f'atr_{period}']
                
                # Calcul True Range
                tr1 = current_candle['high'] - current_candle['low']
                tr2 = abs(current_candle['high'] - prev_close)
                tr3 = abs(current_candle['low'] - prev_close)
                true_range = max(tr1, tr2, tr3)
                
                # ATR incrémental (EMA du True Range)
                if prev_atr is None:
                    new_atr = true_range
                else:
                    alpha = 2.0 / (period + 1)
                    new_atr = alpha * true_range + (1 - alpha) * prev_atr
                
                # Vérifier la validité de l'ATR calculé
                if math.isnan(new_atr) or math.isinf(new_atr):
                    logger.warning(f"ATR incrémental invalide pour {symbol}: {new_atr}, utilisation valeur par défaut")
                    new_atr = true_range  # Fallback sur True Range actuel
                
                cache[f'atr_{period}'] = new_atr
                cache['atr_prev_close'] = current_candle['close']
                return new_atr
                
            elif indicator_type == 'stoch':
                # Stochastic incrémental (K% et D%)
                period_k = params.get('period_k', 14)
                period_d = params.get('period_d', 3)
                
                # Maintenir historique des highs/lows pour K%
                highs_key = f'stoch_highs_{period_k}'
                lows_key = f'stoch_lows_{period_k}'
                
                if highs_key not in cache:
                    cache[highs_key] = []
                if lows_key not in cache:
                    cache[lows_key] = []
                
                # Ajouter valeurs actuelles
                cache[highs_key].append(current_candle['high'])
                cache[lows_key].append(current_candle['low'])
                
                # Maintenir seulement les dernières 'period_k' valeurs
                if len(cache[highs_key]) > period_k:
                    cache[highs_key] = cache[highs_key][-period_k:]
                if len(cache[lows_key]) > period_k:
                    cache[lows_key] = cache[lows_key][-period_k:]
                
                if len(cache[highs_key]) < period_k:
                    return {'stoch_k': 50.0, 'stoch_d': 50.0}  # Valeurs neutres
                
                # Calcul K%
                highest_high = max(cache[highs_key])
                lowest_low = min(cache[lows_key])
                
                if highest_high == lowest_low:
                    stoch_k = 50.0
                else:
                    stoch_k = ((current_candle['close'] - lowest_low) / (highest_high - lowest_low)) * 100
                
                # Calcul D% (SMA de K%)
                k_history_key = f'stoch_k_history_{period_d}'
                if k_history_key not in cache:
                    cache[k_history_key] = []
                    
                cache[k_history_key].append(stoch_k)
                if len(cache[k_history_key]) > period_d:
                    cache[k_history_key] = cache[k_history_key][-period_d:]
                
                stoch_d = sum(cache[k_history_key]) / len(cache[k_history_key])
                
                result = {'stoch_k': stoch_k, 'stoch_d': stoch_d}
                cache['stoch_k'] = stoch_k
                cache['stoch_d'] = stoch_d
                return result
                
            elif indicator_type == 'bollinger':
                # Bollinger Bands incrémental
                period = params.get('period', 20)
                std_dev = params.get('std_dev', 2.0)
                
                # Maintenir historique des prix pour calcul écart-type
                prices_key = f'bb_prices_{period}'
                if prices_key not in cache:
                    cache[prices_key] = []
                    
                cache[prices_key].append(current_candle['close'])
                if len(cache[prices_key]) > period:
                    cache[prices_key] = cache[prices_key][-period:]
                
                if len(cache[prices_key]) < period:
                    return None  # Pas assez de données
                
                # Calcul SMA (middle band)
                sma = sum(cache[prices_key]) / period
                
                # Calcul écart-type
                variance = sum((price - sma) ** 2 for price in cache[prices_key]) / period
                std = variance ** 0.5
                
                # Calcul des bandes
                bb_upper = sma + (std_dev * std)
                bb_lower = sma - (std_dev * std)
                bb_width = (bb_upper - bb_lower) / sma if sma > 0 else 0
                
                # Position relative du prix (0 = bande basse, 1 = bande haute)
                if bb_upper == bb_lower:
                    bb_position = 0.5
                else:
                    bb_position = (current_candle['close'] - bb_lower) / (bb_upper - bb_lower)
                
                result = {
                    'bb_upper': bb_upper,
                    'bb_middle': sma,
                    'bb_lower': bb_lower,
                    'bb_position': bb_position,
                    'bb_width': bb_width
                }
                
                # Mettre à jour le cache
                for key, value in result.items():
                    cache[key] = value
                    
                return result
                
            else:
                # Fallback pour indicateurs non implémentés
                logger.debug(f"Indicateur incrémental non implémenté: {indicator_type}")
                return None
                
        except Exception as e:
            logger.error(f"Erreur calcul incrémental {indicator_type} pour {symbol}: {e}")
            return None