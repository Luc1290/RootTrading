"""
Stratégie EMA Cross Pro
EMA avec contexte multi-timeframes, momentum, structure de marché et confluence avancée.
Intègre ADX, volume, Williams %R, et analyse de régime pour des signaux précis.
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from shared.src.enums import OrderSide, SignalStrength
from shared.src.config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB

from .base_strategy import BaseStrategy

# Import des modules d'analyse avancée
try:
    import redis  # type: ignore
except ImportError:
    redis = None  # type: ignore

logger = logging.getLogger(__name__)

class EMACrossProStrategy(BaseStrategy):
    """
    Stratégie EMA Cross Pro - Croisements avec contexte multi-timeframes et momentum
    BUY: EMA7 croise au-dessus EMA26 + confluence + momentum haussier + structure favorable
    SELL: EMA7 croise en-dessous EMA26 + confluence + momentum baissier + structure favorable
    
    Intègre :
    - Analyse multi-timeframes (confluence)
    - Momentum cross-timeframe
    - Structure de marché
    - Volume confirmation
    - ADX pour force de tendance
    - Williams %R pour timing
    - Régime adaptatif
    """
    
    def __init__(self, symbol: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(symbol, params)
        
        # Paramètres EMA avancés
        symbol_params = self.params.get(symbol, {}) if self.params else {}
        self.min_gap_percent = symbol_params.get('ema_gap_min', 0.001)  # ASSOUPLI de 0.003 à 0.001
        self.min_adx = symbol_params.get('min_adx', 15.0)  # ASSOUPLI de 20 à 15
        self.confluence_threshold = symbol_params.get('confluence_threshold', 30.0)  # ASSOUPLI de 40 à 30
        self.momentum_threshold = symbol_params.get('momentum_threshold', 0.2)  # ASSOUPLI de 0.3 à 0.2
        
        # Connexion Redis pour analyses avancées
        self.redis_client = None
        if redis:
            try:
                self.redis_client = redis.Redis(
                    host=REDIS_HOST, port=REDIS_PORT, 
                    password=REDIS_PASSWORD, db=REDIS_DB, 
                    decode_responses=True
                )
                self.redis_client.ping()  # Test connexion
            except Exception as e:
                logger.warning(f"⚠️ Redis non disponible pour EMA Cross Pro: {e}")
                self.redis_client = None
        
        logger.info(f"🎯 EMA Cross Pro initialisé pour {symbol} (ADX≥{self.min_adx}, Confluence≥{self.confluence_threshold}%)")

    @property
    def name(self) -> str:
        return "EMA_Cross_Pro_Strategy"
    
    def get_min_data_points(self) -> int:
        return 30  # Minimum pour EMA stable
    
    def analyze(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Optional[Dict]:
        """
        Analyse EMA Cross Pro - croisements avec contexte multi-timeframes
        """
        try:
            if len(df) < self.get_min_data_points():
                return None
            
            # PRIORITÉ 1: Vérifier conditions de protection défensive
            defensive_signal = self.check_defensive_conditions(df)
            if defensive_signal:
                return defensive_signal
            
            # Récupérer EMAs pré-calculées
            # MIGRATION BINANCE: EMA 7 directement (plus réactif)
            current_ema7 = self._get_current_indicator(indicators, 'ema_7')
            current_ema26 = self._get_current_indicator(indicators, 'ema_26')
            
            if current_ema7 is None or current_ema26 is None:
                logger.debug(f"❌ {symbol}: EMAs non disponibles")
                return None
            
            # Récupérer EMAs précédentes pour détecter croisement
            # MIGRATION BINANCE: EMA 7 directement
            previous_ema7 = self._get_previous_indicator(indicators, 'ema_7')
            previous_ema26 = self._get_previous_indicator(indicators, 'ema_26')
            
            # Récupérer indicateurs de contexte
            adx = self._get_current_indicator(indicators, 'adx_14')
            williams_r = self._get_current_indicator(indicators, 'williams_r')
            volume_ratio = self._get_current_indicator(indicators, 'volume_ratio')
            volume_spike = indicators.get('volume_spike', False)
            bb_width = self._get_current_indicator(indicators, 'bb_width')
            momentum_10 = self._get_current_indicator(indicators, 'momentum_10')
            
            if previous_ema7 is None or previous_ema26 is None:
                return None
            
            current_price = df['close'].iloc[-1]
            
            # Analyser le contexte multi-timeframes
            context_analysis = self._analyze_market_context(symbol, adx or 0.0, williams_r or 0.0, volume_ratio or 0.0, bb_width or 0.0, momentum_10 or 0.0)
            
            # NOUVEAU: Calculer la position du prix dans son range
            price_position = self.calculate_price_position_in_range(df)
            
            signal = None
            
            # SIGNAL D'ACHAT - Golden Cross avec contexte favorable
            if (previous_ema7 <= previous_ema26 and current_ema7 > current_ema26):
                gap_percent = abs(current_ema7 - current_ema26) / current_ema26
                
                # Vérifier les conditions de contexte
                if self._validate_bullish_context(context_analysis, gap_percent):
                    # Vérifier la position du prix avant de générer le signal
                    if not self.should_filter_signal_by_price_position(OrderSide.BUY, price_position, df):
                        # Calculer confiance avec tous les facteurs
                        base_confidence = min(0.8, gap_percent * 120 + 0.5)
                        context_boost = context_analysis['confidence_boost']
                        final_confidence = min(0.95, base_confidence + context_boost)
                        
                        signal = self.create_signal(
                            side=OrderSide.BUY,
                            price=current_price,
                            confidence=final_confidence,
                            metadata={
                                'ema7': current_ema7,  # MIGRATION BINANCE
                                'ema26': current_ema26,
                                'gap_percent': gap_percent * 100,
                                'adx': adx,
                                'williams_r': williams_r,
                                'volume_ratio': volume_ratio,
                                'volume_spike': volume_spike,
                                'context_score': context_analysis['score'],
                                'confluence_score': context_analysis.get('confluence_score', 0),
                                'momentum_score': context_analysis.get('momentum_score', 0),
                                'price_position': price_position,
                                'reason': f'EMA Golden Cross Pro BINANCE (7: {current_ema7:.4f} > 26: {current_ema26:.4f}) + Contexte favorable'
                            }
                        )
                        # Enregistrer prix d'entrée pour protection défensive
                        self.last_entry_price = current_price
                    else:
                        logger.info(f"📊 EMA Cross Pro {symbol}: Signal BUY techniquement valide mais filtré "
                                  f"(position prix: {price_position:.2f})")
                else:
                    logger.debug(f"🚫 EMA Cross {symbol}: Golden cross détecté mais contexte défavorable (score: {context_analysis['score']:.1f})")
            
            # SIGNAL DE VENTE - Death Cross avec contexte favorable
            elif (previous_ema7 >= previous_ema26 and current_ema7 < current_ema26):
                gap_percent = abs(current_ema26 - current_ema7) / current_ema26
                
                # Vérifier les conditions de contexte
                if self._validate_bearish_context(context_analysis, gap_percent):
                    # Vérifier la position du prix avant de générer le signal
                    if not self.should_filter_signal_by_price_position(OrderSide.SELL, price_position, df):
                        # Calculer confiance avec tous les facteurs
                        base_confidence = min(0.8, gap_percent * 120 + 0.5)
                        context_boost = context_analysis['confidence_boost']
                        final_confidence = min(0.95, base_confidence + context_boost)
                        
                        signal = self.create_signal(
                            side=OrderSide.SELL,
                            price=current_price,
                            confidence=final_confidence,
                            metadata={
                                'ema7': current_ema7,  # MIGRATION BINANCE
                                'ema26': current_ema26,
                                'gap_percent': gap_percent * 100,
                                'adx': adx,
                                'williams_r': williams_r,
                                'volume_ratio': volume_ratio,
                                'volume_spike': volume_spike,
                                'context_score': context_analysis['score'],
                                'confluence_score': context_analysis.get('confluence_score', 0),
                                'momentum_score': context_analysis.get('momentum_score', 0),
                                'price_position': price_position,
                                'reason': f'EMA Death Cross Pro BINANCE (7: {current_ema7:.4f} < 26: {current_ema26:.4f}) + Contexte favorable'
                            }
                        )
                    else:
                        logger.info(f"📊 EMA Cross Pro {symbol}: Signal SELL techniquement valide mais filtré "
                                  f"(position prix: {price_position:.2f})")
                else:
                    logger.debug(f"🚫 EMA Cross {symbol}: Death cross détecté mais contexte défavorable (score: {context_analysis['score']:.1f})")
            
            if signal:
                logger.info(f"🎯 EMA Cross Pro {symbol}: {signal.side} @ {current_price:.4f} "
                          f"(7: {current_ema7:.4f}, 26: {current_ema26:.4f}, ADX: {adx:.1f}, "
                          f"Context: {context_analysis['score']:.1f}, Conf: {signal.confidence:.2f}, {signal.strength})")
                
                # Convertir StrategySignal en dict pour compatibilité
                return {
                    'strategy': signal.strategy,
                    'symbol': signal.symbol,
                    'side': signal.side,
                    'timestamp': signal.timestamp.isoformat(),
                    'price': signal.price,
                    'confidence': signal.confidence,
                    'strength': signal.strength,
                    'metadata': signal.metadata
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Erreur EMA Cross Pro Strategy {symbol}: {e}")
            return None
    
    def _analyze_market_context(self, symbol: str, adx: float, williams_r: float, 
                               volume_ratio: float, bb_width: float, momentum_10: float) -> Dict:
        """Analyse le contexte de marché pour valider les signaux EMA"""
        context = {
            'score': 0.0,
            'confidence_boost': 0.0,
            'confluence_score': 0.0,
            'momentum_score': 0.0,
            'details': []
        }
        
        try:
            # 1. Force de tendance (ADX)
            from shared.src.config import ADX_STRONG_TREND_THRESHOLD, ADX_TREND_THRESHOLD
            if adx >= self.min_adx:
                if adx >= ADX_STRONG_TREND_THRESHOLD:
                    context['score'] += 25
                    context['confidence_boost'] += 0.1
                    context['details'].append(f"ADX fort ({adx:.1f})")
                elif adx >= ADX_TREND_THRESHOLD:
                    context['score'] += 15
                    context['confidence_boost'] += 0.05
                    context['details'].append(f"ADX modéré ({adx:.1f})")
                else:
                    context['score'] += 5
                    context['details'].append(f"ADX faible ({adx:.1f})")
            else:
                context['score'] -= 10
                context['details'].append(f"ADX insuffisant ({adx:.1f})")
            
            # 2. Timing d'entrée (Williams %R)
            if williams_r <= -80:  # Zone de survente
                context['score'] += 15
                context['confidence_boost'] += 0.05
                context['details'].append(f"Williams oversold ({williams_r:.1f})")
            elif williams_r >= -20:  # Zone de surachat
                context['score'] += 15
                context['confidence_boost'] += 0.05
                context['details'].append(f"Williams overbought ({williams_r:.1f})")
            else:
                context['score'] += 5
                context['details'].append(f"Williams neutre ({williams_r:.1f})")
            
            # 3. Confirmation de volume - SEUILS STANDARDISÉS
            if volume_ratio > 1.5:  # STANDARDISÉ: Très bon
                context['score'] += 25
                context['confidence_boost'] += 0.1
                context['details'].append(f"Volume très bon ({volume_ratio:.1f}x)")
            elif volume_ratio > 1.2:  # STANDARDISÉ: Bon
                context['score'] += 20
                context['confidence_boost'] += 0.08
                context['details'].append(f"Volume bon ({volume_ratio:.1f}x)")
            elif volume_ratio > 1.0:  # STANDARDISÉ: Acceptable
                context['score'] += 15
                context['confidence_boost'] += 0.05
                context['details'].append(f"Volume acceptable ({volume_ratio:.1f}x)")
            elif volume_ratio > 0.8:  # Faible mais utilisable
                context['score'] += 10
                context['details'].append(f"Volume faible ({volume_ratio:.1f}x)")
            else:
                context['score'] -= 5
                context['details'].append(f"Volume faible ({volume_ratio:.1f}x)")
            
            # 4. Volatilité (Bollinger width)
            if bb_width > 0.03:  # Expansion
                context['score'] += 15
                context['confidence_boost'] += 0.05
                context['details'].append(f"Volatilité élevée ({bb_width:.3f})")
            elif bb_width < 0.015:  # Compression
                context['score'] -= 5
                context['details'].append(f"Volatilité faible ({bb_width:.3f})")
            else:
                context['score'] += 5
                context['details'].append(f"Volatilité normale ({bb_width:.3f})")
            
            # 5. Momentum directionnel
            momentum_strength = abs(momentum_10)
            if momentum_strength > 1.0:
                context['score'] += 15
                context['momentum_score'] = momentum_strength
                context['confidence_boost'] += 0.05
                context['details'].append(f"Momentum fort ({momentum_10:.2f})")
            elif momentum_strength > 0.5:
                context['score'] += 10
                context['momentum_score'] = momentum_strength
                context['details'].append(f"Momentum modéré ({momentum_10:.2f})")
            else:
                context['score'] += 2
                context['details'].append(f"Momentum faible ({momentum_10:.2f})")
            
            # 6. Analyse multi-timeframes (si Redis disponible)
            if self.redis_client:
                confluence_data = self._get_confluence_analysis(symbol)
                if confluence_data:
                    confluence_score = confluence_data.get('confluence_score', 0)
                    context['confluence_score'] = confluence_score
                    
                    if confluence_score >= 70:
                        context['score'] += 25
                        context['confidence_boost'] += 0.1
                        context['details'].append(f"Confluence forte ({confluence_score:.1f}%)")
                    elif confluence_score >= 55:
                        context['score'] += 15
                        context['confidence_boost'] += 0.05
                        context['details'].append(f"Confluence modérée ({confluence_score:.1f}%)")
                    else:
                        context['score'] -= 5
                        context['details'].append(f"Confluence faible ({confluence_score:.1f}%)")
            
            # Normaliser le score (0-100)
            context['score'] = max(0.0, min(100.0, context['score']))
            context['confidence_boost'] = max(0.0, min(0.25, context['confidence_boost']))
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse contexte EMA: {e}")
            context['score'] = 30  # Score neutre par défaut
        
        return context
    
    def _validate_bullish_context(self, context_analysis: Dict, gap_percent: float) -> bool:
        """Valide si le contexte est favorable pour un signal d'achat"""
        score = context_analysis['score']
        
        # Conditions minimales
        if score < 30:  # ASSOUPLI de 40 à 30
            return False
        
        if gap_percent < self.min_gap_percent:  # Gap EMA minimum
            return False
        
        # Si confluence disponible, l'utiliser
        confluence_score = context_analysis.get('confluence_score', 0)
        if confluence_score > 0 and confluence_score < self.confluence_threshold:
            return False
        
        return True
    
    def _validate_bearish_context(self, context_analysis: Dict, gap_percent: float) -> bool:
        """Valide si le contexte est favorable pour un signal de vente"""
        score = context_analysis['score']
        
        # Conditions minimales assouplis pour SELL
        if score < 30:  # ASSOUPLI de 35 à 30
            return False
        
        if gap_percent < self.min_gap_percent * 0.5:  # Gap EMA minimum encore plus assoupli
            return False
        
        # Si confluence disponible, l'utiliser - assoupli
        confluence_score = context_analysis.get('confluence_score', 0)
        if confluence_score > 0 and confluence_score < (self.confluence_threshold - 10):  # ASSOUPLI de 5 à 10
            return False
        
        return True
    
    def _get_confluence_analysis(self, symbol: str) -> Optional[Dict]:
        """Récupère l'analyse de confluence depuis Redis"""
        try:
            if not self.redis_client:
                return None
            
            cache_key = f"confluence:{symbol}"
            cached = self.redis_client.get(cache_key)
            
            if cached:
                import json
                return json.loads(str(cached))
            
        except Exception as e:
            logger.debug(f"Confluence non disponible pour {symbol}: {e}")
        
        return None
    
    def _get_current_indicator(self, indicators: Dict, key: str) -> Optional[float]:
        """Récupère la valeur actuelle d'un indicateur"""
        value = indicators.get(key)
        if value is None:
            return None
        
        if isinstance(value, (list, np.ndarray)) and len(value) > 0:
            return float(value[-1])
        elif isinstance(value, (int, float)):
            return float(value)
        
        return None
    
    def _get_previous_indicator(self, indicators: Dict, key: str) -> Optional[float]:
        """Récupère la valeur précédente d'un indicateur"""
        value = indicators.get(key)
        if value is None:
            return None
        
        if isinstance(value, (list, np.ndarray)) and len(value) > 1:
            return float(value[-2])
        
        return None