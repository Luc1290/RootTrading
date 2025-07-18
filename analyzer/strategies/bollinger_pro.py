"""
Stratégie Bollinger Pro
Bollinger avec expansion/contraction, squeeze detection, breakouts et confluence.
Intègre ATR, volume, momentum et analyse multi-timeframes pour timing précis.
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
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
    import redis
except ImportError:
    redis = None

logger = logging.getLogger(__name__)

class BollingerProStrategy(BaseStrategy):
    """
    Stratégie Bollinger Pro - Bandes avec squeeze detection et breakouts
    BUY: Squeeze + breakout haussier + volume + momentum favorable + confluence
    SELL: Squeeze + breakout baissier + volume + momentum favorable + confluence
    
    Intègre :
    - Détection de squeeze (contraction)
    - Breakouts avec confirmation volume
    - Expansion/contraction cycles
    - ATR pour volatilité
    - Momentum directionnel
    - Confluence multi-timeframes
    - Mean reversion vs trend following
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        super().__init__(symbol, params)
        
        # Paramètres Bollinger avancés
        symbol_params = self.params.get(symbol, {}) if self.params else {}
        self.squeeze_threshold = symbol_params.get('bb_squeeze_threshold', 0.020)  # Width pour squeeze
        self.expansion_threshold = symbol_params.get('bb_expansion_threshold', 0.035)  # Width pour expansion
        self.breakout_min_volume = symbol_params.get('breakout_min_volume', 1.3)  # Volume minimum
        self.mean_reversion_zone = symbol_params.get('mean_reversion_zone', 0.15)  # Zone mean reversion
        self.trend_follow_zone = symbol_params.get('trend_follow_zone', 0.85)  # Zone trend following
        from shared.src.config import ATR_THRESHOLD_LOW
        self.min_atr = symbol_params.get('min_atr', ATR_THRESHOLD_LOW)  # ATR minimum (0.002)
        self.confluence_threshold = symbol_params.get('confluence_threshold', 30.0)  # ASSOUPLI de 40 à 30
        
        # Historique pour détection de patterns
        self.bb_width_history = []
        self.bb_position_history = []
        self.max_history = 15
        
        # Connexion Redis pour analyses avancées
        self.redis_client = None
        if redis:
            try:
                self.redis_client = redis.Redis(
                    host=REDIS_HOST, port=REDIS_PORT, 
                    password=REDIS_PASSWORD, db=REDIS_DB, 
                    decode_responses=True
                )
                self.redis_client.ping()
            except Exception as e:
                logger.warning(f"⚠️ Redis non disponible pour Bollinger Pro: {e}")
                self.redis_client = None
        
        logger.info(f"🎯 Bollinger Pro initialisé pour {symbol} (Squeeze≤{self.squeeze_threshold:.3f}, Expansion≥{self.expansion_threshold:.3f})")

    @property
    def name(self) -> str:
        return "Bollinger_Pro_Strategy"
    
    def get_min_data_points(self) -> int:
        return 25  # Minimum pour Bollinger et patterns
    
    def analyze(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Optional[Dict]:
        """
        Analyse Bollinger Pro - Squeeze, breakouts et confluence
        """
        try:
            if len(df) < self.get_min_data_points():
                return None
            
            # PRIORITÉ 1: Vérifier conditions de protection défensive
            defensive_signal = self.check_defensive_conditions(df)
            if defensive_signal:
                return defensive_signal
            
            # Récupérer indicateurs Bollinger
            bb_position = self._get_current_indicator(indicators, 'bb_position')
            bb_width = self._get_current_indicator(indicators, 'bb_width')
            bb_upper = self._get_current_indicator(indicators, 'bb_upper')
            bb_lower = self._get_current_indicator(indicators, 'bb_lower')
            bb_middle = self._get_current_indicator(indicators, 'bb_middle')
            
            if any(x is None for x in [bb_position, bb_width, bb_upper, bb_lower, bb_middle]):
                logger.debug(f"❌ {symbol}: Indicateurs Bollinger incomplets")
                return None
            
            current_price = df['close'].iloc[-1]
            
            # Récupérer indicateurs de contexte
            atr = self._get_current_indicator(indicators, 'atr_14')
            volume_ratio = self._get_current_indicator(indicators, 'volume_ratio')
            volume_spike = indicators.get('volume_spike', False)
            momentum_10 = self._get_current_indicator(indicators, 'momentum_10')
            rsi = self._get_current_indicator(indicators, 'rsi_14')
            adx = self._get_current_indicator(indicators, 'adx_14')
            
            # Mettre à jour l'historique
            self._update_history(bb_width, bb_position)
            
            # Analyser le pattern Bollinger
            pattern_analysis = self._analyze_bollinger_pattern()
            
            # Analyser le contexte
            context_analysis = self._analyze_bollinger_context(
                symbol, bb_width, bb_position, atr, volume_ratio, 
                momentum_10, rsi, adx
            )
            
            # NOUVEAU: Calculer la position du prix dans son range
            price_position = self.calculate_price_position_in_range(df)
            
            signal = None
            
            # SIGNAL D'ACHAT - Conditions favorables
            if self._is_bullish_bollinger_signal(
                bb_position, bb_width, pattern_analysis, context_analysis, current_price, bb_middle
            ):
                # Vérifier la position du prix avant de générer le signal
                if not self.should_filter_signal_by_price_position(OrderSide.BUY, price_position, df):
                    confidence = self._calculate_bullish_confidence(
                        bb_position, bb_width, pattern_analysis, context_analysis, 
                        volume_ratio, momentum_10
                    )
                    
                    signal = self.create_signal(
                        side=OrderSide.BUY,
                        price=current_price,
                        confidence=confidence,
                        metadata={
                            'bb_position': bb_position,
                            'bb_width': bb_width,
                            'bb_upper': bb_upper,
                            'bb_lower': bb_lower,
                            'bb_middle': bb_middle,
                            'pattern_type': pattern_analysis['type'],
                            'pattern_strength': pattern_analysis['strength'],
                            'volume_ratio': volume_ratio,
                            'volume_spike': volume_spike,
                            'momentum_10': momentum_10,
                            'context_score': context_analysis['score'],
                            'confluence_score': context_analysis.get('confluence_score', 0),
                            'price_position': price_position,
                            'reason': f'Bollinger Pro BUY ({pattern_analysis["type"]}, pos: {bb_position:.2f}, width: {bb_width:.3f})'
                        }
                    )
                    # Enregistrer prix d'entrée pour protection défensive
                    self.last_entry_price = current_price
                    
                else:
                    logger.info(f"📊 Bollinger Pro {symbol}: Signal BUY techniquement valide mais filtré "
                                f"(position prix: {price_position:.2f})")
            
            # SIGNAL DE VENTE - Conditions favorables
            elif self._is_bearish_bollinger_signal(
                bb_position, bb_width, pattern_analysis, context_analysis, current_price, bb_middle
            ):
                # Vérifier la position du prix avant de générer le signal
                if not self.should_filter_signal_by_price_position(OrderSide.SELL, price_position, df):
                    confidence = self._calculate_bearish_confidence(
                        bb_position, bb_width, pattern_analysis, context_analysis, 
                        volume_ratio, momentum_10
                    )
                    
                    signal = self.create_signal(
                        side=OrderSide.SELL,
                        price=current_price,
                        confidence=confidence,
                        metadata={
                            'bb_position': bb_position,
                            'bb_width': bb_width,
                            'bb_upper': bb_upper,
                            'bb_lower': bb_lower,
                            'bb_middle': bb_middle,
                            'pattern_type': pattern_analysis['type'],
                            'pattern_strength': pattern_analysis['strength'],
                            'volume_ratio': volume_ratio,
                            'volume_spike': volume_spike,
                            'momentum_10': momentum_10,
                            'context_score': context_analysis['score'],
                            'confluence_score': context_analysis.get('confluence_score', 0),
                            'price_position': price_position,
                            'reason': f'Bollinger Pro SELL ({pattern_analysis["type"]}, pos: {bb_position:.2f}, width: {bb_width:.3f})'
                        }
                    )
                else:
                    logger.info(f"📊 Bollinger Pro {symbol}: Signal SELL techniquement valide mais filtré "
                              f"(position prix: {price_position:.2f})")
            
            if signal:
                logger.info(f"🎯 Bollinger Pro {symbol}: {signal.side} @ {current_price:.4f} "
                          f"({pattern_analysis['type']}, pos: {bb_position:.2f}, width: {bb_width:.3f}, "
                          f"Context: {context_analysis['score']:.1f}, Conf: {signal.confidence:.2f})")
                
                # Log metadata pour debug
                logger.info(f"📊 Bollinger Pro metadata avant retour: {signal.metadata}")
                
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
            logger.error(f"❌ Erreur Bollinger Pro Strategy {symbol}: {e}")
            return None
    
    def _update_history(self, bb_width: float, bb_position: float):
        """Met à jour l'historique pour la détection de patterns"""
        self.bb_width_history.append(bb_width)
        self.bb_position_history.append(bb_position)
        
        # Limiter la taille
        if len(self.bb_width_history) > self.max_history:
            self.bb_width_history.pop(0)
        if len(self.bb_position_history) > self.max_history:
            self.bb_position_history.pop(0)
    
    def _analyze_bollinger_pattern(self) -> Dict:
        """Analyse les patterns Bollinger (squeeze, expansion, breakout)"""
        pattern = {
            'type': 'normal',
            'strength': 0.0,
            'confidence': 0.0,
            'signal_type': 'none'  # 'mean_reversion', 'trend_following', 'breakout'
        }
        
        try:
            if len(self.bb_width_history) < 5:
                return pattern
            
            current_width = self.bb_width_history[-1]
            avg_width = np.mean(self.bb_width_history[-5:])
            width_trend = np.mean(np.diff(self.bb_width_history[-5:]))
            
            # Détection de squeeze
            if current_width <= self.squeeze_threshold:
                pattern['type'] = 'squeeze'
                pattern['strength'] = (self.squeeze_threshold - current_width) / self.squeeze_threshold
                pattern['confidence'] = 0.8
                pattern['signal_type'] = 'breakout'  # Préparer breakout
            
            # Détection d'expansion
            elif current_width >= self.expansion_threshold:
                pattern['type'] = 'expansion'
                pattern['strength'] = min(1.0, (current_width - self.expansion_threshold) / self.expansion_threshold)
                pattern['confidence'] = 0.7
                pattern['signal_type'] = 'mean_reversion'  # Retour vers moyenne
            
            # Expansion en cours
            elif width_trend > 0.002:  # Width augmente
                pattern['type'] = 'expanding'
                pattern['strength'] = min(1.0, width_trend * 500)
                pattern['confidence'] = 0.6
                pattern['signal_type'] = 'trend_following'
            
            # Contraction en cours
            elif width_trend < -0.002:  # Width diminue
                pattern['type'] = 'contracting'
                pattern['strength'] = min(1.0, abs(width_trend) * 500)
                pattern['confidence'] = 0.5
                pattern['signal_type'] = 'breakout'  # Se prépare au breakout
            
            # Normal
            else:
                pattern['type'] = 'normal'
                pattern['strength'] = 0.3
                pattern['confidence'] = 0.4
                pattern['signal_type'] = 'mean_reversion'
            
        except Exception as e:
            logger.debug(f"Erreur analyse pattern Bollinger: {e}")
        
        return pattern
    
    def _analyze_bollinger_context(self, symbol: str, bb_width: float, bb_position: float, 
                                  atr: float, volume_ratio: float, momentum_10: float, 
                                  rsi: float, adx: float) -> Dict:
        """Analyse le contexte pour valider les signaux Bollinger"""
        context = {
            'score': 0.0,
            'confidence_boost': 0.0,
            'confluence_score': 0.0,
            'details': []
        }
        
        try:
            # 1. Volatilité (ATR)
            from shared.src.config import ATR_THRESHOLD_HIGH
            if atr and atr >= self.min_atr:
                if atr >= ATR_THRESHOLD_HIGH:  # 0.005 - Volatilité élevée
                    context['score'] += 20
                    context['confidence_boost'] += 0.05
                    context['details'].append(f"ATR élevé ({atr:.4f})")
                else:
                    context['score'] += 10
                    context['details'].append(f"ATR normal ({atr:.4f})")
            else:
                context['score'] -= 10
                context['details'].append(f"ATR faible ({atr or 0:.4f})")
            
            # 2. Volume confirmation - SEUILS STANDARDISÉS
            if volume_ratio and volume_ratio >= 2.0:  # STANDARDISÉ: Excellent (début pump confirmé)
                context['score'] += 25
                context['confidence_boost'] += 0.1
                context['details'].append(f"Volume excellent ({volume_ratio:.1f}x)")
            elif volume_ratio and volume_ratio > 1.5:  # STANDARDISÉ: Très bon
                context['score'] += 20
                context['confidence_boost'] += 0.08
                context['details'].append(f"Volume très bon ({volume_ratio:.1f}x)")
            elif volume_ratio and volume_ratio > 1.2:  # STANDARDISÉ: Bon
                context['score'] += 15
                context['confidence_boost'] += 0.05
                context['details'].append(f"Volume bon ({volume_ratio:.1f}x)")
            elif volume_ratio and volume_ratio > 1.0:  # STANDARDISÉ: Acceptable
                context['score'] += 10
                context['details'].append(f"Volume acceptable ({volume_ratio:.1f}x)")
            elif volume_ratio and volume_ratio > 0.7:  # Faible mais utilisable
                context['score'] += 5
                context['details'].append(f"Volume faible ({volume_ratio:.1f}x)")
            else:
                context['score'] -= 5
                context['details'].append(f"Volume faible ({volume_ratio or 0:.1f}x)")
            
            # 3. Momentum support
            if momentum_10:
                momentum_strength = abs(momentum_10)
                if momentum_strength > 1.5:
                    context['score'] += 20
                    context['confidence_boost'] += 0.08
                    context['details'].append(f"Momentum fort ({momentum_10:.2f})")
                elif momentum_strength > 0.8:
                    context['score'] += 15
                    context['confidence_boost'] += 0.05
                    context['details'].append(f"Momentum modéré ({momentum_10:.2f})")
                else:
                    context['score'] += 5
                    context['details'].append(f"Momentum faible ({momentum_10:.2f})")
            
            # 4. RSI confirmation
            if rsi:
                if rsi <= 30 or rsi >= 70:  # Zones extrêmes
                    context['score'] += 15
                    context['confidence_boost'] += 0.05
                    context['details'].append(f"RSI extrême ({rsi:.1f})")
                elif rsi <= 40 or rsi >= 60:
                    context['score'] += 10
                    context['details'].append(f"RSI actif ({rsi:.1f})")
                else:
                    context['score'] += 5
                    context['details'].append(f"RSI neutre ({rsi:.1f})")
            
            # 5. Tendance (ADX)
            if adx:
                from shared.src.config import ADX_TREND_THRESHOLD, ADX_WEAK_TREND_THRESHOLD
                if adx >= ADX_TREND_THRESHOLD:
                    context['score'] += 15
                    context['confidence_boost'] += 0.05
                    context['details'].append(f"ADX tendance ({adx:.1f})")
                elif adx >= ADX_WEAK_TREND_THRESHOLD:
                    context['score'] += 10
                    context['details'].append(f"ADX modéré ({adx:.1f})")
                else:
                    context['score'] += 5
                    context['details'].append(f"ADX faible ({adx:.1f})")
            
            # 6. Confluence multi-timeframes
            if self.redis_client:
                confluence_data = self._get_confluence_analysis(symbol)
                if confluence_data:
                    confluence_score = confluence_data.get('confluence_score', 0)
                    context['confluence_score'] = confluence_score
                    
                    if confluence_score >= 60:
                        context['score'] += 20
                        context['confidence_boost'] += 0.08
                        context['details'].append(f"Confluence forte ({confluence_score:.1f}%)")
                    elif confluence_score >= 45:
                        context['score'] += 10
                        context['confidence_boost'] += 0.03
                        context['details'].append(f"Confluence modérée ({confluence_score:.1f}%)")
            
            # Normaliser
            context['score'] = max(0, min(100, context['score']))
            context['confidence_boost'] = max(0, min(0.25, context['confidence_boost']))
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse contexte Bollinger: {e}")
            context['score'] = 25
        
        return context
    
    def _is_bullish_bollinger_signal(self, bb_position: float, bb_width: float, 
                                    pattern: Dict, context: Dict, price: float, bb_middle: float) -> bool:
        """Détermine si les conditions d'achat Bollinger sont remplies"""
        
        # Score de contexte minimum assoupli
        if context['score'] < 20:  # ASSOUPLI de 25 à 20
            return False
        
        # Confluence minimum assouplie
        confluence_score = context.get('confluence_score', 0)
        if confluence_score > 0 and confluence_score < self.confluence_threshold:
            return False
        
        signal_type = pattern['signal_type']
        
        # 1. Mean reversion (prix bas, retour vers moyenne) - STANDARDISÉ
        if signal_type == 'mean_reversion' and bb_position <= 0.25:  # STANDARDISÉ: Très bon (bas de bande)
            return context['score'] > 30  # ASSOUPLI de 40 à 30
        
        # 2. Breakout haussier après squeeze - STANDARDISÉ
        if (signal_type == 'breakout' and pattern['type'] in ['squeeze', 'contracting'] and 
            bb_position > 0.25 and price > bb_middle):  # STANDARDISÉ: Au-dessus de "très bon"
            return context['score'] > 35  # ASSOUPLI de 50 à 35
        
        # 3. Trend following - STANDARDISÉ pour capturer les débuts de pump
        if (signal_type == 'trend_following' and bb_position <= 0.45 and  # STANDARDISÉ: Acceptable (position neutre-basse)
            bb_width > self.expansion_threshold):
            return context['score'] > 40  # ASSOUPLI de 50 à 40
        
        # 4. NOUVEAU: Détection de début de pump (oversold qui remonte) - STANDARDISÉ
        if bb_position <= 0.35 and pattern['type'] == 'expanding' and price > bb_middle:  # STANDARDISÉ: Bon (position basse)
            return context['score'] > 35  # ASSOUPLI de 45 à 35
        
        # 5. NOUVEAU: Condition générale pour positions basses - STANDARDISÉ
        if bb_position <= 0.25 and context['score'] > 50:  # STANDARDISÉ: Très bon (bas de bande)
            return True  # Force BUY sur position basse
        
        return False
    
    def _is_bearish_bollinger_signal(self, bb_position: float, bb_width: float, 
                                    pattern: Dict, context: Dict, price: float, bb_middle: float) -> bool:
        """Détermine si les conditions de vente Bollinger sont remplies (fin de pump)"""
        
        # Score de contexte minimum pour fin pump
        if context['score'] < 30:
            return False
        
        # Confluence minimum pour fin pump confirmée
        confluence_score = context.get('confluence_score', 0)
        if confluence_score > 0 and confluence_score < 45:  # Plus strict pour fin pump
            return False
        
        signal_type = pattern['signal_type']
        
        # 1. Mean reversion (prix haut, retour vers moyenne) - STANDARDISÉ
        if signal_type == 'mean_reversion' and bb_position >= 0.75:  # STANDARDISÉ: Très bon (haut de bande)
            return context['score'] > 40  # Était 50, maintenant 40
        
        # 2. Breakout baissier après squeeze - INCHANGÉ
        if (signal_type == 'breakout' and pattern['type'] in ['squeeze', 'contracting'] and 
            bb_position < 0.5 and price < bb_middle):
            return context['score'] > 50  # Était 60, maintenant 50
        
        # 3. Trend following - STANDARDISÉ pour FIN de pump (vendre au sommet)
        if (signal_type == 'trend_following' and bb_position >= 0.75 and  # STANDARDISÉ: Très bon (haut de bande)
            bb_width > self.expansion_threshold):
            return context['score'] > 50  # Fin de pump confirmée
        
        # 4. NOUVEAU : Détection de fin pump (surachat extrême) - STANDARDISÉ
        if bb_position >= 0.85 and pattern['type'] == 'expansion':  # STANDARDISÉ: Excellent (très haut, proche bande haute)
            return context['score'] > 40  # Seuil bas pour capturer fin pump
        
        return False
    
    def _calculate_bullish_confidence(self, bb_position: float, bb_width: float, pattern: Dict, 
                                     context: Dict, volume_ratio: float, momentum_10: float) -> float:
        """Calcule la confiance pour un signal d'achat"""
        base_confidence = 0.5
        
        # Force du pattern
        base_confidence += pattern['strength'] * 0.15
        
        # Position dans les bandes - STANDARDISÉ
        if bb_position <= 0.15:  # STANDARDISÉ: Excellent (très bas, proche bande basse)
            base_confidence += 0.15
        elif bb_position <= 0.25:  # STANDARDISÉ: Très bon (bas de bande)
            base_confidence += 0.1
        elif bb_position <= 0.35:  # STANDARDISÉ: Bon (position basse)
            base_confidence += 0.05
        
        # Contexte
        base_confidence += context['confidence_boost']
        
        # Volume - SEUILS STANDARDISÉS
        if volume_ratio and volume_ratio > 1.5:  # STANDARDISÉ: Très bon
            base_confidence += 0.08
        elif volume_ratio and volume_ratio > 1.2:  # STANDARDISÉ: Bon
            base_confidence += 0.05
        
        # Momentum favorable
        if momentum_10 and momentum_10 > 0.5:
            base_confidence += 0.05
        
        return min(0.95, base_confidence)
    
    def _calculate_bearish_confidence(self, bb_position: float, bb_width: float, pattern: Dict, 
                                     context: Dict, volume_ratio: float, momentum_10: float) -> float:
        """Calcule la confiance pour un signal de vente"""
        base_confidence = 0.6  # Base plus élevée pour SELL
        
        # Force du pattern
        base_confidence += pattern['strength'] * 0.15
        
        # Position dans les bandes - STANDARDISÉ pour les pumps
        if bb_position >= 0.85:  # STANDARDISÉ: Excellent (très haut, proche bande haute)
            base_confidence += 0.15
        elif bb_position >= 0.75:  # STANDARDISÉ: Très bon (haut de bande)
            base_confidence += 0.12
        elif bb_position >= 0.65:  # STANDARDISÉ: Bon (position haute)
            base_confidence += 0.08
        elif bb_position >= 0.55:  # STANDARDISÉ: Acceptable (position neutre-haute)
            base_confidence += 0.05
        
        # Contexte
        base_confidence += context['confidence_boost']
        
        # Volume - plus important pour SELL (confirme le pump) - SEUILS STANDARDISÉS
        if volume_ratio and volume_ratio > 2.0:  # STANDARDISÉ: Excellent (début pump confirmé)
            base_confidence += 0.12
        elif volume_ratio and volume_ratio > 1.5:  # STANDARDISÉ: Très bon
            base_confidence += 0.10
        elif volume_ratio and volume_ratio > 1.2:  # STANDARDISÉ: Bon
            base_confidence += 0.08
        
        # Momentum - accepté même si positif (pump en cours)
        if momentum_10:
            if momentum_10 > 0.5:  # Momentum haussier fort = bon pour SELL au sommet
                base_confidence += 0.08
            elif momentum_10 > 0.2:  # Momentum haussier modéré
                base_confidence += 0.05
            elif momentum_10 < -0.5:  # Momentum baissier
                base_confidence += 0.05
        
        return min(0.95, base_confidence)
    
    def _get_confluence_analysis(self, symbol: str) -> Optional[Dict]:
        """Récupère l'analyse de confluence depuis Redis"""
        try:
            if not self.redis_client:
                return None
            
            cache_key = f"confluence:{symbol}"
            cached = self.redis_client.get(cache_key)
            
            if cached:
                import json
                return json.loads(cached)
            
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