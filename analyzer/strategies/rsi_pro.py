"""
Stratégie RSI Pro
RSI avec contexte multi-timeframes, momentum, divergences et confluence avancée.
Intègre ADX, CCI, Williams %R, VWAP et analyse de structure pour des signaux précis.
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

class RSIProStrategy(BaseStrategy):
    """
    Stratégie RSI Pro - RSI avec contexte multi-timeframes et momentum
    BUY: RSI oversold + divergence haussière + confluence + momentum favorable
    SELL: RSI overbought + divergence baissière + confluence + momentum favorable
    
    Intègre :
    - Analyse de divergences RSI/prix
    - Confluence multi-timeframes
    - Confirmation CCI et Williams %R
    - VWAP pour direction de tendance
    - Volume et momentum
    - Structure de marché
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        super().__init__(symbol, params)
        
        # Paramètres RSI avancés
        symbol_params = self.params.get(symbol, {}) if self.params else {}
        self.oversold_threshold = symbol_params.get('rsi_oversold', 25)  # Plus strict
        self.overbought_threshold = symbol_params.get('rsi_overbought', 75)  # Plus strict
        self.extreme_oversold = symbol_params.get('rsi_extreme_oversold', 15)
        self.extreme_overbought = symbol_params.get('rsi_extreme_overbought', 85)
        self.min_adx = symbol_params.get('min_adx', 20.0)
        self.confluence_threshold = symbol_params.get('confluence_threshold', 50.0)
        
        # Historique pour détection de divergences
        self.price_history = []
        self.rsi_history = []
        self.max_history = 20
        
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
                logger.warning(f"⚠️ Redis non disponible pour RSI Pro: {e}")
                self.redis_client = None
        
        logger.info(f"🎯 RSI Pro initialisé pour {symbol} (RSI: {self.oversold_threshold}-{self.overbought_threshold}, ADX≥{self.min_adx})")

    @property
    def name(self) -> str:
        return "RSI_Pro_Strategy"
    
    def get_min_data_points(self) -> int:
        return 20  # Minimum pour RSI et divergences
    
    def analyze(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Optional[Dict]:
        """
        Analyse RSI Pro - RSI avec contexte et divergences
        """
        try:
            if len(df) < self.get_min_data_points():
                return None
            
            # Récupérer indicateurs principaux
            current_rsi = self._get_current_indicator(indicators, 'rsi_14')
            if current_rsi is None:
                logger.debug(f"❌ {symbol}: RSI non disponible")
                return None
            
            current_price = df['close'].iloc[-1]
            
            # Récupérer indicateurs de contexte
            adx = self._get_current_indicator(indicators, 'adx_14')
            cci = self._get_current_indicator(indicators, 'cci_20')
            williams_r = self._get_current_indicator(indicators, 'williams_r')
            vwap = self._get_current_indicator(indicators, 'vwap_10')
            volume_ratio = self._get_current_indicator(indicators, 'volume_ratio')
            volume_spike = indicators.get('volume_spike', False)
            momentum_10 = self._get_current_indicator(indicators, 'momentum_10')
            bb_position = self._get_current_indicator(indicators, 'bb_position')
            
            # Mettre à jour l'historique pour divergences
            self._update_history(current_price, current_rsi)
            
            # Analyser le contexte multi-timeframes
            context_analysis = self._analyze_rsi_context(
                symbol, current_rsi, adx, cci, williams_r, vwap, 
                volume_ratio, momentum_10, bb_position
            )
            
            # Détecter les divergences
            divergence_analysis = self._detect_rsi_divergences()
            
            signal = None
            
            # SIGNAL D'ACHAT - RSI oversold avec contexte favorable
            if self._is_bullish_rsi_signal(current_rsi, context_analysis, divergence_analysis):
                confidence = self._calculate_bullish_confidence(
                    current_rsi, context_analysis, divergence_analysis, current_price, vwap
                )
                
                signal = self.create_signal(
                    side=OrderSide.BUY,
                    price=current_price,
                    confidence=confidence,
                    metadata={
                        'rsi': current_rsi,
                        'adx': adx,
                        'cci': cci,
                        'williams_r': williams_r,
                        'vwap_distance': ((current_price - vwap) / vwap * 100) if vwap else 0,
                        'volume_ratio': volume_ratio,
                        'volume_spike': volume_spike,
                        'context_score': context_analysis['score'],
                        'divergence_type': divergence_analysis.get('type', 'none'),
                        'divergence_strength': divergence_analysis.get('strength', 0),
                        'confluence_score': context_analysis.get('confluence_score', 0),
                        'reason': f'RSI Pro BUY (RSI: {current_rsi:.1f}, Context: {context_analysis["score"]:.1f})'
                    }
                )
            
            # SIGNAL DE VENTE - RSI overbought avec contexte favorable
            elif self._is_bearish_rsi_signal(current_rsi, context_analysis, divergence_analysis):
                confidence = self._calculate_bearish_confidence(
                    current_rsi, context_analysis, divergence_analysis, current_price, vwap
                )
                
                signal = self.create_signal(
                    side=OrderSide.SELL,
                    price=current_price,
                    confidence=confidence,
                    metadata={
                        'rsi': current_rsi,
                        'adx': adx,
                        'cci': cci,
                        'williams_r': williams_r,
                        'vwap_distance': ((current_price - vwap) / vwap * 100) if vwap else 0,
                        'volume_ratio': volume_ratio,
                        'volume_spike': volume_spike,
                        'context_score': context_analysis['score'],
                        'divergence_type': divergence_analysis.get('type', 'none'),
                        'divergence_strength': divergence_analysis.get('strength', 0),
                        'confluence_score': context_analysis.get('confluence_score', 0),
                        'reason': f'RSI Pro SELL (RSI: {current_rsi:.1f}, Context: {context_analysis["score"]:.1f})'
                    }
                )
            
            if signal:
                logger.info(f"🎯 RSI Pro {symbol}: {signal.side} @ {current_price:.4f} "
                          f"(RSI: {current_rsi:.1f}, Context: {context_analysis['score']:.1f}, "
                          f"Div: {divergence_analysis.get('type', 'none')}, Conf: {signal.confidence:.2f})")
                
                # Convertir StrategySignal en dict pour compatibilité
                return {
                    'strategy': signal.strategy,
                    'symbol': signal.symbol,
                    'side': signal.side,
                    'timestamp': signal.timestamp.isoformat(),
                    'price': signal.price,
                    'confidence': signal.confidence,
                    'strength': signal.strength.value,
                    'metadata': signal.metadata
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Erreur RSI Pro Strategy {symbol}: {e}")
            return None
    
    def _analyze_rsi_context(self, symbol: str, rsi: float, adx: float, cci: float, 
                            williams_r: float, vwap: float, volume_ratio: float, 
                            momentum_10: float, bb_position: float) -> Dict:
        """Analyse le contexte pour valider les signaux RSI"""
        context = {
            'score': 0.0,
            'confidence_boost': 0.0,
            'confluence_score': 0.0,
            'details': []
        }
        
        try:
            # 1. Force de tendance (ADX)
            if adx and adx >= self.min_adx:
                context['score'] += 20
                context['confidence_boost'] += 0.05
                context['details'].append(f"ADX tendance ({adx:.1f})")
            elif adx and adx >= 15:
                context['score'] += 10
                context['details'].append(f"ADX faible ({adx:.1f})")
            else:
                context['score'] -= 5
                context['details'].append(f"ADX insuffisant ({adx or 0:.1f})")
            
            # 2. Confirmation CCI
            if cci:
                if abs(cci) > 100:  # CCI extrême
                    context['score'] += 15
                    context['confidence_boost'] += 0.05
                    context['details'].append(f"CCI extrême ({cci:.1f})")
                elif abs(cci) > 50:
                    context['score'] += 10
                    context['details'].append(f"CCI actif ({cci:.1f})")
                else:
                    context['score'] += 5
                    context['details'].append(f"CCI neutre ({cci:.1f})")
            
            # 3. Williams %R pour timing
            if williams_r:
                if williams_r <= -80:  # Extrême oversold
                    context['score'] += 20
                    context['confidence_boost'] += 0.08
                    context['details'].append(f"Williams oversold ({williams_r:.1f})")
                elif williams_r >= -20:  # Extrême overbought
                    context['score'] += 20
                    context['confidence_boost'] += 0.08
                    context['details'].append(f"Williams overbought ({williams_r:.1f})")
                else:
                    context['score'] += 5
                    context['details'].append(f"Williams neutre ({williams_r:.1f})")
            
            # 4. VWAP direction
            if self.price_history and vwap:
                current_price = self.price_history[-1]
                vwap_distance = (current_price - vwap) / vwap * 100
                
                if abs(vwap_distance) > 2:  # Prix éloigné du VWAP
                    context['score'] += 15
                    context['confidence_boost'] += 0.05
                    context['details'].append(f"Prix vs VWAP ({vwap_distance:+.1f}%)")
                else:
                    context['score'] += 5
                    context['details'].append(f"Prix proche VWAP ({vwap_distance:+.1f}%)")
            
            # 5. Volume confirmation
            if volume_ratio and volume_ratio > 1.3:
                context['score'] += 20
                context['confidence_boost'] += 0.08
                context['details'].append(f"Volume fort ({volume_ratio:.1f}x)")
            elif volume_ratio and volume_ratio > 0.8:
                context['score'] += 10
                context['details'].append(f"Volume normal ({volume_ratio:.1f}x)")
            else:
                context['score'] -= 5
                context['details'].append(f"Volume faible ({volume_ratio or 0:.1f}x)")
            
            # 6. Momentum support
            if momentum_10:
                momentum_strength = abs(momentum_10)
                if momentum_strength > 1.5:
                    context['score'] += 15
                    context['confidence_boost'] += 0.05
                    context['details'].append(f"Momentum fort ({momentum_10:.2f})")
                elif momentum_strength > 0.8:
                    context['score'] += 10
                    context['details'].append(f"Momentum modéré ({momentum_10:.2f})")
                else:
                    context['score'] += 2
                    context['details'].append(f"Momentum faible ({momentum_10:.2f})")
            
            # 7. Position Bollinger
            if bb_position is not None:
                if bb_position <= 0.15 or bb_position >= 0.85:  # Zones extrêmes
                    context['score'] += 15
                    context['confidence_boost'] += 0.05
                    context['details'].append(f"BB position extrême ({bb_position:.2f})")
                else:
                    context['score'] += 5
                    context['details'].append(f"BB position normale ({bb_position:.2f})")
            
            # 8. Confluence multi-timeframes
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
            logger.error(f"❌ Erreur analyse contexte RSI: {e}")
            context['score'] = 25
        
        return context
    
    def _update_history(self, price: float, rsi: float):
        """Met à jour l'historique pour la détection de divergences"""
        self.price_history.append(price)
        self.rsi_history.append(rsi)
        
        # Limiter la taille de l'historique
        if len(self.price_history) > self.max_history:
            self.price_history.pop(0)
        if len(self.rsi_history) > self.max_history:
            self.rsi_history.pop(0)
    
    def _detect_rsi_divergences(self) -> Dict:
        """Détecte les divergences entre prix et RSI"""
        divergence = {
            'type': 'none',
            'strength': 0.0,
            'confidence': 0.0
        }
        
        try:
            if len(self.price_history) < 10 or len(self.rsi_history) < 10:
                return divergence
            
            # Analyser les 10 derniers points
            recent_prices = self.price_history[-10:]
            recent_rsi = self.rsi_history[-10:]
            
            # Trouver les extremums
            price_max_idx = np.argmax(recent_prices)
            price_min_idx = np.argmin(recent_prices)
            rsi_max_idx = np.argmax(recent_rsi)
            rsi_min_idx = np.argmin(recent_rsi)
            
            # Divergence haussière : prix fait un plus bas, RSI fait un plus haut
            if (price_min_idx > 5 and rsi_max_idx > 5 and 
                abs(price_min_idx - rsi_max_idx) <= 3):
                
                price_decline = (recent_prices[price_min_idx] - recent_prices[0]) / recent_prices[0]
                rsi_improvement = recent_rsi[rsi_max_idx] - recent_rsi[0]
                
                if price_decline < -0.01 and rsi_improvement > 2:  # Divergence significative
                    divergence['type'] = 'bullish'
                    divergence['strength'] = min(1.0, abs(price_decline) * 50 + rsi_improvement / 20)
                    divergence['confidence'] = 0.7
            
            # Divergence baissière : prix fait un plus haut, RSI fait un plus bas
            elif (price_max_idx > 5 and rsi_min_idx > 5 and 
                  abs(price_max_idx - rsi_min_idx) <= 3):
                
                price_rise = (recent_prices[price_max_idx] - recent_prices[0]) / recent_prices[0]
                rsi_decline = recent_rsi[0] - recent_rsi[rsi_min_idx]
                
                if price_rise > 0.01 and rsi_decline > 2:  # Divergence significative
                    divergence['type'] = 'bearish'
                    divergence['strength'] = min(1.0, price_rise * 50 + rsi_decline / 20)
                    divergence['confidence'] = 0.7
            
        except Exception as e:
            logger.debug(f"Erreur détection divergence RSI: {e}")
        
        return divergence
    
    def _is_bullish_rsi_signal(self, rsi: float, context: Dict, divergence: Dict) -> bool:
        """Détermine si les conditions d'achat RSI sont remplies"""
        # RSI oversold
        if rsi > self.oversold_threshold:
            return False
        
        # Score de contexte minimum
        if context['score'] < 35:
            return False
        
        # Si confluence disponible, la vérifier
        confluence_score = context.get('confluence_score', 0)
        if confluence_score > 0 and confluence_score < self.confluence_threshold:
            return False
        
        # Bonus pour divergence haussière
        if divergence['type'] == 'bullish' and divergence['strength'] > 0.5:
            return True
        
        # Conditions extrêmes
        if rsi <= self.extreme_oversold and context['score'] > 50:
            return True
        
        # Conditions standard
        return context['score'] > 60
    
    def _is_bearish_rsi_signal(self, rsi: float, context: Dict, divergence: Dict) -> bool:
        """Détermine si les conditions de vente RSI sont remplies"""
        # RSI overbought
        if rsi < self.overbought_threshold:
            return False
        
        # Score de contexte minimum
        if context['score'] < 35:
            return False
        
        # Si confluence disponible, la vérifier
        confluence_score = context.get('confluence_score', 0)
        if confluence_score > 0 and confluence_score < self.confluence_threshold:
            return False
        
        # Bonus pour divergence baissière
        if divergence['type'] == 'bearish' and divergence['strength'] > 0.5:
            return True
        
        # Conditions extrêmes
        if rsi >= self.extreme_overbought and context['score'] > 50:
            return True
        
        # Conditions standard
        return context['score'] > 60
    
    def _calculate_bullish_confidence(self, rsi: float, context: Dict, divergence: Dict, 
                                     price: float, vwap: float) -> float:
        """Calcule la confiance pour un signal d'achat"""
        base_confidence = 0.5
        
        # Force du RSI oversold
        oversold_strength = (self.oversold_threshold - rsi) / self.oversold_threshold
        base_confidence += oversold_strength * 0.2
        
        # Contexte
        base_confidence += context['confidence_boost']
        
        # Divergence
        if divergence['type'] == 'bullish':
            base_confidence += divergence['strength'] * 0.15
        
        # VWAP support
        if vwap and price < vwap:
            vwap_discount = abs(price - vwap) / vwap
            base_confidence += min(0.1, vwap_discount * 5)
        
        return min(0.95, base_confidence)
    
    def _calculate_bearish_confidence(self, rsi: float, context: Dict, divergence: Dict, 
                                     price: float, vwap: float) -> float:
        """Calcule la confiance pour un signal de vente"""
        base_confidence = 0.5
        
        # Force du RSI overbought
        overbought_strength = (rsi - self.overbought_threshold) / (100 - self.overbought_threshold)
        base_confidence += overbought_strength * 0.2
        
        # Contexte
        base_confidence += context['confidence_boost']
        
        # Divergence
        if divergence['type'] == 'bearish':
            base_confidence += divergence['strength'] * 0.15
        
        # VWAP resistance
        if vwap and price > vwap:
            vwap_premium = (price - vwap) / vwap
            base_confidence += min(0.1, vwap_premium * 5)
        
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