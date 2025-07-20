"""
Stratégie MACD Pro
MACD avec détection de divergences, analyse de momentum et confluence multi-timeframes.
Intègre histogramme, signal quality, momentum acceleration et structure de marché.
"""
import logging
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from shared.src.enums import OrderSide
from shared.src.config import (
    REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB,
    MACD_HISTOGRAM_VERY_STRONG, MACD_HISTOGRAM_STRONG, MACD_HISTOGRAM_MODERATE, 
    MACD_HISTOGRAM_WEAK
)
from shared.src.volume_context_detector import volume_context_detector

from .base_strategy import BaseStrategy

# Import des modules d'analyse avancée
try:
    import redis  # type: ignore
except ImportError:
    redis = None  # type: ignore

logger = logging.getLogger(__name__)

class MACDProStrategy(BaseStrategy):
    """
    Stratégie MACD Pro - MACD avec analyse avancée de momentum et divergences
    BUY: MACD crossover bullish + divergence + momentum acceleration + confluence
    SELL: MACD crossover bearish + divergence + momentum deceleration + confluence
    
    Intègre :
    - Détection de divergences MACD/prix
    - Analyse de l'accélération momentum (histogramme)
    - Quality signals (force des crossovers)
    - Volume confirmation
    - ADX pour force de tendance
    - Confluence multi-timeframes
    - Zero line crosses et signal line crosses
    """
    
    def __init__(self, symbol: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(symbol, params)
        
        # Paramètres MACD avancés - SEUILS STANDARDISÉS
        symbol_params = self.params.get(symbol, {}) if self.params else {}
        self.min_histogram_threshold = symbol_params.get('macd_histogram_threshold', MACD_HISTOGRAM_WEAK)  # STANDARDISÉ: Momentum faible
        self.strong_signal_threshold = symbol_params.get('macd_strong_threshold', MACD_HISTOGRAM_VERY_STRONG)  # STANDARDISÉ: Momentum très fort
        self.min_volume_ratio = symbol_params.get('min_volume_ratio', 0.5)  # AJUSTÉ de 0.6 à 0.5 pour plus de flexibilité
        self.min_adx = symbol_params.get('min_adx', 12.0)  # AJUSTÉ de 15 à 12 pour plus de signaux
        self.confluence_threshold = symbol_params.get('confluence_threshold', 25.0)  # AJUSTÉ de 30 à 25 pour plus de signaux
        
        # Historique pour divergences
        self.price_history: List[float] = []
        self.macd_line_history: List[float] = []
        self.macd_histogram_history: List[float] = []
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
                logger.warning(f"⚠️ Redis non disponible pour MACD Pro: {e}")
                self.redis_client = None
        
        logger.info(f"🎯 MACD Pro initialisé pour {symbol} (Threshold min: {self.min_histogram_threshold:.5f}, Strong: {self.strong_signal_threshold:.5f}, ADX≥{self.min_adx})")

    @property
    def name(self) -> str:
        return "MACD_Pro_Strategy"
    
    def get_min_data_points(self) -> int:
        return 35  # Minimum pour MACD stable et divergences
    
    def analyze(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Optional[Dict]:
        """
        Analyse MACD Pro - Crossovers avec divergences et momentum
        """
        try:
            if len(df) < self.get_min_data_points():
                return None
            
            # PRIORITÉ 1: Vérifier conditions de protection défensive
            defensive_signal = self.check_defensive_conditions(df)
            if defensive_signal:
                return defensive_signal
            
            # Récupérer indicateurs MACD
            current_macd_line = self._get_current_indicator(indicators, 'macd_line')
            current_macd_signal = self._get_current_indicator(indicators, 'macd_signal')
            current_histogram = self._get_current_indicator(indicators, 'macd_histogram')
            
            if any(x is None for x in [current_macd_line, current_macd_signal, current_histogram]):
                logger.debug(f"❌ {symbol}: Indicateurs MACD incomplets")
                return None
            
            # Récupérer valeurs précédentes pour crossovers
            previous_macd_line = self._get_previous_indicator(indicators, 'macd_line')
            previous_macd_signal = self._get_previous_indicator(indicators, 'macd_signal')
            previous_histogram = self._get_previous_indicator(indicators, 'macd_histogram')
            
            if any(x is None for x in [previous_macd_line, previous_macd_signal, previous_histogram]):
                return None
            
            current_price = df['close'].iloc[-1]
            
            # Récupérer indicateurs de contexte
            adx = self._get_current_indicator(indicators, 'adx_14')
            volume_ratio = self._get_current_indicator(indicators, 'volume_ratio')
            volume_spike = indicators.get('volume_spike', False)
            momentum_10 = self._get_current_indicator(indicators, 'momentum_10')
            rsi = self._get_current_indicator(indicators, 'rsi_14')
            williams_r = self._get_current_indicator(indicators, 'williams_r')
            
            # Mettre à jour l'historique
            if current_macd_line is not None and current_histogram is not None:
                self._update_history(current_price, current_macd_line, current_histogram)
            
            # Analyser les signaux MACD
            if all(x is not None for x in [current_macd_line, current_macd_signal, current_histogram,
                                           previous_macd_line, previous_macd_signal, previous_histogram]):
                # Assertions pour MyPy
                assert current_macd_line is not None
                assert current_macd_signal is not None  
                assert current_histogram is not None
                assert previous_macd_line is not None
                assert previous_macd_signal is not None
                assert previous_histogram is not None
                
                macd_analysis = self._analyze_macd_signals(
                    current_macd_line, current_macd_signal, current_histogram,  # type: ignore
                    previous_macd_line, previous_macd_signal, previous_histogram  # type: ignore
                )
            else:
                return None
            
            # Analyser les divergences
            divergence_analysis = self._detect_macd_divergences()
            
            # Analyser le contexte
            context_analysis = self._analyze_macd_context(
                symbol, adx or 0.0, volume_ratio or 0.0, momentum_10 or 0.0, 
                rsi or 0.0, williams_r or 0.0
            )
            
            # NOUVEAU: Calculer la position du prix dans son range
            price_position = self.calculate_price_position_in_range(df)
            
            signal = None
            
            # SIGNAL D'ACHAT - MACD bullish avec contexte favorable
            if self._is_bullish_macd_signal(macd_analysis, divergence_analysis, context_analysis):
                # Vérifier la position du prix avant de générer le signal
                if not self.should_filter_signal_by_price_position(OrderSide.BUY, price_position, df):
                    confidence = self._calculate_bullish_confidence(
                        macd_analysis, divergence_analysis, context_analysis, 
                        volume_ratio or 0.0, current_histogram or 0.0
                    )
                    
                    signal = self.create_signal(
                        side=OrderSide.BUY,
                        price=current_price,
                        confidence=confidence,
                        metadata={
                            'macd_line': current_macd_line,
                            'macd_signal': current_macd_signal,
                            'macd_histogram': current_histogram,
                            'signal_type': macd_analysis['signal_type'],
                            'signal_strength': macd_analysis['strength'],
                            'divergence_type': divergence_analysis.get('type', 'none'),
                            'divergence_strength': divergence_analysis.get('strength', 0),
                            'adx': adx,
                            'volume_ratio': volume_ratio,
                            'volume_spike': volume_spike,
                            'context_score': context_analysis['score'],
                            'confluence_score': context_analysis.get('confluence_score', 0),
                            'price_position': price_position,
                            'reason': f'MACD Pro BUY ({macd_analysis["signal_type"]}, hist: {current_histogram:.5f})'
                        }
                    )
                    # Enregistrer prix d'entrée pour protection défensive
                    self.last_entry_price = current_price
                else:
                    logger.info(f"📊 MACD Pro {symbol}: Signal BUY techniquement valide mais filtré "
                              f"(position prix: {price_position:.2f})")
            
            # SIGNAL DE VENTE - MACD bearish avec contexte favorable
            elif self._is_bearish_macd_signal(macd_analysis, divergence_analysis, context_analysis):
                # Vérifier la position du prix avant de générer le signal
                if not self.should_filter_signal_by_price_position(OrderSide.SELL, price_position, df):
                    confidence = self._calculate_bearish_confidence(
                        macd_analysis, divergence_analysis, context_analysis, 
                        volume_ratio or 0.0, current_histogram or 0.0
                    )
                    
                    signal = self.create_signal(
                        side=OrderSide.SELL,
                        price=current_price,
                        confidence=confidence,
                        metadata={
                            'macd_line': current_macd_line,
                            'macd_signal': current_macd_signal,
                            'macd_histogram': current_histogram,
                            'signal_type': macd_analysis['signal_type'],
                            'signal_strength': macd_analysis['strength'],
                            'divergence_type': divergence_analysis.get('type', 'none'),
                            'divergence_strength': divergence_analysis.get('strength', 0),
                            'adx': adx,
                            'volume_ratio': volume_ratio,
                            'volume_spike': volume_spike,
                            'context_score': context_analysis['score'],
                            'confluence_score': context_analysis.get('confluence_score', 0),
                            'price_position': price_position,
                            'reason': f'MACD Pro SELL ({macd_analysis["signal_type"]}, hist: {current_histogram:.5f})'
                        }
                    )
                else:
                    logger.info(f"📊 MACD Pro {symbol}: Signal SELL techniquement valide mais filtré "
                              f"(position prix: {price_position:.2f})")
            
            if signal:
                logger.info(f"🎯 MACD Pro {symbol}: {signal.side} @ {current_price:.4f} "
                          f"({macd_analysis['signal_type']}, hist: {current_histogram:.5f}, "
                          f"Div: {divergence_analysis.get('type', 'none')}, Context: {context_analysis['score']:.1f}, "
                          f"Conf: {signal.confidence:.2f})")
                
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
            logger.error(f"❌ Erreur MACD Pro Strategy {symbol}: {e}")
            return None
    
    def _update_history(self, price: float, macd_line: float, histogram: float):
        """Met à jour l'historique pour divergences"""
        self.price_history.append(price)
        self.macd_line_history.append(macd_line)
        self.macd_histogram_history.append(histogram)
        
        # Limiter la taille
        for history in [self.price_history, self.macd_line_history, self.macd_histogram_history]:
            if len(history) > self.max_history:
                history.pop(0)
    
    def _analyze_macd_signals(self, macd_line: float, macd_signal: float, histogram: float,
                             prev_macd_line: float, prev_macd_signal: float, prev_histogram: float) -> Dict:
        """Analyse les différents types de signaux MACD"""
        analysis = {
            'signal_type': 'none',
            'strength': 0.0,
            'quality': 'low',
            'momentum_direction': 'neutral'
        }
        
        try:
            # 1. Signal Line Crossover (MACD line croise signal line)
            if prev_macd_line <= prev_macd_signal and macd_line > macd_signal:
                analysis['signal_type'] = 'bullish_crossover'
                analysis['strength'] = min(1.0, abs(macd_line - macd_signal) * 2000)
                
                # SEUILS STANDARDISÉS pour quality
                if histogram > self.strong_signal_threshold:  # >= MACD_HISTOGRAM_VERY_STRONG
                    analysis['quality'] = 'high'
                elif histogram > MACD_HISTOGRAM_STRONG:  # STANDARDISÉ: Momentum fort
                    analysis['quality'] = 'medium'
                elif histogram > self.min_histogram_threshold:  # >= MACD_HISTOGRAM_WEAK
                    analysis['quality'] = 'low'
            
            elif prev_macd_line >= prev_macd_signal and macd_line < macd_signal:
                analysis['signal_type'] = 'bearish_crossover'
                analysis['strength'] = min(1.0, abs(macd_signal - macd_line) * 2000)
                
                # SEUILS STANDARDISÉS pour quality
                if histogram < -self.strong_signal_threshold:  # <= -MACD_HISTOGRAM_VERY_STRONG
                    analysis['quality'] = 'high'
                elif histogram < -MACD_HISTOGRAM_STRONG:  # STANDARDISÉ: Momentum fort
                    analysis['quality'] = 'medium'
                elif histogram < -self.min_histogram_threshold:  # <= -MACD_HISTOGRAM_WEAK
                    analysis['quality'] = 'low'
            
            # 2. Zero Line Cross (MACD line croise zéro)
            elif prev_macd_line <= 0 and macd_line > 0:
                analysis['signal_type'] = 'bullish_zero_cross'
                analysis['strength'] = min(1.0, macd_line * 1000)
                analysis['quality'] = 'high'  # Zero cross = signal fort
            
            elif prev_macd_line >= 0 and macd_line < 0:
                analysis['signal_type'] = 'bearish_zero_cross'
                analysis['strength'] = min(1.0, abs(macd_line) * 1000)
                analysis['quality'] = 'high'
            
            # 3. Histogram Change (changement de momentum) - SEUILS STANDARDISÉS
            elif prev_histogram <= 0 and histogram > MACD_HISTOGRAM_MODERATE:  # STANDARDISÉ: Momentum modéré minimum
                analysis['signal_type'] = 'momentum_bullish'
                analysis['strength'] = min(1.0, histogram * 1500)
                analysis['quality'] = 'medium'
            
            elif prev_histogram >= 0 and histogram < -MACD_HISTOGRAM_MODERATE:  # STANDARDISÉ: Momentum modéré minimum
                analysis['signal_type'] = 'momentum_bearish'
                analysis['strength'] = min(1.0, abs(histogram) * 1500)
                analysis['quality'] = 'medium'
            
            # 4. Direction du momentum
            if histogram > prev_histogram:
                analysis['momentum_direction'] = 'accelerating'
            elif histogram < prev_histogram:
                analysis['momentum_direction'] = 'decelerating'
            
        except Exception as e:
            logger.debug(f"Erreur analyse signaux MACD: {e}")
        
        return analysis
    
    def _detect_macd_divergences(self) -> Dict:
        """Détecte les divergences entre prix et MACD"""
        divergence = {
            'type': 'none',
            'strength': 0.0,
            'confidence': 0.0
        }
        
        try:
            if len(self.price_history) < 12 or len(self.macd_line_history) < 12:
                return divergence
            
            # Analyser les 12 derniers points
            recent_prices = self.price_history[-12:]
            recent_macd = self.macd_line_history[-12:]
            
            # Trouver les extremums
            price_max_idx = np.argmax(recent_prices)
            price_min_idx = np.argmin(recent_prices)
            macd_max_idx = np.argmax(recent_macd)
            macd_min_idx = np.argmin(recent_macd)
            
            # Divergence haussière : prix fait plus bas, MACD fait plus haut
            if (price_min_idx > 6 and macd_max_idx > 6 and 
                abs(price_min_idx - macd_max_idx) <= 4):
                
                price_decline = (recent_prices[price_min_idx] - recent_prices[0]) / recent_prices[0]
                macd_improvement = recent_macd[macd_max_idx] - recent_macd[0]
                
                if price_decline < -0.015 and macd_improvement > MACD_HISTOGRAM_WEAK:  # STANDARDISÉ: Momentum faible minimum
                    divergence['type'] = 'bullish'
                    divergence['strength'] = min(1.0, abs(price_decline) * 30 + macd_improvement * 5000)
                    divergence['confidence'] = 0.75
            
            # Divergence baissière : prix fait plus haut, MACD fait plus bas
            elif (price_max_idx > 6 and macd_min_idx > 6 and 
                  abs(price_max_idx - macd_min_idx) <= 4):
                
                price_rise = (recent_prices[price_max_idx] - recent_prices[0]) / recent_prices[0]
                macd_decline = recent_macd[0] - recent_macd[macd_min_idx]
                
                if price_rise > 0.015 and macd_decline > MACD_HISTOGRAM_WEAK:  # STANDARDISÉ: Momentum faible minimum
                    divergence['type'] = 'bearish'
                    divergence['strength'] = min(1.0, price_rise * 30 + macd_decline * 5000)
                    divergence['confidence'] = 0.75
            
        except Exception as e:
            logger.debug(f"Erreur détection divergence MACD: {e}")
        
        return divergence
    
    def _analyze_macd_context(self, symbol: str, adx: float, volume_ratio: float, 
                             momentum_10: float, rsi: float, williams_r: float) -> Dict[str, Any]:
        """Analyse le contexte pour valider les signaux MACD"""
        context: Dict[str, Any] = {
            'score': 0.0,
            'confidence_boost': 0.0,
            'confluence_score': 0.0,
            'details': []
        }
        
        try:
            # 1. Force de tendance (ADX)
            from shared.src.config import ADX_STRONG_TREND_THRESHOLD, ADX_WEAK_TREND_THRESHOLD
            if adx and adx >= self.min_adx:
                if adx >= ADX_STRONG_TREND_THRESHOLD:
                    context['score'] = float(context['score']) + 25
                    context['confidence_boost'] = float(context['confidence_boost']) + 0.08
                    details_list = context.get('details', [])
                    if isinstance(details_list, list):
                        details_list.append(f"ADX fort ({adx:.1f})")
                else:
                    context['score'] = float(context['score']) + 15
                    context['confidence_boost'] = float(context['confidence_boost']) + 0.05
                    details_list = context.get('details', [])
                    if isinstance(details_list, list):
                        details_list.append(f"ADX modéré ({adx:.1f})")
            elif adx and adx >= ADX_WEAK_TREND_THRESHOLD:
                context['score'] = float(context['score']) + 5
                details_list = context.get('details', [])
                if isinstance(details_list, list):
                    details_list.append(f"ADX faible ({adx:.1f})")
            else:
                context['score'] = float(context['score']) - 10
                details_list = context.get('details', [])
                if isinstance(details_list, list):
                    details_list.append(f"ADX insuffisant ({adx or 0:.1f})")
            
            # 2. Volume confirmation - SEUILS CONTEXTUELS ADAPTATIFS
            if volume_ratio:
                try:
                    # Détection du contexte market pour volume adaptatif
                    contextual_threshold, context_name, contextual_score = volume_context_detector.get_contextual_volume_threshold(
                        base_volume_ratio=volume_ratio,
                        rsi=rsi,
                        cci=0.0,
                        adx=adx,
                        signal_type="BUY"
                    )
                    
                    volume_quality = volume_context_detector.get_volume_quality_description(
                        volume_ratio, context_name
                    )
                    
                    # Scoring contextuel du volume
                    if volume_ratio >= 2.0:  # Excellent absolu
                        context['score'] = float(context['score']) + 30
                        context['confidence_boost'] = float(context['confidence_boost']) + 0.12
                        details_list = context.get('details', [])
                        if isinstance(details_list, list):
                            details_list.append(f"Volume excellent ({volume_ratio:.1f}x) - {context_name}")
                    elif volume_ratio >= 1.5:  # Très bon
                        context['score'] = float(context['score']) + 25
                        context['confidence_boost'] = float(context['confidence_boost']) + 0.1
                        details_list = context.get('details', [])
                        if isinstance(details_list, list):
                            details_list.append(f"Volume très bon ({volume_ratio:.1f}x) - {context_name}")
                    elif volume_ratio >= 1.2:  # Bon
                        context['score'] = float(context['score']) + 20
                        context['confidence_boost'] = float(context['confidence_boost']) + 0.08
                        details_list = context.get('details', [])
                        if isinstance(details_list, list):
                            details_list.append(f"Volume bon ({volume_ratio:.1f}x) - {context_name}")
                    elif volume_ratio >= contextual_threshold:  # Contextuel acceptable
                        # Score basé sur la qualité contextuelle
                        score_bonus = int(contextual_score * 20)  # 0-20 points
                        confidence_bonus = contextual_score * 0.06  # 0-0.06 boost
                        context['score'] = float(context['score']) + score_bonus
                        context['confidence_boost'] = float(context['confidence_boost']) + confidence_bonus
                        details_list = context.get('details', [])
                        if isinstance(details_list, list):
                            details_list.append(f"Volume {volume_quality.lower()} ({volume_ratio:.1f}x) - {context_name}")
                    else:
                        # En dessous du seuil contextuel mais pas forcément éliminatoire
                        penalty = min(5, int((1 - volume_ratio/contextual_threshold) * 10))
                        context['score'] = float(context['score']) - penalty
                        details_list = context.get('details', [])
                        if isinstance(details_list, list):
                            details_list.append(f"Volume {volume_quality.lower()} ({volume_ratio:.1f}x) - {context_name}")
                        
                except Exception:
                    # Fallback sur la logique standard si erreur
                    if volume_ratio >= 1.0:
                        context['score'] = float(context['score']) + 15
                        context['confidence_boost'] = float(context['confidence_boost']) + 0.05
                        details_list = context.get('details', [])
                        if isinstance(details_list, list):
                            details_list.append(f"Volume acceptable ({volume_ratio:.1f}x) - standard")
                    else:
                        context['score'] = float(context['score']) - 5
                        details_list = context.get('details', [])
                        if isinstance(details_list, list):
                            details_list.append(f"Volume faible ({volume_ratio:.1f}x) - standard")
                        
            else:
                context['score'] = float(context['score']) - 5
                details_list = context.get('details', [])
                if isinstance(details_list, list):
                    details_list.append("Volume non disponible")
            
            # 3. Momentum général
            if momentum_10:
                momentum_strength = abs(momentum_10)
                if momentum_strength > 1.2:
                    context['score'] = float(context['score']) + 20
                    context['confidence_boost'] = float(context['confidence_boost']) + 0.08
                    details_list = context.get('details', [])
                    if isinstance(details_list, list):
                        details_list.append(f"Momentum très fort ({momentum_10:.2f})")
                elif momentum_strength > 0.6:
                    context['score'] = float(context['score']) + 15
                    context['confidence_boost'] = float(context['confidence_boost']) + 0.05
                    details_list = context.get('details', [])
                    if isinstance(details_list, list):
                        details_list.append(f"Momentum fort ({momentum_10:.2f})")
                elif momentum_strength > 0.3:
                    context['score'] = float(context['score']) + 10
                    details_list = context.get('details', [])
                    if isinstance(details_list, list):
                        details_list.append(f"Momentum modéré ({momentum_10:.2f})")
                else:
                    context['score'] = float(context['score']) + 2
                    details_list = context.get('details', [])
                    if isinstance(details_list, list):
                        details_list.append(f"Momentum faible ({momentum_10:.2f})")
            
            # 4. RSI support
            if rsi:
                if rsi <= 35 or rsi >= 65:  # Zones favorables pour MACD
                    context['score'] = float(context['score']) + 15
                    context['confidence_boost'] = float(context['confidence_boost']) + 0.05
                    details_list = context.get('details', [])
                    if isinstance(details_list, list):
                        details_list.append(f"RSI favorable ({rsi:.1f})")
                else:
                    context['score'] = float(context['score']) + 5
                    details_list = context.get('details', [])
                    if isinstance(details_list, list):
                        details_list.append(f"RSI neutre ({rsi:.1f})")
            
            # 5. Williams %R timing
            if williams_r:
                if williams_r <= -75 or williams_r >= -25:
                    context['score'] = float(context['score']) + 10
                    context['confidence_boost'] = float(context['confidence_boost']) + 0.03
                    details_list = context.get('details', [])
                    if isinstance(details_list, list):
                        details_list.append(f"Williams timing ({williams_r:.1f})")
                else:
                    context['score'] = float(context['score']) + 5
                    details_list = context.get('details', [])
                    if isinstance(details_list, list):
                        details_list.append(f"Williams neutre ({williams_r:.1f})")
            
            # 6. Confluence multi-timeframes
            if self.redis_client:
                confluence_data = self._get_confluence_analysis(symbol)
                if confluence_data:
                    confluence_score = confluence_data.get('confluence_score', 0)
                    context['confluence_score'] = confluence_score
                    
                    if confluence_score >= 60:
                        context['score'] = float(context['score']) + 20
                        context['confidence_boost'] = float(context['confidence_boost']) + 0.08
                        details_list = context.get('details', [])
                        if isinstance(details_list, list):
                            details_list.append(f"Confluence forte ({confluence_score:.1f}%)")
                    elif confluence_score >= 45:
                        context['score'] = float(context['score']) + 10
                        context['confidence_boost'] = float(context['confidence_boost']) + 0.03
                        details_list = context.get('details', [])
                        if isinstance(details_list, list):
                            details_list.append(f"Confluence modérée ({confluence_score:.1f}%)")
            
            # Normaliser
            context['score'] = max(0.0, min(100.0, float(context['score'])))
            context['confidence_boost'] = max(0.0, min(0.25, float(context['confidence_boost'])))
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse contexte MACD: {e}")
            context['score'] = 25
        
        return context
    
    def _is_bullish_macd_signal(self, macd_analysis: Dict, divergence: Dict, context: Dict) -> bool:
        """Détermine si les conditions d'achat MACD sont remplies"""
        signal_type = macd_analysis['signal_type']
        
        # Pas de signal MACD
        if signal_type == 'none':
            return False
        
        # Signals baissiers
        if 'bearish' in signal_type:
            return False
        
        # Score de contexte minimum
        if context['score'] < 20:  # AJUSTÉ de 25 à 20 pour capturer plus de signaux bullish
            return False
        
        # Confluence minimum si disponible
        confluence_score = context.get('confluence_score', 0)
        if confluence_score > 0 and confluence_score < self.confluence_threshold:
            return False
        
        # Signaux forts (zero cross, high quality)
        if signal_type in ['bullish_zero_cross'] or macd_analysis['quality'] == 'high':
            return context['score'] > 30  # AJUSTÉ de 40 à 30 pour plus de réactivité
        
        # Divergence haussière forte
        if divergence['type'] == 'bullish' and divergence['strength'] > 0.4:  # AJUSTÉ de 0.5 à 0.4
            return context['score'] > 25  # AJUSTÉ de 30 à 25
        
        # Signaux standards
        if signal_type in ['bullish_crossover', 'momentum_bullish']:
            return context['score'] > 35 and macd_analysis['strength'] > 0.15  # AJUSTÉ de 45 à 35 et 0.2 à 0.15
        
        return False
    
    def _is_bearish_macd_signal(self, macd_analysis: Dict, divergence: Dict, context: Dict) -> bool:
        """Détermine si les conditions de vente MACD sont remplies"""
        signal_type = macd_analysis['signal_type']
        
        # Pas de signal MACD
        if signal_type == 'none':
            return False
        
        # Signals haussiers
        if 'bullish' in signal_type:
            return False
        
        # Score de contexte minimum assoupli pour SELL
        if context['score'] < 30:
            return False
        
        # Confluence minimum si disponible - assoupli de 5 points
        confluence_score = context.get('confluence_score', 0)
        if confluence_score > 0 and confluence_score < (self.confluence_threshold - 5):
            return False
        
        # Signaux forts (zero cross, high quality) - assoupli de 50 à 45
        if signal_type in ['bearish_zero_cross'] or macd_analysis['quality'] == 'high':
            return context['score'] > 45
        
        # Divergence baissière forte - assoupli de 40 à 35
        if divergence['type'] == 'bearish' and divergence['strength'] > 0.6:
            return context['score'] > 35
        
        # Signaux standards - assoupli de 60 à 55
        if signal_type in ['bearish_crossover', 'momentum_bearish']:
            return context['score'] > 55 and macd_analysis['strength'] > 0.3
        
        return False
    
    def _calculate_bullish_confidence(self, macd_analysis: Dict, divergence: Dict, context: Dict, 
                                     volume_ratio: float, histogram: float) -> float:
        """Calcule la confiance pour un signal d'achat"""
        base_confidence = 0.5
        
        # Force du signal MACD
        base_confidence += macd_analysis['strength'] * 0.2
        
        # Qualité du signal
        if macd_analysis['quality'] == 'high':
            base_confidence += 0.1
        elif macd_analysis['quality'] == 'medium':
            base_confidence += 0.05
        
        # Type de signal (zero cross = plus fort)
        if 'zero_cross' in macd_analysis['signal_type']:
            base_confidence += 0.08
        
        # Divergence
        if divergence['type'] == 'bullish':
            base_confidence += divergence['strength'] * 0.15
        
        # Contexte
        base_confidence += context['confidence_boost']
        
        # Volume - SEUILS STANDARDISÉS
        if volume_ratio and volume_ratio > 1.5:  # STANDARDISÉ: Très bon
            base_confidence += 0.08
        elif volume_ratio and volume_ratio > 1.2:  # STANDARDISÉ: Bon
            base_confidence += 0.05
        
        # Force histogramme
        if histogram > self.strong_signal_threshold:
            base_confidence += 0.05
        
        return min(0.95, base_confidence)
    
    def _calculate_bearish_confidence(self, macd_analysis: Dict, divergence: Dict, context: Dict, 
                                     volume_ratio: float, histogram: float) -> float:
        """Calcule la confiance pour un signal de vente"""
        base_confidence = 0.5
        
        # Force du signal MACD
        base_confidence += macd_analysis['strength'] * 0.2
        
        # Qualité du signal
        if macd_analysis['quality'] == 'high':
            base_confidence += 0.1
        elif macd_analysis['quality'] == 'medium':
            base_confidence += 0.05
        
        # Type de signal (zero cross = plus fort)
        if 'zero_cross' in macd_analysis['signal_type']:
            base_confidence += 0.08
        
        # Divergence
        if divergence['type'] == 'bearish':
            base_confidence += divergence['strength'] * 0.15
        
        # Contexte
        base_confidence += context['confidence_boost']
        
        # Volume - SEUILS STANDARDISÉS
        if volume_ratio and volume_ratio > 1.5:  # STANDARDISÉ: Très bon
            base_confidence += 0.08
        elif volume_ratio and volume_ratio > 1.2:  # STANDARDISÉ: Bon
            base_confidence += 0.05
        
        # Force histogramme
        if histogram < -self.strong_signal_threshold:
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