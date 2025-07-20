"""
Stratégie Reversal Divergence Pro
Divergences multi-indicateurs avec détection fractale, confluence et structure de marché.
Intègre RSI, MACD, CCI, momentum et analyse cross-timeframes pour retournements précis.
"""
import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from shared.src.enums import OrderSide
from shared.src.config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, REDIS_DB

from .base_strategy import BaseStrategy

# Import des modules d'analyse avancée
try:
    import redis  # type: ignore
except ImportError:
    redis = None  # type: ignore

logger = logging.getLogger(__name__)

@dataclass
class DivergenceSignal:
    """Signal de divergence détecté"""
    type: str  # 'bullish', 'bearish', 'hidden_bullish', 'hidden_bearish'
    indicator: str  # 'rsi', 'macd', 'cci', 'momentum'
    strength: float  # 0.0 à 1.0
    price_points: Tuple[float, float]  # (prix_début, prix_fin)
    indicator_points: Tuple[float, float]  # (indicateur_début, indicateur_fin)
    time_span: int  # Nombre de périodes
    confidence: float  # 0.0 à 1.0

class ReversalDivergenceProStrategy(BaseStrategy):
    """
    Stratégie Reversal Divergence Pro - Divergences multi-indicateurs avec analyse avancée
    BUY: Divergence bullish multiple + structure supportive + confluence + momentum favorable
    SELL: Divergence bearish multiple + structure supportive + confluence + momentum favorable
    
    Intègre :
    - Divergences classiques et cachées (RSI, MACD, CCI, Momentum)
    - Détection de fractales pour points pivots précis
    - Confluence multi-timeframes
    - Structure de marché (swing highs/lows)
    - Volume confirmation pour retournements
    - Filtres anti-fausses divergences
    - Multiple divergences simultanées = signal renforcé
    """
    
    def __init__(self, symbol: str, params: Optional[Dict[str, Any]] = None):
        super().__init__(symbol, params)
        
        # Paramètres divergence avancés
        symbol_params = self.params.get(symbol, {}) if self.params else {}
        self.lookback_periods = symbol_params.get('div_lookback', 20)
        self.min_swing_strength = symbol_params.get('min_swing_strength', 2)  # AJUSTÉ de 3 à 2 pour plus de détection
        self.min_price_change = symbol_params.get('min_price_change', 0.7)  # AJUSTÉ de 1.0 à 0.7 pour plus de sensibilité
        self.min_indicator_change = symbol_params.get('min_indicator_change', 1.5)  # AJUSTÉ de 2.0 à 1.5 pour plus de signaux
        self.confluence_threshold = symbol_params.get('confluence_threshold', 25.0)  # AJUSTÉ de 35 à 25 pour plus de signaux
        self.multiple_div_bonus = symbol_params.get('multiple_div_bonus', 0.15)  # Bonus multi-divergences
        
        # Historique pour divergences
        self.price_history: List[float] = []
        self.rsi_history: List[float] = []
        self.macd_history: List[float] = []
        self.cci_history: List[float] = []
        self.momentum_history: List[float] = []
        self.max_history = 30
        
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
                logger.warning(f"⚠️ Redis non disponible pour Divergence Pro: {e}")
                self.redis_client = None
        
        logger.info(f"🎯 Reversal Divergence Pro initialisé pour {symbol} (Lookback: {self.lookback_periods}, Swing: {self.min_swing_strength})")

    @property
    def name(self) -> str:
        return "Reversal_Divergence_Pro_Strategy"
    
    def get_min_data_points(self) -> int:
        return max(40, self.lookback_periods + 15)  # Plus de données pour swings robustes
    
    def analyze(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Optional[Dict]:
        """
        Analyse Reversal Divergence Pro - Divergences multi-indicateurs
        """
        try:
            if len(df) < self.get_min_data_points():
                return None
            
            # PRIORITÉ 1: Vérifier conditions de protection défensive
            defensive_signal = self.check_defensive_conditions(df)
            if defensive_signal:
                return defensive_signal
            
            current_price = df['close'].iloc[-1]
            
            # Récupérer indicateurs
            rsi = self._get_current_indicator(indicators, 'rsi_14')
            macd_line = self._get_current_indicator(indicators, 'macd_line')
            cci = self._get_current_indicator(indicators, 'cci_20')
            momentum_10 = self._get_current_indicator(indicators, 'momentum_10')
            
            if any(x is None for x in [rsi, macd_line, cci, momentum_10]):
                logger.debug(f"❌ {symbol}: Indicateurs divergence incomplets")
                return None
            
            # Récupérer indicateurs de contexte
            volume_ratio = self._get_current_indicator(indicators, 'volume_ratio')
            volume_spike = indicators.get('volume_spike', False)
            adx = self._get_current_indicator(indicators, 'adx_14')
            williams_r = self._get_current_indicator(indicators, 'williams_r')
            bb_position = self._get_current_indicator(indicators, 'bb_position')
            
            # Mettre à jour l'historique
            self._update_history(current_price, rsi or 0.0, macd_line or 0.0, cci or 0.0, momentum_10 or 0.0)
            
            # Détecter toutes les divergences
            all_divergences = self._detect_all_divergences()
            
            # Analyser la structure de marché
            structure_analysis = self._analyze_market_structure_for_divergence(df)
            
            # Analyser le contexte
            context_analysis = self._analyze_divergence_context(
                symbol, volume_ratio or 0.0, adx or 0.0, williams_r or 0.0, bb_position or 0.0
            )
            
            # NOUVEAU: Calculer la position du prix dans son range
            price_position = self.calculate_price_position_in_range(df)
            
            signal = None
            
            # SIGNAL D'ACHAT - Divergences bullish avec contexte favorable
            bullish_divergences = [d for d in all_divergences if d.type in ['bullish', 'hidden_bullish']]
            if self._is_valid_bullish_divergence_signal(bullish_divergences, structure_analysis, context_analysis):
                # Vérifier la position du prix avant de générer le signal
                if not self.should_filter_signal_by_price_position(OrderSide.BUY, price_position, df):
                    confidence = self._calculate_divergence_confidence(
                        bullish_divergences, structure_analysis, context_analysis, volume_ratio or 0.0, True
                    )
                    
                    # Préparer métadonnées détaillées
                    div_summary = self._summarize_divergences(bullish_divergences)
                    
                    signal = self.create_signal(
                        side=OrderSide.BUY,
                        price=current_price,
                        confidence=confidence,
                        metadata={
                            'divergence_count': len(bullish_divergences),
                            'divergence_types': [d.type for d in bullish_divergences],
                            'divergence_indicators': [d.indicator for d in bullish_divergences],
                            'strongest_divergence': div_summary['strongest'],
                            'avg_divergence_strength': div_summary['avg_strength'],
                            'price_swing_strength': structure_analysis['swing_strength'],
                            'volume_ratio': volume_ratio,
                            'volume_spike': volume_spike,
                            'context_score': context_analysis['score'],
                            'confluence_score': context_analysis.get('confluence_score', 0),
                            'structure_favorability': structure_analysis['favorability'],
                            'price_position': price_position,
                            'reason': f'Divergence Pro BUY ({len(bullish_divergences)} div: {div_summary["indicators"]})'
                        }
                    )
                    # Enregistrer prix d'entrée pour protection défensive
                    self.last_entry_price = current_price
                    
                    # Convertir StrategySignal en dict pour compatibilité
                    if signal is not None:
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
                    else:
                        return None
                    
                else:
                    logger.info(f"📊 Divergence Pro {symbol}: Signal BUY techniquement valide mais filtré "
                                f"(position prix: {price_position:.2f})")
            
            # SIGNAL DE VENTE - Divergences bearish avec contexte favorable
            else:
                bearish_divergences = [d for d in all_divergences if d.type in ['bearish', 'hidden_bearish']]
                if self._is_valid_bearish_divergence_signal(bearish_divergences, structure_analysis, context_analysis):
                    # Vérifier la position du prix avant de générer le signal
                    if not self.should_filter_signal_by_price_position(OrderSide.SELL, price_position, df):
                        confidence = self._calculate_divergence_confidence(
                            bearish_divergences, structure_analysis, context_analysis, volume_ratio or 0.0, False
                        )
                        
                        # Préparer métadonnées détaillées
                        div_summary = self._summarize_divergences(bearish_divergences)
                        
                        signal = self.create_signal(
                            side=OrderSide.SELL,
                            price=current_price,
                            confidence=confidence,
                            metadata={
                                'divergence_count': len(bearish_divergences),
                                'divergence_types': [d.type for d in bearish_divergences],
                                'divergence_indicators': [d.indicator for d in bearish_divergences],
                                'strongest_divergence': div_summary['strongest'],
                                'avg_divergence_strength': div_summary['avg_strength'],
                                'price_swing_strength': structure_analysis['swing_strength'],
                                'volume_ratio': volume_ratio,
                                'volume_spike': volume_spike,
                                'context_score': context_analysis['score'],
                                'confluence_score': context_analysis.get('confluence_score', 0),
                                'structure_favorability': structure_analysis['favorability'],
                                'price_position': price_position,
                                'reason': f'Divergence Pro SELL ({len(bearish_divergences)} div: {div_summary["indicators"]})'
                            }
                        )
                    else:
                        logger.info(f"📊 Divergence Pro {symbol}: Signal SELL techniquement valide mais filtré "
                                  f"(position prix: {price_position:.2f})")
            
            if signal:
                metadata = signal.metadata if signal.metadata else {}
                div_count = metadata.get('divergence_count', 0)
                indicators_str = metadata.get('reason', '').split(': ')[-1].rstrip(')')
                logger.info(f"🎯 Divergence Pro {symbol}: {signal.side} @ {current_price:.4f} "
                          f"({div_count} div: {indicators_str}, Context: {context_analysis['score']:.1f}, "
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
            logger.error(f"❌ Erreur Reversal Divergence Pro Strategy {symbol}: {e}")
            return None
    
    def _update_history(self, price: float, rsi: float, macd: float, cci: float, momentum: float):
        """Met à jour l'historique pour divergences"""
        self.price_history.append(price)
        self.rsi_history.append(rsi)
        self.macd_history.append(macd)
        self.cci_history.append(cci)
        self.momentum_history.append(momentum)
        
        # Limiter la taille
        for history in [self.price_history, self.rsi_history, self.macd_history, 
                       self.cci_history, self.momentum_history]:
            if len(history) > self.max_history:
                history.pop(0)
    
    def _detect_all_divergences(self) -> List[DivergenceSignal]:
        """Détecte toutes les divergences sur tous les indicateurs"""
        all_divergences: List[DivergenceSignal] = []
        
        if len(self.price_history) < self.lookback_periods:
            return all_divergences
        
        # Détecter divergences pour chaque indicateur
        indicators_data = {
            'rsi': self.rsi_history,
            'macd': self.macd_history,
            'cci': self.cci_history,
            'momentum': self.momentum_history
        }
        
        for indicator_name, indicator_history in indicators_data.items():
            if len(indicator_history) >= self.lookback_periods:
                divergences = self._detect_divergences_for_indicator(
                    self.price_history, indicator_history, indicator_name
                )
                all_divergences.extend(divergences)
        
        return all_divergences
    
    def _detect_divergences_for_indicator(self, prices: List[float], indicator_values: List[float], 
                                         indicator_name: str) -> List[DivergenceSignal]:
        """Détecte les divergences pour un indicateur spécifique"""
        divergences = []
        
        try:
            # Utiliser les dernières données
            recent_prices = prices[-self.lookback_periods:]
            recent_indicator = indicator_values[-self.lookback_periods:]
            
            # Trouver les swings (fractales)
            price_highs = self._find_swing_highs(recent_prices, self.min_swing_strength)
            price_lows = self._find_swing_lows(recent_prices, self.min_swing_strength)
            indicator_highs = self._find_swing_highs(recent_indicator, self.min_swing_strength)
            indicator_lows = self._find_swing_lows(recent_indicator, self.min_swing_strength)
            
            # Divergence bullish classique : Prix fait lower low, indicateur fait higher low
            if len(price_lows) >= 2 and len(indicator_lows) >= 2:
                for i in range(len(price_lows) - 1):
                    p1_idx, p1_val = price_lows[i]
                    p2_idx, p2_val = price_lows[i + 1]
                    
                    # Trouver indicateur correspondant
                    ind_low = self._find_closest_swing(indicator_lows, p2_idx, tolerance=3)
                    if ind_low:
                        ind1_idx, ind1_val = self._find_closest_swing(indicator_lows, p1_idx, tolerance=3) or (0, recent_indicator[0])
                        ind2_idx, ind2_val = ind_low
                        
                        # Vérifier divergence bullish
                        if p2_val < p1_val and ind2_val > ind1_val:
                            price_change = abs(p2_val - p1_val) / p1_val * 100
                            ind_change = abs(ind2_val - ind1_val)
                            
                            if price_change >= self.min_price_change and ind_change >= self.min_indicator_change:
                                strength = min(1.0, (price_change / 5.0 + ind_change / 10.0) / 2)
                                confidence = min(0.9, strength + 0.3)
                                
                                divergences.append(DivergenceSignal(
                                    type='bullish',
                                    indicator=indicator_name,
                                    strength=strength,
                                    price_points=(p1_val, p2_val),
                                    indicator_points=(ind1_val, ind2_val),
                                    time_span=p2_idx - p1_idx,
                                    confidence=confidence
                                ))
            
            # Divergence bearish classique : Prix fait higher high, indicateur fait lower high
            if len(price_highs) >= 2 and len(indicator_highs) >= 2:
                for i in range(len(price_highs) - 1):
                    p1_idx, p1_val = price_highs[i]
                    p2_idx, p2_val = price_highs[i + 1]
                    
                    # Trouver indicateur correspondant
                    ind_high = self._find_closest_swing(indicator_highs, p2_idx, tolerance=3)
                    if ind_high:
                        ind1_idx, ind1_val = self._find_closest_swing(indicator_highs, p1_idx, tolerance=3) or (0, recent_indicator[0])
                        ind2_idx, ind2_val = ind_high
                        
                        # Vérifier divergence bearish
                        if p2_val > p1_val and ind2_val < ind1_val:
                            price_change = abs(p2_val - p1_val) / p1_val * 100
                            ind_change = abs(ind2_val - ind1_val)
                            
                            if price_change >= self.min_price_change and ind_change >= self.min_indicator_change:
                                strength = min(1.0, (price_change / 5.0 + ind_change / 10.0) / 2)
                                confidence = min(0.9, strength + 0.3)
                                
                                divergences.append(DivergenceSignal(
                                    type='bearish',
                                    indicator=indicator_name,
                                    strength=strength,
                                    price_points=(p1_val, p2_val),
                                    indicator_points=(ind1_val, ind2_val),
                                    time_span=p2_idx - p1_idx,
                                    confidence=confidence
                                ))
            
        except Exception as e:
            logger.debug(f"Erreur détection divergence {indicator_name}: {e}")
        
        return divergences
    
    def _find_swing_highs(self, data: List[float], strength: int = 5) -> List[Tuple[int, float]]:
        """Trouve les swing highs (fractales hautes)"""
        swings = []
        for i in range(strength, len(data) - strength):
            is_swing = True
            for j in range(i - strength, i + strength + 1):
                if j != i and data[j] >= data[i]:
                    is_swing = False
                    break
            if is_swing:
                swings.append((i, data[i]))
        return swings
    
    def _find_swing_lows(self, data: List[float], strength: int = 5) -> List[Tuple[int, float]]:
        """Trouve les swing lows (fractales basses)"""
        swings = []
        for i in range(strength, len(data) - strength):
            is_swing = True
            for j in range(i - strength, i + strength + 1):
                if j != i and data[j] <= data[i]:
                    is_swing = False
                    break
            if is_swing:
                swings.append((i, data[i]))
        return swings
    
    def _find_closest_swing(self, swings: List[Tuple[int, float]], target_idx: int, tolerance: int = 3) -> Optional[Tuple[int, float]]:
        """Trouve le swing le plus proche d'un index donné"""
        closest = None
        min_distance = float('inf')
        
        for idx, val in swings:
            distance = abs(idx - target_idx)
            if distance <= tolerance and distance < min_distance:
                min_distance = distance
                closest = (idx, val)
        
        return closest
    
    def _analyze_market_structure_for_divergence(self, df: pd.DataFrame) -> Dict:
        """Analyse la structure de marché pour valider les divergences"""
        structure = {
            'swing_strength': 0.0,
            'favorability': 0.0,
            'trend_context': 'neutral',
            'reversal_probability': 0.0
        }
        
        try:
            if len(df) < 15:
                return structure
            
            recent_data = df.tail(15)
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            closes = recent_data['close'].values
            
            # Calculer la force des swings
            price_range = np.max(highs) - np.min(lows)
            avg_range = np.mean(highs - lows)
            structure['swing_strength'] = min(1.0, price_range / avg_range / 3)
            
            # Analyser le contexte de tendance
            price_start = closes[0]
            price_end = closes[-1]
            trend_strength = (price_end - price_start) / price_start * 100
            
            if trend_strength > 2:
                structure['trend_context'] = 'uptrend'
                structure['favorability'] = 0.7  # Favorable pour divergence bearish
            elif trend_strength < -2:
                structure['trend_context'] = 'downtrend'
                structure['favorability'] = 0.7  # Favorable pour divergence bullish
            else:
                structure['trend_context'] = 'sideways'
                structure['favorability'] = 0.4
            
            # Probabilité de retournement
            volatility = np.std(closes) / np.mean(closes)
            structure['reversal_probability'] = min(0.9, volatility * 50 + structure['swing_strength'])
            
        except Exception as e:
            logger.debug(f"Erreur analyse structure divergence: {e}")
        
        return structure
    
    def _analyze_divergence_context(self, symbol: str, volume_ratio: float, adx: float, 
                                   williams_r: float, bb_position: float) -> Dict[str, Any]:
        """Analyse le contexte pour valider les divergences"""
        context: Dict[str, Any] = {
            'score': 0.0,
            'confidence_boost': 0.0,
            'confluence_score': 0.0,
            'details': []
        }
        
        try:
            # 1. Volume confirmation (critique pour retournements) - SEUILS STANDARDISÉS
            if volume_ratio > 1.5:  # STANDARDISÉ: Très bon
                context['score'] = float(context['score']) + 25
                context['confidence_boost'] = float(context['confidence_boost']) + 0.1
                details_list = context.get('details', [])
                if isinstance(details_list, list):
                    details_list.append(f"Volume très bon ({volume_ratio:.1f}x)")
            elif volume_ratio > 1.2:  # STANDARDISÉ: Bon
                context['score'] = float(context['score']) + 20
                context['confidence_boost'] = float(context['confidence_boost']) + 0.08
                details_list = context.get('details', [])
                if isinstance(details_list, list):
                    details_list.append(f"Volume bon ({volume_ratio:.1f}x)")
            elif volume_ratio > 1.0:  # STANDARDISÉ: Acceptable
                context['score'] = float(context['score']) + 15
                context['confidence_boost'] = float(context['confidence_boost']) + 0.05
                details_list = context.get('details', [])
                if isinstance(details_list, list):
                    details_list.append(f"Volume acceptable ({volume_ratio:.1f}x)")
            else:
                context['score'] = float(context['score']) - 10
                details_list = context.get('details', [])
                if isinstance(details_list, list):
                    details_list.append(f"Volume faible ({volume_ratio:.1f}x)")
            
            # 2. Force de tendance (ADX) - divergences meilleures en fin de tendance
            from shared.src.config import ADX_STRONG_TREND_THRESHOLD, ADX_TREND_THRESHOLD
            if adx >= ADX_STRONG_TREND_THRESHOLD:  # Tendance très forte = potentiel retournement
                context['score'] = float(context['score']) + 20
                context['confidence_boost'] = float(context['confidence_boost']) + 0.08
                details_list = context.get('details', [])
                if isinstance(details_list, list):
                    details_list.append(f"ADX fort - retournement potentiel ({adx:.1f})")
            elif adx >= ADX_TREND_THRESHOLD:
                context['score'] = float(context['score']) + 15
                context['confidence_boost'] = float(context['confidence_boost']) + 0.05
                details_list = context.get('details', [])
                if isinstance(details_list, list):
                    details_list.append(f"ADX modéré ({adx:.1f})")
            else:
                context['score'] = float(context['score']) + 5
                details_list = context.get('details', [])
                if isinstance(details_list, list):
                    details_list.append(f"ADX faible ({adx:.1f})")
            
            # 3. Williams %R pour zones extrêmes
            if williams_r <= -85 or williams_r >= -15:  # Zones extrêmes = bon pour retournements
                context['score'] = float(context['score']) + 20
                context['confidence_boost'] = float(context['confidence_boost']) + 0.08
                details_list = context.get('details', [])
                if isinstance(details_list, list):
                    details_list.append(f"Williams extrême ({williams_r:.1f})")
            elif williams_r <= -75 or williams_r >= -25:
                context['score'] = float(context['score']) + 15
                context['confidence_boost'] = float(context['confidence_boost']) + 0.05
                details_list = context.get('details', [])
                if isinstance(details_list, list):
                    details_list.append(f"Williams favorable ({williams_r:.1f})")
            else:
                context['score'] = float(context['score']) + 5
                details_list = context.get('details', [])
                if isinstance(details_list, list):
                    details_list.append(f"Williams neutre ({williams_r:.1f})")
            
            # 4. Position Bollinger pour extrêmes - STANDARDISÉ
            if bb_position <= 0.15 or bb_position >= 0.85:  # STANDARDISÉ: Excellent
                context['score'] = float(context['score']) + 20
                context['confidence_boost'] = float(context['confidence_boost']) + 0.08
                details_list = context.get('details', [])
                if isinstance(details_list, list):
                    details_list.append(f"BB position extrême ({bb_position:.2f})")
            elif bb_position <= 0.25 or bb_position >= 0.75:  # STANDARDISÉ: Très bon
                context['score'] = float(context['score']) + 15
                context['confidence_boost'] = float(context['confidence_boost']) + 0.05
                details_list = context.get('details', [])
                if isinstance(details_list, list):
                    details_list.append(f"BB position favorable ({bb_position:.2f})")
            else:
                context['score'] = float(context['score']) + 5
                details_list = context.get('details', [])
                if isinstance(details_list, list):
                    details_list.append(f"BB position neutre ({bb_position:.2f})")
            
            # 5. Confluence multi-timeframes
            if self.redis_client:
                confluence_data = self._get_confluence_analysis(symbol)
                if confluence_data:
                    confluence_score = confluence_data.get('confluence_score', 0)
                    context['confluence_score'] = confluence_score
                    
                    if confluence_score >= 55:
                        context['score'] = float(context['score']) + 15
                        context['confidence_boost'] = float(context['confidence_boost']) + 0.05
                        details_list = context.get('details', [])
                        if isinstance(details_list, list):
                            details_list.append(f"Confluence favorable ({confluence_score:.1f}%)")
                    elif confluence_score >= 40:
                        context['score'] = float(context['score']) + 8
                        details_list = context.get('details', [])
                        if isinstance(details_list, list):
                            details_list.append(f"Confluence modérée ({confluence_score:.1f}%)")
            
            # Normaliser
            context['score'] = max(0.0, min(100.0, float(context['score'])))
            context['confidence_boost'] = max(0.0, min(0.25, float(context['confidence_boost'])))
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse contexte divergence: {e}")
            context['score'] = 25
        
        return context
    
    def _is_valid_bullish_divergence_signal(self, divergences: List[DivergenceSignal], 
                                           structure: Dict, context: Dict) -> bool:
        """Valide si les divergences bullish sont suffisantes"""
        if not divergences:
            return False
        
        # Score de contexte minimum
        if context['score'] < 25:  # AJUSTÉ de 40 à 25 pour plus de signaux
            return False
        
        # Confluence minimum si disponible
        confluence_score = context.get('confluence_score', 0)
        if confluence_score > 0 and confluence_score < (self.confluence_threshold - 10):  # ASSOUPLI de 5 à 10  # Encore plus assoupli
            return False
        
        # Force des divergences
        avg_strength = np.mean([d.strength for d in divergences])
        if avg_strength < 0.25:  # AJUSTÉ de 0.3 à 0.25 pour plus de sensibilité
            return False
        
        # Multiple divergences = bonus
        if len(divergences) >= 2:
            return context['score'] > 30  # AJUSTÉ de 50 à 30
        
        # Divergence unique forte
        if len(divergences) == 1 and divergences[0].strength > 0.5:  # AJUSTÉ de 0.6 à 0.5
            return context['score'] > 35  # AJUSTÉ de 60 à 35
        
        return context['score'] > 40  # AJUSTÉ de 70 à 40
    
    def _is_valid_bearish_divergence_signal(self, divergences: List[DivergenceSignal], 
                                           structure: Dict, context: Dict) -> bool:
        """Valide si les divergences bearish sont suffisantes"""
        if not divergences:
            return False
        
        # Score de contexte minimum assoupli pour SELL
        if context['score'] < 35:
            return False
        
        # Confluence minimum si disponible - assoupli
        confluence_score = context.get('confluence_score', 0)
        if confluence_score > 0 and confluence_score < (self.confluence_threshold - 10):  # ASSOUPLI de 5 à 10
            return False
        
        # Force des divergences
        avg_strength = np.mean([d.strength for d in divergences])
        if avg_strength < 0.3:
            return False
        
        # Multiple divergences = bonus
        if len(divergences) >= 2:
            return context['score'] > 50
        
        # Divergence unique forte
        if len(divergences) == 1 and divergences[0].strength > 0.6:
            return context['score'] > 60
        
        return context['score'] > 70
    
    def _calculate_divergence_confidence(self, divergences: List[DivergenceSignal], structure: Dict, 
                                        context: Dict, volume_ratio: float, is_bullish: bool) -> float:
        """Calcule la confiance pour un signal de divergence"""
        base_confidence = 0.55  # Base élevée pour divergences
        
        # Force moyenne des divergences
        if divergences:
            avg_strength = float(np.mean([d.strength for d in divergences]))
            base_confidence += avg_strength * 0.2
            
            # Bonus pour divergences multiples
            if len(divergences) >= 2:
                base_confidence += self.multiple_div_bonus
            
            # Bonus pour divergences sur indicateurs variés
            unique_indicators = len(set(d.indicator for d in divergences))
            if unique_indicators >= 2:
                base_confidence += 0.08
        
        # Contexte
        base_confidence += context['confidence_boost']
        
        # Structure favorable
        base_confidence += structure['favorability'] * 0.1
        
        # Volume confirmation - SEUILS STANDARDISÉS
        if volume_ratio > 1.5:  # STANDARDISÉ: Très bon
            base_confidence += 0.08
        elif volume_ratio > 1.2:  # STANDARDISÉ: Bon
            base_confidence += 0.05
        
        # Probabilité de retournement
        base_confidence += structure['reversal_probability'] * 0.1
        
        return min(0.95, base_confidence)
    
    def _summarize_divergences(self, divergences: List[DivergenceSignal]) -> Dict:
        """Résume les divergences détectées"""
        if not divergences:
            return {'strongest': 'none', 'avg_strength': 0.0, 'indicators': 'none'}
        
        # Trouver la plus forte
        strongest = max(divergences, key=lambda d: d.strength)
        
        # Force moyenne
        avg_strength = float(np.mean([d.strength for d in divergences]))
        
        # Indicateurs impliqués
        indicators = list(set(d.indicator for d in divergences))
        indicators_str = '+'.join(indicators)
        
        return {
            'strongest': f"{strongest.indicator}({strongest.strength:.2f})",
            'avg_strength': avg_strength,
            'indicators': indicators_str
        }
    
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