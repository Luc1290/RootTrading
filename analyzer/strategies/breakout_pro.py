"""
Stratégie Breakout Pro
Breakout avec détection de niveaux S/R dynamiques, false breakout filter et confluence.
Intègre volume, momentum, structure de marché et analyse multi-timeframes.
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
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

class BreakoutProStrategy(BaseStrategy):
    """
    Stratégie Breakout Pro - Cassures avec analyse avancée et filtres anti-faux breakouts
    BUY: Breakout résistance + volume élevé + momentum + confluence + no false breakout
    SELL: Breakdown support + volume élevé + momentum + confluence + no false breakdown
    
    Intègre :
    - Détection dynamique de S/R (pivot points, fractales)
    - Filtres anti-faux breakouts (volume, momentum, retest)
    - Multiple timeframes pour confluence
    - ATR pour volatilité contexte
    - Structure de marché (HH/HL, LH/LL)
    - Liquidity zones et order blocks
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        super().__init__(symbol, params)
        
        # Paramètres Breakout avancés
        symbol_params = self.params.get(symbol, {}) if self.params else {}
        self.lookback_periods = symbol_params.get('lookback_periods', 50)  # Plus long pour S/R
        self.min_breakout_percent = symbol_params.get('breakout_min_percent', 0.8)  # ASSOUPLI de 1.5 à 0.8
        self.min_volume_multiplier = symbol_params.get('min_volume_multiplier', 1.0)  # ASSOUPLI de 1.3 à 1.0
        self.false_breakout_retest_periods = symbol_params.get('retest_periods', 5)
        self.sr_strength_threshold = symbol_params.get('sr_strength', 2)  # ASSOUPLI de 3 à 2
        self.confluence_threshold = symbol_params.get('confluence_threshold', 35.0)  # ASSOUPLI de 45 à 35
        
        # Historique pour S/R dynamiques
        self.sr_levels = {'supports': [], 'resistances': []}
        self.breakout_history = []
        self.max_sr_levels = 5
        
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
                logger.warning(f"⚠️ Redis non disponible pour Breakout Pro: {e}")
                self.redis_client = None
        
        logger.info(f"🎯 Breakout Pro initialisé pour {symbol} (Lookback: {self.lookback_periods}, Volume≥{self.min_volume_multiplier}x)")

    @property
    def name(self) -> str:
        return "Breakout_Pro_Strategy"
    
    def get_min_data_points(self) -> int:
        return max(60, self.lookback_periods + 10)  # Plus de données pour S/R robustes
    
    def analyze(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Optional[Dict]:
        """
        Analyse Breakout Pro - Cassures avec filtres avancés
        """
        try:
            if len(df) < self.get_min_data_points():
                return None
            
            # PRIORITÉ 1: Vérifier conditions de protection défensive
            defensive_signal = self.check_defensive_conditions(df)
            if defensive_signal:
                return defensive_signal
            
            current_price = df['close'].iloc[-1]
            
            # Récupérer indicateurs de contexte
            volume_ratio = self._get_current_indicator(indicators, 'volume_ratio')
            volume_spike = indicators.get('volume_spike', False)
            atr = self._get_current_indicator(indicators, 'atr_14')
            momentum_10 = self._get_current_indicator(indicators, 'momentum_10')
            adx = self._get_current_indicator(indicators, 'adx_14')
            rsi = self._get_current_indicator(indicators, 'rsi_14')
            bb_width = self._get_current_indicator(indicators, 'bb_width')
            
            # Analyser les niveaux de support/résistance dynamiques
            sr_analysis = self._analyze_dynamic_sr_levels(df)
            
            # Analyser la structure de marché
            structure_analysis = self._analyze_market_structure(df)
            
            # Analyser le contexte breakout
            context_analysis = self._analyze_breakout_context(
                symbol, volume_ratio, atr, momentum_10, adx, rsi, bb_width
            )
            
            # NOUVEAU: Calculer la position du prix dans son range
            price_position = self.calculate_price_position_in_range(df)
            
            signal = None
            
            # SIGNAL D'ACHAT - Breakout résistance avec validations
            breakout_analysis = self._analyze_resistance_breakout(
                current_price, sr_analysis, context_analysis, structure_analysis
            )
            
            if breakout_analysis['is_valid']:
                # Vérifier la position du prix avant de générer le signal
                if not self.should_filter_signal_by_price_position(OrderSide.BUY, price_position, df):
                    confidence = self._calculate_breakout_confidence(
                        breakout_analysis, context_analysis, volume_ratio, momentum_10, True
                    )
                    
                    signal = self.create_signal(
                        side=OrderSide.BUY,
                        price=current_price,
                        confidence=confidence,
                        metadata={
                            'breakout_type': 'resistance',
                            'resistance_level': breakout_analysis['level'],
                            'breakout_strength': breakout_analysis['strength'],
                            'breakout_percent': breakout_analysis['percent'],
                            'sr_touches': breakout_analysis['touches'],
                            'volume_ratio': volume_ratio,
                            'volume_spike': volume_spike,
                            'atr_context': atr,
                            'momentum_10': momentum_10,
                            'structure_type': structure_analysis['type'],
                            'context_score': context_analysis['score'],
                            'confluence_score': context_analysis.get('confluence_score', 0),
                            'false_breakout_risk': breakout_analysis['false_risk'],
                            'price_position': price_position,
                            'reason': f'Breakout Pro résistance ({current_price:.4f} > {breakout_analysis["level"]:.4f})'
                        }
                    )
                    # Enregistrer prix d'entrée pour protection défensive
                    self.last_entry_price = current_price
                else:
                    logger.info(f"📊 Breakout Pro {symbol}: Signal BUY techniquement valide mais filtré "
                              f"(position prix: {price_position:.2f})")
            
            # SIGNAL DE VENTE - Breakdown support avec validations
            breakdown_analysis = self._analyze_support_breakdown(
                current_price, sr_analysis, context_analysis, structure_analysis
            )
            
            if breakdown_analysis['is_valid']:
                # Vérifier la position du prix avant de générer le signal
                if not self.should_filter_signal_by_price_position(OrderSide.SELL, price_position, df):
                    confidence = self._calculate_breakout_confidence(
                        breakdown_analysis, context_analysis, volume_ratio, momentum_10, False
                    )
                    
                    signal = self.create_signal(
                        side=OrderSide.SELL,
                        price=current_price,
                        confidence=confidence,
                        metadata={
                            'breakout_type': 'support',
                            'support_level': breakdown_analysis['level'],
                            'breakdown_strength': breakdown_analysis['strength'],
                            'breakdown_percent': breakdown_analysis['percent'],
                            'sr_touches': breakdown_analysis['touches'],
                            'volume_ratio': volume_ratio,
                            'volume_spike': volume_spike,
                            'atr_context': atr,
                            'momentum_10': momentum_10,
                            'structure_type': structure_analysis['type'],
                            'context_score': context_analysis['score'],
                            'confluence_score': context_analysis.get('confluence_score', 0),
                            'false_breakout_risk': breakdown_analysis['false_risk'],
                            'price_position': price_position,
                            'reason': f'Breakdown Pro support ({current_price:.4f} < {breakdown_analysis["level"]:.4f})'
                        }
                    )
                else:
                    logger.info(f"📊 Breakout Pro {symbol}: Signal SELL techniquement valide mais filtré "
                              f"(position prix: {price_position:.2f})")
            
            # AJOUT CRITIQUE: NO SIGNAL si aucune condition n'est remplie
            # Avant c'était un else forcé - maintenant on peut avoir NO SIGNAL
            
            if signal:
                logger.info(f"🎯 Breakout Pro {symbol}: {signal.side} @ {current_price:.4f} "
                          f"(Niveau: {signal.metadata.get('resistance_level', signal.metadata.get('support_level', 0)):.4f}, "
                          f"Volume: {volume_ratio:.1f}x, Context: {context_analysis['score']:.1f}, "
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
            logger.error(f"❌ Erreur Breakout Pro Strategy {symbol}: {e}")
            return None
    
    def _analyze_dynamic_sr_levels(self, df: pd.DataFrame) -> Dict:
        """Analyse les niveaux de support/résistance dynamiques"""
        sr_analysis = {
            'supports': [],
            'resistances': [],
            'strongest_support': None,
            'strongest_resistance': None
        }
        
        try:
            if len(df) < self.lookback_periods:
                return sr_analysis
            
            # Utiliser les dernières données
            recent_data = df.tail(self.lookback_periods)
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            closes = recent_data['close'].values
            
            # Détecter les pivot points (fractales)
            resistance_levels = self._find_pivot_highs(highs, window=5)
            support_levels = self._find_pivot_lows(lows, window=5)
            
            # Calculer la force des niveaux (nombre de touches)
            for level in resistance_levels:
                touches = self._count_level_touches(closes, level, tolerance=0.002)
                if touches >= self.sr_strength_threshold:
                    sr_analysis['resistances'].append({
                        'level': level,
                        'touches': touches,
                        'strength': touches / len(recent_data) * 100
                    })
            
            for level in support_levels:
                touches = self._count_level_touches(closes, level, tolerance=0.002)
                if touches >= self.sr_strength_threshold:
                    sr_analysis['supports'].append({
                        'level': level,
                        'touches': touches,
                        'strength': touches / len(recent_data) * 100
                    })
            
            # Trier par force et garder les plus forts
            sr_analysis['resistances'] = sorted(sr_analysis['resistances'], 
                                              key=lambda x: x['strength'], reverse=True)[:self.max_sr_levels]
            sr_analysis['supports'] = sorted(sr_analysis['supports'], 
                                           key=lambda x: x['strength'], reverse=True)[:self.max_sr_levels]
            
            # Identifier les niveaux les plus forts
            if sr_analysis['resistances']:
                sr_analysis['strongest_resistance'] = sr_analysis['resistances'][0]
            if sr_analysis['supports']:
                sr_analysis['strongest_support'] = sr_analysis['supports'][0]
            
        except Exception as e:
            logger.debug(f"Erreur analyse S/R dynamiques: {e}")
        
        return sr_analysis
    
    def _find_pivot_highs(self, highs: np.ndarray, window: int = 5) -> List[float]:
        """Trouve les pivot highs (résistances potentielles)"""
        pivots = []
        for i in range(window, len(highs) - window):
            is_pivot = True
            for j in range(i - window, i + window + 1):
                if j != i and highs[j] >= highs[i]:
                    is_pivot = False
                    break
            if is_pivot:
                pivots.append(highs[i])
        return pivots
    
    def _find_pivot_lows(self, lows: np.ndarray, window: int = 5) -> List[float]:
        """Trouve les pivot lows (supports potentiels)"""
        pivots = []
        for i in range(window, len(lows) - window):
            is_pivot = True
            for j in range(i - window, i + window + 1):
                if j != i and lows[j] <= lows[i]:
                    is_pivot = False
                    break
            if is_pivot:
                pivots.append(lows[i])
        return pivots
    
    def _count_level_touches(self, prices: np.ndarray, level: float, tolerance: float = 0.002) -> int:
        """Compte le nombre de fois où le prix touche un niveau"""
        count = 0
        level_min = level * (1 - tolerance)
        level_max = level * (1 + tolerance)
        
        for price in prices:
            if level_min <= price <= level_max:
                count += 1
        
        return count
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """Analyse la structure de marché (HH/HL vs LH/LL)"""
        structure = {
            'type': 'sideways',
            'strength': 0.0,
            'bias': 'neutral'
        }
        
        try:
            if len(df) < 20:
                return structure
            
            recent_data = df.tail(20)
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            # Analyser les 4 derniers swings
            recent_highs = sorted(highs[-8:], reverse=True)[:2]
            recent_lows = sorted(lows[-8:])[:2]
            
            # Higher Highs and Higher Lows = Uptrend
            if len(recent_highs) >= 2 and recent_highs[0] > recent_highs[1]:
                if len(recent_lows) >= 2 and recent_lows[-1] > recent_lows[-2]:
                    structure['type'] = 'uptrend'
                    structure['bias'] = 'bullish'
                    structure['strength'] = 0.8
            
            # Lower Highs and Lower Lows = Downtrend
            elif len(recent_highs) >= 2 and recent_highs[0] < recent_highs[1]:
                if len(recent_lows) >= 2 and recent_lows[-1] < recent_lows[-2]:
                    structure['type'] = 'downtrend'
                    structure['bias'] = 'bearish'
                    structure['strength'] = 0.8
            
            # Mixed = Sideways
            else:
                structure['type'] = 'sideways'
                structure['bias'] = 'neutral'
                structure['strength'] = 0.4
                
        except Exception as e:
            logger.debug(f"Erreur analyse structure marché: {e}")
        
        return structure
    
    def _analyze_breakout_context(self, symbol: str, volume_ratio: float, atr: float, 
                                 momentum_10: float, adx: float, rsi: float, bb_width: float) -> Dict:
        """Analyse le contexte pour valider les breakouts"""
        context = {
            'score': 0.0,
            'confidence_boost': 0.0,
            'confluence_score': 0.0,
            'details': []
        }
        
        try:
            # 1. Volume explosion (critère principal pour breakouts) - SEUILS STANDARDISÉS
            if volume_ratio and volume_ratio >= self.min_volume_multiplier:
                if volume_ratio >= 2.0:  # STANDARDISÉ: Excellent (début pump confirmé)
                    context['score'] += 35
                    context['confidence_boost'] += 0.15
                    context['details'].append(f"Volume excellent ({volume_ratio:.1f}x)")
                elif volume_ratio >= 1.5:  # STANDARDISÉ: Très bon
                    context['score'] += 30
                    context['confidence_boost'] += 0.12
                    context['details'].append(f"Volume très bon ({volume_ratio:.1f}x)")
                else:
                    context['score'] += 20
                    context['confidence_boost'] += 0.08
                    context['details'].append(f"Volume bon ({volume_ratio:.1f}x)")
            elif volume_ratio and volume_ratio >= 1.2:  # STANDARDISÉ: Bon
                context['score'] += 10
                context['details'].append(f"Volume bon ({volume_ratio:.1f}x)")
            else:
                context['score'] -= 15  # Pénalité forte pour volume faible
                context['details'].append(f"Volume insuffisant ({volume_ratio or 0:.1f}x)")
            
            # 2. Volatilité (ATR) - breakouts nécessitent volatilité
            from shared.src.config import ATR_THRESHOLD_VERY_HIGH, ATR_THRESHOLD_MODERATE
            if atr:
                if atr >= ATR_THRESHOLD_VERY_HIGH:  # 0.006 - Volatilité très élevée
                    context['score'] += 20
                    context['confidence_boost'] += 0.08
                    context['details'].append(f"Volatilité haute ({atr:.4f})")
                elif atr >= ATR_THRESHOLD_MODERATE:
                    context['score'] += 15
                    context['confidence_boost'] += 0.05
                    context['details'].append(f"Volatilité normale ({atr:.4f})")
                else:
                    context['score'] -= 5
                    context['details'].append(f"Volatilité faible ({atr:.4f})")
            
            # 3. Momentum directionnel
            if momentum_10:
                momentum_strength = abs(momentum_10)
                if momentum_strength > 1.5:
                    context['score'] += 25
                    context['confidence_boost'] += 0.1
                    context['details'].append(f"Momentum très fort ({momentum_10:.2f})")
                elif momentum_strength > 0.8:
                    context['score'] += 20
                    context['confidence_boost'] += 0.08
                    context['details'].append(f"Momentum fort ({momentum_10:.2f})")
                elif momentum_strength > 0.3:
                    context['score'] += 10
                    context['details'].append(f"Momentum modéré ({momentum_10:.2f})")
                else:
                    context['score'] -= 5
                    context['details'].append(f"Momentum faible ({momentum_10:.2f})")
            
            # 4. Force de tendance (ADX)
            if adx:
                from shared.src.config import ADX_TREND_THRESHOLD, ADX_WEAK_TREND_THRESHOLD
                if adx >= ADX_TREND_THRESHOLD:
                    context['score'] += 20
                    context['confidence_boost'] += 0.08
                    context['details'].append(f"Tendance forte ({adx:.1f})")
                elif adx >= ADX_WEAK_TREND_THRESHOLD:
                    context['score'] += 15
                    context['confidence_boost'] += 0.05
                    context['details'].append(f"Tendance modérée ({adx:.1f})")
                else:
                    context['score'] += 5
                    context['details'].append(f"Tendance faible ({adx:.1f})")
            
            # 5. RSI pour momentum extremes
            if rsi:
                # RSI extrêmes = bon contexte pour breakouts
                if rsi <= 30 or rsi >= 70:
                    context['score'] += 15
                    context['confidence_boost'] += 0.05
                    context['details'].append(f"RSI extrême ({rsi:.1f})")
                else:
                    context['score'] += 5
                    context['details'].append(f"RSI neutre ({rsi:.1f})")
            
            # 6. Bollinger expansion (breakouts aiment l'expansion)
            if bb_width:
                if bb_width >= 0.04:  # Forte expansion
                    context['score'] += 15
                    context['confidence_boost'] += 0.05
                    context['details'].append(f"BB expansion ({bb_width:.3f})")
                elif bb_width >= 0.025:
                    context['score'] += 10
                    context['details'].append(f"BB normale ({bb_width:.3f})")
                else:
                    context['score'] -= 5  # Squeeze = mauvais pour breakouts
                    context['details'].append(f"BB compression ({bb_width:.3f})")
            
            # 7. Confluence multi-timeframes
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
            context['confidence_boost'] = max(0, min(0.3, context['confidence_boost']))
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse contexte breakout: {e}")
            context['score'] = 20
        
        return context
    
    def _analyze_resistance_breakout(self, current_price: float, sr_analysis: Dict, 
                                   context: Dict, structure: Dict) -> Dict:
        """Analyse si le prix casse une résistance de manière valide"""
        analysis = {
            'is_valid': False,
            'level': 0.0,
            'percent': 0.0,
            'strength': 0.0,
            'touches': 0,
            'false_risk': 'low'
        }
        
        try:
            strongest_resistance = sr_analysis.get('strongest_resistance')
            if not strongest_resistance:
                return analysis
            
            resistance_level = strongest_resistance['level']
            breakout_percent = (current_price - resistance_level) / resistance_level * 100
            
            # Vérifier si c'est un vrai breakout
            if breakout_percent >= self.min_breakout_percent:
                analysis['level'] = resistance_level
                analysis['percent'] = breakout_percent
                analysis['strength'] = strongest_resistance['strength']
                analysis['touches'] = strongest_resistance['touches']
                
                # Vérifier les conditions de validation
                conditions_met = []
                
                # 1. Score de contexte minimum - AUGMENTÉ de 50 à 70
                if context['score'] >= 70:
                    conditions_met.append('context')
                
                # 2. Structure favorable (uptrend ou neutre)
                if structure['bias'] in ['bullish', 'neutral']:
                    conditions_met.append('structure')
                
                # 3. Confluence si disponible - Rejeter seulement si données disponibles mais insuffisantes
                confluence_score = context.get('confluence_score', 0)
                if confluence_score > 0 and confluence_score < self.confluence_threshold:
                    # Confluence disponible mais trop faible = signal invalide
                    return analysis
                elif confluence_score >= self.confluence_threshold:
                    conditions_met.append('confluence')
                
                # 4. Force du breakout
                if breakout_percent >= self.min_breakout_percent * 2:
                    conditions_met.append('force')
                
                # Évaluer le risque de faux breakout
                if len(conditions_met) >= 3:
                    analysis['false_risk'] = 'low'
                    analysis['is_valid'] = True
                elif len(conditions_met) >= 2:
                    analysis['false_risk'] = 'medium'
                    analysis['is_valid'] = context['score'] >= 60  # Plus strict
                else:
                    analysis['false_risk'] = 'high'
                    analysis['is_valid'] = False
            
        except Exception as e:
            logger.debug(f"Erreur analyse breakout résistance: {e}")
        
        return analysis
    
    def _analyze_support_breakdown(self, current_price: float, sr_analysis: Dict, 
                                  context: Dict, structure: Dict) -> Dict:
        """Analyse si le prix casse un support de manière valide"""
        analysis = {
            'is_valid': False,
            'level': 0.0,
            'percent': 0.0,
            'strength': 0.0,
            'touches': 0,
            'false_risk': 'low'
        }
        
        try:
            strongest_support = sr_analysis.get('strongest_support')
            if not strongest_support:
                return analysis
            
            support_level = strongest_support['level']
            breakdown_percent = (support_level - current_price) / support_level * 100
            
            # Vérifier si c'est un vrai breakdown
            if breakdown_percent >= self.min_breakout_percent:
                analysis['level'] = support_level
                analysis['percent'] = breakdown_percent
                analysis['strength'] = strongest_support['strength']
                analysis['touches'] = strongest_support['touches']
                
                # Vérifier les conditions de validation
                conditions_met = []
                
                # 1. Score de contexte minimum - AUGMENTÉ de 50 à 70
                if context['score'] >= 70:
                    conditions_met.append('context')
                
                # 2. Structure favorable (downtrend ou neutre)
                if structure['bias'] in ['bearish', 'neutral']:
                    conditions_met.append('structure')
                
                # 3. Confluence si disponible - Rejeter seulement si données disponibles mais insuffisantes
                confluence_score = context.get('confluence_score', 0)
                if confluence_score > 0 and confluence_score < self.confluence_threshold:
                    # Confluence disponible mais trop faible = signal invalide
                    return analysis
                elif confluence_score >= self.confluence_threshold:
                    conditions_met.append('confluence')
                
                # 4. Force du breakdown
                if breakdown_percent >= self.min_breakout_percent * 2:
                    conditions_met.append('force')
                
                # Évaluer le risque de faux breakdown
                if len(conditions_met) >= 3:
                    analysis['false_risk'] = 'low'
                    analysis['is_valid'] = True
                elif len(conditions_met) >= 2:
                    analysis['false_risk'] = 'medium'
                    analysis['is_valid'] = context['score'] >= 60  # Plus strict
                else:
                    analysis['false_risk'] = 'high'
                    analysis['is_valid'] = False
            
        except Exception as e:
            logger.debug(f"Erreur analyse breakdown support: {e}")
        
        return analysis
    
    def _calculate_breakout_confidence(self, breakout_analysis: Dict, context: Dict, 
                                     volume_ratio: float, momentum_10: float, is_bullish: bool) -> float:
        """Calcule la confiance pour un signal de breakout"""
        base_confidence = 0.55  # Base plus élevée pour breakouts
        
        # Force du breakout
        base_confidence += min(0.15, breakout_analysis['percent'] / 100 * 5)
        
        # Force du niveau S/R
        base_confidence += min(0.1, breakout_analysis['strength'] / 100)
        
        # Contexte
        base_confidence += context['confidence_boost']
        
        # Volume (critère principal) - SEUILS STANDARDISÉS
        if volume_ratio and volume_ratio >= 2.0:  # STANDARDISÉ: Excellent
            base_confidence += 0.12
        elif volume_ratio and volume_ratio >= 1.5:  # STANDARDISÉ: Très bon
            base_confidence += 0.08
        
        # Momentum directionnel
        if momentum_10:
            if is_bullish and momentum_10 > 1.0:
                base_confidence += 0.08
            elif not is_bullish and momentum_10 < -1.0:
                base_confidence += 0.08
            elif is_bullish and momentum_10 > 0.5:
                base_confidence += 0.05
            elif not is_bullish and momentum_10 < -0.5:
                base_confidence += 0.05
        
        # Pénalité pour risque de faux breakout
        if breakout_analysis['false_risk'] == 'medium':
            base_confidence -= 0.05
        elif breakout_analysis['false_risk'] == 'high':
            base_confidence -= 0.1
        
        # Bonus pour nombre de touches S/R
        if breakout_analysis['touches'] >= 5:
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