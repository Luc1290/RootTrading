"""
Stratégie Breakout Ultra-Précise
Utilise les indicateurs de la DB pour détecter les cassures après consolidation
avec filtres sophistiqués pour éviter les faux breakouts.
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

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class BreakoutStrategy(BaseStrategy):
    """
    Stratégie Breakout Ultra-Précise qui utilise les indicateurs de la DB
    avec des filtres sophistiqués pour détecter les vraies cassures.
    
    Critères ultra-stricts :
    - Détection consolidation avec ATR et volatilité
    - Breakouts confirmés par volume expansion
    - Validation force cassure et retest
    - Filtres anti-faux breakouts
    - Support/résistance basés sur pivot points
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        super().__init__(symbol, params)
        
        # Paramètres breakout ultra-précis
        self.consolidation_periods = 15           # Périodes pour détection consolidation
        self.min_breakout_strength = 0.008        # Force minimum 0.8%
        self.min_volume_expansion = 1.8           # Volume 80% au-dessus moyenne
        self.max_consolidation_noise = 0.03       # Bruit maximum 3% en consolidation
        
        # Filtres ultra-précis
        self.min_confidence = 0.72                # Confiance minimum 72%
        self.retest_validation_periods = 3        # Périodes pour validation retest
        self.false_breakout_threshold = 0.015     # Seuil détection faux breakout 1.5%
        
        logger.info(f"🎯 Breakout Ultra-Précis initialisé pour {symbol}")
    
    @property
    def name(self) -> str:
        return "Breakout_Ultra_Strategy"
    
    def get_min_data_points(self) -> int:
        return 60  # Minimum pour détection consolidation fiable
    
    def analyze(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Optional[Dict]:
        """
        Analyse breakout ultra-précise avec validation complète
        """
        try:
            if len(df) < self.get_min_data_points():
                return None
            
            current_price = df['close'].iloc[-1]
            
            # 1. Détecter la consolidation avec ATR
            consolidation_analysis = self._detect_consolidation_with_atr(df, indicators)
            if not consolidation_analysis.get('is_consolidating'):
                return None
            
            # 2. Détecter le breakout potentiel
            breakout_analysis = self._detect_breakout_signal(df, consolidation_analysis)
            if not breakout_analysis.get('valid_breakout'):
                return None
            
            # 3. Valider avec expansion de volume
            volume_analysis = self._analyze_volume_expansion(df)
            
            # 4. Analyser la force du breakout
            strength_analysis = self._analyze_breakout_strength(df, breakout_analysis, indicators)
            
            # 5. Analyser le contexte de marché
            market_context = self._analyze_market_context(df, indicators)
            
            # 6. Appliquer les filtres ultra-stricts
            if not self._passes_ultra_filters(volume_analysis, strength_analysis, market_context):
                return None
            
            # 7. Logique de signal ultra-sélective
            signal = None
            side = breakout_analysis['side']
            
            confidence = self._calculate_breakout_confidence(
                consolidation_analysis, breakout_analysis, volume_analysis, 
                strength_analysis, market_context
            )
            
            if confidence >= self.min_confidence:
                signal = self._create_signal(
                    symbol, side, current_price, confidence,
                    consolidation_analysis, breakout_analysis, market_context
                )
            
            if signal:
                level = breakout_analysis.get('level', 0)
                logger.info(f"🎯 Breakout {symbol}: {signal['side'].value} @ {current_price:.4f} "
                          f"(niveau: {level:.4f}, conf: {signal['confidence']:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Erreur Breakout Strategy {symbol}: {e}")
            return None
    
    def _get_current_indicator(self, indicators: Dict, key: str) -> Optional[float]:
        """Récupère valeur actuelle d'un indicateur"""
        value = indicators.get(key)
        if value is None:
            return None
        
        if isinstance(value, (list, np.ndarray)) and len(value) > 0:
            return float(value[-1])
        elif isinstance(value, (int, float)):
            return float(value)
        
        return None
    
    def _get_indicator_series(self, indicators: Dict, key: str, length: int) -> Optional[np.ndarray]:
        """Récupère série d'indicateur"""
        value = indicators.get(key)
        if value is None:
            return None
            
        if isinstance(value, (list, np.ndarray)) and len(value) >= length:
            return np.array(value[-length:])
        
        return None
    
    def _detect_consolidation_with_atr(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Détecte consolidation avec ATR et volatilité"""
        try:
            if len(df) < self.consolidation_periods:
                return {'is_consolidating': False}
            
            # Récupérer ATR de la DB
            atr = self._get_current_indicator(indicators, 'atr_14')
            if atr is None:
                # Calcul fallback ATR simple
                recent_data = df.tail(14)
                true_ranges = []
                for i in range(1, len(recent_data)):
                    high = recent_data.iloc[i]['high']
                    low = recent_data.iloc[i]['low']
                    prev_close = recent_data.iloc[i-1]['close']
                    tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                    true_ranges.append(tr)
                atr = np.mean(true_ranges) if true_ranges else 0
            
            # Analyser consolidation sur période
            recent_data = df.tail(self.consolidation_periods)
            high_price = recent_data['high'].max()
            low_price = recent_data['low'].min()
            avg_price = recent_data['close'].mean()
            
            # Calculer bruit vs ATR
            price_range = (high_price - low_price) / avg_price
            atr_ratio = atr / avg_price if avg_price > 0 else 0
            
            # Consolidation = faible volatilité (prix range < seuil)
            is_consolidating = price_range <= self.max_consolidation_noise
            
            # Vérifier stabilité prix
            price_std = recent_data['close'].std() / avg_price
            is_stable = price_std <= 0.02  # 2% écart-type max
            
            return {
                'is_consolidating': is_consolidating and is_stable,
                'support_level': low_price,
                'resistance_level': high_price,
                'range_size': price_range,
                'atr_ratio': atr_ratio,
                'stability_score': 1 - price_std if price_std <= 0.02 else 0
            }
            
        except Exception as e:
            logger.debug(f"Erreur détection consolidation: {e}")
            return {'is_consolidating': False}
    
    def _detect_breakout_signal(self, df: pd.DataFrame, consolidation: Dict) -> Dict:
        """Détecte signal de breakout après consolidation"""
        try:
            if not consolidation.get('is_consolidating'):
                return {'valid_breakout': False}
            
            current_price = df['close'].iloc[-1]
            support = consolidation['support_level']
            resistance = consolidation['resistance_level']
            
            # Détecter breakout haussier
            bullish_breakout = current_price > resistance * (1 + self.min_breakout_strength)
            
            # Détecter breakout baissier
            bearish_breakout = current_price < support * (1 - self.min_breakout_strength)
            
            if bullish_breakout:
                breakout_strength = (current_price - resistance) / resistance
                return {
                    'valid_breakout': True,
                    'side': OrderSide.BUY,
                    'level': resistance,
                    'strength': breakout_strength,
                    'direction': 'bullish'
                }
            elif bearish_breakout:
                breakout_strength = (support - current_price) / support
                return {
                    'valid_breakout': True,
                    'side': OrderSide.SELL,
                    'level': support,
                    'strength': breakout_strength,
                    'direction': 'bearish'
                }
            
            return {'valid_breakout': False}
            
        except Exception as e:
            logger.debug(f"Erreur détection breakout: {e}")
            return {'valid_breakout': False}
    
    def _analyze_volume_expansion(self, df: pd.DataFrame) -> Dict:
        """Analyse expansion du volume pour validation breakout"""
        try:
            if len(df) < 10:
                return {'expansion_confirmed': False, 'volume_ratio': 1.0}
            
            recent_data = df.tail(10)
            current_volume = recent_data['volume'].iloc[-1]
            avg_volume = recent_data['volume'].iloc[:-1].mean()  # Exclure volume actuel
            
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Confirmer expansion minimum
            expansion_confirmed = volume_ratio >= self.min_volume_expansion
            
            # Score d'expansion
            if volume_ratio >= 3.0:
                expansion_score = 0.95
            elif volume_ratio >= 2.0:
                expansion_score = 0.85
            elif volume_ratio >= 1.8:
                expansion_score = 0.75
            else:
                expansion_score = 0.5
            
            return {
                'expansion_confirmed': expansion_confirmed,
                'volume_ratio': volume_ratio,
                'expansion_score': expansion_score
            }
            
        except Exception as e:
            logger.debug(f"Erreur analyse volume: {e}")
            return {'expansion_confirmed': False, 'volume_ratio': 1.0}
    
    def _analyze_breakout_strength(self, df: pd.DataFrame, breakout: Dict, indicators: Dict) -> Dict:
        """Analyse force et validité du breakout"""
        try:
            if not breakout.get('valid_breakout'):
                return {'strong_breakout': False, 'strength_score': 0}
            
            current_price = df['close'].iloc[-1]
            level = breakout['level']
            breakout_strength = breakout.get('strength', 0)
            
            # Score de force basé sur pénétration
            if breakout_strength >= 0.02:  # 2%+
                strength_score = 0.9
            elif breakout_strength >= 0.015:  # 1.5%+
                strength_score = 0.8
            elif breakout_strength >= 0.01:  # 1%+
                strength_score = 0.7
            else:
                strength_score = 0.5
            
            # Vérifier si c'est un faux breakout potentiel
            is_false_breakout = breakout_strength < self.false_breakout_threshold
            
            # Analyser momentum avec RSI
            rsi = self._get_current_indicator(indicators, 'rsi_14')
            momentum_support = False
            if rsi is not None:
                if breakout['side'] == OrderSide.BUY and rsi > 50:
                    momentum_support = True
                elif breakout['side'] == OrderSide.SELL and rsi < 50:
                    momentum_support = True
            
            # Score composé
            if momentum_support:
                strength_score += 0.1
            
            strong_breakout = (
                not is_false_breakout and
                strength_score >= 0.7 and
                breakout_strength >= self.min_breakout_strength
            )
            
            return {
                'strong_breakout': strong_breakout,
                'strength_score': min(1.0, strength_score),
                'penetration_pct': breakout_strength * 100,
                'momentum_support': momentum_support
            }
            
        except Exception as e:
            logger.debug(f"Erreur analyse force breakout: {e}")
            return {'strong_breakout': False, 'strength_score': 0}
    
    def _analyze_market_context(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyse contexte marché pour breakout"""
        try:
            recent_data = df.tail(20)
            
            # Volatilité
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std()
            
            # ATR contexte
            atr = self._get_current_indicator(indicators, 'atr_14') or 0
            current_price = df['close'].iloc[-1]
            atr_pct = atr / current_price if current_price > 0 else 0
            
            return {
                'volatility': volatility,
                'atr_pct': atr_pct,
                'is_low_volatility': volatility <= 0.04,  # Favorable aux breakouts
                'is_high_atr': atr_pct >= 0.02,           # ATR suffisant pour mouvement
            }
            
        except Exception:
            return {'volatility': 0.03, 'atr_pct': 0.02, 'is_low_volatility': True}
    
    def _passes_ultra_filters(self, volume: Dict, strength: Dict, market: Dict) -> bool:
        """Filtres ultra-stricts pour breakout"""
        return (
            # Volume expansion confirmée
            volume.get('expansion_confirmed', False) and
            # Breakout suffisamment fort
            strength.get('strong_breakout', False) and
            # Contexte marché favorable
            market.get('is_low_volatility', False) and
            market.get('is_high_atr', False)
        )
    
    def _calculate_breakout_confidence(self, consolidation: Dict, breakout: Dict,
                                     volume: Dict, strength: Dict, market: Dict) -> float:
        """Confiance pour signal breakout"""
        confidence = 0.6  # Base
        
        # Qualité consolidation
        stability = consolidation.get('stability_score', 0)
        confidence += min(0.1, stability * 0.1)
        
        # Force breakout
        strength_score = strength.get('strength_score', 0)
        confidence += min(0.15, strength_score * 0.15)
        
        # Expansion volume
        expansion_score = volume.get('expansion_score', 0)
        confidence += min(0.1, expansion_score * 0.1)
        
        # Support momentum
        if strength.get('momentum_support'):
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _create_signal(self, symbol: str, side: OrderSide, price: float,
                      confidence: float, consolidation: Dict, breakout: Dict,
                      market: Dict) -> Dict:
        """Crée signal breakout structuré"""
        
        if confidence >= 0.82:
            strength = SignalStrength.VERY_STRONG
        elif confidence >= 0.78:
            strength = SignalStrength.STRONG
        elif confidence >= 0.72:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK
        
        return {
            'strategy': self.name,
            'symbol': symbol,
            'side': side,
            'price': price,
            'confidence': confidence,
            'strength': strength,
            'timestamp': datetime.now(),
            'metadata': {
                'breakout_level': breakout.get('level', 0),
                'breakout_strength': breakout.get('strength', 0),
                'consolidation_range': consolidation.get('range_size', 0),
                'volume_expansion': market.get('volume_ratio', 1),
                'signal_type': 'breakout_ultra_precise'
            }
        }
    
