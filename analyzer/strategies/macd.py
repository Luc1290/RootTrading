"""
Stratégie MACD Ultra-Précise
Utilise les indicateurs MACD pré-calculés de la DB avec filtres avancés
pour générer des signaux de momentum de très haute qualité.
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
from shared.src.schemas import StrategySignal

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class MACDStrategy(BaseStrategy):
    """
    Stratégie MACD Ultra-Précise qui utilise les indicateurs de la DB
    avec des filtres sophistiqués pour des signaux de momentum fiables.
    
    Critères ultra-stricts :
    - Croisements MACD confirmés par momentum
    - Divergences MACD/Prix validées
    - Confirmation volume et volatilité
    - Histogramme en expansion
    - Contexte de tendance favorable
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        super().__init__(symbol, params)
        
        # Paramètres MACD ultra-précis
        self.min_crossover_strength = 0.0002  # Force minimum du croisement
        self.min_histogram_expansion = 0.0001  # Expansion minimum histogramme
        self.min_volume_confirmation = 1.5    # Volume 50% au-dessus moyenne
        self.max_noise_ratio = 0.25           # Ratio bruit/signal maximum
        
        # Filtres de qualité du marché
        self.min_volatility = 0.008           # Volatilité minimum pour mouvement
        self.max_volatility = 0.10            # Volatilité maximum pour stabilité
        self.min_confidence = 0.80            # Confiance minimum 80%
        
        # Divergences
        self.divergence_lookback = 15
        self.min_divergence_periods = 6
        
        logger.info(f"🎯 MACD Ultra-Précis initialisé pour {symbol}")
    
    @property
    def name(self) -> str:
        return "MACD_Ultra_Strategy"
    
    def get_min_data_points(self) -> int:
        return 60  # Minimum pour MACD fiable
    
    def analyze(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Optional[Dict]:
        """
        Analyse MACD ultra-précise avec validation complète
        """
        try:
            if len(df) < self.get_min_data_points():
                return None
            
            # 1. Récupérer les indicateurs MACD pré-calculés
            macd_line = self._get_current_indicator(indicators, 'macd_line')
            macd_signal = self._get_current_indicator(indicators, 'macd_signal')
            macd_hist = self._get_current_indicator(indicators, 'macd_histogram')
            
            if None in [macd_line, macd_signal, macd_hist]:
                logger.debug(f"❌ {symbol}: Indicateurs MACD non disponibles")
                return None
            
            current_price = df['close'].iloc[-1]
            
            # 2. Analyser les croisements MACD
            crossover_analysis = self._analyze_macd_crossover(indicators)
            
            # 3. Analyser l'expansion de l'histogramme
            histogram_analysis = self._analyze_histogram_momentum(indicators)
            
            # 4. Détecter les divergences MACD/Prix
            divergence_analysis = self._detect_macd_divergence(df, indicators)
            
            # 5. Analyser le contexte de marché
            market_context = self._analyze_market_context(df, indicators)
            
            # 6. Appliquer les filtres de qualité
            if not self._passes_ultra_filters(market_context, histogram_analysis):
                return None
            
            # 7. Logique de signal ultra-sélective
            signal = None
            
            # SIGNAL D'ACHAT MACD - Conditions ultra-strictes
            if (crossover_analysis.get('bullish_crossover') and
                histogram_analysis.get('expanding_bullish') and
                divergence_analysis.get('bullish_divergence', False)):
                
                confidence = self._calculate_buy_confidence(
                    crossover_analysis, histogram_analysis, divergence_analysis, market_context
                )
                
                if confidence >= self.min_confidence:
                    signal = self._create_signal(
                        symbol, OrderSide.BUY, current_price, confidence,
                        macd_line, macd_signal, macd_hist, market_context
                    )
            
            # SIGNAL DE VENTE MACD - Conditions ultra-strictes
            elif (crossover_analysis.get('bearish_crossover') and
                  histogram_analysis.get('expanding_bearish') and
                  divergence_analysis.get('bearish_divergence', False)):
                
                confidence = self._calculate_sell_confidence(
                    crossover_analysis, histogram_analysis, divergence_analysis, market_context
                )
                
                if confidence >= self.min_confidence:
                    signal = self._create_signal(
                        symbol, OrderSide.SELL, current_price, confidence,
                        macd_line, macd_signal, macd_hist, market_context
                    )
            
            if signal:
                logger.info(f"🎯 MACD {symbol}: {signal['side'].value} @ {current_price:.4f} "
                          f"(MACD: {macd_line:.4f}, conf: {signal['confidence']:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Erreur MACD Strategy {symbol}: {e}")
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
    
    def _get_indicator_series(self, indicators: Dict, key: str, length: int) -> Optional[np.ndarray]:
        """Récupère une série d'indicateur"""
        value = indicators.get(key)
        if value is None:
            return None
            
        if isinstance(value, (list, np.ndarray)) and len(value) >= length:
            return np.array(value[-length:])
        
        return None
    
    def _analyze_macd_crossover(self, indicators: Dict) -> Dict:
        """Analyse les croisements MACD avec validation de force"""
        try:
            # Récupérer les séries MACD
            macd_series = self._get_indicator_series(indicators, 'macd_line', 3)
            signal_series = self._get_indicator_series(indicators, 'macd_signal', 3)
            
            if macd_series is None or signal_series is None or len(macd_series) < 3:
                return {'bullish_crossover': False, 'bearish_crossover': False}
            
            # Valeurs actuelles et précédentes
            current_macd, prev_macd = macd_series[-1], macd_series[-2]
            current_signal, prev_signal = signal_series[-1], signal_series[-2]
            
            # Détecter croisements
            bullish_crossover = (prev_macd <= prev_signal and current_macd > current_signal)
            bearish_crossover = (prev_macd >= prev_signal and current_macd < current_signal)
            
            # Mesurer la force du croisement
            crossover_gap = abs(current_macd - current_signal)
            max_reference = max(abs(current_macd), abs(current_signal), 0.001)
            crossover_strength = crossover_gap / max_reference
            
            # Valider la force minimum
            strong_enough = crossover_strength >= self.min_crossover_strength
            
            # Vérifier la persistance (pas de faux signal)
            if len(macd_series) >= 3:
                prev2_macd, prev2_signal = macd_series[-3], signal_series[-3]
                direction_consistent = (
                    (bullish_crossover and current_macd > prev2_macd) or
                    (bearish_crossover and current_macd < prev2_macd)
                )
            else:
                direction_consistent = True
            
            return {
                'bullish_crossover': bullish_crossover and strong_enough and direction_consistent,
                'bearish_crossover': bearish_crossover and strong_enough and direction_consistent,
                'crossover_strength': crossover_strength,
                'direction_consistent': direction_consistent
            }
            
        except Exception as e:
            logger.debug(f"Erreur analyse croisement MACD: {e}")
            return {'bullish_crossover': False, 'bearish_crossover': False}
    
    def _analyze_histogram_momentum(self, indicators: Dict) -> Dict:
        """Analyse l'expansion/contraction de l'histogramme MACD"""
        try:
            hist_series = self._get_indicator_series(indicators, 'macd_histogram', 5)
            if hist_series is None or len(hist_series) < 3:
                return {'expanding_bullish': False, 'expanding_bearish': False}
            
            current_hist = hist_series[-1]
            prev_hist = hist_series[-2]
            
            # Calculer l'expansion
            hist_change = current_hist - prev_hist
            expansion_rate = abs(hist_change) / max(abs(prev_hist), 0.0001)
            
            # Vérifier expansion minimum
            sufficient_expansion = expansion_rate >= self.min_histogram_expansion
            
            # Direction de l'expansion
            expanding_bullish = (current_hist > 0 and hist_change > 0 and sufficient_expansion)
            expanding_bearish = (current_hist < 0 and hist_change < 0 and sufficient_expansion)
            
            # Tendance de l'histogramme sur 5 périodes
            if len(hist_series) >= 5:
                hist_trend = np.polyfit(range(len(hist_series)), hist_series, 1)[0]
                trend_bullish = hist_trend > 0
                trend_bearish = hist_trend < 0
            else:
                trend_bullish = trend_bearish = False
            
            return {
                'expanding_bullish': expanding_bullish and trend_bullish,
                'expanding_bearish': expanding_bearish and trend_bearish,
                'expansion_rate': expansion_rate,
                'histogram_trend': hist_trend if len(hist_series) >= 5 else 0
            }
            
        except Exception as e:
            logger.debug(f"Erreur analyse histogramme: {e}")
            return {'expanding_bullish': False, 'expanding_bearish': False}
    
    def _detect_macd_divergence(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Détecte les divergences MACD/Prix avec validation"""
        try:
            if len(df) < self.divergence_lookback + 5:
                return {'bullish_divergence': False, 'bearish_divergence': False}
            
            # Extraire données récentes
            recent_df = df.tail(self.divergence_lookback)
            recent_prices = recent_df['close'].values
            
            # Récupérer série MACD
            macd_series = self._get_indicator_series(indicators, 'macd_line', self.divergence_lookback)
            if macd_series is None:
                return {'bullish_divergence': False, 'bearish_divergence': False}
            
            # Calculer tendances linéaires
            price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            macd_trend = np.polyfit(range(len(macd_series)), macd_series, 1)[0]
            
            # Normaliser pour comparaison
            price_trend_pct = (price_trend / recent_prices[0]) * 100
            macd_trend_normalized = macd_trend * 1000  # Ajuster échelle
            
            # Seuils pour divergences significatives
            min_price_move = 1.5  # 1.5% mouvement prix minimum
            min_macd_counter = 0.8  # Mouvement MACD contraire minimum
            
            # Détecter divergences
            bullish_divergence = (
                price_trend_pct < -min_price_move and 
                macd_trend_normalized > min_macd_counter
            )
            bearish_divergence = (
                price_trend_pct > min_price_move and 
                macd_trend_normalized < -min_macd_counter
            )
            
            # Force de la divergence
            divergence_strength = abs(price_trend_pct + macd_trend_normalized) / 100
            
            return {
                'bullish_divergence': bullish_divergence,
                'bearish_divergence': bearish_divergence,
                'strength': min(1.0, divergence_strength),
                'price_trend_pct': price_trend_pct,
                'macd_trend': macd_trend_normalized
            }
            
        except Exception as e:
            logger.debug(f"Erreur divergence MACD: {e}")
            return {'bullish_divergence': False, 'bearish_divergence': False}
    
    def _analyze_market_context(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyse le contexte de marché pour MACD"""
        try:
            recent_data = df.tail(20)
            
            # Volume
            current_volume = recent_data['volume'].iloc[-1]
            avg_volume = recent_data['volume'].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Volatilité
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std()
            
            # ATR pour contexte
            atr = self._get_current_indicator(indicators, 'atr_14') or 0
            
            # Tendance EMA
            ema_12 = self._get_current_indicator(indicators, 'ema_12')
            ema_26 = self._get_current_indicator(indicators, 'ema_26')
            
            trend_alignment = None
            if ema_12 and ema_26:
                trend_alignment = "bullish" if ema_12 > ema_26 else "bearish"
            
            # Ratio signal/bruit
            price_range = recent_data['high'].max() - recent_data['low'].min()
            noise_ratio = (atr / price_range) if price_range > 0 else 1.0
            
            return {
                'volume_ratio': volume_ratio,
                'volatility': volatility,
                'atr': atr,
                'trend_alignment': trend_alignment,
                'noise_ratio': noise_ratio,
                'is_clean_market': noise_ratio <= self.max_noise_ratio
            }
            
        except Exception as e:
            logger.debug(f"Erreur contexte marché: {e}")
            return {'volume_ratio': 1.0, 'volatility': 0.1, 'is_clean_market': False}
    
    def _passes_ultra_filters(self, market_context: Dict, histogram_analysis: Dict) -> bool:
        """Applique les filtres ultra-stricts pour la qualité du signal"""
        return (
            # Volume suffisant
            market_context.get('volume_ratio', 0) >= self.min_volume_confirmation and
            # Volatilité dans la plage optimale
            self.min_volatility <= market_context.get('volatility', 0) <= self.max_volatility and
            # Marché propre (peu de bruit)
            market_context.get('is_clean_market', False) and
            # Expansion de l'histogramme
            (histogram_analysis.get('expanding_bullish') or histogram_analysis.get('expanding_bearish'))
        )
    
    def _calculate_buy_confidence(self, crossover: Dict, histogram: Dict,
                                divergence: Dict, market: Dict) -> float:
        """Calcule la confiance pour un signal d'achat MACD"""
        confidence = 0.65  # Base élevée
        
        # Force du croisement
        crossover_strength = crossover.get('crossover_strength', 0)
        confidence += min(0.15, crossover_strength * 50)
        
        # Expansion histogramme
        expansion_rate = histogram.get('expansion_rate', 0)
        confidence += min(0.1, expansion_rate * 100)
        
        # Divergence
        if divergence.get('bullish_divergence'):
            confidence += min(0.1, divergence.get('strength', 0))
        
        # Volume exceptionnel
        vol_ratio = market.get('volume_ratio', 1)
        if vol_ratio >= 2.0:
            confidence += 0.05
        
        # Alignement de tendance
        if market.get('trend_alignment') == 'bullish':
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _calculate_sell_confidence(self, crossover: Dict, histogram: Dict,
                                 divergence: Dict, market: Dict) -> float:
        """Calcule la confiance pour un signal de vente MACD"""
        confidence = 0.65  # Base élevée
        
        # Force du croisement
        crossover_strength = crossover.get('crossover_strength', 0)
        confidence += min(0.15, crossover_strength * 50)
        
        # Expansion histogramme
        expansion_rate = histogram.get('expansion_rate', 0)
        confidence += min(0.1, expansion_rate * 100)
        
        # Divergence
        if divergence.get('bearish_divergence'):
            confidence += min(0.1, divergence.get('strength', 0))
        
        # Volume exceptionnel
        vol_ratio = market.get('volume_ratio', 1)
        if vol_ratio >= 2.0:
            confidence += 0.05
        
        # Alignement de tendance
        if market.get('trend_alignment') == 'bearish':
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _create_signal(self, symbol: str, side: OrderSide, price: float,
                      confidence: float, macd_line: float, macd_signal: float,
                      macd_hist: float, market: Dict) -> Dict:
        """Crée un signal MACD structuré"""
        
        # Déterminer la force
        if confidence >= 0.9:
            strength = SignalStrength.VERY_STRONG
        elif confidence >= 0.85:
            strength = SignalStrength.STRONG
        elif confidence >= 0.8:
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
                'macd_line': macd_line,
                'macd_signal': macd_signal,
                'macd_histogram': macd_hist,
                'volume_ratio': market.get('volume_ratio', 1),
                'volatility': market.get('volatility', 0),
                'trend_alignment': market.get('trend_alignment'),
                'signal_type': 'macd_ultra_precise'
            }
        }