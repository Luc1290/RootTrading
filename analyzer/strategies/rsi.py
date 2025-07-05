"""
Stratégie RSI Ultra-Précise et Pointue
Utilise les indicateurs pré-calculés de la DB avec filtres multi-critères
pour générer des signaux de très haute qualité.
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from shared.src.config import get_strategy_param
from shared.src.enums import OrderSide, SignalStrength
from shared.src.schemas import StrategySignal

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class RSIStrategy(BaseStrategy):
    """
    Stratégie RSI Ultra-Précise qui utilise les indicateurs de la DB
    avec des filtres sophistiqués pour éviter les faux signaux.
    
    Critères stricts :
    - RSI extrême (< 20 ou > 80)
    - Divergences RSI/Prix
    - Confirmation volume
    - Support/Résistance
    - Volatilité contrôlée
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        super().__init__(symbol, params)
        
        # Seuils RSI ultra-précis
        self.extreme_oversold = 20    # Zone d'achat ultra-sélective
        self.extreme_overbought = 80  # Zone de vente ultra-sélective
        self.rsi_exit_low = 35        # Sortie position courte
        self.rsi_exit_high = 65       # Sortie position longue
        
        # Filtres de qualité
        self.min_volume_ratio = 1.4   # Volume 40% au-dessus moyenne
        self.max_volatility = 0.06    # 6% volatilité max
        self.min_confidence = 0.75    # Confiance minimum 75%
        
        # Divergences
        self.divergence_periods = 12
        self.min_divergence_strength = 0.6
        
        logger.info(f"🎯 RSI Ultra-Précis initialisé pour {symbol}")
    
    @property
    def name(self) -> str:
        return "RSI_Ultra_Strategy"
    
    def get_min_data_points(self) -> int:
        return 50  # Minimum pour analyse complète
    
    def analyze(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> Optional[Dict]:
        """
        Analyse RSI ultra-précise avec filtres multiples
        """
        try:
            if len(df) < self.get_min_data_points():
                return None
            
            # 1. Récupérer RSI pré-calculé de la DB
            current_rsi = self._get_current_indicator(indicators, 'rsi_14')
            if current_rsi is None:
                logger.debug(f"❌ {symbol}: RSI non disponible")
                return None
            
            current_price = df['close'].iloc[-1]
            
            # 2. Analyser le contexte de marché
            market_context = self._analyze_market_context(df, indicators)
            
            # 3. Appliquer les filtres de qualité
            if not self._passes_quality_filters(market_context):
                return None
            
            # 4. Détecter les divergences RSI/Prix
            divergence_analysis = self._detect_rsi_divergence(df, indicators)
            
            # 5. Analyser les niveaux de support/résistance
            sr_levels = self._analyze_support_resistance(df)
            
            # 6. Logique de signal ultra-stricte
            signal = None
            
            # SIGNAL D'ACHAT - Conditions ultra-sélectives
            if (current_rsi <= self.extreme_oversold and
                divergence_analysis.get('bullish_divergence', False) and
                sr_levels.get('near_support', False)):
                
                confidence = self._calculate_buy_confidence(
                    current_rsi, divergence_analysis, market_context, sr_levels
                )
                
                if confidence >= self.min_confidence:
                    signal = self._create_signal(
                        symbol, OrderSide.BUY, current_price, confidence,
                        current_rsi, divergence_analysis, market_context
                    )
            
            # SIGNAL DE VENTE - Conditions ultra-sélectives  
            elif (current_rsi >= self.extreme_overbought and
                  divergence_analysis.get('bearish_divergence', False) and
                  sr_levels.get('near_resistance', False)):
                
                confidence = self._calculate_sell_confidence(
                    current_rsi, divergence_analysis, market_context, sr_levels
                )
                
                if confidence >= self.min_confidence:
                    signal = self._create_signal(
                        symbol, OrderSide.SELL, current_price, confidence,
                        current_rsi, divergence_analysis, market_context
                    )
            
            if signal:
                logger.info(f"🎯 RSI {symbol}: {signal['side'].value} @ {current_price:.4f} "
                          f"(RSI: {current_rsi:.1f}, conf: {signal['confidence']:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Erreur RSI Strategy {symbol}: {e}")
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
    
    def _analyze_market_context(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyse le contexte de marché"""
        try:
            recent_data = df.tail(20)
            
            # Volume ratio
            current_volume = recent_data['volume'].iloc[-1]
            avg_volume = recent_data['volume'].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Volatilité
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Tendance
            price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            
            # ATR pour contexte
            atr = self._get_current_indicator(indicators, 'atr_14') or 0
            
            return {
                'volume_ratio': volume_ratio,
                'volatility': volatility,
                'trend_pct': price_change * 100,
                'atr': atr,
                'is_trending': abs(price_change) > 0.02  # 2% mouvement
            }
            
        except Exception as e:
            logger.debug(f"Erreur contexte marché: {e}")
            return {'volume_ratio': 1.0, 'volatility': 0.1}
    
    def _passes_quality_filters(self, market_context: Dict) -> bool:
        """Vérifie si les conditions de marché sont favorables"""
        return (
            market_context.get('volume_ratio', 0) >= self.min_volume_ratio and
            market_context.get('volatility', 1) <= self.max_volatility and
            market_context.get('volatility', 0) >= 0.005  # Minimum de mouvement
        )
    
    def _detect_rsi_divergence(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Détecte les divergences RSI/Prix"""
        try:
            if len(df) < self.divergence_periods + 5:
                return {'bullish_divergence': False, 'bearish_divergence': False}
            
            # Extraire les données récentes
            recent_df = df.tail(self.divergence_periods)
            recent_prices = recent_df['close'].values
            
            # Récupérer RSI de la DB
            rsi_series = indicators.get('rsi_14')
            if rsi_series is None:
                return {'bullish_divergence': False, 'bearish_divergence': False}
            
            if isinstance(rsi_series, (list, np.ndarray)) and len(rsi_series) >= self.divergence_periods:
                recent_rsi = rsi_series[-self.divergence_periods:]
            else:
                return {'bullish_divergence': False, 'bearish_divergence': False}
            
            # Calculer les tendances
            price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            rsi_trend = np.polyfit(range(len(recent_rsi)), recent_rsi, 1)[0]
            
            # Normaliser pour comparaison
            price_trend_norm = (price_trend / recent_prices[0]) * 100
            rsi_trend_norm = rsi_trend
            
            # Détecter divergences
            bullish_div = (price_trend_norm < -1.0 and rsi_trend_norm > 0.3)
            bearish_div = (price_trend_norm > 1.0 and rsi_trend_norm < -0.3)
            
            # Force de la divergence
            divergence_strength = abs(price_trend_norm - rsi_trend_norm) / 10
            
            return {
                'bullish_divergence': bullish_div and divergence_strength >= self.min_divergence_strength,
                'bearish_divergence': bearish_div and divergence_strength >= self.min_divergence_strength,
                'strength': min(1.0, divergence_strength)
            }
            
        except Exception as e:
            logger.debug(f"Erreur divergence RSI: {e}")
            return {'bullish_divergence': False, 'bearish_divergence': False}
    
    def _analyze_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Analyse les niveaux de support/résistance"""
        try:
            recent_data = df.tail(30)
            current_price = df['close'].iloc[-1]
            
            # Identifier support et résistance
            support = recent_data['low'].min()
            resistance = recent_data['high'].max()
            
            # Distances
            support_dist = (current_price - support) / current_price
            resistance_dist = (resistance - current_price) / current_price
            
            return {
                'support_level': support,
                'resistance_level': resistance,
                'near_support': support_dist <= 0.015,  # 1.5%
                'near_resistance': resistance_dist <= 0.015,  # 1.5%
                'support_distance': support_dist,
                'resistance_distance': resistance_dist
            }
            
        except Exception:
            return {'near_support': False, 'near_resistance': False}
    
    def _calculate_buy_confidence(self, rsi: float, divergence: Dict, 
                                market: Dict, sr: Dict) -> float:
        """Calcule la confiance pour un signal d'achat"""
        confidence = 0.6  # Base
        
        # RSI extrême
        if rsi <= 15:
            confidence += 0.2
        elif rsi <= 20:
            confidence += 0.15
        
        # Divergence
        if divergence.get('bullish_divergence'):
            confidence += min(0.15, divergence.get('strength', 0) * 0.15)
        
        # Volume élevé
        vol_ratio = market.get('volume_ratio', 1)
        if vol_ratio >= 2.0:
            confidence += 0.1
        elif vol_ratio >= 1.5:
            confidence += 0.05
        
        # Support proche
        if sr.get('near_support'):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _calculate_sell_confidence(self, rsi: float, divergence: Dict,
                                 market: Dict, sr: Dict) -> float:
        """Calcule la confiance pour un signal de vente"""
        confidence = 0.6  # Base
        
        # RSI extrême
        if rsi >= 85:
            confidence += 0.2
        elif rsi >= 80:
            confidence += 0.15
        
        # Divergence
        if divergence.get('bearish_divergence'):
            confidence += min(0.15, divergence.get('strength', 0) * 0.15)
        
        # Volume élevé
        vol_ratio = market.get('volume_ratio', 1)
        if vol_ratio >= 2.0:
            confidence += 0.1
        elif vol_ratio >= 1.5:
            confidence += 0.05
        
        # Résistance proche
        if sr.get('near_resistance'):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _create_signal(self, symbol: str, side: OrderSide, price: float,
                      confidence: float, rsi: float, divergence: Dict,
                      market: Dict) -> Dict:
        """Crée un signal structuré"""
        
        # Déterminer la force
        if confidence >= 0.9:
            strength = SignalStrength.VERY_STRONG
        elif confidence >= 0.8:
            strength = SignalStrength.STRONG
        elif confidence >= 0.75:
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
                'rsi': rsi,
                'divergence_strength': divergence.get('strength', 0),
                'volume_ratio': market.get('volume_ratio', 1),
                'volatility': market.get('volatility', 0),
                'signal_type': 'rsi_ultra_precise'
            }
        }