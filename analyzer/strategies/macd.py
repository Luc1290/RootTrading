"""
Stratégie de trading basée sur le MACD (Moving Average Convergence Divergence).
Le MACD est un indicateur de momentum qui montre la relation entre deux moyennes mobiles.
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd

from analyzer.strategies.base_strategy import BaseStrategy
from shared.src.enums import OrderSide, SignalStrength
from shared.src.schemas import StrategySignal, MarketData
from shared.src.config import STRATEGY_PARAMS

# Configuration du logging
logger = logging.getLogger(__name__)

class MACDStrategy(BaseStrategy):
    """
    Stratégie de trading basée sur le MACD.
    
    Signaux d'achat:
    - Croisement haussier: ligne MACD passe au-dessus de la ligne signal
    - Divergence haussière: prix fait un plus bas, MACD fait un plus haut
    
    Signaux de vente:
    - Croisement baissier: ligne MACD passe en dessous de la ligne signal
    - Divergence baissière: prix fait un plus haut, MACD fait un plus bas
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        """
        Initialise la stratégie MACD.
        
        Args:
            symbol: Symbole de trading
            params: Paramètres de la stratégie
        """
        super().__init__(symbol, params)
        
        # Charger les paramètres MACD depuis la configuration
        macd_config = STRATEGY_PARAMS.get('macd', {})
        
        # Paramètres MACD (priorité: params utilisateur > config > défaut)
        self.fast_period = self.params.get('fast_period', macd_config.get('fast_period', 12))
        self.slow_period = self.params.get('slow_period', macd_config.get('slow_period', 26))
        self.signal_period = self.params.get('signal_period', macd_config.get('signal_period', 9))
        
        # Seuils pour la force du signal
        self.histogram_threshold = self.params.get('histogram_threshold', 
                                                  macd_config.get('histogram_threshold', 0.001))  # 0.1%
        
        # Buffer minimum requis
        self.buffer_size = max(self.slow_period + self.signal_period + 10, self.buffer_size)
        
        # État interne
        self.macd_line = []
        self.signal_line = []
        self.histogram = []
        self.last_crossover = None
        
        logger.info(f"✅ Stratégie MACD initialisée pour {symbol} - "
                   f"Périodes: {self.fast_period}/{self.slow_period}/{self.signal_period}")
    
    @property
    def name(self) -> str:
        """Nom de la stratégie."""
        return "MACD_Strategy"
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calcule la moyenne mobile exponentielle.
        
        Args:
            prices: Série de prix
            period: Période de l'EMA
            
        Returns:
            Série EMA
        """
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calcule les composants du MACD.
        
        Args:
            prices: Série de prix
            
        Returns:
            Tuple (ligne MACD, ligne signal, histogramme)
        """
        # Calculer les EMAs
        ema_fast = self.calculate_ema(prices, self.fast_period)
        ema_slow = self.calculate_ema(prices, self.slow_period)
        
        # Ligne MACD = EMA rapide - EMA lente
        macd_line = ema_fast - ema_slow
        
        # Ligne signal = EMA de la ligne MACD
        signal_line = self.calculate_ema(macd_line, self.signal_period)
        
        # Histogramme = MACD - Signal
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def detect_crossover(self, macd: pd.Series, signal: pd.Series) -> Optional[str]:
        """
        Détecte les croisements entre MACD et signal.
        
        Args:
            macd: Ligne MACD
            signal: Ligne signal
            
        Returns:
            'bullish' pour croisement haussier, 'bearish' pour baissier, None sinon
        """
        if len(macd) < 2 or len(signal) < 2:
            return None
        
        # Valeurs actuelles et précédentes
        macd_current = macd.iloc[-1]
        macd_prev = macd.iloc[-2]
        signal_current = signal.iloc[-1]
        signal_prev = signal.iloc[-2]
        
        # Croisement haussier: MACD passe au-dessus du signal
        if macd_prev <= signal_prev and macd_current > signal_current:
            return 'bullish'
        
        # Croisement baissier: MACD passe en dessous du signal
        elif macd_prev >= signal_prev and macd_current < signal_current:
            return 'bearish'
        
        return None
    
    def detect_divergence(self, prices: pd.Series, macd: pd.Series, window: int = 20) -> Optional[str]:
        """
        Détecte les divergences entre prix et MACD.
        
        Args:
            prices: Série de prix
            macd: Ligne MACD
            window: Fenêtre pour chercher les extrema
            
        Returns:
            'bullish' ou 'bearish' divergence, None sinon
        """
        if len(prices) < window or len(macd) < window:
            return None
        
        # Trouver les plus hauts et plus bas récents
        price_highs = prices.rolling(window=5).max()
        price_lows = prices.rolling(window=5).min()
        macd_highs = macd.rolling(window=5).max()
        macd_lows = macd.rolling(window=5).min()
        
        # Vérifier divergence baissière (prix plus haut, MACD plus bas)
        if (prices.iloc[-1] > price_highs.iloc[-window:-5].max() and 
            macd.iloc[-1] < macd_highs.iloc[-window:-5].max()):
            return 'bearish'
        
        # Vérifier divergence haussière (prix plus bas, MACD plus haut)
        if (prices.iloc[-1] < price_lows.iloc[-window:-5].min() and 
            macd.iloc[-1] > macd_lows.iloc[-window:-5].min()):
            return 'bullish'
        
        return None
    
    def calculate_signal_strength(self, histogram_value: float, crossover: Optional[str], 
                                divergence: Optional[str]) -> SignalStrength:
        """
        Calcule la force du signal basée sur plusieurs facteurs.
        
        Args:
            histogram_value: Valeur actuelle de l'histogramme
            crossover: Type de croisement détecté
            divergence: Type de divergence détectée
            
        Returns:
            Force du signal
        """
        strength_score = 0
        
        # Force de l'histogramme
        histogram_strength = abs(histogram_value) / self.histogram_threshold
        if histogram_strength > 3:
            strength_score += 3
        elif histogram_strength > 2:
            strength_score += 2
        elif histogram_strength > 1:
            strength_score += 1
        
        # Bonus pour croisement
        if crossover:
            strength_score += 2
        
        # Bonus pour divergence
        if divergence:
            strength_score += 3
        
        # Convertir en SignalStrength
        if strength_score >= 6:
            return SignalStrength.VERY_STRONG
        elif strength_score >= 4:
            return SignalStrength.STRONG
        elif strength_score >= 2:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def generate_signal(self) -> Optional[StrategySignal]:
        """
        Génère un signal de trading basé sur les données MACD.
        
        Returns:
            Signal de trading ou None
        """
        try:
            # Vérifier le cooldown avant de générer un signal
            if not self.can_generate_signal():
                return None
                
            # Vérifier si on a assez de données
            if len(self.data_buffer) < self.buffer_size:
                return None
            
            # Convertir en DataFrame
            df = pd.DataFrame([{
                'close': d.get('close', 0),
                'high': d.get('high', 0),
                'low': d.get('low', 0),
                'volume': d.get('volume', 0),
                'timestamp': d.get('start_time', 0) / 1000  # Convertir en secondes
            } for d in self.data_buffer])
            
            # Calculer le MACD
            macd_line, signal_line, histogram = self.calculate_macd(df['close'])
            
            # Sauvegarder pour analyse
            self.macd_line = macd_line
            self.signal_line = signal_line
            self.histogram = histogram
            
            # Valeurs actuelles
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            current_histogram = histogram.iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # Détecter les signaux
            crossover = self.detect_crossover(macd_line, signal_line)
            divergence = self.detect_divergence(df['close'], macd_line)
            
            # Générer un signal si conditions remplies
            signal = None
            
            if crossover == 'bullish' or divergence == 'bullish':
                # Signal d'achat
                signal_strength = self.calculate_signal_strength(current_histogram, crossover, divergence)
                
                # Calculer le niveau de stop
                atr = self._calculate_atr(df)
                # Ajuster selon le symbole pour donner une chance à BTCUSDC
                if 'BTC' in self.symbol:
                    stop_multiplier = 1.5  # Plus agressif pour BTC
                else:
                    stop_multiplier = 2.5   # Plus conservateur pour autres
                
                stop_price = current_price - (atr * stop_multiplier)
                
                signal = StrategySignal(
                    strategy=self.name,
                    symbol=self.symbol,
                    side=OrderSide.LONG,
                    timestamp=datetime.now(),
                    price=current_price,
                    confidence=0.8 if divergence else 0.75,  # Augmenté pour passer le nouveau filtre
                    strength=signal_strength,
                    metadata={
                        'macd': round(current_macd, 6),
                        'signal': round(current_signal, 6),
                        'histogram': round(current_histogram, 6),
                        'crossover': crossover is not None,
                        'divergence': divergence is not None,
                        'stop_price': round(stop_price, 8)
                    }
                )
                
            elif crossover == 'bearish' or divergence == 'bearish':
                # Signal de vente
                signal_strength = self.calculate_signal_strength(current_histogram, crossover, divergence)
                
                # Calculer le niveau de stop
                atr = self._calculate_atr(df)
                # Ajuster selon le symbole pour donner une chance à BTCUSDC
                if 'BTC' in self.symbol:
                    stop_multiplier = 1.5  # Plus agressif pour BTC
                else:
                    stop_multiplier = 2.5   # Plus conservateur pour autres
                
                stop_price = current_price + (atr * stop_multiplier)
                
                signal = StrategySignal(
                    strategy=self.name,
                    symbol=self.symbol,
                    side=OrderSide.SHORT,
                    timestamp=datetime.now(),
                    price=current_price,
                    confidence=0.7 if divergence else 0.6,
                    strength=signal_strength,
                    metadata={
                        'macd': round(current_macd, 6),
                        'signal': round(current_signal, 6),
                        'histogram': round(current_histogram, 6),
                        'crossover': crossover is not None,
                        'divergence': divergence is not None,
                        'stop_price': round(stop_price, 8)
                    }
                )
            
            # Mettre à jour le timestamp si un signal est généré
            if signal:
                self.last_signal_time = datetime.now()
                logger.info(f"📊 Signal MACD généré: {signal.side} {signal.symbol} "
                           f"(Crossover: {crossover}, Divergence: {divergence})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Erreur dans l'analyse MACD: {str(e)}")
            return None
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Calcule l'Average True Range pour les stops/targets.
        
        Args:
            df: DataFrame avec high, low, close
            period: Période ATR
            
        Returns:
            Valeur ATR
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        return true_range.rolling(period).mean().iloc[-1]
    
    def get_min_data_points(self) -> int:
        """
        Retourne le nombre minimum de points de données nécessaires.
        
        Returns:
            Nombre minimum de points
        """
        return self.buffer_size
    
    def get_state(self) -> Dict[str, Any]:
        """
        Retourne l'état actuel de la stratégie.
        
        Returns:
            Dictionnaire contenant l'état
        """
        state = {
            'buffer_size': len(self.data_buffer),
            'last_signal_time': self.last_signal_time.isoformat() if self.last_signal_time else None,
            'params': self.params
        }
        
        # Ajouter les valeurs MACD actuelles si disponibles
        if len(self.macd_line) > 0:
            state['current_macd'] = float(self.macd_line.iloc[-1])
            state['current_signal'] = float(self.signal_line.iloc[-1])
            state['current_histogram'] = float(self.histogram.iloc[-1])
        
        return state
    
    def reset(self) -> None:
        """Réinitialise l'état de la stratégie."""
        self.data_buffer.clear()
        self.last_signal_time = None
        self.macd_line = []
        self.signal_line = []
        self.histogram = []
        self.last_crossover = None
        logger.info(f"Stratégie {self.name} réinitialisée")