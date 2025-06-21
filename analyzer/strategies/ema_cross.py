"""
Stratégie de trading basée sur le croisement de moyennes mobiles exponentielles (EMA).
Génère des signaux lorsque l'EMA courte croise l'EMA longue.
"""
import logging
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import talib

# Importer les modules partagés
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from shared.src.config import get_strategy_param
from shared.src.enums import OrderSide
from shared.src.schemas import StrategySignal

from .base_strategy import BaseStrategy
from .advanced_filters_mixin import AdvancedFiltersMixin

# Configuration du logging
logger = logging.getLogger(__name__)

class EMACrossStrategy(BaseStrategy, AdvancedFiltersMixin):
    """
    Stratégie basée sur le croisement de moyennes mobiles exponentielles (EMA).
    Génère des signaux d'achat quand l'EMA courte croise l'EMA longue vers le haut
    et des signaux de vente quand l'EMA courte croise l'EMA longue vers le bas.
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        """
        Initialise la stratégie EMA Cross.
        
        Args:
            symbol: Symbole de trading (ex: 'BTCUSDC')
            params: Paramètres spécifiques à la stratégie
        """
        super().__init__(symbol, params)
        
        # Paramètres EMA
        self.SELL_window = self.params.get('SELL_window', get_strategy_param('ema_cross', 'SELL_window', 5))
        self.BUY_window = self.params.get('BUY_window', get_strategy_param('ema_cross', 'BUY_window', 20))
        
        # S'assurer que court < long
        if self.SELL_window >= self.BUY_window:
            logger.warning(f"⚠️ Configuration incorrecte: EMA court ({self.SELL_window}) >= EMA long ({self.BUY_window}). Ajustement automatique.")
            self.SELL_window = min(self.SELL_window, self.BUY_window - 1)

        # Variables pour suivre les tendances
        self.prev_SELL_ema = None
        self.prev_BUY_ema = None

        logger.info(f"🔧 Stratégie EMA Cross initialisée pour {symbol} "
                   f"(SELL={self.SELL_window}, BUY={self.BUY_window})")

    @property
    def name(self) -> str:
        """Nom unique de la stratégie."""
        return "EMA_Cross_Strategy"
    
    def get_min_data_points(self) -> int:
        """
        Nombre minimum de points de données nécessaires pour calculer les EMAs.
        
        Returns:
            Nombre minimum de données requises
        """
        # Besoin d'au moins 3 * la fenêtre longue pour avoir un calcul fiable
        return max(self.long_window * 3, 30)
    
    def calculate_emas(self, prices: np.ndarray) -> tuple:
        """
        Calcule les EMAs courte et longue sur une série de prix.
        
        Args:
            prices: Tableau numpy des prix de clôture
            
        Returns:
            Tuple (SELL_ema, BUY_ema) des valeurs EMA
        """
        try:
            # Utiliser TA-Lib pour calculer les EMAs
            SELL_ema = talib.EMA(prices, timeperiod=self.SELL_window)
            BUY_ema = talib.EMA(prices, timeperiod=self.BUY_window)
            return SELL_ema, BUY_ema
        except Exception as e:
            logger.error(f"Erreur lors du calcul des EMAs: {str(e)}")
            # Implémenter un calcul manuel de secours en cas d'erreur TA-Lib
            return self._calculate_emas_manually(prices)
    
    def _calculate_emas_manually(self, prices: np.ndarray) -> tuple:
        """
        Calcule les EMAs manuellement si TA-Lib n'est pas disponible.
        
        Args:
            prices: Tableau numpy des prix de clôture
            
        Returns:
            Tuple (SELL_ema, BUY_ema) des valeurs EMA
        """
        # Initialiser les tableaux
        SELL_ema = np.zeros_like(prices)
        BUY_ema = np.zeros_like(prices)
        
        # Calculer les EMAs
        # Coefficient de lissage
        SELL_alpha = 2 / (self.SELL_window + 1)
        BUY_alpha = 2 / (self.BUY_window + 1)
        
        # Pour la première valeur, utiliser la moyenne simple (SMA)
        for i in range(len(prices)):
            if i < self.SELL_window - 1:
                SELL_ema[i] = np.nan
            elif i == self.SELL_window - 1:
                SELL_ema[i] = np.mean(prices[:self.SELL_window])
            else:
                SELL_ema[i] = (prices[i] * SELL_alpha) + (SELL_ema[i-1] * (1 - SELL_alpha))
                
            if i < self.BUY_window - 1:
                BUY_ema[i] = np.nan
            elif i == self.BUY_window - 1:
                BUY_ema[i] = np.mean(prices[:self.BUY_window])
            else:
                BUY_ema[i] = (prices[i] * BUY_alpha) + (BUY_ema[i-1] * (1 - BUY_alpha))
        
        return SELL_ema, BUY_ema
    
    def generate_signal(self) -> Optional[StrategySignal]:
        """
        Génère un signal de trading sophistiqué basé sur le croisement d'EMAs.
        Utilise des filtres avancés pour éviter les faux signaux.
        
        Returns:
            Signal de trading ou None si aucun signal n'est généré
        """
        # Vérifier le cooldown avant de générer un signal
        if not self.can_generate_signal():
            return None
        
        # Convertir les données en DataFrame
        df = self.get_data_as_dataframe()
        if df is None or len(df) < self.get_min_data_points():
            return None
        
        # Extraire les données nécessaires
        prices = df['close'].values
        volumes = df['volume'].values if 'volume' in df.columns else None
        
        # Calculer les EMAs
        SELL_ema, BUY_ema = self.calculate_emas(prices)
        
        # Obtenir les dernières valeurs
        current_price = prices[-1]
        current_SELL_ema = SELL_ema[-1]
        current_BUY_ema = BUY_ema[-1]
        
        # Loguer les valeurs actuelles
        precision = 5 if 'BTC' in self.symbol else 3
        logger.debug(f"[EMA Cross] {self.symbol}: Price={current_price:.{precision}f}, "
                    f"SELL EMA={current_SELL_ema:.{precision}f}, BUY EMA={current_BUY_ema:.{precision}f}")
        
        # Vérifications de base
        if np.isnan(current_SELL_ema) or np.isnan(current_BUY_ema):
            return None
        
        # === NOUVEAU SYSTÈME DE FILTRES SOPHISTIQUES ===
        
        # 1. FILTRE SETUP EMA DE BASE
        signal_side = self._detect_ema_setup(SELL_ema, BUY_ema, prices)
        if signal_side is None:
            return None
        
        # 2. FILTRE ADX (force de tendance)
        adx_score = self._analyze_adx_strength(df)
        if adx_score < 0.4:
            logger.debug(f"[EMA Cross] {self.symbol}: Signal rejeté - ADX faible ({adx_score:.2f})")
            return None
        
        # 3. FILTRE VOLUME (confirmation institutionnelle)
        volume_score = self._analyze_volume_confirmation_common(volumes) if volumes is not None else 0.7
        if volume_score < 0.4:
            logger.debug(f"[EMA Cross] {self.symbol}: Signal rejeté - volume insuffisant ({volume_score:.2f})")
            return None
        
        # 4. FILTRE PULLBACK VALIDATION (éviter les faux breakouts)
        pullback_score = self._validate_pullback_quality(df, signal_side, current_SELL_ema, current_BUY_ema)
        
        # 5. FILTRE TREND ALIGNMENT (confirmation tendance supérieure)
        trend_score = self._analyze_trend_alignment_common(df, signal_side)
        
        # 6. FILTRE RSI CONFIRMATION (momentum sous-jacent)
        rsi_score = self._calculate_rsi_confirmation_common(df, signal_side)
        
        # 7. FILTRE SUPPORT/RESISTANCE (confluence niveaux)
        sr_score = self._detect_support_resistance_common(df, current_price, signal_side)
        
        # === CALCUL DE CONFIANCE COMPOSITE ===
        scores = {
            'adx': adx_score,
            'volume': volume_score,
            'pullback': pullback_score,
            'trend': trend_score,
            'rsi': rsi_score,
            'sr': sr_score
        }
        
        weights = {
            'adx': 0.25,      # Force de tendance cruciale pour EMA
            'volume': 0.20,   # Confirmation volume
            'pullback': 0.20, # Qualité du pullback
            'trend': 0.15,    # Tendance supérieure
            'rsi': 0.10,      # Momentum
            'sr': 0.10        # Support/résistance
        }
        
        confidence = self._calculate_composite_confidence_common(scores, weights)
        
        # Seuil minimum de confiance
        if confidence < 0.65:
            logger.debug(f"[EMA Cross] {self.symbol}: Signal rejeté - confiance trop faible ({confidence:.2f})")
            return None
        
        # === CONSTRUCTION DU SIGNAL ===
        signal = self.create_signal(
            side=signal_side,
            price=current_price,
            confidence=confidence
        )
        
        # Ajouter les métadonnées d'analyse
        signal.metadata.update({
            'SELL_ema': current_SELL_ema,
            'BUY_ema': current_BUY_ema,
            'ema_spread_pct': abs(current_SELL_ema - current_BUY_ema) / current_BUY_ema * 100,
            'price_to_SELL_ema_pct': (current_price - current_SELL_ema) / current_SELL_ema * 100,
            'price_to_BUY_ema_pct': (current_price - current_BUY_ema) / current_BUY_ema * 100,
            'adx_score': adx_score,
            'volume_score': volume_score,
            'pullback_score': pullback_score,
            'trend_score': trend_score,
            'rsi_score': rsi_score,
            'sr_score': sr_score
        })
        
        logger.info(f"🎯 [EMA Cross] {self.symbol}: Signal {signal_side} @ {current_price:.{precision}f} "
                   f"(confiance: {confidence:.2f}, scores: ADX={adx_score:.2f}, V={volume_score:.2f}, "
                   f"PB={pullback_score:.2f})")
        
        return signal
    
    def _detect_ema_setup(self, SELL_ema: np.ndarray, BUY_ema: np.ndarray, prices: np.ndarray) -> Optional[OrderSide]:
        """
        Détecte le setup EMA de base avec logique sophistiquée.
        """
        if len(SELL_ema) < 3 or len(BUY_ema) < 3:
            return None
        
        current_SELL = SELL_ema[-1]
        current_BUY = BUY_ema[-1]
        current_price = prices[-1]
        
        # Détecter la direction de la tendance
        is_uptrend = current_SELL > current_BUY
        is_downtrend = current_SELL < current_BUY
        
        # Calculer les distances
        price_to_SELL_pct = (current_price - current_SELL) / current_SELL * 100
        price_to_BUY_pct = (current_price - current_BUY) / current_BUY * 100
        
        # Setup BUY: Pullback dans tendance haussière
        if is_uptrend and price_to_SELL_pct < -0.3:  # Prix sous EMA courte
            # Vérifier que ce n'est pas un breakdown
            if current_price > current_BUY * 0.98:  # Pas trop loin de l'EMA BUYue
                return OrderSide.BUY
        
        # Setup SELL: Rebond dans tendance baissière
        elif is_downtrend and price_to_SELL_pct > 0.3:  # Prix au-dessus EMA courte
            # Vérifier que ce n'est pas un breakout
            if current_price < current_BUY * 1.02:  # Pas trop loin de l'EMA BUYue
                return OrderSide.SELL
        
        return None
    
    def _analyze_adx_strength(self, df: pd.DataFrame) -> float:
        """
        Analyse la force de tendance avec ADX.
        """
        try:
            if len(df) < 30:
                return 0.7
            
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Calculer ADX
            adx = talib.ADX(high, low, close, timeperiod=14)
            
            if np.isnan(adx[-1]):
                return 0.7
            
            current_adx = adx[-1]
            
            # Score basé sur la force ADX
            if current_adx > 40:
                return 0.95  # Tendance très forte
            elif current_adx > 30:
                return 0.85  # Tendance forte
            elif current_adx > 20:
                return 0.75  # Tendance modérée
            elif current_adx > 15:
                return 0.65  # Tendance faible
            else:
                return 0.5   # Pas de tendance claire
                
        except Exception as e:
            logger.warning(f"Erreur ADX: {e}")
            return 0.7
    
    def _validate_pullback_quality(self, df: pd.DataFrame, signal_side: OrderSide, 
                                  SELL_ema: float, BUY_ema: float) -> float:
        """
        Valide la qualité du pullback/rebond.
        """
        try:
            if len(df) < 10:
                return 0.7
            
            prices = df['close'].values
            current_price = prices[-1]
            
            # Analyser la qualité du mouvement
            price_volatility = np.std(prices[-10:]) / np.mean(prices[-10:])
            
            if signal_side == OrderSide.BUY:
                # Pour BUY: prix doit être proche de l'EMA BUYue mais pas en breakdown
                distance_to_BUY = abs(current_price - BUY_ema) / BUY_ema
                
                if distance_to_BUY < 0.01:  # Très proche EMA BUYue
                    score = 0.9
                elif distance_to_BUY < 0.02:  # Proche EMA BUYue
                    score = 0.8
                elif current_price > BUY_ema:  # Au-dessus EMA BUYue
                    score = 0.75
                else:  # En dessous EMA BUYue
                    score = 0.6
            
            else:  # SELL
                # Pour SELL: prix doit être proche de l'EMA BUYue mais pas en breakout
                distance_to_BUY = abs(current_price - BUY_ema) / BUY_ema
                
                if distance_to_BUY < 0.01:  # Très proche EMA BUYue
                    score = 0.9
                elif distance_to_BUY < 0.02:  # Proche EMA BUYue
                    score = 0.8
                elif current_price < BUY_ema:  # En dessous EMA BUYue
                    score = 0.75
                else:  # Au-dessus EMA BUYue
                    score = 0.6
            
            # Ajuster selon la volatilité
            if price_volatility > 0.03:  # Haute volatilité
                score *= 0.9
            
            return min(0.95, score)
            
        except Exception as e:
            logger.warning(f"Erreur validation pullback: {e}")
            return 0.7