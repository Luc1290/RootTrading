"""
Stratégie de trading basée sur le croisement de moyennes mobiles exponentielles (EMA).
Génère des signaux lorsque l'EMA courte croise l'EMA longue.
"""
import logging
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
# Importer les modules partagés
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from shared.src.config import get_strategy_param
from shared.src.enums import OrderSide
from shared.src.schemas import StrategySignal
from shared.src.technical_indicators import calculate_ema

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
        self.short_window = self.params.get('short_window', get_strategy_param('ema_cross', 'short_window', 5))
        self.long_window = self.params.get('long_window', get_strategy_param('ema_cross', 'long_window', 20))
        
        # S'assurer que court < long
        if self.short_window >= self.long_window:
            logger.warning(f"⚠️ Configuration incorrecte: EMA court ({self.short_window}) >= EMA long ({self.long_window}). Ajustement automatique.")
            self.short_window = min(self.short_window, self.long_window - 1)

        # Variables pour suivre les tendances
        self.prev_short_ema = None
        self.prev_long_ema = None

        logger.info(f"🔧 Stratégie EMA Cross initialisée pour {symbol} "
                   f"(short={self.short_window}, long={self.long_window})")

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
            Tuple (short_ema, long_ema) des valeurs EMA
        """
        try:
            # Utiliser le module partagé pour calculer les EMAs
            short_ema = calculate_ema(prices, period=self.short_window)
            long_ema = calculate_ema(prices, period=self.long_window)
            return short_ema, long_ema
        except Exception as e:
            logger.error(f"Erreur lors du calcul des EMAs: {str(e)}")
            # Implémenter un calcul manuel de secours en cas d'erreur
            return self._calculate_emas_manually(prices)
    
    def _calculate_emas_manually(self, prices: np.ndarray) -> tuple:
        """
        Calcule les EMAs manuellement si TA-Lib n'est pas disponible.
        
        Args:
            prices: Tableau numpy des prix de clôture
            
        Returns:
            Tuple (short_ema, long_ema) des valeurs EMA
        """
        # Initialiser les tableaux
        short_ema = np.zeros_like(prices)
        long_ema = np.zeros_like(prices)
        
        # Calculer les EMAs
        # Coefficient de lissage
        short_alpha = 2 / (self.short_window + 1)
        long_alpha = 2 / (self.long_window + 1)
        
        # Pour la première valeur, utiliser la moyenne simple (SMA)
        for i in range(len(prices)):
            if i < self.short_window - 1:
                short_ema[i] = np.nan
            elif i == self.short_window - 1:
                short_ema[i] = np.mean(prices[:self.short_window])
            else:
                short_ema[i] = (prices[i] * short_alpha) + (short_ema[i-1] * (1 - short_alpha))
                
            if i < self.long_window - 1:
                long_ema[i] = np.nan
            elif i == self.long_window - 1:
                long_ema[i] = np.mean(prices[:self.long_window])
            else:
                long_ema[i] = (prices[i] * long_alpha) + (long_ema[i-1] * (1 - long_alpha))
        
        # Retourner seulement les dernières valeurs (comme calculate_ema)
        return short_ema[-1], long_ema[-1]
    
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
        short_ema, long_ema = self.calculate_emas(prices)
        
        # Obtenir les dernières valeurs
        current_price = prices[-1]
        # short_ema et long_ema sont déjà des valeurs float (dernière valeur)
        current_short_ema = short_ema
        current_long_ema = long_ema
        
        # Loguer les valeurs actuelles
        precision = 5 if 'BTC' in self.symbol else 3
        logger.debug(f"[EMA Cross] {self.symbol}: Price={current_price:.{precision}f}, "
                    f"Short EMA={current_short_ema:.{precision}f}, Long EMA={current_long_ema:.{precision}f}")
        
        # Vérifications de base
        if np.isnan(current_short_ema) or np.isnan(current_long_ema):
            return None
        
        # === NOUVEAU SYSTÈME DE FILTRES SOPHISTIQUES ===
        
        # 1. FILTRE SETUP EMA DE BASE
        signal_side = self._detect_ema_setup(current_short_ema, current_long_ema, current_price)
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
        pullback_score = self._validate_pullback_quality(df, signal_side, current_short_ema, current_long_ema)
        
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
            'short_ema': current_short_ema,
            'long_ema': current_long_ema,
            'ema_spread_pct': abs(current_short_ema - current_long_ema) / current_long_ema * 100,
            'price_to_short_ema_pct': (current_price - current_short_ema) / current_short_ema * 100,
            'price_to_long_ema_pct': (current_price - current_long_ema) / current_long_ema * 100,
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
    
    def _detect_ema_setup(self, current_short: float, current_long: float, current_price: float) -> Optional[OrderSide]:
        """
        Détecte le setup EMA de base avec logique sophistiquée ET validation de tendance.
        """
        
        # NOUVEAU: Validation de tendance avant de générer le signal
        trend_alignment = self._validate_trend_alignment_for_signal()
        if trend_alignment is None:
            return None  # Pas assez de données pour valider la tendance
        
        # Détecter la direction de la tendance
        is_uptrend = current_short > current_long
        is_downtrend = current_short < current_long
        
        # Calculer les distances
        price_to_short_pct = (current_price - current_short) / current_short * 100
        price_to_long_pct = (current_price - current_long) / current_long * 100
        
        # Setup BUY: Pullback dans tendance haussière ET tendance compatible
        if is_uptrend and price_to_short_pct < -0.3:  # Prix sous EMA courte
            # Vérifier que ce n'est pas un breakdown
            if current_price > current_long * 0.98:  # Pas trop loin de l'EMA longue
                # NOUVEAU: Ne BUY que si tendance n'est pas fortement baissière
                if trend_alignment in ["STRONG_BEARISH", "WEAK_BEARISH"]:
                    logger.debug(f"[EMA Cross] {self.symbol}: BUY signal supprimé - tendance {trend_alignment}")
                    return None
                return OrderSide.BUY
        
        # Setup SELL: Rebond dans tendance baissière ET tendance compatible
        elif is_downtrend and price_to_short_pct > 0.3:  # Prix au-dessus EMA courte
            # Vérifier que ce n'est pas un breakout
            if current_price < current_long * 1.02:  # Pas trop loin de l'EMA longue
                # NOUVEAU: Ne SELL que si tendance n'est pas fortement haussière
                if trend_alignment in ["STRONG_BULLISH", "WEAK_BULLISH"]:
                    logger.debug(f"[EMA Cross] {self.symbol}: SELL signal supprimé - tendance {trend_alignment}")
                    return None
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
            
            # Calculer ADX manuellement (using advanced_filters_mixin method)
            from shared.src.technical_indicators import calculate_adx
            adx_result = calculate_adx(high, low, close, period=14)
            adx_value = adx_result[0] if adx_result[0] is not None else 20.0  # ADX seulement
            
            if adx_value is None or np.isnan(adx_value):
                return 0.7
            
            current_adx = adx_value
            
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
                                  short_ema: float, long_ema: float) -> float:
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
                # Pour BUY: prix doit être proche de l'EMA longue mais pas en breakdown
                distance_to_long = abs(current_price - long_ema) / long_ema
                
                if distance_to_long < 0.01:  # Très proche EMA longue
                    score = 0.9
                elif distance_to_long < 0.02:  # Proche EMA longue
                    score = 0.8
                elif current_price > long_ema:  # Au-dessus EMA longue
                    score = 0.75
                else:  # En dessous EMA longue
                    score = 0.6
            
            else:  # SELL
                # Pour SELL: prix doit être proche de l'EMA longue mais pas en breakout
                distance_to_long = abs(current_price - long_ema) / long_ema
                
                if distance_to_long < 0.01:  # Très proche EMA longue
                    score = 0.9
                elif distance_to_long < 0.02:  # Proche EMA longue
                    score = 0.8
                elif current_price < long_ema:  # En dessous EMA longue
                    score = 0.75
                else:  # Au-dessus EMA longue
                    score = 0.6
            
            # Ajuster selon la volatilité
            if price_volatility > 0.03:  # Haute volatilité
                score *= 0.9
            
            return min(0.95, score)
            
        except Exception as e:
            logger.warning(f"Erreur validation pullback: {e}")
            return 0.7
    
    def _validate_trend_alignment_for_signal(self) -> Optional[str]:
        """
        Valide la tendance actuelle pour déterminer si un signal est approprié.
        Utilise la même logique que le signal_aggregator pour cohérence.
        """
        try:
            df = self.get_data_as_dataframe()
            if df is None or len(df) < 50:
                return None
            
            prices = df['close'].values
            
            # Calculer EMA 21 vs EMA 50 (harmonisé avec signal_aggregator)
            ema_21 = calculate_ema(prices, period=21)
            ema_50 = calculate_ema(prices, period=50)
            
            if ema_21 is None or ema_50 is None or np.isnan(ema_21) or np.isnan(ema_50):
                return None
            
            current_price = prices[-1]
            trend_21 = ema_21
            trend_50 = ema_50
            
            # Classification sophistiquée de la tendance (même logique que signal_aggregator)
            if trend_21 > trend_50 * 1.015:  # +1.5% = forte haussière
                return "STRONG_BULLISH"
            elif trend_21 > trend_50 * 1.005:  # +0.5% = faible haussière
                return "WEAK_BULLISH"
            elif trend_21 < trend_50 * 0.985:  # -1.5% = forte baissière
                return "STRONG_BEARISH"
            elif trend_21 < trend_50 * 0.995:  # -0.5% = faible baissière
                return "WEAK_BEARISH"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            logger.warning(f"Erreur validation tendance: {e}")
            return None