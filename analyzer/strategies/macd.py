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
from analyzer.strategies.advanced_filters_mixin import AdvancedFiltersMixin
from shared.src.enums import OrderSide, SignalStrength
from shared.src.schemas import StrategySignal, MarketData
from shared.src.config import STRATEGY_PARAMS

# Configuration du logging
logger = logging.getLogger(__name__)

class MACDStrategy(BaseStrategy, AdvancedFiltersMixin):
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
        Génère un signal de trading sophistiqué basé sur MACD.
        Utilise des filtres multi-critères pour éviter les faux signaux.
        
        Returns:
            Signal de trading ou None
        """
        try:
            # Vérifier le cooldown avant de générer un signal
            if not self.can_generate_signal():
                return None
                
            # Convertir les données en DataFrame
            df = self.get_data_as_dataframe()
            if df is None or len(df) < self.buffer_size:
                return None
            
            # Extraire les données nécessaires
            prices = df['close']
            volumes = df['volume'] if 'volume' in df.columns else None
            
            # Calculer le MACD
            macd_line, signal_line, histogram = self.calculate_macd(prices)
            
            # Sauvegarder pour analyse
            self.macd_line = macd_line
            self.signal_line = signal_line
            self.histogram = histogram
            
            # Valeurs actuelles
            current_price = prices.iloc[-1]
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            current_histogram = histogram.iloc[-1]
            
            # Loguer les valeurs actuelles
            precision = 5 if 'BTC' in self.symbol else 3
            logger.debug(f"[MACD] {self.symbol}: Price={current_price:.{precision}f}, "
                        f"MACD={current_macd:.6f}, Signal={current_signal:.6f}, Hist={current_histogram:.6f}")
            
            # === NOUVEAU SYSTÈME DE FILTRES SOPHISTIQUES ===
            
            # 1. FILTRE SETUP MACD DE BASE
            signal_side = self._detect_macd_setup(macd_line, signal_line, histogram)
            if signal_side is None:
                return None
            
            # 2. FILTRE HISTOGRAM MOMENTUM (force du signal)
            histogram_score = self._analyze_histogram_momentum(histogram, signal_side)
            if histogram_score < 0.4:
                logger.debug(f"[MACD] {self.symbol}: Signal rejeté - histogram faible ({histogram_score:.2f})")
                return None
            
            # 3. FILTRE VOLUME CONFIRMATION (validation institutionnelle)
            volume_score = self._analyze_volume_confirmation(volumes) if volumes is not None else 0.7
            if volume_score < 0.4:
                logger.debug(f"[MACD] {self.symbol}: Signal rejeté - volume insuffisant ({volume_score:.2f})")
                return None
            
            # 4. FILTRE BREAKOUT VALIDATION (éviter faux breakouts)
            breakout_score = self._validate_macd_breakout(macd_line, signal_line, df)
            
            # 5. FILTRE DIVERGENCE AVANCÉE (confirmation technique)
            divergence_score = self._detect_advanced_divergence(df, macd_line, signal_side)
            
            # 6. FILTRE TREND ALIGNMENT (tendance supérieure)
            trend_score = self._analyze_trend_alignment(df, signal_side)
            
            # 6.5. FILTRE ADX - DÉSACTIVATION EN RANGE (évite pollution logs)
            adx_analysis = self._analyze_adx_trend_strength_common(df, min_adx_threshold=20.0)
            if adx_analysis['disable_trend_strategies']:
                logger.debug(f"[MACD] {self.symbol}: Signal rejeté - ADX trop faible pour tendance "
                           f"(ADX: {adx_analysis['adx_value']:.1f} < 20, {adx_analysis['reason']})")
                return None
            
            adx_score = adx_analysis['confidence_score']
            
            # 7. FILTRE ZERO LINE CONTEXT (position par rapport à ligne zéro)
            zero_line_score = self._analyze_zero_line_context(macd_line, signal_side)
            
            # === CALCUL DE CONFIANCE COMPOSITE ===
            confidence = self._calculate_composite_confidence(
                histogram_score, volume_score, breakout_score,
                divergence_score, trend_score, adx_score, zero_line_score
            )
            
            # Seuil minimum de confiance pour éviter le trading aléatoire
            if confidence < 0.65:
                logger.debug(f"[MACD] {self.symbol}: Signal rejeté - confiance trop faible ({confidence:.2f})")
                return None
            
            # === CONSTRUCTION DU SIGNAL ===
            signal = self.create_signal(
                side=signal_side,
                price=current_price,
                confidence=confidence
            )
            
            # Ajouter les métadonnées d'analyse
            signal.metadata.update({
                'macd_value': current_macd,
                'signal_value': current_signal,
                'histogram_value': current_histogram,
                'macd_above_zero': current_macd > 0,
                'histogram_score': histogram_score,
                'volume_score': volume_score,
                'breakout_score': breakout_score,
                'divergence_score': divergence_score,
                'trend_score': trend_score,
                'adx_score': adx_score,
                'adx_value': adx_analysis['adx_value'],
                'is_trending': adx_analysis['is_trending'],
                'zero_line_score': zero_line_score,
                'macd_trend': self._get_macd_trend(macd_line),
                'histogram_trend': self._get_histogram_trend(histogram)
            })
            
            logger.info(f"🎯 [MACD] {self.symbol}: Signal {signal_side} @ {current_price:.{precision}f} "
                       f"(MACD: {current_macd:.4f}, confiance: {confidence:.2f}, "
                       f"scores: H={histogram_score:.2f}, V={volume_score:.2f}, B={breakout_score:.2f})")
            
            return signal
            
        except Exception as e:
            logger.error(f"❌ Erreur dans l'analyse MACD: {str(e)}")
            return None
    
    def _detect_macd_setup(self, macd_line: pd.Series, signal_line: pd.Series, 
                          histogram: pd.Series) -> Optional[OrderSide]:
        """
        Détecte le setup MACD de base avec logique sophistiquée ET validation de tendance.
        """
        if len(macd_line) < 3 or len(signal_line) < 3:
            return None
        
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        prev_macd = macd_line.iloc[-2]
        prev_signal = signal_line.iloc[-2]
        current_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]
        
        # NOUVEAU: Validation de tendance avant de générer le signal
        trend_alignment = self._validate_trend_alignment_for_signal()
        if trend_alignment is None:
            return None  # Pas assez de données pour valider la tendance
        
        # Setup BUY: MACD croise au-dessus du signal OU histogram devient positif ET tendance compatible
        if ((prev_macd <= prev_signal and current_macd > current_signal) or  # Croisement classique
            (prev_hist <= 0 and current_hist > 0 and current_macd > current_signal)):  # Histogram breakout
            # Vérifier que ce n'est pas un faux signal
            if len(histogram) >= 5:
                hist_momentum = histogram.iloc[-1] - histogram.iloc[-5]
                if hist_momentum > 0:  # Histogram s'améliore
                    # NOUVEAU: Ne BUY que si tendance n'est pas fortement baissière
                    if trend_alignment in ["STRONG_BEARISH", "WEAK_BEARISH"]:
                        logger.debug(f"[MACD] {self.symbol}: BUY signal supprimé - tendance {trend_alignment}")
                        return None
                    return OrderSide.BUY
        
        # Setup SELL: MACD croise en dessous du signal OU histogram devient négatif ET tendance compatible
        elif ((prev_macd >= prev_signal and current_macd < current_signal) or  # Croisement classique
              (prev_hist >= 0 and current_hist < 0 and current_macd < current_signal)):  # Histogram breakdown
            # Vérifier que ce n'est pas un faux signal
            if len(histogram) >= 5:
                hist_momentum = histogram.iloc[-1] - histogram.iloc[-5]
                if hist_momentum < 0:  # Histogram se détériore
                    # NOUVEAU: Ne SELL que si tendance n'est pas fortement haussière
                    if trend_alignment in ["STRONG_BULLISH", "WEAK_BULLISH"]:
                        logger.debug(f"[MACD] {self.symbol}: SELL signal supprimé - tendance {trend_alignment}")
                        return None
                    return OrderSide.SELL
        
        return None
    
    def _analyze_histogram_momentum(self, histogram: pd.Series, signal_side: OrderSide) -> float:
        """
        Analyse le momentum de l'histogram MACD.
        """
        try:
            if len(histogram) < 10:
                return 0.7
            
            current_hist = histogram.iloc[-1]
            prev_hist = histogram.iloc[-2]
            hist_change = current_hist - prev_hist
            
            # Momentum sur 5 périodes
            hist_momentum_5 = histogram.iloc[-1] - histogram.iloc[-6] if len(histogram) > 5 else 0
            
            if signal_side == OrderSide.BUY:
                score = 0.5  # Base
                
                if current_hist > 0:  # Histogram positif
                    score += 0.2
                if hist_change > 0:  # Histogram s'améliore
                    score += 0.2
                if hist_momentum_5 > 0:  # Momentum positif sur 5 périodes
                    score += 0.1
                
                return min(0.95, score)
            
            else:  # SELL
                score = 0.5  # Base
                
                if current_hist < 0:  # Histogram négatif
                    score += 0.2
                if hist_change < 0:  # Histogram se détériore
                    score += 0.2
                if hist_momentum_5 < 0:  # Momentum négatif sur 5 périodes
                    score += 0.1
                
                return min(0.95, score)
                
        except Exception as e:
            logger.warning(f"Erreur analyse histogram: {e}")
            return 0.7
    
    def _analyze_volume_confirmation(self, volumes: Optional[pd.Series]) -> float:
        """
        Analyse la confirmation par le volume.
        """
        if volumes is None or len(volumes) < 10:
            return 0.7
        
        current_volume = volumes.iloc[-1]
        avg_volume_10 = volumes.iloc[-10:].mean()
        avg_volume_20 = volumes.iloc[-20:].mean() if len(volumes) >= 20 else avg_volume_10
        
        volume_ratio_10 = current_volume / avg_volume_10 if avg_volume_10 > 0 else 1.0
        volume_ratio_20 = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
        
        if volume_ratio_10 > 1.4 and volume_ratio_20 > 1.3:
            return 0.9   # Très forte expansion
        elif volume_ratio_10 > 1.2:
            return 0.8   # Expansion modérée
        elif volume_ratio_10 > 0.8:
            return 0.7   # Volume acceptable
        else:
            return 0.5   # Volume faible
    
    def _validate_macd_breakout(self, macd_line: pd.Series, signal_line: pd.Series, df: pd.DataFrame) -> float:
        """
        Valide que le breakout MACD n'est pas un faux signal.
        """
        try:
            if len(macd_line) < 20:
                return 0.7
            
            # Analyser la consolidation précédente
            recent_macd = macd_line.iloc[-10:]
            recent_signal = signal_line.iloc[-10:]
            
            # Mesurer la volatilité récente du MACD
            macd_volatility = recent_macd.std()
            signal_volatility = recent_signal.std()
            
            # Un bon breakout suit une période de consolidation
            consolidation_score = 0.5
            
            if macd_volatility < macd_line.iloc[-20:].std() * 0.8:
                consolidation_score += 0.2  # MACD était consolidé
            
            if signal_volatility < signal_line.iloc[-20:].std() * 0.8:
                consolidation_score += 0.2  # Signal était consolidé
            
            # Vérifier l'amplitude du breakout
            macd_distance = abs(macd_line.iloc[-1] - signal_line.iloc[-1])
            avg_distance = abs(recent_macd - recent_signal).mean()
            
            if macd_distance > avg_distance * 1.5:
                consolidation_score += 0.1  # Breakout significatif
            
            return min(0.95, consolidation_score)
            
        except Exception as e:
            logger.warning(f"Erreur validation breakout: {e}")
            return 0.7
    
    def _detect_advanced_divergence(self, df: pd.DataFrame, macd_line: pd.Series, signal_side: OrderSide) -> float:
        """
        Détecte les divergences MACD avancées.
        """
        try:
            if len(df) < 30:
                return 0.7
            
            prices = df['close']
            lookback = min(20, len(prices))
            
            recent_prices = prices.iloc[-lookback:]
            recent_macd = macd_line.iloc[-lookback:]
            
            if signal_side == OrderSide.BUY:
                # Divergence bullish: prix fait plus bas, MACD fait plus haut
                price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
                macd_trend = np.polyfit(range(len(recent_macd)), recent_macd, 1)[0]
                
                if price_trend < 0 and macd_trend > 0:
                    return 0.9   # Forte divergence bullish
                elif macd_line.iloc[-1] > macd_line.iloc[-10:].mean():
                    return 0.8   # MACD au-dessus de sa moyenne
                else:
                    return 0.7
            
            else:  # SELL
                # Divergence bearish: prix fait plus haut, MACD fait plus bas
                price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
                macd_trend = np.polyfit(range(len(recent_macd)), recent_macd, 1)[0]
                
                if price_trend > 0 and macd_trend < 0:
                    return 0.9   # Forte divergence bearish
                elif macd_line.iloc[-1] < macd_line.iloc[-10:].mean():
                    return 0.8   # MACD en dessous de sa moyenne
                else:
                    return 0.7
                    
        except Exception as e:
            logger.warning(f"Erreur détection divergence: {e}")
            return 0.7
    
    def _analyze_trend_alignment(self, df: pd.DataFrame, signal_side: OrderSide) -> float:
        """
        Analyse l'alignement avec la tendance générale.
        """
        try:
            if len(df) < 50:
                return 0.7
            
            prices = df['close']
            
            # HARMONISATION: EMA 21 vs EMA 50 pour cohérence avec les autres stratégies
            ema_21 = prices.ewm(span=21).mean()
            ema_50 = prices.ewm(span=50).mean()
            
            current_price = prices.iloc[-1]
            current_ema_21 = ema_21.iloc[-1]
            current_ema_50 = ema_50.iloc[-1]
            
            if signal_side == OrderSide.BUY:
                if current_price > current_ema_21 and current_ema_21 > current_ema_50:
                    return 0.9   # Forte tendance haussière
                elif current_price > current_ema_50:
                    return 0.8   # Tendance modérée
                else:
                    return 0.5   # Contre tendance
            
            else:  # SELL
                if current_price < current_ema_21 and current_ema_21 < current_ema_50:
                    return 0.9   # Forte tendance baissière
                elif current_price < current_ema_50:
                    return 0.8   # Tendance modérée
                else:
                    return 0.5   # Contre tendance
                    
        except Exception as e:
            logger.warning(f"Erreur analyse tendance: {e}")
            return 0.7
    
    def _analyze_zero_line_context(self, macd_line: pd.Series, signal_side: OrderSide) -> float:
        """
        Analyse le contexte par rapport à la ligne zéro MACD.
        """
        try:
            current_macd = macd_line.iloc[-1]
            
            if signal_side == OrderSide.BUY:
                if current_macd > 0:
                    return 0.9   # MACD déjà positif = momentum haussier confirmé
                elif current_macd > -0.001:  # Proche de zéro
                    return 0.8   # Sur le point de devenir positif
                else:
                    return 0.7   # Encore négatif
            
            else:  # SELL
                if current_macd < 0:
                    return 0.9   # MACD déjà négatif = momentum baissier confirmé
                elif current_macd < 0.001:  # Proche de zéro
                    return 0.8   # Sur le point de devenir négatif
                else:
                    return 0.7   # Encore positif
                    
        except Exception as e:
            logger.warning(f"Erreur analyse zero line: {e}")
            return 0.7
    
    def _calculate_composite_confidence(self, histogram_score: float, volume_score: float,
                                       breakout_score: float, divergence_score: float,
                                       trend_score: float, zero_line_score: float) -> float:
        """
        Calcule la confiance composite basée sur tous les filtres.
        """
        weights = {
            'histogram': 0.25,     # Momentum MACD crucial
            'volume': 0.20,        # Volume important
            'breakout': 0.20,      # Validation breakout
            'divergence': 0.15,    # Divergences
            'trend': 0.10,         # Tendance générale
            'zero_line': 0.10      # Position zero line
        }
        
        composite = (
            histogram_score * weights['histogram'] +
            volume_score * weights['volume'] +
            breakout_score * weights['breakout'] +
            divergence_score * weights['divergence'] +
            trend_score * weights['trend'] +
            zero_line_score * weights['zero_line']
        )
        
        return max(0.0, min(1.0, composite))
    
    def _get_macd_trend(self, macd_line: pd.Series) -> str:
        """Retourne la tendance MACD."""
        if len(macd_line) < 3:
            return "unknown"
        
        recent_trend = macd_line.iloc[-1] - macd_line.iloc[-3]
        if recent_trend > 0.0001:
            return "rising"
        elif recent_trend < -0.0001:
            return "falling"
        else:
            return "flat"
    
    def _validate_trend_alignment_for_signal(self) -> Optional[str]:
        """
        Valide la tendance actuelle pour déterminer si un signal est approprié.
        Utilise la même logique que le signal_aggregator pour cohérence.
        """
        try:
            df = self.get_data_as_dataframe()
            if df is None or len(df) < 50:
                return None
            
            prices = df['close']
            
            # Calculer EMA 21 vs EMA 50 (harmonisé avec signal_aggregator)
            ema_21 = prices.ewm(span=21).mean()
            ema_50 = prices.ewm(span=50).mean()
            
            current_price = prices.iloc[-1]
            trend_21 = ema_21.iloc[-1]
            trend_50 = ema_50.iloc[-1]
            
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
    
    def _get_histogram_trend(self, histogram: pd.Series) -> str:
        """Retourne la tendance de l'histogram."""
        if len(histogram) < 3:
            return "unknown"
        
        recent_trend = histogram.iloc[-1] - histogram.iloc[-3]
        if recent_trend > 0.0001:
            return "improving"
        elif recent_trend < -0.0001:
            return "deteriorating"
        else:
            return "stable"
    
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