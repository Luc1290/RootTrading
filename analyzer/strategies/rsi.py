"""
Stratégie de trading basée sur l'indicateur RSI (Relative Strength Index).
Génère des signaux d'achat quand le RSI est survendu et des signaux de vente quand il est suracheté.
"""
import logging
from datetime import datetime
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
# CORRECTION: Les fonctions de convenance existent bien dans le module !
# Importer à la fois la classe ET les fonctions nécessaires
from shared.src.technical_indicators import TechnicalIndicators, calculate_rsi, calculate_ema, calculate_macd

from .base_strategy import BaseStrategy

# Configuration du logging
logger = logging.getLogger(__name__)

class RSIStrategy(BaseStrategy):
    """
    Stratégie sophistiquée basée sur l'indicateur RSI (Relative Strength Index).
    Utilise des filtres multi-critères pour éviter les faux signaux et le trading aléatoire.
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        """
        Initialise la stratégie RSI.
        
        Args:
            symbol: Symbole de trading (ex: 'BTCUSDC')
            params: Paramètres spécifiques à la stratégie
        """
        super().__init__(symbol, params)
        
        # Paramètres RSI adaptatifs
        self.rsi_window = self.params.get('window', get_strategy_param('rsi', 'window', 14))
        self.overbought_threshold = self.params.get('overbought', get_strategy_param('rsi', 'overbought', 70))  # Standard
        self.oversold_threshold = self.params.get('oversold', get_strategy_param('rsi', 'oversold', 30))      # Standard
        
        # Niveaux RSI adaptatifs selon la volatilité
        self.extreme_overbought = 85  # Zone extrême
        self.extreme_oversold = 15    # Zone extrême
        
        # Variables pour suivre les tendances
        self.prev_rsi = None
        self.prev_price = None
        
        logger.info(f"🔧 Stratégie RSI initialisée pour {symbol} "
                   f"(window={self.rsi_window}, overbought={self.overbought_threshold}, "
                   f"oversold={self.oversold_threshold})")
    
    @property
    def name(self) -> str:
        """Nom unique de la stratégie."""
        return "RSI_Strategy"
    
    def get_min_data_points(self) -> int:
        """
        Nombre minimum de points de données nécessaires pour calculer le RSI.
        
        Returns:
            Nombre minimum de données requises
        """
        # Besoin d'au moins 2 * la fenêtre RSI pour avoir un calcul fiable
        return max(self.rsi_window * 2, 15)
    
    def calculate_rsi_series(self, prices: np.ndarray) -> np.ndarray:
        """
        Calcule l'indicateur RSI sur une série de prix.
        Utilise le module partagé pour cohérence.
        
        Args:
            prices: Tableau numpy des prix de clôture
            
        Returns:
            Tableau numpy des valeurs RSI
        """
        # Utiliser le module partagé pour calculer le RSI sur toute la série
        from shared.src.technical_indicators import TechnicalIndicators
        ti = TechnicalIndicators()
        
        # Calculer RSI pour toute la série
        rsi_values = []
        for i in range(len(prices)):
            if i < self.rsi_window:
                rsi_values.append(np.nan)
            else:
                rsi = ti.calculate_rsi(prices[:i+1], self.rsi_window)
                rsi_values.append(rsi if rsi is not None else np.nan)
        
        return np.array(rsi_values)
    
    
    def generate_signal(self) -> Optional[StrategySignal]:
        """
        Génère un signal de trading sophistiqué basé sur l'indicateur RSI.
        Utilise des filtres multi-critères pour éviter les faux signaux.
        
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
        highs = df['high'].values
        lows = df['low'].values
        volumes = df['volume'].values if 'volume' in df.columns else None
        
        # Calculer les indicateurs
        rsi_values = self.calculate_rsi_series(prices)
        current_price = prices[-1]
        current_rsi = rsi_values[-1]
        
        # Loguer les valeurs actuelles
        precision = 5 if 'BTC' in self.symbol else 3
        logger.debug(f"[RSI] {self.symbol}: RSI={current_rsi:.2f}, Price={current_price:.{precision}f}")
        
        # Vérifications de base
        if np.isnan(current_rsi):
            return None
        
        # === NOUVEAU SYSTÈME DE FILTRES SOPHISTIQUES ===
        
        # 1. FILTRE SETUP RSI DE BASE
        signal_side = self._detect_rsi_setup(rsi_values, current_rsi)
        if signal_side is None:
            return None
        
        # 2. FILTRE MOMENTUM (confirmation direction)
        momentum_score = self._analyze_momentum_confirmation(df, signal_side)
        if momentum_score < 0.3:
            logger.debug(f"[RSI] {self.symbol}: Signal rejeté - momentum insuffisant ({momentum_score:.2f})")
            return None
        
        # 3. FILTRE VOLUME (confirmation institutionnelle)
        volume_score = self._analyze_volume_confirmation(volumes) if volumes is not None else 0.7
        if volume_score < 0.3:
            logger.debug(f"[RSI] {self.symbol}: Signal rejeté - volume insuffisant ({volume_score:.2f})")
            return None
        
        # 4. FILTRE PRICE ACTION (structure prix/niveaux)
        price_action_score = self._analyze_price_action_context(df, current_price, signal_side)
        
        # 5. FILTRE DIVERGENCE RSI AVANCÉE (confirmation technique)
        divergence_score = self._detect_advanced_rsi_divergence(df, rsi_values, signal_side)
        
        # 6. FILTRE MULTI-TIMEFRAME (tendance supérieure)
        trend_score = self._analyze_higher_timeframe_alignment(signal_side)
        
        # 7. FILTRE RSI OVERBOUGHT/OVERSOLD ADAPTATIF (environnement marché)
        adaptive_score = self._calculate_adaptive_rsi_threshold(rsi_values, current_rsi, signal_side)
        
        # === CALCUL DE CONFIANCE COMPOSITE ===
        confidence = self._calculate_composite_confidence(
            momentum_score, volume_score, price_action_score,
            divergence_score, trend_score, adaptive_score
        )
        
        # Seuil minimum de confiance pour éviter le trading aléatoire
        if confidence < 0.55:
            logger.debug(f"[RSI] {self.symbol}: Signal rejeté - confiance trop faible ({confidence:.2f})")
            return None
        
        # === CONSTRUCTION DU SIGNAL ===
        signal = self.create_signal(
            side=signal_side,
            price=current_price,
            confidence=confidence
        )
        
        # Ajouter les métadonnées d'analyse
        signal.metadata.update({
            'rsi_value': current_rsi,
            'rsi_zone': self._get_rsi_zone(current_rsi),
            'momentum_score': momentum_score,
            'volume_score': volume_score,
            'price_action_score': price_action_score,
            'divergence_score': divergence_score,
            'trend_score': trend_score,
            'adaptive_score': adaptive_score,
            'rsi_trend': self._get_rsi_trend(rsi_values),
            'oversold_threshold': self.oversold_threshold,
            'overbought_threshold': self.overbought_threshold
        })
        
        logger.info(f"🎯 [RSI] {self.symbol}: Signal {signal_side} @ {current_price:.{precision}f} "
                   f"(RSI: {current_rsi:.1f}, confiance: {confidence:.2f}, "
                   f"scores: M={momentum_score:.2f}, V={volume_score:.2f}, PA={price_action_score:.2f})")
        
        return signal
    
    def _detect_rsi_setup(self, rsi_values: np.ndarray, current_rsi: float) -> Optional[OrderSide]:
        """
        Détecte le setup RSI de base avec logique sophistiquée ET validation de tendance.
        """
        if len(rsi_values) < 3:
            return None
        
        prev_rsi = rsi_values[-2]
        
        # NOUVEAU: Validation de tendance avant de générer le signal
        trend_alignment = self._validate_trend_alignment_for_signal()
        if trend_alignment is None:
            return None  # Pas assez de données pour valider la tendance
        
        # Setup BUY: RSI en zone survente avec momentum favorable ET tendance compatible
        if current_rsi <= self.oversold_threshold:
            # Vérifier que ce n'est pas un couteau qui tombe
            if len(rsi_values) >= 5:
                rsi_momentum = current_rsi - rsi_values[-5]
                # Si RSI chute trop vite (>20 points en 5 périodes), attendre stabilisation
                if rsi_momentum < -20:
                    return None
            
            # NOUVEAU: Ne BUY que si tendance n'est pas fortement baissière
            if trend_alignment in ["STRONG_BEARISH", "WEAK_BEARISH"]:
                logger.debug(f"[RSI] {self.symbol}: BUY signal supprimé - tendance {trend_alignment}")
                return None
            
            return OrderSide.BUY
        
        # Setup SELL: RSI en zone surachat avec momentum favorable ET tendance compatible
        elif current_rsi >= self.overbought_threshold:
            # Vérifier que ce n'est pas un breakout en cours
            if len(rsi_values) >= 5:
                rsi_momentum = current_rsi - rsi_values[-5]
                # Si RSI monte trop vite (>20 points en 5 périodes), attendre ralentissement
                if rsi_momentum > 20:
                    return None
            
            # SELL seulement si tendance baissière/neutre ou en forte haussière (rejet)
            if trend_alignment in ["STRONG_BULLISH"]:
                logger.debug(f"[RSI] {self.symbol}: SELL signal supprimé - tendance {trend_alignment} trop haussière")
                return None
            
            return OrderSide.SELL
        
        return None
    
    def _analyze_momentum_confirmation(self, df: pd.DataFrame, signal_side: OrderSide) -> float:
        """
        Analyse la confirmation du momentum pour le signal RSI.
        """
        try:
            prices = df['close'].values
            if len(prices) < 10:
                return 0.7
            
            # Calculer plusieurs indicateurs de momentum
            roc_5 = ((prices[-1] - prices[-6]) / prices[-6] * 100) if len(prices) > 5 else 0
            roc_10 = ((prices[-1] - prices[-11]) / prices[-11] * 100) if len(prices) > 10 else 0
            
            # MACD pour confirmation
            if len(prices) >= 26:
                macd_data = calculate_macd(prices)
                current_macd = macd_data.get('macd_histogram', 0)
                if current_macd is None:
                    current_macd = 0
            else:
                current_macd = 0
            
            if signal_side == OrderSide.BUY:
                # Pour BUY: chercher momentum haussier naissant
                score = 0.5  # Base
                
                if roc_5 > -2:  # Prix pas en chute libre
                    score += 0.2
                if roc_10 > -5:  # Pas de forte baisse sur 10 périodes
                    score += 0.2
                if current_macd > 0:  # MACD positif
                    score += 0.1
                
                return min(0.95, score)
            
            else:  # SELL
                # Pour SELL: chercher momentum baissier naissant
                score = 0.5  # Base
                
                if roc_5 < 2:  # Prix pas en montée folle
                    score += 0.2
                if roc_10 < 5:  # Pas de forte hausse sur 10 périodes
                    score += 0.2
                if current_macd < 0:  # MACD négatif
                    score += 0.1
                
                return min(0.95, score)
                
        except Exception as e:
            logger.warning(f"Erreur analyse momentum: {e}")
            return 0.7
    
    def _analyze_volume_confirmation(self, volumes: Optional[np.ndarray]) -> float:
        """
        Analyse la confirmation par le volume (réutilise la logique Bollinger).
        """
        if volumes is None or len(volumes) < 10:
            return 0.7
        
        current_volume = volumes[-1]
        avg_volume_10 = np.mean(volumes[-10:])
        avg_volume_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else avg_volume_10
        
        volume_ratio_10 = current_volume / avg_volume_10 if avg_volume_10 > 0 else 1.0
        volume_ratio_20 = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
        
        if volume_ratio_10 > 1.3 and volume_ratio_20 > 1.2:
            return 0.9   # Forte expansion volume
        elif volume_ratio_10 > 1.1:
            return 0.8   # Expansion modérée
        elif volume_ratio_10 > 0.8:
            return 0.7   # Volume acceptable
        else:
            return 0.5   # Volume faible
    
    def _analyze_price_action_context(self, df: pd.DataFrame, current_price: float, signal_side: OrderSide) -> float:
        """
        Analyse le contexte price action et les niveaux clés.
        """
        try:
            prices = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            
            if len(prices) < 20:
                return 0.7
            
            # Analyse des niveaux récents
            recent_period = min(30, len(prices))
            recent_highs = highs[-recent_period:]
            recent_lows = lows[-recent_period:]
            
            if signal_side == OrderSide.BUY:
                # Chercher confluence avec support
                recent_lows_sorted = sorted(recent_lows)
                potential_support = recent_lows_sorted[:3]  # 3 plus bas récents
                
                support_levels = []
                for low in potential_support:
                    if abs(current_price - low) / current_price < 0.05:  # Dans les 5%
                        support_levels.append(low)
                
                if len(support_levels) >= 1:
                    return 0.9   # Proche du support
                elif current_price < np.mean(recent_lows) * 1.02:
                    return 0.8   # Dans zone basse récente
                else:
                    return 0.7
            
            else:  # SELL
                # Chercher confluence avec résistance
                recent_highs_sorted = sorted(recent_highs, reverse=True)
                potential_resistance = recent_highs_sorted[:3]  # 3 plus hauts récents
                
                resistance_levels = []
                for high in potential_resistance:
                    if abs(high - current_price) / current_price < 0.05:  # Dans les 5%
                        resistance_levels.append(high)
                
                if len(resistance_levels) >= 1:
                    return 0.9   # Proche de la résistance
                elif current_price > np.mean(recent_highs) * 0.98:
                    return 0.8   # Dans zone haute récente
                else:
                    return 0.7
                    
        except Exception as e:
            logger.warning(f"Erreur analyse price action: {e}")
            return 0.7
    
    def _detect_advanced_rsi_divergence(self, df: pd.DataFrame, rsi_values: np.ndarray, signal_side: OrderSide) -> float:
        """
        Détecte les divergences RSI avancées.
        """
        try:
            if len(rsi_values) < 20:
                return 0.7
            
            prices = df['close'].values
            lookback = min(15, len(prices))
            
            recent_prices = prices[-lookback:]
            recent_rsi = rsi_values[-lookback:]
            
            # Filtrer les NaN
            valid_mask = ~np.isnan(recent_rsi)
            if np.sum(valid_mask) < 10:
                return 0.7
            
            recent_prices = recent_prices[valid_mask]
            recent_rsi = recent_rsi[valid_mask]
            
            if signal_side == OrderSide.BUY:
                # Divergence bullish: prix fait plus bas, RSI fait plus haut
                price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
                rsi_trend = np.polyfit(range(len(recent_rsi)), recent_rsi, 1)[0]
                
                if price_trend < 0 and rsi_trend > 0:  # Prix baisse, RSI monte
                    return 0.9   # Forte divergence bullish
                elif rsi_values[-1] < 35:  # Zone survente profonde
                    return 0.8
                else:
                    return 0.7
            
            else:  # SELL
                # Divergence bearish: prix fait plus haut, RSI fait plus bas
                price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
                rsi_trend = np.polyfit(range(len(recent_rsi)), recent_rsi, 1)[0]
                
                if price_trend > 0 and rsi_trend < 0:  # Prix monte, RSI baisse
                    return 0.9   # Forte divergence bearish
                elif rsi_values[-1] > 65:  # Zone surachat profonde
                    return 0.8
                else:
                    return 0.7
                    
        except Exception as e:
            logger.warning(f"Erreur détection divergence: {e}")
            return 0.7
    
    def _analyze_higher_timeframe_alignment(self, signal_side: OrderSide) -> float:
        """
        Analyse l'alignement avec timeframe supérieur.
        """
        try:
            df = self.get_data_as_dataframe()
            if df is None or len(df) < 50:
                return 0.7
            
            prices = df['close'].values
            
            # EMA 21 vs EMA 50 pour simuler timeframe supérieur
            ema_21_val = calculate_ema(prices, 21)
            ema_50_val = calculate_ema(prices, 50)
            
            if ema_21_val is None or ema_50_val is None:
                return 0.7
            
            current_price = prices[-1]
            trend_21 = ema_21_val
            trend_50 = ema_50_val
            
            if signal_side == OrderSide.BUY:
                if current_price > trend_21 and trend_21 > trend_50:
                    return 0.9   # Tendance alignée
                elif current_price > trend_50:
                    return 0.8   # Tendance modérée
                else:
                    return 0.5   # Contre tendance
            
            else:  # SELL
                if current_price < trend_21 and trend_21 < trend_50:
                    return 0.9   # Tendance alignée
                elif current_price < trend_50:
                    return 0.8   # Tendance modérée
                else:
                    return 0.5   # Contre tendance
                    
        except Exception as e:
            logger.warning(f"Erreur analyse timeframe: {e}")
            return 0.7
    
    def _calculate_adaptive_rsi_threshold(self, rsi_values: np.ndarray, current_rsi: float, signal_side: OrderSide) -> float:
        """
        Calcule des seuils RSI adaptatifs selon l'environnement de marché.
        """
        try:
            if len(rsi_values) < 20:
                return 0.7
            
            # Analyse de la volatilité RSI
            rsi_volatility = np.std(rsi_values[-20:])
            
            if signal_side == OrderSide.BUY:
                # En haute volatilité, accepter des RSI moins extrêmes
                if rsi_volatility > 15:  # RSI très volatile
                    threshold = 35  # Seuil élargi
                else:
                    threshold = self.oversold_threshold
                
                if current_rsi <= threshold:
                    if current_rsi <= self.extreme_oversold:
                        return 0.95  # Zone extrême
                    else:
                        return 0.8   # Zone normale
                else:
                    return 0.6
            
            else:  # SELL
                if rsi_volatility > 15:
                    threshold = 65  # Seuil élargi
                else:
                    threshold = self.overbought_threshold
                
                if current_rsi >= threshold:
                    if current_rsi >= self.extreme_overbought:
                        return 0.95  # Zone extrême
                    else:
                        return 0.8   # Zone normale
                else:
                    return 0.6
                    
        except Exception as e:
            logger.warning(f"Erreur seuils adaptatifs: {e}")
            return 0.7
    
    def _calculate_composite_confidence(self, momentum_score: float, volume_score: float,
                                       price_action_score: float, divergence_score: float,
                                       trend_score: float, adaptive_score: float) -> float:
        """
        Calcule la confiance composite basée sur tous les filtres.
        """
        weights = {
            'momentum': 0.25,      # Momentum crucial pour RSI
            'volume': 0.20,        # Volume important
            'price_action': 0.20,  # Structure de prix
            'divergence': 0.15,    # Divergences RSI
            'trend': 0.10,         # Tendance supérieure
            'adaptive': 0.10       # Seuils adaptatifs
        }
        
        composite = (
            momentum_score * weights['momentum'] +
            volume_score * weights['volume'] +
            price_action_score * weights['price_action'] +
            divergence_score * weights['divergence'] +
            trend_score * weights['trend'] +
            adaptive_score * weights['adaptive']
        )
        
        return max(0.0, min(1.0, composite))
    
    def _get_rsi_zone(self, rsi_value: float) -> str:
        """Retourne la zone RSI actuelle."""
        if rsi_value <= self.extreme_oversold:
            return "extreme_oversold"
        elif rsi_value <= self.oversold_threshold:
            return "oversold"
        elif rsi_value >= self.extreme_overbought:
            return "extreme_overbought"
        elif rsi_value >= self.overbought_threshold:
            return "overbought"
        else:
            return "neutral"
    
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
            ema_21_val = calculate_ema(prices, 21)
            ema_50_val = calculate_ema(prices, 50)
            
            if ema_21_val is None or ema_50_val is None:
                return None
            
            current_price = prices[-1]
            trend_21 = ema_21_val
            trend_50 = ema_50_val
            
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
    
    def _get_rsi_trend(self, rsi_values: np.ndarray) -> str:
        """Retourne la tendance RSI."""
        if len(rsi_values) < 3:
            return "unknown"
        
        recent_trend = rsi_values[-1] - rsi_values[-3]
        if recent_trend > 2:
            return "rising"
        elif recent_trend < -2:
            return "falling"
        else:
            return "flat"