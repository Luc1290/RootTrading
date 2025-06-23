"""
Stratégie de trading basée sur les bandes de Bollinger.
Génère des signaux lorsque le prix traverse les bandes de Bollinger.
"""
import logging
from datetime import datetime
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

# Configuration du logging
logger = logging.getLogger(__name__)

class BollingerStrategy(BaseStrategy):
    """
    Stratégie basée sur les bandes de Bollinger.
    Génère des signaux d'achat quand le prix traverse la bande inférieure vers le haut
    et des signaux de vente quand il traverse la bande supérieure vers le bas.
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        """
        Initialise la stratégie Bollinger Bands.
        
        Args:
            symbol: Symbole de trading (ex: 'BTCUSDC')
            params: Paramètres spécifiques à la stratégie
        """
        super().__init__(symbol, params)
        
        # Paramètres Bollinger
        self.window = self.params.get('window', get_strategy_param('bollinger', 'window', 20))
        self.num_std = self.params.get('num_std', get_strategy_param('bollinger', 'num_std', 2.0))
        
        # Variables pour suivre les tendances
        self.prev_price = None
        self.prev_upper = None
        self.prev_lower = None
        
        logger.info(f"🔧 Stratégie Bollinger initialisée pour {symbol} "
                   f"(window={self.window}, num_std={self.num_std})")
    
    @property
    def name(self) -> str:
        """Nom unique de la stratégie."""
        return "Bollinger_Strategy"
    
    def get_min_data_points(self) -> int:
        """
        Nombre minimum de points de données nécessaires pour calculer les bandes de Bollinger.
        
        Returns:
            Nombre minimum de données requises
        """
        # Besoin d'au moins 2 * la fenêtre Bollinger pour avoir un calcul fiable
        return max(self.window * 2, 30)
    
    def calculate_bollinger_bands(self, prices: np.ndarray) -> tuple:
        """
        Calcule les bandes de Bollinger sur une série de prix.
        
        Args:
            prices: Tableau numpy des prix de clôture
            
        Returns:
            Tuple (upper, middle, lower) des bandes de Bollinger
        """
        # Utiliser TA-Lib pour calculer les bandes de Bollinger
        try:
            upper, middle, lower = talib.BBANDS(
                prices, 
                timeperiod=self.window,
                nbdevup=self.num_std,
                nbdevdn=self.num_std,
                matype=0  # Simple Moving Average
            )
            return upper, middle, lower
        except Exception as e:
            logger.error(f"Erreur lors du calcul des bandes de Bollinger: {str(e)}")
            # Implémenter un calcul manuel de secours en cas d'erreur TA-Lib
            return self._calculate_bollinger_manually(prices)
    
    def _calculate_bollinger_manually(self, prices: np.ndarray) -> tuple:
        """
        Calcule les bandes de Bollinger manuellement si TA-Lib n'est pas disponible.
        
        Args:
            prices: Tableau numpy des prix de clôture
            
        Returns:
            Tuple (upper, middle, lower) des bandes de Bollinger
        """
        # Initialiser les tableaux
        middle = np.zeros_like(prices)
        upper = np.zeros_like(prices)
        lower = np.zeros_like(prices)
        
        # Calculer la moyenne mobile simple (SMA)
        for i in range(len(prices)):
            if i < self.window - 1:
                # Pas assez de données
                middle[i] = np.nan
                upper[i] = np.nan
                lower[i] = np.nan
            else:
                # Calcul de la SMA
                segment = prices[i-(self.window-1):i+1]
                middle[i] = np.mean(segment)
                
                # Calcul de l'écart-type
                std = np.std(segment)
                
                # Calcul des bandes
                upper[i] = middle[i] + (self.num_std * std)
                lower[i] = middle[i] - (self.num_std * std)
        
        return upper, middle, lower
    
    def generate_signal(self) -> Optional[StrategySignal]:
        """
        Génère un signal de trading sophistiqué basé sur les bandes de Bollinger.
        Utilise des filtres multi-critères pour éviter le trading aléatoire.
        
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
        
        # Calculer les bandes de Bollinger
        upper, middle, lower = self.calculate_bollinger_bands(prices)
        
        # Obtenir les dernières valeurs
        current_price = prices[-1]
        current_upper = upper[-1]
        current_lower = lower[-1]
        current_middle = middle[-1]
        
        # Loguer les valeurs actuelles
        precision = 5 if 'BTC' in self.symbol else 3
        logger.debug(f"[Bollinger] {self.symbol}: Price={current_price:.{precision}f}, "
                    f"Upper={current_upper:.{precision}f}, Lower={current_lower:.{precision}f}")
        
        # Vérifications de base
        if np.isnan(current_upper) or np.isnan(current_lower):
            return None
        
        # === NOUVEAU SYSTÈME DE FILTRES SOPHISTIQUES ===
        
        # 1. FILTRE SETUP DE BASE BOLLINGER
        signal_side = self._detect_bollinger_setup(prices, upper, middle, lower)
        if signal_side is None:
            return None
        
        # 2. FILTRE VOLUME (confirmation institutionnelle)
        volume_score = self._analyze_volume_confirmation(volumes) if volumes is not None else 0.7
        if volume_score < 0.4:
            logger.debug(f"[Bollinger] {self.symbol}: Signal rejeté - volume insuffisant ({volume_score:.2f})")
            return None
        
        # 3. FILTRE DIVERGENCE RSI (momentum sous-jacent)
        rsi_divergence_score = self._detect_rsi_divergence(df, signal_side)
        
        # 4. FILTRE SUPPORT/RESISTANCE (structure de prix)
        sr_score = self._analyze_support_resistance_confluence(df, current_price, signal_side)
        
        # 5. FILTRE MULTI-TIMEFRAME (tendance supérieure)
        trend_score = self._analyze_higher_timeframe_trend(signal_side)
        
        # 6. FILTRE VOLATILITÉ (environnement de marché)
        volatility_score = self._analyze_volatility_environment(upper, middle, lower)
        
        # 7. FILTRE SQUEEZE DETECTION (éviter les faux signaux en range)
        squeeze_score = self._detect_bollinger_squeeze(upper, middle, lower)
        
        # === CALCUL DE CONFIANCE COMPOSITE ===
        confidence = self._calculate_composite_confidence(
            volume_score, rsi_divergence_score, sr_score, 
            trend_score, volatility_score, squeeze_score
        )
        
        # Seuil minimum de confiance pour éviter le trading aléatoire
        if confidence < 0.65:
            logger.debug(f"[Bollinger] {self.symbol}: Signal rejeté - confiance trop faible ({confidence:.2f})")
            return None
        
        # === CONSTRUCTION DU SIGNAL ===
        signal = self.create_signal(
            side=signal_side,
            price=current_price,
            confidence=confidence
        )
        
        # Ajouter les métadonnées d'analyse
        signal.metadata.update({
            'bollinger_position': (current_price - current_lower) / (current_upper - current_lower),
            'volume_score': volume_score,
            'rsi_divergence_score': rsi_divergence_score,
            'sr_score': sr_score,
            'trend_score': trend_score,
            'volatility_score': volatility_score,
            'squeeze_score': squeeze_score,
            'band_width_pct': ((current_upper - current_lower) / current_middle) * 100,
            'price_distance_from_band': self._calculate_band_distance(current_price, upper, lower, signal_side)
        })
        
        logger.info(f"🎯 [Bollinger] {self.symbol}: Signal {signal_side} @ {current_price:.{precision}f} "
                   f"(confiance: {confidence:.2f}, scores: V={volume_score:.2f}, RSI={rsi_divergence_score:.2f}, "
                   f"SR={sr_score:.2f}, Trend={trend_score:.2f})")
        
        return signal
    
    def _detect_bollinger_setup(self, prices: np.ndarray, upper: np.ndarray, 
                               middle: np.ndarray, lower: np.ndarray) -> Optional[OrderSide]:
        """
        Détecte le setup de base Bollinger avec logique sophistiquée ET validation de tendance.
        """
        current_price = prices[-1]
        prev_price = prices[-2] if len(prices) > 1 else current_price
        current_upper = upper[-1]
        current_lower = lower[-1]
        current_middle = middle[-1]
        
        # NOUVEAU: Validation de tendance avant de générer le signal
        trend_alignment = self._validate_trend_alignment_for_signal()
        if trend_alignment is None:
            return None  # Pas assez de données pour valider la tendance
        
        # Setup BUY: Prix près/sous la bande basse avec rebond potentiel ET tendance compatible
        if current_price <= current_lower * 1.003:  # Marge de 0.3%
            # Vérifier si ce n'est pas un couteau qui tombe
            recent_prices = prices[-5:] if len(prices) >= 5 else prices
            if len(recent_prices) >= 3:
                # Le prix doit montrer des signes de stabilisation
                price_momentum = (recent_prices[-1] - recent_prices[-3]) / recent_prices[-3]
                if price_momentum > -0.02:  # Pas de chute > 2% sur 3 périodes
                    # NOUVEAU: Ne BUY que si tendance n'est pas fortement baissière
                    if trend_alignment in ["STRONG_BEARISH", "WEAK_BEARISH"]:
                        logger.debug(f"[Bollinger] {self.symbol}: BUY signal supprimé - tendance {trend_alignment}")
                        return None
                    return OrderSide.BUY
        
        # Setup SELL: Prix près/au-dessus de la bande haute avec rejet potentiel ET tendance compatible
        elif current_price >= current_upper * 0.997:  # Marge de 0.3%
            # Vérifier les signes de rejet
            recent_highs = [prices[i] for i in range(max(0, len(prices)-3), len(prices))]
            if len(recent_highs) >= 2:
                # Prix doit montrer des signes de plafonnement
                if max(recent_highs) - current_price < current_price * 0.005:  # Dans les 0.5% du plus haut
                    # NOUVEAU: Ne SELL que si tendance n'est pas fortement haussière
                    if trend_alignment in ["STRONG_BULLISH", "WEAK_BULLISH"]:
                        logger.debug(f"[Bollinger] {self.symbol}: SELL signal supprimé - tendance {trend_alignment}")
                        return None
                    return OrderSide.SELL
        
        return None
    
    def _analyze_volume_confirmation(self, volumes: Optional[np.ndarray]) -> float:
        """
        Analyse la confirmation par le volume.
        """
        if volumes is None or len(volumes) < 10:
            return 0.7  # Score neutre si pas de données volume
        
        current_volume = volumes[-1]
        avg_volume_10 = np.mean(volumes[-10:])
        avg_volume_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else avg_volume_10
        
        # Volume relatif par rapport aux moyennes
        volume_ratio_10 = current_volume / avg_volume_10 if avg_volume_10 > 0 else 1.0
        volume_ratio_20 = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
        
        # Score basé sur l'expansion du volume
        if volume_ratio_10 > 1.5 and volume_ratio_20 > 1.3:
            return 0.95  # Très forte expansion
        elif volume_ratio_10 > 1.2 and volume_ratio_20 > 1.1:
            return 0.85  # Bonne expansion
        elif volume_ratio_10 > 0.8:
            return 0.75  # Volume acceptable
        else:
            return 0.5   # Volume faible
    
    def _detect_rsi_divergence(self, df: pd.DataFrame, signal_side: OrderSide) -> float:
        """
        Détecte les divergences RSI pour confirmation du signal.
        """
        try:
            prices = df['close'].values
            if len(prices) < 30:
                return 0.7  # Score neutre si pas assez de données
            
            # Calculer RSI
            rsi = talib.RSI(prices, timeperiod=14)
            if np.all(np.isnan(rsi[-10:])):
                return 0.7
            
            current_rsi = rsi[-1]
            
            # Analyser les 20 dernières périodes pour les divergences
            lookback = min(20, len(prices))
            recent_prices = prices[-lookback:]
            recent_rsi = rsi[-lookback:]
            
            # Filtrer les NaN
            valid_mask = ~np.isnan(recent_rsi)
            if np.sum(valid_mask) < 10:
                return 0.7
            
            recent_prices = recent_prices[valid_mask]
            recent_rsi = recent_rsi[valid_mask]
            
            if signal_side == OrderSide.BUY:
                # Chercher divergence bullish: prix fait des plus bas, RSI des plus hauts
                price_min_idx = np.argmin(recent_prices[-10:])
                rsi_in_price_min_zone = recent_rsi[-10:][price_min_idx]
                
                # RSI actuel vs RSI au minimum de prix
                if current_rsi > rsi_in_price_min_zone + 5:  # RSI a monté de 5+ points
                    return 0.9  # Forte divergence bullish
                elif current_rsi > rsi_in_price_min_zone:
                    return 0.8  # Légère divergence bullish
                elif current_rsi < 35:  # Zone survente sans divergence
                    return 0.75
                else:
                    return 0.6
            
            else:  # SELL
                # Chercher divergence bearish: prix fait des plus hauts, RSI des plus bas
                price_max_idx = np.argmax(recent_prices[-10:])
                rsi_in_price_max_zone = recent_rsi[-10:][price_max_idx]
                
                if current_rsi < rsi_in_price_max_zone - 5:  # RSI a chuté de 5+ points
                    return 0.9  # Forte divergence bearish
                elif current_rsi < rsi_in_price_max_zone:
                    return 0.8  # Légère divergence bearish
                elif current_rsi > 65:  # Zone surachat sans divergence
                    return 0.75
                else:
                    return 0.6
                    
        except Exception as e:
            logger.warning(f"Erreur calcul divergence RSI: {e}")
            return 0.7
    
    def _analyze_support_resistance_confluence(self, df: pd.DataFrame, 
                                              current_price: float, signal_side: OrderSide) -> float:
        """
        Analyse la confluence avec les niveaux de support/résistance.
        """
        try:
            prices = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            
            if len(prices) < 50:
                return 0.7  # Score neutre si pas assez de données
            
            # Chercher les pivots sur les 50 dernières périodes
            lookback = min(50, len(prices))
            recent_highs = highs[-lookback:]
            recent_lows = lows[-lookback:]
            
            # Détecter les niveaux pivots
            pivot_highs = []
            pivot_lows = []
            
            for i in range(2, len(recent_highs) - 2):
                # Pivot haut
                if (recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i-2] and
                    recent_highs[i] > recent_highs[i+1] and recent_highs[i] > recent_highs[i+2]):
                    pivot_highs.append(recent_highs[i])
                
                # Pivot bas
                if (recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i-2] and
                    recent_lows[i] < recent_lows[i+1] and recent_lows[i] < recent_lows[i+2]):
                    pivot_lows.append(recent_lows[i])
            
            if signal_side == OrderSide.BUY:
                # Chercher confluence avec support
                if not pivot_lows:
                    return 0.6
                
                # Trouver le support le plus proche
                supports_below = [s for s in pivot_lows if s <= current_price * 1.02]
                if not supports_below:
                    return 0.5  # Pas de support proche
                
                nearest_support = max(supports_below)
                distance_pct = abs(current_price - nearest_support) / current_price * 100
                
                if distance_pct < 0.5:
                    return 0.95  # Très proche du support
                elif distance_pct < 1.0:
                    return 0.85  # Proche du support
                elif distance_pct < 2.0:
                    return 0.75  # Support raisonnable
                else:
                    return 0.6   # Support lointain
            
            else:  # SELL
                # Chercher confluence avec résistance
                if not pivot_highs:
                    return 0.6
                
                resistances_above = [r for r in pivot_highs if r >= current_price * 0.98]
                if not resistances_above:
                    return 0.5  # Pas de résistance proche
                
                nearest_resistance = min(resistances_above)
                distance_pct = abs(nearest_resistance - current_price) / current_price * 100
                
                if distance_pct < 0.5:
                    return 0.95  # Très proche de la résistance
                elif distance_pct < 1.0:
                    return 0.85  # Proche de la résistance
                elif distance_pct < 2.0:
                    return 0.75  # Résistance raisonnable
                else:
                    return 0.6   # Résistance lointaine
                    
        except Exception as e:
            logger.warning(f"Erreur analyse S/R: {e}")
            return 0.7
    
    def _analyze_higher_timeframe_trend(self, signal_side: OrderSide) -> float:
        """
        Analyse la tendance du timeframe supérieur avec la nouvelle logique harmonisée.
        """
        try:
            # Utiliser la méthode harmonisée de validation de tendance
            trend_alignment = self._validate_trend_alignment_for_signal()
            if trend_alignment is None:
                return 0.7  # Score neutre si pas assez de données
            
            # Score selon l'alignement avec la tendance
            if signal_side == OrderSide.BUY:
                if trend_alignment == "STRONG_BULLISH":
                    return 0.95  # Forte tendance alignée
                elif trend_alignment == "WEAK_BULLISH":
                    return 0.85  # Tendance alignée
                elif trend_alignment == "NEUTRAL":
                    return 0.75  # Neutre
                elif trend_alignment == "WEAK_BEARISH":
                    return 0.4   # Contre tendance faible
                else:  # STRONG_BEARISH
                    return 0.2   # Forte contre tendance
            else:  # SELL
                if trend_alignment == "STRONG_BEARISH":
                    return 0.95  # Forte tendance alignée
                elif trend_alignment == "WEAK_BEARISH":
                    return 0.85  # Tendance alignée
                elif trend_alignment == "NEUTRAL":
                    return 0.75  # Neutre
                elif trend_alignment == "WEAK_BULLISH":
                    return 0.4   # Contre tendance faible
                else:  # STRONG_BULLISH
                    return 0.2   # Forte contre tendance
                    
        except Exception as e:
            logger.warning(f"Erreur analyse tendance supérieure: {e}")
            return 0.7
    
    def _analyze_volatility_environment(self, upper: np.ndarray, 
                                       middle: np.ndarray, lower: np.ndarray) -> float:
        """
        Analyse l'environnement de volatilité pour ajuster la confiance.
        """
        try:
            # Calculer la largeur des bandes sur les 20 dernières périodes
            lookback = min(20, len(upper))
            recent_widths = []
            
            for i in range(len(upper) - lookback, len(upper)):
                if not (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i])):
                    width_pct = (upper[i] - lower[i]) / middle[i] * 100
                    recent_widths.append(width_pct)
            
            if len(recent_widths) < 5:
                return 0.7
            
            current_width = recent_widths[-1]
            avg_width = np.mean(recent_widths[:-1])
            
            # Score basé sur l'expansion/contraction
            if current_width > avg_width * 1.3:
                return 0.9   # Volatilité en expansion = bon pour breakouts
            elif current_width > avg_width * 1.1:
                return 0.8   # Légère expansion
            elif current_width > avg_width * 0.7:
                return 0.75  # Volatilité normale
            else:
                return 0.5   # Volatilité trop faible = range
                
        except Exception as e:
            logger.warning(f"Erreur analyse volatilité: {e}")
            return 0.7
    
    def _detect_bollinger_squeeze(self, upper: np.ndarray, 
                                 middle: np.ndarray, lower: np.ndarray) -> float:
        """
        Détecte le Bollinger Squeeze (préparation de breakout).
        """
        try:
            if len(upper) < 20:
                return 0.7
            
            # Calculer les largeurs sur 20 périodes
            widths = []
            for i in range(len(upper) - 20, len(upper)):
                if not (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i])):
                    width = (upper[i] - lower[i]) / middle[i]
                    widths.append(width)
            
            if len(widths) < 10:
                return 0.7
            
            current_width = widths[-1]
            min_width_period = min(widths)
            
            # Squeeze détecté si largeur actuelle proche du minimum
            squeeze_ratio = current_width / min_width_period if min_width_period > 0 else 1
            
            if squeeze_ratio < 1.1:
                return 0.95  # Squeeze fort = prêt pour breakout
            elif squeeze_ratio < 1.3:
                return 0.85  # Squeeze modéré
            else:
                return 0.7   # Pas de squeeze particulier
                
        except Exception as e:
            logger.warning(f"Erreur détection squeeze: {e}")
            return 0.7
    
    def _calculate_composite_confidence(self, volume_score: float, rsi_score: float,
                                       sr_score: float, trend_score: float,
                                       volatility_score: float, squeeze_score: float) -> float:
        """
        Calcule la confiance composite basée sur tous les filtres.
        """
        # Pondération des différents facteurs
        weights = {
            'volume': 0.25,      # Volume très important
            'rsi': 0.20,         # Divergences cruciales
            'sr': 0.20,          # Structure de prix importante
            'trend': 0.15,       # Tendance supérieure
            'volatility': 0.10,  # Environnement de marché
            'squeeze': 0.10      # Préparation breakout
        }
        
        composite = (
            volume_score * weights['volume'] +
            rsi_score * weights['rsi'] +
            sr_score * weights['sr'] +
            trend_score * weights['trend'] +
            volatility_score * weights['volatility'] +
            squeeze_score * weights['squeeze']
        )
        
        # Normaliser entre 0 et 1
        return max(0.0, min(1.0, composite))
    
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
            ema_21 = talib.EMA(prices, timeperiod=21)
            ema_50 = talib.EMA(prices, timeperiod=50)
            
            if np.isnan(ema_21[-1]) or np.isnan(ema_50[-1]):
                return None
            
            current_price = prices[-1]
            trend_21 = ema_21[-1]
            trend_50 = ema_50[-1]
            
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
    
    def _calculate_band_distance(self, price: float, upper: np.ndarray, 
                                lower: np.ndarray, side: OrderSide) -> float:
        """
        Calcule la distance du prix par rapport à la bande pertinente.
        """
        if side == OrderSide.BUY:
            return ((lower[-1] - price) / price * 100) if price > 0 else 0
        else:
            return ((price - upper[-1]) / price * 100) if price > 0 else 0