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
        Génère un signal de trading basé sur les bandes de Bollinger.
        
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
        
        # Extraire les prix de clôture
        prices = df['close'].values
        
        # Calculer les bandes de Bollinger
        upper, middle, lower = self.calculate_bollinger_bands(prices)
        
        # Obtenir les dernières valeurs
        current_price = prices[-1]
        prev_price = prices[-2] if len(prices) > 1 else None
        
        current_upper = upper[-1]
        current_lower = lower[-1]
        
        prev_upper = upper[-2] if len(upper) > 1 else None
        prev_lower = lower[-2] if len(lower) > 1 else None
        
        # Loguer les valeurs actuelles
        # Utiliser plus de précision pour les paires BTC
        precision = 5 if 'BTC' in self.symbol else 2
        logger.info(f"[Bollinger] {self.symbol}: Price={current_price:.{precision}f}, "
                    f"Upper={current_upper:.{precision}f}, Lower={current_lower:.{precision}f}")
        
        # Vérifier les conditions pour générer un signal
        signal = None
        
        if np.isnan(current_upper) or prev_price is None or np.isnan(prev_upper):
            # Pas assez de données pour un signal fiable
            return None
        
        # Calculer des métriques utiles
        band_width = current_upper - current_lower
        band_width_pct = (band_width / middle[-1]) * 100
        # Éviter la division par zéro
        if band_width > 0:
            price_position = (current_price - current_lower) / band_width  # 0 = bande basse, 1 = bande haute
        else:
            # Si les bandes sont identiques (volatilité nulle), position au milieu
            price_position = 0.5
        
        # Filtre de tendance : vérifier la position du prix par rapport à la SMA
        current_middle = middle[-1]
        prev_middle = middle[-2] if len(middle) > 1 else current_middle
        
        # Tendance haussière si SMA monte
        sma_trend_up = current_middle > prev_middle
        # Prix au-dessus de la SMA = tendance haussière
        price_above_sma = current_price > current_middle * 0.98  # 2% de tolérance
        
        # Signal d'ACHAT: Prix touche ou pénètre la bande inférieure (acheter l'extrême)
        # MAIS seulement si on n'est pas dans une forte tendance baissière
        if current_price <= current_lower * 1.002:  # Prix à ou sous la bande basse (avec 0.2% de marge)
            # Plus le prix est sous la bande, plus c'est extrême
            penetration_pct = ((current_lower - current_price) / current_lower) * 100
            
            # NOUVEAU : Ajuster la logique selon la tendance
            if not sma_trend_up and not price_above_sma:
                # Tendance baissière confirmée : réduire la confiance mais ne pas bloquer
                confidence *= 0.4  # Réduire drastiquement la confiance
                signal_reason += "_bearish_market"
                logger.debug(f"[Bollinger] {self.symbol}: Signal LONG réduit en tendance baissière "
                           f"(confiance réduite à {confidence:.2f})")
                # Si la confiance devient trop faible, ne pas générer de signal
                if confidence < 0.3:
                    return None
            
            # Vérifier si les bandes sont suffisamment écartées (éviter les marchés plats)
            if band_width_pct < 1.0:  # Bandes très serrées, marché plat
                confidence = 0.7  # Augmenté de 0.5 à 0.7
                signal_reason = "squeeze_caution"
            elif penetration_pct > 0.5:  # Forte pénétration sous la bande
                confidence = 0.85
                signal_reason = "strong_oversold"
            elif current_price < current_lower:  # Légère pénétration
                confidence = 0.75
                signal_reason = "oversold"
            else:  # Juste touché la bande
                confidence = 0.75  # Augmenté de 0.65 à 0.75
                signal_reason = "band_touch"
            
            # Bonus si on vient de l'extérieur (premier contact)
            if prev_price > prev_lower * 1.01:
                confidence *= 1.1
                signal_reason += "_fresh"
            
            # Stop basé sur la volatilité uniquement
            
            # Stop basé sur la volatilité (pour métadonnées seulement)
            if 'BTC' in self.symbol:
                stop_multiplier = 0.5  # BTC
            else:
                stop_multiplier = 0.6   # Autres
            
            stop_distance = band_width * stop_multiplier
            
            # Forcer un stop minimum en pourcentage absolu - AUGMENTÉ pour éviter les whipsaws
            min_stop_percent = 1.5 if 'BTC' in self.symbol else 2.0  # 1.5% pour BTC, 2% pour autres
            min_stop_distance = current_price * (min_stop_percent / 100)
            stop_distance = max(stop_distance, min_stop_distance)
            
            stop_price = current_price - stop_distance
            
            metadata = {
                "bb_upper": float(current_upper),
                "bb_middle": float(middle[-1]),
                "bb_lower": float(current_lower),
                "band_width_pct": float(band_width_pct),
                "price_position": float(price_position),
                "penetration_pct": float(penetration_pct),
                "reason": signal_reason,
                "stop_price": float(stop_price),
                "sma_trend_up": sma_trend_up,
                "price_above_sma": price_above_sma
            }
            
            signal = self.create_signal(
                side=OrderSide.LONG,
                price=current_price,
                confidence=min(confidence, 0.95),
                metadata=metadata
            )
        
        # Signal de VENTE: Prix touche ou pénètre la bande supérieure (vendre l'extrême)
        # OU signal de vente anticipé en tendance baissière forte
        elif (current_price >= current_upper * 0.998 or  # Condition normale
              (not sma_trend_up and not price_above_sma and current_price >= current_middle * 1.005)):  # Vente anticipée en tendance baissière
            # Identifier le type de signal de vente
            is_early_short = (not sma_trend_up and not price_above_sma and
                              current_price < current_upper * 0.998 and
                              current_price >= current_middle * 1.005)

            if is_early_short:
                # Signal de vente anticipé en tendance baissière
                penetration_pct = ((current_price - current_middle) / current_middle) * 100
                confidence = 0.75  # Confiance modérée pour vente anticipée
                signal_reason = "early_short_bearish_trend"
                logger.info(f"[Bollinger] {self.symbol}: Signal SHORT anticipé en tendance baissière")
            else:
                # Signal de vente normal (bande supérieure)
                penetration_pct = ((current_price - current_upper) / current_upper) * 100
            
            # Calculer la confiance (seulement pour signaux normaux)
            if not is_early_short:
                # Vérifier si les bandes sont suffisamment écartées
                if band_width_pct < 1.0:  # Bandes très serrées, marché plat
                    confidence = 0.7  # Aligné avec les signaux LONG pour cohérence
                    signal_reason = "squeeze_caution"
                elif penetration_pct > 0.5:  # Forte pénétration au-dessus de la bande
                    confidence = 0.85
                    signal_reason = "strong_overbought"
                elif current_price > current_upper:  # Légère pénétration
                    confidence = 0.75
                    signal_reason = "overbought"
                else:  # Juste touché la bande
                    confidence = 0.65
                    signal_reason = "band_touch"
            
            # Bonus si on vient de l'intérieur (premier contact)
            if prev_price < prev_upper * 0.99:
                confidence *= 1.1
                signal_reason += "_fresh"
            
            # Stop basé sur la volatilité uniquement
            
            # Stop basé sur la volatilité (pour métadonnées seulement)
            if 'BTC' in self.symbol:
                stop_multiplier = 0.5  # BTC
            else:
                stop_multiplier = 0.6   # Autres
            
            stop_distance = band_width * stop_multiplier
            
            # Forcer un stop minimum en pourcentage absolu
            min_stop_percent = 0.5 if 'BTC' in self.symbol else 0.8
            min_stop_distance = current_price * (min_stop_percent / 100)
            stop_distance = max(stop_distance, min_stop_distance)
            
            stop_price = current_price + stop_distance
            
            metadata = {
                "bb_upper": float(current_upper),
                "bb_middle": float(middle[-1]),
                "bb_lower": float(current_lower),
                "band_width_pct": float(band_width_pct),
                "price_position": float(price_position),
                "penetration_pct": float(penetration_pct),
                "reason": signal_reason,
                "stop_price": float(stop_price)
            }
            
            signal = self.create_signal(
                side=OrderSide.SHORT,
                price=current_price,
                confidence=min(confidence, 0.95),
                metadata=metadata
            )
        
        # Mettre à jour les valeurs précédentes
        self.prev_price = current_price
        self.prev_upper = current_upper
        self.prev_lower = current_lower
        
        # Mettre à jour le timestamp si un signal est généré
        if signal:
            self.last_signal_time = datetime.now()
        
        return signal
    
    def _calculate_confidence(self, price: float, lower: float, upper: float, side: OrderSide) -> float:
        """
        Calcule le niveau de confiance d'un signal basé sur la position dans les bandes.
        
        Args:
            price: Prix actuel
            lower: Valeur de la bande inférieure
            upper: Valeur de la bande supérieure
            side: Côté du signal (LONG ou SHORT)

        Returns:
            Niveau de confiance entre 0.0 et 1.0
        """
        # Calcul de la distance du prix dans les bandes
        band_width = upper - lower

        if side == OrderSide.LONG:
            # Plus le prix est bas sous la bande inférieure, plus la confiance est élevée
            penetration = (lower - price) / band_width
            # Normaliser entre 0.5 et 1.0
            confidence = 0.5 + min(max(penetration * 2, 0), 0.5)
            return confidence
        else:  # SHORT
            # Plus le prix est haut au-dessus de la bande supérieure, plus la confiance est élevée
            penetration = (price - upper) / band_width
            # Normaliser entre 0.5 et 1.0
            confidence = 0.5 + min(max(penetration * 2, 0), 0.5)
            return confidence