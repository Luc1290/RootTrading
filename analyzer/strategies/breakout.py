"""
Stratégie de trading basée sur les breakouts de consolidation.
Détecte les périodes de consolidation (range) et génère des signaux lorsque le prix casse le range.
"""
import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

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

class BreakoutStrategy(BaseStrategy):
    """
    Stratégie qui détecte les cassures (breakouts) après des périodes de consolidation.
    Génère des signaux d'achat quand le prix casse une résistance et des signaux de vente
    quand le prix casse un support.
    """
    
    def __init__(self, symbol: str, params: Dict[str, Any] = None):
        """
        Initialise la stratégie de Breakout.
        
        Args:
            symbol: Symbole de trading (ex: 'BTCUSDC')
            params: Paramètres spécifiques à la stratégie
        """
        super().__init__(symbol, params)
        
        # Paramètres de la stratégie
        self.min_range_candles = self.params.get('min_range_candles', 5)  # Minimum de chandeliers pour un range
        self.max_range_percent = self.params.get('max_range_percent', 3.0)  # % max pour considérer comme consolidation
        self.breakout_threshold = self.params.get('breakout_threshold', 0.5)  # % au-dessus/en-dessous pour confirmer
        self.max_lookback = self.params.get('max_lookback', 50)  # Nombre max de chandeliers à analyser
        
        # Etat de la stratégie
        self.detected_ranges = []  # [(start_idx, end_idx, support, resistance), ...]
        self.last_breakout_idx = 0
        
        logger.info(f"🔧 Stratégie Breakout initialisée pour {symbol} "
                   f"(min_candles={self.min_range_candles}, threshold={self.breakout_threshold}%)")
    
    @property
    def name(self) -> str:
        """Nom unique de la stratégie."""
        return "Breakout_Strategy"
    
    def get_min_data_points(self) -> int:
        """
        Nombre minimum de points de données nécessaires.
        
        Returns:
            Nombre minimum de données requises
        """
        # Besoin d'au moins 2x le nombre min de chandeliers + détection de breakout
        return self.min_range_candles * 2 + 5
    
    def _find_consolidation_ranges(self, df: pd.DataFrame) -> List[Tuple[int, int, float, float]]:
        """
        Trouve les périodes de consolidation (range) dans les données.
        
        Args:
            df: DataFrame avec les données de prix
            
        Returns:
            Liste de tuples (début, fin, support, résistance)
        """
        ranges = []
        
        # Limiter le nombre de chandeliers à analyser
        lookback = min(len(df), self.max_lookback)
        
        # Récupérer les dernières N bougies
        recent_df = df.iloc[-lookback:]
        
        # Parcourir les périodes possibles pour trouver des ranges
        for i in range(len(recent_df) - self.min_range_candles + 1):
            # Récupérer une fenêtre de taille minimum
            window = recent_df.iloc[i:i+self.min_range_candles]
            
            # Calculer le support et la résistance
            support = window['low'].min()
            resistance = window['high'].max()
            
            # Calculer l'amplitude du range en pourcentage
            range_percent = ((resistance - support) / support) * 100
            
            # Vérifier si c'est un range valide
            if range_percent <= self.max_range_percent:
                # Étendre le range autant que possible
                end_idx = i + self.min_range_candles
                while end_idx < len(recent_df):
                    next_candle = recent_df.iloc[end_idx]
                    # Si le prochain chandelier reste dans le range (ou presque)
                    if (next_candle['low'] >= support * 0.995 and 
                        next_candle['high'] <= resistance * 1.005):
                        end_idx += 1
                    else:
                        break
                
                # Ajuster les indices par rapport au DataFrame complet
                start_idx = len(df) - lookback + i
                end_idx = len(df) - lookback + end_idx - 1
                
                # Ajouter le range à la liste
                ranges.append((start_idx, end_idx, support, resistance))
        
        return ranges
    
    def _detect_breakout(self, df: pd.DataFrame, ranges: List[Tuple[int, int, float, float]]) -> Optional[Dict[str, Any]]:
        """
        Détecte un breakout d'un des ranges identifiés.
        
        Args:
            df: DataFrame avec les données de prix
            ranges: Liste de ranges (début, fin, support, résistance)
            
        Returns:
            Informations sur le breakout ou None
        """
        if not ranges or not df.shape[0]:
            return None
        
        # Obtenir le dernier chandelier
        last_candle = df.iloc[-1]
        last_idx = df.index.get_loc(last_candle.name)
        
        # Vérifier chaque range pour un breakout
        for start_idx, end_idx, support, resistance in ranges:
            # Ne considérer que les ranges qui se terminent récemment
            # et qui n'ont pas déjà produit un breakout
            if last_idx - end_idx <= 3 and end_idx > self.last_breakout_idx:
                last_close = last_candle['close']
                
                # Breakout haussier
                if last_close > resistance * (1 + self.breakout_threshold / 100):
                    self.last_breakout_idx = last_idx
                    
                    # Calculer le profit target basé sur la hauteur du range
                    range_height = resistance - support
                    target_price = last_close + range_height
                    
                    # Stop loss juste sous la résistance cassée
                    stop_loss = resistance * 0.99
                    
                    return {
                        "type": "bullish",
                        "side": OrderSide.BUY,
                        "price": last_close,
                        "support": support,
                        "resistance": resistance,
                        "range_duration": end_idx - start_idx + 1,
                        "target_price": target_price,
                        "stop_price": stop_loss
                    }
                
                # Breakout baissier
                elif last_close < support * (1 - self.breakout_threshold / 100):
                    self.last_breakout_idx = last_idx
                    
                    # Calculer le profit target basé sur la hauteur du range
                    range_height = resistance - support
                    target_price = last_close - range_height
                    
                    # Stop loss juste au-dessus du support cassé
                    stop_loss = support * 1.01
                    
                    return {
                        "type": "bearish",
                        "side": OrderSide.SELL,
                        "price": last_close,
                        "support": support,
                        "resistance": resistance,
                        "range_duration": end_idx - start_idx + 1,
                        "target_price": target_price,
                        "stop_price": stop_loss
                    }
        
        return None
    
    def generate_signal(self) -> Optional[StrategySignal]:
        """
        Génère un signal de trading basé sur les breakouts.
        
        Returns:
            Signal de trading ou None si aucun signal n'est généré
        """
        # Convertir les données en DataFrame
        df = self.get_data_as_dataframe()
        if df is None or len(df) < self.get_min_data_points():
            return None
        
        # Trouver les zones de consolidation
        ranges = self._find_consolidation_ranges(df)
        self.detected_ranges = ranges
        
        # Détecter un breakout
        breakout = self._detect_breakout(df, ranges)
        
        if not breakout:
            return None
        
        # Calculer la confiance basée sur la durée du range
        range_duration = breakout['range_duration']
        
        # Plus le range est long, plus la confiance est grande
        confidence = min(0.5 + (range_duration / 20), 0.95)
        
        # Récupérer les informations du breakout
        side = breakout['side']
        price = breakout['price']
        
        # Créer le signal
        metadata = {
            "type": breakout['type'],
            "support": float(breakout['support']),
            "resistance": float(breakout['resistance']),
            "range_duration": int(breakout['range_duration']),
            "target_price": float(breakout['target_price']),
            "stop_price": float(breakout['stop_price'])
        }
        
        signal = self.create_signal(
            side=side,
            price=float(price),
            confidence=confidence,
            metadata=metadata
        )
        
        logger.info(f"🚀 [Breakout] Signal {side.value} sur {self.symbol}: "
                   f"cassure d'un range de {range_duration} chandeliers")
        
        return signal