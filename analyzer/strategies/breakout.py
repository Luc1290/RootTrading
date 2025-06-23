"""
Stratégie de trading basée sur les breakouts de consolidation.
Détecte les périodes de consolidation (range) et génère des signaux lorsque le prix casse le range.
"""
import logging
from datetime import datetime
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
from .advanced_filters_mixin import AdvancedFiltersMixin

# Configuration du logging
logger = logging.getLogger(__name__)

class BreakoutStrategy(BaseStrategy, AdvancedFiltersMixin):
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
        last_idx = len(df) - 1  # Index basé sur la position, pas sur l'index du DataFrame
        
        # Vérifier chaque range pour un breakout
        for start_idx, end_idx, support, resistance in ranges:
            # Ne considérer que les ranges qui se terminent récemment
            # et qui n'ont pas déjà produit un breakout
            if last_idx - end_idx <= 3 and end_idx > self.last_breakout_idx:
                last_close = last_candle['close']
                
                # Breakout haussier
                if last_close > resistance * (1 + self.breakout_threshold / 100):
                    self.last_breakout_idx = last_idx
                    
                    # Calculer la hauteur du range
                    range_height = resistance - support
                    
                    # Utiliser l'ATR pour des cibles dynamiques
                    atr_percent = self.calculate_atr(df)
                    
                    # Pour les breakouts, utiliser un ratio risque/récompense plus conservateur
                    # car les faux breakouts sont fréquents
                    risk_reward = 1.2 if range_height / support * 100 < 2 else 1.5
                    
                    # Calculer le stop basé sur l'ATR et la résistance cassée
                    atr_stop_distance = atr_percent / 100
                    atr_stop = last_close * (1 - atr_stop_distance)
                    
                    # Le stop peut être ajusté pour être juste sous la résistance cassée
                    # si c'est plus proche que l'ATR stop
                    resistance_stop = resistance * 0.995
                    stop_loss = max(resistance_stop, atr_stop)
                    
                    return {
                        "type": "bullish",
                        "side": OrderSide.BUY,
                        "price": last_close,
                        "support": support,
                        "resistance": resistance,
                        "range_duration": end_idx - start_idx + 1,
                        "range_height_percent": (range_height / support) * 100,
                        "atr_percent": atr_percent,
                        "stop_price": stop_loss
                    }
                
                # Breakout baissier
                elif last_close < support * (1 - self.breakout_threshold / 100):
                    self.last_breakout_idx = last_idx
                    
                    # Calculer la hauteur du range
                    range_height = resistance - support
                    
                    # Utiliser l'ATR pour des cibles dynamiques
                    atr_percent = self.calculate_atr(df)
                    
                    # Pour les breakouts, utiliser un ratio risque/récompense plus conservateur
                    risk_reward = 1.2 if range_height / support * 100 < 2 else 1.5
                    
                    # Calculer le stop basé sur l'ATR et le support cassé
                    atr_stop_distance = atr_percent / 100
                    atr_stop = last_close * (1 + atr_stop_distance)
                    
                    # Le stop peut être ajusté pour être juste au-dessus du support cassé
                    # si c'est plus proche que l'ATR stop
                    support_stop = support * 1.005
                    stop_loss = min(support_stop, atr_stop)
                    
                    return {
                        "type": "bearish",
                        "side": OrderSide.SELL,
                        "price": last_close,
                        "support": support,
                        "resistance": resistance,
                        "range_duration": end_idx - start_idx + 1,
                        "range_height_percent": (range_height / support) * 100,
                        "atr_percent": atr_percent,
                        "stop_price": stop_loss
                    }
        
        return None
    
    def generate_signal(self) -> Optional[StrategySignal]:
        """
        Génère un signal de trading sophistiqué basé sur les breakouts.
        Utilise des filtres avancés pour éviter les faux breakouts.
        
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
        volumes = df['volume'].values if 'volume' in df.columns else None
        current_price = df['close'].iloc[-1]
        
        # === NOUVEAU SYSTÈME DE FILTRES SOPHISTIQUES ===
        
        # 1. FILTRE BREAKOUT DE BASE
        breakout_info = self._detect_valid_breakout(df)
        if breakout_info is None:
            return None
        
        signal_side = breakout_info['side']
        
        # 2. FILTRE VOLUME EXPANSION (confirmation institutionnelle)
        volume_score = self._analyze_volume_confirmation_common(volumes) if volumes is not None else 0.7
        if volume_score < 0.5:  # Plus strict pour breakouts
            logger.debug(f"[Breakout] {self.symbol}: Signal rejeté - volume insuffisant ({volume_score:.2f})")
            return None
        
        # 3. FILTRE FALSE BREAKOUT DETECTION (éviter les faux signaux)
        false_breakout_score = self._detect_false_breakout_patterns(df, breakout_info)
        if false_breakout_score < 0.5:
            logger.debug(f"[Breakout] {self.symbol}: Signal rejeté - pattern faux breakout ({false_breakout_score:.2f})")
            return None
        
        # 4. FILTRE RETEST VALIDATION (confirmation du niveau)
        retest_score = self._validate_retest_opportunity(df, breakout_info)
        
        # 5. FILTRE TREND ALIGNMENT (direction générale)
        trend_score = self._analyze_trend_alignment_common(df, signal_side)
        
        # 6. FILTRE ATR ENVIRONMENT (environnement volatilité)
        atr_score = self._analyze_atr_environment_common(df)
        
        # === CALCUL DE CONFIANCE COMPOSITE ===
        scores = {
            'volume': volume_score,
            'false_breakout': false_breakout_score,
            'retest': retest_score,
            'trend': trend_score,
            'atr': atr_score
        }
        
        weights = {
            'volume': 0.30,        # Volume crucial pour breakouts
            'false_breakout': 0.25, # Éviter faux signaux
            'retest': 0.20,        # Validation niveau
            'trend': 0.15,         # Direction générale
            'atr': 0.10           # Environnement
        }
        
        confidence = self._calculate_composite_confidence_common(scores, weights)
        
        # Seuil minimum de confiance
        if confidence < 0.65:
            logger.debug(f"[Breakout] {self.symbol}: Signal rejeté - confiance trop faible ({confidence:.2f})")
            return None
        
        # === CONSTRUCTION DU SIGNAL ===
        signal = self.create_signal(
            side=signal_side,
            price=current_price,
            confidence=confidence
        )
        
        # Ajouter les métadonnées d'analyse
        signal.metadata.update({
            'breakout_level': breakout_info.get('level'),
            'range_duration': breakout_info.get('duration'),
            'range_height_pct': breakout_info.get('height_pct'),
            'volume_expansion': volume_score,
            'false_breakout_score': false_breakout_score,
            'retest_score': retest_score,
            'trend_score': trend_score,
            'atr_score': atr_score,
            'breakout_strength': self._calculate_breakout_strength(breakout_info)
        })
        
        precision = 5 if 'BTC' in self.symbol else 3
        logger.info(f"🎯 [Breakout] {self.symbol}: Signal {signal_side} @ {current_price:.{precision}f} "
                   f"(confiance: {confidence:.2f}, niveau: {breakout_info.get('level', 'N/A'):.{precision}f}, "
                   f"scores: V={volume_score:.2f}, FB={false_breakout_score:.2f})")
        
        return signal
    
    def _detect_valid_breakout(self, df: pd.DataFrame) -> Optional[Dict]:
        """Version simplifiée de détection de breakout avec validation de tendance."""
        try:
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            
            if len(df) < 20:
                return None
            
            # NOUVEAU: Validation de tendance avant de générer le signal
            trend_alignment = self._validate_trend_alignment_for_signal(df)
            if trend_alignment is None:
                return None  # Pas assez de données pour valider la tendance
            
            # Chercher résistance et support récents
            lookback = min(20, len(df))
            recent_highs = highs[-lookback:]
            recent_lows = lows[-lookback:]
            current_price = closes[-1]
            
            resistance = np.max(recent_highs[:-1])  # Exclure la bougie actuelle
            support = np.min(recent_lows[:-1])
            
            # Breakout haussier ET tendance compatible
            if current_price > resistance * 1.002:  # 0.2% au-dessus
                # NOUVEAU: Ne faire de breakout BUY que si tendance n'est pas fortement baissière
                if trend_alignment in ["STRONG_BEARISH", "WEAK_BEARISH"]:
                    logger.debug(f"[Breakout] {self.symbol}: BUY breakout supprimé - tendance {trend_alignment}")
                    return None
                    
                return {
                    'side': OrderSide.BUY,
                    'level': resistance,
                    'duration': lookback,
                    'height_pct': (resistance - support) / support * 100
                }
            
            # Breakout baissier ET tendance compatible
            elif current_price < support * 0.998:  # 0.2% en dessous
                # NOUVEAU: Ne faire de breakout SELL que si tendance n'est pas fortement haussière
                if trend_alignment in ["STRONG_BULLISH", "WEAK_BULLISH"]:
                    logger.debug(f"[Breakout] {self.symbol}: SELL breakout supprimé - tendance {trend_alignment}")
                    return None
                    
                return {
                    'side': OrderSide.SELL,
                    'level': support,
                    'duration': lookback,
                    'height_pct': (resistance - support) / support * 100
                }
            
            return None
        except:
            return None
    
    def _detect_false_breakout_patterns(self, df: pd.DataFrame, breakout_info: Dict) -> float:
        """Détection simplifiée de faux breakouts."""
        try:
            # Score de base
            score = 0.7
            
            # Vérifier la force du breakout
            current_price = df['close'].iloc[-1]
            level = breakout_info['level']
            
            if breakout_info['side'] == OrderSide.BUY:
                penetration = (current_price - level) / level * 100
            else:
                penetration = (level - current_price) / level * 100
            
            if penetration > 1.0:  # Breakout > 1%
                score += 0.2
            elif penetration > 0.5:  # Breakout > 0.5%
                score += 0.1
            
            return min(0.95, score)
        except:
            return 0.7
    
    def _validate_retest_opportunity(self, df: pd.DataFrame, breakout_info: Dict) -> float:
        """Validation simplifiée du retest."""
        return 0.8  # Score fixe pour simplifier
    
    def _calculate_breakout_strength(self, breakout_info: Dict) -> str:
        """Calcul force du breakout."""
        height_pct = breakout_info.get('height_pct', 0)
        if height_pct > 3:
            return "strong"
        elif height_pct > 1.5:
            return "moderate"
        else:
            return "weak"
    
    def _validate_trend_alignment_for_signal(self, df: pd.DataFrame) -> Optional[str]:
        """
        Valide la tendance actuelle pour déterminer si un signal est approprié.
        Utilise la même logique que le signal_aggregator pour cohérence.
        """
        try:
            if df is None or len(df) < 50:
                return None
            
            prices = df['close'].values
            
            # Calculer EMA 21 vs EMA 50 (harmonisé avec signal_aggregator)
            import talib
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
        
        # Calculer la confiance basée sur la durée du range
        range_duration = breakout['range_duration']
        
        # Plus le range est long, plus la confiance est grande
        confidence = min(0.75 + (range_duration / 20), 0.98)  # Augmenté de 0.5 à 0.75
        
        # Récupérer les informations du breakout
        side = breakout['side']
        price = breakout['price']
        
        # Créer le signal
        metadata = {
            "type": breakout['type'],
            "support": float(breakout['support']),
            "resistance": float(breakout['resistance']),
            "range_duration": int(breakout['range_duration']),
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
        
        # Mettre à jour le timestamp si un signal est généré
        if signal:
            self.last_signal_time = datetime.now()
        
        return signal