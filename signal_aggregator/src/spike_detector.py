"""
Module de détection de spikes et génération de signaux de take-profit automatiques
"""
import logging
from typing import Dict, Tuple
from datetime import datetime
import json

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from shared.src.config import (
    MACD_HISTOGRAM_VERY_STRONG, MACD_HISTOGRAM_STRONG, MACD_HISTOGRAM_MODERATE, 
    MACD_HISTOGRAM_WEAK
)

logger = logging.getLogger(__name__)

class SpikeDetector:
    """
    Détecte les mouvements de prix rapides (spikes) et génère des signaux de sortie
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
        
        # Paramètres de détection de spike
        self.spike_thresholds = {
            '1m': 1.5,   # 1.5% en 1 minute = spike
            '5m': 3.0,   # 3% en 5 minutes
            '15m': 5.0,  # 5% en 15 minutes
            '1h': 8.0    # 8% en 1 heure
        }
        
        # Paramètres de take-profit
        self.take_profit_levels = {
            'conservative': 0.3,  # 30% du mouvement
            'moderate': 0.5,      # 50% du mouvement
            'aggressive': 0.7     # 70% du mouvement
        }
        
        # Cache des positions ouvertes
        self.open_positions = {}
        self.spike_history = {}
        
    def detect_spike(self, symbol: str, current_price: float, 
                    timeframe: str = '5m') -> Tuple[bool, float, str]:
        """
        Détecte si un spike est en cours
        
        Returns:
            (is_spike, spike_magnitude, direction): 
            - is_spike: True si spike détecté
            - spike_magnitude: % de changement
            - direction: 'up' ou 'down'
        """
        try:
            # Récupérer l'historique des prix
            history_key = f"price_history:{symbol}:{timeframe}"
            history = self.redis.get(history_key)
            
            if not history:
                return False, 0.0, 'neutral'
            
            if isinstance(history, str):
                price_history = json.loads(history)
            else:
                price_history = history
            
            if not price_history or len(price_history) < 2:
                return False, 0.0, 'neutral'
            
            # Calculer le changement depuis le début du timeframe
            start_price = price_history[0]['price']
            price_change_pct = ((current_price - start_price) / start_price) * 100
            
            # Vérifier si c'est un spike selon le seuil
            threshold = self.spike_thresholds.get(timeframe, 3.0)
            
            if abs(price_change_pct) >= threshold:
                direction = 'up' if price_change_pct > 0 else 'down'
                logger.info(f"🚀 SPIKE détecté sur {symbol}: {price_change_pct:.2f}% en {timeframe}")
                return True, abs(price_change_pct), direction
            
            return False, 0.0, 'neutral'
            
        except Exception as e:
            logger.error(f"Erreur détection spike: {e}")
            return False, 0.0, 'neutral'
    
    def should_take_profit(self, symbol: str, position_side: str, 
                          entry_price: float, current_price: float,
                          aggressive_mode: bool = False) -> Tuple[bool, str]:
        """
        Détermine si on doit prendre des profits sur une position
        
        Args:
            symbol: Symbole tradé
            position_side: 'BUY' ou 'SELL'
            entry_price: Prix d'entrée
            current_price: Prix actuel
            aggressive_mode: Si True, utilise des seuils plus serrés
            
        Returns:
            (should_exit, reason): True si on doit sortir
        """
        try:
            # Calculer le profit non réalisé
            if position_side == 'BUY':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:  # SELL
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            # Vérifier spike en cours
            is_spike, spike_magnitude, spike_direction = self.detect_spike(
                symbol, current_price, '5m'
            )
            
            # Si spike favorable à notre position
            if is_spike:
                if (position_side == 'BUY' and spike_direction == 'up') or \
                   (position_side == 'SELL' and spike_direction == 'down'):
                    
                    # Utiliser le niveau de TP selon l'agressivité
                    tp_level = self.take_profit_levels['aggressive' if aggressive_mode else 'moderate']
                    tp_threshold = spike_magnitude * tp_level
                    
                    if pnl_pct >= tp_threshold:
                        return True, f"Take-profit spike: +{pnl_pct:.2f}% (seuil: {tp_threshold:.1f}%)"
            
            # Take-profit standard basé sur des seuils fixes
            if aggressive_mode:
                # Mode agressif: TP plus rapide
                tp_levels = [3.0, 5.0, 8.0, 12.0]  # 3%, 5%, 8%, 12%
            else:
                # Mode normal
                tp_levels = [5.0, 8.0, 12.0, 18.0]  # 5%, 8%, 12%, 18%
            
            for tp_level in tp_levels:
                if pnl_pct >= tp_level:
                    # Vérifier la dynamique du marché
                    momentum = self._check_momentum(symbol, position_side)
                    
                    # Si momentum faible, prendre profit
                    if momentum < 0.3:  # Momentum faible
                        return True, f"Take-profit niveau {tp_level}% atteint avec momentum faible"
                    
                    # Si au-dessus de 12%, toujours prendre une partie
                    if pnl_pct >= 12.0:
                        return True, f"Take-profit sécurité: +{pnl_pct:.2f}%"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"Erreur should_take_profit: {e}")
            return False, ""
    
    def _check_momentum(self, symbol: str, position_side: str) -> float:
        """
        Vérifie le momentum actuel (0-1, 1 = fort momentum)
        """
        try:
            # Récupérer RSI et MACD
            indicators_key = f"indicators:{symbol}:15m"
            indicators = self.redis.get(indicators_key)
            
            if not indicators:
                return 0.5  # Momentum neutre par défaut
            
            if isinstance(indicators, str):
                indicators = json.loads(indicators)
            
            rsi = indicators.get('rsi_14', 50)
            macd_histogram = indicators.get('macd_histogram', 0)
            
            # Calculer score momentum
            momentum_score = 0.5
            
            if position_side == 'BUY':
                # Pour BUY: RSI > 50 et MACD > 0 = bon momentum
                if rsi > 60:
                    momentum_score += 0.2
                if rsi > 70:
                    momentum_score += 0.1
                # SEUILS STANDARDISÉS MACD histogram
                if macd_histogram > MACD_HISTOGRAM_VERY_STRONG:  # STANDARDISÉ: Momentum très fort
                    momentum_score += 0.3
                elif macd_histogram > MACD_HISTOGRAM_STRONG:  # STANDARDISÉ: Momentum fort
                    momentum_score += 0.2
                elif macd_histogram > MACD_HISTOGRAM_MODERATE:  # STANDARDISÉ: Momentum modéré
                    momentum_score += 0.1
                elif macd_histogram > MACD_HISTOGRAM_WEAK:  # STANDARDISÉ: Momentum faible
                    momentum_score += 0.05
            else:  # SELL
                # Pour SELL: RSI < 50 et MACD < 0 = bon momentum
                if rsi < 40:
                    momentum_score += 0.2
                if rsi < 30:
                    momentum_score += 0.1
                # SEUILS STANDARDISÉS MACD histogram
                if macd_histogram < -MACD_HISTOGRAM_VERY_STRONG:  # STANDARDISÉ: Momentum très fort
                    momentum_score += 0.3
                elif macd_histogram < -MACD_HISTOGRAM_STRONG:  # STANDARDISÉ: Momentum fort
                    momentum_score += 0.2
                elif macd_histogram < -MACD_HISTOGRAM_MODERATE:  # STANDARDISÉ: Momentum modéré
                    momentum_score += 0.1
                elif macd_histogram < -MACD_HISTOGRAM_WEAK:  # STANDARDISÉ: Momentum faible
                    momentum_score += 0.05
            
            return min(1.0, momentum_score)
            
        except Exception as e:
            logger.error(f"Erreur check momentum: {e}")
            return 0.5
    
    def generate_take_profit_signal(self, symbol: str, position_side: str, 
                                  current_price: float, reason: str) -> Dict:
        """
        Génère un signal de take-profit
        """
        exit_side = 'SELL' if position_side == 'BUY' else 'BUY'
        
        signal = {
            'strategy': 'SpikeTakeProfit',
            'symbol': symbol,
            'side': exit_side,
            'price': current_price,
            'confidence': 0.95,  # Haute confiance pour TP
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'reason': reason,
                'type': 'take_profit',
                'spike_detected': True
            }
        }
        
        logger.info(f"💰 Signal TAKE-PROFIT généré: {exit_side} {symbol} @ {current_price:.4f} - {reason}")
        
        return signal