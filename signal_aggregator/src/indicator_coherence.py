"""
Module de vérification de cohérence entre indicateurs
S'assure que RSI et MACD sont alignés pour confirmer les signaux
"""
import logging
from typing import Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class IndicatorCoherenceValidator:
    """
    Vérifie la cohérence entre différents indicateurs techniques
    """
    
    def __init__(self):
        # Seuils de cohérence
        self.rsi_buy_threshold = 40   # RSI minimum pour BUY
        self.rsi_sell_threshold = 60  # RSI maximum pour SELL
        self.macd_threshold = 0.00005  # MACD minimum significatif
        
        # Poids des indicateurs pour le score de cohérence
        self.indicator_weights = {
            'rsi': 0.3,
            'macd': 0.3,
            'ema_alignment': 0.25,
            'volume': 0.15
        }
    
    def validate_signal_coherence(self, signal_side: str, indicators: Dict) -> Tuple[bool, float, str]:
        """
        Valide la cohérence d'un signal avec les indicateurs
        
        Args:
            signal_side: 'BUY' ou 'SELL'
            indicators: Dict des indicateurs techniques
            
        Returns:
            (is_coherent, coherence_score, reason)
        """
        try:
            coherence_score = 0.0
            reasons = []
            
            # Vérifier RSI
            rsi_score, rsi_reason = self._check_rsi_coherence(signal_side, indicators)
            coherence_score += rsi_score * self.indicator_weights['rsi']
            if rsi_reason:
                reasons.append(rsi_reason)
            
            # Vérifier MACD
            macd_score, macd_reason = self._check_macd_coherence(signal_side, indicators)
            coherence_score += macd_score * self.indicator_weights['macd']
            if macd_reason:
                reasons.append(macd_reason)
            
            # Vérifier alignement EMA
            ema_score, ema_reason = self._check_ema_alignment(signal_side, indicators)
            coherence_score += ema_score * self.indicator_weights['ema_alignment']
            if ema_reason:
                reasons.append(ema_reason)
            
            # Vérifier volume
            volume_score, volume_reason = self._check_volume_confirmation(signal_side, indicators)
            coherence_score += volume_score * self.indicator_weights['volume']
            if volume_reason:
                reasons.append(volume_reason)
            
            # Déterminer si coherent (score > 0.6)
            is_coherent = coherence_score > 0.6
            
            reason = " | ".join(reasons) if reasons else "Cohérence validée"
            
            if is_coherent:
                logger.info(f"✅ Cohérence validée (score: {coherence_score:.2f}): {reason}")
            else:
                logger.warning(f"❌ Cohérence faible (score: {coherence_score:.2f}): {reason}")
            
            return is_coherent, coherence_score, reason
            
        except Exception as e:
            logger.error(f"Erreur validation cohérence: {e}")
            return True, 0.5, "Erreur validation - signal autorisé par défaut"
    
    def _check_rsi_coherence(self, signal_side: str, indicators: Dict) -> Tuple[float, str]:
        """
        Vérifie la cohérence du RSI avec le signal
        
        Returns:
            (score, reason): score 0-1, reason si problème
        """
        try:
            rsi = indicators.get('rsi_14')
            if not rsi:
                return 0.5, "RSI non disponible"
            
            rsi_val = float(rsi)
            
            if signal_side == 'BUY':
                if rsi_val < 30:
                    return 1.0, ""  # Excellent - RSI oversold
                elif rsi_val < 50:
                    return 0.8, ""  # Bon - RSI favorable
                elif rsi_val < 70:
                    return 0.4, "RSI neutre pour BUY"
                else:
                    return 0.1, "RSI trop élevé pour BUY"
            
            else:  # SELL
                if rsi_val > 70:
                    return 1.0, ""  # Excellent - RSI overbought
                elif rsi_val > 50:
                    return 0.8, ""  # Bon - RSI favorable
                elif rsi_val > 30:
                    return 0.4, "RSI neutre pour SELL"
                else:
                    return 0.1, "RSI trop bas pour SELL"
                    
        except Exception as e:
            logger.error(f"Erreur check RSI: {e}")
            return 0.5, "Erreur RSI"
    
    def _check_macd_coherence(self, signal_side: str, indicators: Dict) -> Tuple[float, str]:
        """
        Vérifie la cohérence du MACD avec le signal
        """
        try:
            macd_line = indicators.get('macd_line')
            macd_signal = indicators.get('macd_signal')
            macd_histogram = indicators.get('macd_histogram')
            
            if not all([macd_line, macd_signal, macd_histogram]):
                return 0.5, "MACD incomplet"
            
            macd_line_val = float(macd_line)
            macd_signal_val = float(macd_signal)
            macd_histogram_val = float(macd_histogram)
            
            if signal_side == 'BUY':
                # Pour BUY: MACD line > signal line et histogram > 0
                if macd_line_val > macd_signal_val and macd_histogram_val > 0:
                    return 1.0, ""  # Parfait
                elif macd_line_val > macd_signal_val:
                    return 0.7, ""  # Bon - crossover bullish
                elif macd_histogram_val > 0:
                    return 0.5, ""  # Moyen - momentum positif
                else:
                    return 0.2, "MACD baissier pour BUY"
            
            else:  # SELL
                # Pour SELL: MACD line < signal line et histogram < 0
                if macd_line_val < macd_signal_val and macd_histogram_val < 0:
                    return 1.0, ""  # Parfait
                elif macd_line_val < macd_signal_val:
                    return 0.7, ""  # Bon - crossover bearish
                elif macd_histogram_val < 0:
                    return 0.5, ""  # Moyen - momentum négatif
                else:
                    return 0.2, "MACD haussier pour SELL"
                    
        except Exception as e:
            logger.error(f"Erreur check MACD: {e}")
            return 0.5, "Erreur MACD"
    
    def _check_ema_alignment(self, signal_side: str, indicators: Dict) -> Tuple[float, str]:
        """
        Vérifie l'alignement des EMAs avec le signal
        """
        try:
            ema_12 = indicators.get('ema_12')
            ema_26 = indicators.get('ema_26')
            ema_50 = indicators.get('ema_50')
            
            if not all([ema_12, ema_26, ema_50]):
                return 0.5, "EMAs incomplètes"
            
            ema_12_val = float(ema_12)
            ema_26_val = float(ema_26)
            ema_50_val = float(ema_50)
            
            if signal_side == 'BUY':
                # Pour BUY: EMA12 > EMA26 > EMA50 (bullish alignment)
                if ema_12_val > ema_26_val > ema_50_val:
                    return 1.0, ""  # Parfait
                elif ema_12_val > ema_26_val:
                    return 0.6, ""  # Bon - trend court terme haussier
                elif ema_12_val > ema_50_val:
                    return 0.4, ""  # Moyen
                else:
                    return 0.2, "EMAs baissières pour BUY"
            
            else:  # SELL
                # Pour SELL: EMA12 < EMA26 < EMA50 (bearish alignment)
                if ema_12_val < ema_26_val < ema_50_val:
                    return 1.0, ""  # Parfait
                elif ema_12_val < ema_26_val:
                    return 0.6, ""  # Bon - trend court terme baissier
                elif ema_12_val < ema_50_val:
                    return 0.4, ""  # Moyen
                else:
                    return 0.2, "EMAs haussières pour SELL"
                    
        except Exception as e:
            logger.error(f"Erreur check EMA: {e}")
            return 0.5, "Erreur EMA"
    
    def _check_volume_confirmation(self, signal_side: str, indicators: Dict) -> Tuple[float, str]:
        """
        Vérifie la confirmation du volume
        """
        try:
            volume_ratio = indicators.get('volume_ratio')
            volume_trend = indicators.get('volume_trend')
            
            if not volume_ratio:
                return 0.5, "Volume non disponible"
            
            volume_ratio_val = float(volume_ratio)
            
            # Volume élevé confirme le signal
            if volume_ratio_val > 1.5:
                return 1.0, ""  # Excellent
            elif volume_ratio_val > 1.2:
                return 0.8, ""  # Bon
            elif volume_ratio_val > 0.8:
                return 0.6, ""  # Moyen
            else:
                return 0.3, "Volume trop faible"
                
        except Exception as e:
            logger.error(f"Erreur check volume: {e}")
            return 0.5, "Erreur volume"
    
    def get_coherence_requirements(self, signal_side: str) -> Dict:
        """
        Retourne les exigences de cohérence pour un type de signal
        """
        if signal_side == 'BUY':
            return {
                'rsi': 'RSI < 50 (idéalement < 30)',
                'macd': 'MACD line > signal line, histogram > 0',
                'ema': 'EMA12 > EMA26 > EMA50',
                'volume': 'Volume ratio > 1.2'
            }
        else:  # SELL
            return {
                'rsi': 'RSI > 50 (idéalement > 70)',
                'macd': 'MACD line < signal line, histogram < 0',
                'ema': 'EMA12 < EMA26 < EMA50',
                'volume': 'Volume ratio > 1.2'
            }