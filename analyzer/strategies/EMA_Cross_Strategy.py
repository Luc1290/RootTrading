"""
EMA_Cross_Strategy - Stratégie basée sur les croisements d'EMA.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class EMA_Cross_Strategy(BaseStrategy):
    """
    Stratégie utilisant les croisements d'EMA pour détecter les changements de tendance.
    
    Signaux générés:
    - BUY: EMA rapide croise au-dessus EMA lente + confirmations haussières
    - SELL: EMA rapide croise en-dessous EMA lente + confirmations baissières
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Configuration des EMA - OPTIMISÉES
        self.ema_fast_period = 12      # EMA rapide (info seulement)
        self.ema_slow_period = 26      # EMA lente (info seulement)
        self.ema_filter_period = 50    # EMA filtre pour tendance générale (info seulement)
        # Note: utilise directement ema_12, ema_26, ema_50 de la DB
        self.min_separation_pct = 0.25  # Séparation minimum 0.25% (filtre bruit crypto)
        self.strong_separation_pct = 1.0  # Séparation forte 1.0%
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs EMA."""
        return {
            # EMA disponibles
            'ema_7': self.indicators.get('ema_7'),
            'ema_12': self.indicators.get('ema_12'),
            'ema_26': self.indicators.get('ema_26'),
            'ema_50': self.indicators.get('ema_50'),
            'ema_99': self.indicators.get('ema_99'),
            # SMA pour comparaison
            'sma_20': self.indicators.get('sma_20'),
            'sma_50': self.indicators.get('sma_50'),
            # MACD (basé sur EMA 12/26)
            'macd_line': self.indicators.get('macd_line'),
            'macd_signal': self.indicators.get('macd_signal'),
            'macd_histogram': self.indicators.get('macd_histogram'),
            'macd_zero_cross': self.indicators.get('macd_zero_cross'),
            'macd_signal_cross': self.indicators.get('macd_signal_cross'),
            'macd_trend': self.indicators.get('macd_trend'),
            # Contexte tendance
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'trend_alignment': self.indicators.get('trend_alignment'),
            'momentum_score': self.indicators.get('momentum_score'),
            # Volume pour confirmation
            'volume_ratio': self.indicators.get('volume_ratio'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            # Confluence
            'signal_strength': self.indicators.get('signal_strength'),
            'confluence_score': self.indicators.get('confluence_score')
        }
        
    def _get_current_price(self) -> Optional[float]:
        """Récupère le prix actuel depuis les données OHLCV."""
        try:
            if self.data and 'close' in self.data and self.data['close']:
                return float(self.data['close'][-1])
        except (IndexError, ValueError, TypeError):
            pass
        return None
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur les croisements d'EMA.
        """
        if not self.validate_data():
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Données insuffisantes",
                "metadata": {"strategy": self.name}
            }
            
        values = self._get_current_values()
        current_price = self._get_current_price()
        
        # Vérification des EMA essentielles (12 et 26 pour logique classique)
        try:
            ema_12 = float(values['ema_12']) if values['ema_12'] is not None else None
            ema_26 = float(values['ema_26']) if values['ema_26'] is not None else None
            ema_50 = float(values['ema_50']) if values['ema_50'] is not None else None
        except (ValueError, TypeError) as e:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Erreur conversion EMA: {e}",
                "metadata": {"strategy": self.name}
            }
            
        if ema_12 is None or ema_26 is None or current_price is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "EMA 12/26 ou prix non disponibles",
                "metadata": {"strategy": self.name}
            }
            
        # Analyse du croisement EMA 12/26
        ema_fast_above_slow = ema_12 > ema_26
        ema_distance_pct = abs(ema_12 - ema_26) / ema_26 * 100
        
        # Vérification que les EMA ne sont pas trop proches (éviter faux signaux)
        if ema_distance_pct < self.min_separation_pct:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"EMA trop proches ({ema_distance_pct:.2f}%) - pas de signal clair",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "ema_12": ema_12,
                    "ema_26": ema_26,
                    "separation_pct": ema_distance_pct,
                    "min_required": self.min_separation_pct
                }
            }
            
        signal_side = None
        reason = ""
        base_confidence = 0.65  # Standardisé à 0.65 pour équité avec autres stratégies
        confidence_boost = 0.0
        cross_type = None
        
        # Détection du croisement EMA de base
        if ema_fast_above_slow:
            signal_side = "BUY"
            cross_type = "golden_cross"
            reason = f"EMA12 ({ema_12:.2f}) > EMA26 ({ema_26:.2f})"
            confidence_boost += 0.15  # Base pour croisement
        else:
            signal_side = "SELL"
            cross_type = "death_cross"
            reason = f"EMA12 ({ema_12:.2f}) < EMA26 ({ema_26:.2f})"
            confidence_boost += 0.15  # Base pour croisement
            
        # Filtre EMA50 comme bonus/malus (pas rejet)
        if ema_50 is not None:
            if signal_side == "BUY" and current_price > ema_50 and ema_12 > ema_50:
                # Configuration haussière parfaite
                confidence_boost += 0.20
                reason += " + tendance haussière EMA50"
            elif signal_side == "SELL" and current_price < ema_50 and ema_12 < ema_50:
                # Configuration baissière parfaite  
                confidence_boost += 0.20
                reason += " + tendance baissière EMA50"
            elif (signal_side == "BUY" and (current_price < ema_50 or ema_12 < ema_50)) or \
                 (signal_side == "SELL" and (current_price > ema_50 or ema_12 > ema_50)):
                # Contra-trend = malus mais pas rejet
                confidence_boost -= 0.15
                reason += " MAIS contra-tendance EMA50"
            
        # Bonus selon la force de la séparation - SEUILS RENFORCÉS
        cross_quality = "weak"
        if ema_distance_pct >= self.strong_separation_pct:  # 1.0%
            confidence_boost += 0.20
            reason += f" - séparation FORTE ({ema_distance_pct:.2f}%)"
            cross_quality = "strong"
        elif ema_distance_pct >= 0.5:
            confidence_boost += 0.12
            reason += f" - séparation correcte ({ema_distance_pct:.2f}%)"
            cross_quality = "moderate"
        elif ema_distance_pct >= 0.35:
            confidence_boost += 0.06
            reason += f" - séparation acceptable ({ema_distance_pct:.2f}%)"
            cross_quality = "acceptable"
        else:
            # Séparation faible
            confidence_boost -= 0.05
            reason += f" - séparation faible ({ema_distance_pct:.2f}%)"
            cross_quality = "weak"
            
        # Confirmation stricte avec EMA99 (rejet si forte divergence LT)
        ema_99 = values.get('ema_99')
        if ema_99 is not None:
            try:
                ema99_val = float(ema_99)
                
                # Rejet si contre-tendance LT franche
                if signal_side == "BUY" and current_price < ema99_val * 0.98:  # 2% sous EMA99
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Rejet BUY: prix {current_price:.2f} trop sous EMA99 {ema99_val:.2f} (contra-trend LT)",
                        "metadata": {
                            "strategy": self.name,
                            "symbol": self.symbol,
                            "current_price": current_price,
                            "ema_99": ema99_val,
                            "rejected": "contra_trend_lt"
                        }
                    }
                elif signal_side == "SELL" and current_price > ema99_val * 1.02:  # 2% au-dessus EMA99
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Rejet SELL: prix {current_price:.2f} trop au-dessus EMA99 {ema99_val:.2f} (contra-trend LT)",
                        "metadata": {
                            "strategy": self.name,
                            "symbol": self.symbol,
                            "current_price": current_price,
                            "ema_99": ema99_val,
                            "rejected": "contra_trend_lt"
                        }
                    }
                
                # Confirmations EMA99
                if signal_side == "BUY" and current_price > ema99_val:
                    confidence_boost += 0.08
                    reason += f" + tendance LT haussière"
                elif signal_side == "SELL" and current_price < ema99_val:
                    confidence_boost += 0.08
                    reason += f" + tendance LT baissière"
                else:
                    # Léger malus mais pas rejet (cas limites)
                    confidence_boost -= 0.05
                    reason += f" mais neutre LT"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec MACD (basé sur EMA 12/26)
        macd_line = values.get('macd_line')
        macd_signal = values.get('macd_signal')
        macd_histogram = values.get('macd_histogram')
        
        if macd_line is not None and macd_signal is not None:
            try:
                macd = float(macd_line)
                macd_sig = float(macd_signal)
                macd_cross = macd > macd_sig
                
                # Rejet STRICT si MACD diverge franchement
                if (signal_side == "BUY" and macd < 0 and not macd_cross) or \
                   (signal_side == "SELL" and macd > 0 and macd_cross):
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Rejet croisement EMA: MACD diverge franchement ({macd:.4f})",
                        "metadata": {
                            "strategy": self.name,
                            "symbol": self.symbol,
                            "macd_line": macd,
                            "macd_signal": macd_sig,
                            "rejected": "macd_divergence"
                        }
                    }
                
                # Confirmations MACD
                if (signal_side == "BUY" and macd_cross and macd > 0) or \
                   (signal_side == "SELL" and not macd_cross and macd < 0):
                    confidence_boost += 0.18
                    reason += " + MACD PARFAITEMENT aligné"
                elif (signal_side == "BUY" and macd_cross) or \
                     (signal_side == "SELL" and not macd_cross):
                    confidence_boost += 0.10
                    reason += " + MACD confirme"
                else:
                    confidence_boost -= 0.10  # Pénalité réduite car les cas graves sont rejetés
                    reason += " MAIS MACD mitigé"
            except (ValueError, TypeError):
                pass
                
        # Confirmation avec histogramme MACD
        if macd_histogram is not None:
            try:
                histogram = float(macd_histogram)
                if (signal_side == "BUY" and histogram > 0) or \
                   (signal_side == "SELL" and histogram < 0):
                    confidence_boost += 0.08
                    reason += " + histogram MACD"
            except (ValueError, TypeError):
                pass
                
                
                
                
                
        # Volume pour confirmation - SEUILS RENFORCÉS
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= 2.0:
                    confidence_boost += 0.12
                    reason += f" + volume EXCEPTIONNEL ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.5:
                    confidence_boost += 0.08
                    reason += f" + volume élevé ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.2:
                    confidence_boost += 0.04
                    reason += f" + volume modéré ({vol_ratio:.1f}x)"
                # Pas de bonus sous 1.2x
            except (ValueError, TypeError):
                pass
                
                
                
                
        # Calcul final de confidence avec clamp
        confidence = min(1.0, base_confidence * (1 + confidence_boost))
        
        # Filtre final - rejeter si confidence trop faible
        if confidence < 0.35:  # Seuil minimum abaissé
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Signal EMA rejeté - confiance insuffisante ({confidence:.2f} < 0.35)",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "rejected_signal": signal_side,
                    "rejected_confidence": confidence,
                    "ema_separation": ema_distance_pct
                }
            }
        strength = self.get_strength_from_confidence(confidence)
        
        return {
            "side": signal_side,
            "confidence": confidence,
            "strength": strength,
            "reason": reason,
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "current_price": current_price,
                "ema_12": ema_12,
                "ema_26": ema_26,
                "ema_50": ema_50,
                "cross_type": cross_type,
                "cross_quality": cross_quality,
                "ema_separation_pct": ema_distance_pct,
                "macd_line": macd_line,
                "macd_signal": macd_signal,
                "macd_histogram": macd_histogram,
                "volume_ratio": volume_ratio
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que tous les indicateurs EMA requis sont présents."""
        if not super().validate_data():
            return False
            
        required = ['ema_12', 'ema_26']
        
        for indicator in required:
            if indicator not in self.indicators:
                logger.warning(f"{self.name}: Indicateur manquant: {indicator}")
                return False
            if self.indicators[indicator] is None:
                logger.warning(f"{self.name}: Indicateur null: {indicator}")
                return False
                
        # Vérifier aussi qu'on a des données de prix
        if not self.data or 'close' not in self.data or not self.data['close']:
            logger.warning(f"{self.name}: Données de prix manquantes")
            return False
            
        return True
