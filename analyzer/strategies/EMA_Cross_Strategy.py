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
        self.ema_fast_period = 12      # EMA rapide
        self.ema_slow_period = 26      # EMA lente  
        self.ema_filter_period = 50    # EMA filtre pour tendance générale
        self.cross_confirmation = 3    # Barres de confirmation du croisement
        self.min_separation_pct = 0.3  # Séparation minimum 0.3% (augmenté de 0.1%)
        self.strong_separation_pct = 1.5  # Séparation forte 1.5%
        
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
        base_confidence = 0.3  # Stratégie trend following - conf modérée
        confidence_boost = 0.0
        cross_type = None
        
        # NOUVEAU: Vérifier la direction de la tendance avec EMA50
        trend_filter_passed = False
        if ema_50 is not None:
            if ema_fast_above_slow and current_price > ema_50 and ema_12 > ema_50:
                # Configuration haussière confirmée
                signal_side = "BUY"
                cross_type = "golden_cross"
                reason = f"EMA12 ({ema_12:.2f}) > EMA26 ({ema_26:.2f}) + tendance haussière"
                confidence_boost += 0.20
                trend_filter_passed = True
            elif not ema_fast_above_slow and current_price < ema_50 and ema_12 < ema_50:
                # Configuration baissière confirmée
                signal_side = "SELL"
                cross_type = "death_cross"
                reason = f"EMA12 ({ema_12:.2f}) < EMA26 ({ema_26:.2f}) + tendance baissière"
                confidence_boost += 0.20
                trend_filter_passed = True
            else:
                # Signal contra-trend ou ambigu - REJETER
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Croisement EMA contra-trend ou ambigu (prix vs EMA50)",
                    "metadata": {
                        "strategy": self.name,
                        "symbol": self.symbol,
                        "ema_12": ema_12,
                        "ema_26": ema_26,
                        "ema_50": ema_50,
                        "current_price": current_price,
                        "cross_type": "golden_cross" if ema_fast_above_slow else "death_cross",
                        "rejected": "contra_trend"
                    }
                }
        else:
            # Pas d'EMA50 disponible - utiliser l'ancienne logique mais plus stricte
            if ema_fast_above_slow:
                signal_side = "BUY"
                cross_type = "golden_cross"
                reason = f"EMA12 ({ema_12:.2f}) > EMA26 ({ema_26:.2f})"
                confidence_boost += 0.10  # Réduit de 0.15
            else:
                signal_side = "SELL"
                cross_type = "death_cross"
                reason = f"EMA12 ({ema_12:.2f}) < EMA26 ({ema_26:.2f})"
                confidence_boost += 0.10  # Réduit de 0.15
            
        # Bonus selon la force de la séparation - SEUILS PLUS STRICTS
        if ema_distance_pct >= self.strong_separation_pct:  # 1.5% au lieu de 1.0%
            confidence_boost += 0.20
            reason += f" - séparation FORTE ({ema_distance_pct:.2f}%)"
        elif ema_distance_pct >= 0.8:
            confidence_boost += 0.12
            reason += f" - séparation correcte ({ema_distance_pct:.2f}%)"
        elif ema_distance_pct >= 0.5:
            confidence_boost += 0.06
            reason += f" - séparation acceptable ({ema_distance_pct:.2f}%)"
        else:
            # Séparation faible - pénalité
            confidence_boost -= 0.05
            reason += f" - ATTENTION: séparation faible ({ema_distance_pct:.2f}%)"
            
        # Confirmation supplémentaire avec EMA99 si disponible
        ema_99 = values.get('ema_99')
        if ema_99 is not None:
            try:
                ema99_val = float(ema_99)
                if signal_side == "BUY" and current_price > ema99_val:
                    confidence_boost += 0.08
                    reason += f" + tendance LT haussière"
                elif signal_side == "SELL" and current_price < ema99_val:
                    confidence_boost += 0.08
                    reason += f" + tendance LT baissière"
                elif (signal_side == "BUY" and current_price < ema99_val) or \
                     (signal_side == "SELL" and current_price > ema99_val):
                    confidence_boost -= 0.10
                    reason += f" MAIS contra-trend LT"
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
                
                if (signal_side == "BUY" and macd_cross and macd > 0) or \
                   (signal_side == "SELL" and not macd_cross and macd < 0):
                    confidence_boost += 0.18  # Augmenté si MACD aligné ET bon côté du zéro
                    reason += " + MACD PARFAITEMENT aligné"
                elif (signal_side == "BUY" and macd_cross) or \
                     (signal_side == "SELL" and not macd_cross):
                    confidence_boost += 0.10
                    reason += " + MACD confirme"
                else:
                    confidence_boost -= 0.15  # Pénalité augmentée
                    reason += " ATTENTION: MACD diverge"
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
                
        # Confirmation avec trend_strength (VARCHAR: absent/weak/moderate/strong/very_strong)
        trend_strength = values.get('trend_strength')
        if trend_strength is not None:
            trend_str = str(trend_strength).lower()
            if trend_str in ['strong', 'very_strong']:
                confidence_boost += 0.12
                reason += f" + tendance {trend_str}"
            elif trend_str == 'moderate':
                confidence_boost += 0.08
                reason += f" + tendance {trend_str}"
                
        # Confirmation avec directional_bias
        directional_bias = values.get('directional_bias')
        if directional_bias:
            if (signal_side == "BUY" and directional_bias == "BULLISH") or \
               (signal_side == "SELL" and directional_bias == "BEARISH"):
                confidence_boost += 0.10
                reason += f" + bias {directional_bias}"
                
        # Confirmation avec trend_alignment (toutes les EMA alignées) - format décimal
        trend_alignment = values.get('trend_alignment')
        if trend_alignment is not None:
            try:
                alignment = float(trend_alignment)
                # SEUIL PLUS STRICT pour alignement
                if abs(alignment) > 0.5:  # Plus strict: 0.5 au lieu de 0.3
                    confidence_boost += 0.15
                    reason += " + EMA FORTEMENT alignées"
                elif abs(alignment) > 0.3:
                    confidence_boost += 0.08
                    reason += " + EMA alignées"
                else:
                    confidence_boost -= 0.05
                    reason += " mais EMA peu alignées"
            except (ValueError, TypeError):
                pass
                
        # Momentum pour éviter signaux contre-tendance (format 0-100, 50=neutre)
        momentum_score = values.get('momentum_score')
        if momentum_score is not None:
            try:
                momentum = float(momentum_score)
                # MOMENTUM PLUS STRICT
                if (signal_side == "BUY" and momentum > 60) or \
                   (signal_side == "SELL" and momentum < 40):
                    confidence_boost += 0.12
                    reason += " + momentum FORT"
                elif (signal_side == "BUY" and momentum > 52) or \
                     (signal_side == "SELL" and momentum < 48):
                    confidence_boost += 0.06
                    reason += " + momentum favorable"
                elif (signal_side == "BUY" and momentum < 40) or \
                     (signal_side == "SELL" and momentum > 60):
                    confidence_boost -= 0.20  # Pénalité forte
                    reason += " ATTENTION: momentum CONTRAIRE"
            except (ValueError, TypeError):
                pass
                
        # Volume pour confirmation
        volume_ratio = values.get('volume_ratio')
        if volume_ratio is not None:
            try:
                vol_ratio = float(volume_ratio)
                if vol_ratio >= 1.2:
                    confidence_boost += 0.08
                    reason += f" + volume élevé ({vol_ratio:.1f}x)"
                elif vol_ratio >= 1.1:
                    confidence_boost += 0.05
                    reason += f" + volume modéré ({vol_ratio:.1f}x)"
            except (ValueError, TypeError):
                pass
                
        # Volume quality (champ DB: volume_quality_score - format 0-100)
        volume_quality_score = values.get('volume_quality_score')
        if volume_quality_score is not None:
            try:
                vol_quality = float(volume_quality_score)
                if vol_quality > 70:
                    confidence_boost += 0.08
                    reason += f" + volume qualité ({vol_quality:.0f})"
                elif vol_quality > 50:
                    confidence_boost += 0.05
                    reason += f" + volume correct ({vol_quality:.0f})"
            except (ValueError, TypeError):
                pass
                
        # Signal strength (VARCHAR: WEAK/MODERATE/STRONG)
        signal_strength_calc = values.get('signal_strength')
        if signal_strength_calc is not None:
            sig_str = str(signal_strength_calc).upper()
            if sig_str == 'STRONG':
                confidence_boost += 0.10
                reason += " + signal fort"
            elif sig_str == 'MODERATE':
                confidence_boost += 0.05
                reason += " + signal modéré"
                
        confluence_score = values.get('confluence_score')
        if confluence_score is not None:
            try:
                confluence = float(confluence_score)
                # CONFLUENCE PLUS STRICTE
                if confluence > 75:  # Seuil augmenté
                    confidence_boost += 0.20
                    reason += f" + confluence EXCELLENTE ({confluence:.0f})"
                elif confluence > 65:  # Seuil augmenté
                    confidence_boost += 0.12
                    reason += f" + confluence élevée ({confluence:.0f})"
                elif confluence > 55:  # Seuil augmenté
                    confidence_boost += 0.06
                    reason += f" + confluence correcte ({confluence:.0f})"
                elif confluence < 45:  # Pénalité si faible
                    confidence_boost -= 0.10
                    reason += f" mais confluence FAIBLE ({confluence:.0f})"
            except (ValueError, TypeError):
                pass
                
        # NOUVEAU: Filtre final - rejeter si confidence trop faible
        raw_confidence = base_confidence * (1 + confidence_boost)
        if raw_confidence < 0.40:  # Seuil minimum 40%
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Signal EMA rejeté - confiance insuffisante ({raw_confidence:.2f} < 0.40)",
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "rejected_signal": signal_side,
                    "raw_confidence": raw_confidence,
                    "ema_separation": ema_distance_pct
                }
            }
        
        confidence = self.calculate_confidence(base_confidence, 1 + confidence_boost)
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
                "ema_separation_pct": ema_distance_pct,
                "macd_line": macd_line,
                "macd_signal": macd_signal,
                "macd_histogram": macd_histogram,
                "trend_strength": trend_strength,
                "directional_bias": directional_bias,
                "trend_alignment": trend_alignment,
                "momentum_score": momentum_score,
                "volume_ratio": volume_ratio,
                "confluence_score": confluence_score
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
