"""
RSI_Cross_Strategy - Stratégie basée sur les positions du RSI dans les zones de survente/surachat.
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class RSI_Cross_Strategy(BaseStrategy):
    """
    Stratégie utilisant le RSI et les indicateurs pré-calculés.
    
    Signaux générés:
    - BUY: RSI en zone de survente avec momentum et tendance favorables
    - SELL: RSI en zone de surachat avec momentum et tendance favorables
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Seuils RSI réalistes pour crypto (AJUSTÉS POUR PLUS DE SIGNAUX)
        self.oversold_level = 35      # Crypto oversold accessible (était 25)
        self.overbought_level = 65    # Crypto overbought accessible (était 75)
        self.extreme_oversold = 25    # RSI extreme accessible (était 18)
        self.extreme_overbought = 75  # RSI extreme accessible (était 82)
        self.neutral_low = 45         # Zone neutre basse resserrée (était 38)
        self.neutral_high = 55        # Zone neutre haute resserrée (était 62)
        
    def _get_current_values(self) -> Dict[str, Optional[float]]:
        """Récupère les valeurs actuelles des indicateurs pré-calculés."""
        return {
            'rsi_14': self.indicators.get('rsi_14'),
            'rsi_21': self.indicators.get('rsi_21'),
            'signal_strength': self.indicators.get('signal_strength'),
            'momentum_score': self.indicators.get('momentum_score'),
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'confluence_score': self.indicators.get('confluence_score'),
            'pattern_confidence': self.indicators.get('pattern_confidence'),
            'volume_quality_score': self.indicators.get('volume_quality_score'),
            'adx_14': self.indicators.get('adx_14'),
            'volatility_regime': self.indicators.get('volatility_regime')
        }
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur le RSI et les indicateurs pré-calculés.
        """
        if not self.validate_data():
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "Données insuffisantes",
                "metadata": {}
            }
            
        values = self._get_current_values()
        
        # Vérification des données essentielles
        rsi_14 = values['rsi_14']
        if rsi_14 is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "RSI non disponible",
                "metadata": {"strategy": self.name}
            }
            
        signal_side = None
        reason = ""
        confidence_boost = 0.0
        
        # Logique de signal basée sur RSI et indicateurs pré-calculés
        if rsi_14 <= self.oversold_level:
            # Zone de survente - chercher signal BUY
            signal_side = "BUY"
            zone = "survente extrême" if rsi_14 <= self.extreme_oversold else "survente"
            reason = f"RSI ({rsi_14:.1f}) en zone de {zone}"
            
            # Bonus pour survente extrême (AJUSTÉ)
            if rsi_14 <= self.extreme_oversold:
                confidence_boost += 0.35  # Boost majeur amélioré (était 0.30)
            else:
                confidence_boost += 0.20  # Boost modéré amélioré (était 0.15)
                
        elif rsi_14 >= self.overbought_level:
            # Zone de surachat - chercher signal SELL
            signal_side = "SELL"
            zone = "surachat extrême" if rsi_14 >= self.extreme_overbought else "surachat"
            reason = f"RSI ({rsi_14:.1f}) en zone de {zone}"
            
            # Bonus pour surachat extrême (AJUSTÉ)
            if rsi_14 >= self.extreme_overbought:
                confidence_boost += 0.35  # Boost majeur amélioré (était 0.30)
            else:
                confidence_boost += 0.20  # Boost modéré amélioré (était 0.15)
                
        if signal_side:
            # Utilisation des indicateurs pré-calculés pour ajuster la confiance
            base_confidence = 0.40  # Réduit à 0.40 pour plus d'accessibilité (était 0.50)
            
            # Ajustement avec momentum_score (format 0-100, 50=neutre) - CRYPTO OPTIMISÉ
            momentum_score = values.get('momentum_score', 50)
            if momentum_score:
                try:
                    momentum_val = float(momentum_score)
                    # SEUILS CRYPTO ASSOUPLIS
                    if (signal_side == "BUY" and momentum_val > 60) or \
                       (signal_side == "SELL" and momentum_val < 40):
                        confidence_boost += 0.25  # Amélioré (était 0.22)
                        reason += f" avec momentum EXCELLENT ({momentum_val:.0f})"
                    elif (signal_side == "BUY" and momentum_val > 52) or \
                         (signal_side == "SELL" and momentum_val < 48):
                        confidence_boost += 0.15  # Amélioré (était 0.12)
                        reason += f" avec momentum favorable ({momentum_val:.0f})"
                    elif (signal_side == "BUY" and momentum_val < 40) or \
                         (signal_side == "SELL" and momentum_val > 60):
                        confidence_boost -= 0.15  # Réduit (était -0.25)
                        reason += f" ATTENTION: momentum contraire ({momentum_val:.0f})"
                except (ValueError, TypeError):
                    pass
                    
            # Ajustement avec trend_strength (VARCHAR: weak/moderate/strong/very_strong/extreme)
            trend_strength = values.get('trend_strength')
            if trend_strength:
                trend_str = str(trend_strength).lower()
                if trend_str in ['extreme', 'very_strong']:
                    confidence_boost += 0.15
                    reason += f" et tendance {trend_str}"
                elif trend_str == 'strong':
                    confidence_boost += 0.10
                    reason += f" et tendance {trend_str}"
                elif trend_str == 'moderate':
                    confidence_boost += 0.05
                    reason += f" et tendance {trend_str}"
                
            # Ajustement avec directional_bias
            directional_bias = values.get('directional_bias')
            if directional_bias:
                if (signal_side == "BUY" and directional_bias == "BULLISH") or \
                   (signal_side == "SELL" and directional_bias == "BEARISH"):
                    confidence_boost += 0.1
                    reason += " confirmé par bias directionnel"
                    
            # Ajustement avec confluence_score (format 0-100)
            confluence_score = values.get('confluence_score', 0)
            if confluence_score:
                try:
                    confluence_val = float(confluence_score)
                    # CONFLUENCE CRYPTO RÉALISTE
                    if confluence_val > 75:  # Excellence accessible (était 85)
                        confidence_boost += 0.25
                        reason += f" avec confluence PARFAITE ({confluence_val:.0f})"
                    elif confluence_val > 60:  # Très bon accessible (était 75)
                        confidence_boost += 0.18
                        reason += f" avec confluence EXCELLENTE ({confluence_val:.0f})"
                    elif confluence_val > 50:  # Standard accessible (était 65)
                        confidence_boost += 0.12  # Amélioré (était 0.10)
                        reason += f" avec confluence correcte ({confluence_val:.0f})"
                    elif confluence_val < 45:  # Seuil pénalité réduit (était 55)
                        confidence_boost -= 0.10  # Pénalité réduite (était -0.15)
                        reason += f" mais confluence insuffisante ({confluence_val:.0f})"
                except (ValueError, TypeError):
                    pass
                
            # Ajustement avec signal_strength pré-calculé (VARCHAR: WEAK/MODERATE/STRONG)
            signal_strength_calc = values.get('signal_strength')
            if signal_strength_calc:
                sig_str = str(signal_strength_calc).upper()
                if sig_str == 'STRONG':
                    confidence_boost += 0.1
                    reason += " + signal fort"
                elif sig_str == 'MODERATE':
                    confidence_boost += 0.05
                    reason += " + signal modéré"
                
            # Confirmation avec RSI 21 pour multi-timeframe
            rsi_21 = values.get('rsi_21')
            if rsi_21:
                if (signal_side == "BUY" and rsi_21 <= self.oversold_level) or \
                   (signal_side == "SELL" and rsi_21 >= self.overbought_level):
                    confidence_boost += 0.1
                    reason += " confirmé sur RSI 21"
                    
            # FILTRES SUPPLÉMENTAIRES CRYPTO (AJUSTÉS)
            # Filtre volume quality
            volume_quality = values.get('volume_quality_score', 0)
            if volume_quality and float(volume_quality) < 55:  # Seuil réduit (était 60)
                confidence_boost -= 0.08  # Pénalité réduite (était -0.12)
                reason += f" mais volume faible ({float(volume_quality):.0f})"
            elif volume_quality and float(volume_quality) > 75:  # Seuil réduit (était 80)
                confidence_boost += 0.10  # Bonus amélioré (était 0.08)
                reason += f" avec volume excellent ({float(volume_quality):.0f})"
                
            # Vérifier divergence RSI 14/21 pour confirmation (AJUSTÉ)
            rsi_21 = values.get('rsi_21')
            if rsi_21:
                rsi_diff = abs(rsi_14 - rsi_21)
                if rsi_diff < 6:  # Tolérance augmentée (était 5)
                    confidence_boost += 0.15  # Bonus amélioré (était 0.12)
                    reason += " avec RSI multi-TF alignés"
                elif rsi_diff > 18:  # Tolérance augmentée (était 15)
                    confidence_boost -= 0.06  # Pénalité réduite (était -0.08)
                    reason += " avec divergence multi-TF modérée"
                    
            # Filtre ADX pour éviter signaux en ranging (AJUSTÉ)
            adx = values.get('adx_14')
            if adx and float(adx) < 18:  # Seuil réduit (était 20)
                confidence_boost -= 0.10  # Pénalité réduite (était -0.15)
                reason += f" mais ADX faible ({float(adx):.0f}) - ranging"
            elif adx and float(adx) > 28:  # Seuil réduit (était 30)
                confidence_boost += 0.12  # Bonus amélioré (était 0.10)
                reason += f" avec ADX fort ({float(adx):.0f}) - trending"
            
            # NOUVEAU: Filtre final CRYPTO accessible
            raw_confidence = base_confidence * (1 + confidence_boost)
            if raw_confidence < 0.35:  # Seuil crypto réduit (était 0.45)
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Signal RSI rejeté - confiance insuffisante ({raw_confidence:.2f} < 0.35)",
                    "metadata": {
                        "strategy": self.name,
                        "symbol": self.symbol,
                        "rejected_signal": signal_side,
                        "raw_confidence": raw_confidence,
                        "rsi_14": rsi_14,
                        "volume_quality": volume_quality,
                        "adx_14": adx
                    }
                }
            
            confidence = self.calculate_confidence(base_confidence, 1.0 + confidence_boost)
            strength = self.get_strength_from_confidence(confidence)
            
            return {
                "side": signal_side,
                "confidence": confidence,
                "strength": strength,
                "reason": reason,
                "metadata": {
                    "strategy": self.name,
                    "symbol": self.symbol,
                    "rsi_14": rsi_14,
                    "rsi_21": rsi_21,
                    "momentum_score": momentum_score,
                    "trend_strength": trend_strength,
                    "directional_bias": directional_bias,
                    "confluence_score": confluence_score,
                    "signal_strength_calc": signal_strength_calc,
                    "volume_quality_score": volume_quality,
                    "adx_14": adx
                }
            }
            
        return {
            "side": None,
            "confidence": 0.0,
            "strength": "weak",
            "reason": f"RSI en zone neutre ({rsi_14:.1f}) - pas de signal (seuils ajustés: {self.oversold_level}/{self.overbought_level})",
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "rsi_14": rsi_14
            }
        }
