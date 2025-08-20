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
        # Seuils RSI optimisés pour crypto (plus réactifs)
        self.oversold_level = 25      # Crypto oversold (plus bas que stocks)
        self.overbought_level = 75    # Crypto overbought (plus haut que stocks)  
        self.extreme_oversold = 18    # RSI extreme oversold crypto
        self.extreme_overbought = 82  # RSI extreme overbought crypto
        self.neutral_low = 38         # Zone neutre basse ajustée
        self.neutral_high = 62        # Zone neutre haute ajustée
        
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
            
            # Bonus pour survente extrême
            if rsi_14 <= self.extreme_oversold:
                confidence_boost += 0.30  # Boost majeur pour extrême
            else:
                confidence_boost += 0.15  # Boost modéré
                
        elif rsi_14 >= self.overbought_level:
            # Zone de surachat - chercher signal SELL
            signal_side = "SELL"
            zone = "surachat extrême" if rsi_14 >= self.extreme_overbought else "surachat"
            reason = f"RSI ({rsi_14:.1f}) en zone de {zone}"
            
            # Bonus pour surachat extrême
            if rsi_14 >= self.extreme_overbought:
                confidence_boost += 0.30  # Boost majeur pour extrême
            else:
                confidence_boost += 0.15  # Boost modéré
                
        if signal_side:
            # Utilisation des indicateurs pré-calculés pour ajuster la confiance
            base_confidence = 0.50  # Standardisé à 0.50 pour équité avec autres stratégies
            
            # Ajustement avec momentum_score (format 0-100, 50=neutre) - CRYPTO OPTIMISÉ
            momentum_score = values.get('momentum_score', 50)
            if momentum_score:
                try:
                    momentum_val = float(momentum_score)
                    # SEUILS CRYPTO PLUS STRICTS
                    if (signal_side == "BUY" and momentum_val > 65) or \
                       (signal_side == "SELL" and momentum_val < 35):
                        confidence_boost += 0.22
                        reason += f" avec momentum EXCELLENT ({momentum_val:.0f})"
                    elif (signal_side == "BUY" and momentum_val > 55) or \
                         (signal_side == "SELL" and momentum_val < 45):
                        confidence_boost += 0.12
                        reason += f" avec momentum favorable ({momentum_val:.0f})"
                    elif (signal_side == "BUY" and momentum_val < 35) or \
                         (signal_side == "SELL" and momentum_val > 65):
                        confidence_boost -= 0.25  # Pénalité plus forte
                        reason += f" ATTENTION: momentum CONTRAIRE ({momentum_val:.0f})"
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
                    # CONFLUENCE CRYPTO ULTRA-STRICTE
                    if confluence_val > 85:  # Excellence exigée
                        confidence_boost += 0.25
                        reason += f" avec confluence PARFAITE ({confluence_val:.0f})"
                    elif confluence_val > 75:  # Très bon minimum
                        confidence_boost += 0.18
                        reason += f" avec confluence EXCELLENTE ({confluence_val:.0f})"
                    elif confluence_val > 65:  # Standard crypto
                        confidence_boost += 0.10
                        reason += f" avec confluence correcte ({confluence_val:.0f})"
                    elif confluence_val < 55:  # Pénalité plus forte
                        confidence_boost -= 0.15
                        reason += f" mais confluence INSUFFISANTE ({confluence_val:.0f})"
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
                    
            # FILTRES SUPPLÉMENTAIRES CRYPTO
            # Filtre volume quality
            volume_quality = values.get('volume_quality_score', 0)
            if volume_quality and float(volume_quality) < 60:
                confidence_boost -= 0.12
                reason += f" mais volume FAIBLE ({float(volume_quality):.0f})"
            elif volume_quality and float(volume_quality) > 80:
                confidence_boost += 0.08
                reason += f" avec volume EXCELLENT ({float(volume_quality):.0f})"
                
            # Vérifier divergence RSI 14/21 pour confirmation
            rsi_21 = values.get('rsi_21')
            if rsi_21:
                rsi_diff = abs(rsi_14 - rsi_21)
                if rsi_diff < 5:  # RSI alignés = forte confluence
                    confidence_boost += 0.12
                    reason += " avec RSI multi-TF alignés"
                elif rsi_diff > 15:  # RSI divergents = attention
                    confidence_boost -= 0.08
                    reason += " ATTENTION: divergence multi-TF"
                    
            # Filtre ADX pour éviter signaux en ranging
            adx = values.get('adx_14')
            if adx and float(adx) < 20:  # Market ranging
                confidence_boost -= 0.15
                reason += f" mais ADX faible ({float(adx):.0f}) - market ranging"
            elif adx and float(adx) > 30:  # Trend fort
                confidence_boost += 0.10
                reason += f" avec ADX fort ({float(adx):.0f}) - trending"
            
            # NOUVEAU: Filtre final CRYPTO plus strict
            raw_confidence = base_confidence * (1 + confidence_boost)
            if raw_confidence < 0.45:  # Seuil crypto minimum 45%
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Signal RSI rejeté - confiance crypto insuffisante ({raw_confidence:.2f} < 0.45)",
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
            "reason": f"RSI en zone neutre ({rsi_14:.1f}) - pas de signal crypto (seuils: {self.oversold_level}/{self.overbought_level})",
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "rsi_14": rsi_14
            }
        }
