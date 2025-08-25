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
        # Seuils RSI OPTIMISÉS WINRATE pour crypto (plus sélectifs)
        self.oversold_level = 32      # Plus strict pour qualité (35 -> 32)
        self.overbought_level = 68    # Plus strict pour qualité (65 -> 68)
        self.extreme_oversold = 22    # Zone extrême plus stricte (25 -> 22)
        self.extreme_overbought = 78  # Zone extrême plus stricte (75 -> 78)
        self.neutral_low = 40         # Zone neutre élargie anti-whipsaw (45 -> 40)
        self.neutral_high = 60        # Zone neutre élargie anti-whipsaw (55 -> 60)
        
        # NOUVEAUX FILTRES WINRATE
        self.min_trend_confirmation_required = True  # Obligatoire pour éviter contra-trend
        self.min_volume_quality = 60               # Volume minimum pour signaux qualité
        self.min_confluence_for_signal = 55        # Confluence minimum obligatoire
        
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
            'volume_ratio': self.indicators.get('volume_ratio'),
            'adx_14': self.indicators.get('adx_14'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            'market_regime': self.indicators.get('market_regime'),
            'trend_alignment': self.indicators.get('trend_alignment'),
            'atr_percentile': self.indicators.get('atr_percentile')
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
        
        # FILTRES PRÉLIMINAIRES OBLIGATOIRES - WINRATE FOCUS
        
        # Filtre 1: Régime de marché - éviter RSI en ranging/volatile
        market_regime = values.get('market_regime')
        if market_regime not in ['TRENDING_BULL', 'TRENDING_BEAR', 'BREAKOUT_BULL', 'BREAKOUT_BEAR']:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"RSI désactivé en régime {market_regime} - trop de faux signaux",
                "metadata": {"strategy": self.name, "market_regime": market_regime}
            }
            
        # Filtre 2: Volume qualité minimum obligatoire
        volume_quality = values.get('volume_quality_score', 0)
        if not volume_quality or float(volume_quality) < self.min_volume_quality:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Volume qualité insuffisante ({volume_quality}) < {self.min_volume_quality} - RSI non fiable",
                "metadata": {"strategy": self.name, "volume_quality": volume_quality}
            }
            
        # Filtre 3: Confluence minimum obligatoire  
        confluence_score = values.get('confluence_score', 0)
        if not confluence_score or float(confluence_score) < self.min_confluence_for_signal:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": f"Confluence insuffisante ({confluence_score}) < {self.min_confluence_for_signal} - RSI isolé",
                "metadata": {"strategy": self.name, "confluence_score": confluence_score}
            }
        
        # Logique de signal RSI avec seuils optimisés winrate
        if rsi_14 <= self.oversold_level:
            # Zone de survente - signal BUY avec confirmation obligatoire
            
            # Confirmation tendance requise pour BUY
            directional_bias = values.get('directional_bias')
            if directional_bias != 'BULLISH' and market_regime not in ['TRENDING_BULL', 'BREAKOUT_BULL']:
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"RSI survente ({rsi_14:.1f}) mais tendance non haussière - évite contra-trend",
                    "metadata": {"strategy": self.name, "rsi_14": rsi_14, "directional_bias": directional_bias}
                }
            
            signal_side = "BUY"
            zone = "survente extrême" if rsi_14 <= self.extreme_oversold else "survente"
            reason = f"RSI ({rsi_14:.1f}) {zone} + tendance haussière confirmée"
            
            # Bonus réduits mais plus sélectifs
            if rsi_14 <= self.extreme_oversold:
                confidence_boost += 0.25  # Réduit pour éviter sur-confiance
            else:
                confidence_boost += 0.15  # Réduit pour éviter sur-confiance
                
        elif rsi_14 >= self.overbought_level:
            # Zone de surachat - signal SELL avec confirmation obligatoire
            
            # Confirmation tendance requise pour SELL
            directional_bias = values.get('directional_bias')
            if directional_bias != 'BEARISH' and market_regime not in ['TRENDING_BEAR', 'BREAKOUT_BEAR']:
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"RSI surachat ({rsi_14:.1f}) mais tendance non baissière - évite contra-trend",
                    "metadata": {"strategy": self.name, "rsi_14": rsi_14, "directional_bias": directional_bias}
                }
            
            signal_side = "SELL"
            zone = "surachat extrême" if rsi_14 >= self.extreme_overbought else "surachat"
            reason = f"RSI ({rsi_14:.1f}) {zone} + tendance baissière confirmée"
            
            # Bonus réduits mais plus sélectifs
            if rsi_14 >= self.extreme_overbought:
                confidence_boost += 0.25  # Réduit pour éviter sur-confiance
            else:
                confidence_boost += 0.15  # Réduit pour éviter sur-confiance
                
        if signal_side:
            # Utilisation des indicateurs pré-calculés pour ajuster la confiance
            base_confidence = 0.50  # Harmonisé avec autres stratégies
            
            # Ajustement avec momentum_score (format 0-100, 50=neutre) - CRYPTO OPTIMISÉ
            momentum_score = values.get('momentum_score', 50)
            if momentum_score:
                try:
                    momentum_val = float(momentum_score)
                    # SEUILS MOMENTUM STRICTS POUR WINRATE
                    if (signal_side == "BUY" and momentum_val > 65) or \
                       (signal_side == "SELL" and momentum_val < 35):
                        confidence_boost += 0.20  # Réduit mais plus sélectif
                        reason += f" + momentum EXCELLENT ({momentum_val:.0f})"
                    elif (signal_side == "BUY" and momentum_val > 55) or \
                         (signal_side == "SELL" and momentum_val < 45):
                        confidence_boost += 0.10  # Réduit mais plus sélectif
                        reason += f" + momentum favorable ({momentum_val:.0f})"
                    elif (signal_side == "BUY" and momentum_val < 45) or \
                         (signal_side == "SELL" and momentum_val > 55):
                        # REJET DIRECT si momentum contraire
                        return {
                            "side": None,
                            "confidence": 0.0,
                            "strength": "weak",
                            "reason": f"RSI {signal_side} rejeté - momentum contraire ({momentum_val:.0f})",
                            "metadata": {"strategy": self.name, "rsi_14": rsi_14, "momentum_score": momentum_val}
                        }
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
                    # CONFLUENCE DÉJÀ VALIDÉE - juste bonus progressif
                    if confluence_val > 80:
                        confidence_boost += 0.15
                        reason += f" + confluence PARFAITE ({confluence_val:.0f})"
                    elif confluence_val > 70:
                        confidence_boost += 0.12
                        reason += f" + confluence excellente ({confluence_val:.0f})"
                    elif confluence_val > 60:
                        confidence_boost += 0.08
                        reason += f" + confluence forte ({confluence_val:.0f})"
                    # Pas de pénalité car déjà filtré en amont
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
                    
            # FILTRES SUPPLÉMENTAIRES - Volume déjà validé, juste bonus
            volume_quality = values.get('volume_quality_score', 0)
            if volume_quality and float(volume_quality) > 80:
                confidence_boost += 0.08
                reason += f" + volume exceptionnel ({float(volume_quality):.0f})"
                
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
            
            # Filtre ATR pour éviter signaux en marché mort
            atr_percentile = values.get('atr_percentile')
            if atr_percentile and float(atr_percentile) < 25:
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"RSI rejeté - volatilité trop faible ({float(atr_percentile):.0f}%) - marché inactif",
                    "metadata": {"strategy": self.name, "atr_percentile": atr_percentile}
                }
            
            # Calcul confidence final optimisé
            confidence = min(base_confidence * (1 + confidence_boost), 0.90)
            
            # Filtre final plus strict
            if confidence < 0.55:  # Seuil relevé pour qualité
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Signal RSI rejeté - confiance insuffisante ({confidence:.2f} < 0.55)",
                    "metadata": {
                        "strategy": self.name,
                        "symbol": self.symbol,
                        "rejected_signal": signal_side,
                        "rejected_confidence": confidence,
                        "rsi_14": rsi_14
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
            
        # RSI en zone neutre - élargie pour éviter whipsaw
        return {
            "side": None,
            "confidence": 0.0,
            "strength": "weak",
            "reason": f"RSI neutre ({rsi_14:.1f}) - seuils optimisés winrate: {self.oversold_level}/{self.overbought_level}",
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "rsi_14": rsi_14,
                "market_regime": values.get('market_regime')
            }
        }
