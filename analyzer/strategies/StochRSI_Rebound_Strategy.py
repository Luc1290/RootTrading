"""
StochRSI_Rebound_Strategy - Stratégie basée sur les signaux StochRSI pré-calculés.
OPTIMISÉE POUR CRYPTO SPOT INTRADAY
"""

from typing import Dict, Any, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class StochRSI_Rebound_Strategy(BaseStrategy):
    """
    Stratégie utilisant les signaux StochRSI et indicateurs pré-calculés.
    
    Signaux générés:
    - BUY: StochRSI en zone de survente avec signaux favorables
    - SELL: StochRSI en zone de surachat avec signaux favorables
    """
    
    def __init__(self, symbol: str, data: Dict[str, Any], indicators: Dict[str, Any]):
        super().__init__(symbol, data, indicators)
        # Seuils StochRSI - RESSERREÉS POUR QUALITÉ
        self.oversold_zone = 28       # Resserré (30 -> 28)
        self.overbought_zone = 72     # Resserré (70 -> 72)
        self.extreme_oversold = 15    # Zone extrême
        self.extreme_overbought = 85  # Zone extrême
        self.neutral_low = 35         # Zone neutre basse resserrée (était 40)
        self.neutral_high = 65        # Zone neutre haute resserrée (était 60)
        
        # Seuils RSI assouplis crypto
        self.rsi_oversold = 35        # RSI survente accessible (était 25)
        self.rsi_overbought = 65      # RSI surachat accessible (était 75)
        self.rsi_oversold_strong = 30 # RSI survente forte accessible (était 20)
        self.rsi_overbought_strong = 70 # RSI surachat fort accessible (était 80)
        
        # Seuils momentum assouplis
        self.momentum_bullish = 53    # Momentum haussier modéré (était 58)
        self.momentum_bearish = 47    # Momentum baissier modéré (était 42)
        self.momentum_strong_bull = 60 # Momentum très haussier (était 65)
        self.momentum_strong_bear = 40 # Momentum très baissier (était 35)
        
        # Volume et confluence assouplis
        self.min_volume_ratio = 0.5   # Volume minimum réduit (était 0.7)
        self.min_confluence = 35      # Confluence minimum réduite (était 45)
        self.strong_confluence = 60   # Confluence forte réduite (était 65)
        
    def _safe_float(self, value) -> Optional[float]:
        """Convertit en float de manière sécurisée."""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _get_current_values(self) -> Dict[str, Any]:
        """Récupère les valeurs actuelles des indicateurs pré-calculés."""
        return {
            'stoch_rsi': self._safe_float(self.indicators.get('stoch_rsi')),
            'stoch_k': self._safe_float(self.indicators.get('stoch_k')),
            'stoch_d': self._safe_float(self.indicators.get('stoch_d')),
            'stoch_signal': self.indicators.get('stoch_signal'),
            'stoch_divergence': self.indicators.get('stoch_divergence'),
            'rsi_14': self._safe_float(self.indicators.get('rsi_14')),
            'momentum_score': self._safe_float(self.indicators.get('momentum_score')),
            'trend_strength': self.indicators.get('trend_strength'),
            'directional_bias': self.indicators.get('directional_bias'),
            'confluence_score': self._safe_float(self.indicators.get('confluence_score')),
            'signal_strength': self.indicators.get('signal_strength'),
            'volume_ratio': self._safe_float(self.indicators.get('volume_ratio')),
            'volume_quality_score': self._safe_float(self.indicators.get('volume_quality_score')),
            'market_regime': self.indicators.get('market_regime'),
            'volatility_regime': self.indicators.get('volatility_regime'),
            'adx_14': self._safe_float(self.indicators.get('adx_14'))
        }
        
    def generate_signal(self) -> Dict[str, Any]:
        """
        Génère un signal basé sur les indicateurs StochRSI pré-calculés.
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
        stoch_rsi = values['stoch_rsi']
        if stoch_rsi is None:
            return {
                "side": None,
                "confidence": 0.0,
                "strength": "weak",
                "reason": "StochRSI non disponible",
                "metadata": {"strategy": self.name}
            }
            
        # SUPPRIMÉ: plus de filtres pré-rejet sur volume et confluence
        # Volume et confluence utilisés comme bonus/pénalité seulement
            
        # Pénaliser (mais ne pas bannir) volatilité extrême
        volatility_regime = values.get('volatility_regime')
        volatility_penalty = 0.0
        if volatility_regime == 'extreme':
            volatility_penalty = -0.05  # Pénalité réduite
            
        signal_side = None
        reason = ""
        confidence_boost = 0.0
        
        # Utilisation du signal StochRSI pré-calculé si disponible
        stoch_signal = values.get('stoch_signal')
        if stoch_signal:
            # Conversion des signaux DB vers signaux stratégie
            if stoch_signal == 'OVERSOLD':
                signal_side = "BUY"
                reason = f"Signal StochRSI pré-calculé: {stoch_signal}"
                confidence_boost += 0.20  # Signal pré-calculé fiable (amélioré)
            elif stoch_signal == 'OVERBOUGHT':
                signal_side = "SELL"
                reason = f"Signal StochRSI pré-calculé: {stoch_signal}"
                confidence_boost += 0.20  # Signal pré-calculé fiable (amélioré)
        
        # Si pas de signal pré-calculé, analyse manuelle avec seuils plus stricts
        if not signal_side:
            if stoch_rsi <= self.oversold_zone:
                signal_side = "BUY"
                if stoch_rsi <= self.extreme_oversold:
                    reason = f"StochRSI extrême ({stoch_rsi:.1f}) - rebond attendu"
                    confidence_boost += 0.25  # Amélioré (était 0.22)
                else:
                    reason = f"StochRSI survente ({stoch_rsi:.1f}) - opportunité rebond"
                    confidence_boost += 0.18  # Amélioré (était 0.15)
                    
            elif stoch_rsi >= self.overbought_zone:
                signal_side = "SELL"
                if stoch_rsi >= self.extreme_overbought:
                    reason = f"StochRSI extrême ({stoch_rsi:.1f}) - correction attendue"
                    confidence_boost += 0.25  # Amélioré (était 0.22)
                else:
                    reason = f"StochRSI surachat ({stoch_rsi:.1f}) - opportunité correction"
                    confidence_boost += 0.18  # Amélioré (était 0.15)
                    
        if signal_side:
            base_confidence = 0.65  # Base BEAUCOUP plus élevée
            
            # Bonus MAJEUR pour divergence détectée (AMÉLIORÉ)
            stoch_divergence = values.get('stoch_divergence')
            if stoch_divergence is True:
                confidence_boost += 0.30  # Divergence = signal très puissant (amélioré)
                reason += " avec DIVERGENCE détectée"
                
            # Confirmation avec croisement K/D - PLUS STRICT
            stoch_k = values.get('stoch_k')
            stoch_d = values.get('stoch_d')
            if stoch_k is not None and stoch_d is not None:
                k_d_diff = abs(stoch_k - stoch_d)
                
                # Croisement favorable avec écart minimum (AJUSTÉ)
                if (signal_side == "BUY" and stoch_k > stoch_d and k_d_diff >= 2.5) or \
                   (signal_side == "SELL" and stoch_k < stoch_d and k_d_diff >= 2.5):
                    confidence_boost += 0.18  # Amélioré (était 0.15)
                    reason += " + K/D fort"
                elif (signal_side == "BUY" and stoch_k > stoch_d and k_d_diff >= 0.8) or \
                     (signal_side == "SELL" and stoch_k < stoch_d and k_d_diff >= 0.8):
                    confidence_boost += 0.10  # Amélioré (était 0.08)
                    reason += " + K/D"
                # Pénalité si croisement défavorable (TRÈS RÉDUITE)
                elif (signal_side == "BUY" and stoch_k <= stoch_d) or \
                     (signal_side == "SELL" and stoch_k >= stoch_d):
                    confidence_boost -= 0.05  # Réduit de -0.10 à -0.05
                    
            # Confirmation avec RSI - SEUILS CRYPTO
            rsi_14 = values.get('rsi_14')
            if rsi_14:
                if signal_side == "BUY":
                    if rsi_14 <= self.rsi_oversold_strong:
                        confidence_boost += 0.20
                        reason += " + RSI survente"
                    elif rsi_14 <= self.rsi_oversold:
                        confidence_boost += 0.12
                        reason += " + RSI survente"
                    elif rsi_14 <= 40:  # Seuil élargi (était 35)
                        confidence_boost += 0.08  # Amélioré (était 0.06)
                        reason += " + RSI"
                    elif rsi_14 > 60:
                        confidence_boost -= 0.05  # Pénalité très réduite
                        
                elif signal_side == "SELL":
                    if rsi_14 >= self.rsi_overbought_strong:
                        confidence_boost += 0.20
                        reason += " + RSI surachat"
                    elif rsi_14 >= self.rsi_overbought:
                        confidence_boost += 0.12
                        reason += " + RSI surachat"
                    elif rsi_14 >= 60:  # Seuil réduit (était 65)
                        confidence_boost += 0.08  # Amélioré (était 0.06)
                        reason += " + RSI"
                    elif rsi_14 < 40:
                        confidence_boost -= 0.05  # Pénalité très réduite
                    
            # Utilisation du momentum_score - SEUILS CRYPTO
            momentum_score = values.get('momentum_score', 50)
            if momentum_score:
                if signal_side == "BUY":
                    if momentum_score >= self.momentum_strong_bull:
                        confidence_boost += 0.18
                        reason += " + momentum"
                    elif momentum_score >= self.momentum_bullish:
                        confidence_boost += 0.10
                        reason += " + momentum"
                    elif momentum_score < 40:  # Rejet momentum contradictoire
                        return {
                            "side": None,
                            "confidence": 0.0,
                            "strength": "weak",
                            "reason": f"Rejet StochRSI BUY: momentum trop faible ({momentum_score:.0f})",
                            "metadata": {"strategy": self.name, "stoch_rsi": stoch_rsi, "momentum_score": momentum_score}
                        }
                        
                elif signal_side == "SELL":
                    if momentum_score <= self.momentum_strong_bear:
                        confidence_boost += 0.18
                        reason += " + momentum"
                    elif momentum_score <= self.momentum_bearish:
                        confidence_boost += 0.10
                        reason += " + momentum"
                    elif momentum_score > 60:  # Rejet momentum contradictoire
                        return {
                            "side": None,
                            "confidence": 0.0,
                            "strength": "weak",
                            "reason": f"Rejet StochRSI SELL: momentum trop fort ({momentum_score:.0f})",
                            "metadata": {"strategy": self.name, "stoch_rsi": stoch_rsi, "momentum_score": momentum_score}
                        }
                    
            # Utilisation du trend_strength - BONUS AJUSTÉS
            trend_strength = values.get('trend_strength')
            if trend_strength:
                trend_str = str(trend_strength).lower()
                if trend_str in ['extreme', 'very_strong']:
                    confidence_boost += 0.25  # Bonus majeur pour tendance extrême
                    reason += " + tendance"
                elif trend_str == 'strong':
                    confidence_boost += 0.15
                    reason += " + tendance"
                elif trend_str == 'moderate':
                    confidence_boost += 0.08
                    reason += " + tendance"
                elif trend_str in ['weak', 'absent']:
                    confidence_boost -= 0.08  # Pénalité réduite (était -0.12)
                    pass  # Tendance faible
                
            # Utilisation du directional_bias - CRITIQUE EN CRYPTO
            directional_bias = values.get('directional_bias')
            if directional_bias:
                bias_upper = directional_bias.upper()
                if (signal_side == "BUY" and bias_upper == "BULLISH") or \
                   (signal_side == "SELL" and bias_upper == "BEARISH"):
                    confidence_boost += 0.15  # Bonus important si aligné
                    reason += " + bias"
                elif signal_side == "BUY" and bias_upper == "BEARISH":
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": "Rejet StochRSI BUY: bias contraire (BEARISH)",
                        "metadata": {"strategy": self.name, "stoch_rsi": stoch_rsi, "directional_bias": directional_bias}
                    }
                elif signal_side == "SELL" and bias_upper == "BULLISH":
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": "Rejet StochRSI SELL: bias contraire (BULLISH)",
                        "metadata": {"strategy": self.name, "stoch_rsi": stoch_rsi, "directional_bias": directional_bias}
                    }
                    
            # Confluence plus stricte - rejet <40
            confluence_score = values.get('confluence_score')
            if confluence_score and confluence_score < 40:
                return {
                    "side": None,
                    "confidence": 0.0,
                    "strength": "weak",
                    "reason": f"Rejet StochRSI: confluence insuffisante ({confluence_score:.0f})",
                    "metadata": {"strategy": self.name, "stoch_rsi": stoch_rsi, "confluence_score": confluence_score}
                }
            # Bonus confluence si bonne
            if confluence_score >= 60:
                confidence_boost += 0.20
                reason += " + confluence"
            elif confluence_score >= 45:
                confidence_boost += 0.10
                
            # Utilisation du signal_strength pré-calculé
            signal_strength_calc = values.get('signal_strength')
            if signal_strength_calc:
                sig_str = str(signal_strength_calc).upper()
                if sig_str == 'STRONG':
                    confidence_boost += 0.12
                    reason += " + signal FORT"
                elif sig_str == 'MODERATE':
                    confidence_boost += 0.06
                    reason += " + signal modéré"
                elif sig_str == 'WEAK':
                    confidence_boost -= 0.05  # Pénalité réduite (était -0.08)
                    reason += " mais signal faible"
                    
            # Volume plus strict - neutre ≥1.0
            volume_ratio = values.get('volume_ratio')
            if volume_ratio is not None:
                if volume_ratio >= 1.8:  # Volume fort
                    confidence_boost += 0.25
                    reason += " + volume"
                elif volume_ratio >= 1.3:  # Volume élevé
                    confidence_boost += 0.15
                elif volume_ratio >= 1.0:  # Volume neutre
                    pass  # Pas de bonus/malus
                else:  # Volume faible = pénalité
                    confidence_boost -= 0.08
                    
            # Volume quality (AJUSTÉ)
            volume_quality_score = values.get('volume_quality_score')
            if volume_quality_score is not None and volume_quality_score >= 65:  # Seuil réduit (était 70)
                confidence_boost += 0.10  # Amélioré (était 0.08)
                reason += " + volume HQ"
                
            # Filtre ADX pour StochRSI
            adx_14 = values.get('adx_14')
            if adx_14:
                if adx_14 < 15:  # Range trop mou
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Rejet StochRSI: ADX trop faible ({adx_14:.0f}) - range mou",
                        "metadata": {"strategy": self.name, "stoch_rsi": stoch_rsi, "adx_14": adx_14}
                    }
                elif adx_14 > 40:  # Tendance trop forte = StochRSI trompeur
                    return {
                        "side": None,
                        "confidence": 0.0,
                        "strength": "weak",
                        "reason": f"Rejet StochRSI: ADX trop fort ({adx_14:.0f}) - tendance trop forte",
                        "metadata": {"strategy": self.name, "stoch_rsi": stoch_rsi, "adx_14": adx_14}
                    }
            
            # Market regime bonus - StochRSI EXCELLENT en ranging
            market_regime = values.get('market_regime')
            if market_regime == 'RANGING':
                confidence_boost += 0.20
                reason += " + ranging"
            elif market_regime in ['TRENDING_BULL', 'TRENDING_BEAR']:
                confidence_boost += 0.05
            elif market_regime == 'VOLATILE':
                confidence_boost -= 0.05
                    
            # Filtre final - seuil accessible pour StochRSI
            # Appliquer pénalité volatilité
            confidence_boost += volatility_penalty
            
            # SUPPRIMÉ: plus de filtre de rejet final - laisser aggregator décider
            raw_confidence = base_confidence * (1 + confidence_boost)
            
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
                    "stoch_rsi": stoch_rsi,
                    "stoch_k": stoch_k,
                    "stoch_d": stoch_d,
                    "stoch_signal": stoch_signal,
                    "stoch_divergence": stoch_divergence,
                    "rsi_14": rsi_14,
                    "momentum_score": momentum_score,
                    "trend_strength": trend_strength,
                    "confluence_score": confluence_score,
                    "volume_ratio": volume_ratio,
                    "directional_bias": directional_bias,
                    "market_regime": market_regime
                }
            }
            
        return {
            "side": None,
            "confidence": 0.0,
            "strength": "weak",
            "reason": f"StochRSI en zone neutre ({stoch_rsi:.1f}) - seuils: {self.oversold_zone}/{self.overbought_zone}",
            "metadata": {
                "strategy": self.name,
                "symbol": self.symbol,
                "stoch_rsi": stoch_rsi,
                "oversold_zone": self.oversold_zone,
                "overbought_zone": self.overbought_zone
            }
        }
        
    def validate_data(self) -> bool:
        """Valide que les indicateurs StochRSI requis sont présents."""
        if not super().validate_data():
            return False
            
        # Vérifier StochRSI obligatoire
        if 'stoch_rsi' not in self.indicators or self.indicators['stoch_rsi'] is None:
            logger.warning(f"{self.name}: StochRSI manquant")
            return False
            
        return True